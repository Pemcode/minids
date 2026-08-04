"""Transport HTTP vers le pod, en stdlib pure (urllib).

Le proxy RunPod ferme toute connexion à 100 s. Tout est donc découpé :

* upload en chunks de 8 Mo, chacun repris individuellement en cas de coupure ;
* suivi par polling de petites réponses JSON ;
* téléchargement par requêtes `Range`, reprise sur le fichier `.part` existant.

Conséquence pratique : une coupure réseau ne coûte jamais plus qu'un chunk.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CHUNK = 8 * 1024 * 1024
MAX_CHUNK = 64 * 1024 * 1024
DEFAULT_TIMEOUT = 90  # sous les 100 s du proxy
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 520, 522, 524}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Sans en-tête explicite, urllib s'annonce « Python-urllib/3.x » — que Cloudflare,
# devant le proxy RunPod, bloque en 403 avant même d'atteindre le pod. Le
# symptôme est déroutant : /health répond dans un navigateur, et le client se
# voit refuser tout accès avec une configuration pourtant correcte.
USER_AGENT = "minids-client/0.1.0"


class MinidsError(RuntimeError):
    pass


class MinidsHTTPError(MinidsError):
    """Erreur HTTP dont le code est exploitable par l'appelant (416 notamment)."""

    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers or {}


@dataclass
class Progress:
    """Affichage d'avancement sur une ligne, sans dépendance."""

    label: str
    total: int
    quiet: bool = False
    _last: float = 0.0

    def update(self, done: int) -> None:
        if self.quiet:
            return
        now = time.monotonic()
        if done < self.total and now - self._last < 0.2:
            return
        self._last = now
        done = max(0, min(done, self.total)) if self.total > 0 else 0
        ratio = done / self.total if self.total else 1.0
        filled = int(28 * ratio)
        bar = "█" * filled + "·" * (28 - filled)
        if self.total < 1_000_000:
            size = f"{done / 1e3:.0f}/{self.total / 1e3:.0f} Ko"
        else:
            size = f"{done / 1e6:.1f}/{self.total / 1e6:.1f} Mo"
        print(f"\r{self.label} [{bar}] {ratio * 100:5.1f}% ({size})", end="")
        if done >= self.total:
            print()


def sha256_file(
    path: Path,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    digest = hashlib.sha256()
    done = 0
    with Path(path).open("rb") as handle:
        while True:
            if should_stop is not None and should_stop():
                raise MinidsError("calcul de l'empreinte interrompu")
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if progress:
                progress.update(done)
    return digest.hexdigest()


def artifact_destination(directory: Path, name: str) -> Path:
    """Construit un chemin d'artefact qui reste strictement dans ``directory``."""
    if not isinstance(name, str) or not name or Path(name).name != name or Path(name).is_absolute():
        raise MinidsError(f"nom d'artefact invalide : {name!r}")
    if name in {".", ".."}:
        raise MinidsError(f"nom d'artefact invalide : {name!r}")
    return Path(directory) / name


def validate_job_id(value: str) -> str:
    if not isinstance(value, str) or JOB_ID_RE.fullmatch(value) is None:
        raise MinidsError("identifiant de job invalide (32 caractères hexadécimaux attendus)")
    return value


def job_destination(directory: Path, job_id: str) -> Path:
    return Path(directory) / validate_job_id(job_id)


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    port = parsed.port if parsed.port is not None else {"http": 80, "https": 443}.get(scheme)
    return scheme, (parsed.hostname or "").lower(), port


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Autorise les redirections sans jamais transmettre le bearer à une autre origine."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        target = urllib.parse.urljoin(req.full_url, newurl)
        if _origin(req.full_url) != _origin(target):
            raise urllib.error.HTTPError(
                req.full_url,
                code,
                "redirection vers une autre origine refusée",
                headers,
                fp,
            )
        return super().redirect_request(req, fp, code, msg, headers, target)


class MinidsClient:
    def __init__(self, url: str, token: str, timeout: int = DEFAULT_TIMEOUT, retries: int = 5) -> None:
        if not isinstance(url, str) or not url.strip():
            raise MinidsError("URL du pod manquante (--url ou MINIDS_URL)")
        if not isinstance(token, str) or not token.strip():
            raise MinidsError("token manquant (--token ou MINIDS_TOKEN)")
        url = url.strip()
        token = token.strip()
        try:
            parsed = urllib.parse.urlsplit(url)
            _ = parsed.port  # force la validation d'un port éventuellement mal formé
        except ValueError as exc:
            raise MinidsError(f"URL du pod invalide : {exc}") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise MinidsError("URL du pod invalide : une URL http(s) complète est requise")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise MinidsError("URL du pod invalide : identifiants, requête et fragment sont interdits")
        if parsed.path not in {"", "/"}:
            raise MinidsError("URL du pod invalide : ne pas ajouter de chemin après le nom d'hôte")
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
            raise MinidsError("HTTP non chiffré refusé : utilise HTTPS (HTTP reste autorisé sur localhost)")
        if not isinstance(timeout, (int, float)) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise MinidsError("timeout invalide : une durée strictement positive est requise")
        if not isinstance(retries, int) or isinstance(retries, bool) or retries < 1:
            raise MinidsError("retries invalide : au moins une tentative est requise")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self._opener = urllib.request.build_opener(_SameOriginRedirectHandler())

    # -- couche HTTP ------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        parse_json: bool = True,
        timeout: int | None = None,
    ) -> Any:
        url = f"{self.url}{path}"
        all_headers = {
            "Authorization": f"Bearer {self.token}",
            "User-Agent": USER_AGENT,
            **(headers or {}),
        }
        last_error: Exception | None = None
        retry_allowed = method.upper() in {"GET", "PUT"}

        for attempt in range(self.retries):
            # Base validée http(s) au constructeur ; ``path`` est produit par le client.
            request = urllib.request.Request(url, data=body, method=method, headers=all_headers)  # noqa: S310
            try:
                effective_timeout = self.timeout if timeout is None else timeout
                with self._opener.open(request, timeout=effective_timeout) as response:
                    payload = response.read()
                    if not parse_json:
                        return payload, dict(response.headers), response.status
                    if not payload:
                        return {}
                    try:
                        return json.loads(payload)
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise MinidsError(f"{method} {path} : réponse JSON invalide") from exc
            except urllib.error.HTTPError as exc:
                detail = _describe_http_error(exc)
                error_headers = dict(exc.headers or {})
                if retry_allowed and exc.code in RETRY_STATUSES and attempt < self.retries - 1:
                    last_error = MinidsHTTPError(exc.code, detail, error_headers)
                    _backoff(attempt)
                    continue
                raise MinidsHTTPError(exc.code, detail, error_headers) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = MinidsError(f"{method} {path} : {type(exc).__name__} {exc}")
                if retry_allowed and attempt < self.retries - 1:
                    _backoff(attempt)
                    continue
                raise last_error from exc

        raise last_error or MinidsError("échec inconnu")

    # -- endpoints ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return _dict_response(self._request("GET", "/health", timeout=20), "/health")

    def create_job(self, filename: str, size: int, chunk_size: int, sha256: str | None, params: dict[str, Any]) -> str:
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            raise MinidsError("nom de fichier invalide")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise MinidsError("taille de fichier invalide")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or not 64 * 1024 <= chunk_size <= MAX_CHUNK:
            raise MinidsError("taille de chunk invalide (64 Kio à 64 Mio)")
        if sha256 is not None and re.fullmatch(r"[0-9a-fA-F]{64}", sha256) is None:
            raise MinidsError("empreinte sha256 invalide")
        if not isinstance(params, dict):
            raise MinidsError("paramètres de job invalides")
        payload = {
            "filename": filename,
            "size": size,
            "chunk_size": chunk_size,
            "sha256": sha256,
            "params": {key: value for key, value in params.items() if value is not None},
        }
        response = _dict_response(
            self._request("POST", "/jobs", json.dumps(payload).encode(), {"Content-Type": "application/json"}),
            "création du job",
        )
        job_id = response.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            raise MinidsError("création du job : identifiant absent de la réponse")
        return validate_job_id(job_id)

    def uploaded_chunks(self, job_id: str) -> set[int]:
        response = _dict_response(self._request("GET", f"/jobs/{_segment(job_id)}/chunks"), "liste des chunks")
        received = response.get("received", [])
        if not isinstance(received, list):
            raise MinidsError("liste des chunks : champ 'received' invalide")
        if any(not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in received):
            raise MinidsError("liste des chunks : index invalide")
        chunks = set(received)
        total = response.get("total")
        if total is not None:
            if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                raise MinidsError("liste des chunks : total invalide")
            if any(index >= total for index in chunks):
                raise MinidsError("liste des chunks : index hors bornes")
        return chunks

    def upload(
        self,
        job_id: str,
        path: Path,
        chunk_size: int = DEFAULT_CHUNK,
        quiet: bool = False,
        on_progress: Callable[[int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> None:
        """Téléverse par morceaux, en sautant ceux déjà reçus.

        `on_progress(envoyé, total)` alimente une barre externe (interface
        graphique) ; `should_stop()` est consulté entre deux morceaux, ce qui
        borne le délai d'annulation à un seul chunk.
        """
        path = Path(path)
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or not 64 * 1024 <= chunk_size <= MAX_CHUNK:
            raise MinidsError("taille de chunk invalide (64 Kio à 64 Mio)")
        if not path.is_file():
            raise MinidsError(f"fichier à envoyer introuvable : {path}")
        size = path.stat().st_size
        if size < 1:
            raise MinidsError(f"fichier à envoyer vide : {path}")
        total_chunks = (size + chunk_size - 1) // chunk_size
        already = self.uploaded_chunks(job_id)
        incompatible = {index for index in already if index >= total_chunks}
        if incompatible:
            raise MinidsError("reprise impossible : les chunks distants ne correspondent pas à ce fichier")
        progress = Progress(f"upload  {path.name}", size, quiet)
        sent = sum(min(chunk_size, size - index * chunk_size) for index in already)
        progress.update(sent)
        if on_progress is not None and sent:
            on_progress(sent, size)

        with path.open("rb") as handle:
            for index in range(total_chunks):
                if should_stop is not None and should_stop():
                    raise MinidsError("upload interrompu")
                if index in already:
                    continue
                handle.seek(index * chunk_size)
                block = handle.read(chunk_size)
                self._request(
                    "PUT",
                    f"/jobs/{_segment(job_id)}/chunks/{index}",
                    block,
                    {"Content-Type": "application/octet-stream"},
                )
                sent = min(size, sent + len(block))
                progress.update(sent)
                if on_progress is not None:
                    on_progress(sent, size)
        progress.update(size)
        if on_progress is not None:
            on_progress(size, size)

    def start(self, job_id: str) -> dict[str, Any]:
        job_id = validate_job_id(job_id)
        try:
            return _dict_response(self._request("POST", f"/jobs/{job_id}/start", b""), "démarrage du job")
        except MinidsHTTPError as exc:
            # Les erreurs de proxy après un POST peuvent arriver après que le
            # serveur a reçu la requête. Les 4xx métier restent non ambigus.
            if exc.status not in RETRY_STATUSES:
                raise
            start_error: MinidsError = exc
        except MinidsError as exc:
            start_error = exc

        # L'assemblage et la validation peuvent durer plus longtemps que le
        # timeout du client alors que le serveur a bien placé le job en file.
        # Un GET idempotent permet de récupérer ce cas sans rejouer le POST.
        try:
            state = self.status(job_id, logs=False)
        except MinidsError as status_error:
            raise MinidsError(
                f"{start_error}; impossible de vérifier si le démarrage a abouti ({status_error})"
            ) from start_error
        if state["status"] in {"starting", "queued", "running", "done"}:
            return state
        if state["status"] == "created":
            raise MinidsError(
                f"{start_error}; démarrage non confirmé pour le job {job_id}. "
                "Attendez quelques instants puis rattachez-vous au job avant de relancer."
            ) from start_error
        raise MinidsError(f"{start_error}; le job est désormais dans l'état {state['status']!r}") from start_error

    def status(self, job_id: str, logs: bool = True) -> dict[str, Any]:
        state = _dict_response(
            self._request("GET", f"/jobs/{_segment(job_id)}?logs={'true' if logs else 'false'}", timeout=30),
            "état du job",
        )
        status = state.get("status")
        allowed = {"created", "starting", "queued", "running", "done", "failed", "cancelled"}
        if not isinstance(status, str) or status not in allowed:
            raise MinidsError(f"état du job : statut inconnu {status!r}")
        if "logs" in state and not isinstance(state["logs"], list):
            raise MinidsError("état du job : journal invalide")
        for key in ("log_count", "logs_offset"):
            value = state.get(key)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
                raise MinidsError(f"état du job : {key} invalide")
        if (
            state.get("log_count") is not None
            and state.get("logs_offset") is not None
            and state["logs_offset"] > state["log_count"]
        ):
            raise MinidsError("état du job : fenêtre de journal incohérente")
        if logs:
            state.setdefault("logs", [])
        return state

    def cancel(self, job_id: str) -> dict[str, Any]:
        return _dict_response(self._request("POST", f"/jobs/{_segment(job_id)}/cancel", b""), "annulation du job")

    def jobs(self) -> list[dict[str, Any]]:
        """Jobs connus du pod, du plus récent au plus ancien.

        Sert à rejoindre un scan lancé depuis une autre session : le pod garde
        l'état sur disque, donc un client qui a été fermé peut retrouver le job
        sans en connaître l'identifiant par cœur.
        """
        payload = _dict_response(self._request("GET", "/jobs"), "liste des jobs")
        jobs = payload.get("jobs", [])
        if not isinstance(jobs, list) or any(not isinstance(job, dict) for job in jobs):
            raise MinidsError("liste des jobs : réponse invalide")
        return sorted(jobs, key=_created_at, reverse=True)

    def artifacts(self, job_id: str) -> list[dict[str, Any]]:
        payload = _dict_response(self._request("GET", f"/jobs/{_segment(job_id)}/artifacts"), "liste des artefacts")
        artifacts = payload.get("artifacts", [])
        if not isinstance(artifacts, list) or any(not isinstance(item, dict) for item in artifacts):
            raise MinidsError("liste des artefacts : réponse invalide")
        return artifacts

    def wait(
        self,
        job_id: str,
        poll_seconds: float = 5.0,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll jusqu'à l'état terminal. Chaque requête est courte : jamais de timeout proxy."""
        if not isinstance(poll_seconds, (int, float)) or not math.isfinite(float(poll_seconds)) or poll_seconds <= 0:
            raise MinidsError("intervalle de polling invalide")
        previous_logs: list[Any] = []
        log_cursor: int | None = None
        while True:
            state = self.status(job_id)
            logs = state.get("logs", [])
            if not isinstance(logs, list):
                raise MinidsError("état du job : journal invalide")
            if on_update:
                logs_offset = state.get("logs_offset")
                state = {
                    **state,
                    "new_logs": new_log_lines(
                        previous_logs,
                        logs,
                        previous_end=log_cursor,
                        current_offset=logs_offset if isinstance(logs_offset, int) else None,
                    ),
                }
                on_update(state)
            logs_offset = state.get("logs_offset")
            log_cursor = (
                logs_offset + len(logs) if isinstance(logs_offset, int) and not isinstance(logs_offset, bool) else None
            )
            previous_logs = list(logs)
            status = state.get("status")
            if not isinstance(status, str):
                raise MinidsError("état du job : statut absent de la réponse")
            if status == "created":
                raise MinidsError("upload incomplet : ce job n'a pas été démarré")
            if status in {"done", "failed", "cancelled"}:
                return state
            if status not in {"starting", "queued", "running"}:
                raise MinidsError(f"état du job : statut inconnu {status!r}")
            time.sleep(poll_seconds)

    def download(
        self,
        job_id: str,
        name: str,
        destination: Path,
        chunk_size: int = DEFAULT_CHUNK,
        quiet: bool = False,
        expected_sha256: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> Path:
        """Télécharge un artefact par `Range`, en reprenant un `.part` existant."""
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or not 1 <= chunk_size <= MAX_CHUNK:
            raise MinidsError("taille de chunk invalide (maximum 64 Mio)")
        if not isinstance(name, str) or not name:
            raise MinidsError("nom d'artefact invalide")
        if expected_sha256 is not None and (
            not isinstance(expected_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None
        ):
            raise MinidsError("empreinte sha256 attendue invalide")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")

        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise MinidsError(f"destination invalide : {destination}")
        if partial.is_symlink() or (partial.exists() and not partial.is_file()):
            raise MinidsError(f"fichier de reprise invalide : {partial}")

        offset = partial.stat().st_size if partial.exists() else 0
        total: int | None = None
        progress: Progress | None = None

        while total is None or offset < total:
            if should_stop is not None and should_stop():
                raise MinidsError("téléchargement interrompu")
            end = offset + chunk_size - 1
            try:
                payload, headers, status = self._request(
                    "GET",
                    f"/jobs/{_segment(job_id)}/artifacts/{urllib.parse.quote(name, safe='')}",
                    headers={"Range": f"bytes={offset}-{end}"},
                    parse_json=False,
                )
            except MinidsHTTPError as exc:
                if exc.status != 416:
                    raise
                remote_total = _unsatisfied_range_total(exc.headers)
                if remote_total is not None and offset == remote_total:
                    total = remote_total
                    partial.touch(exist_ok=True)
                    break
                if offset > 0:
                    # Le `.part` déborde de l'artefact (reste d'un autre fichier).
                    partial.unlink(missing_ok=True)
                    offset = 0
                    continue
                if remote_total == 0:
                    total = 0
                    partial.touch(exist_ok=True)
                    break
                raise

            if status not in {200, 206}:
                raise MinidsError(f"{name} : statut HTTP inattendu {status}")
            if status == 200 and offset:
                # Certains serveurs/proxies ignorent Range. Ne jamais ajouter le
                # fichier complet derrière un préfixe déjà présent.
                partial.unlink(missing_ok=True)
                offset = 0

            response_total = _total_from_headers(headers, len(payload), status)
            if status == 206:
                range_info = _content_range(headers)
                if range_info is None:
                    raise MinidsError(f"{name} : réponse 206 sans Content-Range valide")
                range_start, range_end, range_total = range_info
                if range_start != offset:
                    raise MinidsError(f"{name} : reprise incohérente (octet {range_start}, attendu {offset})")
                if len(payload) != range_end - range_start + 1:
                    raise MinidsError(f"{name} : morceau HTTP tronqué")
                response_total = range_total
            if total is None:
                total = response_total
                progress = Progress(f"download {name}", total, quiet)
                if offset and offset >= total:  # `.part` périmé, mais accepté par le serveur
                    partial.unlink(missing_ok=True)
                    offset = 0
                    total = None
                    continue
            elif response_total != total:
                raise MinidsError(f"{name} : taille distante modifiée pendant le téléchargement")
            if status == 200 and len(payload) != total:
                raise MinidsError(f"{name} : réponse complète tronquée ({len(payload)} octets sur {total})")
            if not payload:
                break
            with partial.open("ab") as handle:
                handle.write(payload)
            offset += len(payload)
            if progress:
                progress.update(min(offset, total))
            if on_progress is not None:
                on_progress(min(offset, total), total)

        if total is not None and offset != total:
            raise MinidsError(f"{name} : {offset} octets reçus sur {total}")

        if expected_sha256 is not None:
            actual = sha256_file(partial, should_stop=should_stop)
            if actual.lower() != expected_sha256.lower():
                partial.unlink(missing_ok=True)
                raise MinidsError(f"{name} : sha256 incorrect ({actual[:12]}… ≠ {expected_sha256[:12]}…)")

        partial.replace(destination)
        return destination


def _total_from_headers(headers: dict[str, str], payload_size: int, status: int) -> int:
    content_range = headers.get("Content-Range") or headers.get("content-range")
    if content_range and "/" in content_range:
        try:
            value = int(content_range.rsplit("/", 1)[1])
            if value >= 0:
                return value
        except ValueError:
            pass
    for key in ("X-Content-Length", "x-content-length", "Content-Length", "content-length"):
        if key in headers:
            try:
                value = int(headers[key])
                if value >= 0:
                    return value
            except ValueError:
                continue
    if status == 200:
        return payload_size
    raise MinidsError("réponse partielle sans taille totale exploitable")


_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+)$", re.IGNORECASE)
_UNSATISFIED_RANGE_RE = re.compile(r"^bytes\s+\*/(\d+)$", re.IGNORECASE)


def _content_range(headers: dict[str, str]) -> tuple[int, int, int] | None:
    value = headers.get("Content-Range") or headers.get("content-range") or ""
    match = _CONTENT_RANGE_RE.fullmatch(value.strip())
    if match is None:
        return None
    start, end, total = (int(part) for part in match.groups())
    if start > end or end >= total:
        return None
    return start, end, total


def _unsatisfied_range_total(headers: dict[str, str]) -> int | None:
    value = headers.get("Content-Range") or headers.get("content-range") or ""
    match = _UNSATISFIED_RANGE_RE.fullmatch(value.strip())
    return int(match.group(1)) if match else None


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
        decoded = json.loads(body)
        detail: Any = body
        if isinstance(decoded, dict):
            detail = decoded.get("error") or decoded.get("detail") or body
        if isinstance(detail, list):
            messages = []
            for item in detail:
                if isinstance(item, dict):
                    location = ".".join(str(part) for part in item.get("loc", []))
                    message = str(item.get("msg") or item.get("message") or item)
                    messages.append(f"{location}: {message}" if location else message)
                else:
                    messages.append(str(item))
            detail = "; ".join(messages)
    except Exception:  # noqa: BLE001
        detail = exc.reason
    return f"HTTP {exc.code} : {str(detail)[:1000]}"


def _dict_response(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MinidsError(f"{context} : réponse invalide")
    return value


def _segment(value: str) -> str:
    return urllib.parse.quote(validate_job_id(value), safe="")


def new_log_lines(
    previous: list[Any],
    current: list[Any],
    *,
    previous_end: int | None = None,
    current_offset: int | None = None,
) -> list[Any]:
    """Retourne la partie nouvelle, avec curseur absolu ou fallback par chevauchement."""
    if previous_end is not None and current_offset is not None:
        current_end = current_offset + len(current)
        if current_offset <= previous_end <= current_end:
            return current[previous_end - current_offset :]
        return list(current)
    overlap_limit = min(len(previous), len(current))
    for overlap in range(overlap_limit, 0, -1):
        if previous[-overlap:] == current[:overlap]:
            return current[overlap:]
    return list(current)


def _created_at(job: dict[str, Any]) -> float:
    try:
        value = float(job.get("created_at") or 0.0)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _is_loopback(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _backoff(attempt: int) -> None:
    time.sleep(min(30.0, 1.5 * (2**attempt)))
