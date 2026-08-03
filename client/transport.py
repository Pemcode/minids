"""Transport HTTP vers le pod, en stdlib pure (urllib).

Le proxy RunPod ferme toute connexion à 100 s. Tout est donc découpé :

* upload en chunks de 8 Mo, chacun repris individuellement en cas de coupure ;
* suivi par polling de petites réponses JSON ;
* téléchargement par requêtes `Range`, reprise sur le fichier `.part` existant.

Conséquence pratique : une coupure réseau ne coûte jamais plus qu'un chunk.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CHUNK = 8 * 1024 * 1024
DEFAULT_TIMEOUT = 90  # sous les 100 s du proxy
RETRY_STATUSES = {408, 429, 500, 502, 503, 504, 520, 522, 524}


class MinidsError(RuntimeError):
    pass


class MinidsHTTPError(MinidsError):
    """Erreur HTTP dont le code est exploitable par l'appelant (416 notamment)."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


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
        now = time.time()
        if done < self.total and now - self._last < 0.2:
            return
        self._last = now
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


def sha256_file(path: Path, progress: Progress | None = None) -> str:
    digest = hashlib.sha256()
    done = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            done += len(block)
            if progress:
                progress.update(done)
    return digest.hexdigest()


class MinidsClient:
    def __init__(self, url: str, token: str, timeout: int = DEFAULT_TIMEOUT, retries: int = 5) -> None:
        if not url:
            raise MinidsError("URL du pod manquante (--url ou MINIDS_URL)")
        if not token:
            raise MinidsError("token manquant (--token ou MINIDS_TOKEN)")
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self.retries = retries

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
        all_headers = {"Authorization": f"Bearer {self.token}", **(headers or {})}
        last_error: Exception | None = None

        for attempt in range(self.retries):
            request = urllib.request.Request(url, data=body, method=method, headers=all_headers)
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    payload = response.read()
                    if not parse_json:
                        return payload, dict(response.headers), response.status
                    return json.loads(payload) if payload else {}
            except urllib.error.HTTPError as exc:
                detail = _describe_http_error(exc)
                if exc.code in RETRY_STATUSES and attempt < self.retries - 1:
                    last_error = MinidsHTTPError(exc.code, detail)
                    _backoff(attempt)
                    continue
                raise MinidsHTTPError(exc.code, detail) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = MinidsError(f"{method} {path} : {type(exc).__name__} {exc}")
                if attempt < self.retries - 1:
                    _backoff(attempt)
                    continue
                raise last_error from exc

        raise last_error or MinidsError("échec inconnu")

    # -- endpoints ---------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health", timeout=20)

    def create_job(self, filename: str, size: int, chunk_size: int, sha256: str | None, params: dict[str, Any]) -> str:
        payload = {
            "filename": filename,
            "size": size,
            "chunk_size": chunk_size,
            "sha256": sha256,
            "params": {key: value for key, value in params.items() if value is not None},
        }
        response = self._request("POST", "/jobs", json.dumps(payload).encode(), {"Content-Type": "application/json"})
        return str(response["job_id"])

    def uploaded_chunks(self, job_id: str) -> set[int]:
        response = self._request("GET", f"/jobs/{job_id}/chunks")
        return {int(index) for index in response.get("received", [])}

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
        size = path.stat().st_size
        total_chunks = (size + chunk_size - 1) // chunk_size
        already = self.uploaded_chunks(job_id)
        progress = Progress(f"upload  {path.name}", size, quiet)
        sent = min(size, len(already) * chunk_size)

        with path.open("rb") as handle:
            for index in range(total_chunks):
                if should_stop is not None and should_stop():
                    raise MinidsError("upload interrompu")
                if index in already:
                    continue
                handle.seek(index * chunk_size)
                block = handle.read(chunk_size)
                self._request(
                    "PUT", f"/jobs/{job_id}/chunks/{index}", block,
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
        return self._request("POST", f"/jobs/{job_id}/start", b"")

    def status(self, job_id: str, logs: bool = True) -> dict[str, Any]:
        return self._request("GET", f"/jobs/{job_id}?logs={'true' if logs else 'false'}", timeout=30)

    def cancel(self, job_id: str) -> dict[str, Any]:
        return self._request("POST", f"/jobs/{job_id}/cancel", b"")

    def jobs(self) -> list[dict[str, Any]]:
        """Jobs connus du pod, du plus récent au plus ancien.

        Sert à rejoindre un scan lancé depuis une autre session : le pod garde
        l'état sur disque, donc un client qui a été fermé peut retrouver le job
        sans en connaître l'identifiant par cœur.
        """
        jobs = self._request("GET", "/jobs").get("jobs", [])
        return sorted(jobs, key=lambda job: job.get("created_at") or 0, reverse=True)

    def artifacts(self, job_id: str) -> list[dict[str, Any]]:
        return self._request("GET", f"/jobs/{job_id}/artifacts").get("artifacts", [])

    def wait(
        self,
        job_id: str,
        poll_seconds: float = 5.0,
        on_update: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll jusqu'à l'état terminal. Chaque requête est courte : jamais de timeout proxy."""
        seen_logs = 0
        while True:
            state = self.status(job_id)
            if on_update:
                logs = state.get("logs", [])
                state = {**state, "new_logs": logs[seen_logs:]}
                seen_logs = len(logs)
                on_update(state)
            if state["status"] in {"done", "failed", "cancelled"}:
                return state
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
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")

        offset = partial.stat().st_size if partial.exists() else 0
        total: int | None = None
        progress: Progress | None = None

        while total is None or offset < total:
            if should_stop is not None and should_stop():
                raise MinidsError("téléchargement interrompu")
            end = offset + chunk_size - 1
            try:
                payload, headers, status = self._request(
                    "GET", f"/jobs/{job_id}/artifacts/{urllib.parse.quote(name)}",
                    headers={"Range": f"bytes={offset}-{end}"}, parse_json=False,
                )
            except MinidsHTTPError as exc:
                # 416 = le `.part` déborde de l'artefact (reste d'un autre fichier).
                # On repart de zéro plutôt que d'échouer.
                if exc.status != 416 or offset == 0:
                    raise
                partial.unlink(missing_ok=True)
                offset = 0
                continue
            if total is None:
                total = _total_from_headers(headers, len(payload), status)
                progress = Progress(f"download {name}", total, quiet)
                if offset and offset >= total:  # `.part` périmé, mais accepté par le serveur
                    partial.unlink(missing_ok=True)
                    offset = 0
                    total = None
                    continue
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

        if expected_sha256:
            actual = sha256_file(partial)
            if actual != expected_sha256:
                raise MinidsError(f"{name} : sha256 incorrect ({actual[:12]}… ≠ {expected_sha256[:12]}…)")

        destination.unlink(missing_ok=True)
        partial.replace(destination)
        return destination


def _total_from_headers(headers: dict[str, str], payload_size: int, status: int) -> int:
    content_range = headers.get("Content-Range") or headers.get("content-range")
    if content_range and "/" in content_range:
        try:
            return int(content_range.rsplit("/", 1)[1])
        except ValueError:
            pass
    for key in ("X-Content-Length", "x-content-length", "Content-Length", "content-length"):
        if key in headers:
            try:
                return int(headers[key])
            except ValueError:
                continue
    return payload_size


def _describe_http_error(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", "replace")
        detail = json.loads(body).get("error", body)
    except Exception:  # noqa: BLE001
        detail = exc.reason
    return f"HTTP {exc.code} : {detail}"


def _backoff(attempt: int) -> None:
    time.sleep(min(30.0, 1.5 * (2**attempt)))
