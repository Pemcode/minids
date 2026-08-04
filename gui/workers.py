"""Tâches de fond.

Règle absolue en Qt : le fil d'exécution de l'interface ne doit jamais bloquer.
Un scan dure une quinzaine de minutes et enchaîne ffmpeg, des requêtes réseau et
de l'attente — tout cela vit ici, et ne communique avec l'interface que par
signaux.

`MinidsClient` étant synchrone à base de rappels, il s'intègre naturellement :
on lui passe des fonctions qui réémettent en signaux Qt.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from client.payload import build_payload
from client.transport import (
    MAX_CHUNK,
    MinidsClient,
    MinidsError,
    artifact_destination,
    job_destination,
    new_log_lines,
    sha256_file,
    validate_job_id,
)

DEFAULT_ARTIFACTS = ["mesh.glb", "report.json", "preview.png"]


@dataclass
class ScanRequest:
    """Tout ce qu'il faut pour lancer un scan, collecté depuis le formulaire.

    `job_id` renseigné signifie « rejoindre ce job » : la source et l'envoi sont
    alors sans objet, le travail commence directement au suivi.
    """

    url: str
    token: str
    source: Path
    output_dir: Path
    params: dict[str, Any]
    long_side: int = 1024
    send_video: bool = False
    chunk_size: int = 8 * 1024 * 1024
    poll_seconds: float = 3.0
    fetch_raw: bool = False
    fetch_all: bool = False
    job_id: str = ""


@dataclass
class ScanOutcome:
    """Résultat d'un scan, y compris les mesures faites côté client."""

    job_id: str
    status: str
    directory: Path
    report: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    payload_bytes: int = 0
    extract_seconds: float = 0.0
    upload_seconds: float = 0.0
    server_seconds: float = 0.0
    download_seconds: float = 0.0
    download_bytes: int = 0
    total_seconds: float = 0.0
    error: str = ""

    @property
    def upload_rate(self) -> float:
        return self.payload_bytes / self.upload_seconds if self.upload_seconds > 0 else 0.0

    @property
    def download_rate(self) -> float:
        return self.download_bytes / self.download_seconds if self.download_seconds > 0 else 0.0


class HealthWorker(QThread):
    """Interroge `/health`. Court, mais réseau : donc hors du fil graphique."""

    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, url: str, token: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url
        self._token = token

    def run(self) -> None:
        try:
            client = MinidsClient(url=self._url, token=self._token, timeout=20, retries=2)
            self.succeeded.emit(client.health())
        except MinidsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class JobListWorker(QThread):
    """Récupère la liste des jobs du pod, pour le dialogue de reprise."""

    succeeded = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, url: str, token: str, parent=None) -> None:
        super().__init__(parent)
        self._url = url
        self._token = token

    def run(self) -> None:
        try:
            client = MinidsClient(url=self._url, token=self._token, timeout=20, retries=2)
            self.succeeded.emit(client.jobs())
        except MinidsError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class ScanWorker(QThread):
    """Chaîne complète : extraction locale → upload → attente → récupération.

    Si `request.job_id` est renseigné, les deux premières étapes sont sautées et
    le worker rejoint un job déjà en cours sur le pod. Le reste — signaux,
    progression, historique — est rigoureusement identique : c'est ce qui permet
    à l'interface de câbler un seul chemin.
    """

    log = pyqtSignal(str)
    phase_changed = pyqtSignal(str)  # extraction | upload | attente | téléchargement
    # ``int`` Qt est signé sur 32 bits : une vidéo/artefact > 2 Gio débordait le
    # signal. ``object`` transporte les entiers Python sans troncature.
    transfer_progress = pyqtSignal(str, object, object, float)  # libellé, fait, total, débit
    job_created = pyqtSignal(str)
    server_state = pyqtSignal(dict)
    finished_ok = pyqtSignal(object)  # ScanOutcome
    failed = pyqtSignal(object)  # ScanOutcome

    def __init__(self, request: ScanRequest, parent=None) -> None:
        super().__init__(parent)
        self.request = request
        self._cancelled = False
        self._cancel_sent = False
        self._cancel_confirmed = False
        self._detach_requested = False
        self._created_locally = False
        self._start_confirmed = False
        self._client: MinidsClient | None = None
        self._job_id: str = request.job_id

    def cancel(self) -> None:
        """Demande l'arrêt. Les rappels `should_stop` la voient entre deux chunks."""
        self._cancelled = True
        self.log.emit("annulation demandée…")

    def detach(self) -> None:
        """Ferme le suivi sans lancer après coup un job encore local."""
        self._detach_requested = True
        if self.request.job_id or self._start_confirmed:
            self.log.emit("fermeture demandée : le job distant n'est pas annulé ; son ID reste disponible…")
        else:
            self.log.emit("fermeture demandée : arrêt avant tout démarrage GPU…")

    def _stop_requested(self) -> bool:
        # Avant confirmation du démarrage, fermer doit interrompre hash/upload.
        return self._cancelled or (self._detach_requested and not self._start_confirmed)

    def _download_stop_requested(self) -> bool:
        return self._cancelled or self._detach_requested

    # -- exécution ---------------------------------------------------------
    def run(self) -> None:
        started = time.monotonic()
        request = self.request
        outcome = ScanOutcome(job_id="", status="client_error", directory=request.output_dir)
        workdir = Path(tempfile.mkdtemp(prefix="minids-gui-"))

        try:
            self._validate_request(request)
            self._client = MinidsClient(url=request.url, token=request.token)

            if request.job_id:
                self.log.emit(f"reprise du job {request.job_id}")
                self.job_created.emit(request.job_id)
            else:
                payload = self._prepare(request, workdir, outcome)
                self._upload(payload, outcome)
            state = self._wait(outcome)

            outcome.job_id = self._job_id
            outcome.status = state.get("status", "?")
            if outcome.status != "done":
                outcome.error = state.get("error") or f"job {outcome.status}"
                outcome.total_seconds = time.monotonic() - started
                self.failed.emit(outcome)
                return

            self._download(outcome)
            outcome.total_seconds = time.monotonic() - started
            self.finished_ok.emit(outcome)

        except MinidsError as exc:
            self._emit_failure(outcome, exc, started)
        except Exception as exc:  # noqa: BLE001 - tout doit remonter à l'interface
            self._emit_failure(outcome, exc, started, unexpected=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _validate_request(self, request: ScanRequest) -> None:
        if not isinstance(request.params, dict):
            raise MinidsError("paramètres de scan invalides")
        if request.job_id:
            validate_job_id(request.job_id)
        if not request.job_id and not Path(request.source).is_file() and not Path(request.source).is_dir():
            raise MinidsError(f"source introuvable : {request.source}")
        if (
            not isinstance(request.chunk_size, int)
            or isinstance(request.chunk_size, bool)
            or not 64 * 1024 <= request.chunk_size <= MAX_CHUNK
        ):
            raise MinidsError("taille de chunk invalide (64 Kio à 64 Mio)")
        if (
            not isinstance(request.poll_seconds, (int, float))
            or not math.isfinite(float(request.poll_seconds))
            or request.poll_seconds <= 0
        ):
            raise MinidsError("intervalle de polling invalide")
        if (
            not isinstance(request.long_side, int)
            or isinstance(request.long_side, bool)
            or request.long_side <= 0
            or request.long_side % 2
        ):
            raise MinidsError("résolution d'envoi invalide")

    def _emit_failure(
        self,
        outcome: ScanOutcome,
        exc: Exception,
        started: float,
        unexpected: bool = False,
    ) -> None:
        if self._detach_requested and not self._cancelled:
            if self._created_locally and not self._start_confirmed:
                self._try_cancel_remote()
            if self._cancel_confirmed:
                outcome.status = "cancelled"
                outcome.error = "arrêté avant démarrage"
            else:
                outcome.status = "detached"
                outcome.error = "suivi local détaché"
        elif self._cancelled and outcome.status == "done":
            outcome.status = "download_interrupted"
            outcome.error = "téléchargement local interrompu ; le job terminé peut être rejoint pour reprendre"
        elif self._cancelled and (not self._job_id or self._cancel_confirmed):
            outcome.status = "cancelled"
            outcome.error = "annulé"
        else:
            if self._cancelled:
                self._try_cancel_remote()
            if self._cancelled and self._cancel_confirmed:
                outcome.status = "cancelled"
                outcome.error = "annulé"
            else:
                outcome.status = "client_error"
                detail = f"{type(exc).__name__}: {exc}" if unexpected else str(exc)
                outcome.error = f"annulation non confirmée : {detail}" if self._cancelled else detail
        outcome.total_seconds = time.monotonic() - started
        outcome.job_id = self._job_id
        self.failed.emit(outcome)

    def _try_cancel_remote(self) -> None:
        if self._cancel_sent or not self._job_id or self._client is None:
            return
        try:
            response = self._client.cancel(self._job_id)
        except MinidsError as exc:
            self.log.emit(f"annulation distante non confirmée : {exc}")
            return
        self._cancel_sent = True
        self._cancel_confirmed = response.get("status") == "cancelled"

    def _client_or_error(self) -> MinidsClient:
        if self._client is None:
            raise MinidsError("client HTTP non initialisé")
        return self._client

    # -- étapes -------------------------------------------------------------
    def _prepare(self, request: ScanRequest, workdir: Path, outcome: ScanOutcome) -> Path:
        self.phase_changed.emit("extraction")
        self.log.emit(f"source : {request.source.name}")
        step = time.monotonic()
        payload = build_payload(
            source=request.source,
            workdir=workdir,
            frames=int(request.params.get("frames") or 120),
            long_side=request.long_side,
            send_video=request.send_video,
            log=self.log.emit,
            should_stop=self._stop_requested,
        )
        outcome.extract_seconds = time.monotonic() - step
        outcome.payload_bytes = payload.stat().st_size
        if self._stop_requested():
            raise MinidsError("arrêté avant l'envoi")
        return payload

    def _upload(self, payload: Path, outcome: ScanOutcome) -> None:
        client = self._client_or_error()
        self.phase_changed.emit("upload")
        self.log.emit("empreinte sha256 du paquet…")
        digest = sha256_file(payload, should_stop=self._stop_requested)

        self._job_id = client.create_job(
            payload.name, payload.stat().st_size, self.request.chunk_size, digest, self.request.params
        )
        self._created_locally = True
        self.job_created.emit(self._job_id)
        self.log.emit(f"job {self._job_id}")

        step = time.monotonic()

        def on_progress(done: int, total: int) -> None:
            elapsed = max(1e-6, time.monotonic() - step)
            self.transfer_progress.emit("upload", done, total, done / elapsed)

        client.upload(
            self._job_id,
            payload,
            self.request.chunk_size,
            quiet=True,
            on_progress=on_progress,
            should_stop=self._stop_requested,
        )
        outcome.upload_seconds = max(1e-9, time.monotonic() - step)

        if self._detach_requested:
            raise MinidsError("fermeture demandée avant le démarrage")
        response = client.start(self._job_id)
        self._start_confirmed = response.get("status") in {"starting", "queued", "running", "done"}
        self.log.emit(f"lancé (file d'attente : {response.get('queue_position', 0)})")

    def _wait(self, outcome: ScanOutcome) -> dict[str, Any]:
        client = self._client_or_error()
        self.phase_changed.emit("attente")
        step = time.monotonic()
        previous_logs: list[Any] = []
        log_cursor: int | None = None

        while True:
            if self._detach_requested and not self._cancelled:
                raise MinidsError("suivi local détaché")
            if self._cancelled and not self._cancel_sent:
                # Une seule fois : la boucle tourne toutes les 3 s, et le serveur
                # a déjà noté la demande. On continue à interroger même si l'appel
                # échoue — c'est le pod qui confirme l'état final.
                self.log.emit("annulation du job côté pod…")
                self._try_cancel_remote()

            state = client.status(self._job_id)
            logs = state.get("logs", [])
            if not isinstance(logs, list):
                raise MinidsError("état du job : journal invalide")
            logs_offset = state.get("logs_offset")
            for line in new_log_lines(
                previous_logs,
                logs,
                previous_end=log_cursor,
                current_offset=logs_offset if isinstance(logs_offset, int) else None,
            ):
                self.log.emit(str(line))
            log_cursor = (
                logs_offset + len(logs) if isinstance(logs_offset, int) and not isinstance(logs_offset, bool) else None
            )
            previous_logs = list(logs)
            self.server_state.emit(state)

            status = state.get("status")
            if status not in {"created", "starting", "queued", "running", "done", "failed", "cancelled"}:
                raise MinidsError(f"état du job : statut inconnu {status!r}")
            if status == "created" and self.request.job_id:
                raise MinidsError(
                    "upload incomplet : ce job créé hors de cette session ne peut pas être repris sans sa source"
                )
            if status in {"done", "failed", "cancelled"}:
                if status == "cancelled":
                    self._cancel_confirmed = True
                outcome.server_seconds = time.monotonic() - step
                return state
            self.msleep(max(1, int(self.request.poll_seconds * 1000)))

    def _download(self, outcome: ScanOutcome) -> None:
        client = self._client_or_error()
        self.phase_changed.emit("téléchargement")
        destination = job_destination(self.request.output_dir, self._job_id)
        destination.mkdir(parents=True, exist_ok=True)
        outcome.directory = destination

        available: dict[str, dict[str, Any]] = {}
        for item in client.artifacts(self._job_id):
            name = item.get("name")
            if isinstance(name, str) and name:
                # Valide dès la liste : aucune métadonnée distante ne peut faire
                # sortir l'écriture du dossier du job.
                artifact_destination(destination, name)
                available[name] = item
        missing = [name for name in ("mesh.glb", "report.json") if name not in available]
        if missing:
            raise MinidsError(f"job terminé sans artefact requis : {', '.join(missing)}")
        if self.request.fetch_all:
            wanted = list(available)
        else:
            wanted = [name for name in DEFAULT_ARTIFACTS if name in available]
            if self.request.fetch_raw and "vggt_raw.npz" in available:
                wanted.append("vggt_raw.npz")

        step = time.monotonic()
        for name in wanted:
            entry = available[name]

            # `started` et `label` sont liés en arguments par défaut : sans cela,
            # la fermeture lirait la valeur de l'itération courante de la boucle.
            def on_progress(done: int, total: int, label: str = name, started: float = time.monotonic()) -> None:
                elapsed = max(1e-6, time.monotonic() - started)
                self.transfer_progress.emit(f"↓ {label}", done, total, done / elapsed)

            downloaded = client.download(
                self._job_id,
                name,
                artifact_destination(destination, name),
                quiet=True,
                expected_sha256=entry.get("sha256"),
                on_progress=on_progress,
                should_stop=self._download_stop_requested,
            )
            actual_size = downloaded.stat().st_size
            outcome.download_bytes += actual_size
            self.log.emit(f"récupéré {name} ({actual_size / 1e6:.2f} Mo)")

        outcome.download_seconds = time.monotonic() - step
        outcome.artifacts = [available[name] for name in wanted]
        outcome.report = _read_report(destination / "report.json")


def _read_report(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
