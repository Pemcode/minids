"""Tâches de fond.

Règle absolue en Qt : le fil d'exécution de l'interface ne doit jamais bloquer.
Un scan dure une quinzaine de minutes et enchaîne ffmpeg, des requêtes réseau et
de l'attente — tout cela vit ici, et ne communique avec l'interface que par
signaux.

`MinidsClient` étant synchrone à base de rappels, il s'intègre naturellement :
on lui passe des fonctions qui réémettent en signaux Qt.
"""

from __future__ import annotations

import contextlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThread, pyqtSignal

from client.payload import build_payload
from client.transport import MinidsClient, MinidsError, sha256_file

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
    transfer_progress = pyqtSignal(str, int, int, float)  # libellé, fait, total, débit
    job_created = pyqtSignal(str)
    server_state = pyqtSignal(dict)
    finished_ok = pyqtSignal(object)  # ScanOutcome
    failed = pyqtSignal(object)  # ScanOutcome

    def __init__(self, request: ScanRequest, parent=None) -> None:
        super().__init__(parent)
        self.request = request
        self._cancelled = False
        self._cancel_sent = False
        self._client: MinidsClient | None = None
        self._job_id: str = request.job_id

    def cancel(self) -> None:
        """Demande l'arrêt. Les rappels `should_stop` la voient entre deux chunks."""
        self._cancelled = True
        self.log.emit("annulation demandée…")

    def _stop_requested(self) -> bool:
        return self._cancelled

    # -- exécution ---------------------------------------------------------
    def run(self) -> None:
        started = time.time()
        request = self.request
        outcome = ScanOutcome(job_id="", status="failed", directory=request.output_dir)
        workdir = Path(tempfile.mkdtemp(prefix="minids-gui-"))

        try:
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
                outcome.total_seconds = time.time() - started
                self.failed.emit(outcome)
                return

            self._download(outcome)
            outcome.total_seconds = time.time() - started
            self.finished_ok.emit(outcome)

        except MinidsError as exc:
            outcome.error = str(exc)
            outcome.total_seconds = time.time() - started
            outcome.job_id = self._job_id
            self.failed.emit(outcome)
        except Exception as exc:  # noqa: BLE001 - tout doit remonter à l'interface
            outcome.error = f"{type(exc).__name__}: {exc}"
            outcome.total_seconds = time.time() - started
            outcome.job_id = self._job_id
            self.failed.emit(outcome)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # -- étapes -------------------------------------------------------------
    def _prepare(self, request: ScanRequest, workdir: Path, outcome: ScanOutcome) -> Path:
        self.phase_changed.emit("extraction")
        self.log.emit(f"source : {request.source.name}")
        step = time.time()
        payload = build_payload(
            source=request.source,
            workdir=workdir,
            frames=int(request.params.get("frames") or 120),
            long_side=request.long_side,
            send_video=request.send_video,
            log=self.log.emit,
        )
        outcome.extract_seconds = time.time() - step
        outcome.payload_bytes = payload.stat().st_size
        if self._cancelled:
            raise MinidsError("annulé avant l'envoi")
        return payload

    def _upload(self, payload: Path, outcome: ScanOutcome) -> None:
        assert self._client is not None
        self.phase_changed.emit("upload")
        self.log.emit("empreinte sha256 du paquet…")
        digest = sha256_file(payload)

        self._job_id = self._client.create_job(
            payload.name, payload.stat().st_size, self.request.chunk_size, digest, self.request.params
        )
        self.job_created.emit(self._job_id)
        self.log.emit(f"job {self._job_id}")

        step = time.time()

        def on_progress(done: int, total: int) -> None:
            elapsed = max(1e-6, time.time() - step)
            self.transfer_progress.emit("upload", done, total, done / elapsed)

        self._client.upload(
            self._job_id, payload, self.request.chunk_size, quiet=True,
            on_progress=on_progress, should_stop=self._stop_requested,
        )
        outcome.upload_seconds = time.time() - step

        response = self._client.start(self._job_id)
        self.log.emit(f"lancé (file d'attente : {response.get('queue_position', 0)})")

    def _wait(self, outcome: ScanOutcome) -> dict[str, Any]:
        assert self._client is not None
        self.phase_changed.emit("attente")
        step = time.time()
        seen = 0

        while True:
            if self._cancelled and not self._cancel_sent:
                # Une seule fois : la boucle tourne toutes les 3 s, et le serveur
                # a déjà noté la demande. On continue à interroger même si l'appel
                # échoue — c'est le pod qui confirme l'état final.
                self._cancel_sent = True
                self.log.emit("annulation du job côté pod…")
                with contextlib.suppress(MinidsError):
                    self._client.cancel(self._job_id)

            state = self._client.status(self._job_id)
            logs = state.get("logs", [])
            for line in logs[seen:]:
                self.log.emit(line)
            seen = len(logs)
            self.server_state.emit(state)

            if state.get("status") in {"done", "failed", "cancelled"}:
                outcome.server_seconds = time.time() - step
                return state
            self.msleep(int(self.request.poll_seconds * 1000))

    def _download(self, outcome: ScanOutcome) -> None:
        assert self._client is not None
        self.phase_changed.emit("téléchargement")
        destination = self.request.output_dir / self._job_id
        destination.mkdir(parents=True, exist_ok=True)
        outcome.directory = destination

        available = {item["name"]: item for item in self._client.artifacts(self._job_id)}
        if self.request.fetch_all:
            wanted = list(available)
        else:
            wanted = [name for name in DEFAULT_ARTIFACTS if name in available]
            if self.request.fetch_raw and "vggt_raw.npz" in available:
                wanted.append("vggt_raw.npz")

        step = time.time()
        for name in wanted:
            entry = available[name]

            # `started` et `label` sont liés en arguments par défaut : sans cela,
            # la fermeture lirait la valeur de l'itération courante de la boucle.
            def on_progress(done: int, total: int, label: str = name, started: float = time.time()) -> None:
                elapsed = max(1e-6, time.time() - started)
                self.transfer_progress.emit(f"↓ {label}", done, total, done / elapsed)

            self._client.download(
                self._job_id, name, destination / name, quiet=True,
                expected_sha256=entry.get("sha256"), on_progress=on_progress,
                should_stop=self._stop_requested,
            )
            outcome.download_bytes += int(entry.get("size", 0))
            self.log.emit(f"récupéré {name} ({entry.get('size', 0) / 1e6:.2f} Mo)")

        outcome.download_seconds = time.time() - step
        outcome.artifacts = [available[name] for name in wanted]
        outcome.report = _read_report(destination / "report.json")


def _read_report(path: Path) -> dict[str, Any]:
    import json

    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
