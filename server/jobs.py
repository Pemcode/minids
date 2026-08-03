"""Gestion des jobs : arborescence disque, file d'attente mono-worker, progression.

Un seul job tourne à la fois : le GPU du pod ne se partage pas, et la file garde
l'ordre d'arrivée. L'état est persisté dans `state.json` pour survivre à un
redémarrage du conteneur.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
import threading
import time
import uuid
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from queue import Queue
from typing import Any

from .config import Settings

log = logging.getLogger("minids.jobs")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
ARCHIVE_SUFFIXES = {".tar", ".tgz", ".tar.gz", ".zip"}

# Étapes du pipeline avec leur poids relatif, pour une ETA honnête.
STAGES: list[tuple[str, float]] = [
    ("ingest", 0.02),
    ("frames", 0.03),
    ("vggt", 0.08),
    ("segment", 0.07),
    ("colmap", 0.03),
    ("refine", 0.55),
    ("mesh", 0.10),
    ("cleanup", 0.04),
    ("texture", 0.06),
    ("export", 0.02),
]
STAGE_NAMES = [name for name, _ in STAGES]


class JobCancelled(RuntimeError):
    """Levée depuis le pipeline quand l'utilisateur annule."""


@dataclass
class Reporter:
    """Passé au pipeline pour remonter progression et logs."""

    job: Job
    _stage: str = ""
    _stage_start: float = 0.0

    def stage(self, name: str) -> None:
        if name not in STAGE_NAMES:
            raise ValueError(f"étape inconnue: {name}")
        self._stage = name
        self._stage_start = time.time()
        self.job.set_stage(name)
        self.log(f"→ {name}")

    def progress(self, fraction: float) -> None:
        self.job.set_stage_progress(max(0.0, min(1.0, fraction)))

    def log(self, message: str) -> None:
        log.info("[%s] %s", self.job.job_id[:8], message)
        self.job.append_log(message)

    def check_cancelled(self) -> None:
        if self.job.cancel_requested:
            raise JobCancelled("annulé par l'utilisateur")

    def timings(self) -> dict[str, float]:
        return dict(self.job.stage_timings)


@dataclass
class Job:
    job_id: str
    root: Path
    params: dict[str, Any]
    upload: dict[str, Any]
    status: str = "created"  # created|queued|running|done|failed|cancelled
    stage: str = ""
    stage_progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: bool = False
    stage_timings: dict[str, float] = field(default_factory=dict)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # -- arborescence --------------------------------------------------
    @property
    def upload_dir(self) -> Path:
        return self.root / "upload"

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    @property
    def frames_dir(self) -> Path:
        return self.input_dir / "frames"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    def mkdirs(self) -> None:
        for d in (self.upload_dir, self.input_dir, self.frames_dir, self.work_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- progression ---------------------------------------------------
    def set_stage(self, name: str) -> None:
        with self._lock:
            if self.stage and self.started_at is not None:
                pass
            self.stage = name
            self.stage_progress = 0.0
            self.stage_timings.setdefault(name, 0.0)
            self._stage_started = time.time()
        self.persist()

    def set_stage_progress(self, fraction: float) -> None:
        with self._lock:
            self.stage_progress = fraction
            if self.stage:
                started = getattr(self, "_stage_started", None)
                if started:
                    self.stage_timings[self.stage] = time.time() - started

    def append_log(self, message: str) -> None:
        with self._lock:
            self.logs.append(f"{time.strftime('%H:%M:%S')} {message}")

    def overall_progress(self) -> float:
        done = 0.0
        for name, weight in STAGES:
            if name == self.stage:
                return min(1.0, done + weight * self.stage_progress)
            done += weight
            if name not in self.stage_timings:
                # étape jamais atteinte : on s'arrête là
                return min(1.0, done - weight)
        return 1.0 if self.status == "done" else min(0.99, done)

    def eta_seconds(self) -> float | None:
        if self.status != "running" or self.started_at is None:
            return None
        progress = self.overall_progress()
        if progress <= 0.01:
            return None
        elapsed = time.time() - self.started_at
        return max(0.0, elapsed / progress - elapsed)

    # -- sérialisation --------------------------------------------------
    def to_dict(self, include_logs: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "stage_progress": round(self.stage_progress, 4),
            "progress": round(self.overall_progress(), 4),
            "eta_seconds": self.eta_seconds(),
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "params": self.params,
            "upload": {
                "filename": self.upload.get("filename"),
                "size": self.upload.get("size"),
                "chunk_size": self.upload.get("chunk_size"),
                "received_chunks": len(self.upload.get("received", [])),
                "total_chunks": self.upload.get("total_chunks"),
            },
            "stage_timings": {k: round(v, 2) for k, v in self.stage_timings.items()},
            "artifacts": [p.name for p in sorted(self.artifacts_dir.glob("*")) if p.is_file()],
        }
        if include_logs:
            data["logs"] = list(self.logs)[-40:]
        return data

    def persist(self) -> None:
        payload = self.to_dict()
        payload["upload_full"] = self.upload
        tmp = self.root / "state.json.tmp"
        try:
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self.root / "state.json")
        except OSError as exc:  # pragma: no cover - disque plein / job supprimé
            log.warning("persist échoué pour %s: %s", self.job_id, exc)


class JobStore:
    """Registre en mémoire + worker unique."""

    def __init__(self, settings: Settings, runner: Callable[[Job, Reporter], None]) -> None:
        self.settings = settings
        self.runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: Queue[str] = Queue()
        self._worker = threading.Thread(target=self._work_loop, name="minids-worker", daemon=True)
        self._worker.start()

    # -- cycle de vie ---------------------------------------------------
    def create(self, params: dict[str, Any], upload: dict[str, Any]) -> Job:
        job_id = uuid.uuid4().hex
        root = self.settings.jobs_dir / job_id
        job = Job(job_id=job_id, root=root, params=params, upload=upload)
        job.mkdirs()
        job.append_log("job créé")
        job.persist()
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def enqueue(self, job: Job) -> None:
        job.status = "queued"
        job.persist()
        self._queue.put(job.job_id)

    def cancel(self, job: Job) -> None:
        job.cancel_requested = True
        if job.status in {"created", "queued"}:
            job.status = "cancelled"
            job.finished_at = time.time()
        job.append_log("annulation demandée")
        job.persist()

    def delete(self, job: Job) -> None:
        with self._lock:
            self._jobs.pop(job.job_id, None)
        shutil.rmtree(job.root, ignore_errors=True)

    def queue_position(self, job: Job) -> int:
        return max(0, self._queue.qsize())

    # -- worker ----------------------------------------------------------
    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None or job.cancel_requested:
                self._queue.task_done()
                continue
            reporter = Reporter(job=job)
            job.status = "running"
            job.started_at = time.time()
            job.persist()
            try:
                self.runner(job, reporter)
                job.status = "done"
                job.append_log("terminé")
            except JobCancelled:
                job.status = "cancelled"
                job.append_log("annulé")
            except Exception as exc:  # noqa: BLE001 - on veut tout remonter au client
                log.exception("job %s en échec", job_id)
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.append_log(f"ERREUR {job.error}")
            finally:
                job.finished_at = time.time()
                job.persist()
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Assemblage de l'upload
# ---------------------------------------------------------------------------

def assemble_upload(job: Job) -> Path:
    """Recolle les chunks reçus en un fichier unique dans `input/`."""
    filename = job.upload["filename"]
    total = job.upload["total_chunks"]
    received = set(job.upload.get("received", []))
    missing = sorted(set(range(total)) - received)
    if missing:
        raise ValueError(f"chunks manquants: {missing[:10]}{'…' if len(missing) > 10 else ''}")

    target = job.input_dir / Path(filename).name
    with target.open("wb") as out:
        for index in range(total):
            chunk = job.upload_dir / f"{index:06d}.part"
            with chunk.open("rb") as handle:
                shutil.copyfileobj(handle, out, length=1024 * 1024)
    shutil.rmtree(job.upload_dir, ignore_errors=True)
    job.upload_dir.mkdir(parents=True, exist_ok=True)
    return target


def unpack_input(path: Path, frames_dir: Path) -> tuple[str, Path]:
    """Retourne ('video', chemin) ou ('frames', dossier) selon ce qui a été uploadé."""
    suffixes = "".join(path.suffixes[-2:]).lower()
    suffix = path.suffix.lower()

    if suffix in VIDEO_SUFFIXES:
        return "video", path

    frames_dir.mkdir(parents=True, exist_ok=True)
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            _safe_extract_zip(archive, frames_dir)
    elif suffix in {".tar", ".tgz"} or suffixes in {".tar.gz"}:
        with tarfile.open(path) as archive:
            _safe_extract_tar(archive, frames_dir)
    else:
        raise ValueError(f"format d'entrée non supporté: {path.name}")

    # Aplatit un éventuel dossier racine dans l'archive.
    entries = [p for p in frames_dir.rglob("*") if p.is_file()]
    images = [p for p in entries if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        raise ValueError("archive sans image exploitable")
    for image in images:
        if image.parent != frames_dir:
            destination = frames_dir / image.name
            if not destination.exists():
                image.replace(destination)
    for directory in sorted((p for p in frames_dir.iterdir() if p.is_dir()), reverse=True):
        shutil.rmtree(directory, ignore_errors=True)
    return "frames", frames_dir


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    for member in archive.infolist():
        if member.is_dir():
            continue
        out = destination / Path(member.filename).name
        if not _is_within(destination, out):
            raise ValueError(f"chemin d'archive suspect: {member.filename}")
        with archive.open(member) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _safe_extract_tar(archive: tarfile.TarFile, destination: Path) -> None:
    for member in archive.getmembers():
        if not member.isfile():
            continue
        out = destination / Path(member.name).name
        if not _is_within(destination, out):
            raise ValueError(f"chemin d'archive suspect: {member.name}")
        extracted = archive.extractfile(member)
        if extracted is None:
            continue
        with extracted as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
