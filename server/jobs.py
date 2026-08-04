"""Gestion des jobs : arborescence disque, file d'attente mono-worker, progression.

Un seul job tourne à la fois : le GPU du pod ne se partage pas, et la file garde
l'ordre d'arrivée. L'état est persisté dans `state.json` pour survivre à un
redémarrage du conteneur.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import tarfile
import threading
import time
import uuid
import zipfile
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from queue import Queue
from typing import Any

from .config import MAX_CHUNK_SIZE, MAX_UPLOAD_BYTES, MIN_CHUNK_SIZE, Settings

log = logging.getLogger("minids.jobs")

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
ARCHIVE_SUFFIXES = {".tar", ".tgz", ".tar.gz", ".zip"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
UPLOAD_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
JOB_STATUSES = {"created", "starting", "queued", "running", "done", "failed", "cancelled"}
MAX_ARCHIVE_FILES = 10_000
DEFAULT_MAX_EXTRACTED_BYTES = MAX_UPLOAD_BYTES

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
    status: str = "created"  # created|starting|queued|running|done|failed|cancelled
    stage: str = ""
    stage_progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    cancel_requested: bool = False
    stage_timings: dict[str, float] = field(default_factory=dict)
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=200))
    log_count: int = 0
    _lock: Any = field(default_factory=threading.RLock, repr=False)

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
            self.log_count += 1
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
        with self._lock:
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
                "log_count": self.log_count,
                "artifacts": [
                    path.name
                    for path in sorted(self.artifacts_dir.glob("*"))
                    if path.is_file() and not path.name.endswith(".sha256")
                ],
            }
            if include_logs:
                visible_logs = list(self.logs)[-40:]
                data["logs"] = visible_logs
                data["logs_offset"] = max(0, self.log_count - len(visible_logs))
            return data

    def persist(self) -> None:
        with self._lock:
            payload = self.to_dict()
            payload["upload_full"] = self.upload
            payload["cancel_requested"] = self.cancel_requested
            tmp = self.root / "state.json.tmp"
            try:
                tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                tmp.replace(self.root / "state.json")
            except OSError as exc:  # pragma: no cover - disque plein / job supprimé
                log.warning("persist échoué pour %s: %s", self.job_id, exc)


class JobStore:
    """Registre persistant en mémoire + worker unique arrêtables proprement."""

    def __init__(self, settings: Settings, runner: Callable[[Job, Reporter], None]) -> None:
        self.settings = settings
        self.runner = runner
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._queue: Queue[str | None] = Queue()
        self._worker: threading.Thread | None = None
        self._closed = False
        self.settings.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._restore()

    # -- cycle de vie ---------------------------------------------------
    def _restore(self) -> None:
        for state_path in sorted(self.settings.jobs_dir.glob("*/state.json")):
            try:
                job = self._load_job(state_path)
                if job.status in {"starting", "queued", "running"}:
                    job.status = "failed"
                    job.error = "serveur redémarré pendant le traitement"
                    job.cancel_requested = False
                    job.finished_at = time.time()
                    job.append_log("ERREUR traitement interrompu par le redémarrage du serveur")
                    job.persist()
                elif job.status == "created":
                    self._reconcile_chunks(job)
                self._jobs[job.job_id] = job
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                log.warning("état de job ignoré (%s): %s", state_path, exc)

    def _load_job(self, state_path: Path) -> Job:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("state.json n'est pas un objet")

        root = state_path.parent
        job_id = payload.get("job_id")
        if job_id != root.name or not isinstance(job_id, str) or not JOB_ID_RE.fullmatch(job_id):
            raise ValueError("identifiant de job invalide")
        status = payload.get("status")
        if status not in JOB_STATUSES:
            raise ValueError(f"statut de job invalide: {status!r}")
        params = payload.get("params")
        upload = payload.get("upload_full")
        if not isinstance(params, dict) or not isinstance(upload, dict):
            raise ValueError("paramètres ou upload persistés invalides")
        for key in ("filename", "size", "chunk_size", "total_chunks"):
            if key not in upload:
                raise ValueError(f"champ d'upload absent: {key}")
        filename = upload["filename"]
        size = upload["size"]
        chunk_size = upload["chunk_size"]
        total_chunks = upload["total_chunks"]
        json.dumps(params, allow_nan=False)
        json.dumps(upload, allow_nan=False)
        if (
            not isinstance(filename, str)
            or PurePosixPath(filename.replace("\\", "/")).name != filename
            or len(filename.encode("utf-8")) > 255
        ):
            raise ValueError("nom d'upload persisté invalide")
        if (
            type(size) is not int
            or type(chunk_size) is not int
            or type(total_chunks) is not int
            or size <= 0
            or chunk_size <= 0
            or total_chunks != (size + chunk_size - 1) // chunk_size
        ):
            raise ValueError("métadonnées d'upload persistées incohérentes")
        if status == "created" and (
            size > self.settings.max_upload_bytes or not MIN_CHUNK_SIZE <= chunk_size <= MAX_CHUNK_SIZE
        ):
            raise ValueError("upload créé hors des limites serveur")
        received = upload.get("received", [])
        if not isinstance(received, list) or any(
            type(index) is not int or not 0 <= index < total_chunks for index in received
        ):
            raise ValueError("liste de chunks persistée invalide")
        upload["received"] = sorted(set(received))
        expected_sha = upload.get("sha256")
        if expected_sha is not None:
            if not isinstance(expected_sha, str) or not UPLOAD_SHA256_RE.fullmatch(expected_sha):
                raise ValueError("sha256 persisté invalide")
            upload["sha256"] = expected_sha.lower()

        def finite_float(name: str, default: float | None = None) -> float | None:
            value = payload.get(name, default)
            if value is None:
                return None
            if type(value) not in {int, float}:
                raise ValueError(f"{name} persisté invalide")
            converted = float(value)
            if not math.isfinite(converted):
                raise ValueError(f"{name} persisté non fini")
            return converted

        logs = payload.get("logs", [])
        timings = payload.get("stage_timings", {})
        if (
            not isinstance(logs, list)
            or any(not isinstance(line, str) for line in logs)
            or not isinstance(timings, dict)
        ):
            raise ValueError("journaux ou timings persistés invalides")
        log_count = payload.get("log_count", len(logs))
        if type(log_count) is not int or log_count < len(logs):
            raise ValueError("compteur de logs persisté invalide")
        stage = payload.get("stage", "")
        if not isinstance(stage, str) or stage not in {"", *STAGE_NAMES}:
            raise ValueError("étape persistée invalide")
        stage_progress = finite_float("stage_progress", 0.0)
        if stage_progress is None or not 0.0 <= stage_progress <= 1.0:
            raise ValueError("progression persistée hors limites")
        normalized_timings: dict[str, float] = {}
        for name, seconds in timings.items():
            if (
                name not in STAGE_NAMES
                or type(seconds) not in {int, float}
                or not math.isfinite(float(seconds))
                or float(seconds) < 0
            ):
                raise ValueError("timing persisté invalide")
            normalized_timings[name] = float(seconds)

        created_at = finite_float("created_at", state_path.stat().st_mtime)
        started_at = finite_float("started_at")
        finished_at = finite_float("finished_at")
        if (
            created_at is None
            or created_at < 0
            or started_at is not None
            and started_at < 0
            or finished_at is not None
            and finished_at < 0
        ):
            raise ValueError("timestamp persisté invalide")
        cancel_requested = payload.get("cancel_requested", False)
        if type(cancel_requested) is not bool:
            raise ValueError("indicateur d'annulation persisté invalide")
        error = payload.get("error")
        if error is not None and not isinstance(error, str):
            raise ValueError("erreur persistée invalide")

        job = Job(
            job_id=job_id,
            root=root,
            params=params,
            upload=upload,
            status=status,
            stage=stage,
            stage_progress=stage_progress,
            error=error,
            created_at=created_at,
            started_at=started_at,
            finished_at=finished_at,
            cancel_requested=cancel_requested,
            stage_timings=normalized_timings,
            logs=deque((line for line in logs if isinstance(line, str)), maxlen=200),
            log_count=log_count,
        )
        job.mkdirs()
        return job

    @staticmethod
    def _reconcile_chunks(job: Job) -> None:
        total = job.upload.get("total_chunks")
        if type(total) is not int or total <= 0:
            job.upload["received"] = []
            job.persist()
            return

        actual = []
        for index in range(total):
            path = job.upload_dir / f"{index:06d}.part"
            if path.is_file() and path.stat().st_size == expected_chunk_size(job, index):
                actual.append(index)
        job.upload["received"] = actual
        job.persist()

    def create(self, params: dict[str, Any], upload: dict[str, Any]) -> Job:
        with self._lock:
            if self._closed:
                raise RuntimeError("registre de jobs fermé")
            job_id = uuid.uuid4().hex
            root = self.settings.jobs_dir / job_id
            job = Job(job_id=job_id, root=root, params=params, upload=upload)
            job.mkdirs()
            job.append_log("job créé")
            job.persist()
            self._jobs[job_id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def enqueue(self, job: Job) -> None:
        with job._lock:
            if job.status != "starting":
                raise ValueError(f"job non démarrable (statut {job.status})")
            with self._lock:
                if self._closed:
                    raise RuntimeError("registre de jobs fermé")
                if self._worker is None:
                    self._worker = threading.Thread(target=self._work_loop, name="minids-worker", daemon=True)
                    self._worker.start()
                job.status = "queued"
                job.persist()
                self._queue.put(job.job_id)

    def cancel(self, job: Job) -> bool:
        with job._lock:
            with self._lock:
                if self._jobs.get(job.job_id) is not job:
                    return False
            if job.status in {"done", "failed", "cancelled"}:
                return True
            job.cancel_requested = True
            if job.status in {"created", "queued"}:
                job.status = "cancelled"
                job.finished_at = time.time()
            job.append_log("annulation demandée")
            job.persist()
            return True

    def delete(self, job: Job) -> bool:
        with job._lock:
            if job.status in {"starting", "running"}:
                return False
            job.cancel_requested = True
            with self._lock:
                if self._jobs.get(job.job_id) is not job:
                    return False
                self._jobs.pop(job.job_id)
            shutil.rmtree(job.root, ignore_errors=True)
            return True

    def queue_position(self, _job: Job) -> int:
        return max(0, self._queue.qsize())

    @property
    def worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            jobs = list(self._jobs.values())
            worker = self._worker
        for job in jobs:
            if job.status in {"starting", "queued", "running"}:
                self.cancel(job)
        if worker is None:
            return
        self._queue.put(None)
        if threading.current_thread() is not worker:
            worker.join(timeout=timeout)
        if worker.is_alive():  # pragma: no cover - runner tiers non coopératif
            log.warning("le worker ne s'est pas arrêté dans le délai de %.1f s", timeout)

    # -- worker ----------------------------------------------------------
    def _work_loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                if job_id is None:
                    return
                job = self.get(job_id)
                if job is None:
                    continue
                with job._lock:
                    if job.status != "queued" or job.cancel_requested:
                        continue
                    reporter = Reporter(job=job)
                    job.status = "running"
                    job.started_at = time.time()
                    job.persist()
                try:
                    self.runner(job, reporter)
                    with job._lock:
                        if job.cancel_requested:
                            job.status = "cancelled"
                            job.append_log("annulé")
                        else:
                            job.status = "done"
                            job.append_log("terminé")
                except JobCancelled:
                    with job._lock:
                        job.status = "cancelled"
                        job.append_log("annulé")
                except Exception as exc:  # noqa: BLE001 - on veut tout remonter au client
                    log.exception("job %s en échec", job_id)
                    with job._lock:
                        job.status = "failed"
                        job.error = f"{type(exc).__name__}: {exc}"
                        job.append_log(f"ERREUR {job.error}")
                finally:
                    with job._lock:
                        job.finished_at = time.time()
                        job.persist()
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Assemblage de l'upload
# ---------------------------------------------------------------------------


def expected_chunk_size(job: Job, index: int) -> int:
    size = job.upload.get("size")
    chunk_size = job.upload.get("chunk_size")
    total = job.upload.get("total_chunks")
    if type(size) is not int or type(chunk_size) is not int or type(total) is not int:
        raise ValueError("métadonnées d'upload invalides")
    if size <= 0 or chunk_size <= 0 or total != (size + chunk_size - 1) // chunk_size:
        raise ValueError("métadonnées d'upload incohérentes")
    if not 0 <= index < total:
        raise ValueError(f"index de chunk hors bornes: {index}")
    return chunk_size if index < total - 1 else size - chunk_size * (total - 1)


def assemble_upload(job: Job) -> Path:
    """Recolle atomiquement les chunks après validation de leurs tailles exactes."""
    filename = job.upload.get("filename")
    if not isinstance(filename, str):
        raise ValueError("nom de fichier d'upload invalide")
    normalized = filename.replace("\\", "/")
    basename = PurePosixPath(normalized).name
    if basename != filename or basename in {"", ".", ".."} or len(filename.encode("utf-8")) > 255:
        raise ValueError("nom de fichier d'upload invalide")

    total = job.upload.get("total_chunks")
    if type(total) is not int or total <= 0:
        raise ValueError("métadonnées d'upload invalides")
    received = {index for index in job.upload.get("received", []) if type(index) is int}
    missing = sorted(set(range(total)) - received)
    if missing:
        raise ValueError(f"chunks manquants: {missing[:10]}{'…' if len(missing) > 10 else ''}")

    chunks = []
    for index in range(total):
        if job.cancel_requested:
            raise JobCancelled("annulation pendant l'assemblage")
        chunk = job.upload_dir / f"{index:06d}.part"
        expected = expected_chunk_size(job, index)
        if not chunk.is_file() or chunk.stat().st_size != expected:
            actual = chunk.stat().st_size if chunk.is_file() else 0
            raise ValueError(f"chunk {index}: {actual} octets, {expected} attendus")
        chunks.append(chunk)

    target = job.input_dir / basename
    temporary = target.with_name(f"{target.name}.assembling")
    try:
        with temporary.open("wb") as out:
            for chunk in chunks:
                if job.cancel_requested:
                    raise JobCancelled("annulation pendant l'assemblage")
                with chunk.open("rb") as handle:
                    shutil.copyfileobj(handle, out, length=1024 * 1024)
        expected_total = job.upload["size"]
        if temporary.stat().st_size != expected_total:
            raise ValueError(f"upload assemblé: {temporary.stat().st_size} octets, {expected_total} attendus")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    shutil.rmtree(job.upload_dir, ignore_errors=True)
    job.upload_dir.mkdir(parents=True, exist_ok=True)
    return target


def validate_input_archive(
    path: Path,
    max_extracted_bytes: int,
    max_files: int = MAX_ARCHIVE_FILES,
) -> None:
    """Valide format, chemins aplatis, doublons et taille avant mise en file."""
    suffixes = "".join(path.suffixes[-2:]).lower()
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return
    if max_extracted_bytes <= 0 or max_files <= 0:
        raise ValueError("limites d'archive invalides")
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                members = _validated_zip_members(archive, max_extracted_bytes, max_files)
                _validate_zip_images(archive, members)
            return
        if suffix in {".tar", ".tgz"} or suffixes == ".tar.gz":
            with tarfile.open(path) as archive:  # noqa: S202 - aucune extraction native
                members = _validated_tar_members(archive, max_extracted_bytes, max_files)
                _validate_tar_images(archive, members)
            return
    except (OSError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
        raise ValueError(f"archive invalide: {type(exc).__name__}: {exc}") from exc
    raise ValueError(f"format d'entrée non supporté: {path.name}")


def unpack_input(path: Path, frames_dir: Path) -> tuple[str, Path]:
    """Retourne ('video', chemin) ou ('frames', dossier) selon ce qui a été uploadé."""
    suffixes = "".join(path.suffixes[-2:]).lower()
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return "video", path

    shutil.rmtree(frames_dir, ignore_errors=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                _safe_extract_zip(archive, frames_dir)
        elif suffix in {".tar", ".tgz"} or suffixes == ".tar.gz":
            with tarfile.open(path) as archive:  # noqa: S202 - membres copiés manuellement
                _safe_extract_tar(archive, frames_dir)
        else:
            raise ValueError(f"format d'entrée non supporté: {path.name}")
    except ValueError:
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        raise
    except (OSError, tarfile.TarError, zipfile.BadZipFile, RuntimeError) as exc:
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        raise ValueError(f"archive invalide: {type(exc).__name__}: {exc}") from exc

    images = [path for path in frames_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        shutil.rmtree(frames_dir, ignore_errors=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        raise ValueError("archive sans image exploitable")
    return "frames", frames_dir


def _archive_output_name(raw_name: str) -> str:
    normalized = raw_name.replace("\\", "/")
    member_path = PurePosixPath(normalized)
    if (
        not normalized
        or "\x00" in normalized
        or member_path.is_absolute()
        or ".." in member_path.parts
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise ValueError(f"chemin d'archive suspect: {raw_name}")
    name = member_path.name
    if name in {"", ".", ".."}:
        raise ValueError(f"chemin d'archive suspect: {raw_name}")
    if len(name.encode("utf-8")) > 255:
        raise ValueError(f"nom d'archive trop long après aplatissement: {name[:80]}")
    return name


def _validated_flat_members(
    entries: list[tuple[Any, str, int]],
    member_count: int,
    max_extracted_bytes: int,
    max_files: int,
) -> list[tuple[Any, str]]:
    if member_count > max_files or len(entries) > max_files:
        raise ValueError(f"archive avec trop d'entrées (maximum {max_files})")
    total = 0
    seen: set[str] = set()
    validated = []
    for member, raw_name, size in entries:
        if type(size) is not int or size < 0:
            raise ValueError(f"taille d'entrée invalide: {raw_name}")
        total += size
        if total > max_extracted_bytes:
            raise ValueError(f"archive décompressée > {max_extracted_bytes} octets")
        name = _archive_output_name(raw_name)
        key = name.casefold()
        if key in seen:
            raise ValueError(f"nom dupliqué après aplatissement: {name}")
        seen.add(key)
        validated.append((member, name))
    return validated


def _validated_zip_members(
    archive: zipfile.ZipFile, max_extracted_bytes: int, max_files: int
) -> list[tuple[zipfile.ZipInfo, str]]:
    members = archive.infolist()
    files = []
    for member in members:
        if member.is_dir():
            continue
        if member.flag_bits & 0x1:
            raise ValueError(f"entrée ZIP chiffrée non supportée: {member.filename}")
        files.append((member, member.filename, member.file_size))
    return _validated_flat_members(files, len(members), max_extracted_bytes, max_files)


def _validated_tar_members(
    archive: tarfile.TarFile, max_extracted_bytes: int, max_files: int
) -> list[tuple[tarfile.TarInfo, str]]:
    members = archive.getmembers()
    files = [(member, member.name, member.size) for member in members if member.isfile()]
    return _validated_flat_members(files, len(members), max_extracted_bytes, max_files)


def _valid_image_prefix(name: str, prefix: bytes) -> bool:
    suffix = PurePosixPath(name).suffix.lower()
    if suffix == ".png":
        return prefix.startswith(b"\x89PNG\r\n\x1a\n")
    if suffix in {".jpg", ".jpeg"}:
        return prefix.startswith(b"\xff\xd8\xff")
    return False


def _validate_zip_images(archive: zipfile.ZipFile, members: list[tuple[zipfile.ZipInfo, str]]) -> None:
    images = [
        (member, name) for member, name in members if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        raise ValueError("archive sans image exploitable")
    for member, name in images:
        with archive.open(member) as source:
            prefix = source.read(8)
        if member.file_size <= 0 or not _valid_image_prefix(name, prefix):
            raise ValueError(f"image d'archive invalide: {name}")


def _validate_tar_images(archive: tarfile.TarFile, members: list[tuple[tarfile.TarInfo, str]]) -> None:
    images = [
        (member, name) for member, name in members if PurePosixPath(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if not images:
        raise ValueError("archive sans image exploitable")
    for member, name in images:
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"image TAR illisible: {name}")
        with source:
            prefix = source.read(8)
        if member.size <= 0 or not _valid_image_prefix(name, prefix):
            raise ValueError(f"image d'archive invalide: {name}")


def _copy_exact(source: Any, destination: Path, expected_size: int) -> None:
    written = 0
    with destination.open("wb") as target:
        while True:
            block = source.read(min(1024 * 1024, expected_size - written + 1))
            if not block:
                break
            written += len(block)
            if written > expected_size:
                raise ValueError(f"entrée d'archive plus grande qu'annoncée: {destination.name}")
            target.write(block)
    if written != expected_size:
        raise ValueError(f"entrée d'archive tronquée: {destination.name}")


def _safe_extract_zip(
    archive: zipfile.ZipFile,
    destination: Path,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    max_files: int = MAX_ARCHIVE_FILES,
) -> None:
    for member, name in _validated_zip_members(archive, max_extracted_bytes, max_files):
        with archive.open(member) as source:
            _copy_exact(source, destination / name, member.file_size)


def _safe_extract_tar(
    archive: tarfile.TarFile,
    destination: Path,
    max_extracted_bytes: int = DEFAULT_MAX_EXTRACTED_BYTES,
    max_files: int = MAX_ARCHIVE_FILES,
) -> None:
    for member, name in _validated_tar_members(archive, max_extracted_bytes, max_files):
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"entrée TAR illisible: {member.name}")
        with extracted as source:
            _copy_exact(source, destination / name, member.size)
