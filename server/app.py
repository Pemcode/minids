"""API HTTP miniDS.

Conçue pour le proxy RunPod, qui coupe toute connexion à 100 s (Cloudflare) :
aucune requête ne doit être longue. L'upload est découpé en chunks, l'inférence
est asynchrone, le suivi se fait par polling léger et le téléchargement des
artefacts passe par des requêtes `Range` reprises côté client.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import re
import secrets
import time
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator, model_validator

from .config import MAX_CHUNK_SIZE, MIN_CHUNK_SIZE, Settings, get_settings
from .jobs import (
    ARCHIVE_SUFFIXES,
    JOB_ID_RE,
    VIDEO_SUFFIXES,
    Job,
    JobCancelled,
    JobStore,
    Reporter,
    assemble_upload,
    expected_chunk_size,
    validate_input_archive,
)

logging.basicConfig(
    level=os.environ.get("MINIDS_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("minids.api")

VERSION = "0.1.0"
ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RANGE_RE = re.compile(r"^bytes=([0-9]*)-([0-9]*)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
STREAM_BLOCK = 1024 * 1024


class JobParams(BaseModel):
    """Paramètres de reconstruction, tous optionnels côté client."""

    model_config = ConfigDict(extra="forbid")

    prompt: str | None = Field(default=None, max_length=512, description="Prompt texte SAM 3, ex: 'the sneaker'")
    frames: int = Field(default=0, ge=0, le=600, description="0 = valeur par défaut du serveur")
    refine: bool = Field(default=True, description="Raffinement 2DGS avant fusion TSDF")
    gs_iters: int = Field(default=12000, ge=500, le=60000)
    mesh_backends: list[Literal["tsdf2dgs", "tsdf", "poisson"]] = Field(
        default_factory=lambda: ["tsdf2dgs"], min_length=1, max_length=3
    )
    segmentation: Literal["auto", "sam3", "geometric", "none"] = "auto"
    texture: Literal["bake", "vertex"] = "bake"
    texture_size: int = Field(default=2048, ge=512, le=4096)
    target_triangles: int = Field(default=200_000, ge=5_000, le=2_000_000)
    voxel_divisor: int = Field(default=512, ge=64, le=2048)
    ref_size: FiniteFloat | None = Field(default=None, gt=0, description="Plus grande dimension réelle en mètres")
    bundle_adjustment: Literal[False] = False
    watertight: bool = True

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt vide")
        return normalized

    @field_validator("mesh_backends")
    @classmethod
    def deduplicate_mesh_backends(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @model_validator(mode="after")
    def validate_sam3_prompt(self) -> JobParams:
        if self.segmentation == "sam3" and self.prompt is None:
            raise ValueError("un prompt non vide est requis avec segmentation=sam3")
        return self


class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(ge=1)
    chunk_size: int = Field(default=8 * 1024 * 1024, ge=MIN_CHUNK_SIZE, le=MAX_CHUNK_SIZE)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{64}$")
    params: JobParams = Field(default_factory=JobParams)

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        normalized = value.strip()
        portable = normalized.replace("\\", "/")
        if (
            not normalized
            or any(ord(character) < 32 for character in normalized)
            or PurePosixPath(portable).name != normalized
            or len(normalized.encode("utf-8")) > 255
        ):
            raise ValueError("nom de fichier invalide")
        suffixes = "".join(PurePosixPath(normalized).suffixes[-2:]).lower()
        suffix = PurePosixPath(normalized).suffix.lower()
        if suffix not in VIDEO_SUFFIXES and suffix not in ARCHIVE_SUFFIXES and suffixes not in ARCHIVE_SUFFIXES:
            raise ValueError("format de fichier non supporté")
        return normalized

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str | None) -> str | None:
        return value.lower() if value is not None else None


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    from .pipeline.run import run_pipeline  # import tardif : évite de charger torch au boot

    def runner(job: Job, reporter: Reporter) -> None:
        run_pipeline(job, reporter, settings)

    store = JobStore(settings, runner)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            yield
        finally:
            store.close()

    app = FastAPI(title="miniDS", version=VERSION, docs_url="/docs", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store

    def check_token(authorization: str | None) -> bool:
        if not settings.token or authorization is None:
            return False
        return secrets.compare_digest(authorization.strip(), f"Bearer {settings.token}")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if not settings.token:
            # Pas de token configuré : on n'ouvre pas une API publique sans le dire.
            raise HTTPException(status_code=503, detail="MINIDS_TOKEN non configuré sur le serveur")
        if not check_token(authorization):
            raise HTTPException(status_code=401, detail="token invalide")

    auth = Depends(require_auth)

    def get_job(job_id: str) -> Job:
        if not JOB_ID_RE.fullmatch(job_id):
            raise HTTPException(status_code=400, detail="identifiant de job invalide")
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job inconnu")
        return job

    # -- santé ----------------------------------------------------------
    @app.get("/health")
    def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        # Le HEALTHCHECK public ne publie aucune information d'infrastructure.
        # Les diagnostics détaillés sont réservés à un bearer valide.
        if authorization is None:
            return {"status": "ok"}
        if not check_token(authorization):
            raise HTTPException(status_code=401, detail="token invalide")
        info: dict[str, Any] = {
            "status": "ok",
            "version": VERSION,
            "fake_gpu": settings.fake_gpu,
            "chunk_size": settings.chunk_size,
            "default_frames": settings.default_frames,
            "jobs": len(store.list()),
            "auth_configured": bool(settings.token),
            # Seule la présence des prérequis modèles est publiée, jamais leur valeur.
            "hf_token_configured": bool(settings.hf_token),
            "checkpoint_configured": bool(settings.checkpoint),
        }
        try:  # pragma: no cover - dépend du GPU
            import torch

            info["torch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                info["gpu"] = torch.cuda.get_device_name(0)
                info["vram_free_gb"] = round(free / 1e9, 2)
                info["vram_total_gb"] = round(total / 1e9, 2)
        except Exception as exc:  # noqa: BLE001
            info["torch"] = f"indisponible ({type(exc).__name__})"
        return info

    # -- création / upload ----------------------------------------------
    @app.post("/jobs", dependencies=[auth])
    def create_job(request: CreateJobRequest) -> dict[str, Any]:
        if request.size > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail=f"upload > {settings.max_upload_bytes} octets")
        chunk_size = request.chunk_size if "chunk_size" in request.model_fields_set else settings.chunk_size
        total_chunks = (request.size + chunk_size - 1) // chunk_size
        upload = {
            "filename": request.filename,
            "size": request.size,
            "chunk_size": chunk_size,
            "sha256": request.sha256,
            "total_chunks": total_chunks,
            "received": [],
        }
        job = store.create(request.params.model_dump(), upload)
        return {"job_id": job.job_id, "total_chunks": total_chunks, "chunk_size": chunk_size}

    @app.put("/jobs/{job_id}/chunks/{index}", dependencies=[auth])
    async def put_chunk(job_id: str, index: int, request: Request) -> dict[str, Any]:
        job = get_job(job_id)
        with job._lock:
            if job.status != "created" or job.cancel_requested:
                raise HTTPException(status_code=409, detail=f"upload fermé (statut {job.status})")
            total = job.upload["total_chunks"]
            if not 0 <= index < total:
                raise HTTPException(status_code=416, detail=f"index hors bornes (0..{total - 1})")
            try:
                expected = expected_chunk_size(job, index)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        body = await _read_exact_body(request, expected, index)
        with job._lock:
            if job.status != "created" or job.cancel_requested or store.get(job_id) is not job:
                raise HTTPException(status_code=409, detail=f"upload fermé (statut {job.status})")
            target = job.upload_dir / f"{index:06d}.part"
            if target.is_file() and target.stat().st_size == expected:
                if not _same_content(target, body):
                    raise HTTPException(status_code=409, detail=f"chunk {index} déjà reçu avec un contenu différent")
            else:
                tmp = target.with_suffix(".tmp")
                tmp.write_bytes(body)
                tmp.replace(target)
            if index not in job.upload["received"]:
                job.upload["received"].append(index)
                job.upload["received"].sort()
            job.persist()
            return {"received": len(job.upload["received"]), "total": total}

    @app.get("/jobs/{job_id}/chunks", dependencies=[auth])
    def list_chunks(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        with job._lock:
            return {
                "received": sorted(job.upload.get("received", [])),
                "total": job.upload["total_chunks"],
                "chunk_size": job.upload["chunk_size"],
            }

    @app.post("/jobs/{job_id}/start", dependencies=[auth])
    def start_job(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        with job._lock:
            if store.get(job_id) is not job:
                raise HTTPException(status_code=404, detail="job inconnu")
            if job.status in {"starting", "queued", "running", "done"}:
                return {
                    "job_id": job.job_id,
                    "status": job.status,
                    "queue_position": store.queue_position(job),
                }
            if job.status != "created" or job.cancel_requested:
                raise HTTPException(status_code=409, detail=f"job déjà démarré (statut {job.status})")
            job.status = "starting"
            job.error = None
            job.finished_at = None
            job.persist()

        path: Path | None = None
        try:
            path = assemble_upload(job)
            _raise_if_cancelled(job)
            expected_sha = job.upload.get("sha256")
            if expected_sha and _sha256(path, job) != expected_sha:
                raise ValueError("sha256 de l'upload incorrect")
            _raise_if_cancelled(job)
            validate_input_archive(path, settings.max_upload_bytes)
            _raise_if_cancelled(job)
            assembled_size = path.stat().st_size

            with job._lock:
                _raise_if_cancelled(job)
                job.upload["assembled_path"] = str(path)
                job.append_log(f"upload assemblé: {path.name} ({assembled_size} octets)")
                store.enqueue(job)
                return {"job_id": job.job_id, "status": job.status, "queue_position": store.queue_position(job)}
        except JobCancelled:
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            with job._lock:
                job.status = "cancelled"
                job.finished_at = time.time()
                job.persist()
            return {"job_id": job.job_id, "status": "cancelled", "queue_position": 0}
        except ValueError as exc:
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            with job._lock:
                if path is None:
                    job.status = "created"
                else:
                    job.status = "failed"
                    job.error = str(exc)
                    job.finished_at = time.time()
                    job.append_log(f"ERREUR {exc}")
                job.persist()
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            with job._lock:
                job.status = "failed"
                job.error = str(exc)
                job.finished_at = time.time()
                job.persist()
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except OSError as exc:
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            with job._lock:
                job.status = "failed"
                job.error = f"erreur disque pendant la préparation: {exc}"
                job.finished_at = time.time()
                job.persist()
            raise HTTPException(status_code=500, detail="erreur disque pendant la préparation") from exc
        except Exception as exc:
            log.exception("préparation inattendue en échec pour %s", job.job_id)
            if path is not None:
                with contextlib.suppress(OSError):
                    path.unlink(missing_ok=True)
            with job._lock:
                job.status = "cancelled" if job.cancel_requested else "failed"
                job.error = None if job.cancel_requested else f"{type(exc).__name__}: {exc}"
                job.finished_at = time.time()
                job.persist()
            raise

    # -- suivi -------------------------------------------------------------
    @app.get("/jobs/{job_id}", dependencies=[auth])
    def job_status(job_id: str, logs: bool = True) -> dict[str, Any]:
        return get_job(job_id).to_dict(include_logs=logs)

    @app.get("/jobs", dependencies=[auth])
    def list_jobs() -> dict[str, Any]:
        return {"jobs": [job.to_dict(include_logs=False) for job in store.list()]}

    @app.post("/jobs/{job_id}/cancel", dependencies=[auth])
    def cancel_job(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        if not store.cancel(job):
            raise HTTPException(status_code=404, detail="job inconnu")
        with job._lock:
            return {"job_id": job.job_id, "status": job.status}

    @app.delete("/jobs/{job_id}", dependencies=[auth])
    def delete_job(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        if not store.delete(job):
            if store.get(job_id) is None:
                raise HTTPException(status_code=404, detail="job inconnu")
            raise HTTPException(
                status_code=409,
                detail="job en cours d'exécution; annulez-le puis attendez son arrêt avant suppression",
            )
        return {"deleted": job_id}

    # -- artefacts ----------------------------------------------------------
    @app.get("/jobs/{job_id}/artifacts", dependencies=[auth])
    def list_artifacts(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        items = []
        with job._lock:
            for path in sorted(job.artifacts_dir.glob("*")):
                if path.is_file() and not path.is_symlink() and not path.name.endswith(".sha256"):
                    items.append({"name": path.name, "size": path.stat().st_size, "sha256": _cached_sha256(path)})
        return {"job_id": job_id, "artifacts": items}

    @app.get("/jobs/{job_id}/artifacts/{name}", dependencies=[auth])
    def get_artifact(job_id: str, name: str, request: Request) -> Response:
        job = get_job(job_id)
        path = _artifact_path(job, name)
        size = path.stat().st_size
        headers = {
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'attachment; filename="{path.name}"',
            "X-Content-SHA256": _cached_sha256(path),
            "X-Content-Length": str(size),
        }

        range_header = request.headers.get("range")
        if not range_header:
            headers["Content-Length"] = str(size)
            return StreamingResponse(
                _iter_range(path, 0, size - 1), media_type="application/octet-stream", headers=headers
            )

        match = RANGE_RE.match(range_header.strip())
        if not match:
            raise HTTPException(status_code=400, detail="en-tête Range malformé")
        raw_start, raw_end = match.groups()
        if raw_start == "":
            if raw_end == "":
                raise HTTPException(status_code=400, detail="en-tête Range malformé")
            length = min(int(raw_end), size)
            start, end = size - length, size - 1
        else:
            start = int(raw_start)
            end = int(raw_end) if raw_end else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            headers["Content-Range"] = f"bytes */{size}"
            headers["Content-Length"] = "0"
            return Response(status_code=416, headers=headers)

        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        return StreamingResponse(
            _iter_range(path, start, end), status_code=206, media_type="application/octet-stream", headers=headers
        )

    @app.exception_handler(RequestValidationError)
    def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {
                "location": [str(part) for part in issue.get("loc", ())],
                "message": str(issue.get("msg", "valeur invalide")),
                "type": str(issue.get("type", "validation_error")),
            }
            for issue in exc.errors()
        ]
        summary = (
            "; ".join(f"{'.'.join(detail['location'])}: {detail['message']}" for detail in details)
            or "requête invalide"
        )
        return JSONResponse(
            status_code=422,
            content={"error": summary, "status": 422, "details": details},
        )

    @app.exception_handler(HTTPException)
    def http_error(request: Request, exc: HTTPException) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "status": exc.status_code})

    return app


async def _read_exact_body(request: Request, expected: int, index: int) -> bytes:
    raw_length = request.headers.get("content-length")
    if raw_length is not None:
        try:
            declared = int(raw_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Content-Length invalide") from exc
        if declared != expected:
            raise HTTPException(
                status_code=400,
                detail=f"chunk {index}: Content-Length {declared}, {expected} attendu",
            )

    body = bytearray()
    async for block in request.stream():
        if len(body) + len(block) > expected:
            raise HTTPException(
                status_code=400,
                detail=f"chunk {index}: plus de {expected} octets reçus",
            )
        body.extend(block)
    if len(body) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"chunk {index}: {len(body)} octets, {expected} attendus",
        )
    return bytes(body)


def _same_content(path: Path, expected: bytes) -> bool:
    offset = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            if block != expected[offset : offset + len(block)]:
                return False
            offset += len(block)
    return offset == len(expected)


def _artifact_path(job: Job, name: str) -> Path:
    if not ARTIFACT_NAME_RE.fullmatch(name) or name.endswith(".sha256"):
        raise HTTPException(status_code=400, detail="nom d'artefact invalide")
    path = job.artifacts_dir / name
    if path.is_symlink():
        raise HTTPException(status_code=400, detail="lien symbolique d'artefact interdit")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(job.artifacts_dir.resolve(strict=True))
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail=f"artefact absent: {name}") from None
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail=f"artefact absent: {name}")
    return path


def _iter_range(path: Path, start: int, end: int) -> Iterator[bytes]:
    remaining = end - start + 1
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining > 0:
            block = handle.read(min(STREAM_BLOCK, remaining))
            if not block:
                break
            remaining -= len(block)
            yield block


def _raise_if_cancelled(job: Job) -> None:
    if job.cancel_requested:
        raise JobCancelled("annulation pendant la préparation")


def _sha256(path: Path, job: Job | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if job is not None:
                _raise_if_cancelled(job)
            digest.update(block)
    return digest.hexdigest()


def _cached_sha256(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    stat = path.stat()
    if sidecar.is_file() and sidecar.stat().st_mtime >= stat.st_mtime:
        try:
            cached = sidecar.read_text(encoding="utf-8").strip().lower()
        except (OSError, UnicodeError):
            cached = ""
        if SHA256_RE.fullmatch(cached):
            return cached
    value = _sha256(path)
    with contextlib.suppress(OSError):  # cache best-effort : disque plein ou job supprimé
        sidecar.write_text(value, encoding="utf-8")
    return value


app = build_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    # Écoute toutes les interfaces intentionnellement : le conteneur publie ce port.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MINIDS_PORT", "8000")))  # noqa: S104
