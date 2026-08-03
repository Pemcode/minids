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
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .jobs import Job, JobStore, Reporter, assemble_upload

logging.basicConfig(level=os.environ.get("MINIDS_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("minids.api")

VERSION = "0.1.0"
ARTIFACT_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")
STREAM_BLOCK = 1024 * 1024


class JobParams(BaseModel):
    """Paramètres de reconstruction, tous optionnels côté client."""

    prompt: str | None = Field(default=None, description="Prompt texte SAM 3, ex: 'the sneaker'")
    frames: int = Field(default=0, ge=0, le=600, description="0 = valeur par défaut du serveur")
    refine: bool = Field(default=True, description="Raffinement 2DGS avant fusion TSDF")
    gs_iters: int = Field(default=12000, ge=500, le=60000)
    mesh_backends: list[Literal["tsdf2dgs", "tsdf", "poisson"]] = Field(default_factory=lambda: ["tsdf2dgs"])
    segmentation: Literal["auto", "sam3", "geometric", "none"] = "auto"
    texture: Literal["bake", "vertex"] = "bake"
    texture_size: int = Field(default=2048, ge=512, le=8192)
    target_triangles: int = Field(default=200_000, ge=5_000, le=2_000_000)
    voxel_divisor: int = Field(default=512, ge=64, le=2048)
    ref_size: float | None = Field(default=None, gt=0, description="Plus grande dimension réelle en mètres")
    bundle_adjustment: bool = False
    watertight: bool = True


class CreateJobRequest(BaseModel):
    filename: str
    size: int = Field(ge=1)
    chunk_size: int = Field(default=8 * 1024 * 1024, ge=64 * 1024)
    sha256: str | None = None
    params: JobParams = Field(default_factory=JobParams)


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="miniDS", version=VERSION, docs_url="/docs")

    from .pipeline.run import run_pipeline  # import tardif : évite de charger torch au boot

    def runner(job: Job, reporter: Reporter) -> None:
        run_pipeline(job, reporter, settings)

    store = JobStore(settings, runner)
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
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job inconnu")
        return job

    # -- santé ----------------------------------------------------------
    @app.get("/health")
    def health(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        # Volontairement ouvert : le HEALTHCHECK Docker et la vérification « le pod
        # répond-il ? » doivent marcher sans secret. En revanche, un token *présenté*
        # doit être valide, sinon `minids health` validerait un token erroné.
        if authorization is not None and not check_token(authorization):
            raise HTTPException(status_code=401, detail="token invalide")
        info: dict[str, Any] = {
            "status": "ok",
            "version": VERSION,
            "fake_gpu": settings.fake_gpu,
            "chunk_size": settings.chunk_size,
            "default_frames": settings.default_frames,
            "jobs": len(store.list()),
            "auth_configured": bool(settings.token),
            # Les deux réglages sans lesquels l'étape `vggt` échoue. Les exposer
            # ici permet de le voir avant de lancer un scan, plutôt qu'après cinq
            # étapes de pipeline sur un GPU facturé. Seule leur présence est
            # publiée : `HF_TOKEN` est un secret.
            "hf_token_configured": bool(settings.hf_token),
            "checkpoint": settings.checkpoint,
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
        total_chunks = (request.size + request.chunk_size - 1) // request.chunk_size
        upload = {
            "filename": Path(request.filename).name,
            "size": request.size,
            "chunk_size": request.chunk_size,
            "sha256": request.sha256,
            "total_chunks": total_chunks,
            "received": [],
        }
        job = store.create(request.params.model_dump(), upload)
        return {"job_id": job.job_id, "total_chunks": total_chunks, "chunk_size": request.chunk_size}

    @app.put("/jobs/{job_id}/chunks/{index}", dependencies=[auth])
    async def put_chunk(job_id: str, index: int, request: Request) -> dict[str, Any]:
        job = get_job(job_id)
        if job.status != "created":
            raise HTTPException(status_code=409, detail=f"upload fermé (statut {job.status})")
        total = job.upload["total_chunks"]
        if not 0 <= index < total:
            raise HTTPException(status_code=416, detail=f"index hors bornes (0..{total - 1})")

        body = await request.body()
        expected = job.upload["chunk_size"]
        is_last = index == total - 1
        if not is_last and len(body) != expected:
            raise HTTPException(status_code=400, detail=f"chunk {index}: {len(body)} octets, {expected} attendus")
        if is_last and not 0 < len(body) <= expected:
            raise HTTPException(status_code=400, detail=f"dernier chunk invalide ({len(body)} octets)")

        target = job.upload_dir / f"{index:06d}.part"
        tmp = target.with_suffix(".tmp")
        tmp.write_bytes(body)
        tmp.replace(target)
        if index not in job.upload["received"]:
            job.upload["received"].append(index)
        job.persist()
        return {"received": len(job.upload["received"]), "total": total}

    @app.get("/jobs/{job_id}/chunks", dependencies=[auth])
    def list_chunks(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        return {
            "received": sorted(job.upload.get("received", [])),
            "total": job.upload["total_chunks"],
            "chunk_size": job.upload["chunk_size"],
        }

    @app.post("/jobs/{job_id}/start", dependencies=[auth])
    def start_job(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        if job.status != "created":
            raise HTTPException(status_code=409, detail=f"job déjà démarré (statut {job.status})")
        try:
            path = assemble_upload(job)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        expected_sha = job.upload.get("sha256")
        if expected_sha:
            actual = _sha256(path)
            if actual != expected_sha:
                path.unlink(missing_ok=True)
                job.status = "failed"
                job.error = "sha256 de l'upload incorrect"
                job.persist()
                raise HTTPException(status_code=400, detail="sha256 de l'upload incorrect")

        job.upload["assembled_path"] = str(path)
        job.append_log(f"upload assemblé: {path.name} ({path.stat().st_size} octets)")
        store.enqueue(job)
        return {"job_id": job.job_id, "status": job.status, "queue_position": store.queue_position(job)}

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
        store.cancel(job)
        return {"job_id": job.job_id, "status": job.status}

    @app.delete("/jobs/{job_id}", dependencies=[auth])
    def delete_job(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        store.cancel(job)
        store.delete(job)
        return {"deleted": job_id}

    # -- artefacts ----------------------------------------------------------
    @app.get("/jobs/{job_id}/artifacts", dependencies=[auth])
    def list_artifacts(job_id: str) -> dict[str, Any]:
        job = get_job(job_id)
        items = []
        for path in sorted(job.artifacts_dir.glob("*")):
            if path.is_file() and not path.name.endswith(".sha256"):
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
            return StreamingResponse(_iter_range(path, 0, size - 1), media_type="application/octet-stream", headers=headers)

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
            return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})

        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
        headers["Content-Length"] = str(end - start + 1)
        return StreamingResponse(
            _iter_range(path, start, end), status_code=206, media_type="application/octet-stream", headers=headers
        )

    @app.exception_handler(HTTPException)
    def http_error(request: Request, exc: HTTPException) -> JSONResponse:  # noqa: ARG001
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail, "status": exc.status_code})

    return app


def _artifact_path(job: Job, name: str) -> Path:
    if not ARTIFACT_NAME_RE.match(name) or name.endswith(".sha256"):
        raise HTTPException(status_code=400, detail="nom d'artefact invalide")
    path = job.artifacts_dir / name
    if not path.is_file():
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cached_sha256(path: Path) -> str:
    sidecar = path.with_name(path.name + ".sha256")
    stat = path.stat()
    if sidecar.is_file() and sidecar.stat().st_mtime >= stat.st_mtime:
        return sidecar.read_text(encoding="utf-8").strip()
    value = _sha256(path)
    with contextlib.suppress(OSError):  # cache best-effort : disque plein ou job supprimé
        sidecar.write_text(value, encoding="utf-8")
    return value


app = build_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MINIDS_PORT", "8000")))
