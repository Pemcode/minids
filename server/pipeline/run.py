"""Orchestration : vidéo → GLB.

L'ordre diffère d'une intuition naturelle sur un point : la segmentation passe
**après** VGGT-Ω. Le repli géométrique a besoin de la profondeur pour isoler
l'objet, et SAM 3 travaille aussi bien sur les images pré-traitées par le modèle
— ce qui garantit en prime un alignement pixel à pixel avec les intrinsèques.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Settings
from ..jobs import Job, Reporter, unpack_input
from . import cleanup as cleanup_module
from . import mesh_poisson, mesh_tsdf, preview
from . import texture as texture_module
from .colmap_export import build_sparse_cloud, refine_with_bundle_adjustment, write_colmap
from .geometry import compute_normalization
from .refine_2dgs import RefineConfig, RefineResult
from .segment import segment
from .vggt import VGGTResult, run_inference, world_points

log = logging.getLogger("minids.run")

BACKEND_LABELS = {
    "tsdf2dgs": "TSDF sur profondeurs rendues par 2DGS",
    "tsdf": "TSDF sur profondeur VGGT-Ω brute",
    "poisson": "Poisson screened",
}


def run_pipeline(job: Job, reporter: Reporter, settings: Settings) -> None:
    if settings.fake_gpu:
        from .fake import run_fake_pipeline

        run_fake_pipeline(job, reporter, settings)
        return

    started = time.time()
    params = job.params
    report: dict[str, Any] = {
        "job_id": job.job_id,
        "params": params,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "backends": {},
    }

    # -- 1. ingestion ----------------------------------------------------
    reporter.stage("ingest")
    source = Path(job.upload["assembled_path"])
    kind, target = unpack_input(source, job.frames_dir)
    reporter.log(f"entrée : {kind} ({source.name})")
    reporter.progress(1.0)

    # -- 2. images -------------------------------------------------------
    reporter.stage("frames")
    frame_count = params.get("frames") or settings.default_frames
    frames = _prepare_frames(kind, target, job.frames_dir, frame_count, reporter)
    report["frames"] = {"count": len(frames), "source": kind}
    reporter.progress(1.0)
    reporter.check_cancelled()

    # -- 3. VGGT-Ω -------------------------------------------------------
    reporter.stage("vggt")
    result = run_inference(
        frames=frames,
        checkpoint=settings.checkpoint,
        device=settings.device,
        hf_token=settings.hf_token,
        cache_dir=settings.cache_dir,
        image_resolution=settings.image_resolution,
        log_fn=reporter.log,
    )
    dense_points = world_points(result)
    normalization = compute_normalization(dense_points[np.isfinite(dense_points).all(axis=-1)])
    reporter.log(f"échelle de scène normalisée (facteur {normalization.scale:.4f})")
    report["normalization"] = normalization.to_dict()
    reporter.progress(1.0)
    reporter.check_cancelled()

    # -- 4. isolation de l'objet -----------------------------------------
    reporter.stage("segment")
    segmentation = segment(
        result=result,
        normalization=normalization,
        method=params.get("segmentation", "auto"),
        prompt=params.get("prompt"),
        log_fn=reporter.log,
    )
    report["segmentation"] = segmentation.to_dict()
    segmentation.save_pngs(job.work_dir / "masks")
    reporter.progress(1.0)
    reporter.check_cancelled()

    # La sortie brute part avec tout ce qu'il faut pour re-mailler en local.
    raw_path = job.artifacts_dir / "vggt_raw.npz"
    result.save(
        raw_path,
        extra={
            "scene_center": normalization.center.astype(np.float32),
            "scene_scale": np.float32(normalization.scale),
            "masks_packed": np.packbits(segmentation.masks),
            "masks_shape": np.array(segmentation.masks.shape, dtype=np.int64),
        },
    )
    reporter.log(f"vggt_raw.npz écrit ({raw_path.stat().st_size / 1e6:.1f} Mo)")

    # Toute la suite travaille dans le repère normalisé.
    normalized_extrinsics = normalization.apply_extrinsics(result.extrinsics)
    normalized_depth = normalization.apply_depth(result.depth)

    # -- 5. export COLMAP -------------------------------------------------
    reporter.stage("colmap")
    cloud = build_sparse_cloud(result, normalization, segmentation)
    sparse_dir = write_colmap(job.work_dir, result, normalization, cloud, reporter.log)
    if params.get("bundle_adjustment"):
        refine_with_bundle_adjustment(sparse_dir, job.frames_dir, reporter.log)
    report["sparse_points"] = int(len(cloud.points))
    reporter.progress(1.0)
    reporter.check_cancelled()

    # -- 6. raffinement 2DGS ----------------------------------------------
    reporter.stage("refine")
    refined: RefineResult | None = None
    if params.get("refine", True):
        refined = _run_refinement(
            result, segmentation, normalized_depth, normalized_extrinsics, cloud, params, settings, reporter
        )
        report["refine"] = {"gaussians": refined.num_gaussians, **refined.losses}
    else:
        reporter.log("raffinement désactivé : fusion sur la profondeur VGGT-Ω brute")
    reporter.progress(1.0)
    reporter.check_cancelled()

    # -- 7-9. maillage, nettoyage, texture --------------------------------
    backends = params.get("mesh_backends") or ["tsdf2dgs"]
    if refined is None:
        backends = ["tsdf" if backend == "tsdf2dgs" else backend for backend in backends]
        backends = list(dict.fromkeys(backends))

    voxel_size = mesh_tsdf.voxel_size_from_bbox(
        segmentation.bbox_min if segmentation.bbox_min is not None else cloud.points.min(axis=0),
        segmentation.bbox_max if segmentation.bbox_max is not None else cloud.points.max(axis=0),
        int(params.get("voxel_divisor", 512)),
    )
    reporter.log(f"taille de voxel TSDF : {voxel_size:.5f} (unités scène normalisées)")

    reporter.stage("mesh")
    meshes: dict[str, Any] = {}
    for index, backend in enumerate(backends):
        reporter.log(f"maillage — {BACKEND_LABELS.get(backend, backend)}")
        meshes[backend] = _build_mesh(
            backend, result, segmentation, refined, normalized_depth, normalized_extrinsics,
            voxel_size, settings, reporter,
        )
        reporter.progress((index + 1) / len(backends))
        reporter.check_cancelled()

    reporter.stage("cleanup")
    cleanup_config = cleanup_module.CleanupConfig(
        target_triangles=int(params.get("target_triangles", 200_000)),
        watertight=bool(params.get("watertight", True)),
    )
    for index, (backend, mesh) in enumerate(list(meshes.items())):
        meshes[backend] = cleanup_module.clean(
            mesh, cleanup_config, segmentation.bbox_min, segmentation.bbox_max,
            segmentation.plane, voxel_size, reporter.log,
        )
        report["backends"].setdefault(backend, {})["metrics"] = cleanup_module.mesh_metrics(meshes[backend])
        reporter.progress((index + 1) / len(meshes))
        reporter.check_cancelled()

    primary_backend = backends[0]
    primary_mesh = meshes[primary_backend]
    if not len(primary_mesh.triangles):
        raise ValueError(f"maillage vide pour le backend {primary_backend} : scan inexploitable")

    reporter.stage("texture")
    bake = _bake_texture(primary_mesh, result, segmentation, refined, normalized_extrinsics, params, reporter)
    report["texture"] = {"method": bake.method, "coverage": round(bake.coverage, 4)}
    reporter.progress(1.0)

    # -- 10. export --------------------------------------------------------
    reporter.stage("export")
    scale_to_real = _real_scale(primary_mesh, params.get("ref_size"))
    report["export_scale"] = scale_to_real
    _write_glb(job.artifacts_dir / "mesh.glb", bake, scale_to_real)
    for backend, mesh in meshes.items():
        if backend != primary_backend:
            _write_backend_glb(job.artifacts_dir / f"mesh_{backend}.glb", mesh, scale_to_real)

    _write_preview(job.artifacts_dir / "preview.png", primary_mesh, result, normalized_extrinsics, reporter)

    report["primary_backend"] = primary_backend
    report["timings"] = reporter.timings()
    report["total_seconds"] = round(time.time() - started, 2)
    (job.artifacts_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    reporter.progress(1.0)
    reporter.log(f"terminé en {report['total_seconds']:.0f}s → mesh.glb")


# ---------------------------------------------------------------------------
# Étapes
# ---------------------------------------------------------------------------

def _prepare_frames(kind: str, target: Path, frames_dir: Path, count: int, reporter: Reporter) -> list[Path]:
    from common.video import extract_frames

    if kind == "video":
        return extract_frames(target, frames_dir, count=count, log=reporter.log)

    frames = sorted(p for p in frames_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        raise ValueError("aucune image trouvée dans l'archive")
    if len(frames) > count:
        indices = np.linspace(0, len(frames) - 1, count).round().astype(int)
        frames = [frames[i] for i in sorted(set(indices.tolist()))]
    reporter.log(f"{len(frames)} images pré-extraites côté client")
    return frames


def _run_refinement(
    result: VGGTResult,
    segmentation: Any,
    normalized_depth: np.ndarray,
    normalized_extrinsics: np.ndarray,
    cloud: Any,
    params: dict[str, Any],
    settings: Settings,
    reporter: Reporter,
) -> RefineResult:
    from .refine_2dgs import refine

    config = RefineConfig(iterations=int(params.get("gs_iters", 12_000)))
    # La profondeur n'ancre l'optimisation que là où le modèle est confiant.
    confidence = result.depth_conf
    threshold = float(np.percentile(confidence, 40.0))
    weights = np.clip((confidence - threshold) / max(1e-6, confidence.max() - threshold), 0.0, 1.0)

    return refine(
        images=result.images,
        masks=segmentation.masks,
        depths=normalized_depth,
        depth_weights=weights.astype(np.float32),
        viewmats=normalized_extrinsics.astype(np.float32),
        intrinsics=result.intrinsics.astype(np.float32),
        init_points=cloud.points,
        init_colors=cloud.colors.astype(np.float32) / 255.0,
        config=config,
        device=settings.device,
        log_fn=reporter.log,
        progress_fn=reporter.progress,
        should_stop=reporter.check_cancelled,
    )


def _build_mesh(
    backend: str,
    result: VGGTResult,
    segmentation: Any,
    refined: RefineResult | None,
    normalized_depth: np.ndarray,
    normalized_extrinsics: np.ndarray,
    voxel_size: float,
    settings: Settings,
    reporter: Reporter,
) -> Any:
    device = "CUDA:0" if settings.device.startswith("cuda") else "CPU:0"
    tsdf_config = mesh_tsdf.TSDFConfig(voxel_size=voxel_size)

    if backend == "tsdf2dgs":
        if refined is None:
            raise ValueError("backend tsdf2dgs demandé sans raffinement")
        return mesh_tsdf.fuse(
            depths=refined.depths, colors=refined.colors,
            intrinsics=result.intrinsics, extrinsics=normalized_extrinsics,
            config=tsdf_config, masks=segmentation.masks, alphas=refined.alphas,
            device=device, log_fn=reporter.log,
        )

    if backend == "tsdf":
        return mesh_tsdf.fuse(
            depths=normalized_depth, colors=result.images,
            intrinsics=result.intrinsics, extrinsics=normalized_extrinsics,
            config=tsdf_config, masks=segmentation.masks,
            device=device, log_fn=reporter.log,
        )

    if backend == "poisson":
        depths = refined.depths if refined is not None else normalized_depth
        colors = refined.colors if refined is not None else result.images
        alphas = refined.alphas if refined is not None else None
        points, point_colors = mesh_poisson.point_cloud_from_depths(
            depths, colors, result.intrinsics, normalized_extrinsics,
            masks=segmentation.masks, alphas=alphas,
        )
        return mesh_poisson.reconstruct(
            points, point_colors, normalized_extrinsics,
            mesh_poisson.PoissonConfig(voxel_size=voxel_size), reporter.log,
        )

    raise ValueError(f"backend de maillage inconnu: {backend}")


def _bake_texture(
    mesh: Any,
    result: VGGTResult,
    segmentation: Any,
    refined: RefineResult | None,
    normalized_extrinsics: np.ndarray,
    params: dict[str, Any],
    reporter: Reporter,
) -> texture_module.BakeResult:
    if params.get("texture") == "vertex":
        vertices = np.asarray(mesh.vertices)
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        return texture_module.BakeResult(
            vertices=vertices,
            faces=np.asarray(mesh.triangles),
            normals=np.asarray(mesh.vertex_normals),
            uvs=None,
            texture=None,
            vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
            method="vertex",
        )

    config = texture_module.TextureConfig(size=int(params.get("texture_size", 2048)))
    return texture_module.bake(
        mesh=mesh,
        images=result.images,
        intrinsics=result.intrinsics,
        extrinsics=normalized_extrinsics,
        config=config,
        masks=segmentation.masks,
        log_fn=reporter.log,
    )


def _real_scale(mesh: Any, ref_size: float | None) -> float:
    """Facteur ramenant le maillage à sa taille réelle si l'utilisateur l'a donnée."""
    if not ref_size:
        return 1.0
    vertices = np.asarray(mesh.vertices)
    if not len(vertices):
        return 1.0
    extent = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    return float(ref_size) / extent if extent > 1e-9 else 1.0


def _write_glb(path: Path, bake: texture_module.BakeResult, scale: float) -> None:
    from common.glb import encode_png, write_glb

    write_glb(
        path=path,
        vertices=np.asarray(bake.vertices) * scale,
        faces=np.asarray(bake.faces),
        normals=bake.normals,
        uvs=bake.uvs,
        vertex_colors=bake.vertex_colors,
        texture_png=encode_png(bake.texture) if bake.texture is not None else None,
        name="minids_object",
    )


def _write_backend_glb(path: Path, mesh: Any, scale: float) -> None:
    from common.glb import write_glb

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    write_glb(
        path=path,
        vertices=np.asarray(mesh.vertices) * scale,
        faces=np.asarray(mesh.triangles),
        normals=np.asarray(mesh.vertex_normals),
        vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
        name=path.stem,
    )


def _write_preview(
    path: Path, mesh: Any, result: VGGTResult, extrinsics: np.ndarray, reporter: Reporter
) -> None:
    from common.glb import encode_png

    try:
        image = preview.render(mesh, result.intrinsics, extrinsics)
        path.write_bytes(encode_png(image))
    except Exception as exc:  # noqa: BLE001 - un aperçu raté ne doit pas perdre le scan
        reporter.log(f"aperçu non généré ({type(exc).__name__}: {exc})")
