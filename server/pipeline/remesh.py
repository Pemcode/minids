"""Re-maillage hors-ligne depuis `vggt_raw.npz`.

Permet de rejouer TSDF ou Poisson sans repasser par le pod : c'est la raison
d'être du rapatriement de la sortie brute. Ne dépend que de numpy et Open3D
(CPU), donc utilisable directement sur la machine Windows.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import cleanup as cleanup_module
from . import mesh_poisson, mesh_tsdf
from .geometry import SceneNormalization
from .vggt import VGGTResult


def load_raw(path: Path) -> tuple[VGGTResult, SceneNormalization, np.ndarray | None]:
    """Relit la sortie brute, sa normalisation et les masques compressés."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"archive VGGT-Ω introuvable: {path}")
    result = VGGTResult.load(path)
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        if {"scene_center", "scene_scale"} <= keys:
            normalization = SceneNormalization(
                center=np.asarray(data["scene_center"], dtype=np.float64),
                scale=float(data["scene_scale"]),
            )
        else:
            from .geometry import compute_normalization
            from .vggt import world_points

            points = world_points(result)
            valid = np.isfinite(result.depth) & (result.depth > 0) & np.isfinite(points).all(axis=-1)
            normalization = compute_normalization(points[valid])

        masks = None
        if {"masks_packed", "masks_shape"} <= keys:
            shape = tuple(int(v) for v in data["masks_shape"])
            if shape != result.depth.shape:
                raise ValueError(f"forme de masques {shape} incompatible avec les profondeurs {result.depth.shape}")
            count = int(np.prod(shape, dtype=np.int64))
            packed = np.asarray(data["masks_packed"])
            if packed.dtype != np.uint8:
                raise ValueError(f"masques compressés de type inattendu: {packed.dtype}")
            unpacked = np.unpackbits(packed)
            if len(unpacked) < count:
                raise ValueError("masques compressés tronqués dans l'archive VGGT-Ω")
            masks = unpacked[:count].astype(bool).reshape(shape)
    return result, normalization, masks


def remesh(
    npz_path: Path,
    output: Path,
    backend: str = "tsdf",
    voxel_divisor: int = 512,
    target_triangles: int = 200_000,
    use_masks: bool = True,
    watertight: bool = True,
    ref_size: float | None = None,
    device: str = "CPU:0",
    log_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Reconstruit un GLB depuis la sortie VGGT-Ω brute. Retourne les métriques."""
    from common.glb import write_glb

    if backend not in {"tsdf", "poisson"}:
        raise ValueError(f"backend local inconnu: {backend} (tsdf ou poisson)")
    if (
        not isinstance(voxel_divisor, (int, np.integer))
        or isinstance(voxel_divisor, (bool, np.bool_))
        or voxel_divisor <= 0
    ):
        raise ValueError(f"diviseur de voxel invalide: {voxel_divisor!r}")
    if (
        not isinstance(target_triangles, (int, np.integer))
        or isinstance(target_triangles, (bool, np.bool_))
        or target_triangles <= 0
    ):
        raise ValueError(f"budget de triangles invalide: {target_triangles!r}")
    if ref_size is not None and (not np.isfinite(ref_size) or ref_size <= 0):
        raise ValueError(f"taille de référence invalide: {ref_size!r}")
    output = Path(output)

    result, normalization, masks = load_raw(Path(npz_path))
    if not use_masks:
        masks = None
    log_fn(f"{len(result.depth)} vues, masques {'oui' if masks is not None else 'non'}")

    extrinsics = normalization.apply_extrinsics(result.extrinsics)
    depths = normalization.apply_depth(result.depth)

    bbox_min, bbox_max = _object_bounds(depths, result, extrinsics, masks)
    voxel_size = mesh_tsdf.voxel_size_from_bbox(bbox_min, bbox_max, voxel_divisor)
    log_fn(f"boîte objet {np.round(bbox_min, 3)} → {np.round(bbox_max, 3)}, voxel {voxel_size:.5f}")

    if backend == "tsdf":
        mesh = mesh_tsdf.fuse(
            depths=depths,
            colors=result.images,
            intrinsics=result.intrinsics,
            extrinsics=extrinsics,
            config=mesh_tsdf.TSDFConfig(voxel_size=voxel_size),
            masks=masks,
            device=device,
            log_fn=log_fn,
        )
    elif backend == "poisson":
        points, colors = mesh_poisson.point_cloud_from_depths(
            depths, result.images, result.intrinsics, extrinsics, masks=masks
        )
        mesh = mesh_poisson.reconstruct(
            points, colors, extrinsics, mesh_poisson.PoissonConfig(voxel_size=voxel_size), log_fn
        )
    mesh = cleanup_module.clean(
        mesh,
        cleanup_module.CleanupConfig(target_triangles=target_triangles, watertight=watertight),
        bbox_min,
        bbox_max,
        None,
        voxel_size,
        log_fn,
    )
    if not len(mesh.vertices) or not len(mesh.triangles):
        raise ValueError(f"maillage {backend} vide après nettoyage")

    scale = 1.0
    if ref_size is not None:
        vertices = np.asarray(mesh.vertices)
        extent = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
        if extent <= 1e-9:
            raise ValueError("maillage d'étendue nulle, mise à l'échelle impossible")
        scale = float(ref_size) / extent

    metrics = cleanup_module.scale_metrics_for_export(cleanup_module.mesh_metrics(mesh), scale)

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    write_glb(
        path=output,
        vertices=np.asarray(mesh.vertices) * scale,
        faces=np.asarray(mesh.triangles),
        normals=np.asarray(mesh.vertex_normals),
        vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
        name="minids_object",
    )
    log_fn(f"écrit {output} — {metrics['triangles']} triangles, watertight={metrics['watertight']}")
    return metrics


def _object_bounds(
    depths: np.ndarray, result: VGGTResult, extrinsics: np.ndarray, masks: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Boîte englobante robuste des points retenus (percentiles, pas min/max)."""
    from .geometry import unproject

    collected = []
    for index in range(len(depths)):
        depth = depths[index]
        keep = np.isfinite(depth) & (depth > 0)
        if masks is not None:
            keep &= masks[index]
        if not keep.any():
            continue
        points = unproject(depth, result.intrinsics[index], extrinsics[index])
        collected.append(points[keep][::7])
    if not collected:
        raise ValueError("aucun point valide dans vggt_raw.npz")
    points = np.concatenate(collected)
    return np.percentile(points, 0.5, axis=0), np.percentile(points, 99.5, axis=0)
