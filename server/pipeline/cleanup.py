"""Nettoyage du maillage brut : découpe, composantes, lissage, décimation, trous.

Le marching cubes sort toujours des îlots parasites (bruit de profondeur derrière
l'objet, morceaux de plan de support). Sans cette étape, le GLB contient autant
de déchets que d'objet.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("minids.cleanup")


@dataclass
class CleanupConfig:
    target_triangles: int = 200_000
    smooth_iterations: int = 8
    smooth_lambda: float = 0.5
    smooth_mu: float = -0.53
    component_ratio: float = 0.05
    plane_margin: float = 0.01
    bbox_margin: float = 0.02
    watertight: bool = True
    hole_size_voxels: float = 30.0


def clean(
    mesh: Any,
    config: CleanupConfig,
    bbox_min: np.ndarray | None = None,
    bbox_max: np.ndarray | None = None,
    plane: np.ndarray | None = None,
    voxel_size: float = 0.004,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> Any:
    import open3d as o3d

    before = (len(mesh.vertices), len(mesh.triangles))

    if bbox_min is not None and bbox_max is not None:
        margin = config.bbox_margin * float(np.linalg.norm(np.asarray(bbox_max) - np.asarray(bbox_min)))
        box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.asarray(bbox_min, dtype=np.float64) - margin,
            max_bound=np.asarray(bbox_max, dtype=np.float64) + margin,
        )
        mesh = mesh.crop(box)
        log_fn(f"découpe boîte objet → {len(mesh.triangles)} triangles")

    if plane is not None and len(mesh.vertices):
        vertices = np.asarray(mesh.vertices)
        signed = vertices @ np.asarray(plane[:3]) + float(plane[3])
        # On retire ce qui est sur ou sous le plan de support.
        mesh.remove_vertices_by_mask(signed < config.plane_margin)
        log_fn(f"plan de support retiré → {len(mesh.triangles)} triangles")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    mesh = keep_main_components(mesh, config.component_ratio, log_fn)

    if config.smooth_iterations > 0 and len(mesh.triangles):
        colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
        smoothed = mesh.filter_smooth_taubin(
            number_of_iterations=config.smooth_iterations, lambda_filter=config.smooth_lambda, mu=config.smooth_mu
        )
        if colors is not None and len(colors) == len(smoothed.vertices):
            smoothed.vertex_colors = o3d.utility.Vector3dVector(colors)
        mesh = smoothed

    if config.target_triangles and len(mesh.triangles) > config.target_triangles:
        mesh = mesh.simplify_quadric_decimation(int(config.target_triangles))
        log_fn(f"décimation → {len(mesh.triangles)} triangles")

    if config.watertight:
        mesh = fill_holes(mesh, config.hole_size_voxels * voxel_size, log_fn)

    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    log_fn(f"nettoyage : {before[0]}→{len(mesh.vertices)} sommets, {before[1]}→{len(mesh.triangles)} triangles")
    return mesh


def keep_main_components(mesh: Any, ratio: float, log_fn: Callable[[str], None]) -> Any:
    """Ne garde que les composantes dont l'aire dépasse `ratio` × la plus grande."""
    import open3d as o3d

    if not len(mesh.triangles):
        return mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels, counts, areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    areas = np.asarray(areas)
    if areas.size <= 1:
        return mesh

    threshold = ratio * areas.max()
    drop = np.isin(labels, np.nonzero(areas < threshold)[0])
    if drop.any():
        mesh.remove_triangles_by_mask(drop)
        mesh.remove_unreferenced_vertices()
        log_fn(f"{int(drop.sum())} triangles d'îlots parasites retirés ({areas.size} composantes)")
    return mesh


def fill_holes(mesh: Any, hole_size: float, log_fn: Callable[[str], None]) -> Any:
    """Bouche les trous (API tensorielle) pour obtenir un objet fermé."""
    import open3d as o3d

    try:
        colors = np.asarray(mesh.vertex_colors).copy() if mesh.has_vertex_colors() else None
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        filled = tensor_mesh.fill_holes(hole_size=float(hole_size)).to_legacy()
        if not len(filled.triangles):
            return mesh
        # `fill_holes` ajoute des sommets en fin de tableau : on prolonge la couleur.
        if colors is not None and len(filled.vertices) >= len(colors):
            padding = np.tile(colors.mean(axis=0), (len(filled.vertices) - len(colors), 1))
            filled.vertex_colors = o3d.utility.Vector3dVector(np.vstack([colors, padding]))
        log_fn(f"trous bouchés : {len(mesh.triangles)}→{len(filled.triangles)} triangles")
        return filled
    except Exception as exc:  # noqa: BLE001 - fill_holes est capricieux sur maillage non-manifold
        log_fn(f"bouchage de trous ignoré ({type(exc).__name__}: {exc})")
        return mesh


def mesh_metrics(mesh: Any) -> dict[str, Any]:
    """Métriques reportées dans `report.json` et le benchmark."""

    vertices = np.asarray(mesh.vertices)
    metrics: dict[str, Any] = {
        "vertices": int(len(vertices)),
        "triangles": int(len(mesh.triangles)),
        "watertight": bool(mesh.is_watertight()),
        "edge_manifold": bool(mesh.is_edge_manifold()),
        "vertex_manifold": bool(mesh.is_vertex_manifold()),
        "self_intersecting": None,
    }
    if len(vertices):
        metrics["bbox_min"] = vertices.min(axis=0).tolist()
        metrics["bbox_max"] = vertices.max(axis=0).tolist()
        metrics["bbox_diagonal"] = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    try:
        metrics["surface_area"] = float(mesh.get_surface_area())
        if metrics["watertight"]:
            metrics["volume"] = float(mesh.get_volume())
    except Exception:  # noqa: BLE001
        pass
    return metrics
