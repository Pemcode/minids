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

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target_triangles, (int, np.integer))
            or isinstance(self.target_triangles, (bool, np.bool_))
            or self.target_triangles <= 0
        ):
            raise ValueError(f"budget de triangles invalide: {self.target_triangles!r}")
        if (
            not isinstance(self.smooth_iterations, (int, np.integer))
            or isinstance(self.smooth_iterations, (bool, np.bool_))
            or self.smooth_iterations < 0
        ):
            raise ValueError(f"nombre d'itérations de lissage invalide: {self.smooth_iterations!r}")
        for name in ("smooth_lambda", "smooth_mu"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"paramètre de lissage non fini: {name}")
        if not np.isfinite(self.component_ratio) or not 0.0 <= self.component_ratio <= 1.0:
            raise ValueError(f"ratio de composante invalide: {self.component_ratio!r}")
        for name in ("plane_margin", "bbox_margin", "hole_size_voxels"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} invalide: {value!r}")


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

    if not np.isfinite(voxel_size) or voxel_size <= 0:
        raise ValueError(f"taille de voxel invalide: {voxel_size!r}")
    if (bbox_min is None) != (bbox_max is None):
        raise ValueError("les deux bornes de découpe sont requises ensemble")

    before = (len(mesh.vertices), len(mesh.triangles))

    if bbox_min is not None and bbox_max is not None:
        bbox_min = np.asarray(bbox_min, dtype=np.float64)
        bbox_max = np.asarray(bbox_max, dtype=np.float64)
        if bbox_min.shape != (3,) or bbox_max.shape != (3,) or not np.isfinite([bbox_min, bbox_max]).all():
            raise ValueError("boîte de découpe invalide")
        if np.any(bbox_max < bbox_min) or np.linalg.norm(bbox_max - bbox_min) <= 1e-12:
            raise ValueError("boîte de découpe vide ou inversée")
        margin = config.bbox_margin * float(np.linalg.norm(bbox_max - bbox_min))
        box = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox_min - margin,
            max_bound=bbox_max + margin,
        )
        mesh = mesh.crop(box)
        log_fn(f"découpe boîte objet → {len(mesh.triangles)} triangles")

    if plane is not None and len(mesh.vertices):
        plane = np.asarray(plane, dtype=np.float64)
        if plane.shape != (4,) or not np.isfinite(plane).all() or np.linalg.norm(plane[:3]) <= 1e-12:
            raise ValueError("plan de support invalide")
        vertices = np.asarray(mesh.vertices)
        signed = vertices @ plane[:3] + float(plane[3])
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

    if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio de composante invalide: {ratio!r}")

    if not len(mesh.triangles):
        return mesh
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
        labels, _counts, areas = mesh.cluster_connected_triangles()
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

    if not np.isfinite(hole_size) or hole_size < 0:
        raise ValueError(f"taille de trou invalide: {hole_size!r}")
    if hole_size == 0 or not len(mesh.triangles):
        return mesh

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
    except (RuntimeError, ValueError) as exc:
        log.warning("aire de surface non calculable: %s", exc)
    if metrics["watertight"]:
        try:
            metrics["volume"] = float(mesh.get_volume())
        except (RuntimeError, ValueError) as exc:
            log.warning("volume non calculable: %s", exc)
    return metrics


def scale_metrics_for_export(metrics: dict[str, Any], scale: float) -> dict[str, Any]:
    """Ramène les métriques géométriques dans les unités du GLB exporté."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"échelle d'export invalide: {scale!r}")
    scaled = dict(metrics)
    scaled["export_scale"] = float(scale)
    for key in ("bbox_min", "bbox_max"):
        if key in scaled:
            scaled[key] = (np.asarray(scaled[key], dtype=np.float64) * scale).tolist()
    if "bbox_diagonal" in scaled:
        scaled["bbox_diagonal"] = float(scaled["bbox_diagonal"]) * scale
    if "surface_area" in scaled:
        scaled["surface_area"] = float(scaled["surface_area"]) * scale**2
    if "volume" in scaled:
        scaled["volume"] = float(scaled["volume"]) * scale**3
    return scaled
