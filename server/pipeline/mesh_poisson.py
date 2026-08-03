"""Reconstruction de Poisson screened — backend de comparaison du benchmark.

Points forts face au TSDF : tolère un échantillonnage non uniforme et produit
toujours une surface fermée. Points faibles : arrondit les arêtes vives et
« gonfle » là où l'objet est peu observé, d'où le seuillage par densité.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import camera_centers, unproject


@dataclass
class PoissonConfig:
    depth: int = 10
    scale: float = 1.1
    density_quantile: float = 0.03
    normal_radius_voxels: float = 4.0
    voxel_size: float = 0.004
    max_points: int = 1_500_000


def point_cloud_from_depths(
    depths: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    masks: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
    min_alpha: float = 0.5,
    depth_max: float = 8.0,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Nuage monde + couleurs, issu des cartes de profondeur retenues."""
    all_points, all_colors = [], []
    for index in range(len(depths)):
        depth = depths[index]
        keep = np.isfinite(depth) & (depth > 0) & (depth < depth_max)
        if masks is not None:
            keep &= masks[index]
        if alphas is not None:
            keep &= alphas[index] >= min_alpha
        if stride > 1:
            sub = np.zeros_like(keep)
            sub[::stride, ::stride] = True
            keep &= sub
        if not keep.any():
            continue
        points = unproject(depth, intrinsics[index], extrinsics[index])
        all_points.append(points[keep])
        all_colors.append(np.clip(colors[index][keep], 0, 1))

    if not all_points:
        raise ValueError("aucun point exploitable pour Poisson")
    return np.concatenate(all_points), np.concatenate(all_colors)


def reconstruct(
    points: np.ndarray,
    colors: np.ndarray,
    extrinsics: np.ndarray,
    config: PoissonConfig,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> Any:
    """Estime les normales (orientées vers les caméras) puis reconstruit."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if len(points) > config.max_points:
        ratio = max(1, len(points) // config.max_points)
        cloud = cloud.uniform_down_sample(ratio)
    cloud = cloud.voxel_down_sample(config.voxel_size)
    cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    radius = config.normal_radius_voxels * config.voxel_size
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    # Sans orientation cohérente, Poisson produit une surface retournée par morceaux.
    centroid = camera_centers(extrinsics).mean(axis=0)
    cloud.orient_normals_towards_camera_location(centroid)
    cloud.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        cloud, depth=config.depth, scale=config.scale, linear_fit=False
    )
    densities = np.asarray(densities)
    if config.density_quantile > 0 and densities.size:
        threshold = np.quantile(densities, config.density_quantile)
        mesh.remove_vertices_by_mask(densities < threshold)

    log_fn(
        f"Poisson depth={config.depth} sur {len(cloud.points)} points → "
        f"{len(mesh.vertices)} sommets / {len(mesh.triangles)} triangles"
    )
    return mesh
