"""Reconstruction de Poisson screened — backend de comparaison du benchmark.

Points forts face au TSDF : tolère un échantillonnage non uniforme et produit
toujours une surface fermée. Points faibles : arrondit les arêtes vives et
« gonfle » là où l'objet est peu observé, d'où le seuillage par densité.
"""

from __future__ import annotations

import math
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

    def __post_init__(self) -> None:
        if not isinstance(self.depth, (int, np.integer)) or isinstance(self.depth, (bool, np.bool_)) or self.depth <= 0:
            raise ValueError(f"profondeur Poisson invalide: {self.depth!r}")
        if not np.isfinite(self.scale) or self.scale <= 1.0:
            raise ValueError(f"facteur d'échelle Poisson invalide: {self.scale!r}")
        if not np.isfinite(self.density_quantile) or not 0.0 <= self.density_quantile <= 1.0:
            raise ValueError(f"quantile de densité invalide: {self.density_quantile!r}")
        if not np.isfinite(self.normal_radius_voxels) or self.normal_radius_voxels <= 0:
            raise ValueError(f"rayon de normales invalide: {self.normal_radius_voxels!r}")
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError(f"taille de voxel invalide: {self.voxel_size!r}")
        if (
            not isinstance(self.max_points, (int, np.integer))
            or isinstance(self.max_points, (bool, np.bool_))
            or self.max_points <= 0
        ):
            raise ValueError(f"nombre maximal de points invalide: {self.max_points!r}")


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
    depths = np.asarray(depths)
    colors = np.asarray(colors)
    intrinsics = np.asarray(intrinsics)
    extrinsics = np.asarray(extrinsics)
    masks = None if masks is None else np.asarray(masks)
    alphas = None if alphas is None else np.asarray(alphas)
    if depths.ndim != 3 or any(size <= 0 for size in depths.shape):
        raise ValueError(f"profondeurs de forme inattendue: {depths.shape}")
    sequence, height, width = depths.shape
    expected = {
        "couleurs": (sequence, height, width, 3),
        "intrinsèques": (sequence, 3, 3),
        "extrinsèques": (sequence, 4, 4),
    }
    observed = {"couleurs": colors.shape, "intrinsèques": intrinsics.shape, "extrinsèques": extrinsics.shape}
    if masks is not None:
        expected["masques"] = depths.shape
        observed["masques"] = masks.shape
    if alphas is not None:
        expected["alphas"] = depths.shape
        observed["alphas"] = alphas.shape
    mismatches = [
        f"{name}={observed[name]} (attendu {shape})" for name, shape in expected.items() if observed[name] != shape
    ]
    if mismatches:
        raise ValueError("entrées Poisson incompatibles: " + ", ".join(mismatches))
    if not isinstance(stride, (int, np.integer)) or isinstance(stride, (bool, np.bool_)) or stride <= 0:
        raise ValueError(f"pas d'échantillonnage invalide: {stride!r}")
    if not np.isfinite(depth_max) or depth_max <= 0:
        raise ValueError(f"profondeur maximale invalide: {depth_max!r}")
    if not np.isfinite(min_alpha) or not 0.0 <= min_alpha <= 1.0:
        raise ValueError(f"seuil alpha invalide: {min_alpha!r}")
    if not np.isfinite(colors).all() or not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("couleurs, intrinsèques et extrinsèques Poisson doivent être finies")

    all_points, all_colors = [], []
    for index in range(len(depths)):
        depth = depths[index]
        keep = np.isfinite(depth) & (depth > 0) & (depth < depth_max)
        if masks is not None:
            keep &= masks[index].astype(bool, copy=False)
        if alphas is not None:
            keep &= np.isfinite(alphas[index]) & (alphas[index] >= min_alpha)
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

    points = np.asarray(points, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 10 or not np.isfinite(points).all():
        raise ValueError(f"nuage Poisson invalide ou trop petit: {points.shape}")
    if colors.shape != points.shape or not np.isfinite(colors).all():
        raise ValueError(f"couleurs Poisson invalides: {colors.shape}")
    if (
        extrinsics.ndim != 3
        or extrinsics.shape[1:] != (4, 4)
        or not len(extrinsics)
        or not np.isfinite(extrinsics).all()
    ):
        raise ValueError(f"extrinsèques Poisson invalides: {extrinsics.shape}")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if len(points) > config.max_points:
        ratio = math.ceil(len(points) / config.max_points)
        cloud = cloud.uniform_down_sample(ratio)
    cloud = cloud.voxel_down_sample(config.voxel_size)
    if len(cloud.points) >= 4:
        neighbours = min(20, len(cloud.points) - 1)
        cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=neighbours, std_ratio=2.0)
    if len(cloud.points) < 10:
        raise ValueError("moins de 10 points subsistent après filtrage Poisson")

    radius = config.normal_radius_voxels * config.voxel_size
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    # Sans orientation cohérente, Poisson produit une surface retournée par morceaux.
    centroid = camera_centers(extrinsics).mean(axis=0)
    cloud.orient_normals_towards_camera_location(centroid)
    cloud.orient_normals_consistent_tangent_plane(k=min(15, len(cloud.points) - 1))

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
