"""Helpers géométriques partagés par les étapes du pipeline.

Convention de caméra : celle de COLMAP / VGGT — les extrinsèques sont
*world-to-camera*, `X_cam = R @ X_world + t`, axes `+x` droite, `+y` bas,
`+z` devant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def to_4x4(extrinsics: np.ndarray) -> np.ndarray:
    """Complète des extrinsèques (..., 3, 4) en matrices homogènes (..., 4, 4)."""
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim < 2:
        raise ValueError(f"extrinsèques de forme inattendue: {extrinsics.shape}")
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    if extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"extrinsèques de forme inattendue: {extrinsics.shape}")
    shape = extrinsics.shape[:-2]
    bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (*shape, 1, 4))
    return np.concatenate([extrinsics, bottom], axis=-2)


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    """Positions des caméras dans le repère monde, (..., 3)."""
    matrices = to_4x4(extrinsics)
    rotations = matrices[..., :3, :3]
    translations = matrices[..., :3, 3]
    return -np.einsum("...ji,...j->...i", rotations, translations)


def pixel_grid(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Coordonnées de pixels au centre de chaque texel."""
    if not isinstance(height, (int, np.integer)) or isinstance(height, (bool, np.bool_)) or height <= 0:
        raise ValueError(f"hauteur d'image invalide: {height!r}")
    if not isinstance(width, (int, np.integer)) or isinstance(width, (bool, np.bool_)) or width <= 0:
        raise ValueError(f"largeur d'image invalide: {width!r}")
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64) + 0.5, np.arange(height, dtype=np.float64) + 0.5)
    return us, vs


def unproject(depth: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
    """Dé-projette une carte de profondeur (H, W) en points monde (H, W, 3)."""
    depth = np.asarray(depth)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f"profondeur de forme inattendue: {depth.shape}, (H, W) attendu")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsèques de forme inattendue: {intrinsic.shape}, (3, 3) attendu")
    if extrinsic.shape not in {(3, 4), (4, 4)}:
        raise ValueError(f"extrinsèque de forme inattendue: {extrinsic.shape}")
    height, width = depth.shape
    us, vs = pixel_grid(height, width)
    fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
    cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
    if not np.isfinite([fx, fy, cx, cy]).all() or abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise ValueError("intrinsèques invalides: focales non nulles et paramètres finis requis")

    x_cam = (us - cx) / fx * depth
    y_cam = (vs - cy) / fy * depth
    points_cam = np.stack([x_cam, y_cam, depth], axis=-1)

    matrix = to_4x4(extrinsic[None])[0]
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    return (points_cam - translation) @ rotation


def project(points_world: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projette des points monde (N, 3) → pixels (N, 2) et profondeur caméra (N,)."""
    points_world = np.asarray(points_world, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    extrinsic = np.asarray(extrinsic, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1:] != (3,):
        raise ValueError(f"points de forme inattendue: {points_world.shape}, (N, 3) attendu")
    if intrinsic.shape != (3, 3):
        raise ValueError(f"intrinsèques de forme inattendue: {intrinsic.shape}, (3, 3) attendu")
    if extrinsic.shape not in {(3, 4), (4, 4)}:
        raise ValueError(f"extrinsèque de forme inattendue: {extrinsic.shape}")
    fx, fy = intrinsic[[0, 1], [0, 1]]
    cx, cy = intrinsic[[0, 1], [2, 2]]
    if not np.isfinite([fx, fy, cx, cy]).all() or abs(fx) <= 1e-12 or abs(fy) <= 1e-12:
        raise ValueError("intrinsèques invalides: focales non nulles et finies requises")
    matrix = to_4x4(extrinsic[None])[0]
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    points_cam = points_world @ rotation.T + translation
    depth = points_cam[:, 2]
    safe = np.where(np.abs(depth) < 1e-9, np.copysign(1e-9, depth), depth)
    u = intrinsic[0, 0] * points_cam[:, 0] / safe + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_cam[:, 1] / safe + intrinsic[1, 2]
    return np.stack([u, v], axis=-1), depth


@dataclass
class SceneNormalization:
    """Recentrage + mise à l'échelle de la scène.

    VGGT-Ω prédit une géométrie *up-to-scale* : sans normalisation, la taille de
    voxel du TSDF et les hyperparamètres du 2DGS n'ont aucun sens stable d'un
    scan à l'autre. On ramène donc la scène dans une boule de rayon ~1.
    """

    center: np.ndarray
    scale: float

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64)
        self.scale = float(self.scale)
        if self.center.shape != (3,) or not np.isfinite(self.center).all():
            raise ValueError(f"centre de normalisation invalide: {self.center}")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"échelle de normalisation invalide: {self.scale}")

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        return (points - self.center) / self.scale

    def apply_extrinsics(self, extrinsics: np.ndarray) -> np.ndarray:
        """Adapte les extrinsèques world-to-camera à la transformation du monde."""
        matrices = to_4x4(extrinsics).copy()
        rotations = matrices[..., :3, :3]
        translations = matrices[..., :3, 3]
        # X'_world = (X_world - c) / s  ⇒  t' = (t + R c) / s
        matrices[..., :3, 3] = (translations + np.einsum("...ij,j->...i", rotations, self.center)) / self.scale
        return matrices

    def apply_depth(self, depth: np.ndarray) -> np.ndarray:
        return depth / self.scale

    def to_dict(self) -> dict[str, object]:
        return {"center": self.center.tolist(), "scale": float(self.scale)}


def compute_normalization(points: np.ndarray, percentile: float = 95.0) -> SceneNormalization:
    """Centre robuste (médiane) + rayon robuste (percentile) du nuage."""
    if not np.isfinite(percentile) or not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile invalide: {percentile!r}")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        raise ValueError("nuage de points vide : la sortie VGGT-Ω est inexploitable")
    center = np.median(finite, axis=0)
    radii = np.linalg.norm(finite - center, axis=1)
    scale = float(np.percentile(radii, percentile))
    if not np.isfinite(scale) or scale <= 1e-9:
        scale = 1.0
    return SceneNormalization(center=center, scale=scale)


def confidence_threshold(confidence: np.ndarray, keep_fraction: float) -> float:
    """Seuil gardant `keep_fraction` des pixels les plus confiants."""
    if not np.isfinite(keep_fraction) or not 0.0 <= keep_fraction <= 1.0:
        raise ValueError(f"fraction de confiance invalide: {keep_fraction!r}")
    values = np.asarray(confidence, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -np.inf
    if keep_fraction == 0:
        return np.inf
    if keep_fraction == 1:
        return -np.inf
    return float(np.percentile(values, 100.0 * (1.0 - keep_fraction)))


def largest_cluster_mask(points: np.ndarray, eps: float, min_points: int = 30) -> np.ndarray:
    """Masque booléen du plus gros amas DBSCAN (Open3D), pour isoler l'objet."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(f"nuage de forme inattendue: {points.shape}, (N, 3) attendu")
    if not np.isfinite(points).all():
        raise ValueError("DBSCAN requiert un nuage de points fini")
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError(f"rayon DBSCAN invalide: {eps!r}")
    if not isinstance(min_points, (int, np.integer)) or isinstance(min_points, (bool, np.bool_)) or min_points <= 0:
        raise ValueError(f"minimum de points DBSCAN invalide: {min_points!r}")
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return np.ones(len(points), dtype=bool)
    counts = np.bincount(labels[labels >= 0])
    return labels == int(np.argmax(counts))
