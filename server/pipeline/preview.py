"""Aperçu PNG du maillage, rendu par lancer de rayons.

Pas d'OpenGL : un pod n'a ni serveur X ni contexte GL utilisable sans bricolage.
`RaycastingScene` tourne sur CPU et donne un rendu ombré suffisant pour vérifier
d'un coup d'œil que le GLB n'est pas une bouillie.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def render(
    mesh: Any,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    view_indices: list[int] | None = None,
    height: int = 320,
    width: int = 320,
) -> np.ndarray:
    """Grille d'aperçus depuis plusieurs caméras d'origine, (H, W*n, 3) uint8."""
    import open3d as o3d
    import open3d.core as o3c

    if not len(mesh.triangles):
        return np.zeros((height, width, 3), dtype=np.uint8)

    count = len(extrinsics)
    if view_indices is None:
        view_indices = [int(round(i * count / 4.0)) % count for i in range(4)]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    tiles = []
    for index in view_indices:
        intrinsic = _resize_intrinsic(intrinsics[index], height, width)
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3c.Tensor(np.ascontiguousarray(intrinsic, dtype=np.float64)),
            extrinsic_matrix=o3c.Tensor(np.ascontiguousarray(extrinsics[index], dtype=np.float64)),
            width_px=width,
            height_px=height,
        )
        result = scene.cast_rays(rays)
        hit = np.isfinite(result["t_hit"].numpy())
        normals = result["primitive_normals"].numpy()

        # Éclairage frontal simple, dans le repère caméra (+z devant).
        rotation = np.asarray(extrinsics[index], dtype=np.float64)[:3, :3]
        normals_camera = normals @ rotation.T
        shading = np.clip(np.abs(normals_camera[..., 2]), 0.0, 1.0) * 0.75 + 0.25
        tile = np.where(hit[..., None], shading[..., None], 0.12)
        tiles.append((np.repeat(tile, 3, axis=-1) * 255).astype(np.uint8))

    return np.concatenate(tiles, axis=1)


def _resize_intrinsic(intrinsic: np.ndarray, height: int, width: int) -> np.ndarray:
    """Recadre les intrinsèques sur la taille de l'aperçu (approximation carrée)."""
    scaled = np.array(intrinsic, dtype=np.float64, copy=True)
    source_cx, source_cy = scaled[0, 2], scaled[1, 2]
    scale_x = width / (2.0 * source_cx) if source_cx > 0 else 1.0
    scale_y = height / (2.0 * source_cy) if source_cy > 0 else 1.0
    scaled[0, 0] *= scale_x
    scaled[0, 2] = width / 2.0
    scaled[1, 1] *= scale_y
    scaled[1, 2] = height / 2.0
    return scaled
