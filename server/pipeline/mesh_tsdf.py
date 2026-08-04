"""Fusion TSDF des cartes de profondeur → maillage (marching cubes).

Utilise `open3d.t.geometry.VoxelBlockGrid`, la version tensorielle qui tourne sur
GPU, avec repli sur l'ancienne `ScalableTSDFVolume` en CPU — c'est ce repli qui
permet de re-mailler en local sur Windows depuis `vggt_raw.npz`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("minids.tsdf")


@dataclass
class TSDFConfig:
    voxel_size: float = 0.004
    trunc_multiplier: float = 4.0
    depth_max: float = 8.0
    block_count: int = 30_000
    block_resolution: int = 16
    min_alpha: float = 0.5

    def __post_init__(self) -> None:
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError(f"taille de voxel invalide: {self.voxel_size!r}")
        if not np.isfinite(self.trunc_multiplier) or self.trunc_multiplier <= 0:
            raise ValueError(f"multiplicateur de troncature invalide: {self.trunc_multiplier!r}")
        if not np.isfinite(self.depth_max) or self.depth_max <= 0:
            raise ValueError(f"profondeur maximale invalide: {self.depth_max!r}")
        for name in ("block_count", "block_resolution"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
                raise ValueError(f"{name} invalide: {value!r}")
        if not np.isfinite(self.min_alpha) or not 0.0 <= self.min_alpha <= 1.0:
            raise ValueError(f"seuil alpha invalide: {self.min_alpha!r}")


def voxel_size_from_bbox(bbox_min: np.ndarray, bbox_max: np.ndarray, divisor: int) -> float:
    """Taille de voxel dérivée de la diagonale de l'objet : indépendante de l'échelle du scan."""
    bbox_min = np.asarray(bbox_min, dtype=np.float64)
    bbox_max = np.asarray(bbox_max, dtype=np.float64)
    if bbox_min.shape != (3,) or bbox_max.shape != (3,) or not np.isfinite([bbox_min, bbox_max]).all():
        raise ValueError("boîte objet invalide pour le calcul du voxel")
    if np.any(bbox_max < bbox_min) or np.linalg.norm(bbox_max - bbox_min) <= 1e-12:
        raise ValueError("boîte objet vide ou inversée")
    if not isinstance(divisor, (int, np.integer)) or isinstance(divisor, (bool, np.bool_)) or divisor <= 0:
        raise ValueError(f"diviseur de voxel invalide: {divisor!r}")
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    return max(1e-4, diagonal / max(16, int(divisor)))


def _validate_inputs(
    depths: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    masks: np.ndarray | None,
    alphas: np.ndarray | None,
) -> None:
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
        raise ValueError("entrées TSDF incompatibles: " + ", ".join(mismatches))
    if not np.isfinite(colors).all() or not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("couleurs, intrinsèques et extrinsèques TSDF doivent être finies")


def _prepare_depth(
    depth: np.ndarray,
    mask: np.ndarray | None,
    alpha: np.ndarray | None,
    config: TSDFConfig,
) -> np.ndarray:
    """Met à zéro (= ignoré par Open3D) tout ce qui n'est pas une surface fiable."""
    prepared = np.array(depth, dtype=np.float32, copy=True)
    invalid = ~np.isfinite(prepared) | (prepared <= 0) | (prepared > config.depth_max)
    if mask is not None:
        invalid |= ~np.asarray(mask, dtype=bool)
    if alpha is not None:
        invalid |= ~np.isfinite(alpha) | (alpha < config.min_alpha)
    prepared[invalid] = 0.0
    return prepared


def fuse_gpu(
    depths: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    config: TSDFConfig,
    masks: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
    device: str = "CUDA:0",
    log_fn: Callable[[str], None] = lambda _m: None,
) -> Any:
    """Fusion tensorielle. `colors` en float [0,1], (S, H, W, 3)."""
    import open3d as o3d
    import open3d.core as o3c

    depths = np.asarray(depths)
    colors = np.asarray(colors)
    intrinsics = np.asarray(intrinsics)
    extrinsics = np.asarray(extrinsics)
    masks = None if masks is None else np.asarray(masks)
    alphas = None if alphas is None else np.asarray(alphas)
    _validate_inputs(depths, colors, intrinsics, extrinsics, masks, alphas)

    try:
        o3d_device = o3c.Device(device)
        if device.upper().startswith("CUDA") and not o3d.core.cuda.is_available():
            raise RuntimeError("Open3D compilé sans CUDA")
    except Exception as exc:  # noqa: BLE001
        log_fn(f"TSDF GPU indisponible ({exc}), bascule CPU")
        return fuse_cpu(depths, colors, intrinsics, extrinsics, config, masks, alphas, log_fn)

    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=(1, 1, 3),
        voxel_size=config.voxel_size,
        block_resolution=config.block_resolution,
        block_count=config.block_count,
        device=o3d_device,
    )

    integrated = 0
    for index in range(len(depths)):
        depth_np = _prepare_depth(
            depths[index],
            None if masks is None else masks[index],
            None if alphas is None else alphas[index],
            config,
        )
        if not np.any(depth_np > 0):
            continue
        # L'API tensor n'accepte que deux combinaisons : (float, float) ou
        # (uint16, uint8). La profondeur étant en float32, la couleur doit
        # l'être aussi — une couleur uint8 fait échouer `integrate`. Et elle
        # doit rester dans [0, 1] : en 0-255 la fusion réussit, mais les
        # couleurs ressortent en 0-255 et le maillage sort saturé.
        color_np = np.clip(colors[index], 0.0, 1.0).astype(np.float32)

        depth_image = o3d.t.geometry.Image(o3c.Tensor(depth_np, device=o3d_device))
        color_image = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(color_np), device=o3d_device))
        intrinsic = o3c.Tensor(np.ascontiguousarray(intrinsics[index], dtype=np.float64))
        extrinsic = o3c.Tensor(np.ascontiguousarray(extrinsics[index], dtype=np.float64))

        coordinates = grid.compute_unique_block_coordinates(depth_image, intrinsic, extrinsic, 1.0, config.depth_max)
        try:
            grid.integrate(
                coordinates,
                depth_image,
                color_image,
                intrinsic,
                intrinsic,
                extrinsic,
                1.0,
                config.depth_max,
                config.trunc_multiplier,
            )
        except TypeError:  # versions d'Open3D sans le multiplicateur de troncature
            grid.integrate(
                coordinates, depth_image, color_image, intrinsic, intrinsic, extrinsic, 1.0, config.depth_max
            )
        integrated += 1

    if integrated == 0:
        raise ValueError("aucune profondeur exploitable pour la fusion TSDF")

    mesh = grid.extract_triangle_mesh().to_legacy()
    log_fn(
        f"TSDF GPU : {integrated} vues, voxel {config.voxel_size:.4f}, "
        f"{len(mesh.vertices)} sommets / {len(mesh.triangles)} triangles"
    )
    return mesh


def fuse_cpu(
    depths: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    config: TSDFConfig,
    masks: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> Any:
    """Repli historique `ScalableTSDFVolume` (CPU), pour le re-maillage local."""
    import open3d as o3d

    depths = np.asarray(depths)
    colors = np.asarray(colors)
    intrinsics = np.asarray(intrinsics)
    extrinsics = np.asarray(extrinsics)
    masks = None if masks is None else np.asarray(masks)
    alphas = None if alphas is None else np.asarray(alphas)
    _validate_inputs(depths, colors, intrinsics, extrinsics, masks, alphas)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config.voxel_size,
        sdf_trunc=config.voxel_size * config.trunc_multiplier,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    height, width = depths.shape[1:3]
    integrated = 0
    for index in range(len(depths)):
        depth_np = _prepare_depth(
            depths[index],
            None if masks is None else masks[index],
            None if alphas is None else alphas[index],
            config,
        )
        if not np.any(depth_np > 0):
            continue
        color_np = (np.clip(colors[index], 0, 1) * 255).round().astype(np.uint8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color_np)),
            o3d.geometry.Image(depth_np),
            depth_scale=1.0,
            depth_trunc=config.depth_max,
            convert_rgb_to_intensity=False,
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(intrinsics[index][0, 0]),
            float(intrinsics[index][1, 1]),
            float(intrinsics[index][0, 2]),
            float(intrinsics[index][1, 2]),
        )
        volume.integrate(rgbd, intrinsic, np.ascontiguousarray(extrinsics[index], dtype=np.float64))
        integrated += 1

    if integrated == 0:
        raise ValueError("aucune profondeur exploitable pour la fusion TSDF")

    mesh = volume.extract_triangle_mesh()
    log_fn(f"TSDF CPU : {integrated} vues, {len(mesh.vertices)} sommets / {len(mesh.triangles)} triangles")
    return mesh


def fuse(
    depths: np.ndarray,
    colors: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    config: TSDFConfig,
    masks: np.ndarray | None = None,
    alphas: np.ndarray | None = None,
    device: str = "CUDA:0",
    log_fn: Callable[[str], None] = lambda _m: None,
) -> Any:
    if device.upper().startswith("CPU"):
        return fuse_cpu(depths, colors, intrinsics, extrinsics, config, masks, alphas, log_fn)
    return fuse_gpu(depths, colors, intrinsics, extrinsics, config, masks, alphas, device, log_fn)
