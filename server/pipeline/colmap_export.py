"""Export au format COLMAP + nuage d'initialisation pour le splatting.

Format texte volontairement (pas de dépendance à pycolmap pour écrire) : il est
lu tel quel par nerfstudio, gsplat, 3DGS et l'outillage COLMAP, et il reste
inspectable à la main quand une pose part de travers.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import SceneNormalization, confidence_threshold, project, to_4x4, unproject
from .segment import Segmentation
from .vggt import VGGTResult


@dataclass
class SparseCloud:
    points: np.ndarray  # (N, 3) repère monde normalisé
    colors: np.ndarray  # (N, 3) uint8
    frame_index: np.ndarray  # (N,) image d'origine


def rotmat_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Matrice de rotation → quaternion COLMAP (qw, qx, qy, qz)."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"rotation invalide: {rotation.shape}")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=5e-3) or np.linalg.det(rotation) <= 0:
        raise ValueError("la pose contient une matrice qui n'est pas une rotation propre")
    trace = float(np.trace(rotation))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def build_sparse_cloud(
    result: VGGTResult,
    normalization: SceneNormalization,
    segmentation: Segmentation,
    max_points: int = 200_000,
    conf_quantile: float = 0.4,
    seed: int = 0,
) -> SparseCloud:
    """Échantillonne les pixels confiants et masqués pour initialiser les gaussiennes."""
    if not isinstance(max_points, (int, np.integer)) or isinstance(max_points, (bool, np.bool_)) or max_points <= 0:
        raise ValueError(f"nombre maximal de points invalide: {max_points!r}")
    if not np.isfinite(conf_quantile) or not 0.0 <= conf_quantile <= 1.0:
        raise ValueError(f"quantile de confiance invalide: {conf_quantile!r}")
    if segmentation.masks.shape != result.depth.shape:
        raise ValueError(f"masques {segmentation.masks.shape} incompatibles avec les profondeurs {result.depth.shape}")
    threshold = confidence_threshold(result.depth_conf, 1.0 - conf_quantile)
    all_points, all_colors, all_frames = [], [], []

    for index in range(len(result.depth)):
        depth = result.depth[index]
        keep = (
            segmentation.masks[index]
            & np.isfinite(result.depth_conf[index])
            & (result.depth_conf[index] >= threshold)
            & np.isfinite(depth)
            & (depth > 0)
        )
        if not keep.any():
            continue
        points = unproject(depth, result.intrinsics[index], result.extrinsics[index])
        all_points.append(normalization.apply_points(points[keep]))
        all_colors.append((np.clip(result.images[index][keep], 0, 1) * 255).round().astype(np.uint8))
        all_frames.append(np.full(int(keep.sum()), index, dtype=np.int32))

    if not all_points:
        raise ValueError("nuage d'initialisation vide (masque ou confiance trop stricts)")

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    frames = np.concatenate(all_frames)

    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        selection = np.sort(rng.choice(len(points), size=max_points, replace=False))
        points, colors, frames = points[selection], colors[selection], frames[selection]

    return SparseCloud(points=points, colors=colors, frame_index=frames)


def write_colmap(
    directory: Path,
    result: VGGTResult,
    normalization: SceneNormalization,
    cloud: SparseCloud,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> Path:
    """Écrit `sparse/0/{cameras,images,points3D}.txt` dans `directory`."""
    points = np.asarray(cloud.points, dtype=np.float64)
    colors = np.asarray(cloud.colors)
    frame_index = np.asarray(cloud.frame_index)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError(f"nuage COLMAP invalide: {points.shape}")
    if (
        colors.shape != points.shape
        or not np.issubdtype(colors.dtype, np.number)
        or not np.isfinite(colors).all()
        or np.any(colors < 0)
        or np.any(colors > 255)
    ):
        raise ValueError(f"couleurs COLMAP invalides: {colors.shape}")
    if frame_index.shape != (len(points),) or not np.issubdtype(frame_index.dtype, np.integer):
        raise ValueError(f"indices de frame COLMAP invalides: {frame_index.shape}")
    if len(points) and (int(frame_index.min()) < 0 or int(frame_index.max()) >= len(result.depth)):
        raise ValueError("indice de frame hors bornes dans le nuage COLMAP")

    sparse = directory / "sparse" / "0"
    sparse.mkdir(parents=True, exist_ok=True)
    height, width = result.image_size
    extrinsics = normalization.apply_extrinsics(result.extrinsics)

    with (sparse / "cameras.txt").open("w", encoding="utf-8") as handle:
        handle.write("# Camera list with one line of data per camera:\n")
        handle.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for index, intrinsic in enumerate(result.intrinsics):
            fx, fy = float(intrinsic[0, 0]), float(intrinsic[1, 1])
            cx, cy = float(intrinsic[0, 2]), float(intrinsic[1, 2])
            handle.write(f"{index + 1} PINHOLE {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    # Un point3D COLMAP ne peut référencer une observation inexistante. On
    # reprojette donc chaque point dans sa frame d'origine et on écrit les deux
    # côtés de la relation (POINTS2D dans images.txt et TRACK dans points3D.txt).
    observations: dict[int, tuple[int, int]] = {}
    image_rows: list[tuple[str, str]] = []
    with (sparse / "images.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Image list with two lines of data per image:\n")
        handle.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[]\n")
        for index, matrix in enumerate(to_4x4(extrinsics)):
            quat = rotmat_to_quat(matrix[:3, :3])
            t = matrix[:3, 3]
            name = result.frame_names[index] if index < len(result.frame_names) else f"frame_{index:05d}.jpg"
            if "\n" in name or "\r" in name:
                raise ValueError(f"nom de frame COLMAP invalide: {name!r}")

            point_ids = np.flatnonzero(frame_index == index)
            pixels, depths = project(points[point_ids], result.intrinsics[index], matrix)
            valid = np.isfinite(pixels).all(axis=1) & np.isfinite(depths) & (depths > 0)
            point_ids = point_ids[valid]
            pixels = pixels[valid]
            tokens = []
            for point2d_index, (point_id, pixel) in enumerate(zip(point_ids, pixels, strict=True)):
                observations[int(point_id)] = (index + 1, point2d_index)
                tokens.append(f"{pixel[0]:.6f} {pixel[1]:.6f} {int(point_id) + 1}")
            image_rows.append(
                (
                    f"{index + 1} {quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f} "
                    f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {index + 1} {name}",
                    " ".join(tokens),
                )
            )
        for image_row, points2d_row in image_rows:
            handle.write(f"{image_row}\n{points2d_row}\n")

    with (sparse / "points3D.txt").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# 3D point list with one line of data per point:\n")
        handle.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        rows = zip(cloud.points, cloud.colors, cloud.frame_index, strict=True)
        for index, (point, color, _frame) in enumerate(rows):
            track = observations.get(index)
            track_text = "" if track is None else f" {track[0]} {track[1]}"
            handle.write(
                f"{index + 1} {point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 1.0{track_text}\n"
            )

    log_fn(f"COLMAP écrit : {len(result.depth)} caméras, {len(cloud.points)} points → {sparse}")
    return sparse
