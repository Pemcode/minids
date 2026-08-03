"""Conventions de caméra : une erreur ici ne se voit qu'à la fin, sur un maillage
retourné ou explosé. On les verrouille par aller-retour."""

from __future__ import annotations

import numpy as np
import pytest

from server.pipeline.geometry import (
    SceneNormalization,
    camera_centers,
    compute_normalization,
    project,
    to_4x4,
    unproject,
)


def make_camera(distance: float = 3.0, angle: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    """Caméra regardant l'origine depuis un cercle, en convention world-to-camera."""
    center = np.array([distance * np.cos(angle), 0.4, distance * np.sin(angle)])
    forward = -center / np.linalg.norm(center)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])  # lignes = axes caméra
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ center
    intrinsic = np.array([[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]])
    return extrinsic, intrinsic


def test_camera_center_recovered():
    extrinsic, _ = make_camera()
    expected = -extrinsic[:3, :3].T @ extrinsic[:3, 3]
    assert np.allclose(camera_centers(extrinsic[None])[0], expected)


def test_unproject_then_project_is_identity():
    extrinsic, intrinsic = make_camera()
    depth = np.full((240, 320), 2.5)
    depth[100:140, 100:140] = 3.1  # un relief, pour ne pas tester qu'un plan

    points = unproject(depth, intrinsic, extrinsic)
    pixels, recovered_depth = project(points.reshape(-1, 3), intrinsic, extrinsic)

    height, width = depth.shape
    us, vs = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    assert np.allclose(pixels[:, 0], us.ravel(), atol=1e-6)
    assert np.allclose(pixels[:, 1], vs.ravel(), atol=1e-6)
    assert np.allclose(recovered_depth, depth.ravel(), atol=1e-6)


def test_to_4x4_accepts_both_shapes():
    extrinsic, _ = make_camera()
    assert to_4x4(extrinsic[None]).shape == (1, 4, 4)
    assert to_4x4(extrinsic[None, :3, :]).shape == (1, 4, 4)
    with pytest.raises(ValueError):
        to_4x4(np.zeros((1, 2, 4)))


def test_normalization_keeps_geometry_consistent():
    """Après normalisation, dé-projeter la profondeur mise à l'échelle doit
    redonner exactement les points du monde mis à l'échelle."""
    extrinsic, intrinsic = make_camera()
    depth = np.full((60, 80), 2.5)
    depth[20:40, 30:50] = 3.0
    points = unproject(depth, intrinsic, extrinsic)

    normalization = compute_normalization(points.reshape(-1, 3))
    new_extrinsic = normalization.apply_extrinsics(extrinsic[None])[0]
    new_depth = normalization.apply_depth(depth)

    expected = normalization.apply_points(points.reshape(-1, 3))
    obtained = unproject(new_depth, intrinsic, new_extrinsic).reshape(-1, 3)
    assert np.allclose(obtained, expected, atol=1e-9)


def test_normalization_rejects_empty_cloud():
    with pytest.raises(ValueError):
        compute_normalization(np.full((10, 3), np.nan))


def test_normalization_scale_is_robust_to_outliers():
    points = np.random.default_rng(0).normal(scale=0.1, size=(5000, 3))
    points[0] = [1000.0, 1000.0, 1000.0]  # un point aberrant ne doit pas piloter l'échelle
    normalization = compute_normalization(points)
    assert normalization.scale < 1.0
    assert isinstance(normalization, SceneNormalization)
