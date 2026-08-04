"""Tests CPU ciblés des contrats et replis du pipeline.

Ces tests restent indépendants d'Open3D, de VGGT-Ω et de gsplat afin que les
erreurs d'entrée soient détectées sur toutes les versions de Python supportées.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from server.pipeline import cleanup, fake, mesh_tsdf, refine_2dgs, texture, vggt
from server.pipeline import run as run_module
from server.pipeline.colmap_export import build_sparse_cloud, rotmat_to_quat, write_colmap
from server.pipeline.geometry import SceneNormalization
from server.pipeline.remesh import load_raw
from server.pipeline.segment import Segmentation, segment


def small_result(sequence: int = 1, height: int = 2, width: int = 3) -> vggt.VGGTResult:
    intrinsic = np.array([[2.0, 0.0, width / 2], [0.0, 2.0, height / 2], [0.0, 0.0, 1.0]])
    return vggt.VGGTResult(
        depth=np.ones((sequence, height, width), dtype=np.float32),
        depth_conf=np.ones((sequence, height, width), dtype=np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (sequence, 1, 1)),
        intrinsics=np.tile(intrinsic.astype(np.float32), (sequence, 1, 1)),
        images=np.full((sequence, height, width, 3), 0.25, dtype=np.float32),
        frame_names=[f"vue_{index}.png" for index in range(sequence)],
    )


def test_vggt_npz_uses_unicode_and_loads_legacy_names_without_pickle(tmp_path):
    result = small_result()
    current = result.save(tmp_path / "current.npz")
    with np.load(current, allow_pickle=False) as data:
        assert data["frame_names"].dtype.kind == "U"
    assert vggt.VGGTResult.load(current).frame_names == ["vue_0.png"]

    legacy = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy,
        depth=result.depth,
        depth_conf=result.depth_conf,
        extrinsics=result.extrinsics,
        intrinsics=result.intrinsics,
        images=(result.images * 255).astype(np.uint8),
        frame_names=np.array(["ancien.png"], dtype=object),
    )
    loaded = vggt.VGGTResult.load(legacy)
    assert loaded.frame_names == ["frame_00000.jpg"]


def test_vggt_npz_rejects_reserved_or_object_extras(tmp_path):
    result = small_result()
    with pytest.raises(ValueError, match="réservés"):
        result.save(tmp_path / "bad.npz", extra={"depth": np.ones(1)})
    with pytest.raises(TypeError, match="objets Python"):
        result.save(tmp_path / "bad.npz", extra={"payload": np.array([object()], dtype=object)})


def test_load_raw_rejects_masks_with_incompatible_shape(tmp_path):
    result = small_result()
    path = result.save(
        tmp_path / "bad_masks.npz",
        extra={
            "scene_center": np.zeros(3, dtype=np.float32),
            "scene_scale": np.float32(1.0),
            "masks_packed": np.packbits(np.ones(4, dtype=bool)),
            "masks_shape": np.array([1, 2, 2], dtype=np.int64),
        },
    )
    with pytest.raises(ValueError, match="incompatible"):
        load_raw(path)


def test_colmap_export_writes_matching_observations_and_tracks(tmp_path):
    result = small_result(height=2, width=3)
    segmentation = Segmentation(np.ones_like(result.depth, dtype=bool), method="none")
    cloud = build_sparse_cloud(result, SceneNormalization(np.zeros(3), 1.0), segmentation)
    sparse = write_colmap(tmp_path, result, SceneNormalization(np.zeros(3), 1.0), cloud)

    image_lines = [
        line for line in (sparse / "images.txt").read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]
    assert len(image_lines) == 2
    observations = image_lines[1].split()
    assert len(observations) == len(cloud.points) * 3

    point_lines = [
        line for line in (sparse / "points3D.txt").read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]
    assert len(point_lines) == len(cloud.points)
    assert all(len(line.split()) == 10 for line in point_lines)  # 8 champs + IMAGE_ID + POINT2D_IDX
    assert [int(line.split()[-1]) for line in point_lines] == list(range(len(cloud.points)))


def test_sparse_cloud_ignores_nonfinite_confidence():
    result = small_result(height=2, width=2)
    result.depth_conf[0, 0, 0] = np.inf
    result.depth_conf[0, 0, 1] = np.nan
    segmentation = Segmentation(np.ones_like(result.depth, dtype=bool), method="none")

    cloud = build_sparse_cloud(result, SceneNormalization(np.zeros(3), 1.0), segmentation)

    assert len(cloud.points) == 2


def test_rotation_to_quaternion_rejects_reflection():
    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="rotation propre"):
        rotmat_to_quat(reflection)


def test_texture_dilation_does_not_wrap_across_edges():
    source = np.zeros((3, 3, 3), dtype=np.float32)
    source[0, 0] = [1.0, 0.0, 0.0]
    painted = np.zeros((3, 3), dtype=bool)
    painted[0, 0] = True

    dilated = texture._dilate(source, painted, iterations=1)

    assert np.array_equal(dilated[0, 1], [1.0, 0.0, 0.0])
    assert np.array_equal(dilated[1, 0], [1.0, 0.0, 0.0])
    assert np.array_equal(dilated[0, -1], [0.0, 0.0, 0.0])
    assert np.array_equal(dilated[-1, 0], [0.0, 0.0, 0.0])


def test_texture_size_is_capped_before_large_allocations():
    with pytest.raises(ValueError, match="taille de texture"):
        texture.TextureConfig(size=4097)


def test_texture_visibility_uses_pixel_centres_not_bankers_rounding():
    positions = np.array([[1.0, 0.0, 1.0]])  # projection (1.5, 0.5)
    normals = np.array([[-1.0, 0.0, -1.0]]) / np.sqrt(2.0)
    scores = np.zeros(1, dtype=np.float32)
    colors = np.zeros((1, 3), dtype=np.float32)
    image = np.zeros((1, 3, 3), dtype=np.float32)
    image[0, 1] = [0.0, 1.0, 0.0]
    image[0, 2] = [1.0, 0.0, 0.0]
    mask = np.zeros((1, 3), dtype=bool)
    mask[0, 1] = True
    intrinsic = np.array([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])

    texture._accumulate_view(
        positions,
        normals,
        scores,
        colors,
        np.ones((1, 3), dtype=np.float32),
        image,
        mask,
        intrinsic,
        np.eye(4),
        np.zeros(3),
        1.0,
        1e-6,
        texture.TextureConfig(size=8, chunk=8),
    )

    assert scores[0] > 0
    assert np.allclose(colors[0], [0.0, 1.0, 0.0])


def test_small_images_have_finite_neutral_sharpness():
    scores = texture.image_sharpness(np.zeros((2, 2, 1, 3), dtype=np.float32))
    assert np.array_equal(scores, np.ones(2, dtype=np.float32))


def test_invalid_pipeline_options_fail_before_optional_dependencies(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="aucune frame"):
        vggt.run_inference([], "", "cpu", "", tmp_path)
    with pytest.raises(ValueError, match="inconnue"):
        segment(small_result(), SceneNormalization(np.zeros(3), 1.0), method="typo")
    with pytest.raises(ValueError, match="prompt"):
        segment(small_result(), SceneNormalization(np.zeros(3), 1.0), method="sam3", prompt="  ")
    monkeypatch.setenv("MINIDS_FAKE_SPEED", "vite")
    with pytest.raises(ValueError, match="MINIDS_FAKE_SPEED"):
        fake._stage_duration("mesh")


def test_segmentation_stats_cannot_override_contract_fields():
    segmentation = Segmentation(
        np.ones((1, 2, 2), dtype=bool),
        method="geometric",
        stats={"method": "faux", "coverage": -1},
    )
    payload = segmentation.to_dict()
    assert payload["method"] == "geometric"
    assert payload["coverage"] == 1.0


def test_tsdf_preparation_rejects_nonfinite_alpha_and_bad_bbox():
    depth = np.ones((2, 2), dtype=np.float32)
    alpha = np.ones((2, 2), dtype=np.float32)
    alpha[0, 0] = np.nan
    prepared = mesh_tsdf._prepare_depth(depth, None, alpha, mesh_tsdf.TSDFConfig())
    assert prepared[0, 0] == 0
    with pytest.raises(ValueError, match="inversée"):
        mesh_tsdf.voxel_size_from_bbox(np.ones(3), np.zeros(3), 128)
    planar = mesh_tsdf.voxel_size_from_bbox(np.zeros(3), np.array([1.0, 1.0, 0.0]), 128)
    assert planar > 0


def test_cleanup_requires_a_positive_triangle_budget():
    with pytest.raises(ValueError, match="budget de triangles"):
        cleanup.CleanupConfig(target_triangles=0)


def test_mesh_metrics_are_scaled_to_export_units():
    metrics = {
        "bbox_min": [1.0, 2.0, 3.0],
        "bbox_max": [2.0, 4.0, 6.0],
        "bbox_diagonal": 4.0,
        "surface_area": 5.0,
        "volume": 6.0,
        "triangles": 12,
    }

    scaled = cleanup.scale_metrics_for_export(metrics, 2.0)

    assert scaled["bbox_min"] == [2.0, 4.0, 6.0]
    assert scaled["bbox_max"] == [4.0, 8.0, 12.0]
    assert scaled["bbox_diagonal"] == 8.0
    assert scaled["surface_area"] == 20.0
    assert scaled["volume"] == 48.0
    assert scaled["triangles"] == 12
    assert scaled["export_scale"] == 2.0
    assert metrics["bbox_min"] == [1.0, 2.0, 3.0]


def test_refine_config_and_render_unpack_fail_early():
    with pytest.raises(ValueError, match="densify_every"):
        refine_2dgs.RefineConfig(densify_every=0)
    with pytest.raises(RuntimeError, match="clés"):
        refine_2dgs._unpack_render({"colors": np.zeros(1)})


def test_prepare_frames_is_deterministic_and_validates_count(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for index in range(5):
        (frames_dir / f"frame_{index}.jpg").write_bytes(b"")
    reporter = SimpleNamespace(log=lambda _message: None)

    selected = run_module._prepare_frames("frames", frames_dir, frames_dir, 3, reporter)
    assert [path.name for path in selected] == ["frame_0.jpg", "frame_2.jpg", "frame_4.jpg"]
    with pytest.raises(ValueError, match="nombre de frames"):
        run_module._prepare_frames("frames", frames_dir, frames_dir, 0, reporter)


def test_refinement_weights_ignore_nan_and_handle_constant_confidence(monkeypatch):
    result = small_result(height=2, width=2)
    result.depth_conf[:] = 5.0
    result.depth_conf[0, 0, 0] = np.nan
    captured: dict[str, object] = {}

    def fake_refine(**kwargs):
        captured.update(kwargs)
        return "raffiné"

    monkeypatch.setattr(refine_2dgs, "refine", fake_refine)
    monkeypatch.setattr(run_module, "_select_device", lambda _device: "cpu")
    output = run_module._run_refinement(
        result,
        SimpleNamespace(masks=np.ones_like(result.depth, dtype=bool)),
        result.depth,
        result.extrinsics,
        SimpleNamespace(points=np.zeros((1, 3)), colors=np.zeros((1, 3), dtype=np.uint8)),
        {"gs_iters": 1},
        SimpleNamespace(device="cuda"),
        SimpleNamespace(log=lambda _message: None, progress=lambda _value: None, check_cancelled=lambda: None),
    )

    assert output == "raffiné"
    weights = captured["depth_weights"]
    assert np.isfinite(weights).all()
    assert weights[0, 0, 0] == 0
    assert np.all(weights[np.isfinite(result.depth_conf)] == 1)


def test_mesh_device_selection_is_case_insensitive(monkeypatch):
    result = small_result(height=2, width=2)
    monkeypatch.setattr(mesh_tsdf, "fuse", lambda **kwargs: kwargs["device"])

    device = run_module._build_mesh(
        "tsdf",
        result,
        SimpleNamespace(masks=np.ones_like(result.depth, dtype=bool)),
        None,
        result.depth,
        result.extrinsics,
        0.01,
        SimpleNamespace(device="CUDA:0"),
        SimpleNamespace(log=lambda _message: None),
    )

    assert device == "CUDA:0"


def test_tsdf_gpu_availability_check_is_case_insensitive(monkeypatch):
    result = small_result(height=2, width=2)
    fake_open3d = SimpleNamespace(core=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    fake_core = SimpleNamespace(Device=lambda _device: object())
    monkeypatch.setitem(sys.modules, "open3d", fake_open3d)
    monkeypatch.setitem(sys.modules, "open3d.core", fake_core)
    monkeypatch.setattr(mesh_tsdf, "fuse_cpu", lambda *args, **kwargs: "cpu")

    fused = mesh_tsdf.fuse_gpu(
        result.depth,
        result.images,
        result.intrinsics,
        result.extrinsics,
        mesh_tsdf.TSDFConfig(voxel_size=0.01),
        device="cuda:0",
    )

    assert fused == "cpu"
