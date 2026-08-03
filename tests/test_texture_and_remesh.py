"""Bake de texture et re-maillage local, sur la même vérité terrain synthétique.

Le bake est validé de bout en bout : on peint l'objet avec une fonction connue
`f(position)`, on laisse le pipeline retrouver ces couleurs par rétro-projection,
puis on compare la texture obtenue à `f` au bon endroit. Une erreur de convention
(profondeur, UV, visibilité) casse ce test immédiatement.
"""

from __future__ import annotations

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="Open3D requis (venv Python 3.12)")
xatlas = pytest.importorskip("xatlas", reason="xatlas requis")

from common.glb import read_glb_summary  # noqa: E402
from server.pipeline import cleanup as cleanup_module  # noqa: E402
from server.pipeline import mesh_tsdf  # noqa: E402
from server.pipeline import texture as texture_module  # noqa: E402
from server.pipeline.geometry import SceneNormalization  # noqa: E402
from server.pipeline.remesh import load_raw, remesh  # noqa: E402
from server.pipeline.vggt import VGGTResult  # noqa: E402
from tests.test_meshing import IMAGE_SIZE, ground_truth_mesh, orbit_cameras, render_depths  # noqa: E402


def surface_color(points: np.ndarray) -> np.ndarray:
    """Motif déterministe et à variation lente : permet de comparer sans ambiguïté."""
    points = np.atleast_2d(points)
    return np.stack(
        [
            0.5 + 0.45 * np.sin(4.0 * points[:, 0]),
            0.5 + 0.45 * np.sin(4.0 * points[:, 1] + 1.0),
            0.5 + 0.45 * np.sin(4.0 * points[:, 2] + 2.0),
        ],
        axis=-1,
    ).astype(np.float32)


@pytest.fixture(scope="module")
def painted_scene():
    """Objet connu + images rendues avec des couleurs fonction de la position 3D."""
    mesh = ground_truth_mesh()
    extrinsics, intrinsics = orbit_cameras(count=20)
    depths, masks = render_depths(mesh, extrinsics, intrinsics)

    height, width = IMAGE_SIZE
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    images = np.zeros((len(extrinsics), height, width, 3), dtype=np.float32)
    for index, (extrinsic, intrinsic) in enumerate(zip(extrinsics, intrinsics, strict=True)):
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(np.ascontiguousarray(intrinsic)),
            extrinsic_matrix=o3d.core.Tensor(np.ascontiguousarray(extrinsic)),
            width_px=width,
            height_px=height,
        ).numpy()
        t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        hit = np.isfinite(t_hit)
        positions = rays[..., :3] + np.where(hit, t_hit, 0.0)[..., None] * rays[..., 3:]
        colors = surface_color(positions.reshape(-1, 3)).reshape(height, width, 3)
        images[index] = np.where(hit[..., None], colors, 0.0)

    return {
        "truth": mesh, "extrinsics": extrinsics, "intrinsics": intrinsics,
        "depths": depths, "masks": masks, "images": images,
    }


@pytest.fixture(scope="module")
def reconstructed(painted_scene):
    mesh = mesh_tsdf.fuse(
        depths=painted_scene["depths"], colors=painted_scene["images"],
        intrinsics=painted_scene["intrinsics"], extrinsics=painted_scene["extrinsics"],
        config=mesh_tsdf.TSDFConfig(voxel_size=0.006, depth_max=5.0),
        masks=painted_scene["masks"], device="CPU:0",
    )
    return cleanup_module.clean(
        mesh, cleanup_module.CleanupConfig(target_triangles=30_000, watertight=False), voxel_size=0.006
    )


def test_bake_recovers_the_painted_colors(painted_scene, reconstructed):
    result = texture_module.bake(
        mesh=reconstructed,
        images=painted_scene["images"],
        intrinsics=painted_scene["intrinsics"],
        extrinsics=painted_scene["extrinsics"],
        config=texture_module.TextureConfig(size=1024, max_triangles=30_000),
        masks=painted_scene["masks"],
    )

    assert result.method == "bake"
    assert result.texture is not None and result.texture.shape == (1024, 1024, 3)
    assert result.uvs is not None and len(result.uvs) == len(result.vertices)
    assert result.uvs.min() >= -1e-6 and result.uvs.max() <= 1.0 + 1e-6
    assert result.coverage > 0.75, f"couverture trop faible : {result.coverage:.2%}"

    # Pour un échantillon de faces : la couleur lue au barycentre UV doit
    # correspondre à `surface_color` évaluée au barycentre 3D de la même face.
    rng = np.random.default_rng(0)
    faces = rng.choice(len(result.faces), size=300, replace=False)
    corners_3d = result.vertices[result.faces[faces]]
    corners_uv = result.uvs[result.faces[faces]]
    centres_3d = corners_3d.mean(axis=1)
    centres_uv = corners_uv.mean(axis=1)

    size = result.texture.shape[0]
    columns = np.clip((centres_uv[:, 0] * size - 0.5).round().astype(int), 0, size - 1)
    rows = np.clip((centres_uv[:, 1] * size - 0.5).round().astype(int), 0, size - 1)
    sampled = result.texture[rows, columns].astype(np.float32) / 255.0

    error = np.abs(sampled - surface_color(centres_3d)).mean(axis=1)
    assert np.median(error) < 0.08, f"erreur médiane de couleur : {np.median(error):.3f}"
    assert (error < 0.2).mean() > 0.85, "trop de faces mal peintes"


def test_bake_falls_back_to_vertex_colors_without_xatlas(painted_scene, reconstructed, monkeypatch):
    """xatlas absent ne doit pas faire échouer le job, juste dégrader la sortie."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "xatlas":
            raise ImportError("simulé")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    result = texture_module.bake(
        mesh=reconstructed,
        images=painted_scene["images"],
        intrinsics=painted_scene["intrinsics"],
        extrinsics=painted_scene["extrinsics"],
        config=texture_module.TextureConfig(size=256),
    )

    assert result.method == "vertex"
    assert result.texture is None
    assert result.uvs is None


def test_image_sharpness_ranks_blurred_images_lower():
    sharp = np.zeros((2, 64, 64, 3), dtype=np.float32)
    sharp[0, ::4, :, :] = 1.0  # rayures fines
    sharp[1] = 0.5  # image plate

    scores = texture_module.image_sharpness(sharp)

    assert scores[0] > scores[1]
    assert scores.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Re-maillage local depuis vggt_raw.npz
# ---------------------------------------------------------------------------

def write_raw_npz(path, painted_scene, normalization: SceneNormalization):
    result = VGGTResult(
        depth=painted_scene["depths"],
        depth_conf=np.ones_like(painted_scene["depths"]),
        extrinsics=painted_scene["extrinsics"].astype(np.float32),
        intrinsics=painted_scene["intrinsics"].astype(np.float32),
        images=painted_scene["images"],
        frame_names=[f"frame_{i:05d}.jpg" for i in range(len(painted_scene["depths"]))],
    )
    masks = painted_scene["masks"]
    return result.save(
        path,
        extra={
            "scene_center": normalization.center.astype(np.float32),
            "scene_scale": np.float32(normalization.scale),
            "masks_packed": np.packbits(masks),
            "masks_shape": np.array(masks.shape, dtype=np.int64),
        },
    )


def test_raw_npz_roundtrip_preserves_masks(tmp_path, painted_scene):
    normalization = SceneNormalization(center=np.zeros(3), scale=1.0)
    path = write_raw_npz(tmp_path / "vggt_raw.npz", painted_scene, normalization)

    result, loaded_normalization, masks = load_raw(path)

    assert result.depth.shape == painted_scene["depths"].shape
    assert np.allclose(result.depth, painted_scene["depths"])
    assert np.allclose(result.extrinsics, painted_scene["extrinsics"], atol=1e-5)
    assert loaded_normalization.scale == pytest.approx(1.0)
    assert masks is not None
    # `packbits` travaille sur un tableau aplati : la forme doit être restituée exactement.
    assert masks.shape == painted_scene["masks"].shape
    assert np.array_equal(masks, painted_scene["masks"])


def test_remesh_produces_a_valid_glb(tmp_path, painted_scene):
    normalization = SceneNormalization(center=np.zeros(3), scale=1.0)
    raw = write_raw_npz(tmp_path / "vggt_raw.npz", painted_scene, normalization)
    output = tmp_path / "remesh.glb"

    metrics = remesh(
        npz_path=raw, output=output, backend="tsdf", voxel_divisor=200,
        target_triangles=20_000, watertight=False, log_fn=lambda _m: None,
    )

    assert metrics["triangles"] > 1_000
    summary = read_glb_summary(output)
    assert summary["triangles"] == metrics["triangles"]
    assert "COLOR_0" in summary["attributes"]
    assert output.stat().st_size > 10_000


def test_remesh_scales_to_real_world_size(tmp_path, painted_scene):
    """`--ref-size` doit produire un GLB à la bonne taille métrique."""
    normalization = SceneNormalization(center=np.zeros(3), scale=1.0)
    raw = write_raw_npz(tmp_path / "vggt_raw.npz", painted_scene, normalization)
    output = tmp_path / "remesh.glb"

    remesh(
        npz_path=raw, output=output, backend="tsdf", voxel_divisor=128,
        target_triangles=10_000, watertight=False, ref_size=0.28, log_fn=lambda _m: None,
    )

    summary = read_glb_summary(output)
    extent = max(b - a for a, b in zip(summary["min"], summary["max"], strict=True))
    assert extent == pytest.approx(0.28, rel=0.02)
