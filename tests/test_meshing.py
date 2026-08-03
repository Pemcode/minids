"""Chaîne de maillage sur une vérité terrain synthétique.

On part d'un objet connu (sphère + boîte), on rend des profondeurs *exactes*
depuis des caméras en orbite, puis on demande au pipeline de le reconstruire.
Comme la géométrie d'entrée est parfaite, tout écart mesuré vient du code de
fusion, de nettoyage ou de bake — pas du modèle.

Ces tests nécessitent Open3D, absent des wheels Python 3.13 : ils tournent dans
le venv 3.12 (`.venv312`) et sont ignorés ailleurs.
"""

from __future__ import annotations

import numpy as np
import pytest

o3d = pytest.importorskip("open3d", reason="Open3D requis (venv Python 3.12)")

from server.pipeline import cleanup as cleanup_module  # noqa: E402
from server.pipeline import mesh_poisson, mesh_tsdf  # noqa: E402
from server.pipeline.geometry import unproject  # noqa: E402

IMAGE_SIZE = (240, 320)  # hauteur, largeur
RADIUS = 0.35
ORBIT = 1.6


def ground_truth_mesh():
    """Sphère + petite boîte posée dessus : arêtes vives *et* courbure."""
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=RADIUS, resolution=40)
    box = o3d.geometry.TriangleMesh.create_box(width=0.25, height=0.25, depth=0.25)
    box.translate((-0.125, -RADIUS - 0.25, -0.125))
    mesh = sphere + box
    mesh.compute_vertex_normals()
    return mesh


def orbit_cameras(count: int = 24):
    """Caméras réparties sur deux hauteurs, regardant l'origine (convention COLMAP)."""
    height, width = IMAGE_SIZE
    focal = 300.0
    intrinsic = np.array([[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]])

    extrinsics = []
    for index in range(count):
        angle = 2 * np.pi * index / count
        elevation = 0.25 if index % 2 else -0.35
        center = np.array([ORBIT * np.cos(angle), elevation, ORBIT * np.sin(angle)])
        forward = -center / np.linalg.norm(center)
        right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right, down, forward])
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = -rotation @ center
        extrinsics.append(matrix)

    return np.stack(extrinsics), np.tile(intrinsic, (count, 1, 1))


def render_depths(mesh, extrinsics, intrinsics):
    """Profondeur *z*, telle que l'attend `unproject`.

    `create_rays_pinhole` produit des directions de composante z égale à 1 (elles
    ne sont *pas* normalisées), donc `t_hit` est déjà une profondeur z. Aucune
    conversion à appliquer — s'en convaincre évite une erreur de ~20 % en bord
    d'image.
    """
    height, width = IMAGE_SIZE
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    depths, masks = [], []
    for extrinsic, intrinsic in zip(extrinsics, intrinsics, strict=True):
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(np.ascontiguousarray(intrinsic)),
            extrinsic_matrix=o3d.core.Tensor(np.ascontiguousarray(extrinsic)),
            width_px=width,
            height_px=height,
        )
        t_hit = scene.cast_rays(rays)["t_hit"].numpy()
        hit = np.isfinite(t_hit)
        depths.append(np.where(hit, t_hit, 0.0).astype(np.float32))
        masks.append(hit)

    return np.stack(depths), np.stack(masks)


@pytest.fixture(scope="module")
def synthetic():
    mesh = ground_truth_mesh()
    extrinsics, intrinsics = orbit_cameras()
    depths, masks = render_depths(mesh, extrinsics, intrinsics)
    colors = np.zeros((*depths.shape, 3), dtype=np.float32)
    colors[..., 0] = 0.8
    colors[..., 1] = 0.4
    return {
        "truth": mesh,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "depths": depths,
        "masks": masks,
        "colors": colors,
    }


def chamfer_percent(mesh, truth) -> float:
    """Chamfer symétrique, en % de la diagonale de la vérité terrain."""
    sampled = mesh.sample_points_uniformly(number_of_points=40_000)
    reference = truth.sample_points_uniformly(number_of_points=40_000)
    forward = np.asarray(sampled.compute_point_cloud_distance(reference)).mean()
    backward = np.asarray(reference.compute_point_cloud_distance(sampled)).mean()
    bounds = np.asarray(truth.vertices)
    diagonal = np.linalg.norm(bounds.max(axis=0) - bounds.min(axis=0))
    return 100.0 * (forward + backward) / 2 / diagonal


def test_unprojection_matches_ground_truth_surface(synthetic):
    """Verrouille la convention de profondeur : les points dé-projetés doivent
    tomber sur la surface de départ, pas à côté."""
    points = unproject(synthetic["depths"][0], synthetic["intrinsics"][0], synthetic["extrinsics"][0])
    valid = points[synthetic["masks"][0]]

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(synthetic["truth"]))
    distances = scene.compute_distance(o3d.core.Tensor(valid.astype(np.float32))).numpy()

    assert distances.max() < 1e-3, f"écart max à la surface : {distances.max():.5f}"


def test_tsdf_reconstructs_the_object(synthetic):
    voxel = mesh_tsdf.voxel_size_from_bbox(
        np.asarray(synthetic["truth"].vertices).min(axis=0),
        np.asarray(synthetic["truth"].vertices).max(axis=0),
        divisor=256,
    )
    mesh = mesh_tsdf.fuse(
        depths=synthetic["depths"],
        colors=synthetic["colors"],
        intrinsics=synthetic["intrinsics"],
        extrinsics=synthetic["extrinsics"],
        config=mesh_tsdf.TSDFConfig(voxel_size=voxel, depth_max=5.0),
        masks=synthetic["masks"],
        device="CPU:0",
    )

    assert len(mesh.triangles) > 5_000
    error = chamfer_percent(mesh, synthetic["truth"])
    assert error < 1.0, f"TSDF trop éloigné de la vérité terrain : {error:.3f} % de la diagonale"

    reconstructed = np.asarray(mesh.vertices)
    truth = np.asarray(synthetic["truth"].vertices)
    assert np.allclose(reconstructed.min(axis=0), truth.min(axis=0), atol=0.05)
    assert np.allclose(reconstructed.max(axis=0), truth.max(axis=0), atol=0.05)


def test_poisson_reconstructs_the_object(synthetic):
    points, colors = mesh_poisson.point_cloud_from_depths(
        synthetic["depths"], synthetic["colors"], synthetic["intrinsics"], synthetic["extrinsics"],
        masks=synthetic["masks"], depth_max=5.0, stride=2,
    )
    assert len(points) > 10_000

    mesh = mesh_poisson.reconstruct(
        points, colors, synthetic["extrinsics"],
        mesh_poisson.PoissonConfig(depth=8, voxel_size=0.005),
    )

    assert len(mesh.triangles) > 1_000
    error = chamfer_percent(mesh, synthetic["truth"])
    assert error < 2.5, f"Poisson trop éloigné : {error:.3f} %"


def test_cleanup_removes_stray_islands(synthetic):
    """Un îlot parasite loin de l'objet doit disparaître, sans abîmer l'objet."""
    mesh = mesh_tsdf.fuse(
        depths=synthetic["depths"], colors=synthetic["colors"],
        intrinsics=synthetic["intrinsics"], extrinsics=synthetic["extrinsics"],
        config=mesh_tsdf.TSDFConfig(voxel_size=0.008, depth_max=5.0),
        masks=synthetic["masks"], device="CPU:0",
    )
    island = o3d.geometry.TriangleMesh.create_sphere(radius=0.02, resolution=6)
    island.translate((0.9, 0.9, 0.9))
    polluted = mesh + island
    triangles_before = len(polluted.triangles)

    cleaned = cleanup_module.clean(
        polluted,
        cleanup_module.CleanupConfig(target_triangles=40_000, smooth_iterations=3, watertight=False),
        voxel_size=0.008,
    )

    vertices = np.asarray(cleaned.vertices)
    assert len(cleaned.triangles) < triangles_before
    assert vertices.max() < 0.7, "l'îlot parasite est resté dans le maillage"
    assert chamfer_percent(cleaned, synthetic["truth"]) < 1.5


def test_cleanup_respects_triangle_budget(synthetic):
    mesh = mesh_tsdf.fuse(
        depths=synthetic["depths"], colors=synthetic["colors"],
        intrinsics=synthetic["intrinsics"], extrinsics=synthetic["extrinsics"],
        config=mesh_tsdf.TSDFConfig(voxel_size=0.004, depth_max=5.0),
        masks=synthetic["masks"], device="CPU:0",
    )
    cleaned = cleanup_module.clean(
        mesh, cleanup_module.CleanupConfig(target_triangles=8_000, watertight=False), voxel_size=0.004
    )

    assert len(cleaned.triangles) <= 8_600  # la décimation quadrique vise sans garantir l'exactitude
    metrics = cleanup_module.mesh_metrics(cleaned)
    assert metrics["triangles"] == len(cleaned.triangles)
    assert metrics["surface_area"] > 0
