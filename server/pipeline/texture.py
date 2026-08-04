"""Bake de texture : dépliage UV (xatlas) + rétro-projection depuis les vues.

C'est l'étape qui fait la différence visuelle. Les couleurs par sommet du TSDF
sont limitées par la densité du maillage ; une texture 2k échantillonne l'image
source à sa vraie résolution.

Pour chaque texel : position 3D par coordonnées barycentriques, test de
visibilité par lancer de rayons, puis choix de la meilleure vue selon
`frontalité × netteté`. Les texels non vus sont dilatés depuis leurs voisins
pour éviter les coutures noires en bord de chart.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import camera_centers

log = logging.getLogger("minids.texture")


@dataclass
class TextureConfig:
    size: int = 2048
    padding: int = 6
    visibility_tolerance: float = 0.01  # fraction de la diagonale de l'objet
    min_cos: float = 0.10
    chunk: int = 1_000_000
    max_triangles: int = 300_000

    def __post_init__(self) -> None:
        if (
            not isinstance(self.size, (int, np.integer))
            or isinstance(self.size, (bool, np.bool_))
            or not 1 <= self.size <= 4096
        ):
            raise ValueError(f"taille de texture invalide: {self.size!r}")
        if (
            not isinstance(self.padding, (int, np.integer))
            or isinstance(self.padding, (bool, np.bool_))
            or self.padding < 0
        ):
            raise ValueError(f"padding de texture invalide: {self.padding!r}")
        if not np.isfinite(self.visibility_tolerance) or self.visibility_tolerance < 0:
            raise ValueError(f"tolérance de visibilité invalide: {self.visibility_tolerance!r}")
        if not np.isfinite(self.min_cos) or not -1.0 <= self.min_cos <= 1.0:
            raise ValueError(f"cosinus minimal invalide: {self.min_cos!r}")
        if not isinstance(self.chunk, (int, np.integer)) or isinstance(self.chunk, (bool, np.bool_)) or self.chunk <= 0:
            raise ValueError(f"taille de chunk invalide: {self.chunk!r}")
        if (
            not isinstance(self.max_triangles, (int, np.integer))
            or isinstance(self.max_triangles, (bool, np.bool_))
            or self.max_triangles <= 0
        ):
            raise ValueError(f"budget de triangles invalide: {self.max_triangles!r}")


@dataclass
class BakeResult:
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray | None
    texture: np.ndarray | None  # (size, size, 3) uint8
    vertex_colors: np.ndarray | None
    method: str
    coverage: float = 0.0


def image_sharpness(images: np.ndarray) -> np.ndarray:
    """Variance du laplacien par image, normalisée — sert à départager les vues."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
        raise ValueError(f"images de forme inattendue: {images.shape}, (S, H, W, 3) attendu")
    if not np.isfinite(images).all():
        raise ValueError("images non finies pour le calcul de netteté")
    if images.shape[1] < 3 or images.shape[2] < 3:
        return np.ones(len(images), dtype=np.float32)
    grey = images.mean(axis=-1)
    laplacian = (
        -4.0 * grey[:, 1:-1, 1:-1] + grey[:, :-2, 1:-1] + grey[:, 2:, 1:-1] + grey[:, 1:-1, :-2] + grey[:, 1:-1, 2:]
    )
    scores = laplacian.var(axis=(1, 2))
    maximum = float(scores.max()) if scores.size else 1.0
    return scores / maximum if maximum > 0 else np.ones_like(scores)


def bake(
    mesh: Any,
    images: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    config: TextureConfig,
    masks: np.ndarray | None = None,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> BakeResult:
    """Déplie puis peint le maillage. Retombe sur les couleurs par sommet si besoin."""
    import open3d as o3d

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.triangles, dtype=np.int64)
    images = np.asarray(images, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    masks = None if masks is None else np.asarray(masks, dtype=bool)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f"sommets invalides pour le bake: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces):
        raise ValueError(f"faces invalides pour le bake: {faces.shape}")
    if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
        raise ValueError("indices de faces hors bornes pour le bake")
    if images.ndim != 4 or images.shape[-1] != 3 or len(images) == 0:
        raise ValueError(f"images de forme inattendue: {images.shape}")
    sequence, height, width, _channels = images.shape
    expected = {"intrinsèques": (sequence, 3, 3), "extrinsèques": (sequence, 4, 4)}
    observed = {"intrinsèques": intrinsics.shape, "extrinsèques": extrinsics.shape}
    if masks is not None:
        expected["masques"] = (sequence, height, width)
        observed["masques"] = masks.shape
    mismatches = [
        f"{name}={observed[name]} (attendu {shape})" for name, shape in expected.items() if observed[name] != shape
    ]
    if mismatches:
        raise ValueError("entrées de bake incompatibles: " + ", ".join(mismatches))
    if not np.isfinite(images).all() or not np.isfinite(intrinsics).all() or not np.isfinite(extrinsics).all():
        raise ValueError("images et caméras du bake doivent être finies")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    if len(faces) > config.max_triangles:
        log_fn(f"décimation avant bake ({len(faces)} → {config.max_triangles} triangles)")
        mesh = mesh.simplify_quadric_decimation(config.max_triangles)
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)
        vertex_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    try:
        import xatlas
    except ImportError:
        log_fn("xatlas absent : export en couleurs par sommet")
        return BakeResult(vertices, faces, normals, None, None, vertex_colors, "vertex")

    try:
        vmapping, indices, uvs = xatlas.parametrize(vertices, faces)
    except Exception as exc:  # noqa: BLE001 - xatlas peut rejeter un maillage dégénéré
        log_fn(f"dépliage xatlas échoué ({type(exc).__name__}: {exc}) : export en couleurs par sommet")
        return BakeResult(vertices, faces, normals, None, None, vertex_colors, "vertex")
    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    if (
        vmapping.ndim != 1
        or not len(vmapping)
        or indices.ndim != 2
        or indices.shape[1:] != (3,)
        or uvs.shape != (len(vmapping), 2)
        or not len(indices)
        or not np.isfinite(uvs).all()
        or int(vmapping.min()) < 0
        or int(vmapping.max()) >= len(vertices)
        or int(indices.min()) < 0
        or int(indices.max()) >= len(vmapping)
    ):
        log_fn("sortie xatlas invalide : export en couleurs par sommet")
        return BakeResult(vertices, faces, normals, None, None, vertex_colors, "vertex")
    vertices_uv = vertices[vmapping]
    normals_uv = normals[vmapping]
    colors_uv = None if vertex_colors is None else vertex_colors[vmapping]
    faces_uv = indices
    log_fn(f"atlas UV : {len(vertices_uv)} sommets, {len(faces_uv)} triangles")

    face_id, bary = _rasterize_uv(uvs, faces_uv, config.size)
    valid = face_id >= 0
    if not valid.any():
        log_fn("rastérisation UV vide : export en couleurs par sommet")
        return BakeResult(vertices, faces, normals, None, None, vertex_colors, "vertex")

    corner = vertices_uv[faces_uv[face_id[valid]]]  # (M, 3, 3)
    positions = np.einsum("mij,mi->mj", corner, bary[valid])
    corner_normals = normals_uv[faces_uv[face_id[valid]]]
    texel_normals = np.einsum("mij,mi->mj", corner_normals, bary[valid])
    lengths = np.linalg.norm(texel_normals, axis=1, keepdims=True)
    texel_normals = texel_normals / np.where(lengths < 1e-12, 1.0, lengths)

    diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0
    tolerance = config.visibility_tolerance * diagonal

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    sharpness = image_sharpness(images)
    centers = camera_centers(extrinsics)

    best_score = np.zeros(len(positions), dtype=np.float32)
    best_color = np.zeros((len(positions), 3), dtype=np.float32)

    for view in range(len(images)):
        depth_map = _raycast_depth(scene, intrinsics[view], extrinsics[view], images.shape[1], images.shape[2])
        _accumulate_view(
            positions,
            texel_normals,
            best_score,
            best_color,
            depth_map,
            images[view],
            None if masks is None else masks[view],
            intrinsics[view],
            extrinsics[view],
            centers[view],
            float(sharpness[view]),
            tolerance,
            config,
        )

    texture = np.zeros((config.size, config.size, 3), dtype=np.float32)
    painted = np.zeros((config.size, config.size), dtype=bool)
    texture[valid] = best_color
    painted[valid] = best_score > 0
    coverage = float(painted.sum()) / max(1, int(valid.sum()))

    if colors_uv is not None:  # trous internes : on retombe sur la couleur du TSDF
        fallback = np.zeros_like(texture)
        corner_colors = colors_uv[faces_uv[face_id[valid]]]
        fallback[valid] = np.einsum("mij,mi->mj", corner_colors, bary[valid])
        missing = valid & ~painted
        texture[missing] = fallback[missing]
        painted |= missing

    texture = _dilate(texture, painted, config.padding)
    log_fn(f"texture {config.size}² bakée, {coverage:.1%} des texels vus par au moins une caméra")

    return BakeResult(
        vertices=vertices_uv,
        faces=faces_uv,
        normals=normals_uv,
        uvs=uvs,
        texture=(np.clip(texture, 0, 1) * 255).astype(np.uint8),
        vertex_colors=colors_uv,
        method="bake",
        coverage=coverage,
    )


def _rasterize_uv(uvs: np.ndarray, faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Rastérise l'atlas : identifiant de face + coordonnées barycentriques par texel."""
    face_id = np.full((size, size), -1, dtype=np.int32)
    bary = np.zeros((size, size, 3), dtype=np.float32)

    # Convention glTF : u vers la droite, v vers le bas, origine au coin haut-gauche.
    xs = uvs[:, 0] * size - 0.5
    ys = uvs[:, 1] * size - 0.5

    for index, (i0, i1, i2) in enumerate(faces):
        x0, x1, x2 = xs[i0], xs[i1], xs[i2]
        y0, y1, y2 = ys[i0], ys[i1], ys[i2]
        min_x = max(0, int(np.floor(min(x0, x1, x2))))
        max_x = min(size - 1, int(np.ceil(max(x0, x1, x2))))
        min_y = max(0, int(np.floor(min(y0, y1, y2))))
        max_y = min(size - 1, int(np.ceil(max(y0, y1, y2))))
        if min_x > max_x or min_y > max_y:
            continue

        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denominator) < 1e-12:
            continue

        grid_x = np.arange(min_x, max_x + 1, dtype=np.float64)[None, :]
        grid_y = np.arange(min_y, max_y + 1, dtype=np.float64)[:, None]
        l0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        l1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue

        window = (slice(min_y, max_y + 1), slice(min_x, max_x + 1))
        target_id = face_id[window]
        target_bary = bary[window]
        target_id[inside] = index
        # l0/l1/l2 sont déjà en (ny, nx) par diffusion de grid_x/grid_y.
        target_bary[inside] = np.stack([l0, l1, l2], axis=-1).astype(np.float32)[inside]

    return face_id, bary


def _raycast_depth(scene: Any, intrinsic: np.ndarray, extrinsic: np.ndarray, height: int, width: int) -> np.ndarray:
    """Profondeur *z* caméra→surface, par pixel (inf si rien touché).

    `create_rays_pinhole` renvoie des directions **non normalisées**, de composante
    z égale à 1 (mesuré : norme 1.000 au centre, 1.201 dans un coin). `t_hit` est
    donc déjà une profondeur z, pas une distance euclidienne — confondre les deux
    fausserait le test de visibilité de ~20 % en périphérie et trouerait la texture.
    """
    import open3d.core as o3c

    rays = scene.create_rays_pinhole(
        intrinsic_matrix=o3c.Tensor(np.ascontiguousarray(intrinsic, dtype=np.float64)),
        extrinsic_matrix=o3c.Tensor(np.ascontiguousarray(extrinsic, dtype=np.float64)),
        width_px=int(width),
        height_px=int(height),
    )
    return scene.cast_rays(rays)["t_hit"].numpy()


def _accumulate_view(
    positions: np.ndarray,
    normals: np.ndarray,
    best_score: np.ndarray,
    best_color: np.ndarray,
    depth_map: np.ndarray,
    image: np.ndarray,
    mask: np.ndarray | None,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    center: np.ndarray,
    sharpness: float,
    tolerance: float,
    config: TextureConfig,
) -> None:
    """Met à jour la meilleure couleur pour chaque texel, vue par vue."""
    from .geometry import project

    height, width = image.shape[:2]
    for start in range(0, len(positions), config.chunk):
        stop = min(start + config.chunk, len(positions))
        chunk_positions = positions[start:stop]
        chunk_normals = normals[start:stop]

        to_camera = center[None, :] - chunk_positions
        distance = np.linalg.norm(to_camera, axis=1)
        safe_distance = np.where(distance < 1e-9, 1e-9, distance)
        cosine = np.einsum("ij,ij->i", to_camera / safe_distance[:, None], chunk_normals)

        pixels, depth = project(chunk_positions, intrinsic, extrinsic)
        # Les intrinsèques utilisent des centres de pixels à n + 0.5 : le
        # texel contenant la projection est donc floor(coordonnée), pas round.
        columns = np.floor(pixels[:, 0]).astype(np.int64)
        rows = np.floor(pixels[:, 1]).astype(np.int64)
        candidate = (
            (cosine > config.min_cos) & (depth > 0) & (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        )
        if not candidate.any():
            continue

        rows_c = np.clip(rows, 0, height - 1)
        columns_c = np.clip(columns, 0, width - 1)
        hit = depth_map[rows_c, columns_c]
        # `hit` et `depth` sont tous deux des profondeurs z : comparables directement.
        candidate &= np.isfinite(hit) & (np.abs(hit - depth) <= tolerance)
        if mask is not None:
            candidate &= mask[rows_c, columns_c]
        if not candidate.any():
            continue

        score = (cosine * sharpness).astype(np.float32)
        score[~candidate] = 0.0
        improved = score > best_score[start:stop]
        if not improved.any():
            continue

        sampled = _bilinear_sample(image, pixels[improved])
        indices = np.nonzero(improved)[0] + start
        best_score[indices] = score[improved]
        best_color[indices] = sampled


def _bilinear_sample(image: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Échantillonnage bilinéaire de l'image source aux coordonnées demandées."""
    image = np.asarray(image)
    pixels = np.asarray(pixels, dtype=np.float64)
    if image.ndim != 3 or image.shape[-1] != 3 or not image.shape[0] or not image.shape[1]:
        raise ValueError(f"image de forme inattendue: {image.shape}")
    if pixels.ndim != 2 or pixels.shape[1:] != (2,) or not np.isfinite(pixels).all():
        raise ValueError(f"pixels de forme inattendue: {pixels.shape}")
    height, width = image.shape[:2]
    x = np.clip(pixels[:, 0] - 0.5, 0, width - 1)
    y = np.clip(pixels[:, 1] - 0.5, 0, height - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = (x - x0)[:, None]
    wy = (y - y0)[:, None]
    top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
    return (top * (1 - wy) + bottom * wy).astype(np.float32)


def _dilate(texture: np.ndarray, painted: np.ndarray, iterations: int) -> np.ndarray:
    """Étale les texels peints sur leurs voisins vides (anti-couture)."""
    texture = texture.copy()
    filled = painted.copy()
    for _ in range(max(0, iterations)):
        empty = ~filled
        if not empty.any():
            break
        accumulated = np.zeros_like(texture)
        counts = np.zeros(filled.shape, dtype=np.float32)
        for shift_y, shift_x in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            shifted_texture = np.roll(texture, (shift_y, shift_x), axis=(0, 1))
            shifted_filled = np.roll(filled, (shift_y, shift_x), axis=(0, 1))
            if shift_y > 0:
                shifted_filled[:shift_y, :] = False
            elif shift_y < 0:
                shifted_filled[shift_y:, :] = False
            if shift_x > 0:
                shifted_filled[:, :shift_x] = False
            elif shift_x < 0:
                shifted_filled[:, shift_x:] = False
            accumulated += shifted_texture * shifted_filled[..., None]
            counts += shifted_filled
        usable = empty & (counts > 0)
        texture[usable] = accumulated[usable] / counts[usable][:, None]
        filled |= usable
    return texture
