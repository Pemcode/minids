"""Raffinement 2DGS : le contrat de forme passé à gsplat.

Le GPU manque ici, mais ce qui a cassé en production n'était pas du calcul :
c'était une **forme de tenseur**. `rasterization_2dgs` accepte des couleurs en
(N, D) dans ses assertions, puis, en mode `RGB+ED`, exécute

    colors = torch.cat((colors, depths[..., None]), dim=-1)

avec `depths` en (C, N). Une couleur en (N, 3) fait donc échouer la
concaténation au premier rendu — après VGGT-Ω, COLMAP et l'écriture du
`.npz`, c'est-à-dire au plus mauvais moment.

Le faux rasteriseur ci-dessous rejoue cette concaténation à l'identique. Il ne
simule aucune physique : il vérifie qu'on appelle gsplat avec des formes qu'il
sait traiter, ce qui est précisément ce qu'un GPU absent empêche de vérifier.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch requis")
pytest.importorskip("open3d", reason="open3d requis (dimensionnement des gaussiennes)")

from server.pipeline.refine_2dgs import RefineConfig, refine  # noqa: E402

SEQUENCE, HEIGHT, WIDTH, POINTS = 2, 8, 8, 64


def gsplat_like_rasterization(
    means, quats, scales, opacities, colors, viewmats, Ks, width, height, render_mode, packed
):
    """Reproduit les formes et la concaténation de gsplat 1.5.3.

    `means2d` et `radii` sortent en (C, N, 2) comme dans cette version — c'est
    ce que doit digérer le contrôle adaptatif de densité.
    """
    cameras = viewmats.shape[0]
    count = means.shape[0]

    assert means.shape == (count, 3)
    assert opacities.shape == (count,)
    assert viewmats.shape == (cameras, 4, 4)
    assert Ks.shape == (cameras, 3, 3)

    # L'assertion de gsplat : (N, D) ou (C, N, D). Les deux sont « acceptées »…
    assert colors.dim() in (2, 3), colors.shape

    # … mais seule (C, N, D) survit à cette ligne, qui est celle de gsplat.
    depths = torch.rand(cameras, count)
    colors = torch.cat((colors, depths[..., None]), dim=-1)
    channels = colors.shape[-1]

    # Les gradients doivent atteindre means2d : c'est sur eux que repose la
    # densification. Le coefficient les porte au-delà de `grad_threshold`.
    means2d = torch.rand(cameras, count, 2, requires_grad=True)

    image = colors.mean(dim=1)[:, None, None, :].expand(cameras, height, width, channels)
    image = image + means2d.abs().mean()
    alphas = opacities.mean()[None, None, None, None].expand(cameras, height, width, 1)
    normals = torch.zeros(cameras, height, width, 3)
    distort = torch.zeros(cameras, height, width, 1)
    median = torch.rand(cameras, height, width, 1)
    meta = {"means2d": means2d, "radii": torch.ones(cameras, count, 2)}
    return image, alphas, normals, normals.clone(), distort, median, meta


@pytest.fixture
def scene():
    rng = np.random.default_rng(0)
    viewmats = np.stack([np.eye(4, dtype=np.float32) for _ in range(SEQUENCE)])
    viewmats[:, 2, 3] = 3.0
    intrinsics = np.stack(
        [np.array([[8.0, 0, WIDTH / 2], [0, 8.0, HEIGHT / 2], [0, 0, 1]], dtype=np.float32)] * SEQUENCE
    )
    return {
        "images": rng.random((SEQUENCE, HEIGHT, WIDTH, 3)).astype(np.float32),
        "masks": np.ones((SEQUENCE, HEIGHT, WIDTH), dtype=bool),
        "depths": np.full((SEQUENCE, HEIGHT, WIDTH), 3.0, dtype=np.float32),
        "depth_weights": np.ones((SEQUENCE, HEIGHT, WIDTH), dtype=np.float32),
        "viewmats": viewmats,
        "intrinsics": intrinsics,
        "init_points": rng.normal(scale=0.3, size=(POINTS, 3)).astype(np.float32),
        "init_colors": rng.random((POINTS, 3)).astype(np.float32),
    }


def run(scene, monkeypatch, rasterization, config=None):
    from server.pipeline import refine_2dgs

    monkeypatch.setattr(refine_2dgs, "_import_gsplat", lambda: rasterization)
    return refine(
        **scene,
        config=config or RefineConfig(iterations=3, densify=False, opacity_reset_every=0),
        device="cpu",
    )


def test_refinement_calls_gsplat_with_per_camera_colors(scene, monkeypatch):
    """Le rendu doit aboutir, entraînement et rendu final compris."""
    result = run(scene, monkeypatch, gsplat_like_rasterization)

    assert result.depths.shape == (SEQUENCE, HEIGHT, WIDTH)
    assert result.alphas.shape == (SEQUENCE, HEIGHT, WIDTH)
    assert result.colors.shape == (SEQUENCE, HEIGHT, WIDTH, 3)
    assert result.num_gaussians == POINTS
    assert np.isfinite(result.depths).all()


def test_densification_survives_the_gsplat_tensor_shapes(scene, monkeypatch):
    """Le contrôle adaptatif de densité démarre à l'itération 500 en production.

    Il consomme `means2d.grad` et `radii` en (C, N, 2), puis reconstruit chaque
    paramètre et son état Adam par concaténation. Autant d'occasions de se
    tromper de forme — et une erreur ici ne se verrait qu'après plusieurs
    minutes de GPU.
    """
    config = RefineConfig(
        iterations=6,
        densify=True,
        densify_from=1,
        densify_every=1,
        densify_until_ratio=1.0,
        opacity_reset_every=0,
        max_gaussians=POINTS * 4,
    )
    result = run(scene, monkeypatch, gsplat_like_rasterization, config)

    assert result.num_gaussians != POINTS, "aucune densification n'a eu lieu : test sans portée"
    assert result.depths.shape == (SEQUENCE, HEIGHT, WIDTH)
    assert np.isfinite(result.colors).all()


def test_colors_are_shaped_per_camera(scene, monkeypatch):
    """Garde-fou explicite : (C, N, 3), jamais (N, 3).

    Sans cette forme, gsplat lève « Tensors must have same number of dimensions ».
    Le test précédent le détecterait, mais pas de façon lisible : celui-ci nomme
    l'invariant.
    """
    seen: list[tuple[int, ...]] = []

    def spy(**kwargs):
        seen.append(tuple(kwargs["colors"].shape))
        return gsplat_like_rasterization(**kwargs)

    run(scene, monkeypatch, spy)

    assert seen, "gsplat n'a jamais été appelé"
    cameras = 1  # une vue par appel, à l'entraînement comme au rendu final
    assert all(shape == (cameras, POINTS, 3) for shape in seen), seen
