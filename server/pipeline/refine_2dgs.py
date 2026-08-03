"""Raffinement 2D Gaussian Splatting (gsplat) puis rendu des profondeurs.

Pourquoi cette étape : le TSDF n'invente rien, il moyenne ce qu'on lui donne.
La profondeur brute de VGGT-Ω est cohérente mais bruitée à l'échelle du pixel ;
optimiser des surfels 2D sur les images réelles, puis *rendre* la profondeur,
donne des cartes nettement plus propres — c'est exactement ce que font 2DGS,
GOF et GS2Mesh avant leur propre fusion TSDF.

Trois écarts assumés par rapport à un entraînement 2DGS canonique :

* **Initialisation dense** depuis le nuage VGGT-Ω (et non des points SfM épars),
  ce qui rend la densification adaptative bien moins critique.
* **Pas d'harmoniques sphériques** : on cherche de la géométrie, la couleur
  vue-dépendante nuirait au bake de texture qui suit.
* **Perte de profondeur ancrée sur VGGT-Ω**, décroissante : elle stabilise les
  premières itérations, puis laisse la photométrie décider.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("minids.refine")


@dataclass
class RefineConfig:
    iterations: int = 12_000
    max_gaussians: int = 500_000
    init_points: int = 300_000
    densify: bool = True
    densify_from: int = 500
    densify_until_ratio: float = 0.6
    densify_every: int = 200
    grad_threshold: float = 2e-4
    prune_opacity: float = 0.05
    opacity_reset_every: int = 3000
    lambda_ssim: float = 0.2
    lambda_alpha: float = 0.5
    lambda_normal: float = 0.05
    lambda_distort: float = 100.0
    lambda_depth: float = 0.5
    depth_decay_ratio: float = 0.5
    lr_means: float = 1.6e-4
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_colors: float = 2.5e-3
    seed: int = 0


@dataclass
class RefineResult:
    depths: np.ndarray  # (S, H, W) profondeur rendue, 0 = vide
    alphas: np.ndarray  # (S, H, W)
    colors: np.ndarray  # (S, H, W, 3) rendu, dans [0, 1]
    num_gaussians: int
    losses: dict[str, float] = field(default_factory=dict)


def _import_gsplat() -> Any:
    try:
        from gsplat import rasterization_2dgs
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gsplat est requis pour le raffinement (pip install gsplat). "
            "Utiliser --no-refine pour rester sur la profondeur VGGT-Ω brute."
        ) from exc
    return rasterization_2dgs


def _unpack_render(outputs: Any) -> dict[str, Any]:
    """`rasterization_2dgs` renvoie 7 valeurs ; on tolère les variantes."""
    if isinstance(outputs, dict):
        return outputs
    values = list(outputs)
    meta = values[-1]
    keys = ["colors", "alphas", "normals", "surf_normals", "distort", "median_depth"]
    unpacked: dict[str, Any] = {"meta": meta}
    # strict=False assumé : une variante de gsplat peut renvoyer moins de sorties.
    for key, value in zip(keys, values[:-1], strict=False):
        unpacked[key] = value
    if "colors" not in unpacked or "alphas" not in unpacked:
        raise RuntimeError(f"sortie gsplat inattendue ({len(values)} valeurs)")
    return unpacked


def _nearest_neighbour_scale(points: np.ndarray) -> np.ndarray:
    """Distance au plus proche voisin, pour dimensionner les gaussiennes initiales."""
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    distances = np.asarray(cloud.compute_nearest_neighbor_distance())
    distances = np.where(np.isfinite(distances) & (distances > 1e-6), distances, np.median(distances))
    return np.clip(distances, 1e-4, 0.05)


def _ssim(prediction: Any, target: Any) -> Any:
    """SSIM 11×11 gaussien, sur des tenseurs (1, 3, H, W)."""
    import torch
    import torch.nn.functional as F

    window_size, sigma, channels = 11, 1.5, prediction.shape[1]
    coords = torch.arange(window_size, dtype=prediction.dtype, device=prediction.device) - window_size // 2
    gauss = torch.exp(-(coords**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    window = (gauss[:, None] @ gauss[None, :]).expand(channels, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(prediction, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channels)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1 = F.conv2d(prediction * prediction, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2 = F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1, c2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1 + sigma2 + c2))
    return ssim_map.mean()


class GaussianModel:
    """Paramètres bruts + optimiseur, sans harmoniques sphériques."""

    def __init__(self, points: np.ndarray, colors: np.ndarray, config: RefineConfig, device: str) -> None:
        import torch

        self.config = config
        self.device = device
        scales = _nearest_neighbour_scale(points)

        def parameter(array: np.ndarray) -> Any:
            return torch.nn.Parameter(torch.tensor(array, dtype=torch.float32, device=device))

        self.means = parameter(points)
        self.scales = parameter(np.log(np.repeat(scales[:, None], 3, axis=1)))
        quats = np.zeros((len(points), 4), dtype=np.float32)
        quats[:, 0] = 1.0
        self.quats = parameter(quats)
        self.opacities = parameter(np.full(len(points), _inverse_sigmoid(0.5), dtype=np.float32))
        self.colors = parameter(_inverse_sigmoid_array(np.clip(colors, 1e-3, 1 - 1e-3)))

        self.optimizer = torch.optim.Adam(
            [
                {"params": [self.means], "lr": config.lr_means, "name": "means"},
                {"params": [self.scales], "lr": config.lr_scales, "name": "scales"},
                {"params": [self.quats], "lr": config.lr_quats, "name": "quats"},
                {"params": [self.opacities], "lr": config.lr_opacities, "name": "opacities"},
                {"params": [self.colors], "lr": config.lr_colors, "name": "colors"},
            ],
            eps=1e-15,
        )
        self._grad_accum = torch.zeros(len(points), device=device)
        self._grad_count = torch.zeros(len(points), device=device)

    def __len__(self) -> int:
        return self.means.shape[0]

    def render_parameters(self) -> dict[str, Any]:
        import torch

        return {
            "means": self.means,
            "quats": torch.nn.functional.normalize(self.quats, dim=-1),
            "scales": torch.exp(self.scales),
            "opacities": torch.sigmoid(self.opacities),
            "colors": torch.sigmoid(self.colors),
        }

    # -- contrôle adaptatif de densité ---------------------------------
    def accumulate_gradients(self, means2d: Any, radii: Any) -> None:
        import torch

        if means2d.grad is None:
            return
        grad = means2d.grad.detach()
        if grad.ndim == 3:  # (C, N, 2)
            grad = grad[0]
        norms = torch.norm(grad, dim=-1)
        visible = radii.detach()
        if visible.ndim == 3:
            visible = visible[0]
        elif visible.ndim == 2:
            visible = visible[0] if visible.shape[0] == 1 else visible
        if visible.ndim == 2:
            visible = visible.max(dim=-1).values
        mask = visible > 0
        self._grad_accum[mask] += norms[mask]
        self._grad_count[mask] += 1

    def _prune_and_extend(self, keep: Any, extensions: dict[str, Any]) -> None:
        """Applique un masque de conservation puis concatène de nouvelles gaussiennes."""
        import torch

        for group in self.optimizer.param_groups:
            name = group["name"]
            parameter = group["params"][0]
            state = self.optimizer.state.get(parameter)
            new_values = torch.cat([parameter.detach()[keep], extensions[name]], dim=0)
            new_parameter = torch.nn.Parameter(new_values)
            if state is not None:
                zeros = torch.zeros((len(extensions[name]), *parameter.shape[1:]), device=self.device)
                state["exp_avg"] = torch.cat([state["exp_avg"][keep], zeros], dim=0)
                state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"][keep], zeros], dim=0)
                del self.optimizer.state[parameter]
                self.optimizer.state[new_parameter] = state
            group["params"] = [new_parameter]
            setattr(self, name, new_parameter)

        count = len(self.means)
        self._grad_accum = torch.zeros(count, device=self.device)
        self._grad_count = torch.zeros(count, device=self.device)

    def densify_and_prune(self, scene_extent: float) -> tuple[int, int]:
        import torch

        config = self.config
        average_grad = self._grad_accum / self._grad_count.clamp(min=1)
        opacity = torch.sigmoid(self.opacities.detach())
        scales = torch.exp(self.scales.detach())
        largest = scales[:, :2].max(dim=1).values

        selected = (average_grad > config.grad_threshold) & (self._grad_count > 0)
        headroom = max(0, config.max_gaussians - len(self))
        if headroom == 0:
            selected = torch.zeros_like(selected)

        small = selected & (largest <= 0.01 * scene_extent)
        big = selected & (largest > 0.01 * scene_extent)
        if int(small.sum() + big.sum()) > headroom:  # on privilégie les clones, moins destructeurs
            indices = torch.nonzero(small | big).squeeze(-1)[:headroom]
            keep_selected = torch.zeros_like(small)
            keep_selected[indices] = True
            small &= keep_selected
            big &= keep_selected

        extensions: dict[str, Any] = {}
        clone_index = torch.nonzero(small).squeeze(-1)
        split_index = torch.nonzero(big).squeeze(-1)

        noise = torch.randn((len(split_index), 3), device=self.device) * scales[split_index] * 0.5
        extensions["means"] = torch.cat([self.means.detach()[clone_index], self.means.detach()[split_index] + noise])
        split_scales = torch.log(scales[split_index] / 1.6)
        extensions["scales"] = torch.cat([self.scales.detach()[clone_index], split_scales])
        for name in ("quats", "opacities", "colors"):
            source = getattr(self, name).detach()
            extensions[name] = torch.cat([source[clone_index], source[split_index]])

        keep = opacity.squeeze() > config.prune_opacity
        keep &= largest < 0.5 * scene_extent  # gaussiennes géantes = artefacts de fond
        if int(keep.sum()) < 100:  # garde-fou : ne jamais vider le modèle
            keep = torch.ones_like(keep)
        # Les nouvelles gaussiennes issues d'un split remplacent leurs parents.
        keep[split_index] = False

        pruned = int((~keep).sum())
        self._prune_and_extend(keep, extensions)
        return len(clone_index) + len(split_index), pruned

    def reset_opacity(self) -> None:
        import torch

        with torch.no_grad():
            ceiling = torch.full_like(self.opacities, _inverse_sigmoid(0.1))
            self.opacities.copy_(torch.minimum(self.opacities, ceiling))


def _inverse_sigmoid(value: float) -> float:
    return math.log(value / (1.0 - value))


def _inverse_sigmoid_array(array: np.ndarray) -> np.ndarray:
    return np.log(array / (1.0 - array)).astype(np.float32)


def refine(
    images: np.ndarray,
    masks: np.ndarray,
    depths: np.ndarray,
    depth_weights: np.ndarray,
    viewmats: np.ndarray,
    intrinsics: np.ndarray,
    init_points: np.ndarray,
    init_colors: np.ndarray,
    config: RefineConfig,
    device: str = "cuda",
    log_fn: Callable[[str], None] = lambda _m: None,
    progress_fn: Callable[[float], None] = lambda _f: None,
    should_stop: Callable[[], None] = lambda: None,
) -> RefineResult:
    """Optimise les surfels puis rend profondeur/alpha/couleur pour chaque vue."""
    import torch

    rasterization_2dgs = _import_gsplat()
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    sequence, height, width = depths.shape
    if len(init_points) > config.init_points:
        selection = rng.choice(len(init_points), size=config.init_points, replace=False)
        init_points, init_colors = init_points[selection], init_colors[selection]

    model = GaussianModel(init_points, init_colors, config, device)
    log_fn(f"{len(model)} gaussiennes initiales, {config.iterations} itérations")

    images_t = torch.tensor(images, dtype=torch.float32, device=device)
    masks_t = torch.tensor(masks.astype(np.float32), dtype=torch.float32, device=device)
    depths_t = torch.tensor(depths, dtype=torch.float32, device=device)
    weights_t = torch.tensor(depth_weights, dtype=torch.float32, device=device)
    viewmats_t = torch.tensor(viewmats, dtype=torch.float32, device=device)
    ks_t = torch.tensor(intrinsics, dtype=torch.float32, device=device)

    scene_extent = float(np.percentile(np.linalg.norm(init_points - init_points.mean(0), axis=1), 95)) or 1.0
    densify_until = int(config.iterations * config.densify_until_ratio)
    depth_until = max(1, int(config.iterations * config.depth_decay_ratio))
    losses: dict[str, float] = {}

    for iteration in range(config.iterations):
        should_stop()
        view = int(rng.integers(sequence))
        parameters = model.render_parameters()

        outputs = rasterization_2dgs(
            means=parameters["means"],
            quats=parameters["quats"],
            scales=parameters["scales"],
            opacities=parameters["opacities"],
            colors=parameters["colors"],
            viewmats=viewmats_t[view : view + 1],
            Ks=ks_t[view : view + 1],
            width=width,
            height=height,
            render_mode="RGB+ED",
            packed=False,
        )
        render = _unpack_render(outputs)
        meta = render["meta"]
        means2d = meta.get("means2d") if isinstance(meta, dict) else None
        if means2d is not None and means2d.requires_grad:
            means2d.retain_grad()

        rendered = render["colors"][0]
        rgb = rendered[..., :3].clamp(0, 1)
        rendered_depth = rendered[..., 3] if rendered.shape[-1] > 3 else None
        alpha = render["alphas"][0, ..., 0]

        target_rgb = images_t[view]
        mask = masks_t[view]
        mask_sum = mask.sum().clamp(min=1.0)

        # Photométrie, restreinte à l'objet (le fond n'a aucune raison d'être appris).
        l1 = ((rgb - target_rgb).abs().mean(dim=-1) * mask).sum() / mask_sum
        ssim = _ssim((rgb * mask[..., None]).permute(2, 0, 1)[None], (target_rgb * mask[..., None]).permute(2, 0, 1)[None])
        loss = (1.0 - config.lambda_ssim) * l1 + config.lambda_ssim * (1.0 - ssim)

        # L'opacité doit suivre le masque : c'est ce qui élimine les gaussiennes de fond.
        loss = loss + config.lambda_alpha * (alpha - mask).abs().mean()

        if "normals" in render and "surf_normals" in render and config.lambda_normal > 0:
            normals = render["normals"][0]
            surf_normals = render["surf_normals"][0]
            consistency = (1.0 - (normals * surf_normals).sum(dim=-1)) * mask
            loss = loss + config.lambda_normal * consistency.mean()

        if "distort" in render and config.lambda_distort > 0:
            loss = loss + config.lambda_distort * (render["distort"][0].squeeze(-1) * mask).mean()

        if rendered_depth is not None and config.lambda_depth > 0 and iteration < depth_until:
            decay = 1.0 - iteration / depth_until
            weight = weights_t[view] * mask
            depth_loss = ((rendered_depth - depths_t[view]).abs() * weight).sum() / weight.sum().clamp(min=1.0)
            loss = loss + config.lambda_depth * decay * depth_loss

        model.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.densify and means2d is not None and iteration < densify_until:
            radii = meta.get("radii") if isinstance(meta, dict) else None
            if radii is not None:
                model.accumulate_gradients(means2d, radii)

        model.optimizer.step()
        _decay_means_lr(model, iteration, config)

        if (
            config.densify
            and config.densify_from <= iteration < densify_until
            and iteration % config.densify_every == 0
            and iteration > 0
        ):
            added, pruned = model.densify_and_prune(scene_extent)
            if iteration % (config.densify_every * 5) == 0:
                log_fn(f"iter {iteration}: {len(model)} gaussiennes (+{added} / -{pruned})")
        if config.opacity_reset_every and iteration > 0 and iteration % config.opacity_reset_every == 0:
            model.reset_opacity()

        if iteration % 50 == 0:
            progress_fn(iteration / config.iterations)
            losses = {"total": float(loss.detach()), "l1": float(l1.detach())}

    log_fn(f"optimisation terminée : {len(model)} gaussiennes, perte {losses.get('total', float('nan')):.4f}")
    rendered = render_views(model, viewmats_t, ks_t, width, height, rasterization_2dgs)
    rendered.losses = losses
    return rendered


def _decay_means_lr(model: GaussianModel, iteration: int, config: RefineConfig) -> None:
    """Décroissance exponentielle du pas sur les positions (×0.01 en fin d'entraînement)."""
    factor = 0.01 ** (iteration / max(1, config.iterations))
    for group in model.optimizer.param_groups:
        if group["name"] == "means":
            group["lr"] = config.lr_means * factor


def render_views(
    model: GaussianModel,
    viewmats: Any,
    ks: Any,
    width: int,
    height: int,
    rasterization_2dgs: Any,
) -> RefineResult:
    """Rendu final : profondeur médiane (surface) plutôt qu'espérée (moyenne floue)."""
    import torch

    depths, alphas, colors = [], [], []
    parameters = model.render_parameters()
    with torch.no_grad():
        for index in range(viewmats.shape[0]):
            outputs = rasterization_2dgs(
                means=parameters["means"],
                quats=parameters["quats"],
                scales=parameters["scales"],
                opacities=parameters["opacities"],
                colors=parameters["colors"],
                viewmats=viewmats[index : index + 1],
                Ks=ks[index : index + 1],
                width=width,
                height=height,
                render_mode="RGB+ED",
                packed=False,
            )
            render = _unpack_render(outputs)
            rendered = render["colors"][0]
            alpha = render["alphas"][0, ..., 0]

            median = render.get("median_depth")
            depth = None
            if median is not None:
                depth = median[0]
                depth = depth[..., 0] if depth.ndim == 3 else depth
            if depth is None or not torch.isfinite(depth).any() or float(depth.max()) <= 0:
                depth = rendered[..., 3] if rendered.shape[-1] > 3 else torch.zeros_like(alpha)

            depths.append(depth.float().cpu().numpy())
            alphas.append(alpha.float().cpu().numpy())
            colors.append(rendered[..., :3].clamp(0, 1).float().cpu().numpy())

    return RefineResult(
        depths=np.stack(depths).astype(np.float32),
        alphas=np.stack(alphas).astype(np.float32),
        colors=np.stack(colors).astype(np.float32),
        num_gaussians=len(model),
    )
