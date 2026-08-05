"""Isolation de l'objet : masques par image.

Deux stratégies, la seconde servant toujours de filet :

* **sam3** — SAM 3 (Meta, nov. 2025) avec un prompt texte (« the sneaker »).
  Segmente le concept demandé, y compris quand il touche le plan de support.
* **geometric** — aucune dépendance modèle : plan de support par RANSAC, plus
  gros amas DBSCAN, boîte englobante, reprojection. Suffisant pour un objet
  posé et filmé en orbite, et parfaitement déterministe.

Le résultat sert trois fois : perte masquée du 2DGS, restriction de la fusion
TSDF, et découpe finale du maillage.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import SceneNormalization, confidence_threshold, largest_cluster_mask, unproject
from .vggt import VGGTResult

log = logging.getLogger("minids.segment")

SAM3_MODEL_ID = "facebook/sam3"


@dataclass
class Segmentation:
    masks: np.ndarray  # (S, H, W) booléens, True = objet
    method: str
    bbox_min: np.ndarray | None = None  # boîte objet, repère monde normalisé
    bbox_max: np.ndarray | None = None
    plane: np.ndarray | None = None  # (a, b, c, d) du plan de support
    stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.masks = np.asarray(self.masks, dtype=bool)
        if self.masks.ndim != 3 or any(size <= 0 for size in self.masks.shape):
            raise ValueError(f"masques de forme inattendue: {self.masks.shape}, (S, H, W) attendu")
        if not self.method:
            raise ValueError("méthode de segmentation vide")
        if (self.bbox_min is None) != (self.bbox_max is None):
            raise ValueError("les deux bornes de la boîte objet sont requises ensemble")
        if self.bbox_min is not None and self.bbox_max is not None:
            self.bbox_min = np.asarray(self.bbox_min, dtype=np.float64)
            self.bbox_max = np.asarray(self.bbox_max, dtype=np.float64)
            if self.bbox_min.shape != (3,) or self.bbox_max.shape != (3,):
                raise ValueError("la boîte objet doit contenir deux vecteurs 3D")
            if not np.isfinite(self.bbox_min).all() or not np.isfinite(self.bbox_max).all():
                raise ValueError("boîte objet non finie")
            if np.any(self.bbox_max < self.bbox_min) or np.linalg.norm(self.bbox_max - self.bbox_min) <= 1e-12:
                raise ValueError("boîte objet vide ou inversée")
        if self.plane is not None:
            self.plane = np.asarray(self.plane, dtype=np.float64)
            if self.plane.shape != (4,) or not np.isfinite(self.plane).all():
                raise ValueError("plan de support invalide")

    @property
    def coverage(self) -> float:
        return float(self.masks.mean()) if self.masks.size else 0.0

    def save_pngs(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            return
        for index, mask in enumerate(self.masks):
            Image.fromarray((mask * 255).astype(np.uint8)).save(directory / f"mask_{index:05d}.png")

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.stats,
            "method": self.method,
            "coverage": round(self.coverage, 4),
            "bbox_min": None if self.bbox_min is None else self.bbox_min.tolist(),
            "bbox_max": None if self.bbox_max is None else self.bbox_max.tolist(),
            "plane": None if self.plane is None else self.plane.tolist(),
        }


def segment(
    result: VGGTResult,
    normalization: SceneNormalization,
    method: str = "auto",
    prompt: str | None = None,
    conf_quantile: float = 0.5,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> Segmentation:
    """Point d'entrée : choisit la stratégie et garantit un résultat exploitable."""
    allowed = {"auto", "sam3", "geometric", "none"}
    if method not in allowed:
        raise ValueError(f"méthode de segmentation inconnue: {method!r} ({', '.join(sorted(allowed))})")
    if not np.isfinite(conf_quantile) or not 0.0 <= conf_quantile <= 1.0:
        raise ValueError(f"quantile de confiance invalide: {conf_quantile!r}")
    prompt = prompt.strip() if prompt else None
    if method == "sam3" and not prompt:
        raise ValueError("un prompt texte non vide est requis pour la segmentation SAM 3")

    sequence, height, width = result.depth.shape

    if method == "none":
        return Segmentation(masks=np.ones((sequence, height, width), dtype=bool), method="none")

    if method in {"auto", "sam3"} and prompt:
        try:
            masks = _segment_sam3(result, prompt, log_fn)
            segmentation = _refine_with_geometry(result, normalization, masks, "sam3", conf_quantile, log_fn)
            if segmentation.coverage > 0.002:
                return segmentation
            log_fn(f"masques SAM 3 quasi vides (couverture {segmentation.coverage:.4f}), repli géométrique")
        except Exception as exc:  # noqa: BLE001 - dépendance externe, on ne casse pas le job
            if method == "sam3":
                raise
            log_fn(f"SAM 3 indisponible ({type(exc).__name__}: {exc}), repli géométrique")
    elif method in {"auto", "sam3"} and not prompt:
        log_fn("pas de prompt texte fourni : segmentation géométrique")

    return _segment_geometric(result, normalization, conf_quantile, log_fn)


# ---------------------------------------------------------------------------
# SAM 3
# ---------------------------------------------------------------------------


def _segment_sam3(result: VGGTResult, prompt: str, log_fn: Callable[[str], None]) -> np.ndarray:
    import torch
    import transformers
    from PIL import Image

    model_cls = getattr(transformers, "Sam3Model", None)
    if model_cls is None:
        raise ImportError("transformers ne fournit pas Sam3Model (mettre à jour transformers)")
    processor = transformers.AutoProcessor.from_pretrained(SAM3_MODEL_ID)
    model = model_cls.from_pretrained(SAM3_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    log_fn(f"SAM 3 chargé sur {device}, prompt: {prompt!r}")

    sequence, height, width = result.depth.shape
    masks = np.zeros((sequence, height, width), dtype=bool)
    try:
        for index in range(sequence):
            pixels = (np.clip(result.images[index], 0, 1) * 255).round().astype(np.uint8)
            image = Image.fromarray(pixels)
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            parsed = processor.post_process_instance_segmentation(
                outputs, threshold=0.4, mask_threshold=0.5, target_sizes=[(height, width)]
            )[0]
            instance_masks = parsed.get("masks")
            scores = parsed.get("scores")
            if instance_masks is None or len(instance_masks) == 0:
                continue
            best = int(torch.as_tensor(scores).argmax()) if scores is not None else 0
            mask = np.asarray(torch.as_tensor(instance_masks[best]).cpu()).squeeze()
            if mask.shape != (height, width):
                raise ValueError(f"masque SAM 3 de forme inattendue: {mask.shape}")
            masks[index] = mask > 0.5
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    log_fn(f"SAM 3 : {int(masks.any(axis=(1, 2)).sum())}/{sequence} images avec détection")
    return masks


def _refine_with_geometry(
    result: VGGTResult,
    normalization: SceneNormalization,
    masks: np.ndarray,
    method: str,
    conf_quantile: float,
    log_fn: Callable[[str], None],
) -> Segmentation:
    """Complète des masques 2D par la boîte englobante 3D de l'objet.

    SAM 3 donne des masques, pas de géométrie. Or la suite du pipeline a besoin
    d'une boîte : elle fixe la taille de voxel du TSDF et sert à découper le
    maillage final. On la déduit des points masqués et confiants.
    """
    masks = _clean_masks(masks)
    threshold = confidence_threshold(result.depth_conf, 1.0 - conf_quantile)

    collected = []
    for index in range(len(result.depth)):
        depth = result.depth[index]
        keep = masks[index] & (result.depth_conf[index] >= threshold) & np.isfinite(depth) & (depth > 0)
        if not keep.any():
            continue
        points = unproject(depth, result.intrinsics[index], result.extrinsics[index])
        collected.append(normalization.apply_points(points[keep][::3]))

    if not collected:
        log_fn("masques sans profondeur exploitable : pas de boîte objet")
        return Segmentation(masks=masks, method=method)

    points = np.concatenate(collected)
    # Un masque légèrement débordant capte des points d'arrière-plan très éloignés :
    # l'amas principal les écarte avant de calculer la boîte.
    if len(points) > 200:
        points = points[largest_cluster_mask(points, eps=0.05, min_points=20)]

    lower = np.percentile(points, 0.5, axis=0)
    upper = np.percentile(points, 99.5, axis=0)
    margin = 0.05 * np.maximum(upper - lower, 1e-3)
    log_fn(f"boîte objet depuis {len(points)} points masqués")

    return Segmentation(
        masks=masks,
        method=method,
        bbox_min=lower - margin,
        bbox_max=upper + margin,
        stats={"object_points": int(len(points)), "conf_threshold": threshold},
    )


# ---------------------------------------------------------------------------
# Repli géométrique
# ---------------------------------------------------------------------------


def _confident_points(
    result: VGGTResult, normalization: SceneNormalization, conf_quantile: float, stride: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Nuage normalisé des pixels les plus confiants + seuil retenu."""
    if not isinstance(stride, (int, np.integer)) or isinstance(stride, (bool, np.bool_)) or stride <= 0:
        raise ValueError(f"pas d'échantillonnage invalide: {stride!r}")
    threshold = confidence_threshold(result.depth_conf, 1.0 - conf_quantile)
    clouds = []
    for index in range(len(result.depth)):
        depth = result.depth[index]
        sub = np.zeros(depth.shape, dtype=bool)
        sub[::stride, ::stride] = True
        keep = sub & (result.depth_conf[index] >= threshold) & np.isfinite(depth) & (depth > 0)
        if not keep.any():
            continue
        points = unproject(depth, result.intrinsics[index], result.extrinsics[index])
        clouds.append(normalization.apply_points(points[keep]))
    if not clouds:
        raise ValueError("aucun point confiant : la reconstruction VGGT-Ω a échoué")
    return np.concatenate(clouds, axis=0), np.array([threshold])


def _segment_geometric(
    result: VGGTResult,
    normalization: SceneNormalization,
    conf_quantile: float,
    log_fn: Callable[[str], None],
) -> Segmentation:
    import open3d as o3d

    random_api = getattr(o3d.utility, "random", None)
    if random_api is not None and hasattr(random_api, "seed"):
        random_api.seed(0)

    points, threshold = _confident_points(result, normalization, conf_quantile)
    log_fn(f"segmentation géométrique sur {len(points)} points")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud = cloud.voxel_down_sample(voxel_size=0.01)
    reduced = np.asarray(cloud.points)
    # Sans cette ligne, le journal enchaîne « segmentation sur 4 938 967 points »
    # puis « amas objet: 3242 points » : deux échelles différentes qu'on lit comme
    # une perte catastrophique, alors que le sous-échantillonnage est voulu.
    log_fn(f"sous-échantillonnage à 0.01 : {len(reduced)} points conservés")

    plane_model: np.ndarray | None = None
    remaining = reduced
    if len(reduced) > 500:
        model, inliers = cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=800)
        inlier_ratio = len(inliers) / len(reduced)
        if inlier_ratio > 0.12:  # un vrai plan de support, pas trois points alignés
            plane_model = np.asarray(model, dtype=np.float64)
            centers_side = _camera_side(plane_model, result, normalization)
            # Convention transmise au nettoyage : l'objet et les caméras sont
            # toujours du côté positif, quel que soit le signe choisi par RANSAC.
            if centers_side < 0:
                plane_model = -plane_model
            signed = reduced @ plane_model[:3] + plane_model[3]
            keep = signed > 0.02
            remaining = reduced[keep]
            log_fn(f"plan de support retiré ({inlier_ratio:.0%} des points)")

    if len(remaining) < 100:
        remaining = reduced
        plane_model = None

    cluster = largest_cluster_mask(remaining, eps=0.035, min_points=20)
    object_points = remaining[cluster]
    if len(object_points) < 50:
        object_points = remaining
    log_fn(f"amas objet: {len(object_points)} points ({len(object_points) / max(1, len(remaining)):.0%})")

    lower = np.percentile(object_points, 0.5, axis=0)
    upper = np.percentile(object_points, 99.5, axis=0)
    margin = 0.05 * np.maximum(upper - lower, 1e-3)
    bbox_min, bbox_max = lower - margin, upper + margin

    masks = _masks_from_box(result, normalization, bbox_min, bbox_max, plane_model, float(threshold[0]))
    masks = _clean_masks(masks)
    return Segmentation(
        masks=masks,
        method="geometric",
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        plane=plane_model,
        stats={"object_points": int(len(object_points)), "conf_threshold": float(threshold[0])},
    )


def _camera_side(plane: np.ndarray, result: VGGTResult, normalization: SceneNormalization) -> float:
    """De quel côté du plan se trouvent les caméras (donc l'objet)."""
    from .geometry import camera_centers

    centers = normalization.apply_points(camera_centers(result.extrinsics))
    signed = centers @ plane[:3] + plane[3]
    return float(np.sign(np.median(signed)) or 1.0)


def _masks_from_box(
    result: VGGTResult,
    normalization: SceneNormalization,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    plane: np.ndarray | None,
    conf_threshold: float,
) -> np.ndarray:
    sequence, height, width = result.depth.shape
    masks = np.zeros((sequence, height, width), dtype=bool)
    for index in range(sequence):
        depth = result.depth[index]
        points = normalization.apply_points(unproject(depth, result.intrinsics[index], result.extrinsics[index]))
        inside = np.all((points >= bbox_min) & (points <= bbox_max), axis=-1)
        valid = np.isfinite(depth) & (depth > 0) & (result.depth_conf[index] >= conf_threshold)
        mask = inside & valid
        if plane is not None:
            signed = points @ plane[:3] + plane[3]
            mask &= np.abs(signed) > 0.015
        masks[index] = mask
    return masks


def _clean_masks(masks: np.ndarray, min_area_ratio: float = 0.15) -> np.ndarray:
    """Fermeture morphologique + plus grande composante connexe 2D."""
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3:
        raise ValueError(f"masques de forme inattendue: {masks.shape}, (S, H, W) attendu")
    if not np.isfinite(min_area_ratio) or not 0.0 <= min_area_ratio <= 1.0:
        raise ValueError(f"ratio d'aire invalide: {min_area_ratio!r}")
    try:
        import cv2
    except ImportError:  # pragma: no cover - OpenCV absent : on garde le masque brut
        return masks.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = np.zeros_like(masks)
    for index, mask in enumerate(masks):
        binary = (mask * 255).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        if mask.any() and not binary.any():
            binary = (mask * 255).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if count <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = int(np.argmax(areas)) + 1
        keep = areas >= min_area_ratio * areas.max()
        selected = np.zeros_like(binary, dtype=bool)
        for label in np.nonzero(keep)[0] + 1:
            selected |= labels == label
        selected |= labels == biggest
        cleaned[index] = selected
    return cleaned
