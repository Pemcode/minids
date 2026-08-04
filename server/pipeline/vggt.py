"""Inférence VGGT-Ω : images → poses caméra + cartes de profondeur.

Le modèle reste chargé en VRAM entre deux jobs (singleton) : le rechargement
coûte plus cher que l'inférence elle-même.

Les formes exactes renvoyées par le modèle peuvent évoluer d'une révision à
l'autre du dépôt ; `_as_sequence` normalise tout en (S, ...) et lève une erreur
explicite avec la forme observée plutôt que de propager un tenseur mal
interprété.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("minids.vggt")

_MODEL_LOCK = threading.Lock()
_MODEL: Any = None
_MODEL_KEY: str | None = None


@dataclass
class VGGTResult:
    """Sortie brute normalisée, telle qu'elle est sauvegardée dans `vggt_raw.npz`."""

    depth: np.ndarray  # (S, H, W) float32
    depth_conf: np.ndarray  # (S, H, W) float32
    extrinsics: np.ndarray  # (S, 4, 4) world-to-camera
    intrinsics: np.ndarray  # (S, 3, 3) pour la résolution (H, W)
    images: np.ndarray  # (S, H, W, 3) float32 dans [0, 1], telles que vues par le modèle
    frame_names: list[str]

    def __post_init__(self) -> None:
        self.depth = np.asarray(self.depth, dtype=np.float32)
        self.depth_conf = np.asarray(self.depth_conf, dtype=np.float32)
        self.extrinsics = np.asarray(self.extrinsics, dtype=np.float32)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float32)
        self.images = np.asarray(self.images, dtype=np.float32)
        self.frame_names = [str(name) for name in self.frame_names]

        if self.depth.ndim != 3 or any(size <= 0 for size in self.depth.shape):
            raise ValueError(f"profondeurs de forme inattendue: {self.depth.shape}, (S, H, W) attendu")
        sequence, height, width = self.depth.shape
        expected = {
            "depth_conf": (sequence, height, width),
            "extrinsics": (sequence, 4, 4),
            "intrinsics": (sequence, 3, 3),
            "images": (sequence, height, width, 3),
        }
        observed = {
            "depth_conf": self.depth_conf.shape,
            "extrinsics": self.extrinsics.shape,
            "intrinsics": self.intrinsics.shape,
            "images": self.images.shape,
        }
        mismatches = [
            f"{name}={observed[name]} (attendu {shape})" for name, shape in expected.items() if observed[name] != shape
        ]
        if mismatches:
            raise ValueError("sortie VGGT-Ω incohérente: " + ", ".join(mismatches))
        if len(self.frame_names) != sequence:
            raise ValueError(f"{len(self.frame_names)} noms de frames pour {sequence} vues")
        if not np.isfinite(self.extrinsics).all() or not np.isfinite(self.intrinsics).all():
            raise ValueError("poses ou intrinsèques VGGT-Ω non finies")
        if np.any(np.abs(self.intrinsics[:, (0, 1), (0, 1)]) <= 1e-12):
            raise ValueError("focale VGGT-Ω nulle")
        if not np.isfinite(self.images).all():
            raise ValueError("images VGGT-Ω non finies")
        if np.any(self.images < 0) or np.any(self.images > 1):
            raise ValueError("images VGGT-Ω hors de la plage [0, 1]")

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.depth.shape[1]), int(self.depth.shape[2])

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        reserved = {"depth", "depth_conf", "extrinsics", "intrinsics", "images", "frame_names"}
        overlap = reserved.intersection(extra or {})
        if overlap:
            raise ValueError(f"champs VGGT réservés dans extra: {', '.join(sorted(overlap))}")
        payload: dict[str, Any] = {
            "depth": self.depth.astype(np.float32),
            "depth_conf": self.depth_conf.astype(np.float32),
            "extrinsics": self.extrinsics.astype(np.float32),
            "intrinsics": self.intrinsics.astype(np.float32),
            "images": (self.images * 255.0).clip(0, 255).round().astype(np.uint8),
            "frame_names": np.asarray(self.frame_names, dtype=np.str_),
        }
        for key, value in (extra or {}).items():
            if np.asarray(value).dtype.hasobject:
                raise TypeError(f"le champ NPZ {key!r} ne peut pas contenir d'objets Python")
            payload[key] = value
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def load(cls, path: Path) -> VGGTResult:
        with np.load(path, allow_pickle=False) as data:
            depth = data["depth"]
            try:
                frame_names = [str(name) for name in data["frame_names"]]
            except (KeyError, ValueError):
                # Les premières archives miniDS utilisaient un tableau objet. Ne
                # jamais le dé-pickler : les noms ne sont pas requis au re-maillage.
                frame_names = [f"frame_{index:05d}.jpg" for index in range(len(depth))]
            raw_images = data["images"]
            images = raw_images.astype(np.float32)
            if np.issubdtype(raw_images.dtype, np.integer) or (images.size and float(images.max()) > 1.5):
                images = images / 255.0
            return cls(
                depth=depth,
                depth_conf=data["depth_conf"],
                extrinsics=data["extrinsics"],
                intrinsics=data["intrinsics"],
                images=images,
                frame_names=frame_names,
            )


def scale_intrinsics(intrinsics: np.ndarray, from_size: tuple[int, int], to_size: tuple[int, int]) -> np.ndarray:
    """Adapte des intrinsèques à une autre résolution (facteurs H et W séparés)."""
    if len(from_size) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in from_size):
        raise ValueError(f"résolution source invalide: {from_size!r}")
    if len(to_size) != 2 or any(not isinstance(value, (int, np.integer)) or value <= 0 for value in to_size):
        raise ValueError(f"résolution cible invalide: {to_size!r}")
    intrinsics = np.asarray(intrinsics)
    if intrinsics.ndim < 2 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"intrinsèques de forme inattendue: {intrinsics.shape}")
    scale_y = to_size[0] / from_size[0]
    scale_x = to_size[1] / from_size[1]
    scaled = np.array(intrinsics, dtype=np.float64, copy=True)
    scaled[..., 0, 0] *= scale_x
    scaled[..., 0, 2] *= scale_x
    scaled[..., 1, 1] *= scale_y
    scaled[..., 1, 2] *= scale_y
    return scaled


def _as_sequence(tensor: Any, name: str, expected_trailing: int) -> np.ndarray:
    """Ramène un tenseur (B, S, ...) ou (S, ...) à (S, ...) en numpy float32."""
    import torch

    if isinstance(tensor, torch.Tensor):
        array = tensor.detach().float().cpu().numpy()
    else:
        array = np.asarray(tensor, dtype=np.float32)

    # Supprime une dimension de canal finale à 1 (ex: profondeur en (B, S, H, W, 1)).
    while array.ndim > expected_trailing + 1 and array.shape[-1] == 1:
        array = array[..., 0]
    # Supprime la dimension de batch de tête.
    while array.ndim > expected_trailing + 1 and array.shape[0] == 1:
        array = array[0]

    if array.ndim != expected_trailing + 1:
        raise ValueError(
            f"'{name}' a la forme {np.shape(tensor)} → {array.shape} après normalisation ; "
            f"{expected_trailing + 1} dimensions attendues (S, ...). "
            "L'API de VGGT-Ω a probablement changé : adapter server/pipeline/vggt.py."
        )
    return np.ascontiguousarray(array)


def _select_device(preferred: str) -> str:
    import torch

    preferred = str(preferred).strip().lower()
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA indisponible, bascule sur CPU (l'inférence sera très lente)")
        return "cpu"
    return preferred


def _autocast_dtype(device: str) -> Any:
    import torch

    if not device.startswith("cuda"):
        return torch.float32
    major = torch.cuda.get_device_capability(device)[0]
    return torch.bfloat16 if major >= 8 else torch.float16


def _resolve_checkpoint(checkpoint: str, hf_token: str, cache_dir: Path, image_resolution: int = 512) -> str | None:
    """Chemin local du checkpoint, téléchargé depuis Hugging Face si nécessaire.

    `facebook/VGGT-Omega` héberge plusieurs variantes (512 et 256-text) dans le
    même dépôt. Sans nom de fichier explicite, on choisit celle qui correspond à
    la résolution de travail : prendre la mauvaise dégraderait la profondeur
    silencieusement.
    """
    if not checkpoint:
        return None
    candidate = Path(checkpoint)
    if candidate.is_file():
        return str(candidate)
    if "/" not in checkpoint:
        raise FileNotFoundError(f"checkpoint introuvable: {checkpoint}")

    from huggingface_hub import hf_hub_download, list_repo_files

    repo_id, _, filename = checkpoint.partition(":")
    token = hf_token or None
    if not filename:
        files = [f for f in list_repo_files(repo_id, token=token) if f.endswith((".pt", ".safetensors", ".bin"))]
        if not files:
            raise FileNotFoundError(f"aucun poids trouvé dans {repo_id}")
        matching = [name for name in files if str(image_resolution) in name]
        if not matching:
            raise FileNotFoundError(
                f"aucun poids en {image_resolution} px dans {repo_id} (disponibles : {', '.join(files)}). "
                "Préciser MINIDS_CKPT sous la forme 'repo_id:fichier', ou ajuster MINIDS_IMAGE_RESOLUTION."
            )
        filename = sorted(matching, key=len)[0]
    elif str(image_resolution) not in filename:
        log.warning(
            "checkpoint '%s' et MINIDS_IMAGE_RESOLUTION=%d semblent incohérents : "
            "vérifier que la variante correspond bien à la résolution de travail.",
            filename,
            image_resolution,
        )
    log.info("téléchargement du checkpoint %s:%s", repo_id, filename)
    return hf_hub_download(repo_id=repo_id, filename=filename, token=token, cache_dir=str(cache_dir))


def load_model(checkpoint: str, device: str, hf_token: str, cache_dir: Path, image_resolution: int = 512) -> Any:
    """Charge (et met en cache) VGGT-Ω."""
    global _MODEL, _MODEL_KEY

    key = f"{checkpoint}@{device}@{image_resolution}"
    with _MODEL_LOCK:
        if _MODEL is not None and key == _MODEL_KEY:
            return _MODEL

        import torch
        from vggt_omega.models import VGGTOmega

        path = _resolve_checkpoint(checkpoint, hf_token, cache_dir, image_resolution)
        if path is None and hasattr(VGGTOmega, "from_pretrained"):
            model = VGGTOmega.from_pretrained("facebook/VGGT-Omega", token=hf_token or None)
        else:
            if path is None:
                raise ValueError(
                    "Aucun checkpoint : renseigner MINIDS_CKPT (chemin local ou 'repo_id:fichier'). "
                    "Les poids VGGT-Ω sont sous accès restreint sur Hugging Face."
                )
            model = VGGTOmega()
            if Path(path).suffix.lower() == ".safetensors":
                from safetensors.torch import load_file

                state = load_file(path, device="cpu")
            else:
                try:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                except TypeError as exc:  # PyTorch trop ancien pour un chargement sûr
                    raise RuntimeError(
                        "cette version de PyTorch ne prend pas en charge weights_only=True ; "
                        "mettre PyTorch à jour avant de charger un checkpoint local"
                    ) from exc
            state = state.get("model", state) if isinstance(state, dict) else state
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                log.warning("poids manquants (%d), ex: %s", len(missing), missing[:3])
            if unexpected:
                log.warning("poids inattendus (%d), ex: %s", len(unexpected), unexpected[:3])

        model = model.to(device).eval()
        _MODEL, _MODEL_KEY = model, key
        return model


def run_inference(
    frames: list[Path],
    checkpoint: str,
    device: str,
    hf_token: str,
    cache_dir: Path,
    image_resolution: int = 512,
    log_fn: Callable[[str], None] = lambda _m: None,
) -> VGGTResult:
    """Une passe VGGT-Ω sur toute la séquence (l'attention est globale : pas de découpage)."""
    if not frames:
        raise ValueError("aucune frame fournie à VGGT-Ω")
    if not isinstance(image_resolution, int) or isinstance(image_resolution, bool) or image_resolution <= 0:
        raise ValueError(f"résolution VGGT-Ω invalide: {image_resolution!r}")
    missing = [str(path) for path in frames if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"frame introuvable: {missing[0]}")

    import torch
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    device = _select_device(device)
    model = load_model(checkpoint, device, hf_token, cache_dir, image_resolution)

    names = [str(path) for path in frames]
    images = load_and_preprocess_images(names, image_resolution=image_resolution).to(device)
    if images.ndim == 4:
        images = images[None]
    log_fn(f"inférence sur {images.shape[1]} images en {tuple(images.shape[-2:])}")

    dtype = _autocast_dtype(device)
    with torch.inference_mode():
        if device.startswith("cuda"):
            with torch.autocast("cuda", dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)

    if not isinstance(predictions, dict):
        raise TypeError(f"sortie VGGT-Ω inattendue: {type(predictions).__name__}, dictionnaire attendu")
    if "pose_enc" not in predictions or "depth" not in predictions:
        raise KeyError(f"sortie VGGT-Ω inattendue, clés disponibles: {sorted(predictions)}")

    model_images = predictions.get("images", images)
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], model_images.shape[-2:])

    depth = _as_sequence(predictions["depth"], "depth", 2)
    conf_key = "depth_conf" if "depth_conf" in predictions else "conf"
    if conf_key not in predictions:
        raise KeyError("sortie VGGT-Ω sans carte de confiance ('depth_conf' ou 'conf')")
    depth_conf = _as_sequence(predictions[conf_key], conf_key, 2)
    extrinsics_np = _as_sequence(extrinsics, "extrinsics", 2)
    intrinsics_np = _as_sequence(intrinsics, "intrinsics", 2)
    images_np = _as_sequence(model_images, "images", 3)

    if images_np.shape[-1] != 3:
        if images_np.shape[1] == 3:  # (S, 3, H, W) → (S, H, W, 3)
            images_np = np.transpose(images_np, (0, 2, 3, 1))
        else:
            raise ValueError(f"images VGGT-Ω sans canal RGB identifiable: {images_np.shape}")
    minimum, maximum = float(images_np.min()), float(images_np.max())
    if minimum < -1e-3 or maximum > 255.5:
        raise ValueError(f"plage d'images VGGT-Ω inattendue: [{minimum:.3g}, {maximum:.3g}]")
    if maximum > 1.5:
        images_np = images_np / 255.0
    images_np = np.clip(images_np, 0.0, 1.0)

    from .geometry import to_4x4

    result = VGGTResult(
        depth=depth.astype(np.float32),
        depth_conf=depth_conf.astype(np.float32),
        extrinsics=to_4x4(extrinsics_np).astype(np.float32),
        intrinsics=intrinsics_np.astype(np.float32),
        images=images_np.astype(np.float32),
        frame_names=[Path(name).name for name in names],
    )

    if device.startswith("cuda"):
        peak = torch.cuda.max_memory_allocated() / 1e9
        log_fn(f"profondeur {result.depth.shape}, pic VRAM {peak:.1f} Go")
        torch.cuda.reset_peak_memory_stats()
    return result


def world_points(result: VGGTResult) -> np.ndarray:
    """Nuage dense (S, H, W, 3) dé-projeté depuis les profondeurs."""
    from .geometry import unproject

    if len(result.depth) == 0:
        raise ValueError("aucune vue à dé-projeter")
    return np.stack(
        [unproject(result.depth[i], result.intrinsics[i], result.extrinsics[i]) for i in range(len(result.depth))]
    )
