"""Configuration centrale du serveur miniDS (tout vient de l'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

MIN_CHUNK_SIZE = 64 * 1024
MAX_CHUNK_SIZE = 64 * 1024 * 1024

MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name}: booléen invalide {raw!r} (attendu: 1/0, true/false, yes/no, on/off)")


def _bounded_int(name: str, value: int, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ValueError(f"{name}: entier invalide {value!r}")
    if value < minimum or maximum is not None and value > maximum:
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name}: {value} hors limites ({bounds})")
    return value


def _env_int(name: str, default: int, minimum: int, maximum: int | None = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return _bounded_int(name, default, minimum, maximum)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: entier invalide {raw!r}") from exc
    return _bounded_int(name, value, minimum, maximum)


def _default_data_dir() -> Path:
    # Sur le pod RunPod, /workspace est le volume persistant.
    workspace = Path("/workspace")
    if workspace.is_dir():
        return workspace / "minids"
    return Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Settings:
    token: str = ""
    data_dir: Path = field(default_factory=_default_data_dir)
    fake_gpu: bool = False
    device: str = "cuda"
    checkpoint: str = ""
    hf_token: str = ""
    image_resolution: int = 512
    default_frames: int = 120
    chunk_size: int = 8 * 1024 * 1024
    max_upload_bytes: int = 4 * 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("token", "device", "checkpoint", "hf_token"):
            object.__setattr__(self, name, getattr(self, name).strip())
        if not self.device:
            raise ValueError("device ne peut pas être vide")

        _bounded_int("image_resolution", self.image_resolution, 1, 8192)
        _bounded_int("default_frames", self.default_frames, 1, 600)
        _bounded_int("chunk_size", self.chunk_size, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE)
        _bounded_int("max_upload_bytes", self.max_upload_bytes, 1, MAX_UPLOAD_BYTES)

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    data_dir = os.environ.get("MINIDS_DATA_DIR")
    settings = Settings(
        token=os.environ.get("MINIDS_TOKEN", ""),
        data_dir=Path(data_dir).expanduser() if data_dir else _default_data_dir(),
        fake_gpu=_bool("MINIDS_FAKE_GPU"),
        device=os.environ.get("MINIDS_DEVICE", "cuda"),
        checkpoint=os.environ.get("MINIDS_CKPT", ""),
        hf_token=os.environ.get("HF_TOKEN", ""),
        image_resolution=_env_int("MINIDS_IMAGE_RESOLUTION", 512, 1, 8192),
        default_frames=_env_int("MINIDS_FRAMES", 120, 1, 600),
        chunk_size=_env_int("MINIDS_CHUNK_SIZE", 8 * 1024 * 1024, MIN_CHUNK_SIZE, MAX_CHUNK_SIZE),
        max_upload_bytes=_env_int("MINIDS_MAX_UPLOAD", MAX_UPLOAD_BYTES, 1, MAX_UPLOAD_BYTES),
    )
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
