"""Configuration centrale du serveur miniDS (tout vient de l'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
    keep_work_dir: bool = False

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
        image_resolution=int(os.environ.get("MINIDS_IMAGE_RESOLUTION", "512")),
        default_frames=int(os.environ.get("MINIDS_FRAMES", "120")),
        chunk_size=int(os.environ.get("MINIDS_CHUNK_SIZE", str(8 * 1024 * 1024))),
        max_upload_bytes=int(os.environ.get("MINIDS_MAX_UPLOAD", str(4 * 1024 * 1024 * 1024))),
        keep_work_dir=_bool("MINIDS_KEEP_WORK", True),
    )
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
