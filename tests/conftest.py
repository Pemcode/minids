"""Fixtures partagées : serveur en mode factice et charge utile d'images."""

from __future__ import annotations

import io
import random
import struct
import tarfile
import zlib
from pathlib import Path

import pytest

TEST_TOKEN = "jeton-de-test"
# Plancher imposé par l'API (`CreateJobRequest.chunk_size`, ge=64 Kio).
CHUNK_SIZE = 64 * 1024


def _noise_png(seed: int, size: int = 128) -> bytes:
    """PNG peu compressible : l'archive doit couvrir plusieurs chunks de 64 Kio."""
    generator = random.Random(seed)
    raw = bytearray()
    for _ in range(size):
        raw.append(0)  # octet de filtre
        raw.extend(generator.randbytes(size * 3))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw))) + chunk(b"IEND", b"")


@pytest.fixture
def frames_archive(tmp_path: Path) -> Path:
    """Archive tar de 5 images, telle que la produit le client."""
    archive = tmp_path / "frames.tar"
    with tarfile.open(archive, "w") as tar:
        for index in range(5):
            payload = _noise_png(seed=index)
            info = tarfile.TarInfo(name=f"frame_{index:05d}.png")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return archive


@pytest.fixture(autouse=True)
def fast_fake_pipeline(monkeypatch):
    """Le pipeline factice simule des durées réalistes ; inutile de les subir ici."""
    monkeypatch.setenv("MINIDS_FAKE_SPEED", "0.05")


@pytest.fixture
def settings(tmp_path: Path):
    from server.config import Settings

    return Settings(
        token=TEST_TOKEN,
        data_dir=tmp_path / "data",
        fake_gpu=True,
        device="cpu",
        chunk_size=CHUNK_SIZE,
    )


@pytest.fixture
def app(settings):
    from server.app import build_app

    return build_app(settings)
