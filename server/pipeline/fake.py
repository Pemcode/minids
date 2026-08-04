"""Pipeline factice (`MINIDS_FAKE_GPU=1`).

Rejoue les mêmes étapes, la même progression et les mêmes artefacts qu'un vrai
scan, mais sans GPU ni modèle : c'est ce qui permet de valider tout le transport
(upload chunké, polling, download par Range, reprise) sur une machine Windows
avant de payer une seule minute de pod.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import Settings
from ..jobs import STAGES, Job, Reporter, unpack_input

# Durées simulées : assez longues pour que le client voie une vraie progression,
# réductibles par les tests via MINIDS_FAKE_SPEED.
_STAGE_SECONDS = {"refine": 6.0, "vggt": 2.0, "mesh": 1.5}


def _stage_duration(name: str) -> float:
    raw_scale = os.environ.get("MINIDS_FAKE_SPEED", "1.0")
    try:
        scale = float(raw_scale)
    except ValueError as exc:
        raise ValueError(f"MINIDS_FAKE_SPEED invalide: {raw_scale!r}") from exc
    if not np.isfinite(scale) or scale < 0:
        raise ValueError(f"MINIDS_FAKE_SPEED doit être un nombre fini positif ou nul: {raw_scale!r}")
    return _STAGE_SECONDS.get(name, 0.6) * scale


def run_fake_pipeline(job: Job, reporter: Reporter, settings: Settings) -> None:
    started = time.time()
    source = Path(job.upload["assembled_path"])

    for name, _weight in STAGES:
        reporter.stage(name)
        if name == "ingest":
            kind, target = unpack_input(source, job.frames_dir)
            reporter.log(f"[factice] entrée {kind}: {target.name}")
        duration = _stage_duration(name)
        steps = 12
        for step in range(steps):
            reporter.check_cancelled()
            time.sleep(duration / steps)
            reporter.progress((step + 1) / steps)

    frames = sorted(path for path in job.frames_dir.glob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    reporter.log(f"[factice] {len(frames)} images vues, génération du cube de test")

    _write_fake_artifacts(job, frames_count=len(frames))
    report = {
        "job_id": job.job_id,
        "fake": True,
        "params": job.params,
        "frames": {"count": len(frames)},
        "timings": reporter.timings(),
        "total_seconds": round(time.time() - started, 2),
        "note": "MINIDS_FAKE_GPU=1 — artefacts synthétiques, aucune reconstruction réelle",
    }
    (job.artifacts_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def _write_fake_artifacts(job: Job, frames_count: int) -> None:
    from common.glb import encode_png, write_glb

    vertices, faces, uvs, normals = _unit_cube()
    texture = _checkerboard(512)
    write_glb(
        path=job.artifacts_dir / "mesh.glb",
        vertices=vertices * 0.1,
        faces=faces,
        normals=normals,
        uvs=uvs,
        texture_png=encode_png(texture),
        name="minids_fake_cube",
    )
    (job.artifacts_dir / "preview.png").write_bytes(encode_png(_checkerboard(256, tiles=4)))

    count = max(1, frames_count)
    height, width = 48, 64
    np.savez_compressed(
        job.artifacts_dir / "vggt_raw.npz",
        depth=np.ones((count, height, width), dtype=np.float32),
        depth_conf=np.ones((count, height, width), dtype=np.float32),
        extrinsics=np.tile(np.eye(4, dtype=np.float32), (count, 1, 1)),
        intrinsics=np.tile(np.eye(3, dtype=np.float32), (count, 1, 1)),
        images=np.zeros((count, height, width, 3), dtype=np.uint8),
        frame_names=np.asarray([f"frame_{i:05d}.jpg" for i in range(count)], dtype=np.str_),
        scene_center=np.zeros(3, dtype=np.float32),
        scene_scale=np.float32(1.0),
        masks_packed=np.packbits(np.ones((count, height, width), dtype=bool)),
        masks_shape=np.array([count, height, width], dtype=np.int64),
        fake=np.array([True]),
    )


def _unit_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cube à faces séparées (24 sommets) pour des UV et normales propres."""
    directions = [
        ((0, 0, 1), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
        ((0, 0, -1), ((1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1))),
        ((1, 0, 0), ((1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1))),
        ((-1, 0, 0), ((-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1))),
        ((0, 1, 0), ((-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1))),
        ((0, -1, 0), ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ]
    vertices, normals, uvs, faces = [], [], [], []
    for index, (normal, corners) in enumerate(directions):
        base = index * 4
        vertices.extend(corners)
        normals.extend([normal] * 4)
        uvs.extend([(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)])
        faces.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
    return (
        np.array(vertices, dtype=np.float64),
        np.array(faces, dtype=np.uint32),
        np.array(uvs, dtype=np.float32),
        np.array(normals, dtype=np.float64),
    )


def _checkerboard(size: int, tiles: int = 8) -> np.ndarray:
    if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)) or size <= 0:
        raise ValueError(f"taille de damier invalide: {size!r}")
    if not isinstance(tiles, (int, np.integer)) or isinstance(tiles, (bool, np.bool_)) or tiles <= 0:
        raise ValueError(f"nombre de cases invalide: {tiles!r}")
    step = max(1, size // tiles)
    ys, xs = np.mgrid[0:size, 0:size]
    pattern = ((xs // step + ys // step) % 2).astype(np.uint8)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = 40 + pattern * 180
    image[..., 1] = 60 + pattern * 140
    image[..., 2] = 90 + pattern * 100
    return image
