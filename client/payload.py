"""Préparation de la charge utile envoyée au pod.

Extrait de la CLI pour être partagé avec l'interface graphique : les deux
doivent produire exactement la même archive, sinon un scan lancé depuis le GUI
ne serait pas comparable à un scan lancé en ligne de commande.
"""

from __future__ import annotations

import shutil
import tarfile
from collections.abc import Callable
from pathlib import Path

from common.video import extract_frames

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def build_payload(
    source: Path,
    workdir: Path,
    frames: int = 120,
    long_side: int = 1024,
    send_video: bool = False,
    log: Callable[[str], None] = lambda _m: None,
) -> Path:
    """Retourne le fichier à téléverser : archive d'images, ou vidéo brute.

    Extraire les images en local plutôt que d'envoyer la vidéo divise le volume
    transféré par un facteur ~25 sur une prise 4K.
    """
    source = Path(source)
    workdir = Path(workdir)
    if not source.exists():
        raise FileNotFoundError(f"introuvable: {source}")

    if source.is_file() and send_video:
        log(f"envoi de la vidéo brute ({source.stat().st_size / 1e6:.0f} Mo) — extraction côté pod")
        return source

    frames_dir = workdir / "frames"
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"aucune image dans {source}")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            shutil.copy2(image, frames_dir / f"frame_{index:05d}{image.suffix.lower()}")
        log(f"{len(images)} images reprises depuis {source.name}")
    else:
        extracted = extract_frames(source, frames_dir, count=frames, long_side=long_side, log=log)
        if not extracted:
            raise ValueError("extraction ffmpeg vide")

    archive = workdir / "frames.tar"
    with tarfile.open(archive, "w") as tar:
        for image in sorted(frames_dir.iterdir()):
            tar.add(image, arcname=image.name)
    log(f"archive prête : {archive.stat().st_size / 1e6:.1f} Mo")
    return archive
