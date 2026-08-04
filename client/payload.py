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

from common.video import extract_frames, probe

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def build_payload(
    source: Path,
    workdir: Path,
    frames: int = 120,
    long_side: int = 1024,
    send_video: bool = False,
    log: Callable[[str], None] = lambda _m: None,
    should_stop: Callable[[], bool] | None = None,
) -> Path:
    """Retourne le fichier à téléverser : archive d'images, ou vidéo brute.

    Extraire les images en local plutôt que d'envoyer la vidéo divise le volume
    transféré par un facteur ~25 sur une prise 4K.
    """
    source = Path(source)
    workdir = Path(workdir)
    if not source.exists():
        raise FileNotFoundError(f"introuvable: {source}")
    if not source.is_file() and not source.is_dir():
        raise ValueError(f"source non prise en charge: {source}")

    workdir.mkdir(parents=True, exist_ok=True)

    if source.is_file() and send_video:
        if source.stat().st_size == 0:
            raise ValueError(f"vidéo vide: {source}")
        if source.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"format vidéo non pris en charge: {source.suffix or '(sans extension)'}")
        info = probe(source)
        if info.duration <= 0 or info.fps <= 0:
            raise ValueError(f"métadonnées vidéo incomplètes: {source}")
        log(f"envoi de la vidéo brute ({source.stat().st_size / 1e6:.0f} Mo) — extraction côté pod")
        return source

    if not isinstance(frames, int) or isinstance(frames, bool) or frames <= 0:
        raise ValueError("le nombre d'images doit être strictement positif")
    if not isinstance(long_side, int) or isinstance(long_side, bool) or long_side <= 0 or long_side % 2:
        raise ValueError("la résolution d'envoi doit être un entier pair strictement positif")

    frames_dir = workdir / "frames"
    prepared: list[Path]
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"aucune image dans {source}")
        source_count = len(images)
        if source_count > frames:
            if frames == 1:
                images = [images[0]]
            else:
                indices = [round(index * (source_count - 1) / (frames - 1)) for index in range(frames)]
                images = [images[index] for index in indices]
        frames_dir.mkdir(parents=True, exist_ok=True)
        prepared = []
        for index, image in enumerate(images):
            if should_stop is not None and should_stop():
                raise ValueError("préparation interrompue")
            target = frames_dir / f"frame_{index:05d}{image.suffix.lower()}"
            if image.resolve() != target.resolve():
                shutil.copy2(image, target)
            prepared.append(target)
        detail = f" sur {source_count}" if source_count != len(images) else ""
        log(f"{len(images)} images reprises{detail} depuis {source.name}")
    else:
        prepared = extract_frames(
            source,
            frames_dir,
            count=frames,
            long_side=long_side,
            log=log,
            should_stop=should_stop,
        )
        if not prepared:
            raise ValueError("extraction ffmpeg vide")

    archive = workdir / "frames.tar"
    with tarfile.open(archive, "w") as tar:
        # N'archive que les fichiers préparés pendant cet appel. Un workdir
        # réutilisé peut contenir d'anciennes frames, qui ne doivent jamais se
        # glisser silencieusement dans un nouveau job.
        for image in prepared:
            if should_stop is not None and should_stop():
                raise ValueError("préparation interrompue")
            tar.add(image, arcname=image.name)
    log(f"archive prête : {archive.stat().st_size / 1e6:.1f} Mo")
    return archive
