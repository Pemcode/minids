"""Extraction d'images depuis une vidéo de téléphone, avec ffmpeg uniquement.

Trois choses que fait ce module et qui comptent pour la qualité finale :

1. **Échantillonnage uniforme** de la séquence (couverture angulaire régulière de
   l'objet), et pas un simple `-r`.
2. **Rejet des images floues** : on extrait 3× plus de candidates que nécessaire,
   on mesure le flou avec le filtre `blurdetect` de ffmpeg, puis on garde la plus
   nette de chaque intervalle temporel. `lavfi.blur` monte quand l'image est floue
   (vérifié : ~4.5 sur une mire nette, ~32 après `boxblur`), donc on minimise.
3. **Tonemap HDR→SDR** : les Samsung filment en HDR10+ par défaut ; donner du
   PQ/HLG brut à un réseau entraîné sur du sRGB dégrade franchement la profondeur.

Aucune dépendance hors stdlib : le client Windows n'a besoin que de ffmpeg.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

FRAME_GLOB = "frame_*.jpg"
_METADATA_RE = re.compile(r"^frame:(\d+)\s")
_BLUR_RE = re.compile(r"^lavfi\.blur=([0-9.eE+-]+)")
HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}
EXTRACTION_TIMEOUT_SECONDS = 30 * 60


class FFmpegError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    duration: float
    fps: float
    codec: str
    color_transfer: str
    rotation: int

    @property
    def is_hdr(self) -> bool:
        return self.color_transfer in HDR_TRANSFERS


def _tool(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise FFmpegError(
            f"{name} introuvable dans le PATH. Windows : winget install Gyan.FFmpeg ; Debian : apt install ffmpeg"
        )
    return resolved


def probe(video: Path) -> VideoInfo:
    """Lit les métadonnées du flux vidéo via ffprobe."""
    video = Path(video)
    if not video.is_file():
        raise FFmpegError(f"vidéo introuvable : {video}")
    command = [
        _tool("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,codec_name,color_transfer,duration",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "side_data=rotation",
        "-of",
        "json",
        str(video),
    ]
    try:
        # argv sans shell, exécutable résolu par shutil.which.
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(f"ffprobe ne répond pas sur {video.name}") from exc
    if result.returncode != 0:
        raise FFmpegError(f"ffprobe a échoué sur {video.name}: {result.stderr.strip()[:400]}")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(f"réponse ffprobe invalide sur {video.name}") from exc
    if not isinstance(payload, dict):
        raise FFmpegError(f"réponse ffprobe invalide sur {video.name}")
    streams = payload.get("streams")
    if not isinstance(streams, list) or not streams:
        raise FFmpegError(f"aucun flux vidéo dans {video.name}")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise FFmpegError(f"métadonnées du flux vidéo invalides dans {video.name}")
    format_info = payload.get("format") if isinstance(payload.get("format"), dict) else {}

    duration = _to_float(stream.get("duration")) or _to_float(format_info.get("duration")) or 0.0
    fps = _parse_rate(stream.get("r_frame_rate", "0/1"))
    rotation = 0
    side_data = stream.get("side_data_list") if isinstance(stream.get("side_data_list"), list) else []
    for side in side_data:
        if isinstance(side, dict) and "rotation" in side:
            try:
                rotation = int(side["rotation"])
            except (TypeError, ValueError):
                rotation = 0

    width = _to_int(stream.get("width"))
    height = _to_int(stream.get("height"))
    if width <= 0 or height <= 0:
        raise FFmpegError(f"dimensions vidéo invalides dans {video.name}")

    return VideoInfo(
        path=video,
        width=width,
        height=height,
        duration=duration,
        fps=fps,
        codec=str(stream.get("codec_name") or "?"),
        color_transfer=str(stream.get("color_transfer") or ""),
        rotation=rotation,
    )


def _to_float(value: object) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
        return parsed if math.isfinite(parsed) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0


def _parse_rate(rate: object) -> float:
    rate = str(rate or "0/1")
    if "/" in rate:
        num, _, den = rate.partition("/")
        denominator = _to_float(den)
        return _to_float(num) / denominator if denominator else 0.0
    return _to_float(rate)


def _scale_filter(long_side: int) -> str:
    # Côté le plus long ramené à `long_side`, dimensions paires (exigence yuv420p).
    return (
        f"scale=w='if(gt(iw,ih),{long_side},-2)':h='if(gt(iw,ih),-2,{long_side})'"
        ":force_original_aspect_ratio=decrease:flags=lanczos"
    )


TONEMAP_CHAIN = (
    "zscale=transfer=linear:npl=100,format=gbrpf32le,zscale=primaries=bt709,"
    "tonemap=tonemap=hable:desat=0,zscale=transfer=bt709:matrix=bt709:range=tv"
)


def _build_filters(info: VideoInfo, sample_fps: float, long_side: int, tonemap: bool) -> str:
    chain = [f"fps={sample_fps:.6f}"]
    if tonemap:
        chain.append(TONEMAP_CHAIN)
    chain.append(_scale_filter(long_side))
    chain.append("blurdetect=block_pct=90")
    chain.append("metadata=mode=print:key=lavfi.blur:file=-")
    chain.append("format=yuvj420p")
    return ",".join(chain)


def _run_extraction(
    video: Path,
    out_dir: Path,
    filters: str,
    quality: int,
    should_stop: Callable[[], bool] | None = None,
) -> str:
    command = [
        _tool("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video),
        "-vf",
        filters,
        "-fps_mode",
        "passthrough",
        "-q:v",
        str(quality),
        str(out_dir / "cand_%05d.jpg"),
    ]
    if should_stop is not None and should_stop():
        raise FFmpegError("extraction interrompue")
    process = subprocess.Popen(  # noqa: S603 - argv sans shell, ffmpeg résolu par shutil.which
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + EXTRACTION_TIMEOUT_SECONDS
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.2)
            break
        except subprocess.TimeoutExpired as exc:
            interrupted = should_stop is not None and should_stop()
            timed_out = time.monotonic() >= deadline
            if not interrupted and not timed_out:
                continue
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            if interrupted:
                raise FFmpegError("extraction interrompue") from exc
            raise FFmpegError(
                f"ffmpeg n'a pas terminé après {EXTRACTION_TIMEOUT_SECONDS // 60} min "
                "(média corrompu ou anormalement long ?)"
            ) from exc
    if process.returncode != 0:
        raise FFmpegError(f"ffmpeg a échoué: {stderr.strip()[:600]}")
    return stdout


def _parse_blur(stdout: str) -> dict[int, float]:
    """`metadata=print` écrit `frame:N ...` puis `lavfi.blur=x` sur la ligne suivante."""
    blur: dict[int, float] = {}
    current: int | None = None
    for line in stdout.splitlines():
        frame_match = _METADATA_RE.match(line)
        if frame_match:
            current = int(frame_match.group(1))
            continue
        blur_match = _BLUR_RE.match(line.strip())
        if blur_match and current is not None:
            value = float(blur_match.group(1))
            if math.isfinite(value):
                blur[current] = value
            current = None
    return blur


def select_sharpest(candidates: list[Path], blur: dict[int, float], count: int) -> list[Path]:
    """Découpe la séquence en `count` intervalles et garde la plus nette de chacun.

    Conserve la couverture temporelle (donc angulaire) tout en éliminant les
    images bougées, ce qu'un simple tri global sur la netteté ne ferait pas.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("le nombre d'images doit être strictement positif")
    if len(candidates) <= count:
        return list(candidates)

    total = len(candidates)
    selected: list[Path] = []
    for bucket in range(count):
        start = math.floor(bucket * total / count)
        end = max(start + 1, math.floor((bucket + 1) * total / count))
        window = candidates[start:end]
        if not window:
            continue
        # `cand_00001.jpg` ↔ frame 0 côté métadonnées.
        best = min(window, key=lambda p: blur.get(_candidate_index(p) - 1, float("inf")))
        selected.append(best)
    return selected


def _candidate_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def extract_frames(
    video: Path,
    out_dir: Path,
    count: int = 120,
    long_side: int = 1024,
    oversample: int = 3,
    quality: int = 2,
    log: Callable[[str], None] = lambda _msg: None,
    should_stop: Callable[[], bool] | None = None,
) -> list[Path]:
    """Extrait `count` images nettes et uniformément réparties dans `out_dir`."""
    video = Path(video)
    out_dir = Path(out_dir)
    if not video.is_file():
        raise FFmpegError(f"vidéo introuvable : {video}")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("count doit être un entier strictement positif")
    if not isinstance(long_side, int) or isinstance(long_side, bool) or long_side <= 0 or long_side % 2:
        raise ValueError("long_side doit être un entier pair strictement positif")
    if not isinstance(oversample, int) or isinstance(oversample, bool) or oversample <= 0:
        raise ValueError("oversample doit être un entier strictement positif")
    if not isinstance(quality, int) or isinstance(quality, bool) or not 1 <= quality <= 31:
        raise ValueError("quality doit être un entier compris entre 1 et 31")

    info = probe(video)
    if should_stop is not None and should_stop():
        raise FFmpegError("extraction interrompue")
    log(
        f"vidéo {info.width}x{info.height} {info.fps:.1f} fps, {info.duration:.1f}s, "
        f"codec {info.codec}{', HDR' if info.is_hdr else ''}"
    )
    if info.duration <= 0:
        raise FFmpegError("durée de la vidéo inconnue (fichier corrompu ?)")

    # Ne demande jamais plus d'images que la source n'en contient : le filtre
    # fps dupliquerait sinon une courte séquence jusqu'à atteindre count.
    source_frames = max(1, math.ceil(info.duration * info.fps)) if info.fps > 0 else count
    candidates_wanted = min(count * oversample, source_frames)
    sample_fps = max(0.1, candidates_wanted / info.duration)

    out_dir.mkdir(parents=True, exist_ok=True)
    # L'extraction se fait à part : une vidéo invalide ou un ffmpeg en erreur ne
    # doit jamais effacer les images déjà présentes dans le dossier de sortie.
    with tempfile.TemporaryDirectory(prefix=".minids-frames-", dir=out_dir) as staging_raw:
        staging = Path(staging_raw)
        tonemap = info.is_hdr
        try:
            stdout = _run_extraction(
                video,
                staging,
                _build_filters(info, sample_fps, long_side, tonemap),
                quality,
                should_stop,
            )
        except FFmpegError as exc:
            if not tonemap:
                raise
            # zimg absent ou chaîne refusée : on repasse en conversion simple.
            log(f"tonemap HDR indisponible ({str(exc)[:120]}), extraction en conversion directe")
            for stale in staging.glob("cand_*.jpg"):
                stale.unlink()
            stdout = _run_extraction(
                video,
                staging,
                _build_filters(info, sample_fps, long_side, False),
                quality,
                should_stop,
            )

        candidates = sorted(staging.glob("cand_*.jpg"))
        if not candidates:
            raise FFmpegError("aucune image extraite (vidéo illisible ?)")
        blur = _parse_blur(stdout)
        log(f"{len(candidates)} candidates, mesure de flou sur {len(blur)}")

        selected = select_sharpest(candidates, blur, count)
        staged_frames: list[Path] = []
        for position, candidate in enumerate(selected):
            target = staging / f"frame_{position:05d}.jpg"
            candidate.replace(target)
            staged_frames.append(target)

        measured = [blur[_candidate_index(path) - 1] for path in selected if _candidate_index(path) - 1 in blur]
        rejected = len(candidates) - len(selected)
        if measured:
            log(
                f"{len(staged_frames)} images gardées ({rejected} rejetées), "
                f"flou moyen {sum(measured) / len(measured):.2f}"
            )
        else:
            log(f"{len(staged_frames)} images gardées ({rejected} rejetées), flou non mesuré")

        for stale in list(out_dir.glob("cand_*.jpg")) + list(out_dir.glob(FRAME_GLOB)):
            if stale.is_file() or stale.is_symlink():
                stale.unlink()
        frames: list[Path] = []
        for staged in staged_frames:
            target = out_dir / staged.name
            staged.replace(target)
            frames.append(target)
        return frames
