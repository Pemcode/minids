"""Extraction ffmpeg : sélection des images et lecture des métadonnées de flou.

Le test d'intégration fabrique une vidéo où l'on sait *où* est le flou, puis
vérifie que l'extraction a bien gardé les images nettes. C'est la seule façon
de valider le sens de la métrique `lavfi.blur` (élevée = floue).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from common.video import _parse_blur, extract_frames, probe, select_sharpest

pytestmark = pytest.mark.filterwarnings("ignore")
HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def test_parse_blur_pairs_frames_and_values():
    stdout = (
        "frame:0    pts:0       pts_time:0\n"
        "lavfi.blur=4.529226\n"
        "frame:1    pts:1       pts_time:0.2\n"
        "lavfi.blur=31.747578\n"
    )
    assert _parse_blur(stdout) == {0: 4.529226, 1: 31.747578}


def test_parse_blur_ignores_other_metadata():
    stdout = "frame:0 pts:0\nlavfi.entropy.entropy=3.2\nlavfi.blur=1.5\n"
    assert _parse_blur(stdout) == {0: 1.5}


def test_select_sharpest_keeps_one_per_interval():
    candidates = [Path(f"cand_{i + 1:05d}.jpg") for i in range(9)]
    # La plus nette de chaque groupe de trois : indices 1, 4, 7 (base 0 côté métadonnées).
    blur = {0: 9.0, 1: 1.0, 2: 8.0, 3: 9.0, 4: 2.0, 5: 7.0, 6: 9.0, 7: 3.0, 8: 6.0}

    selected = select_sharpest(candidates, blur, count=3)

    assert [p.name for p in selected] == ["cand_00002.jpg", "cand_00005.jpg", "cand_00008.jpg"]


def test_select_sharpest_preserves_temporal_order():
    candidates = [Path(f"cand_{i + 1:05d}.jpg") for i in range(20)]
    blur = {i: float(20 - i) for i in range(20)}  # les dernières sont les plus nettes
    selected = select_sharpest(candidates, blur, count=4)

    assert len(selected) == 4
    assert selected == sorted(selected, key=lambda p: p.name)
    # Malgré le gradient de netteté, la couverture temporelle est conservée.
    assert selected[0].name < "cand_00006.jpg"


def test_select_sharpest_returns_all_when_not_enough():
    candidates = [Path("cand_00001.jpg"), Path("cand_00002.jpg")]
    assert select_sharpest(candidates, {}, count=10) == candidates


def _make_video(path: Path, blurred_second_half: bool) -> None:
    """4 s de mire ; option : seconde moitié floutée."""
    filters = "testsrc2=size=320x240:rate=10:duration=4"
    command = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", filters]
    if blurred_second_half:
        command += ["-vf", "boxblur=12:2:enable='gte(t,2)'"]
    command += ["-pix_fmt", "yuv420p", str(path)]
    subprocess.run(command, check=True, capture_output=True)


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe absents")
def test_probe_reads_stream_metadata(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, blurred_second_half=False)

    info = probe(video)

    assert (info.width, info.height) == (320, 240)
    assert info.duration == pytest.approx(4.0, abs=0.3)
    assert info.fps == pytest.approx(10.0, abs=0.1)
    assert not info.is_hdr


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe absents")
def test_extract_frames_produces_requested_count(tmp_path):
    video = tmp_path / "clip.mp4"
    _make_video(video, blurred_second_half=False)

    frames = extract_frames(video, tmp_path / "frames", count=8, long_side=160)

    assert len(frames) == 8
    assert [p.name for p in frames] == [f"frame_{i:05d}.jpg" for i in range(8)]
    assert all(p.stat().st_size > 0 for p in frames)
    assert not list((tmp_path / "frames").glob("cand_*.jpg"))  # candidates nettoyées


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg/ffprobe absents")
def test_extract_frames_prefers_sharp_images(tmp_path):
    """Sur une vidéo dont la seconde moitié est floue, les images retenues
    doivent être nettement plus nettes que la moyenne des candidates."""
    video = tmp_path / "clip.mp4"
    _make_video(video, blurred_second_half=True)

    logs: list[str] = []
    frames = extract_frames(video, tmp_path / "frames", count=6, long_side=160, oversample=4, log=logs.append)

    assert len(frames) == 6
    summary = next(line for line in logs if "images gardées" in line)
    kept_blur = float(summary.rsplit(" ", 1)[1])
    # Le flou moyen des candidates est ~2× supérieur (moitié nette, moitié floutée).
    assert kept_blur < 20.0, summary
