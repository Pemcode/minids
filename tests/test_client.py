"""Cas limites du payload et de la CLI locale."""

from __future__ import annotations

import tarfile
from types import SimpleNamespace

import pytest

from client import minids
from client import payload as payload_module
from client.payload import build_payload
from client.transport import MinidsError
from common.video import FFmpegError


def test_build_payload_excludes_stale_files_from_a_reused_workdir(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "one.png").write_bytes(b"new")
    workdir = tmp_path / "work"
    frames = workdir / "frames"
    frames.mkdir(parents=True)
    (frames / "frame_99999.jpg").write_bytes(b"stale")

    archive = build_payload(source, workdir)

    with tarfile.open(archive) as handle:
        assert handle.getnames() == ["frame_00000.png"]


def test_build_payload_uniformly_limits_a_large_image_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(10):
        (source / f"{index:02d}.png").write_bytes(bytes([index]))

    archive = build_payload(source, tmp_path / "work", frames=3)

    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        values = [handle.extractfile(member).read() for member in members]
    assert values == [b"\x00", b"\x04", b"\x09"]


def test_send_video_probes_the_source_before_upload(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    calls = []
    monkeypatch.setattr(
        payload_module,
        "probe",
        lambda path: calls.append(path) or SimpleNamespace(duration=1.0, fps=25.0),
    )

    assert build_payload(source, tmp_path / "work", send_video=True) == source
    assert calls == [source]


def test_send_video_rejects_a_non_video_suffix(tmp_path, monkeypatch):
    source = tmp_path / "payload.tar"
    source.write_bytes(b"not a video")
    monkeypatch.setattr(payload_module, "probe", lambda _path: pytest.fail("probe ne doit pas être appelé"))

    with pytest.raises(ValueError, match="format vidéo"):
        build_payload(source, tmp_path / "work", send_video=True)


@pytest.mark.parametrize(
    "option",
    [
        ["--frames", "601"],
        ["--gs-iters", "499"],
        ["--texture-size", "4097"],
        ["--target-triangles", "4999"],
        ["--voxel-divisor", "63"],
    ],
)
def test_cli_rejects_server_bounds_before_preparing_payload(option):
    with pytest.raises(SystemExit):
        minids.build_parser().parse_args(
            ["--url", "https://pod.example", "--token", "x", "submit", "video.mp4", *option]
        )


def test_cli_rejects_an_oversized_chunk():
    with pytest.raises(SystemExit):
        minids.build_parser().parse_args(
            [
                "--url",
                "https://pod.example",
                "--token",
                "x",
                "--chunk-size",
                str(64 * 1024 * 1024 + 1),
                "submit",
                "video.mp4",
            ]
        )


def test_no_refine_maps_2dgs_to_tsdf_without_losing_other_backends():
    args = minids.build_parser().parse_args(
        [
            "--url",
            "https://pod.example",
            "--token",
            "x",
            "submit",
            "video.mp4",
            "--no-refine",
            "--backends",
            "tsdf2dgs,poisson",
        ]
    )
    assert minids.job_params(args)["mesh_backends"] == ["tsdf", "poisson"]


def test_sam3_requires_a_prompt_before_job_creation():
    args = minids.build_parser().parse_args(
        ["--url", "https://pod.example", "--token", "x", "submit", "video.mp4", "--segmentation", "sam3"]
    )
    with pytest.raises(MinidsError, match="prompt"):
        minids.job_params(args)


def test_cli_turns_ffmpeg_failure_into_a_clean_exit(tmp_path, monkeypatch, capsys):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"invalid")
    monkeypatch.setattr(minids, "extract_frames", lambda *args, **kwargs: (_ for _ in ()).throw(FFmpegError("boom")))

    assert minids.main(["--url", "https://pod.example", "--token", "x", "extract", str(video)]) == 2
    assert "boom" in capsys.readouterr().err
