from __future__ import annotations

import argparse
from types import SimpleNamespace

import numpy as np
import pytest

from bench import compare_meshes


@pytest.mark.parametrize("value", ["0", "-3", "abc"])
def test_positive_int_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        compare_meshes._positive_int(value)


@pytest.mark.parametrize("value", ["0", "-3", "nan", "inf", "abc"])
def test_positive_float_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        compare_meshes._positive_float(value)


def test_parse_backends_normalizes_and_deduplicates():
    assert compare_meshes._parse_backends(" TSDF,poisson,tsdf ") == ["tsdf", "poisson"]


@pytest.mark.parametrize("value", ["", "tsdf,inconnu"])
def test_parse_backends_rejects_unusable_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        compare_meshes._parse_backends(value)


def test_build_meshes_reports_empty_depths_clearly(monkeypatch, tmp_path):
    result = SimpleNamespace(
        depth=np.zeros((1, 2, 2), dtype=np.float32),
        extrinsics=np.eye(4, dtype=np.float32)[None],
        intrinsics=np.eye(3, dtype=np.float32)[None],
    )
    normalization = SimpleNamespace(
        apply_extrinsics=lambda value: value,
        apply_depth=lambda value: value,
    )
    monkeypatch.setattr(compare_meshes, "load_raw", lambda _path: (result, normalization, None))

    with pytest.raises(ValueError, match="aucun point 3D valide"):
        compare_meshes.build_meshes(tmp_path / "raw.npz", ["tsdf"], 512, 20_000, "CPU:0", True)


def test_build_meshes_rejects_unknown_backend_before_loading(monkeypatch, tmp_path):
    monkeypatch.setattr(compare_meshes, "load_raw", lambda _path: pytest.fail("le NPZ ne doit pas être chargé"))

    with pytest.raises(ValueError, match="backend de benchmark inconnu"):
        compare_meshes.build_meshes(tmp_path / "raw.npz", ["inconnu"], 512, 20_000, "CPU:0", True)


def test_render_markdown_handles_missing_and_float_values():
    rendered = compare_meshes.render_markdown(
        [{"backend": "tsdf", "seconds": 1.23456, "triangles": 42, "watertight": True}]
    )

    assert "| tsdf | 1.235 | 42 | True |" in rendered
    assert rendered.endswith("\n")


def test_normalized_scale_rejects_degenerate_mesh():
    mesh = SimpleNamespace(vertices=np.ones((3, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="dégénéré"):
        compare_meshes.normalized_scale(mesh)


def test_reference_export_scale_uses_explicit_or_sibling_report(tmp_path):
    reference = tmp_path / "mesh.glb"
    reference.write_bytes(b"glTF")
    (tmp_path / "report.json").write_text('{"export_scale": 2.5}', encoding="utf-8")

    assert compare_meshes.reference_export_scale(reference) == pytest.approx(2.5)
    assert compare_meshes.reference_export_scale(reference, 4.0) == pytest.approx(4.0)


@pytest.mark.parametrize("payload", ['{"export_scale": 0}', '{"export_scale": "nan"}', "[]", "{"])
def test_reference_export_scale_rejects_invalid_report_values(tmp_path, payload):
    reference = tmp_path / "mesh.glb"
    reference.write_bytes(b"glTF")
    (tmp_path / "report.json").write_text(payload, encoding="utf-8")

    if payload == "[]":
        assert compare_meshes.reference_export_scale(reference) == 1.0
    else:
        with pytest.raises(ValueError):
            compare_meshes.reference_export_scale(reference)
