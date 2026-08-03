"""Le .glb est le livrable final : s'il est mal formé, tout le reste est perdu."""

from __future__ import annotations

import json
import struct
import zlib

import numpy as np
import pytest

from common.glb import COLMAP_TO_GLTF, encode_png, read_glb_summary, write_glb


def triangle_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.uint32)
    return vertices, faces


def test_glb_roundtrip_with_vertex_colors(tmp_path):
    vertices, faces = triangle_mesh()
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])

    path = write_glb(tmp_path / "mesh.glb", vertices, faces, vertex_colors=colors)
    summary = read_glb_summary(path)

    assert summary["vertices"] == 4
    assert summary["triangles"] == 4
    assert "COLOR_0" in summary["attributes"]
    assert not summary["has_texture"]
    assert summary["file_size"] == path.stat().st_size


def test_glb_with_texture(tmp_path):
    vertices, faces = triangle_mesh()
    uvs = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    texture = encode_png(np.full((8, 8, 3), 128, dtype=np.uint8))

    path = write_glb(tmp_path / "mesh.glb", vertices, faces, uvs=uvs, texture_png=texture)
    summary = read_glb_summary(path)

    assert "TEXCOORD_0" in summary["attributes"]
    assert summary["has_texture"]


def test_chunks_are_four_byte_aligned(tmp_path):
    """Un décalage non aligné casse silencieusement certains visualiseurs."""
    vertices, faces = triangle_mesh()
    path = write_glb(tmp_path / "mesh.glb", vertices, faces, vertex_colors=np.ones((4, 3)))

    data = path.read_bytes()
    json_length = struct.unpack("<I", data[12:16])[0]
    assert json_length % 4 == 0
    binary_length = struct.unpack("<I", data[20 + json_length : 24 + json_length])[0]
    assert binary_length % 4 == 0
    assert 12 + 8 + json_length + 8 + binary_length == len(data)

    gltf = json.loads(data[20 : 20 + json_length])
    for view in gltf["bufferViews"]:
        assert view["byteOffset"] % 4 == 0


def test_axis_conversion_puts_object_upright(tmp_path):
    """VGGT a Y vers le bas, glTF Y vers le haut : sans rotation, tout est renversé."""
    vertices = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    path = write_glb(tmp_path / "mesh.glb", vertices, faces, vertex_colors=np.ones((3, 3)))
    with path.open("rb") as handle:
        handle.seek(12)
        json_length = struct.unpack("<I", handle.read(4))[0]
        handle.read(4)
        gltf = json.loads(handle.read(json_length))
    position = gltf["accessors"][gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]]

    # Le sommet à y=-1 (bas en convention VGGT) doit se retrouver à y=+1.
    assert position["max"][1] == pytest.approx(1.0)
    assert np.linalg.det(COLMAP_TO_GLTF) == pytest.approx(1.0)  # rotation propre : winding conservé


def test_write_glb_rejects_bad_indices(tmp_path):
    vertices, _ = triangle_mesh()
    with pytest.raises(ValueError, match="hors bornes"):
        write_glb(tmp_path / "bad.glb", vertices, np.array([[0, 1, 99]], dtype=np.uint32))


def test_encode_png_is_decodable():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    image[..., 0] = np.arange(6, dtype=np.uint8)[None, :]

    payload = encode_png(image)
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"

    width, height, depth, color_type = struct.unpack(">IIBB", payload[16:26])
    assert (width, height, depth, color_type) == (6, 4, 8, 2)

    idat_start = payload.index(b"IDAT") + 4
    idat_length = struct.unpack(">I", payload[idat_start - 8 : idat_start - 4])[0]
    raw = zlib.decompress(payload[idat_start : idat_start + idat_length])
    assert len(raw) == height * (1 + width * 3)
    assert raw[0] == 0  # octet de filtre
    assert np.frombuffer(raw[1 : 1 + width * 3], dtype=np.uint8).reshape(width, 3)[:, 0].tolist() == list(range(6))
