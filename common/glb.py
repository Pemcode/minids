"""Écriture de fichiers .glb (glTF 2.0 binaire), sans dépendance hors numpy.

Écrire le conteneur à la main plutôt que de passer par trimesh évite d'imposer
la même pile au serveur et au re-maillage local, et garantit la conversion de
repère : COLMAP/VGGT travaille en Y bas / Z devant, glTF en Y haut / Z derrière.
Sans cette rotation, l'objet apparaît à l'envers dans tous les visualiseurs.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np

# Rotation de 180° autour de X : (x, y, z) → (x, -y, -z).
COLMAP_TO_GLTF = np.diag([1.0, -1.0, -1.0])

_COMPONENT_FLOAT = 5126
_COMPONENT_UINT32 = 5125
_COMPONENT_UBYTE = 5121
_TARGET_ARRAY_BUFFER = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963


def encode_png(image: np.ndarray) -> bytes:
    """Encode un tableau RGB uint8 (H, W, 3) en PNG (stdlib uniquement)."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image RGB (H, W, 3) attendue, reçu {image.shape}")
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape[:2]

    raw = bytearray()
    for row in image:
        raw.append(0)  # type de filtre : aucun
        raw.extend(row.tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


class _BufferBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.views: list[dict[str, Any]] = []

    def add_view(self, payload: bytes, target: int | None = None) -> int:
        while len(self.data) % 4:
            self.data.append(0)
        offset = len(self.data)
        self.data.extend(payload)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            view["target"] = target
        self.views.append(view)
        return len(self.views) - 1


def write_glb(
    path: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    vertex_colors: np.ndarray | None = None,
    texture_png: bytes | None = None,
    convert_axes: bool = True,
    name: str = "minids",
) -> Path:
    """Écrit un .glb à partir d'un maillage triangulaire.

    `vertex_colors` en float [0,1] ou uint8 ; ignoré si une texture est fournie.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float64)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"sommets (N, 3) attendus, reçu {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces (M, 3) attendues, reçu {faces.shape}")
    if len(faces) and int(faces.max()) >= len(vertices):
        raise ValueError("indice de face hors bornes")

    if convert_axes:
        vertices = vertices @ COLMAP_TO_GLTF.T
        if normals is not None:
            normals = np.asarray(normals, dtype=np.float64) @ COLMAP_TO_GLTF.T
        # Le miroir sur deux axes préserve l'orientation : pas de réindexation des faces.

    builder = _BufferBuilder()
    accessors: list[dict[str, Any]] = []
    attributes: dict[str, int] = {}

    positions = vertices.astype(np.float32)
    view = builder.add_view(positions.tobytes(), _TARGET_ARRAY_BUFFER)
    accessors.append({
        "bufferView": view, "componentType": _COMPONENT_FLOAT, "count": len(positions), "type": "VEC3",
        "min": positions.min(axis=0).tolist(), "max": positions.max(axis=0).tolist(),
    })
    attributes["POSITION"] = len(accessors) - 1

    if normals is not None:
        array = np.ascontiguousarray(normals, dtype=np.float32)
        lengths = np.linalg.norm(array, axis=1, keepdims=True)
        array = array / np.where(lengths < 1e-12, 1.0, lengths)
        view = builder.add_view(array.astype(np.float32).tobytes(), _TARGET_ARRAY_BUFFER)
        accessors.append({"bufferView": view, "componentType": _COMPONENT_FLOAT, "count": len(array), "type": "VEC3"})
        attributes["NORMAL"] = len(accessors) - 1

    if uvs is not None and texture_png is not None:
        array = np.ascontiguousarray(uvs, dtype=np.float32)
        view = builder.add_view(array.tobytes(), _TARGET_ARRAY_BUFFER)
        accessors.append({"bufferView": view, "componentType": _COMPONENT_FLOAT, "count": len(array), "type": "VEC2"})
        attributes["TEXCOORD_0"] = len(accessors) - 1
    elif vertex_colors is not None:
        colors = np.ascontiguousarray(vertex_colors)
        if colors.dtype != np.uint8:
            colors = (np.clip(colors, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        if colors.shape[1] == 3:
            colors = np.concatenate([colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1)
        view = builder.add_view(colors.tobytes(), _TARGET_ARRAY_BUFFER)
        accessors.append({
            "bufferView": view, "componentType": _COMPONENT_UBYTE, "count": len(colors),
            "type": "VEC4", "normalized": True,
        })
        attributes["COLOR_0"] = len(accessors) - 1

    indices = faces.reshape(-1).astype(np.uint32)
    view = builder.add_view(indices.tobytes(), _TARGET_ELEMENT_ARRAY_BUFFER)
    accessors.append({
        "bufferView": view, "componentType": _COMPONENT_UINT32, "count": len(indices), "type": "SCALAR"
    })
    indices_accessor = len(accessors) - 1

    material: dict[str, Any] = {
        "name": f"{name}_material",
        "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.9},
        "doubleSided": True,
    }
    gltf: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "miniDS"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name}],
        "meshes": [{"name": name, "primitives": [{"attributes": attributes, "indices": indices_accessor, "material": 0, "mode": 4}]}],
        "accessors": accessors,
        "materials": [material],
    }

    if "TEXCOORD_0" in attributes and texture_png is not None:
        image_view = builder.add_view(texture_png)
        gltf["images"] = [{"bufferView": image_view, "mimeType": "image/png", "name": f"{name}_basecolor"}]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0, "texCoord": 0}
    else:
        material["pbrMetallicRoughness"]["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]

    gltf["bufferViews"] = builder.views
    gltf["buffers"] = [{"byteLength": len(builder.data)}]

    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    binary = bytes(builder.data)
    binary += b"\x00" * ((4 - len(binary) % 4) % 4)

    total = 12 + 8 + len(json_bytes) + 8 + len(binary)
    path = Path(path)
    with path.open("wb") as handle:
        handle.write(struct.pack("<III", 0x46546C67, 2, total))
        handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        handle.write(json_bytes)
        handle.write(struct.pack("<II", len(binary), 0x004E4942))
        handle.write(binary)
    return path


def read_glb_summary(path: Path) -> dict[str, Any]:
    """Relit l'en-tête d'un .glb : sert aux tests et au rapport de benchmark."""
    with Path(path).open("rb") as handle:
        magic, version, total = struct.unpack("<III", handle.read(12))
        if magic != 0x46546C67:
            raise ValueError("ce n'est pas un fichier GLB")
        length, kind = struct.unpack("<II", handle.read(8))
        if kind != 0x4E4F534A:
            raise ValueError("premier chunk non-JSON")
        gltf = json.loads(handle.read(length))

    primitive = gltf["meshes"][0]["primitives"][0]
    accessors = gltf["accessors"]
    position = accessors[primitive["attributes"]["POSITION"]]
    return {
        "version": version,
        "file_size": total,
        "vertices": position["count"],
        "triangles": accessors[primitive["indices"]]["count"] // 3,
        "attributes": sorted(primitive["attributes"]),
        "has_texture": bool(gltf.get("images")),
        "min": position["min"],
        "max": position["max"],
    }
