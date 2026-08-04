"""Écriture de fichiers .glb (glTF 2.0 binaire), sans dépendance hors numpy.

Écrire le conteneur à la main plutôt que de passer par trimesh évite d'imposer
la même pile au serveur et au re-maillage local, et garantit la conversion de
repère : COLMAP/VGGT travaille en Y bas / Z devant, glTF en Y haut / Z derrière.
Sans cette rotation, l'objet apparaît à l'envers dans tous les visualiseurs.
"""

from __future__ import annotations

import json
import os
import struct
import tempfile
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
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image RGB (H, W, 3) attendue, reçu {image.shape}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError("une image PNG ne peut pas être vide")
    if image.dtype != np.uint8:
        raise ValueError(f"image uint8 attendue, reçu {image.dtype}")
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]

    raw = bytearray()
    for row in image:
        raw.append(0)  # type de filtre : aucun
        raw.extend(row.tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

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
    vertices = _float_attribute(vertices, "sommets", 3)
    if len(vertices) == 0:
        raise ValueError("le maillage ne contient aucun sommet")

    raw_faces = np.asarray(faces)
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3:
        raise ValueError(f"faces (M, 3) attendues, reçu {raw_faces.shape}")
    if len(raw_faces) == 0:
        raise ValueError("le maillage ne contient aucune face")
    if not np.issubdtype(raw_faces.dtype, np.integer):
        raise ValueError(f"indices de faces entiers attendus, reçu {raw_faces.dtype}")
    if bool(np.any(raw_faces < 0)) or int(raw_faces.max()) >= len(vertices):
        raise ValueError("indice de face hors bornes")
    faces = np.ascontiguousarray(raw_faces, dtype=np.uint32)

    if normals is not None:
        normals = _float_attribute(normals, "normales", 3, len(vertices))
    if uvs is not None and texture_png is not None:
        uvs = _float_attribute(uvs, "UV", 2, len(vertices))
    if texture_png is not None:
        if uvs is None:
            raise ValueError("une texture nécessite des coordonnées UV")
        _validate_png(texture_png)
    if vertex_colors is not None and texture_png is None:
        colors_raw = np.asarray(vertex_colors)
        if colors_raw.ndim != 2 or colors_raw.shape[0] != len(vertices) or colors_raw.shape[1] not in {3, 4}:
            raise ValueError(f"couleurs (N, 3) ou (N, 4) attendues avec N={len(vertices)}, reçu {colors_raw.shape}")
        if colors_raw.dtype != np.uint8 and not np.issubdtype(colors_raw.dtype, np.floating):
            raise ValueError("les couleurs doivent être en float [0,1] ou uint8")
        if not bool(np.all(np.isfinite(colors_raw))):
            raise ValueError("les couleurs contiennent une valeur non finie")
        if np.issubdtype(colors_raw.dtype, np.floating) and bool(np.any((colors_raw < 0) | (colors_raw > 1))):
            raise ValueError("les couleurs float doivent rester dans l'intervalle [0,1]")
        vertex_colors = colors_raw

    if convert_axes:
        vertices = vertices @ COLMAP_TO_GLTF.T
        if normals is not None:
            normals = normals @ COLMAP_TO_GLTF.T
        # Le miroir sur deux axes préserve l'orientation : pas de réindexation des faces.

    builder = _BufferBuilder()
    accessors: list[dict[str, Any]] = []
    attributes: dict[str, int] = {}

    positions = vertices.astype(np.float32)
    view = builder.add_view(positions.tobytes(), _TARGET_ARRAY_BUFFER)
    accessors.append(
        {
            "bufferView": view,
            "componentType": _COMPONENT_FLOAT,
            "count": len(positions),
            "type": "VEC3",
            "min": positions.min(axis=0).tolist(),
            "max": positions.max(axis=0).tolist(),
        }
    )
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
        accessors.append(
            {
                "bufferView": view,
                "componentType": _COMPONENT_UBYTE,
                "count": len(colors),
                "type": "VEC4",
                "normalized": True,
            }
        )
        attributes["COLOR_0"] = len(accessors) - 1

    indices = faces.reshape(-1).astype(np.uint32)
    view = builder.add_view(indices.tobytes(), _TARGET_ELEMENT_ARRAY_BUFFER)
    accessors.append({"bufferView": view, "componentType": _COMPONENT_UINT32, "count": len(indices), "type": "SCALAR"})
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
        "meshes": [
            {
                "name": name,
                "primitives": [{"attributes": attributes, "indices": indices_accessor, "material": 0, "mode": 4}],
            }
        ],
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(struct.pack("<III", 0x46546C67, 2, total))
            handle.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
            handle.write(json_bytes)
            handle.write(struct.pack("<II", len(binary), 0x004E4942))
            handle.write(binary)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def read_glb_summary(path: Path) -> dict[str, Any]:
    """Relit l'en-tête d'un .glb : sert aux tests et au rapport de benchmark."""
    path = Path(path)
    actual_size = path.stat().st_size
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError("en-tête GLB tronqué")
        magic, version, total = struct.unpack("<III", header)
        if magic != 0x46546C67:
            raise ValueError("ce n'est pas un fichier GLB")
        if version != 2:
            raise ValueError(f"version GLB non prise en charge : {version}")
        if total != actual_size:
            raise ValueError(f"taille GLB incohérente : en-tête {total}, fichier {actual_size}")
        chunk_header = handle.read(8)
        if len(chunk_header) != 8:
            raise ValueError("en-tête du chunk JSON tronqué")
        length, kind = struct.unpack("<II", chunk_header)
        if kind != 0x4E4F534A:
            raise ValueError("premier chunk non-JSON")
        payload = handle.read(length)
        if len(payload) != length:
            raise ValueError("chunk JSON tronqué")
        try:
            gltf = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("chunk JSON invalide") from exc

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


def _float_attribute(value: np.ndarray, label: str, columns: int, count: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != columns:
        raise ValueError(f"{label} (N, {columns}) attendus, reçu {raw.shape}")
    if count is not None and len(raw) != count:
        raise ValueError(f"{label} : {count} lignes attendues, reçu {len(raw)}")
    try:
        array = np.ascontiguousarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} numériques attendus") from exc
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{label} : valeur non finie")
    if bool(np.any(np.abs(array) > np.finfo(np.float32).max)):
        raise ValueError(f"{label} : valeur hors plage float32")
    return array


def _validate_png(payload: bytes) -> None:
    if not isinstance(payload, bytes) or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("texture_png doit contenir une image PNG valide")
    offset = 8
    seen_idat = False
    first_chunk = True
    while offset + 12 <= len(payload):
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(payload):
            break
        tag = payload[offset + 4 : offset + 8]
        data = payload[offset + 8 : offset + 8 + length]
        checksum = struct.unpack(">I", payload[offset + 8 + length : end])[0]
        if checksum != zlib.crc32(tag + data) & 0xFFFFFFFF:
            break
        if first_chunk and (tag != b"IHDR" or length != 13):
            break
        first_chunk = False
        if tag == b"IHDR":
            width, height = struct.unpack(">II", data[:8])
            if width == 0 or height == 0:
                break
        elif tag == b"IDAT":
            seen_idat = True
        elif tag == b"IEND":
            if length == 0 and seen_idat and end == len(payload):
                return
            break
        offset = end
    raise ValueError("texture_png doit contenir une image PNG valide")
