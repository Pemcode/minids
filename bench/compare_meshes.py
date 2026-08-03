"""Benchmark des backends de maillage sur *tes* données.

Le classement annoncé dans le README vient de la littérature ; celui-ci vient de
ton scan. On repart de `vggt_raw.npz`, on reconstruit avec chaque backend
demandé, et on compare au maillage de référence (par défaut le `mesh.glb`
produit par la chaîne 2DGS→TSDF, qui est la plus coûteuse).

    python bench/compare_meshes.py --raw out/<job>/vggt_raw.npz \\
        --reference out/<job>/mesh.glb --backends tsdf,poisson --out bench_out

Métriques : temps, triangles, étanchéité, aire, et distance de Chamfer
symétrique (moyenne des deux distances point-à-surface).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.glb import COLMAP_TO_GLTF  # noqa: E402
from server.pipeline import cleanup as cleanup_module  # noqa: E402
from server.pipeline import mesh_poisson, mesh_tsdf  # noqa: E402
from server.pipeline.remesh import load_raw  # noqa: E402

SAMPLE_POINTS = 200_000


def build_meshes(
    raw: Path, backends: list[str], voxel_divisor: int, target_triangles: int, device: str, use_masks: bool
) -> dict[str, dict[str, Any]]:
    from server.pipeline.geometry import unproject

    result, normalization, masks = load_raw(raw)
    if not use_masks:
        masks = None
    extrinsics = normalization.apply_extrinsics(result.extrinsics)
    depths = normalization.apply_depth(result.depth)

    collected = []
    for index in range(len(depths)):
        keep = np.isfinite(depths[index]) & (depths[index] > 0)
        if masks is not None:
            keep &= masks[index]
        if keep.any():
            collected.append(unproject(depths[index], result.intrinsics[index], extrinsics[index])[keep][::7])
    points = np.concatenate(collected)
    bbox_min = np.percentile(points, 0.5, axis=0)
    bbox_max = np.percentile(points, 99.5, axis=0)
    voxel_size = mesh_tsdf.voxel_size_from_bbox(bbox_min, bbox_max, voxel_divisor)

    outputs: dict[str, dict[str, Any]] = {}
    for backend in backends:
        started = time.time()
        if backend == "tsdf":
            mesh = mesh_tsdf.fuse(
                depths=depths, colors=result.images, intrinsics=result.intrinsics, extrinsics=extrinsics,
                config=mesh_tsdf.TSDFConfig(voxel_size=voxel_size), masks=masks, device=device, log_fn=print,
            )
        elif backend == "poisson":
            cloud_points, cloud_colors = mesh_poisson.point_cloud_from_depths(
                depths, result.images, result.intrinsics, extrinsics, masks=masks
            )
            mesh = mesh_poisson.reconstruct(
                cloud_points, cloud_colors, extrinsics,
                mesh_poisson.PoissonConfig(voxel_size=voxel_size), print,
            )
        else:
            print(f"backend '{backend}' ignoré (non reproductible hors pipeline complet)")
            continue

        mesh = cleanup_module.clean(
            mesh, cleanup_module.CleanupConfig(target_triangles=target_triangles),
            bbox_min, bbox_max, None, voxel_size, print,
        )
        outputs[backend] = {"mesh": mesh, "seconds": round(time.time() - started, 2)}
    return outputs


def load_reference(path: Path, from_glb: bool) -> Any:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not len(mesh.triangles):
        raise ValueError(f"maillage de référence vide: {path}")
    if from_glb:
        # Annule la conversion d'axes appliquée à l'export (rotation involutive).
        mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) @ COLMAP_TO_GLTF.T)
    return mesh


def chamfer(mesh_a: Any, mesh_b: Any, samples: int = SAMPLE_POINTS) -> dict[str, float]:
    """Distance de Chamfer symétrique entre deux maillages échantillonnés."""
    import open3d as o3d

    # `sample_points_uniformly` n'a pas de paramètre `seed` (vérifié en 0.19) :
    # le déterminisme passe par le générateur global d'Open3D.
    o3d.utility.random.seed(0)
    cloud_a = mesh_a.sample_points_uniformly(number_of_points=samples)
    cloud_b = mesh_b.sample_points_uniformly(number_of_points=samples)
    forward = np.asarray(cloud_a.compute_point_cloud_distance(cloud_b))
    backward = np.asarray(cloud_b.compute_point_cloud_distance(cloud_a))
    return {
        "chamfer_mean": float((forward.mean() + backward.mean()) / 2),
        "chamfer_median": float((np.median(forward) + np.median(backward)) / 2),
        "hausdorff_p95": float(max(np.percentile(forward, 95), np.percentile(backward, 95))),
    }


def normalized_scale(mesh: Any) -> float:
    vertices = np.asarray(mesh.vertices)
    return float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))) or 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True, help="vggt_raw.npz du job")
    parser.add_argument("--reference", type=Path, default=None, help="mesh.glb de référence (2DGS→TSDF)")
    parser.add_argument("--backends", default="tsdf,poisson")
    parser.add_argument("--out", type=Path, default=Path("bench_out"))
    parser.add_argument("--voxel-divisor", type=int, default=512)
    parser.add_argument("--target-triangles", type=int, default=200_000)
    parser.add_argument("--device", default="CPU:0", help="CUDA:0 sur le pod, CPU:0 en local")
    parser.add_argument("--no-masks", action="store_true")
    args = parser.parse_args(argv)

    from common.glb import write_glb

    args.out.mkdir(parents=True, exist_ok=True)
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = build_meshes(
        args.raw, backends, args.voxel_divisor, args.target_triangles, args.device, not args.no_masks
    )

    reference = None
    if args.reference and Path(args.reference).exists():
        reference = load_reference(Path(args.reference), Path(args.reference).suffix.lower() == ".glb")
        results["reference (2dgs→tsdf)"] = {"mesh": reference, "seconds": None}

    rows: list[dict[str, Any]] = []
    for name, entry in results.items():
        mesh = entry["mesh"]
        row: dict[str, Any] = {"backend": name, "seconds": entry["seconds"], **cleanup_module.mesh_metrics(mesh)}
        if reference is not None and mesh is not reference:
            scale = normalized_scale(reference)
            distances = chamfer(mesh, reference)
            # Exprimé en % de la diagonale : comparable d'un scan à l'autre.
            row.update({key: round(100.0 * value / scale, 3) for key, value in distances.items()})
        rows.append(row)

        if mesh is not reference:
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
            write_glb(
                path=args.out / f"mesh_{name}.glb",
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles),
                normals=np.asarray(mesh.vertex_normals),
                vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
                name=f"bench_{name}",
            )

    (args.out / "report.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    (args.out / "report.md").write_text(render_markdown(rows), encoding="utf-8")
    print("\n" + render_markdown(rows))
    print(f"→ {args.out / 'report.md'}")
    return 0


def render_markdown(rows: list[dict[str, Any]]) -> str:
    headers = [
        ("backend", "Backend"), ("seconds", "Temps (s)"), ("triangles", "Triangles"),
        ("watertight", "Étanche"), ("surface_area", "Aire"),
        ("chamfer_mean", "Chamfer moy. (% diag)"), ("hausdorff_p95", "P95 (% diag)"),
    ]
    lines = [
        "# Benchmark des backends de maillage",
        "",
        "Distances exprimées en pourcentage de la diagonale de l'objet — plus bas = plus proche de la référence.",
        "",
        "| " + " | ".join(label for _key, label in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        cells = []
        for key, _label in headers:
            value = row.get(key)
            if value is None:
                cells.append("—")
            elif isinstance(value, float):
                cells.append(f"{value:.3f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
