"""CLI miniDS — pilote un pod RunPod depuis Windows, sans terminal web.

    python client/minids.py run video.mp4 --prompt "the sneaker"

Le client extrait les images en local avec ffmpeg : uploader 120 JPEG plutôt
qu'une vidéo 4K fait passer le transfert de ~500 Mo à ~20 Mo, ce qui change tout
sur une liaison montante domestique.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # exécution directe : `python client/minids.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.payload import build_payload  # noqa: E402
from client.transport import (  # noqa: E402
    DEFAULT_CHUNK,
    MAX_CHUNK,
    MinidsClient,
    MinidsError,
    artifact_destination,
    job_destination,
    sha256_file,
)
from common.video import FFmpegError, extract_frames  # noqa: E402

DEFAULT_ARTIFACTS = ["mesh.glb", "report.json", "preview.png"]


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("un entier est requis") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("la valeur doit être strictement positive")
    return parsed


def _even_positive_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed % 2:
        raise argparse.ArgumentTypeError("la valeur doit être paire")
    return parsed


def _bounded_int(value: str, minimum: int, maximum: int, label: str) -> int:
    parsed = _positive_int(value)
    if not minimum <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"{label} doit être compris entre {minimum} et {maximum}")
    return parsed


def _frames(value: str) -> int:
    return _bounded_int(value, 1, 600, "frames")


def _gs_iters(value: str) -> int:
    return _bounded_int(value, 500, 60_000, "gs-iters")


def _texture_size(value: str) -> int:
    return _bounded_int(value, 512, 4096, "texture-size")


def _target_triangles(value: str) -> int:
    return _bounded_int(value, 5_000, 2_000_000, "target-triangles")


def _voxel_divisor(value: str) -> int:
    return _bounded_int(value, 64, 2048, "voxel-divisor")


def _chunk_size(value: str) -> int:
    parsed = _positive_int(value)
    if not 64 * 1024 <= parsed <= MAX_CHUNK:
        raise argparse.ArgumentTypeError("la taille d'un chunk doit être comprise entre 65536 et 67108864 octets")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("un nombre est requis") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("la valeur doit être un nombre strictement positif")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minids",
        description="Reconstruction 3D d'un objet filmé au téléphone, via un pod RunPod.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", default=os.environ.get("MINIDS_URL", ""), help="https://<podid>-8000.proxy.runpod.net")
    parser.add_argument("--token", default=os.environ.get("MINIDS_TOKEN", ""), help="doit correspondre au pod")
    parser.add_argument("--chunk-size", type=_chunk_size, default=DEFAULT_CHUNK, help="taille des chunks (octets)")
    parser.add_argument("--quiet", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    health = sub.add_parser("health", help="état du pod (GPU, VRAM, version)")
    health.set_defaults(func=command_health)

    run = sub.add_parser("run", help="scan complet : extraction, upload, attente, récupération")
    run.add_argument("video", type=Path, help="vidéo, ou dossier d'images déjà extraites")
    _add_job_arguments(run)
    run.add_argument("--out", type=Path, default=Path("out"), help="dossier de sortie local")
    run.add_argument("--also-raw", action="store_true", help="récupère aussi vggt_raw.npz")
    run.add_argument("--all-artifacts", action="store_true", help="récupère tous les artefacts")
    run.add_argument("--poll", type=_positive_float, default=5.0, help="intervalle de polling (s)")
    run.set_defaults(func=command_run)

    submit = sub.add_parser("submit", help="prépare et lance un job, sans attendre")
    submit.add_argument("video", type=Path)
    _add_job_arguments(submit)
    submit.set_defaults(func=command_submit)

    status = sub.add_parser("status", help="état d'un job")
    status.add_argument("job_id")
    status.add_argument("--watch", action="store_true", help="suit jusqu'à la fin")
    status.add_argument("--poll", type=_positive_float, default=5.0)
    status.set_defaults(func=command_status)

    fetch = sub.add_parser("fetch", help="télécharge les artefacts d'un job terminé")
    fetch.add_argument("job_id")
    fetch.add_argument("--out", type=Path, default=Path("out"))
    fetch.add_argument("--names", nargs="+", default=None, help="par défaut : mesh.glb, report.json, preview.png")
    fetch.add_argument("--all-artifacts", action="store_true")
    fetch.add_argument("--also-raw", action="store_true")
    fetch.set_defaults(func=command_fetch)

    jobs = sub.add_parser("jobs", help="liste les jobs du pod")
    jobs.set_defaults(func=command_jobs)

    cancel = sub.add_parser("cancel", help="annule un job")
    cancel.add_argument("job_id")
    cancel.set_defaults(func=command_cancel)

    remesh = sub.add_parser("remesh", help="re-maille en local depuis vggt_raw.npz (Open3D CPU, sans pod)")
    remesh.add_argument("npz", type=Path)
    remesh.add_argument("--out", type=Path, default=Path("out/remesh.glb"))
    remesh.add_argument("--backend", choices=["tsdf", "poisson"], default="tsdf")
    remesh.add_argument("--voxel-divisor", type=_positive_int, default=512)
    remesh.add_argument("--target-triangles", type=_positive_int, default=200_000)
    remesh.add_argument(
        "--ref-size", type=_positive_float, default=None, help="plus grande dimension réelle, en mètres"
    )
    remesh.add_argument("--no-masks", action="store_true")
    remesh.add_argument("--no-watertight", action="store_true")
    remesh.set_defaults(func=command_remesh)

    extract = sub.add_parser("extract", help="extrait seulement les images (débogage ffmpeg)")
    extract.add_argument("video", type=Path)
    extract.add_argument("--out", type=Path, default=Path("out/frames"))
    extract.add_argument("--frames", type=_frames, default=120)
    extract.add_argument("--long-side", type=_even_positive_int, default=1024)
    extract.set_defaults(func=command_extract)

    return parser


def _add_job_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", default=None, help='prompt SAM 3, ex: "the sneaker"')
    parser.add_argument("--frames", type=_frames, default=120, help="images envoyées au modèle")
    parser.add_argument("--long-side", type=_even_positive_int, default=1024, help="résolution d'upload (côté long)")
    parser.add_argument("--no-refine", action="store_true", help="saute le 2DGS (rapide, moins précis)")
    parser.add_argument("--gs-iters", type=_gs_iters, default=12_000)
    parser.add_argument(
        "--backends",
        default="tsdf2dgs",
        help="backends séparés par des virgules : tsdf2dgs, tsdf, poisson",
    )
    parser.add_argument("--segmentation", choices=["auto", "sam3", "geometric", "none"], default="auto")
    parser.add_argument("--texture", choices=["bake", "vertex"], default="bake")
    parser.add_argument("--texture-size", type=_texture_size, default=2048)
    parser.add_argument("--target-triangles", type=_target_triangles, default=200_000)
    parser.add_argument("--voxel-divisor", type=_voxel_divisor, default=512)
    parser.add_argument(
        "--ref-size", type=_positive_float, default=None, help="plus grande dimension réelle, en mètres"
    )
    parser.add_argument("--no-watertight", action="store_true")
    parser.add_argument("--send-video", action="store_true", help="envoie la vidéo brute (extraction sur le pod)")


def job_params(args: argparse.Namespace) -> dict[str, Any]:
    allowed = {"tsdf2dgs", "tsdf", "poisson"}
    backends = list(dict.fromkeys(part.strip() for part in args.backends.split(",") if part.strip()))
    invalid = [backend for backend in backends if backend not in allowed]
    if invalid:
        raise MinidsError(f"backend inconnu : {', '.join(invalid)}")
    if not backends:
        raise MinidsError("au moins un backend de maillage est requis")
    if args.no_refine:
        backends = list(dict.fromkeys("tsdf" if backend == "tsdf2dgs" else backend for backend in backends))
    prompt = args.prompt.strip() if args.prompt else None
    if args.segmentation == "sam3" and not prompt:
        raise MinidsError("la segmentation sam3 exige un prompt texte non vide")
    return {
        "prompt": prompt,
        "frames": args.frames,
        "refine": not args.no_refine,
        "gs_iters": args.gs_iters,
        "mesh_backends": backends,
        "segmentation": args.segmentation,
        "texture": args.texture,
        "texture_size": args.texture_size,
        "target_triangles": args.target_triangles,
        "voxel_divisor": args.voxel_divisor,
        "ref_size": args.ref_size,
        "watertight": not args.no_watertight,
    }


def make_client(args: argparse.Namespace) -> MinidsClient:
    return MinidsClient(url=args.url, token=args.token)


# ---------------------------------------------------------------------------
# Préparation de l'entrée
# ---------------------------------------------------------------------------


def prepare_payload(args: argparse.Namespace, workdir: Path) -> Path:
    """Retourne le fichier à envoyer : archive d'images, ou vidéo brute."""
    try:
        return build_payload(
            source=Path(args.video),
            workdir=workdir,
            frames=args.frames,
            long_side=args.long_side,
            send_video=args.send_video,
            log=lambda message: print(f"  {message}"),
        )
    except (FFmpegError, OSError, ValueError) as exc:
        raise MinidsError(str(exc)) from exc


def submit_job(
    client: MinidsClient,
    args: argparse.Namespace,
    payload: Path,
    params: dict[str, Any] | None = None,
) -> str:
    size = payload.stat().st_size
    print("empreinte sha256…", end=" ", flush=True)
    digest = sha256_file(payload)
    print(digest[:16] + "…")

    job_id = client.create_job(
        payload.name, size, args.chunk_size, digest, params if params is not None else job_params(args)
    )
    print(f"job {job_id}")
    client.upload(job_id, payload, args.chunk_size, args.quiet)
    response = client.start(job_id)
    print(f"lancé (file d'attente : {response.get('queue_position', 0)})")
    return job_id


# ---------------------------------------------------------------------------
# Commandes
# ---------------------------------------------------------------------------


def command_health(args: argparse.Namespace) -> int:
    print(json.dumps(make_client(args).health(), indent=2, ensure_ascii=False))
    return 0


def command_submit(args: argparse.Namespace) -> int:
    params = job_params(args)  # validation avant extraction/empreinte coûteuse
    client = make_client(args)
    with tempfile.TemporaryDirectory(prefix="minids-") as tmp:
        job_id = submit_job(client, args, prepare_payload(args, Path(tmp)), params)
    print(f"\nsuivi : python client/minids.py status {job_id} --watch")
    return 0


def command_run(args: argparse.Namespace) -> int:
    params = job_params(args)  # validation avant extraction/empreinte coûteuse
    client = make_client(args)
    with tempfile.TemporaryDirectory(prefix="minids-") as tmp:
        job_id = submit_job(client, args, prepare_payload(args, Path(tmp)), params)

    state = watch(client, job_id, args.poll)
    if state["status"] != "done":
        print(f"\njob {state['status']} : {state.get('error') or 'annulé'}")
        return 1

    destination = job_destination(Path(args.out), job_id)
    names = _artifact_names(client, job_id, args.all_artifacts, args.also_raw, None)
    available_names = {str(item.get("name")) for item in names}
    missing_required = [name for name in ("mesh.glb", "report.json") if name not in available_names]
    if missing_required:
        raise MinidsError(f"job terminé sans artefact requis : {', '.join(missing_required)}")
    download_many(client, job_id, names, destination, args.quiet)

    glb = destination / "mesh.glb"
    print(f"\n✔ {glb if glb.exists() else destination}")
    _print_report(destination / "report.json")
    return 0


def command_status(args: argparse.Namespace) -> int:
    client = make_client(args)
    if args.watch:
        state = watch(client, args.job_id, args.poll)
    else:
        state = client.status(args.job_id)
        print(_format_state(state))
        for line in state.get("logs", [])[-15:]:
            print(f"  {line}")
    return 0 if state["status"] in {"done", "running", "queued", "starting", "created"} else 1


def command_fetch(args: argparse.Namespace) -> int:
    client = make_client(args)
    names = _artifact_names(client, args.job_id, args.all_artifacts, args.also_raw, args.names)
    if args.names:
        found = {str(item.get("name")) for item in names}
        missing = [name for name in args.names if name not in found]
        if missing:
            raise MinidsError(f"artefact demandé absent : {', '.join(missing)}")
    download_many(client, args.job_id, names, job_destination(Path(args.out), args.job_id), args.quiet)
    return 0


def command_jobs(args: argparse.Namespace) -> int:
    for job in make_client(args).jobs():
        job_id = str(job.get("job_id") or "?")
        status = str(job.get("status") or "?")
        stage = str(job.get("stage") or "")
        progress = _finite_float(job.get("progress"), 0.0)
        print(f"{job_id[:8]}  {status:<9} {stage:<9} {progress * 100:5.1f}%")
    return 0


def command_cancel(args: argparse.Namespace) -> int:
    print(json.dumps(make_client(args).cancel(args.job_id), indent=2))
    return 0


def command_extract(args: argparse.Namespace) -> int:
    try:
        frames = extract_frames(
            args.video, args.out, count=args.frames, long_side=args.long_side, log=lambda m: print(f"  {m}")
        )
    except (FFmpegError, OSError, ValueError) as exc:
        raise MinidsError(str(exc)) from exc
    print(f"{len(frames)} images → {args.out}")
    return 0


def command_remesh(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from server.pipeline.remesh import remesh

    try:
        metrics = remesh(
            npz_path=args.npz,
            output=args.out,
            backend=args.backend,
            voxel_divisor=args.voxel_divisor,
            target_triangles=args.target_triangles,
            use_masks=not args.no_masks,
            watertight=not args.no_watertight,
            ref_size=args.ref_size,
            log_fn=lambda message: print(f"  {message}"),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise MinidsError(str(exc)) from exc
    print(json.dumps(metrics, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Suivi et téléchargement
# ---------------------------------------------------------------------------


def watch(client: MinidsClient, job_id: str, poll: float) -> dict[str, Any]:
    started = time.time()

    def on_update(state: dict[str, Any]) -> None:
        logs = state.get("new_logs", [])
        if logs:
            print("\r" + " " * 78, end="\r")  # efface la ligne de progression avant les logs
            for line in logs:
                print(f"  {line}")
        print(f"\r{_format_state(state)}  [{time.time() - started:.0f}s]", end="", flush=True)

    state = client.wait(job_id, poll_seconds=poll, on_update=on_update)
    print()
    return state


def _format_state(state: dict[str, Any]) -> str:
    eta = _finite_float(state.get("eta_seconds"), 0.0)
    suffix = f", reste ~{eta / 60:.1f} min" if eta > 0 else ""
    status = str(state.get("status") or "?")
    stage = str(state.get("stage") or "")
    progress = max(0.0, min(1.0, _finite_float(state.get("progress"), 0.0)))
    return f"{status:<9} {stage:<9} {progress * 100:5.1f}%{suffix}"


def _artifact_names(
    client: MinidsClient, job_id: str, all_artifacts: bool, also_raw: bool, explicit: list[str] | None
) -> list[dict[str, Any]]:
    available: dict[str, dict[str, Any]] = {}
    for item in client.artifacts(job_id):
        name = item.get("name")
        if isinstance(name, str) and name:
            available[name] = item
    if explicit:
        wanted = explicit
    elif all_artifacts:
        wanted = list(available)
    else:
        wanted = [name for name in DEFAULT_ARTIFACTS if name in available]
        if also_raw and "vggt_raw.npz" in available:
            wanted.append("vggt_raw.npz")
    missing = [name for name in wanted if name not in available]
    if missing:
        print(f"absent du pod : {', '.join(missing)}")
    return [available[name] for name in wanted if name in available]


def download_many(
    client: MinidsClient, job_id: str, artifacts: list[dict[str, Any]], destination: Path, quiet: bool
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for artifact in artifacts:
        name = artifact.get("name")
        if not isinstance(name, str) or not name:
            raise MinidsError("liste des artefacts : nom invalide")
        client.download(
            job_id,
            name,
            artifact_destination(destination, name),
            quiet=quiet,
            expected_sha256=artifact.get("sha256"),
        )


def _print_report(path: Path) -> None:
    if not path.exists():
        return
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return
    if not isinstance(report, dict):
        return
    backend = report.get("primary_backend", "?")
    backends = report.get("backends") if isinstance(report.get("backends"), dict) else {}
    backend_report = backends.get(backend) if isinstance(backends.get(backend), dict) else {}
    metrics = backend_report.get("metrics") if isinstance(backend_report.get("metrics"), dict) else {}
    if metrics:
        print(
            f"  {backend} : {metrics.get('triangles', '?')} triangles, "
            f"watertight={metrics.get('watertight')}, {report.get('total_seconds', '?')}s"
        )


def _finite_float(value: object, fallback: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
        return parsed if math.isfinite(parsed) else fallback
    except (TypeError, ValueError):
        return fallback


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (MinidsError, FFmpegError, OSError) as exc:
        print(f"\nerreur : {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrompu", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
