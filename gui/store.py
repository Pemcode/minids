"""Persistance locale : réglages de connexion et historique des scans.

Rangé dans `~/.minids/` et non dans le dépôt : ces fichiers sont propres à la
machine, et l'un d'eux peut contenir un jeton.

L'historique n'est pas un simple journal — c'est la base des métriques
comparatives. Sans lui, impossible de dire si un changement de paramètre a
réellement amélioré quoi que ce soit.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

STORE_DIR = Path.home() / ".minids"
SETTINGS_PATH = STORE_DIR / "gui-settings.json"
HISTORY_PATH = STORE_DIR / "gui-history.json"
MAX_RECORDS = 200


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return fallback


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    with contextlib.suppress(OSError):
        tmp.chmod(0o600)
    tmp.replace(path)


@dataclass
class Settings:
    url: str = ""
    token: str = ""
    remember_token: bool = False
    output_dir: str = ""
    last_source_dir: str = ""
    # Job encore en cours à la dernière fermeture. C'est lui qu'on repropose de
    # rejoindre : un scan de 15 min sur GPU facturé ne doit pas être perdu parce
    # que la fenêtre s'est fermée.
    last_job_id: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    long_side: int = 1024
    send_video: bool = False
    fetch_raw: bool = False

    @classmethod
    def load(cls) -> Settings:
        data = _read_json(SETTINGS_PATH, {})
        if not isinstance(data, dict):
            return cls()
        settings = cls()
        for key in ("url", "token", "output_dir", "last_source_dir", "last_job_id"):
            value = data.get(key)
            if isinstance(value, str):
                setattr(settings, key, value)
        for key in ("remember_token", "send_video", "fetch_raw"):
            value = data.get(key)
            if isinstance(value, bool):
                setattr(settings, key, value)
        if isinstance(data.get("params"), dict):
            settings.params = data["params"]
        long_side = data.get("long_side")
        if isinstance(long_side, int) and not isinstance(long_side, bool) and long_side > 0:
            settings.long_side = long_side
        if not settings.remember_token:
            settings.token = ""
        return settings

    def save(self) -> None:
        payload = asdict(self)
        if not self.remember_token:
            # Le jeton donne accès à un GPU facturé : on ne l'écrit que sur demande.
            payload["token"] = ""
        _write_json(SETTINGS_PATH, payload)


@dataclass
class ScanRecord:
    """Une ligne d'historique : ce qu'on a demandé, et ce qu'on a obtenu."""

    job_id: str
    timestamp: float
    source: str
    status: str
    directory: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    client: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def when(self) -> str:
        timestamp = _number(self.timestamp)
        if timestamp is None:
            return "—"
        try:
            return time.strftime("%d/%m %H:%M", time.localtime(timestamp))
        except (OSError, OverflowError, ValueError):
            return "—"

    @property
    def total_seconds(self) -> float:
        return _number(self.client.get("total_seconds")) or 0.0

    @property
    def triangles(self) -> int | None:
        value = _number(self.metrics.get("triangles"))
        return int(value) if value is not None and value >= 0 else None

    @property
    def backend(self) -> str:
        backends = self.params.get("mesh_backends")
        return str(backends[0]) if isinstance(backends, list) and backends else "?"


def record_from_outcome(outcome: Any, source: str) -> ScanRecord:
    """Construit une entrée d'historique depuis un `ScanOutcome`."""
    report = outcome.report if isinstance(outcome.report, dict) else {}
    backend = report.get("primary_backend", "")
    backends = report.get("backends") if isinstance(report.get("backends"), dict) else {}
    backend_report = backends.get(backend) if isinstance(backends.get(backend), dict) else {}
    metrics = backend_report.get("metrics") if isinstance(backend_report.get("metrics"), dict) else {}
    texture = report.get("texture") if isinstance(report.get("texture"), dict) else {}
    refine = report.get("refine") if isinstance(report.get("refine"), dict) else {}
    frames = report.get("frames") if isinstance(report.get("frames"), dict) else {}

    return ScanRecord(
        job_id=outcome.job_id,
        timestamp=time.time(),
        source=source,
        status=outcome.status,
        directory=str(outcome.directory),
        params=report.get("params") if isinstance(report.get("params"), dict) else {},
        timings=report.get("timings") if isinstance(report.get("timings"), dict) else {},
        metrics={
            **metrics,
            "primary_backend": backend,
            "texture_coverage": texture.get("coverage"),
            "gaussians": refine.get("gaussians"),
            "frames": frames.get("count"),
            "server_total": report.get("total_seconds"),
        },
        client={
            "total_seconds": outcome.total_seconds,
            "extract_seconds": outcome.extract_seconds,
            "upload_seconds": outcome.upload_seconds,
            "server_seconds": outcome.server_seconds,
            "download_seconds": outcome.download_seconds,
            "payload_bytes": outcome.payload_bytes,
            "download_bytes": outcome.download_bytes,
            "upload_rate": outcome.upload_rate,
            "download_rate": outcome.download_rate,
        },
        error=outcome.error,
    )


class History:
    def __init__(self) -> None:
        self.records: list[ScanRecord] = []
        self.load()

    def load(self) -> None:
        raw = _read_json(HISTORY_PATH, [])
        self.records = []
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                record = ScanRecord(**item)
            except TypeError:
                continue  # entrée d'une version antérieure : ignorée sans bruit
            if not all(isinstance(value, str) for value in (record.job_id, record.source, record.status)):
                continue
            if not all(
                isinstance(value, dict) for value in (record.params, record.timings, record.metrics, record.client)
            ):
                continue
            if _number(record.timestamp) is None:
                continue
            self.records.append(record)

    def save(self) -> None:
        _write_json(HISTORY_PATH, [asdict(record) for record in self.records[:MAX_RECORDS]])

    def add(self, record: ScanRecord) -> None:
        self.records.insert(0, record)
        del self.records[MAX_RECORDS:]
        self.save()

    def clear(self) -> None:
        self.records = []
        self.save()

    # -- agrégats -----------------------------------------------------------
    def successful(self) -> list[ScanRecord]:
        return [record for record in self.records if record.status == "done"]

    def average_timings(self) -> dict[str, float]:
        """Durée moyenne par étape sur les scans réussis."""
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for record in self.successful():
            for stage, value in record.timings.items():
                numeric = _number(value)
                if not isinstance(stage, str) or numeric is None or numeric < 0:
                    continue
                totals[stage] = totals.get(stage, 0.0) + numeric
                counts[stage] = counts.get(stage, 0) + 1
        return {stage: totals[stage] / counts[stage] for stage in totals if counts[stage]}

    def summary(self) -> dict[str, Any]:
        done = self.successful()
        durations = [record.total_seconds for record in done if record.total_seconds > 0]
        triangles = [record.triangles for record in done if record.triangles]
        rates = [_number(record.client.get("upload_rate")) for record in done]
        rates = [value for value in rates if value is not None and value > 0]
        return {
            "count": len(self.records),
            "done": len(done),
            "failed": sum(1 for record in self.records if record.status not in {"done", "cancelled"}),
            "median_duration": _median(durations),
            "median_triangles": _median([float(value) for value in triangles]),
            "median_upload_rate": _median(rates),
        }


def _median(values: list[float]) -> float | None:
    cleaned = [number for value in values if (number := _number(value)) is not None]
    if not cleaned:
        return None
    ordered = sorted(cleaned)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
