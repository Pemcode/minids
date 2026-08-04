"""Formatage des durées, tailles et débits.

S'appuie sur les fonctions statiques de tqdm plutôt que de réécrire un formateur
maison : on obtient exactement les mêmes chaînes que la barre de progression du
terminal, donc des mesures comparables entre la CLI et l'interface.
"""

from __future__ import annotations

import math

from tqdm.std import tqdm


def duration(seconds: float | None) -> str:
    """`92.4` → `01:32`. Renvoie `—` si la durée est inconnue."""
    value = _number(seconds)
    if value is None or value < 0:
        return "—"
    return tqdm.format_interval(round(value))


def size(num_bytes: float | None) -> str:
    """`8388608` → `8.39M`."""
    value = _number(num_bytes)
    if value is None or value < 0:
        return "—"
    return f"{tqdm.format_sizeof(value, 'o')}"


def rate(bytes_per_second: float | None) -> str:
    value = _number(bytes_per_second)
    if value is None or value <= 0:
        return "—"
    return f"{tqdm.format_sizeof(value, 'o')}/s"


def percent(fraction: float | None) -> str:
    value = _number(fraction)
    if value is None:
        return "—"
    return f"{value * 100:.1f} %"


def boolean(value: object) -> str:
    if value is None:
        return "—"
    return "oui" if value else "non"


def compact_number(value: float | int | None) -> str:
    """`203451` → `203 k`, pour les compteurs de triangles et de gaussiennes."""
    numeric = _number(value)
    if numeric is None:
        return "—"
    if abs(numeric) >= 1e6:
        return f"{numeric / 1e6:.2f} M"
    if abs(numeric) >= 1e3:
        return f"{numeric / 1e3:.0f} k"
    return f"{numeric:.0f}"


def _number(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None
