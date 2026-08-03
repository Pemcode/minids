"""Formatage des durées, tailles et débits.

S'appuie sur les fonctions statiques de tqdm plutôt que de réécrire un formateur
maison : on obtient exactement les mêmes chaînes que la barre de progression du
terminal, donc des mesures comparables entre la CLI et l'interface.
"""

from __future__ import annotations

from tqdm.std import tqdm


def duration(seconds: float | None) -> str:
    """`92.4` → `01:32`. Renvoie `—` si la durée est inconnue."""
    if seconds is None or seconds < 0:
        return "—"
    return tqdm.format_interval(int(round(seconds)))


def size(num_bytes: float | None) -> str:
    """`8388608` → `8.39M`."""
    if num_bytes is None:
        return "—"
    return f"{tqdm.format_sizeof(float(num_bytes), 'o')}"


def rate(bytes_per_second: float | None) -> str:
    if not bytes_per_second or bytes_per_second <= 0:
        return "—"
    return f"{tqdm.format_sizeof(bytes_per_second, 'o')}/s"


def percent(fraction: float | None) -> str:
    if fraction is None:
        return "—"
    return f"{fraction * 100:.1f} %"


def boolean(value: object) -> str:
    if value is None:
        return "—"
    return "oui" if value else "non"


def compact_number(value: float | int | None) -> str:
    """`203451` → `203 k`, pour les compteurs de triangles et de gaussiennes."""
    if value is None:
        return "—"
    value = float(value)
    if abs(value) >= 1e6:
        return f"{value / 1e6:.2f} M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f} k"
    return f"{value:.0f}"
