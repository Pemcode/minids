"""Widgets personnalisés.

Le seul non trivial est `StageTimeline` : une barre empilée qui montre où part
le temps dans le pipeline. C'est la visualisation qui compte pour ce projet —
sur un scan de 15 minutes, savoir que 11 sont passées dans le raffinement 2DGS
dit immédiatement quel paramètre ajuster.
"""

from __future__ import annotations

import math
import time

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QFont, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPlainTextEdit,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import theme
from .formatting import duration, percent

STAGE_ORDER = list(theme.STAGE_COLORS)


class MetricCard(QFrame):
    """Grand nombre + intitulé, pour les chiffres qu'on regarde en premier."""

    def __init__(self, title: str, value: str = "—", hint: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background:{theme.SURFACE_HIGH}; border:1px solid {theme.BORDER}; border-radius:8px; }}"
        )
        self.setMinimumWidth(120)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        self._title = QLabel(title.upper())
        self._title.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:10px; letter-spacing:1px; border:none;")
        self._value = QLabel(value)
        value_font = QFont()
        value_font.setPointSize(17)
        value_font.setWeight(QFont.Weight.DemiBold)
        self._value.setFont(value_font)
        self._value.setStyleSheet("border:none;")
        self._hint = QLabel(hint)
        self._hint.setStyleSheet(f"color:{theme.TEXT_DIM}; font-size:11px; border:none;")

        layout.addWidget(self._title)
        layout.addWidget(self._value)
        layout.addWidget(self._hint)
        self.setAccessibleName(title)
        self.setAccessibleDescription(f"{title} : {value}")

    def set_value(self, value: str, hint: str = "", color: str | None = None) -> None:
        self._value.setText(value)
        self._value.setStyleSheet(f"border:none; color:{color or theme.TEXT};")
        self._hint.setText(hint)
        self.setAccessibleDescription(f"{self._title.text()} : {value}{f', {hint}' if hint else ''}")


class StageTimeline(QWidget):
    """Barre empilée des durées par étape, avec légende et survol."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._timings: dict[str, float] = {}
        self._current: str = ""
        self.setMinimumHeight(74)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self.setAccessibleName("Répartition du temps par étape")

    def set_timings(self, timings: dict[str, float], current: str = "") -> None:
        self._timings = {}
        for key, value in (timings or {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if isinstance(key, str) and math.isfinite(numeric) and numeric >= 0:
                self._timings[key] = numeric
        self._current = current
        self._refresh_tooltip()
        self.update()

    def clear(self) -> None:
        self.set_timings({}, "")

    def _refresh_tooltip(self) -> None:
        total = sum(self._timings.values())
        if total <= 0:
            self.setToolTip("Aucune étape terminée")
            return
        lines = [
            f"{name} : {duration(value)}  ({value / total * 100:.0f} %)"
            for name, value in sorted(self._timings.items(), key=lambda kv: -kv[1])
            if value > 0
        ]
        self.setToolTip("\n".join(lines))
        self.setAccessibleDescription("; ".join(lines))

    def _ordered_stages(self) -> list[str]:
        return STAGE_ORDER + [name for name in self._timings if name not in STAGE_ORDER]

    def paintEvent(self, event) -> None:  # noqa: N802 - API Qt
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bar_height = 26
        width = max(1, self.width())
        total = sum(self._timings.values())

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(theme.SURFACE_HIGH))
        painter.drawRoundedRect(QRectF(0, 0, width, bar_height), 6, 6)

        if total > 0:
            clip = QPainterPath()
            clip.addRoundedRect(QRectF(0, 0, width, bar_height), 6, 6)
            painter.save()
            painter.setClipPath(clip)
            offset = 0.0
            for name in self._ordered_stages():
                value = self._timings.get(name, 0.0)
                if value <= 0:
                    continue
                segment = value / total * width
                color = QColor(theme.STAGE_COLORS.get(name, theme.ACCENT))
                if name == self._current:
                    color = color.lighter(130)
                painter.setBrush(color)
                painter.drawRect(QRectF(offset, 0, max(1.0, segment), bar_height))
                offset += segment
            painter.restore()
            # Réapplique les coins arrondis sur les extrémités.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QColor(theme.BORDER))
            painter.drawRoundedRect(QRectF(0.5, 0.5, width - 1, bar_height - 1), 6, 6)
        else:
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(
                QRectF(0, 0, width, bar_height), Qt.AlignmentFlag.AlignCenter, "en attente du premier scan"
            )

        # Légende : uniquement les étapes réellement chronométrées.
        painter.setFont(QFont("Segoe UI", 8))
        x, y = 0.0, bar_height + 10.0
        for name in self._ordered_stages():
            value = self._timings.get(name, 0.0)
            if value <= 0:
                continue
            label = f"{name} {duration(value)}"
            text_width = painter.fontMetrics().horizontalAdvance(label) + 20
            if x + text_width > width:
                x, y = 0.0, y + 16.0
                if y > self.height() - 8:
                    break
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(theme.STAGE_COLORS.get(name, theme.ACCENT)))
            painter.drawEllipse(QRectF(x, y + 3, 7, 7))
            painter.setPen(QColor(theme.TEXT_DIM))
            painter.drawText(QRectF(x + 11, y - 2, text_width, 16), Qt.AlignmentFlag.AlignVCenter, label)
            x += text_width
        painter.end()


class LogView(QPlainTextEdit):
    """Journal en lecture seule, avec défilement automatique respectueux."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("log")
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setAccessibleName("Journal du scan")

    def append_line(self, text: str) -> None:
        # Ne force le défilement que si l'utilisateur était déjà en bas : sinon
        # il devient impossible de relire un message pendant un scan actif.
        scrollbar = self.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4
        self.appendPlainText(text.rstrip())
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())


class PreviewPane(QLabel):
    """Affiche `preview.png` en conservant le rapport d'aspect."""

    clicked = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(180)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setAccessibleName("Aperçu du résultat")
        self.setToolTip("Cliquer ou appuyer sur Entrée pour ouvrir l'aperçu")
        self.setStyleSheet(
            f"background:{theme.SURFACE_HIGH}; border:1px solid {theme.BORDER};"
            f"border-radius:8px; color:{theme.TEXT_DIM};"
        )
        self.setText("aucun aperçu")

    def load(self, path) -> bool:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._pixmap = None
            self.setText("aperçu illisible")
            return False
        self._pixmap = pixmap
        self.setText("")
        self.setAccessibleDescription(f"Aperçu chargé depuis {path}")
        self._rescale()
        return True

    def clear_preview(self) -> None:
        self._pixmap = None
        self.setPixmap(QPixmap())
        self.setText("aucun aperçu")
        self.setAccessibleDescription("Aucun aperçu disponible")

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 - API Qt
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - API Qt
        if self._pixmap is not None and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - API Qt
        if self._pixmap is not None and event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class JobPickerDialog(QDialog):
    """Liste les jobs du pod pour en rejoindre un.

    Le pod persiste l'état de ses jobs sur disque : on peut donc retrouver un
    scan lancé depuis une autre session — ou depuis la CLI — sans en avoir noté
    l'identifiant.
    """

    def __init__(self, jobs: list[dict], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Jobs du pod")
        self.resize(720, 420)
        self._jobs = jobs

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.table = QTableWidget(len(jobs), 5)
        self.table.setHorizontalHeaderLabels(["Job", "État", "Étape", "Avancement", "Créé"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAccessibleName("Jobs disponibles sur le pod")
        self.table.doubleClicked.connect(self.accept)

        for row, job in enumerate(jobs):
            created = job.get("created_at")
            values = [
                job.get("job_id", ""),
                job.get("status", ""),
                job.get("stage") or "—",
                percent(job.get("progress")),
                _format_timestamp(created),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        if jobs:
            self.table.selectRow(0)
        layout.addWidget(self.table, 1)

        hint = QLabel(
            "Un job « starting » ou « running » se rejoint en direct ; un job « done » se contente de "
            "retélécharger ses artefacts. Un job « created » a un upload incomplet et ne peut pas être repris ici."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color:{theme.TEXT_DIM};")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        # Qt libelle ses boutons standards en anglais tant qu'aucune traduction
        # n'est chargée : on les nomme donc explicitement.
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Rejoindre")
        buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(bool(jobs))
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Annuler")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_job_id(self) -> str:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return ""
        return str(self._jobs[rows[0].row()].get("job_id", ""))


class StatusDot(QWidget):
    """Pastille d'état + libellé, pour la barre de connexion."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = theme.TEXT_DIM
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)
        self._dot = QLabel("●")
        self._dot.setStyleSheet(f"color:{self._color}; font-size:14px;")
        self._label = QLabel("non connecté")
        self._label.setStyleSheet(f"color:{theme.TEXT_DIM};")
        layout.addWidget(self._dot)
        layout.addWidget(self._label)
        layout.addStretch(1)
        self.setAccessibleName("État de la connexion")
        self.setAccessibleDescription("non connecté")

    def set_state(self, text: str, color: str) -> None:
        self._color = color
        self._dot.setStyleSheet(f"color:{color}; font-size:14px;")
        self._label.setText(text)
        self._label.setStyleSheet(f"color:{theme.TEXT};")
        self.setAccessibleDescription(text)


def _format_timestamp(value: object) -> str:
    try:
        timestamp = float(value)  # type: ignore[arg-type]
        if not math.isfinite(timestamp):
            return "—"
        return time.strftime("%d/%m %H:%M", time.localtime(timestamp))
    except (TypeError, ValueError, OSError, OverflowError):
        return "—"
