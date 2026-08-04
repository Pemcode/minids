"""Fenêtre principale : connexion, paramétrage, suivi, résultats, historique."""

from __future__ import annotations

import html
import importlib.util
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from client.transport import MinidsError, validate_job_id

from . import theme
from .formatting import boolean, compact_number, duration, percent, rate, size
from .store import History, Settings, record_from_outcome
from .widgets import JobPickerDialog, LogView, MetricCard, PreviewPane, StageTimeline, StatusDot
from .workers import HealthWorker, JobListWorker, ScanRequest, ScanWorker

VIDEO_FILTER = "Vidéos et images (*.mp4 *.mov *.m4v *.mkv *.avi *.webm);;Tous les fichiers (*)"

# Réglages recommandés : valider la chaîne avant de payer un raffinement de 15 min.
PRESETS: dict[str, dict[str, Any]] = {
    "Validation rapide  (~2 min)": {
        "frames": 100,
        "refine": False,
        "backend": "tsdf",
        "compare": False,
        "texture": "vertex",
        "gs_iters": 12000,
        "texture_size": 2048,
        "target_triangles": 200000,
        "voxel_divisor": 512,
        "watertight": True,
    },
    "Qualité maximale  (~15 min)": {
        "frames": 120,
        "refine": True,
        "backend": "tsdf2dgs",
        "compare": False,
        "texture": "bake",
        "gs_iters": 12000,
        "texture_size": 2048,
        "target_triangles": 200000,
        "voxel_divisor": 512,
        "watertight": True,
    },
    "Comparatif de backends  (~18 min)": {
        "frames": 120,
        "refine": True,
        "backend": "tsdf2dgs",
        "compare": True,
        "texture": "bake",
        "gs_iters": 12000,
        "texture_size": 2048,
        "target_triangles": 200000,
        "voxel_divisor": 512,
        "watertight": True,
    },
    "Personnalisé": {},
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("miniDS — vidéo → objet 3D")
        self.resize(1280, 860)

        self.settings = Settings.load()
        self.history = History()
        self.scan_worker: ScanWorker | None = None
        self.health_worker: HealthWorker | None = None
        self.job_list_worker: JobListWorker | None = None
        self.last_outcome: Any = None
        self._scan_started_at = 0.0
        self._loading_preset = False
        self._source_label = ""
        self._model_access_warning = ""
        self._close_requested = False

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_connection_tab(), "Connexion")
        self.tabs.addTab(self._build_scan_tab(), "Scan")
        self.tabs.addTab(self._build_results_tab(), "Résultats")
        self.tabs.addTab(self._build_history_tab(), "Historique")
        self.setCentralWidget(self.tabs)

        self.status_dot = StatusDot()
        self.statusBar().addPermanentWidget(self.status_dot)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._apply_settings()
        self._refresh_history_view()

    # ------------------------------------------------------------------
    # Onglet Connexion
    # ------------------------------------------------------------------
    def _build_connection_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        box = QGroupBox("Pod RunPod")
        form = self._form(box)

        self.url_edit = QLineEdit()
        self.url_edit.setPlaceholderText("https://<POD_ID>-8000.proxy.runpod.net")
        self.token_edit = QLineEdit()
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.token_edit.setPlaceholderText("MINIDS_TOKEN du template")

        token_row = QWidget()
        token_layout = QHBoxLayout(token_row)
        token_layout.setContentsMargins(0, 0, 0, 0)
        self.show_token_button = QPushButton("Afficher")
        self.show_token_button.setCheckable(True)
        self.show_token_button.setMaximumWidth(90)
        self.show_token_button.setAccessibleName("Afficher ou masquer le jeton")
        self.show_token_button.toggled.connect(self._toggle_token_visibility)
        token_layout.addWidget(self.token_edit)
        token_layout.addWidget(self.show_token_button)

        self.remember_token = QCheckBox("Mémoriser le jeton")
        self.remember_token.setToolTip(
            "Le jeton donne accès à un GPU facturé. Décoché, il est redemandé à chaque lancement."
        )

        form.addRow("URL du pod", self.url_edit)
        form.addRow("Jeton", token_row)
        form.addRow("", self.remember_token)

        buttons = QHBoxLayout()
        self.test_button = QPushButton("Tester la connexion")
        self.test_button.setObjectName("primary")
        self.test_button.clicked.connect(self.check_health)
        env_button = QPushButton("Charger depuis l'environnement")
        env_button.setToolTip("Reprend MINIDS_URL et MINIDS_TOKEN des variables d'environnement")
        env_button.clicked.connect(self._load_from_env)
        buttons.addWidget(self.test_button)
        buttons.addWidget(env_button)
        buttons.addStretch(1)
        form.addRow("", self._wrap(buttons))

        layout.addWidget(box)

        health_box = QGroupBox("État du pod")
        health_layout = QVBoxLayout(health_box)
        cards = QHBoxLayout()
        self.card_gpu = MetricCard("GPU")
        self.card_vram = MetricCard("VRAM libre")
        self.card_cuda = MetricCard("CUDA")
        self.card_version = MetricCard("Version miniDS")
        for card in (self.card_gpu, self.card_vram, self.card_cuda, self.card_version):
            cards.addWidget(card)
        health_layout.addLayout(cards)

        self.health_detail = QLabel("Aucun test effectué.")
        self.health_detail.setWordWrap(True)
        self.health_detail.setTextFormat(Qt.TextFormat.RichText)
        self.health_detail.setStyleSheet(f"color:{theme.TEXT_DIM};")
        health_layout.addWidget(self.health_detail)
        layout.addWidget(health_box)

        hint = QLabel(
            "<b>502</b> : le serveur n'a pas fini de démarrer, ou le port 8000 n'est pas exposé dans le template.<br>"
            "<b>401</b> : le jeton ne correspond pas à celui du pod.<br>"
            "<b>503</b> : aucun <code>MINIDS_TOKEN</code> n'est configuré côté pod."
        )
        hint.setStyleSheet(
            f"color:{theme.TEXT_DIM}; background:{theme.SURFACE_HIGH};"
            f"border:1px solid {theme.BORDER}; border-radius:8px; padding:12px;"
        )
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------
    # Onglet Scan
    # ------------------------------------------------------------------
    def _build_scan_tab(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # -- panneau des paramètres --
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        form_host = QWidget()
        form_layout = QVBoxLayout(form_host)
        form_layout.setContentsMargins(16, 16, 16, 16)
        form_layout.setSpacing(12)

        source_box = QGroupBox("Source")
        source_form = self._form(source_box, stacked=True)
        self.source_edit = QLineEdit()
        self.source_edit.setPlaceholderText("vidéo, ou dossier d'images déjà extraites")
        source_form.addRow("Fichier", self.source_edit)

        # Boutons sur leur propre ligne : côte à côte avec le champ, ils
        # imposaient une largeur minimale qui faisait déborder le panneau.
        source_buttons = QHBoxLayout()
        source_buttons.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Vidéo…")
        browse.clicked.connect(self._pick_source)
        browse_dir = QPushButton("Dossier d'images…")
        browse_dir.clicked.connect(self._pick_source_dir)
        source_buttons.addStretch(1)
        source_buttons.addWidget(browse)
        source_buttons.addWidget(browse_dir)
        source_form.addRow(self._wrap(source_buttons))

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(PRESETS.keys())
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        source_form.addRow("Préréglage", self.preset_combo)

        self.frames_spin = QSpinBox()
        self.frames_spin.setRange(20, 600)
        self.frames_spin.setSingleStep(10)
        self.frames_spin.setToolTip("120 images ≈ 16 Go de VRAM. Au-delà de 200, il faut plus de 24 Go.")
        self.frames_spin.valueChanged.connect(self._on_manual_change)
        source_form.addRow("Images envoyées", self.frames_spin)

        self.long_side_spin = QSpinBox()
        self.long_side_spin.setRange(512, 4096)
        self.long_side_spin.setSingleStep(128)
        self.long_side_spin.setToolTip("Côté long des JPEG téléversés. Influe surtout sur le bake de texture.")
        source_form.addRow("Résolution d'envoi", self.long_side_spin)

        self.send_video_check = QCheckBox("Envoyer la vidéo brute")
        self.send_video_check.setToolTip("Multiplie le volume transféré par ~25. À réserver au débogage.")
        source_form.addRow(self.send_video_check)
        form_layout.addWidget(source_box)

        object_box = QGroupBox("Objet")
        object_form = self._form(object_box, stacked=True)
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("the sneaker")
        self.prompt_edit.setToolTip("Prompt texte SAM 3. Vide → segmentation géométrique.")
        object_form.addRow("Prompt SAM 3", self.prompt_edit)

        self.segmentation_combo = QComboBox()
        self.segmentation_combo.addItems(["auto", "sam3", "geometric", "none"])
        object_form.addRow("Segmentation", self.segmentation_combo)

        self.ref_size_spin = QDoubleSpinBox()
        self.ref_size_spin.setRange(0.0, 100.0)
        self.ref_size_spin.setDecimals(3)
        self.ref_size_spin.setSingleStep(0.01)
        self.ref_size_spin.setSuffix(" m")
        self.ref_size_spin.setSpecialValueText("échelle relative")
        self.ref_size_spin.setToolTip(
            "Plus grande dimension réelle de l'objet. VGGT-Ω prédit à un facteur près :\n"
            "sans cette valeur, le GLB n'est pas à l'échelle métrique."
        )
        object_form.addRow("Taille réelle", self.ref_size_spin)
        form_layout.addWidget(object_box)

        quality_box = QGroupBox("Qualité")
        quality_form = self._form(quality_box, stacked=True)
        self.refine_check = QCheckBox("Raffinement 2DGS")
        self.refine_check.setToolTip(
            "Optimise des surfels sur les vraies images, puis fusionne les profondeurs rendues.\n"
            "C'est ce qui distingue un maillage propre d'un maillage bruité — mais coûte ~12 min."
        )
        self.refine_check.toggled.connect(self._on_manual_change)
        self.refine_check.toggled.connect(self._sync_quality_controls)
        quality_form.addRow(self.refine_check)

        self.gs_iters_spin = QSpinBox()
        self.gs_iters_spin.setRange(500, 60000)
        self.gs_iters_spin.setSingleStep(1000)
        self.gs_iters_spin.valueChanged.connect(self._on_manual_change)
        quality_form.addRow("Itérations 2DGS", self.gs_iters_spin)

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["tsdf2dgs", "tsdf", "poisson"])
        self.backend_combo.setToolTip(
            "tsdf2dgs : TSDF sur profondeurs rendues par 2DGS (défaut)\n"
            "tsdf     : TSDF sur profondeur VGGT-Ω brute\n"
            "poisson  : Poisson screened"
        )
        self.backend_combo.currentTextChanged.connect(self._on_manual_change)
        quality_form.addRow("Backend principal", self.backend_combo)

        self.compare_check = QCheckBox("Comparer les autres backends")
        self.compare_check.toggled.connect(self._on_manual_change)
        quality_form.addRow(self.compare_check)
        form_layout.addWidget(quality_box)

        mesh_box = QGroupBox("Maillage et texture")
        mesh_form = self._form(mesh_box, stacked=True)
        self.target_tris_spin = QSpinBox()
        self.target_tris_spin.setRange(5000, 2000000)
        self.target_tris_spin.setSingleStep(50000)
        self.target_tris_spin.setGroupSeparatorShown(True)
        self.target_tris_spin.valueChanged.connect(self._on_manual_change)
        mesh_form.addRow("Triangles visés", self.target_tris_spin)

        self.voxel_spin = QSpinBox()
        self.voxel_spin.setRange(64, 2048)
        self.voxel_spin.setSingleStep(64)
        self.voxel_spin.setToolTip("Diagonale de l'objet ÷ ce nombre = taille de voxel TSDF.")
        self.voxel_spin.valueChanged.connect(self._on_manual_change)
        mesh_form.addRow("Diviseur de voxel", self.voxel_spin)

        self.texture_combo = QComboBox()
        self.texture_combo.addItems(["bake", "vertex"])
        self.texture_combo.currentTextChanged.connect(self._on_manual_change)
        mesh_form.addRow("Texture", self.texture_combo)

        self.texture_size_combo = QComboBox()
        self.texture_size_combo.addItems(["1024", "2048", "4096"])
        self.texture_size_combo.currentTextChanged.connect(self._on_manual_change)
        mesh_form.addRow("Taille de texture", self.texture_size_combo)

        self.watertight_check = QCheckBox("Boucher les trous")
        self.watertight_check.toggled.connect(self._on_manual_change)
        mesh_form.addRow(self.watertight_check)
        form_layout.addWidget(mesh_box)

        output_box = QGroupBox("Sortie")
        output_form = self._form(output_box, stacked=True)
        output_row = QHBoxLayout()
        self.output_edit = QLineEdit()
        pick_out = QPushButton("Parcourir…")
        pick_out.clicked.connect(self._pick_output)
        output_row.addWidget(self.output_edit)
        output_row.addWidget(pick_out)
        output_form.addRow("Dossier", self._wrap(output_row))
        self.fetch_raw_check = QCheckBox("Rapatrier vggt_raw.npz")
        output_form.addRow(self.fetch_raw_check)
        self.fetch_all_check = QCheckBox("Tous les artefacts")
        output_form.addRow(self.fetch_all_check)
        form_layout.addWidget(output_box)
        form_layout.addStretch(1)

        scroll.setWidget(form_host)

        # -- panneau de suivi --
        monitor = QWidget()
        monitor_layout = QVBoxLayout(monitor)
        monitor_layout.setContentsMargins(16, 16, 16, 16)
        monitor_layout.setSpacing(12)

        actions = QHBoxLayout()
        self.launch_button = QPushButton("Lancer le scan")
        self.launch_button.setObjectName("primary")
        self.launch_button.setMinimumHeight(38)
        self.launch_button.clicked.connect(self.start_scan)
        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setMinimumHeight(38)
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_scan)
        actions.addWidget(self.launch_button, 2)
        actions.addWidget(self.cancel_button, 1)
        monitor_layout.addLayout(actions)

        # Reprise : le pod garde l'état de ses jobs sur disque. Une fenêtre fermée
        # au milieu d'un scan de 15 min ne doit pas coûter le scan.
        resume = QHBoxLayout()
        self.job_id_edit = QLineEdit()
        self.job_id_edit.setPlaceholderText("identifiant d'un job à rejoindre")
        self.browse_jobs_button = QPushButton("Jobs du pod…")
        self.browse_jobs_button.clicked.connect(self.browse_jobs)
        self.attach_button = QPushButton("Rejoindre")
        self.attach_button.setToolTip(
            "Reprend le suivi d'un job déjà lancé sur le pod, puis récupère ses artefacts.\n"
            "Le scan continue côté pod même quand cette fenêtre est fermée."
        )
        self.attach_button.clicked.connect(self.attach_to_job)
        resume.addWidget(self.job_id_edit, 1)
        resume.addWidget(self.browse_jobs_button)
        resume.addWidget(self.attach_button)
        monitor_layout.addLayout(resume)

        self.phase_label = QLabel("prêt")
        phase_font = QFont()
        phase_font.setPointSize(14)
        phase_font.setWeight(QFont.Weight.DemiBold)
        self.phase_label.setFont(phase_font)
        monitor_layout.addWidget(self.phase_label)

        self.server_bar = QProgressBar()
        self.server_bar.setRange(0, 1000)
        self.server_bar.setFormat("pipeline  %p%")
        monitor_layout.addWidget(self.server_bar)

        self.transfer_bar = QProgressBar()
        self.transfer_bar.setRange(0, 1000)
        self.transfer_bar.setFormat("transfert  %p%")
        monitor_layout.addWidget(self.transfer_bar)

        stats = QHBoxLayout()
        self.card_elapsed = MetricCard("Écoulé")
        self.card_eta = MetricCard("Reste estimé")
        self.card_stage = MetricCard("Étape")
        self.card_rate = MetricCard("Débit")
        for card in (self.card_elapsed, self.card_eta, self.card_stage, self.card_rate):
            stats.addWidget(card)
        monitor_layout.addLayout(stats)

        timeline_box = QGroupBox("Répartition du temps par étape")
        timeline_layout = QVBoxLayout(timeline_box)
        self.live_timeline = StageTimeline()
        timeline_layout.addWidget(self.live_timeline)
        monitor_layout.addWidget(timeline_box)

        self.log_view = LogView()
        monitor_layout.addWidget(self.log_view, 1)

        scroll.setMinimumWidth(430)
        splitter.addWidget(scroll)
        splitter.addWidget(monitor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([470, 810])
        return splitter

    # ------------------------------------------------------------------
    # Onglet Résultats
    # ------------------------------------------------------------------
    def _build_results_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        cards = QGridLayout()
        self.result_cards = {
            "triangles": MetricCard("Triangles"),
            "watertight": MetricCard("Étanche"),
            "duration": MetricCard("Durée totale"),
            "glb": MetricCard("Taille GLB"),
            "texture": MetricCard("Couverture texture"),
            "gaussians": MetricCard("Gaussiennes"),
            "upload": MetricCard("Débit montant"),
            "download": MetricCard("Débit descendant"),
        }
        for index, card in enumerate(self.result_cards.values()):
            cards.addWidget(card, index // 4, index % 4)
        layout.addLayout(cards)

        middle = QHBoxLayout()
        preview_box = QGroupBox("Aperçu (rendu par lancer de rayons)")
        preview_layout = QVBoxLayout(preview_box)
        self.preview = PreviewPane()
        self.preview.clicked.connect(self._open_preview)
        preview_layout.addWidget(self.preview)
        middle.addWidget(preview_box, 3)

        artifacts_box = QGroupBox("Artefacts récupérés")
        artifacts_layout = QVBoxLayout(artifacts_box)
        self.artifacts_table = QTableWidget(0, 2)
        self.artifacts_table.setHorizontalHeaderLabels(["Fichier", "Taille"])
        self.artifacts_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.artifacts_table.verticalHeader().setVisible(False)
        self.artifacts_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.artifacts_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.artifacts_table.setAccessibleName("Artefacts récupérés")
        artifacts_layout.addWidget(self.artifacts_table)
        middle.addWidget(artifacts_box, 2)
        layout.addLayout(middle, 1)

        timeline_box = QGroupBox("Temps par étape sur ce scan")
        timeline_layout = QVBoxLayout(timeline_box)
        self.result_timeline = StageTimeline()
        timeline_layout.addWidget(self.result_timeline)
        layout.addWidget(timeline_box)

        buttons = QHBoxLayout()
        self.open_folder_button = QPushButton("Ouvrir le dossier")
        self.open_folder_button.clicked.connect(self._open_folder)
        self.open_glb_button = QPushButton("Ouvrir le GLB (application système)")
        self.open_glb_button.clicked.connect(self._open_glb)
        self.view_3d_button = QPushButton("Visionneuse 3D interactive")
        self.view_3d_button.setObjectName("primary")
        self.view_3d_button.setToolTip("Ouvre le maillage dans le visualiseur Open3D, dans un processus séparé")
        self.view_3d_button.clicked.connect(self._open_3d_viewer)
        for button in (self.open_folder_button, self.open_glb_button, self.view_3d_button):
            button.setEnabled(False)
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        return page

    # ------------------------------------------------------------------
    # Onglet Historique
    # ------------------------------------------------------------------
    def _build_history_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        cards = QHBoxLayout()
        self.history_cards = {
            "count": MetricCard("Scans"),
            "done": MetricCard("Réussis"),
            "duration": MetricCard("Durée médiane"),
            "triangles": MetricCard("Triangles médians"),
            "upload": MetricCard("Débit montant médian"),
        }
        for card in self.history_cards.values():
            cards.addWidget(card)
        layout.addLayout(cards)

        self.history_table = QTableWidget(0, 8)
        self.history_table.setHorizontalHeaderLabels(
            ["Date", "Source", "Backend", "État", "Durée", "Triangles", "Étanche", "Images"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.setAccessibleName("Historique des scans")
        self.history_table.itemSelectionChanged.connect(self._on_history_selection)
        layout.addWidget(self.history_table, 1)

        timeline_box = QGroupBox("Temps moyen par étape (scans réussis)")
        timeline_layout = QVBoxLayout(timeline_box)
        self.history_timeline = StageTimeline()
        timeline_layout.addWidget(self.history_timeline)
        layout.addWidget(timeline_box)

        buttons = QHBoxLayout()
        open_selected = QPushButton("Ouvrir le dossier du scan sélectionné")
        open_selected.clicked.connect(self._open_history_folder)
        clear_button = QPushButton("Vider l'historique")
        clear_button.setObjectName("danger")
        clear_button.clicked.connect(self._clear_history)
        buttons.addWidget(open_selected)
        buttons.addStretch(1)
        buttons.addWidget(clear_button)
        layout.addLayout(buttons)
        return page

    # ------------------------------------------------------------------
    # Réglages
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap(layout) -> QWidget:
        """Enveloppe un layout dans un widget, pour l'insérer dans un formulaire.

        Le fond doit être transparent : un `QWidget` nu peint la couleur de la
        fenêtre, ce qui dessine un rectangle sombre au milieu du groupe — et se
        lit comme un champ de saisie vide.

        Le sélecteur est nommé, et non un simple `background: transparent` : une
        feuille de style posée sur un widget s'applique aussi à sa descendance,
        ce qui effacerait le remplissage des boutons `#primary` qu'il contient.
        """
        holder = QWidget()
        holder.setObjectName("formRow")
        holder.setLayout(layout)
        holder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        holder.setStyleSheet("QWidget#formRow { background: transparent; }")
        return holder

    @staticmethod
    def _form(parent: QWidget, stacked: bool = False) -> QFormLayout:
        """Formulaire qui tient dans un panneau étroit.

        `stacked` place l'intitulé **au-dessus** du champ. C'est ce qui rend le
        panneau de paramètres utilisable : avec deux colonnes, la largeur
        minimale de la colonne d'étiquettes suffit à faire déborder le panneau,
        et le contenu se retrouve rogné.
        """
        form = QFormLayout(parent)
        form.setSpacing(9)
        form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows if stacked else QFormLayout.RowWrapPolicy.WrapLongRows
        )
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return form

    def _apply_settings(self) -> None:
        settings = self.settings
        self.url_edit.setText(settings.url or os.environ.get("MINIDS_URL", ""))
        self.token_edit.setText(settings.token or os.environ.get("MINIDS_TOKEN", ""))
        self.remember_token.setChecked(settings.remember_token)
        self.output_edit.setText(settings.output_dir or str(Path.cwd() / "out"))
        self.long_side_spin.setValue(settings.long_side if settings.long_side % 2 == 0 else 1024)
        self.send_video_check.setChecked(settings.send_video)
        self.fetch_raw_check.setChecked(settings.fetch_raw)
        # Un job resté en cours à la dernière fermeture : proposé d'emblée.
        self.job_id_edit.setText(settings.last_job_id)

        params = settings.params or {}
        self._loading_preset = True
        self.frames_spin.setValue(_bounded_int(params.get("frames"), 120, 20, 600))
        prompt = params.get("prompt")
        self.prompt_edit.setText(prompt if isinstance(prompt, str) else "")
        self.segmentation_combo.setCurrentText(
            _choice(params.get("segmentation"), {"auto", "sam3", "geometric", "none"}, "auto")
        )
        self.refine_check.setChecked(bool(params.get("refine", True)))
        self.gs_iters_spin.setValue(_bounded_int(params.get("gs_iters"), 12000, 500, 60000))
        raw_backends = params.get("mesh_backends")
        backends = raw_backends if isinstance(raw_backends, list) and raw_backends else ["tsdf2dgs"]
        self.backend_combo.setCurrentText(_choice(backends[0], {"tsdf2dgs", "tsdf", "poisson"}, "tsdf2dgs"))
        self.compare_check.setChecked(len(backends) > 1)
        self.texture_combo.setCurrentText(_choice(params.get("texture"), {"bake", "vertex"}, "bake"))
        self.texture_size_combo.setCurrentText(
            str(_choice(str(params.get("texture_size", 2048)), {"1024", "2048", "4096"}, "2048"))
        )
        self.target_tris_spin.setValue(_bounded_int(params.get("target_triangles"), 200000, 5000, 2000000))
        self.voxel_spin.setValue(_bounded_int(params.get("voxel_divisor"), 512, 64, 2048))
        self.watertight_check.setChecked(bool(params.get("watertight", True)))
        self.ref_size_spin.setValue(_bounded_float(params.get("ref_size"), 0.0, 0.0, 100.0))
        self.preset_combo.setCurrentText("Personnalisé" if params else "Qualité maximale  (~15 min)")
        self._loading_preset = False
        self._sync_quality_controls(self.refine_check.isChecked())

    def _collect_params(self) -> dict[str, Any]:
        primary = self.backend_combo.currentText()
        if not self.refine_check.isChecked() and primary == "tsdf2dgs":
            primary = "tsdf"
        backends = [primary]
        if self.compare_check.isChecked():
            backends += [b for b in ("tsdf2dgs", "tsdf", "poisson") if b not in backends]
            if not self.refine_check.isChecked():
                backends = [b for b in backends if b != "tsdf2dgs"] or ["tsdf"]
        return {
            "prompt": self.prompt_edit.text().strip() or None,
            "frames": self.frames_spin.value(),
            "refine": self.refine_check.isChecked(),
            "gs_iters": self.gs_iters_spin.value(),
            "mesh_backends": backends,
            "segmentation": self.segmentation_combo.currentText(),
            "texture": self.texture_combo.currentText(),
            "texture_size": int(self.texture_size_combo.currentText()),
            "target_triangles": self.target_tris_spin.value(),
            "voxel_divisor": self.voxel_spin.value(),
            "ref_size": self.ref_size_spin.value() or None,
            "watertight": self.watertight_check.isChecked(),
        }

    def _save_settings(self) -> None:
        self.settings.url = self.url_edit.text().strip()
        self.settings.token = self.token_edit.text().strip()
        self.settings.remember_token = self.remember_token.isChecked()
        self.settings.output_dir = self.output_edit.text().strip()
        self.settings.long_side = self.long_side_spin.value()
        self.settings.send_video = self.send_video_check.isChecked()
        self.settings.fetch_raw = self.fetch_raw_check.isChecked()
        self.settings.params = self._collect_params()
        try:
            self.settings.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Réglages non enregistrés : {exc}", 10000)

    def _load_from_env(self) -> None:
        self.url_edit.setText(os.environ.get("MINIDS_URL", ""))
        self.token_edit.setText(os.environ.get("MINIDS_TOKEN", ""))
        self.statusBar().showMessage("Valeurs reprises de l'environnement", 4000)

    def _toggle_token_visibility(self, visible: bool) -> None:
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password)
        self.show_token_button.setText("Masquer" if visible else "Afficher")

    def _sync_quality_controls(self, refine: bool) -> None:
        self.gs_iters_spin.setEnabled(refine)
        if not refine and self.backend_combo.currentText() == "tsdf2dgs":
            self.backend_combo.setCurrentText("tsdf")

    def _apply_preset(self, name: str) -> None:
        preset = PRESETS.get(name) or {}
        if not preset:
            return
        self._loading_preset = True
        self.frames_spin.setValue(preset["frames"])
        self.refine_check.setChecked(preset["refine"])
        self.backend_combo.setCurrentText(preset["backend"])
        self.compare_check.setChecked(preset["compare"])
        self.texture_combo.setCurrentText(preset["texture"])
        self.gs_iters_spin.setValue(preset["gs_iters"])
        self.texture_size_combo.setCurrentText(str(preset["texture_size"]))
        self.target_tris_spin.setValue(preset["target_triangles"])
        self.voxel_spin.setValue(preset["voxel_divisor"])
        self.watertight_check.setChecked(preset["watertight"])
        self._loading_preset = False

    def _on_manual_change(self, *_args) -> None:
        if not self._loading_preset and self.preset_combo.currentText() != "Personnalisé":
            self.preset_combo.setCurrentText("Personnalisé")

    # ------------------------------------------------------------------
    # Sélecteurs de fichiers
    # ------------------------------------------------------------------
    def _pick_source(self) -> None:
        start = self.settings.last_source_dir or str(Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Choisir une vidéo", start, VIDEO_FILTER)
        if path:
            self.source_edit.setText(path)
            self.settings.last_source_dir = str(Path(path).parent)

    def _pick_source_dir(self) -> None:
        start = self.settings.last_source_dir or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choisir un dossier d'images", start)
        if path:
            self.source_edit.setText(path)
            self.settings.last_source_dir = path

    def _pick_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Dossier de sortie", self.output_edit.text() or str(Path.cwd()))
        if path:
            self.output_edit.setText(path)

    # ------------------------------------------------------------------
    # Connexion
    # ------------------------------------------------------------------
    def check_health(self) -> None:
        url = self.url_edit.text().strip()
        token = self.token_edit.text().strip()
        if not url or not token:
            QMessageBox.warning(self, "miniDS", "URL et jeton sont requis.")
            return
        self.test_button.setEnabled(False)
        self.status_dot.set_state("test en cours…", theme.WARNING)
        self.health_worker = HealthWorker(url, token)
        self.health_worker.succeeded.connect(self._on_health_ok)
        self.health_worker.failed.connect(self._on_health_failed)
        self.health_worker.finished.connect(lambda: self.test_button.setEnabled(True))
        self.health_worker.finished.connect(self._maybe_finish_close)
        self.health_worker.start()

    @pyqtSlot(dict)
    def _on_health_ok(self, info: dict) -> None:
        cuda = info.get("cuda_available")
        self.card_gpu.set_value(str(info.get("gpu", "—")))
        free = info.get("vram_free_gb")
        total = info.get("vram_total_gb")
        self.card_vram.set_value(f"{free} Go" if free else "—", f"sur {total} Go" if total else "")
        self.card_cuda.set_value(boolean(cuda), str(info.get("torch", "")), theme.SUCCESS if cuda else theme.WARNING)
        self.card_version.set_value(str(info.get("version", "—")))

        # Un scan atteint l'étape `vggt` après cinq étapes déjà facturées. Si les
        # identifiants du modèle manquent, il vaut mieux le savoir maintenant.
        self._model_access_warning = "" if info.get("fake_gpu") else self._missing_model_access(info)

        if info.get("fake_gpu"):
            self.status_dot.set_state("connecté — mode factice", theme.WARNING)
            self.health_detail.setText(
                "Le pod tourne avec MINIDS_FAKE_GPU=1 : les artefacts sont synthétiques. "
                "Idéal pour valider le transport, inutile pour une vraie reconstruction."
            )
        elif self._model_access_warning:
            self.status_dot.set_state("connecté — modèle inaccessible", theme.WARNING)
            self.health_detail.setText(self._model_access_warning)
        elif cuda:
            self.status_dot.set_state("connecté", theme.SUCCESS)
            self.health_detail.setText(f"{info.get('jobs', 0)} job(s) connus du pod.")
        else:
            self.status_dot.set_state("connecté — sans CUDA", theme.WARNING)
            self.health_detail.setText("Le pod répond mais ne voit aucun GPU : l'inférence serait inutilisable.")
        self._save_settings()

    @staticmethod
    def _missing_model_access(info: dict) -> str:
        """Ce qui manque au pod pour que l'étape `vggt` puisse aboutir.

        Les anciens pods ne publient pas ces champs : dans le doute, on ne dit
        rien plutôt que d'inventer une alerte.
        """
        missing = []
        if "hf_token_configured" in info and not info["hf_token_configured"]:
            missing.append(
                "<b>HF_TOKEN</b> n'est pas défini sur le pod — les poids VGGT-Ω sont sous accès "
                "restreint, et le job échouera à l'étape <code>vggt</code> avec « Please log in »."
            )
        checkpoint_missing = ("checkpoint_configured" in info and not bool(info["checkpoint_configured"])) or (
            "checkpoint_configured" not in info and "checkpoint" in info and not info["checkpoint"]
        )
        if checkpoint_missing:
            missing.append(
                "<b>MINIDS_CKPT</b> est vide — attendu : <code>facebook/VGGT-Omega:vggt_omega_1b_512.pt</code>."
            )
        if not missing:
            return ""
        return (
            "<br>".join(missing) + "<br><br>Corrige les variables du template RunPod, puis <b>recrée le pod</b> : "
            "modifier un template ne change pas l'environnement d'un pod déjà lancé."
        )

    @pyqtSlot(str)
    def _on_health_failed(self, message: str) -> None:
        self.status_dot.set_state("échec", theme.DANGER)
        self.health_detail.setText(html.escape(message))
        for card in (self.card_gpu, self.card_vram, self.card_cuda, self.card_version):
            card.set_value("—")

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------
    def _connection(self) -> tuple[str, str] | None:
        """URL et jeton, ou `None` après avoir renvoyé l'utilisateur vers Connexion."""
        url = self.url_edit.text().strip()
        token = self.token_edit.text().strip()
        if not url or not token:
            QMessageBox.warning(self, "miniDS", "Renseigne l'URL et le jeton dans l'onglet Connexion.")
            self.tabs.setCurrentIndex(0)
            return None
        return url, token

    def _busy(self) -> bool:
        return self.scan_worker is not None and self.scan_worker.isRunning()

    def start_scan(self) -> None:
        if self._busy():
            return
        connection = self._connection()
        if connection is None:
            return
        url, token = connection
        source = Path(self.source_edit.text().strip()).expanduser()
        if not source.is_file() and not source.is_dir():
            QMessageBox.warning(self, "miniDS", "Choisis une vidéo ou un dossier d'images existant.")
            return
        if self.segmentation_combo.currentText() == "sam3" and not self.prompt_edit.text().strip():
            QMessageBox.warning(self, "miniDS", "La segmentation SAM 3 exige un prompt texte non vide.")
            return
        output_dir = Path(self.output_edit.text().strip() or "out").expanduser()
        if output_dir.exists() and not output_dir.is_dir():
            QMessageBox.warning(self, "miniDS", "Le chemin de sortie existe mais n'est pas un dossier.")
            return
        if not self._confirm_despite_missing_model_access():
            return

        self.settings.last_source_dir = str(source if source.is_dir() else source.parent)
        self._save_settings()
        request = ScanRequest(
            url=url,
            token=token,
            source=source,
            output_dir=output_dir,
            params=self._collect_params(),
            long_side=self.long_side_spin.value(),
            send_video=self.send_video_check.isChecked(),
            fetch_raw=self.fetch_raw_check.isChecked(),
            fetch_all=self.fetch_all_check.isChecked(),
        )
        self._launch(request, source.name)

    def _confirm_despite_missing_model_access(self) -> bool:
        """Prévient avant de payer un scan condamné à échouer sur `vggt`.

        Le dernier test de connexion a vu qu'il manquait un identifiant au pod.
        On n'interdit pas — le pod a pu être corrigé entre-temps — mais le défaut
        est de ne pas lancer.
        """
        if not self._model_access_warning:
            return True
        answer = QMessageBox.question(
            self,
            "miniDS",
            "Au dernier test de connexion, le pod ne pouvait pas accéder aux poids du modèle :\n\n"
            "le scan ira jusqu'à l'étape « vggt » avant d'échouer, GPU facturé.\n\n"
            "Lancer quand même ?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def browse_jobs(self) -> None:
        """Demande la liste des jobs au pod, hors du fil graphique."""
        connection = self._connection()
        if connection is None:
            return
        self.browse_jobs_button.setEnabled(False)
        self.job_list_worker = JobListWorker(*connection)
        self.job_list_worker.succeeded.connect(self._on_jobs_listed)
        self.job_list_worker.failed.connect(lambda message: QMessageBox.warning(self, "miniDS", message))
        self.job_list_worker.finished.connect(lambda: self.browse_jobs_button.setEnabled(True))
        self.job_list_worker.finished.connect(self._maybe_finish_close)
        self.job_list_worker.start()

    @pyqtSlot(list)
    def _on_jobs_listed(self, jobs: list) -> None:
        if not jobs:
            QMessageBox.information(self, "miniDS", "Le pod ne connaît aucun job.")
            return
        dialog = JobPickerDialog(jobs, self)
        if dialog.exec() == JobPickerDialog.DialogCode.Accepted and dialog.selected_job_id():
            self.job_id_edit.setText(dialog.selected_job_id())

    def attach_to_job(self) -> None:
        """Rejoint un job déjà lancé : suivi puis récupération, sans rien renvoyer."""
        if self._busy():
            return
        connection = self._connection()
        if connection is None:
            return
        job_id = self.job_id_edit.text().strip()
        if not job_id:
            QMessageBox.warning(self, "miniDS", "Indique l'identifiant du job à rejoindre.")
            return
        try:
            validate_job_id(job_id)
        except MinidsError as exc:
            QMessageBox.warning(self, "miniDS", str(exc))
            return

        self._save_settings()
        request = ScanRequest(
            url=connection[0],
            token=connection[1],
            source=Path(),
            output_dir=Path(self.output_edit.text().strip() or "out"),
            params={},
            job_id=job_id,
            fetch_raw=self.fetch_raw_check.isChecked(),
            fetch_all=self.fetch_all_check.isChecked(),
        )
        self._launch(request, f"(repris) {job_id}")

    def _launch(self, request: ScanRequest, source_label: str) -> None:
        """Remise à zéro de l'affichage et démarrage du worker.

        Partagé par le lancement et la reprise : un seul câblage de signaux, donc
        un job rejoint alimente exactement les mêmes barres, métriques et
        historique qu'un scan lancé ici.
        """
        self._source_label = source_label

        self.log_view.clear()
        self.live_timeline.clear()
        self.server_bar.setValue(0)
        self.transfer_bar.setValue(0)
        self.card_eta.set_value("—")
        self.card_rate.set_value("—")
        self._scan_started_at = time.time()
        self._elapsed_timer.start()

        self.scan_worker = ScanWorker(request)
        self.scan_worker.log.connect(self.log_view.append_line)
        self.scan_worker.phase_changed.connect(self._on_phase)
        self.scan_worker.transfer_progress.connect(self._on_transfer)
        self.scan_worker.server_state.connect(self._on_server_state)
        self.scan_worker.job_created.connect(self._on_job_created)
        self.scan_worker.finished_ok.connect(self._on_scan_done)
        self.scan_worker.failed.connect(self._on_scan_failed)
        self.scan_worker.finished.connect(self._on_worker_finished)
        self.scan_worker.start()

        self.launch_button.setEnabled(False)
        self.attach_button.setEnabled(False)
        self.cancel_button.setEnabled(True)

    @pyqtSlot(str)
    def _on_job_created(self, job_id: str) -> None:
        # Mémorisé immédiatement : c'est ce qui permet de rejoindre le job après
        # une fermeture brutale, sans avoir eu à noter l'identifiant.
        self.job_id_edit.setText(job_id)
        self.settings.last_job_id = job_id
        try:
            self.settings.save()
        except OSError as exc:
            self.statusBar().showMessage(f"Identifiant du job non enregistré : {exc}", 10000)
        self.statusBar().showMessage(f"job {job_id}", 8000)

    def cancel_scan(self) -> None:
        if self.scan_worker is not None:
            self.scan_worker.cancel()
            self.cancel_button.setEnabled(False)

    @pyqtSlot(str)
    def _on_phase(self, phase: str) -> None:
        self.phase_label.setText(phase)
        if phase in {"extraction", "attente"}:
            self.transfer_bar.setValue(0)

    @pyqtSlot(str, object, object, float)
    def _on_transfer(self, label: str, done: int, total: int, speed: float) -> None:
        done = max(0, _bounded_int(done, 0, 0, 2**63 - 1))
        total = max(0, _bounded_int(total, 0, 0, 2**63 - 1))
        fraction = max(0.0, min(1.0, done / total if total else 0.0))
        self.transfer_bar.setValue(int(fraction * 1000))
        self.transfer_bar.setFormat(f"{label}  {size(done)} / {size(total)}  (%p%)")
        self.card_rate.set_value(rate(speed))

    @pyqtSlot(dict)
    def _on_server_state(self, state: dict) -> None:
        progress = _bounded_float(state.get("progress"), 0.0, 0.0, 1.0)
        self.server_bar.setValue(round(progress * 1000))
        stage = state.get("stage") or ""
        self.card_stage.set_value(stage or "—", state.get("status", ""))
        self.card_eta.set_value(duration(_optional_float(state.get("eta_seconds"))))
        timings = state.get("stage_timings") if isinstance(state.get("stage_timings"), dict) else {}
        self.live_timeline.set_timings(timings, stage)

    def _tick_elapsed(self) -> None:
        self.card_elapsed.set_value(duration(time.time() - self._scan_started_at))

    @pyqtSlot(object)
    def _on_scan_done(self, outcome) -> None:
        self.phase_label.setText("terminé")
        self.server_bar.setValue(1000)
        self.last_outcome = outcome
        self._record_history(outcome)
        self._refresh_history_view()
        self._show_results(outcome)
        self.tabs.setCurrentIndex(2)
        self._forget_pending_job()
        self.statusBar().showMessage(f"Scan terminé en {duration(outcome.total_seconds)}", 10000)

    @pyqtSlot(object)
    def _on_scan_failed(self, outcome) -> None:
        if outcome.status == "detached":
            self.phase_label.setText("détaché")
            self.log_view.append_line("suivi local fermé — le job distant n'a pas été annulé")
            self.statusBar().showMessage("Job non annulé ; son identifiant reste disponible pour le rejoindre", 10000)
            return
        if outcome.status == "download_interrupted":
            self.phase_label.setText("téléchargement interrompu")
            self.log_view.append_line(outcome.error)
            self.statusBar().showMessage("Rejoins le job terminé pour reprendre les téléchargements", 10000)
            return
        cancelled = outcome.status == "cancelled"
        stopped_before_start = cancelled and outcome.error == "arrêté avant démarrage"
        self.phase_label.setText(
            "arrêté avant démarrage"
            if stopped_before_start
            else ("annulé" if cancelled else f"échec — {outcome.status}")
        )
        if outcome.job_id:
            self._record_history(outcome)
            self._refresh_history_view()

        if cancelled:
            # Une annulation est une décision de l'utilisateur, pas un incident :
            # une boîte d'erreur lui ferait relire ce qu'il vient de demander.
            message = "scan arrêté avant démarrage" if stopped_before_start else "scan annulé"
            self.log_view.append_line(message)
            self._forget_pending_job()
            self.statusBar().showMessage(message.capitalize(), 8000)
            return

        if outcome.status in {"failed", "cancelled"}:
            self._forget_pending_job()

        self.log_view.append_line(f"ERREUR {outcome.error}")
        QMessageBox.critical(self, "miniDS", outcome.error or "Le scan a échoué.")

    def _forget_pending_job(self) -> None:
        """Le job n'est plus en cours : plus rien à reproposer au prochain démarrage."""
        self.settings.last_job_id = ""
        try:
            self.settings.save()
        except OSError as exc:
            self.statusBar().showMessage(f"État du job non enregistré : {exc}", 10000)

    def _record_history(self, outcome: Any) -> None:
        try:
            self.history.add(record_from_outcome(outcome, self._source_label))
        except (OSError, TypeError, ValueError) as exc:
            self.statusBar().showMessage(f"Historique non enregistré : {exc}", 10000)

    def _on_worker_finished(self) -> None:
        self._elapsed_timer.stop()
        self.launch_button.setEnabled(True)
        self.attach_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self._maybe_finish_close()

    def _workers_running(self) -> bool:
        return any(
            worker is not None and worker.isRunning()
            for worker in (self.scan_worker, self.health_worker, self.job_list_worker)
        )

    def _maybe_finish_close(self) -> None:
        if self._close_requested and not self._workers_running():
            QTimer.singleShot(0, self.close)

    # ------------------------------------------------------------------
    # Résultats
    # ------------------------------------------------------------------
    def _show_results(self, outcome) -> None:
        report = outcome.report if isinstance(outcome.report, dict) else {}
        backend = report.get("primary_backend") if isinstance(report.get("primary_backend"), str) else ""
        backends = report.get("backends") if isinstance(report.get("backends"), dict) else {}
        backend_report = backends.get(backend) if isinstance(backends.get(backend), dict) else {}
        metrics = backend_report.get("metrics") if isinstance(backend_report.get("metrics"), dict) else {}
        texture = report.get("texture") if isinstance(report.get("texture"), dict) else {}
        refine = report.get("refine") if isinstance(report.get("refine"), dict) else {}
        timings = report.get("timings") if isinstance(report.get("timings"), dict) else {}

        watertight = metrics.get("watertight")
        self.result_cards["triangles"].set_value(compact_number(metrics.get("triangles")), backend)
        self.result_cards["watertight"].set_value(
            boolean(watertight), color=theme.SUCCESS if watertight else theme.WARNING
        )
        self.result_cards["duration"].set_value(
            duration(outcome.total_seconds), f"dont {duration(outcome.server_seconds)} sur le pod"
        )

        glb = outcome.directory / "mesh.glb"
        self.result_cards["glb"].set_value(size(glb.stat().st_size) if glb.is_file() else "—")
        self.result_cards["texture"].set_value(percent(texture.get("coverage")))
        self.result_cards["gaussians"].set_value(compact_number(refine.get("gaussians")))
        self.result_cards["upload"].set_value(rate(outcome.upload_rate), size(outcome.payload_bytes))
        self.result_cards["download"].set_value(rate(outcome.download_rate), size(outcome.download_bytes))

        self.result_timeline.set_timings(timings)

        preview = outcome.directory / "preview.png"
        if preview.is_file():
            self.preview.load(preview)
        else:
            self.preview.clear_preview()

        self.artifacts_table.setRowCount(0)
        artifacts = outcome.artifacts if isinstance(outcome.artifacts, list) else []
        for entry in artifacts:
            if not isinstance(entry, dict):
                continue
            row = self.artifacts_table.rowCount()
            self.artifacts_table.insertRow(row)
            self.artifacts_table.setItem(row, 0, QTableWidgetItem(str(entry.get("name") or "")))
            self.artifacts_table.setItem(row, 1, QTableWidgetItem(size(entry.get("size"))))

        self.open_folder_button.setEnabled(outcome.directory.is_dir())
        self.open_glb_button.setEnabled(glb.is_file())
        self.view_3d_button.setEnabled(glb.is_file() and importlib.util.find_spec("open3d") is not None)

    def _current_directory(self) -> Path | None:
        if self.last_outcome is not None:
            return Path(self.last_outcome.directory)
        return None

    def _open_folder(self) -> None:
        directory = self._current_directory()
        if directory and directory.is_dir():
            try:
                os.startfile(str(directory))  # noqa: S606 - ouverture de dossier voulue
            except OSError as exc:
                QMessageBox.warning(self, "miniDS", f"Ouverture impossible : {exc}")

    def _open_glb(self) -> None:
        directory = self._current_directory()
        if directory is None:
            return
        glb = directory / "mesh.glb"
        if glb.is_file():
            try:
                os.startfile(str(glb))  # noqa: S606
            except OSError as exc:
                QMessageBox.warning(self, "miniDS", f"Ouverture impossible : {exc}")
        else:
            QMessageBox.information(self, "miniDS", "mesh.glb absent de ce scan.")

    def _open_preview(self) -> None:
        directory = self._current_directory()
        preview = directory / "preview.png" if directory is not None else None
        if preview is not None and preview.is_file():
            try:
                os.startfile(str(preview))  # noqa: S606
            except OSError as exc:
                QMessageBox.warning(self, "miniDS", f"Ouverture impossible : {exc}")

    def _open_3d_viewer(self) -> None:
        """Ouvre Open3D dans un processus séparé.

        Deux boucles d'événements graphiques ne peuvent pas cohabiter dans le
        même processus : lancer le visualiseur Open3D ici figerait Qt.
        """
        directory = self._current_directory()
        if directory is None:
            return
        glb = directory / "mesh.glb"
        if not glb.is_file():
            QMessageBox.information(self, "miniDS", "mesh.glb absent de ce scan.")
            return
        if importlib.util.find_spec("open3d") is None:
            QMessageBox.warning(self, "miniDS", "Visionneuse indisponible : Open3D n'est pas installé.")
            return
        script = (
            "import sys, open3d as o3d;"
            "m = o3d.io.read_triangle_mesh(sys.argv[1], enable_post_processing=True);"
            "m.compute_vertex_normals();"
            "o3d.visualization.draw_geometries([m], window_name='miniDS', width=1100, height=800)"
        )
        try:
            subprocess.Popen(  # noqa: S603 - interpréteur courant, script constant, chemin en argv
                [sys.executable, "-c", script, str(glb)]
            )
        except OSError as exc:
            QMessageBox.warning(self, "miniDS", f"Visionneuse indisponible : {exc}")

    # ------------------------------------------------------------------
    # Historique
    # ------------------------------------------------------------------
    def _refresh_history_view(self) -> None:
        summary = self.history.summary()
        self.history_cards["count"].set_value(str(summary["count"]))
        self.history_cards["done"].set_value(
            str(summary["done"]), f"{summary['failed']} en échec" if summary["failed"] else ""
        )
        self.history_cards["duration"].set_value(duration(summary["median_duration"]))
        self.history_cards["triangles"].set_value(compact_number(summary["median_triangles"]))
        self.history_cards["upload"].set_value(rate(summary["median_upload_rate"]))

        self.history_table.setRowCount(0)
        for record in self.history.records:
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            values = [
                record.when,
                record.source,
                record.metrics.get("primary_backend") or record.backend,
                record.status,
                duration(record.total_seconds),
                compact_number(record.triangles),
                boolean(record.metrics.get("watertight")),
                str(record.metrics.get("frames") or "—"),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 3 and record.status != "done":
                    item.setToolTip(record.error or record.status)
                self.history_table.setItem(row, column, item)

        self.history_timeline.set_timings(self.history.average_timings())

    def _on_history_selection(self) -> None:
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            return
        record = self.history.records[rows[0].row()]
        self.statusBar().showMessage(f"{record.job_id} — {record.directory}", 8000)

    def _open_history_folder(self) -> None:
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "miniDS", "Sélectionne une ligne.")
            return
        directory = Path(self.history.records[rows[0].row()].directory)
        if directory.is_dir():
            try:
                os.startfile(str(directory))  # noqa: S606
            except OSError as exc:
                QMessageBox.warning(self, "miniDS", f"Ouverture impossible : {exc}")
        else:
            QMessageBox.information(self, "miniDS", "Ce dossier n'existe plus.")

    def _clear_history(self) -> None:
        confirm = QMessageBox.question(
            self, "miniDS", "Vider l'historique ? Les fichiers de scan sur disque ne sont pas touchés."
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.history.clear()
            self._refresh_history_view()

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 - API Qt
        if self.scan_worker is not None and self.scan_worker.isRunning():
            # Fermer signifie se détacher, jamais annuler un GPU distant. Le
            # bouton « Annuler » reste l'action explicite dédiée à cela.
            self.scan_worker.detach()
        if self._workers_running():
            # Détruire un QThread actif peut faire terminer brutalement Python.
            # On laisse les appels courts rendre la main et on referme via leur
            # signal ``finished`` ; la fenêtre est gelée entre-temps.
            self._close_requested = True
            self.setEnabled(False)
            self.statusBar().showMessage("Détachement du suivi puis fermeture… le job continue sur le pod")
            event.ignore()
            return
        self._save_settings()
        super().closeEvent(event)


def _bounded_int(value: object, default: int, minimum: int, maximum: int) -> int:
    try:
        if isinstance(value, bool):
            return default
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: object, default: float, minimum: float, maximum: float) -> float:
    numeric = _optional_float(value)
    if numeric is None:
        return default
    return max(minimum, min(maximum, numeric))


def _optional_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _choice(value: object, allowed: set[str], default: str) -> str:
    return value if isinstance(value, str) and value in allowed else default
