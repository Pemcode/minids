"""Palette et feuille de style de l'interface.

Un thème sombre volontairement sobre : l'écran affiche des rendus 3D et des
aperçus photo, et un fond clair fausse la perception des couleurs de texture.
"""

from __future__ import annotations

BACKGROUND = "#14161a"
SURFACE = "#1c1f26"
SURFACE_HIGH = "#242832"
BORDER = "#2f3542"
TEXT = "#e6e9ef"
TEXT_DIM = "#98a1b3"
ACCENT = "#4c8dff"
ACCENT_DIM = "#2f5fb0"
SUCCESS = "#3ecf8e"
WARNING = "#f2b545"
DANGER = "#ff6b6b"

# Couleurs des étapes du pipeline, dans leur ordre d'affichage côté client.
# Progression du bleu (acquisition) au vert (production du livrable).
STAGE_COLORS = {
    "ingest": "#5b7cfa",
    "frames": "#4c8dff",
    "vggt": "#3aa0ff",
    "segment": "#00b4d8",
    "colmap": "#0bc5b8",
    "refine": "#f2b545",
    "mesh": "#ff8f5c",
    "cleanup": "#e0699a",
    "texture": "#a97bd6",
    "export": "#3ecf8e",
}

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{ background-color: {BACKGROUND}; }}

/* Les intitulés doivent laisser voir le fond du groupe qui les porte : sinon
   chaque étiquette de formulaire traîne son propre rectangle sombre. */
QLabel {{ background: transparent; }}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background-color: {SURFACE};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 9px 20px;
    margin-right: 2px;
    border: 1px solid transparent;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    color: {TEXT};
    background: {SURFACE};
    border-color: {BORDER};
    border-bottom-color: {SURFACE};
}}
QTabBar::tab:hover:!selected {{ color: {TEXT}; }}

QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {TEXT_DIM};
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background-color: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border-color: {ACCENT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {TEXT_DIM};
    background-color: {SURFACE};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT_DIM};
    outline: none;
}}

QPushButton {{
    background-color: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 7px 16px;
    color: {TEXT};
}}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:focus {{ border: 2px solid {ACCENT}; padding: 6px 15px; }}
QPushButton:pressed {{ background-color: {BORDER}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; background: {SURFACE}; }}
QPushButton#primary {{
    background-color: {ACCENT};
    border-color: {ACCENT};
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#primary:hover {{ background-color: #5c9aff; }}
QPushButton#primary:disabled {{ background-color: {ACCENT_DIM}; color: #c9d6ef; }}
QPushButton#danger {{ border-color: {DANGER}; color: {DANGER}; }}
QPushButton#danger:hover {{ background-color: {DANGER}; color: #ffffff; }}

QProgressBar {{
    background-color: {SURFACE_HIGH};
    border: 1px solid {BORDER};
    border-radius: 6px;
    height: 18px;
    text-align: center;
    color: {TEXT};
}}
QProgressBar::chunk {{ background-color: {ACCENT}; border-radius: 5px; }}

QCheckBox {{ background: transparent; spacing: 7px; }}
QCheckBox:focus {{ color: #ffffff; }}
/* L'indicateur natif de Fusion ne survit pas à une feuille de style posée sur
   QCheckBox : Qt cesse alors d'en peindre la boîte, et une case cochée ne se
   distingue plus d'une case vide sur ce fond sombre (corriger la palette n'y
   change rien — vérifié). On le redessine donc explicitement. L'état ne tient
   pas qu'à la teinte : boîte vide sombre contre boîte pleine claire restent
   distinctes en niveaux de gris. */
QCheckBox::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE_HIGH};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
/* L'état se place sur le sous-contrôle : `QCheckBox:focus::indicator` dessine
   un cadre permanent autour de la case entière, dans tous les états. */
QCheckBox::indicator:focus {{ border: 2px solid #ffffff; }}

QPlainTextEdit#log {{
    font-family: "Cascadia Mono", "Consolas", monospace;
    font-size: 12px;
    background-color: #0f1115;
    border: 1px solid {BORDER};
}}

QTableWidget {{
    background-color: {SURFACE};
    alternate-background-color: {SURFACE_HIGH};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT_DIM};
}}
QHeaderView::section {{
    background-color: {SURFACE_HIGH};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 7px;
    font-weight: 600;
}}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #3d465a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 5px; min-width: 30px; }}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 3px; }}

QStatusBar {{ background: {SURFACE}; border-top: 1px solid {BORDER}; color: {TEXT_DIM}; }}
QToolTip {{
    background-color: {SURFACE_HIGH};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px;
}}
QScrollArea {{ border: none; background: transparent; }}
"""
