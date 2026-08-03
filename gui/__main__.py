"""Point d'entrée : `python -m gui`."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # exécution directe du fichier
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import QLibraryInfo, QLocale, Qt, QTranslator  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui import theme  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402


def install_french(app: QApplication) -> QTranslator | None:
    """Traduit les libellés fournis par Qt (Oui/Non/Annuler, sélecteurs de fichiers).

    Sans cela, une interface entièrement française se retrouve avec des boutons
    « Yes » et « Cancel ». Le traducteur doit rester référencé : Qt ne le garde
    pas en vie à notre place.
    """
    translator = QTranslator()
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    if translator.load(QLocale(QLocale.Language.French), "qtbase", "_", path):
        app.installTranslator(translator)
        return translator
    return None


def main() -> int:
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("miniDS")
    app.setStyle("Fusion")  # base neutre : la feuille de style s'applique de façon homogène
    app.setStyleSheet(theme.STYLESHEET)
    translator = install_french(app)  # noqa: F841 - doit survivre à la durée de l'application

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
