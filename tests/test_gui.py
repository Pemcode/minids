"""Interface graphique, pilotée sans écran.

Qt sait tourner en `offscreen` : on peut donc construire la vraie fenêtre,
remplir les champs, lancer un scan contre le serveur factice et vérifier que les
résultats remontent — c'est-à-dire tester la chaîne signaux/threads, qui est
l'endroit où une interface Qt casse réellement.

Les chemins de persistance sont redirigés vers un dossier temporaire : un test
ne doit jamais écraser les réglages ni l'historique de l'utilisateur.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6", reason="PyQt6 requis")
uvicorn = pytest.importorskip("uvicorn")

from PyQt6.QtGui import QPixmap  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from gui import store  # noqa: E402
from gui.formatting import boolean, compact_number, duration, percent, rate, size  # noqa: E402
from gui.store import History, ScanRecord, Settings  # noqa: E402
from gui.widgets import MetricCard, StageTimeline  # noqa: E402
from tests.conftest import CHUNK_SIZE, TEST_TOKEN, _noise_png  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Redirige réglages et historique — jamais dans le vrai ~/.minids."""
    monkeypatch.setattr(store, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(store, "HISTORY_PATH", tmp_path / "history.json")
    return tmp_path


@pytest.fixture
def live_server(app):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 20
    while not server.started and time.time() < deadline:
        time.sleep(0.05)
    if not server.started:
        raise RuntimeError("uvicorn n'a pas démarré")

    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=15)


@pytest.fixture
def frames_dir(tmp_path):
    """Dossier d'images : accepté directement par `build_payload`, sans ffmpeg."""
    directory = tmp_path / "frames_src"
    directory.mkdir()
    for index in range(4):
        (directory / f"frame_{index:05d}.png").write_bytes(_noise_png(seed=index, size=64))
    return directory


def pump(application, predicate, timeout: float = 120.0) -> bool:
    """Fait tourner la boucle d'événements jusqu'à ce que `predicate` soit vraie."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        application.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------

def test_formatting_handles_missing_values():
    assert duration(None) == "—"
    assert size(None) == "—"
    assert rate(0) == "—"
    assert percent(None) == "—"
    assert boolean(None) == "—"
    assert compact_number(None) == "—"


def test_formatting_values():
    assert duration(92) == "01:32"
    assert boolean(True) == "oui" and boolean(False) == "non"
    assert percent(0.874) == "87.4 %"
    assert compact_number(203451) == "203 k"
    assert compact_number(2_500_000) == "2.50 M"
    assert "o" in size(8 * 1024 * 1024)


# ---------------------------------------------------------------------------
# Persistance
# ---------------------------------------------------------------------------

def test_settings_never_persist_token_unless_asked(isolated_store):
    settings = Settings(url="http://pod", token="secret", remember_token=False)
    settings.save()

    reloaded = Settings.load()
    assert reloaded.url == "http://pod"
    assert reloaded.token == ""


def test_settings_persist_token_when_asked(isolated_store):
    Settings(url="http://pod", token="secret", remember_token=True).save()
    assert Settings.load().token == "secret"


def test_history_summary_and_averages(isolated_store):
    history = History()
    history.add(ScanRecord(
        job_id="a", timestamp=time.time(), source="v1.mp4", status="done",
        timings={"vggt": 10.0, "refine": 90.0}, metrics={"triangles": 100_000},
        client={"total_seconds": 120.0, "upload_rate": 1_000_000.0},
    ))
    history.add(ScanRecord(
        job_id="b", timestamp=time.time(), source="v2.mp4", status="done",
        timings={"vggt": 20.0, "refine": 110.0}, metrics={"triangles": 200_000},
        client={"total_seconds": 180.0, "upload_rate": 3_000_000.0},
    ))
    history.add(ScanRecord(
        job_id="c", timestamp=time.time(), source="v3.mp4", status="failed", error="boum",
    ))

    summary = history.summary()
    assert summary == {
        "count": 3, "done": 2, "failed": 1,
        "median_duration": 150.0, "median_triangles": 150_000.0, "median_upload_rate": 2_000_000.0,
    }
    assert history.average_timings() == {"vggt": 15.0, "refine": 100.0}

    # Rechargement depuis le disque : les entrées doivent survivre.
    assert len(History().records) == 3


def test_history_ignores_records_from_other_versions(isolated_store):
    store.HISTORY_PATH.write_text('[{"champ_inconnu": 1}]', encoding="utf-8")
    assert History().records == []


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

def test_stage_timeline_paints_with_and_without_data(qapp):
    timeline = StageTimeline()
    timeline.resize(600, 74)

    timeline.clear()
    assert not QPixmap(timeline.size()).isNull()
    timeline.render(QPixmap(timeline.size()))  # ne doit pas lever

    timeline.set_timings({"vggt": 12.0, "refine": 300.0, "mesh": 20.0}, current="refine")
    timeline.render(QPixmap(timeline.size()))
    assert "refine" in timeline.toolTip()
    assert "%" in timeline.toolTip()


def test_qt_labels_are_translated(qapp):
    """Sans les traductions Qt, une interface française affiche « Yes » et « Cancel »."""
    from PyQt6.QtCore import QCoreApplication

    from gui.__main__ import install_french

    translator = install_french(qapp)
    assert translator is not None, "qtbase_fr.qm absent de l'installation PyQt6"
    assert QCoreApplication.translate("QPlatformTheme", "Cancel") == "Annuler"
    qapp.removeTranslator(translator)


def test_job_picker_lists_and_returns_the_selection(qapp):
    from gui.widgets import JobPickerDialog

    dialog = JobPickerDialog([
        {"job_id": "aaa", "status": "running", "stage": "refine", "progress": 0.61},
        {"job_id": "bbb", "status": "done", "stage": "export", "progress": 1.0},
    ])

    assert dialog.table.rowCount() == 2
    assert dialog.selected_job_id() == "aaa"  # première ligne présélectionnée
    dialog.table.selectRow(1)
    assert dialog.selected_job_id() == "bbb"
    dialog.close()


def test_metric_card_updates(qapp):
    card = MetricCard("Triangles")
    card.set_value("203 k", "tsdf2dgs")
    assert card._value.text() == "203 k"
    assert card._hint.text() == "tsdf2dgs"


# ---------------------------------------------------------------------------
# Scan complet par l'interface
# ---------------------------------------------------------------------------

def test_full_scan_through_the_window(qapp, live_server, frames_dir, tmp_path, isolated_store):
    from gui.main_window import MainWindow

    window = MainWindow()
    window.url_edit.setText(live_server)
    window.token_edit.setText(TEST_TOKEN)
    window.source_edit.setText(str(frames_dir))
    window.output_edit.setText(str(tmp_path / "out"))
    window.preset_combo.setCurrentText("Validation rapide  (~2 min)")
    window.frames_spin.setValue(20)

    window.start_scan()
    assert window.scan_worker is not None
    assert not window.launch_button.isEnabled(), "le bouton doit être verrouillé pendant un scan"

    assert pump(qapp, lambda: not window.scan_worker.isRunning()), "le scan n'a pas abouti"
    qapp.processEvents()

    outcome = window.last_outcome
    assert outcome is not None, "aucun résultat remonté à l'interface"
    assert outcome.status == "done", outcome.error

    # Les artefacts sont bien arrivés sur le disque local.
    directory = Path(outcome.directory)
    assert (directory / "mesh.glb").is_file()
    assert (directory / "report.json").is_file()
    assert (directory / "mesh.glb").read_bytes()[:4] == b"glTF"

    # Les mesures côté client ont été calculées.
    assert outcome.payload_bytes > 0
    assert outcome.total_seconds > 0
    assert outcome.upload_rate > 0

    # L'interface a été rafraîchie.
    assert window.tabs.currentIndex() == 2  # bascule sur Résultats
    assert window.open_glb_button.isEnabled()
    assert window.artifacts_table.rowCount() >= 2
    assert window.launch_button.isEnabled()

    # L'historique a enregistré le scan.
    assert len(window.history.records) == 1
    assert window.history.records[0].status == "done"
    assert window.history_table.rowCount() == 1

    window.close()


def test_scan_refuses_missing_connection(qapp, tmp_path, isolated_store, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from gui.main_window import MainWindow

    warnings: list[str] = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args, **kwargs: warnings.append(args[2]))

    window = MainWindow()
    window.url_edit.setText("")
    window.token_edit.setText("")
    window.source_edit.setText(str(tmp_path))

    window.start_scan()

    assert window.scan_worker is None
    assert warnings and "Connexion" in warnings[0]
    assert window.tabs.currentIndex() == 0  # renvoyé vers l'onglet Connexion
    window.close()


def test_attach_to_a_job_started_elsewhere(qapp, live_server, frames_dir, tmp_path, isolated_store):
    """Rejoindre un job lancé hors de la fenêtre.

    C'est le scénario qui protège une minute de pod : la fenêtre s'est fermée
    pendant un scan de 15 min, le job continue côté serveur, et l'interface doit
    pouvoir reprendre le suivi puis récupérer les artefacts.
    """
    from client.payload import build_payload
    from client.transport import MinidsClient, sha256_file
    from gui.main_window import MainWindow

    # Le job est soumis sans passer par l'interface, comme le ferait la CLI.
    client = MinidsClient(url=live_server, token=TEST_TOKEN, timeout=30)
    payload = build_payload(source=frames_dir, workdir=tmp_path / "work")
    job_id = client.create_job(
        payload.name, payload.stat().st_size, CHUNK_SIZE, sha256_file(payload), {"frames": 20}
    )
    client.upload(job_id, payload, CHUNK_SIZE, quiet=True)
    client.start(job_id)

    window = MainWindow()
    window.url_edit.setText(live_server)
    window.token_edit.setText(TEST_TOKEN)
    window.output_edit.setText(str(tmp_path / "out"))
    window.job_id_edit.setText(job_id)

    window.attach_to_job()
    assert window.scan_worker is not None
    assert not window.launch_button.isEnabled(), "lancer et rejoindre s'excluent"

    assert pump(qapp, lambda: not window.scan_worker.isRunning()), "la reprise n'a pas abouti"
    qapp.processEvents()

    outcome = window.last_outcome
    assert outcome is not None and outcome.status == "done", getattr(outcome, "error", "aucun résultat")
    assert outcome.job_id == job_id
    assert (Path(outcome.directory) / "mesh.glb").is_file()

    # Rien n'a été téléversé par cette fenêtre : le débit montant reste vide.
    assert outcome.payload_bytes == 0
    assert outcome.upload_rate == 0.0
    assert outcome.download_bytes > 0

    # L'historique distingue une reprise d'un scan lancé ici.
    assert window.history.records[0].source == f"(repris) {job_id}"
    # Job terminé : plus rien à reproposer au prochain démarrage.
    assert Settings.load().last_job_id == ""
    window.close()


def test_a_crash_mid_scan_leaves_the_job_reachable(
    qapp, live_server, frames_dir, tmp_path, isolated_store
):
    """Un scan en cours doit survivre à la disparition brutale de la fenêtre.

    L'identifiant est écrit sur disque dès que le pod l'attribue — et non à la
    fermeture, qui n'a pas lieu quand le processus meurt. La fenêtre suivante le
    propose donc d'emblée, prêt à être rejoint.
    """
    from gui.main_window import MainWindow

    window = MainWindow()
    window.url_edit.setText(live_server)
    window.token_edit.setText(TEST_TOKEN)
    window.source_edit.setText(str(frames_dir))
    window.output_edit.setText(str(tmp_path / "out"))
    window.start_scan()

    assert pump(qapp, lambda: bool(window.settings.last_job_id)), "aucun job créé"
    job_id = window.settings.last_job_id

    # Ce que trouverait un processus relancé après un plantage : rien n'a été
    # sauvegardé à la fermeture, tout l'a été à la création du job.
    assert Settings.load().last_job_id == job_id
    reopened = MainWindow()
    assert reopened.job_id_edit.text() == job_id
    reopened.close()

    window.cancel_scan()
    assert pump(qapp, lambda: not window.scan_worker.isRunning(), timeout=60)
    window.close()


def test_cancelled_scan_is_not_reported_as_a_crash(qapp, tmp_path, isolated_store, monkeypatch):
    """Annuler est une décision, pas un incident : pas de boîte d'erreur."""
    from PyQt6.QtWidgets import QMessageBox

    from gui.main_window import MainWindow
    from gui.workers import ScanOutcome

    criticals: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: criticals.append(args[2]))

    window = MainWindow()
    window.settings.last_job_id = "job-en-cours"
    window._on_scan_failed(
        ScanOutcome(job_id="job-en-cours", status="cancelled", directory=tmp_path, error="annulé")
    )

    assert not criticals, "une annulation volontaire ne doit pas s'afficher comme une erreur"
    assert window.phase_label.text() == "annulé"
    assert len(window.history.records) == 1  # la trace reste dans l'historique
    assert Settings.load().last_job_id == ""
    window.close()


def test_failed_scan_still_raises_a_dialog(qapp, tmp_path, isolated_store, monkeypatch):
    """Le garde-fou du test précédent : un vrai échec doit rester visible."""
    from PyQt6.QtWidgets import QMessageBox

    from gui.main_window import MainWindow
    from gui.workers import ScanOutcome

    criticals: list[str] = []
    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: criticals.append(args[2]))

    window = MainWindow()
    window._on_scan_failed(
        ScanOutcome(job_id="abc", status="failed", directory=tmp_path, error="VRAM insuffisante")
    )

    assert criticals == ["VRAM insuffisante"]
    window.close()


def test_collect_params_matches_server_schema(qapp, isolated_store):
    """Les paramètres produits doivent être acceptés par le modèle Pydantic du serveur."""
    from gui.main_window import MainWindow
    from server.app import JobParams

    window = MainWindow()
    window.prompt_edit.setText("the sneaker")
    window.ref_size_spin.setValue(0.0)  # « échelle relative »
    window.compare_check.setChecked(True)
    window.refine_check.setChecked(True)
    window.backend_combo.setCurrentText("tsdf2dgs")

    params = window._collect_params()
    validated = JobParams(**params)

    assert validated.prompt == "the sneaker"
    assert validated.ref_size is None  # 0 doit devenir None, pas 0.0 (rejeté par gt=0)
    assert validated.mesh_backends[0] == "tsdf2dgs"
    assert set(validated.mesh_backends) == {"tsdf2dgs", "tsdf", "poisson"}
    window.close()


def test_compare_without_refine_drops_the_2dgs_backend(qapp, isolated_store):
    """tsdf2dgs exige le raffinement : le proposer sans lui ferait échouer le job."""
    from gui.main_window import MainWindow

    window = MainWindow()
    window.refine_check.setChecked(False)
    window.backend_combo.setCurrentText("tsdf")
    window.compare_check.setChecked(True)

    params = window._collect_params()

    assert "tsdf2dgs" not in params["mesh_backends"]
    assert params["mesh_backends"][0] == "tsdf"
    window.close()
