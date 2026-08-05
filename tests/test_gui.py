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
    assert duration(float("nan")) == "—"
    assert percent(float("inf")) == "—"


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


def test_settings_recover_from_valid_json_with_the_wrong_shape(isolated_store):
    store.SETTINGS_PATH.write_text("[]", encoding="utf-8")
    assert Settings.load() == Settings()

    store.SETTINGS_PATH.write_text('{"long_side":"huge","params":[],"remember_token":"yes"}', encoding="utf-8")
    loaded = Settings.load()
    assert loaded.long_side == 1024
    assert loaded.params == {}
    assert loaded.remember_token is False


def test_history_summary_and_averages(isolated_store):
    history = History()
    history.add(
        ScanRecord(
            job_id="a",
            timestamp=time.time(),
            source="v1.mp4",
            status="done",
            timings={"vggt": 10.0, "refine": 90.0},
            metrics={"triangles": 100_000},
            client={"total_seconds": 120.0, "upload_rate": 1_000_000.0},
        )
    )
    history.add(
        ScanRecord(
            job_id="b",
            timestamp=time.time(),
            source="v2.mp4",
            status="done",
            timings={"vggt": 20.0, "refine": 110.0},
            metrics={"triangles": 200_000},
            client={"total_seconds": 180.0, "upload_rate": 3_000_000.0},
        )
    )
    history.add(
        ScanRecord(
            job_id="c",
            timestamp=time.time(),
            source="v3.mp4",
            status="failed",
            error="boum",
        )
    )

    summary = history.summary()
    assert summary == {
        "count": 3,
        "done": 2,
        "failed": 1,
        "median_duration": 150.0,
        "median_triangles": 150_000.0,
        "median_upload_rate": 2_000_000.0,
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
    timeline.set_timings({"future_stage": 5.0})
    assert "future_stage" in timeline.toolTip()


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

    dialog = JobPickerDialog(
        [
            {"job_id": "aaa", "status": "running", "stage": "refine", "progress": 0.61},
            {"job_id": "bbb", "status": "done", "stage": "export", "progress": 1.0},
        ]
    )

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
    job_id = client.create_job(payload.name, payload.stat().st_size, CHUNK_SIZE, sha256_file(payload), {"frames": 20})
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


def test_a_crash_mid_scan_leaves_the_job_reachable(qapp, live_server, frames_dir, tmp_path, isolated_store):
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


def test_a_pod_without_model_credentials_is_flagged_before_scanning(qapp, tmp_path, isolated_store, monkeypatch):
    """Un scan condamné à échouer sur `vggt` ne doit pas partir sans avertissement."""
    from PyQt6.QtWidgets import QMessageBox

    from gui.main_window import MainWindow

    window = MainWindow()
    window.url_edit.setText("http://pod")
    window.token_edit.setText("jeton")
    window.source_edit.setText(str(tmp_path))

    window._on_health_ok({"cuda_available": True, "hf_token_configured": False, "checkpoint": "facebook/X:y.pt"})
    assert "HF_TOKEN" in window._model_access_warning

    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.StandardButton.No)[1],
    )
    window.start_scan()

    assert asked and "vggt" in asked[0]
    assert window.scan_worker is None, "le scan ne doit pas partir sur un refus"

    # Un pod correctement configuré ne pose aucune question.
    window._on_health_ok({"cuda_available": True, "hf_token_configured": True, "checkpoint": "facebook/X:y.pt"})
    assert window._model_access_warning == ""

    # Un pod d'une version antérieure ne publie pas ces champs : pas d'alerte inventée.
    window._on_health_ok({"cuda_available": True})
    assert window._model_access_warning == ""
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
    window._on_scan_failed(ScanOutcome(job_id="job-en-cours", status="cancelled", directory=tmp_path, error="annulé"))

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
    window._on_scan_failed(ScanOutcome(job_id="abc", status="failed", directory=tmp_path, error="VRAM insuffisante"))

    assert criticals == ["VRAM insuffisante"]
    window.close()


def test_worker_reports_local_cancellation_as_cancelled(qapp, tmp_path, monkeypatch):
    from client.transport import MinidsError
    from gui.workers import ScanRequest, ScanWorker

    request = ScanRequest(
        url="http://127.0.0.1:8000",
        token="token",
        source=tmp_path,
        output_dir=tmp_path / "out",
        params={},
    )
    worker = ScanWorker(request)
    outcomes = []
    worker.failed.connect(outcomes.append)

    def cancelled_prepare(*_args):
        worker._cancelled = True
        raise MinidsError("interrompu")

    monkeypatch.setattr(worker, "_prepare", cancelled_prepare)
    worker.run()

    assert outcomes and outcomes[0].status == "cancelled"
    assert outcomes[0].error == "annulé"


def test_worker_keeps_job_reachable_when_remote_cancellation_fails(qapp, tmp_path):
    from client.transport import MinidsError
    from gui.workers import ScanOutcome, ScanRequest, ScanWorker

    class OfflineClient:
        def cancel(self, _job_id):
            raise MinidsError("réseau indisponible")

    worker = ScanWorker(
        ScanRequest(
            url="http://127.0.0.1:8000",
            token="token",
            source=tmp_path,
            output_dir=tmp_path / "out",
            params={},
            job_id="job-still-running",
        )
    )
    worker._client = OfflineClient()
    worker._job_id = "job-still-running"
    worker._cancelled = True
    outcomes = []
    worker.failed.connect(outcomes.append)

    worker._emit_failure(
        ScanOutcome("job-still-running", "client_error", tmp_path),
        MinidsError("polling interrompu"),
        time.monotonic(),
    )

    assert outcomes[0].status == "client_error"
    assert outcomes[0].job_id == "job-still-running"
    assert "non confirmée" in outcomes[0].error


def test_detach_before_start_reports_confirmed_remote_cancellation(qapp, tmp_path):
    from client.transport import MinidsError
    from gui.workers import ScanOutcome, ScanRequest, ScanWorker

    class CreatedClient:
        def cancel(self, _job_id):
            return {"status": "cancelled"}

    worker = ScanWorker(
        ScanRequest(
            url="http://127.0.0.1:8000",
            token="token",
            source=tmp_path,
            output_dir=tmp_path / "out",
            params={},
        )
    )
    worker._client = CreatedClient()
    worker._job_id = "a" * 32
    worker._created_locally = True
    worker._detach_requested = True
    outcomes = []
    worker.failed.connect(outcomes.append)

    worker._emit_failure(
        ScanOutcome("", "client_error", tmp_path),
        MinidsError("fermeture demandée"),
        time.monotonic(),
    )

    assert outcomes[0].status == "cancelled"
    assert outcomes[0].error == "arrêté avant démarrage"


def test_worker_rejects_an_invalid_attached_job_id(qapp, tmp_path):
    from client.transport import MinidsError
    from gui.workers import ScanRequest, ScanWorker

    request = ScanRequest(
        url="http://127.0.0.1:8000",
        token="token",
        source=tmp_path,
        output_dir=tmp_path / "out",
        params={},
        job_id="../escape",
    )

    with pytest.raises(MinidsError, match="identifiant"):
        ScanWorker(request)._validate_request(request)


def test_attach_rejects_a_created_job_instead_of_polling_forever(qapp, tmp_path):
    from client.transport import MinidsError
    from gui.workers import ScanOutcome, ScanRequest, ScanWorker

    class CreatedClient:
        def status(self, _job_id):
            return {"status": "created", "logs": [], "progress": 0.0}

    job_id = "a" * 32
    worker = ScanWorker(
        ScanRequest(
            url="http://127.0.0.1:8000",
            token="token",
            source=tmp_path,
            output_dir=tmp_path / "out",
            params={},
            job_id=job_id,
        )
    )
    worker._client = CreatedClient()
    with pytest.raises(MinidsError, match="upload incomplet"):
        worker._wait(ScanOutcome(job_id, "client_error", tmp_path))


def test_a_silent_pod_does_not_end_the_tracking(qapp, tmp_path, monkeypatch):
    """Un sondage qui échoue n'est pas un scan qui échoue.

    Le pod se tait pendant les étapes lourdes — écriture du `.npz`, fusion,
    bake — et le proxy coupe alors la requête. Abandonner à la première coupure
    faisait perdre le suivi d'un job qui tournait parfaitement.
    """
    from gui.workers import MinidsError, ScanOutcome, ScanRequest, ScanWorker

    class FlakyClient:
        def __init__(self):
            self.calls = 0

        def status(self, _job_id):
            self.calls += 1
            if self.calls <= 3:
                raise MinidsError("TimeoutError The read operation timed out")
            return {"status": "done", "logs": [], "progress": 1.0}

    job_id = "b" * 32
    worker = ScanWorker(
        ScanRequest(
            url="http://127.0.0.1:8000",
            token="token",
            source=tmp_path,
            output_dir=tmp_path / "out",
            params={},
            job_id=job_id,
            poll_seconds=0.01,
        )
    )
    client = FlakyClient()
    worker._client = client

    assert worker._wait(ScanOutcome(job_id, "client_error", tmp_path))["status"] == "done"
    assert client.calls == 4, "les trois coupures auraient dû être retentées"


def test_a_pod_silent_too_long_gives_up_with_the_job_id(qapp, tmp_path, monkeypatch):
    """Le garde-fou du test précédent : le silence ne peut pas durer indéfiniment.

    Le message d'abandon doit porter l'identifiant, puisque le job continue
    côté pod et se rejoint.
    """
    from gui import workers
    from gui.workers import MinidsError, ScanOutcome, ScanRequest, ScanWorker

    class DeadClient:
        def status(self, _job_id):
            raise MinidsError("TimeoutError The read operation timed out")

    job_id = "c" * 32
    worker = ScanWorker(
        ScanRequest(
            url="http://127.0.0.1:8000",
            token="token",
            source=tmp_path,
            output_dir=tmp_path / "out",
            params={},
            job_id=job_id,
            poll_seconds=0.001,
        )
    )
    worker._client = DeadClient()
    monkeypatch.setattr(workers, "SILENCE_TOLERANCE_SECONDS", 0.05)

    with pytest.raises(MinidsError, match=job_id):
        worker._wait(ScanOutcome(job_id, "client_error", tmp_path))


def test_attach_waits_through_the_starting_state(qapp, tmp_path, monkeypatch):
    from gui import workers
    from gui.workers import ScanOutcome, ScanRequest, ScanWorker

    class StartingClient:
        def __init__(self):
            self.states = iter(
                [
                    {"status": "starting", "logs": [], "progress": 0.0},
                    {"status": "done", "logs": [], "progress": 1.0},
                ]
            )

        def status(self, _job_id):
            return next(self.states)

    job_id = "a" * 32
    worker = ScanWorker(
        ScanRequest(
            url="http://127.0.0.1:8000",
            token="token",
            source=tmp_path,
            output_dir=tmp_path / "out",
            params={},
            job_id=job_id,
            poll_seconds=0.01,
        )
    )
    worker._client = StartingClient()
    monkeypatch.setattr(workers.time, "sleep", lambda _delay: None)

    assert worker._wait(ScanOutcome(job_id, "client_error", tmp_path))["status"] == "done"


def test_every_preset_pins_the_segmentation(qapp, isolated_store):
    """Le trou par lequel un scan payant a reconstruit la pièce entière.

    Tant que `segmentation` manquait aux préréglages, un `none` choisi une fois
    survivait dans `~/.minids/gui-settings.json` et repartait tel quel : la
    scène était reconstruite au lieu de l'objet, et le nettoyage n'avait ni plan
    de support ni boîte pour la recadrer.
    """
    from gui.main_window import PRESETS

    named = {name: preset for name, preset in PRESETS.items() if preset}
    missing = [name for name, preset in named.items() if "segmentation" not in preset]

    assert not missing, f"préréglages sans segmentation : {missing}"
    assert all(preset["segmentation"] != "none" for preset in named.values())


def test_baseline_preset_matches_the_validated_recipe(qapp, isolated_store):
    """Le préréglage « Baseline VGGT » reprend la recette classée première par SP-012.

    Une vingtaine de vues, isolation géométrique, maillage Poisson, aucun
    raffinement 2DGS.
    """
    from gui.main_window import MainWindow

    window = MainWindow()
    window.preset_combo.setCurrentText("Baseline VGGT  (~2 min)")

    assert window.frames_spin.value() == 20
    assert window.segmentation_combo.currentText() == "geometric"
    assert window.backend_combo.currentText() == "poisson"
    assert not window.refine_check.isChecked()

    params = window._collect_params()
    assert params["mesh_backends"] == ["poisson"]
    assert params["segmentation"] == "geometric"
    window.close()


def test_disabled_segmentation_asks_before_reconstructing_the_scene(qapp, tmp_path, isolated_store, monkeypatch):
    """Reconstruire la scène entière est légitime, mais jamais par accident."""
    from PyQt6.QtWidgets import QMessageBox

    from gui.main_window import MainWindow

    asked: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: (asked.append(args[2]), QMessageBox.StandardButton.No)[1],
    )

    window = MainWindow()
    window.url_edit.setText("http://pod")
    window.token_edit.setText("jeton")
    window.source_edit.setText(str(tmp_path))
    window.segmentation_combo.setCurrentText("none")

    window.start_scan()

    assert asked and "plan de support" in asked[0]
    assert window.scan_worker is None, "un refus ne doit pas lancer le scan"

    # « geometric » ne demande rien : c'est le chemin sans friction, sans prompt.
    asked.clear()
    window.segmentation_combo.setCurrentText("geometric")
    assert window._confirm_despite_disabled_segmentation() is True
    assert not asked
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


def test_no_refine_maps_a_2dgs_primary_to_tsdf(qapp, isolated_store):
    from gui.main_window import MainWindow

    window = MainWindow()
    window._loading_preset = True
    window.refine_check.setChecked(False)
    window.backend_combo.setCurrentText("tsdf2dgs")
    window._loading_preset = False
    assert window._collect_params()["mesh_backends"] == ["tsdf"]
    window.close()


def test_checkpoint_configured_boolean_is_supported(qapp, isolated_store):
    from gui.main_window import MainWindow

    assert "MINIDS_CKPT" in MainWindow._missing_model_access({"checkpoint_configured": False})
    assert MainWindow._missing_model_access({"checkpoint_configured": True}) == ""
