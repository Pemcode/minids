"""API : upload chunké, cycle de vie du job, téléchargement par Range.

Tourne en `MINIDS_FAKE_GPU=1`, donc sans GPU ni modèle : c'est exactement le
chemin réseau que le client empruntera sur le pod.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from tests.conftest import CHUNK_SIZE, TEST_TOKEN

fastapi_testclient = pytest.importorskip("fastapi.testclient")
TestClient = fastapi_testclient.TestClient

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}


def upload_and_start(client, archive: Path, chunk_size: int = CHUNK_SIZE, params: dict | None = None) -> str:
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    response = client.post(
        "/jobs",
        json={
            "filename": archive.name,
            "size": len(payload),
            "chunk_size": chunk_size,
            "sha256": digest,
            "params": params or {"prompt": "the test object"},
        },
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    job = response.json()
    assert job["total_chunks"] == (len(payload) + chunk_size - 1) // chunk_size

    for index in range(job["total_chunks"]):
        block = payload[index * chunk_size : (index + 1) * chunk_size]
        put = client.put(f"/jobs/{job['job_id']}/chunks/{index}", content=block, headers=AUTH)
        assert put.status_code == 200, put.text

    assert client.post(f"/jobs/{job['job_id']}/start", headers=AUTH).status_code == 200
    return job["job_id"]


def wait_for(client, job_id: str, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if state["status"] in {"done", "failed", "cancelled"}:
            return state
        time.sleep(0.25)
    raise AssertionError(f"job {job_id} toujours en {state['status']} après {timeout}s")


def create_payload_job(
    client,
    filename: str,
    payload: bytes,
    *,
    chunk_size: int = CHUNK_SIZE,
    sha256: str | None = None,
    params: dict | None = None,
) -> dict:
    response = client.post(
        "/jobs",
        json={
            "filename": filename,
            "size": len(payload),
            "chunk_size": chunk_size,
            "sha256": sha256,
            "params": params or {},
        },
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    return response.json()


def upload_payload(client, job: dict, payload: bytes, chunk_size: int = CHUNK_SIZE) -> None:
    for index in range(job["total_chunks"]):
        block = payload[index * chunk_size : (index + 1) * chunk_size]
        response = client.put(f"/jobs/{job['job_id']}/chunks/{index}", content=block, headers=AUTH)
        assert response.status_code == 200, response.text


def add_artifact(app, name: str, payload: bytes) -> str:
    job = app.state.store.create(
        {},
        {
            "filename": "input.tar",
            "size": 1,
            "chunk_size": CHUNK_SIZE,
            "sha256": None,
            "total_chunks": 1,
            "received": [],
        },
    )
    job.status = "done"
    job.finished_at = time.time()
    (job.artifacts_dir / name).write_bytes(payload)
    job.persist()
    return job.job_id


def test_health_reports_fake_mode(app):
    with TestClient(app) as client:
        public = client.get("/health").json()
        body = client.get("/health", headers=AUTH).json()
        invalid = client.get("/health", headers={"Authorization": "Bearer faux"})
    assert public == {"status": "ok"}
    assert invalid.status_code == 401
    assert body["status"] == "ok"
    assert body["fake_gpu"] is True
    assert body["auth_configured"] is True


def test_health_reports_missing_model_credentials(settings):
    """Sans HF_TOKEN, le job ne meurt qu'à l'étape `vggt`, GPU déjà facturé.

    `/health` doit donc le dire d'avance — sans publier le secret lui-même.
    """
    from server.app import build_app

    with TestClient(build_app(replace(settings, fake_gpu=False))) as client:
        body = client.get("/health", headers=AUTH).json()
    assert body["hf_token_configured"] is False
    assert body["checkpoint_configured"] is False

    configured = replace(settings, fake_gpu=False, hf_token="hf_secret", checkpoint="facebook/X:y.pt")
    with TestClient(build_app(configured)) as client:
        body = client.get("/health", headers=AUTH).json()
    assert body["hf_token_configured"] is True
    assert body["checkpoint_configured"] is True
    serialized = json.dumps(body)
    assert "hf_secret" not in serialized, "le jeton ne doit jamais être publié"
    assert "facebook/X:y.pt" not in serialized, "le chemin du checkpoint ne doit jamais être publié"


def test_authentication_is_required(app):
    with TestClient(app) as client:
        assert client.get("/jobs").status_code == 401
        assert client.get("/jobs", headers={"Authorization": "Bearer faux"}).status_code == 401
        assert client.get("/jobs", headers=AUTH).status_code == 200


def test_full_job_produces_artifacts(app, frames_archive):
    with TestClient(app) as client:
        job_id = upload_and_start(client, frames_archive)
        state = wait_for(client, job_id)

        assert state["status"] == "done", state.get("error")
        assert state["progress"] == pytest.approx(1.0, abs=0.01)
        assert state["stage_timings"]

        names = {item["name"] for item in client.get(f"/jobs/{job_id}/artifacts", headers=AUTH).json()["artifacts"]}
        assert {"mesh.glb", "vggt_raw.npz", "report.json", "preview.png"} <= names


def test_download_full_and_ranged_agree(app, frames_archive):
    with TestClient(app) as client:
        job_id = upload_and_start(client, frames_archive)
        assert wait_for(client, job_id)["status"] == "done"

        entry = next(
            item
            for item in client.get(f"/jobs/{job_id}/artifacts", headers=AUTH).json()["artifacts"]
            if item["name"] == "mesh.glb"
        )
        full = client.get(f"/jobs/{job_id}/artifacts/mesh.glb", headers=AUTH)
        assert full.status_code == 200
        assert full.headers["accept-ranges"] == "bytes"
        assert len(full.content) == entry["size"]
        assert hashlib.sha256(full.content).hexdigest() == entry["sha256"]

        # Reconstruit le même fichier par tranches, comme le fait le client.
        assembled = b""
        step = max(1, entry["size"] // 3)
        while len(assembled) < entry["size"]:
            end = min(len(assembled) + step - 1, entry["size"] - 1)
            part = client.get(
                f"/jobs/{job_id}/artifacts/mesh.glb",
                headers={**AUTH, "Range": f"bytes={len(assembled)}-{end}"},
            )
            assert part.status_code == 206
            assert part.headers["content-range"] == f"bytes {len(assembled)}-{end}/{entry['size']}"
            assembled += part.content
        assert assembled == full.content
        assert assembled[:4] == b"glTF"


def test_range_beyond_end_returns_416(app, frames_archive):
    with TestClient(app) as client:
        job_id = upload_and_start(client, frames_archive)
        assert wait_for(client, job_id)["status"] == "done"

        size = next(
            item["size"]
            for item in client.get(f"/jobs/{job_id}/artifacts", headers=AUTH).json()["artifacts"]
            if item["name"] == "report.json"
        )
        response = client.get(f"/jobs/{job_id}/artifacts/report.json", headers={**AUTH, "Range": f"bytes={size + 10}-"})
        assert response.status_code == 416
        assert response.headers["content-range"] == f"bytes */{size}"


def test_suffix_range_returns_tail(app, frames_archive):
    with TestClient(app) as client:
        job_id = upload_and_start(client, frames_archive)
        assert wait_for(client, job_id)["status"] == "done"

        full = client.get(f"/jobs/{job_id}/artifacts/report.json", headers=AUTH).content
        tail = client.get(f"/jobs/{job_id}/artifacts/report.json", headers={**AUTH, "Range": "bytes=-16"})
        assert tail.status_code == 206
        assert tail.content == full[-16:]


def test_artifact_name_traversal_is_rejected(app, frames_archive):
    with TestClient(app) as client:
        job_id = upload_and_start(client, frames_archive)
        assert wait_for(client, job_id)["status"] == "done"

        for name in ("..%2Fstate.json", "mesh.glb.sha256"):
            assert client.get(f"/jobs/{job_id}/artifacts/{name}", headers=AUTH).status_code in {400, 404}


def test_missing_chunk_blocks_start(app, frames_archive):
    payload = frames_archive.read_bytes()
    with TestClient(app) as client:
        job = client.post(
            "/jobs",
            json={"filename": "frames.tar", "size": len(payload), "chunk_size": CHUNK_SIZE, "params": {}},
            headers=AUTH,
        ).json()
        client.put(f"/jobs/{job['job_id']}/chunks/0", content=payload[:CHUNK_SIZE], headers=AUTH)

        response = client.post(f"/jobs/{job['job_id']}/start", headers=AUTH)
        assert response.status_code == 400
        assert "manquants" in response.json()["error"]


def test_wrong_sha256_is_detected(app, frames_archive):
    payload = frames_archive.read_bytes()
    with TestClient(app) as client:
        job = client.post(
            "/jobs",
            json={
                "filename": "frames.tar",
                "size": len(payload),
                "chunk_size": len(payload),
                "sha256": "0" * 64,
                "params": {},
            },
            headers=AUTH,
        ).json()
        client.put(f"/jobs/{job['job_id']}/chunks/0", content=payload, headers=AUTH)

        response = client.post(f"/jobs/{job['job_id']}/start", headers=AUTH)
        assert response.status_code == 400
        assert "sha256" in response.json()["error"]


def test_chunk_resume_reports_received_indices(app, frames_archive):
    payload = frames_archive.read_bytes()
    with TestClient(app) as client:
        job = client.post(
            "/jobs",
            json={"filename": "frames.tar", "size": len(payload), "chunk_size": CHUNK_SIZE, "params": {}},
            headers=AUTH,
        ).json()
        client.put(f"/jobs/{job['job_id']}/chunks/0", content=payload[:CHUNK_SIZE], headers=AUTH)
        client.put(f"/jobs/{job['job_id']}/chunks/2", content=payload[2 * CHUNK_SIZE : 3 * CHUNK_SIZE], headers=AUTH)

        listing = client.get(f"/jobs/{job['job_id']}/chunks", headers=AUTH).json()
        assert listing["received"] == [0, 2]
        assert listing["total"] == job["total_chunks"]


def test_cancel_stops_a_running_job(app, frames_archive):
    with TestClient(app) as client:
        job_id = upload_and_start(client, frames_archive)
        deadline = time.time() + 20
        status = ""
        while time.time() < deadline:
            status = client.get(f"/jobs/{job_id}", headers=AUTH).json()["status"]
            if status == "running":
                break
            time.sleep(0.1)
        assert status == "running"

        root = app.state.store.get(job_id).root
        refused = client.delete(f"/jobs/{job_id}", headers=AUTH)
        assert refused.status_code == 409
        assert root.is_dir(), "DELETE ne doit pas supprimer les fichiers utilisés par le runner"

        client.post(f"/jobs/{job_id}/cancel", headers=AUTH)
        assert wait_for(client, job_id, timeout=30)["status"] == "cancelled"
        assert client.delete(f"/jobs/{job_id}", headers=AUTH).status_code == 200
        assert not root.exists()


@pytest.mark.parametrize(
    "override",
    [
        {"filename": "../frames.tar"},
        {"filename": "frames.exe"},
        {"filename": "é" * 250 + ".tar"},
        {"chunk_size": 64 * 1024 * 1024 + 1},
        {"sha256": "invalide"},
        {"params": {"segmentation": "sam3", "prompt": "   "}},
        {"params": {"mesh_backends": []}},
        {"params": {"bundle_adjustment": True}},
        {"params": {"texture_size": 4097}},
        {"params": {"option_inconnue": True}},
    ],
)
def test_create_job_rejects_invalid_contract_values(app, override):
    request = {"filename": "frames.tar", "size": 1, "chunk_size": CHUNK_SIZE, "params": {}}
    request.update(override)
    with TestClient(app) as client:
        response = client.post("/jobs", json=request, headers=AUTH)
    assert response.status_code == 422


def test_create_job_rejects_non_finite_real_scale(app):
    body = json.dumps(
        {"filename": "frames.tar", "size": 1, "chunk_size": CHUNK_SIZE, "params": {"ref_size": float("inf")}}
    )
    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            content=body,
            headers={**AUTH, "Content-Type": "application/json"},
        )
    assert response.status_code == 422


def test_server_chunk_default_is_really_used(app):
    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            json={"filename": "frames.tar", "size": CHUNK_SIZE + 1, "params": {}},
            headers=AUTH,
        )
    assert response.status_code == 200
    assert response.json()["chunk_size"] == CHUNK_SIZE
    assert response.json()["total_chunks"] == 2


def test_malformed_job_identifier_is_rejected(app):
    with TestClient(app) as client:
        response = client.get("/jobs/not-a-job-id", headers=AUTH)
    assert response.status_code == 400


def test_last_chunk_must_have_its_exact_declared_size(app):
    payload = b"a" * (CHUNK_SIZE + 5)
    with TestClient(app) as client:
        job = create_payload_job(client, "frames.tar", payload)
        first = client.put(f"/jobs/{job['job_id']}/chunks/0", content=payload[:CHUNK_SIZE], headers=AUTH)
        short = client.put(f"/jobs/{job['job_id']}/chunks/1", content=b"1234", headers=AUTH)
        listing = client.get(f"/jobs/{job['job_id']}/chunks", headers=AUTH).json()
        exact = client.put(f"/jobs/{job['job_id']}/chunks/1", content=b"12345", headers=AUTH)
    assert first.status_code == 200
    assert short.status_code == 400
    assert "Content-Length" in short.json()["error"]
    assert listing["received"] == [0]
    assert exact.status_code == 200


def test_retrying_a_chunk_with_different_content_is_rejected(app):
    payload = b"a" * CHUNK_SIZE
    with TestClient(app) as client:
        job = create_payload_job(client, "frames.tar", payload)
        assert client.put(f"/jobs/{job['job_id']}/chunks/0", content=payload, headers=AUTH).status_code == 200
        same = client.put(f"/jobs/{job['job_id']}/chunks/0", content=payload, headers=AUTH)
        conflict = client.put(f"/jobs/{job['job_id']}/chunks/0", content=b"b" * CHUNK_SIZE, headers=AUTH)
    assert same.status_code == 200
    assert conflict.status_code == 409


def test_start_rechecks_chunk_files_and_declared_total(app, frames_archive):
    payload = frames_archive.read_bytes()
    with TestClient(app) as client:
        job = create_payload_job(client, frames_archive.name, payload)
        upload_payload(client, job, payload)
        stored = app.state.store.get(job["job_id"])
        last = stored.upload_dir / f"{job['total_chunks'] - 1:06d}.part"
        last.write_bytes(last.read_bytes()[:-1])

        response = client.post(f"/jobs/{job['job_id']}/start", headers=AUTH)
        state = client.get(f"/jobs/{job['job_id']}", headers=AUTH).json()
    assert response.status_code == 400
    assert "attendus" in response.json()["error"]
    assert state["status"] == "created", "un chunk corrompu doit pouvoir être renvoyé"


def test_concurrent_start_is_idempotent_and_enqueues_once(app, frames_archive, monkeypatch):
    import server.app as app_module

    payload = frames_archive.read_bytes()
    real_assemble = app_module.assemble_upload
    entered = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def slow_assemble(job):
        calls.append(job.job_id)
        entered.set()
        assert release.wait(timeout=5)
        return real_assemble(job)

    monkeypatch.setattr(app_module, "assemble_upload", slow_assemble)
    with TestClient(app) as client:
        job = create_payload_job(client, frames_archive.name, payload, sha256=hashlib.sha256(payload).hexdigest())
        upload_payload(client, job, payload)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(client.post, f"/jobs/{job['job_id']}/start", headers=AUTH)
            assert entered.wait(timeout=2)
            second_future = pool.submit(client.post, f"/jobs/{job['job_id']}/start", headers=AUTH)
            second = second_future.result(timeout=2)
            assert second.status_code == 200
            assert second.json()["status"] == "starting"
            release.set()
            first = first_future.result(timeout=10)
        assert first.status_code == 200
        assert calls == [job["job_id"]]
        client.post(f"/jobs/{job['job_id']}/cancel", headers=AUTH)
        assert wait_for(client, job["job_id"], timeout=30)["status"] == "cancelled"


def test_created_upload_and_chunks_are_restored_after_restart(settings):
    from server.app import build_app

    payload = b"a" * (CHUNK_SIZE + 1)
    first_app = build_app(settings)
    with TestClient(first_app) as client:
        job = create_payload_job(client, "frames.tar", payload)
        put = client.put(f"/jobs/{job['job_id']}/chunks/0", content=payload[:CHUNK_SIZE], headers=AUTH)
        assert put.status_code == 200
        assert first_app.state.store.worker_alive is False

    second_app = build_app(settings)
    with TestClient(second_app) as client:
        state = client.get(f"/jobs/{job['job_id']}", headers=AUTH).json()
        chunks = client.get(f"/jobs/{job['job_id']}/chunks", headers=AUTH).json()
    assert state["status"] == "created"
    assert chunks["received"] == [0]


@pytest.mark.parametrize("interrupted_status", ["starting", "queued", "running"])
def test_interrupted_job_is_marked_failed_on_restart(settings, interrupted_status):
    from server.app import build_app

    first_app = build_app(settings)
    job = first_app.state.store.create(
        {},
        {
            "filename": "frames.tar",
            "size": 1,
            "chunk_size": CHUNK_SIZE,
            "sha256": None,
            "total_chunks": 1,
            "received": [],
        },
    )
    job.status = interrupted_status
    job.persist()

    second_app = build_app(settings)
    restored = second_app.state.store.get(job.job_id)
    try:
        assert restored is not None
        assert restored.status == "failed"
        assert "redémarré" in restored.error
        assert restored.finished_at is not None
    finally:
        second_app.state.store.close()


def test_corrupt_persisted_upload_does_not_block_server_boot(settings):
    from server.app import build_app

    job_id = "0" * 32
    root = settings.jobs_dir / job_id
    root.mkdir(parents=True)
    (root / "state.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "created",
                "params": {},
                "upload_full": {"filename": "x.tar", "size": "x", "chunk_size": CHUNK_SIZE, "total_chunks": 1},
            }
        ),
        encoding="utf-8",
    )
    restored_app = build_app(settings)
    try:
        assert restored_app.state.store.list() == []
    finally:
        restored_app.state.store.close()


@pytest.mark.parametrize("first_name", ["a/frame.png", "../frame.png"])
def test_archive_flattening_rejects_duplicates_and_traversal(app, tmp_path, first_name):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr(first_name, b"first")
        if ".." not in first_name:
            handle.writestr("b/frame.png", b"second")
    payload = archive.read_bytes()

    with TestClient(app) as client:
        job = create_payload_job(client, archive.name, payload)
        upload_payload(client, job, payload)
        response = client.post(f"/jobs/{job['job_id']}/start", headers=AUTH)
        state = client.get(f"/jobs/{job['job_id']}", headers=AUTH).json()
    assert response.status_code == 400
    assert "dupliqué" in response.json()["error"] or "suspect" in response.json()["error"]
    assert state["status"] == "failed"


def test_archive_expanded_size_is_limited_before_enqueue(settings, tmp_path):
    from server.app import build_app

    archive = tmp_path / "bomb.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("frame.png", b"x" * 200_000)
    payload = archive.read_bytes()
    assert len(payload) < 100_000
    limited_app = build_app(replace(settings, max_upload_bytes=100_000))
    with TestClient(limited_app) as client:
        job = create_payload_job(client, archive.name, payload)
        upload_payload(client, job, payload)
        response = client.post(f"/jobs/{job['job_id']}/start", headers=AUTH)
    assert response.status_code == 400
    assert "décompressée" in response.json()["error"]
    assert limited_app.state.store.worker_alive is False


def test_archive_member_count_limit_is_enforced(tmp_path):
    from server.jobs import validate_input_archive

    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("one.png", b"1")
        handle.writestr("two.png", b"2")
    with pytest.raises(ValueError, match="trop d'entrées"):
        validate_input_archive(archive, max_extracted_bytes=1024, max_files=1)


def test_archive_member_name_is_limited_in_utf8_bytes(tmp_path):
    from server.jobs import validate_input_archive

    archive = tmp_path / "long-name.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("é" * 130 + ".png", b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError, match="trop long"):
        validate_input_archive(archive, max_extracted_bytes=1024)


def test_archive_requires_a_nonempty_image_with_valid_signature(tmp_path):
    from server.jobs import validate_input_archive

    archive = tmp_path / "invalid-image.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("frame.png", b"")
    with pytest.raises(ValueError, match="image d'archive invalide"):
        validate_input_archive(archive, max_extracted_bytes=1024)


def test_invalid_recent_sha_sidecar_is_recomputed_and_hidden(app):
    payload = b"artifact-content"
    job_id = add_artifact(app, "result.bin", payload)
    artifact = app.state.store.get(job_id).artifacts_dir / "result.bin"
    sidecar = artifact.with_name("result.bin.sha256")
    sidecar.write_text("not-a-digest", encoding="utf-8")

    with TestClient(app) as client:
        listing = client.get(f"/jobs/{job_id}/artifacts", headers=AUTH).json()["artifacts"]
        state = client.get(f"/jobs/{job_id}", headers=AUTH).json()
    expected = hashlib.sha256(payload).hexdigest()
    assert listing == [{"name": "result.bin", "size": len(payload), "sha256": expected}]
    assert state["artifacts"] == ["result.bin"]
    assert sidecar.read_text(encoding="utf-8") == expected


def test_range_edge_cases_return_consistent_headers(app):
    payload = b"0123456789"
    job_id = add_artifact(app, "range.bin", payload)
    url = f"/jobs/{job_id}/artifacts/range.bin"
    with TestClient(app) as client:
        zero_suffix = client.get(url, headers={**AUTH, "Range": "bytes=-0"})
        reversed_range = client.get(url, headers={**AUTH, "Range": "bytes=8-3"})
        multiple = client.get(url, headers={**AUTH, "Range": "bytes=0-1,3-4"})
        large_suffix = client.get(url, headers={**AUTH, "Range": "bytes=-999"})
    for response in (zero_suffix, reversed_range):
        assert response.status_code == 416
        assert response.headers["content-range"] == f"bytes */{len(payload)}"
        assert response.headers["accept-ranges"] == "bytes"
        assert response.headers["x-content-length"] == str(len(payload))
    assert multiple.status_code == 400
    assert large_suffix.status_code == 206
    assert large_suffix.content == payload


def test_worker_thread_stops_with_application_lifespan(settings, frames_archive):
    from server.app import build_app

    application = build_app(settings)
    store = application.state.store
    with TestClient(application) as client:
        job_id = upload_and_start(client, frames_archive)
        assert wait_for(client, job_id, timeout=30)["status"] == "done"
        assert store.worker_alive is True
    assert store.worker_alive is False


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("MINIDS_FAKE_GPU", "tru"),
        ("MINIDS_CHUNK_SIZE", str(64 * 1024 * 1024 + 1)),
        ("MINIDS_FRAMES", "0"),
        ("MINIDS_IMAGE_RESOLUTION", "not-an-int"),
    ],
)
def test_invalid_environment_configuration_fails_fast(monkeypatch, name, value):
    from server.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv(name, value)
    try:
        with pytest.raises(ValueError, match=name):
            get_settings()
    finally:
        get_settings.cache_clear()


def test_settings_normalize_secrets_and_have_no_dead_keep_work_flag(tmp_path):
    from server.config import Settings

    configured = Settings(token="  secret\n", device=" cpu ", hf_token=" hf_test ", data_dir=tmp_path)
    assert configured.token == "secret"
    assert configured.device == "cpu"
    assert configured.hf_token == "hf_test"
    assert not hasattr(configured, "keep_work_dir")
