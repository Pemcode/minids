"""API : upload chunké, cycle de vie du job, téléchargement par Range.

Tourne en `MINIDS_FAKE_GPU=1`, donc sans GPU ni modèle : c'est exactement le
chemin réseau que le client empruntera sur le pod.
"""

from __future__ import annotations

import hashlib
import time
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


def test_health_reports_fake_mode(app):
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["fake_gpu"] is True
    assert body["auth_configured"] is True


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
            item for item in client.get(f"/jobs/{job_id}/artifacts", headers=AUTH).json()["artifacts"]
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
            item["size"] for item in client.get(f"/jobs/{job_id}/artifacts", headers=AUTH).json()["artifacts"]
            if item["name"] == "report.json"
        )
        response = client.get(
            f"/jobs/{job_id}/artifacts/report.json", headers={**AUTH, "Range": f"bytes={size + 10}-"}
        )
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
                "filename": "frames.tar", "size": len(payload), "chunk_size": len(payload),
                "sha256": "0" * 64, "params": {},
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
        while time.time() < deadline and client.get(f"/jobs/{job_id}", headers=AUTH).json()["status"] != "running":
            time.sleep(0.1)

        client.post(f"/jobs/{job_id}/cancel", headers=AUTH)
        assert wait_for(client, job_id, timeout=30)["status"] == "cancelled"
