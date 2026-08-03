"""Client de bout en bout contre un vrai serveur HTTP.

`TestClient` court-circuite la pile réseau ; ici on lance uvicorn et on parle en
urllib, comme depuis Windows. C'est le seul moyen de vérifier réellement la
reprise d'un téléchargement interrompu — le scénario qui fait perdre un scan
quand le proxy RunPod coupe à 100 s.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from client.transport import MinidsClient, MinidsError, sha256_file
from tests.conftest import CHUNK_SIZE, TEST_TOKEN

uvicorn = pytest.importorskip("uvicorn")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def live_server(app):
    port = _free_port()
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
def client(live_server):
    return MinidsClient(url=live_server, token=TEST_TOKEN, timeout=30)


def run_job(client: MinidsClient, archive: Path, chunk_size: int = CHUNK_SIZE) -> str:
    digest = sha256_file(archive)
    job_id = client.create_job(archive.name, archive.stat().st_size, chunk_size, digest, {"prompt": "objet"})
    client.upload(job_id, archive, chunk_size, quiet=True)
    client.start(job_id)
    state = client.wait(job_id, poll_seconds=0.3)
    assert state["status"] == "done", state.get("error")
    return job_id


def test_health(client):
    assert client.health()["fake_gpu"] is True


def test_bad_token_is_reported_clearly(live_server):
    """/health reste ouvert (HEALTHCHECK Docker), mais refuse un token erroné :
    sinon `minids health` validerait une configuration cassée."""
    with pytest.raises(MinidsError, match="401"):
        MinidsClient(url=live_server, token="mauvais").health()

    import urllib.request

    with urllib.request.urlopen(f"{live_server}/health", timeout=10) as response:
        assert response.status == 200  # sans en-tête : autorisé


def test_missing_url_or_token():
    with pytest.raises(MinidsError, match="URL"):
        MinidsClient(url="", token="x")
    with pytest.raises(MinidsError, match="token"):
        MinidsClient(url="http://x", token="")


def test_end_to_end_download(client, frames_archive, tmp_path):
    job_id = run_job(client, frames_archive)

    artifacts = {item["name"]: item for item in client.artifacts(job_id)}
    assert "mesh.glb" in artifacts

    destination = client.download(
        job_id, "mesh.glb", tmp_path / "mesh.glb", quiet=True,
        expected_sha256=artifacts["mesh.glb"]["sha256"],
    )
    assert destination.stat().st_size == artifacts["mesh.glb"]["size"]
    assert destination.read_bytes()[:4] == b"glTF"
    assert not destination.with_suffix(".glb.part").exists()


def test_download_resumes_from_partial_file(client, frames_archive, tmp_path):
    """Un `.part` déjà rempli ne doit pas être re-téléchargé depuis zéro."""
    job_id = run_job(client, frames_archive)
    artifact = next(item for item in client.artifacts(job_id) if item["name"] == "mesh.glb")

    reference = client.download(job_id, "mesh.glb", tmp_path / "reference.glb", quiet=True)
    payload = reference.read_bytes()

    destination = tmp_path / "resumed.glb"
    partial = destination.with_suffix(".glb.part")
    prefix = len(payload) // 2
    partial.write_bytes(payload[:prefix])

    client.download(job_id, "mesh.glb", destination, chunk_size=CHUNK_SIZE, quiet=True,
                    expected_sha256=artifact["sha256"])

    assert destination.read_bytes() == payload


def test_download_discards_oversized_partial(client, frames_archive, tmp_path):
    """Un `.part` plus gros que l'artefact (fichier périmé) doit être jeté."""
    job_id = run_job(client, frames_archive)
    reference = client.download(job_id, "report.json", tmp_path / "reference.json", quiet=True)

    destination = tmp_path / "report.json"
    destination.with_suffix(".json.part").write_bytes(b"X" * (reference.stat().st_size + 5000))

    client.download(job_id, "report.json", destination, quiet=True)

    assert destination.read_bytes() == reference.read_bytes()


def test_download_detects_corrupted_content(client, frames_archive, tmp_path):
    job_id = run_job(client, frames_archive)
    with pytest.raises(MinidsError, match="sha256"):
        client.download(job_id, "mesh.glb", tmp_path / "mesh.glb", quiet=True, expected_sha256="0" * 64)


def test_upload_skips_already_received_chunks(client, frames_archive):
    """Reprise d'upload : les chunks déjà envoyés ne repartent pas."""
    size = frames_archive.stat().st_size
    chunk_size = CHUNK_SIZE
    job_id = client.create_job(frames_archive.name, size, chunk_size, None, {})

    payload = frames_archive.read_bytes()
    client._request("PUT", f"/jobs/{job_id}/chunks/0", payload[:chunk_size],
                    {"Content-Type": "application/octet-stream"})
    assert client.uploaded_chunks(job_id) == {0}

    client.upload(job_id, frames_archive, chunk_size, quiet=True)

    expected = (size + chunk_size - 1) // chunk_size
    assert client.uploaded_chunks(job_id) == set(range(expected))
    assert client.start(job_id)["status"] == "queued"


def test_cancel_from_client(client, frames_archive):
    job_id = client.create_job(frames_archive.name, frames_archive.stat().st_size, CHUNK_SIZE, None, {})
    client.upload(job_id, frames_archive, CHUNK_SIZE, quiet=True)
    client.start(job_id)

    client.cancel(job_id)
    assert client.wait(job_id, poll_seconds=0.3)["status"] == "cancelled"


def test_jobs_lists_the_pod_state_most_recent_first(client, frames_archive):
    """Retrouver un job sans en connaître l'identifiant : c'est ce qui permet de
    rejoindre un scan lancé depuis une autre session."""
    older = client.create_job("a.tar", frames_archive.stat().st_size, CHUNK_SIZE, None, {})
    newer = client.create_job("b.tar", frames_archive.stat().st_size, CHUNK_SIZE, None, {})

    jobs = client.jobs()
    identifiers = [job["job_id"] for job in jobs]

    assert identifiers.index(newer) < identifiers.index(older)
    assert {older, newer} <= set(identifiers)
    assert jobs[0]["status"] == "created"  # pas encore démarré
