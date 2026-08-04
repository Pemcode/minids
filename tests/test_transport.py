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
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import pytest

from client.transport import (
    MAX_CHUNK,
    MinidsClient,
    MinidsError,
    MinidsHTTPError,
    _describe_http_error,
    _SameOriginRedirectHandler,
    new_log_lines,
    sha256_file,
)
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
    with pytest.raises(MinidsError, match="URL"):
        MinidsClient(url=None, token="x")
    with pytest.raises(MinidsError, match="token"):
        MinidsClient(url="http://127.0.0.1", token=None)


def test_fastapi_validation_error_is_human_readable():
    payload = b'{"detail":[{"loc":["body","filename"],"msg":"Field required"}]}'
    error = urllib.error.HTTPError(
        "https://example.invalid/jobs",
        422,
        "Unprocessable Entity",
        {},
        BytesIO(payload),
    )

    assert _describe_http_error(error) == "HTTP 422 : body.filename: Field required"


def test_end_to_end_download(client, frames_archive, tmp_path):
    job_id = run_job(client, frames_archive)

    artifacts = {item["name"]: item for item in client.artifacts(job_id)}
    assert "mesh.glb" in artifacts

    destination = client.download(
        job_id,
        "mesh.glb",
        tmp_path / "mesh.glb",
        quiet=True,
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

    client.download(
        job_id, "mesh.glb", destination, chunk_size=CHUNK_SIZE, quiet=True, expected_sha256=artifact["sha256"]
    )

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
    destination = tmp_path / "mesh.glb"
    with pytest.raises(MinidsError, match="sha256"):
        client.download(job_id, "mesh.glb", destination, quiet=True, expected_sha256="0" * 64)
    assert not destination.with_suffix(".glb.part").exists(), "une reprise corrompue échouerait indéfiniment"


def test_download_rejects_an_invalid_expected_hash_before_network(tmp_path):
    client = MinidsClient("https://example.invalid", "token")

    with pytest.raises(MinidsError, match="empreinte"):
        client.download("a" * 32, "mesh.glb", tmp_path / "mesh.glb", expected_sha256="bad")


def test_download_rejects_an_oversized_chunk_before_network(tmp_path):
    client = MinidsClient("https://example.invalid", "token")

    with pytest.raises(MinidsError, match="chunk"):
        client.download(
            "a" * 32,
            "mesh.glb",
            tmp_path / "mesh.glb",
            chunk_size=MAX_CHUNK + 1,
        )


def test_upload_skips_already_received_chunks(client, frames_archive):
    """Reprise d'upload : les chunks déjà envoyés ne repartent pas."""
    size = frames_archive.stat().st_size
    chunk_size = CHUNK_SIZE
    job_id = client.create_job(frames_archive.name, size, chunk_size, None, {})

    payload = frames_archive.read_bytes()
    client._request(
        "PUT", f"/jobs/{job_id}/chunks/0", payload[:chunk_size], {"Content-Type": "application/octet-stream"}
    )
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


def test_requests_announce_an_explicit_user_agent(client, monkeypatch):
    """Cloudflare, devant le proxy RunPod, renvoie 403 à l'agent par défaut d'urllib.

    Le pod n'est alors jamais atteint : `/health` répond dans un navigateur mais
    le client se voit tout refuser, avec une configuration pourtant correcte.
    """
    seen: list[str | None] = []
    real_open = client._opener.open

    def spy(request, *args, **kwargs):
        seen.append(request.get_header("User-agent"))
        return real_open(request, *args, **kwargs)

    monkeypatch.setattr(client._opener, "open", spy)
    client.health()

    assert seen, "aucune requête émise"
    assert seen[0] and "Python-urllib" not in seen[0]


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


def test_plain_http_is_only_allowed_for_localhost():
    with pytest.raises(MinidsError, match="HTTP non chiffré"):
        MinidsClient("http://example.com", "secret")
    assert MinidsClient("http://127.0.0.1:8000", "secret").url.endswith(":8000")


def test_cross_origin_redirect_is_refused_before_reusing_authorization():
    handler = _SameOriginRedirectHandler()
    request = urllib.request.Request("https://pod.example/health", headers={"Authorization": "Bearer secret"})
    with pytest.raises(urllib.error.HTTPError, match="autre origine"):
        handler.redirect_request(
            request,
            BytesIO(b""),
            302,
            "Found",
            {"Location": "https://attacker.example/collect"},
            "https://attacker.example/collect",
        )


def test_upload_progress_counts_non_contiguous_resumed_chunks_exactly(tmp_path, monkeypatch):
    chunk = 64 * 1024
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x" * (2 * chunk + 17))
    client = MinidsClient("http://127.0.0.1:8000", "secret")
    monkeypatch.setattr(client, "uploaded_chunks", lambda _job: {0, 2})
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: {})
    progress: list[tuple[int, int]] = []

    client.upload(
        "a" * 32, payload, chunk_size=chunk, quiet=True, on_progress=lambda done, total: progress.append((done, total))
    )

    assert progress[0] == (chunk + 17, payload.stat().st_size)
    assert progress[-1] == (payload.stat().st_size, payload.stat().st_size)


def test_download_restarts_if_a_proxy_ignores_range(tmp_path, monkeypatch):
    payload = b"complete artifact"
    destination = tmp_path / "artifact.bin"
    destination.with_suffix(".bin.part").write_bytes(b"stale prefix")
    client = MinidsClient("http://127.0.0.1:8000", "secret")
    monkeypatch.setattr(
        client,
        "_request",
        lambda *args, **kwargs: (payload, {"Content-Length": str(len(payload))}, 200),
    )

    client.download("a" * 32, "artifact.bin", destination, chunk_size=64 * 1024, quiet=True)

    assert destination.read_bytes() == payload


def test_download_supports_an_empty_artifact(tmp_path, monkeypatch):
    destination = tmp_path / "empty.bin"
    client = MinidsClient("http://127.0.0.1:8000", "secret")

    def empty_response(*args, **kwargs):
        raise MinidsHTTPError(416, "vide", {"Content-Range": "bytes */0"})

    monkeypatch.setattr(client, "_request", empty_response)
    client.download("a" * 32, "empty.bin", destination, chunk_size=64 * 1024, quiet=True)
    assert destination.read_bytes() == b""


@pytest.mark.parametrize("job_id", ["..", "../escape", "/absolute", "g" * 32, "a" * 31])
def test_job_ids_cannot_escape_the_output_directory(job_id, tmp_path):
    from client.transport import job_destination

    with pytest.raises(MinidsError, match="identifiant"):
        job_destination(tmp_path, job_id)


def test_wait_rejects_invalid_protocol_state(monkeypatch):
    client = MinidsClient("http://127.0.0.1:8000", "secret")
    monkeypatch.setattr(client, "status", lambda _job: {"status": "mystery", "logs": []})
    with pytest.raises(MinidsError, match="statut inconnu"):
        client.wait("job", poll_seconds=0.01)

    monkeypatch.setattr(client, "status", lambda _job: {"status": "running", "logs": "not-a-list"})
    with pytest.raises(MinidsError, match="journal invalide"):
        client.wait("job", poll_seconds=0.01)

    monkeypatch.setattr(client, "status", lambda _job: {"status": "created", "logs": []})
    with pytest.raises(MinidsError, match="upload incomplet"):
        client.wait("job", poll_seconds=0.01)


def test_starting_is_a_valid_non_terminal_protocol_state(monkeypatch):
    client = MinidsClient("http://127.0.0.1:8000", "secret")
    job_id = "a" * 32
    states = iter(
        [
            {"status": "starting", "logs": ["assemblage"]},
            {"status": "done", "logs": ["assemblage", "terminé"]},
        ]
    )
    monkeypatch.setattr(client, "status", lambda _job: next(states))
    monkeypatch.setattr("client.transport.time.sleep", lambda _delay: None)

    assert client.wait(job_id, poll_seconds=0.01)["status"] == "done"


def test_non_idempotent_post_is_never_retried_after_network_failure(monkeypatch):
    client = MinidsClient("http://127.0.0.1:8000", "secret", retries=5)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise urllib.error.URLError("connection lost")

    monkeypatch.setattr(client._opener, "open", fail)
    with pytest.raises(MinidsError, match="POST"):
        client._request("POST", "/jobs", b"{}")
    assert calls == 1, "retenter /jobs pourrait créer plusieurs jobs facturés"


def test_start_recovers_a_lost_response_when_job_is_queued(monkeypatch):
    client = MinidsClient("https://example.invalid", "token")
    job_id = "a" * 32
    calls: list[tuple[str, str]] = []

    def fake_request(method, path, *args, **kwargs):
        calls.append((method, path))
        if method == "POST":
            raise MinidsError("POST /start : timeout")
        return {"status": "queued"}

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.start(job_id)["status"] == "queued"
    assert calls == [("POST", f"/jobs/{job_id}/start"), ("GET", f"/jobs/{job_id}?logs=false")]


def test_start_does_not_replay_an_ambiguous_post(monkeypatch):
    client = MinidsClient("https://example.invalid", "token")
    job_id = "a" * 32
    post_calls = 0

    def fake_request(method, path, *args, **kwargs):
        nonlocal post_calls
        if method == "POST":
            post_calls += 1
            raise MinidsError("POST /start : timeout")
        return {"status": "created"}

    monkeypatch.setattr(client, "_request", fake_request)

    with pytest.raises(MinidsError, match="non confirmé"):
        client.start(job_id)
    assert post_calls == 1


def test_start_recovers_an_ambiguous_proxy_error_via_status(monkeypatch):
    client = MinidsClient("https://example.invalid", "token")
    job_id = "a" * 32
    calls = []

    def fake_request(method, path, *args, **kwargs):
        calls.append((method, path))
        if method == "POST":
            raise MinidsHTTPError(524, "proxy timeout")
        return {"status": "starting"}

    monkeypatch.setattr(client, "_request", fake_request)

    assert client.start(job_id)["status"] == "starting"
    assert calls == [("POST", f"/jobs/{job_id}/start"), ("GET", f"/jobs/{job_id}?logs=false")]


def test_new_log_lines_handles_a_sliding_fixed_size_window():
    previous = [f"log {index}" for index in range(40)]
    current = [f"log {index}" for index in range(1, 41)]

    assert new_log_lines(previous, current) == ["log 40"]


def test_new_log_lines_uses_server_offset_when_entries_repeat():
    previous = ["même ligne"] * 40
    current = ["même ligne"] * 40

    assert new_log_lines(previous, current, previous_end=40, current_offset=1) == ["même ligne"]
