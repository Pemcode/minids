"""Le diagnostic Hugging Face doit rester utile sans divulguer le jeton."""

from __future__ import annotations

from scripts import check_access


def test_check_repo_ignores_malformed_sibling_entries(monkeypatch):
    monkeypatch.setattr(
        check_access,
        "request",
        lambda _path, _token: (
            200,
            {"siblings": [{"rfilename": "model_512.pt"}, {}, "bad", {"rfilename": 42}]},
        ),
    )
    assert check_access.check_repo("owner/model", "token") == ["model_512.pt"]


def test_pick_checkpoint_is_deterministic_and_prefers_512():
    assert check_access.pick_checkpoint(["z_256.pt", "b_512.safetensors", "a_512.pt"]) == "a_512.pt"


def test_main_never_prints_the_complete_token(monkeypatch, capsys):
    token = "hf_super_secret_1234"
    monkeypatch.setattr(check_access, "check_identity", lambda _token: True)
    monkeypatch.setattr(check_access, "check_repo", lambda _repo, _token: ["model_512.pt"])

    monkeypatch.setattr("sys.argv", ["check_access.py", "--token", token])
    assert check_access.main() == 0
    output = capsys.readouterr().out
    assert token not in output
    assert "hf_…1234" in output


def test_short_tokens_are_not_echoed_verbatim():
    assert check_access._mask_token("abc") != "abc"
