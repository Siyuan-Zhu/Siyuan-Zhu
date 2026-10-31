"""Release acceptance at the visitor CLI and publication boundaries."""
import shutil
from pathlib import Path
from unittest.mock import patch

from scripts import process_visitor_event
from src.state_store import InMemoryStateStore, git_commit_and_push_with_retry


ROOT = Path(__file__).resolve().parents[1]


def test_staging_success_assertion_rejects_an_unaccepted_move(tmp_path):
    for name in ("data", "assets", "config"):
        shutil.copytree(ROOT / name, tmp_path / name)
    assert process_visitor_event.main([
        "--repo-root", str(tmp_path), "--login", "staging-bot",
        "--title", "connect4: drop 4", "--require-accepted",
    ]) != 0


def test_publication_command_fails_when_git_publication_is_exhausted():
    from scripts import publish_readme
    with patch.object(publish_readme, "git_commit_and_push_with_retry", return_value=False):
        assert publish_readme.main(["--message", "test publication"]) != 0


def test_publication_command_succeeds_after_git_publication():
    from scripts import publish_readme
    with patch.object(publish_readme, "git_commit_and_push_with_retry", return_value=True):
        assert publish_readme.main(["--message", "test publication"]) == 0


def test_incomplete_writer_configuration_cannot_silently_use_local_state():
    from scripts.check_runtime_configuration import configuration_errors
    assert configuration_errors({}, visitor=True)
    env = {
        "GITHUB_REPOSITORY": "Siyuan-Zhu/readme-live-sandbox",
        "TARGET_REPO_SLUG": "Siyuan-Zhu/readme-live-sandbox",
        "STATE_STORE_BACKEND": "firestore", "FIRESTORE_DATABASE_ID": "(default)",
        "GCP_WIF_PROVIDER": "//iam.googleapis.com/projects/277213321965/locations/global/workloadIdentityPools/github-pool/providers/github-provider",
        "PULSE_VOTE_SALT": "test-fixture-only",
        "FIRESTORE_COLLECTION_PREFIX": "staging_release_",
        "GCP_PROJECT_ID": "git-readme-features",
    }
    assert configuration_errors(env, visitor=True) == []
    env["TARGET_REPO_SLUG"] = "Siyuan-Zhu/Siyuan-Zhu"
    assert configuration_errors(env, visitor=True)


def test_failed_git_commit_cannot_be_reported_as_successful_publication(tmp_path):
    from types import SimpleNamespace
    commands = []
    def git(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=1 if command[1] in ("commit", "diff") else 0)
    store = InMemoryStateStore()
    with patch("subprocess.run", side_effect=git):
        result = git_commit_and_push_with_retry("test", paths=["README.md"],
            repo_root=tmp_path, store=store, sleep_fn=lambda _: None)
    assert result is False
    assert not any(command[1] == "push" for command in commands)
    assert store.needs_git_reconcile is True


def test_sandbox_writer_must_never_use_unprefixed_production_state():
    from scripts.check_runtime_configuration import configuration_errors
    env = {"GITHUB_REPOSITORY": "Siyuan-Zhu/readme-live-sandbox",
        "TARGET_REPO_SLUG": "Siyuan-Zhu/readme-live-sandbox",
        "STATE_STORE_BACKEND": "firestore", "FIRESTORE_DATABASE_ID": "(default)",
        "GCP_PROJECT_ID": "git-readme-features", "FIRESTORE_COLLECTION_PREFIX": "",
        "GCP_WIF_PROVIDER": "//iam.googleapis.com/projects/277213321965/locations/global/workloadIdentityPools/github-pool/providers/github-provider"}
    assert configuration_errors(env)


def test_recovery_restores_publication_manifest_and_bundled_font_bytes(tmp_path):
    from scripts.migrate_release_state import create_release_backup, rollback_release_state
    root = tmp_path / "repo"
    (root / "assets/fonts").mkdir(parents=True)
    manifest = root / "assets/pulse_manifest.json"
    font = root / "assets/fonts/CormorantGaramond.ttf"
    manifest.write_text('{"publication_id":"before"}')
    font.write_bytes(b"original-font")
    create_release_backup(root, tmp_path / "backup")
    manifest.write_text('{"publication_id":"after"}')
    font.write_bytes(b"changed-font")
    rollback_release_state(root, tmp_path / "backup")
    assert manifest.read_text() == '{"publication_id":"before"}'
    assert font.read_bytes() == b"original-font"


def test_recovery_removes_publication_manifest_absent_before_cutover(tmp_path):
    from scripts.migrate_release_state import create_release_backup, rollback_release_state
    root = tmp_path / "repo"
    (root / "assets").mkdir(parents=True)
    create_release_backup(root, tmp_path / "backup")
    manifest = root / "assets/pulse_manifest.json"
    manifest.write_text('{"publication_id":"created-after-backup"}')
    rollback_release_state(root, tmp_path / "backup")
    assert not manifest.exists()


def test_migration_preserves_actual_generated_image_medium(tmp_path):
    import json
    from scripts.migrate_release_state import migrate_release_state
    (tmp_path / "data").mkdir()
    for medium in ("IMAGEN_3", "FLUX_PRO"):
        (tmp_path / "data/zeitgeist_state.json").write_text(json.dumps({
            "date": "2026-10-29", "word": "Counterpoise", "visual_medium": medium,
        }))
        report = migrate_release_state(tmp_path)
        assert report["documents"]["profile_state"]["daily_pulse"]["visual_medium"] == medium


def test_offline_publication_never_uses_paid_keys_in_the_runner(tmp_path, monkeypatch):
    from src.zeitgeist.spend_guard import SpendBlocked
    from src.zeitgeist.visual_router import select_and_render_pulse_assets
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-do-not-call")
    monkeypatch.setenv("BFL_API_KEY", "test-key-do-not-call")
    with patch("src.zeitgeist.spend_guard.get_spend_guard", side_effect=SpendBlocked("Offline request reached provider")):
        pulse = select_and_render_pulse_assets(
            {"word": "Counterpoise", "gloss": "A steady balance."},
            repo_root=tmp_path, candidates=[],
        )
    assert pulse.visual_medium == "BESPOKE_SVG"


def test_flux_unpriced_dimensions_cannot_reach_a_provider(monkeypatch):
    import pytest
    from src.zeitgeist import visual_router
    from src.zeitgeist.spend_guard import SpendBlocked
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    with patch("httpx.Client") as client, pytest.raises(SpendBlocked, match="dimensions"):
        visual_router._generate_live_flux_bytes("A workshop", api_key="fixture", width=2048, height=2048)
    client.assert_not_called()


def test_flux_override_credential_uses_only_its_verified_prepaid_balance(monkeypatch):
    import hashlib
    from types import SimpleNamespace
    from datetime import datetime
    from tests.test_spend_provisioning import candidate, NOW
    from src.zeitgeist import spend_guard, visual_router
    class FrozenClock(datetime):
        @classmethod
        def now(cls, tz=None): return NOW
    monkeypatch.setattr(spend_guard, "datetime", FrozenClock)
    key = "fixture-override-key"
    document = candidate()
    document["providers"]["bfl"]["credential_fingerprint"] = hashlib.sha256(key.encode()).hexdigest()[:16]
    ledger = spend_guard.MemoryLedger(document)
    monkeypatch.setattr(spend_guard, "FirestoreLedger", lambda *args: ledger)
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("BFL_API_KEY", "different-environment-key")
    class Http:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, *, headers, json):
            assert headers["x-key"] == key
            return SimpleNamespace(status_code=200, json=lambda: {})
    monkeypatch.setattr("httpx.Client", lambda **kwargs: Http())
    assert visual_router._generate_live_flux_bytes("A workshop", api_key=key) is None
    assert len(ledger.document["reservations"]) == 1
    assert next(iter(ledger.document["reservations"].values()))["provider"] == "bfl"
