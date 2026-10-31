"""Real application/CLI paths over an atomic offline Firestore wire server."""

from io import BytesIO
import json
from pathlib import Path
import shutil

import pytest

from src import state_store as state
from tests.test_publication_atomic_commit import WireFirestore

PREFIX = "staging_123456_1_application_"
ROOT = Path(__file__).resolve().parents[1]


class ApplicationWire(WireFirestore):
    def __init__(self, monkeypatch):
        super().__init__(monkeypatch, {})
        self.touched = []
        self.reject_cleanup = False
        self.cleanup_response_error = None
        self.delete_batches = 0

    def urlopen(self, request, **kwargs):
        path = request.full_url.split("/documents", 1)[1]
        self.touched.append((request.method, path))
        if request.method == "PATCH":
            key = path[1:]
            self.docs[key] = {
                k: state._firestore_value_to_py(v)
                for k, v in json.loads(request.data)["fields"].items()
            }
            self.versions[key] = self.versions.get(key, 0) + 1
            return Response(b"{}")
        if request.method == "POST":
            writes = json.loads(request.data)["writes"]
            if writes and all("delete" in write for write in writes):
                self.delete_batches += 1
                if self.reject_cleanup:
                    raise state.PublicationOutcomeUnknown("cleanup response unknown")
                keys = []
                for write in writes:
                    key = write["delete"].split("/documents/", 1)[1]
                    assert write["currentDocument"] == {
                        "updateTime": str(self.versions[key])
                    }
                    keys.append(key)
                for key in keys:
                    del self.docs[key]
                    del self.versions[key]
                response = Response(
                    json.dumps(
                        {"commitTime": "now", "writeResults": [{} for _ in writes]}
                    ).encode()
                )
                if self.cleanup_response_error:

                    def failed_read(*args):
                        raise self.cleanup_response_error

                    response.read = failed_read
                return response
        return super().urlopen(request, **kwargs)


class Response(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def setup(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    (root / "config").mkdir(parents=True)
    (root / "config/orbit_sources.json").write_text(
        '{"active_interjection":{"enabled":false}}'
    )
    shutil.copy2(ROOT / "PROJECTS.md", root / "PROJECTS.md")
    wire = ApplicationWire(monkeypatch)

    def factory(**kwargs):
        return state.FirestoreStateStore(client=wire.client, **kwargs)

    return root, wire, factory


def test_application_rehearsal_uses_real_publication_cli_cache_and_conditional_cleanup(
    tmp_path, monkeypatch
):
    from scripts.rehearse_staging_application import rehearse

    root, wire, factory = setup(tmp_path, monkeypatch)
    result = rehearse(
        root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
    )
    assert result["status"] == "ok"
    assert result["backend"] == "firestore"
    assert result["publication"]["canonical_verified"]
    assert result["publication"]["explicit_history_feed_ids"]
    assert result["interactions"] == {
        "accepted_move": True,
        "bot_response": True,
        "stale_rejected": True,
        "issue_replay_rejected": True,
        "vote_counted_once": True,
    }
    assert result["cache"]["fixture_items_verified"] == 501
    assert result["cache"]["fixture_page_count"] >= 2
    assert result["remaining_document_count"] == 0
    assert wire.docs == {}
    assert all("spend_control" not in path for _, path in wire.touched)


@pytest.mark.parametrize(
    "prefix",
    [
        "",
        "staging_",
        "prod_",
        "staging_123_1_",
        "staging_123_0_application_",
        "staging_123_1_application_../",
    ],
)
def test_invalid_application_namespace_never_reaches_store(tmp_path, prefix):
    from scripts.rehearse_staging_application import rehearse

    def forbidden(**kwargs):
        pytest.fail("Invalid namespace must fail before storage")

    with pytest.raises(ValueError, match="namespace"):
        rehearse(tmp_path, prefix, store_factory=forbidden)


def test_existing_application_namespace_is_never_modified(tmp_path, monkeypatch):
    from scripts.rehearse_staging_application import rehearse

    root, wire, factory = setup(tmp_path, monkeypatch)
    wire.docs[PREFIX + "ingestion_cache/keep"] = {"date": "2026-10-30"}
    wire.versions[PREFIX + "ingestion_cache/keep"] = 1
    with pytest.raises(ValueError, match="occupied"):
        rehearse(root, PREFIX, store_factory=factory, private_root=tmp_path / "private")
    assert wire.docs == {PREFIX + "ingestion_cache/keep": {"date": "2026-10-30"}}
    assert not any(method in {"PATCH", "POST", "DELETE"} for method, _ in wire.touched)


def test_ambiguous_application_commit_retains_namespace_and_private_journal(
    tmp_path, monkeypatch
):
    from scripts.rehearse_staging_application import rehearse, ApplicationRehearsalError

    root, wire, factory = setup(tmp_path, monkeypatch)
    from http.client import IncompleteRead

    wire.response_error = IncompleteRead(b"private partial", 1)
    with pytest.raises(ApplicationRehearsalError) as error:
        rehearse(root, PREFIX, store_factory=factory, private_root=tmp_path / "private")
    assert error.value.report["namespace_retained"] is True
    assert wire.docs
    assert Path(error.value.report["private_journal_dir"]).is_dir()
    assert "private partial" not in json.dumps(error.value.report)
    assert not any(method == "DELETE" for method, _ in wire.touched)


def test_failed_conditional_cleanup_keeps_recorded_namespace_for_reconciliation(
    tmp_path, monkeypatch
):
    from scripts.rehearse_staging_application import rehearse, ApplicationRehearsalError

    root, wire, factory = setup(tmp_path, monkeypatch)
    wire.reject_cleanup = True
    with pytest.raises(ApplicationRehearsalError) as failure:
        rehearse(root, PREFIX, store_factory=factory, private_root=tmp_path / "private")
    report = failure.value.report
    assert report["failed_phase"] == "conditional_owned_cleanup"
    assert report["namespace_retained"] is None and wire.docs
    assert (
        report["cleanup_attempted"] is True and report["cleanup_outcome"] == "unknown"
    )
    journal = json.loads(
        (Path(report["private_journal_dir"]) / "application-journal.json").read_text()
    )
    assert journal["phase"] == "known_success_owned_cleanup_pending"
    assert set(journal["documents"]) == {
        path.removeprefix(PREFIX) for path in wire.docs
    }


def test_canonical_target_mismatch_blocks_before_read_or_write(tmp_path, monkeypatch):
    from scripts.rehearse_staging_application import rehearse

    root, wire, factory = setup(tmp_path, monkeypatch)

    def wrong(**kwargs):
        store = factory(**kwargs)
        store._database_id = "another-database"
        return store

    with pytest.raises(ValueError, match="target differs"):
        rehearse(root, PREFIX, store_factory=wrong)
    assert wire.touched == []


def test_json_fallback_store_is_rejected_before_any_application_call(
    tmp_path, monkeypatch
):
    from scripts.rehearse_staging_application import rehearse

    root, wire, factory = setup(tmp_path, monkeypatch)

    def fallback(**kwargs):
        store = factory(**kwargs)
        store._fallback_store = state.JsonFileStateStore(root)
        return store

    with pytest.raises(ValueError, match="canonical-only"):
        rehearse(root, PREFIX, store_factory=fallback)
    assert wire.touched == []


def test_owned_cleanup_cannot_target_production_or_spending_collections(
    tmp_path, monkeypatch
):
    from scripts.rehearse_staging_application import _delete_owned

    root, wire, factory = setup(tmp_path, monkeypatch)
    store = factory(project_id="git-readme-features", collection_prefix="")
    with pytest.raises(ValueError, match="target differs"):
        _delete_owned(store, {})
    store.collection_prefix = PREFIX
    snapshot = state._FirestoreRestDocSnapshot("2026-10", True, {}, update_time="1")
    with pytest.raises(RuntimeError, match="safe identity"):
        _delete_owned(store, {"spend_control/2026-10": snapshot})
    assert wire.touched == []


def test_workflow_uses_separate_application_namespace_and_sanitized_artifacts():
    text = (ROOT / ".github/workflows/staging_zeitgeist_dryrun.yml").read_text()
    assert "scripts/rehearse_staging_application.py" in text
    assert (
        "staging_"
        + chr(36)
        + "{{ github.run_id }}_"
        + chr(36)
        + "{{ github.run_attempt }}_application_"
        in text
    )
    artifact = text.split("name: Upload isolated staging", 1)[1]
    assert "/tmp/zeitgeist-staging-state/application.json" in artifact
    assert (
        "private-staging" not in artifact and "application-journal.json" not in artifact
    )


def test_create_only_bootstrap_is_journaled_before_dispatch(tmp_path, monkeypatch):
    from scripts.rehearse_staging_application import rehearse

    root, wire, factory = setup(tmp_path, monkeypatch)

    def inspect(server):
        plan = json.loads((tmp_path / "private/create-only-bootstrap.json").read_text())
        assert set(plan["expected_absent"]) == set(plan["desired_documents"])
        assert plan["desired_documents"]["profile_state/daily_pulse"][
            "visitor_mood_tally"
        ] == {"spot_on": 0, "too_sunny": 0, "too_heavy": 0}
        assert plan["desired_documents"]["profile_state/elo_ledger"]["players"] == {}

    wire.before_commit = inspect
    assert (
        rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
        )["status"]
        == "ok"
    )


def test_real_cli_rehearsal_restores_owner_salt_and_prevents_issue_side_effects(
    tmp_path, monkeypatch
):
    import os
    from scripts.rehearse_staging_application import rehearse
    from scripts import process_visitor_event

    root, wire, factory = setup(tmp_path, monkeypatch)
    monkeypatch.setenv("PULSE_VOTE_SALT", "private production salt")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("ISSUE_NUMBER", "91919")
    monkeypatch.setattr(
        process_visitor_event.subprocess,
        "run",
        lambda *a, **k: pytest.fail("No Issue or Git subprocess may be sent"),
    )
    result = rehearse(
        root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
    )
    assert result["status"] == "ok"
    assert os.environ["PULSE_VOTE_SALT"] == "private production salt"
    assert os.environ["GITHUB_ACTIONS"] == "true"
    assert "private production salt" not in json.dumps(result)


def test_empty_success_does_not_delete_other_namespaces_or_global_budget(
    tmp_path, monkeypatch
):
    from scripts.rehearse_staging_application import rehearse

    root, wire, factory = setup(tmp_path, monkeypatch)
    external = {
        "profile_state/daily_pulse": {"word": "PRODUCTION"},
        "spend_control/2026-10": {"cash_ceiling": "2.50"},
        "staging_123456_1_profile_state/daily_pulse": {"word": "MIGRATION"},
    }
    wire.docs.update(external)
    wire.versions.update({key: 1 for key in external})
    assert (
        rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
        )["remaining_document_count"]
        == 0
    )
    assert wire.docs == external


def test_cleanup_cas_race_does_not_overwrite_modified_owned_documents(
    tmp_path, monkeypatch
):
    from scripts import rehearse_staging_application as app

    root, wire, factory = setup(tmp_path, monkeypatch)
    original = app._delete_owned

    def race(store, snapshots):
        key = PREFIX + "profile_state/daily_pulse"
        wire.docs[key]["external_change"] = "retain"
        wire.versions[key] += 1
        # Real REST precondition failure is simulated by the wire's assertion.
        original(store, snapshots)

    monkeypatch.setattr(app, "_delete_owned", race)
    with pytest.raises(app.ApplicationRehearsalError) as failure:
        app.rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
        )
    assert (
        wire.docs[PREFIX + "profile_state/daily_pulse"]["external_change"] == "retain"
    )
    assert failure.value.report["cleanup_attempted"]


def test_lost_actual_publication_response_retains_application_namespace(
    tmp_path, monkeypatch
):
    from http.client import IncompleteRead
    from scripts.rehearse_staging_application import rehearse, ApplicationRehearsalError

    root, wire, factory = setup(tmp_path, monkeypatch)
    original = wire.urlopen

    def lost_publication(request, **kwargs):
        if request.method == "POST":
            writes = json.loads(request.data)["writes"]
            if any(
                "publication_id" in write.get("update", {}).get("fields", {})
                for write in writes
            ):
                wire.response_error = IncompleteRead(b"private provider response", 1)
        return original(request, **kwargs)

    monkeypatch.setattr(state.urllib.request, "urlopen", lost_publication)
    with pytest.raises(ApplicationRehearsalError) as failure:
        rehearse(root, PREFIX, store_factory=factory, private_root=tmp_path / "private")
    report = failure.value.report
    assert report["failed_phase"] == "actual_orchestrator"
    assert report["namespace_retained"] is True and not report["cleanup_attempted"]
    assert wire.docs[PREFIX + "profile_state/daily_pulse"]["publication_id"]
    assert (root / "data/publication_reconcile.json").exists()
    with pytest.raises(ValueError, match="occupied"):
        rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "other-private"
        )
    assert "private provider response" not in json.dumps(report)


def test_delete_response_lost_reports_unknown_presence_without_second_delete(
    tmp_path, monkeypatch
):
    from http.client import IncompleteRead
    from scripts.rehearse_staging_application import rehearse, ApplicationRehearsalError

    root, wire, factory = setup(tmp_path, monkeypatch)
    wire.cleanup_response_error = IncompleteRead(b"partial delete response", 1)
    with pytest.raises(ApplicationRehearsalError) as failure:
        rehearse(root, PREFIX, store_factory=factory, private_root=tmp_path / "private")
    assert wire.docs == {}
    assert failure.value.report["namespace_retained"] is None
    assert failure.value.report["cleanup_outcome"] == "unknown"
    # Exactly one delete batch: there is no blind retry after a lost response.
    assert wire.delete_batches == 1


def test_default_rehearsal_uses_actual_runtime_factory_and_canonical_only_mirrors(
    tmp_path, monkeypatch
):
    from scripts import rehearse_staging_application as app

    root, wire, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        state.FirestoreStateStore, "_get_client", lambda self: wire.client
    )
    calls, adapters = [], []

    def runtime_factory(**kwargs):
        calls.append(kwargs)
        adapter = state.get_state_store(**kwargs, firestore_client=wire.client)
        assert adapter.canonical_only is True and adapter._fallback_store is None
        assert adapter._mirror_store.repo_root == root
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(app, "get_state_store", runtime_factory, raising=False)
    # These deployment-environment values must not redirect the fixed staging target.
    monkeypatch.setenv("GCP_PROJECT_ID", "another-project")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "another-project")
    monkeypatch.setenv("FIRESTORE_DATABASE_ID", "another-database")
    report = app.rehearse(root, PREFIX, private_root=tmp_path / "private")
    assert calls == [
        {"repo_root": root, "backend": "firestore", "collection_prefix": PREFIX}
    ]
    assert report["status"] == "ok" and report["remaining_document_count"] == 0
    assert wire.docs == {} and wire.delete_batches == 1
    assert adapters[0]._mirror_store.repo_root == root


def test_default_runtime_factory_cannot_return_legacy_semantics_even_without_json_fallback(
    tmp_path, monkeypatch
):
    from scripts import rehearse_staging_application as app

    root, wire, factory = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        state.FirestoreStateStore, "_get_client", lambda self: wire.client
    )
    legacy = factory(
        project_id="git-readme-features",
        database_id="(default)",
        collection_prefix=PREFIX,
    )
    monkeypatch.setattr(app, "get_state_store", lambda **kwargs: legacy, raising=False)
    with pytest.raises(ValueError, match="runtime factory"):
        app.rehearse(root, PREFIX, private_root=tmp_path / "private")
    assert not wire.touched


def test_default_runtime_factory_still_checks_empty_owned_namespace_before_bootstrap(
    tmp_path, monkeypatch
):
    from scripts import rehearse_staging_application as app

    root, wire, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        state.FirestoreStateStore, "_get_client", lambda self: wire.client
    )
    key = PREFIX + "ingestion_cache/keep"
    wire.docs[key], wire.versions[key] = {"date": "2026-10-30"}, 1

    def runtime_factory(**kwargs):
        return state.get_state_store(**kwargs, firestore_client=wire.client)

    monkeypatch.setattr(app, "get_state_store", runtime_factory, raising=False)
    with pytest.raises(ValueError, match="occupied"):
        app.rehearse(root, PREFIX, private_root=tmp_path / "private")
    assert wire.docs == {key: {"date": "2026-10-30"}}
    assert not any(method in {"POST", "PATCH", "DELETE"} for method, _ in wire.touched)


@pytest.mark.parametrize(
    "change",
    ["project", "database", "prefix", "fallback", "mirror_root", "mirror_missing"],
)
def test_default_runtime_factory_rejects_target_or_authority_mismatch_before_storage(
    tmp_path, monkeypatch, change
):
    from scripts import rehearse_staging_application as app

    root, wire, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        state.FirestoreStateStore, "_get_client", lambda self: wire.client
    )

    def runtime_factory(**kwargs):
        adapter = state.get_state_store(**kwargs, firestore_client=wire.client)
        if change == "project":
            adapter._project_id = "another-project"
        elif change == "database":
            adapter._database_id = "another-database"
        elif change == "prefix":
            adapter.collection_prefix = "staging_999_1_application_"
        elif change == "fallback":
            adapter._fallback_store = state.JsonFileStateStore(root)
        elif change == "mirror_root":
            adapter._mirror_store = state.JsonFileStateStore(tmp_path / "other")
        else:
            adapter._mirror_store = None
        return adapter

    monkeypatch.setattr(app, "get_state_store", runtime_factory)
    with pytest.raises(ValueError):
        app.rehearse(root, PREFIX, private_root=tmp_path / "private")
    assert wire.touched == [] and wire.docs == {}
