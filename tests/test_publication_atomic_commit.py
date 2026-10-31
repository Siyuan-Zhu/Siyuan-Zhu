"""Canonical publication writes must be conditional and indivisible."""

import io
import json
from urllib.error import HTTPError

import pytest

from src import state_store as stores


def test_rest_commit_has_all_write_preconditions_and_no_patch(monkeypatch):
    requests = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def urlopen(request, **kwargs):
        requests.append(request)
        return Response(b'{"commitTime":"now","writeResults":[{"updateTime":"now"},{"updateTime":"now"}]}')

    monkeypatch.setattr(stores.urllib.request, "urlopen", urlopen)
    client = stores._FirestoreRestClient("fixture", access_token="private-token")
    current = stores._FirestoreRestDocSnapshot(
        "daily_pulse", True, {"word": "OLD"}, update_time="2026-10-30T00:00:00Z"
    )
    absent = stores._FirestoreRestDocSnapshot("2026-10-30", False)
    stores._commit_documents(
        client,
        [
            ("staging_profile_state", "daily_pulse", {"word": "NEW"}, current),
            ("staging_pulse_history", "2026-10-30", {"word": "NEW"}, absent),
        ],
    )
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].full_url.endswith("/documents:commit")
    writes = json.loads(requests[0].data)["writes"]
    assert writes[0]["currentDocument"] == {"updateTime": current.update_time}
    assert writes[1]["currentDocument"] == {"exists": False}
    assert writes[0]["updateMask"] == {"fieldPaths": ["word"]}


def test_rest_commit_precondition_failure_is_retriable_without_partial_fallback(
    monkeypatch,
):
    def fail(request, **kwargs):
        raise HTTPError(
            request.full_url, 412, "precondition", {}, io.BytesIO(b"private")
        )

    monkeypatch.setattr(stores.urllib.request, "urlopen", fail)
    client = stores._FirestoreRestClient("fixture", access_token="private-token")
    with pytest.raises(stores.PublicationConflict):
        stores._commit_documents(
            client,
            [
                (
                    "pulse_feed",
                    "v1",
                    {"entries": []},
                    stores._FirestoreRestDocSnapshot("v1", False),
                )
            ],
        )


def test_existing_document_without_revision_fails_closed():
    client = stores._FirestoreRestClient("fixture", access_token="private-token")
    with pytest.raises(RuntimeError, match="update_time"):
        stores._commit_documents(
            client,
            [("pulse_feed", "v1", {}, stores._FirestoreRestDocSnapshot("v1", True))],
        )


class WireFirestore:
    """Exercise the real REST encoding/transport with an atomic mock server."""

    def __init__(self, monkeypatch, initial):
        import copy

        self.docs = copy.deepcopy(initial)
        self.versions = {path: 1 for path in self.docs}
        self.commits = []
        self.before_commit = None
        self.reject = None
        self.response_error = None
        self.response_bytes = None
        self.response_close_error = None
        monkeypatch.setattr(stores.urllib.request, "urlopen", self.urlopen)
        self.client = stores._FirestoreRestClient(
            "fixture", access_token="private-token"
        )

    def urlopen(self, request, **kwargs):
        import copy

        path = request.full_url.split("/documents", 1)[1]
        if request.method == "GET":
            path = path[1:]
            if "/" not in path:
                collection = path.split("?", 1)[0]
                body = {"documents": [{
                    "name": "projects/fixture/databases/(default)/documents/" + key,
                    "fields": {k: stores._py_to_firestore_value(v, in_array=False) for k, v in data.items()},
                    "updateTime": str(self.versions[key]),
                } for key, data in self.docs.items() if key.startswith(collection + "/")]}
            else:
                if path not in self.docs:
                    raise HTTPError(request.full_url, 404, "missing", {}, io.BytesIO())
                body = {"fields": {k: stores._py_to_firestore_value(v, in_array=False) for k, v in self.docs[path].items()},
                        "updateTime": str(self.versions[path])}
        else:
            assert request.method == "POST" and path == ":commit"
            body = json.loads(request.data)
            self.commits.append(copy.deepcopy(body))
            if self.before_commit:
                callback, self.before_commit = self.before_commit, None
                callback(self)
            if self.reject:
                raise HTTPError(
                    request.full_url,
                    self.reject,
                    "rejected",
                    {},
                    io.BytesIO(b"private"),
                )
            decoded = []
            for write in body["writes"]:
                key = write["update"]["name"].split("/documents/", 1)[1]
                condition = write["currentDocument"]
                if (condition.get("exists") is False and key in self.docs) or (
                    "updateTime" in condition
                    and str(self.versions.get(key)) != condition["updateTime"]
                ):
                    raise HTTPError(request.full_url, 412, "conflict", {}, io.BytesIO())
                decoded.append(
                    (
                        key,
                        {
                            k: stores._firestore_value_to_py(v)
                            for k, v in write["update"]["fields"].items()
                        },
                    )
                )
            for key, data in decoded:
                self.docs[key] = dict(self.docs.get(key, {})) | data
                self.versions[key] = self.versions.get(key, 0) + 1
            body = {"commitTime": "now", "writeResults": [{"updateTime": "now"} for _ in decoded]}

        response_error = self.response_error if request.method == "POST" else None
        response_bytes = self.response_bytes if request.method == "POST" else None
        response_close_error = self.response_close_error if request.method == "POST" else None

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.close()
                if response_close_error is not None:
                    raise response_close_error

            def read(self, *args):
                if response_error is not None:
                    raise response_error
                return super().read(*args)

        return Response(response_bytes if response_bytes is not None else json.dumps(body).encode())


def _publication_fixture(tmp_path, monkeypatch):
    import copy
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator

    monkeypatch.setenv("STATE_STORE_BACKEND", "json")
    monkeypatch.setenv("ZEITGEIST_OFFLINE_DEMO", "1")
    monkeypatch.setenv("ZEITGEIST_LIVE_UNSPLASH", "0")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "0")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text(
        '{"active_interjection":{"enabled":false}}'
    )
    ZeitgeistADKOrchestrator(tmp_path).run(
        "2026-10-29", prefer_visual_mode="blueprint", offline_demo=True
    )
    old = json.loads((tmp_path / "data/zeitgeist_state.json").read_text())
    feed = json.loads((tmp_path / "data/pulse_feed.json").read_text())
    from src.game_engine import create_initial_game_state
    initial = {
        "staging_profile_state/active_game": create_initial_game_state(),
        "staging_profile_state/elo_ledger": {"state_version": 0, "players": {}, "recent_rounds": []},
        "staging_profile_state/daily_pulse": copy.deepcopy(old),
        "staging_pulse_history/2026-10-29": old["published_pulse"],
        "staging_pulse_feed/v1": feed,
        "staging_pulse_votes/2026-10-30": {
            "spot_on": 30,
            "too_sunny": 3,
            "too_heavy": 4,
        },
    }
    wire = WireFirestore(monkeypatch, initial)
    store = stores.FirestoreStateStore(
        client=wire.client, collection_prefix="staging_", repo_root=tmp_path
    )
    return wire, store


def _published_files(root):
    return {
        str(p.relative_to(root)): p.read_bytes()
        for pattern in (
            "README.md",
            "variants/*.md",
            "assets/**/*",
            "data/zeitgeist_state.json",
            "data/pulse_feed.json",
        )
        for p in root.glob(pattern)
        if p.is_file()
    }


@pytest.mark.parametrize("boundary", ["readme", "canonical_commit"])
def test_real_orchestrator_failure_preserves_canonical_and_local_publication(
    tmp_path, monkeypatch, boundary
):
    import copy
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from scripts import build_readme

    wire, store = _publication_fixture(tmp_path, monkeypatch)
    before_remote = copy.deepcopy(wire.docs)
    before_local = _published_files(tmp_path)
    if boundary == "readme":

        def fail(*args, **kwargs):
            raise ValueError("README invalid")

        monkeypatch.setattr(build_readme, "build_all_readmes", fail)
    else:
        wire.reject = 403
    with pytest.raises((ValueError, RuntimeError)):
        ZeitgeistADKOrchestrator(tmp_path, store).run(
            "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
        )
    assert wire.docs == before_remote
    assert _published_files(tmp_path) == before_local
    assert len(wire.commits) == (0 if boundary == "readme" else 1)


def test_real_orchestrator_retries_vote_conflict_and_commits_one_coherent_bundle(
    tmp_path, monkeypatch
):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from src.zeitgeist.publication import verify_publication_bundle

    wire, store = _publication_fixture(tmp_path, monkeypatch)

    def vote(server):
        key = "staging_pulse_votes/2026-10-30"
        server.docs[key]["spot_on"] = 31
        server.versions[key] += 1

    wire.before_commit = vote
    orchestrator = ZeitgeistADKOrchestrator(tmp_path, store)
    orchestrator.run("2026-10-30", prefer_visual_mode="blueprint", offline_demo=True)
    assert len(wire.commits) == 2
    for key in (
        "staging_profile_state/daily_pulse",
        "staging_pulse_history/2026-10-30",
    ):
        document = wire.docs[key]
        assert document["word"] == orchestrator.last_published_pulse.word
        assert document["mood_votes"]["spot_on"] == 31
        assert document["publication_id"] == wire.docs["staging_profile_state/daily_pulse"]["publication_id"]
    assert (
        wire.docs["staging_pulse_feed/v1"]["entries"][0]["mood_votes"]["spot_on"] == 31
    )
    assert wire.docs["staging_pulse_feed/v1"]["entries"][0]["publication_id"] == wire.docs["staging_profile_state/daily_pulse"]["publication_id"]
    assert "Spot On 31" in (tmp_path / "README.md").read_text()
    assert verify_publication_bundle(tmp_path, require_current_pulse=True)
    assert wire.docs["staging_profile_state/daily_pulse"]["schema_version"] == 3
    assert json.loads((tmp_path / "data/zeitgeist_state.json").read_text())["schema_version"] == 3
    assert store.needs_git_reconcile


def test_stale_editorial_candidate_cannot_overwrite_newer_publication(
    tmp_path, monkeypatch
):
    from src.zeitgeist.publication_commit import publish_bundle

    wire, store = _publication_fixture(tmp_path, monkeypatch)
    old = store.load_pulse_state()
    wire.docs["staging_profile_state/daily_pulse"][
        "publication_id"
    ] = "newer-publication"
    wire.versions["staging_profile_state/daily_pulse"] += 1
    with pytest.raises(stores.PublicationConflict, match="newer editorial"):
        publish_bundle(
            store,
            old,
            old["published_pulse"],
            expected=old,
            local_feed={"entries": []},
            prepare=lambda *args: pytest.fail(
                "Stale candidate must fail before prepare"
            ),
        )
    assert not wire.commits


@pytest.mark.parametrize("entrypoint", ["daily_cli", "manual"])
def test_readme_failure_gates_both_entrypoints_before_remote_save(
    tmp_path, monkeypatch, entrypoint
):
    import copy
    from argparse import Namespace
    from scripts import build_readme, orbit, run_zeitgeist

    wire, store = _publication_fixture(tmp_path, monkeypatch)
    before_remote, before_local = copy.deepcopy(wire.docs), _published_files(tmp_path)
    before_config = (tmp_path / "config/orbit_sources.json").read_bytes()
    monkeypatch.setattr(stores, "get_state_store", lambda **kwargs: store)
    monkeypatch.setattr(run_zeitgeist, "get_state_store", lambda **kwargs: store)
    # Avoid the separate CLI startup reconciliation in this isolated wire fixture.
    monkeypatch.setattr(store, "reconcile_to_git", lambda: False)

    def fail(*args, **kwargs):
        raise ValueError("README invalid")

    monkeypatch.setattr(build_readme, "build_all_readmes", fail)
    with pytest.raises(ValueError, match="README invalid"):
        if entrypoint == "daily_cli":
            run_zeitgeist.main(
                [
                    "--repo-root",
                    str(tmp_path),
                    "--date",
                    "2026-10-30",
                    "--prefer-visual-mode",
                    "blueprint",
                    "--skip-readme",
                ]
            )
        else:
            orbit.cmd_interject(
                tmp_path,
                Namespace(
                    word="ANCHORED",
                    phrase="Steady craft.",
                    mode="blueprint",
                    image_url=None,
                    note=None,
                    channel="test",
                ),
            )
    assert wire.docs == before_remote
    assert _published_files(tmp_path) == before_local
    assert (tmp_path / "config/orbit_sources.json").read_bytes() == before_config
    assert not wire.commits


def test_post_commit_install_failure_retains_bundle_without_remote_compensation(
    tmp_path, monkeypatch
):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from src.zeitgeist.publication_commit import (
        PublicationWorkspace,
        PublicationMirrorError,
    )

    wire, store = _publication_fixture(tmp_path, monkeypatch)

    def fail(*args):
        raise OSError("disk failed")

    monkeypatch.setattr(PublicationWorkspace, "install", fail)
    with pytest.raises(PublicationMirrorError):
        ZeitgeistADKOrchestrator(tmp_path, store).run(
            "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
        )
    assert len(wire.commits) == 1
    assert wire.docs["staging_profile_state/daily_pulse"]["date"] == "2026-10-30"
    marker = json.loads((tmp_path / "data/publication_reconcile.json").read_text())
    assert marker["status"] == "canonical_committed_local_install_failed"
    recovery = __import__("pathlib").Path(marker["recovery_path"])
    assert (recovery / "assets/pulse_manifest.json").exists()
    assert not (recovery / "config").exists()
    assert store.needs_git_reconcile


def test_ambiguous_commit_failure_is_not_retried_or_compensated(tmp_path, monkeypatch):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from src.zeitgeist.publication_commit import PublicationMirrorError

    wire, store = _publication_fixture(tmp_path, monkeypatch)
    wire.reject = 503
    with pytest.raises(PublicationMirrorError):
        ZeitgeistADKOrchestrator(tmp_path, store).run(
            "2026-10-30", prefer_visual_mode="blueprint", offline_demo=True
        )
    assert len(wire.commits) == 1
    marker = json.loads((tmp_path / "data/publication_reconcile.json").read_text())
    assert marker["status"] == "canonical_commit_outcome_unknown"


def test_pending_unknown_publication_blocks_new_provider_work(tmp_path, monkeypatch):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from src.zeitgeist import adk_pipeline

    (tmp_path / "data").mkdir()
    (tmp_path / "data/publication_reconcile.json").write_text('{"status":"unknown"}')
    monkeypatch.setattr(
        adk_pipeline,
        "collect_96h_signals",
        lambda *a, **k: pytest.fail("No new provider work before reconciliation"),
    )
    with pytest.raises(RuntimeError, match="pending canonical"):
        ZeitgeistADKOrchestrator(tmp_path, stores.InMemoryStateStore()).run()


def test_recovery_journal_exists_before_any_canonical_dispatch(tmp_path, monkeypatch):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    def inspect(server):
        marker = json.loads((tmp_path / "data/publication_reconcile.json").read_text())
        assert marker["status"] == "canonical_commit_pending"
        assert (__import__("pathlib").Path(marker["recovery_path"]) / "README.md").exists()
    wire.before_commit = inspect
    ZeitgeistADKOrchestrator(tmp_path, store).run("2026-10-30", prefer_visual_mode="blueprint", offline_demo=True)
    assert not (tmp_path / "data/publication_reconcile.json").exists()


def test_sdk_batch_disables_transport_retries_and_preserves_preconditions(monkeypatch):
    from datetime import datetime, timezone
    from google.auth.credentials import AnonymousCredentials
    from google.cloud import firestore
    from types import SimpleNamespace
    client = firestore.Client(project="fixture", credentials=AnonymousCredentials())
    batch = client.batch()
    captured = {}
    def commit(**kwargs):
        captured.update(kwargs)
        captured["writes"] = list(batch._write_pbs)
        return [SimpleNamespace(update_time="2026-10-30T00:00:00Z") for _ in batch._write_pbs]
    monkeypatch.setattr(batch, "commit", commit)
    monkeypatch.setattr(client, "batch", lambda: batch)
    stores._commit_documents(client, [
        ("pulse_history", "2026-10-30", {"24h_shift": 1.0}, SimpleNamespace(exists=True, update_time=datetime(2026, 10, 30, tzinfo=timezone.utc))),
        ("pulse_feed", "v1", {"entries": []}, SimpleNamespace(exists=False)),
    ])
    assert captured["retry"] is None and captured["timeout"] == 15
    assert captured["writes"][0].update_mask.field_paths == ["`24h_shift`"]
    assert captured["writes"][0].current_document.update_time.year == 2026
    assert captured["writes"][1].current_document.exists is False


def test_missing_canonical_feed_bootstraps_remote_history_not_stale_json(tmp_path, monkeypatch):
    import copy
    from src.zeitgeist.publication_commit import publish_bundle
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    wire.docs.pop("staging_pulse_feed/v1")
    wire.versions.pop("staging_pulse_feed/v1")
    remote_entry = copy.deepcopy(wire.docs["staging_pulse_history/2026-10-29"])
    remote_entry.update(date="2026-10-30", word="REMOTE_ONLY")
    wire.docs["staging_pulse_history/2026-10-30"] = remote_entry
    wire.versions["staging_pulse_history/2026-10-30"] = 1
    old = store.load_pulse_state()
    entry = copy.deepcopy(old["published_pulse"])
    entry.update(date="2026-10-31", word="NEW")
    stale_feed = {"entries": [dict(entry, date="2026-10-27", word="STALE_JSON")]}
    publish_bundle(store, old, entry, expected=old, local_feed=stale_feed, prepare=lambda *args: None)
    words = [item["word"] for item in wire.docs["staging_pulse_feed/v1"]["entries"]]
    assert words[:2] == ["NEW", "REMOTE_ONLY"]
    assert "STALE_JSON" not in words
    assert old["word"] in words


def test_requested_live_director_failure_blocks_before_render_or_commit(tmp_path, monkeypatch):
    import copy
    from src.zeitgeist import adk_pipeline
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    before_remote, before_local = copy.deepcopy(wire.docs), _published_files(tmp_path)
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.delenv("ZEITGEIST_OFFLINE_DEMO")
    # Isolate this gate from the independently tested live collection requirement.
    monkeypatch.setattr(adk_pipeline, "collect_96h_signals", lambda *args, **kwargs: [])
    monkeypatch.setattr(adk_pipeline, "select_and_render_pulse_assets", lambda **kwargs: pytest.fail("Live Director failure must gate rendering"))
    with pytest.raises(ValueError, match="Live.*commission"):
        adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run("2026-10-30", prefer_visual_mode="blueprint", live_collect=False, offline_demo=False)
    assert wire.docs == before_remote
    assert _published_files(tmp_path) == before_local
    assert not wire.commits


def test_existing_canonical_feed_preserves_its_entries_and_ignores_local_json(tmp_path, monkeypatch):
    import copy
    from src.zeitgeist.publication_commit import publish_bundle
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    old = store.load_pulse_state()
    existing = dict(old["published_pulse"], date="2026-10-28", word="PRESERVE_CANONICAL")
    wire.docs["staging_pulse_feed/v1"]["entries"].append(existing)
    monkeypatch.setattr(store, "load_pulse_feed_history", lambda **kwargs: pytest.fail("Existing canonical feed must preserve its own snapshot"))
    entry = dict(old["published_pulse"], date="2026-10-30", word="NEW")
    publish_bundle(store, old, entry, expected=old, local_feed={"entries": [dict(existing, word="STALE_JSON")]}, prepare=lambda *args: None)
    words = [item["word"] for item in wire.docs["staging_pulse_feed/v1"]["entries"]]
    assert "PRESERVE_CANONICAL" in words and "STALE_JSON" not in words


def test_canonical_history_failure_never_bootstraps_from_json(tmp_path, monkeypatch):
    from src.zeitgeist.publication_commit import publish_bundle
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    wire.docs.pop("staging_pulse_feed/v1")
    wire.versions.pop("staging_pulse_feed/v1")
    def fail(**kwargs): raise RuntimeError("History unavailable")
    monkeypatch.setattr(store, "load_pulse_feed_history", fail)
    old = store.load_pulse_state()
    with pytest.raises(RuntimeError, match="History unavailable"):
        publish_bundle(store, old, old["published_pulse"], expected=old, local_feed={"entries": []}, prepare=lambda *args: pytest.fail("No validation with unverified history"))
    assert not wire.commits


def test_explicit_offline_demo_can_use_fallback_with_live_flag_set(tmp_path, monkeypatch):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    orchestrator = ZeitgeistADKOrchestrator(tmp_path, store)
    orchestrator.run("2026-10-30", prefer_visual_mode="blueprint", offline_demo=True)
    assert orchestrator.last_commission.fallback_used
    assert len(wire.commits) == 1


@pytest.mark.parametrize("failure", ["incomplete_read", "malformed_json", "truncated_json", "invalid_ack", "unexpected_read_error", "response_close_error", "cancellation", "read_http_4xx"])
def test_rest_committed_but_bad_response_keeps_recovery_and_blocks_regeneration(tmp_path, monkeypatch, failure):
    from http.client import IncompleteRead
    from pathlib import Path
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from src.zeitgeist.publication_commit import PublicationMirrorError
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    before_local = _published_files(tmp_path)
    if failure == "incomplete_read":
        wire.response_error = IncompleteRead(b"partial private response", 999)
    elif failure == "unexpected_read_error":
        wire.response_error = ValueError("private response decoder failed")
    elif failure == "response_close_error":
        wire.response_close_error = ValueError("private close failed")
    elif failure == "cancellation":
        wire.response_error = KeyboardInterrupt()
    elif failure == "read_http_4xx":
        wire.response_error = HTTPError("https://private.invalid", 400, "private read error after HTTP success", {}, io.BytesIO())
    else:
        wire.response_bytes = {"malformed_json": b"not-json", "truncated_json": b'{"commitTime":', "invalid_ack": b'{}'}[failure]
    with pytest.raises(PublicationMirrorError):
        ZeitgeistADKOrchestrator(tmp_path, store).run("2026-10-30", prefer_visual_mode="blueprint", offline_demo=True)
    assert wire.docs["staging_profile_state/daily_pulse"]["date"] == "2026-10-30"
    assert wire.docs["staging_pulse_history/2026-10-30"]["date"] == "2026-10-30"
    assert _published_files(tmp_path) == before_local
    assert len(wire.commits) == 1
    marker = json.loads((tmp_path / "data/publication_reconcile.json").read_text())
    assert marker["status"] == "canonical_commit_outcome_unknown"
    assert (Path(marker["recovery_path"]) / "assets/pulse_manifest.json").exists()
    with pytest.raises(RuntimeError, match="pending canonical"):
        ZeitgeistADKOrchestrator(tmp_path, store).run("2026-10-30", prefer_visual_mode="blueprint", offline_demo=True)
    assert len(wire.commits) == 1


@pytest.mark.parametrize("error_kind", ["decoder", "missing_ack", "short_ack"])
def test_sdk_committed_but_decoder_error_keeps_recovery_and_blocks_regeneration(tmp_path, monkeypatch, error_kind):
    from pathlib import Path
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from src.zeitgeist.publication_commit import PublicationMirrorError
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    before_local = _published_files(tmp_path)

    class SDKBatch:
        def __init__(self): self.pending = []
        def update(self, reference, data, option):
            self.pending.append((reference._collection_id, reference._doc_id, data, reference.get()))
        def create(self, reference, data):
            self.pending.append((reference._collection_id, reference._doc_id, data, reference.get()))
        def commit(self, **kwargs):
            assert kwargs == {"retry": None, "timeout": 15}
            # The mock server applies the actual conditional bundle first;
            # the SDK then fails while decoding the already received response.
            stores._commit_documents(wire.client, self.pending)
            if error_kind == "decoder":
                raise ValueError("private SDK response decoding detail")
            return None if error_kind == "missing_ack" else []

    class SDKClient:
        def collection(self, name): return wire.client.collection(name)
        def batch(self): return SDKBatch()

    store._client = SDKClient()
    with pytest.raises(PublicationMirrorError):
        ZeitgeistADKOrchestrator(tmp_path, store).run("2026-10-30", prefer_visual_mode="blueprint", offline_demo=True)
    assert wire.docs["staging_profile_state/daily_pulse"]["date"] == "2026-10-30"
    assert _published_files(tmp_path) == before_local
    assert len(wire.commits) == 1
    marker = json.loads((tmp_path / "data/publication_reconcile.json").read_text())
    assert marker["status"] == "canonical_commit_outcome_unknown"
    assert (Path(marker["recovery_path"]) / "README.md").exists()
    with pytest.raises(RuntimeError, match="pending canonical"):
        ZeitgeistADKOrchestrator(tmp_path, store).run("2026-10-30", prefer_visual_mode="blueprint", offline_demo=True)
    assert len(wire.commits) == 1


@pytest.mark.parametrize("error_name", ["BadRequest", "Forbidden", "Unauthorized", "TooManyRequests"])
def test_sdk_definitive_rejection_is_sanitized_and_not_outcome_unknown(monkeypatch, error_name):
    from google.auth.credentials import AnonymousCredentials
    from google.cloud import firestore
    from google.api_core import exceptions
    client = firestore.Client(project="fixture", credentials=AnonymousCredentials())
    batch = client.batch()
    def reject(**kwargs): raise getattr(exceptions, error_name)("private provider detail")
    monkeypatch.setattr(batch, "commit", reject)
    monkeypatch.setattr(client, "batch", lambda: batch)
    with pytest.raises(RuntimeError) as failure:
        stores._commit_documents(client, [("pulse_feed", "v1", {"entries": []}, stores._FirestoreRestDocSnapshot("v1", False))])
    assert not isinstance(failure.value, stores.PublicationOutcomeUnknown)
    assert "private" not in str(failure.value)
