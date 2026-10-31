#!/usr/bin/env python3
"""Rehearse the actual application in an empty, per-run Firestore namespace.

Invocation performs isolated staging writes. Imports never contact services.
Private recovery journals stay outside the checkout and are never upload artifacts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
from urllib.parse import urlsplit
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import process_visitor_event
from scripts.build_readme import build_all_readmes, check_all_readmes, resolve_repo_slug
from scripts.validate_release_links import validate_release_bundle
from src.game_engine import create_initial_game_state
from src.state_store import (
    FirestoreStateStore,
    JsonFileStateStore,
    get_state_store,
    PublicationConflict,
    PublicationOutcomeUnknown,
    _FirestoreRestClient,
    _commit_documents,
    _encode_json_safe,
)
from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
from src.zeitgeist.collectors import (
    IngestionUnavailable,
    public_ingestion_diagnostics,
    public_collection_health,
)
from src.zeitgeist.publication import verify_publication_bundle
from src.zeitgeist.spend_guard import SpendBlocked

PROJECT = "git-readme-features"
DATABASE = "(default)"
PREFIX = re.compile(r"staging_[1-9][0-9]*_[1-9][0-9]*_application_\Z")
COLLECTIONS = (
    "profile_state",
    "pulse_feed",
    "pulse_history",
    "pulse_votes",
    "vote_rate_limits",
    "processed_events",
    "ingestion_cache",
)
ZERO_VOTES = {"spot_on": 0, "too_sunny": 0, "too_heavy": 0}


class ApplicationRehearsalError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            "Application rehearsal failed; inspect canonical namespace and private recovery journal"
        )


def _failure_classification(error):
    """Expose only fixed categories and typed status from recognized endpoints.

    Exception messages, response bodies, URLs and arbitrary attributes are never
    serialized or used to infer a domain. Generic transport errors stay unknown.
    """
    for error_type, domain in (
        (IngestionUnavailable, "ingestion"),
        (SpendBlocked, "budget_control"),
        (PublicationConflict, "storage_conflict"),
        (PublicationOutcomeUnknown, "storage_outcome_unknown"),
    ):
        if isinstance(error, error_type):
            return {"failure_domain": domain}
    unknown = {"failure_domain": "unknown"}
    if not isinstance(error, urllib.error.HTTPError):
        return unknown
    status = error.code
    if type(status) is not int or not 100 <= status <= 599:
        return unknown
    url = error.url
    if not isinstance(url, str) or len(url) > 8192:
        return unknown
    try:
        parsed = urlsplit(url)
        host = parsed.hostname
    except ValueError:
        return unknown
    if parsed.scheme != "https":
        return unknown
    if host == "firestore.googleapis.com":
        domain = "storage_authentication" if status in (401, 403) else "storage_api"
    elif host in ("sts.googleapis.com", "iamcredentials.googleapis.com"):
        domain = "authentication"
    else:
        return unknown
    return {"failure_domain": domain, "http_status": status}


def _private_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        os.chmod(handle.name, 0o600)
        json.dump(_encode_json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _target(store):
    if (store._project_id, store._database_id) != (
        PROJECT,
        DATABASE,
    ) or not PREFIX.fullmatch(store.collection_prefix):
        raise ValueError("Application target differs from fixed staging identity")


def _snapshot(store):
    _target(store)
    client = store._get_client()
    if client is None or store._fallback_store is not None:
        raise RuntimeError(
            "Canonical-only Firestore store required; JSON fallback forbidden"
        )
    documents = {}
    for logical in COLLECTIONS:
        for snapshot in client.collection(store._col(logical)).stream():
            doc_id = str(snapshot.id)
            if not doc_id or "/" in doc_id:
                raise RuntimeError("Invalid canonical document identity")
            documents[f"{logical}/{doc_id}"] = snapshot
    return documents


def _record(path, phase, documents):
    _private_json(
        path,
        {
            "phase": phase,
            "documents": {
                name: {
                    "update_time": str(getattr(snapshot, "update_time", "")),
                    "payload": snapshot.to_dict(),
                }
                for name, snapshot in documents.items()
            },
        },
    )


def _refresh_mirrors(root, documents):
    # Exact canonical projections, never enrich from the old Git seed.
    names = {
        "profile_state/active_game": "game_state.json",
        "profile_state/daily_pulse": "zeitgeist_state.json",
        "pulse_feed/v1": "pulse_feed.json",
        "profile_state/elo_ledger": "hall_of_records.json",
        "profile_state/bot_memory": "bot_brain.json",
        "profile_state/orbit_roster": "orbit_roster.json",
    }
    for key, filename in names.items():
        if key in documents:
            target = root / "data" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(documents[key].to_dict(), indent=2) + "\n")


@contextmanager
def _event_environment(prefix):
    keys = (
        "GITHUB_ACTIONS",
        "GITHUB_EVENT_PATH",
        "ISSUE_NUMBER",
        "ISSUE_TITLE",
        "VISITOR_LOGIN",
        "PULSE_VOTE_SALT",
        "STATE_STORE_BACKEND",
        "FIRESTORE_COLLECTION_PREFIX",
        "GCP_PROJECT_ID",
        "GOOGLE_CLOUD_PROJECT",
        "FIRESTORE_PROJECT_ID",
        "FIRESTORE_DATABASE_ID",
    )
    previous = {key: os.environ.get(key) for key in keys}
    for key in keys:
        os.environ.pop(key, None)
    os.environ.update(
        STATE_STORE_BACKEND="firestore",
        FIRESTORE_COLLECTION_PREFIX=prefix,
        GCP_PROJECT_ID=PROJECT,
        GOOGLE_CLOUD_PROJECT=PROJECT,
        FIRESTORE_PROJECT_ID=PROJECT,
        FIRESTORE_DATABASE_ID=DATABASE,
    )
    os.environ["GITHUB_ACTIONS"] = "false"
    os.environ["PULSE_VOTE_SALT"] = "isolated-application-fixture:" + prefix
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _seed(store, private):
    _target(store)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    seeds = {
        "profile_state/active_game": create_initial_game_state(),
        "profile_state/daily_pulse": {
            "state_version": 0,
            "date_iso": date,
            "visitor_mood_tally": dict(ZERO_VOTES),
            "mood_votes": dict(ZERO_VOTES),
            "recent_words": [],
        },
        "profile_state/elo_ledger": {
            "state_version": 0,
            "bot_elo": 1200,
            "players": {},
            "recent_rounds": [],
        },
        "profile_state/bot_memory": {
            "state_version": 0,
            "games_learned": 0,
            "states": {},
        },
        "profile_state/game_rate_limits": {"users": {}},
        "profile_state/orbit_roster": {
            "state_version": 0,
            "handles": {},
            "curated_o3_cursor": {},
        },
        "pulse_feed/v1": {"version": "2.0", "schema_version": "2.0", "entries": []},
    }
    client = store._get_client()
    writes = []
    for key, payload in seeds.items():
        logical, doc_id = key.split("/", 1)
        snapshot = client.collection(store._col(logical)).document(doc_id).get()
        if snapshot.exists:
            raise ValueError("Staging namespace occupied before create-only bootstrap")
        writes.append((store._col(logical), doc_id, payload, snapshot))
    _private_json(
        private / "create-only-bootstrap.json",
        {"expected_absent": list(seeds), "desired_documents": seeds},
    )
    _commit_documents(client, writes)


def _verify_publication(root, documents):
    manifest = verify_publication_bundle(root, require_current_pulse=True)
    profile = documents["profile_state/daily_pulse"].to_dict()
    entry = documents["pulse_history/" + manifest["date"]].to_dict()
    feed = documents["pulse_feed/v1"].to_dict()
    feed_entry = next(row for row in feed["entries"] if row["date"] == manifest["date"])
    if profile.get("publication_id") != manifest["publication_id"]:
        raise RuntimeError(
            "Canonical publication identity differs from rendered manifest"
        )
    for candidate in (profile, entry, feed_entry):
        for key, manifest_key in (
            ("date", "date"),
            ("word", "word"),
            ("gloss", "gloss"),
            ("visual_medium", "medium"),
        ):
            if candidate.get(key) != manifest[manifest_key]:
                raise RuntimeError("Canonical publication projections differ")
        if candidate.get("publication_id") != manifest["publication_id"]:
            raise RuntimeError("Canonical publication ID differs across projections")
    return {
        "publication_id": manifest["publication_id"],
        "canonical_verified": True,
        "history_feed_fields_verified": True,
        "explicit_history_feed_ids": entry.get("publication_id")
        == feed_entry.get("publication_id")
        == manifest["publication_id"],
    }


def _events(root, store, private):
    next_issue = 900000000000

    def event(actor, title, require=False, *, issue_number=None):
        nonlocal next_issue
        if issue_number is None:
            next_issue += 1
            issue_number = next_issue
        output, error = StringIO(), StringIO()
        args = [
            "--repo-root",
            str(root),
            "--login",
            actor,
            "--title",
            title,
            "--issue-number",
            str(issue_number),
        ]
        if require:
            args.append("--require-accepted")
        with redirect_stdout(output), redirect_stderr(error):
            code = process_visitor_event.main(args, store=store)
        with (private / "application.log").open("a") as handle:
            handle.write(error.getvalue())
        if code != 0:
            raise RuntimeError("Actual visitor CLI rejected its success-path fixture")
        return json.loads(output.getvalue())

    before = store.load_game_state()
    turn = int(before["turn"])
    title = f"game:move:connect4:col:4:turn={turn}"
    move = event("staging-player", title, True)
    after = store.load_game_state()
    pieces = [cell for row in after["board"] for cell in row]
    if not (
        move.get("accepted")
        and move.get("bot_move")
        and int(after["turn"]) > turn
        and sum(cell in (1, "C") for cell in pieces) == 1
        and sum(cell in (2, "B") for cell in pieces) == 1
    ):
        raise RuntimeError(
            "Canonical fresh board lacks accepted visitor and bot response"
        )
    stale = event("staging-second", title)
    if stale.get("accepted") is not False or store.load_game_state() != after:
        raise RuntimeError("Stale move changed canonical game state")
    replay = event("staging-player", title, issue_number=900000000001)
    if (replay.get("accepted") is not False
            or replay.get("status") != "already_processed"
            or store.load_game_state() != after):
        raise RuntimeError("Replayed Issue changed canonical game state")
    first = event("staging-voter", "pulse:vote:spot_on", True)
    duplicate = event("staging-voter", "pulse:vote:spot_on")
    expected = dict(ZERO_VOTES, spot_on=1)
    current = store.load_pulse_state()
    if (
        first.get("status") != "recorded_vote"
        or duplicate.get("status") != "ignored"
        or first.get("tally") != expected
        or duplicate.get("tally") != expected
        or current.get("visitor_mood_tally") != expected
    ):
        raise RuntimeError("Canonical vote was not counted exactly once from zero")
    return {
        "accepted_move": True,
        "bot_response": True,
        "stale_rejected": True,
        "issue_replay_rejected": True,
        "vote_counted_once": True,
    }


def _cache_probe(store):
    now = datetime.now(timezone.utc)
    date = now.strftime("%Y-%m-%d")
    before = store.load_ingestion_cache(now)
    vector = b"\x00\x3c" * 16
    rows = {
        "application-fixture-"
        + str(index): {
            "url_hash": "application-fixture-" + str(index),
            "bucket": "ORBIT",
            "tier": "O1",
            "origin": "offline_demo",
            "source_id": "explicit_rehearsal_fixture",
            "published_at": now.isoformat(),
            "embedding_f16": vector,
            "embedding_source": "offline_demo",
        }
        for index in range(501)
    }
    # This runs after publication and is explicitly not live ingestion evidence.
    store.save_ingestion_cache_day(date, rows)
    actual = store.load_ingestion_cache(now)
    if any(actual.get(key, {}).get("embedding_f16") != vector for key in rows):
        raise RuntimeError(
            "Canonical paged ingestion cache roundtrip changed binary vectors"
        )
    pages = list(store._get_client().collection(store._col("ingestion_cache")).stream())
    fixture_pages = [
        page
        for page in pages
        if set(page.to_dict().get("items") or {}).intersection(rows)
    ]
    if len(fixture_pages) < 2 or any(
        "embedding_f32" in item
        for page in pages
        for item in (page.to_dict().get("items") or {}).values()
    ):
        raise RuntimeError(
            "Canonical cache paging/vector storage contract not exercised"
        )
    return {
        "application_items_read_before_fixture": len(before),
        "fixture_items_verified": len(rows),
        "fixture_page_count": len(fixture_pages),
        "fixture_scope": "explicit_postpublication_offline_probe",
    }


def _delete_owned(store, snapshots):
    """One conditional delete batch; never enumerate or touch shared spending docs."""
    _target(store)
    client = store._get_client()
    if not snapshots or len(snapshots) > 500:
        raise RuntimeError("Owned cleanup requires 1..500 recorded documents")
    for path, snapshot in snapshots.items():
        logical, doc_id = path.split("/", 1)
        if (
            logical not in COLLECTIONS
            or "/" in doc_id
            or not getattr(snapshot, "update_time", None)
        ):
            raise RuntimeError("Owned cleanup lacks a safe identity/revision")
    if isinstance(client, _FirestoreRestClient):
        name = f"projects/{PROJECT}/databases/{DATABASE}/documents/"
        writes = [
            {
                "delete": name
                + store._col(path.split("/", 1)[0])
                + "/"
                + path.split("/", 1)[1],
                "currentDocument": {"updateTime": str(snapshot.update_time)},
            }
            for path, snapshot in snapshots.items()
        ]
        request = urllib.request.Request(
            client._base_url() + ":commit",
            data=json.dumps({"writes": writes}).encode(),
            headers={
                "Authorization": "Bearer " + client._resolve_token(),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                ack = json.loads(response.read())
            if not ack.get("commitTime") or len(ack.get("writeResults", [])) != len(
                writes
            ):
                raise ValueError("Incomplete cleanup acknowledgement")
        except BaseException:
            raise PublicationOutcomeUnknown(
                "Owned namespace cleanup outcome unknown; reconcile privately"
            ) from None
    else:
        from google.cloud.firestore_v1 import LastUpdateOption

        batch = client.batch()
        for path, snapshot in snapshots.items():
            logical, doc_id = path.split("/", 1)
            batch.delete(
                client.collection(store._col(logical)).document(doc_id),
                option=LastUpdateOption(snapshot.update_time),
            )
        try:
            results = batch.commit(retry=None, timeout=15)
            if not isinstance(results, list) or len(results) != len(snapshots):
                raise ValueError("Incomplete SDK cleanup acknowledgement")
        except BaseException:
            raise PublicationOutcomeUnknown(
                "Owned namespace cleanup outcome unknown; reconcile privately"
            ) from None


def rehearse(
    repo_root,
    collection_prefix,
    *,
    store_factory=None,
    private_root=None,
):
    if not PREFIX.fullmatch(collection_prefix):
        raise ValueError("Expected new staging_<run>_<attempt>_application_ namespace")
    root = Path(repo_root).resolve()
    flags = tuple(
        os.environ.get(key, "0")
        for key in (
            "ZEITGEIST_LIVE_COLLECT",
            "ZEITGEIST_LIVE_LLM",
            "ZEITGEIST_LIVE_UNSPLASH",
        )
    )
    if flags not in (("0", "0", "0"), ("1", "1", "1")) or os.environ.get(
        "ZEITGEIST_OFFLINE_DEMO", "0"
    ) != ("1" if flags[0] == "0" else "0"):
        raise ValueError(
            "Application rehearsal requires explicit all-off demo or all-on live flags"
        )
    if store_factory is None:
        # Instantiate the production adapter through its actual runtime seam,
        # fixing only the explicitly isolated staging identity for this rehearsal.
        with _event_environment(collection_prefix):
            store = get_state_store(
                repo_root=root, backend="firestore", collection_prefix=collection_prefix
            )
        mirror = getattr(store, "_mirror_store", None)
        if (
            not isinstance(store, FirestoreStateStore)
            or not store.canonical_only
            or store._fallback_store is not None
            or not isinstance(mirror, JsonFileStateStore)
            or Path(mirror.repo_root).resolve() != root
        ):
            raise ValueError(
                "Canonical runtime factory must provide canonical-only authority and exact write-only mirrors"
            )
    else:
        # Explicit test seam: real application logic can run over a wire fake.
        store = store_factory(
            project_id=PROJECT,
            database_id=DATABASE,
            collection_prefix=collection_prefix,
            fallback_store=None,
        )
    if (store._project_id, store._database_id, store.collection_prefix) != (
        PROJECT,
        DATABASE,
        collection_prefix,
    ) or store._fallback_store is not None:
        raise ValueError(
            "Application target differs from canonical-only fixed staging identity"
        )
    if _snapshot(store):
        raise ValueError("Refusing occupied application staging namespace")
    if private_root is None:
        private = Path(tempfile.mkdtemp(prefix="egm-application-rehearsal-"))
    else:
        private = Path(private_root).resolve()
        if private.is_relative_to(root) or (
            private.exists() and any(private.iterdir())
        ):
            raise ValueError(
                "Private journal must be a new directory outside the checkout"
            )
        private.mkdir(parents=True, exist_ok=True)
    os.chmod(private, 0o700)
    journal = private / "application-journal.json"
    report = {
        "status": "running",
        "backend": "firestore",
        "project": PROJECT,
        "database": DATABASE,
        "collection_prefix": collection_prefix,
        "providers_live": flags[0] == "1",
        "private_journal_dir": str(private),
        "namespace_retained": True,
        "evidence_scope": (
            "live_application"
            if flags[0] == "1"
            else "offline_application_over_firestore"
        ),
        "release_decision": "NO_GO_pending_separate_release_gates_and_approval",
    }
    phase = "create_only_bootstrap_pending"
    _record(journal, phase, {})
    try:
        with _event_environment(collection_prefix):
            _seed(store, private)
            documents = _snapshot(store)
            _record(journal, "bootstrap_verified", documents)
            _refresh_mirrors(root, documents)
            phase = "actual_orchestrator"
            with (private / "application.log").open("a") as log, redirect_stdout(
                log
            ), redirect_stderr(log):
                ZeitgeistADKOrchestrator(root, state_store=store).run()
            documents = _snapshot(store)
            _record(journal, "publication_verified", documents)
            report["publication"] = _verify_publication(root, documents)
            health = (
                documents["profile_state/daily_pulse"].to_dict().get("telemetry") or {}
            ).get("collection_health") or {}
            report["collection"] = public_collection_health(health)
            if flags[0] == "1" and report["collection"].get("publishable") is not True:
                raise RuntimeError(
                    "Live application lacks explicit publishable source-coverage evidence"
                )
            if store.load_elo_ledger().get("players"):
                raise RuntimeError(
                    "Fresh application namespace contains invented player ratings"
                )
            phase = "actual_visitor_cli"
            report["interactions"] = _events(root, store, private)
            phase = "canonical_cache_probe"
            report["cache"] = _cache_probe(store)
            documents = _snapshot(store)
            _refresh_mirrors(root, documents)
            build_all_readmes(root, state_store=store)
            audit = validate_release_bundle(
                root,
                expected_repo_slug=resolve_repo_slug(),
                require_publication_manifest=True,
            )
            if not audit.get("ok") or not check_all_readmes(root, state_store=store):
                raise RuntimeError(
                    "Actual Firestore application failed link/manifest/README gates"
                )
            report["link_manifest_verified"] = True
            _record(journal, "known_success_owned_cleanup_pending", documents)
            phase = "conditional_owned_cleanup"
            _delete_owned(store, documents)
            remaining = _snapshot(store)
            if remaining:
                raise RuntimeError("Owned namespace cleanup left unexpected documents")
            _record(journal, "complete_empty_namespace", {})
        report.update(
            status="ok",
            namespace_retained=False,
            owned_document_count=len(documents),
            remaining_document_count=0,
        )
        return report
    except BaseException as error:
        report.update(
            status="failed",
            failed_phase=phase,
            error_type=type(error).__name__,
            cleanup_attempted=phase == "conditional_owned_cleanup",
            **_failure_classification(error),
        )
        if isinstance(error, IngestionUnavailable):
            detail = public_ingestion_diagnostics(error.public_diagnostics)
            if detail:
                report["provider_diagnostics"] = detail
            health = public_collection_health(error.public_collection_health)
            if health:
                report["collection"] = health
        elif isinstance(error, SpendBlocked):
            report["budget_diagnostics"] = public_ingestion_diagnostics(error.public_diagnostics)
        if phase == "conditional_owned_cleanup":
            # The delete batch itself may have committed before its reply failed.
            report.update(namespace_retained=None, cleanup_outcome="unknown")
        try:
            _private_json(
                private / "failure.json",
                {
                    "phase": phase,
                    "error_type": type(error).__name__,
                    **_failure_classification(error),
                },
            )
        except OSError:
            report["private_failure_log_written"] = False
        raise ApplicationRehearsalError(report) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--collection-prefix", required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args(argv)
    try:
        report = rehearse(args.repo_root, args.collection_prefix)
        code = 0
    except ApplicationRehearsalError as error:
        report, code = error.report, 1
    except BaseException as error:
        report, code = {
            "status": "failed",
            "error_type": type(error).__name__,
            "phase": "preflight",
            **_failure_classification(error),
        }, 1
    text = json.dumps(report, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
