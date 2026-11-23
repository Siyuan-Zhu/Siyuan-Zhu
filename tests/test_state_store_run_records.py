"""The private run-record collection in every state store."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.state_store import (
    FirestoreStateStore,
    InMemoryStateStore,
    JsonFileStateStore,
    StateStore,
)
from tests.test_firestore_store import _RecordingFirestoreClient

REPO_ROOT = Path(__file__).resolve().parents[1]

OLDER = {"record_schema": 1, "kind": "daily_run", "written_at": "2026-11-21T08:00:00+00:00", "value": 1.5}
NEWER = {"record_schema": 1, "kind": "private_run", "written_at": "2026-11-22T08:00:00+00:00", "value": 2}
OLDER_ID = "daily_run:2026-11-21:local:1"
NEWER_ID = "private_run:2026-11-22:local:1"


def _firestore(tmp_path: Path) -> FirestoreStateStore:
    return FirestoreStateStore(
        client=_RecordingFirestoreClient(), project_id="test-proj", canonical_only=True
    )


STORES = {
    "memory": lambda tmp_path: InMemoryStateStore(),
    "json": lambda tmp_path: JsonFileStateStore(repo_root=tmp_path),
    "firestore": _firestore,
}


@pytest.fixture(params=sorted(STORES))
def store(request, tmp_path):
    return STORES[request.param](tmp_path)


def test_every_store_satisfies_the_protocol(store) -> None:
    for method in ("save_run_record", "load_run_records"):
        assert hasattr(StateStore, method)
        assert callable(getattr(store, method))
    assert isinstance(store, StateStore)


def test_empty_store_loads_no_records(store) -> None:
    assert store.load_run_records() == []


def test_two_records_round_trip_newest_first(store) -> None:
    store.save_run_record(OLDER_ID, OLDER)
    store.save_run_record(NEWER_ID, NEWER)
    assert store.load_run_records() == [NEWER, OLDER]


def test_records_filter_by_kind(store) -> None:
    store.save_run_record(OLDER_ID, OLDER)
    store.save_run_record(NEWER_ID, NEWER)
    assert store.load_run_records(kind="daily_run") == [OLDER]
    assert store.load_run_records(kind="private_run") == [NEWER]
    assert store.load_run_records(kind="other") == []


def test_limit_keeps_the_newest(store) -> None:
    store.save_run_record(OLDER_ID, OLDER)
    store.save_run_record(NEWER_ID, NEWER)
    assert store.load_run_records(limit=1) == [NEWER]


def test_saving_the_same_run_id_replaces_the_record(store) -> None:
    store.save_run_record(OLDER_ID, OLDER)
    store.save_run_record(OLDER_ID, dict(OLDER, value=3))
    assert store.load_run_records() == [dict(OLDER, value=3)]


def test_saved_and_loaded_records_are_independent_copies(store) -> None:
    document = dict(OLDER, nested={"a": [1]})
    store.save_run_record(OLDER_ID, document)
    document["nested"]["a"].append(2)
    loaded = store.load_run_records()
    assert loaded == [dict(OLDER, nested={"a": [1]})]
    loaded[0]["nested"]["a"].append(3)
    assert store.load_run_records() == [dict(OLDER, nested={"a": [1]})]


BAD_IDS = ("", "a" * 151, "run/record", "../escape", "run record", "run\\record", "r\xe9cord", None, 7)


@pytest.mark.parametrize("bad", BAD_IDS)
def test_a_bad_run_id_raises_without_touching_the_store(store, bad, tmp_path) -> None:
    with pytest.raises(ValueError):
        store.save_run_record(bad, OLDER)
    assert store.load_run_records() == []
    if isinstance(store, FirestoreStateStore):
        assert store._client.docs == {}
    assert not (tmp_path / "data").exists()


def test_the_longest_allowed_run_id_is_accepted(store) -> None:
    store.save_run_record("A" * 150, OLDER)
    assert store.load_run_records() == [OLDER]


def test_json_record_lands_under_data_run_records(tmp_path) -> None:
    store = JsonFileStateStore(repo_root=tmp_path)
    store.save_run_record(OLDER_ID, OLDER)
    assert [p.name for p in (tmp_path / "data" / "run_records").iterdir()] == [f"{OLDER_ID}.json"]
    assert not (tmp_path / "data" / "ingestion_cache").exists()


def test_firestore_record_is_one_document_per_run_id_in_run_records() -> None:
    client = _RecordingFirestoreClient()
    store = FirestoreStateStore(client=client, project_id="test-proj", canonical_only=True)
    store.save_run_record(OLDER_ID, OLDER)
    store.save_run_record(NEWER_ID, NEWER)
    assert client.docs == {f"run_records/{OLDER_ID}": OLDER, f"run_records/{NEWER_ID}": NEWER}


def test_firestore_record_honours_collection_prefix() -> None:
    client = _RecordingFirestoreClient()
    store = FirestoreStateStore(
        client=client, project_id="test-proj", canonical_only=True, collection_prefix="staging_"
    )
    store.save_run_record(OLDER_ID, OLDER)
    assert set(client.docs) == {f"staging_run_records/{OLDER_ID}"}
    assert store.load_run_records() == [OLDER]


def test_firestore_record_is_never_mirrored_to_the_checkout(tmp_path) -> None:
    store = FirestoreStateStore(
        client=_RecordingFirestoreClient(),
        project_id="test-proj",
        canonical_only=True,
        mirror_store=JsonFileStateStore(repo_root=tmp_path),
    )
    store.save_run_record(OLDER_ID, OLDER)
    assert store.load_run_records() == [OLDER]
    data_dir = tmp_path / "data"
    assert not data_dir.exists() or list(data_dir.rglob("*")) == []


def test_firestore_without_client_delegates_to_fallback(monkeypatch) -> None:
    fallback = InMemoryStateStore()
    store = FirestoreStateStore(project_id="test-proj", fallback_store=fallback)
    monkeypatch.setattr(store, "_get_client", lambda: None)
    store.save_run_record(OLDER_ID, OLDER)
    assert fallback.load_run_records() == [OLDER]
    assert store.load_run_records(kind="daily_run") == [OLDER]


def test_json_run_records_are_gitignored(tmp_path) -> None:
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    result = subprocess.run(
        ["git", "-C", str(tmp_path), "check-ignore", "-q", "data/run_records/x.json"],
        check=False,
    )
    assert result.returncode == 0
