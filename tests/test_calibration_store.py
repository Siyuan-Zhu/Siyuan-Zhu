"""Named private calibration documents in every state store (embedding and judge)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.state_store import (
    CALIBRATION_DOCUMENT_NAMES,
    FirestoreStateStore,
    InMemoryStateStore,
    JsonFileStateStore,
    StateStore,
)
from tests.test_firestore_store import _RecordingFirestoreClient

NAMES = ("embedding_calibration", "judge_calibration")
REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)
SAMPLE = {"schema_version": 1, "status": "rejected"}


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


def test_names_are_exactly_the_two_calibration_documents() -> None:
    assert CALIBRATION_DOCUMENT_NAMES == frozenset(NAMES)


def test_every_store_satisfies_the_protocol(store) -> None:
    assert isinstance(store, StateStore)
    for method in ("load_calibration", "save_calibration", "clear_calibration"):
        assert hasattr(StateStore, method)


@pytest.mark.parametrize("name", NAMES)
def test_absent_document_loads_none(store, name) -> None:
    assert store.load_calibration(name) is None


@pytest.mark.parametrize("name", NAMES)
def test_save_then_load_returns_an_independent_copy(store, name) -> None:
    document = {"schema_version": 1, "status": "rejected"}
    store.save_calibration(name, document)
    document["status"] = "mutated_after_save"

    first = store.load_calibration(name)
    assert first == SAMPLE
    first["status"] = "mutated_after_load"
    first["extra"] = [1]
    assert store.load_calibration(name) == SAMPLE


@pytest.mark.parametrize("name", NAMES)
def test_save_replaces_the_whole_document(store, name) -> None:
    store.save_calibration(name, {"schema_version": 1, "status": "rejected", "old": 0.5})
    store.save_calibration(name, {"schema_version": 1, "status": "accepted"})
    assert store.load_calibration(name) == {"schema_version": 1, "status": "accepted"}


@pytest.mark.parametrize("name", NAMES)
def test_clear_reports_whether_a_document_was_removed(store, name) -> None:
    store.save_calibration(name, SAMPLE)
    assert store.clear_calibration(name) is True
    assert store.load_calibration(name) is None
    assert store.clear_calibration(name) is False


def test_the_two_names_are_separate_documents(store) -> None:
    store.save_calibration("embedding_calibration", {"schema_version": 1, "status": "accepted"})
    assert store.load_calibration("judge_calibration") is None

    store.save_calibration("judge_calibration", {"schema_version": 1, "status": "rejected"})
    assert store.load_calibration("embedding_calibration") == {"schema_version": 1, "status": "accepted"}
    assert store.load_calibration("judge_calibration") == {"schema_version": 1, "status": "rejected"}

    assert store.clear_calibration("embedding_calibration") is True
    assert store.load_calibration("embedding_calibration") is None
    assert store.load_calibration("judge_calibration") == {"schema_version": 1, "status": "rejected"}


BAD_NAMES = ("orbit_roster", "bot_memory", "", "Embedding_calibration", "embedding_calibration ",
             "../embedding_calibration", "ingestion_cache/embedding_calibration", None, 1)


@pytest.mark.parametrize("bad", BAD_NAMES)
def test_other_names_are_refused_without_touching_the_store(store, bad, tmp_path) -> None:
    with pytest.raises(ValueError):
        store.load_calibration(bad)
    with pytest.raises(ValueError):
        store.save_calibration(bad, SAMPLE)
    with pytest.raises(ValueError):
        store.clear_calibration(bad)
    for name in NAMES:
        assert store.load_calibration(name) is None
    if isinstance(store, FirestoreStateStore):
        assert store._client.docs == {}
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("name", NAMES)
def test_firestore_document_is_profile_state_name(name) -> None:
    client = _RecordingFirestoreClient()
    store = FirestoreStateStore(client=client, project_id="test-proj", canonical_only=True)
    store.save_calibration(name, SAMPLE)
    assert client.docs == {f"profile_state/{name}": SAMPLE}


@pytest.mark.parametrize("name", NAMES)
def test_firestore_document_honours_collection_prefix(name) -> None:
    client = _RecordingFirestoreClient()
    store = FirestoreStateStore(
        client=client, project_id="test-proj", canonical_only=True, collection_prefix="staging_"
    )
    store.save_calibration(name, SAMPLE)
    assert set(client.docs) == {f"staging_profile_state/{name}"}
    assert store.load_calibration(name) == SAMPLE
    assert store.clear_calibration(name) is True
    assert client.docs == {}


@pytest.mark.parametrize("name", NAMES)
def test_firestore_save_is_never_mirrored_to_the_checkout(name, tmp_path) -> None:
    client = _RecordingFirestoreClient()
    store = FirestoreStateStore(
        client=client,
        project_id="test-proj",
        canonical_only=True,
        mirror_store=JsonFileStateStore(repo_root=tmp_path),
    )
    store.save_calibration(name, SAMPLE)
    assert store.load_calibration(name) == SAMPLE
    data_dir = tmp_path / "data"
    assert not data_dir.exists() or list(data_dir.rglob("*")) == []


@pytest.mark.parametrize("name", NAMES)
def test_firestore_without_client_delegates_to_fallback(name, monkeypatch) -> None:
    fallback = InMemoryStateStore()
    store = FirestoreStateStore(project_id="test-proj", fallback_store=fallback)
    monkeypatch.setattr(store, "_get_client", lambda: None)
    store.save_calibration(name, SAMPLE)
    assert fallback.load_calibration(name) == SAMPLE
    assert store.load_calibration(name) == SAMPLE
    assert store.clear_calibration(name) is True
    assert fallback.load_calibration(name) is None


@pytest.mark.parametrize("name", NAMES)
def test_json_file_sits_in_data_and_leaves_ingestion_cache_alone(name, tmp_path) -> None:
    store = JsonFileStateStore(repo_root=tmp_path)
    store.save_ingestion_cache_day(
        "2026-11-22",
        {"hash_0001": {"bucket": "ORBIT", "published_at": NOW.isoformat(), "summary": "synthetic"}},
    )
    before = store.load_ingestion_cache(NOW)

    store.save_calibration(name, SAMPLE)

    assert store.load_ingestion_cache(NOW) == before
    assert (tmp_path / "data" / f"{name}.json").is_file()
    assert not list((tmp_path / "data" / "ingestion_cache").glob("*calibration*"))


@pytest.mark.parametrize("name", NAMES)
def test_json_calibration_files_are_gitignored(name) -> None:
    lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert f"data/{name}.json" in lines
