"""Large date caches remain complete, bounded, and readable across repacks."""
import base64
import copy
from datetime import datetime, timedelta, timezone

import pytest

from src import state_store as stores
from tests.test_firestore_store import _RecordingFirestoreClient

DATE = "2026-10-30"
NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def items(count, start=0):
    raw = bytes(1536)
    return {f"hash_{i:04d}": {"bucket": "ORBIT", "published_at": NOW.isoformat(),
        "summary": "Complete source context " * 30, "embedding_f16": raw,
        "embedding_f16_b64": base64.b64encode(raw).decode(), "embedding_model": "gemini-embedding-001"}
        for i in range(start, start + count)}


def test_large_cache_is_binary_only_bounded_and_preserves_every_item():
    client = _RecordingFirestoreClient()
    store = stores.FirestoreStateStore(client=client)
    expected = items(450)
    store.save_ingestion_cache_day(DATE, expected)
    pages = {key: value for key, value in client.docs.items() if key.startswith("ingestion_cache/")}
    assert len(pages) > 1
    assert all(stores._cache_payload_bytes(value) <= 800 * 1024 for value in pages.values())
    assert all("embedding_f16_b64" not in row for page in pages.values() for row in page["items"].values())
    restored = store.load_ingestion_cache(NOW)
    assert set(restored) == set(expected)
    assert all(restored[key]["summary"] == expected[key]["summary"] for key in expected)
    assert all(restored[key]["embedding_f16"] == expected[key]["embedding_f16"] for key in expected)


def test_same_day_delta_merges_previous_items_instead_of_dropping_them():
    client = _RecordingFirestoreClient()
    store = stores.FirestoreStateStore(client=client)
    store.save_ingestion_cache_day(DATE, items(2))
    store.save_ingestion_cache_day(DATE, items(1, start=2))
    assert set(store.load_ingestion_cache(NOW)) == {"hash_0000", "hash_0001", "hash_0002"}


def test_loader_reads_legacy_pages_and_ignores_retired_pages():
    client = _RecordingFirestoreClient()
    client.docs.update({"ingestion_cache/" + DATE: {"date": DATE, "items": items(1), "page_count": 1},
        "ingestion_cache/" + DATE + "__001": {"date": DATE, "items": items(1, start=2)},
        "ingestion_cache/2026-10-29": {"date": "2026-10-29", "items": items(1, start=3)}})
    store = stores.FirestoreStateStore(client=client)
    assert set(store.load_ingestion_cache(NOW)) == {"hash_0000", "hash_0003"}


def test_page_pruning_keeps_original_date_and_asymmetric_retention():
    client = _RecordingFirestoreClient()
    page = {"date": DATE, "items": {
        "weekly": {"bucket": "ORBIT", "published_at": (NOW - timedelta(hours=200)).isoformat()},
        "expired_macro": {"bucket": "M3", "published_at": (NOW - timedelta(hours=100)).isoformat()}}}
    client.docs["ingestion_cache/" + DATE + "__001"] = page
    client.docs["ingestion_cache/2026-10-01__001"] = {"date": "2026-10-01", "items": items(1)}
    store = stores.FirestoreStateStore(client=client)
    store.prune_expired_cache_items(NOW)
    assert "ingestion_cache/2026-10-01__001" not in client.docs
    retained = client.docs["ingestion_cache/" + DATE + "__001"]
    assert retained["date"] == DATE and set(retained["items"]) == {"weekly"}


def test_oversized_single_item_rejects_without_writes_or_truncation():
    client = _RecordingFirestoreClient()
    store = stores.FirestoreStateStore(client=client)
    giant = {"huge": {"summary": "x" * (900 * 1024)}}
    with pytest.raises(ValueError, match="Single ingestion item"):
        store.save_ingestion_cache_day(DATE, giant)
    assert client.docs == {}
    assert len(giant["huge"]["summary"]) == 900 * 1024


def test_preembedding_size_guard_reserves_vector_payload():
    source = {"large": {"summary": "x" * (799 * 1024)}}
    with pytest.raises(ValueError, match="Single ingestion item"):
        stores.pack_ingestion_cache_pages(DATE, source, reserve_embedding=True)


def test_atomic_conflict_does_not_partially_replace_pages():
    client = _RecordingFirestoreClient()
    store = stores.FirestoreStateStore(client=client)
    store.save_ingestion_cache_day(DATE, items(1))
    before = copy.deepcopy(client.docs)
    def conflict(writes):
        raise stores.PublicationConflict("Concurrent page generation")
    client.commit_documents = conflict
    with pytest.raises(stores.PublicationConflict):
        store.save_ingestion_cache_day(DATE, items(450))
    assert client.docs == before


def test_huge_source_is_rejected_before_embedding_call(monkeypatch, tmp_path):
    from src.zeitgeist import adk_pipeline as pipeline
    from tests.test_pipeline_launch_acceptance import _live_fixture
    store = _live_fixture(monkeypatch, tmp_path)
    cache = store.load_ingestion_cache()
    giant = dict(next(iter(cache.values())), url_hash="giant", summary="x" * (900 * 1024))
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", lambda *args, **kwargs: [giant])
    calls = []
    monkeypatch.setattr(pipeline, "embed_missing_items", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="Single ingestion item"):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True,
            now_epoch=datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc).timestamp())
    assert calls == []


def test_budgeted_batch_ceiling_rejects_before_any_remote_write():
    client = _RecordingFirestoreClient()
    store = stores.FirestoreStateStore(client=client)
    with pytest.raises(ValueError, match="atomic commit byte ceiling"):
        store.save_ingestion_cache_day(DATE, items(3000))
    assert client.docs == {}


def test_repack_retires_extra_pages_without_deleting_reusable_ids():
    client = _RecordingFirestoreClient()
    client.docs.update({"ingestion_cache/" + DATE: {"date": DATE, "items": items(1), "page_count": 2},
        "ingestion_cache/" + DATE + "__001": {"date": DATE, "items": {}}})
    store = stores.FirestoreStateStore(client=client)
    store.save_ingestion_cache_day(DATE, items(1, start=1))
    assert client.docs["ingestion_cache/" + DATE]["page_count"] == 1
    assert "ingestion_cache/" + DATE + "__001" in client.docs
    assert len(store.load_ingestion_cache(NOW)) == 2


def test_real_rest_packer_submits_all_pages_in_one_conditional_commit(monkeypatch):
    import io
    import json
    from urllib.error import HTTPError
    requests = []
    class Response(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
    def urlopen(request, **kwargs):
        requests.append(request)
        if request.method == "POST":
            count = len(json.loads(request.data)["writes"])
            return Response(json.dumps({"commitTime": "2026-10-30T00:00:00Z",
                "writeResults": [{"updateTime": "2026-10-30T00:00:00Z"}] * count}).encode())
        if request.full_url.endswith("pageSize=100"):
            return Response(b'{"documents":[]}')
        raise HTTPError(request.full_url, 404, "missing", {}, io.BytesIO())
    monkeypatch.setattr(stores.urllib.request, "urlopen", urlopen)
    adapter = stores.FirestoreStateStore(client=stores._FirestoreRestClient("fixture", access_token="fixture-only"))
    adapter.save_ingestion_cache_day(DATE, items(450))
    commits = [request for request in requests if request.method == "POST"]
    assert len(commits) == 1
    writes = json.loads(commits[0].data)["writes"]
    assert len(writes) > 1
    assert sum(len(write["update"]["fields"]["items"]["mapValue"]["fields"]) for write in writes) == 450
    assert all(write["currentDocument"] == {"exists": False} for write in writes)
    assert all(len(json.dumps({"fields": write["update"]["fields"]}).encode()) <= 800 * 1024 for write in writes)
