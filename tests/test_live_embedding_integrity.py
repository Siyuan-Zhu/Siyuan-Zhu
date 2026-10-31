"""Live semantic clustering never admits hash vectors or another model's space."""
from datetime import datetime, timedelta, timezone
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from src.zeitgeist import collectors, adk_pipeline as pipeline
from tests.test_pipeline_launch_acceptance import _live_fixture


def row(key="story", **extra):
    return dict(url_hash=key, title=key, summary="Physical craft and systems.", **extra)


def provider(failure=None):
    calls = []
    def embed(**kwargs):
        calls.append(kwargs)
        if failure:
            raise failure
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.25] * 768)
            for _ in kwargs["contents"]])
    return SimpleNamespace(models=SimpleNamespace(embed_content=embed)), calls


@pytest.mark.parametrize("bad", [None, "gemini-embedding-2-preview", "deterministic-sha256-v1"])
def test_live_cache_with_unknown_or_other_model_is_reembedded(bad):
    legacy = row(embedding_f16=np.ones(768, dtype=np.float16).tobytes(), embedding_source="gemini")
    if bad:
        legacy["embedding_model"] = bad
    client, calls = provider()
    result = collectors.embed_missing_items([row()], {"story": legacy}, client, require_live=True)
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-embedding-001"
    assert result[0]["embedding_model"] == "gemini-embedding-001"
    assert result[0]["embedding_source"] == "gemini"


def test_live_known_matching_cache_needs_no_paid_request():
    packed = collectors._deterministic_fallback_embedding("unit fixture only")
    cached = row(embedding_f16=packed, embedding_source="gemini", embedding_model="gemini-embedding-001")
    client, calls = provider()
    assert collectors.embed_missing_items([row()], {"story": cached}, client, require_live=True)[0]["embedding_f16"] == packed
    assert calls == []


@pytest.mark.parametrize("failure", [None, RuntimeError("provider unavailable")])
def test_live_missing_or_failed_provider_never_substitutes_hash_vectors(failure):
    client, calls = provider(failure)
    if failure is None:
        client = None
    cache = {}
    with pytest.raises(collectors.IngestionUnavailable, match="embedding"):
        collectors.embed_missing_items([row()], cache, client, require_live=True)
    assert cache == {}
    assert len(calls) <= 1


@pytest.mark.parametrize("values", [[0.0] * 768, [float("nan")] * 768, [1.0] * 767])
def test_live_invalid_vector_is_rejected(values):
    client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **kwargs:
        SimpleNamespace(embeddings=[SimpleNamespace(values=values)])))
    with pytest.raises(collectors.IngestionUnavailable, match="embedding"):
        collectors.embed_missing_items([row()], {}, client, require_live=True)


def test_live_collector_filters_dates_before_actual_embedding_call(monkeypatch, tmp_path):
    store = _live_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "embed_missing_items", collectors.embed_missing_items)
    now = datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc)
    cache = store.load_ingestion_cache()
    fresh_macro = copy.deepcopy(cache["m1_source_0"])
    stale = [dict(fresh_macro, url_hash=key, title=key, published_at=stamp) for key, stamp in (
        ("stale_macro", (now - timedelta(hours=97)).isoformat()),
        ("future", (now + timedelta(hours=1)).isoformat()), ("malformed", "invalid"))]
    stale.append(dict(fresh_macro, bucket="ORBIT", url_hash="stale_orbit", title="stale_orbit",
        published_at=(now - timedelta(hours=337)).isoformat()))
    retained_orbit = dict(fresh_macro, bucket="ORBIT", url_hash="weekly_orbit", title="weekly_orbit",
        published_at=(now - timedelta(hours=200)).isoformat())
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", lambda *a, **k: stale + [retained_orbit])
    client, calls = provider()
    persisted = {}
    store.save_ingestion_cache_day = lambda date, rows: persisted.update(rows)
    result = pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True,
        genai_client=client, now_epoch=now.timestamp())
    submitted = "\n".join(text for call in calls for text in call["contents"])
    assert "weekly_orbit" in submitted
    assert all(key not in submitted for key in ("stale_macro", "stale_orbit", "future", "malformed"))
    assert len(result) == 14
    assert all(item.embedding_f16 for item in result)
    assert set(cache).issubset(persisted)  # Legacy model provenance is durably refreshed.
    assert all(item["embedding_model"] == "gemini-embedding-001" for item in persisted.values())


def test_live_collector_provider_failure_cannot_reach_clusters(monkeypatch, tmp_path):
    store = _live_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "embed_missing_items", collectors.embed_missing_items)
    persisted = {}
    store.save_ingestion_cache_day = lambda date, rows: persisted.update(rows)
    client, calls = provider(RuntimeError("failure"))
    with pytest.raises(collectors.IngestionUnavailable, match="embedding"):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, genai_client=client,
            now_epoch=datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc).timestamp())
    assert len(calls) == 1 and persisted == {}


def test_offline_hash_vectors_are_explicitly_tagged():
    result = collectors.embed_missing_items([row()], {}, None)
    assert result[0]["embedding_source"] == "deterministic_fallback"
    assert result[0]["embedding_model"] == "deterministic-sha256-v1"
