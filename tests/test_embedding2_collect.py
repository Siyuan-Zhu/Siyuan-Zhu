"""Live embedding under gemini-embedding-2: task prefix, exact model tag, unchanged 001 requests."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import state_store as stores
from src.zeitgeist import collectors
from src.zeitgeist.spend_guard import SpendBlocked, refused_before_dispatch

DATE = "2026-10-30"
PREFIX = "task: clustering | query: "
E2 = "gemini-embedding-2"
E1 = "gemini-embedding-001"


def row(key="story", title=None, **extra):
    return dict(url_hash=key, title=title or key, summary="Synthetic systems summary.", **extra)


def provider(failure=None):
    calls = []
    def embed(**kwargs):
        calls.append(kwargs)
        if failure:
            raise failure
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.25] * 768)
            for _ in kwargs["contents"]])
    return SimpleNamespace(models=SimpleNamespace(embed_content=embed)), calls


def cached(model):
    return row(embedding_f16=np.ones(768, dtype=np.float16).tobytes(),
        embedding_source="gemini", embedding_model=model)


def test_constants_name_both_live_models_and_the_exact_prefix():
    assert collectors.EMBEDDING_2_TASK_PREFIX == PREFIX
    assert collectors.LIVE_EMBEDDING_MODELS == frozenset({E1, E2})


def test_embedding_2_sends_prefixed_001_text_and_tags_rows():
    items = [row("alpha", title="Alpha"), row("beta", title="Beta")]
    client, calls = provider()
    result = collectors.embed_missing_items(items, {}, client, model=E2, require_live=True)
    sent = [text for call in calls for text in call["contents"]]
    assert sent == [PREFIX + "Alpha\nSynthetic systems summary.", PREFIX + "Beta\nSynthetic systems summary."]
    assert all(call["model"] == E2 for call in calls)
    assert all(call["config"] == {"output_dimensionality": 768} for call in calls)
    assert [r["embedding_model"] for r in result] == [E2, E2]
    assert [r["embedding_source"] for r in result] == ["gemini", "gemini"]


def test_embedding_001_request_carries_no_prefix():
    client, calls = provider()
    collectors.embed_missing_items([row("alpha", title="Alpha")], {}, client, model=E1, require_live=True)
    assert calls == [{"model": E1, "contents": ["Alpha\nSynthetic systems summary."],
        "config": {"output_dimensionality": 768}}]


def test_embedding_2_single_retry_is_prefixed_too():
    calls = []
    def embed(**kwargs):
        calls.append(kwargs)
        if len(kwargs["contents"]) > 1:
            raise RuntimeError("synthetic batch failure")
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.25] * 768)])
    client = SimpleNamespace(models=SimpleNamespace(embed_content=embed))
    result = collectors.embed_missing_items([row("alpha", title="Alpha"), row("beta", title="Beta")],
        {}, client, model=E2)
    assert len(calls) == 3
    singles = [call["contents"] for call in calls[1:]]
    assert singles == [[PREFIX + "Alpha\nSynthetic systems summary."], [PREFIX + "Beta\nSynthetic systems summary."]]
    assert all(call["config"] == {"output_dimensionality": 768} for call in calls)
    assert [r["embedding_model"] for r in result] == [E2, E2]


def test_embedding_2_cache_row_is_reused_under_embedding_2_only():
    client, calls = provider()
    result = collectors.embed_missing_items([row()], {"story": cached(E2)}, client, model=E2, require_live=True)
    assert calls == []
    assert result[0]["embedding_model"] == E2
    client, calls = provider()
    result = collectors.embed_missing_items([row()], {"story": cached(E2)}, client, model=E1, require_live=True)
    assert len(calls) == 1
    assert calls[0]["model"] == E1
    assert result[0]["embedding_model"] == E1


def test_001_cache_row_is_reembedded_under_embedding_2():
    client, calls = provider()
    result = collectors.embed_missing_items([row()], {"story": cached(E1)}, client, model=E2, require_live=True)
    assert len(calls) == 1
    assert calls[0]["model"] == E2
    assert calls[0]["contents"] == [PREFIX + "story\nSynthetic systems summary."]
    assert result[0]["embedding_model"] == E2


def test_other_live_model_is_still_a_mismatch_with_no_call():
    client, calls = provider()
    with pytest.raises(collectors.IngestionUnavailable) as caught:
        collectors.embed_missing_items([row()], {}, client, model="gemini-embedding-2-preview", require_live=True)
    assert caught.value.public_diagnostics["code"] == "embedding-model-mismatch"
    assert calls == []


def test_spend_blocked_before_dispatch_propagates_unchanged():
    refusal = SpendBlocked("Unpriced SDK request option")
    refusal.refused_before_dispatch = True
    client, calls = provider(refusal)
    with pytest.raises(SpendBlocked) as caught:
        collectors.embed_missing_items([row()], {}, client, model=E2, require_live=True)
    assert caught.value is refusal
    assert refused_before_dispatch(caught.value)
    assert len(calls) == 1


def test_reserve_tags_the_model_in_force():
    source = {"a": {"summary": "x"}, "b": {"summary": "y"}}
    pages = stores.pack_ingestion_cache_pages(DATE, source, reserve_embedding=True, embedding_model=E2)
    packed = [r for page in pages.values() for r in page["items"].values()]
    assert len(packed) == 2
    assert all(r["embedding_model"] == E2 and r["embedding_source"] == "gemini" for r in packed)
    default = stores.pack_ingestion_cache_pages(DATE, source, reserve_embedding=True)
    assert all(r["embedding_model"] == E1 for page in default.values() for r in page["items"].values())
