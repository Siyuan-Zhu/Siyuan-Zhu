"""Collection under an embedding run: migrated, switch-over night, fallback to 001."""
from datetime import datetime, timezone
import copy
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from src import state_store as stores
from src.zeitgeist import collectors, embedding_migration, adk_pipeline as pipeline
from src.zeitgeist.embedding_migration import EMBED_001, EMBED_2, EmbeddingRun
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_pipeline_launch_acceptance import _live_fixture

PREFIX = "task: clustering | query: "
NOW = datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc).timestamp()
VECTORS = {EMBED_001: [0.25] * 768, EMBED_2: [0.5 if i % 2 else -0.125 for i in range(768)]}


def provider(fail=None):
    """Fake client whose vectors depend on the model; fail maps a model to the exception it raises."""
    calls = []
    def embed(**kwargs):
        calls.append(kwargs)
        failure = (fail or {}).get(kwargs["model"])
        if failure is not None:
            raise failure
        return SimpleNamespace(embeddings=[SimpleNamespace(values=VECTORS[kwargs["model"]])
            for _ in kwargs["contents"]])
    return SimpleNamespace(models=SimpleNamespace(embed_content=embed)), calls


def packed(model):
    client, _ = provider()
    return collectors.embed_missing_items([{"url_hash": "unit", "title": "unit"}], {}, client,
        model=model, require_live=True)[0]["embedding_f16"]


def refusal(marked):
    error = SpendBlocked("Unpriced SDK request option")
    if marked:
        error.refused_before_dispatch = True
    return error


@pytest.fixture
def live(monkeypatch, tmp_path):
    store = _live_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, "embed_missing_items", collectors.embed_missing_items)
    persisted = {}
    store.save_ingestion_cache_day = lambda date, rows: persisted.update(copy.deepcopy(rows))
    reserves = []
    real_pack = stores.pack_ingestion_cache_pages
    def pack(*args, **kwargs):
        reserves.append(kwargs)
        return real_pack(*args, **kwargs)
    monkeypatch.setattr(stores, "pack_ingestion_cache_pages", pack)
    # "ig" is too short to search for without false hits ("config"); every other key is checked.
    secrets = [key for key in sorted(store.load_ingestion_cache()) if len(key) > 2]
    secrets += ["Craft and policy", "Public Instagram post"]
    return SimpleNamespace(store=store, tmp_path=tmp_path, persisted=persisted, reserves=reserves,
        secrets=secrets)


def collect(live, client, run=None):
    kwargs = {} if run is None else {"embedding_run": run}
    return pipeline.collect_96h_signals(live.tmp_path, store=live.store, live_collect=True,
        genai_client=client, now_epoch=NOW, **kwargs)


def texts(calls, model):
    # Embedding-2 items are one-text Contents; 001 items are bare strings.
    return [item if isinstance(item, str) else item.parts[0].text
            for call in calls if call["model"] == model for item in call["contents"]]


@pytest.fixture(autouse=True)
def no_private_text(live, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    yield
    out = capsys.readouterr()
    seen = out.out + out.err + "\n".join(r.getMessage() for r in caplog.records)
    assert all(secret not in seen for secret in live.secrets)


def test_no_run_is_todays_001_path(live):
    client, calls = provider()
    result = collect(live, client)
    assert calls and all(call["model"] == EMBED_001 for call in calls)
    assert not any(text.startswith(PREFIX) for text in texts(calls, EMBED_001))
    assert live.persisted and all(r["embedding_model"] == EMBED_001 for r in live.persisted.values())
    assert all(item.embedding_f16 == packed(EMBED_001) for item in result)
    assert live.reserves == [{"reserve_embedding": True}]


def test_legacy_run_is_todays_001_path(live):
    client, calls = provider()
    run = EmbeddingRun(mode="legacy", calibration_state=embedding_migration.CAL_REJECTED)
    result = collect(live, client, run)
    assert calls and all(call["model"] == EMBED_001 for call in calls)
    assert all(r["embedding_model"] == EMBED_001 for r in live.persisted.values())
    assert live.reserves == [{"reserve_embedding": True}]
    assert run.used == EMBED_001 and not run.fallback and not run.deferred
    assert run.items == len(result) and run.date_str == "2026-10-29"
    assert run.pending_rows is None and run.pending_items is None


def test_migrated_embeds_and_caches_embedding_2_only(live):
    client, calls = provider()
    run = EmbeddingRun(mode="migrated", calibration_state=embedding_migration.CAL_ACCEPTED,
        percentiles=(0.9, 0.8, 0.7, 0.95))
    result = collect(live, client, run)
    assert calls and all(call["model"] == EMBED_2 for call in calls)
    assert all(text.startswith(PREFIX) for text in texts(calls, EMBED_2))
    assert len(texts(calls, EMBED_2)) == len(result)
    assert set(live.persisted) == {item.url_hash for item in result}
    assert all(r["embedding_model"] == EMBED_2 for r in live.persisted.values())
    assert all(item.embedding_f16 == packed(EMBED_2) for item in result)
    assert live.reserves == [{"reserve_embedding": True, "embedding_model": EMBED_2}]
    assert run.used == EMBED_2 and not run.fallback
    assert run.items == len(result) and run.date_str == "2026-10-29"


@pytest.mark.parametrize("failure", [RuntimeError("synthetic provider failure"), refusal(marked=True)])
def test_migrated_embedding_2_failure_falls_back_to_001_in_memory(live, failure):
    client, calls = provider({EMBED_2: failure})
    run = EmbeddingRun(mode="migrated", calibration_state=embedding_migration.CAL_ACCEPTED,
        percentiles=(0.9, 0.8, 0.7, 0.95))
    result = collect(live, client, run)
    models = [call["model"] for call in calls]
    assert models[0] == EMBED_2 and EMBED_001 in models
    assert models.index(EMBED_001) > max(i for i, m in enumerate(models) if m == EMBED_2)
    assert live.persisted == {}
    assert run.fallback and run.used == EMBED_001
    assert result and all(item.embedding_f16 == packed(EMBED_001) for item in result)
    assert len(texts(calls, EMBED_001)) == len(result) == run.items


def test_migrated_spend_blocked_after_dispatch_propagates(live):
    error = refusal(marked=False)
    client, calls = provider({EMBED_2: error})
    run = EmbeddingRun(mode="migrated", calibration_state=embedding_migration.CAL_ACCEPTED,
        percentiles=(0.9, 0.8, 0.7, 0.95))
    with pytest.raises(SpendBlocked) as caught:
        collect(live, client, run)
    assert caught.value is error
    assert all(call["model"] == EMBED_2 for call in calls)
    assert live.persisted == {}
    assert not run.fallback


def test_migrated_001_fallback_failure_propagates(live):
    client, calls = provider({EMBED_2: RuntimeError("synthetic"), EMBED_001: RuntimeError("synthetic")})
    run = EmbeddingRun(mode="migrated", calibration_state=embedding_migration.CAL_ACCEPTED,
        percentiles=(0.9, 0.8, 0.7, 0.95))
    with pytest.raises(collectors.IngestionUnavailable, match="embedding"):
        collect(live, client, run)
    assert live.persisted == {}


def test_switch_over_caches_001_and_holds_embedding_2_pending(live, monkeypatch):
    monkeypatch.setattr(embedding_migration, "MIN_CALIBRATION_ITEMS", 2)
    client, calls = provider()
    run = EmbeddingRun(mode="switch_over", calibration_state=embedding_migration.CAL_NONE)
    result = collect(live, client, run)
    models = [call["model"] for call in calls]
    first_e2 = models.index(EMBED_2)
    assert EMBED_001 in models[:first_e2] and EMBED_001 not in models[first_e2:]
    assert live.persisted and all(r["embedding_model"] == EMBED_001 for r in live.persisted.values())
    assert all(item.embedding_f16 == packed(EMBED_001) for item in result)
    e2_texts = texts(calls, EMBED_2)
    assert len(e2_texts) == len(set(e2_texts)) == len(result)
    assert all(text.startswith(PREFIX) for text in e2_texts)
    assert set(run.pending_rows) == {item.url_hash for item in result}
    assert all(r["embedding_model"] == EMBED_2 and r["embedding_source"] == "gemini"
        for r in run.pending_rows.values())
    assert [item.url_hash for item in run.pending_items] == [item.url_hash for item in result]
    assert all(item.embedding_f16 == packed(EMBED_2) for item in run.pending_items)
    assert live.reserves == [{"reserve_embedding": True}]
    assert run.used == EMBED_001 and not run.fallback and not run.deferred
    assert run.items == len(result) and run.date_str == "2026-10-29"


def test_switch_over_with_too_few_items_defers_with_no_embedding_2_call(live):
    assert embedding_migration.MIN_CALIBRATION_ITEMS == 100
    client, calls = provider()
    run = EmbeddingRun(mode="switch_over", calibration_state=embedding_migration.CAL_NONE)
    result = collect(live, client, run)
    assert result and all(call["model"] == EMBED_001 for call in calls)
    assert run.deferred and not run.fallback
    assert run.pending_rows is None and run.pending_items is None
    assert all(r["embedding_model"] == EMBED_001 for r in live.persisted.values())


@pytest.mark.parametrize("failure", [RuntimeError("synthetic provider failure"), refusal(marked=True)])
def test_switch_over_embedding_2_failure_keeps_001(live, monkeypatch, failure):
    monkeypatch.setattr(embedding_migration, "MIN_CALIBRATION_ITEMS", 2)
    client, calls = provider({EMBED_2: failure})
    run = EmbeddingRun(mode="switch_over", calibration_state=embedding_migration.CAL_NONE)
    result = collect(live, client, run)
    assert result and all(item.embedding_f16 == packed(EMBED_001) for item in result)
    assert run.fallback and run.used == EMBED_001
    assert run.pending_rows is None and run.pending_items is None
    assert all(r["embedding_model"] == EMBED_001 for r in live.persisted.values())


def test_switch_over_spend_blocked_after_dispatch_propagates(live, monkeypatch):
    monkeypatch.setattr(embedding_migration, "MIN_CALIBRATION_ITEMS", 2)
    error = refusal(marked=False)
    client, _ = provider({EMBED_2: error})
    run = EmbeddingRun(mode="switch_over", calibration_state=embedding_migration.CAL_NONE)
    with pytest.raises(SpendBlocked) as caught:
        collect(live, client, run)
    assert caught.value is error
    assert run.pending_rows is None


def test_unset_variable_after_migration_reembeds_embedding_2_cache_with_001(live):
    """Review Focus 3: every cached row carries an embedding-2 vector; no run object."""
    e2 = packed(EMBED_2)
    tagged = {key: dict(row, embedding_f16=e2, embedding_source="gemini", embedding_model=EMBED_2)
        for key, row in live.store.load_ingestion_cache().items()}
    live.store.load_ingestion_cache = lambda **k: copy.deepcopy(tagged)
    client, calls = provider()
    result = collect(live, client)
    assert all(call["model"] == EMBED_001 for call in calls)
    assert len(texts(calls, EMBED_001)) == len(result) == len(tagged)
    assert set(live.persisted) == set(tagged)
    assert all(r["embedding_model"] == EMBED_001 for r in live.persisted.values())
    assert all(item.embedding_f16 == packed(EMBED_001) for item in result)


def test_migrated_reuses_cached_embedding_2_rows(live):
    e2 = packed(EMBED_2)
    tagged = {key: dict(row, embedding_f16=e2, embedding_source="gemini", embedding_model=EMBED_2)
        for key, row in live.store.load_ingestion_cache().items()}
    live.store.load_ingestion_cache = lambda **k: copy.deepcopy(tagged)
    client, calls = provider()
    run = EmbeddingRun(mode="migrated", calibration_state=embedding_migration.CAL_ACCEPTED,
        percentiles=(0.9, 0.8, 0.7, 0.95))
    result = collect(live, client, run)
    assert calls == []  # every vector, the re-collected live row's included, is reused
    assert set(live.persisted) == {"ig"}  # the live row is refreshed in today's shard, as on 001
    assert live.persisted["ig"]["embedding_model"] == EMBED_2
    assert all(item.embedding_f16 == e2 for item in result) and run.used == EMBED_2
