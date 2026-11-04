"""A required-source outage must stop before buying unusable embeddings."""
import copy
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist.collectors import IngestionUnavailable
from tests.test_pipeline_launch_acceptance import _live_fixture


@pytest.mark.parametrize('missing', ['instagram', 'O2', 'M2'])
def test_required_coverage_failure_precedes_embedding_and_cache_write(tmp_path, monkeypatch, missing):
    store = _live_fixture(monkeypatch, tmp_path, omit_tier=None if missing == 'instagram' else missing)
    rows = copy.deepcopy(store.load_ingestion_cache())
    if missing == 'instagram':
        rows.pop('ig')
    store.load_ingestion_cache = lambda **kwargs: copy.deepcopy(rows)
    monkeypatch.setattr(pipeline, 'fetch_instagram_batch', lambda *a, **k: [])
    calls = []
    monkeypatch.setattr(pipeline, 'embed_missing_items', lambda rows, **kwargs: calls.append('embedding') or rows)
    store.save_ingestion_cache_day = lambda *a, **k: calls.append('cache_write')
    health = {}
    with pytest.raises(IngestionUnavailable, match='diverse') as error:
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health)
    assert calls == []
    assert health['publishable'] is False
    assert error.value.public_collection_health['publishable'] is False


def test_empty_batch_can_use_existing_fresh_instagram_without_weakening_gate(tmp_path, monkeypatch):
    store = _live_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, 'fetch_instagram_batch', lambda *a, **k: [])
    health = {}
    items = pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health)
    assert len(items) == 13 and health['publishable'] is True
    assert health['instagram'] == 'empty' and health['instagram_item_count'] == 1


def test_post_embedding_check_still_rejects_loss_of_required_source(tmp_path, monkeypatch):
    store = _live_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(pipeline, 'embed_missing_items',
        lambda rows, **kwargs: [row for row in rows if '_ig_' not in row['source_id']])
    with pytest.raises(IngestionUnavailable, match='diverse'):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True)


@pytest.mark.parametrize('hours,eligible', [(72, True), (72.001, False), (-0.001, False)])
def test_retained_instagram_fetch_window_is_not_extended(tmp_path, monkeypatch, hours, eligible):
    store = _live_fixture(monkeypatch, tmp_path)
    rows = copy.deepcopy(store.load_ingestion_cache())
    rows['ig']['fetched_at'] = (datetime(2026,10,29,17,30,tzinfo=timezone.utc)-timedelta(hours=hours)).isoformat()
    store.load_ingestion_cache = lambda **kwargs: copy.deepcopy(rows)
    monkeypatch.setattr(pipeline, 'fetch_instagram_batch', lambda *a, **k: [])
    calls=[]
    monkeypatch.setattr(pipeline, 'embed_missing_items', lambda rows, **kwargs: calls.append(1) or rows)
    if eligible:
        pipeline.collect_96h_signals(tmp_path,store=store,live_collect=True)
        assert calls == [1]
    else:
        with pytest.raises(IngestionUnavailable):pipeline.collect_96h_signals(tmp_path,store=store,live_collect=True)
        assert calls == []


def test_successful_apify_error_dataset_stops_real_collector_before_embedding(tmp_path, monkeypatch):
    import httpx
    from unittest.mock import MagicMock
    from src.zeitgeist import collectors
    from tests.test_ingestion_release_safety import ApprovedGuard
    store = _live_fixture(monkeypatch,tmp_path)
    rows=store.load_ingestion_cache();rows.pop('ig')
    store.load_ingestion_cache=lambda **kwargs:copy.deepcopy(rows)
    response=httpx.Response(200,json=[{'username':'demo','url':'https://www.instagram.com/demo',
        'error':'not_found','errorDescription':'Profile does not exist'}],request=httpx.Request('POST','https://api.apify.com'))
    client=MagicMock();client.__enter__.return_value.post.return_value=response
    monkeypatch.setattr(collectors.httpx,'Client',lambda **kwargs:client)
    monkeypatch.setattr('src.zeitgeist.spend_guard.get_spend_guard',lambda **kwargs:ApprovedGuard())
    monkeypatch.setattr(pipeline,'fetch_instagram_batch',collectors.fetch_instagram_batch)
    monkeypatch.setenv('APIFY_INSTAGRAM_BUILD','0.0.648')
    calls=[]
    monkeypatch.setattr(pipeline,'embed_missing_items',lambda rows,**kwargs:calls.append(1) or rows)
    health={}
    with pytest.raises(IngestionUnavailable):pipeline.collect_96h_signals(tmp_path,store=store,live_collect=True,health=health)
    assert health['instagram_dataset_record_count']==health['instagram_unavailable_handle_count']==1
    assert health['instagram_item_count']==0 and calls==[]
    assert client.__enter__.return_value.post.call_count==1

