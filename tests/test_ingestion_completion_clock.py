"""Live collection validates timestamps against observed completion, without skew."""
import copy
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist.collectors import IngestionUnavailable
from tests.test_pipeline_launch_acceptance import _live_fixture

START = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
COMPLETE = START + timedelta(seconds=30)


def _advancing_sources(monkeypatch, tmp_path, *, fetched_offset=10, published_offset=-3600):
    store = _live_fixture(monkeypatch, tmp_path)
    cache = store.load_ingestion_cache()
    cache.pop("ig")  # No older Instagram row may conceal a failed fresh batch.
    store.load_ingestion_cache = lambda **kwargs: copy.deepcopy(cache)
    clock = [START]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock[0]
    monkeypatch.setattr(pipeline, "datetime", Clock)
    def feeds(*args, **kwargs):
        clock[0] = START + timedelta(seconds=5)
        return []
    def instagram(*args, **kwargs):
        clock[0] = START + timedelta(seconds=10)
        return [{"url_hash": "just_collected_ig", "source_id": "o3_ig_demo", "bucket": "ORBIT", "tier": "O3",
                 "title": "Just collected", "published_at": (START + timedelta(seconds=published_offset)).isoformat(),
                 "fetched_at": (START + timedelta(seconds=fetched_offset)).isoformat()}]
    def macro(*args, **kwargs):
        clock[0] = COMPLETE
        return []
    embedded = []
    def embed(rows, **kwargs):
        embedded.extend(rows)
        return rows
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", feeds)
    monkeypatch.setattr(pipeline, "fetch_instagram_batch", instagram)
    monkeypatch.setattr(pipeline, "fetch_macro_scout_signals", macro)
    monkeypatch.setattr(pipeline, "embed_missing_items", embed)
    return store, embedded


def test_live_fetch_after_start_is_fresh_at_observed_collection_completion(monkeypatch, tmp_path):
    store, _ = _advancing_sources(monkeypatch, tmp_path)
    health = {}
    items = pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health)
    assert any(item.url_hash == "just_collected_ig" for item in items)
    assert health["instagram_item_count"] == 1 and health["publishable"] is True
    assert health["reference_time"] == COMPLETE.isoformat()


@pytest.mark.parametrize("future_field", ["fetched", "published"])
def test_timestamps_after_observed_completion_remain_future(monkeypatch, tmp_path, future_field):
    kwargs = {"fetched_offset": 31} if future_field == "fetched" else {"published_offset": 31}
    store, embedded = _advancing_sources(monkeypatch, tmp_path, **kwargs)
    health = {}
    with pytest.raises(IngestionUnavailable, match="diverse"):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health)
    assert health["instagram_item_count"] == 0 and health["publishable"] is False
    if future_field == "published":
        assert all(row["url_hash"] != "just_collected_ig" for row in embedded)


def test_explicit_reference_clock_stays_deterministic_while_collectors_advance(monkeypatch, tmp_path):
    store, _ = _advancing_sources(monkeypatch, tmp_path)
    health = {}
    with pytest.raises(IngestionUnavailable, match="diverse"):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health,
                                     now_epoch=START.timestamp())
    assert health["reference_time"] == START.isoformat()
    assert health["instagram_item_count"] == 0


def test_daily_orchestrator_does_not_turn_its_live_start_into_a_fixed_replay_clock(monkeypatch, tmp_path):
    from src.state_store import InMemoryStateStore
    source_store, _ = _advancing_sources(monkeypatch, tmp_path, published_offset=20)
    store = InMemoryStateStore()
    store.load_ingestion_cache = source_store.load_ingestion_cache
    store.load_orbit_roster = source_store.load_orbit_roster
    scoring_times = []
    actual_score = pipeline.score_and_cluster_signals
    def score(*args, **kwargs):
        scoring_times.append(kwargs["now_utc"])
        return actual_score(*args, **kwargs)
    monkeypatch.setattr(pipeline, "score_and_cluster_signals", score)
    _, _, telemetry = pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
        date_iso=START.date().isoformat(), live_collect=True, offline_demo=False)
    assert telemetry.collection_health["instagram_item_count"] == 1
    assert telemetry.collection_health["reference_time"] == COMPLETE.isoformat()
    assert scoring_times == [COMPLETE]


@pytest.mark.parametrize("instagram_cursor,rss_cursor", [(0, 0), (8, 2), (4, 4)])
def test_live_v3_fetches_all_sixty_curated_feeds_regardless_legacy_cursors(monkeypatch, tmp_path, instagram_cursor, rss_cursor):
    from src.zeitgeist import collectors
    store, _ = _advancing_sources(monkeypatch, tmp_path)
    config = json.loads((Path(pipeline.__file__).resolve().parents[2] / "config/orbit_sources.json").read_text())
    (tmp_path / "config/orbit_sources.json").write_text(json.dumps(config))
    store.load_orbit_roster = lambda: {"handles": {}, "curated_o3_cursor": instagram_cursor,
                                     "rss_o3_cursor": rss_cursor}
    selected = []
    def fetch_worker(client, feed, timeout, now):
        selected.append(feed)
        return [{"url_hash": feed["id"], "source_id": feed["id"], "bucket": feed["bucket"],
                 "tier": feed["tier"], "title": "A curated signal", "origin": feed.get("origin"),
                 "published_at": (START - timedelta(hours=1)).isoformat()}]
    # Exercise actual selection/concurrency; only the network boundary is replaced.
    monkeypatch.setattr(collectors, "_fetch_single_feed", fetch_worker)
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", collectors.fetch_all_curated_feeds)
    health = {}
    pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health)
    assert {feed["id"] for feed in selected} == {feed["id"] for feed in config["feeds"]}
    assert Counter(feed["tier"] for feed in selected) == {tier: 10 for tier in ("O1", "O2", "O3", "M1", "M2", "M3")}
    assert health["publishable"] is True
