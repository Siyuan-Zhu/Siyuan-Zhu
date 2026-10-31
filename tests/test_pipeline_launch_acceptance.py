"""Acceptance checks at the required-source and single-publication boundaries."""
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist.collectors import IngestionUnavailable

REQUIREMENTS = {"min_orbit_sources": 6, "min_macro_sources": 6,
                "required_orbit_tiers": ["O1", "O2", "O3"],
                "required_macro_tiers": ["M1", "M2", "M3"]}


def _live_fixture(monkeypatch, tmp_path, omit_tier=None, sparse=False, requirements=REQUIREMENTS):
    stamp = "2026-10-29T17:00:00Z"
    rows = {}
    for bucket, prefix in (("ORBIT", "O"), ("MACRO", "M")):
        for i in range(6):
            tier = prefix + str(i % 3 + 1)
            if tier == omit_tier or sparse:
                continue
            key = f"{tier.lower()}_source_{i}"
            rows[key] = {"url_hash": key, "source_id": key, "bucket": bucket, "tier": tier,
                         "published_at": stamp, "fetched_at": stamp, "title": key, "summary": "Craft and policy."}
    ig = {"url_hash": "ig", "source_id": "o3_ig_demo", "bucket": "ORBIT", "tier": "O3",
          "published_at": stamp, "fetched_at": stamp, "title": "Public Instagram post."}
    rows["ig"] = ig
    if sparse:
        rows["hn"] = {**ig, "url_hash": "hn", "source_id": "hn", "bucket": "MACRO", "tier": "M1"}
    config = {"curated_o3_instagram_handles": ["demo"]}
    if requirements is not None:
        config["coverage_requirements"] = requirements
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text(json.dumps(config))
    monkeypatch.setenv("APIFY_TOKEN", "dummy")
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "fetch_instagram_batch", lambda *a, **k: [ig])
    monkeypatch.setattr(pipeline, "fetch_macro_scout_signals", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "embed_missing_items", lambda rows, **k: rows)
    store = SimpleNamespace(load_ingestion_cache=lambda **k: dict(rows), load_orbit_roster=lambda: {})
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(pipeline, "datetime", Clock)
    return store


def test_afternoon_live_data_uses_actual_clock_and_full_diverse_coverage(monkeypatch, tmp_path):
    store = _live_fixture(monkeypatch, tmp_path)
    health = {}
    items = pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True,
                                      date_iso="2026-10-28", health=health)
    assert len(items) == 13
    assert health["publishable"] is True
    assert health["reference_time"].startswith("2026-10-29T17:30")
    assert health["nonsocial_orbit_source_count"] == health["nonsocial_macro_source_count"] == 6


@pytest.mark.parametrize("omit_tier,sparse", [(None, True), ("O2", False), ("M2", False)])
def test_sparse_or_missing_bucket_never_passes_live_publication(monkeypatch, tmp_path, omit_tier, sparse):
    store = _live_fixture(monkeypatch, tmp_path, omit_tier=omit_tier, sparse=sparse)
    with pytest.raises(IngestionUnavailable, match="diverse"):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True,
            now_epoch=datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc).timestamp())


def test_missing_coverage_config_blocks_before_instagram_request(monkeypatch, tmp_path):
    store = _live_fixture(monkeypatch, tmp_path, requirements=None)
    def unexpected(*a, **k):
        pytest.fail("No paid provider request is allowed before configuration validation")
    monkeypatch.setattr(pipeline, "fetch_instagram_batch", unexpected)
    with pytest.raises(IngestionUnavailable, match="coverage_requirements"):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True)


def test_seeded_corpus_requires_explicit_offline_demo(tmp_path, monkeypatch):
    monkeypatch.delenv("ZEITGEIST_OFFLINE_DEMO", raising=False)
    store = SimpleNamespace(load_ingestion_cache=lambda **k: {})
    with pytest.raises(IngestionUnavailable, match="explicit offline_demo"):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=False)
    health = {}
    assert pipeline.collect_96h_signals(tmp_path, store=store, live_collect=False, offline_demo=True, health=health)
    assert health["seeded"] is True
    assert health["publishable"] is False


def test_v3_output_never_calls_legacy_renderer_and_matches_actual_publication(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    def forbidden(*a, **k):
        pytest.fail("The legacy renderer cannot overwrite a v3 publication")
    monkeypatch.setattr(pipeline, "route_and_render_visual", forbidden)
    orchestrator = pipeline.ZeitgeistADKOrchestrator(tmp_path)
    analysis, decision, telemetry = orchestrator.run(date_iso="2026-10-28", offline_demo=True)
    actual = orchestrator.last_published_pulse
    assert analysis.date_iso == actual.date == "2026-10-28"
    assert analysis.word_of_the_day == decision.word_of_the_day == actual.word
    assert decision.explanation_phrase == actual.gloss
    assert decision.local_svg_path == str(tmp_path / actual.svg_asset_path)
    assert decision.composite_score == actual.composite_critic_score
    assert telemetry.estimated_monthly_cost_index is None and telemetry.within_budget is False


def test_unsupported_manual_override_fails_before_state_changes(tmp_path):
    with pytest.raises(ValueError, match="Visual mode"):
        pipeline.ZeitgeistADKOrchestrator(tmp_path).run(prefer_visual_mode="video")
    assert not list(tmp_path.iterdir())


def test_live_batches_use_v3_seed_rotation_independent_rss_and_ranked_scouts(monkeypatch, tmp_path):
    from collections import Counter
    import copy
    store = _live_fixture(monkeypatch, tmp_path)
    path = tmp_path / "config/orbit_sources.json"
    config = json.loads(path.read_text())
    seeds = [f"seed{i}" for i in range(12)]
    config.update(curated_o3_instagram_handles=seeds, handle_blocklist=["voice00"])
    path.write_text(json.dumps(config))
    roster = {"handles": {f"voice{i:02}": {"vitality_score": .9-i*.01,
        "last_vitality_decay_date": "2026-10-29"} for i in range(12)}, "curated_o3_cursor": 0}
    store.load_orbit_roster = lambda: copy.deepcopy(roster)
    def save(value):
        roster.clear()
        roster.update(copy.deepcopy(value))
    store.save_orbit_roster = save
    batches, rss = [], []
    def scrape(handles, **kwargs):
        batches.append(handles)
        return [{"url_hash": "live_ig", "source_id": "o3_ig_seed0", "bucket": "ORBIT", "tier": "O3",
                 "published_at": "2026-10-29T17:00:00Z", "fetched_at": "2026-10-29T17:00:00Z"}]
    monkeypatch.setattr(pipeline, "fetch_instagram_batch", scrape)
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", lambda *a, **k: rss.append(k.get("curated_o3_cursor")) or [])
    for day in (29, 30, 31):
        health = {}
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health,
            now_epoch=datetime(2026, 10, day, 17, 30, tzinfo=timezone.utc).timestamp())
        assert health["apify_batch_succeeded"] is True
    assert Counter(handle for batch in batches for handle in batch[:8]) == Counter({h:2 for h in seeds})
    assert batches[0][8:] == [f"voice{i:02}" for i in range(1,9)]
    assert rss == [None, None, None]  # V3 fetches every feed, independently of legacy RSS metadata.
    assert roster["curated_o3_cursor"] == 0 and roster["rss_o3_cursor"] == 0


def test_failed_live_batch_does_not_write_or_advance_roster(monkeypatch, tmp_path):
    import copy
    store = _live_fixture(monkeypatch, tmp_path)
    roster = {"curated_o3_cursor": 8, "rss_o3_cursor": 4, "handles": {"voice": {"vitality_score": .5}}}
    before = copy.deepcopy(roster)
    store.load_orbit_roster = lambda: copy.deepcopy(roster)
    def forbidden(*a):
        pytest.fail("An unknown/failed Instagram run cannot mutate its roster")
    store.save_orbit_roster = forbidden
    def fail(*a, **k):
        raise IngestionUnavailable("HTTP400 or unknown outcome")
    monkeypatch.setattr(pipeline, "fetch_instagram_batch", fail)
    health = {}
    with pytest.raises(IngestionUnavailable):
        pipeline.collect_96h_signals(tmp_path, store=store, live_collect=True, health=health)
    assert roster == before and health["apify_batch_succeeded"] is False


def test_orchestrator_credits_actual_winning_items_after_confirmed_batch(monkeypatch, tmp_path):
    from src.state_store import InMemoryStateStore
    from src.zeitgeist.methodology import ScoredCluster
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    store = InMemoryStateStore()
    store.save_orbit_roster({"handles": {"voice": {"vitality_score": .5, "tier": "O4_SCOUT"}},
                            "curated_o3_cursor": 8, "rss_o3_cursor": 2})
    item = SimpleNamespace(source_id="o4_ig_voice", bucket="ORBIT", tier="O4_SCOUT",
                           title="Material craft", summary="A quiet architectural study")
    def confirmed(*args, **kwargs):
        kwargs["health"]["apify_batch_succeeded"] = True
        return [item]
    cluster = ScoredCluster("actual_winner", [item], ["craft"], .8, 1, 1, 1, 1, 1, .8,
                            ("o4_ig_voice",), 1, 1)
    monkeypatch.setattr(pipeline, "collect_96h_signals", confirmed)
    monkeypatch.setattr(pipeline, "score_and_cluster_signals", lambda *a, **k: [cluster])
    orchestrator = pipeline.ZeitgeistADKOrchestrator(tmp_path, store)
    for _ in range(2):
        orchestrator.run(date_iso="2026-10-30", live_collect=True, offline_demo=False,
                         now_epoch=datetime(2026, 10, 30, 12, tzinfo=timezone.utc).timestamp())
        assert store.load_orbit_roster()["handles"]["voice"]["vitality_score"] == pytest.approx(.85)
