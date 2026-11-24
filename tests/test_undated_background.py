"""Items whose feed gives no publish date count as background, never as new momentum (switch-gated)."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import math
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest

from src.zeitgeist import collectors
from src.zeitgeist.adk_pipeline import _row_to_cached_item
from src.zeitgeist.methodology import (
    UNDATED_ENV,
    CachedItem,
    cluster_and_score,
    item_date_known,
    score_and_cluster_signals,
    undated_as_background_enabled,
)
from tests.test_ingestion_release_safety import ApprovedGuard
from tests.test_run_record_wiring import _job_lines

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
STAMP = "%Y-%m-%dT%H:%M:%SZ"
ROOT = Path(__file__).resolve().parents[1]
DOLLAR = chr(36)
SWITCH_LINE = UNDATED_ENV + ": " + DOLLAR + "{{ vars." + UNDATED_ENV + " || '0' }}"
CANDIDATE_LINE = "ZEITGEIST_REFLECTOR_CANDIDATE: " + DOLLAR + "{{ vars.ZEITGEIST_REFLECTOR_CANDIDATE || '0' }}"


def _rss(*items: str) -> str:
    return '<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>' + "".join(items) + "</channel></rss>"


def _rss_item(n: int, date: str | None) -> str:
    when = f"<pubDate>{date}</pubDate>" if date is not None else ""
    return f"<item><title>Synthetic {n}</title><link>https://example.org/p/{n}</link>{when}</item>"


FEED_CFG = {"id": "o1_synthetic", "name": "Synthetic", "tier": "O1", "bucket": "ORBIT", "type": "rss"}


def _by_url(items):
    return {item["url"].rsplit("/", 1)[1]: item for item in items}


# ---- collection records whether the date was real ----


def test_rss_entry_without_a_date_is_marked_undated_and_a_dated_one_known() -> None:
    xml = _rss(_rss_item(1, None), _rss_item(2, "Thu, 19 Nov 2026 08:00:00 GMT"), _rss_item(3, "not a date"))
    items = _by_url(collectors._normalize_feed_entries(collectors._parse_xml_feed_entries(xml, NOW), FEED_CFG, NOW))
    assert items["1"]["date_known"] is False
    assert items["1"]["published_at"] == items["1"]["fetched_at"] == NOW.strftime(STAMP)
    assert items["2"]["date_known"] is True
    assert items["2"]["published_at"] == "2026-11-19T08:00:00Z"
    # An unreadable date falls back to the fetch time just as a missing one does.
    assert items["3"]["date_known"] is False


def test_atom_entry_without_a_date_is_marked_undated() -> None:
    xml = ('<feed xmlns="http://www.w3.org/2005/Atom">'
           '<entry><title>A</title><link href="https://example.org/a/1"/></entry>'
           '<entry><title>B</title><link href="https://example.org/a/2"/><updated>2026-11-18T06:00:00Z</updated></entry>'
           "</feed>")
    items = _by_url(collectors._normalize_feed_entries(collectors._parse_xml_feed_entries(xml, NOW), FEED_CFG, NOW))
    assert items["1"]["date_known"] is False
    assert items["2"]["date_known"] is True


def test_json_feed_entry_without_a_date_is_marked_undated() -> None:
    payload = {"items": [{"id": "1", "title": "A", "url": "https://example.org/j/1"},
                         {"id": "2", "title": "B", "url": "https://example.org/j/2",
                          "date_published": "2026-11-18T06:00:00Z"}]}
    items = _by_url(collectors._normalize_feed_entries(collectors._parse_json_feed_entries(payload, NOW), FEED_CFG, NOW))
    assert items["1"]["date_known"] is False
    assert items["2"]["date_known"] is True


def test_the_curated_fetch_and_the_macro_scouts_carry_the_mark(monkeypatch) -> None:
    response = MagicMock(text=_rss(_rss_item(1, None), _rss_item(2, "Thu, 19 Nov 2026 08:00:00 GMT")))
    client = MagicMock()
    client.get.return_value = response
    monkeypatch.setattr(collectors.httpx, "Client", lambda **kwargs: MagicMock(__enter__=lambda s: client,
                                                                             __exit__=lambda s, *a: False))
    curated = _by_url(collectors.fetch_all_curated_feeds({"feeds": [dict(FEED_CFG, url="https://example.org/f")]}))
    assert (curated["1"]["date_known"], curated["2"]["date_known"]) == (False, True)
    scouts = collectors.fetch_macro_scout_signals()
    assert scouts and {item["date_known"] for item in scouts} == {False, True}
    assert all(item["date_known"] is (item["url"].endswith("/2")) for item in scouts)


def _instagram(monkeypatch, posts):
    response = httpx.Response(200, json=posts, request=httpx.Request("POST", "https://api.apify.com"))
    client = MagicMock()
    client.__enter__.return_value.post.return_value = response
    monkeypatch.setattr(collectors.httpx, "Client", lambda **kwargs: client)
    return collectors.fetch_instagram_batch(["requested_a"], "not-a-real-token", spend_guard=ApprovedGuard(),
                                            actor_build="0.0.648")


def test_instagram_post_without_a_timestamp_is_marked_undated(monkeypatch) -> None:
    posts = [{"ownerUsername": "requested_a", "shortCode": "undated", "caption": "Synthetic."},
             {"ownerUsername": "requested_a", "shortCode": "dated", "caption": "Synthetic.",
              "timestamp": "2026-11-18T17:00:00Z"}]
    items = _by_url(_instagram(monkeypatch, posts))
    assert items["undated"]["date_known"] is False
    assert items["undated"]["published_at"] == items["undated"]["fetched_at"]
    assert items["dated"]["date_known"] is True


def test_the_cache_schema_keeps_the_mark_and_derives_it_for_old_rows() -> None:
    assert "date_known" in collectors.FEED_CACHE_ALLOWED_KEYS
    base = {"url": "https://example.org/x", "source_id": "o1_synthetic", "tier": "O1"}
    old_undated = dict(base, published_at="2026-11-19T10:00:00Z", fetched_at="2026-11-19T10:00:00Z")
    old_dated = dict(base, published_at="2026-11-18T10:00:00Z", fetched_at="2026-11-19T10:00:00Z")
    assert collectors.sanitize_feed_cache_item(old_undated)["date_known"] is False
    assert collectors.sanitize_feed_cache_item(old_dated)["date_known"] is True
    assert collectors.sanitize_feed_cache_item(dict(old_undated, date_known=True))["date_known"] is True
    assert set(collectors.sanitize_feed_cache_item(old_dated)) == set(collectors.FEED_CACHE_ALLOWED_KEYS)


# ---- the undated rule, for new and stored rows ----


@pytest.mark.parametrize("row, known", [
    ({"date_known": False, "published_at": "2026-11-18T10:00:00Z", "fetched_at": "2026-11-19T10:00:00Z"}, False),
    ({"date_known": True, "published_at": "2026-11-19T10:00:00Z", "fetched_at": "2026-11-19T10:00:00Z"}, True),
    # Stored before the field existed: the fallback shows as published_at == fetched_at to the second.
    ({"published_at": "2026-11-19T10:00:00Z", "fetched_at": "2026-11-19T10:00:00.400000+00:00"}, False),
    ({"published_at": "2026-11-19T10:00:00Z", "fetched_at": "2026-11-19T10:00:01Z"}, True),
    ({"published_at": "2026-11-19T10:00:00Z"}, True),
    ({"published_at": "garbled", "fetched_at": "garbled"}, True),
])
def test_a_row_is_undated_by_its_mark_or_by_equal_stamps(row, known) -> None:
    assert item_date_known(row) is known


def test_a_stored_row_reaches_scoring_with_what_it_needs_to_be_judged() -> None:
    stamp = "2026-11-19T10:00:00Z"
    item = _row_to_cached_item({"source_id": "o1_s", "published_at": stamp, "fetched_at": stamp}, 0, stamp)
    assert item_date_known(item) is False
    item = _row_to_cached_item({"source_id": "o1_s", "published_at": stamp, "fetched_at": stamp,
                                "date_known": True}, 0, stamp)
    assert item_date_known(item) is True
    assert item_date_known(CachedItem(url_hash="h", source_id="s", published_at=stamp)) is True


def test_the_switch_is_read_like_the_other_switches(monkeypatch) -> None:
    assert undated_as_background_enabled({UNDATED_ENV: "1"}) is True
    for value in ("0", "", "true"):
        assert undated_as_background_enabled({UNDATED_ENV: value}) is False
    assert undated_as_background_enabled({}) is False
    monkeypatch.setenv(UNDATED_ENV, "1")
    assert undated_as_background_enabled() is True


# ---- scoring ----


def _axis(*components: tuple[int, float]) -> list[float]:
    vec = np.zeros(768, dtype=np.float32)
    for index, value in components:
        vec[index] = value
    return vec.tolist()


def _corpus() -> list[dict]:
    """One undated O1 essay alone on its axis; a dated two-source Macro group on another."""
    stored = (NOW - timedelta(hours=2)).strftime(STAMP)
    fetched = (NOW - timedelta(hours=1)).strftime(STAMP)
    tilt = math.radians(10.0)
    return [
        {"url_hash": "essay", "source_id": "o1_orbit_feed", "bucket": "ORBIT", "tier": "O1",
         "title": "Synthetic essay", "published_at": stored, "fetched_at": stored,
         "embedding_f32": _axis((0, 1.0))},
        {"url_hash": "news_1", "source_id": "m1_feed", "bucket": "MACRO", "tier": "M1",
         "title": "Synthetic report", "published_at": (NOW - timedelta(hours=10)).strftime(STAMP),
         "fetched_at": fetched, "date_known": True, "embedding_f32": _axis((5, 1.0))},
        {"url_hash": "news_2", "source_id": "m2_feed", "bucket": "MACRO", "tier": "M2",
         "title": "Synthetic report two", "published_at": (NOW - timedelta(hours=40)).strftime(STAMP),
         "fetched_at": fetched, "date_known": True,
         "embedding_f32": _axis((5, math.cos(tilt)), (6, math.sin(tilt)))},
    ]


def _cluster_of(clusters, url_hash):
    return next(c for c in clusters if any(item["url_hash"] == url_hash for item in c.items))


def _fingerprint(clusters) -> list[tuple]:
    return [(c.cluster_id, [item["url_hash"] for item in c.items], c.w_recent, c.w_base_daily, c.breakout_surge,
             c.velocity, c.orbit_resonance, c.s_intra, c.unique_sources, c.commission_score, c.threshold_used,
             c.legacy_commission_score, list(c.cluster_keywords), c.silhouette_score) for c in clusters]


def test_switch_on_an_undated_single_item_no_longer_outranks_the_dated_group_it_used_to() -> None:
    off = cluster_and_score(_corpus(), now_utc=NOW, undated_as_background=False)
    on = cluster_and_score(_corpus(), now_utc=NOW, undated_as_background=True)
    assert [c.items[0]["url_hash"] for c in off] == ["essay", "news_1"]
    assert [c.items[0]["url_hash"] for c in on] == ["news_1", "essay"]
    essay_off, essay_on = _cluster_of(off, "essay"), _cluster_of(on, "essay")
    assert essay_off.w_recent > 0 and essay_off.w_base_daily == 0
    # Counted as base inside the 14-day Orbit window: never recent, still a source, still grouped.
    assert essay_on.w_recent == 0
    assert essay_on.w_base_daily == pytest.approx(essay_off.w_recent / 11.0)
    assert essay_on.unique_sources == ("o1_orbit_feed",)
    assert [item["url_hash"] for item in essay_on.items] == ["essay"]
    # The dated group is untouched.
    assert _fingerprint([_cluster_of(on, "news_1")]) == _fingerprint([_cluster_of(off, "news_1")])


def test_switch_on_an_undated_macro_item_counts_as_base_not_as_a_fresh_source() -> None:
    corpus = _corpus()
    corpus[2]["date_known"] = False
    corpus[2]["published_at"] = (NOW - timedelta(hours=3)).strftime(STAMP)
    off = _cluster_of(cluster_and_score(copy.deepcopy(corpus), now_utc=NOW, undated_as_background=False), "news_2")
    on = _cluster_of(cluster_and_score(copy.deepcopy(corpus), now_utc=NOW, undated_as_background=True), "news_2")
    assert off.w_recent == pytest.approx(2.0) and off.w_base_daily == 0
    assert on.w_recent == pytest.approx(1.0) and on.w_base_daily == pytest.approx(1.0 / 3.0)
    assert on.unique_sources == off.unique_sources == ("m1_feed", "m2_feed")


def test_switch_off_scoring_is_what_it_was_and_ignores_the_new_fields(monkeypatch) -> None:
    stripped = [{k: v for k, v in row.items() if k not in {"date_known", "fetched_at"}} for row in _corpus()]
    reference = _fingerprint(cluster_and_score(copy.deepcopy(stripped), now_utc=NOW))
    monkeypatch.delenv(UNDATED_ENV, raising=False)
    assert _fingerprint(cluster_and_score(_corpus(), now_utc=NOW)) == reference
    monkeypatch.setenv(UNDATED_ENV, "0")
    assert _fingerprint(cluster_and_score(_corpus(), now_utc=NOW)) == reference
    assert _fingerprint(score_and_cluster_signals(_corpus(), now_utc=NOW)) == _fingerprint(
        score_and_cluster_signals(copy.deepcopy(stripped), now_utc=NOW))
    # Today the undated essay is the top group, counted entirely as recent.
    top = cluster_and_score(_corpus(), now_utc=NOW)[0]
    assert [item["url_hash"] for item in top.items] == ["essay"] and top.w_base_daily == 0


def test_the_switch_from_the_environment_reaches_the_night_s_scoring(monkeypatch) -> None:
    monkeypatch.setenv(UNDATED_ENV, "1")
    on = score_and_cluster_signals(_corpus(), now_utc=NOW)
    assert [c.items[0]["url_hash"] for c in on] == ["news_1", "essay"]
    # The window is unchanged: the undated item is still in it, with the same weight.
    monkeypatch.setenv(UNDATED_ENV, "0")
    off = score_and_cluster_signals(_corpus(), now_utc=NOW)
    assert sorted(i["url_hash"] for c in on for i in c.items) == sorted(i["url_hash"] for c in off for i in c.items)
    assert _cluster_of(on, "essay").items[0]["weight"] == _cluster_of(off, "essay").items[0]["weight"]


# ---- the workflows pass the switch, off by default ----


def test_the_daily_workflow_passes_the_switch_to_the_edition_job_only() -> None:
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition[edition.index(CANDIDATE_LINE) + 1] == SWITCH_LINE
    assert SWITCH_LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(SWITCH_LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_workflows_pass_the_switch(name) -> None:
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(SWITCH_LINE) == 1
    assert lines[lines.index(CANDIDATE_LINE) + 1] == SWITCH_LINE


def test_the_provider_canary_does_not_pass_the_switch() -> None:
    assert UNDATED_ENV not in (ROOT / ".github/workflows/provider_canary.yml").read_text()
