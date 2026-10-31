"""Unit tests for WS2 Orbit & Macro Ingestion Collectors (Ticket #22)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ORBIT_SOURCES_PATH = REPO_ROOT / "config" / "orbit_sources.json"


def test_orbit_sources_60_feed_schema() -> None:
    """Verify config/orbit_sources.json contains 60 curated feeds (30 ORBIT + 30 MACRO), tier weights, O3 categories, Instagram handles, and blocklist."""
    from src.zeitgeist.collectors import O3_CATEGORIES, TIER_WEIGHTS

    data = json.loads(ORBIT_SOURCES_PATH.read_text(encoding="utf-8"))
    feeds = data.get("feeds", [])
    assert len(feeds) == 60

    orbit_feeds = [f for f in feeds if f.get("bucket") == "ORBIT"]
    macro_feeds = [f for f in feeds if f.get("bucket") == "MACRO"]
    assert len(orbit_feeds) == 30
    assert len(macro_feeds) == 30

    tiers = {}
    o3_cats = set()
    for f in feeds:
        assert {"id", "name", "tier", "bucket", "type", "url", "weight"}.issubset(f.keys())
        tier = f["tier"]
        tiers[tier] = tiers.get(tier, 0) + 1
        assert f["weight"] == pytest.approx(TIER_WEIGHTS[tier])
        if tier == "O3":
            o3_cats.add(f.get("category"))
    assert tiers == {"O1": 10, "O2": 10, "O3": 10, "M1": 10, "M2": 10, "M3": 10}
    assert o3_cats == set(O3_CATEGORIES)

    ig_handles = data.get("curated_o3_instagram_handles", [])
    assert len(ig_handles) == 12
    blocklist = set(data.get("handle_blocklist", []))
    assert {"instagram", "explore", "reels", "github", "youtube", "nytimes"}.issubset(blocklist)


def test_per_source_daily_cap_and_url_normalization() -> None:
    """A single prolific RSS feed returning 10 items today is capped at MAX_ITEMS_PER_SOURCE_PER_DAY = 3 newest items with normalized URLs and full feed_cache schema."""
    from src.zeitgeist.collectors import (
        FEED_CACHE_ALLOWED_KEYS,
        MAX_ITEMS_PER_SOURCE_PER_DAY,
        fetch_all_curated_feeds,
        sanitize_feed_cache_item,
    )

    assert MAX_ITEMS_PER_SOURCE_PER_DAY == 3

    rss_items_xml = []
    for idx in range(10):
        hour = 10 + idx
        rss_items_xml.append(
            f"""
            <item>
              <title>Dispatch Item {idx}</title>
              <link>https://simonwillison.net/2026/Oct/27/item-{idx}/?utm_source=rss&amp;utm_medium=feed&amp;fbclid=123&amp;igshid=abc&amp;keep=1</link>
              <description>Summary for item {idx}</description>
              <pubDate>Tue, 27 Oct 2026 {hour:02d}:00:00 GMT</pubDate>
            </item>
            """
        )
    rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Simon Willison</title>
        {''.join(rss_items_xml)}
      </channel>
    </rss>
    """

    sources_config = {
        "feeds": [
            {
                "id": "o1_simon_willison",
                "name": "Simon Willison",
                "tier": "O1",
                "bucket": "ORBIT",
                "type": "rss",
                "url": "https://simonwillison.net/atom/everything/",
                "weight": 2.0,
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = rss_xml
    mock_resp.content = rss_xml.encode("utf-8")
    mock_resp.raise_for_status = MagicMock()

    with patch("src.zeitgeist.collectors.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        items = fetch_all_curated_feeds(sources_config, timeout_s=5.0, max_workers=4)

    assert len(items) == 3
    titles = [item["title"] for item in items]
    # Newest 3 items (hours 19, 18, 17 -> indices 9, 8, 7)
    assert titles == ["Dispatch Item 9", "Dispatch Item 8", "Dispatch Item 7"]

    expected_canonical_9 = "https://simonwillison.net/2026/Oct/27/item-9?keep=1"
    assert items[0]["url"] == expected_canonical_9
    assert items[0]["url_hash"] == hashlib.sha256(expected_canonical_9.encode("utf-8")).hexdigest()[:32]
    assert len(items[0]["url_hash"]) == 32
    assert items[0]["source_id"] == "o1_simon_willison"
    assert items[0]["tier"] == "O1"
    assert items[0]["source_tier"] == "O1"
    assert items[0]["bucket"] == "ORBIT"
    assert items[0]["origin"] == "curated_O1_O3"
    assert items[0]["weight"] == pytest.approx(2.0)
    assert items[0]["orbit_weight"] == pytest.approx(2.0)
    assert items[0]["schema_version"] == 1
    assert isinstance(items[0]["expires_at"], int)
    assert items[0][" fetched_at".strip()].endswith("Z")

    sanitized = sanitize_feed_cache_item(items[0])
    assert set(sanitized.keys()) == set(FEED_CACHE_ALLOWED_KEYS)
    assert sanitized["source_tier"] == "O1"
    assert sanitized["orbit_weight"] == pytest.approx(2.0)
    assert sanitized["schema_version"] == 1


def test_extract_one_hop_mentions_and_blocklist_filtering() -> None:
    """Verify @handle regex extraction, blocklist filtering (@instagram, @explore, @github), curated handle filtering, and 1-hop O1-O3 limit."""
    from src.zeitgeist.collectors import extract_one_hop_mentions

    curated_handles = {"hubermanlab", "lexfridman", "karpathy"}
    blocklist = {"instagram", "explore", "reels", "github", "youtube", "nytimes", "wsj", "bloomberg"}

    items = [
        {
            "source_id": "o3_ig_hubermanlab",
            "tier": "O3",
            "title": "New episode with @peterattiamd and @FoundMyFitness.",
            "summary": "Follow @instagram @explore @github and @lexfridman for more updates! Also @new_neuro_lab",
        },
        {
            "source_id": "o3_ig_lexfridman",
            "tier": "O3",
            "title": "Conversation with @peterattiamd",
            "summary": "Check @youtube and @nytimes",
        },
        {
            # 2-hop scout item must NOT be expanded further
            "source_id": "o4_scout_someone",
            "tier": "O4_SCOUT",
            "title": "Mentioning @two_hop_target",
            "summary": "Should be ignored because tier is O4_SCOUT",
        },
    ]

    mentions = extract_one_hop_mentions(items, curated_handles=curated_handles, blocklist=blocklist)

    # Blocked handles must not appear
    for blocked in ("instagram", "explore", "github", "youtube", "nytimes"):
        assert blocked not in mentions

    # Existing curated handles must not appear
    assert "lexfridman" not in mentions
    assert "hubermanlab" not in mentions

    # 2-hop mention from O4_SCOUT must not appear
    assert "two_hop_target" not in mentions

    # Valid 1-hop mentions with distinct parent source sets
    assert mentions["peterattiamd"] == {"o3_ig_hubermanlab", "o3_ig_lexfridman"}
    assert mentions["foundmyfitness"] == {"o3_ig_hubermanlab"}
    assert mentions["new_neuro_lab"] == {"o3_ig_hubermanlab"}


def test_one_hop_promotion_threshold() -> None:
    """A handle mentioned by 1 parent with score 1.0 stays CANDIDATE; 2 distinct parents with score >= 2.0 promotes to ACTIVE_O4_SCOUT."""
    from datetime import datetime, timezone
    from src.zeitgeist.collectors import update_orbit_roster

    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    mention_parents = {
        "single_parent_handle": {"o3_ig_hubermanlab"},
        "dual_parent_handle": {"o3_ig_hubermanlab", "o3_ig_lexfridman"},
    }

    roster = update_orbit_roster({}, mention_parents, now_utc=now_utc)

    assert "single_parent_handle" in roster
    assert roster["single_parent_handle"]["decayed_score"] == pytest.approx(1.0)
    assert roster["single_parent_handle"]["status"] == "CANDIDATE"

    assert "dual_parent_handle" in roster
    assert roster["dual_parent_handle"]["decayed_score"] == pytest.approx(2.0)
    assert len(roster["dual_parent_handle"]["distinct_parent_sources"]) == 2
    assert roster["dual_parent_handle"]["status"] == "ACTIVE_O4_SCOUT"


def test_seven_day_half_life_decay_and_eviction() -> None:
    """A scout with score 4.0 and zero new mentions decays to 2.0 after 7 days, 0.5 after 21 days, and is evicted (< 0.25) at day 29."""
    from datetime import datetime, timedelta, timezone
    from src.zeitgeist.collectors import update_orbit_roster

    t0 = datetime(2026, 10, 1, 0, 0, 0, tzinfo=timezone.utc)
    roster = {
        "dr_scout": {
            "handle": "dr_scout",
            "decayed_score": 4.0,
            "distinct_parent_sources": ["o1_simon_willison", "o1_karpathy"],
            "first_seen_utc": "2026-10-01T00:00:00Z",
            "last_seen_utc": "2026-10-01T00:00:00Z",
            "status": "ACTIVE_O4_SCOUT",
        }
    }

    # Day 7: 4.0 * 0.5^(7/7) = 2.0
    roster_d7 = update_orbit_roster(roster, {}, now_utc=t0 + timedelta(days=7))
    assert "dr_scout" in roster_d7
    assert roster_d7["dr_scout"]["decayed_score"] == pytest.approx(2.0, rel=1e-5)

    # Day 21: 2.0 * 0.5^(14/7) = 0.5
    roster_d21 = update_orbit_roster(roster_d7, {}, now_utc=t0 + timedelta(days=21))
    assert "dr_scout" in roster_d21
    assert roster_d21["dr_scout"]["decayed_score"] == pytest.approx(0.5, rel=1e-5)

    # Day 29: 0.5 * 0.5^(8/7) = 0.2264 < 0.25 -> evicted
    roster_d29 = update_orbit_roster(roster_d21, {}, now_utc=t0 + timedelta(days=29))
    assert "dr_scout" not in roster_d29


def test_roster_cap_max_active_scouts_15() -> None:
    """Adding a 16th scout with a higher score evicts the lowest-scoring scout to maintain MAX_ACTIVE_SCOUTS = 15."""
    from datetime import datetime, timezone
    from src.zeitgeist.collectors import MAX_ACTIVE_SCOUTS, update_orbit_roster

    assert MAX_ACTIVE_SCOUTS == 15
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)

    initial_roster = {}
    for i in range(15):
        handle = f"scout_{i:02d}"
        # scout_00 has lowest score (1.5), scout_01..14 have scores 2.0..15.0
        score = 1.5 if i == 0 else float(i + 1)
        initial_roster[handle] = {
            "handle": handle,
            "decayed_score": score,
            "distinct_parent_sources": ["o1_simon_willison", "o1_karpathy"],
            "first_seen_utc": "2026-10-27T12:00:00Z",
            "last_seen_utc": "2026-10-27T12:00:00Z",
            "status": "ACTIVE_O4_SCOUT",
        }

    # Add a 16th scout mentioned by 3 distinct parents today (score = 3.0 > 1.5)
    mention_parents = {
        "scout_new_16": {"o1_simon_willison", "o1_karpathy", "o3_ig_hubermanlab"}
    }
    updated = update_orbit_roster(initial_roster, mention_parents, now_utc=now_utc)

    assert len(updated) == 15
    assert "scout_new_16" in updated
    assert updated["scout_new_16"]["decayed_score"] == pytest.approx(3.0)
    assert updated["scout_new_16"]["status"] == "ACTIVE_O4_SCOUT"
    assert "scout_00" not in updated


def test_apify_graceful_fallback_and_batch_scrape(monkeypatch) -> None:
    """Missing/empty APIFY_API_TOKEN or HTTP timeout returns [] without raising; valid response parses capped posts with tier weights."""
    from src.zeitgeist.collectors import APIFY_INSTAGRAM_ENDPOINT, fetch_instagram_batch, IngestionUnavailable
    from tests.test_ingestion_release_safety import ApprovedGuard
    monkeypatch.setenv("APIFY_INSTAGRAM_BUILD", "1.2.3")
    monkeypatch.setattr("src.zeitgeist.spend_guard.get_spend_guard", lambda **kwargs: ApprovedGuard())

    assert (
        APIFY_INSTAGRAM_ENDPOINT
        == "https://api.apify.com/v2/acts/apify~instagram-post-scraper/run-sync-get-dataset-items"
    )

    # 1. Missing or empty token -> returns [] without making network calls
    assert fetch_instagram_batch(["hubermanlab", "lexfridman"], apify_token=None) == []
    assert fetch_instagram_batch(["hubermanlab", "lexfridman"], apify_token="   ") == []

    # 2. HTTP timeout / exception -> returns [] gracefully
    with patch("src.zeitgeist.collectors.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.side_effect = TimeoutError("Apify timed out")
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with pytest.raises(IngestionUnavailable):
            fetch_instagram_batch(["hubermanlab"], apify_token="apify_test_token", timeout_s=1.0)
        assert mock_client.post.call_count == 1

    # 3. Valid batched response -> caps handles at 20 and posts at max_posts_per_handle=3
    fake_dataset = [
        {
            "username": "hubermanlab",
            "latestPosts": [
                {
                    "url": f"https://www.instagram.com/p/post_{i}/?igshid=xyz",
                    "caption": f"Protocol {i} with @peterattiamd",
                    "timestamp": f"2026-10-27T1{i}:00:00Z",
                }
                for i in range(5)
            ],
        }
    ]
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = fake_dataset
    mock_resp.raise_for_status = MagicMock()

    with patch("src.zeitgeist.collectors.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        posts = fetch_instagram_batch(
            ["hubermanlab"] + [f"handle_{i}" for i in range(25)],
            apify_token="apify_test_token",
            max_posts_per_handle=3,
        )
        # Verify at most 20 usernames sent in batch request
        call_kwargs = mock_client.post.call_args.kwargs
        assert len(call_kwargs["json"]["username"]) == 20

    assert len(posts) == 3
    assert posts[0]["source_id"] == "o3_ig_hubermanlab"
    assert posts[0]["source_name"] == "Instagram Orbit Dispatch"
    assert posts[0]["tier"] == "O3"
    assert posts[0]["source_tier"] == "O3"
    assert posts[0]["weight"] == pytest.approx(1.3)
    assert posts[0]["orbit_weight"] == pytest.approx(1.3)
    assert posts[0]["origin"] == "curated_O1_O3"
    assert posts[0]["schema_version"] == 1
    assert isinstance(posts[0]["expires_at"], int)
    assert "@" not in posts[0]["source_name"]
    assert "@" not in posts[0]["title"]
    assert "@" not in posts[0]["summary"]
    assert "@" not in posts[0]["caption"]
    assert posts[0]["mentioned_handles"] == ["peterattiamd"]
    assert posts[0]["url"] == "https://www.instagram.com/p/post_4"

    # 4. Tier map overrides (O1=2.0, O2=1.6, O4_SCOUT=1.1)
    tiered_dataset = [
        {"ownerUsername": "karpathy", "url": "https://www.instagram.com/p/k1", "caption": "O1 post"},
        {"ownerUsername": "lexfridman", "url": "https://www.instagram.com/p/l1", "caption": "O2 post"},
        {"ownerUsername": "peterattiamd", "url": "https://www.instagram.com/p/p1", "caption": "O4 post"},
    ]
    mock_resp_tiered = MagicMock()
    mock_resp_tiered.status_code = 200
    mock_resp_tiered.json.return_value = tiered_dataset
    mock_resp_tiered.raise_for_status = MagicMock()

    with patch("src.zeitgeist.collectors.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp_tiered
        mock_client_cls.return_value.__enter__.return_value = mock_client

        tiered_posts = fetch_instagram_batch(
            ["karpathy", "lexfridman", "peterattiamd"],
            apify_token="apify_test_token",
            tier_map={"karpathy": "O1", "lexfridman": "O2", "peterattiamd": "O4_SCOUT"},
        )

    by_tier = {p["tier"]: p for p in tiered_posts}
    assert by_tier["O1"]["orbit_weight"] == pytest.approx(2.0)
    assert by_tier["O2"]["orbit_weight"] == pytest.approx(1.6)
    assert by_tier["O4_SCOUT"]["orbit_weight"] == pytest.approx(1.1)
    assert by_tier["O4_SCOUT"]["origin"] == "orbit_scout_1hop"


def test_o3_cursor_round_robin_rotation_and_state_schema() -> None:
    """Verify 6 O3 categories rotate 2 per run via (cursor + 2) % 6 and update_orbit_roster tracks structured state."""
    from datetime import datetime, timedelta, timezone
    from src.zeitgeist.collectors import (
        O3_CATEGORIES,
        ORBIT_ROSTER_FIRESTORE_KEYS,
         sanitize_orbit_roster_state,
        select_o3_sources_for_cursor,
        update_orbit_roster,
    )

    data = json.loads(ORBIT_SOURCES_PATH.read_text(encoding="utf-8"))
    feeds = data["feeds"]

    seen_o3_ids: set[str] = set()
    cursor = 0
    for expected_next in (2, 4, 0):
        selected, next_cursor = select_o3_sources_for_cursor(feeds, cursor)
        assert next_cursor == expected_next
        o3_selected = [s for s in selected if s["tier"] == "O3"]
        active_cats = {O3_CATEGORIES[cursor], O3_CATEGORIES[(cursor + 1) % 6]}
        assert {s["category"] for s in o3_selected} == active_cats
        seen_o3_ids.update(s["id"] for s in o3_selected)
        cursor = next_cursor

    # Across 3 runs (cursor 0 -> 2 -> 4 -> 0), all 10 O3 feeds across all 6 categories were selected
    all_o3_ids = {s["id"] for s in feeds if s["tier"] == "O3"}
    assert seen_o3_ids == all_o3_ids

    # Verify update_orbit_roster advances curated_o3_cursor and tracks empty runs + quarantine_until
    t0 = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    state_0 = update_orbit_roster(
        {"curated_o3_cursor": 0},
        {"scout_alpha": {"o1_simon_willison", "o1_karpathy"}},
        now_utc=t0,
    )
    assert state_0["curated_o3_cursor"] == 2
    assert state_0["tier_map"]["scout_alpha"] == "O4_SCOUT"
    assert state_0["consecutive_empty_runs"]["scout_alpha"] == 0
    assert state_0["quarantine_until"]["scout_alpha"] is None

    # 3 consecutive empty runs trigger quarantine_until (7 days out) while cursor rotates 2 -> 4 -> 0 -> 2
    state_1 = update_orbit_roster(state_0, {}, now_utc=t0 + timedelta(days=1))
    assert state_1["curated_o3_cursor"] == 4
    assert state_1["consecutive_empty_runs"]["scout_alpha"] == 1
    assert state_1["quarantine_until"]["scout_alpha"] is None

    state_2 = update_orbit_roster(state_1, {}, now_utc=t0 + timedelta(days=2))
    assert state_2["curated_o3_cursor"] == 0
    assert state_2["consecutive_empty_runs"]["scout_alpha"] == 2
    assert state_2["quarantine_until"]["scout_alpha"] is None

    state_3 = update_orbit_roster(state_2, {}, now_utc=t0 + timedelta(days=3))
    assert state_3["curated_o3_cursor"] == 2
    assert state_3["consecutive_empty_runs"]["scout_alpha"] == 3
    assert state_3["quarantine_until"]["scout_alpha"] == "2026-11-06T12:00:00Z"

    firestore_doc = sanitize_orbit_roster_state(state_3)
    assert set(firestore_doc.keys()) == set(ORBIT_ROSTER_FIRESTORE_KEYS)
    assert firestore_doc["curated_o3_cursor"] == 2
    assert "scout_alpha" in firestore_doc["handles"]


def test_fetch_all_curated_feeds_async_with_semaphore_and_o3_cursor() -> None:
    """Verify fetch_all_curated_feeds_async uses httpx.AsyncClient + asyncio.Semaphore(10) and filters O3 by cursor."""
    import asyncio
    from unittest.mock import AsyncMock
    from src.zeitgeist.collectors import fetch_all_curated_feeds_async

    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Test Feed</title>
        <item>
          <title>Async Item 1</title>
          <link>https://example.com/post-1?utm_source=rss</link>
          <description>Summary with @async_scout mention</description>
          <pubDate>Tue, 27 Oct 2026 12:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """
    sources_config = {
        "feeds": [
            {
                "id": "o1_builder",
                "name": "Builder O1",
                "tier": "O1",
                "bucket": "ORBIT",
                "type": "rss",
                "url": "https://example.com/o1.xml",
                "weight": 2.0,
            },
            {
                "id": "o3_tech",
                "name": "Tech O3",
                "tier": "O3",
                "bucket": "ORBIT",
                "category": "tech_ai",
                "type": "rss",
                "url": "https://example.com/o3-tech.xml",
                "weight": 1.3,
            },
            {
                "id": "o3_design",
                "name": "Design O3",
                "tier": "O3",
                "bucket": "ORBIT",
                "category": "design",
                "type": "rss",
                "url": "https://example.com/o3-design.xml",
                "weight": 1.3,
            },
        ]
    }

    async def fake_async_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        resp.text = rss_xml.replace(
            "https://example.com/post-1?utm_source=rss",
            f"{url}/post-1?utm_source=rss",
        )
        return resp

    with patch("src.zeitgeist.collectors.httpx.AsyncClient") as mock_async_cls:
        mock_async_client = AsyncMock()
        mock_async_client.get.side_effect = fake_async_get
        mock_async_cls.return_value.__aenter__.return_value = mock_async_client

        # cursor=0 selects O3 categories ('tech_ai', 'economics'), excluding 'design'
        items = asyncio.run(
            fetch_all_curated_feeds_async(
                sources_config,
                timeout_s=12.0,
                max_concurrency=10,
                curated_o3_cursor=0,
            )
        )

    source_ids = {it["source_id"] for it in items}
    assert source_ids == {"o1_builder", "o3_tech"}
    by_src = {it["source_id"]: it for it in items}
    assert by_src["o1_builder"]["orbit_weight"] == pytest.approx(2.0)
    assert by_src["o3_tech"]["orbit_weight"] == pytest.approx(1.3)
    assert by_src["o1_builder"]["mentioned_handles"] == ["async_scout"]


def test_fetch_macro_scout_signals_m4_tag_and_cap() -> None:
    """Verify fetch_macro_scout_signals queries HN Algolia and Google Trends RSS, tagging M4_SCOUT and capping at 3 per source."""
    from src.zeitgeist.collectors import fetch_macro_scout_signals

    hn_payload = {
        "hits": [
            {
                "objectID": f"900{i}",
                "title": f"HN Breakthrough {i}",
                "url": f"https://example.org/hn-{i}?utm_source=hn",
                "points": 250 + i,
                "num_comments": 80,
                "created_at": f"2026-10-27T1{i}:00:00Z",
            }
            for i in range(6)
        ]
    }
    trends_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Daily Search Trends</title>
        <item>
          <title>Trend Alpha</title>
          <link>https://trends.google.com/trends/explore?q=alpha&amp;utm_source=trends</link>
          <pubDate>Tue, 27 Oct 2026 15:00:00 GMT</pubDate>
        </item>
        <item>
          <title>Trend Beta</title>
          <link>https://trends.google.com/trends/explore?q=beta</link>
          <pubDate>Tue, 27 Oct 2026 14:00:00 GMT</pubDate>
        </item>
        <item>
          <title>Trend Gamma</title>
          <link>https://trends.google.com/trends/explore?q=gamma</link>
          <pubDate>Tue, 27 Oct 2026 13:00:00 GMT</pubDate>
        </item>
        <item>
          <title>Trend Delta</title>
          <link>https://trends.google.com/trends/explore?q=delta</link>
          <pubDate>Tue, 27 Oct 2026 12:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
    """

    def fake_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status = MagicMock()
        if "algolia" in url:
            resp.text = json.dumps(hn_payload)
            resp.json.return_value = hn_payload
        else:
            resp.text = trends_xml
        return resp

    with patch("src.zeitgeist.collectors.httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.side_effect = fake_get
        mock_client_cls.return_value.__enter__.return_value = mock_client

        signals = fetch_macro_scout_signals(timeout_s=5.0)

    # 3 from HN Algolia + 3 from Google Trends US = 6 total
    assert len(signals) == 6
    assert all(s["tier"] == "M4_SCOUT" for s in signals)
    assert all(s["bucket"] == "MACRO" for s in signals)


def test_embedding_cache_hit_and_uncached_batching() -> None:
    """Offline legacy cache remains readable; uncached items use the approved embedding model."""
    from types import SimpleNamespace
    import numpy as np
    from src.zeitgeist.collectors import embed_missing_items

    cached_vec = np.ones(768, dtype=np.float16).tobytes()
    existing_cache = {
        "cached_hash_0001": {
            "url_hash": "cached_hash_0001",
            "title": "Cached Story",
            "embedding_f16": cached_vec,
        }
    }

    items = [
        {
            "url_hash": "cached_hash_0001",
            "url": "https://example.com/cached",
            "title": "Cached Story",
            "summary": "Already embedded yesterday",
        }
    ]
    for i in range(120):
        items.append(
            {
                "url_hash": f"uncached_hash_{i:04d}",
                "url": f"https://example.com/new-{i}",
                "title": f"Uncached Story {i}",
                "summary": f"Summary {i}",
            }
        )

    mock_client = MagicMock()

    def fake_embed_content(*, model: str, contents: list[str], config=None, **kwargs):
        assert model == "gemini-embedding-001"
        assert len(contents) <= 100
        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.25] * 768)
                for _ in contents
            ]
        )

    mock_client.models.embed_content.side_effect = fake_embed_content

    enriched = embed_missing_items(items, existing_cache=existing_cache, genai_client=mock_client)

    # 120 uncached items with batch_size=100 -> exactly 2 API calls (100 + 20); cached item was NOT re-embedded
    assert mock_client.models.embed_content.call_count == 2
    first_call_contents = mock_client.models.embed_content.call_args_list[0].kwargs["contents"]
    second_call_contents = mock_client.models.embed_content.call_args_list[1].kwargs["contents"]
    assert len(first_call_contents) == 100
    assert len(second_call_contents) == 20

    assert len(enriched) == 121
    assert enriched[0]["embedding_f16"] == cached_vec
    for item in enriched:
        assert isinstance(item["embedding_f16"], bytes)
        assert len(item["embedding_f16"]) == 1536

    # Also verify deterministic offline fallback when genai_client is None
    offline_items = [
        {
            "url_hash": "offline_hash_0001",
            "url": "https://example.com/offline",
            "title": "Offline Story",
            "summary": "Deterministic fallback test",
        }
    ]
    offline_enriched = embed_missing_items(offline_items, existing_cache={}, genai_client=None)
    assert len(offline_enriched) == 1
    assert offline_enriched[0]["embedding_source"] == "deterministic_fallback"
    emb_bytes = offline_enriched[0]["embedding_f16"]
    assert isinstance(emb_bytes, bytes)
    assert len(emb_bytes) == 1536
    vec = np.frombuffer(emb_bytes, dtype=np.float16).astype(np.float32)
    assert np.linalg.norm(vec) == pytest.approx(1.0, rel=1e-2)


def test_collector_warning_logging_and_embedding_single_item_retry(caplog: pytest.LogCaptureFixture, monkeypatch) -> None:
    """Verify collectors log warnings on network errors and embed_missing_items retries failed batches item-by-item before fallback."""
    import logging
    from types import SimpleNamespace
    from src.zeitgeist.collectors import embed_missing_items, fetch_instagram_batch, IngestionUnavailable
    from tests.test_ingestion_release_safety import ApprovedGuard
    monkeypatch.setenv("APIFY_INSTAGRAM_BUILD", "1.2.3")
    monkeypatch.setattr("src.zeitgeist.spend_guard.get_spend_guard", lambda **kwargs: ApprovedGuard())

    caplog.set_level(logging.WARNING, logger="src.zeitgeist.collectors")

    with patch("src.zeitgeist.collectors.httpx.Client") as mock_client_cls:
        mock_http = MagicMock()
        mock_http.post.side_effect = TimeoutError("Apify transient timeout")
        mock_client_cls.return_value.__enter__.return_value = mock_http

        with pytest.raises(IngestionUnavailable):
            fetch_instagram_batch(["hubermanlab"], apify_token="apify_test_token", timeout_s=0.5)
        assert any("Apify" in rec.message or "instagram" in rec.message.lower() for rec in caplog.records)

    # Batch call fails when len(contents) > 1, succeeds on single-item retry for item 0, fails for item 1
    mock_genai = MagicMock()

    def flaky_embed(*, model: str, contents: list[str], config=None, **kwargs):
        if len(contents) > 1:
            raise RuntimeError("Transient 503 batch failure")
        if "Recoverable Story" in contents[0]:
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.5] * 768)])
        raise RuntimeError("Permanent item failure")

    mock_genai.models.embed_content.side_effect = flaky_embed

    batch_items = [
        {
            "url_hash": "a" * 32,
            "url": "https://example.com/recoverable",
            "title": "Recoverable Story",
            "summary": "Succeeds on single-item retry",
        },
        {
            "url_hash": "b" * 32,
            "url": "https://example.com/unrecoverable",
            "title": "Unrecoverable Story",
            "summary": "Falls back after single-item retry",
        },
    ]

    enriched = embed_missing_items(batch_items, existing_cache={}, genai_client=mock_genai)
    assert len(enriched) == 2
    assert enriched[0]["embedding_source"] == "gemini"
    assert enriched[1]["embedding_source"] == "deterministic_fallback"
    assert any("embed" in rec.message.lower() for rec in caplog.records)





