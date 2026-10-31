"""The live Instagram roster policy: independent cursors and successful batches."""
from collections import Counter
from datetime import datetime, timedelta, timezone

import pytest

from src.zeitgeist.orbit_roster import advance_roster_after_batch, select_instagram_batch

NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def test_three_successful_batches_evenly_rotate_seeds_and_independent_rss_cursor():
    seeds = [f"seed{i}" for i in range(12)]
    roster = {"handles": {}, "curated_o3_cursor": 0, "state_version": 7}
    counts = Counter()
    cursors = []
    for day in range(3):
        counts.update(select_instagram_batch(roster, seeds))
        roster = advance_roster_after_batch(roster, {}, NOW + timedelta(days=day),
                    apify_batch_succeeded=True, curated_handles=seeds)
        cursors.append((roster["curated_o3_cursor"], roster["rss_o3_cursor"]))
    assert counts == Counter({seed: 2 for seed in seeds})
    assert cursors == [(8, 2), (4, 4), (0, 0)]
    assert roster["state_version"] == 10


def test_one_parent_discovery_is_active_and_daily_vitality_changes_only_once():
    roster = advance_roster_after_batch({"handles": {}}, {"newvoice": {"o1_source"}}, NOW,
                                      apify_batch_succeeded=True)
    scout = roster["handles"]["newvoice"]
    assert scout["vitality_score"] == .50 and scout["promoted"] is True
    assert select_instagram_batch(roster, []) == ["newvoice"]
    # Repeated owner runs on the discovery day must not mint another bonus.
    again = advance_roster_after_batch(roster, {"newvoice": {"o1_source"}}, NOW,
                                       apify_batch_succeeded=True)
    assert again["handles"]["newvoice"]["vitality_score"] == .50
    tomorrow = advance_roster_after_batch(again, {"newvoice": {"o2_source"}}, NOW + timedelta(days=1),
                                          apify_batch_succeeded=True)
    assert tomorrow["handles"]["newvoice"]["vitality_score"] == pytest.approx(.63)
    assert tomorrow["handles"]["newvoice"]["parent_sources"] == ["o1_source", "o2_source"]


def test_failed_batch_cannot_age_or_advance_roster_and_skipped_days_do_not_decay():
    roster = {"curated_o3_cursor": 8, "rss_o3_cursor": 4, "state_version": 9,
              "handles": {"oldvoice": {"vitality_score": .5, "decayed_score": 2.0, "updated_at": NOW.isoformat(),
                                       "last_vitality_decay_date": "2026-10-30", "legacy_field": "retain"}}}
    failed = advance_roster_after_batch(roster, {"newvoice": {"o1_source"}}, NOW + timedelta(days=3),
                                        apify_batch_succeeded=False)
    assert failed == roster
    known = advance_roster_after_batch(failed, {}, NOW + timedelta(days=4), apify_batch_succeeded=True)
    assert known["handles"]["oldvoice"]["vitality_score"] == pytest.approx(.38)
    assert known["handles"]["oldvoice"]["legacy_field"] == "retain"
    assert known["handles"]["oldvoice"]["decayed_score"] == 2.0


def test_winning_cluster_bonus_uses_actual_instagram_contributors_once_per_day():
    from src.zeitgeist.orbit_roster import apply_winning_cluster_bonus
    from types import SimpleNamespace

    roster = advance_roster_after_batch({"handles": {}}, {"winningvoice": {"parent"}, "other": {"parent"}},
                                      NOW, apify_batch_succeeded=True)
    items = [SimpleNamespace(source_id="o4_ig_winningvoice"), {"source_id": "o3_ig_winningvoice"},
             {"source_id": "m1_ig_other"}, {"source_id": "o4_ig_unknown"}]
    result = apply_winning_cluster_bonus(roster, items, NOW, apify_batch_succeeded=True)
    assert result["handles"]["winningvoice"]["vitality_score"] == pytest.approx(.85)
    assert result["handles"]["other"]["vitality_score"] == .5
    assert "unknown" not in result["handles"]
    assert apply_winning_cluster_bonus(result, items, NOW, apify_batch_succeeded=True) == result
    assert apply_winning_cluster_bonus(roster, items, NOW, apify_batch_succeeded=False) == roster


def test_top_twenty_survive_and_only_top_eight_unblocked_scouts_are_scraped():
    roster = {"handles": {f"voice{i:02}": {"vitality_score": .30 + i*.02,
                "last_vitality_decay_date": "2026-10-30"} for i in range(22)}}
    roster = advance_roster_after_batch(roster, {}, NOW, apify_batch_succeeded=True)
    assert len(roster["handles"]) == 20 and "voice00" not in roster["handles"]
    assert select_instagram_batch(roster, [], blocklist={"voice21"}) == [f"voice{i:02}" for i in range(20, 12, -1)]


def test_v3_sanitizer_preserves_independent_cursors_and_private_record_fields():
    from src.zeitgeist.collectors import sanitize_orbit_roster_state
    result = sanitize_orbit_roster_state({"curated_o3_cursor": 8, "rss_o3_cursor": 4,
               "state_version": 14, "handles": {"voice": {"legacy": "retained"}}})
    assert result["curated_o3_cursor"] == 8 and result["rss_o3_cursor"] == 4
    assert result["handles"]["voice"]["legacy"] == "retained"


def test_live_discovery_accepts_only_one_hop_curated_origin_and_handle_mentions():
    from src.zeitgeist.collectors import extract_one_hop_mentions
    rows = [{"source_id": "o1_feed", "tier": "O1", "origin": "curated_O1_O3",
             "title": "@fresh @blocked @seed https://example.com https://x.com/outsider"},
            {"source_id": "o4_ig_scout", "tier": "O4_SCOUT", "origin": "orbit_scout", "title": "@secondhop"},
            {"source_id": "m1_feed", "tier": "M1", "origin": "curated_O1_O3", "title": "@macro"},
            {"source_id": "o2_fake", "tier": "O2", "title": "@unverified"}]
    assert extract_one_hop_mentions(rows, {"seed"}, {"blocked"}, instagram_only=True,
                                    require_curated_origin=True) == {"fresh": {"o1_feed"}}
