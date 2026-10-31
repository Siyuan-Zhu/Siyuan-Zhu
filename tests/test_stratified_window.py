"""Unit tests for WS3 Seam 1: Stratified Asymmetric Window & Farthest-Point Diversity."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from src.zeitgeist.methodology import (
    CachedItem,
    farthest_point_diversity_sample,
    load_asymmetric_window,
)


def _make_unit_vec(dim: int, angle_rad: float, plane: tuple[int, int] = (0, 1)) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    i, j = plane
    vec[i] = math.cos(angle_rad)
    vec[j] = math.sin(angle_rad)
    norm = float(np.linalg.norm(vec))
    return vec / norm


def _vec_to_f16_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float16).tobytes()


def test_farthest_point_diversity_sample_collapses_near_duplicates_and_picks_k4() -> None:
    """12 items over 96h: 6 near-duplicates (cos sim >= 0.85) collapse to newest, then greedy k=4."""
    base_time = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    dim = 768

    # 6 near-duplicate breaking news rewrites clustered tightly around angle 0.0 (cos(0.05) ~ 0.998 >= 0.85)
    dup_items: list[CachedItem] = []
    for idx in range(6):
        vec = _make_unit_vec(dim, angle_rad=0.02 * idx, plane=(0, 1))
        dup_items.append(
            CachedItem(
                url_hash=f"dup_{idx}",
                source_id="m1_reuters",
                bucket="MACRO",
                tier="M1",
                title=f"Breaking news rewrite {idx}",
                summary="Same underlying story rewrite.",
                published_at=(base_time - timedelta(hours=idx * 2)).isoformat(),
                embedding_f16=_vec_to_f16_bytes(vec),
            )
        )

    # 5 distinct orthogonal/distant items along independent basis axes (axes 2, 3, 4, 5, 6)
    # Plus 1 item with moderate similarity (e.g., cos = 0.5 with axis 0, below 0.85 threshold)
    distinct_items: list[CachedItem] = []
    for axis_offset, axis_idx in enumerate([2, 3, 4]):
        vec = np.zeros(dim, dtype=np.float32)
        vec[axis_idx] = 1.0
        distinct_items.append(
            CachedItem(
                url_hash=f"ortho_{axis_idx}",
                source_id="m1_reuters",
                bucket="MACRO",
                tier="M1",
                title=f"Distinct orthogonal topic {axis_idx}",
                summary=f"Orthogonal dispatch {axis_idx}",
                published_at=(base_time - timedelta(hours=20 + axis_offset * 5)).isoformat(),
                embedding_f16=_vec_to_f16_bytes(vec),
            )
        )

    # 3 more distinct items that have positive similarity (~0.50 to 0.70) to already chosen items
    # so greedy farthest-point must prefer the orthogonal ones (similarity 0.0) over these!
    for mod_idx, angle in enumerate([math.acos(0.60), math.acos(0.50), math.acos(0.40)]):
        vec = _make_unit_vec(dim, angle_rad=angle, plane=(0, 2))
        distinct_items.append(
            CachedItem(
                url_hash=f"moderate_{mod_idx}",
                source_id="m1_reuters",
                bucket="MACRO",
                tier="M1",
                title=f"Moderately similar topic {mod_idx}",
                summary="Moderately related dispatch.",
                published_at=(base_time - timedelta(hours=14 + mod_idx)).isoformat(),
                embedding_f16=_vec_to_f16_bytes(vec),
            )
        )

    all_12_items = distinct_items + dup_items
    assert len(all_12_items) == 12

    sampled = farthest_point_diversity_sample(all_12_items, k=4)
    assert len(sampled) == 4

    sampled_hashes = [item.url_hash for item in sampled]
    # x_1 must be the newest item overall (dup_0)
    assert sampled_hashes[0] == "dup_0"
    # None of dup_1..dup_5 may be present (collapsed at >= 0.85)
    for idx in range(1, 6):
        assert f"dup_{idx}" not in sampled_hashes
    # The remaining 3 selected items must be the mutually orthogonal items (max sim = 0.0)
    # rather than moderate_0..2 (which have max sim >= 0.40 to dup_0 / ortho_2)
    assert set(sampled_hashes) == {"dup_0", "ortho_2", "ortho_3", "ortho_4"}


def test_load_asymmetric_window_retains_orbit_336h_and_caps_macro_with_exact_weights() -> None:
    """Verify 14d Orbit retention vs 4d Macro diversity sampling and exact sub-linear source damping."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    dim = 768

    # 4 Orbit items from source 'o1_willison' spread up to 300h (within 336h = 14d)
    # Plus 1 expired Orbit item at 350h (> 336h)
    orbit_items: list[CachedItem] = []
    for idx, age_h in enumerate([10, 80, 200, 320, 350]):
        vec = np.zeros(dim, dtype=np.float32)
        vec[10 + idx] = 1.0
        orbit_items.append(
            CachedItem(
                url_hash=f"orbit_hash_{idx}",
                source_id="o1_willison",
                bucket="ORBIT",
                tier="O1",
                title=f"Orbit note {idx}",
                summary="Personal orbit post.",
                published_at=(now_utc - timedelta(hours=age_h)).isoformat(),
                embedding_f16=_vec_to_f16_bytes(vec),
            )
        )

    # 6 distinct Macro items from source 'm1_ars' within 96h (capped to k=4 via farthest-point)
    # Plus 1 expired Macro item at 110h (> 96h)
    macro_items: list[CachedItem] = []
    for idx, age_h in enumerate([5, 15, 30, 50, 70, 90, 110]):
        vec = np.zeros(dim, dtype=np.float32)
        vec[30 + idx] = 1.0
        macro_items.append(
            CachedItem(
                url_hash=f"macro_hash_{idx}",
                source_id="m1_ars",
                bucket="MACRO",
                tier="M1",
                title=f"Macro article {idx}",
                summary="Macro news item.",
                published_at=(now_utc - timedelta(hours=age_h)).isoformat(),
                embedding_f16=_vec_to_f16_bytes(vec),
            )
        )

    # Mark orbit_hash_0 and macro_hash_0 as already used to anchor a word in the last 14 days
    anchored_14d = {"orbit_hash_0", "macro_hash_0"}

    window = load_asymmetric_window(
        orbit_items + macro_items,
        anchored_hashes_14d=anchored_14d,
        now_utc=now_utc,
    )

    # 4 non-expired Orbit items + 4 sampled Macro items = 8 items
    assert len(window) == 8
    by_hash = {item.url_hash: item for item in window}

    assert "orbit_hash_4" not in by_hash  # 350h > 336h
    assert "macro_hash_6" not in by_hash  # 110h > 96h

    # N_win('o1_willison') = 4 -> sqrt(4) = 2.0
    # Unanchored Orbit weight: w_orbit * 1.00 / 2.0 (0.80 for w_orbit=1.60 or 1.00 for w_orbit=2.00)
    # Anchored Orbit weight:   w_orbit * 0.50 / 2.0 (0.40 for w_orbit=1.60 or 0.50 for w_orbit=2.00)
    assert by_hash["orbit_hash_0"].weight in (
        pytest.approx(0.40, abs=1e-6),
        pytest.approx(0.50, abs=1e-6),
    )
    assert by_hash["orbit_hash_1"].weight in (
        pytest.approx(0.80, abs=1e-6),
        pytest.approx(1.00, abs=1e-6),
    )
    assert by_hash["orbit_hash_2"].weight == pytest.approx(by_hash["orbit_hash_1"].weight, abs=1e-6)
    assert by_hash["orbit_hash_3"].weight == pytest.approx(by_hash["orbit_hash_1"].weight, abs=1e-6)
    assert by_hash["orbit_hash_1"].weight == pytest.approx(
        2.0 * by_hash["orbit_hash_0"].weight, abs=1e-6
    )

    # N_win('m1_ars') = 4 -> sqrt(4) = 2.0
    # Unanchored Macro weight: 1.00 * 1.00 / 2.0 = 0.50
    # Anchored Macro weight:   1.00 * 0.50 / 2.0 = 0.25
    assert by_hash["macro_hash_0"].weight == pytest.approx(0.25, abs=1e-6)
    for h in by_hash:
        if h.startswith("macro_hash_") and h != "macro_hash_0":
            assert by_hash[h].weight == pytest.approx(0.50, abs=1e-6)


def test_collector_and_config_tier_weights_consistency() -> None:
    """Verify tier-specific weights (O1=2.0, O2=1.6, O3=1.3, O4_SCOUT=1.1, M1-M3=1.0) across config and collectors."""
    import json
    from pathlib import Path
    from src.zeitgeist.collectors import (
        FEED_CACHE_ALLOWED_KEYS,
        TIER_WEIGHTS,
        _resolve_orbit_weight,
        sanitize_feed_cache_item,
    )

    assert TIER_WEIGHTS["O1"] == pytest.approx(2.0)
    assert TIER_WEIGHTS["O2"] == pytest.approx(1.6)
    assert TIER_WEIGHTS["O3"] == pytest.approx(1.3)
    assert TIER_WEIGHTS["O4_SCOUT"] == pytest.approx(1.1)
    assert TIER_WEIGHTS["M1"] == pytest.approx(1.0)
    assert TIER_WEIGHTS["M2"] == pytest.approx(1.0)
    assert TIER_WEIGHTS["M3"] == pytest.approx(1.0)

    cfg_path = Path(__file__).resolve().parent.parent / "config" / "orbit_sources.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for feed in cfg.get("feeds", []):
        tier = feed["tier"]
        assert feed["weight"] == pytest.approx(TIER_WEIGHTS[tier])
        assert _resolve_orbit_weight(tier, feed["weight"]) == pytest.approx(TIER_WEIGHTS[tier])

    for tier, expected_w in [("O1", 2.0), ("O2", 1.6), ("O3", 1.3), ("O4_SCOUT", 1.1), ("M1", 1.0)]:
        doc = sanitize_feed_cache_item(
            {
                "url": f"https://example.com/{tier.lower()}",
                "title": f"Tier {tier} item",
                "source_id": f"src_{tier.lower()}",
                "tier": tier,
                "published_at": "2026-10-27T12:00:00Z",
                "summary": "Summary text",
            }
        )
        assert set(doc.keys()) == set(FEED_CACHE_ALLOWED_KEYS)
        assert doc["orbit_weight"] == pytest.approx(expected_w)
        assert doc["source_tier"] == (tier if tier.startswith("O") else "MACRO")


def test_load_asymmetric_window_enforces_220_orbit_and_180_macro_caps_with_dicts() -> None:
    """Verify total window <= 400 (max 220 Orbit, max 180 Macro) and dict item support."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    dim = 16
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = 1.0
    f16_bytes = _vec_to_f16_bytes(vec)

    orbit_dicts = [
        {
            "url_hash": f"orb_{i}",
            "source_id": f"o1_src_{i % 10}",
            "bucket": "ORBIT",
            "published_at": (now_utc - timedelta(hours=i)).isoformat(),
            "embedding_f16": f16_bytes,
        }
        for i in range(250)
    ]

    macro_dicts = []
    for s in range(50):
        for j in range(4):
            v = np.zeros(dim, dtype=np.float32)
            v[(j + 1) % dim] = 1.0
            macro_dicts.append(
                {
                    "url_hash": f"mac_{s}_{j}",
                    "source_id": f"m1_src_{s}",
                    "bucket": "MACRO",
                    "published_at": (now_utc - timedelta(hours=j + 1)).isoformat(),
                    "embedding_f16": _vec_to_f16_bytes(v),
                }
            )

    window = load_asymmetric_window(orbit_dicts + macro_dicts, now_utc=now_utc)
    assert len(window) == 400
    orbit_kept = [it for it in window if it["bucket"] == "ORBIT"]
    macro_kept = [it for it in window if it["bucket"] == "MACRO"]
    assert len(orbit_kept) == 220
    assert len(macro_kept) == 180

