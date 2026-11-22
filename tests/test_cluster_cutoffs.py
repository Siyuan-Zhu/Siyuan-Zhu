"""Task E4: clustering and near-copy cutoffs are parameters with today's defaults."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from src.zeitgeist import embedding_migration
from src.zeitgeist.methodology import (
    DYNAMIC_CLUSTER_THRESHOLDS,
    NEAR_DUPLICATE_THRESHOLD,
    CachedItem,
    cluster_and_score,
    load_asymmetric_window,
    score_and_cluster_signals,
)
from tests.test_zeitgeist_cluster import _vec_to_f16_bytes

NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
DIM = 768


def _unit(angle_deg: float, plane: tuple[int, int] = (0, 1)) -> np.ndarray:
    vec = np.zeros(DIM, dtype=np.float32)
    vec[plane[0]] = math.cos(math.radians(angle_deg))
    vec[plane[1]] = math.sin(math.radians(angle_deg))
    return vec


def _item(url_hash: str, source_id: str, bucket: str, hours: float, vec: np.ndarray) -> CachedItem:
    return CachedItem(
        url_hash=url_hash,
        source_id=source_id,
        bucket=bucket,
        title=f"Synthetic title {url_hash}",
        summary=f"Synthetic summary {url_hash}.",
        published_at=(NOW - timedelta(hours=hours)).isoformat(),
        embedding_f16=_vec_to_f16_bytes(vec),
    )


def _mixed_fixture() -> list[CachedItem]:
    """Fresh synthetic items: two topics, a near-copy pair, and a loner."""
    return [
        _item("t1_a", "o1_src", "ORBIT", 2, _unit(0.0)),
        _item("t1_b", "m1_src", "MACRO", 3, _unit(20.0)),
        _item("t1_c", "m2_src", "MACRO", 5, _unit(38.0)),
        _item("t2_a", "o2_src", "ORBIT", 4, _unit(0.0, (2, 3))),
        _item("t2_b", "m3_src", "MACRO", 6, _unit(25.0, (2, 3))),
        _item("dup_a", "m4_src", "MACRO", 1, _unit(0.0, (4, 5))),
        _item("dup_b", "m4_src", "MACRO", 7, _unit(10.0, (4, 5))),
        _item("lone", "m5_src", "MACRO", 8, _unit(0.0, (6, 7))),
    ]


def _summary(clusters) -> list[tuple]:
    return [
        (
            c.cluster_id,
            [it.url_hash for it in c.items],
            c.commission_score,
            c.legacy_commission_score,
            c.threshold_used,
        )
        for c in clusters
    ]


def test_explicit_default_cutoffs_match_the_defaults() -> None:
    implicit = score_and_cluster_signals(_mixed_fixture(), now_utc=NOW)
    explicit = score_and_cluster_signals(
        _mixed_fixture(),
        now_utc=NOW,
        cutoffs=(0.78, 0.72, 0.68),
        near_duplicate_threshold=0.85,
    )
    assert implicit, "fixture must produce clusters"
    assert _summary(explicit) == _summary(implicit)


def test_higher_cutoffs_keep_a_080_pair_apart() -> None:
    sim = 0.80
    angle = math.degrees(math.acos(sim))

    def pair() -> list[CachedItem]:
        return [
            _item("p_a", "o1_src", "ORBIT", 2, _unit(0.0)),
            _item("p_b", "m1_src", "MACRO", 3, _unit(angle)),
        ]

    default_clusters = score_and_cluster_signals(pair(), now_utc=NOW)
    assert [sorted(it.url_hash for it in c.items) for c in default_clusters] == [["p_a", "p_b"]]

    high = score_and_cluster_signals(pair(), now_utc=NOW, cutoffs=(0.90, 0.85, 0.82))
    assert high, "singletons still come back as clusters"
    assert all(len(c.items) == 1 for c in high)
    assert all(c.threshold_used == pytest.approx(0.82) for c in high)


def test_near_duplicate_threshold_controls_the_same_source_collapse() -> None:
    sim = 0.86
    angle = math.degrees(math.acos(sim))

    def same_source_pair() -> list[CachedItem]:
        return [
            _item("n_a", "m1_src", "MACRO", 2, _unit(0.0)),
            _item("n_b", "m1_src", "MACRO", 3, _unit(angle)),
        ]

    default_window = load_asymmetric_window(same_source_pair(), now_utc=NOW)
    assert [it.url_hash for it in default_window] == ["n_a"]

    loose = load_asymmetric_window(same_source_pair(), now_utc=NOW, near_duplicate_threshold=0.90)
    assert sorted(it.url_hash for it in loose) == ["n_a", "n_b"]


@pytest.mark.parametrize("bad", [(), (float("nan"),), (0.78, float("inf"))])
def test_invalid_cutoffs_raise_before_any_work(bad) -> None:
    with pytest.raises(ValueError):
        cluster_and_score(_mixed_fixture(), now_utc=NOW, cutoffs=bad)
    with pytest.raises(ValueError):
        score_and_cluster_signals(_mixed_fixture(), now_utc=NOW, cutoffs=bad)
    with pytest.raises(ValueError):
        cluster_and_score([], now_utc=NOW, cutoffs=bad)


def test_cluster_and_score_validation_leaves_items_untouched() -> None:
    items = _mixed_fixture()
    with pytest.raises(ValueError):
        cluster_and_score(items, now_utc=NOW, cutoffs=())
    assert all(it.weight is None for it in items)


def test_legacy_cutoffs_pin_the_methodology_constants() -> None:
    assert NEAR_DUPLICATE_THRESHOLD == 0.85
    assert embedding_migration.LEGACY_CUTOFFS == (*DYNAMIC_CLUSTER_THRESHOLDS, NEAR_DUPLICATE_THRESHOLD)
