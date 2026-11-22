"""Off-diagonal cohesion mode with the legacy score kept (synthetic data only)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest

from src.zeitgeist.methodology import (
    CachedItem,
    ScoredCluster,
    _cohesion,
    cluster_and_score,
    legacy_top_cluster,
)

NOW = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
DIM = 16


def _unit(axis: int, angle_deg: float = 0.0, other: int | None = None) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[axis] = math.cos(math.radians(angle_deg))
    if other is not None:
        v[other] = math.sin(math.radians(angle_deg))
    return v


def _item(name: str, source: str, vec: np.ndarray) -> CachedItem:
    return CachedItem(
        url_hash=name,
        source_id=source,
        bucket="ORBIT",
        title=f"synthetic title {name}",
        summary=f"synthetic summary {name}",
        published_at=(NOW - timedelta(hours=3)).isoformat(),
        embedding_f16=vec.astype(np.float16).tobytes(),
    )


def _mixed_items() -> list[CachedItem]:
    return [
        _item("a1", "s_a1", _unit(0)),
        _item("a2", "s_a2", _unit(0, 20.0, 1)),
        _item("a3", "s_a3", _unit(0, 10.0, 1)),
        _item("b1", "s_b1", _unit(2)),
        _item("b2", "s_b2", _unit(2, 30.0, 3)),
        _item("c1", "s_c1", _unit(4)),
    ]


def test_legacy_mode_is_default_and_unchanged() -> None:
    default = cluster_and_score(_mixed_items(), now_utc=NOW)
    explicit = cluster_and_score(_mixed_items(), now_utc=NOW, cohesion_mode="legacy")
    assert [c.cluster_id for c in default] == [c.cluster_id for c in explicit]
    for d, e in zip(default, explicit):
        assert d.s_intra == e.s_intra
        assert d.commission_score == e.commission_score
        assert d.legacy_s_intra == d.s_intra
        assert d.legacy_commission_score == d.commission_score


def test_offdiag_two_items() -> None:
    items = [_item("p", "s_p", _unit(0)), _item("q", "s_q", _unit(0, math.degrees(math.acos(0.8)), 1))]
    legacy = cluster_and_score(items, now_utc=NOW)
    off = cluster_and_score(items, now_utc=NOW, cohesion_mode="offdiag")
    assert len(legacy) == len(off) == 1
    assert legacy[0].s_intra == pytest.approx(0.90, abs=1e-3)
    assert off[0].s_intra == pytest.approx(0.80, abs=1e-3)
    assert off[0].legacy_s_intra == pytest.approx(0.90, abs=1e-3)
    assert off[0].legacy_commission_score == pytest.approx(legacy[0].commission_score)


def test_offdiag_identical_vectors_is_one() -> None:
    items = [_item("p", "s_p", _unit(0)), _item("q", "s_q", _unit(0))]
    for mode in ("legacy", "offdiag"):
        res = cluster_and_score(items, now_utc=NOW, cohesion_mode=mode)
        assert res[0].s_intra == pytest.approx(1.0, abs=1e-3)


def test_offdiag_singleton_is_zero_and_last() -> None:
    res = cluster_and_score(_mixed_items(), now_utc=NOW, cohesion_mode="offdiag")
    singles = [c for c in res if len(c.items) == 1]
    multis = [c for c in res if len(c.items) > 1]
    assert singles and multis
    for c in singles:
        assert c.s_intra == 0.0
        assert c.commission_score == 0.0
    last_multi = max(i for i, c in enumerate(res) if len(c.items) > 1)
    first_single = min(i for i, c in enumerate(res) if len(c.items) == 1)
    assert last_multi < first_single
    for c in res:
        for v in (c.s_intra, c.commission_score, c.legacy_s_intra, c.legacy_commission_score):
            assert math.isfinite(v)


def test_offdiag_floors_negative_at_zero() -> None:
    sub = np.array([[1.0, -0.2], [-0.2, 1.0]], dtype=np.float32)
    assert _cohesion(sub, "offdiag") == 0.0


def _stub(cid: str, legacy_cs: float, legacy_s: float, cs: float) -> ScoredCluster:
    return ScoredCluster(
        cluster_id=cid, items=[], cluster_keywords=[], silhouette_score=0.0,
        w_recent=1.0, w_base_daily=0.0, breakout_surge=1.0, velocity=1.0,
        orbit_resonance=1.0, s_intra=legacy_s, unique_sources=(), unique_sources_count=2,
        commission_score=cs, legacy_s_intra=legacy_s, legacy_commission_score=legacy_cs,
    )


def test_legacy_top_cluster_follows_legacy_score() -> None:
    a = _stub("cluster_a", legacy_cs=1.0, legacy_s=0.6, cs=9.0)
    b = _stub("cluster_b", legacy_cs=2.0, legacy_s=0.7, cs=3.0)
    ranked_by_fields = sorted([a, b], key=lambda c: -c.commission_score)
    assert ranked_by_fields[0] is a
    assert legacy_top_cluster(ranked_by_fields) is b
    assert legacy_top_cluster([]) is None


def test_unknown_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        cluster_and_score(_mixed_items(), now_utc=NOW, cohesion_mode="bogus")
    with pytest.raises(ValueError):
        _cohesion(np.ones((2, 2), dtype=np.float32), "bogus")
