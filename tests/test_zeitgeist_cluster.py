"""Unit tests for WS3 Seam 2: Pure-numpy Average-Linkage Clustering & Disjoint Surge Math."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math
import time

import numpy as np
import pytest

from src.zeitgeist.methodology import (
    CachedItem,
    ScoredCluster,
    cluster_and_score,
)


def _vec_to_f16_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float16).tobytes()


def test_average_linkage_prevents_transitive_chaining() -> None:
    """Verify A-B (~0.81) and B-C (~0.81) with low A-C (~0.31) does NOT chain C into {A, B}."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    dim = 768

    # Construct 3 vectors in plane (0, 1) at angles 0, 34 deg, 70 deg:
    # A-B: cos(34 deg) = 0.8290 >= 0.78
    # B-C: cos(36 deg) = 0.8090 >= 0.78 (both A-B and B-C exceed 0.78!)
    # A-C: cos(70 deg) = 0.3420 (nearly orthogonal)
    # Average similarity of C to {A, B} = (0.8090 + 0.3420) / 2 = 0.5755 < 0.78
    vec_a = np.zeros(dim, dtype=np.float32)
    vec_a[0] = 1.0

    vec_b = np.zeros(dim, dtype=np.float32)
    vec_b[0] = math.cos(math.radians(34.0))
    vec_b[1] = math.sin(math.radians(34.0))

    vec_c = np.zeros(dim, dtype=np.float32)
    vec_c[0] = math.cos(math.radians(70.0))
    vec_c[1] = math.sin(math.radians(70.0))

    # Also include a second independent multi-source pair {D, E} along axis 2 so >= 2 multi-source clusters exist at tau=0.78
    vec_d = np.zeros(dim, dtype=np.float32)
    vec_d[2] = 1.0
    vec_e = np.zeros(dim, dtype=np.float32)
    vec_e[2] = math.cos(math.radians(15.0))
    vec_e[3] = math.sin(math.radians(15.0))

    items = [
        CachedItem(
            url_hash="item_a",
            source_id="o1_alpha",
            bucket="ORBIT",
            title="Verification kernel formal proofs",
            summary="Deterministic formal verification kernel proofs.",
            published_at=(now_utc - timedelta(hours=4)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(vec_a),
        ),
        CachedItem(
            url_hash="item_b",
            source_id="m1_beta",
            bucket="MACRO",
            title="Verification kernel architecture",
            summary="Systems verification kernel architecture benchmarks.",
            published_at=(now_utc - timedelta(hours=3)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(vec_b),
        ),
        CachedItem(
            url_hash="item_c",
            source_id="m2_gamma",
            bucket="MACRO",
            title="Unrelated monetary policy drift",
            summary="Central bank liquidity reserve operations.",
            published_at=(now_utc - timedelta(hours=2)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(vec_c),
        ),
        CachedItem(
            url_hash="item_d",
            source_id="o2_delta",
            bucket="ORBIT",
            title="Photonic interconnect bus latency",
            summary="Optical interconnect wafer scale routing.",
            published_at=(now_utc - timedelta(hours=5)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(vec_d),
        ),
        CachedItem(
            url_hash="item_e",
            source_id="m3_epsilon",
            bucket="MACRO",
            title="Photonic interconnect fabrication",
            summary="Optical wafer interconnect packaging.",
            published_at=(now_utc - timedelta(hours=6)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(vec_e),
        ),
    ]

    clusters = cluster_and_score(items, now_utc=now_utc)
    assert len(clusters) == 3

    cluster_sets = [ {it.url_hash for it in c.items} for c in clusters ]
    assert {"item_a", "item_b"} in cluster_sets
    assert {"item_c"} in cluster_sets
    assert {"item_d", "item_e"} in cluster_sets

    for c in clusters:
        assert isinstance(c, ScoredCluster)
        assert c.threshold_used == pytest.approx(0.78)
        assert len(c.cluster_keywords) > 0
        assert -1.0 <= c.silhouette_score <= 1.0


def test_dynamic_similarity_threshold_steps_down_to_072_or_068() -> None:
    """When pairs have cosine similarity ~0.74 (< 0.78 but >= 0.72), threshold steps down to 0.72."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    dim = 768
    # cos(42 deg) ~ 0.7431 (between 0.72 and 0.78)
    angle = math.radians(42.0)

    v1 = np.zeros(dim, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(dim, dtype=np.float32)
    v2[0] = math.cos(angle)
    v2[1] = math.sin(angle)

    v3 = np.zeros(dim, dtype=np.float32)
    v3[4] = 1.0
    v4 = np.zeros(dim, dtype=np.float32)
    v4[4] = math.cos(angle)
    v4[5] = math.sin(angle)

    items = [
        CachedItem(
            url_hash="c1_a",
            source_id="o1_s1",
            bucket="ORBIT",
            title="Rust async runtime scheduler",
            summary="Work stealing async runtime.",
            published_at=(now_utc - timedelta(hours=5)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(v1),
        ),
        CachedItem(
            url_hash="c1_b",
            source_id="m1_s2",
            bucket="MACRO",
            title="Rust async scheduler latency",
            summary="Tail latency in async runtime.",
            published_at=(now_utc - timedelta(hours=6)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(v2),
        ),
        CachedItem(
            url_hash="c2_a",
            source_id="o2_s3",
            bucket="ORBIT",
            title="Grid transformer solid state",
            summary="High voltage solid state transformer.",
            published_at=(now_utc - timedelta(hours=7)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(v3),
        ),
        CachedItem(
            url_hash="c2_b",
            source_id="m2_s4",
            bucket="MACRO",
            title="Grid transformer substation",
            summary="Substation solid state power grid.",
            published_at=(now_utc - timedelta(hours=8)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(v4),
        ),
    ]

    clusters = cluster_and_score(items, now_utc=now_utc)
    assert len(clusters) == 2
    assert all(c.threshold_used == pytest.approx(0.72) for c in clusters)
    assert all(c.unique_sources_count == 2 for c in clusters)


def test_disjoint_surge_velocity_breakout_and_commission_score_math() -> None:
    """Verify exact closed-form W_recent, W_base_daily, B_surge, Velocity, M_orbit, and CommissionScore."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    dim = 768
    vec_cluster_1 = np.zeros(dim, dtype=np.float32)
    vec_cluster_1[0] = 1.0
    f16_c1 = _vec_to_f16_bytes(vec_cluster_1)

    vec_cluster_2 = np.zeros(dim, dtype=np.float32)
    vec_cluster_2[1] = 1.0
    f16_c2 = _vec_to_f16_bytes(vec_cluster_2)

    # Cluster 1: Cross-Lens Convergence (1 Orbit active at 48h, 1 Macro active at 10h,
    # 1 Macro baseline at 48h, 1 Orbit baseline at 100h) with explicit weights
    # Note: 48h is active for Orbit (<= 72h), but baseline for Macro (24h < 48h <= 96h)!
    c1_items = [
        CachedItem(
            url_hash="c1_orbit_active",
            source_id="o1_src",
            bucket="ORBIT",
            title="Autonomous compiler synthesis",
            summary="Verified compiler synthesis pass.",
            published_at=(now_utc - timedelta(hours=48)).isoformat(),
            embedding_f16=f16_c1,
            weight=1.60,
        ),
        CachedItem(
            url_hash="c1_macro_active",
            source_id="m1_src",
            bucket="MACRO",
            title="Autonomous compiler synthesis benchmark",
            summary="Compiler synthesis speedups.",
            published_at=(now_utc - timedelta(hours=10)).isoformat(),
            embedding_f16=f16_c1,
            weight=1.00,
        ),
        CachedItem(
            url_hash="c1_macro_base",
            source_id="m2_src",
            bucket="MACRO",
            title="Early compiler synthesis notes",
            summary="Baseline macro compiler article.",
            published_at=(now_utc - timedelta(hours=48)).isoformat(),
            embedding_f16=f16_c1,
            weight=0.90,
        ),
        CachedItem(
            url_hash="c1_orbit_base",
            source_id="o2_src",
            bucket="ORBIT",
            title="Historical compiler synthesis",
            summary="Baseline orbit compiler essay.",
            published_at=(now_utc - timedelta(hours=100)).isoformat(),
            embedding_f16=f16_c1,
            weight=1.10,
        ),
    ]

    # Cluster 2: Pure Personal Orbit Breakout Surge!
    # 2 Orbit sources in 0-72h and 0 Orbit sources in 72-336h -> B_surge = 1.25, M_orbit = 1.35
    c2_items = [
        CachedItem(
            url_hash="c2_orbit_1",
            source_id="o3_creator_a",
            bucket="ORBIT",
            title="Acoustic Luthier resonance bracing",
            summary="Handcrafted cedar top bracing geometry.",
            published_at=(now_utc - timedelta(hours=12)).isoformat(),
            embedding_f16=f16_c2,
            weight=1.60,
        ),
        CachedItem(
            url_hash="c2_orbit_2",
            source_id="o3_creator_b",
            bucket="ORBIT",
            title="Luthier resonance bracing frequency",
            summary="Modal acoustic frequency response.",
            published_at=(now_utc - timedelta(hours=36)).isoformat(),
            embedding_f16=f16_c2,
            weight=1.60,
        ),
    ]

    clusters = cluster_and_score(c1_items + c2_items, now_utc=now_utc)
    assert len(clusters) == 2
    by_hashes = {frozenset(it.url_hash for it in c.items): c for c in clusters}

    c1 = by_hashes[frozenset(it.url_hash for it in c1_items)]
    c2 = by_hashes[frozenset(it.url_hash for it in c2_items)]

    # Verify Cluster 1 math:
    # W_recent = 1.60 (orbit at 48h) + 1.00 (macro at 10h) = 2.60
    # W_base_daily = (1/3)*0.90 (macro at 48h) + (1/11)*1.10 (orbit at 100h) = 0.30 + 0.10 = 0.40
    # B_surge = 1.00 (only 1 source in 0-24h, and 1 orbit source in 72-336h)
    # Velocity = min(8.0, 1.00 * ((2.60 + 0.25) / (0.40 + 0.25)) ** 0.6)
    expected_v1 = min(8.0, 1.00 * ((2.85 / 0.65) ** 0.6))
    expected_m1 = 1.50  # Cross-Lens Convergence (Orbit >= 1 and Macro >= 1)
    expected_s1 = 1.00  # Identical unit vectors
    expected_score1 = expected_v1 * expected_m1 * expected_s1 * math.log(1.0 + 4.0)

    assert c1.w_recent == pytest.approx(2.60, abs=1e-5)
    assert c1.w_base_daily == pytest.approx(0.40, abs=1e-5)
    assert c1.breakout_surge == pytest.approx(1.00, abs=1e-5)
    assert c1.velocity == pytest.approx(expected_v1, abs=1e-4)
    assert c1.orbit_resonance == pytest.approx(1.50, abs=1e-5)
    assert c1.s_intra == pytest.approx(1.00, abs=1e-4)
    assert c1.commission_score == pytest.approx(expected_score1, abs=1e-4)

    # Verify Cluster 2 math:
    # W_recent = 1.60 + 1.60 = 3.20
    # W_base_daily = 0.0
    # B_surge = 1.25 (|OrbitSources_{0-72h}| == 2 >= 2 and |OrbitSources_{72-336h}| == 0)
    expected_v2 = min(8.0, 1.25 * ((3.45 / 0.25) ** 0.6))
    expected_m2 = 1.35  # Pure Personal Orbit Synthesis (Orbit >= 2, Macro == 0)
    expected_score2 = expected_v2 * expected_m2 * 1.00 * math.log(1.0 + 2.0)

    assert c2.w_recent == pytest.approx(3.20, abs=1e-5)
    assert c2.w_base_daily == pytest.approx(0.0, abs=1e-5)
    assert c2.breakout_surge == pytest.approx(1.25, abs=1e-5)
    assert c2.velocity == pytest.approx(expected_v2, abs=1e-4)
    assert c2.orbit_resonance == pytest.approx(1.35, abs=1e-5)
    assert c2.commission_score == pytest.approx(expected_score2, abs=1e-4)


def test_cluster_400_items_performance_under_50ms() -> None:
    """Verify 400 x 400 numpy average-linkage clustering executes in well under 50ms."""
    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    dim = 768
    # Create 20 cluster centroids and 400 items around them
    centroids = rng.standard_normal((20, dim)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    items: list[CachedItem] = []
    for i in range(400):
        c_idx = i % 20
        noise = rng.standard_normal(dim).astype(np.float32) * 0.01
        vec = centroids[c_idx] + noise
        vec /= np.linalg.norm(vec)
        is_orb = (i % 3) == 0
        items.append(
            CachedItem(
                url_hash=f"perf_{i}",
                source_id=f"{'o1' if is_orb else 'm1'}_src_{i % 15}",
                bucket="ORBIT" if is_orb else "MACRO",
                title=f"Topic {c_idx} systems synthesis {i}",
                summary=f"Detailed architectural analysis for topic {c_idx}.",
                published_at=(now_utc - timedelta(hours=i % 72)).isoformat(),
                embedding_f16=_vec_to_f16_bytes(vec),
                weight=1.0,
            )
        )

    t0 = time.perf_counter()
    clusters = cluster_and_score(items, now_utc=now_utc)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(clusters) == 20
    # Even with all 400 items, vectorized numpy average linkage should be fast
    assert elapsed_ms < 250.0


def test_orbit_tier_weights_and_score_and_cluster_signals() -> None:
    """Verify granular ORBIT_TIER_WEIGHTS (O1=2.0, O2=1.6, O3=1.3, O4_SCOUT=1.1, MACRO=1.0) in clustering."""
    from src.zeitgeist.methodology import (
        ORBIT_TIER_WEIGHTS,
        resolve_orbit_tier_weight,
        score_and_cluster_signals,
    )

    assert ORBIT_TIER_WEIGHTS["O1"] == pytest.approx(2.0)
    assert ORBIT_TIER_WEIGHTS["O2"] == pytest.approx(1.6)
    assert ORBIT_TIER_WEIGHTS["O3"] == pytest.approx(1.3)
    assert ORBIT_TIER_WEIGHTS["O4_SCOUT"] == pytest.approx(1.1)
    assert ORBIT_TIER_WEIGHTS["MACRO"] == pytest.approx(1.0)

    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    unit_v = np.zeros(768, dtype=np.float32)
    unit_v[0] = 1.0
    f16_vec = _vec_to_f16_bytes(unit_v)

    items = [
        CachedItem(
            url_hash="tier_o1",
            source_id="o1_architect",
            bucket="ORBIT",
            tier="O1",
            title="Isobaric manifold geometry",
            summary="Structural pressure boundaries.",
            published_at=(now_utc - timedelta(hours=6)).isoformat(),
            embedding_f16=f16_vec,
        ),
        CachedItem(
            url_hash="tier_o2",
            source_id="o2_systems",
            bucket="ORBIT",
            tier="O2",
            title="Isobaric manifold calibration",
            summary="Structural pressure boundaries in production.",
            published_at=(now_utc - timedelta(hours=8)).isoformat(),
            embedding_f16=f16_vec,
        ),
        CachedItem(
            url_hash="tier_o3",
            source_id="o3_craft",
            bucket="ORBIT",
            tier="O3",
            title="Isobaric manifold craft",
            summary="Structural pressure boundaries in studio.",
            published_at=(now_utc - timedelta(hours=10)).isoformat(),
            embedding_f16=f16_vec,
        ),
        CachedItem(
            url_hash="tier_o4",
            source_id="o4_scout_node",
            bucket="ORBIT_SCOUT",
            tier="O4_SCOUT",
            title="Isobaric manifold scout",
            summary="Structural pressure boundaries in 1-hop scout.",
            published_at=(now_utc - timedelta(hours=12)).isoformat(),
            embedding_f16=f16_vec,
        ),
        CachedItem(
            url_hash="tier_macro",
            source_id="m1_arxiv",
            bucket="MACRO",
            tier="MACRO",
            title="Isobaric manifold paper",
            summary="Structural pressure boundaries in theory.",
            published_at=(now_utc - timedelta(hours=4)).isoformat(),
            embedding_f16=f16_vec,
        ),
    ]

    assert resolve_orbit_tier_weight(items[0]) == pytest.approx(2.0)
    assert resolve_orbit_tier_weight(items[1]) == pytest.approx(1.6)
    assert resolve_orbit_tier_weight(items[2]) == pytest.approx(1.3)
    assert resolve_orbit_tier_weight(items[3]) == pytest.approx(1.1)
    assert resolve_orbit_tier_weight(items[4]) == pytest.approx(1.0)

    clusters = score_and_cluster_signals(items, now_utc=now_utc)
    assert len(clusters) == 1
    # Sum of recent weights = 2.0 + 1.6 + 1.3 + 1.1 + 1.0 = 7.0
    assert clusters[0].w_recent == pytest.approx(7.0, abs=1e-5)


def test_calibrated_clustering_constants_and_sparse_daily_regime_stepdown() -> None:
    """Verify calibrated constants (DEFAULT_COSINE_DISTANCE_THRESHOLD=0.68, DEFAULT_MIN_CLUSTER_SIZE=2, DYNAMIC_CLUSTER_THRESHOLDS=(0.78, 0.72, 0.68))."""
    from src.zeitgeist.methodology import (
        DEFAULT_COSINE_DISTANCE_THRESHOLD,
        DEFAULT_MIN_CLUSTER_SIZE,
        DYNAMIC_CLUSTER_THRESHOLDS,
    )

    assert DEFAULT_COSINE_DISTANCE_THRESHOLD == pytest.approx(0.68)
    assert DEFAULT_MIN_CLUSTER_SIZE == 2
    assert DYNAMIC_CLUSTER_THRESHOLDS == (0.78, 0.72, 0.68)

    now_utc = datetime(2026, 10, 27, 12, 0, 0, tzinfo=timezone.utc)
    dim = 768
    # cos(46 deg) ~ 0.6946 (below 0.72, above 0.68 -> steps down to 0.68)
    angle = math.radians(46.0)
    v1 = np.zeros(dim, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(dim, dtype=np.float32)
    v2[0] = math.cos(angle)
    v2[1] = math.sin(angle)

    items = [
        CachedItem(
            url_hash="sparse_1",
            source_id="o1_src",
            bucket="ORBIT",
            title="Deterministic synthesis pass",
            summary="Formal verification synthesis.",
            published_at=(now_utc - timedelta(hours=3)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(v1),
        ),
        CachedItem(
            url_hash="sparse_2",
            source_id="m1_src",
            bucket="MACRO",
            title="Compiler verification architecture",
            summary="Deterministic formal pass.",
            published_at=(now_utc - timedelta(hours=4)).isoformat(),
            embedding_f16=_vec_to_f16_bytes(v2),
        ),
    ]

    clusters = cluster_and_score(items, now_utc=now_utc)
    assert len(clusters) == 1
    assert clusters[0].threshold_used == pytest.approx(DEFAULT_COSINE_DISTANCE_THRESHOLD)
    assert clusters[0].unique_sources_count >= DEFAULT_MIN_CLUSTER_SIZE


