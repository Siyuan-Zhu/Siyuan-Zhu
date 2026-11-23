"""Tests for the pure undercurrents arithmetic (realms, eligibility, centroids)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.zeitgeist import undercurrents as uc


def _item(source_id, vec=None, url_hash="h"):
    item = {"source_id": source_id, "url_hash": url_hash}
    if vec is not None:
        item["embedding_f32"] = vec
    return item


def _cluster(cid, items, score=1.0):
    return SimpleNamespace(cluster_id=cid, items=items, commission_score=score)


REALMS = {"m1_feed_a": "tech", "m1_feed_b": "markets", "m1_feed_c": "civic"}


def test_default_category_realms_exact():
    assert uc.DEFAULT_CATEGORY_REALMS == {
        "builder_creator": "tech",
        "tech_community": "tech",
        "tech_ai": "tech",
        "financial_markets": "markets",
        "economics": "markets",
        "policy": "civic",
        "macro_institution": "civic",
        "global_news": "civic",
        "culture": "culture",
        "cultural_pulse": "culture",
        "design": "design",
        "visual_aesthetic": "design",
        "science": "science",
    }
    assert len(uc.DEFAULT_CATEGORY_REALMS) == 13


def test_source_realms_from_feeds_with_overrides_and_unknown():
    feeds = [
        {"id": "m1_feed_a", "category": "tech_ai"},
        {"id": "m1_feed_b", "category": "economics"},
        {"id": "m1_feed_x", "category": "unmapped_category"},
        {"id": "m1_feed_d", "category": "science"},
    ]
    out = uc.source_realms_from_feeds(feeds, overrides={"m1_feed_d": "design"})
    assert out == {"m1_feed_a": "tech", "m1_feed_b": "markets", "m1_feed_d": "design"}
    assert "m1_feed_x" not in out


def test_cluster_sources_sorted_distinct():
    c = _cluster("c1", [_item("m1_feed_b"), _item("m1_feed_a"), _item("m1_feed_b")])
    assert uc.cluster_sources(c) == ["m1_feed_a", "m1_feed_b"]


def test_cluster_sources_skips_empty_and_none_ids():
    c = _cluster("c1", [_item("m1_feed_a"), _item(""), _item(None), _item("m1_feed_b")])
    assert uc.cluster_sources(c) == ["m1_feed_a", "m1_feed_b"]


def test_cluster_sources_accepts_object_items():
    c = _cluster("c1", [SimpleNamespace(source_id="m1_feed_a"), SimpleNamespace(source_id="m1_feed_a")])
    assert uc.cluster_sources(c) == ["m1_feed_a"]


def test_realm_profile_counts_distinct_sources_and_ignores_unlabelled():
    c = _cluster(
        "c1",
        [_item("m1_feed_a"), _item("m1_feed_a"), _item("m1_feed_a2"), _item("m1_feed_b"), _item("m1_feed_zzz")],
    )
    realms = {"m1_feed_a": "tech", "m1_feed_a2": "tech", "m1_feed_b": "markets"}
    assert uc.realm_profile(c, realms) == {"tech": 2, "markets": 1}


def test_unlabelled_source_never_counts_toward_a_realm():
    c = _cluster("c1", [_item("m1_feed_zzz"), _item("m1_feed_yyy")])
    assert uc.realm_profile(c, REALMS) == {}
    assert uc.dominant_realm(c, REALMS) is None


def test_dominant_realm_most_sources():
    c = _cluster("c1", [_item("s1"), _item("s2"), _item("s3")])
    assert uc.dominant_realm(c, {"s1": "tech", "s2": "tech", "s3": "markets"}) == "tech"


def test_dominant_realm_even_split_is_lexically_smaller_and_stable():
    items = [_item("s1"), _item("s2")]
    realms = {"s1": "tech", "s2": "markets"}
    results = set()
    for order in (items, list(reversed(items))):
        results.add(uc.dominant_realm(_cluster("c1", order), realms))
    assert results == {"markets"}


def test_is_eligible_thresholds():
    assert not uc.is_eligible(_cluster("c", [_item("a"), _item("b")]))
    assert not uc.is_eligible(_cluster("c", [_item("a"), _item("a"), _item("a")]))
    assert uc.is_eligible(_cluster("c", [_item("a"), _item("a"), _item("b")]))


def test_cluster_centroid_unit_mean():
    c = _cluster("c", [_item("a", [1.0, 0.0]), _item("b", [0.0, 1.0])])
    cen = uc.cluster_centroid(c)
    assert cen.dtype == np.float32
    np.testing.assert_allclose(cen, [2 ** -0.5, 2 ** -0.5], atol=1e-6)
    assert abs(float(np.linalg.norm(cen)) - 1.0) < 1e-6


def test_cluster_centroid_missing_embeddings_is_zero_vector():
    c = _cluster("c", [_item("a"), _item("b"), _item("c")])
    cen = uc.cluster_centroid(c)
    assert not np.any(cen)


def test_cluster_centroid_mixed_dimensions_keeps_modal_dimension():
    # the missing-embedding fallback is a 768-d zero vector; it must be dropped
    c = _cluster("c", [_item("a"), _item("b", [1.0, 0.0]), _item("c", [0.0, 1.0])])
    cen = uc.cluster_centroid(c)
    assert cen.shape == (2,)
    np.testing.assert_allclose(cen, [2 ** -0.5, 2 ** -0.5], atol=1e-6)
    # a dimension tie goes to the larger dimension
    tie = _cluster("t", [_item("a", [1.0, 0.0]), _item("b", [0.0, 1.0, 0.0])])
    np.testing.assert_allclose(uc.cluster_centroid(tie), [0.0, 1.0, 0.0], atol=1e-6)


def test_cluster_centroid_cancelling_vectors_is_zero():
    c = _cluster("c", [_item("a", [1.0, 0.0]), _item("b", [-1.0, 0.0])])
    cen = uc.cluster_centroid(c)
    assert cen.dtype == np.float32
    assert not np.any(cen)


# ---- Task 2: links and undercurrents ----

_DIM = 8


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _vec_cluster(cid, sources, vec, score=1.0, hashes=None):
    vec = [float(x) for x in vec]
    items = [
        _item(src, vec, url_hash=(hashes[i] if hashes else f"{cid}_h{i}"))
        for i, src in enumerate(sources)
    ]
    return _cluster(cid, items, score)


def _axis(i, extra=None, eps=0.0):
    v = np.zeros(_DIM)
    v[i] = 1.0
    if extra is not None:
        v = v + eps * np.asarray(extra, dtype=np.float64)
    return _unit(v)


FIX_REALMS = {
    "s_tech": "tech",
    "s_tech2": "tech",
    "s_mkt": "markets",
    "s_mkt2": "markets",
    "s_civ": "civic",
    "s_civ2": "civic",
    "s_cul": "culture",
    "s_cul2": "culture",
    "s_sci": "science",
    "s_sci2": "science",
}


def _fixture(shared_mix=0.0, relabel=None):
    """Six eligible clusters; c1,c2,c3 share a direction across three realms."""
    shared = _unit(np.ones(_DIM))
    group = np.zeros(_DIM)
    group[0] = 1.0
    specs = [
        ("c1", ("s_tech", "s_tech2"), group + 0.02 * _axis(5)),
        ("c2", ("s_mkt", "s_mkt2"), group + 0.02 * _axis(6)),
        ("c3", ("s_civ", "s_civ2"), group + 0.02 * _axis(7)),
        ("c4", ("s_cul", "s_cul2"), _axis(1)),
        ("c5", ("s_sci", "s_sci2"), _axis(2)),
        ("c6", ("s_tech", "s_mkt2"), _axis(3)),
    ]
    clusters = []
    for cid, sources, vec in specs:
        v = _unit(vec)
        if shared_mix:
            v = _unit(v + shared_mix * shared)
        clusters.append(
            _vec_cluster(cid, [sources[0], sources[0], sources[1]], v, score=2.0)
        )
    realms = dict(FIX_REALMS)
    if relabel is not None:
        realms = {k: relabel for k in realms}
    return clusters, realms


def _ids(clusters, links):
    return [(clusters[i].cluster_id, clusters[j].cluster_id) for i, j, _ in links]


def test_find_undercurrents_fixture_one_group():
    clusters, realms = _fixture()
    out = uc.find_undercurrents(clusters, realms, percentile=80.0)
    assert len(out) == 1
    u = out[0]
    assert u.cluster_ids == ("c1", "c2", "c3")
    assert u.realms == ("civic", "markets", "tech")
    assert u.source_count == 6
    assert u.score == 6.0 * float(np.log(4))
    assert u.mean_link_similarity > 0.9
    assert u.item_keys == tuple(sorted(u.item_keys))
    assert len(u.item_keys) == 9


def test_link_clusters_returns_sorted_i_lt_j_with_similarity():
    clusters, realms = _fixture()
    links = uc.link_clusters(clusters, realms, percentile=80.0)
    assert [(i, j) for i, j, _ in links] == [(0, 1), (0, 2), (1, 2)]
    assert all(isinstance(s, float) for _, _, s in links)


def test_same_realm_everywhere_yields_nothing():
    clusters, realms = _fixture(relabel="tech")
    assert uc.link_clusters(clusters, realms, percentile=80.0) == []
    assert uc.find_undercurrents(clusters, realms, percentile=80.0) == []


def test_two_realms_in_component_is_rejected():
    clusters, realms = _fixture()
    realms = dict(realms, s_civ="tech", s_civ2="tech")
    assert uc.find_undercurrents(clusters, realms, percentile=80.0) == []


def test_min_sources_gate():
    clusters, realms = _fixture()
    assert len(uc.find_undercurrents(clusters, realms, percentile=80.0, min_sources=6)) == 1
    assert uc.find_undercurrents(clusters, realms, percentile=80.0, min_sources=7) == []


def test_min_clusters_gate():
    clusters, realms = _fixture()
    assert uc.find_undercurrents(clusters, realms, percentile=80.0, min_clusters=4) == []


def test_no_absolute_cutoff_links_survive_uniform_shift():
    base, realms = _fixture()
    shifted, _ = _fixture(shared_mix=1.5)
    cents_base = [uc.cluster_centroid(c) for c in base]
    cents_shift = [uc.cluster_centroid(c) for c in shifted]
    sims_base = [float(cents_base[i] @ cents_base[j]) for i in range(6) for j in range(i + 1, 6)]
    sims_shift = [float(cents_shift[i] @ cents_shift[j]) for i in range(6) for j in range(i + 1, 6)]
    assert min(sims_shift) > min(sims_base) + 0.1  # the whole day moved
    a = [(i, j) for i, j, _ in uc.link_clusters(base, realms, percentile=80.0)]
    b = [(i, j) for i, j, _ in uc.link_clusters(shifted, realms, percentile=80.0)]
    assert a == b == [(0, 1), (0, 2), (1, 2)]


def test_shuffled_input_gives_same_ids():
    clusters, realms = _fixture()
    ref = uc.find_undercurrents(clusters, realms, percentile=80.0)
    rng = np.random.default_rng(7)
    for _ in range(5):
        order = list(rng.permutation(len(clusters)))
        shuffled = [clusters[i] for i in order]
        out = uc.find_undercurrents(shuffled, realms, percentile=80.0)
        assert [u.cluster_ids for u in out] == [u.cluster_ids for u in ref]
        assert [u.score for u in out] == [u.score for u in ref]


def test_fewer_than_two_usable_clusters_is_empty():
    clusters, realms = _fixture()
    assert uc.link_clusters(clusters[:1], realms) == []
    assert uc.link_clusters([], realms) == []
    # ineligible (one source) and zero-centroid clusters are not usable
    thin = _vec_cluster("t", ["s_tech"] * 3, _axis(0))
    zero = _cluster("z", [_item("s_mkt"), _item("s_civ"), _item("s_civ2")])
    assert uc.link_clusters([clusters[0], thin, zero], realms) == []


def test_percentile_population_includes_clusters_without_dominant_realm():
    # Two cross-realm clusters sit at similarity ~0.5; two unlabelled clusters
    # form a very-high-similarity pair that never links but raises the bar.
    a = _vec_cluster("a", ["s_tech", "s_tech", "s_tech2"], _unit([1, 1, 0, 0]))
    b = _vec_cluster("b", ["s_mkt", "s_mkt", "s_mkt2"], _unit([1, 0, 1, 0]))
    u1 = _vec_cluster("u1", ["x1", "x2", "x2"], _unit([0, 0, 0, 1]))
    u2 = _vec_cluster("u2", ["x3", "x4", "x4"], _unit([0, 0, 0.01, 1]))
    clusters = [a, b, u1, u2]
    assert [(i, j) for i, j, _ in uc.link_clusters(clusters, FIX_REALMS, percentile=90.0)] == []
    # without the unlabelled pair in the population, the same pair links
    assert [(i, j) for i, j, _ in uc.link_clusters([a, b], FIX_REALMS, percentile=90.0)] == [(0, 1)]


def test_k_limits_to_most_similar_other_realm_neighbours():
    # a's nearest other-realm neighbours are b then c; with k=1 only a-b can link.
    a = _vec_cluster("a", ["s_tech", "s_tech", "s_tech2"], _unit([1, 0, 0]))
    b = _vec_cluster("b", ["s_mkt", "s_mkt", "s_mkt2"], _unit([1, 0.1, 0]))
    c = _vec_cluster("c", ["s_civ", "s_civ", "s_civ2"], _unit([1, 0.3, 0]))
    clusters = [a, b, c]
    k1 = [(i, j) for i, j, _ in uc.link_clusters(clusters, FIX_REALMS, percentile=0.0, k=1)]
    k2 = [(i, j) for i, j, _ in uc.link_clusters(clusters, FIX_REALMS, percentile=0.0, k=2)]
    assert k1 == [(0, 1)]
    assert k2 == [(0, 1), (0, 2), (1, 2)]


def test_ties_break_on_lower_index():
    a = _vec_cluster("a", ["s_tech", "s_tech", "s_tech2"], _unit([1, 0, 0]))
    b = _vec_cluster("b", ["s_mkt", "s_mkt", "s_mkt2"], _unit([1, 0, 0]))
    c = _vec_cluster("c", ["s_civ", "s_civ", "s_civ2"], _unit([1, 0, 0]))
    links = uc.link_clusters([a, b, c], FIX_REALMS, percentile=0.0, k=1)
    assert [(i, j) for i, j, _ in links] == [(0, 1)]


# ---- Task 3: cross-day null and realm report ----

_NULL_REALM_SOURCES = [
    ("s_tech", "s_tech2"),
    ("s_mkt", "s_mkt2"),
    ("s_civ", "s_civ2"),
]


def _day(prefix, directions, noise, rng):
    """One cluster per direction; realms cycle tech, markets, civic."""
    clusters = []
    for n, vec in enumerate(directions):
        a, b = _NULL_REALM_SOURCES[n % 3]
        v = _unit(np.asarray(vec) + noise * rng.standard_normal(len(vec)))
        clusters.append(_vec_cluster(f"{prefix}{n}", [a, a, b], v))
    return clusters


def _grouped_day(prefix, rng, dim=64, groups=3):
    """Real within-day structure: each group of three realms shares a direction."""
    directions = []
    for _ in range(groups):
        base = _unit(rng.standard_normal(dim))
        directions.extend([base, base, base])
    return _day(prefix, directions, 0.05, rng)


def test_null_pass_ratio_constant():
    assert uc.NULL_PASS_RATIO == 2.0


def test_cross_day_null_unrelated_days_pass():
    rng = np.random.default_rng(11)
    day_a = _grouped_day("a", rng)
    day_b = _grouped_day("b", rng)
    out = uc.cross_day_null(day_a, day_b, FIX_REALMS, percentile=90.0, k=3)
    assert set(out) == {
        "within_links",
        "between_links",
        "within_pairs",
        "between_pairs",
        "within_rate",
        "between_rate",
        "ratio",
    }
    assert out["within_pairs"] == 2 * 27  # 9 clusters, 3 per realm: 27 cross-realm pairs per day
    assert out["between_pairs"] == 9 * 6
    assert out["within_links"] > 0
    assert out["between_links"] == 0
    assert out["ratio"] >= uc.NULL_PASS_RATIO


def test_cross_day_null_finite_ratio_from_nonzero_between_rate():
    # seeded, so the link counts are exact: 18 within, 4 between, over 54 pairs each
    rng = np.random.default_rng(11)
    day_a = _grouped_day("a", rng)
    day_b = _grouped_day("b", rng)
    out = uc.cross_day_null(day_a, day_b, FIX_REALMS, percentile=50.0, k=3)
    assert out["within_links"] == 18 and out["between_links"] == 4
    assert out["within_pairs"] == 54 and out["between_pairs"] == 54
    assert out["between_rate"] > 0
    assert np.isfinite(out["ratio"])
    assert out["ratio"] == out["within_rate"] / out["between_rate"]
    assert abs(out["ratio"] - 4.5) < 1e-9
    assert out["ratio"] >= uc.NULL_PASS_RATIO


def test_cross_day_null_stock_direction_fails():
    # The seed is load-bearing: the stock-direction days must link across days
    # (between_rate > 0) so the ratio lands under NULL_PASS_RATIO. Changing it
    # can silently turn this into a vacuous case.
    rng = np.random.default_rng(5)
    stock = _unit(rng.standard_normal(64))
    day_a = _day("a", [stock] * 9, 0.3, rng)
    day_b = _day("b", [stock] * 9, 0.3, rng)
    out = uc.cross_day_null(day_a, day_b, FIX_REALMS, percentile=90.0, k=3)
    assert out["within_links"] + out["between_links"] > 0
    assert out["between_rate"] > 0
    assert out["ratio"] < uc.NULL_PASS_RATIO


def test_link_clusters_mixed_centroid_dimensions_do_not_raise():
    realms = {"s_tech": "tech", "s_mkt": "markets", "s_civ": "civic", "s_cul": "culture"}
    srcs = lambda s: [s, s, s + "2"]
    realms.update({"s_tech2": "tech", "s_mkt2": "markets", "s_civ2": "civic", "s_cul2": "culture"})
    c0 = _vec_cluster("c0", srcs("s_tech"), _unit([1, 0.1, 0, 0]))
    c1 = _vec_cluster("c1", srcs("s_mkt"), _unit([1, 0.2, 0, 0]))
    c2 = _vec_cluster("c2", srcs("s_civ"), _unit([1, 0.3, 0, 0]))
    odd = _vec_cluster("odd", srcs("s_cul"), _unit([1, 0.1, 0]))
    clusters = [odd, c0, c1, c2]
    links = uc.link_clusters(clusters, realms, percentile=0.0, k=3)
    assert links
    assert all(0 not in (i, j) for i, j, _ in links)
    assert links == uc.link_clusters(list(clusters), realms, percentile=0.0, k=3)


def test_cross_day_null_zero_division_cases():
    # no usable clusters at all: both rates and the ratio are zero
    empty = uc.cross_day_null([], [], FIX_REALMS)
    assert empty == {
        "within_links": 0,
        "between_links": 0,
        "within_pairs": 0,
        "between_pairs": 0,
        "within_rate": 0.0,
        "between_rate": 0.0,
        "ratio": 0.0,
    }
    # within links but no between links: infinite ratio
    a = _vec_cluster("a", ["s_tech", "s_tech", "s_tech2"], _unit([1, 0, 0, 0]))
    b = _vec_cluster("b", ["s_mkt", "s_mkt", "s_mkt2"], _unit([1, 0.01, 0, 0]))
    c = _vec_cluster("c", ["s_tech", "s_tech", "s_tech2"], _unit([0, 0, 1, 0]))
    d = _vec_cluster("d", ["s_mkt", "s_mkt", "s_mkt2"], _unit([0, 0, 1, 0.01]))
    out = uc.cross_day_null([a, b], [c, d], FIX_REALMS, percentile=60.0, k=1)
    assert out["within_links"] == 2 and out["between_links"] == 0
    assert out["between_pairs"] == 2 and out["between_rate"] == 0.0
    assert out["ratio"] == float("inf")
    # pairs exist but nothing links: both rates and the ratio are zero
    out = uc.cross_day_null([a, b], [c, d], FIX_REALMS, percentile=100.0, k=0)
    assert out["within_links"] == 0 and out["between_links"] == 0
    assert out["within_pairs"] == 2 and out["between_pairs"] == 2
    assert out["ratio"] == 0.0


def test_cross_day_null_identical_days_do_not_divide_by_zero():
    rng = np.random.default_rng(3)
    day = _grouped_day("a", rng)
    out = uc.cross_day_null(day, list(day), FIX_REALMS, percentile=90.0, k=3)
    assert out["within_pairs"] > 0 and out["between_pairs"] > 0
    assert np.isfinite(out["within_rate"]) and np.isfinite(out["between_rate"])
    assert out["ratio"] > 0.0  # identical days link within and between: finite or inf, never an error
    assert np.isfinite(out["ratio"]) or out["ratio"] == float("inf")
    # a day against itself with nothing linkable stays at zero
    thin = [_vec_cluster("t", ["s_tech"] * 3, _axis(0))]
    assert uc.cross_day_null(thin, thin, FIX_REALMS)["ratio"] == 0.0


def test_cross_day_null_ignores_unlabelled_and_ineligible_in_pairs():
    clusters, realms = _fixture()
    unlabelled = _vec_cluster("u", ["x1", "x2", "x2"], _axis(4))
    thin = _vec_cluster("t", ["s_tech"] * 3, _axis(0))
    base = uc.cross_day_null(clusters, [], realms, percentile=80.0)
    extra = uc.cross_day_null(clusters + [unlabelled, thin], [], realms, percentile=80.0)
    assert base["within_pairs"] == extra["within_pairs"]
    assert base["between_pairs"] == 0


def test_realm_report_exact_dictionary():
    clusters, realms = _fixture()
    unlabelled = _vec_cluster("u", ["x1", "x2", "x2"], _axis(4))
    found = uc.find_undercurrents(clusters, realms, percentile=80.0)
    clusters = clusters + [unlabelled]
    assert [u.cluster_ids for u in found] == [("c1", "c2", "c3")]
    assert uc.realm_report(clusters, realms, found) == {
        "tech": {"items": 5, "eligible_clusters": 1, "undercurrent_clusters": 1},
        "markets": {"items": 4, "eligible_clusters": 2, "undercurrent_clusters": 1},
        "civic": {"items": 3, "eligible_clusters": 1, "undercurrent_clusters": 1},
        "culture": {"items": 3, "eligible_clusters": 1, "undercurrent_clusters": 0},
        "science": {"items": 3, "eligible_clusters": 1, "undercurrent_clusters": 0},
        "unlabelled": {"items": 3, "eligible_clusters": 1, "undercurrent_clusters": 0},
    }


def test_realm_report_empty_input():
    assert uc.realm_report([], FIX_REALMS, []) == {}
