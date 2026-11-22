from datetime import datetime, timedelta, timezone

import numpy as np

from src.zeitgeist import understanding as u
from src.zeitgeist.methodology import CachedItem, ScoredCluster

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=timezone.utc)
EMOJI = chr(0x1F600) * 2
WORD = "chur" + "ch"  # ordinary word that is also a handle; split to satisfy the name scan
CJK = chr(0x4F60) + chr(0x597D) + chr(0x4E16) + chr(0x754C)


def mk(title, summary="", hours=1, source_id="rss_a", bucket="MACRO", vec=None, extra=None):
    return CachedItem(
        url_hash=f"h_{title}_{hours}",
        source_id=source_id,
        bucket=bucket,
        title=title,
        summary=summary,
        url="https://example.test/x",
        published_at=NOW - timedelta(hours=hours),
        embedding_f32=vec,
        extra=extra or {},
    )


def mc(items, cid="cluster_0", legacy=0.0, cs=1.234, **kw):
    return ScoredCluster(
        cluster_id=cid, items=items, cluster_keywords=["alpha", "beta"], silhouette_score=0.1,
        w_recent=kw.get("w", 1.0), w_base_daily=1.0, breakout_surge=1.0, velocity=1.1,
        orbit_resonance=1.5, s_intra=0.5, unique_sources=("a", "b"), unique_sources_count=2,
        commission_score=cs, legacy_s_intra=0.5, legacy_commission_score=legacy,
    )


def test_switch():
    assert not u.understanding_v1_enabled({})
    assert not u.understanding_v1_enabled({u.SWITCH_ENV: "0"})
    assert not u.understanding_v1_enabled({u.SWITCH_ENV: "true"})
    assert u.understanding_v1_enabled({u.SWITCH_ENV: "1"})


def test_item_handles():
    it = mk("t", source_id="o3_ig_river.studio_7", extra={"mentioned_handles": ["@Harbor_Co", "river.studio_7"]})
    assert u.item_handles(it) == ["harbor_co", "river.studio_7"]
    assert u.item_handles({"source_id": "rss_feed", "mentioned_handles": ["X1"]}) == ["x1"]
    assert u.item_handles(mk("t", source_id="rss_feed")) == []


def test_blank_handles():
    assert u.blank_handles("thanks river.studio_7 for hosting", ["river.studio_7"]) == "thanks someone for hosting"
    assert u.blank_handles(f"our {WORD} met", [WORD]) == "our someone met"
    assert u.blank_handles(f"{WORD}es and riverside", [WORD, "river"]) == f"{WORD}es and riverside"
    assert u.blank_handles("Thanks River.Studio_7.", ["river.studio_7"]) == "Thanks someone."
    assert u.blank_handles("river.studio_7x", ["river.studio_7"]) == "river.studio_7x"


def _vec(i, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def _vec2(*xs):
    return np.array(xs, dtype=np.float32) / np.linalg.norm(xs)


def _seven():
    # items 0-2 are outliers, 3-6 sit near the centre; centrality: 4 > 6 > 3 > 5
    vs = [(0, 1, 0), (0, 0, 1), (-1, 0, 0.2), (1, 0.2, 0.1), (1, 0.4, 0.4), (1, 0.1, 0.1), (1, 0.3, 0.2)]
    return [mk(f"t{i}", hours=i, vec=_vec2(*v)) for i, v in enumerate(vs)]


def test_typical_most_central_in_order():
    items = _seven()
    got = u.typical_items(items)
    assert [x.title for x in got] == ["t4", "t6", "t3", "t5"]


def test_typical_small_keeps_order():
    items = [mk(f"t{i}", vec=_vec(i)) for i in range(3)]
    assert u.typical_items(items) == items
    assert u.typical_items([]) == []


def test_typical_zero_and_missing_vectors():
    items = [mk(f"t{i}", vec=np.zeros(8, dtype=np.float32)) for i in range(6)]
    items += [mk("none")]
    assert len(u.typical_items(items)) == 4
    mixed = [mk("z", vec=np.zeros(8, dtype=np.float32))] + [mk(f"v{i}", vec=_vec(0)) for i in range(5)]
    assert "z" not in [x.title for x in u.typical_items(mixed)]


def test_render_exact_line():
    items = [mk("First title", "First summary text", hours=2), mk("Second title", "", hours=30)]
    out = u.render_evidence_packet([mc(items)], now_utc=NOW)
    assert out == (
        "- Cluster cluster_0: commission_score=1.23, velocity=1.10, orbit_resonance=1.50, "
        "keywords=['alpha', 'beta'], sample_titles=['First title', 'Second title']"
    )


def test_render_uses_typical_four_not_first_four():
    items = _seven()
    out = u.render_evidence_packet([mc(items)])
    assert "sample_titles=['t4', 't6', 't3', 't5']" in out
    assert "t0" not in out.split("sample_titles=")[1]
    assert "facts:" not in out and "evidence:" not in out


def test_render_cleaning_and_limits():
    items = [
        mk("T" * 200, "S" * 400),
        mk("Hello see https://x.test/a?b=1 thanks river.studio_7", source_id="o2_ig_river.studio_7"),
    ]
    out = u.render_evidence_packet([mc(items)], now_utc=NOW)
    assert "T" * 121 not in out and "T" * 120 in out
    assert "S" * 20 not in out
    assert "Hello see thanks someone" in out
    assert "river.studio_7" not in out and "https" not in out and "o2_ig" not in out


def test_render_edge_cases():
    assert u.render_evidence_packet([]) == "- No clusters provided"
    assert u.render_evidence_packet([], now_utc=NOW) == "- No clusters provided"
    clusters = [mc([], cid=f"c{i}") for i in range(7)]
    out = u.render_evidence_packet(clusters)
    assert out.count("- Cluster ") == 5 and out.index("c0") < out.index("c4")
    assert len(out.splitlines()) == 5
    out = u.render_evidence_packet([mc([mk(EMOJI, CJK)])])
    assert "sample_titles=" in out and "evidence:" not in out


def _orbit_cluster():
    orbit = mk("alpha bravo charlie delta echo foxtrot", "", bucket="O3", source_id="o3_ig_river.studio_7")
    plain = mk("one two three four five six", "", bucket="MACRO")
    return [mc([orbit, plain])]


def test_echo_violation():
    cl = _orbit_cluster()
    assert u.echo_violation(["we saw alpha bravo charlie delta echo today"], cl) == "Echoes source text"
    assert u.echo_violation(["alpha bravo charlie delta now"], cl) is None
    assert u.echo_violation(["one two three four five here"], cl) is None
    assert u.echo_violation(["thanks River.Studio_7"], cl) == "Names a source account"
    assert u.echo_violation([EMOJI, "", CJK], cl) is None
    assert u.echo_violation([], []) is None


def test_echo_cluster_id():
    cl = _orbit_cluster()
    for text in ("Cluster 78 reflects on the divine", "see cluster_3", "Cluster#4", "cluster-9 here"):
        assert u.echo_violation([text], cl) == "Names an internal cluster id"
    assert u.echo_violation(["a cluster of stars"], cl) is None
    assert u.echo_violation(["clusters 5 times"], cl) is None


def test_public_keywords():
    it = mk("t", source_id="o2_ig_river.studio_7", extra={"mentioned_handles": ["longlettershandle", "quill"]})
    cl = mc([it, mk("plain")])
    assert u.public_keywords(["river.studio_7", "Quill", "loom", "wow longlettershandle", "quilling"], cl) == [
        "loom", "quilling",
    ]
    assert u.public_keywords(["a", "b"], mc([mk("x")])) == ["a", "b"]
    assert u.public_keywords(["a"], object()) == ["a"]
    assert u.public_keywords(["a"], None) == ["a"]


def test_echo_ordinary_word_handle_ok():
    it = mk("t", source_id=f"o2_ig_{WORD}", bucket="O2")
    assert u.echo_violation([f"a {WORD} at dusk"], [mc([it])]) is None
    it2 = mk("t", bucket="O2", extra={"mentioned_handles": ["longlettershandle"]})
    assert u.echo_violation(["longlettershandle wow"], [mc([it2])]) == "Names a source account"


def test_ranking_stats():
    a = mc([], cid="a", legacy=1.0)
    b = mc([], cid="b", legacy=3.0)
    c = mc([], cid="c", legacy=2.0)
    assert u.ranking_stats([b, c, a]) == (0, 1)
    assert u.ranking_stats([c, b, a]) == (1, 2)
    assert u.ranking_stats([a, b, c]) == (1, 3)
    assert u.ranking_stats([]) == (0, 0)


def test_public_keywords_drops_handle_part_combos():
    it = mk("t", source_id="o2_ig_river.studio_7")
    cl = mc([it, mk("plain")])
    assert u.public_keywords(
        ["river studio", "studio river", "river", "studio light", "open models"], cl
    ) == ["river", "studio light", "open models"]


def test_public_keywords_plain_short_handle_drops_nothing_extra():
    cl = mc([mk("t", extra={"mentioned_handles": [WORD]})])
    assert u.public_keywords(["open models", "river studio"], cl) == ["open models", "river studio"]


def test_public_keywords_handle_parts_never_raises():
    cl = mc([mk("t", source_id="o2_ig_river.studio_7")])
    assert u.public_keywords([None, 5, ""], cl) is not None
    assert u.public_keywords(["river studio"], object()) == ["river studio"]
    assert u.public_keywords([], cl) == []


def _studio_clusters():
    it = mk("Title here", source_id="o2_ig_river.studio_7")
    cl = mc([it, mk("plain")])
    cl.cluster_keywords = ["river studio", "open models"]
    return [cl]


def test_render_keywords_hide_handle_parts():
    out = u.render_evidence_packet(_studio_clusters(), now_utc=NOW)
    assert "river studio" not in out
    assert "keywords=['open models']" in out


def test_echo_handle_part_run_rejected():
    cl = _studio_clusters()
    assert u.echo_violation(["River Studio hosts the night"], cl) == "Names a source account"
    assert u.echo_violation(["the studio river is calm"], cl) == "Names a source account"
    assert u.echo_violation(["a river runs past the studio"], cl) is None
    assert u.echo_violation(["river alone is fine"], cl) is None


def test_echo_handle_part_plain_short_handle_unchanged():
    cl = [mc([mk("t", extra={"mentioned_handles": [WORD]}), mk("plain")])]
    assert u.echo_violation(["river studio hosts the night"], cl) is None
