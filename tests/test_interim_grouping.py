"""SIY-221: the interim grouping repair behind ZEITGEIST_INTERIM_GROUPING. Synthetic vectors only; no network."""

from __future__ import annotations

import math
import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import private_run
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import embedding_migration as em
from src.zeitgeist import methodology
from src.zeitgeist.methodology import CachedItem, cluster_and_score, score_and_cluster_signals
from tests.test_embedding_migration_run import DATE, NOW, RecordingStore, _items as _six_groups
from tests.test_instagram_partial_results import _post, _run as _fetch_instagram
from tests.test_understanding_pipeline import _FakeStore

ROOT = Path(__file__).resolve().parents[1]
DIM = 768
SWITCH = "ZEITGEIST_INTERIM_GROUPING"
STAND_IN = "Instagram orbit dispatch"
CALIBRATED = (0.917, 0.887, 0.858)
GROUPING_LINE = re.compile(r"\[grouping\] captionless_items=\d+")


@pytest.fixture(autouse=True)
def _switch_unset(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)


@pytest.fixture(autouse=True)
def _small_minimum(monkeypatch):
    monkeypatch.setattr(em, "MIN_CALIBRATION_ITEMS", 2)


def _basis(i):
    vec = np.zeros(DIM)
    vec[i] = 1.0
    return vec


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _eps(sim):
    """Noise weight that gives two members of one group (own noise dims each) this cosine."""
    return math.sqrt(1.0 / sim - 1.0)


def _member(base, noise_dim, sim):
    return _unit(base + _eps(sim) * _basis(noise_dim))


def _item(key, source, vec, *, title=None, summary=None, hours=2.0, **extra):
    return CachedItem(
        url_hash=key, source_id=source, bucket="ORBIT", tier="O1", origin="orbit",
        title="Synthetic title %s" % key if title is None else title,
        summary="Synthetic summary %s" % key if summary is None else summary,
        published_at=(NOW - timedelta(hours=hours)).isoformat(),
        embedding_f32=np.asarray(vec, dtype=np.float32), extra=dict(extra),
    )


def _stand_ins(count=3, vec_dim=500):
    """Captionless Instagram posts: the stand-in title, no summary, one shared vector."""
    vec = _basis(vec_dim)
    return [_item("standin_%d" % i, "o1_ig_standin_%d" % i, vec, title=STAND_IN, summary="")
            for i in range(count)]


def _world(with_stand_ins=True):
    """Small group A (3 items, 2 sources) next to a larger group D; tight B; loose pair C; two loners."""
    a_base = _unit(0.6 * _basis(0) + 0.8 * _basis(3))  # A leans towards D: centroid cosine 0.8
    items = [
        _item("a_0", "o1_src_a1", _member(a_base, 100, 0.95)),
        _item("a_1", "o1_src_a2", _member(a_base, 101, 0.95)),
        _item("a_2", "o1_src_a1", _member(a_base, 102, 0.95)),
        _item("b_0", "o1_src_b1", _member(_basis(1), 110, 0.95)),
        _item("b_1", "o1_src_b2", _member(_basis(1), 111, 0.95)),
        _item("c_0", "o1_src_c1", _member(_basis(2), 120, 0.87)),
        _item("c_1", "o1_src_c2", _member(_basis(2), 121, 0.87)),
    ]
    items += [_item("d_%d" % i, "o1_src_d%d" % i, _member(_basis(3), 130 + i, 0.95)) for i in range(5)]
    items += [_item("lone_%d" % i, "o1_src_lone_%d" % i, _basis(200 + i)) for i in range(2)]
    if with_stand_ins:
        items += _stand_ins()
    return items


def _groups(clusters):
    return sorted(tuple(sorted(it.url_hash for it in c.items)) for c in clusters)


def _full(clusters):
    """Every public field of every cluster, in order, so two runs can be compared exactly."""
    out = []
    for c in clusters:
        out.append((
            c.cluster_id, [it.url_hash for it in c.items], list(c.cluster_keywords), c.silhouette_score,
            c.w_recent, c.w_base_daily, c.breakout_surge, c.velocity, c.orbit_resonance, c.s_intra,
            c.unique_sources, c.unique_sources_count, c.commission_score, c.threshold_used,
            c.legacy_s_intra, c.legacy_commission_score,
        ))
    return out


# ---- the switch and the stand-in test ----

@pytest.mark.parametrize("value, on", [("1", True), ("0", False), ("", False), ("true", False), (None, False)])
def test_the_switch_is_on_only_for_1(value, on):
    environ = {} if value is None else {SWITCH: value}
    assert methodology.interim_grouping_enabled(environ) is on


def test_a_captionless_post_is_known_by_its_mark_or_for_older_rows_by_the_stand_in_title():
    vec = _basis(0)
    assert methodology.is_captionless(_item("x", "s", vec, title=STAND_IN, summary=""))
    assert methodology.is_captionless({"title": STAND_IN, "summary": ""})
    assert methodology.is_captionless(_item("x", "s", vec, title="A real title", text_missing=True))
    assert not methodology.is_captionless(_item("x", "s", vec, title=STAND_IN, summary="", text_missing=False))
    assert not methodology.is_captionless(_item("x", "s", vec, title=STAND_IN, summary="a caption"))
    assert not methodology.is_captionless(_item("x", "s", vec))


# ---- part 1: stand-ins leave the similarity matrix but are still counted ----

def test_switch_off_the_stand_ins_group_together_as_today():
    groups = _groups(score_and_cluster_signals(_world(), now_utc=NOW))
    assert ("standin_0", "standin_1", "standin_2") in groups


def test_excluded_stand_ins_join_no_group_and_every_other_group_is_unchanged():
    off = _groups(score_and_cluster_signals(_world(), now_utc=NOW))
    on = _groups(score_and_cluster_signals(_world(), now_utc=NOW, exclude_captionless=True))
    assert not any(key.startswith("standin_") for group in on for key in group)
    assert on == [g for g in off if not g[0].startswith("standin_")]


def test_a_stand_in_still_counts_towards_its_sources_weight():
    """The stand-in's source keeps its window count, so a captioned post of that source keeps its weight."""
    def world():
        items = _world()
        items[0].source_id = items[-1].source_id  # a_0 shares a source with a stand-in
        return items

    def group_a(clusters):
        return next(c for c in clusters if any(it.url_hash == "a_0" for it in c.items))

    off = group_a(score_and_cluster_signals(world(), now_utc=NOW))
    on = group_a(score_and_cluster_signals(world(), now_utc=NOW, exclude_captionless=True))
    assert [it.url_hash for it in on.items] == [it.url_hash for it in off.items]
    assert [it.weight for it in on.items] == [it.weight for it in off.items]
    assert on.commission_score == off.commission_score and on.w_recent == off.w_recent


def test_only_stand_ins_give_no_groups():
    assert score_and_cluster_signals(_stand_ins(), now_utc=NOW, exclude_captionless=True) == []


def test_switch_off_grouping_is_byte_for_byte_todays():
    implicit = _full(score_and_cluster_signals(_world(), now_utc=NOW, cutoffs=CALIBRATED))
    explicit = _full(score_and_cluster_signals(_world(), now_utc=NOW, cutoffs=CALIBRATED,
                                               exclude_captionless=False, use_loosest_cutoff=False))
    assert implicit and implicit == explicit


# ---- part 2: calibrated cutoffs run to their loosest tier ----

def test_the_tier_loop_stops_at_the_strictest_tier_by_default():
    clusters = cluster_and_score(_world(False), now_utc=NOW, cutoffs=CALIBRATED)
    assert clusters and all(c.threshold_used == pytest.approx(0.917) for c in clusters)
    assert ("c_0",) in _groups(clusters) and ("c_1",) in _groups(clusters)


def test_use_loosest_cutoff_groups_at_the_loosest_tier():
    clusters = cluster_and_score(_world(False), now_utc=NOW, cutoffs=CALIBRATED, use_loosest_cutoff=True)
    assert clusters and all(c.threshold_used == pytest.approx(0.858) for c in clusters)
    assert ("c_0", "c_1") in _groups(clusters)


def test_the_small_group_survives_the_loosest_tier_and_is_not_merged_into_its_neighbour():
    small = ("a_0", "a_1", "a_2")
    neighbour = ("d_0", "d_1", "d_2", "d_3", "d_4")
    world = _world(False)
    vec = {it.url_hash: np.asarray(it.embedding_f32, dtype=np.float64) for it in world}
    cross = [float(vec[a] @ vec[d]) for a in small for d in neighbour]
    assert 0.7 < min(cross) and max(cross) < 0.858  # related to D, but below every tier
    off = _groups(cluster_and_score(_world(False), now_utc=NOW, cutoffs=CALIBRATED))
    on = _groups(cluster_and_score(_world(False), now_utc=NOW, cutoffs=CALIBRATED, use_loosest_cutoff=True))
    assert small in off and small in on
    assert neighbour in off and neighbour in on
    # Every group of today is kept; the loosest tier only joins the loose pair.
    assert set(off) - set(on) == {("c_0",), ("c_1",)} and set(on) - set(off) == {("c_0", "c_1")}


def test_legacy_cutoffs_are_unchanged_without_the_loosest_flag():
    assert _full(score_and_cluster_signals(_world(), now_utc=NOW)) == _full(
        score_and_cluster_signals(_world(), now_utc=NOW, use_loosest_cutoff=False))


# ---- cluster_for_run: what reaches the clustering, and the calibration maths ----

@pytest.fixture
def score_spy(monkeypatch):
    calls = []
    real = em.score_and_cluster_signals

    def spy(rows, **kw):
        calls.append(dict(kw))
        return real(rows, **kw)

    monkeypatch.setattr(em, "score_and_cluster_signals", spy)
    return calls


PERCENTILES = (0.95, 0.9, 0.88, 0.99)


def _migrated():
    return em.EmbeddingRun(mode="migrated", calibration_state=em.CAL_ACCEPTED, percentiles=PERCENTILES,
                           used=em.EMBED_2, date_str=DATE, items=31)


def _with_stand_ins(count=3):
    return _six_groups() + _stand_ins(count)


def test_migrated_with_interim_grouping_runs_the_calibrated_cutoffs_to_the_loosest_tier(score_spy):
    run = _migrated()
    out = em.cluster_for_run(_six_groups(), run, store=RecordingStore(), now_utc=NOW, interim_grouping=True)
    assert len(score_spy) == 1
    assert score_spy[0]["use_loosest_cutoff"] is True and score_spy[0]["exclude_captionless"] is True
    assert tuple(score_spy[0]["cutoffs"]) == run.cutoffs[:3]
    assert score_spy[0]["near_duplicate_threshold"] == run.cutoffs[3]  # the near-copy cutoff is unchanged
    assert run.cutoffs[2] < run.cutoffs[0]
    assert out and all(c.threshold_used == pytest.approx(run.cutoffs[2]) for c in out)


def test_migrated_without_interim_grouping_is_todays_call(score_spy):
    today = _migrated()
    off = em.cluster_for_run(_six_groups(), today, store=RecordingStore(), now_utc=NOW)
    explicit = _migrated()
    again = em.cluster_for_run(_six_groups(), explicit, store=RecordingStore(), now_utc=NOW, interim_grouping=False)
    assert score_spy[0] == score_spy[1]
    assert set(score_spy[0]) == {"now_utc", "cutoffs", "near_duplicate_threshold"}
    assert _full(off) == _full(again) and today.cutoffs == explicit.cutoffs


def test_migrated_cutoffs_are_taken_without_the_stand_ins():
    # Ten identical stand-ins are enough pairs at similarity 1.0 to lift the strictest cutoff to 1.0.
    with_interim = _migrated()
    em.cluster_for_run(_with_stand_ins(10), with_interim, store=RecordingStore(), now_utc=NOW, interim_grouping=True)
    without = _migrated()
    em.cluster_for_run(_with_stand_ins(10), without, store=RecordingStore(), now_utc=NOW)
    clean = em.cutoffs_at(em.pairwise_similarities(_six_groups()), PERCENTILES)
    assert with_interim.cutoffs == clean
    assert without.cutoffs == em.cutoffs_at(em.pairwise_similarities(_with_stand_ins(10)), PERCENTILES)
    assert without.cutoffs[0] > clean[0]


@pytest.mark.parametrize("run", [
    em.EmbeddingRun(mode="legacy", calibration_state=em.CAL_REJECTED),
    em.EmbeddingRun(mode="switch_over", calibration_state=em.CAL_NONE, deferred=True),
    em.EmbeddingRun(mode="migrated", calibration_state=em.CAL_ACCEPTED, percentiles=PERCENTILES, fallback=True),
], ids=["legacy", "deferred", "migrated_fallback"])
def test_the_legacy_cutoff_path_keeps_its_tier_loop_with_the_switch_on(run, score_spy):
    out = em.cluster_for_run(_six_groups(), run, store=RecordingStore(), now_utc=NOW, interim_grouping=True)
    assert score_spy == [{"now_utc": NOW, "exclude_captionless": True}]
    today = score_and_cluster_signals(_six_groups(), now_utc=NOW)
    assert _full(out) == _full(today)  # no stand-ins here, so nothing else may move


def test_a_maths_failure_falls_back_to_the_legacy_path_and_still_leaves_out_the_stand_ins(monkeypatch, score_spy):
    def broken(*a, **k):
        raise ValueError("synthetic")

    monkeypatch.setattr(em, "cutoffs_at", broken)
    run = _migrated()
    out = em.cluster_for_run(_with_stand_ins(), run, store=RecordingStore(), now_utc=NOW, interim_grouping=True)
    assert run.fallback and score_spy == [{"now_utc": NOW, "exclude_captionless": True}]
    assert not any(it.url_hash.startswith("standin_") for c in out for it in c.items)


def _switch_over():
    pending = _six_groups() + _stand_ins()
    return em.EmbeddingRun(mode="switch_over", calibration_state=em.CAL_NONE, date_str=DATE, items=31,
                           pending_items=pending, pending_rows={it.url_hash: {"url_hash": it.url_hash} for it in pending})


def test_switch_over_percentiles_leave_out_the_stand_ins_and_only_the_calibrated_call_runs_loosest(score_spy):
    run = _switch_over()
    store = RecordingStore()
    em.cluster_for_run(_with_stand_ins(), run, store=store, now_utc=NOW, interim_grouping=True)
    clean = em.pairwise_similarities(_six_groups())
    assert run.percentiles == em.percentile_ranks(clean)
    assert run.cutoffs == em.cutoffs_at(clean, run.percentiles)
    assert run.percentiles != em.percentile_ranks(em.pairwise_similarities(_with_stand_ins()))
    assert [kw.get("use_loosest_cutoff") for kw in score_spy] == [None, True]
    assert all(kw["exclude_captionless"] is True for kw in score_spy)
    saved = [call for call in store.calls if call[0] == "save_calibration"]
    assert saved and saved[0][2]["item_count"] == 31  # the stand-ins still count as items
    assert saved[0][2]["percentiles"] == list(run.percentiles)


# ---- collection marks the stand-in ----

def _collected(monkeypatch):
    posts = [_post("requested_a", "with_caption"), _post("requested_a", "no_caption", caption="")]
    return _fetch_instagram(monkeypatch, posts, ["requested_a"])


def test_collection_marks_captionless_posts_with_the_switch_on(monkeypatch):
    monkeypatch.setenv(SWITCH, "1")
    rows = {row["url"].rsplit("/", 1)[-1]: row for row in _collected(monkeypatch)}
    assert rows["no_caption"]["title"] == STAND_IN and rows["no_caption"]["text_missing"] is True
    assert rows["with_caption"]["text_missing"] is False


@pytest.mark.parametrize("value", [None, "0"])
def test_collection_rows_are_todays_with_the_switch_off(monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(SWITCH, value)
    off = _collected(monkeypatch)
    monkeypatch.setenv(SWITCH, "1")
    on = _collected(monkeypatch)
    assert all("text_missing" not in row for row in off)
    strip = lambda rows: [{k: v for k, v in row.items() if k not in ("text_missing", "fetched_at")} for row in rows]
    assert strip(off) == strip(on)


def test_a_stored_mark_reaches_the_grouping_item():
    marked = pipeline._row_to_cached_item({"url_hash": "h", "title": "t", "text_missing": True}, 0, NOW.isoformat())
    assert methodology.is_captionless(marked)
    unmarked = pipeline._row_to_cached_item({"url_hash": "h", "title": "t"}, 0, NOW.isoformat())
    assert unmarked.extra == {}


# ---- the edition run: wiring and the public line ----

def _staged(tmp_path, monkeypatch, *, switch, model=None, document=None):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    if model is None:
        monkeypatch.delenv("ZEITGEIST_EMBEDDING_MODEL", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_EMBEDDING_MODEL", model)
    calls = {"score": [], "run": []}
    real_score, real_run = pipeline.score_and_cluster_signals, em.cluster_for_run

    def score(rows, **kw):
        calls["score"].append(dict(kw))
        return real_score(rows, **kw)

    def for_run(rows, run, **kw):
        calls["run"].append(dict(kw))
        return real_run(rows, run, **kw)

    def collect(*a, health=None, **kw):
        if health is not None:
            health["apify_batch_succeeded"] = True
        return _with_stand_ins()

    monkeypatch.setattr(pipeline, "score_and_cluster_signals", score)
    monkeypatch.setattr(em, "cluster_for_run", for_run)
    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)
    monkeypatch.setattr(pipeline, "apply_winning_cluster_bonus", lambda roster, items, *a, **kw: roster)
    roster = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    store = _FakeStore(tmp_path, roster)
    if document is not None:
        store.save_calibration(em.CALIBRATION_DOCUMENT, document)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    staged._run_staged(DATE, "blueprint", live_collect=True, offline_demo=True, now_epoch=NOW.timestamp())
    return calls


def _grouping_lines(out):
    return [line for line in out.splitlines() if line.startswith("[grouping]")]


def _rejected():
    return em.calibration_document(em.Calibration("rejected", "2026-11-20", 120, (0.9, 0.8, 0.7, 0.95), 1, 0.5, 1))


@pytest.mark.parametrize("switch", [None, "0"])
def test_the_edition_with_the_switch_off_makes_todays_calls_and_prints_no_line(tmp_path, monkeypatch, capsys, switch):
    calls = _staged(tmp_path, monkeypatch, switch=switch)
    assert [set(kw) for kw in calls["score"]] == [{"now_utc"}] and calls["run"] == []
    assert _grouping_lines(capsys.readouterr().out) == []


def test_the_edition_with_the_switch_on_leaves_out_and_counts_the_stand_ins(tmp_path, monkeypatch, capsys):
    calls = _staged(tmp_path, monkeypatch, switch="1")
    assert [kw.get("exclude_captionless") for kw in calls["score"]] == [True]
    lines = _grouping_lines(capsys.readouterr().out)
    assert lines == ["[grouping] captionless_items=3"] and GROUPING_LINE.fullmatch(lines[0])


@pytest.mark.parametrize("switch, passed", [(None, None), ("1", True)])
def test_the_edition_passes_the_switch_to_the_embedding_run(tmp_path, monkeypatch, capsys, switch, passed):
    calls = _staged(tmp_path, monkeypatch, switch=switch, model=em.EMBED_2, document=_rejected())
    assert len(calls["run"]) == 1 and calls["run"][0].get("interim_grouping") is passed
    assert set(calls["run"][0]) == ({"store", "now_utc"} | ({"interim_grouping"} if passed else set()))
    assert len(_grouping_lines(capsys.readouterr().out)) == (1 if passed else 0)


# ---- the private run groups as the edition does ----

def _accepted():
    return em.calibration_document(em.Calibration("accepted", "2026-11-20", 120, PERCENTILES, 2, 0.3, 3))


@pytest.mark.parametrize("switch, model", [(None, em.EMBED_2), ("1", em.EMBED_2), (None, em.EMBED_001), ("1", em.EMBED_001)])
def test_the_private_run_passes_the_switch_to_its_grouping(monkeypatch, switch, model):
    if switch is not None:
        monkeypatch.setenv(SWITCH, switch)
    seen = []
    monkeypatch.setattr(private_run.em, "cluster_for_run", lambda rows, run, **kw: seen.append(("run", kw)) or [])
    monkeypatch.setattr(private_run.pipeline, "score_and_cluster_signals", lambda rows, **kw: seen.append(("score", kw)) or [])
    store = SimpleNamespace(load_calibration=lambda name: _accepted())
    private_run._cluster_stored(store, [{"embedding_model": model}], [], NOW)
    kind, kw = seen[0]
    assert len(seen) == 1 and kind == ("run" if model == em.EMBED_2 else "score")
    extra = {"run": {"interim_grouping": True}, "score": {"exclude_captionless": True}}[kind] if switch else {}
    base = {"store": None, "now_utc": NOW} if kind == "run" else {"now_utc": NOW}
    assert kw == {**base, **extra}


# ---- workflows pass the switch, defaulting to off ----

DOLLAR = chr(36)
SWITCH_LINE = "ZEITGEIST_INTERIM_GROUPING: " + DOLLAR + "{{ vars.ZEITGEIST_INTERIM_GROUPING || '0' }}"


@pytest.mark.parametrize("name", ["daily_zeitgeist.yml", "private_run.yml", "comparison_run.yml"])
def test_the_workflows_pass_the_switch_once_defaulting_to_off(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(SWITCH_LINE) == 1
