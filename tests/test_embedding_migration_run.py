"""E6: clustering choice, the switch-over gate and the numbers-only line. Synthetic data only."""

from __future__ import annotations

import copy
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.state_store import InMemoryStateStore
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import embedding_migration as em
from src.zeitgeist import methodology
from src.zeitgeist.methodology import CachedItem
from tests.test_understanding_pipeline import _FakeStore

NOW = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
DIM = 768  # the size methodology assumes for an item without a vector
TITLE = "SYNTHETIC_TITLE_%d_%d"
DATE = "2026-11-21"


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float64)
    return vec / np.linalg.norm(vec)


def _vectors():
    """Six groups of four close vectors (one basis dim each, one noise dim each), then four singletons."""
    basis = np.eye(DIM)
    out = []
    for group in range(6):
        for member in range(4):
            out.append(_unit(basis[group] + 0.3 * basis[6 + group * 4 + member]))
    for single in range(4):
        out.append(basis[30 + single].copy())
    return out


def _items(vectors=None):
    vectors = _vectors() if vectors is None else vectors
    items = []
    for index, vec in enumerate(vectors):
        group, member = divmod(index, 4) if index < 24 else (6 + index - 24, 0)
        source = "o1_src_%d_%d" % (group, member % 2) if index < 24 else "o1_single_%d" % index
        items.append(CachedItem(
            url_hash="hash_%02d" % index, source_id=source, bucket="ORBIT", tier="O1",
            origin="orbit", title=TITLE % (group, member), summary="synthetic body %d" % index,
            published_at=(NOW - timedelta(hours=2 + index)).isoformat(),
            embedding_f32=np.asarray(vec, dtype=np.float32),
        ))
    return items


def _rotation(seed=7):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((DIM, DIM)))
    return q


def _pending(kind):
    if kind == "pass":
        rot = _rotation()
        vectors = [rot @ v for v in _vectors()]
    else:
        rng = np.random.default_rng(11)
        vectors = [_unit(rng.standard_normal(DIM)) for _ in _vectors()]
    return _items(vectors)


def _rows(pending):
    return {item.url_hash: {"url_hash": item.url_hash, "embedding_model": em.EMBED_2}
            for item in pending}


def _switch_over_run(kind):
    pending = _pending(kind)
    return em.EmbeddingRun(mode="switch_over", calibration_state=em.CAL_NONE, date_str=DATE,
                           items=28, pending_items=pending, pending_rows=_rows(pending))


class RecordingStore(InMemoryStateStore):
    def __init__(self, *, calibration_error=None, cache_error=None):
        super().__init__()
        self.calls = []
        self._calibration_error = calibration_error
        self._cache_error = cache_error

    def save_calibration(self, name, document):
        self.calls.append(("save_calibration", name, copy.deepcopy(document)))
        if self._calibration_error is not None:
            raise self._calibration_error
        super().save_calibration(name, document)

    def save_ingestion_cache_day(self, date_str, items_by_hash):
        self.calls.append(("save_ingestion_cache_day", date_str, items_by_hash))
        if self._cache_error is not None:
            raise self._cache_error
        super().save_ingestion_cache_day(date_str, items_by_hash)

    def clear_calibration(self, name):
        self.calls.append(("clear_calibration", name))
        return super().clear_calibration(name)


@pytest.fixture(autouse=True)
def _small_minimum(monkeypatch):
    monkeypatch.setattr(em, "MIN_CALIBRATION_ITEMS", 2)


@pytest.fixture
def score_spy(monkeypatch):
    calls = []
    real = em.score_and_cluster_signals

    def spy(rows, **kw):
        calls.append(dict(kw))
        return real(rows, **kw)

    monkeypatch.setattr(em, "score_and_cluster_signals", spy)
    return calls


def _signature(clusters):
    return [(c.cluster_id, sorted(methodology._get_item_hash(i) for i in c.items)) for c in clusters]


def _fields(line):
    assert line.startswith("[embedding] ")
    return dict(part.split("=", 1) for part in line[len("[embedding] "):].split(" "))


def test_switch_over_pass_uses_embedding2_and_saves_calibration_then_cache():
    items = _items()
    run = _switch_over_run("pass")
    store = RecordingStore()
    expected_p = list(em.percentile_ranks(em.pairwise_similarities(_items())))

    out = em.cluster_for_run(items, run, store=store, now_utc=NOW)

    pending_ids = {id(item) for item in run.pending_items}
    assert out and all(id(item) in pending_ids for c in out for item in c.items)
    assert [c[0] for c in store.calls] == ["save_calibration", "save_ingestion_cache_day"]
    _, name, document = store.calls[0]
    assert name == em.CALIBRATION_DOCUMENT
    assert document["status"] == "accepted" and document["percentiles"] == expected_p
    assert document["date"] == DATE and document["item_count"] == 28
    assert em.parse_calibration(store.load_calibration(em.CALIBRATION_DOCUMENT))[0] == em.CAL_ACCEPTED
    assert store.calls[1][1:] == (run.date_str, run.pending_rows)
    assert run.used == em.EMBED_2 and run.gate.passed and run.percentiles == tuple(expected_p)
    assert (run.multi_source, run.largest_share) == em.cluster_shape(out)
    fields = _fields(em.embedding_line(run))
    assert (fields["used"], fields["switch_over"], fields["pass"], fields["write_failed"]) == ("2", "1", "1", "0")


def test_switch_over_fail_returns_001_clusters_and_writes_rejected_only():
    items = _items()
    fresh = methodology.score_and_cluster_signals(_items(), now_utc=NOW)
    run = _switch_over_run("fail")
    store = RecordingStore()

    out = em.cluster_for_run(items, run, store=store, now_utc=NOW)

    assert _signature(out) == _signature(fresh)
    assert [c[0] for c in store.calls] == ["save_calibration"]
    document = store.calls[0][2]
    assert document["status"] == "rejected" and document["top5_survivors"] < 3
    assert run.used == em.EMBED_001 and not run.gate.passed
    fields = _fields(em.embedding_line(run))
    assert (fields["used"], fields["switch_over"], fields["pass"], fields["write_failed"]) == ("1", "1", "0", "0")


def test_migrated_uses_cutoffs_at_stored_percentiles_without_writes(score_spy):
    items = _items()
    percentiles = (0.95, 0.9, 0.88, 0.99)
    run = em.EmbeddingRun(mode="migrated", calibration_state=em.CAL_ACCEPTED, percentiles=percentiles,
                          used=em.EMBED_2, date_str=DATE, items=28)
    store = RecordingStore()
    cut = em.cutoffs_at(em.pairwise_similarities(_items()), percentiles)

    out = em.cluster_for_run(items, run, store=store, now_utc=NOW)

    assert len(score_spy) == 1
    assert tuple(score_spy[0]["cutoffs"]) == cut[:3]
    assert score_spy[0]["near_duplicate_threshold"] == cut[3]
    assert run.cutoffs == cut and store.calls == []
    assert (run.multi_source, run.largest_share) == em.cluster_shape(out)
    assert _fields(em.embedding_line(run))["used"] == "2"


@pytest.mark.parametrize("run", [
    em.EmbeddingRun(mode="legacy", calibration_state=em.CAL_REJECTED),
    em.EmbeddingRun(mode="switch_over", calibration_state=em.CAL_NONE, deferred=True),
    em.EmbeddingRun(mode="switch_over", calibration_state=em.CAL_NONE, fallback=True),
    em.EmbeddingRun(mode="migrated", calibration_state=em.CAL_ACCEPTED,
                    percentiles=(0.95, 0.9, 0.88, 0.99), fallback=True),
], ids=["legacy", "deferred", "switch_over_fallback", "migrated_fallback"])
def test_fallback_deferred_and_legacy_use_todays_cutoffs_without_writes(run, score_spy):
    store = RecordingStore()
    out = em.cluster_for_run(_items(), run, store=store, now_utc=NOW, cohesion_mode="offdiag")
    assert len(score_spy) == 1
    assert score_spy[0] == {"now_utc": NOW, "cohesion_mode": "offdiag"}
    assert store.calls == [] and run.cutoffs is None
    assert (run.multi_source, run.largest_share) == em.cluster_shape(out)


def test_score_kwargs_reach_every_switch_over_call(score_spy):
    em.cluster_for_run(_items(), _switch_over_run("pass"), store=RecordingStore(), now_utc=NOW,
                       cohesion_mode="offdiag")
    assert len(score_spy) == 2 and all(kw["cohesion_mode"] == "offdiag" for kw in score_spy)
    assert "cutoffs" not in score_spy[0] and "cutoffs" in score_spy[1]


def test_calibration_write_failure_returns_001_clusters_and_skips_cache():
    fresh = methodology.score_and_cluster_signals(_items(), now_utc=NOW)
    run = _switch_over_run("pass")
    store = RecordingStore(calibration_error=RuntimeError("synthetic write failure"))

    out = em.cluster_for_run(_items(), run, store=store, now_utc=NOW)

    assert _signature(out) == _signature(fresh)
    assert [c[0] for c in store.calls] == ["save_calibration"]
    assert run.used == em.EMBED_001 and run.write_failed
    fields = _fields(em.embedding_line(run))
    assert (fields["used"], fields["write_failed"]) == ("1", "1")


def test_cache_write_failure_after_pass_still_returns_embedding2_clusters():
    run = _switch_over_run("pass")
    store = RecordingStore(cache_error=RuntimeError("synthetic write failure"))

    out = em.cluster_for_run(_items(), run, store=store, now_utc=NOW)

    pending_ids = {id(item) for item in run.pending_items}
    assert out and all(id(item) in pending_ids for c in out for item in c.items)
    assert [c[0] for c in store.calls] == ["save_calibration", "save_ingestion_cache_day"]
    assert run.used == em.EMBED_2 and run.write_failed
    fields = _fields(em.embedding_line(run))
    assert (fields["used"], fields["write_failed"], fields["pass"]) == ("2", "1", "1")


@pytest.mark.parametrize("document", [
    {"status": "accepted"},
    ["not", "a", "dict"],
    "truncated",
    dict(em.calibration_document(em.Calibration("accepted", DATE, 28, (0.9, 0.8, 0.7, 0.95), 3, 0.2, 4)),
         schema_version=2),
], ids=["truncated", "list", "string", "future_schema"])
def test_invalid_document_runs_legacy_and_is_never_overwritten(document, score_spy):
    store = RecordingStore()
    store._calibrations[em.CALIBRATION_DOCUMENT] = copy.deepcopy(document)
    run = em.plan_embedding_run(store, em.EMBED_2)
    assert (run.mode, run.calibration_state) == ("legacy", em.CAL_INVALID)

    em.cluster_for_run(_items(), run, store=store, now_utc=NOW)

    assert store.calls == [] and store._calibrations[em.CALIBRATION_DOCUMENT] == document
    assert all("cutoffs" not in kw for kw in score_spy)
    assert _fields(em.embedding_line(run))["calibration"] == "3"


def test_unreadable_document_runs_legacy_without_raising():
    class Unreadable(RecordingStore):
        def load_calibration(self, name):
            raise RuntimeError("synthetic read failure")

    store = Unreadable()
    run = em.plan_embedding_run(store, em.EMBED_2)
    assert (run.mode, run.calibration_state) == ("legacy", em.CAL_UNREADABLE)
    em.cluster_for_run(_items(), run, store=store, now_utc=NOW)
    assert store.calls == []


@pytest.mark.parametrize("pending", [None, []], ids=["none", "empty"])
def test_switch_over_without_pending_items_is_no_decision(pending, score_spy):
    run = em.EmbeddingRun(mode="switch_over", calibration_state=em.CAL_NONE, date_str=DATE,
                          items=0, pending_items=pending, pending_rows={} if pending == [] else None)
    store = RecordingStore()
    out = em.cluster_for_run(_items(), run, store=store, now_utc=NOW)
    assert store.calls == [] and run.gate is None and run.used == em.EMBED_001
    assert len(score_spy) == 1 and "cutoffs" not in score_spy[0]
    assert (run.multi_source, run.largest_share) == em.cluster_shape(out)
    assert _fields(em.embedding_line(run))["pass"] == "-1"


def _strip_embedding(item):
    item.embedding_f32 = None
    item.embedding_f16 = None
    return item


@pytest.mark.parametrize("which", ["items", "pending"])
def test_switch_over_with_fewer_than_two_usable_vectors_is_no_decision(which, score_spy):
    items = _items()
    run = _switch_over_run("pass")
    target = items if which == "items" else run.pending_items
    for item in target[1:]:
        _strip_embedding(item)
    store = RecordingStore()
    em.cluster_for_run(items, run, store=store, now_utc=NOW)
    assert store.calls == [] and run.gate is None and run.used == em.EMBED_001
    assert len(score_spy) == 1 and "cutoffs" not in score_spy[0]


def test_migrated_with_fewer_than_two_usable_vectors_uses_todays_cutoffs(score_spy):
    items = [_strip_embedding(item) for item in _items()]
    run = em.EmbeddingRun(mode="migrated", calibration_state=em.CAL_ACCEPTED,
                          percentiles=(0.95, 0.9, 0.88, 0.99), used=em.EMBED_2)
    store = RecordingStore()
    em.cluster_for_run(items, run, store=store, now_utc=NOW)
    assert store.calls == [] and run.cutoffs is None
    assert len(score_spy) == 1 and "cutoffs" not in score_spy[0]


def test_items_without_vectors_are_left_out_of_similarities():
    items = _items() + [_strip_embedding(CachedItem(url_hash="hash_bare_%d" % n, source_id="o1_bare_%d" % n,
                                                    bucket="ORBIT", tier="O1", published_at=NOW.isoformat()))
                        for n in range(2)]
    run = _switch_over_run("pass")
    run.pending_items = run.pending_items + [_strip_embedding(CachedItem(url_hash="hash_bare_x", source_id="o1_bare_x",
                                                                         bucket="ORBIT", tier="O1"))]
    store = RecordingStore()
    em.cluster_for_run(items, run, store=store, now_utc=NOW)
    expected_p = list(em.percentile_ranks(em.pairwise_similarities(_items())))
    assert store.calls[0][2]["percentiles"] == expected_p
    assert all(np.isfinite(c) for c in run.cutoffs)


# ---- _run_staged wiring ----

def _rejected_document():
    return em.calibration_document(em.Calibration("rejected", "2026-11-20", 120, (0.9, 0.8, 0.7, 0.95), 1, 0.5, 1))


def _staged(tmp_path, monkeypatch, *, model, document=None):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    if model is None:
        monkeypatch.delenv("ZEITGEIST_EMBEDDING_MODEL", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_EMBEDDING_MODEL", model)
    monkeypatch.setattr(em, "MIN_CALIBRATION_ITEMS", 100)
    seen = {}

    def collect(*a, health=None, **kw):
        seen["embedding_run"] = kw.get("embedding_run")
        seen["has_keyword"] = "embedding_run" in kw
        if health is not None:
            health["apify_batch_succeeded"] = True
        return _items()

    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)
    monkeypatch.setattr(pipeline, "apply_winning_cluster_bonus", lambda roster, items, *a, **kw: roster)
    roster = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    store = _FakeStore(tmp_path, roster)
    if document is not None:
        store.save_calibration(em.CALIBRATION_DOCUMENT, document)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    staged._run_staged(DATE, "blueprint", live_collect=True, offline_demo=True, now_epoch=NOW.timestamp())
    return seen


def _embedding_lines(out):
    return [line for line in out.splitlines() if line.startswith("[embedding]")]


def _assert_no_titles(out):
    assert not re.search(r"SYNTHETIC_TITLE|synthetic body|hash_\d", out)


def test_staged_without_variable_passes_no_run_and_prints_no_line(tmp_path, monkeypatch, capsys):
    seen = _staged(tmp_path, monkeypatch, model=None)
    out = capsys.readouterr().out
    assert seen["embedding_run"] is None and not seen["has_keyword"]
    assert _embedding_lines(out) == []
    _assert_no_titles(out)


def test_staged_with_embedding2_and_rejected_document_prints_one_line(tmp_path, monkeypatch, capsys):
    seen = _staged(tmp_path, monkeypatch, model=em.EMBED_2, document=_rejected_document())
    out = capsys.readouterr().out
    assert isinstance(seen["embedding_run"], em.EmbeddingRun)
    lines = _embedding_lines(out)
    assert len(lines) == 1
    fields = _fields(lines[0])
    assert (fields["calibration"], fields["used"]) == ("2", "1")
    _assert_no_titles(out)


def test_staged_with_unrecognized_model_passes_no_run(tmp_path, monkeypatch, capsys):
    seen = _staged(tmp_path, monkeypatch, model="gemini-embedding-5")
    out = capsys.readouterr().out
    assert seen["embedding_run"] is None
    assert _embedding_lines(out) == []
    _assert_no_titles(out)


def test_staged_offline_path_never_plans_a_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    monkeypatch.setenv("ZEITGEIST_EMBEDDING_MODEL", em.EMBED_2)
    seen = {}

    def collect(*a, health=None, **kw):
        seen["embedding_run"] = kw.get("embedding_run")
        return _items()

    monkeypatch.setattr(pipeline, "collect_96h_signals", collect)
    monkeypatch.setattr(pipeline, "apply_winning_cluster_bonus", lambda roster, items, *a, **kw: roster)
    roster = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=_FakeStore(tmp_path, roster))
    staged._run_staged(DATE, "blueprint", live_collect=False, offline_demo=True, now_epoch=NOW.timestamp())
    assert seen["embedding_run"] is None
    assert _embedding_lines(capsys.readouterr().out) == []



# ---- Task T5: the switch-over survives any failure after the embedding-2 call ----

import hashlib  # noqa: E402

from src.zeitgeist import collectors  # noqa: E402
from src.zeitgeist import model_choice  # noqa: E402
from src.zeitgeist import spend_guard  # noqa: E402
from src.zeitgeist.spend_guard import SpendBlocked  # noqa: E402
from tests.test_pipeline_launch_acceptance import _live_fixture  # noqa: E402

COLLECT_NOW = datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc).timestamp()


class StopAfterClustering(Exception):
    pass


def _vector(model, text, dim=768):
    text = text if isinstance(text, str) else text.parts[0].text  # embedding-2 sends one-text Contents
    seed = int(hashlib.sha256((model + text).encode()).hexdigest()[:8], 16)
    return [float(v) for v in np.random.default_rng(seed).standard_normal(dim)]


def _e2_reply(kind):
    def reply(contents):
        if kind == "short":
            return [SimpleNamespace(values=_vector(em.EMBED_2, t)) for t in contents[:-1]]
        if kind == "dimension":
            return [SimpleNamespace(values=_vector(em.EMBED_2, t, dim=767)) for t in contents]
        if kind == "nonfinite":
            rows = [_vector(em.EMBED_2, t) for t in contents]
            rows[0][5] = float("nan")
            return [SimpleNamespace(values=row) for row in rows]
        return [SimpleNamespace(values=_vector(em.EMBED_2, t)) for t in contents]
    return reply


def _night_client(e2=None):
    """001 always answers; e2 is a reply builder or an exception raised after the call is recorded."""
    calls = []

    def embed(*, model, contents, config):
        calls.append(model)
        if model == em.EMBED_2 and isinstance(e2, BaseException):
            raise e2
        build = _e2_reply(e2) if model == em.EMBED_2 and isinstance(e2, str) else None
        embeddings = build(contents) if build else [SimpleNamespace(values=_vector(model, t)) for t in contents]
        return SimpleNamespace(embeddings=embeddings)

    return SimpleNamespace(models=SimpleNamespace(embed_content=embed)), calls


class NightStore(InMemoryStateStore):
    """Holds the calibration across nights; records cache days; stops the run after clustering."""

    def __init__(self, rows):
        super().__init__()
        self._rows = rows
        self.cache_days = []

    def load_ingestion_cache(self, **kwargs):
        return copy.deepcopy(self._rows)

    def save_ingestion_cache_day(self, date_str, items_by_hash):
        self.cache_days.append(copy.deepcopy(items_by_hash))

    def load_pulse_feed_history(self, days=30):
        raise StopAfterClustering()


def _night(tmp_path, monkeypatch, capsys, store, client):
    monkeypatch.setattr(pipeline, "embed_missing_items", collectors.embed_missing_items)
    monkeypatch.setattr("google.genai.Client", lambda **kw: client)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda c, **kw: c)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    with pytest.raises(StopAfterClustering):
        staged._run_staged("2026-10-29", live_collect=True, now_epoch=COLLECT_NOW)
    return _embedding_lines(capsys.readouterr().out)


@pytest.fixture
def night(tmp_path, monkeypatch):
    seed = _live_fixture(monkeypatch, tmp_path)
    for name in (model_choice.DIRECTOR_ENV, model_choice.JUDGE_ENV, "ZEITGEIST_UNDERSTANDING_V1",
                 "ZEITGEIST_OFFLINE_DEMO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(model_choice.EMBEDDING_ENV, em.EMBED_2)
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    return NightStore(seed.load_ingestion_cache())


def _e2_rows_written(store):
    return [row for day in store.cache_days for row in day.values() if row.get("embedding_model") == em.EMBED_2]


FAILURES = {
    "short": "short",
    "dimension": "dimension",
    "nonfinite": "nonfinite",
    "generic-after-call": RuntimeError("synthetic failure after the call"),
    "value-error-after-call": ValueError("synthetic failure after the call"),
    "marked-refusal": None,
}


def _failure(name):
    if name == "marked-refusal":
        error = SpendBlocked("Unpriced SDK request option")
        error.refused_before_dispatch = True
        return error
    return FAILURES[name]


@pytest.mark.parametrize("name", list(FAILURES))
def test_switch_over_failure_after_the_call_clusters_on_001_and_writes_nothing(
        tmp_path, monkeypatch, capsys, night, name):
    client, calls = _night_client(_failure(name))
    lines = _night(tmp_path, monkeypatch, capsys, night, client)
    assert em.EMBED_2 in calls
    assert night.load_calibration(em.CALIBRATION_DOCUMENT) is None
    assert _e2_rows_written(night) == []
    assert len(lines) == 1
    fields = _fields(lines[0])
    assert (fields["used"], fields["switch_over"], fields["fallback"], fields["pass"]) == ("1", "1", "1", "-1")
    assert fields["cut"] == "-1"


@pytest.mark.parametrize("name", ["short", "dimension", "nonfinite", "generic-after-call"])
def test_migrated_failure_after_the_call_clusters_on_001_and_writes_no_embedding2_rows(
        tmp_path, monkeypatch, capsys, night, name):
    night.save_calibration(em.CALIBRATION_DOCUMENT, em.calibration_document(
        em.Calibration("accepted", "2026-10-28", 120, (0.9, 0.8, 0.7, 0.95), 3, 0.2, 4)))
    client, calls = _night_client(_failure(name))
    lines = _night(tmp_path, monkeypatch, capsys, night, client)
    assert calls[0] == em.EMBED_2 and em.EMBED_001 in calls
    assert night.cache_days == []
    fields = _fields(lines[0])
    assert (fields["used"], fields["switch_over"], fields["fallback"], fields["cut"]) == ("1", "0", "1", "-1")


def test_switch_over_unmarked_spend_block_after_the_call_propagates(tmp_path, monkeypatch, night):
    error = SpendBlocked("Receipt mismatch")
    client, calls = _night_client(error)
    monkeypatch.setattr(pipeline, "embed_missing_items", collectors.embed_missing_items)
    monkeypatch.setattr("google.genai.Client", lambda **kw: client)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda c, **kw: c)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=night)
    with pytest.raises(SpendBlocked) as caught:
        staged._run_staged("2026-10-29", live_collect=True, now_epoch=COLLECT_NOW)
    assert caught.value is error
    assert night.load_calibration(em.CALIBRATION_DOCUMENT) is None and _e2_rows_written(night) == []


def test_switch_over_failure_in_the_held_rows_falls_back(tmp_path, monkeypatch, capsys, night):
    real = pipeline._row_to_cached_item

    def broken(row, idx, ref_dt_str):
        if isinstance(row, dict) and row.get("embedding_model") == em.EMBED_2:
            raise ValueError("synthetic failure after the call")
        return real(row, idx, ref_dt_str)

    monkeypatch.setattr(pipeline, "_row_to_cached_item", broken)
    client, calls = _night_client()
    lines = _night(tmp_path, monkeypatch, capsys, night, client)
    assert em.EMBED_2 in calls
    assert night.load_calibration(em.CALIBRATION_DOCUMENT) is None and _e2_rows_written(night) == []
    fields = _fields(lines[0])
    assert (fields["used"], fields["fallback"], fields["pass"]) == ("1", "1", "-1")


@pytest.mark.parametrize("migrated", [False, True], ids=["switch_over", "migrated"])
def test_an_error_escaping_the_embedding2_embedder_falls_back(tmp_path, monkeypatch, capsys, night, migrated):
    if migrated:
        night.save_calibration(em.CALIBRATION_DOCUMENT, em.calibration_document(
            em.Calibration("accepted", "2026-10-28", 120, (0.9, 0.8, 0.7, 0.95), 3, 0.2, 4)))
    before = night.load_calibration(em.CALIBRATION_DOCUMENT)
    client, calls = _night_client()
    real = collectors.embed_missing_items

    def embedder(items, **kwargs):
        if kwargs.get("model") == em.EMBED_2:
            real(items, **kwargs)
            raise KeyError("synthetic failure after the call")
        return real(items, **kwargs)

    monkeypatch.setattr("google.genai.Client", lambda **kw: client)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda c, **kw: c)
    monkeypatch.setattr(pipeline, "embed_missing_items", embedder)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=night)
    with pytest.raises(StopAfterClustering):
        staged._run_staged("2026-10-29", live_collect=True, now_epoch=COLLECT_NOW)
    lines = _embedding_lines(capsys.readouterr().out)
    assert em.EMBED_2 in calls
    assert night.load_calibration(em.CALIBRATION_DOCUMENT) == before
    assert _e2_rows_written(night) == []
    fields = _fields(lines[0])
    assert (fields["used"], fields["fallback"], fields["cut"]) == ("1", "1", "-1")


def test_the_night_after_a_failed_switch_over_performs_a_normal_switch_over(tmp_path, monkeypatch, capsys, night):
    broken, _ = _night_client(RuntimeError("synthetic failure after the call"))
    first = _night(tmp_path, monkeypatch, capsys, night, broken)
    assert _fields(first[0])["fallback"] == "1"
    assert night.load_calibration(em.CALIBRATION_DOCUMENT) is None
    working, calls = _night_client()
    second = _night(tmp_path, monkeypatch, capsys, night, working)
    fields = _fields(second[0])
    assert (fields["switch_over"], fields["fallback"], fields["calibration"]) == ("1", "0", "0")
    assert fields["pass"] in ("0", "1") and fields["cut"] != "-1"
    state, _ = em.parse_calibration(night.load_calibration(em.CALIBRATION_DOCUMENT))
    assert state in (em.CAL_ACCEPTED, em.CAL_REJECTED)



# ---- a failure in the cutoff or gate maths falls back instead of stopping the run ----


def _boom(*args, **kwargs):
    raise RuntimeError("synthetic maths failure")


def _migrated_run():
    return em.EmbeddingRun(mode="migrated", calibration_state=em.CAL_ACCEPTED, percentiles=(0.95, 0.9, 0.88, 0.99),
                           used=em.EMBED_2, date_str=DATE, items=28)


@pytest.mark.parametrize("target", ["percentile_ranks", "cutoffs_at", "gate_report"])
def test_switch_over_maths_failure_returns_todays_001_clusters_and_writes_nothing(monkeypatch, target):
    fresh = methodology.score_and_cluster_signals(_items(), now_utc=NOW)
    monkeypatch.setattr(em, target, _boom)
    run = _switch_over_run("pass")
    store = RecordingStore()

    out = em.cluster_for_run(_items(), run, store=store, now_utc=NOW)

    assert _signature(out) == _signature(fresh)
    assert store.calls == []
    assert run.fallback and run.used == em.EMBED_001
    assert (run.percentiles, run.cutoffs, run.gate) == (None, None, None)
    assert (run.multi_source, run.largest_share) == em.cluster_shape(out)
    fields = _fields(em.embedding_line(run))
    assert (fields["used"], fields["switch_over"], fields["fallback"], fields["pass"], fields["cut"], fields["p"]) == (
        "1", "1", "1", "-1", "-1", "-1")


@pytest.mark.parametrize("target", ["cutoffs_at", "_usable_similarities"])
def test_migrated_maths_failure_returns_fixed_cutoff_clusters_and_shows_the_fallback(monkeypatch, target):
    fixed = methodology.score_and_cluster_signals(_items(), now_utc=NOW, cohesion_mode="offdiag")
    monkeypatch.setattr(em, target, _boom)
    run = _migrated_run()
    store = RecordingStore()

    out = em.cluster_for_run(_items(), run, store=store, now_utc=NOW, cohesion_mode="offdiag")

    assert out and store.calls == []
    assert _signature(out) == _signature(fixed)
    assert run.fallback and run.cutoffs is None and run.gate is None
    assert run.percentiles == (0.95, 0.9, 0.88, 0.99)
    fields = _fields(em.embedding_line(run))
    assert (fields["fallback"], fields["cut"], fields["switch_over"]) == ("1", "-1", "0")


def test_migrated_failure_after_cutoffs_are_set_leaves_no_half_set_state(monkeypatch):
    real = em.score_and_cluster_signals
    calls = []

    def score(rows, **kw):
        calls.append(dict(kw))
        if "cutoffs" in kw:
            raise RuntimeError("synthetic maths failure")
        return real(rows, **kw)

    monkeypatch.setattr(em, "score_and_cluster_signals", score)
    run = _migrated_run()
    out = em.cluster_for_run(_items(), run, store=RecordingStore(), now_utc=NOW)
    assert out and run.fallback and run.cutoffs is None and run.gate is None
    assert "cutoffs" in calls[0] and calls[-1] == {"now_utc": NOW}


def test_switch_over_failure_in_the_embedding2_clustering_leaves_no_half_set_state(monkeypatch):
    real = em.score_and_cluster_signals

    def score(rows, **kw):
        if "cutoffs" in kw:
            raise RuntimeError("synthetic maths failure")
        return real(rows, **kw)

    monkeypatch.setattr(em, "score_and_cluster_signals", score)
    run = _switch_over_run("pass")
    store = RecordingStore()
    out = em.cluster_for_run(_items(), run, store=store, now_utc=NOW)
    assert _signature(out) == _signature(methodology.score_and_cluster_signals(_items(), now_utc=NOW))
    assert store.calls == [] and run.fallback
    assert (run.percentiles, run.cutoffs, run.gate) == (None, None, None)


@pytest.mark.parametrize("make_run", [lambda: _switch_over_run("pass"), _migrated_run], ids=["switch_over", "migrated"])
def test_spend_block_in_the_maths_still_propagates(monkeypatch, make_run):
    def blocked(*args, **kwargs):
        raise SpendBlocked("synthetic block")

    monkeypatch.setattr(em, "cutoffs_at", blocked)
    run = make_run()
    store = RecordingStore()
    with pytest.raises(SpendBlocked):
        em.cluster_for_run(_items(), run, store=store, now_utc=NOW)
    assert store.calls == [] and not run.fallback


def test_variable_unset_never_reaches_the_maths(tmp_path, monkeypatch, capsys):
    for target in ("percentile_ranks", "cutoffs_at", "gate_report", "cluster_for_run"):
        monkeypatch.setattr(em, target, _boom)
    seen = _staged(tmp_path, monkeypatch, model=None)
    assert seen["embedding_run"] is None
    assert _embedding_lines(capsys.readouterr().out) == []
