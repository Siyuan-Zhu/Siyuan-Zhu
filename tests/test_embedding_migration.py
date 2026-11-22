"""Embedding-2 calibration: similarity percentiles, the gate, the private document and the run plan."""
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest

from src.state_store import InMemoryStateStore
from src.zeitgeist import embedding_migration as em

E1 = "gemini-embedding-001"
E2 = "gemini-embedding-2"
DOC = "embedding_calibration"


def cl(hashes, sources=1):
    return SimpleNamespace(items=[{"url_hash": h} for h in hashes], unique_sources_count=sources)


def e2_clusters(sizes, multi):
    clusters, start = [], 0
    for index, size in enumerate(sizes):
        hashes = ["e%d" % n for n in range(start, start + size)]
        clusters.append(cl(hashes, 2 if index < multi else 1))
        start += size
    return clusters


def top5(survivors):
    return [cl(["e0"]) for _ in range(survivors)] + [
        cl(["absent%d" % n]) for n in range(5 - survivors)
    ]


def calibration(status="accepted"):
    return em.Calibration(
        status=status,
        date="2026-11-22",
        item_count=140,
        percentiles=(0.97, 0.93, 0.9, 0.995),
        multi_source=3,
        largest_share=0.21,
        top5_survivors=4,
    )


def test_constants_and_codes():
    assert em.EMBED_001 == E1 and em.EMBED_2 == E2
    assert em.LEGACY_CUTOFFS == (0.78, 0.72, 0.68, 0.85)
    assert (em.MIN_CALIBRATION_ITEMS, em.GATE_MIN_MULTI_SOURCE, em.GATE_MAX_LARGEST_SHARE,
            em.GATE_MIN_SURVIVORS, em.CALIBRATION_SCHEMA_VERSION) == (100, 2, 0.35, 3, 1)
    assert (em.CAL_NONE, em.CAL_ACCEPTED, em.CAL_REJECTED, em.CAL_INVALID,
            em.CAL_UNREADABLE, em.CAL_UNSUPPORTED) == (0, 1, 2, 3, 4, 5)


def test_pairwise_similarities_upper_triangle_in_order():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.6, 0.8, 0.0])
    c = np.array([0.0, 0.6, 0.8])
    sims = em.pairwise_similarities([{"embedding": a}, {"embedding": b * 5}, {"embedding": c}])
    assert sims.dtype == np.float64
    np.testing.assert_allclose(sims, [a @ b, a @ c, b @ c], atol=1e-6)
    assert em.pairwise_similarities([{"embedding": a}]).size == 0
    assert em.pairwise_similarities([]).size == 0


def test_percentile_ranks_fraction_strictly_below_each_cutoff():
    assert em.percentile_ranks(np.array([0.1, 0.5, 0.7, 0.8, 0.9])) == (0.6, 0.6, 0.4, 0.8)


def test_cutoffs_at_inverts_percentile_ranks():
    sims = np.linspace(0, 1, 100001)
    cut = em.cutoffs_at(sims, em.percentile_ranks(sims))
    assert len(cut) == 4
    for got, want in zip(cut, em.LEGACY_CUTOFFS):
        assert abs(got - want) < 1e-4


def test_cutoff_for_percentile_one_is_reached_by_no_pair():
    cut = em.cutoffs_at(np.array([0.2, 0.3]), [1.0])
    assert cut[0] > 1.0


def test_cluster_shape():
    assert em.cluster_shape([]) == (0, 0.0)
    assert em.cluster_shape(e2_clusters([3, 1], multi=1)) == (1, 0.75)


def test_gate_passes_at_the_boundaries():
    report = em.gate_report(top5(3), e2_clusters([17, 11, 11, 11], multi=2))
    assert report == em.GateReport(multi_source=2, largest_share=0.34, top5_survivors=3, passed=True)


def test_gate_fails_with_one_multi_source_cluster():
    report = em.gate_report(top5(3), e2_clusters([17, 11, 11, 11], multi=1))
    assert report.multi_source == 1 and not report.passed


def test_gate_fails_at_a_share_of_exactly_035():
    report = em.gate_report(top5(3), e2_clusters([7, 7, 6], multi=2))
    assert report.largest_share == 0.35 and not report.passed


def test_gate_fails_with_two_survivors_and_reads_only_the_top_five():
    report = em.gate_report(top5(2) + [cl(["e0"])], e2_clusters([17, 11, 11, 11], multi=2))
    assert report.top5_survivors == 2 and not report.passed


def test_survival_needs_more_than_half_in_one_cluster():
    e2 = e2_clusters([17, 11, 11, 11], multi=2)
    half = cl(["e0", "e17"])
    majority = cl(["e0", "e1", "e17"])
    assert em.gate_report([half], e2).top5_survivors == 0
    assert em.gate_report([majority], e2).top5_survivors == 1


def test_one_item_cluster_survives_only_when_its_hash_is_present():
    e2 = e2_clusters([17, 11, 11, 11], multi=2)
    assert em.gate_report([cl(["e30"])], e2).top5_survivors == 1
    assert em.gate_report([cl(["absent"])], e2).top5_survivors == 0


def test_fewer_than_three_001_clusters_always_fail():
    e2 = e2_clusters([17, 11, 11, 11], multi=2)
    assert not em.gate_report([cl(["e0"]), cl(["e17"])], e2).passed
    assert not em.gate_report([], e2).passed


@pytest.mark.parametrize("status,code", [("accepted", 1), ("rejected", 2)])
def test_document_round_trips(status, code):
    cal = calibration(status)
    assert em.parse_calibration(em.calibration_document(cal)) == (code, cal)


def test_document_has_exactly_the_public_numeric_keys():
    doc = em.calibration_document(calibration())
    assert doc == {
        "schema_version": 1,
        "status": "accepted",
        "date": "2026-11-22",
        "item_count": 140,
        "from_model": E1,
        "to_model": E2,
        "percentiles": [0.97, 0.93, 0.9, 0.995],
        "multi_source_clusters": 3,
        "largest_share": 0.21,
        "top5_survivors": 4,
    }
    allowed = {"accepted", "rejected", "2026-11-22", E1, E2}
    for value in doc.values():
        if isinstance(value, str):
            assert value in allowed
        elif isinstance(value, list):
            assert all(type(v) is float for v in value)
        else:
            assert type(value) in (int, float)


def _mutated(**changes):
    doc = em.calibration_document(calibration())
    for key, value in changes.items():
        if value is KeyError:
            del doc[key]
        else:
            doc[key] = value
    return doc


@pytest.mark.parametrize(
    "document",
    [
        _mutated(title="x"),
        _mutated(top5_survivors=KeyError),
        _mutated(schema_version=2),
        _mutated(status="pending"),
        _mutated(percentiles=[0.97, 0.93, 0.9, 1.2]),
        _mutated(percentiles=[0.97, 0.93, float("nan"), 0.995]),
        _mutated(percentiles=[0.93, 0.97, 0.9, 0.995]),
        _mutated(item_count=99),
        _mutated(item_count=True),
        _mutated(date="22-11-2026"),
        _mutated(from_model=E2),
        _mutated(largest_share=1.5),
        _mutated(multi_source_clusters=-1),
        _mutated(percentiles=[0.97, 0.93, 0.9]),
        _mutated(percentiles="0.9"),
        ["not", "a", "dict"],
        "accepted",
        42,
    ],
    ids=[
        "extra_title",
        "missing_key",
        "schema_2",
        "status_pending",
        "percentile_1_2",
        "percentile_nan",
        "percentiles_out_of_order",
        "item_count_99",
        "item_count_true",
        "bad_date",
        "wrong_from_model",
        "share_above_one",
        "negative_count",
        "three_percentiles",
        "percentiles_not_list",
        "list_document",
        "string_document",
        "int_document",
    ],
)
def test_invalid_documents(document):
    assert em.parse_calibration(document) == (em.CAL_INVALID, None)


def test_absent_document_is_none_state():
    assert em.parse_calibration(None) == (em.CAL_NONE, None)


class _RecordingStore(InMemoryStateStore):
    def __init__(self):
        super().__init__()
        self.writes = []

    def save_calibration(self, name, document):
        self.writes.append(name)
        super().save_calibration(name, document)

    def clear_calibration(self, name):
        self.writes.append(name)
        return super().clear_calibration(name)


def _store_with(document=None):
    store = _RecordingStore()
    if document is not None:
        InMemoryStateStore.save_calibration(store, DOC, document)
    return store


def test_plan_returns_none_unless_embedding_2_is_requested():
    assert em.plan_embedding_run(_store_with(), E1) is None
    assert em.plan_embedding_run(_store_with(), "") is None


def test_plan_switch_over_on_an_empty_store():
    store = _store_with()
    run = em.plan_embedding_run(store, E2)
    assert (run.mode, run.calibration_state, run.percentiles) == ("switch_over", em.CAL_NONE, None)
    assert run.used == E1 and store.writes == []


def test_plan_migrated_with_stored_percentiles():
    store = _store_with(em.calibration_document(calibration("accepted")))
    run = em.plan_embedding_run(store, E2)
    assert (run.mode, run.calibration_state) == ("migrated", em.CAL_ACCEPTED)
    assert run.percentiles == (0.97, 0.93, 0.9, 0.995)
    assert store.writes == []


def test_plan_legacy_on_a_rejected_document():
    store = _store_with(em.calibration_document(calibration("rejected")))
    run = em.plan_embedding_run(store, E2)
    assert (run.mode, run.calibration_state, run.percentiles) == ("legacy", em.CAL_REJECTED, None)
    assert store.writes == []


def test_plan_legacy_on_an_invalid_document_left_in_place():
    bad = _mutated(schema_version=2)
    store = _store_with(bad)
    run = em.plan_embedding_run(store, E2)
    assert (run.mode, run.calibration_state) == ("legacy", em.CAL_INVALID)
    assert store.writes == [] and store.load_calibration(DOC) == bad


def test_plan_legacy_invalid_when_load_returns_a_non_dict():
    store = _store_with()
    store.load_calibration = lambda name: ["corrupt"]
    run = em.plan_embedding_run(store, E2)
    assert (run.mode, run.calibration_state) == ("legacy", em.CAL_INVALID)


def test_plan_legacy_unreadable_when_load_raises():
    store = _store_with()

    def boom(name):
        raise ValueError("corrupt json")

    store.load_calibration = boom
    run = em.plan_embedding_run(store, E2)
    assert (run.mode, run.calibration_state) == ("legacy", em.CAL_UNREADABLE)
    assert store.writes == []


def test_plan_legacy_unsupported_without_calibration_methods():
    run = em.plan_embedding_run(object(), E2)
    assert (run.mode, run.calibration_state) == ("legacy", em.CAL_UNSUPPORTED)


def test_plan_reads_the_embedding_calibration_document():
    names = []
    store = SimpleNamespace(load_calibration=lambda name: names.append(name))
    em.plan_embedding_run(store, E2)
    assert names == [DOC]


def test_embedding_line_for_a_default_legacy_run():
    run = em.EmbeddingRun(mode="legacy", calibration_state=2)
    assert em.embedding_line(run) == (
        "[embedding] used=1 switch_over=0 fallback=0 calibration=2 items=0 p=-1 cut=-1 "
        "multi_source=-1 largest_share=-1 gate_multi_source=-1 gate_largest_share=-1 "
        "top5_survivors=-1 pass=-1 write_failed=0"
    )


def test_embedding_line_for_a_populated_switch_over_run():
    run = em.EmbeddingRun(
        mode="switch_over",
        calibration_state=0,
        percentiles=(0.97, 0.93, 0.9, 1.0),
        used=E2,
        items=140,
        cutoffs=(0.81, 0.77, 0.74, 1.000001),
        multi_source=3,
        largest_share=0.2,
        gate=em.GateReport(multi_source=3, largest_share=0.25, top5_survivors=4, passed=True),
        write_failed=True,
    )
    line = em.embedding_line(run)
    assert re.fullmatch(r"\[embedding\]( [a-z0-9_]+=-?[0-9.,]+)+", line)
    assert line == (
        "[embedding] used=2 switch_over=1 fallback=0 calibration=0 items=140 "
        "p=0.970000,0.930000,0.900000,1.000000 cut=0.8100,0.7700,0.7400,1.0000 "
        "multi_source=3 largest_share=0.200 gate_multi_source=3 gate_largest_share=0.250 "
        "top5_survivors=4 pass=1 write_failed=1"
    )
