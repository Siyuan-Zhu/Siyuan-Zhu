"""The reflector: the standing group, and what the reflector needs from a run. Synthetic data only."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from src.state_store import JsonFileStateStore
from src.zeitgeist import reflector, run_record
from src.zeitgeist.understanding import _render_cluster


def _item(url_hash, published_at, vector=(1.0, 0.0, 0.0), title=None):
    return {"url_hash": url_hash, "published_at": published_at, "embedding_f32": list(vector),
            "title": title or ("Synthetic story " + url_hash), "source_id": "m1_feed"}


def _days(*days):
    return ["2026-11-%02dT06:00:00Z" % day for day in days]


def _cluster(cluster_id, dates, *, score=1.0, vector=(1.0, 0.0, 0.0)):
    items = [_item("%s_h%d" % (cluster_id, index), when, vector) for index, when in enumerate(dates)]
    return SimpleNamespace(cluster_id=cluster_id, items=items, commission_score=score, velocity=0.5,
                           orbit_resonance=0.25, cluster_keywords=["synthetic", "keyword"])


def _ids(clusters):
    return [cluster.cluster_id for cluster in clusters]


# ---- the switch ----


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("true", False), ("1", True)])
def test_the_switch_is_on_only_for_one(value, expected):
    environ = {} if value is None else {"ZEITGEIST_REFLECTOR": value}
    assert reflector.reflector_enabled(environ) is expected


def test_the_switch_reads_the_process_environment(monkeypatch):
    monkeypatch.delenv("ZEITGEIST_REFLECTOR", raising=False)
    assert reflector.reflector_enabled() is False
    monkeypatch.setenv("ZEITGEIST_REFLECTOR", "1")
    assert reflector.reflector_enabled() is True


# ---- the standing groups ----


def test_a_group_on_five_distinct_days_stands_and_one_on_four_does_not():
    clusters = [_cluster("lead", _days(1, 2, 3, 4, 5, 6)), _cluster("five", _days(1, 3, 5, 7, 9)),
                _cluster("four", _days(1, 2, 3, 4, 4, 4, 4))]
    assert _ids(reflector.standing_groups(clusters)) == ["five"]


def test_the_leader_never_stands_however_many_days_it_spans():
    clusters = [_cluster("lead", _days(*range(1, 15))), _cluster("short", _days(1, 2))]
    assert reflector.standing_groups(clusters) == []


def test_several_items_on_one_date_count_once_and_dates_are_read_in_utc():
    # 23:30 at minus five hours is the next UTC date; the two Z times on the 3rd are one date.
    dates = ["2026-11-01T06:00:00Z", "2026-11-01T23:30:00-05:00", "2026-11-03T01:00:00Z",
             "2026-11-03T22:00:00Z", "2026-11-04T06:00:00+00:00", "2026-11-05T06:00:00"]
    clusters = [_cluster("lead", []), _cluster("g", dates)]
    assert _ids(reflector.standing_groups(clusters)) == ["g"]
    assert _ids(reflector.standing_groups(clusters, min_days=6)) == []


def test_items_without_a_readable_date_are_ignored():
    dates = _days(1, 2, 3, 4) + ["", None, "not a date", "2026-13-40T00:00:00Z", 12345]
    clusters = [_cluster("lead", []), _cluster("g", dates)]
    assert reflector.standing_groups(clusters) == []
    assert _ids(reflector.standing_groups(clusters, min_days=4)) == ["g"]


def test_most_days_first_then_the_higher_score_then_the_smaller_id():
    clusters = [
        _cluster("lead", _days(1)),
        _cluster("c_b", _days(1, 2, 3, 4, 5), score=2.0),
        _cluster("c_a", _days(1, 2, 3, 4, 5), score=2.0),
        _cluster("c_low", _days(1, 2, 3, 4, 5), score=1.0),
        _cluster("c_wide", _days(1, 2, 3, 4, 5, 6), score=0.1),
    ]
    assert _ids(reflector.standing_groups(clusters)) == ["c_wide", "c_a", "c_b", "c_low"]


def test_standing_groups_is_pure():
    clusters = [_cluster("lead", _days(1)), _cluster("g", _days(1, 2, 3, 4, 5))]
    before = json.dumps([[vars(c)["cluster_id"], c.items] for c in clusters], sort_keys=True)
    reflector.standing_groups(clusters)
    assert json.dumps([[vars(c)["cluster_id"], c.items] for c in clusters], sort_keys=True) == before
    assert reflector.standing_groups([]) == [] and reflector.standing_groups(None) == []


# ---- the centre ----


def test_the_centre_is_the_unit_mean_of_the_unit_vectors():
    cluster = SimpleNamespace(cluster_id="g", items=[_item("a", None, (2.0, 0.0, 0.0)), _item("b", None, (0.0, 5.0, 0.0))])
    centre = reflector.centre(cluster)
    assert centre == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2), 0.0])
    assert all(type(value) is float for value in centre)


def test_items_without_a_vector_are_left_out_of_the_centre_and_none_gives_none():
    no_vector = {"url_hash": "x", "published_at": None}
    cluster = SimpleNamespace(cluster_id="g", items=[no_vector, _item("a", None, (0.0, 0.0, 3.0))])
    assert reflector.centre(cluster) == pytest.approx([0.0, 0.0, 1.0])
    assert reflector.centre(SimpleNamespace(cluster_id="g", items=[no_vector])) is None
    assert reflector.centre(SimpleNamespace(cluster_id="g", items=[])) is None


# ---- what the reflector needs from a run ----


def _night():
    return [
        _cluster("cluster_0", _days(1, 2, 3, 4, 5, 6), score=3.0, vector=(1.0, 0.0, 0.0)),
        _cluster("cluster_1", _days(1, 2), score=2.0, vector=(0.0, 1.0, 0.0)),
        _cluster("cluster_2", _days(1, 2, 3, 4, 5, 6, 7), score=1.5, vector=(0.0, 0.0, 1.0)),
        _cluster("cluster_3", _days(1, 2, 3, 4, 5), score=1.0, vector=(0.0, 1.0, 1.0)),
    ]


def test_the_inputs_hold_the_standing_group_and_the_chosen_one():
    clusters = _night()
    inputs = reflector.reflector_inputs(clusters, "cluster_1", "gemini-embedding-2")
    assert set(inputs) == {"standing", "chosen", "embedding_model"}
    assert inputs["embedding_model"] == "gemini-embedding-2"
    standing = inputs["standing"]
    assert set(standing) == {"cluster_id", "rank", "days", "item_hashes", "block", "centre"}
    assert (standing["cluster_id"], standing["rank"], standing["days"]) == ("cluster_2", 3, 7)
    assert standing["item_hashes"] == ["cluster_2_h%d" % index for index in range(7)]
    assert standing["block"] == _render_cluster(clusters[2])
    assert standing["centre"] == pytest.approx([0.0, 0.0, 1.0])
    chosen = inputs["chosen"]
    assert set(chosen) == {"cluster_id", "block", "centre"}
    assert chosen["cluster_id"] == "cluster_1" and chosen["block"] == _render_cluster(clusters[1])
    assert chosen["centre"] == pytest.approx([0.0, 1.0, 0.0])


def test_the_standing_group_is_never_the_chosen_one():
    inputs = reflector.reflector_inputs(_night(), "cluster_2", "gemini-embedding-2")
    assert inputs["standing"]["cluster_id"] == "cluster_3" and inputs["chosen"]["cluster_id"] == "cluster_2"
    only = _night()[:3]
    assert reflector.reflector_inputs(only, "cluster_2", "gemini-embedding-2")["standing"] is None


def test_no_standing_group_gives_null():
    clusters = [_cluster("cluster_0", _days(1)), _cluster("cluster_1", _days(1, 2))]
    assert reflector.reflector_inputs(clusters, "cluster_0", "gemini-embedding-001")["standing"] is None


@pytest.mark.parametrize("chosen_id", ["cluster_9", None, ""])
def test_the_chosen_entry_is_null_for_an_unknown_id(chosen_id):
    assert reflector.reflector_inputs(_night(), chosen_id, "gemini-embedding-2")["chosen"] is None


def test_the_chosen_one_is_looked_for_among_the_first_five_only():
    clusters = _night() + [_cluster("cluster_%d" % index, _days(1)) for index in range(4, 7)]
    assert reflector.reflector_inputs(clusters, "cluster_4", "m")["chosen"]["cluster_id"] == "cluster_4"
    assert reflector.reflector_inputs(clusters, "cluster_5", "m")["chosen"] is None


def test_a_numeric_chosen_id_matches_its_string_form():
    clusters = [_cluster(0, _days(1)), _cluster(1, _days(1, 2, 3, 4, 5))]
    inputs = reflector.reflector_inputs(clusters, 0, "m")
    assert inputs["chosen"]["cluster_id"] == "0" and inputs["standing"]["cluster_id"] == "1"


# ---- stored on the record ----


def _written_record(store, run_id):
    record = {"kind": "daily_run", "date": "2026-11-21", "written_at": "2026-11-21T12:00:00+00:00",
              "word": "Synthword", "gloss": "A synthetic gloss.", "tests": {"own_group": {"status": "ok"}}}
    assert run_record.write_run_record(store, run_id, record)
    return store.load_run_record(run_id)


RUN_ID = "daily_run:2026-11-21:local:1"


def test_the_stored_inputs_round_trip_through_a_json_store(tmp_path, capsys):
    store = JsonFileStateStore(repo_root=tmp_path)
    before = _written_record(store, RUN_ID)
    clusters = _night()
    assert reflector.store_reflector_inputs(store, RUN_ID, clusters, "cluster_1", "gemini-embedding-2") is True
    after = JsonFileStateStore(repo_root=tmp_path).load_run_record(RUN_ID)
    expected = json.loads(json.dumps(reflector.reflector_inputs(clusters, "cluster_1", "gemini-embedding-2")))
    assert after["tests"]["reflector_inputs"] == expected
    assert after["tests"]["own_group"] == {"status": "ok"}
    assert {k: v for k, v in after.items() if k != "tests"} == {k: v for k, v in before.items() if k != "tests"}
    out = capsys.readouterr()
    assert (out.out, out.err) == ("", "")


def test_without_a_record_nothing_is_stored(tmp_path):
    store = JsonFileStateStore(repo_root=tmp_path)
    assert reflector.store_reflector_inputs(store, RUN_ID, _night(), "cluster_1", "m") is False
    assert store.load_run_record(RUN_ID) is None


class _Halt(BaseException):
    """A failure outside Exception."""


class _FailingStore(JsonFileStateStore):
    def __init__(self, root, error, on):
        super().__init__(repo_root=root)
        self.error, self.on = error, on

    def save_run_record(self, run_id, document):
        if self.on == "save" and getattr(self, "armed", False):
            raise self.error
        super().save_run_record(run_id, document)

    def load_run_record(self, run_id):
        if self.on == "load" and getattr(self, "armed", False):
            raise self.error
        return super().load_run_record(run_id)


@pytest.mark.parametrize("on", ["load", "save"])
@pytest.mark.parametrize("error", [RuntimeError("Zqsecret failure"), _Halt("Zqsecret halt")])
def test_a_failing_store_returns_false_and_changes_nothing(tmp_path, capsys, on, error):
    store = _FailingStore(tmp_path, error, on)
    before = _written_record(store, RUN_ID)
    store.armed = True
    assert reflector.store_reflector_inputs(store, RUN_ID, _night(), "cluster_1", "m") is False
    assert JsonFileStateStore(repo_root=tmp_path).load_run_record(RUN_ID) == before
    out = capsys.readouterr()
    assert (out.out, out.err) == ("", "")


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_or_an_exit_is_raised_again(tmp_path, error):
    store = _FailingStore(tmp_path, error(), "load")
    _written_record(store, RUN_ID)
    store.armed = True
    with pytest.raises(error):
        reflector.store_reflector_inputs(store, RUN_ID, _night(), "cluster_1", "m")


def test_clusters_that_cannot_be_read_store_nothing(tmp_path):
    store = JsonFileStateStore(repo_root=tmp_path)
    before = _written_record(store, RUN_ID)

    class Unreadable:
        @property
        def items(self):
            raise RuntimeError("Zqsecret unreadable")

    assert reflector.store_reflector_inputs(store, RUN_ID, [Unreadable(), Unreadable()], "x", "m") is False
    assert store.load_run_record(RUN_ID) == before


# ---- has it been named? the arithmetic answer ----


def test_the_any_day_glosses_are_three_fixed_strings():
    assert reflector.ANY_DAY_GLOSSES == ("A quiet tension beneath ordinary days.", "Change arriving slower than expected.",
                                         "What we carry and what we set down.")


def test_a_past_gloss_as_close_as_tonights_own_names_the_group():
    # Tonight's own gloss sits at cosine 0.8 to its candidate; a past gloss sits at 0.8 to the standing group.
    numbers = reflector.named_by_arithmetic([1, 0, 0], [[0, 1, 0], [0.8, 0.6, 0]], [0.8, 0, 0.6], [1, 0, 0])
    assert numbers == {"named": True, "margin": pytest.approx(0.0), "best_index": 1}
    assert type(numbers["margin"]) is float and type(numbers["named"]) is bool


def test_the_rule_allows_two_hundredths_below_the_reference():
    reference = [0.8, 0, 0.6]
    just_inside = [0.785, math.sqrt(1 - 0.785 ** 2), 0]
    just_outside = [0.775, math.sqrt(1 - 0.775 ** 2), 0]
    inside = reflector.named_by_arithmetic([1, 0, 0], [just_inside], reference, [1, 0, 0])
    outside = reflector.named_by_arithmetic([1, 0, 0], [just_outside], reference, [1, 0, 0])
    assert inside["named"] is True and inside["margin"] == pytest.approx(-0.015)
    assert outside["named"] is False and outside["margin"] == pytest.approx(-0.025)


def test_vectors_are_normalised_before_the_cosine():
    numbers = reflector.named_by_arithmetic([5, 0, 0], [[0, 0, 2], [3, 3, 0]], [0, 4, 0], [0, 9, 0])
    assert numbers["best_index"] == 1
    assert numbers["margin"] == pytest.approx(1 / math.sqrt(2) - 1.0)
    assert numbers["named"] is False


@pytest.mark.parametrize("args", [
    ([1, 0], [], [1, 0], [1, 0]),
    ([0, 0], [[1, 0]], [1, 0], [1, 0]),
    ([1, 0], [[1, 0]], [1, 0], [1, 0, 0]),
    ([1, 0], [[float("nan"), 0]], [1, 0], [1, 0]),
])
def test_an_arithmetic_answer_needs_usable_vectors(args):
    with pytest.raises(ValueError):
        reflector.named_by_arithmetic(*args)


# ---- the pointing task ----


def test_the_prompt_head_is_fixed():
    assert reflector.PROMPT_HEAD == (
        "Below is one group of items and a numbered list of short glosses. Reply with JSON of the form "
        '{"answer": N}, where N is the number of the gloss that was written about this group, or "none" if no '
        "gloss was.")


def test_the_question_is_the_head_the_block_then_the_glosses_numbered_in_the_given_order():
    question = reflector.build_question("- Cluster g: synthetic block", ["First.", "Second.", "Third."], [2, 0, 1])
    assert question == (reflector.PROMPT_HEAD + "\n\n- Cluster g: synthetic block\n\n"
                        "1. Third.\n2. First.\n3. Second.")


def test_a_gloss_with_line_breaks_stays_on_its_numbered_line():
    question = reflector.build_question("block", ["One\nline.", "Two."], [0, 1])
    assert question.endswith("\n\n1. One line.\n2. Two.")


@pytest.mark.parametrize("order", [[0, 1], [0, 1, 1], [0, 1, 3], [1, 2, 3]])
def test_an_order_that_is_not_a_permutation_is_refused(order):
    with pytest.raises(ValueError):
        reflector.build_question("block", ["a", "b", "c"], order)


@pytest.mark.parametrize("text, expected", [
    ('{"answer": 2}', 2),
    ('  {"answer": 1}\n', 1),
    ('{"answer": 3, "why": "synthetic"}', 3),
    ('{"answer": "none"}', "none"),
    ('{"answer": "None"}', "none"),
    ('{"answer": 0}', None),
    ('{"answer": 4}', None),
    ('{"answer": -1}', None),
    ('{"answer": true}', None),
    ('{"answer": 2.0}', None),
    ('{"answer": "2"}', None),
    ('{"answer": null}', None),
    ('{"answer": [1]}', None),
    ('{"other": 1}', None),
    ('[1]', None),
    ('Answer: {"answer": 2}', None),
    ('{"answer": 2} because it fits', None),
    ('```json\n{"answer": 2}\n```', None),
    ("", None),
    (None, None),
    ("not json", None),
])
def test_an_answer_is_a_number_in_range_none_or_nothing(text, expected):
    assert reflector.parse_answer(text, 3) == expected


def test_parse_answer_never_raises_on_odd_input():
    for text in (b'{"answer": 1}', 12, object(), '{"answer": 1e999}', "{" * 5000):
        assert reflector.parse_answer(text, 3) in (None, 1)


def test_an_answer_maps_back_to_the_gloss_it_pointed_at():
    order = [2, 0, 1]
    assert [reflector.answer_identity(answer, order) for answer in (1, 2, 3, "none", None)] == [2, 0, 1, "none", None]


# ---- the verdict ----


def _verdict(control=(0, 0, 0), expected=(0, 0, 0), real=("none",) * 3, plants=(5, 6, 7)):
    return reflector.verdict(list(control), list(expected), list(real), list(plants))


def test_clean_plants_and_three_nones_give_a_pointer():
    assert _verdict() == {"plants_ok": True, "agreement": True, "status": "pointer"}


def test_clean_plants_and_three_answers_on_one_gloss_give_named():
    assert _verdict(real=(3, 3, 3)) == {"plants_ok": True, "agreement": True, "status": "named"}


def test_a_wrong_control_gives_void():
    assert _verdict(control=(0, 2, 0)) == {"plants_ok": False, "agreement": True, "status": "void"}


def test_an_unreadable_control_counts_as_wrong():
    assert _verdict(control=(0, None, 0))["status"] == "void"


def test_a_chosen_any_day_plant_gives_void():
    # The plant of the second run is chosen in the second run.
    assert _verdict(real=("none", 6, "none")) == {"plants_ok": False, "agreement": False, "status": "void"}
    assert _verdict(real=(6, 6, 6)) == {"plants_ok": False, "agreement": True, "status": "void"}


def test_three_real_answers_that_differ_give_void():
    assert _verdict(real=(3, 3, "none")) == {"plants_ok": True, "agreement": False, "status": "void"}
    assert _verdict(real=(1, 2, 3))["status"] == "void"


def test_an_unreadable_real_answer_voids_the_run():
    assert _verdict(real=("none", None, "none")) == {"plants_ok": True, "agreement": False, "status": "void"}
    assert _verdict(real=(None, None, None)) == {"plants_ok": True, "agreement": False, "status": "void"}


def test_missing_runs_give_void():
    assert reflector.verdict([0, 0], [0, 0, 0], ["none"] * 3, [5, 6, 7])["status"] == "void"
    assert reflector.verdict([0] * 3, [0] * 3, ["none"] * 2, [5, 6, 7])["status"] == "void"
    assert reflector.verdict([], [], [], [])["status"] == "void"
