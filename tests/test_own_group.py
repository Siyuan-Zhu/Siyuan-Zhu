"""The own-group check: how close a word and gloss sit to their own candidate. Synthetic vectors only."""

from __future__ import annotations

import math

import pytest

from src.zeitgeist import own_group


def _item(*vector):
    return {"embedding_f32": list(vector)}


def _unit(*vector):
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector]


# ---- the switch ----


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("true", False), ("1", True)])
def test_the_switch_is_on_only_for_one(value, expected):
    environ = {} if value is None else {"ZEITGEIST_OWN_GROUP": value}
    assert own_group.own_group_enabled(environ) is expected


def test_the_switch_reads_the_process_environment(monkeypatch):
    monkeypatch.delenv("ZEITGEIST_OWN_GROUP", raising=False)
    assert own_group.own_group_enabled() is False
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    assert own_group.own_group_enabled() is True


# ---- the arithmetic ----


def test_the_three_figures_of_a_hand_computed_case():
    own = [_item(1, 0, 0), _item(0.6, 0.8, 0)]
    others = [[_item(0, 1, 0)], [_item(0.8, 0.6, 0), _item(0, 0, 1)]]
    numbers = own_group.own_group_numbers([1, 0, 0], own, others)
    assert numbers["status"] == "ok"
    assert numbers["own"] == pytest.approx(0.8)
    assert numbers["others"] == pytest.approx([0.0, 0.4])
    assert numbers["margin"] == pytest.approx(0.4)
    assert numbers["n_own"] == 2
    assert set(numbers) == {"status", "own", "others", "margin", "n_own"}
    assert all(type(value) is float for value in [numbers["own"], numbers["margin"], *numbers["others"]])


def test_the_text_vector_is_normalised_before_the_cosine():
    numbers = own_group.own_group_numbers([3, 0, 0], [_item(1, 0, 0)], [[_item(0, 1, 0)]])
    assert numbers["own"] == pytest.approx(1.0) and numbers["margin"] == pytest.approx(1.0)


def test_a_text_at_its_own_centre_scores_a_positive_margin():
    own = [_item(*_unit(1, 0.1, 0)), _item(*_unit(1, -0.1, 0))]
    others = [[_item(0, 1, 0), _item(0, 1, 0.1)], [_item(0, 0, 1)]]
    numbers = own_group.own_group_numbers([1, 0, 0], own, others)
    assert numbers["margin"] > 0.5


def test_a_text_that_fits_any_candidate_scores_a_margin_near_zero():
    own = [_item(1, 0, 0)]
    others = [[_item(0, 1, 0)], [_item(0, 0, 1)]]
    numbers = own_group.own_group_numbers(_unit(1, 1, 1), own, others)
    assert numbers["own"] == pytest.approx(1 / math.sqrt(3))
    assert numbers["others"] == pytest.approx([1 / math.sqrt(3)] * 2)
    assert numbers["margin"] == pytest.approx(0.0, abs=1e-6)


def test_one_candidate_only_has_no_others_and_a_null_margin():
    numbers = own_group.own_group_numbers([1, 0, 0], [_item(1, 0, 0)], [])
    assert numbers == {"status": "ok", "own": pytest.approx(1.0), "others": [], "margin": None, "n_own": 1}


def test_empty_own_items_are_skipped():
    assert own_group.own_group_numbers([1, 0, 0], [], [[_item(1, 0, 0)]]) == {"status": "skipped", "reason": "no_items"}


def test_own_items_without_vectors_are_skipped():
    numbers = own_group.own_group_numbers([1, 0, 0], [{"title": "Zqtitle"}, {"embedding_f32": [0, 0, 0]}], [])
    assert numbers == {"status": "skipped", "reason": "no_vector"}


@pytest.mark.parametrize("text_vector", [[0, 0, 0], [float("nan"), 0, 0], []])
def test_a_text_vector_that_cannot_be_used_is_skipped(text_vector):
    numbers = own_group.own_group_numbers(text_vector, [_item(1, 0, 0)], [])
    assert numbers == {"status": "skipped", "reason": "no_vector"}


def test_a_dimension_mismatch_with_the_own_items_is_skipped():
    numbers = own_group.own_group_numbers([1, 0, 0, 0], [_item(1, 0, 0)], [[_item(0, 1, 0)]])
    assert numbers == {"status": "skipped", "reason": "dimension"}


def test_a_dimension_mismatch_with_another_candidate_is_skipped():
    numbers = own_group.own_group_numbers([1, 0, 0], [_item(1, 0, 0)], [[_item(0, 1, 0, 0)]])
    assert numbers == {"status": "skipped", "reason": "dimension"}


def test_items_without_vectors_are_left_out_of_each_mean():
    own = [_item(1, 0, 0), {"title": "no vector"}]
    others = [[_item(0, 1, 0), {"title": "no vector"}], [{"title": "no vector"}]]
    numbers = own_group.own_group_numbers([1, 0, 0], own, others)
    assert numbers["own"] == pytest.approx(1.0) and numbers["n_own"] == 1
    # A candidate with no vector at all has no figure.
    assert numbers["others"] == pytest.approx([0.0])
    assert numbers["margin"] == pytest.approx(1.0)


def test_the_items_may_be_objects_with_an_f16_vector():
    from types import SimpleNamespace

    from src.zeitgeist.collectors import pack_f16_base64

    own = [SimpleNamespace(embedding_f16=pack_f16_base64([1.0, 0.0, 0.0]))]
    numbers = own_group.own_group_numbers([1, 0, 0], own, [])
    assert numbers["own"] == pytest.approx(1.0, abs=1e-3)


# ---- the guarded step ----


from types import SimpleNamespace  # noqa: E402

from src.zeitgeist import model_choice as mc  # noqa: E402
from src.zeitgeist.collectors import EMBEDDING_2_TASK_PREFIX  # noqa: E402
from src.zeitgeist.spend_guard import SpendBlocked  # noqa: E402

RUN_ID = "daily_run:2026-11-21:local:1"
WORD, GLOSS = "Selvedge", "A finished edge that holds."
TEXT = WORD + ". " + GLOSS


def _axis(index, dim=768):
    vector = [0.0] * dim
    vector[index] = 1.0
    return vector


def _cluster(index, *axes, dim=768):
    return SimpleNamespace(cluster_id="cluster_%d" % index, items=[{"embedding_f32": _axis(a, dim)} for a in axes])


def _night(count=5, dim=768):
    # Candidate i holds two items on axis i and one on axis 10 + i.
    return [_cluster(i, i, i, 10 + i, dim=dim) for i in range(count)]


class MemoryStore:
    """One stored run record, every store call noted."""

    def __init__(self, record=None, fail_on=None, error=RuntimeError):
        self.records = {} if record is False else {RUN_ID: record or {"run_id": RUN_ID, "tests": {}}}
        self.calls = []
        self.fail_on = fail_on
        self.error = error

    def load_run_record(self, run_id):
        self.calls.append("load_run_record")
        if self.fail_on == "load":
            raise self.error("synthetic load failure")
        record = self.records.get(run_id)
        return None if record is None else {**record, "tests": dict(record["tests"])}

    def save_run_record(self, run_id, document):
        self.calls.append("save_run_record")
        if self.fail_on == "save":
            raise self.error("synthetic save failure")
        self.records[run_id] = document


class EmbedClient:
    """A fake embedding endpoint: records each request and answers one vector per text."""

    def __init__(self, vector=None, error=None, count=None):
        self.requests = []
        self.vector = _axis(0) if vector is None else vector
        self.error = error
        self.count = count
        self.models = self

    def embed_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if self.error is not None:
            raise self.error
        count = len(contents) if self.count is None else self.count
        return SimpleNamespace(embeddings=[SimpleNamespace(values=list(self.vector)) for _ in range(count)])


def _check(store=None, client=None, **over):
    kwargs = dict(word=WORD, gloss=GLOSS, chosen_cluster_id="cluster_0", clusters=_night(),
                  embedding_model=mc.EMBED_001, genai_client=EmbedClient() if client is None else client)
    kwargs.update(over)
    store = MemoryStore() if store is None else store
    return own_group.run_own_group_check(store, RUN_ID, **kwargs), store, kwargs["genai_client"]


def _stored(store):
    return store.records[RUN_ID]["tests"].get("own_group")


def test_the_stored_numbers_are_the_arithmetic_of_one_embedding_request(capsys, caplog):
    vector = [0.0] * 768
    vector[0], vector[1] = 0.8, 0.6
    ok, store, client = _check(client=EmbedClient(vector=vector))
    assert ok is True
    [request] = client.requests
    assert request == {"model": mc.EMBED_001, "contents": [TEXT], "config": {"output_dimensionality": 768}}
    clusters = _night()
    expected = own_group.own_group_numbers(vector, clusters[0].items, [c.items for c in clusters[1:]])
    stored = _stored(store)
    assert stored["status"] == "ok" and stored["n_own"] == 3
    assert stored["own"] == pytest.approx(expected["own"], abs=1e-3)
    assert stored["others"] == pytest.approx(expected["others"], abs=1e-3) and len(stored["others"]) == 4
    assert stored["margin"] == pytest.approx(expected["margin"], abs=1e-3)
    # Nothing else in the record moves.
    assert set(store.records[RUN_ID]) == {"run_id", "tests"} and set(store.records[RUN_ID]["tests"]) == {"own_group"}
    out = capsys.readouterr()
    assert out.out == out.err == "" and caplog.records == []


def test_the_embedding_2_request_is_the_one_the_collection_step_makes():
    ok, store, client = _check(embedding_model=mc.EMBED_2)
    assert ok is True and _stored(store)["status"] == "ok"
    [request] = client.requests
    assert request["model"] == mc.EMBED_2 and request["config"] == {"output_dimensionality": 768}
    [content] = request["contents"]
    assert [part.text for part in content.parts] == [EMBEDDING_2_TASK_PREFIX + TEXT]


def test_the_check_reads_only_the_first_five_candidates():
    clusters = _night(6)
    ok, store, client = _check(chosen_cluster_id="cluster_5", clusters=clusters)
    assert ok is False and client.requests == []
    assert _stored(store) == {"status": "skipped", "reason": "unknown_candidate"}
    ok, store, _ = _check(chosen_cluster_id="cluster_4", clusters=clusters)
    assert ok is True and len(_stored(store)["others"]) == 4


@pytest.mark.parametrize("chosen", ["cluster_9", None, ""])
def test_an_unknown_chosen_candidate_is_skipped_without_a_call(chosen):
    ok, store, client = _check(chosen_cluster_id=chosen)
    assert ok is False and client.requests == []
    assert _stored(store) == {"status": "skipped", "reason": "unknown_candidate"}


def test_a_chosen_candidate_without_vectors_is_skipped_without_a_call():
    clusters = _night()
    clusters[0] = SimpleNamespace(cluster_id="cluster_0", items=[{"title": "no vector"}])
    ok, store, client = _check(clusters=clusters)
    assert ok is False and client.requests == []
    assert _stored(store) == {"status": "skipped", "reason": "no_vector"}
    clusters[0] = SimpleNamespace(cluster_id="cluster_0", items=[])
    ok, store, client = _check(clusters=clusters)
    assert ok is False and client.requests == []
    assert _stored(store) == {"status": "skipped", "reason": "no_items"}


def test_a_night_with_one_candidate_stores_empty_others_and_a_null_margin():
    ok, store, _ = _check(clusters=_night(1))
    assert ok is True
    stored = _stored(store)
    assert stored["status"] == "ok" and stored["others"] == [] and stored["margin"] is None


def test_item_vectors_of_another_length_are_skipped():
    ok, store, client = _check(clusters=_night(dim=32))
    assert ok is False and len(client.requests) == 1
    assert _stored(store) == {"status": "skipped", "reason": "dimension"}


@pytest.mark.parametrize("model", ["mixed", "unknown", "", None])
def test_vectors_of_no_single_model_are_skipped_without_a_call(model):
    ok, store, client = _check(embedding_model=model)
    assert ok is False and client.requests == []
    assert _stored(store) == {"status": "skipped", "reason": "embedding_model"}


def test_no_client_is_skipped():
    ok, store, _ = _check(genai_client=None)
    assert ok is False
    assert _stored(store) == {"status": "skipped", "reason": "no_client"}


def _refusal(pre):
    error = SpendBlocked("Unregistered paid SDK method")
    if pre:
        error.refused_before_dispatch = True
    return error


@pytest.mark.parametrize("pre", [True, False])
def test_a_refused_call_is_stored_as_skipped(pre, capsys):
    ok, store, client = _check(client=EmbedClient(error=_refusal(pre)))
    assert ok is False and len(client.requests) == 1
    assert _stored(store) == {"status": "skipped", "reason": "refused"}
    out = capsys.readouterr()
    assert out.out == out.err == ""


class _Halt(BaseException):
    """A failure outside Exception, as a library or the runtime might raise."""


@pytest.mark.parametrize("error", [RuntimeError("Zqsecret failure"), _Halt("Zqsecret halt")])
def test_a_raising_client_is_stored_as_skipped(error, capsys, caplog):
    ok, store, _ = _check(client=EmbedClient(error=error))
    assert ok is False
    assert _stored(store) == {"status": "skipped", "reason": "failed"}
    out = capsys.readouterr()
    assert out.out == out.err == "" and caplog.records == []


@pytest.mark.parametrize("count", [0, 2])
def test_a_reply_without_exactly_one_vector_is_stored_as_skipped(count):
    ok, store, _ = _check(client=EmbedClient(count=count), embedding_model=mc.EMBED_2)
    assert ok is False
    assert _stored(store) == {"status": "skipped", "reason": "failed"}


def test_a_vector_of_another_length_from_the_provider_is_stored_as_skipped():
    ok, store, _ = _check(client=EmbedClient(vector=_axis(0, 12)))
    assert ok is False
    assert _stored(store) == {"status": "skipped", "reason": "failed"}


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_an_interrupt_or_exit_is_raised_again(error):
    with pytest.raises(error):
        _check(client=EmbedClient(error=error()))


def test_without_a_record_nothing_is_embedded_or_stored():
    ok, store, client = _check(store=MemoryStore(record=False))
    assert ok is False and client.requests == []
    assert store.calls == ["load_run_record"]


@pytest.mark.parametrize("fail_on, error", [("load", RuntimeError), ("save", RuntimeError),
                                            ("load", _Halt), ("save", _Halt)])
def test_a_failing_store_returns_false_and_raises_nothing(fail_on, error):
    ok, _, _ = _check(store=MemoryStore(fail_on=fail_on, error=error))
    assert ok is False


def test_a_missing_word_or_gloss_is_skipped_without_a_call():
    for over in ({"word": None}, {"gloss": ""}):
        ok, store, client = _check(**over)
        assert ok is False and client.requests == []
        assert _stored(store) == {"status": "skipped", "reason": "no_text"}


# ---- the time budget ----


import signal  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402


def test_a_function_that_finishes_in_time_returns_true_on_a_daemon_thread():
    seen = []
    assert own_group.run_with_budget(lambda: seen.append(threading.current_thread()), seconds=5.0) is True
    [thread] = seen
    assert thread is not threading.main_thread() and thread.daemon is True


def test_the_default_budget_is_forty_five_seconds():
    assert own_group.BUDGET_SECONDS == 45.0
    assert own_group.run_with_budget.__defaults__ == (45.0,)


def test_a_function_past_the_budget_returns_false_quickly(capsys):
    release = threading.Event()
    started = time.monotonic()
    try:
        assert own_group.run_with_budget(lambda: release.wait(10), seconds=0.1) is False
        assert time.monotonic() - started < 2
    finally:
        release.set()
    out = capsys.readouterr()
    assert out.out == out.err == ""


@pytest.mark.parametrize("error", [RuntimeError("Zqsecret failure"), _Halt("Zqsecret halt"), KeyboardInterrupt(),
                                   SystemExit(1)])
def test_an_exception_inside_the_function_is_swallowed_in_its_thread(error, capsys):
    def fn():
        raise error

    assert own_group.run_with_budget(fn, seconds=5.0) is True
    out = capsys.readouterr()
    assert out.out == out.err == ""


def test_an_interrupt_in_the_main_thread_while_waiting_propagates():
    import _thread

    release = threading.Event()

    def fn():
        _thread.interrupt_main()
        release.wait(10)

    try:
        with pytest.raises(KeyboardInterrupt):
            own_group.run_with_budget(fn, seconds=5.0)
    finally:
        release.set()


def test_an_exit_in_the_main_thread_while_waiting_propagates():
    import _thread

    def leave(signum, frame):
        raise SystemExit(3)

    previous = signal.signal(signal.SIGUSR1, leave)
    release = threading.Event()

    def fn():
        _thread.interrupt_main(signal.SIGUSR1)
        release.wait(10)

    try:
        with pytest.raises(SystemExit):
            own_group.run_with_budget(fn, seconds=5.0)
    finally:
        release.set()
        signal.signal(signal.SIGUSR1, previous)
