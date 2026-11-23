"""Part B: the pointed group is found for tonight and offered to the Director. Synthetic data only."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.state_store import JsonFileStateStore
from src.zeitgeist import reflector, run_record

NOW = datetime(2026, 11, 22, 4, 0, tzinfo=timezone.utc)
HASHES = ["p1", "p2", "p3", "p4"]


def _cluster(cid, hashes):
    return SimpleNamespace(cluster_id=cid, items=[{"url_hash": h} for h in hashes], commission_score=1.0,
                           velocity=0.5, orbit_resonance=1.0, cluster_keywords=["kw"])


def _night(extra_hashes=("p1", "p2", "p3")):
    groups = [_cluster("g_%d" % n, ["h%d_%d" % (n, k) for k in range(3)]) for n in range(7)]
    groups[6].items += [{"url_hash": h} for h in extra_hashes]
    return groups


def _record(kind="daily_run", *, hours=2, status="pointer", hashes=HASHES, key="reflector", written=None, arm=None,
            same_model=False):
    tests = {"reflector_inputs": {}}
    tests[key] = {"status": status, "pointer_item_hashes": list(hashes) if isinstance(hashes, (list, tuple)) else hashes,
                  "same_model": same_model}
    moment = NOW - timedelta(hours=hours) if written is None else written
    return {"kind": kind, "date": "2026-11-21", "written_at": moment.isoformat() if isinstance(moment, datetime)
            else moment, "tests": tests, "arm": arm}


def _store(tmp_path, *records):
    store = JsonFileStateStore(repo_root=tmp_path)
    for index, record in enumerate(records):
        assert run_record.write_run_record(store, "%s:2026-11-21:local:%d" % (record["kind"], index), record)
    return store


# ---- the switch ----


@pytest.mark.parametrize("candidate, repair, expected", [
    (None, None, False), ("1", None, False), (None, "1", False), ("1", "0", False), ("0", "1", False),
    ("true", "1", False), ("1", "1", True),
])
def test_the_candidate_is_on_only_with_its_switch_and_the_keyword_repair(candidate, repair, expected):
    environ = {}
    if candidate is not None:
        environ["ZEITGEIST_REFLECTOR_CANDIDATE"] = candidate
    if repair is not None:
        environ["ZEITGEIST_CHOSEN_GROUP_FIELDS"] = repair
    assert reflector.candidate_enabled(environ) is expected


def test_the_candidate_switch_reads_the_process_environment(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_REFLECTOR_CANDIDATE", "1")
    monkeypatch.delenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", raising=False)
    assert reflector.candidate_enabled() is False
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    assert reflector.candidate_enabled() is True


# ---- which pointers count, through the production path ----


def _pointer(store, night=None):
    """The pointer the night's lookup used: its run id, or None."""
    return reflector.candidate_for_night(store, _night() if night is None else night, NOW)["pointer_run_id"]


@pytest.mark.parametrize("kind", ["daily_run", "private_run"])
def test_a_fresh_pointer_of_either_kind_is_found(tmp_path, kind):
    store = _store(tmp_path, _record(kind))
    found = reflector.candidate_for_night(store, _night(), NOW)
    assert found["pointer_run_id"] == "%s:2026-11-21:local:0" % kind and found["offered"] is True


@pytest.mark.parametrize("hours, found", [(35.9, True), (36.1, False), (37, False), (-1, False)])
def test_only_a_pointer_written_in_the_last_36_hours_counts(tmp_path, hours, found):
    assert (_pointer(_store(tmp_path, _record(hours=hours))) is not None) is found


@pytest.mark.parametrize("status", ["named", "void", "disputed", "skipped", "failed", None])
def test_any_other_status_is_not_a_pointer(tmp_path, status):
    assert _pointer(_store(tmp_path, _record(status=status))) is None


def test_a_trial_result_is_never_read(tmp_path):
    assert _pointer(_store(tmp_path, _record(key="reflector_trial"))) is None


def test_another_kind_of_record_is_not_read(tmp_path):
    assert _pointer(_store(tmp_path, _record("other_run"))) is None


@pytest.mark.parametrize("hashes", [None, [], [""], [3, None], "p1"])
def test_a_malformed_pointer_is_not_offered(tmp_path, hashes):
    assert _pointer(_store(tmp_path, _record(hashes=hashes))) is None


@pytest.mark.parametrize("written", ["not a time", None, ""])
def test_an_unreadable_moment_is_not_offered(tmp_path, written):
    record = _record()
    record["written_at"] = written
    assert _pointer(_store(tmp_path, record)) is None


def test_the_newest_pointer_wins_across_kinds(tmp_path):
    store = _store(tmp_path, _record("daily_run", hours=10, hashes=["old"]), _record("private_run", hours=3),
                   _record("daily_run", hours=1, status="void"))
    assert _pointer(store) == "private_run:2026-11-21:local:1"


def test_pointers_written_at_one_moment_go_to_the_smaller_run_id(tmp_path):
    moment = NOW - timedelta(hours=2)
    store = JsonFileStateStore(repo_root=tmp_path)
    for run_id, hashes in (("private_run:b", ["p1"]), ("daily_run:z", HASHES), ("private_run:a", HASHES)):
        record = _record("private_run" if run_id.startswith("private") else "daily_run", written=moment,
                         hashes=hashes)
        assert run_record.write_run_record(store, run_id, record)
    assert _pointer(store) == "daily_run:z"


class _Raising:
    def load_run_records(self, **kwargs):
        raise RuntimeError("Zqsecret store")


class _Counting:
    def __init__(self, records):
        self.records, self.calls = records, []

    def load_run_records(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.records)


def test_the_lookup_reads_the_run_records_once_and_filters_the_kinds_itself():
    records = [dict(_record("other_run"), run_id="other_run:x"), dict(_record("private_run", hours=3), run_id="p:1"),
               dict(_record("daily_run", hours=1), run_id="d:1")]
    store = _Counting(records)
    assert _pointer(store) == "d:1"
    [call] = store.calls
    assert call.get("kind") is None and call["limit"] == reflector.POINTER_LOAD_LIMIT


def test_newest_pointer_is_gone():
    assert not hasattr(reflector, "newest_pointer")


# ---- which records can supply a pointer ----


def _supplier(store):
    return reflector.candidate_for_night(store, _night(), NOW)["pointer_run_id"]


@pytest.mark.parametrize("kind, arm", [("daily_run", None), ("private_run", None), ("private_run", "current")])
def test_the_edition_the_single_private_run_and_the_current_arm_can_supply_a_pointer(tmp_path, kind, arm):
    assert _supplier(_store(tmp_path, _record(kind, arm=arm))) == "%s:2026-11-21:local:0" % kind


@pytest.mark.parametrize("arm", ["previous", "current_links", "current_reflector", "Zqunknown"])
def test_no_other_comparison_arm_can_supply_a_pointer(tmp_path, arm):
    assert _supplier(_store(tmp_path, _record("private_run", arm=arm))) is None


@pytest.mark.parametrize("kind, arm", [("daily_run", None), ("private_run", None), ("private_run", "current")])
@pytest.mark.parametrize("same_model", [True, None, "false", 0])
def test_a_reflector_on_the_directors_own_model_never_supplies_a_pointer(tmp_path, kind, arm, same_model):
    record = _record(kind, arm=arm, same_model=same_model)
    if same_model is None:
        del record["tests"]["reflector"]["same_model"]
    assert _supplier(_store(tmp_path, record)) is None


def test_an_excluded_newer_pointer_leaves_the_older_eligible_one(tmp_path):
    store = _store(tmp_path, _record("daily_run", hours=5), _record("private_run", hours=1, arm="previous"),
                   _record("private_run", hours=2, same_model=True))
    assert _supplier(store) == "daily_run:2026-11-21:local:0"


# ---- matching tonight's groups ----


def test_a_match_needs_at_least_half_of_the_pointers_items():
    pointer = {"run_id": "r", "item_hashes": HASHES}
    group, overlap = reflector.match_pointer(pointer, _night(extra_hashes=("p1", "p2")))
    assert group.cluster_id == "g_6" and overlap == pytest.approx(0.5)
    group, overlap = reflector.match_pointer(pointer, _night(extra_hashes=("p1",)))
    assert group is None and overlap == pytest.approx(0.25)
    group, overlap = reflector.match_pointer(pointer, _night(extra_hashes=()))
    assert group is None and overlap == 0.0


def test_the_largest_share_wins_and_a_tie_goes_to_the_higher_rank():
    pointer = {"run_id": "r", "item_hashes": HASHES}
    night = _night(extra_hashes=("p1", "p2", "p3"))
    night[2].items += [{"url_hash": "p4"}, {"url_hash": "p1"}]
    assert reflector.match_pointer(pointer, night)[0].cluster_id == "g_6"
    tie = _night(extra_hashes=("p1", "p2"))
    tie[3].items += [{"url_hash": "p3"}, {"url_hash": "p4"}]
    group, overlap = reflector.match_pointer(pointer, tie)
    assert group.cluster_id == "g_3" and overlap == pytest.approx(0.5)


def test_items_as_objects_are_read_too():
    pointer = {"run_id": "r", "item_hashes": ["a", "b"]}
    cluster = SimpleNamespace(cluster_id="g", items=[SimpleNamespace(url_hash="a"), SimpleNamespace(url_hash="b")])
    assert reflector.match_pointer(pointer, [cluster]) == (cluster, 1.0)


# ---- the candidate for the night ----

CANDIDATE_KEYS = {"pointer_run_id", "matched_cluster_id", "overlap", "already_shown", "offered", "group", "lookup"}


def test_a_match_beyond_the_five_is_offered(tmp_path):
    night = _night()
    found = reflector.candidate_for_night(_store(tmp_path, _record()), night, NOW)
    assert set(found) == CANDIDATE_KEYS
    assert found == {"pointer_run_id": "daily_run:2026-11-21:local:0", "matched_cluster_id": "g_6",
                     "overlap": pytest.approx(0.75), "already_shown": False, "offered": True, "group": night[6],
                     "lookup": "ok"}


def test_a_match_among_the_five_is_already_shown_and_nothing_is_offered(tmp_path):
    night = _night(extra_hashes=())
    night[4].items += [{"url_hash": h} for h in HASHES]
    found = reflector.candidate_for_night(_store(tmp_path, _record()), night, NOW)
    assert found["matched_cluster_id"] == "g_4" and found["already_shown"] is True
    assert found["offered"] is False and found["group"] is None


def test_no_match_keeps_the_pointer_but_offers_nothing(tmp_path):
    found = reflector.candidate_for_night(_store(tmp_path, _record()), _night(extra_hashes=("p1",)), NOW)
    assert found == {"pointer_run_id": "daily_run:2026-11-21:local:0", "matched_cluster_id": None,
                     "overlap": pytest.approx(0.25), "already_shown": False, "offered": False, "group": None,
                     "lookup": "ok"}


NOTHING = {"pointer_run_id": None, "matched_cluster_id": None, "overlap": None, "already_shown": False,
           "offered": False, "group": None, "lookup": "ok"}
FAILED_LOOKUP = dict(NOTHING, lookup="failed")


def test_no_pointer_offers_nothing(tmp_path):
    assert reflector.candidate_for_night(_store(tmp_path), _night(), NOW) == NOTHING


def test_the_lookup_never_raises():
    class Unreadable:
        @property
        def items(self):
            raise RuntimeError("Zqsecret")

    store = SimpleNamespace(load_run_records=lambda **kw: [dict(_record(), run_id="daily_run:x")])
    assert reflector.candidate_for_night(_Raising(), _night(), NOW) == FAILED_LOOKUP
    assert reflector.candidate_for_night(object(), _night(), NOW) == FAILED_LOOKUP
    assert reflector.candidate_for_night(store, [Unreadable(), Unreadable()], NOW) == FAILED_LOOKUP
    assert reflector.candidate_for_night(store, None, NOW)["offered"] is False


def test_a_read_that_raises_fails_the_whole_lookup():
    assert reflector.candidate_for_night(_Raising(), _night(), NOW) == FAILED_LOOKUP


def test_a_finished_lookup_says_ok_whatever_it_found(tmp_path):
    assert reflector.candidate_for_night(_store(tmp_path / "a", _record()), _night(), NOW)["lookup"] == "ok"
    assert reflector.candidate_for_night(_store(tmp_path / "b"), _night(), NOW)["lookup"] == "ok"
    junk = SimpleNamespace(load_run_records=lambda **kw: [1, "x", None])
    assert reflector.candidate_for_night(junk, _night(), NOW)["lookup"] == "ok"


def test_the_record_value_leaves_the_group_out_and_says_whether_it_was_shown(tmp_path):
    found = reflector.candidate_for_night(_store(tmp_path, _record()), _night(), NOW)
    plain = {k: v for k, v in found.items() if k != "group"}
    assert reflector.candidate_record(found, shown=True) == dict(plain, shown=True)
    assert reflector.candidate_record(found) == dict(plain, shown=False)
    nothing = reflector.candidate_for_night(_store(tmp_path / "none"), _night(), NOW)
    # Nothing offered is never shown.
    assert reflector.candidate_record(nothing, shown=True)["shown"] is False
    assert reflector.candidate_record(None) is None


# ---- the edition: the lookup runs before the Director call, inside a 20-second budget ----

import threading  # noqa: E402
import time  # noqa: E402

from src.zeitgeist import adk_pipeline as pipeline  # noqa: E402
from src.zeitgeist.adk_pipeline import commission_daily_zeitgeist as _REAL_COMMISSION  # noqa: E402
from src.zeitgeist.adk_pipeline import score_and_cluster_signals as _REAL_SCORE  # noqa: E402
from src.zeitgeist.adk_pipeline import select_and_render_pulse_assets as _REAL_RENDER  # noqa: E402
from src.zeitgeist import own_group  # noqa: E402
from src.zeitgeist.understanding import public_keywords  # noqa: E402
from tests.test_run_record_wiring import SLATE, _edition_setup, _MemoryRecordStore, _published_state  # noqa: E402
from tests.test_understanding_pipeline import _reply  # noqa: E402

SWITCHES = ("ZEITGEIST_REFLECTOR_CANDIDATE", "ZEITGEIST_CHOSEN_GROUP_FIELDS", "ZEITGEIST_REFLECTOR",
            "ZEITGEIST_OWN_GROUP", "ZEITGEIST_LINKS")


@pytest.fixture(autouse=True)
def _switches_off_unless_set(monkeypatch):
    for name in SWITCHES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)


class PointerStore(_MemoryRecordStore):
    """Run records in memory; the pointer record names the items of tonight's group at `rank` (from 0)."""

    def __init__(self, root, events, *, seen, rank=5, fail=None, stall=None):
        super().__init__(root, events)
        self.seen, self.rank, self.fail, self.stall = seen, rank, fail, stall

    def load_run_records(self, *, kind=None, limit=50):
        self.events.append("load_run_records")
        if self.stall is not None:
            self.stall.wait(10)
        if self.fail is not None:
            raise self.fail
        if kind not in (None, "daily_run") or self.rank is None:
            return []
        group = self.seen[0][self.rank]
        moment = datetime.now(timezone.utc) - timedelta(hours=2)
        return [{"kind": "daily_run", "run_id": "daily_run:2026-11-20:local:1", "written_at": moment.isoformat(),
                 "arm": None, "tests": {"reflector": {"status": "pointer", "same_model": False,
                                                      "pointer_item_hashes": [item.url_hash for item in group.items]}}}]


def _switches(monkeypatch, candidate, repair="1"):
    for name, value in (("ZEITGEIST_REFLECTOR_CANDIDATE", candidate), ("ZEITGEIST_CHOSEN_GROUP_FIELDS", repair)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def _candidate_edition(tmp_path, monkeypatch, *, choose="cluster_0", **store_kwargs):
    """A full offline edition with the run record on; the Director replies `choose` (a callable gets the groups)."""
    seen, renders = [], []
    # Each spy wraps the real function, so a second run in one test is spied on its own.
    def score(rows, **kw):
        clusters = _REAL_SCORE(rows, **kw)
        # The offline fixture gives every item one url_hash; a pointer needs them distinct.
        for cluster in clusters:
            for index, item in enumerate(cluster.items):
                object.__setattr__(item, "url_hash", "u_%s_%d" % (cluster.cluster_id, index))
        seen.append(clusters)
        return clusters

    monkeypatch.setattr(pipeline, "score_and_cluster_signals", score)
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets",
                        lambda *a, **kw: renders.append(kw) or _REAL_RENDER(*a, **kw))
    # _edition_setup wraps the function it finds; a second run in one test must wrap the real one again.
    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", _REAL_COMMISSION)
    client, events = _edition_setup(tmp_path, monkeypatch, switch="1")

    class Chooser:
        def __getitem__(self, index):
            chosen = choose(seen[0]) if callable(choose) else choose
            return _reply(selected_cluster_id=chosen, candidate_words=SLATE)

    client.replies = Chooser()

    def generate(*, model, contents, config):
        client.prompts.append(contents)
        return SimpleNamespace(text=client.replies[0])

    client.generate_content = generate
    store = PointerStore(tmp_path, events, seen=seen, **store_kwargs)
    analysis, decision, _ = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run(
        "2026-11-21", "blueprint", offline_demo=True)
    return SimpleNamespace(store=store, client=client, seen=seen, renders=renders,
                           result=(analysis.word_of_the_day, analysis.explanation_phrase, decision.selected_mode),
                           record=store.records[0][1] if store.records else None)


def _cluster_lines(prompt):
    return [line for line in prompt.split("\n") if line.startswith("- Cluster ")]


def _published(run, root):
    state = run.store.load_pulse_state()["published_pulse"]
    return _published_state(run.store, root), {key: state.get(key) for key in sorted(state)
                                              if key not in ("published_at", "updated_at", "run_id", "timestamp_utc")}


@pytest.mark.parametrize("candidate, repair", [(None, "1"), ("0", "1"), ("1", None), ("1", "0")])
def test_off_means_off_no_read_no_thread_and_the_same_bytes(tmp_path, monkeypatch, candidate, repair):
    _switches(monkeypatch, None, None)
    baseline = _candidate_edition(tmp_path / "base", monkeypatch)
    _switches(monkeypatch, candidate, repair)
    started = []

    class _Thread(threading.Thread):
        def start(self):
            started.append(self.name)
            super().start()

    monkeypatch.setattr(own_group, "threading", SimpleNamespace(Thread=_Thread, Event=threading.Event))
    run = _candidate_edition(tmp_path / "run", monkeypatch)
    assert "load_run_records" not in run.store.events and started == []
    if repair == "1":
        assert run.client.prompts == baseline.client.prompts
    assert run.result == baseline.result
    assert run.record["reflector_candidate"] is None


def test_a_pointed_group_beyond_the_five_is_one_more_plain_cluster_line_after_them(tmp_path, monkeypatch):
    _switches(monkeypatch, "0")
    off = _candidate_edition(tmp_path / "off", monkeypatch)
    _switches(monkeypatch, "1")
    on = _candidate_edition(tmp_path / "on", monkeypatch)
    extra = on.seen[0][5]
    [off_prompt], [on_prompt] = off.client.prompts, on.client.prompts
    off_lines, on_lines = off_prompt.split("\n"), on_prompt.split("\n")
    added = [line for line in on_lines if line not in off_lines]
    assert added == [line for line in _cluster_lines(on_prompt) if line.startswith("- Cluster %s:" % extra.cluster_id)]
    assert len(added) == 1
    # The same prompt with that one line after the fifth group, and nothing else: no sentence, no marker.
    at = on_lines.index(added[0])
    assert on_lines[:at] + on_lines[at + 1:] == off_lines
    assert on_lines[at - 1] == _cluster_lines(off_prompt)[-1]
    assert on.record["reflector_candidate"] == {
        "pointer_run_id": "daily_run:2026-11-20:local:1", "matched_cluster_id": str(extra.cluster_id),
        "overlap": 1.0, "already_shown": False, "offered": True, "lookup": "ok", "shown": True}
    assert on.result == off.result


def test_a_match_among_the_five_offers_nothing_and_says_so(tmp_path, monkeypatch):
    _switches(monkeypatch, "0")
    off = _candidate_edition(tmp_path / "off", monkeypatch)
    _switches(monkeypatch, "1")
    on = _candidate_edition(tmp_path / "on", monkeypatch, rank=2)
    assert on.client.prompts == off.client.prompts
    found = on.record["reflector_candidate"]
    assert found["already_shown"] is True and found["offered"] is False
    assert found["matched_cluster_id"] == str(on.seen[0][2].cluster_id)


@pytest.mark.parametrize("case", ["raises", "stalls", "no_pointer"])
def test_a_failing_or_stalled_lookup_leaves_the_edition_exactly_as_with_the_switch_off(tmp_path, monkeypatch, case,
                                                                                       capsys):
    _switches(monkeypatch, "0")
    off = _candidate_edition(tmp_path / "off", monkeypatch)
    off_out = capsys.readouterr()
    _switches(monkeypatch, "1")
    monkeypatch.setattr(reflector, "CANDIDATE_BUDGET_SECONDS", 0.2)
    release = threading.Event()
    kwargs = {"raises": {"fail": RuntimeError("Zqsecret store")}, "stalls": {"stall": release},
              "no_pointer": {"rank": None}}[case]
    started = time.monotonic()
    try:
        on = _candidate_edition(tmp_path / "on", monkeypatch, **kwargs)
    finally:
        release.set()
    elapsed = time.monotonic() - started
    on_out = capsys.readouterr()
    assert elapsed < 5
    assert on.result == off.result
    assert on.client.prompts == off.client.prompts and len(on.client.prompts) == len(off.client.prompts) == 1
    assert _published(on, tmp_path / "on")[1] == _published(off, tmp_path / "off")[1]
    assert on.renders and [_figures(kw) for kw in on.renders] == [_figures(kw) for kw in off.renders]
    assert on.record["reflector_candidate"]["offered"] is False
    assert on.record["reflector_candidate"]["lookup"] == {"raises": "failed", "stalls": "timeout",
                                                          "no_pointer": "ok"}[case]
    assert (on_out.out, on_out.err) == (off_out.out, off_out.err)


def test_an_interrupt_while_the_edition_waits_on_the_lookup_propagates(tmp_path, monkeypatch):
    import _thread

    _switches(monkeypatch, "1")
    release = threading.Event()

    def lookup(store, clusters, now):
        _thread.interrupt_main()
        release.wait(10)

    monkeypatch.setattr(reflector, "candidate_for_night", lookup)
    try:
        with pytest.raises(KeyboardInterrupt):
            _candidate_edition(tmp_path, monkeypatch)
    finally:
        release.set()


def _figures(kw):
    return (kw["orbit_resonance_score"], kw["velocity_score"], kw["breakout_surge"], kw["cluster_silhouette"],
            kw["cluster_keywords"])


def _of(cluster):
    return (float(cluster.orbit_resonance), float(cluster.velocity), bool(getattr(cluster, "breakout_surge", False)),
            float(cluster.silhouette_score), public_keywords(list(cluster.cluster_keywords), cluster))


def test_a_chosen_extra_group_publishes_its_own_keywords_and_figures(tmp_path, monkeypatch):
    _switches(monkeypatch, "1")
    run = _candidate_edition(tmp_path, monkeypatch, choose=lambda groups: groups[5].cluster_id)
    extra = run.seen[0][5]
    assert run.record["chosen_cluster_id"] == str(extra.cluster_id) and run.record["chosen_mapped"] is False
    [kw] = run.renders
    assert _figures(kw) == _of(extra)


def test_with_the_switch_off_the_same_reply_is_mapped_to_the_top_group(tmp_path, monkeypatch):
    _switches(monkeypatch, "0")
    run = _candidate_edition(tmp_path, monkeypatch, choose=lambda groups: groups[5].cluster_id)
    top = run.seen[0][0]
    assert run.record["chosen_cluster_id"] == str(top.cluster_id) and run.record["chosen_mapped"] is True
    [kw] = run.renders
    assert _figures(kw) == _of(top)


def test_the_own_group_check_and_the_reflector_inputs_take_a_chosen_extra_group(tmp_path, monkeypatch):
    _switches(monkeypatch, "1")
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    monkeypatch.setenv("ZEITGEIST_REFLECTOR", "1")
    checks = []
    real_check = own_group.run_own_group_check

    def check(store, run_id, **kwargs):
        checks.append(kwargs)
        return real_check(store, run_id, **kwargs)

    monkeypatch.setattr(own_group, "run_own_group_check", check)
    run = _candidate_edition(tmp_path, monkeypatch, choose=lambda groups: groups[5].cluster_id)
    extra = run.seen[0][5]
    [kwargs] = checks
    assert kwargs["chosen_cluster_id"] == str(extra.cluster_id) and kwargs["extra_group"] is extra
    tests = run.store.records[-1][1]["tests"]
    # The offline edition has no live client, so the check stops before its paid call, not at the candidate.
    assert tests["own_group"] == {"status": "skipped", "reason": "no_client"}
    assert tests["reflector_inputs"]["chosen"]["cluster_id"] == str(extra.cluster_id)


# ---- a chosen extra group through the staged run: published fields, roster bonus, photo recovery ----

from src.zeitgeist.adk_pipeline import DirectorCommission  # noqa: E402
from tests import test_director_cluster_choice as choice  # noqa: E402


def _offer_sixth(monkeypatch, calls=None):
    def lookup(store, clusters, now):
        if calls is not None:
            calls.append([c.cluster_id for c in clusters])
        group = clusters[5]
        return {"pointer_run_id": "daily_run:x", "matched_cluster_id": str(group.cluster_id), "overlap": 1.0,
                "already_shown": False, "offered": True, "group": group, "lookup": "ok"}

    monkeypatch.setattr(reflector, "candidate_for_night", lookup)


@pytest.fixture
def candidate_on(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_REFLECTOR_CANDIDATE", "1")
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")


def _choose_extra(clusters, kw):
    assert kw["extra_group"] is clusters[5]
    return clusters[5].cluster_id


def test_a_chosen_extra_group_publishes_its_fields_in_the_staged_run(tmp_path, monkeypatch, candidate_on):
    lookups = []
    _offer_sixth(monkeypatch, lookups)
    seen = choice._staged(tmp_path, monkeypatch, choose=_choose_extra)
    clusters = seen["scored"][0]
    assert len(clusters) > 5 and lookups == [[c.cluster_id for c in clusters]]
    [kw] = seen["render"]
    assert choice._figures(kw) == choice._of(clusters[5], public_keywords(list(clusters[5].cluster_keywords),
                                                                         clusters[5]))


def test_the_roster_bonus_goes_to_the_same_group_when_the_extra_group_is_chosen(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    off = choice._staged(tmp_path / "off", monkeypatch, choose=lambda clusters, kw: clusters[0].cluster_id)
    monkeypatch.setenv("ZEITGEIST_REFLECTOR_CANDIDATE", "1")
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    _offer_sixth(monkeypatch)
    on = choice._staged(tmp_path / "on", monkeypatch, choose=_choose_extra)

    def hashes(seen):
        return [[getattr(item, "url_hash", None) for item in items] for items in seen["bonus"]]

    assert on["bonus"] and hashes(on) == hashes(off)


def _extra_photo_run(tmp_path, monkeypatch, *, revise_calls):
    """The live photo branch with fake clients: the Director chooses the extra group; the photo needs a revision."""
    import sys
    import types

    from src.zeitgeist import photo_recovery, spend_guard

    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    fake_genai = types.ModuleType("google.genai")
    fake_genai.Client = lambda **kw: object()
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    google = sys.modules.get("google")
    if google is None:
        google = types.ModuleType("google")
        monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setattr(google, "genai", fake_genai, raising=False)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda client, **kw: client)
    _offer_sixth(monkeypatch)

    def commission(clusters, **kw):
        if "max_retries" in kw:
            revise_calls.append([c.cluster_id for c in clusters])
            chosen = clusters[0].cluster_id
        else:
            chosen = kw["extra_group"].cluster_id
        return DirectorCommission(word="Loom", gloss="steady weaving", commissioned_thesis="Looms keep time.",
                                  selected_cluster_id=chosen, attempts_used=1, packet_used=True)

    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", commission)
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets",
                        lambda *a, **kw: _REAL_RENDER(*a, **{**kw, "offline_demo": True, "strict_photo_errors": False}))

    def recover(current, *, render, revise, history, photo_required):
        revised = revise(current, change_word=False)
        return revised, render(revised), {"recovered": True}

    monkeypatch.setattr(photo_recovery, "recover_visual", recover)
    real_collect = pipeline.collect_96h_signals
    monkeypatch.setattr(pipeline, "collect_96h_signals",
                        lambda *a, **kw: real_collect(*a, **{**kw, "offline_demo": True}))
    scored = []
    monkeypatch.setattr(pipeline, "score_and_cluster_signals",
                        lambda rows, **kw: scored.append(_REAL_SCORE(rows, **kw)) or scored[-1])
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    roster = SimpleNamespace(load_orbit_roster=lambda: {}, save_orbit_roster=lambda r: None)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=choice._FakeStore(tmp_path, roster))
    staged._run_staged("2026-11-21", None, offline_demo=False)
    return staged, scored[0]


def test_photo_recovery_after_an_extra_group_choice_revises_that_group_and_does_not_raise(
        tmp_path, monkeypatch, candidate_on):
    revise_calls = []
    staged, clusters = _extra_photo_run(tmp_path, monkeypatch, revise_calls=revise_calls)
    assert revise_calls == [[clusters[5].cluster_id]]
    assert staged.last_commission.selected_cluster_id == clusters[5].cluster_id


def test_with_the_packet_path_off_the_switch_reads_nothing(tmp_path, monkeypatch):
    # The extra group is shown only on the packet path, so without it nothing is looked up.
    _switches(monkeypatch, "1")
    monkeypatch.delenv("ZEITGEIST_UNDERSTANDING_V1", raising=False)
    lookups, kwargs = [], []
    monkeypatch.setattr(reflector, "candidate_for_night", lambda *a: lookups.append(1))

    def choose(clusters, kw):
        kwargs.append(kw)
        return clusters[0].cluster_id

    seen = choice._staged(tmp_path, monkeypatch, choose=choose)
    assert lookups == [] and seen["render"] and "extra_group" not in kwargs[0]



# ---- the record says whether the lookup finished; the lookup's thread carries its own name ----


def test_a_lookup_that_raises_in_its_thread_is_recorded_as_failed(tmp_path, monkeypatch):
    _switches(monkeypatch, "1")

    def lookup(store, clusters, now):
        raise RuntimeError("Zqsecret lookup")

    monkeypatch.setattr(reflector, "candidate_for_night", lookup)
    run = _candidate_edition(tmp_path, monkeypatch)
    assert run.record["reflector_candidate"]["lookup"] == "failed"
    assert run.record["reflector_candidate"]["offered"] is False


def test_a_budget_that_cannot_start_is_recorded_as_failed(tmp_path, monkeypatch):
    _switches(monkeypatch, "1")

    def broken(fn, seconds, name="own-group-check"):
        raise RuntimeError("Zqsecret thread")

    monkeypatch.setattr(own_group, "run_with_budget", broken)
    run = _candidate_edition(tmp_path, monkeypatch)
    assert run.record["reflector_candidate"]["lookup"] == "failed"


def test_the_lookup_runs_on_a_thread_named_for_it(tmp_path, monkeypatch):
    _switches(monkeypatch, "1")
    names = []

    class _Thread(threading.Thread):
        def start(self):
            names.append(self.name)
            super().start()

    monkeypatch.setattr(own_group, "threading", SimpleNamespace(Thread=_Thread, Event=threading.Event))
    _candidate_edition(tmp_path, monkeypatch)
    assert names == [reflector.LOOKUP_THREAD_NAME] and reflector.LOOKUP_THREAD_NAME == "reflector-candidate-lookup"



# ---- shown: a packet attempt rendered the extra group ----

from tests.test_director_cluster_choice import _commission, _groups  # noqa: E402
from tests.test_director_cluster_choice import _reply as _choice_reply  # noqa: E402


def test_the_extra_group_counts_as_shown_only_when_a_packet_attempt_rendered_it(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    groups = _groups()
    shown, prompts = _commission([_choice_reply()], groups=groups, extra_group=groups[5])
    assert "- Cluster g_5: " in prompts[0] and pipeline.extra_group_shown(shown) is True
    older, prompts = _commission([_choice_reply()], groups=groups, extra_group=groups[5], packet=False)
    assert "- Cluster g_5: " not in prompts[0] and pipeline.extra_group_shown(older) is False
    owner = {"enabled": True, "override_word": "Selvedge", "personal_note": "A synthetic note."}
    owner_night, prompts = _commission([_choice_reply()], groups=groups, extra_group=groups[5], owner_override=owner)
    assert "- Cluster g_5: " not in "".join(prompts) and pipeline.extra_group_shown(owner_night) is False
    none_offered, _ = _commission([_choice_reply()], groups=groups)
    assert pipeline.extra_group_shown(none_offered) is False
    assert pipeline.extra_group_shown(None) is False


def test_the_edition_records_shown_false_when_no_packet_attempt_rendered_the_group(tmp_path, monkeypatch):
    _switches(monkeypatch, "1")
    run = _candidate_edition(tmp_path / "packet", monkeypatch)
    assert run.record["reflector_candidate"]["offered"] is True and run.record["reflector_candidate"]["shown"] is True

    def no_packet(*args, **kwargs):
        raise RuntimeError("synthetic packet failure")

    # The packet prompt cannot be built: the older prompt's round runs, without the extra group.
    monkeypatch.setattr(pipeline, "render_evidence_packet", no_packet)
    older = _candidate_edition(tmp_path / "older", monkeypatch)
    assert all("Cluster %s:" % older.seen[0][5].cluster_id not in prompt for prompt in older.client.prompts)
    assert older.record["reflector_candidate"]["offered"] is True
    assert older.record["reflector_candidate"]["shown"] is False


# ---- the final echo diagnostic covers the extra group ----


def test_the_final_echo_line_counts_an_echo_of_the_offered_group(capsys, monkeypatch):
    from tests.test_director_cluster_choice import ECHO, _echoing

    # The legacy ranking needs fields these synthetic groups lack; this test is about final_echo only.
    monkeypatch.setattr(pipeline, "ranking_stats", lambda clusters: (0, 1))
    groups = _echoing(_groups(), 5)
    commission = DirectorCommission(word="Echoword", gloss="g", commissioned_thesis="They wrote " + ECHO + " here.",
                                    selected_cluster_id="g_5")
    pipeline._print_understanding_line(groups, commission, commission)
    assert "final_echo=0" in capsys.readouterr().out
    pipeline._print_understanding_line(groups, commission, commission, extra_group=groups[5])
    assert "final_echo=1" in capsys.readouterr().out


def test_the_staged_run_passes_the_offered_group_to_the_final_echo_line(tmp_path, monkeypatch, candidate_on):
    _offer_sixth(monkeypatch)
    calls = []
    monkeypatch.setattr(pipeline, "_print_understanding_line", lambda *a, **kw: calls.append(kw))
    seen = choice._staged(tmp_path, monkeypatch, choose=_choose_extra)
    [kw] = calls
    assert kw["extra_group"] is seen["scored"][0][5]
