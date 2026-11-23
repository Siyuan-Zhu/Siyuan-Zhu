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


def _record(kind="daily_run", *, hours=2, status="pointer", hashes=HASHES, key="reflector", written=None):
    tests = {"reflector_inputs": {}}
    tests[key] = {"status": status, "pointer_item_hashes": list(hashes) if isinstance(hashes, (list, tuple)) else hashes}
    moment = NOW - timedelta(hours=hours) if written is None else written
    return {"kind": kind, "date": "2026-11-21", "written_at": moment.isoformat() if isinstance(moment, datetime)
            else moment, "tests": tests}


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


# ---- which pointers count ----


@pytest.mark.parametrize("kind", ["daily_run", "private_run"])
def test_a_fresh_pointer_of_either_kind_is_found(tmp_path, kind):
    store = _store(tmp_path, _record(kind))
    assert reflector.newest_pointer(store, NOW) == {"run_id": "%s:2026-11-21:local:0" % kind, "item_hashes": HASHES}


@pytest.mark.parametrize("hours, found", [(35.9, True), (36.1, False), (-1, False)])
def test_only_a_pointer_written_in_the_last_36_hours_counts(tmp_path, hours, found):
    store = _store(tmp_path, _record(hours=hours))
    assert (reflector.newest_pointer(store, NOW) is not None) is found


@pytest.mark.parametrize("status", ["named", "void", "disputed", "skipped", "failed", None])
def test_any_other_status_is_not_a_pointer(tmp_path, status):
    assert reflector.newest_pointer(_store(tmp_path, _record(status=status)), NOW) is None


def test_a_trial_result_is_never_read(tmp_path):
    store = _store(tmp_path, _record(key="reflector_trial"))
    assert reflector.newest_pointer(store, NOW) is None


def test_another_kind_of_record_is_not_read(tmp_path):
    assert reflector.newest_pointer(_store(tmp_path, _record("other_run")), NOW) is None


@pytest.mark.parametrize("hashes", [None, [], [""], [3, None], "p1"])
def test_a_malformed_pointer_is_not_offered(tmp_path, hashes):
    assert reflector.newest_pointer(_store(tmp_path, _record(hashes=hashes)), NOW) is None


@pytest.mark.parametrize("written", ["not a time", None, ""])
def test_an_unreadable_moment_is_not_offered(tmp_path, written):
    record = _record()
    record["written_at"] = written
    assert reflector.newest_pointer(_store(tmp_path, record), NOW) is None


def test_the_newest_pointer_wins_across_kinds(tmp_path):
    store = _store(tmp_path, _record("daily_run", hours=10, hashes=["old"]), _record("private_run", hours=3),
                   _record("daily_run", hours=1, status="void"))
    assert reflector.newest_pointer(store, NOW) == {"run_id": "private_run:2026-11-21:local:1", "item_hashes": HASHES}


def test_duplicate_hashes_count_once(tmp_path):
    store = _store(tmp_path, _record(hashes=["p1", "p1", "p2"]))
    assert reflector.newest_pointer(store, NOW)["item_hashes"] == ["p1", "p2"]


class _Raising:
    def load_run_records(self, **kwargs):
        raise RuntimeError("Zqsecret store")


def test_a_failing_or_missing_store_read_finds_nothing():
    assert reflector.newest_pointer(_Raising(), NOW) is None
    assert reflector.newest_pointer(object(), NOW) is None
    assert reflector.newest_pointer(SimpleNamespace(load_run_records=lambda **kw: [1, "x", None]), NOW) is None


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

CANDIDATE_KEYS = {"pointer_run_id", "matched_cluster_id", "overlap", "already_shown", "offered", "group"}


def test_a_match_beyond_the_five_is_offered(tmp_path):
    night = _night()
    found = reflector.candidate_for_night(_store(tmp_path, _record()), night, NOW)
    assert set(found) == CANDIDATE_KEYS
    assert found == {"pointer_run_id": "daily_run:2026-11-21:local:0", "matched_cluster_id": "g_6",
                     "overlap": pytest.approx(0.75), "already_shown": False, "offered": True, "group": night[6]}


def test_a_match_among_the_five_is_already_shown_and_nothing_is_offered(tmp_path):
    night = _night(extra_hashes=())
    night[4].items += [{"url_hash": h} for h in HASHES]
    found = reflector.candidate_for_night(_store(tmp_path, _record()), night, NOW)
    assert found["matched_cluster_id"] == "g_4" and found["already_shown"] is True
    assert found["offered"] is False and found["group"] is None


def test_no_match_keeps_the_pointer_but_offers_nothing(tmp_path):
    found = reflector.candidate_for_night(_store(tmp_path, _record()), _night(extra_hashes=("p1",)), NOW)
    assert found == {"pointer_run_id": "daily_run:2026-11-21:local:0", "matched_cluster_id": None,
                     "overlap": pytest.approx(0.25), "already_shown": False, "offered": False, "group": None}


NOTHING = {"pointer_run_id": None, "matched_cluster_id": None, "overlap": None, "already_shown": False,
           "offered": False, "group": None}


def test_no_pointer_offers_nothing(tmp_path):
    assert reflector.candidate_for_night(_store(tmp_path), _night(), NOW) == NOTHING


def test_the_lookup_never_raises():
    class Unreadable:
        @property
        def items(self):
            raise RuntimeError("Zqsecret")

    store = SimpleNamespace(load_run_records=lambda **kw: [dict(_record(), run_id="daily_run:x")])
    assert reflector.candidate_for_night(_Raising(), _night(), NOW) == NOTHING
    assert reflector.candidate_for_night(store, [Unreadable(), Unreadable()], NOW)["offered"] is False
    assert reflector.candidate_for_night(store, None, NOW)["offered"] is False


def test_the_record_value_leaves_the_group_out(tmp_path):
    found = reflector.candidate_for_night(_store(tmp_path, _record()), _night(), NOW)
    assert reflector.candidate_record(found) == {k: v for k, v in found.items() if k != "group"}
    assert reflector.candidate_record(None) is None
