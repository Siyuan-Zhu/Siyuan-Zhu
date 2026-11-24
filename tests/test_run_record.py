"""The private run record: built from plain values, written once, never fails a run, never prints."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.zeitgeist import run_record as rr
from src.zeitgeist.adk_pipeline import DirectorCommission
from src.zeitgeist.methodology import CachedItem, ScoredCluster
from src.zeitgeist.model_choice import RunModels

NOW = datetime(2026, 11, 22, 8, 30, tzinfo=timezone.utc)
SWITCH_NAMES = (
    "ZEITGEIST_LIVE_LLM", "ZEITGEIST_LIVE_COLLECT", "ZEITGEIST_LIVE_UNSPLASH", "ZEITGEIST_OFFLINE_DEMO",
    "ZEITGEIST_UNDERSTANDING_V1", "ZEITGEIST_RUN_RECORD", "ZEITGEIST_DIRECTOR_MODEL", "ZEITGEIST_JUDGE_MODEL",
    "ZEITGEIST_EMBEDDING_MODEL", "ZEITGEIST_LINKS", "ZEITGEIST_CHOSEN_GROUP_FIELDS", "ZEITGEIST_OWN_NIGHT",
    "ZEITGEIST_OWN_GROUP", "ZEITGEIST_REFLECTOR", "ZEITGEIST_REFLECTOR_MODEL", "ZEITGEIST_REFLECTOR_CANDIDATE",
    "ZEITGEIST_UNDATED_AS_BACKGROUND",
    "ZEITGEIST_COPY_CAP",
    "ZEITGEIST_INTERIM_GROUPING",
)
ENVIRON = {
    "GITHUB_SHA": "0123abcd",
    "GITHUB_RUN_ID": "777",
    "GITHUB_RUN_ATTEMPT": "2",
    "ZEITGEIST_LIVE_LLM": "1",
    "ZEITGEIST_UNDERSTANDING_V1": "1",
    "ZEITGEIST_RUN_RECORD": "1",
    "ZEITGEIST_DIRECTOR_MODEL": "gemini-4-hydrogen",
    "ZEITGEIST_NOT_A_LISTED_SWITCH": "synthetic",
    "UNRELATED": "synthetic",
}
EXPECTED_KEYS = {
    "record_schema", "kind", "date", "written_at", "commit", "switches", "models", "prompt_path",
    "attempts_used", "groups", "evidence_text", "chosen_cluster_id", "chosen_rank", "candidate_words",
    "word", "gloss", "thesis", "commission_sha256", "prompt_sha256", "revised_by_photo_recovery",
    "main_pass", "intake_hash", "cutoffs_source", "arm", "repeat", "tests", "links", "realm_counts",
    "chosen_mapped", "reflector_candidate",
}
GROUP_KEYS = {
    "cluster_id", "rank", "commission_score", "velocity", "orbit_resonance", "s_intra",
    "unique_sources_count", "item_count", "item_hashes", "instagram_item_hashes",
}
CANDIDATES = [{"word": f"synthword{i}", "fit": f"synthetic fit {i}"} for i in range(8)]


def _item(n: int, source_id: str) -> CachedItem:
    return CachedItem(url_hash=f"hash_{n:04d}", source_id=source_id, title=f"synthetic title {n}")


def _cluster(index: int) -> ScoredCluster:
    items = [_item(index * 10 + 1, f"m1_feed_{index}"), _item(index * 10 + 2, "o3_ig_synthetic_account"),
             {"url_hash": f"hash_{index * 10 + 3:04d}", "source_id": "o4_ig_scout_synthetic"}]
    return ScoredCluster(
        cluster_id=f"cluster_{index}", items=items, cluster_keywords=["synthetic"],
        silhouette_score=np.float32(0.5), w_recent=1.0, w_base_daily=1.0, breakout_surge=0.0,
        velocity=np.float64(2.0 + index), orbit_resonance=np.float32(1.25), s_intra=np.float64(0.75),
        unique_sources=("a", "b"), unique_sources_count=np.int64(2 + index),
        commission_score=np.float64(10.0 - index),
    )


CLUSTERS = [_cluster(i) for i in range(6)]


def _commission(**overrides) -> DirectorCommission:
    values = dict(word="Synthword", gloss="A synthetic gloss", commissioned_thesis="A synthetic thesis.",
                  selected_cluster_id="cluster_2", attempts_used=2, packet_used=True)
    values.update(overrides)
    return DirectorCommission(**values)


def _models(**overrides) -> RunModels:
    values = dict(director="gemini-4-hydrogen", judge="gemini-4-hydrogen", embedding="gemini-embedding-2")
    values.update(overrides)
    return RunModels(**values)


def _build(**overrides) -> dict:
    values = dict(
        kind="daily_run", date_iso="2026-11-22", now_utc=NOW, environ=ENVIRON, run_models=_models(),
        embedding_model="gemini-embedding-2", clusters=CLUSTERS, evidence_text="synthetic evidence block",
        commission=_commission(), candidate_words=CANDIDATES, chosen_cluster_id="cluster_2",
        prompt_sha256="f" * 64,
    )
    values.update(overrides)
    return rr.build_run_record(**values)


class _Store:
    def __init__(self) -> None:
        self.saved: list[tuple[str, dict]] = []

    def save_run_record(self, run_id: str, document: dict) -> None:
        self.saved.append((run_id, document))


class _RaisingStore:
    def save_run_record(self, run_id: str, document: dict) -> None:
        raise RuntimeError("synthetic store failure")


@pytest.fixture(autouse=True)
def _nothing_printed_or_logged(capsys, caplog):
    caplog.set_level(logging.DEBUG)
    yield
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    assert caplog.records == []


def test_constants() -> None:
    assert rr.RECORD_SCHEMA == 1
    assert rr.MAX_RECORD_BYTES == 900_000


@pytest.mark.parametrize("value, expected", [("1", True), ("0", False), ("", False), ("true", False), (" 1", False)])
def test_switch_is_on_only_for_exactly_one(value, expected) -> None:
    assert rr.run_record_enabled({"ZEITGEIST_RUN_RECORD": value}) is expected


def test_switch_is_off_when_unset(monkeypatch) -> None:
    assert rr.run_record_enabled({}) is False
    monkeypatch.delenv("ZEITGEIST_RUN_RECORD", raising=False)
    assert rr.run_record_enabled() is False
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    assert rr.run_record_enabled() is True


def test_run_id_from_the_actions_run() -> None:
    assert rr.make_run_id("daily_run", "2026-11-22", ENVIRON) == "daily_run:2026-11-22:777:2"
    assert rr.make_run_id("private_run", "2026-11-22", ENVIRON, arm="current", repeat=3) == (
        "private_run:2026-11-22:777:2:current:3")


def test_run_id_outside_actions_is_local() -> None:
    assert rr.make_run_id("daily_run", "2026-11-22", {}) == "daily_run:2026-11-22:local:1"


def test_every_key_is_present_with_its_value() -> None:
    commission = _commission()
    record = _build(commission=commission, intake_hash="a" * 64, arm="current", repeat=1,
                    extra={"own_night": 0.25})
    assert set(record) == EXPECTED_KEYS
    assert record["record_schema"] == 1
    assert record["kind"] == "daily_run"
    assert record["date"] == "2026-11-22"
    assert record["written_at"] == "2026-11-22T08:30:00+00:00"
    assert record["commit"] == "0123abcd"
    assert record["models"] == {"director": "gemini-4-hydrogen", "judge": "gemini-4-hydrogen",
                                "embedding": "gemini-embedding-2", "fallback_fired": False,
                                "unrecognized": False}
    assert record["prompt_path"] == "packet"
    assert record["attempts_used"] == 2
    assert record["evidence_text"] == "synthetic evidence block"
    assert record["chosen_cluster_id"] == "cluster_2"
    assert record["chosen_rank"] == 3
    assert record["candidate_words"] == CANDIDATES
    assert (record["word"], record["gloss"], record["thesis"]) == (
        "Synthword", "A synthetic gloss", "A synthetic thesis.")
    assert record["commission_sha256"] == hashlib.sha256(json.dumps(
        commission.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    assert record["prompt_sha256"] == "f" * 64
    assert record["revised_by_photo_recovery"] is False
    assert record["main_pass"] == {"word": "Synthword", "gloss": "A synthetic gloss"}
    assert record["intake_hash"] == "a" * 64
    assert (record["arm"], record["repeat"]) == ("current", 1)
    assert record["tests"] == {"own_night": 0.25}


def test_switches_are_the_fixed_list_as_strings_or_null() -> None:
    switches = _build()["switches"]
    assert tuple(switches) == SWITCH_NAMES
    assert switches["ZEITGEIST_LIVE_LLM"] == "1"
    assert switches["ZEITGEIST_DIRECTOR_MODEL"] == "gemini-4-hydrogen"
    assert switches["ZEITGEIST_LINKS"] is None
    assert switches["ZEITGEIST_REFLECTOR_CANDIDATE"] is None
    assert switches["ZEITGEIST_COPY_CAP"] is None
    assert "ZEITGEIST_NOT_A_LISTED_SWITCH" not in switches


def test_the_first_five_groups_with_every_score_term_and_item_pointers() -> None:
    groups = _build()["groups"]
    assert [g["cluster_id"] for g in groups] == [f"cluster_{i}" for i in range(5)]
    for index, group in enumerate(groups):
        assert set(group) == GROUP_KEYS
        assert group["rank"] == index + 1
        assert group["commission_score"] == 10.0 - index
        assert group["velocity"] == 2.0 + index
        assert group["orbit_resonance"] == 1.25
        assert group["s_intra"] == 0.75
        assert group["unique_sources_count"] == 2 + index
        assert group["item_count"] == 3
        assert group["item_hashes"] == [f"hash_{index * 10 + 1:04d}"]
        assert group["instagram_item_hashes"] == [f"hash_{index * 10 + 2:04d}", f"hash_{index * 10 + 3:04d}"]
        for key in ("commission_score", "velocity", "orbit_resonance", "s_intra"):
            assert type(group[key]) is float
        for key in ("rank", "unique_sources_count", "item_count"):
            assert type(group[key]) is int


def test_instagram_test_mirrors_the_pipeline() -> None:
    assert rr.is_instagram_item({"source_id": "o3_ig_synthetic"}) is True
    assert rr.is_instagram_item(SimpleNamespace(source_id="o1_ig_x")) is True
    assert rr.is_instagram_item({"source_id": "m1_feed"}) is False
    assert rr.is_instagram_item({}) is False


def test_chosen_rank_is_null_for_an_unknown_id() -> None:
    assert _build(chosen_cluster_id="cluster_99")["chosen_rank"] is None
    # A group past the first five is not among the groups the record holds.
    assert _build(chosen_cluster_id="cluster_5")["chosen_rank"] is None


def test_older_prompt_path_without_candidate_words() -> None:
    record = _build(commission=_commission(packet_used=False, attempts_used=1), candidate_words=None,
                    evidence_text="- Cluster cluster_0: synthetic")
    assert record["prompt_path"] == "legacy"
    assert record["attempts_used"] == 1
    assert record["candidate_words"] == []
    assert record["evidence_text"] == "- Cluster cluster_0: synthetic"


def test_a_fallback_night_says_so() -> None:
    models = _models()
    models.give_up("director")
    record = _build(run_models=models)
    assert record["models"]["fallback_fired"] is True
    assert record["models"]["director"] == "gemini-3.8-flash"
    assert record["models"]["judge"] == "gemini-3.8-flash"


def test_a_photo_recovery_revision_keeps_the_main_pass() -> None:
    main = _commission(word="Mainword", gloss="Main synthetic gloss", packet_used=True, attempts_used=2)
    revised = _commission(word="Revisedword", gloss="Revised synthetic gloss",
                          commissioned_thesis="Revised synthetic thesis.", packet_used=False, attempts_used=1)
    record = _build(commission=revised, main_commission=main)
    assert (record["word"], record["gloss"], record["thesis"]) == (
        "Revisedword", "Revised synthetic gloss", "Revised synthetic thesis.")
    assert record["revised_by_photo_recovery"] is True
    assert record["main_pass"] == {"word": "Mainword", "gloss": "Main synthetic gloss"}
    assert record["prompt_path"] == "packet"
    assert record["attempts_used"] == 2
    assert record["commission_sha256"] == hashlib.sha256(json.dumps(
        revised.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def test_the_same_commission_as_main_pass_is_not_a_revision() -> None:
    commission = _commission()
    record = _build(commission=commission, main_commission=commission)
    assert record["revised_by_photo_recovery"] is False
    assert record["main_pass"] == {"word": record["word"], "gloss": record["gloss"]}


def test_awkward_values_do_not_break_the_write() -> None:
    record = _build(extra={"when": datetime(2026, 11, 22, tzinfo=timezone.utc), "seen": {"b", "a"},
                           "score": np.float32(0.5), "count": np.int32(3), "nan": float("nan")})
    store = _Store()
    assert rr.write_run_record(store, "daily_run:2026-11-22:local:1", record) is True
    [(run_id, document)] = store.saved
    assert run_id == "daily_run:2026-11-22:local:1"
    assert json.loads(json.dumps(document, allow_nan=False)) == document
    assert document["tests"] == {"when": "2026-11-22T00:00:00+00:00", "seen": ["a", "b"], "score": 0.5,
                                 "count": 3, "nan": None}


def test_awkward_values_handed_straight_to_the_write_are_made_plain() -> None:
    store = _Store()
    record = {"written_at": NOW, "value": np.float64(1.5), "tags": {"x"}, "pair": (1, 2)}
    assert rr.write_run_record(store, "daily_run:2026-11-22:local:1", record) is True
    assert store.saved[0][1] == {"written_at": "2026-11-22T08:30:00+00:00", "value": 1.5, "tags": ["x"],
                                 "pair": [1, 2], "run_id": "daily_run:2026-11-22:local:1"}


def test_a_store_that_raises_returns_false() -> None:
    assert rr.write_run_record(_RaisingStore(), "daily_run:2026-11-22:local:1", _build()) is False


def test_a_store_without_the_method_returns_false() -> None:
    assert rr.write_run_record(object(), "daily_run:2026-11-22:local:1", _build()) is False


def test_a_record_over_the_size_cap_is_not_written() -> None:
    store = _Store()
    record = _build(evidence_text="x" * (rr.MAX_RECORD_BYTES + 1))
    assert rr.write_run_record(store, "daily_run:2026-11-22:local:1", record) is False
    assert store.saved == []


def test_a_record_that_cannot_be_measured_is_not_written() -> None:
    store = _Store()
    assert rr.write_run_record(store, "daily_run:2026-11-22:local:1", {"bad": object()}) is False
    assert store.saved == []


def test_a_build_that_raises_inside_never_escapes_write() -> None:
    class _Exploding(dict):
        def items(self):
            raise RuntimeError("synthetic")

    store = _Store()
    assert rr.write_run_record(store, "daily_run:2026-11-22:local:1", _Exploding(a=1)) is False
    assert store.saved == []


def test_the_module_makes_no_network_or_provider_call(monkeypatch) -> None:
    import socket
    import urllib.request

    def _refuse(*args, **kwargs):
        raise AssertionError("network call attempted")

    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(urllib.request, "urlopen", _refuse)
    store = _Store()
    record = _build()
    assert rr.write_run_record(store, rr.make_run_id("daily_run", "2026-11-22", ENVIRON), record) is True
    assert [run_id for run_id, _ in store.saved] == ["daily_run:2026-11-22:777:2"]
    source = open(rr.__file__, encoding="utf-8").read()
    for forbidden in ("import socket", "urllib", "requests", "genai", "generate_content", "embed_content",
                      "print(", "logging", "logger"):
        assert forbidden not in source


# ---- adding test numbers to a stored record ----


def _stored(record=None):
    from src.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    rr.write_run_record(store, RUN_ID, _build() if record is None else record)
    return store


RUN_ID = "daily_run:2026-11-22:777:2"


def test_add_test_numbers_sets_only_that_test_key() -> None:
    store = _stored()
    before = store.load_run_record(RUN_ID)
    assert rr.add_test_numbers(store, RUN_ID, "specificity", {"margin": 0.25, "groups": 5}) is True
    after = store.load_run_record(RUN_ID)
    assert after["tests"] == {"specificity": {"margin": 0.25, "groups": 5}}
    assert {key: value for key, value in after.items() if key != "tests"} == {
        key: value for key, value in before.items() if key != "tests"}


def test_add_test_numbers_keeps_the_other_tests() -> None:
    store = _stored(_build(extra={"first": {"n": 1}}))
    assert rr.add_test_numbers(store, RUN_ID, "second", {"n": 2.5}) is True
    assert store.load_run_record(RUN_ID)["tests"] == {"first": {"n": 1}, "second": {"n": 2.5}}


def test_add_test_numbers_on_a_missing_record_returns_false_and_writes_nothing() -> None:
    store = _stored()
    assert rr.add_test_numbers(store, "daily_run:2026-11-22:local:9", "t", {"n": 1}) is False
    assert store.load_run_record("daily_run:2026-11-22:local:9") is None


class _FailingRecordStore:
    def __init__(self, *, load_fails=False, record=None) -> None:
        self.load_fails = load_fails
        self.record = record

    def load_run_record(self, run_id: str) -> dict:
        if self.load_fails:
            raise RuntimeError("synthetic store failure")
        return self.record

    def save_run_record(self, run_id: str, document: dict) -> None:
        raise RuntimeError("synthetic store failure")


@pytest.mark.parametrize("store", [
    _FailingRecordStore(load_fails=True),
    _FailingRecordStore(record={"tests": {}}),
    _FailingRecordStore(record={"tests": "not a mapping"}),
    _FailingRecordStore(record="not a record"),
    _Store(),
    object(),
], ids=["load-raises", "save-raises", "tests-not-a-mapping", "not-a-record", "no-load", "no-methods"])
def test_add_test_numbers_on_a_failing_store_returns_false(store) -> None:
    assert rr.add_test_numbers(store, RUN_ID, "t", {"n": 1}) is False


def test_add_test_numbers_with_unplain_numbers_returns_false_and_keeps_the_record() -> None:
    store = _stored()
    before = store.load_run_record(RUN_ID)
    assert rr.add_test_numbers(store, RUN_ID, "t", {"n": object()}) is False
    assert store.load_run_record(RUN_ID) == before


# ---- records carry their id; pruning by age ----


def test_a_written_record_carries_its_run_id() -> None:
    store = _Store()
    assert rr.write_run_record(store, RUN_ID, _build()) is True
    [(run_id, document)] = store.saved
    assert run_id == document["run_id"] == RUN_ID
    assert {key: value for key, value in document.items() if key != "run_id"} == _build()


def test_a_written_record_does_not_change_the_callers_record() -> None:
    record = _build()
    rr.write_run_record(_Store(), RUN_ID, record)
    assert "run_id" not in record


def _aged(store, run_id, written_at):
    rr.write_run_record(store, run_id, dict(_build(), written_at=written_at))


def test_prune_removes_only_records_older_than_the_cutoff() -> None:
    from src.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    _aged(store, "daily_run:2026-01-01:local:1", "2026-01-01T08:00:00+00:00")
    _aged(store, "daily_run:2026-05-20:local:1", "2026-05-20T08:00:00+00:00")
    _aged(store, "private_run:2026-11-01:local:1", "2026-11-01T08:00:00+00:00")
    assert rr.prune_run_records(store, now_utc=NOW) == 2
    assert [record["run_id"] for record in store.load_run_records()] == ["private_run:2026-11-01:local:1"]
    assert rr.prune_run_records(store, older_than_days=10, now_utc=NOW) == 1
    assert store.load_run_records() == []


def test_prune_keeps_records_it_cannot_date_or_name() -> None:
    from src.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    store.save_run_record("daily_run:2026-01-01:local:1", dict(_build(), written_at="2026-01-01T08:00:00+00:00"))
    _aged(store, "daily_run:2026-01-02:local:1", "not a date")
    _aged(store, "daily_run:2026-01-03:local:1", "2026-01-03T08:00:00")
    assert rr.prune_run_records(store, now_utc=NOW) == 1
    assert sorted(record.get("run_id") or "-" for record in store.load_run_records()) == [
        "-", "daily_run:2026-01-02:local:1"]


def test_prune_defaults_to_the_current_time() -> None:
    from src.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    _aged(store, "daily_run:2020-01-01:local:1", "2020-01-01T08:00:00+00:00")
    assert rr.prune_run_records(store) == 1


class _PruneFailingStore:
    def __init__(self, *, load_fails=False) -> None:
        self.load_fails = load_fails
        self.deleted = []

    def load_run_records(self, *, kind=None, limit=50):
        if self.load_fails:
            raise RuntimeError("synthetic store failure")
        return [{"run_id": "daily_run:2026-01-0%d:local:1" % day, "written_at": "2026-01-0%dT08:00:00+00:00" % day}
                for day in (1, 2)]

    def delete_run_record(self, run_id):
        if run_id.startswith("daily_run:2026-01-01"):
            raise RuntimeError("synthetic store failure")
        self.deleted.append(run_id)


def test_prune_never_raises_and_counts_only_what_it_deleted() -> None:
    assert rr.prune_run_records(_PruneFailingStore(load_fails=True), now_utc=NOW) == 0
    store = _PruneFailingStore()
    assert rr.prune_run_records(store, now_utc=NOW) == 1
    assert store.deleted == ["daily_run:2026-01-02:local:1"]
    assert rr.prune_run_records(object(), now_utc=NOW) == 0


def test_the_record_says_which_cutoffs_grouped_the_night() -> None:
    assert _build()["cutoffs_source"] is None
    assert _build(cutoffs_source="calibrated")["cutoffs_source"] == "calibrated"
    assert _build(cutoffs_source="legacy")["cutoffs_source"] == "legacy"


# ---- lone surrogates never reach the store ----

HIGH, LOW, OTHER_LOW, REPLACEMENT = chr(0xD800), chr(0xDFFF), chr(0xDC80), chr(0xFFFD)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def test_lone_surrogates_are_replaced_before_the_write() -> None:
    candidates = [dict(entry) for entry in CANDIDATES]
    candidates[1]["word"] = "synth" + HIGH + "word"
    record = _build(candidate_words=candidates, evidence_text="synthetic " + LOW + " evidence block",
                    extra={"key" + OTHER_LOW: {"n": 1}})
    store = _Store()
    assert rr.write_run_record(store, RUN_ID, record) is True
    [(_, document)] = store.saved
    for text in _strings(document):
        text.encode("utf-8")
    assert document["candidate_words"][1]["word"] == "synth" + REPLACEMENT + "word"
    assert document["evidence_text"] == "synthetic " + REPLACEMENT + " evidence block"
    assert document["tests"] == {"key" + REPLACEMENT: {"n": 1}}
    expected = _build()
    for key in set(document) - {"candidate_words", "evidence_text", "tests", "run_id"}:
        assert document[key] == expected[key], key
    assert document["candidate_words"][0] == CANDIDATES[0] and document["candidate_words"][2:] == CANDIDATES[2:]


def test_the_callers_record_keeps_its_own_text() -> None:
    record = _build(evidence_text="synthetic " + HIGH + " evidence")
    rr.write_run_record(_Store(), RUN_ID, record)
    assert record["evidence_text"] == "synthetic " + HIGH + " evidence"


# ---- links and realm counts (links plan, Task 3) ----


def _link(number: int, members: tuple) -> SimpleNamespace:
    return SimpleNamespace(link_id=f"link_{number}", member_ids=members, realms=("culture_and_community", "tech", "markets"),
                           source_count=np.int64(5), mean_link_similarity=np.float32(0.875),
                           members=tuple(CLUSTERS[int(m[-1])] for m in members))


def test_links_and_realm_counts_are_null_without_the_switch() -> None:
    record = _build()
    assert record["links"] is None and record["realm_counts"] is None


def test_the_record_carries_each_link_and_the_realm_counts() -> None:
    counts = {"intake": {"tech": np.int64(4), "none": 1},
              "candidates": {"cluster_0": {"tech": 2}, "link_1": {"tech": 2, "culture_and_community": 3}}}
    record = _build(links=[_link(1, ("cluster_0", "cluster_5", "cluster_3"))], realm_counts=counts)
    assert record["links"] == [{
        "link_id": "link_1", "member_ids": ["cluster_0", "cluster_5", "cluster_3"],
        "realms": ["culture_and_community", "tech", "markets"], "source_count": 5,
        "mean_link_similarity": 0.875,
    }]
    assert type(record["links"][0]["source_count"]) is int
    assert record["realm_counts"] == {"intake": {"tech": 4, "none": 1},
                                      "candidates": {"cluster_0": {"tech": 2},
                                                     "link_1": {"tech": 2, "culture_and_community": 3}}}
    json.dumps(record, allow_nan=False)


def test_a_night_with_the_switch_on_and_no_links_records_an_empty_list() -> None:
    record = _build(links=[], realm_counts={"intake": {}, "candidates": {}})
    assert record["links"] == [] and record["realm_counts"] == {"intake": {}, "candidates": {}}



@pytest.mark.parametrize("value", [None, True, False])
def test_the_record_says_whether_the_chosen_candidate_was_mapped(value) -> None:
    record = _build(chosen_mapped=value)
    assert record["chosen_mapped"] is value
    assert _build()["chosen_mapped"] is None



def test_the_reflector_candidate_is_null_unless_given_and_kept_as_plain_values() -> None:
    assert _build()["reflector_candidate"] is None
    found = {"pointer_run_id": "daily_run:2026-11-21:777:1", "matched_cluster_id": "cluster_7", "overlap": 0.75,
             "already_shown": False, "offered": True}
    record = _build(reflector_candidate=found)
    assert record["reflector_candidate"] == found and record["reflector_candidate"] is not found
