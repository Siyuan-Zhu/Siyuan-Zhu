"""Task 3: the candidate words and evidence kept off the public commission, and the record written by the edition."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.state_store import JsonFileStateStore
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import model_choice as mc
from src.zeitgeist.adk_pipeline import DirectorCommission, commission_daily_zeitgeist
from src.zeitgeist.publication_commit import publish_bundle as _real_publish
from src.zeitgeist.understanding import render_evidence_packet
from tests.test_understanding_pipeline import HISTORY, _clusters, _reply, _run_photo_recovery

ROOT = Path(__file__).resolve().parents[1]
NOW = "2026-11-21T12:00:00Z"
COMMISSION_KEYS = {
    "word", "gloss", "commissioned_thesis", "visual_anchor", "unsplash_queries", "imagen_commission_prompt",
    "flux_3_commission_prompts", "palette_hex", "selected_cluster_id", "attempts_used", "fallback_used",
    "packet_used", "echo_rejections", "slate_count",
}
SLATE = [{"word": f"Synthword{i}", "fit": f"synthetic fit {i}"} for i in range(10)]
# The repository forbids a literal dollar character in test files; build the expression prefix instead.
DOLLAR = chr(36)
RECORD_LINE = "ZEITGEIST_RUN_RECORD: " + DOLLAR + "{{ vars.ZEITGEIST_RUN_RECORD || '0' }}"
UNDERSTANDING_LINE = "ZEITGEIST_UNDERSTANDING_V1: " + DOLLAR + "{{ vars.ZEITGEIST_UNDERSTANDING_V1 || '0' }}"

# sha256 of each prompt as the builders rendered it before this change: not one character may move.
GOLDEN = {
    "legacy": "bee670594934d778f9c85b04ebe405f14cd4e1d620fd2d254a1f945fac609ecb",
    "legacy_retry": "7b0aadc02a960ccaef33f59c07d4927b8c1a4462f6f541e4abbec5b97daa8a83",
    "legacy_empty": "225333f76fa7206247cb60bdbffc89cf342d15beaa1db1de4cc7e1eb29eb4100",
    "packet": "b4989b0b8b7098a9989f2f80d3b8cdc8012f4cb0b40d30ab94fecf9e676ad2f4",
    "packet_slate_retry": "433824acfaaf7f5762046420ca7a9b345b6d69508eec7a7597a5d7bff6cdc8c8",
}
CASES = {
    "legacy": ((_clusters, HISTORY), {}),
    "legacy_retry": ((_clusters, HISTORY, "Synthetic reason"), {}),
    "legacy_empty": ((list, []), {}),
    "packet": ((_clusters, HISTORY), {"evidence_packet": True, "now_utc": NOW}),
    "packet_slate_retry": ((_clusters, HISTORY, "Synthetic reason"),
                           {"evidence_packet": True, "now_utc": NOW, "slate": True}),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _call(builder, name):
    (make, *args), kwargs = CASES[name]
    return builder(make(), *args, **kwargs)


def _block_of(prompt: str) -> str:
    return re.search(r"Top Candidate Clusters:\n(.*)\nReturn JSON with keys", prompt, re.S).group(1)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_the_prompt_text_is_unchanged_to_the_character(name):
    assert _sha(_call(pipeline._build_director_prompt, name)) == GOLDEN[name]
    prompt, block = _call(pipeline._director_prompt_and_block, name)
    assert _sha(prompt) == GOLDEN[name]
    assert block == _block_of(prompt)


def test_the_packet_block_is_the_rendered_packet():
    _, block = pipeline._director_prompt_and_block(_clusters(), HISTORY, evidence_packet=True, now_utc=NOW)
    assert block == render_evidence_packet(_clusters()[:5], now_utc=NOW)


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.prompts.append(contents)
        return SimpleNamespace(text=self.replies[min(len(self.prompts), len(self.replies)) - 1])


def _hydrogen():
    return mc.RunModels(mc.HYDROGEN, mc.HYDROGEN, mc.EMBED_001)


def _packet_commission(replies):
    client = _Client(replies)
    commission = commission_daily_zeitgeist(
        _clusters(), HISTORY, genai_client=client, model_name=mc.HYDROGEN, run_models=_hydrogen(),
        evidence_packet=True, now_utc=NOW,
    )
    return commission, client


@pytest.fixture
def record_on(monkeypatch):
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")


def test_candidate_words_and_evidence_are_kept_off_the_public_commission(record_on):
    commission, client = _packet_commission([_reply(candidate_words=SLATE)])
    assert set(asdict(commission)) == COMMISSION_KEYS
    assert "candidate_words" not in commission.to_dict()
    assert "candidate_words" not in json.dumps(commission.to_dict())
    assert commission.slate_count == 10
    evidence = pipeline.director_evidence(commission)
    assert evidence["candidate_words"] == SLATE
    assert evidence["evidence_text"] == _block_of(client.prompts[-1])
    assert evidence["evidence_text"] == render_evidence_packet(_clusters()[:5], now_utc=NOW)
    assert evidence["prompt_sha256"] == _sha(client.prompts[-1])


def test_the_evidence_comes_from_the_accepted_attempt(record_on):
    commission, client = _packet_commission(["not json", _reply(candidate_words=SLATE)])
    assert len(client.prompts) == 2 and "PREVIOUS ATTEMPT REJECTED" in client.prompts[1]
    assert pipeline.director_evidence(commission)["prompt_sha256"] == _sha(client.prompts[1])


def test_the_older_prompt_path_keeps_its_cluster_lines_and_no_words(record_on):
    prompts = []

    def llm(prompt):
        prompts.append(prompt)
        return _reply()

    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=llm)
    assert set(asdict(commission)) == COMMISSION_KEYS
    evidence = pipeline.director_evidence(commission)
    assert evidence["candidate_words"] == []
    assert evidence["evidence_text"] == _block_of(prompts[0])
    assert evidence["evidence_text"].startswith("- Cluster c_one: commission_score=1.50")
    assert evidence["prompt_sha256"] == _sha(prompts[0])


@pytest.mark.parametrize("switch", [None, "0"])
def test_with_the_switch_off_nothing_is_kept_beside_the_commission(monkeypatch, switch):
    if switch is None:
        monkeypatch.delenv("ZEITGEIST_RUN_RECORD", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_RUN_RECORD", switch)
    commission, _ = _packet_commission([_reply(candidate_words=SLATE)])
    assert commission.slate_count == 10
    assert not hasattr(commission, "_director_evidence")
    assert pipeline.director_evidence(commission) == {"candidate_words": [], "evidence_text": None,
                                                      "prompt_sha256": None}


def test_a_fallback_commission_has_no_evidence(record_on):
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, llm_callable=lambda prompt: "not json")
    assert commission.fallback_used is True
    assert pipeline.director_evidence(commission) == {"candidate_words": [], "evidence_text": None,
                                                      "prompt_sha256": None}
    assert pipeline.director_evidence(None) == {"candidate_words": [], "evidence_text": None,
                                                "prompt_sha256": None}


# ---- the edition writes one record, only once it is published ----


class _RecordingStore(JsonFileStateStore):
    """The real JSON store; run records are kept in memory and their moment is noted among the events."""

    def __init__(self, root, events):
        super().__init__(repo_root=root)
        self.events = events
        self.records = []

    def save_run_record(self, run_id, document):
        self.events.append("save_run_record")
        self.records.append((run_id, document))


class _RaisingStore(_RecordingStore):
    def save_run_record(self, run_id, document):
        self.events.append("save_run_record")
        raise RuntimeError("synthetic store failure")


def _edition_setup(tmp_path, monkeypatch, *, switch, publish_fails=False):
    from src.zeitgeist import publication_commit

    if switch is None:
        monkeypatch.delenv("ZEITGEIST_RUN_RECORD", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_RUN_RECORD", switch)
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    monkeypatch.setenv(mc.DIRECTOR_ENV, mc.HYDROGEN)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config/orbit_sources.json").write_text("{}")
    client = _Client([_reply(selected_cluster_id="cluster_0", candidate_words=SLATE)])
    real_commission = pipeline.commission_daily_zeitgeist
    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist",
                        lambda clusters, **kw: real_commission(clusters, **{**kw, "genai_client": client}))
    events = []

    def publish(*args, **kwargs):
        events.append("publish_bundle:called")
        if publish_fails:
            raise RuntimeError("synthetic publication failure")
        state = _real_publish(*args, **kwargs)
        events.append("publish_bundle:returned")
        return state

    monkeypatch.setattr(publication_commit, "publish_bundle", publish)
    return client, events


def _edition(tmp_path, monkeypatch, store_cls=_RecordingStore, *, switch, publish_fails=False):
    client, events = _edition_setup(tmp_path, monkeypatch, switch=switch, publish_fails=publish_fails)
    store = store_cls(tmp_path, events)
    orchestrator = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    analysis, decision, _ = orchestrator.run("2026-11-21", "blueprint", offline_demo=True)
    return store, client, (analysis.word_of_the_day, analysis.explanation_phrase, decision.selected_mode)


def test_switch_on_writes_one_daily_record_only_after_publication(tmp_path, monkeypatch, capsys):
    store, client, result = _edition(tmp_path, monkeypatch, switch="1")
    out = capsys.readouterr()
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]
    [(run_id, record)] = store.records
    assert run_id == "daily_run:2026-11-21:local:1"
    assert record["kind"] == "daily_run"
    assert record["word"] == result[0] == "Selvedge"
    assert store.load_pulse_state()["published_pulse"]["word"] == record["word"]
    assert record["chosen_cluster_id"] == "cluster_0"
    assert record["candidate_words"] == SLATE
    assert record["evidence_text"] == _block_of(client.prompts[-1])
    assert record["prompt_sha256"] == _sha(client.prompts[-1])
    assert record["prompt_path"] == "packet"
    assert record["revised_by_photo_recovery"] is False
    assert record["main_pass"] == {"word": record["word"], "gloss": record["gloss"]}
    assert record["switches"]["ZEITGEIST_RUN_RECORD"] == "1"
    assert record["models"]["director"] == mc.HYDROGEN and record["models"]["fallback_fired"] is False
    assert 1 <= len(record["groups"]) <= 5 and record["groups"][0]["rank"] == 1
    for text in ("Synthword", "synthetic fit", record["gloss"]):
        assert text not in out.out + out.err


def test_a_failed_publication_writes_no_record(tmp_path, monkeypatch):
    events_store = []
    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        _edition(tmp_path, monkeypatch, lambda root, events: events_store.append(_RecordingStore(root, events))
                 or events_store[0], switch="1", publish_fails=True)
    [store] = events_store
    assert store.events == ["publish_bundle:called"]
    assert store.records == []


@pytest.mark.parametrize("switch", [None, "0", "true"])
def test_switch_off_writes_no_record(tmp_path, monkeypatch, switch):
    store, _, _ = _edition(tmp_path, monkeypatch, switch=switch)
    assert store.events == ["publish_bundle:called", "publish_bundle:returned"]
    assert store.records == []
    assert not (tmp_path / "data" / "run_records").exists()


def test_a_store_that_raises_leaves_the_edition_unchanged(tmp_path, monkeypatch, capsys):
    on_store, _, on = _edition(tmp_path / "on", monkeypatch, _RaisingStore, switch="1")
    on_out = capsys.readouterr()
    off_store, _, off = _edition(tmp_path / "off", monkeypatch, _RecordingStore, switch="0")
    off_out = capsys.readouterr()
    assert on == off
    assert on_store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]
    assert on_store.load_pulse_state()["published_pulse"]["word"] == off_store.load_pulse_state()["published_pulse"]["word"]
    assert not (tmp_path / "on" / "data" / "publication_reconcile.json").exists()
    assert (on_out.out, on_out.err) == (off_out.out, off_out.err)


class _Halt(BaseException):
    """A failure outside Exception, as a library or the runtime might raise."""


class _HaltingStore(_RecordingStore):
    error = _Halt

    def save_run_record(self, run_id, document):
        self.events.append("save_run_record")
        raise self.error("synthetic halt")


class _InterruptingStore(_HaltingStore):
    error = KeyboardInterrupt


def _published_state(store, root):
    return (store.load_pulse_state()["published_pulse"]["word"], store.needs_git_reconcile,
            (root / "data" / "publication_reconcile.json").exists())


def test_a_base_exception_from_the_record_store_leaves_the_published_edition(tmp_path, monkeypatch, capsys):
    clean_store, _, clean = _edition(tmp_path / "clean", monkeypatch, _RecordingStore, switch="1")
    clean_out = capsys.readouterr()
    halted_store, _, halted = _edition(tmp_path / "halted", monkeypatch, _HaltingStore, switch="1")
    halted_out = capsys.readouterr()
    assert halted == clean
    assert halted_store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]
    assert _published_state(halted_store, tmp_path / "halted") == _published_state(clean_store, tmp_path / "clean")
    assert _published_state(halted_store, tmp_path / "halted")[1:] == (False, False)
    assert (halted_out.out, halted_out.err) == (clean_out.out, clean_out.err)


def test_an_interrupt_during_the_record_write_is_not_a_failed_publication(tmp_path, monkeypatch):
    from src.zeitgeist.publication_commit import PublicationMirrorError

    holder = []

    def make(root, events):
        holder.append(_InterruptingStore(root, events))
        return holder[0]

    with pytest.raises(KeyboardInterrupt) as raised:
        _edition(tmp_path, monkeypatch, make, switch="1")
    assert not isinstance(raised.value, PublicationMirrorError)
    [store] = holder
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]
    assert _published_state(store, tmp_path) == ("Selvedge", False, False)


def test_the_json_backend_record_lands_in_the_gitignored_folder(tmp_path, monkeypatch):
    _edition(tmp_path, monkeypatch, lambda root, events: JsonFileStateStore(repo_root=root), switch="1")
    records = JsonFileStateStore(repo_root=tmp_path).load_run_records(kind="daily_run")
    assert len(records) == 1 and records[0]["word"] == "Selvedge"
    assert [p.name for p in (tmp_path / "data" / "run_records").iterdir()] == ["daily_run:2026-11-21:local:1.json"]


def test_a_staged_run_alone_writes_nothing(tmp_path, monkeypatch):
    _, events = _edition_setup(tmp_path, monkeypatch, switch="1")
    store = _RecordingStore(tmp_path, events)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
    assert store.records == [] and events == []
    assert staged._run_record_inputs is not None


def test_a_photo_recovery_revision_is_recorded_with_its_main_pass(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")

    def recover(commission, *, render, revise, history, photo_required):
        revised = DirectorCommission(word="Warp", gloss="threads held taut", commissioned_thesis="Synthetic thesis.",
                                     selected_cluster_id=commission.selected_cluster_id)
        return revised, render(revised), {"recovered": True}

    staged = _run_photo_recovery(tmp_path, monkeypatch, recover=recover)
    store = JsonFileStateStore(repo_root=tmp_path / "records")
    pipeline._write_pending_run_record(store, staged)
    [record] = store.load_run_records()
    assert (record["word"], record["gloss"], record["thesis"]) == ("Warp", "threads held taut", "Synthetic thesis.")
    assert record["revised_by_photo_recovery"] is True
    assert record["main_pass"] == {"word": "Loom", "gloss": "steady weaving"}
    assert record["prompt_path"] == "packet" and record["attempts_used"] == 3


# ---- the workflows pass the switch, off by default ----


def _job_lines(text, job):
    """The stripped lines of one top-level job of a workflow."""
    lines = text.splitlines()
    start = lines.index("  " + job + ":")
    block = []
    for line in lines[start + 1:]:
        if re.fullmatch(r"  [A-Za-z0-9_-]+:", line):
            break
        block.append(line.strip())
    return block


def test_the_daily_workflow_passes_the_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition.count(RECORD_LINE) == 1
    assert edition[edition.index(UNDERSTANDING_LINE) + 1] == RECORD_LINE
    # The read-only preflight writes no record, and the provider canary mirrors it.
    preflight = _job_lines(text, "profile-preflight")
    assert UNDERSTANDING_LINE in preflight and RECORD_LINE not in preflight
    assert [line.strip() for line in text.splitlines()].count(RECORD_LINE) == 1


def test_the_private_run_workflow_passes_the_switch():
    lines = [line.strip() for line in (ROOT / ".github/workflows/private_run.yml").read_text().splitlines()]
    assert lines.count(RECORD_LINE) == 1
    assert lines[lines.index(UNDERSTANDING_LINE) + 1] == RECORD_LINE


# ---- the own-group check runs after the record, only with its switch on ----

import threading  # noqa: E402
import time  # noqa: E402

from src.zeitgeist import own_group  # noqa: E402

OWN_GROUP_LINE = "ZEITGEIST_OWN_GROUP: " + DOLLAR + "{{ vars.ZEITGEIST_OWN_GROUP || '0' }}"


@pytest.fixture(autouse=True)
def _own_group_switch_off_unless_set(monkeypatch):
    """A shell that exports ZEITGEIST_OWN_GROUP must not change these tests; those that want it set it."""
    monkeypatch.delenv("ZEITGEIST_OWN_GROUP", raising=False)


class _MemoryRecordStore(_RecordingStore):
    """Run records kept in memory and read back from there; every load is noted among the events."""

    def load_run_record(self, run_id):
        self.events.append("load_run_record")
        found = [document for saved_id, document in self.records if saved_id == run_id]
        return json.loads(json.dumps(found[-1])) if found else None


def _own_group_switch(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ZEITGEIST_OWN_GROUP", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_OWN_GROUP", value)


def _spy_check(monkeypatch, events, error=None):
    calls = []

    def spy(store, run_id, **kwargs):
        events.append("own_group_check")
        calls.append((run_id, kwargs))
        if error is not None:
            raise error

    monkeypatch.setattr(own_group, "run_own_group_check", spy)
    return calls


def _spied_edition(tmp_path, monkeypatch, *, error=None, store_cls=_MemoryRecordStore):
    holder = []

    def make(root, events):
        holder.append(store_cls(root, events))
        holder.append(_spy_check(monkeypatch, events, error))
        return holder[0]

    _, _, result = _edition(tmp_path, monkeypatch, make, switch="1")
    return holder[0], holder[1], result


def test_switch_on_checks_once_after_the_record_with_the_records_inputs(tmp_path, monkeypatch, capsys):
    _own_group_switch(monkeypatch, "1")
    store, calls, result = _spied_edition(tmp_path, monkeypatch)
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record", "own_group_check"]
    [(run_id, kwargs)] = calls
    [(record_id, record)] = store.records
    assert run_id == record_id == "daily_run:2026-11-21:local:1"
    assert (kwargs["word"], kwargs["gloss"]) == (record["word"], record["gloss"]) and kwargs["word"] == result[0]
    assert kwargs["chosen_cluster_id"] == record["chosen_cluster_id"] == "cluster_0"
    assert [str(c.cluster_id) for c in kwargs["clusters"]][:5] == [g["cluster_id"] for g in record["groups"]]
    assert kwargs["embedding_model"] == record["models"]["embedding"]
    # The offline edition has no live client, so the real check would store "skipped".
    assert kwargs["genai_client"] is None
    assert capsys.readouterr().out.count(record["gloss"]) == 0


def test_switch_on_without_a_client_stores_skipped_and_keeps_the_edition(tmp_path, monkeypatch, capsys):
    _own_group_switch(monkeypatch, "0")
    off_store, _, off = _edition(tmp_path / "off", monkeypatch, _MemoryRecordStore, switch="1")
    off_out = capsys.readouterr()
    _own_group_switch(monkeypatch, "1")
    on_store, _, on = _edition(tmp_path / "on", monkeypatch, _MemoryRecordStore, switch="1")
    on_out = capsys.readouterr()
    assert on == off
    assert on_store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record",
                               "load_run_record", "load_run_record", "save_run_record"]
    saved = on_store.records[-1][1]
    assert saved["tests"] == {"own_group": {"status": "skipped", "reason": "no_client"}}
    assert {k: v for k, v in saved.items() if k != "tests"} == {k: v for k, v in on_store.records[0][1].items()
                                                                if k != "tests"}
    assert _published_state(on_store, tmp_path / "on") == _published_state(off_store, tmp_path / "off")
    assert (on_out.out, on_out.err) == (off_out.out, off_out.err)


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_switch_off_makes_no_new_call(tmp_path, monkeypatch, value):
    _own_group_switch(monkeypatch, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("own-group check called with the switch off")

    monkeypatch.setattr(own_group, "run_own_group_check", forbidden)
    monkeypatch.setattr(pipeline, "_daily_own_group_check", forbidden, raising=False)
    store, _, _ = _edition(tmp_path, monkeypatch, _MemoryRecordStore, switch="1")
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]
    assert len(store.records) == 1 and store.records[0][1]["tests"] == {}


@pytest.mark.parametrize("error", [RuntimeError("Zqsecret failure"), _Halt("Zqsecret halt")])
def test_a_failing_check_leaves_the_published_edition(tmp_path, monkeypatch, capsys, error):
    _own_group_switch(monkeypatch, "0")
    clean_store, _, clean = _edition(tmp_path / "clean", monkeypatch, _MemoryRecordStore, switch="1")
    clean_out = capsys.readouterr()
    _own_group_switch(monkeypatch, "1")
    store, _, result = _spied_edition(tmp_path / "failing", monkeypatch, error=error)
    out = capsys.readouterr()
    assert result == clean
    assert store.events[-2:] == ["save_run_record", "own_group_check"]
    assert _published_state(store, tmp_path / "failing") == _published_state(clean_store, tmp_path / "clean")
    assert _published_state(store, tmp_path / "failing")[1:] == (False, False)
    assert (out.out, out.err) == (clean_out.out, clean_out.err)


def test_an_exception_inside_the_check_never_reaches_the_edition(tmp_path, monkeypatch):
    # Raised in the check's own thread, even an interrupt or an exit stays there.
    _own_group_switch(monkeypatch, "1")
    for error in (KeyboardInterrupt(), SystemExit(1)):
        store, _, result = _spied_edition(tmp_path / type(error).__name__, monkeypatch, error=error)
        assert result[0] == "Selvedge"
        assert store.events[-2:] == ["save_run_record", "own_group_check"]


def test_an_interrupt_while_the_edition_waits_is_not_a_failed_publication(tmp_path, monkeypatch):
    import _thread

    from src.zeitgeist.publication_commit import PublicationMirrorError

    _own_group_switch(monkeypatch, "1")
    release = threading.Event()
    holder = []

    def check(store, run_id, **kwargs):
        holder[0].events.append("own_group_check")
        _thread.interrupt_main()
        release.wait(10)

    def make(root, events):
        holder.append(_MemoryRecordStore(root, events))
        monkeypatch.setattr(own_group, "run_own_group_check", check)
        return holder[0]

    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            _edition(tmp_path, monkeypatch, make, switch="1")
    finally:
        release.set()
    assert not isinstance(raised.value, PublicationMirrorError)
    [store] = holder
    assert store.events[-2:] == ["save_run_record", "own_group_check"]
    assert _published_state(store, tmp_path) == ("Selvedge", False, False)


# ---- what runs after publication has a hard time budget ----


def _clean_edition(tmp_path, monkeypatch, capsys):
    _own_group_switch(monkeypatch, "0")
    store, _, result = _edition(tmp_path / "clean", monkeypatch, _MemoryRecordStore, switch="1")
    out = capsys.readouterr()
    return store, result, out


def _saved_state(store, root):
    state = store.load_pulse_state()
    return state, store.needs_git_reconcile, (root / "data" / "publication_reconcile.json").exists()


def _comparable(state):
    """The saved pulse state without the fields that differ between any two runs (ids and moments)."""
    published = dict(state["published_pulse"])
    return {"word": published.get("word"), "keys": sorted(state), "published_keys": sorted(published),
            "tally": state.get("visitor_mood_tally")}


def test_a_check_past_its_budget_leaves_the_edition_as_a_clean_run_and_stores_nothing(tmp_path, monkeypatch, capsys):
    clean_store, clean, clean_out = _clean_edition(tmp_path, monkeypatch, capsys)
    _own_group_switch(monkeypatch, "1")
    monkeypatch.setattr(own_group, "BUDGET_SECONDS", 0.2)
    release, late = threading.Event(), []
    real = own_group.run_own_group_check

    def stalled(store, run_id, **kwargs):
        release.wait(10)
        # Once the edition has moved on, the abandoned check stores nothing.
        late.append(real(store, run_id, **{**kwargs, "genai_client": None}))

    monkeypatch.setattr(own_group, "run_own_group_check", stalled)
    started = time.monotonic()
    store, _, result = _edition(tmp_path / "slow", monkeypatch, _MemoryRecordStore, switch="1")
    elapsed = time.monotonic() - started
    out = capsys.readouterr()
    events_at_return = list(store.events)
    release.set()
    deadline = time.monotonic() + 5
    while not late and time.monotonic() < deadline:
        time.sleep(0.01)
    assert late == [False]
    assert elapsed < 5
    assert result == clean
    assert events_at_return == clean_store.events == ["publish_bundle:called", "publish_bundle:returned",
                                                      "save_run_record"]
    slow_state, slow_flag, slow_journal = _saved_state(store, tmp_path / "slow")
    clean_state, clean_flag, clean_journal = _saved_state(clean_store, tmp_path / "clean")
    assert _comparable(slow_state) == _comparable(clean_state)
    assert (slow_flag, slow_journal) == (clean_flag, clean_journal) == (False, False)
    assert len(store.records) == 1 and store.records[0][1]["tests"] == {}
    assert "save_run_record" not in store.events[3:]
    assert (out.out, out.err) == (clean_out.out, clean_out.err)


class _EmbedClient:
    """A fake embedding endpoint: one fixed vector per text."""

    def __init__(self):
        self.models = self
        self.requests = []

    def embed_content(self, *, model, contents, config):
        self.requests.append(model)
        vector = [0.0] * 768
        vector[0] = 1.0
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector) for _ in contents])


def test_a_check_inside_its_budget_returns_true_and_its_numbers_are_stored(tmp_path, monkeypatch, capsys):
    clean_store, clean, clean_out = _clean_edition(tmp_path, monkeypatch, capsys)
    _own_group_switch(monkeypatch, "1")
    real_check, real_budget = own_group.run_own_group_check, own_group.run_with_budget
    client, budgets, measured = _EmbedClient(), [], []

    def check(store, run_id, **kwargs):
        measured.append(kwargs)
        return real_check(store, run_id, **{**kwargs, "genai_client": client})

    def budget(fn, seconds):
        budgets.append((seconds, real_budget(fn, seconds)))
        return budgets[-1][1]

    monkeypatch.setattr(own_group, "run_own_group_check", check)
    monkeypatch.setattr(own_group, "run_with_budget", budget)
    store, _, result = _edition(tmp_path / "on", monkeypatch, _MemoryRecordStore, switch="1")
    out = capsys.readouterr()
    assert budgets == [(own_group.BUDGET_SECONDS, True)] and own_group.BUDGET_SECONDS == 45.0
    assert result == clean
    [kwargs] = measured
    clusters = list(kwargs["clusters"])[:5]
    own = next(c for c in clusters if str(c.cluster_id) == str(kwargs["chosen_cluster_id"]))
    numbers = own_group.own_group_numbers([1.0] + [0.0] * 767, own.items, [c.items for c in clusters if c is not own])
    stored = store.records[-1][1]["tests"]["own_group"]
    assert stored["status"] == numbers["status"] == "ok" and client.requests == [kwargs["embedding_model"]]
    assert stored["own"] == pytest.approx(numbers["own"], abs=1e-3)
    assert stored["others"] == pytest.approx(numbers["others"], abs=1e-3)
    assert stored["margin"] == pytest.approx(numbers["margin"], abs=1e-3)
    assert (out.out, out.err) == (clean_out.out, clean_out.err)


def test_with_the_switch_off_no_thread_is_started(tmp_path, monkeypatch):
    _own_group_switch(monkeypatch, "0")
    started = []

    class _Thread(threading.Thread):
        def start(self):
            started.append(self.name)
            super().start()

    monkeypatch.setattr(own_group, "threading", SimpleNamespace(Thread=_Thread, Event=threading.Event))
    budgets = []
    monkeypatch.setattr(own_group, "run_with_budget", lambda *a, **k: budgets.append(a))
    store, _, _ = _edition(tmp_path, monkeypatch, _MemoryRecordStore, switch="1")
    assert started == [] and budgets == []
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]


def test_a_failed_publication_runs_no_check(tmp_path, monkeypatch):
    _own_group_switch(monkeypatch, "1")
    holder = []

    def make(root, events):
        holder.append(_MemoryRecordStore(root, events))
        _spy_check(monkeypatch, events)
        return holder[0]

    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        _edition(tmp_path, monkeypatch, make, switch="1", publish_fails=True)
    assert holder[0].events == ["publish_bundle:called"]


@pytest.mark.parametrize("record, value, kept", [("1", None, False), ("1", "0", False), ("1", "1", True),
                                                 ("0", "1", False), (None, "1", False)])
def test_a_staged_run_keeps_the_check_inputs_only_with_both_switches_on(tmp_path, monkeypatch, record, value, kept):
    _own_group_switch(monkeypatch, value)
    _, events = _edition_setup(tmp_path, monkeypatch, switch=record)
    store = _RecordingStore(tmp_path, events)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
    assert store.records == [] and events == []
    assert (staged._own_group_inputs is not None) is kept


def test_the_daily_workflow_passes_the_own_group_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition[edition.index(RECORD_LINE) + 1] == OWN_GROUP_LINE
    assert OWN_GROUP_LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(OWN_GROUP_LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_workflows_pass_the_own_group_switch(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(OWN_GROUP_LINE) == 1
    assert lines[lines.index(RECORD_LINE) + 1] == OWN_GROUP_LINE


def test_the_provider_canary_does_not_pass_the_own_group_switch():
    assert "ZEITGEIST_OWN_GROUP" not in (ROOT / ".github/workflows/provider_canary.yml").read_text()



# ---- links (links plan, Task 3): with no link the prompt is today's; the record carries the night's links ----

from src.zeitgeist.understanding import LinkCandidate  # noqa: E402


def _links_switch(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ZEITGEIST_LINKS", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_LINKS", value)


@pytest.mark.parametrize("name", ["packet", "packet_slate_retry"])
def test_with_links_on_and_none_found_the_packet_prompt_is_unchanged_to_the_character(name):
    (make, *args), kwargs = CASES[name]
    prompt, block = pipeline._director_prompt_and_block(make(), *args, **kwargs, links=[])
    assert _sha(prompt) == GOLDEN[name]
    assert _sha(pipeline._build_director_prompt(make(), *args, **kwargs, links=[])) == GOLDEN[name]
    packet_args = (make(), args[0], args[1] if len(args) > 1 else None, NOW)
    assert _sha(pipeline._build_packet_director_prompt(*packet_args, slate=kwargs.get("slate", False),
                                                       links=[])) == GOLDEN[name]


def test_with_links_on_and_none_found_the_director_is_sent_todays_prompt():
    plain, plain_client = _packet_commission([_reply(candidate_words=SLATE)])
    client = _Client([_reply(candidate_words=SLATE)])
    commission_daily_zeitgeist(
        _clusters(), HISTORY, genai_client=client, model_name=mc.HYDROGEN, run_models=_hydrogen(),
        evidence_packet=True, now_utc=NOW, links=[],
    )
    assert client.prompts == plain_client.prompts


def _one_link(clusters, realm_map):
    members = tuple(clusters[:3])
    return [LinkCandidate("link_1", tuple(str(c.cluster_id) for c in members), ("civic", "markets", "tech"), 4,
                          0.91, members)]


@pytest.mark.parametrize("value", [None, "0"])
def test_links_off_records_null_links_and_realm_counts(tmp_path, monkeypatch, value):
    _links_switch(monkeypatch, value)
    store, _, _ = _edition(tmp_path, monkeypatch, switch="1")
    [(_, record)] = store.records
    assert record["links"] is None and record["realm_counts"] is None


def test_links_on_with_none_found_sends_todays_prompt_and_records_an_empty_list(tmp_path, monkeypatch):
    _links_switch(monkeypatch, None)
    off_store, _, off = _edition(tmp_path / "off", monkeypatch, switch="1")
    _links_switch(monkeypatch, "1")
    monkeypatch.setattr(pipeline, "link_candidates", lambda clusters, realm_map: [])
    store, _, on = _edition(tmp_path / "on", monkeypatch, switch="1")
    [(_, off_record)] = off_store.records
    [(_, record)] = store.records
    # The prompt the Director answered is today's to the character.
    assert on == off and record["prompt_sha256"] == off_record["prompt_sha256"]
    assert record["evidence_text"] == off_record["evidence_text"]
    assert record["links"] == []
    assert set(record["realm_counts"]) == {"intake", "candidates"}
    assert sum(record["realm_counts"]["intake"].values()) > 0
    assert set(record["realm_counts"]["candidates"]) == {g["cluster_id"] for g in record["groups"]}


def test_links_on_shows_the_links_to_the_director_and_records_them(tmp_path, monkeypatch, capsys):
    _links_switch(monkeypatch, "1")
    monkeypatch.setattr(pipeline, "link_candidates", _one_link)
    store, client, _ = _edition(tmp_path, monkeypatch, switch="1")
    out = capsys.readouterr()
    assert "\n- Link link_1: groups=" in client.prompts[-1]
    [(_, record)] = store.records
    member_ids = [g["cluster_id"] for g in record["groups"]][:3]
    assert record["links"] == [{"link_id": "link_1", "member_ids": member_ids,
                                "realms": ["civic", "markets", "tech"], "source_count": 4,
                                "mean_link_similarity": 0.91}]
    assert "link_1" in record["realm_counts"]["candidates"]
    assert "- Link link_1" in record["evidence_text"]
    assert "Link" not in out.out + out.err


def test_links_on_gives_the_own_group_check_the_links(tmp_path, monkeypatch):
    _links_switch(monkeypatch, "1")
    _own_group_switch(monkeypatch, "1")
    monkeypatch.setattr(pipeline, "link_candidates", _one_link)
    _, calls, _ = _spied_edition(tmp_path, monkeypatch)
    [(_, kwargs)] = calls
    assert [link.link_id for link in kwargs["links"]] == ["link_1"]


def test_links_off_gives_the_own_group_check_no_links(tmp_path, monkeypatch):
    _links_switch(monkeypatch, None)
    _own_group_switch(monkeypatch, "1")
    _, calls, _ = _spied_edition(tmp_path, monkeypatch)
    [(_, kwargs)] = calls
    assert kwargs.get("links") is None


LINKS_LINE = "ZEITGEIST_LINKS: " + DOLLAR + "{{ vars.ZEITGEIST_LINKS || '0' }}"


def test_the_daily_workflow_passes_the_links_switch_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    assert edition[edition.index(OWN_GROUP_LINE) + 1] == LINKS_LINE
    assert LINKS_LINE not in _job_lines(text, "profile-preflight")
    assert [line.strip() for line in text.splitlines()].count(LINKS_LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_workflows_pass_the_links_switch(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(LINKS_LINE) == 1
    assert lines[lines.index(OWN_GROUP_LINE) + 1] == LINKS_LINE


def test_the_provider_canary_does_not_pass_the_links_switch():
    assert "ZEITGEIST_LINKS" not in (ROOT / ".github/workflows/provider_canary.yml").read_text()


def test_a_photo_recovery_revision_is_measured_against_the_published_commissions_candidate(tmp_path, monkeypatch):
    # In the edition, a revision must keep the main pass's candidate (revise_for_photo refuses any other), so
    # the two ids agree. A revision that named another candidate would be about that candidate, and the
    # published word and gloss are measured against the one the published commission is about.
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)

    def recover(commission, *, render, revise, history, photo_required):
        assert commission.selected_cluster_id == "cluster_0"
        revised = DirectorCommission(word="Warp", gloss="threads held taut", commissioned_thesis="Synthetic thesis.",
                                     selected_cluster_id="cluster_1")
        return revised, render(revised), {"recovered": True}

    staged = _run_photo_recovery(tmp_path, monkeypatch, recover=recover)
    inputs = staged._own_group_inputs
    assert staged._run_record_inputs["main_commission"].selected_cluster_id == "cluster_0"
    assert "cluster_1" in [str(cluster.cluster_id) for cluster in inputs["clusters"][:5]]
    calls = []
    monkeypatch.setattr(own_group, "run_own_group_check", lambda store, run_id, **kw: calls.append((run_id, kw)))
    pipeline._daily_own_group_check(object(), **inputs)
    [(run_id, kwargs)] = calls
    assert run_id == "daily_run:2026-11-21:local:1"
    assert (kwargs["word"], kwargs["gloss"]) == ("Warp", "threads held taut")
    assert kwargs["chosen_cluster_id"] == "cluster_1"


def test_a_chosen_link_is_measured_by_the_own_group_check_from_the_published_commission(tmp_path, monkeypatch):
    _links_switch(monkeypatch, "1")
    _own_group_switch(monkeypatch, "1")
    monkeypatch.setattr(pipeline, "link_candidates", _one_link)
    client, events = _edition_setup(tmp_path, monkeypatch, switch="1")
    client.replies = [_reply(selected_cluster_id="link_1", candidate_words=SLATE)]
    store = _MemoryRecordStore(tmp_path, events)
    calls = _spy_check(monkeypatch, events)
    orchestrator = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    orchestrator.run("2026-11-21", "blueprint", offline_demo=True)
    [(_, kwargs)] = calls
    [(_, record)] = store.records
    assert kwargs["chosen_cluster_id"] == record["chosen_cluster_id"] == "link_1"
    assert [link.link_id for link in kwargs["links"]] == ["link_1"]



# ---- the record says when the chosen candidate was mapped ----


def _mapped_edition(tmp_path, monkeypatch, chosen):
    client, events = _edition_setup(tmp_path, monkeypatch, switch="1")
    client.replies = [_reply(selected_cluster_id=chosen, candidate_words=SLATE)]
    store = _RecordingStore(tmp_path, events)
    pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run("2026-11-21", "blueprint", offline_demo=True)
    [(_, record)] = store.records
    return record, client


def test_with_neither_links_nor_the_repair_switch_chosen_mapped_is_null(tmp_path, monkeypatch):
    _links_switch(monkeypatch, None)
    monkeypatch.delenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", raising=False)
    record, _ = _mapped_edition(tmp_path, monkeypatch, "cluster_nowhere")
    assert record["chosen_mapped"] is None and record["chosen_cluster_id"] == "cluster_nowhere"


def test_with_the_repair_switch_an_unshown_id_on_the_accepted_attempt_is_recorded_as_mapped(tmp_path, monkeypatch):
    _links_switch(monkeypatch, None)
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    record, client = _mapped_edition(tmp_path, monkeypatch, "cluster_nowhere")
    assert record["chosen_mapped"] is True
    assert record["chosen_cluster_id"] == record["groups"][0]["cluster_id"]
    assert record["attempts_used"] == len(client.prompts) == 1


@pytest.mark.parametrize("links, repair", [("1", None), (None, "1"), ("1", "1")])
def test_a_shown_choice_is_recorded_as_not_mapped(tmp_path, monkeypatch, links, repair):
    _links_switch(monkeypatch, links)
    if repair is None:
        monkeypatch.delenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", repair)
    record, _ = _mapped_edition(tmp_path, monkeypatch, "cluster_0")
    assert record["chosen_mapped"] is False and record["chosen_cluster_id"] == "cluster_0"



# ---- the record lists only links that were shown; ids are stored stripped ----


def test_links_on_with_the_packet_path_off_records_no_links_as_offered(tmp_path, monkeypatch):
    _links_switch(monkeypatch, "1")
    _own_group_switch(monkeypatch, "1")
    monkeypatch.setattr(pipeline, "link_candidates", _one_link)
    client, events = _edition_setup(tmp_path, monkeypatch, switch="1")
    monkeypatch.delenv("ZEITGEIST_UNDERSTANDING_V1", raising=False)
    store = _MemoryRecordStore(tmp_path, events)
    calls = _spy_check(monkeypatch, events)
    pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store).run("2026-11-21", "blueprint", offline_demo=True)
    [(_, record)] = store.records
    assert "- Link" not in client.prompts[-1]
    assert record["links"] is None and record["prompt_path"] == "legacy"
    assert "link_1" not in record["realm_counts"]["candidates"]
    assert record["chosen_mapped"] is False
    [(_, kwargs)] = calls
    assert kwargs["links"] is None


def test_a_padded_id_is_stored_stripped(tmp_path, monkeypatch):
    _links_switch(monkeypatch, None)
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    record, client = _mapped_edition(tmp_path, monkeypatch, "  cluster_0 ")
    assert record["chosen_cluster_id"] == "cluster_0" and record["chosen_rank"] == 1
    assert record["chosen_mapped"] is False and len(client.prompts) == 1


# ---- the reflector's inputs are stored after the record and the own-group check (SIY-210) ----

from src.zeitgeist import reflector  # noqa: E402


@pytest.fixture(autouse=True)
def _reflector_switch_off_unless_set(monkeypatch):
    """A shell that exports ZEITGEIST_REFLECTOR must not change these tests; those that want it set it."""
    monkeypatch.delenv("ZEITGEIST_REFLECTOR", raising=False)


def _reflector_switch(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ZEITGEIST_REFLECTOR", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_REFLECTOR", value)


def _spy_inputs(monkeypatch, events, error=None):
    calls = []

    def spy(store, run_id, clusters, chosen_cluster_id, embedding_model, extra_group=None):
        events.append("reflector_inputs")
        calls.append((run_id, clusters, chosen_cluster_id, embedding_model))
        if error is not None:
            raise error

    monkeypatch.setattr(reflector, "store_reflector_inputs", spy)
    return calls


def _reflector_edition(tmp_path, monkeypatch, *, error=None, own_group_error=None):
    holder = []

    def make(root, events):
        holder.append(_MemoryRecordStore(root, events))
        _spy_check(monkeypatch, events, own_group_error)
        holder.append(_spy_inputs(monkeypatch, events, error))
        return holder[0]

    _, _, result = _edition(tmp_path, monkeypatch, make, switch="1")
    return holder[0], holder[1], result


def test_reflector_on_stores_the_inputs_once_after_the_record_and_the_own_group_check(tmp_path, monkeypatch,
                                                                                      capsys):
    _own_group_switch(monkeypatch, "1")
    _reflector_switch(monkeypatch, "1")
    store, calls, result = _reflector_edition(tmp_path, monkeypatch)
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record",
                            "own_group_check", "reflector_inputs"]
    [(run_id, clusters, chosen, model)] = calls
    [(record_id, record)] = store.records
    assert run_id == record_id == "daily_run:2026-11-21:local:1"
    assert chosen == record["chosen_cluster_id"] == "cluster_0"
    assert [str(c.cluster_id) for c in clusters][:5] == [g["cluster_id"] for g in record["groups"]]
    assert model == record["models"]["embedding"]
    assert capsys.readouterr().out.count(record["gloss"]) == 0


def test_reflector_on_alone_stores_the_inputs_on_the_record_and_keeps_the_edition(tmp_path, monkeypatch, capsys):
    _reflector_switch(monkeypatch, "0")
    off_store, _, off = _edition(tmp_path / "off", monkeypatch, _MemoryRecordStore, switch="1")
    off_out = capsys.readouterr()
    _reflector_switch(monkeypatch, "1")
    on_store, _, on = _edition(tmp_path / "on", monkeypatch, _MemoryRecordStore, switch="1")
    on_out = capsys.readouterr()
    assert on == off
    assert on_store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record",
                               "load_run_record", "save_run_record"]
    first, saved = on_store.records[0][1], on_store.records[-1][1]
    assert set(saved["tests"]) == {"reflector_inputs"}
    inputs = saved["tests"]["reflector_inputs"]
    assert set(inputs) == {"standing", "chosen", "embedding_model"}
    assert inputs["chosen"]["cluster_id"] == first["chosen_cluster_id"]
    assert inputs["embedding_model"] == first["models"]["embedding"]
    assert {k: v for k, v in saved.items() if k != "tests"} == {k: v for k, v in first.items() if k != "tests"}
    assert _published_state(on_store, tmp_path / "on") == _published_state(off_store, tmp_path / "off")
    assert (on_out.out, on_out.err) == (off_out.out, off_out.err)


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_reflector_off_makes_no_new_call(tmp_path, monkeypatch, value):
    _reflector_switch(monkeypatch, value)

    def forbidden(*args, **kwargs):
        raise AssertionError("reflector inputs stored with the switch off")

    monkeypatch.setattr(reflector, "store_reflector_inputs", forbidden)
    monkeypatch.setattr(reflector, "reflector_inputs", forbidden)
    monkeypatch.setattr(pipeline, "_daily_reflector_inputs", forbidden, raising=False)
    store, _, _ = _edition(tmp_path, monkeypatch, _MemoryRecordStore, switch="1")
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]
    assert len(store.records) == 1 and store.records[0][1]["tests"] == {}


def test_a_failing_own_group_check_still_stores_the_inputs_after_it(tmp_path, monkeypatch):
    _own_group_switch(monkeypatch, "1")
    _reflector_switch(monkeypatch, "1")
    store, calls, _ = _reflector_edition(tmp_path, monkeypatch, own_group_error=RuntimeError("Zqsecret failure"))
    assert store.events[-3:] == ["save_run_record", "own_group_check", "reflector_inputs"] and len(calls) == 1


@pytest.mark.parametrize("error", [RuntimeError("Zqsecret failure"), _Halt("Zqsecret halt")])
def test_a_failing_inputs_step_leaves_the_published_edition(tmp_path, monkeypatch, capsys, error):
    _reflector_switch(monkeypatch, "0")
    clean_store, _, clean = _edition(tmp_path / "clean", monkeypatch, _MemoryRecordStore, switch="1")
    clean_out = capsys.readouterr()
    _reflector_switch(monkeypatch, "1")
    store, _, result = _reflector_edition(tmp_path / "failing", monkeypatch, error=error)
    out = capsys.readouterr()
    assert result == clean
    assert store.events[-2:] == ["save_run_record", "reflector_inputs"]
    assert _published_state(store, tmp_path / "failing") == _published_state(clean_store, tmp_path / "clean")
    assert _published_state(store, tmp_path / "failing")[1:] == (False, False)
    assert (out.out, out.err) == (clean_out.out, clean_out.err)


def test_an_interrupt_or_exit_inside_the_inputs_step_never_reaches_the_edition(tmp_path, monkeypatch):
    # Raised in the budget's own thread, even an interrupt or an exit stays there.
    _reflector_switch(monkeypatch, "1")
    for error in (KeyboardInterrupt(), SystemExit(1)):
        store, _, result = _reflector_edition(tmp_path / type(error).__name__, monkeypatch, error=error)
        assert result[0] == "Selvedge"
        assert store.events[-2:] == ["save_run_record", "reflector_inputs"]


def test_an_interrupt_while_the_edition_waits_on_the_inputs_is_not_a_failed_publication(tmp_path, monkeypatch):
    import _thread

    from src.zeitgeist.publication_commit import PublicationMirrorError

    _reflector_switch(monkeypatch, "1")
    release = threading.Event()
    holder = []

    def inputs(store, *args, **kwargs):
        holder[0].events.append("reflector_inputs")
        _thread.interrupt_main()
        release.wait(10)

    def make(root, events):
        holder.append(_MemoryRecordStore(root, events))
        monkeypatch.setattr(reflector, "store_reflector_inputs", inputs)
        return holder[0]

    try:
        with pytest.raises(KeyboardInterrupt) as raised:
            _edition(tmp_path, monkeypatch, make, switch="1")
    finally:
        release.set()
    assert not isinstance(raised.value, PublicationMirrorError)
    [store] = holder
    assert store.events[-2:] == ["save_run_record", "reflector_inputs"]
    assert _published_state(store, tmp_path) == ("Selvedge", False, False)


def test_a_failed_publication_stores_no_inputs(tmp_path, monkeypatch):
    _reflector_switch(monkeypatch, "1")
    holder = []

    def make(root, events):
        holder.append(_MemoryRecordStore(root, events))
        _spy_inputs(monkeypatch, events)
        return holder[0]

    with pytest.raises(RuntimeError, match="synthetic publication failure"):
        _edition(tmp_path, monkeypatch, make, switch="1", publish_fails=True)
    assert holder[0].events == ["publish_bundle:called"]


@pytest.mark.parametrize("value, kept", [(None, False), ("0", False), ("1", True)])
def test_a_staged_run_keeps_the_reflector_inputs_only_with_the_switch_on(tmp_path, monkeypatch, value, kept):
    _reflector_switch(monkeypatch, value)
    _, events = _edition_setup(tmp_path, monkeypatch, switch="1")
    store = _RecordingStore(tmp_path, events)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
    assert store.records == [] and events == []
    assert (staged._reflector_inputs is not None) is kept


# ---- one post-publication budget covers the own-group check, then the reflector's inputs ----


def test_a_stalled_own_group_check_leaves_no_reflector_inputs_and_the_edition_clean(tmp_path, monkeypatch, capsys):
    clean_store, clean, clean_out = _clean_edition(tmp_path, monkeypatch, capsys)
    _own_group_switch(monkeypatch, "1")
    _reflector_switch(monkeypatch, "1")
    monkeypatch.setattr(own_group, "BUDGET_SECONDS", 0.2)
    release, late = threading.Event(), []
    real_inputs = reflector.store_reflector_inputs

    def stalled(store, run_id, **kwargs):
        release.wait(10)

    def inputs(*args, **kwargs):
        # Reached only once the abandoned check returns; by then the store refuses the save.
        late.append(real_inputs(*args, **kwargs))

    monkeypatch.setattr(own_group, "run_own_group_check", stalled)
    monkeypatch.setattr(reflector, "store_reflector_inputs", inputs)
    started = time.monotonic()
    store, _, result = _edition(tmp_path / "slow", monkeypatch, _MemoryRecordStore, switch="1")
    elapsed = time.monotonic() - started
    out = capsys.readouterr()
    events_at_return = list(store.events)
    release.set()
    deadline = time.monotonic() + 5
    while not late and time.monotonic() < deadline:
        time.sleep(0.01)
    assert late == [False]
    assert elapsed < 5
    assert result == clean
    assert events_at_return == clean_store.events == ["publish_bundle:called", "publish_bundle:returned",
                                                      "save_run_record"]
    slow_state, slow_flag, slow_journal = _saved_state(store, tmp_path / "slow")
    clean_state, clean_flag, clean_journal = _saved_state(clean_store, tmp_path / "clean")
    assert _comparable(slow_state) == _comparable(clean_state)
    assert (slow_flag, slow_journal) == (clean_flag, clean_journal) == (False, False)
    assert len(store.records) == 1 and store.records[0][1]["tests"] == {}
    assert "save_run_record" not in store.events[3:]
    assert (out.out, out.err) == (clean_out.out, clean_out.err)


def test_stalled_inputs_alone_are_held_to_the_same_budget(tmp_path, monkeypatch, capsys):
    clean_store, clean, clean_out = _clean_edition(tmp_path, monkeypatch, capsys)
    _reflector_switch(monkeypatch, "1")
    monkeypatch.setattr(own_group, "BUDGET_SECONDS", 0.2)
    release, late = threading.Event(), []
    real_inputs = reflector.store_reflector_inputs

    def inputs(*args, **kwargs):
        release.wait(10)
        late.append(real_inputs(*args, **kwargs))

    monkeypatch.setattr(reflector, "store_reflector_inputs", inputs)
    started = time.monotonic()
    store, _, result = _edition(tmp_path / "slow", monkeypatch, _MemoryRecordStore, switch="1")
    elapsed = time.monotonic() - started
    release.set()
    deadline = time.monotonic() + 5
    while not late and time.monotonic() < deadline:
        time.sleep(0.01)
    assert late == [False] and elapsed < 5 and result == clean
    assert len(store.records) == 1 and store.records[0][1]["tests"] == {}


def test_a_clean_run_stores_both_inside_one_budget(tmp_path, monkeypatch, capsys):
    clean_store, clean, clean_out = _clean_edition(tmp_path, monkeypatch, capsys)
    _own_group_switch(monkeypatch, "1")
    _reflector_switch(monkeypatch, "1")
    real_check, real_budget = own_group.run_own_group_check, own_group.run_with_budget
    client, budgets, order = _EmbedClient(), [], []
    real_inputs = reflector.store_reflector_inputs

    def check(store, run_id, **kwargs):
        order.append("own_group")
        return real_check(store, run_id, **{**kwargs, "genai_client": client})

    def inputs(*args, **kwargs):
        order.append("reflector_inputs")
        return real_inputs(*args, **kwargs)

    def budget(fn, seconds):
        budgets.append((seconds, real_budget(fn, seconds)))
        return budgets[-1][1]

    monkeypatch.setattr(own_group, "run_own_group_check", check)
    monkeypatch.setattr(reflector, "store_reflector_inputs", inputs)
    monkeypatch.setattr(own_group, "run_with_budget", budget)
    store, _, result = _edition(tmp_path / "on", monkeypatch, _MemoryRecordStore, switch="1")
    out = capsys.readouterr()
    assert budgets == [(own_group.BUDGET_SECONDS, True)] and own_group.BUDGET_SECONDS == 45.0
    assert order == ["own_group", "reflector_inputs"]
    assert result == clean
    tests = store.records[-1][1]["tests"]
    assert set(tests) == {"own_group", "reflector_inputs"} and tests["own_group"]["status"] == "ok"
    assert tests["reflector_inputs"]["chosen"]["cluster_id"] == store.records[0][1]["chosen_cluster_id"]
    assert (out.out, out.err) == (clean_out.out, clean_out.err)


@pytest.mark.parametrize("own, refl, budgets_expected", [("0", "0", 0), ("1", "0", 1), ("0", "1", 1), ("1", "1", 1)])
def test_one_budget_at_most_and_none_with_both_switches_off(tmp_path, monkeypatch, own, refl, budgets_expected):
    _own_group_switch(monkeypatch, own)
    _reflector_switch(monkeypatch, refl)
    started = []

    class _Thread(threading.Thread):
        def start(self):
            started.append(self.name)
            super().start()

    monkeypatch.setattr(own_group, "threading", SimpleNamespace(Thread=_Thread, Event=threading.Event))
    store, _, _ = _edition(tmp_path, monkeypatch, _MemoryRecordStore, switch="1")
    assert len(started) == budgets_expected


def test_the_inputs_name_the_published_commissions_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    _reflector_switch(monkeypatch, "1")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)

    def recover(commission, *, render, revise, history, photo_required):
        assert commission.selected_cluster_id == "cluster_0"
        revised = DirectorCommission(word="Warp", gloss="threads held taut", commissioned_thesis="Synthetic thesis.",
                                     selected_cluster_id="cluster_1")
        return revised, render(revised), {"recovered": True}

    staged = _run_photo_recovery(tmp_path, monkeypatch, recover=recover)
    inputs = staged._reflector_inputs
    assert "cluster_1" in [str(cluster.cluster_id) for cluster in inputs["clusters"][:5]]
    calls = []
    monkeypatch.setattr(reflector, "store_reflector_inputs", lambda *args, **kwargs: calls.append(args))
    pipeline._daily_reflector_inputs(object(), **inputs)
    [(_, run_id, _, chosen, _)] = calls
    assert run_id == "daily_run:2026-11-21:local:1" and chosen == "cluster_1"


# ---- the reflector's switch and model reach the edition, the private run and the comparison ----

REFLECTOR_LINE = "ZEITGEIST_REFLECTOR: " + DOLLAR + "{{ vars.ZEITGEIST_REFLECTOR || '0' }}"
# The reflector's two lines follow the links switches, which follow the own-group line.
CHOSEN_FIELDS_LINE = "ZEITGEIST_CHOSEN_GROUP_FIELDS: " + DOLLAR + "{{ vars.ZEITGEIST_CHOSEN_GROUP_FIELDS || '0' }}"
REFLECTOR_MODEL_LINE = "ZEITGEIST_REFLECTOR_MODEL: " + DOLLAR + "{{ vars.ZEITGEIST_REFLECTOR_MODEL || '' }}"
CANDIDATE_LINE = "ZEITGEIST_REFLECTOR_CANDIDATE: " + DOLLAR + "{{ vars.ZEITGEIST_REFLECTOR_CANDIDATE || '0' }}"


def test_the_daily_workflow_passes_the_reflector_switch_and_model_to_the_edition_job_only():
    text = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    edition = _job_lines(text, "synthesize-pulse")
    at = edition.index(CHOSEN_FIELDS_LINE)
    assert edition[at + 1:at + 4] == [REFLECTOR_LINE, REFLECTOR_MODEL_LINE, CANDIDATE_LINE]
    preflight = _job_lines(text, "profile-preflight")
    assert REFLECTOR_LINE not in preflight and REFLECTOR_MODEL_LINE not in preflight and CANDIDATE_LINE not in preflight
    stripped = [line.strip() for line in text.splitlines()]
    assert stripped.count(REFLECTOR_LINE) == 1 and stripped.count(REFLECTOR_MODEL_LINE) == 1
    assert stripped.count(CANDIDATE_LINE) == 1


@pytest.mark.parametrize("name", ["private_run.yml", "comparison_run.yml"])
def test_the_private_and_comparison_workflows_pass_the_reflector_switch_and_model(name):
    lines = [line.strip() for line in (ROOT / ".github/workflows" / name).read_text().splitlines()]
    assert lines.count(REFLECTOR_LINE) == 1 and lines.count(REFLECTOR_MODEL_LINE) == 1
    assert lines.count(CANDIDATE_LINE) == 1
    at = lines.index(CHOSEN_FIELDS_LINE)
    assert lines[at + 1:at + 4] == [REFLECTOR_LINE, REFLECTOR_MODEL_LINE, CANDIDATE_LINE]


def test_the_provider_canary_does_not_pass_the_reflector_switch_or_model():
    assert "ZEITGEIST_REFLECTOR" not in (ROOT / ".github/workflows/provider_canary.yml").read_text()


# ---- the edition stores the reflector's inputs only when its record was written ----


class _FailingSaveStore(_MemoryRecordStore):
    """An older record of the same id is already stored; tonight's record write fails."""

    def __init__(self, root, events):
        super().__init__(root, events)
        self.records.append(("daily_run:2026-11-21:local:1", {"tests": {}, "word": "Zqsecret older"}))

    def save_run_record(self, run_id, document):
        self.events.append("save_run_record")
        raise RuntimeError("synthetic store failure")


def _forbid_inputs(monkeypatch):
    """Notes every call: an error raised in the budget's thread would stay there unseen."""
    called = []
    monkeypatch.setattr(reflector, "store_reflector_inputs", lambda *a, **k: called.append("store_reflector_inputs"))
    monkeypatch.setattr(pipeline, "_daily_reflector_inputs", lambda *a, **k: called.append("_daily_reflector_inputs"))
    return called


def test_a_failed_record_write_leaves_no_reflector_store_access(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    _reflector_switch(monkeypatch, "1")
    called = _forbid_inputs(monkeypatch)
    store, _, result = _edition(tmp_path, monkeypatch, _FailingSaveStore, switch="1")
    assert result[0] == "Selvedge" and called == []
    assert store.events == ["publish_bundle:called", "publish_bundle:returned", "save_run_record"]


@pytest.mark.parametrize("record", [None, "0"])
def test_with_the_record_switch_off_the_reflector_keeps_and_stores_nothing(tmp_path, monkeypatch, record):
    _reflector_switch(monkeypatch, "1")
    called = _forbid_inputs(monkeypatch)
    store, _, _ = _edition(tmp_path, monkeypatch, _MemoryRecordStore, switch=record)
    assert called == []
    assert store.events == ["publish_bundle:called", "publish_bundle:returned"]


@pytest.mark.parametrize("record, value, kept", [("1", None, False), ("1", "0", False), ("1", "1", True),
                                                 ("0", "1", False), (None, "1", False)])
def test_a_staged_run_keeps_the_reflector_inputs_only_with_both_switches_on(tmp_path, monkeypatch, record, value,
                                                                           kept):
    _reflector_switch(monkeypatch, value)
    _, events = _edition_setup(tmp_path, monkeypatch, switch=record)
    store = _RecordingStore(tmp_path, events)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
    assert (staged._reflector_inputs is not None) is kept


def test_the_pending_record_write_reports_whether_it_wrote(tmp_path, monkeypatch):
    _, events = _edition_setup(tmp_path, monkeypatch, switch="1")
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=_RecordingStore(tmp_path, events))
    staged._run_staged("2026-11-21", "blueprint", offline_demo=True)
    assert pipeline._write_pending_run_record(_MemoryRecordStore(tmp_path, []), staged) is True
    assert pipeline._write_pending_run_record(_FailingSaveStore(tmp_path, []), staged) is False
    assert pipeline._write_pending_run_record(_MemoryRecordStore(tmp_path, []), object()) is False


def test_a_chosen_link_stores_no_chosen_group_for_the_reflector(tmp_path, monkeypatch):
    # A link is not one of the five groups: the reflector has no control block for it.
    _links_switch(monkeypatch, "1")
    _reflector_switch(monkeypatch, "1")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    monkeypatch.setattr(pipeline, "link_candidates", _one_link)
    client, events = _edition_setup(tmp_path, monkeypatch, switch="1")
    client.replies = [_reply(selected_cluster_id="link_1", candidate_words=SLATE)]
    store = _MemoryRecordStore(tmp_path, events)
    orchestrator = pipeline.ZeitgeistADKOrchestrator(tmp_path, state_store=store)
    orchestrator.run("2026-11-21", "blueprint", offline_demo=True)
    run_id, record = store.records[-1]
    assert run_id == "daily_run:2026-11-21:local:1" and record["chosen_cluster_id"] == "link_1"
    inputs = record["tests"]["reflector_inputs"]
    assert inputs["chosen"] is None
