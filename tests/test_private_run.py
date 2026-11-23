"""Task C9: a production private run of the Director and the photo path. Synthetic data only; no network."""

from __future__ import annotations

import json
import re
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from scripts import private_run
from src.zeitgeist import model_choice as mc
from src.zeitgeist import visual_router as vr
from src.zeitgeist.collectors import pack_f16_base64
from src.zeitgeist.spend_guard import SpendBlocked, refused_before_dispatch
from tests.test_understanding_pipeline import _reply

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/private_run.yml"
DAILY = ROOT / ".github/workflows/daily_zeitgeist.yml"
NOW = datetime(2026, 11, 21, 12, tzinfo=timezone.utc)
TEXT_MODEL = r"(gemini-3\.8-flash|gemini-4-hydrogen)"
FORMATS = [
    re.compile(r"\[model_choice\] director=" + TEXT_MODEL + " judge=" + TEXT_MODEL
               + r" embedding=(gemini-embedding-001|gemini-embedding-2) fallback=[01] unrecognized=[01] slate=\d+"),
    re.compile(r"\[private_run\] director_ok=[01] attempts=\d+ packet=[01] fallback=[01] items=\d+ clusters=\d+"),
    re.compile(r"\[private_run\] photo_ok=[01] candidates=\d+ safe=\d+ best_meaning=\d\.\d{3} best_composite=\d\.\d{3}"),
    re.compile(r"\[private_run\] failed_stage=[1-4]"),
]
# Every synthetic private string: item text, keywords, the word, the gloss and the reading.
PRIVATE = ("Zqtitle", "Zqsummary", "zqtitle", "zqsummary", "Selvedge", "finished edge", "Quiet craft",
           "linen", "Zqsecret")
SCORES = {"semantic_alignment": 0.9, "compositional_clarity": 0.9, "editorial_memorability": 0.9, "safety_pass": True}


@pytest.fixture(autouse=True)
def _record_switch_off_unless_set(monkeypatch):
    """A shell that exports ZEITGEIST_RUN_RECORD must not change these tests; those that want it set it."""
    monkeypatch.delenv("ZEITGEIST_RUN_RECORD", raising=False)


def _rows():
    rng = np.random.default_rng(7)
    rows = {}
    for group in (0, 1):
        for index in range(4):
            vector = np.zeros(768)
            vector[group] = 1.0
            vector += rng.normal(0, 0.02, 768)
            vector /= np.linalg.norm(vector)
            key = f"h{group}{index}"
            rows[key] = {"url_hash": key, "source_id": f"src_{group}_{index}", "bucket": "MACRO", "tier": "M1",
                         "title": f"Zqtitle{group}{index} story", "summary": f"Zqsummary{group}{index} words",
                         "url": f"https://example.test/{key}", "published_at": "2026-11-21T06:00:00Z",
                         "embedding_f16": pack_f16_base64(vector)}
    return rows


class ReadOnlyStore:
    """Only the two reads exist; any other attribute (every save_* and prune_*) fails the test."""

    def __init__(self, rows=None, history=(), fail=None):
        self.rows = _rows() if rows is None else rows
        self.history = list(history)
        self.fail = fail
        self.reads = []

    def load_ingestion_cache(self, now_utc):
        self.reads.append(("load_ingestion_cache", now_utc))
        if self.fail is not None:
            raise self.fail
        return {key: dict(row) for key, row in self.rows.items()}

    def load_pulse_feed_history(self, days=30):
        self.reads.append(("load_pulse_feed_history", days))
        return list(self.history)

    def __getattr__(self, name):
        raise AssertionError("private run touched store." + name)


class Director:
    """Fake genai client for the Director; records every model it is sent."""

    def __init__(self, reply):
        self.calls = []
        self.prompts = []
        self.reply = reply
        self.models = self

    def generate_content(self, *, model, contents, config):
        self.calls.append(model)
        self.prompts.append(contents)
        reply = self.reply(model) if callable(self.reply) else self.reply
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(text=reply)


def _refusal(pre=True):
    error = SpendBlocked("Unregistered paid SDK method")
    if pre:
        error.refused_before_dispatch = True
    return error


def _jpeg(size=(1200, 800)):
    stream = BytesIO()
    Image.new("RGB", size, (90, 110, 120)).save(stream, format="JPEG")
    return stream.getvalue()


def _photo(photo_id):
    return {"id": photo_id, "urls": {"small": f"https://images.unsplash.com/{photo_id}_small",
            "raw": f"https://images.unsplash.com/{photo_id}_raw?ixid=keep"},
            "links": {"html": f"https://unsplash.com/photos/{photo_id}",
                      "download_location": f"https://api.unsplash.com/photos/{photo_id}/download"},
            "user": {"name": "Synthetic Photographer", "links": {"html": "https://unsplash.com/@synthetic"}},
            "exif": {"make": "Synthetic Camera", "model": "Fixture"}}


def _photo_world(monkeypatch, *, scores=SCORES, refuse_judge=None, critic_raises=None):
    """Fake Unsplash (search, thumbnails, detail, winner) and a fake critic; no request leaves the process."""
    requests, critic = [], []
    queries = {}

    def get(client, url, *, params=None, **kwargs):
        requests.append(url)
        if url.endswith("/search/photos"):
            slot = queries.setdefault(params["query"], len(queries))
            return SimpleNamespace(status_code=200, json=lambda: {"results": [_photo(f"p{slot}{i}") for i in range(3)]})
        if url.startswith("https://api.unsplash.com/photos/"):
            photo_id = url.rsplit("/", 1)[-1]
            return SimpleNamespace(status_code=200, json=lambda: _photo(photo_id))
        if url.startswith("https://api.unsplash.com/"):
            raise AssertionError("unexpected Unsplash call " + url)
        return SimpleNamespace(status_code=200, content=_jpeg())

    def generate(*, model, contents, config):
        critic.append(model)
        if critic_raises is not None:
            critic_raises()
        if refuse_judge is not None and model == mc.HYDROGEN:
            raise _refusal(refuse_judge)
        count = sum(isinstance(part, str) and part.startswith("Candidate index") for part in contents)
        return SimpleNamespace(text=json.dumps({"candidates": [dict(scores, index=i) for i in range(count)]}),
                               candidates=[])

    monkeypatch.setattr("httpx.Client.get", get)
    monkeypatch.setattr("httpx.Client.post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no POST")))
    monkeypatch.setattr("google.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    monkeypatch.setattr(vr, "_ping_unsplash_download_location",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download tracking")))
    return requests, critic


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (mc.DIRECTOR_ENV, mc.JUDGE_ENV, mc.EMBEDDING_ENV, "ZEITGEIST_BEST_SAFE_PHOTO_DATE", "ZEITGEIST_OFFLINE_DEMO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("ZEITGEIST_LIVE_UNSPLASH", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "synthetic-key")


def _models(director=mc.HYDROGEN, judge=mc.HYDROGEN):
    return mc.resolve_run_models({}, {mc.DIRECTOR_ENV: director, mc.JUDGE_ENV: judge})


def _run(store=None, reply=None, run_models=None):
    client = Director(_reply(selected_cluster_id="cluster_0") if reply is None else reply)
    result = private_run.run_private(store or ReadOnlyStore(), genai_client=client, now=NOW,
                                     run_models=run_models or _models())
    return result, client


def _assert_public(lines):
    assert lines
    for line in lines:
        assert any(pattern.fullmatch(line) for pattern in FORMATS), line
        assert not any(text in line for text in PRIVATE), line


def test_happy_run_prints_only_known_numeric_lines_on_the_chosen_models(monkeypatch):
    _, critic = _photo_world(monkeypatch)
    store = ReadOnlyStore()
    result, director = _run(store)
    _assert_public(result.lines)
    assert result.lines[0] == ("[model_choice] director=gemini-4-hydrogen judge=gemini-4-hydrogen "
                               "embedding=gemini-embedding-001 fallback=0 unrecognized=0 slate=0")
    assert re.fullmatch(r"\[private_run\] director_ok=1 attempts=1 packet=1 fallback=0 items=8 clusters=\d+",
                        result.lines[1])
    assert result.lines[2] == ("[private_run] photo_ok=1 candidates=9 safe=9 best_meaning=0.900 "
                               "best_composite=0.900")
    assert len(result.lines) == 3
    assert director.calls == [mc.HYDROGEN] and critic == [mc.HYDROGEN]
    assert result.exit_code == 0
    assert [read[0] for read in store.reads] == ["load_ingestion_cache", "load_pulse_feed_history"]
    assert store.reads[0][1] == NOW and store.reads[1][1] == 30


def test_scored_candidates_stay_in_memory_with_their_thumbnails_and_scores(monkeypatch):
    _photo_world(monkeypatch)
    result, _ = _run()
    assert len(result.scored_candidates) == 9
    for row in result.scored_candidates:
        assert isinstance(row["thumb_bytes"], bytes) and row["thumb_bytes"]
        assert row["semantic_alignment"] == 0.9 and row["safety_pass"] is True
        assert row["composite_score"] == pytest.approx(0.9)
    assert result.commission.word == "Selvedge"


def test_photo_stage_renders_outside_the_repository_and_leaves_nothing(monkeypatch):
    _photo_world(monkeypatch)
    roots = []
    real = vr.select_and_render_pulse_assets

    def spy(**kwargs):
        roots.append(Path(kwargs["repo_root"]))
        assert kwargs["requested_medium"] == "photo" and kwargs["judge_model"] == mc.HYDROGEN
        return real(**kwargs)

    monkeypatch.setattr(vr, "select_and_render_pulse_assets", spy)
    result, _ = _run()
    assert result.exit_code == 0 and len(roots) == 1
    assert ROOT not in roots[0].resolve().parents and not roots[0].exists()


def test_no_photo_passing_reports_photo_ok_zero_and_keeps_the_scored_rows(monkeypatch):
    _photo_world(monkeypatch, scores=dict(SCORES, semantic_alignment=0.4))
    result, _ = _run()
    _assert_public(result.lines)
    assert result.lines[2] == ("[private_run] photo_ok=0 candidates=9 safe=9 best_meaning=0.400 "
                               "best_composite=0.650")
    assert len(result.scored_candidates) == 9
    assert result.exit_code == 0


def test_director_fallback_exits_3_without_a_photo_stage(monkeypatch):
    _, critic = _photo_world(monkeypatch)
    result, director = _run(reply="not json")
    _assert_public(result.lines)
    assert result.lines[1].startswith("[private_run] director_ok=0 ")
    assert len(result.lines) == 2 and critic == [] and result.scored_candidates == []
    assert result.exit_code == 3


def test_refused_hydrogen_director_moves_to_flash_and_exits_3(monkeypatch):
    _, critic = _photo_world(monkeypatch)
    reply = lambda model: _refusal() if model == mc.HYDROGEN else _reply(selected_cluster_id="cluster_0")
    result, director = _run(reply=reply)
    _assert_public(result.lines)
    assert director.calls == [mc.HYDROGEN, mc.FLASH]
    assert "fallback=1" in result.lines[0] and " fallback=1 " in result.lines[1]
    assert result.exit_code == 3


def test_a_director_fallback_to_flash_skips_the_photo_stage(monkeypatch):
    _, critic = _photo_world(monkeypatch)
    reply = lambda model: _refusal() if model == mc.HYDROGEN else _reply(selected_cluster_id="cluster_0")
    result, _ = _run(reply=reply)
    assert critic == [] and result.scored_candidates == []
    assert len(result.lines) == 2 and result.lines[1].startswith("[private_run] director_ok=1 ")
    assert result.exit_code == 3


# ---- the packet follows the edition's own switch ----


PACKET_MARKER = "The evidence lines are private working material."


def test_switch_on_sends_the_stage0_packet(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    result, director = _run()
    assert PACKET_MARKER in director.prompts[0]
    assert " packet=1 " in result.lines[1]


@pytest.mark.parametrize("value", [None, "0"])
def test_switch_off_sends_the_legacy_prompt_like_the_edition(monkeypatch, value):
    _photo_world(monkeypatch)
    if value is None:
        monkeypatch.delenv("ZEITGEIST_UNDERSTANDING_V1", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", value)
    result, director = _run()
    assert PACKET_MARKER not in director.prompts[0]
    assert "Select the most compelling topical cluster" in director.prompts[0]
    assert " packet=0 " in result.lines[1]
    assert result.exit_code == 0


def test_judge_refusal_in_the_photo_stage_prints_a_second_model_choice_line(monkeypatch):
    _, critic = _photo_world(monkeypatch, refuse_judge=True)
    result, _ = _run(run_models=_models(director=mc.FLASH, judge=mc.HYDROGEN))
    _assert_public(result.lines)
    assert critic == [mc.HYDROGEN, mc.FLASH]
    assert result.lines[-1] == ("[model_choice] director=gemini-3.8-flash judge=gemini-3.8-flash "
                                "embedding=gemini-embedding-001 fallback=1 unrecognized=0 slate=0")
    assert result.exit_code == 3


def test_a_block_after_dispatch_is_a_failed_director_stage(monkeypatch):
    _, critic = _photo_world(monkeypatch)
    result, director = _run(reply=lambda model: _refusal(pre=False))
    assert result.lines[-1] == "[private_run] failed_stage=3"
    assert director.calls == [mc.HYDROGEN] and critic == []
    assert result.exit_code == 3


def test_a_store_failure_is_a_failed_read_stage_without_its_text(monkeypatch):
    _photo_world(monkeypatch)
    result, director = _run(store=ReadOnlyStore(fail=RuntimeError("Zqsecret")))
    assert result.lines == ["[private_run] failed_stage=2"] and director.calls == []
    assert result.exit_code == 3


def _main(monkeypatch, capsys, store, client):
    monkeypatch.setattr(private_run, "_store", lambda: store)
    monkeypatch.setattr(private_run, "_genai_client", lambda: client)
    code = private_run.main([])
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.out + out.err


def test_main_prints_only_the_lines_and_hides_everything_else(monkeypatch, capsys):
    _photo_world(monkeypatch)
    monkeypatch.setenv(mc.DIRECTOR_ENV, mc.HYDROGEN)

    def noisy(model):
        print("Zqtitle00 story leaked by a library")
        import logging
        logging.getLogger("zeitgeist").warning("Zqsummary00 words leaked by a logger")
        return _reply(selected_cluster_id="cluster_0")

    code, lines, everything = _main(monkeypatch, capsys, ReadOnlyStore(), Director(noisy))
    _assert_public(lines)
    assert len(lines) == 3 and code == 0
    assert not any(text in everything for text in PRIVATE)


def test_main_director_fallback_returns_3(monkeypatch, capsys):
    _photo_world(monkeypatch)
    code, lines, _ = _main(monkeypatch, capsys, ReadOnlyStore(), Director("not json"))
    _assert_public(lines)
    assert code == 3


def test_main_setup_failure_returns_3_with_a_numeric_line(monkeypatch, capsys):
    def broken():
        raise RuntimeError("Zqsecret credential text")

    monkeypatch.setattr(private_run, "_store", broken)
    monkeypatch.setattr(private_run, "_genai_client", lambda: None)
    code = private_run.main([])
    out = capsys.readouterr()
    assert out.out.splitlines() == ["[private_run] failed_stage=1"] and code == 3
    assert "Zqsecret" not in out.out + out.err


def test_main_takes_no_embedding_option():
    with pytest.raises(SystemExit):
        private_run.main(["--measure-embedding-2"])


# ---- the visual router hook that exposes the scored candidates ----


def test_renderer_hands_its_scored_candidates_to_the_caller_even_when_none_passes(tmp_path, monkeypatch):
    monkeypatch.setenv("ZEITGEIST_LIVE_UNSPLASH", "0")
    out = []
    rows = [dict(SCORES, id="a", semantic_alignment=0.4, thumb_bytes=_jpeg((8, 8)))]
    with pytest.raises(vr.PhotoSelectionUnavailable):
        vr.select_and_render_pulse_assets({"word": "W", "gloss": "G"}, repo_root=tmp_path, candidates=rows,
                                          requested_medium="photo", photo_history_30d=[], scored_candidates_out=out)
    assert len(out) == 1 and out[0]["thumb_bytes"] == rows[0]["thumb_bytes"]
    assert out[0]["composite_score"] == pytest.approx(0.65)


# ---- the workflow ----


def _step_env(text, step_name):
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "- name: " + step_name)
    env_at = next(i for i in range(start, len(lines)) if lines[i].strip() == "env:")
    indent = len(lines[env_at]) - len(lines[env_at].lstrip())
    block = []
    for line in lines[env_at + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        if line.strip():
            block.append(line.strip())
    return block


def test_workflow_is_a_dispatch_only_read_only_private_run():
    text = WORKFLOW.read_text()
    assert "workflow_dispatch:" in text and "schedule:" not in text and "inputs:" not in text
    assert "concurrency:\n  group: readme-state-writer\n  cancel-in-progress: false\n  queue: max\n" in text
    assert text.count("concurrency:") == 1
    assert "contents: read" in text and "id-token: write" in text
    for banned in ("upload-artifact", "contents: write", "publish_readme", "git push", "git commit",
                   "refresh_daily_digest", "--measure-embedding-2"):
        assert banned not in text, banned


def test_workflow_runs_maintenance_then_the_private_run():
    text = WORKFLOW.read_text()
    maintenance = "python3 scripts/maintain_spend_control.py --apply --private-report /tmp/private-spend-maintenance/evidence.json"
    assert text.count(maintenance) == 1 and text.count("python3 scripts/private_run.py") == 1
    assert text.index(maintenance) < text.index("python3 scripts/private_run.py")
    assert "pip install -r requirements.txt" in text and "actions/checkout@v4" in text and "actions/setup-python@v5" in text


def test_workflow_env_mirrors_the_daily_synthesis_step_and_its_pin():
    private = _step_env(WORKFLOW.read_text(), "Private run of the Director and the photo path")
    daily = _step_env(DAILY.read_text(), "Run Daily Zeitgeist Pipeline & Rebuild README")
    assert private == daily
    for name in (mc.DIRECTOR_ENV, mc.JUDGE_ENV, mc.EMBEDDING_ENV, "ZEITGEIST_UNDERSTANDING_V1"):
        assert any(line.startswith(name + ":") for line in private), name
    pin = re.compile(r"SPEND_POLICY_SHA256:\s*['\"]([a-f0-9]{64})['\"]")
    assert pin.findall(WORKFLOW.read_text()) == pin.findall(DAILY.read_text())[:1]


# ---- an interrupt at any stage prints one numeric line and no traceback ----


MARKER = "Zqsecret interrupt marker"


def _interrupt(*args, **kwargs):
    try:
        raise RuntimeError(MARKER)
    except RuntimeError:
        raise KeyboardInterrupt(MARKER)


class InterruptingStore(ReadOnlyStore):
    def load_ingestion_cache(self, now_utc):
        _interrupt()


def _interrupted_main(monkeypatch, capsys, stage):
    critic_raises = _interrupt if stage == 4 else None
    _photo_world(monkeypatch, critic_raises=critic_raises)
    store = InterruptingStore() if stage == 2 else ReadOnlyStore()
    client = Director(_interrupt if stage == 3 else _reply(selected_cluster_id="cluster_0"))
    if stage == 1:
        monkeypatch.setattr(private_run, "_store", _interrupt)
    else:
        monkeypatch.setattr(private_run, "_store", lambda: store)
    monkeypatch.setattr(private_run, "_genai_client", lambda: client)
    code = private_run.main([])
    captured = capsys.readouterr()
    return code, captured


@pytest.mark.parametrize("stage", [1, 2, 3, 4], ids=["setup", "read", "director", "photo"])
def test_an_interrupt_at_any_stage_prints_one_numeric_line_and_nothing_private(monkeypatch, capsys, stage):
    code, captured = _interrupted_main(monkeypatch, capsys, stage)
    lines = captured.out.splitlines()
    _assert_public(lines)
    assert [line for line in lines if "failed_stage" in line] == [f"[private_run] failed_stage={stage}"]
    assert lines[-1] == f"[private_run] failed_stage={stage}"
    assert captured.err == ""
    assert MARKER not in captured.out and "Traceback" not in captured.out
    assert isinstance(code, int) and code == 3


@pytest.mark.parametrize("error", [SystemExit(MARKER), GeneratorExit(MARKER)], ids=["system-exit", "base-exception"])
def test_other_base_exceptions_in_the_director_are_a_failed_stage(monkeypatch, capsys, error):
    _photo_world(monkeypatch)

    def raise_it(model):
        try:
            raise RuntimeError(MARKER)
        except RuntimeError:
            raise error

    monkeypatch.setattr(private_run, "_store", lambda: ReadOnlyStore())
    monkeypatch.setattr(private_run, "_genai_client", lambda: Director(raise_it))
    code = private_run.main([])
    captured = capsys.readouterr()
    assert captured.out.splitlines()[-1] == "[private_run] failed_stage=3"
    assert captured.err == "" and MARKER not in captured.out and code == 3


# ---- the private run record, behind its own switch ----


SLATE = [{"word": f"Zqsecret{i}", "fit": f"Zqsecret fit {i}"} for i in range(10)]


class RecordingStore(ReadOnlyStore):
    """The two reads, plus run records; every other store attribute still fails the test."""

    def __init__(self, *args, raise_on_save=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.records = []
        self.raise_on_save = raise_on_save

    def save_run_record(self, run_id, document):
        if self.raise_on_save:
            raise RuntimeError("Zqsecret store failure")
        self.records.append((run_id, document))


def _slate_reply():
    return _reply(selected_cluster_id="cluster_0", candidate_words=SLATE)


RECORD_LINE = re.compile(r"\[run_record\] written=[01]")


def _assert_public_with_record(lines):
    """The switch-on lines: today's lines plus one numbers-only record line."""
    assert lines
    for line in lines:
        assert any(pattern.fullmatch(line) for pattern in FORMATS + [RECORD_LINE]), line
        assert not any(text in line for text in PRIVATE), line


def _with_record_line(baseline_lines, written):
    """The switch-off lines with the record line after the Director line."""
    return baseline_lines[:2] + ["[run_record] written=%d" % written] + baseline_lines[2:]


def test_switch_on_writes_one_private_record_after_the_director(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    baseline, _ = _run(reply=_slate_reply())
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    store = RecordingStore()
    result, director = _run(store, reply=_slate_reply())
    assert result.lines == _with_record_line(baseline.lines, 1)
    assert result.exit_code == baseline.exit_code == 0
    _assert_public_with_record(result.lines)
    assert [read[0] for read in store.reads] == ["load_ingestion_cache", "load_pulse_feed_history"]
    [(run_id, record)] = store.records
    assert run_id == "private_run:2026-11-21:local:1"
    assert record["kind"] == "private_run" and record["date"] == "2026-11-21"
    assert record["word"] == result.commission.word == "Selvedge"
    assert record["chosen_cluster_id"] == "cluster_0"
    assert record["candidate_words"] == SLATE
    sent = director.prompts[0]
    assert record["evidence_text"] and ("Top Candidate Clusters:\n" + record["evidence_text"] + "\nReturn JSON") in sent
    assert record["prompt_path"] == "packet" and record["revised_by_photo_recovery"] is False
    assert record["models"]["director"] == mc.HYDROGEN


@pytest.mark.parametrize("value", [None, "0"])
def test_switch_off_makes_no_new_store_call(monkeypatch, value):
    _photo_world(monkeypatch)
    if value is None:
        monkeypatch.delenv("ZEITGEIST_RUN_RECORD", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_RUN_RECORD", value)
    store = RecordingStore()
    result, _ = _run(store, reply=_slate_reply())
    assert store.records == [] and result.exit_code == 0
    assert [read[0] for read in store.reads] == ["load_ingestion_cache", "load_pulse_feed_history"]


def test_switch_on_with_a_failing_record_store_keeps_the_run(monkeypatch):
    _photo_world(monkeypatch)
    baseline, _ = _run(reply=_slate_reply())
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    for store in (RecordingStore(raise_on_save=True), ReadOnlyStore()):
        result, _ = _run(store, reply=_slate_reply())
        assert result.lines == _with_record_line(baseline.lines, 0) and result.exit_code == baseline.exit_code
        _assert_public_with_record(result.lines)


class _Halt(BaseException):
    """A failure outside Exception, raised by the record store."""


class HaltingStore(RecordingStore):
    def save_run_record(self, run_id, document):
        raise _Halt("Zqsecret halt")


def test_a_base_exception_from_the_record_store_keeps_the_lines_and_exit_code(monkeypatch):
    _photo_world(monkeypatch)
    baseline, _ = _run(reply=_slate_reply())
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    result, _ = _run(HaltingStore(), reply=_slate_reply())
    assert result.lines == _with_record_line(baseline.lines, 0) and result.exit_code == baseline.exit_code == 0
    _assert_public_with_record(result.lines)


def test_switch_on_records_a_director_fallback_and_still_exits_3(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    reply = lambda model: _refusal() if model == mc.HYDROGEN else _slate_reply()
    store = RecordingStore()
    result, _ = _run(store, reply=reply)
    assert result.exit_code == 3
    [(_, record)] = store.records
    assert record["models"]["fallback_fired"] is True and record["models"]["director"] == mc.FLASH


def test_main_with_the_switch_on_prints_the_same_public_lines(monkeypatch, capsys):
    _photo_world(monkeypatch)
    code_off, lines_off, _ = _main(monkeypatch, capsys, RecordingStore(), Director(_slate_reply()))
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    store = RecordingStore()
    code_on, lines_on, everything = _main(monkeypatch, capsys, store, Director(_slate_reply()))
    assert (code_on, lines_on) == (code_off, _with_record_line(lines_off, 1))
    _assert_public(lines_off)
    _assert_public_with_record(lines_on)
    assert len(store.records) == 1
    assert not any(text in everything for text in PRIVATE)


# ---- stored vectors are grouped with the cutoffs of the model that made them ----


from src.zeitgeist import embedding_migration as em  # noqa: E402

# Percentiles that, on the synthetic window, put the cutoffs between the two groups' similarities.
CAL_PERCENTILES = (0.5, 0.4, 0.3, 0.6)


def _accepted_calibration():
    return em.calibration_document(em.Calibration("accepted", "2026-11-20", 120, CAL_PERCENTILES, 2, 0.3, 3))


class CalibratedStore(ReadOnlyStore):
    """The two reads and the calibration read; every write still fails the test."""

    def __init__(self, *args, calibration=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.calibration = calibration

    def load_calibration(self, name):
        self.reads.append(("load_calibration", name))
        return self.calibration


def _marked_rows(model):
    return {key: dict(row, embedding_model=model, embedding_source="gemini") for key, row in _rows().items()}


def _clusters(line):
    return int(re.search(r" clusters=(\d+)", line).group(1))


def test_embedding_2_rows_are_grouped_with_the_calibrated_cutoffs(monkeypatch):
    _photo_world(monkeypatch)
    legacy, _ = _run(CalibratedStore(rows=_marked_rows(em.EMBED_001), calibration=_accepted_calibration()))
    store = CalibratedStore(rows=_marked_rows(em.EMBED_2), calibration=_accepted_calibration())
    result, _ = _run(store)
    _assert_public(result.lines)
    assert result.exit_code == legacy.exit_code == 0
    # The legacy cutoffs split the two groups into six; the calibrated ones find the two groups.
    assert _clusters(legacy.lines[1]) == 6 and _clusters(result.lines[1]) == 2
    assert ("load_calibration", em.CALIBRATION_DOCUMENT) in store.reads


def test_embedding_001_rows_keep_the_legacy_cutoffs_and_read_no_calibration(monkeypatch):
    _photo_world(monkeypatch)
    unmarked, _ = _run(ReadOnlyStore())
    store = CalibratedStore(rows=_marked_rows(em.EMBED_001), calibration=_accepted_calibration())
    result, _ = _run(store)
    assert result.lines == unmarked.lines and result.exit_code == 0
    assert [read[0] for read in store.reads] == ["load_ingestion_cache", "load_pulse_feed_history"]


# ---- comparison arms on one frozen intake (SIY-208) ----


ARM_FORMATS = [
    re.compile(r"\[private_run\] arm=\d+ repeat=[1-3] director_ok=[01] attempts=\d+ packet=[01] fallback=[01]"),
    re.compile(r"\[private_run\] arm=\d+ repeat=[0-3] failed_stage=[1-4]"),
    re.compile(r"\[private_run\] records_written=\d+"),
    re.compile(r"\[private_run\] arms_refused=\d+"),
]


def _assert_arms_public(lines, *also_forbidden):
    assert lines
    for line in lines:
        assert any(pattern.fullmatch(line) for pattern in FORMATS + ARM_FORMATS), line
        assert not any(text in line for text in PRIVATE + also_forbidden), line


def _embedding_2_vector(text):
    """A synthetic embedding-2 vector: the group is read from the synthetic title."""
    group = int(re.search(r"Zqtitle(\d)", text).group(1))
    rng = np.random.default_rng(sum(text.encode("utf-8")))
    vector = np.zeros(768)
    vector[2 + group] = 1.0
    vector += rng.normal(0, 0.02, 768)
    return list(vector / np.linalg.norm(vector))


class ArmClient(Director):
    """The fake Director plus a fake embedding endpoint; records each embedding model and batch size."""

    def __init__(self, reply):
        super().__init__(reply)
        self.embeds = []

    def embed_content(self, *, model, contents, config):
        self.embeds.append((model, len(contents)))
        texts = [content.parts[0].text if hasattr(content, "parts") else str(content) for content in contents]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=_embedding_2_vector(text)) for text in texts])


class ArmStore(CalibratedStore):
    """The reads, the calibration read and run records; every other store attribute still fails the test."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("rows", _marked_rows(em.EMBED_2))
        kwargs.setdefault("calibration", _accepted_calibration())
        super().__init__(*args, **kwargs)
        self.records = []

    def save_run_record(self, run_id, document):
        self.records.append((run_id, document))


def _arms_main(monkeypatch, capsys, argv, store=None, client=None):
    store = ArmStore() if store is None else store
    client = ArmClient(_slate_reply()) if client is None else client
    monkeypatch.setattr(private_run, "_store", lambda: store)
    monkeypatch.setattr(private_run, "_genai_client", lambda: client)
    monkeypatch.setattr(private_run, "_now", lambda: NOW)
    code = private_run.main(argv)
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.out + out.err, store, client


def _spy_shown_groups(monkeypatch):
    shown = []
    real = private_run.pipeline.commission_daily_zeitgeist

    def spy(top_clusters, *args, **kwargs):
        shown.append([cluster.cluster_id for cluster in top_clusters[:5]])
        return real(top_clusters, *args, **kwargs)

    monkeypatch.setattr(private_run.pipeline, "commission_daily_zeitgeist", spy)
    return shown


def test_two_arms_three_repeats_write_six_records_on_one_intake(monkeypatch, capsys):
    _photo_world(monkeypatch)
    monkeypatch.delenv("ZEITGEIST_RUN_RECORD", raising=False)
    monkeypatch.delenv("ZEITGEIST_UNDERSTANDING_V1", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    code, lines, everything, store, client = _arms_main(
        monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "3"])
    _assert_arms_public(lines)
    assert not any(text in everything for text in PRIVATE)
    assert code == 0
    assert [line for line in lines if " repeat=" in line] == [
        "[private_run] arm=1 repeat=1 director_ok=1 attempts=1 packet=0 fallback=0",
        "[private_run] arm=1 repeat=2 director_ok=1 attempts=1 packet=0 fallback=0",
        "[private_run] arm=1 repeat=3 director_ok=1 attempts=1 packet=0 fallback=0",
        "[private_run] arm=2 repeat=1 director_ok=1 attempts=1 packet=1 fallback=0",
        "[private_run] arm=2 repeat=2 director_ok=1 attempts=1 packet=1 fallback=0",
        "[private_run] arm=2 repeat=3 director_ok=1 attempts=1 packet=1 fallback=0",
    ]
    assert lines[-1] == "[private_run] records_written=6"
    # The intake is read once, and each arm's Director answers on its own model.
    assert [read[0] for read in store.reads].count("load_ingestion_cache") == 1
    assert client.calls == [mc.FLASH] * 3 + [mc.HYDROGEN] * 3
    # Only the previous arm embeds again, in memory, with the first model; the current arm uses the stored vectors.
    assert {model for model, _ in client.embeds} == {em.EMBED_001}
    assert sum(size for _, size in client.embeds) == 8
    ids = [run_id for run_id, _ in store.records]
    assert ids == [f"private_run:2026-11-21:local:1:{arm}:{repeat}"
                   for arm in ("previous", "current") for repeat in (1, 2, 3)]
    records = [record for _, record in store.records]
    expected_hash = private_run.intake_hash(list(_marked_rows(em.EMBED_2).values()))
    assert {record["intake_hash"] for record in records} == {expected_hash}
    assert [(record["arm"], record["repeat"]) for record in records] == [
        (arm, repeat) for arm in ("previous", "current") for repeat in (1, 2, 3)]
    for record in records:
        assert record["kind"] == "private_run"
    previous, current = records[0], records[3]
    assert previous["models"]["director"] == previous["models"]["judge"] == mc.FLASH
    assert previous["models"]["embedding"] == em.EMBED_001 and previous["prompt_path"] == "legacy"
    assert current["models"]["director"] == current["models"]["judge"] == mc.HYDROGEN
    assert current["models"]["embedding"] == em.EMBED_2 and current["prompt_path"] == "packet"
    # The previous arm groups on the legacy cutoffs; the current arm on the calibrated embedding-2 ones.
    assert {record["cutoffs_source"] for record in records[:3]} == {"legacy"}
    assert {record["cutoffs_source"] for record in records[3:]} == {"calibrated"}
    assert len(previous["groups"]) == 5 and len(current["groups"]) == 2


def test_the_intake_hash_ignores_vectors_and_row_order_and_follows_the_text():
    rows = list(_marked_rows(em.EMBED_001).values())
    digest = private_run.intake_hash(rows)
    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    revectored = [dict(row, embedding_f16="AAAA", embedding_model=em.EMBED_2) for row in rows]
    assert private_run.intake_hash(list(reversed(revectored))) == digest
    retitled = [dict(rows[0], title="Zqtitle00 another story")] + rows[1:]
    assert private_run.intake_hash(retitled) != digest


def test_repeats_show_the_same_five_groups_in_a_different_order(monkeypatch, capsys):
    _photo_world(monkeypatch)
    shown = _spy_shown_groups(monkeypatch)
    code, _, _, store, _ = _arms_main(monkeypatch, capsys, ["--arms", "previous", "--repeats", "3"])
    assert code == 0 and len(shown) == 3
    ranked = [group["cluster_id"] for group in store.records[0][1]["groups"]]
    assert shown[0] == ranked
    assert sorted(shown[1]) == sorted(shown[2]) == sorted(ranked)
    assert shown[1] != shown[0] and shown[2] != shown[0]
    # The record keeps the true ranks whatever order the Director saw.
    assert [[group["cluster_id"] for group in record["groups"]] for _, record in store.records] == [ranked] * 3
    # The order is fixed by the intake and the repeat number.
    first = [list(order) for order in shown]
    again = _spy_shown_groups(monkeypatch)
    _arms_main(monkeypatch, capsys, ["--arms", "previous", "--repeats", "3"])
    assert again == first


def test_candidate_order_is_a_fixed_permutation_per_intake_and_repeat():
    assert private_run.candidate_order(5, "a" * 64, 1) == [0, 1, 2, 3, 4]
    for repeat in (2, 3):
        order = private_run.candidate_order(5, "a" * 64, repeat)
        assert sorted(order) == [0, 1, 2, 3, 4] and order != [0, 1, 2, 3, 4]
        assert order == private_run.candidate_order(5, "a" * 64, repeat)
    assert private_run.candidate_order(2, "b" * 64, 2) == [1, 0]
    assert private_run.candidate_order(1, "b" * 64, 3) == [0]


def test_a_store_write_other_than_the_record_fails_that_arm_only(monkeypatch, capsys):
    _photo_world(monkeypatch)

    def writing_plan(store, requested):
        store.save_calibration(em.CALIBRATION_DOCUMENT, {})

    monkeypatch.setattr(em, "plan_embedding_run", writing_plan)
    code, lines, _, store, _ = _arms_main(monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "2"])
    _assert_arms_public(lines)
    assert code == 3
    assert "[private_run] arm=2 repeat=0 failed_stage=2" in lines
    assert not any(line.startswith("[private_run] arm=2 repeat=1 director_ok") for line in lines)
    assert [line for line in lines if line.startswith("[private_run] arm=1 ")] == [
        "[private_run] arm=1 repeat=1 director_ok=1 attempts=1 packet=0 fallback=0",
        "[private_run] arm=1 repeat=2 director_ok=1 attempts=1 packet=0 fallback=0",
    ]
    assert [record["arm"] for _, record in store.records] == ["previous", "previous"]
    assert lines[-1] == "[private_run] records_written=2"


def test_the_arm_store_passes_reads_and_the_record_and_refuses_every_other_write():
    store = ArmStore()
    arm_store = private_run.ArmStore(store)
    assert arm_store.load_calibration(em.CALIBRATION_DOCUMENT) == _accepted_calibration()
    arm_store.save_run_record("private_run:2026-11-21:local:1:previous:1", {"kind": "private_run"})
    assert store.records == [("private_run:2026-11-21:local:1:previous:1", {"kind": "private_run"})]
    for name in ("save_calibration", "save_ingestion_cache_day", "save_orbit_roster", "save_pulse_state",
                 "save_pulse_feed_entry", "delete_run_record", "clear_calibration", "prune_ingestion_cache"):
        with pytest.raises(private_run.ArmWriteRefused):
            getattr(arm_store, name)
    assert arm_store.refused == 8


def test_a_failing_arm_does_not_stop_the_next_one(monkeypatch, capsys):
    _photo_world(monkeypatch)
    reply = lambda model: RuntimeError("Zqsecret provider text") if model == mc.FLASH else _slate_reply()
    code, lines, everything, store, _ = _arms_main(
        monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "1"], client=ArmClient(reply))
    _assert_arms_public(lines)
    assert "Zqsecret" not in everything
    assert code == 3
    assert "[private_run] arm=1 repeat=1 failed_stage=3" in lines
    assert "[private_run] arm=2 repeat=1 director_ok=1 attempts=1 packet=1 fallback=0" in lines
    assert [record["arm"] for _, record in store.records] == ["current"]


def _spy_photo(monkeypatch):
    calls = []
    real = vr.select_and_render_pulse_assets

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(vr, "select_and_render_pulse_assets", spy)
    return calls


def test_the_photo_path_runs_once_per_arm_on_the_first_repeat(monkeypatch, capsys):
    _, critic = _photo_world(monkeypatch)
    calls = _spy_photo(monkeypatch)
    code, lines, _, _, _ = _arms_main(monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "3"])
    assert code == 0
    assert [kwargs["judge_model"] for kwargs in calls] == [mc.FLASH, mc.HYDROGEN]
    assert not any("download_ping_callback" in kwargs for kwargs in calls)
    photo = [index for index, line in enumerate(lines) if line.startswith("[private_run] photo_ok=")]
    assert len(photo) == 2
    assert lines[photo[0] - 1].startswith("[private_run] arm=1 repeat=1 ")
    assert lines[photo[1] - 1].startswith("[private_run] arm=2 repeat=1 ")


def test_the_photo_path_runs_for_at_most_the_first_two_arms(monkeypatch, capsys):
    _photo_world(monkeypatch)
    calls = _spy_photo(monkeypatch)
    table = dict(private_run.ARMS)
    table["current_links"] = private_run.Arm(mc.HYDROGEN, mc.HYDROGEN, em.EMBED_2, packet=True)
    monkeypatch.setattr(private_run, "ARMS", table)
    code, lines, _, store, _ = _arms_main(
        monkeypatch, capsys, ["--arms", "previous,current,current_links", "--repeats", "1"])
    assert code == 0 and len(store.records) == 3
    assert len(calls) == 2
    assert sum(line.startswith("[private_run] photo_ok=") for line in lines) == 2


@pytest.mark.parametrize("argv, refusal", [
    (["--arms", "previous", "--repeats", "0"], 1),
    (["--arms", "previous", "--repeats", "4"], 1),
    (["--arms", "previous", "--repeats", "two"], 1),
    (["--arms", "previous,Zqsecretarm"], 2),
    (["--arms", "previous,,current"], 2),
    (["--arms", "previous,current,previous,current,previous", "--repeats", "3"], 4),
    (["--arms", "previous,previous", "--repeats", "1"], 5),
    (["--arms", "current_reflector", "--repeats", "1"], 3),
], ids=["zero", "four", "word", "unknown", "empty-part", "over-twelve", "duplicate", "reflector"])
def test_a_refused_dispatch_ends_with_one_numeric_line_before_any_call(monkeypatch, capsys, argv, refusal):
    _photo_world(monkeypatch)
    opened = []
    store, client = ArmStore(), ArmClient(_slate_reply())
    monkeypatch.setattr(private_run, "_store", lambda: opened.append(1) or store)
    monkeypatch.setattr(private_run, "_genai_client", lambda: opened.append(2) or client)
    code = private_run.main(argv)
    out = capsys.readouterr()
    assert out.out.splitlines() == [f"[private_run] arms_refused={refusal}"]
    assert code == 3 and opened == [] and store.reads == [] and client.calls == [] and client.embeds == []
    assert "Zqsecret" not in out.out + out.err


def test_arms_never_measure_the_judges(monkeypatch, capsys):
    _photo_world(monkeypatch)

    def refuse(*args, **kwargs):
        raise AssertionError("measure_judges called in arms mode")

    monkeypatch.setattr(private_run, "measure_judges", refuse)
    code, lines, _, store, _ = _arms_main(monkeypatch, capsys, ["--arms", "current", "--repeats", "1"])
    assert code == 0 and len(store.records) == 1
    assert not any(line.startswith("[judge_calibration]") for line in lines)


def test_empty_arms_and_repeats_run_the_single_private_run(monkeypatch, capsys):
    _photo_world(monkeypatch)
    code_none, lines_none, _ = _main(monkeypatch, capsys, ReadOnlyStore(), Director(_slate_reply()))
    monkeypatch.setattr(private_run, "_store", lambda: ReadOnlyStore())
    monkeypatch.setattr(private_run, "_genai_client", lambda: Director(_slate_reply()))
    code = private_run.main(["--arms", "", "--repeats", ""])
    assert (code, capsys.readouterr().out.splitlines()) == (code_none, lines_none)


def test_repeats_without_arms_is_an_argument_error():
    with pytest.raises(SystemExit):
        private_run.main(["--repeats", "2"])


def test_a_failed_intake_read_ends_the_dispatch_with_a_numeric_line(monkeypatch, capsys):
    _photo_world(monkeypatch)
    store = ArmStore(fail=RuntimeError("Zqsecret"))
    code, lines, everything, store, client = _arms_main(monkeypatch, capsys, ["--arms", "previous,current"],
                                                        store=store)
    assert lines == ["[private_run] failed_stage=2"] and code == 3
    assert client.calls == [] and store.records == [] and "Zqsecret" not in everything


# ---- the record says which cutoffs and which embedding model grouped the night ----


@pytest.mark.parametrize("rows_model, calibration, source", [
    (em.EMBED_2, True, "calibrated"),
    (em.EMBED_2, False, "legacy"),
    (em.EMBED_001, True, "legacy"),
], ids=["embedding-2-calibrated", "embedding-2-uncalibrated", "embedding-001"])
def test_the_private_record_names_the_cutoffs_and_the_vectors_model(monkeypatch, rows_model, calibration, source):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    store = ArmStore(rows=_marked_rows(rows_model), calibration=_accepted_calibration() if calibration else None)
    # The environment names the first model; the record must name the model on the rows.
    result, _ = _run(store, reply=_slate_reply(), run_models=_models())
    assert result.exit_code == 0
    [(_, record)] = store.records
    assert record["cutoffs_source"] == source
    assert record["models"]["embedding"] == rows_model


def test_the_private_record_does_not_name_one_model_for_mixed_rows(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    rows = _marked_rows(em.EMBED_2)
    rows["h00"]["embedding_model"] = em.EMBED_001
    store = ArmStore(rows=rows)
    _run(store, reply=_slate_reply())
    [(_, record)] = store.records
    assert record["cutoffs_source"] == "legacy" and record["models"]["embedding"] == "mixed"


@pytest.mark.parametrize("rows_model, calibration", [(em.EMBED_001, True), (em.EMBED_2, False)],
                         ids=["first-model-rows", "no-accepted-calibration"])
def test_the_current_arm_fails_before_any_call_without_its_vectors_and_calibration(
        monkeypatch, capsys, rows_model, calibration):
    _photo_world(monkeypatch)
    store = ArmStore(rows=_marked_rows(rows_model), calibration=_accepted_calibration() if calibration else None)
    code, lines, _, store, client = _arms_main(
        monkeypatch, capsys, ["--arms", "current,previous", "--repeats", "1"], store=store)
    _assert_arms_public(lines)
    assert code == 3
    assert lines[0] == "[private_run] arm=1 repeat=0 failed_stage=2"
    assert not any(line.startswith("[private_run] arm=1 repeat=1") for line in lines)
    assert mc.HYDROGEN not in client.calls and em.EMBED_2 not in {model for model, _ in client.embeds}
    # The previous arm still runs, on the first model's vectors and the legacy cutoffs.
    assert [(record["arm"], record["cutoffs_source"], record["models"]["embedding"])
            for _, record in store.records] == [("previous", "legacy", em.EMBED_001)]


def test_the_previous_arm_on_first_model_rows_makes_no_embedding_call(monkeypatch, capsys):
    _photo_world(monkeypatch)
    store = ArmStore(rows=_marked_rows(em.EMBED_001))
    code, _, _, store, client = _arms_main(monkeypatch, capsys, ["--arms", "previous", "--repeats", "1"],
                                           store=store)
    assert code == 0 and client.embeds == []
    [(_, record)] = store.records
    assert record["cutoffs_source"] == "legacy" and record["models"]["embedding"] == em.EMBED_001


# ---- a comparison stops after 24 Director calls ----


ARM_FORMATS.append(re.compile(r"\[private_run\] arm=\d+ repeat=[1-3] skipped=1"))


def test_a_comparison_starts_no_attempt_after_24_director_calls(monkeypatch, capsys):
    _photo_world(monkeypatch)
    client = ArmClient("not json")
    code, lines, _, store, client = _arms_main(
        monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "3"], client=client)
    _assert_arms_public(lines)
    assert code == 3
    # Every try counts: the previous arm makes 3 per attempt, the current arm 5 on Hydrogen and 3 on Flash.
    # The twenty-fifth call, the last Flash try of the current arm's second attempt, is refused before dispatch.
    assert client.calls == [mc.FLASH] * 9 + [mc.HYDROGEN] * 5 + [mc.FLASH] * 3 + [mc.HYDROGEN] * 5 + [mc.FLASH] * 2
    assert len(client.calls) == 24
    assert "[private_run] arm=2 repeat=2 failed_stage=3" in lines
    assert "[private_run] arm=2 repeat=3 skipped=1" in lines
    assert not any(line.startswith("[private_run] arm=2 repeat=2 director_ok") for line in lines)
    assert not any(line.startswith("[private_run] arm=2 repeat=3 director_ok") for line in lines)
    assert [(record["arm"], record["repeat"]) for _, record in store.records] == [
        ("previous", 1), ("previous", 2), ("previous", 3), ("current", 1)]
    assert lines[-1] == "[private_run] records_written=4"


def test_an_arm_reached_after_the_cap_is_skipped_without_embedding(monkeypatch, capsys):
    _photo_world(monkeypatch)
    table = dict(private_run.ARMS)
    table["current_links"] = private_run.Arm(mc.HYDROGEN, mc.HYDROGEN, em.EMBED_2, packet=True)
    table["current_reflector"] = private_run.Arm(mc.FLASH, mc.FLASH, em.EMBED_001, packet=False)
    monkeypatch.setattr(private_run, "ARMS", table)
    client = ArmClient("not json")
    code, lines, _, _, client = _arms_main(
        monkeypatch, capsys, ["--arms", "current,current_links,current_reflector", "--repeats", "2"], client=client)
    _assert_arms_public(lines)
    assert code == 3 and len(client.calls) == 24 and client.embeds == []
    assert [line for line in lines if "skipped" in line] == [
        "[private_run] arm=2 repeat=2 skipped=1", "[private_run] arm=3 repeat=1 skipped=1",
        "[private_run] arm=3 repeat=2 skipped=1"]


def test_the_director_counter_counts_every_made_call_but_not_a_refusal_before_dispatch():
    replies = iter([_slate_reply(), RuntimeError("Zqsecret timeout"), _refusal(), _refusal(pre=False)])
    client = ArmClient(lambda model: next(replies))
    counter = private_run.DirectorCounter(client)
    counter.models.generate_content(model=mc.FLASH, contents="x", config={})
    for _ in range(3):
        with pytest.raises(Exception):
            counter.models.generate_content(model=mc.FLASH, contents="x", config={})
    assert counter.calls == 3
    counter.models.embed_content(model=em.EMBED_001, contents=["Zqtitle00"], config={})
    assert counter.calls == 3 and client.embeds == [(em.EMBED_001, 1)]


def test_the_director_counter_refuses_the_twenty_fifth_call_before_dispatch():
    client = ArmClient(lambda model: _slate_reply())
    counter = private_run.DirectorCounter(client)
    for _ in range(private_run.MAX_DIRECTOR_CALLS):
        counter.models.generate_content(model=mc.FLASH, contents="x", config={})
    assert counter.calls == 24 and len(client.calls) == 24
    with pytest.raises(SpendBlocked) as caught:
        counter.models.generate_content(model=mc.HYDROGEN, contents="x", config={})
    assert refused_before_dispatch(caught.value) is True
    assert counter.calls == 24 and len(client.calls) == 24


# ---- a spend guard that reports an uncertain call ends the comparison ----


class UncertainOnSecondCall(ArmClient):
    """The fake client; its second paid call, Director or embedding, raises a block after dispatch."""

    def __init__(self, reply):
        super().__init__(reply)
        self.paid = 0

    def _second(self):
        self.paid += 1
        if self.paid == 2:
            raise _refusal(pre=False)

    def generate_content(self, *, model, contents, config):
        self._second()
        return super().generate_content(model=model, contents=contents, config=config)

    def embed_content(self, *, model, contents, config):
        self._second()
        return super().embed_content(model=model, contents=contents, config=config)


def test_an_uncertain_director_call_ends_the_whole_comparison(monkeypatch, capsys):
    _photo_world(monkeypatch)
    client = UncertainOnSecondCall(_slate_reply())
    code, lines, _, store, client = _arms_main(
        monkeypatch, capsys, ["--arms", "current,previous", "--repeats", "3"], client=client)
    _assert_arms_public(lines)
    assert code == 3
    assert [line for line in lines if " repeat=" in line] == [
        "[private_run] arm=1 repeat=1 director_ok=1 attempts=1 packet=1 fallback=0",
        "[private_run] arm=1 repeat=2 failed_stage=3",
        "[private_run] arm=1 repeat=3 skipped=1",
        "[private_run] arm=2 repeat=1 skipped=1",
        "[private_run] arm=2 repeat=2 skipped=1",
        "[private_run] arm=2 repeat=3 skipped=1",
    ]
    # No Flash round after the block, no further repeat or arm, and the previous arm embeds nothing.
    assert client.paid == 2 and client.calls == [mc.HYDROGEN] and client.embeds == []
    assert [(record["arm"], record["repeat"]) for _, record in store.records] == [("current", 1)]
    assert lines[-1] == "[private_run] records_written=1"


def test_an_uncertain_embedding_call_ends_the_whole_comparison(monkeypatch, capsys):
    _photo_world(monkeypatch)
    table = dict(private_run.ARMS)
    table["current_reflector"] = private_run.Arm(mc.FLASH, mc.FLASH, em.EMBED_001, packet=False)
    monkeypatch.setattr(private_run, "ARMS", table)
    client = UncertainOnSecondCall(_slate_reply())
    code, lines, _, store, client = _arms_main(
        monkeypatch, capsys, ["--arms", "current,previous,current_reflector", "--repeats", "1"], client=client)
    _assert_arms_public(lines)
    assert code == 3
    assert [line for line in lines if " repeat=" in line] == [
        "[private_run] arm=1 repeat=1 director_ok=1 attempts=1 packet=1 fallback=0",
        "[private_run] arm=2 repeat=0 failed_stage=2",
        "[private_run] arm=2 repeat=1 skipped=1",
        "[private_run] arm=3 repeat=1 skipped=1",
    ]
    assert client.paid == 2 and client.calls == [mc.HYDROGEN] and client.embeds == []
    assert [(record["arm"], record["repeat"]) for _, record in store.records] == [("current", 1)]
    assert lines[-1] == "[private_run] records_written=1"


def test_a_refusal_before_dispatch_in_the_re_embedding_fails_that_arm_only(monkeypatch, capsys):
    _photo_world(monkeypatch)

    class RefusingEmbeddings(ArmClient):
        def embed_content(self, *, model, contents, config):
            raise _refusal()

    code, lines, _, store, _ = _arms_main(
        monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "1"],
        client=RefusingEmbeddings(_slate_reply()))
    _assert_arms_public(lines)
    assert code == 3
    assert "[private_run] arm=1 repeat=0 failed_stage=2" in lines
    assert "[private_run] arm=2 repeat=1 director_ok=1 attempts=1 packet=1 fallback=0" in lines
    assert [record["arm"] for _, record in store.records] == ["current"]


# ---- the switch-on run says by number whether its record was written ----


def test_a_director_fallback_with_the_switch_on_still_says_whether_the_record_was_written(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    result, _ = _run(RecordingStore(), reply="not json")
    assert result.lines[2] == "[run_record] written=1" and len(result.lines) == 3
    assert result.exit_code == 3


@pytest.mark.parametrize("value", [None, "0"])
def test_switch_off_prints_no_record_line(monkeypatch, value):
    _photo_world(monkeypatch)
    if value is None:
        monkeypatch.delenv("ZEITGEIST_RUN_RECORD", raising=False)
    else:
        monkeypatch.setenv("ZEITGEIST_RUN_RECORD", value)
    result, _ = _run(RecordingStore(), reply=_slate_reply())
    assert not any(line.startswith("[run_record]") for line in result.lines)
    _assert_public(result.lines)


# ---- the own-group check after the record (SIY-213) ----


from src.zeitgeist import own_group  # noqa: E402


@pytest.fixture(autouse=True)
def _own_group_switch_off_unless_set(monkeypatch):
    """A shell that exports ZEITGEIST_OWN_GROUP must not change these tests; those that want it set it."""
    monkeypatch.delenv("ZEITGEIST_OWN_GROUP", raising=False)


class OwnGroupStore(RecordingStore):
    """Rows marked with one embedding model, run records kept and read back; every store call is noted."""

    def __init__(self, *args, model=em.EMBED_001, **kwargs):
        kwargs.setdefault("rows", _marked_rows(model))
        super().__init__(*args, **kwargs)
        self.calls = []

    def save_run_record(self, run_id, document):
        self.calls.append("save_run_record")
        super().save_run_record(run_id, document)

    def load_run_record(self, run_id):
        self.calls.append("load_run_record")
        found = [document for saved_id, document in self.records if saved_id == run_id]
        return json.loads(json.dumps(found[-1])) if found else None


def _group_vector(group):
    vector = np.zeros(768)
    vector[group] = 1.0
    return list(vector)


class EmbeddingDirector(Director):
    """The fake Director plus a fake embedding endpoint that answers the own-group text on group 0's axis."""

    def __init__(self, reply, embed_error=None):
        super().__init__(reply)
        self.embeds = []
        self.embed_error = embed_error

    def embed_content(self, *, model, contents, config):
        texts = [content.parts[0].text if hasattr(content, "parts") else str(content) for content in contents]
        self.embeds.append((model, texts, config))
        if self.embed_error is not None:
            raise self.embed_error
        vectors = [_embedding_2_vector(text) if "Zqtitle" in text else _group_vector(0) for text in texts]
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vector) for vector in vectors])


def _own_group_run(store, embed_error=None):
    client = EmbeddingDirector(_slate_reply(), embed_error=embed_error)
    result = private_run.run_private(store, genai_client=client, now=NOW, run_models=_models())
    return result, client


def _own_group(record):
    return record["tests"].get("own_group")


def test_own_group_on_stores_the_numbers_after_the_record_and_keeps_the_lines(monkeypatch, capsys):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    baseline, _ = _own_group_run(OwnGroupStore())
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    store = OwnGroupStore()
    result, client = _own_group_run(store)
    assert (result.lines, result.exit_code) == (baseline.lines, baseline.exit_code)
    _assert_public_with_record(result.lines)
    assert client.embeds == [(em.EMBED_001, ["Selvedge. A finished edge that holds."], {"output_dimensionality": 768})]
    assert store.calls == ["save_run_record", "load_run_record", "load_run_record", "save_run_record"]
    first, last = store.records[0][1], store.records[-1][1]
    assert store.records[0][0] == store.records[-1][0] == "private_run:2026-11-21:local:1"
    numbers = _own_group(last)
    assert numbers["status"] == "ok"
    chosen = next(group for group in first["groups"] if group["cluster_id"] == first["chosen_cluster_id"])
    assert numbers["n_own"] == chosen["item_count"]
    assert len(numbers["others"]) == len(first["groups"]) - 1
    assert {k: v for k, v in last.items() if k != "tests"} == {k: v for k, v in first.items() if k != "tests"}
    out = capsys.readouterr()
    assert not any(text in out.out + out.err for text in PRIVATE)


@pytest.mark.parametrize("error", [None, "refused", "raises"])
def test_a_refused_or_failing_embedding_leaves_the_run_unchanged(monkeypatch, error):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    baseline, _ = _own_group_run(OwnGroupStore())
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    store = OwnGroupStore()
    embed_error = {None: None, "refused": _refusal(), "raises": RuntimeError("Zqsecret embedding failure")}[error]
    result, client = _own_group_run(store, embed_error=embed_error)
    assert (result.lines, result.exit_code) == (baseline.lines, baseline.exit_code)
    assert len(client.embeds) == 1
    expected = {None: "ok", "refused": "skipped", "raises": "skipped"}[error]
    assert _own_group(store.records[-1][1])["status"] == expected


def test_vectors_of_several_models_are_skipped_without_a_call(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    rows = _marked_rows(em.EMBED_001)
    rows["h00"]["embedding_model"] = em.EMBED_2
    store = OwnGroupStore(rows=rows)
    result, client = _own_group_run(store)
    assert result.exit_code == 0 and client.embeds == []
    assert _own_group(store.records[-1][1]) == {"status": "skipped", "reason": "embedding_model"}


@pytest.mark.parametrize("value", [None, "0"])
def test_own_group_off_makes_no_embedding_and_no_extra_store_call(monkeypatch, value):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    if value is not None:
        monkeypatch.setenv("ZEITGEIST_OWN_GROUP", value)
    store = OwnGroupStore()
    result, client = _own_group_run(store)
    assert result.exit_code == 0 and client.embeds == []
    assert store.calls == ["save_run_record"] and _own_group(store.records[0][1]) is None


def test_own_group_on_with_the_record_off_makes_no_call(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    store = OwnGroupStore()
    result, client = _own_group_run(store)
    assert result.exit_code == 0 and client.embeds == [] and store.calls == []


def test_an_unwritten_record_gets_no_check(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    store = OwnGroupStore(raise_on_save=True)
    result, client = _own_group_run(store)
    assert result.exit_code == 0 and client.embeds == []
    assert store.calls == ["save_run_record"]


class OwnGroupArmStore(ArmStore):
    """The arm test store with its run records read back."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = []

    def save_run_record(self, run_id, document):
        self.calls.append("save_run_record")
        super().save_run_record(run_id, document)

    def load_run_record(self, run_id):
        self.calls.append("load_run_record")
        found = [document for saved_id, document in self.records if saved_id == run_id]
        return json.loads(json.dumps(found[-1])) if found else None


class OwnGroupArmClient(ArmClient):
    """The arm client; the own-group text is answered on one fixed axis."""

    def embed_content(self, *, model, contents, config):
        texts = [content.parts[0].text if hasattr(content, "parts") else str(content) for content in contents]
        if any("Zqtitle" in text for text in texts):
            return super().embed_content(model=model, contents=contents, config=config)
        self.embeds.append((model, len(contents)))
        return SimpleNamespace(embeddings=[SimpleNamespace(values=_group_vector(2)) for _ in texts])


def _own_group_arms(monkeypatch, capsys):
    store, client = OwnGroupArmStore(), OwnGroupArmClient(_slate_reply())
    code, lines, everything, _, _ = _arms_main(monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "2"],
                                               store=store, client=client)
    return code, lines, everything, store, client


def test_each_arm_and_repeat_is_checked_after_its_record(monkeypatch, capsys):
    _photo_world(monkeypatch)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    code_off, lines_off, _, store_off, client_off = _own_group_arms(monkeypatch, capsys)
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    code, lines, everything, store, client = _own_group_arms(monkeypatch, capsys)
    assert (code, lines) == (code_off, lines_off) and code == 0
    _assert_arms_public(lines)
    assert not any(text in everything for text in PRIVATE)
    # One request of one text per arm and repeat, on the model that made that arm's vectors.
    assert client.embeds == client_off.embeds + [(em.EMBED_001, 1)] * 2 + [(em.EMBED_2, 1)] * 2
    assert store_off.calls == ["save_run_record"] * 4
    assert store.calls == ["save_run_record", "load_run_record", "load_run_record", "save_run_record"] * 4
    finals = {}
    for run_id, record in store.records:
        finals[run_id] = record
    assert list(finals) == [f"private_run:2026-11-21:local:1:{arm}:{repeat}"
                            for arm in ("previous", "current") for repeat in (1, 2)]
    assert all(_own_group(record)["status"] == "ok" for record in finals.values())



# ---- the links arm (links plan, Task 3) ----


def test_the_links_arm_is_runnable_and_the_reflector_arm_is_known_but_not():
    assert private_run.ARMS["current_links"] == private_run.Arm(mc.HYDROGEN, mc.HYDROGEN, em.EMBED_2, packet=True,
                                                               links=True)
    assert private_run.ARMS["current"].links is False and private_run.ARMS["previous"].links is False
    assert private_run.ARMS["current_reflector"].runnable is False
    assert private_run.parse_arms("current,current_links", "1") == (["current", "current_links"], 1, 0)


class LinksArmStore(ArmStore):
    """The arm store plus the two reads the night's links make."""

    def load_calibration(self, name):
        self.reads.append(("load_calibration", name))
        return {"synthetic_handle": "design"} if name == "orbit_realms" else self.calibration

    def load_orbit_roster(self):
        self.reads.append(("load_orbit_roster",))
        return {"handles": {}}


def test_the_links_arm_is_the_current_arm_with_the_links_the_edition_would_show(monkeypatch, capsys):
    from src.zeitgeist.understanding import LinkCandidate

    _photo_world(monkeypatch)
    monkeypatch.delenv("ZEITGEIST_LINKS", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ATTEMPT", raising=False)
    shown = []
    real_commission = private_run.pipeline.commission_daily_zeitgeist

    def spy_commission(top_clusters, *args, **kwargs):
        shown.append(kwargs.get("links"))
        return real_commission(top_clusters, *args, **kwargs)

    built = []
    real_night_links = private_run.pipeline._night_links

    def spy_night_links(store, config, clusters):
        found = real_night_links(store, config, clusters)
        built.append((config, list(clusters), found))
        return found

    def one_link(clusters, realm_map):
        members = tuple(clusters[:2])
        return [LinkCandidate("link_1", tuple(str(c.cluster_id) for c in members), ("markets", "tech", "civic"),
                              4, 0.93, members)]

    monkeypatch.setattr(private_run.pipeline, "commission_daily_zeitgeist", spy_commission)
    monkeypatch.setattr(private_run.pipeline, "_night_links", spy_night_links)
    monkeypatch.setattr(private_run.pipeline, "link_candidates", one_link)
    config = {"feeds": [{"id": "src_0_0", "category": "tech_ai"}]}
    monkeypatch.setattr(private_run, "_orbit_config", lambda: config)
    store = LinksArmStore()
    code, lines, everything, store, client = _arms_main(
        monkeypatch, capsys, ["--arms", "current,current_links", "--repeats", "1"], store=store)
    _assert_arms_public(lines)
    assert not any(text in everything for text in PRIVATE + ("synthetic_handle", "Link"))
    assert code == 0 and len(store.records) == 2
    # The current arm is shown no links; the links arm the night's links, built as the edition builds them.
    assert shown[0] is None and [link.link_id for link in shown[1]] == ["link_1"]
    [(config_seen, clusters_seen, (links, _))] = built
    assert config_seen is config and links is shown[1]
    assert ("load_calibration", "orbit_realms") in store.reads and ("load_orbit_roster",) in store.reads
    assert "- Link link_1" not in client.prompts[0] and "\n- Link link_1: groups=" in client.prompts[1]
    current, with_links = [record for _, record in store.records]
    assert current["links"] is None and current["realm_counts"] is None
    assert with_links["arm"] == "current_links"
    assert with_links["links"] == [{"link_id": "link_1", "member_ids": [str(c.cluster_id) for c in clusters_seen[:2]],
                                    "realms": ["markets", "tech", "civic"], "source_count": 4,
                                    "mean_link_similarity": 0.93}]
    assert sum(with_links["realm_counts"]["intake"].values()) == 8
    differ = ("arm", "links", "realm_counts", "chosen_mapped", "evidence_text", "prompt_sha256", "written_at",
              "run_id")
    assert {k: v for k, v in current.items() if k not in differ} == {
        k: v for k, v in with_links.items() if k not in differ}


def test_a_failing_night_links_read_leaves_the_links_arm_running_without_links(monkeypatch, capsys):
    _photo_world(monkeypatch)
    monkeypatch.setattr(private_run, "_orbit_config", lambda: {"feeds": []})
    # The plain arm store has no roster read: the night's links run on the defaults and find none here.
    code, lines, everything, store, client = _arms_main(
        monkeypatch, capsys, ["--arms", "current_links", "--repeats", "1"])
    _assert_arms_public(lines)
    assert code == 0 and len(store.records) == 1
    [(_, record)] = store.records
    assert record["links"] == [] and "- Link" not in client.prompts[0]


class LinksRecordingStore(RecordingStore):
    """The recording store plus the two reads the night's links make."""

    def load_calibration(self, name):
        self.reads.append(("load_calibration", name))
        return {}

    def load_orbit_roster(self):
        self.reads.append(("load_orbit_roster",))
        return {"handles": {}}


def _first_two_linked(clusters, realm_map):
    from src.zeitgeist.understanding import LinkCandidate

    members = tuple(clusters[:2])
    return [LinkCandidate("link_1", tuple(str(c.cluster_id) for c in members), ("markets", "tech", "civic"), 4,
                          0.93, members)]


def test_the_private_run_shows_the_links_as_the_edition_does_under_the_links_switch(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    monkeypatch.setenv("ZEITGEIST_LINKS", "1")
    monkeypatch.setattr(private_run.pipeline, "link_candidates", _first_two_linked)
    monkeypatch.setattr(private_run, "_orbit_config", lambda: {"feeds": []})
    store = LinksRecordingStore()
    result, director = _run(store, reply=_slate_reply())
    _assert_public_with_record(result.lines)
    assert result.exit_code == 0
    assert "\n- Link link_1: groups=" in director.prompts[0]
    assert ("load_calibration", "orbit_realms") in store.reads and ("load_orbit_roster",) in store.reads
    [(_, record)] = store.records
    assert [link["link_id"] for link in record["links"]] == ["link_1"]
    assert set(record["realm_counts"]) == {"intake", "candidates"}


def test_the_private_run_without_the_links_switch_reads_nothing_more(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    monkeypatch.delenv("ZEITGEIST_LINKS", raising=False)
    store = LinksRecordingStore()
    result, director = _run(store, reply=_slate_reply())
    assert [read[0] for read in store.reads] == ["load_ingestion_cache", "load_pulse_feed_history"]
    [(_, record)] = store.records
    assert record["links"] is None and record["realm_counts"] is None
    assert "- Link" not in director.prompts[0]


@pytest.mark.parametrize("case", ["director_failed", "fallback_fired"])
def test_no_check_is_paid_for_when_the_run_cannot_succeed(monkeypatch, case):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    reply = {"director_failed": "not json",
             "fallback_fired": lambda model: _refusal() if model == mc.HYDROGEN else _slate_reply()}[case]
    store = OwnGroupStore()
    client = EmbeddingDirector(reply)
    result = private_run.run_private(store, genai_client=client, now=NOW, run_models=_models())
    assert result.exit_code == 3
    assert "[run_record] written=1" in result.lines
    assert client.embeds == [] and store.calls == ["save_run_record"]
    assert _own_group(store.records[0][1]) is None


def test_an_arm_whose_director_failed_is_not_checked(monkeypatch, capsys):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_OWN_GROUP", "1")
    store, client = OwnGroupArmStore(), OwnGroupArmClient("not json")
    code, lines, _, _, _ = _arms_main(monkeypatch, capsys, ["--arms", "previous,current", "--repeats", "1"],
                                      store=store, client=client)
    assert code == 3
    assert lines[-1] == "[private_run] records_written=2"
    # Only the previous arm's in-memory re-embedding; no own-group request.
    assert {model for model, _ in client.embeds} == {em.EMBED_001} and sum(size for _, size in client.embeds) == 8
    assert store.calls == ["save_run_record"] * 2



def test_the_arm_records_say_whether_the_chosen_candidate_was_mapped(monkeypatch, capsys):
    _photo_world(monkeypatch)
    monkeypatch.delenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", raising=False)
    monkeypatch.setattr(private_run, "_orbit_config", lambda: {"feeds": []})
    code, lines, _, store, _ = _arms_main(monkeypatch, capsys, ["--arms", "current,current_links", "--repeats", "1"])
    assert code == 0
    current, with_links = [record for _, record in store.records]
    assert current["chosen_mapped"] is None and with_links["chosen_mapped"] is False


def test_the_private_run_record_says_whether_the_chosen_candidate_was_mapped(monkeypatch):
    _photo_world(monkeypatch)
    monkeypatch.setenv("ZEITGEIST_RUN_RECORD", "1")
    monkeypatch.delenv("ZEITGEIST_LINKS", raising=False)
    monkeypatch.delenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", raising=False)
    store = RecordingStore()
    _run(store, reply=_slate_reply())
    assert store.records[-1][1]["chosen_mapped"] is None
    monkeypatch.setenv("ZEITGEIST_CHOSEN_GROUP_FIELDS", "1")
    store = RecordingStore()
    _run(store, reply=_reply(selected_cluster_id="cluster_nowhere", candidate_words=SLATE))
    [(_, record)] = store.records
    assert record["chosen_mapped"] is True and record["chosen_cluster_id"] == "cluster_0"
