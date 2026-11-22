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
from src.zeitgeist.spend_guard import SpendBlocked
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
