"""Task G2: the private run measures both judges and writes the matched pass marks. Synthetic data only."""

from __future__ import annotations

import json
import re
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts import private_run
from src.zeitgeist import judge_calibration as jc
from src.zeitgeist import model_choice as mc
from src.zeitgeist import visual_router as vr
from tests.test_private_run import NOW, Director, ReadOnlyStore, _models, _photo, _refusal
from tests.test_understanding_pipeline import _reply

LINE = re.compile(r"\[judge_calibration\] photos=\d+ flash_pass_meaning=\d+ flash_pass_composite=\d+ "
                  r"meaning_min=\d\.\d{2} composite_min=\d\.\d{2} written=[01]")
DATE = NOW.strftime("%Y-%m-%d")


def _row(meaning, clarity=None):
    clarity = meaning if clarity is None else clarity
    return {"semantic_alignment": meaning, "compositional_clarity": clarity, "editorial_memorability": clarity,
            "safety_pass": True}


# Flash passes five photos on meaning (0.70 and up) and six on composite (the 0.69 photo scores 0.795).
FLASH_ROWS = [_row(m, c) for m, c in ((0.90, 0.90), (0.85, 0.85), (0.80, 0.80), (0.75, 0.75), (0.70, 0.70),
                                      (0.69, 0.90), (0.50, 0.50), (0.40, 0.40), (0.30, 0.30))]
# Hydrogen scores every photo higher; its composite equals its meaning.
HYDROGEN_ROWS = [_row(m) for m in (0.95, 0.92, 0.88, 0.84, 0.81, 0.78, 0.60, 0.50, 0.40)]


class Store(ReadOnlyStore):
    """The private run's two reads plus the one calibration write this task adds."""

    def __init__(self, save_error=None, **kwargs):
        super().__init__(**kwargs)
        self.saved = []
        self.save_error = save_error

    def save_calibration(self, name, document):
        if self.save_error is not None:
            raise self.save_error
        self.saved.append((name, document))


def _thumb(photo_id):
    stream = BytesIO()
    slot, index = int(photo_id[1]), int(photo_id[2:])
    Image.new("RGB", (64, 48), (40 * slot, 40 * index, 80)).save(stream, format="JPEG")
    return stream.getvalue()


def _world(monkeypatch, *, per_query=3, flash=FLASH_ROWS, hydrogen=HYDROGEN_ROWS, flash_raises=None,
           refuse_hydrogen=False, flash_blocked_batches=()):
    """Fake Unsplash and a fake critic for both models; each photo has its own thumbnail.

    Scores are dealt by the photo's position in the model's own sequence of batches. Returns the critic calls
    as (model, prompt texts, image bytes).
    """
    calls = []
    queries = {}
    rows = {mc.HYDROGEN: hydrogen, mc.FLASH: flash}

    def get(client, url, *, params=None, **kwargs):
        if url.endswith("/search/photos"):
            slot = queries.setdefault(params["query"], len(queries))
            return SimpleNamespace(status_code=200,
                                   json=lambda: {"results": [_photo(f"p{slot}{i}") for i in range(per_query)]})
        if url.startswith("https://api.unsplash.com/photos/"):
            photo_id = url.rsplit("/", 1)[-1]
            return SimpleNamespace(status_code=200, json=lambda: _photo(photo_id))
        if url.startswith("https://api.unsplash.com/"):
            raise AssertionError("unexpected Unsplash call " + url)
        photo_id = url.rsplit("/", 1)[-1].split("?")[0].replace("_small", "").replace("_raw", "")
        return SimpleNamespace(status_code=200, content=_thumb(photo_id))

    def generate(*, model, contents, config):
        texts = [part for part in contents if isinstance(part, str)]
        images = [part.inline_data.data for part in contents if not isinstance(part, str)]
        done = sum(len(call[2]) for call in calls if call[0] == model)
        calls.append((model, texts, images))
        if model == mc.FLASH and flash_raises is not None:
            raise flash_raises
        if model == mc.HYDROGEN and refuse_hydrogen:
            raise _refusal()
        dealt = rows[model][done:done + len(images)]
        batch = sum(call[0] == model for call in calls) - 1
        blocked = model == mc.FLASH and batch in flash_blocked_batches
        return SimpleNamespace(text=json.dumps({"candidates": [dict(r, index=i) for i, r in enumerate(dealt)]}),
                               candidates=[SimpleNamespace(finish_reason="SAFETY")] if blocked else [])

    monkeypatch.setattr("httpx.Client.get", get)
    monkeypatch.setattr("httpx.Client.post", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no POST")))
    monkeypatch.setattr("google.genai.Client",
                        lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    monkeypatch.setattr(vr, "_ping_unsplash_download_location",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no download tracking")))
    return calls


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in (mc.DIRECTOR_ENV, mc.JUDGE_ENV, mc.EMBEDDING_ENV, "ZEITGEIST_BEST_SAFE_PHOTO_DATE", "ZEITGEIST_OFFLINE_DEMO"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ZEITGEIST_UNDERSTANDING_V1", "1")
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("ZEITGEIST_LIVE_UNSPLASH", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "synthetic-key")


def _measure(store, run_models=None):
    """The private run, then the calibration step on its in-memory results."""
    run_models = run_models or _models(director=mc.FLASH, judge=mc.HYDROGEN)
    result = private_run.run_private(store, genai_client=Director(_reply(selected_cluster_id="cluster_0")),
                                     now=NOW, run_models=run_models)
    return result, private_run.measure_judges(store, result, run_models, DATE)


def _main(monkeypatch, capsys, store, judge=mc.HYDROGEN):
    if judge is not None:
        monkeypatch.setenv(mc.JUDGE_ENV, judge)
    monkeypatch.setattr(private_run, "_store", lambda: store)
    monkeypatch.setattr(private_run, "_genai_client", lambda: Director(_reply(selected_cluster_id="cluster_0")))
    code = private_run.main([])
    out = capsys.readouterr()
    return code, out.out.splitlines(), out.out + out.err


def test_hydrogen_run_scores_again_with_flash_and_writes_the_matched_marks(monkeypatch):
    calls = _world(monkeypatch)
    store = Store()
    result, line = _measure(store)
    assert line == ("[judge_calibration] photos=9 flash_pass_meaning=5 flash_pass_composite=6 "
                    "meaning_min=0.81 composite_min=0.78 written=1")
    assert [call[0] for call in calls] == [mc.HYDROGEN, mc.FLASH]
    assert store.saved == [("judge_calibration", {
        "schema_version": 1, "model": "gemini-4-hydrogen", "meaning_min": 0.81, "composite_min": 0.78,
        "photos": 9, "flash_pass_meaning": 5, "flash_pass_composite": 6, "date": DATE})]
    document = store.saved[0][1]
    assert all(type(document[key]) is float for key in ("meaning_min", "composite_min"))
    assert all(type(document[key]) is int for key in ("photos", "flash_pass_meaning", "flash_pass_composite"))
    assert jc.marks_for(mc.HYDROGEN, document) == (0.81, 0.78)
    assert result.exit_code == 0


def test_flash_sees_the_same_batches_and_the_same_prompt(monkeypatch):
    calls = _world(monkeypatch, per_query=6, flash=FLASH_ROWS * 2, hydrogen=HYDROGEN_ROWS * 2)
    store = Store()
    _, line = _measure(store)
    hydrogen = [call for call in calls if call[0] == mc.HYDROGEN]
    flash = [call for call in calls if call[0] == mc.FLASH]
    assert [len(call[2]) for call in hydrogen] == [9, 9]
    assert [call[1:] for call in flash] == [call[1:] for call in hydrogen]
    assert len({image for call in hydrogen for image in call[2]}) == 18
    assert line.startswith("[judge_calibration] photos=18 flash_pass_meaning=10 flash_pass_composite=12 ")
    assert line.endswith(" written=1") and len(store.saved) == 1


def test_main_prints_the_calibration_line_after_the_run_lines(monkeypatch, capsys):
    _world(monkeypatch)
    store = Store()
    code, lines, everything = _main(monkeypatch, capsys, store)
    assert len(lines) == 4 and lines[2].startswith("[private_run] photo_ok=1 ")
    assert lines[3] == ("[judge_calibration] photos=9 flash_pass_meaning=5 flash_pass_composite=6 "
                        "meaning_min=0.81 composite_min=0.78 written=1")
    assert code == 0 and len(store.saved) == 1
    for private in ("p00", "p01", "p10", "Selvedge", "Zqtitle", "Zqsummary", "unsplash"):
        assert private not in everything


def test_fewer_than_nine_photos_writes_nothing_and_spends_nothing(monkeypatch):
    calls = _world(monkeypatch, per_query=2)
    store = Store()
    result, line = _measure(store)
    assert line == ("[judge_calibration] photos=6 flash_pass_meaning=0 flash_pass_composite=0 "
                    "meaning_min=0.70 composite_min=0.68 written=0")
    assert [call[0] for call in calls] == [mc.HYDROGEN]
    assert store.saved == [] and result.exit_code == 0


@pytest.mark.parametrize("error", [
    ValueError("Zqsecret broken reply"), _refusal(), _refusal(pre=False), KeyboardInterrupt("Zqsecret"),
], ids=["error", "refused-before-dispatch", "blocked-after-dispatch", "interrupt"])
def test_a_flash_scoring_failure_prints_written_0_and_keeps_the_run_green(monkeypatch, capsys, error):
    calls = _world(monkeypatch, flash_raises=error)
    store = Store()
    code, lines, everything = _main(monkeypatch, capsys, store)
    assert lines[-1] == ("[judge_calibration] photos=9 flash_pass_meaning=0 flash_pass_composite=0 "
                         "meaning_min=0.70 composite_min=0.68 written=0")
    assert [call[0] for call in calls] == [mc.HYDROGEN, mc.FLASH]
    assert store.saved == [] and code == 0
    assert "Zqsecret" not in everything and "Traceback" not in everything


def test_a_flash_reply_that_does_not_map_the_candidates_writes_nothing(monkeypatch):
    _world(monkeypatch, flash=FLASH_ROWS[:4])
    store = Store()
    result, line = _measure(store)
    assert line.endswith(" written=0") and store.saved == [] and result.exit_code == 0


def test_a_store_write_failure_prints_written_0_and_keeps_the_run_green(monkeypatch, capsys):
    _world(monkeypatch)
    store = Store(save_error=RuntimeError("Zqsecret firestore text"))
    code, lines, everything = _main(monkeypatch, capsys, store)
    assert lines[-1] == ("[judge_calibration] photos=9 flash_pass_meaning=5 flash_pass_composite=6 "
                         "meaning_min=0.81 composite_min=0.78 written=0")
    assert code == 0 and "Zqsecret" not in everything


@pytest.mark.parametrize("flash, expected", [
    ([_row(0.10)] * 9, "flash_pass_meaning=0 flash_pass_composite=0 meaning_min=0.85 composite_min=0.85"),
    ([_row(0.99)] * 9, "flash_pass_meaning=9 flash_pass_composite=9 meaning_min=0.55 composite_min=0.55"),
], ids=["flash-passes-none", "flash-passes-all"])
def test_matched_marks_stay_within_bounds_when_flash_passes_none_or_all(monkeypatch, flash, expected):
    _world(monkeypatch, flash=flash)
    store = Store()
    _, line = _measure(store)
    assert line == f"[judge_calibration] photos=9 {expected} written=1"
    assert LINE.fullmatch(line)
    document = store.saved[0][1]
    assert jc.MARK_FLOOR <= document["meaning_min"] <= jc.MARK_CEILING
    assert jc.MARK_FLOOR <= document["composite_min"] <= jc.MARK_CEILING


def test_a_flash_judged_run_does_not_measure(monkeypatch, capsys):
    calls = _world(monkeypatch)
    store = Store()
    code, lines, _ = _main(monkeypatch, capsys, store, judge=None)
    assert len(lines) == 3 and not any(line.startswith("[judge_calibration]") for line in lines)
    assert [call[0] for call in calls] == [mc.FLASH] and store.saved == [] and code == 0


def test_hydrogen_falling_back_to_flash_mid_run_does_not_measure(monkeypatch):
    calls = _world(monkeypatch, refuse_hydrogen=True)
    store = Store()
    result, line = _measure(store)
    assert line is None and store.saved == [] and result.exit_code == 3
    assert [call[0] for call in calls] == [mc.HYDROGEN, mc.FLASH]


def test_a_failed_private_run_does_not_measure(monkeypatch):
    calls = _world(monkeypatch)
    store = Store()
    run_models = _models(director=mc.FLASH, judge=mc.HYDROGEN)
    result = private_run.run_private(store, genai_client=Director("not json"), now=NOW, run_models=run_models)
    assert private_run.measure_judges(store, result, run_models, DATE) is None
    assert calls == [] and store.saved == []


def test_the_edition_run_never_measures(tmp_path, monkeypatch):
    from pathlib import Path

    from src.state_store import InMemoryStateStore
    from src.zeitgeist import adk_pipeline
    from tests.test_model_choice_pipeline import Stop, _live

    saved = []
    monkeypatch.setattr(InMemoryStateStore, "save_calibration", lambda self, name, doc: saved.append(name))
    monkeypatch.setattr(private_run, "measure_judges",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("edition measured the judges")))
    rendered = []

    def render(**kwargs):
        rendered.append(kwargs["judge_model"])
        raise Stop()

    def recover(commission, *, render, **kw):
        render(commission, photo_only=True)

    monkeypatch.setattr("src.zeitgeist.adk_pipeline.select_and_render_pulse_assets", render)
    _live(tmp_path, monkeypatch, [_reply()], judge=mc.HYDROGEN, recover=recover)
    assert rendered == [mc.HYDROGEN] and "judge_calibration" not in saved
    source = Path(adk_pipeline.__file__).read_text()
    assert "private_run" not in source and "measure_judges" not in source


def test_no_marks_are_written_when_the_judge_scored_fewer_photos_than_flash_passes(monkeypatch):
    # Hydrogen scores every photo zero while Flash passes four: there is no fourth Hydrogen score to match.
    _world(monkeypatch, flash=[_row(0.90)] * 4 + [_row(0.30)] * 5, hydrogen=[_row(0.0)] * 9)
    store = Store()
    result, line = _measure(store)
    assert line == ("[judge_calibration] photos=9 flash_pass_meaning=4 flash_pass_composite=4 "
                    "meaning_min=0.70 composite_min=0.68 written=0")
    assert store.saved == [] and result.exit_code == 0


@pytest.mark.parametrize("per_query, blocked, photos", [(3, (0,), 9), (6, (1,), 18)],
                         ids=["only-batch-blocked", "second-batch-blocked"])
def test_no_marks_are_written_when_a_flash_batch_fails_closed(monkeypatch, per_query, blocked, photos):
    calls = _world(monkeypatch, per_query=per_query, flash=FLASH_ROWS * 2, hydrogen=HYDROGEN_ROWS * 2,
                   flash_blocked_batches=blocked)
    store = Store()
    result, line = _measure(store)
    assert line == (f"[judge_calibration] photos={photos} flash_pass_meaning=0 flash_pass_composite=0 "
                    "meaning_min=0.70 composite_min=0.68 written=0")
    assert sum(call[0] == mc.FLASH for call in calls) == len(range(0, photos, 9))
    assert store.saved == [] and result.exit_code == 0
