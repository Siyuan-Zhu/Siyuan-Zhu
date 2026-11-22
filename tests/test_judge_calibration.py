"""Task G1: the critic's pass marks are values a per-model calibration can set. Synthetic data only."""

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from src.zeitgeist import judge_calibration as jc
from src.zeitgeist import model_choice as mc
from src.zeitgeist import visual_router as vr
from src.zeitgeist.spend_guard import SpendBlocked


def _photo(color):
    stream = BytesIO()
    Image.new("RGB", (1200, 800), color).save(stream, format="JPEG", quality=85)
    return stream.getvalue()


def _candidate(name, meaning, clarity, memorability, **extra):
    """A synthetic, already-scored candidate; the name travels as the photographer credit."""
    row = {
        "id": f"synthetic-{name}",
        "photographer_name": name,
        "photographer_url": f"https://example.invalid/{name}",
        "unsplash_url": f"https://example.invalid/photos/{name}",
        "download_location": f"https://example.invalid/photos/{name}/download",
        "exif": {"make": "Fixture Camera", "model": "Fixture camera model"},
        "image_bytes": _photo((30 + len(name), 60, 90)),
        "semantic_alignment": meaning,
        "compositional_clarity": clarity,
        "editorial_memorability": memorability,
        "safety_pass": True,
    }
    row.update(extra)
    return row


_COMMISSION = {"word": "Zqword", "gloss": "A synthetic gloss.", "commissioned_thesis": "A synthetic thesis.",
               "unsplash_queries": ["synthetic query"]}


def _winner(tmp_path, candidates, **kwargs):
    """The credited photographer of the chosen photo, or None when the gate passes nothing."""
    try:
        pulse = vr.select_and_render_pulse_assets(
            commission=_COMMISSION, repo_root=tmp_path, date_str="2026-11-22", candidates=candidates,
            requested_medium="photo", photo_history_30d=[], **kwargs)
    except vr.PhotoSelectionUnavailable:
        return None
    return pulse.attribution["photographer_name"]


# Composite = 0.50 meaning + 0.30 clarity + 0.20 memorability.
# "edge": meaning 0.70, composite 0.70 -- passes today's marks exactly at the meaning edge.
# "lowmeaning": meaning 0.69, composite 0.79 -- fails today's meaning mark only.
# "lowcomposite": meaning 0.80, composite 0.67 -- fails today's composite mark only.
# "middling": meaning 0.65, composite 0.64 -- fails both of today's marks.
EDGE = ("edge", 0.70, 0.70, 0.70)
LOW_MEANING = ("lowmeaning", 0.69, 0.90, 0.90)
LOW_COMPOSITE = ("lowcomposite", 0.80, 0.50, 0.60)
MIDDLING = ("middling", 0.65, 0.65, 0.60)


def test_default_gate_passes_at_today_marks_and_rejects_below(tmp_path):
    assert _winner(tmp_path, [_candidate(*EDGE), _candidate(*LOW_MEANING)]) == "edge"
    assert _winner(tmp_path, [_candidate(*LOW_MEANING), _candidate(*LOW_COMPOSITE), _candidate(*MIDDLING)]) is None


def test_default_gate_ranks_passing_photos_by_composite(tmp_path):
    strong = ("strong", 0.80, 0.90, 0.90)  # composite 0.85
    assert _winner(tmp_path, [_candidate(*EDGE), _candidate(*strong)]) == "strong"


# ---- matched_marks ----

def _pairs(meanings, composites):
    return list(zip(meanings, composites))


# Twelve synthetic photos. Flash passes 4 on meaning (>= 0.70) and 3 on composite (>= 0.68).
FLASH_12 = _pairs([0.91, 0.80, 0.75, 0.70, 0.69, 0.66, 0.60, 0.55, 0.50, 0.42, 0.30, 0.20],
                  [0.88, 0.71, 0.68, 0.67, 0.66, 0.60, 0.58, 0.52, 0.49, 0.40, 0.31, 0.22])
# The same photos under the judge, in the same order. Sorted meanings: 0.83 0.79 0.70 0.6789 0.65 ...
# Sorted composites: 0.81 0.77 0.6459 0.63 ...
JUDGE_12 = _pairs([0.70, 0.83, 0.65, 0.79, 0.6789, 0.58, 0.40, 0.33, 0.31, 0.25, 0.20, 0.10],
                  [0.6459, 0.81, 0.77, 0.63, 0.61, 0.57, 0.50, 0.44, 0.38, 0.30, 0.21, 0.12])


def test_constants_are_today_marks_and_spec_bounds():
    assert jc.DEFAULT_MARKS == (0.70, 0.68)
    assert (jc.MARK_FLOOR, jc.MARK_CEILING, jc.MIN_PHOTOS) == (0.55, 0.85, 9)


def test_matched_marks_take_the_kth_highest_judge_score_rounded_down():
    # Meaning: k = 4, 4th highest judge meaning = 0.6789 -> 0.67. Composite: k = 3, 3rd highest = 0.6459 -> 0.64.
    assert jc.matched_marks(FLASH_12, JUDGE_12) == (0.67, 0.64)


def test_matched_marks_when_flash_passes_none_sits_just_above_the_best_judge_score():
    flash = _pairs([0.60] * 10, [0.50] * 10)
    judge = _pairs([0.723, 0.70, 0.66, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], [0.641] + [0.6] * 9)
    assert jc.matched_marks(flash, judge) == (0.73, 0.65)


def test_matched_marks_when_flash_passes_all_take_the_lowest_judge_score():
    flash = _pairs([0.95] * 10, [0.90] * 10)
    judge = _pairs([0.80, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.612],
                   [0.80, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.597])
    assert jc.matched_marks(flash, judge) == (0.61, 0.59)


def test_matched_marks_clamp_at_the_ceiling_and_the_floor():
    none_pass = _pairs([0.10] * 9, [0.10] * 9)
    high_judge = _pairs([0.97] * 9, [0.96] * 9)
    assert jc.matched_marks(none_pass, high_judge) == (0.85, 0.85)
    all_pass = _pairs([0.99] * 9, [0.99] * 9)
    low_judge = _pairs([0.30] * 9, [0.12] * 9)
    assert jc.matched_marks(all_pass, low_judge) == (0.55, 0.55)


def test_matched_marks_refuse_a_match_on_a_zero_judge_score():
    # Flash passes four on meaning, but the judge scored only three photos above zero: no valid fourth mark.
    flash = _pairs([0.90] * 4 + [0.30] * 5, [0.30] * 9)
    judge = _pairs([0.95, 0.90, 0.85] + [0.0] * 6, [0.60] * 9)
    assert jc.matched_marks(flash, judge) is None
    # The same on the composite axis alone.
    flash = _pairs([0.30] * 9, [0.90] * 2 + [0.30] * 7)
    judge = _pairs([0.60] * 9, [0.80] + [0.0] * 8)
    assert jc.matched_marks(flash, judge) is None
    # A zero judge score below the k-th position does not matter.
    flash = _pairs([0.90] * 2 + [0.30] * 7, [0.30] * 9)
    judge = _pairs([0.95, 0.90] + [0.0] * 7, [0.60] * 9)
    assert jc.matched_marks(flash, judge) == (0.85, 0.61)


def test_matched_marks_refuse_a_judge_whose_best_score_is_zero_even_when_flash_passes_none():
    # Every judge batch blocked (all scores zero) and Flash passes nothing: the floor marks would be invented.
    flash = _pairs([0.10] * 9, [0.10] * 9)
    assert jc.matched_marks(flash, _pairs([0.0] * 9, [0.0] * 9)) is None
    assert jc.matched_marks(flash, _pairs([-0.2] * 9, [-0.1] * 9)) is None
    # Either axis alone with no score above zero is enough to refuse.
    assert jc.matched_marks(flash, _pairs([0.0] * 9, [0.60] * 9)) is None
    assert jc.matched_marks(flash, _pairs([0.60] * 9, [0.0] * 9)) is None


def test_matched_marks_need_nine_photos():
    assert jc.matched_marks(FLASH_12[:8], JUDGE_12[:8]) is None
    assert jc.matched_marks([], []) is None
    assert jc.matched_marks(FLASH_12[:9], JUDGE_12[:9]) is not None


def test_matched_marks_refuse_unpaired_lists():
    assert jc.matched_marks(FLASH_12, JUDGE_12[:11]) is None


# ---- build_document and marks_for ----


def _document(**overrides):
    doc = jc.build_document(mc.HYDROGEN, (0.60, 0.58),
                            {"photos": 12, "flash_pass_meaning": 4, "flash_pass_composite": 3}, "2026-11-22")
    doc.update(overrides)
    return doc


def test_build_document_holds_numbers_and_the_model_name_only():
    assert _document() == {
        "schema_version": 1, "model": mc.HYDROGEN, "meaning_min": 0.60, "composite_min": 0.58,
        "photos": 12, "flash_pass_meaning": 4, "flash_pass_composite": 3, "date": "2026-11-22",
    }
    json.dumps(_document())


def test_marks_for_a_good_document_returns_its_marks():
    assert jc.marks_for(mc.HYDROGEN, _document()) == (0.60, 0.58)


def test_marks_for_another_model_returns_today_marks():
    assert jc.marks_for(mc.FLASH, _document()) == jc.DEFAULT_MARKS
    assert jc.marks_for(mc.HYDROGEN, _document(model=mc.FLASH)) == jc.DEFAULT_MARKS


@pytest.mark.parametrize("overrides", [
    {"meaning_min": 0.54}, {"meaning_min": 0.86}, {"composite_min": 0.50}, {"composite_min": 0.90},
    {"meaning_min": float("nan")}, {"composite_min": float("inf")},
], ids=["meaning-low", "meaning-high", "composite-low", "composite-high", "nan", "inf"])
def test_marks_for_an_out_of_range_mark_returns_today_marks(overrides):
    assert jc.marks_for(mc.HYDROGEN, _document(**overrides)) == jc.DEFAULT_MARKS


class _Exploding(dict):
    def get(self, *args, **kwargs):
        raise RuntimeError("synthetic")


@pytest.mark.parametrize("document", [
    None, "not a document", [], 0.6, {},
    {"model": mc.HYDROGEN, "meaning_min": 0.6, "composite_min": 0.58},
    "schema", "mark-text", "mark-bool", "photos-few", "photos-missing", "exploding",
], ids=["none", "string", "list", "number", "empty", "no-schema", "schema-2", "mark-text", "mark-bool",
        "photos-few", "photos-missing", "exploding"])
def test_marks_for_a_malformed_document_returns_today_marks_without_raising(document):
    document = {
        "schema": _document(schema_version=2),
        "mark-text": _document(meaning_min="0.6"),
        "mark-bool": _document(composite_min=True),
        "photos-few": _document(photos=8),
        "photos-missing": {k: v for k, v in _document().items() if k != "photos"},
        "exploding": _Exploding(_document()),
    }.get(document, document) if isinstance(document, str) else document
    assert jc.marks_for(mc.HYDROGEN, document) == jc.DEFAULT_MARKS


def test_load_document_returns_none_when_the_store_fails():
    class Broken:
        def load_calibration(self, name):
            raise ValueError("synthetic corrupt document")

    assert jc.load_document(Broken()) is None
    assert jc.load_document(object()) is None


def test_load_document_reads_the_judge_calibration_by_name():
    from src.state_store import InMemoryStateStore

    store = InMemoryStateStore()
    assert jc.load_document(store) is None
    store.save_calibration("judge_calibration", _document())
    assert jc.load_document(store) == _document()


# ---- the gate reads its marks from a parameter ----


def test_gate_applies_marks_passed_as_values(tmp_path):
    assert _winner(tmp_path, [_candidate(*MIDDLING)], pass_marks=(0.60, 0.58)) == "middling"
    assert _winner(tmp_path, [_candidate(*EDGE)], pass_marks=(0.71, 0.58)) is None


def test_gate_with_default_marks_parameter_matches_the_default_path(tmp_path):
    rows = [_candidate(*EDGE), _candidate(*LOW_MEANING), _candidate(*LOW_COMPOSITE)]
    assert _winner(tmp_path, rows, pass_marks=jc.DEFAULT_MARKS) == _winner(tmp_path, rows) == "edge"


def test_lowered_marks_do_not_change_the_ranking(tmp_path):
    # Both pass at 0.60/0.58; the higher composite (0.79 > 0.64) wins as today.
    assert _winner(tmp_path, [_candidate(*MIDDLING), _candidate(*LOW_MEANING)], pass_marks=(0.60, 0.58)) == "lowmeaning"


# ---- Review Focus 5: a mid-run fallback to Flash applies Flash's marks ----


def _refusal():
    error = SpendBlocked("synthetic refusal")
    error.refused_before_dispatch = True
    return error


def _critic(monkeypatch, *, refuse_hydrogen):
    """A fake critic: every model scores the one candidate as middling (0.65 meaning, 0.64 composite)."""
    calls = []

    def generate(*, model, contents, config):
        calls.append(model)
        if refuse_hydrogen and model == mc.HYDROGEN:
            raise _refusal()
        row = {"index": 0, "semantic_alignment": 0.65, "compositional_clarity": 0.65,
               "editorial_memorability": 0.60, "safety_pass": True}
        return SimpleNamespace(text=json.dumps({"candidates": [row]}), candidates=[])

    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("ZEITGEIST_LIVE_UNSPLASH", "0")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setattr("google.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    return calls


def _unscored(name):
    row = _candidate(name, 0.0, 0.0, 0.0, _requires_critic=True)
    row["thumb_bytes"] = row["image_bytes"]
    return row


@pytest.mark.parametrize("refuse, winner, judge_after", [
    (False, "unscored", mc.HYDROGEN),
    (True, None, mc.FLASH),
], ids=["hydrogen-answers", "hydrogen-refused-mid-run"])
def test_marks_follow_the_judge_actually_used_at_gate_time(tmp_path, monkeypatch, refuse, winner, judge_after):
    calls = _critic(monkeypatch, refuse_hydrogen=refuse)
    rm = mc.resolve_run_models({}, {mc.JUDGE_ENV: mc.HYDROGEN})
    document = _document()  # Hydrogen marks 0.60/0.58 would pass the middling photo; Flash's would not.
    with mc.activate(rm):
        chosen = _winner(tmp_path, [_unscored("unscored")], judge_model=rm.current("judge"),
                         pass_marks=lambda: jc.marks_for(rm.current("judge"), document))
    assert chosen == winner
    assert rm.current("judge") == judge_after and rm.fallback_fired is refuse
    assert calls == ([mc.HYDROGEN, mc.FLASH] if refuse else [mc.HYDROGEN])


# ---- pipeline wiring ----


def _pipeline_marks(tmp_path, monkeypatch, *, judge=None, director=None, document=None, load_error=False,
                    refuse_judge_in_render=False, replies=None):
    """Runs the live pipeline to the renderer; returns (marks resolved at the gate, calibration names loaded)."""
    from src.state_store import InMemoryStateStore
    from tests.test_model_choice_pipeline import Stop, _live, _reply

    loaded = []

    def load_calibration(self, name):
        loaded.append(name)
        if load_error:
            raise ValueError("synthetic corrupt document")
        return document if name == "judge_calibration" else None

    monkeypatch.setattr(InMemoryStateStore, "load_calibration", load_calibration)
    seen = []

    def render(**kwargs):
        if refuse_judge_in_render:
            mc.active().call("judge", lambda m: (_ for _ in ()).throw(_refusal()) if m == mc.HYDROGEN else None)
        marks = kwargs.get("pass_marks", jc.DEFAULT_MARKS)
        seen.append(marks() if callable(marks) else marks)
        raise Stop()

    def recover(commission, *, render, **kw):
        render(commission, photo_only=True)

    monkeypatch.setattr("src.zeitgeist.adk_pipeline.select_and_render_pulse_assets", render)
    _live(tmp_path, monkeypatch, replies or [_reply()], director=director, judge=judge, recover=recover)
    return seen, loaded


def test_pipeline_hydrogen_judge_uses_the_stored_hydrogen_marks(tmp_path, monkeypatch):
    seen, loaded = _pipeline_marks(tmp_path, monkeypatch, judge=mc.HYDROGEN, document=_document())
    assert seen == [(0.60, 0.58)] and loaded == ["judge_calibration"]


def test_pipeline_hydrogen_judge_without_a_document_uses_today_marks(tmp_path, monkeypatch):
    assert _pipeline_marks(tmp_path, monkeypatch, judge=mc.HYDROGEN)[0] == [jc.DEFAULT_MARKS]


def test_pipeline_load_error_uses_today_marks_and_does_not_raise(tmp_path, monkeypatch):
    seen, loaded = _pipeline_marks(tmp_path, monkeypatch, judge=mc.HYDROGEN, load_error=True)
    assert seen == [jc.DEFAULT_MARKS] and loaded == ["judge_calibration"]


def test_pipeline_document_naming_another_model_uses_today_marks(tmp_path, monkeypatch):
    seen, _ = _pipeline_marks(tmp_path, monkeypatch, judge=mc.HYDROGEN, document=_document(model=mc.FLASH))
    assert seen == [jc.DEFAULT_MARKS]


def test_pipeline_flash_judge_does_not_read_the_document(tmp_path, monkeypatch):
    seen, loaded = _pipeline_marks(tmp_path, monkeypatch, document=_document(model=mc.FLASH))
    assert seen == [jc.DEFAULT_MARKS] and loaded == []


def test_pipeline_judge_fallback_during_render_applies_flash_marks(tmp_path, monkeypatch):
    seen, _ = _pipeline_marks(tmp_path, monkeypatch, judge=mc.HYDROGEN, document=_document(),
                              refuse_judge_in_render=True)
    assert seen == [jc.DEFAULT_MARKS]


def test_pipeline_director_fallback_before_render_applies_flash_marks(tmp_path, monkeypatch):
    from tests.test_model_choice_pipeline import _refuse_hydrogen, _reply

    seen, loaded = _pipeline_marks(tmp_path, monkeypatch, judge=mc.HYDROGEN, director=mc.HYDROGEN,
                                   document=_document(), replies=[_refuse_hydrogen(), _reply()])
    assert seen == [jc.DEFAULT_MARKS] and loaded == []
