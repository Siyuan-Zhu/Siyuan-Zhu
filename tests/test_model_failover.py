"""Task T3: a failing or unusable non-default model hands the run to 3.8 Flash. Synthetic data only."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from src.zeitgeist import judge_calibration
from src.zeitgeist import model_choice as mc
from src.zeitgeist import visual_router as vr
from src.zeitgeist.adk_pipeline import commission_daily_zeitgeist
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_model_choice_pipeline import _client, _lines, _live, _refusal, _thumb
from tests.test_understanding_pipeline import _clusters, _reply

NOW = "2026-11-21T12:00:00Z"
HISTORY = ["Entropy", "Tessellation"]
# sha256 of the calls and result of an all-malformed Flash run, taken before Task T3.
FLASH_LEGACY_SHA256 = "89cfbbffe7d8d3c6cd749a19ba8580ee3a52562f5405ce06463972ed06948160"
FLASH_PACKET_SHA256 = "111b9d78b233708a694d7d1f64ef6785bd4ea92bc5a825e49bf5e5e8f6df7167"


def _transport(model):
    if model == mc.HYDROGEN:
        return RuntimeError("synthetic transport failure")
    return _reply()


def _hydrogen():
    return mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN, mc.JUDGE_ENV: mc.HYDROGEN})


# ---- RunModels ----


def test_any_failure_on_a_non_default_model_moves_the_run_to_flash():
    rm = _hydrogen()
    seen = []

    def fn(model):
        seen.append(model)
        if model == mc.HYDROGEN:
            raise RuntimeError("synthetic transport failure")
        return "ok"

    assert rm.call("director", fn) == "ok"
    assert seen == [mc.HYDROGEN, mc.FLASH]
    assert rm.fallback_fired and rm.current("director") == mc.FLASH and rm.current("judge") == mc.FLASH


def test_flash_failure_propagates_after_one_call():
    rm = mc.resolve_run_models({}, {})
    calls = []

    def fn(model):
        calls.append(model)
        raise RuntimeError("synthetic transport failure")

    with pytest.raises(RuntimeError):
        rm.call("director", fn)
    assert calls == [mc.FLASH] and not rm.fallback_fired


def test_unmarked_spend_block_still_propagates_from_a_non_default_model():
    rm = _hydrogen()
    calls = []

    def fn(model):
        calls.append(model)
        raise _refusal(pre=False)

    with pytest.raises(SpendBlocked):
        rm.call("judge", fn)
    assert calls == [mc.HYDROGEN] and not rm.fallback_fired


def test_embedding_role_failure_still_propagates():
    rm = mc.resolve_run_models({}, {mc.EMBEDDING_ENV: mc.EMBED_2})
    with pytest.raises(RuntimeError):
        rm.call("embedding", lambda model: (_ for _ in ()).throw(RuntimeError("synthetic")))
    assert not rm.fallback_fired and rm.embedding == mc.EMBED_2


def test_give_up_moves_a_non_default_run_to_flash_once():
    rm = mc.resolve_run_models({}, {mc.JUDGE_ENV: mc.HYDROGEN})
    assert rm.give_up("judge") is True
    assert rm.fallback_fired and rm.current("judge") == mc.FLASH and rm.current("director") == mc.FLASH
    assert rm.give_up("judge") is False and rm.give_up("director") is False


def test_give_up_on_flash_changes_nothing():
    rm = mc.resolve_run_models({}, {})
    assert rm.give_up("director") is False
    assert not rm.fallback_fired and rm.current("director") == mc.FLASH


# ---- Director ----


def test_hydrogen_director_transport_error_is_answered_by_one_flash_call():
    client = _client([_transport])
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, genai_client=client, run_models=rm)
    calls = client.models.calls
    assert [c["model"] for c in calls] == [mc.HYDROGEN, mc.FLASH]
    assert calls[0]["contents"] == calls[1]["contents"]
    assert commission.word == "Selvedge" and not commission.fallback_used and commission.attempts_used == 1
    assert rm.fallback_fired and "fallback=1" in mc.model_choice_line(rm, commission.slate_count)


def test_hydrogen_director_truncated_on_every_attempt_runs_the_flash_legacy_loop():
    truncated = _reply()[:40]
    replies = [truncated] * 5 + [_reply()]
    client = _client(replies)
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, genai_client=client, run_models=rm,
                                            evidence_packet=True, now_utc=NOW)
    calls = client.models.calls
    assert [c["model"] for c in calls] == [mc.HYDROGEN] * 5 + [mc.FLASH]
    legacy_prompt = calls[2]["contents"]
    assert calls[5]["contents"] != calls[0]["contents"]
    assert "candidate_words" not in calls[5]["contents"]
    assert "Malformed director response" not in calls[5]["contents"]
    assert calls[5]["contents"].split("\n")[0] == legacy_prompt.split("\n")[0]
    assert commission.word == "Selvedge" and not commission.fallback_used and not commission.packet_used
    assert commission.attempts_used == 6 and commission.slate_count == 0
    assert rm.fallback_fired and rm.current("judge") == mc.FLASH


def test_flash_legacy_round_runs_in_full_then_exhausts_as_today():
    client = _client(["not json"] * 6)
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, genai_client=client, run_models=rm)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN] * 3 + [mc.FLASH] * 3
    assert client.models.calls[3]["contents"] == client.models.calls[0]["contents"]
    assert commission.fallback_used and rm.fallback_fired


def test_unmarked_spend_block_from_hydrogen_director_propagates_without_flash_call():
    client = _client([lambda model: _refusal(pre=False)])
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    with pytest.raises(SpendBlocked):
        commission_daily_zeitgeist(_clusters(), HISTORY, genai_client=client, run_models=rm)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN]
    assert not rm.fallback_fired


def test_flash_director_transport_error_propagates_as_today():
    client = _client([RuntimeError("synthetic transport failure")])
    rm = mc.resolve_run_models({}, {})
    with pytest.raises(RuntimeError):
        commission_daily_zeitgeist(_clusters(), HISTORY, genai_client=client, run_models=rm)
    assert [c["model"] for c in client.models.calls] == [mc.FLASH]
    assert not rm.fallback_fired


@pytest.mark.parametrize("packet, expected_calls, digest", [
    (False, 3, FLASH_LEGACY_SHA256),
    (True, 5, FLASH_PACKET_SHA256),
], ids=["legacy", "packet"])
def test_variables_unset_calls_and_result_are_byte_identical(packet, expected_calls, digest):
    client = _client(["not json"] * 9)
    rm = mc.resolve_run_models({}, {})
    kwargs = {"evidence_packet": True, "now_utc": NOW} if packet else {}
    commission = commission_daily_zeitgeist(_clusters(), HISTORY, genai_client=client, run_models=rm, **kwargs)
    blob = json.dumps({"calls": client.models.calls, "commission": commission.to_dict(),
                       "attempts": commission.attempts_used, "fallback_used": commission.fallback_used,
                       "fired": rm.fallback_fired}, sort_keys=True, default=str)
    assert len(client.models.calls) == expected_calls
    assert hashlib.sha256(blob.encode()).hexdigest() == digest


def test_pipeline_line_shows_fallback_after_a_hydrogen_transport_error(tmp_path, monkeypatch, capsys):
    client, _ = _live(tmp_path, monkeypatch, [_transport], director=mc.HYDROGEN)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN, mc.FLASH]
    lines = _lines(capsys)
    assert len(lines) == 1 and "director=gemini-3.8-flash judge=gemini-3.8-flash" in lines[0]
    assert "fallback=1" in lines[0]


# ---- Judge ----


_SCORES = {"semantic_alignment": 0.9, "compositional_clarity": 0.9, "editorial_memorability": 0.9, "safety_pass": True}


def _good_batch(count):
    return json.dumps({"candidates": [dict(_SCORES, index=i) for i in range(count)]})


def _judge_client(monkeypatch, hydrogen_reply, flash_reply=None):
    calls = []

    def generate(*, model, contents, config):
        calls.append({"model": model, "contents": list(contents)})
        count = sum(1 for part in contents if isinstance(part, str) and part.startswith("Candidate index"))
        reply = hydrogen_reply if model == mc.HYDROGEN else (flash_reply or (lambda n: (_good_batch(n), None)))
        if isinstance(reply, BaseException):
            raise reply
        text, finish = reply(count)
        return SimpleNamespace(text=text, candidates=[SimpleNamespace(finish_reason=finish)] if finish else [])

    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setattr("google.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    return calls


def _two_batch():
    return vr._score_photo_batch([{"thumb_bytes": _thumb()}, {"thumb_bytes": _thumb()}], word="ANCHOR",
                                 gloss="Rooted craft", thesis="Physical stability", model_name=mc.FLASH,
                                 strict_errors=True)


UNUSABLE = {
    "bad-json": lambda n: ('{"candidates": [', None),
    "wrong-count": lambda n: (_good_batch(n - 1), None),
    "index-mismatch": lambda n: (json.dumps({"candidates": [dict(_SCORES, index=i + 1) for i in range(n)]}), None),
    "malformed-scores": lambda n: (json.dumps({"candidates": [dict(_SCORES, index=i, safety_pass="yes")
                                                              for i in range(n)]}), None),
    "safety-block": lambda n: (_good_batch(n), "SAFETY"),
}


@pytest.mark.parametrize("reply", list(UNUSABLE.values()), ids=list(UNUSABLE))
def test_unusable_hydrogen_batch_is_scored_once_on_flash(monkeypatch, reply):
    calls = _judge_client(monkeypatch, reply)
    rm = _hydrogen()
    with mc.activate(rm):
        scores = _two_batch()
    assert [c["model"] for c in calls] == [mc.HYDROGEN, mc.FLASH]
    assert calls[0]["contents"] == calls[1]["contents"]
    assert [s["safety_pass"] for s in scores] == [True, True]
    assert rm.fallback_fired and rm.current("director") == mc.FLASH
    assert "fallback=1" in mc.model_choice_line(rm, 0)


def test_flash_rescore_that_is_also_unusable_applies_todays_strict_behaviour(monkeypatch):
    calls = _judge_client(monkeypatch, UNUSABLE["bad-json"], flash_reply=UNUSABLE["bad-json"])
    rm = _hydrogen()
    with mc.activate(rm), pytest.raises(vr.PhotoCriticUnavailable):
        _two_batch()
    assert [c["model"] for c in calls] == [mc.HYDROGEN, mc.FLASH]


def test_hydrogen_judge_transport_error_is_scored_on_flash(monkeypatch):
    calls = _judge_client(monkeypatch, RuntimeError("synthetic transport failure"))
    rm = _hydrogen()
    with mc.activate(rm):
        scores = _two_batch()
    assert [c["model"] for c in calls] == [mc.HYDROGEN, mc.FLASH]
    assert [s["safety_pass"] for s in scores] == [True, True] and rm.fallback_fired


def test_hydrogen_single_critic_transport_error_is_scored_on_flash(monkeypatch):
    def generate(*, model, contents, config):
        calls.append(model)
        if model == mc.HYDROGEN:
            raise RuntimeError("synthetic transport failure")
        return SimpleNamespace(text=json.dumps(_SCORES), candidates=[])

    calls = []
    _judge_client(monkeypatch, None)
    monkeypatch.setattr("google.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    rm = _hydrogen()
    with mc.activate(rm):
        verdict = vr._score_candidate_with_gemini_vision(_thumb(), "ANCHOR", "Rooted craft", "Physical stability",
                                                         model_name=mc.HYDROGEN)
    assert calls == [mc.HYDROGEN, mc.FLASH] and verdict["safety_pass"] is True and rm.fallback_fired


@pytest.mark.parametrize("reply", [UNUSABLE["bad-json"], UNUSABLE["wrong-count"]], ids=["bad-json", "wrong-count"])
def test_flash_judge_unusable_batch_raises_after_one_call_as_today(monkeypatch, reply):
    calls = _judge_client(monkeypatch, None, flash_reply=reply)
    rm = mc.resolve_run_models({}, {})
    with mc.activate(rm), pytest.raises(vr.PhotoCriticUnavailable):
        _two_batch()
    assert [c["model"] for c in calls] == [mc.FLASH] and not rm.fallback_fired


def test_flash_judge_safety_block_fails_closed_after_one_call_as_today(monkeypatch):
    calls = _judge_client(monkeypatch, None, flash_reply=UNUSABLE["safety-block"])
    rm = mc.resolve_run_models({}, {})
    with mc.activate(rm):
        scores = _two_batch()
    assert [c["model"] for c in calls] == [mc.FLASH]
    assert [s["safety_pass"] for s in scores] == [False, False] and not rm.fallback_fired


def test_unusable_hydrogen_batch_without_strict_errors_is_not_rescored(monkeypatch):
    calls = _judge_client(monkeypatch, UNUSABLE["bad-json"])
    rm = _hydrogen()
    with mc.activate(rm):
        scores = vr._score_photo_batch([{"thumb_bytes": _thumb()}], word="ANCHOR", gloss="Rooted craft",
                                       thesis="Physical stability", model_name=mc.HYDROGEN, strict_errors=False)
    assert [c["model"] for c in calls] == [mc.HYDROGEN]
    assert scores[0]["safety_pass"] is False and not rm.fallback_fired


def test_unmarked_spend_block_from_hydrogen_judge_propagates_without_flash_call(monkeypatch):
    calls = _judge_client(monkeypatch, _refusal(pre=False))
    rm = _hydrogen()
    with mc.activate(rm), pytest.raises(SpendBlocked):
        _two_batch()
    assert [c["model"] for c in calls] == [mc.HYDROGEN] and not rm.fallback_fired


# ---- Photo stage: after a hand-over to Flash, Flash judges every batch (Task T6) ----


_GREY_STEP = 9


def _grey(k):
    """A synthetic thumbnail whose grey level names candidate k, so the fake judge can tell batches apart."""
    stream = BytesIO()
    Image.new("RGB", (48, 32), (_GREY_STEP * k,) * 3).save(stream, format="JPEG")
    return stream.getvalue()


def _candidate_numbers(contents):
    numbers = []
    for part in contents:
        data = getattr(getattr(part, "inline_data", None), "data", None)
        if data:
            with Image.open(BytesIO(data)) as image:
                numbers.append(round(image.convert("L").getpixel((0, 0)) / _GREY_STEP))
    return numbers


def _scores(meaning, other=None):
    other = meaning if other is None else other
    return {"semantic_alignment": meaning, "compositional_clarity": other, "editorial_memorability": other,
            "safety_pass": True}


def _stage_judge(monkeypatch, scores, broken=lambda model, numbers: None):
    """A fake critic: scores(model, k) per candidate; broken(model, numbers) may fail a whole batch."""
    calls = []

    def generate(*, model, contents, config):
        numbers = _candidate_numbers(contents)
        calls.append((model, numbers))
        trouble = broken(model, numbers)
        if isinstance(trouble, BaseException):
            raise trouble
        if trouble is not None:
            return SimpleNamespace(text=trouble, candidates=[])
        rows = [dict(scores(model, k), index=i) for i, k in enumerate(numbers)]
        return SimpleNamespace(text=json.dumps({"candidates": rows}), candidates=[])

    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setattr("google.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    return calls


def _photo(photo_id):
    return {"id": photo_id, "urls": {"small": f"https://images.unsplash.com/{photo_id}_small",
                                     "raw": f"https://images.unsplash.com/{photo_id}_raw"},
            "links": {"html": f"https://unsplash.com/photos/{photo_id}",
                      "download_location": f"https://api.unsplash.com/photos/{photo_id}/download"},
            "user": {"name": "Synthetic Photographer", "links": {"html": "https://unsplash.com/@synthetic"}},
            "exif": {"make": "Synthetic Camera", "model": "Fixture"}}


def _unsplash(monkeypatch, per_query=5):
    """Two synthetic queries of per_query photos each: candidate k is photo q{k}, thumbnail grey k."""
    numbers = {}

    def get(client, url, *, params=None, **kwargs):
        if url.endswith("/search/photos"):
            query = params["query"]
            items = [_photo(f"{query}{i}") for i in range(per_query)]
            for item in items:
                numbers.setdefault(item["id"], len(numbers))
            return SimpleNamespace(status_code=200, json=lambda: {"results": items})
        photo_id = url.rsplit("/", 1)[-1].removesuffix("_small")
        return SimpleNamespace(status_code=200, content=_grey(numbers[photo_id]))

    monkeypatch.setattr("httpx.Client.get", get)
    monkeypatch.setenv("ZEITGEIST_LIVE_UNSPLASH", "1")
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "synthetic-unsplash-key")
    return numbers


class _Chosen(Exception):
    pass


def _photo_stage(monkeypatch, tmp_path, rm, document):
    """One live photo stage on two batches (9 + 1) with the pipeline's marks; returns (chosen, rows, marks)."""
    _unsplash(monkeypatch)
    seen_marks = []

    def pass_marks():
        seen_marks.append(judge_calibration.marks_for(rm.current("judge"), document))
        return seen_marks[-1]

    def chosen(candidate, **kwargs):
        raise _Chosen(candidate["id"])

    monkeypatch.setattr(vr, "_download_winning_unsplash_photo", chosen)
    rows = []
    with mc.activate(rm):
        try:
            vr.select_and_render_pulse_assets({"word": "ANCHOR", "gloss": "Rooted craft", "commissioned_thesis": "Stability",
                                               "unsplash_queries": ["a", "b"]},
                                              repo_root=tmp_path, date_str="2026-11-21", requested_medium="photo",
                                              photo_history_30d=[], judge_model=rm.current("judge"),
                                              strict_photo_errors=True, scored_candidates_out=rows, pass_marks=pass_marks)
        except _Chosen as picked:
            return str(picked), rows, seen_marks
        except vr.PhotoSelectionUnavailable:
            return None, rows, seen_marks
    raise AssertionError("the stage neither chose nor rejected")


def _calibration(marks):
    return judge_calibration.build_document(mc.HYDROGEN, marks, {"photos": 18, "flash_pass_meaning": 4,
                                                                "flash_pass_composite": 4}, "2026-11-20")


def _flash_table(model, k):
    # Candidate 0: Flash meaning 0.66 (below 0.70) with a high composite; candidate 1: Flash meaning 0.75 and
    # composite 0.685 (at least 0.68). The rest fail under any mark here.
    return {0: _scores(0.66, 0.95), 1: _scores(0.75, 0.62)}.get(k, _scores(0.2))


BATCH_1, BATCH_2 = list(range(9)), [9]

HYDROGEN_SAID = {
    # Hydrogen marks above Flash's: Hydrogen loved candidate 0 and hated candidate 1.
    "hydrogen-marks-above": ((0.80, 0.78), {0: _scores(0.90), 1: _scores(0.10)}),
    # Hydrogen marks below Flash's: candidate 0 would pass at Hydrogen's marks on Flash's own score.
    "hydrogen-marks-below": ((0.60, 0.58), {0: _scores(0.62), 1: _scores(0.10)}),
}

BATCH_2_FAILURES = {
    "bad-json": lambda: '{"candidates": [',
    "transport": lambda: RuntimeError("synthetic transport failure"),
    "refused-before-dispatch": lambda: _refusal(pre=True),
}


def _hydrogen_fails_batch_2(failure):
    return lambda model, numbers: failure() if model == mc.HYDROGEN and numbers == BATCH_2 else None


def _two_tables(hydrogen_scores):
    return lambda model, k: _flash_table(model, k) if model == mc.FLASH else hydrogen_scores.get(k, _scores(0.2))


@pytest.mark.parametrize("failure", list(BATCH_2_FAILURES.values()), ids=list(BATCH_2_FAILURES))
@pytest.mark.parametrize("marks, hydrogen_scores", list(HYDROGEN_SAID.values()), ids=list(HYDROGEN_SAID))
def test_after_a_hand_over_flash_scores_and_marks_apply_to_the_whole_stage(monkeypatch, tmp_path, marks,
                                                                           hydrogen_scores, failure):
    calls = _stage_judge(monkeypatch, _two_tables(hydrogen_scores), _hydrogen_fails_batch_2(failure))
    rm = _hydrogen()
    chosen, rows, seen_marks = _photo_stage(monkeypatch, tmp_path, rm, _calibration(marks))
    assert calls == [(mc.HYDROGEN, BATCH_1), (mc.HYDROGEN, BATCH_2), (mc.FLASH, BATCH_2), (mc.FLASH, BATCH_1)]
    assert [row["semantic_alignment"] for row in rows] == [_flash_table(mc.FLASH, k)["semantic_alignment"]
                                                           for k in range(10)]
    assert seen_marks == [(0.70, 0.68)]
    # Candidate 0 (Flash meaning 0.66) fails and candidate 1 (Flash 0.75) passes, whatever Hydrogen said.
    assert chosen == "a1"
    assert rm.fallback_fired and rm.current("judge") == mc.FLASH


def test_a_stage_judged_wholly_by_hydrogen_is_not_scored_again(monkeypatch, tmp_path):
    calls = _stage_judge(monkeypatch, _two_tables({0: _scores(0.85), 1: _scores(0.10)}))
    rm = _hydrogen()
    chosen, rows, seen_marks = _photo_stage(monkeypatch, tmp_path, rm, _calibration((0.80, 0.78)))
    assert calls == [(mc.HYDROGEN, BATCH_1), (mc.HYDROGEN, BATCH_2)]
    assert seen_marks == [(0.80, 0.78)] and chosen == "a0" and not rm.fallback_fired


def _batches(count, **kwargs):
    candidates = [{"thumb_bytes": _grey(k)} for k in range(count)]
    return vr._score_in_batches(candidates, word="ANCHOR", gloss="Rooted craft", thesis="Stability",
                                model_name=mc.HYDROGEN, strict_errors=True, **kwargs)


def test_three_batches_with_the_failure_on_the_third_rescore_the_first_two_once_each(monkeypatch):
    third = list(range(18, 27))
    calls = _stage_judge(monkeypatch, _two_tables({}),
                         lambda model, numbers: '{"candidates": [' if model == mc.HYDROGEN and numbers == third else None)
    rm = _hydrogen()
    with mc.activate(rm):
        scores = _batches(27)
    assert calls == [(mc.HYDROGEN, list(range(9))), (mc.HYDROGEN, list(range(9, 18))), (mc.HYDROGEN, third),
                     (mc.FLASH, third), (mc.FLASH, list(range(9))), (mc.FLASH, list(range(9, 18)))]
    assert [s["semantic_alignment"] for s in scores] == [_flash_table(mc.FLASH, k)["semantic_alignment"]
                                                         for k in range(27)]


def test_a_flash_rescore_of_an_earlier_batch_that_fails_applies_todays_strict_behaviour(monkeypatch):
    def broken(model, numbers):
        if model == mc.HYDROGEN and numbers == BATCH_2:
            return '{"candidates": ['
        if model == mc.FLASH and numbers == BATCH_1:
            return '{"candidates": ['
        return None

    calls = _stage_judge(monkeypatch, _two_tables({}), broken)
    with mc.activate(_hydrogen()), pytest.raises(vr.PhotoCriticUnavailable):
        _batches(10)
    assert calls == [(mc.HYDROGEN, BATCH_1), (mc.HYDROGEN, BATCH_2), (mc.FLASH, BATCH_2), (mc.FLASH, BATCH_1)]


def test_a_spend_block_after_the_call_on_batch_2_still_propagates(monkeypatch):
    calls = _stage_judge(monkeypatch, _two_tables({}),
                         lambda model, numbers: _refusal(pre=False) if model == mc.HYDROGEN and numbers == BATCH_2 else None)
    rm = _hydrogen()
    with mc.activate(rm), pytest.raises(SpendBlocked):
        _batches(10)
    assert calls == [(mc.HYDROGEN, BATCH_1), (mc.HYDROGEN, BATCH_2)] and not rm.fallback_fired


# sha256 of the calls and rows of a two-batch Flash photo stage, taken before Task T6.
FLASH_STAGE_SHA256 = "82918acc58bb5dc3e9e573f6b0475f094df89dbd940b710ab9fd35b74dc78d9f"


@pytest.mark.parametrize("active", [False, True], ids=["no-run", "variables-unset"])
def test_variables_unset_photo_stage_calls_and_rows_are_byte_identical(monkeypatch, active):
    calls = _stage_judge(monkeypatch, _two_tables({}))
    _unsplash(monkeypatch)
    rm = mc.resolve_run_models({}, {})

    def stage():
        return vr._fetch_live_unsplash_candidates(["a", "b"], access_key="synthetic-unsplash-key", word="ANCHOR",
                                                  gloss="Rooted craft", thesis="Stability", strict_errors=True)

    if active:
        with mc.activate(rm):
            rows = stage()
    else:
        rows = stage()
    blob = json.dumps({"calls": calls, "rows": rows}, sort_keys=True, default=str)
    assert calls == [(mc.FLASH, BATCH_1), (mc.FLASH, BATCH_2)]
    assert hashlib.sha256(blob.encode()).hexdigest() == FLASH_STAGE_SHA256
    assert not rm.fallback_fired
