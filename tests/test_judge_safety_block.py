"""Task F2: a safety block from the judge fails the batch closed, whatever the judge model. Synthetic data only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google.genai import types

from src.zeitgeist import model_choice as mc
from src.zeitgeist import visual_router as vr
from tests.test_model_failover import (BATCH_1, BATCH_2, _calibration, _candidate_numbers, _grey, _hydrogen,
                                       _photo_stage, _scores)

FAIL_CLOSED = {"semantic_alignment": 0.0, "compositional_clarity": 0.0, "editorial_memorability": 0.0,
               "safety_pass": False}


def _rows(numbers, table):
    return json.dumps({"candidates": [dict(table.get(k, _scores(0.2)), index=i) for i, k in enumerate(numbers)]})


def _good(numbers, table):
    return SimpleNamespace(text=_rows(numbers, table), candidates=[])


def _finished(reason, numbers, table):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        finish_reason=reason, content=types.Content(role="model", parts=[types.Part(text=_rows(numbers, table))]))])


def _finish(reason):
    return lambda numbers, table: _finished(reason, numbers, table)


def _prompt_block(numbers, table):
    return types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY))


BLOCKS = {
    "safety-finish": _finish(types.FinishReason.SAFETY),
    "image-safety-finish": _finish(types.FinishReason.IMAGE_SAFETY),
    "prompt-block": _prompt_block,
}


def _judge(monkeypatch, reply):
    """A fake critic: reply(model, numbers) returns a response or an exception; calls are (model, numbers)."""
    calls = []

    def generate(*, model, contents, config):
        numbers = _candidate_numbers(contents)
        calls.append((model, numbers))
        response = reply(model, numbers)
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setattr("google.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    return calls


def _batch(model_name, count=2):
    return vr._score_photo_batch([{"thumb_bytes": _grey(k)} for k in range(count)], word="ANCHOR",
                                 gloss="Rooted craft", thesis="Stability", model_name=model_name, strict_errors=True)


def _batches(count):
    return vr._score_in_batches([{"thumb_bytes": _grey(k)} for k in range(count)], word="ANCHOR",
                                gloss="Rooted craft", thesis="Stability", model_name=mc.HYDROGEN, strict_errors=True)


@pytest.mark.parametrize("block", list(BLOCKS.values()), ids=list(BLOCKS))
def test_hydrogen_safety_block_fails_the_batch_closed_without_a_hand_over(monkeypatch, block):
    calls = _judge(monkeypatch, lambda model, numbers: block(numbers, {}))
    rm = _hydrogen()
    with mc.activate(rm):
        scores = _batch(mc.HYDROGEN)
    assert calls == [(mc.HYDROGEN, [0, 1])]
    assert scores == [FAIL_CLOSED, FAIL_CLOSED]
    assert not rm.fallback_fired
    assert rm.current("judge") == mc.HYDROGEN and rm.current("director") == mc.HYDROGEN


def test_hydrogen_malformed_json_still_hands_over_to_flash(monkeypatch):
    calls = _judge(monkeypatch, lambda model, numbers: SimpleNamespace(text='{"candidates": [', candidates=[])
                   if model == mc.HYDROGEN else _good(numbers, {}))
    rm = _hydrogen()
    with mc.activate(rm):
        scores = _batch(mc.HYDROGEN)
    assert calls == [(mc.HYDROGEN, [0, 1]), (mc.FLASH, [0, 1])]
    assert [s["safety_pass"] for s in scores] == [True, True] and rm.fallback_fired


@pytest.mark.parametrize("block", list(BLOCKS.values()), ids=list(BLOCKS))
def test_flash_safety_block_fails_the_batch_closed_after_one_call(monkeypatch, block):
    calls = _judge(monkeypatch, lambda model, numbers: block(numbers, {}))
    rm = mc.resolve_run_models({}, {})
    with mc.activate(rm):
        scores = _batch(mc.FLASH)
    assert calls == [(mc.FLASH, [0, 1])]
    assert scores == [FAIL_CLOSED, FAIL_CLOSED] and not rm.fallback_fired


def test_a_safety_blocked_batch_cannot_win_and_the_next_batch_stays_with_hydrogen(monkeypatch, tmp_path):
    hydrogen_table = {0: _scores(0.95), 9: _scores(0.90)}

    def reply(model, numbers):
        if model == mc.HYDROGEN and numbers == BATCH_1:
            return _finished(types.FinishReason.SAFETY, numbers, hydrogen_table)
        return _good(numbers, hydrogen_table if model == mc.HYDROGEN else {})

    calls = _judge(monkeypatch, reply)
    rm = _hydrogen()
    chosen, rows, seen_marks = _photo_stage(monkeypatch, tmp_path, rm, _calibration((0.80, 0.78)))
    assert calls == [(mc.HYDROGEN, BATCH_1), (mc.HYDROGEN, BATCH_2)]
    assert [row["safety_pass"] for row in rows] == [False] * 9 + [True]
    assert chosen == "b4" and seen_marks == [(0.80, 0.78)] and not rm.fallback_fired


def test_a_later_hand_over_does_not_ask_flash_about_a_safety_blocked_batch(monkeypatch):
    def reply(model, numbers):
        if model == mc.HYDROGEN and numbers == BATCH_1:
            return _finished(types.FinishReason.SAFETY, numbers, {})
        if model == mc.HYDROGEN and numbers == BATCH_2:
            return RuntimeError("synthetic transport failure")
        return _good(numbers, {})

    calls = _judge(monkeypatch, reply)
    rm = _hydrogen()
    with mc.activate(rm):
        scores = _batches(10)
    assert calls == [(mc.HYDROGEN, BATCH_1), (mc.HYDROGEN, BATCH_2), (mc.FLASH, BATCH_2)]
    assert scores[:9] == [FAIL_CLOSED] * 9 and scores[9]["safety_pass"] is True
    assert rm.fallback_fired
