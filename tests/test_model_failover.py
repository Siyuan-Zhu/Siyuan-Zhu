"""Task T3: a failing or unusable non-default model hands the run to 3.8 Flash. Synthetic data only."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

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
