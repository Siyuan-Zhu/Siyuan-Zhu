"""Task C2: the Director runs on the chosen model with a run-wide Flash fallback. Synthetic data only."""

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest

from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import model_choice as mc
from src.zeitgeist import photo_recovery
from src.zeitgeist import spend_guard
from src.zeitgeist import visual_router as vr
from src.zeitgeist.adk_pipeline import commission_daily_zeitgeist
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_understanding_pipeline import _clusters, _reply


def _refusal(pre=True):
    error = SpendBlocked("Unregistered paid SDK method")
    if pre:
        error.refused_before_dispatch = True
    return error


class FakeModels:
    """Records every generate_content call; replies come from a script keyed by call order."""

    def __init__(self, replies):
        self.calls = []
        self._replies = list(replies)

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": dict(config)})
        reply = self._replies.pop(0) if self._replies else _reply()
        if callable(reply):
            reply = reply(model)
        if isinstance(reply, BaseException):
            raise reply
        return SimpleNamespace(text=reply)


def _refuse_hydrogen(pre=True):
    return lambda model: _refusal(pre) if model == mc.HYDROGEN else _reply()


def _client(replies):
    return SimpleNamespace(models=FakeModels(replies))


# ---- commission_daily_zeitgeist ----


def test_hydrogen_director_sends_hydrogen_without_temperature():
    client = _client([_reply()])
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    commission_daily_zeitgeist(_clusters(), [], genai_client=client, run_models=rm)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN]
    assert client.models.calls[0]["config"] == {"response_mime_type": "application/json"}


def test_director_config_adds_temperature_only_where_sampling_is_supported():
    assert pipeline._director_config("gemini-2.5-flash", 0.3) == {"response_mime_type": "application/json", "temperature": 0.3}
    for model in (mc.FLASH, mc.HYDROGEN):
        assert pipeline._director_config(model, 0.3) == {"response_mime_type": "application/json"}


def test_pre_dispatch_refusal_resends_same_attempt_on_flash_then_stays_on_flash():
    client = _client([_refuse_hydrogen(), "not json", _reply()])
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    commission = commission_daily_zeitgeist(_clusters(), [], genai_client=client, run_models=rm)
    calls = client.models.calls
    assert [c["model"] for c in calls] == [mc.HYDROGEN, mc.FLASH, mc.FLASH]
    assert calls[0]["contents"] == calls[1]["contents"]
    assert all("temperature" not in c["config"] for c in calls)
    assert commission.attempts_used == 2 and not commission.fallback_used
    assert rm.fallback_fired and rm.current("director") == mc.FLASH and rm.current("judge") == mc.FLASH
    assert "fallback=1" in mc.model_choice_line(rm, 0)


def test_post_dispatch_block_propagates_without_flash_call():
    client = _client([_refuse_hydrogen(pre=False)])
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN})
    with pytest.raises(SpendBlocked):
        commission_daily_zeitgeist(_clusters(), [], genai_client=client, run_models=rm)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN]
    assert not rm.fallback_fired


def test_without_run_models_the_request_is_unchanged():
    client = _client([_reply()])
    commission_daily_zeitgeist(_clusters(), [], genai_client=client, model_name="gemini-2.5-flash", temperature=0.3)
    assert client.models.calls == [{"model": "gemini-2.5-flash", "contents": client.models.calls[0]["contents"],
                                    "config": {"temperature": 0.3, "response_mime_type": "application/json"}}]


# ---- pipeline wiring ----


class Stop(Exception):
    pass


def _live(tmp_path, monkeypatch, replies, *, director=None, judge=None, recover=None, telemetry=None):
    from src.state_store import InMemoryStateStore

    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text(json.dumps({
        "budget_guardrails": {"primary_text_model": mc.FLASH, "multimodal_judge_model": mc.FLASH},
        "active_interjection": {"enabled": False},
    }))
    for name in (mc.DIRECTOR_ENV, mc.JUDGE_ENV, mc.EMBEDDING_ENV, "ZEITGEIST_UNDERSTANDING_V1", "ZEITGEIST_OFFLINE_DEMO"):
        monkeypatch.delenv(name, raising=False)
    if director:
        monkeypatch.setenv(mc.DIRECTOR_ENV, director)
    if judge:
        monkeypatch.setenv(mc.JUDGE_ENV, judge)
    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    client = _client(replies)
    monkeypatch.setattr("google.genai.Client", lambda **kw: client)
    monkeypatch.setattr(spend_guard, "guard_genai_client", lambda c, **kw: c)
    clusters = _clusters()
    for cluster in clusters:
        cluster.silhouette_score = 0.5
    monkeypatch.setattr(pipeline, "collect_96h_signals", lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, "score_and_cluster_signals", lambda rows, **kw: clusters)
    commissions = []
    real_commission = pipeline.commission_daily_zeitgeist

    def spy(*args, **kwargs):
        commissions.append(dict(kwargs))
        return real_commission(*args, **kwargs)

    monkeypatch.setattr(pipeline, "commission_daily_zeitgeist", spy)

    def stop_recovery(commission, *, revise, **kw):
        raise Stop()

    monkeypatch.setattr(photo_recovery, "recover_visual", recover or stop_recovery)
    if telemetry is not None:
        monkeypatch.setattr(pipeline, "ADKRunTelemetry", telemetry)
    staged = pipeline.ZeitgeistADKOrchestrator(tmp_path, InMemoryStateStore())
    with pytest.raises(Stop):
        staged._run_staged("2026-11-21", now_epoch=1795262400.0)
    return client, commissions


def _lines(capsys):
    return [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[model_choice]")]


def test_pipeline_director_variable_reaches_the_client(tmp_path, monkeypatch, capsys):
    client, _ = _live(tmp_path, monkeypatch, [_reply()], director=mc.HYDROGEN)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN]
    assert "temperature" not in client.models.calls[0]["config"]
    assert _lines(capsys) == [
        "[model_choice] director=gemini-4-hydrogen judge=gemini-3.8-flash embedding=gemini-embedding-001 "
        "fallback=0 unrecognized=0 slate=0"
    ]


def test_pipeline_refusal_line_shows_fallback(tmp_path, monkeypatch, capsys):
    client, _ = _live(tmp_path, monkeypatch, [_refuse_hydrogen(), "not json", _reply()], director=mc.HYDROGEN)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN, mc.FLASH, mc.FLASH]
    lines = _lines(capsys)
    assert len(lines) == 1
    assert "director=gemini-3.8-flash judge=gemini-3.8-flash" in lines[0] and "fallback=1" in lines[0]


def test_pipeline_variables_unset_sends_flash_as_before(tmp_path, monkeypatch, capsys):
    client, commissions = _live(tmp_path, monkeypatch, [_reply()])
    assert client.models.calls[0]["model"] == mc.FLASH
    assert client.models.calls[0]["config"] == {"response_mime_type": "application/json"}
    assert commissions[0]["model_name"] == mc.FLASH
    assert len(_lines(capsys)) == 1


def test_pipeline_offline_prints_no_model_choice_line(tmp_path, monkeypatch, capsys):
    from src.state_store import InMemoryStateStore

    (tmp_path / "config").mkdir()
    (tmp_path / "config/orbit_sources.json").write_text(json.dumps({"active_interjection": {"enabled": False}}))
    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", lambda **kw: (_ for _ in ()).throw(Stop()))
    with pytest.raises(Stop):
        pipeline.ZeitgeistADKOrchestrator(tmp_path, InMemoryStateStore()).run("2026-10-30", offline_demo=True)
    assert _lines(capsys) == []


def test_photo_recovery_uses_director_in_force(tmp_path, monkeypatch):
    def recover(commission, *, revise, **kw):
        revise(commission, change_word=False)
        raise Stop()

    client, commissions = _live(tmp_path, monkeypatch, [_refuse_hydrogen(), _reply(), _reply()],
                                director=mc.HYDROGEN, recover=recover)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN, mc.FLASH, mc.FLASH]
    assert len(commissions) == 2
    assert commissions[1]["model_name"] == mc.FLASH
    assert commissions[1]["run_models"] is commissions[0]["run_models"]


def test_photo_recovery_without_fallback_stays_on_hydrogen(tmp_path, monkeypatch):
    def recover(commission, *, revise, **kw):
        revise(commission, change_word=False)
        raise Stop()

    client, commissions = _live(tmp_path, monkeypatch, [_reply(), _reply()], director=mc.HYDROGEN, recover=recover)
    assert [c["model"] for c in client.models.calls] == [mc.HYDROGEN, mc.HYDROGEN]
    assert commissions[1]["model_name"] == mc.HYDROGEN


def _published():
    return SimpleNamespace(word="Selvedge", gloss="A finished edge that holds.", attribution={},
                           visual_medium="PROCEDURAL", composite_critic_score=0.8, semantic_alignment_score=0.8)


@pytest.mark.parametrize("replies, expected", [
    ([_reply()], mc.HYDROGEN),
    ([_refuse_hydrogen(), _reply()], mc.FLASH),
])
def test_telemetry_model_id_is_director_in_force(tmp_path, monkeypatch, replies, expected):
    seen = {}

    def recover(commission, **kw):
        return commission, _published(), []

    def telemetry(**kwargs):
        seen.update(kwargs)
        raise Stop()

    _live(tmp_path, monkeypatch, replies, director=mc.HYDROGEN, recover=recover, telemetry=telemetry)
    assert seen["model_id"] == expected



# ---- Task C3: photo critic on the chosen model ----


_SCORES = {"semantic_alignment": 0.9, "compositional_clarity": 0.9, "editorial_memorability": 0.9, "safety_pass": True}


def _thumb():
    from PIL import Image

    encoded = BytesIO()
    Image.new("RGB", (8, 8), (120, 130, 140)).save(encoded, format="JPEG")
    return encoded.getvalue()


def _critic_client(monkeypatch, refuse=None):
    calls = []

    def generate(*, model, contents, config):
        calls.append({"model": model, "config": dict(config)})
        if refuse is not None and model == mc.HYDROGEN:
            raise _refusal(refuse)
        batch = any(isinstance(part, str) and "Candidate index" in part for part in contents)
        return SimpleNamespace(text=json.dumps({"candidates": [dict(_SCORES, index=0)]} if batch else _SCORES),
                               candidates=[])

    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "synthetic-key")
    monkeypatch.setattr("google.genai.Client", lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    return calls


def _single(model_name=mc.FLASH):
    return vr._score_candidate_with_gemini_vision(_thumb(), "ANCHOR", "Rooted craft", "Physical stability",
                                                  model_name=model_name)


def _batch(model_name=mc.FLASH):
    return vr._score_photo_batch([{"thumb_bytes": _thumb()}], word="ANCHOR", gloss="Rooted craft",
                                 thesis="Physical stability", model_name=model_name, strict_errors=True)[0]


_LEAVES = pytest.mark.parametrize("leaf", [_single, _batch], ids=["single", "batch"])


def test_critic_config_adds_temperature_only_where_sampling_is_supported():
    assert vr.critic_config("gemini-2.5-flash") == {"response_mime_type": "application/json", "temperature": 0.1}
    for model in (mc.FLASH, mc.HYDROGEN):
        assert vr.critic_config(model) == {"response_mime_type": "application/json"}


@_LEAVES
def test_hydrogen_judge_sends_hydrogen_without_temperature(monkeypatch, leaf):
    calls = _critic_client(monkeypatch)
    rm = mc.resolve_run_models({}, {mc.JUDGE_ENV: mc.HYDROGEN})
    with mc.activate(rm):
        assert leaf()["safety_pass"] is True
    assert calls == [{"model": mc.HYDROGEN, "config": {"response_mime_type": "application/json"}}]
    assert not rm.fallback_fired


@_LEAVES
def test_judge_refusal_retries_on_flash_and_moves_the_run_to_flash(monkeypatch, leaf):
    calls = _critic_client(monkeypatch, refuse=True)
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: mc.HYDROGEN, mc.JUDGE_ENV: mc.HYDROGEN})
    with mc.activate(rm):
        assert leaf(mc.HYDROGEN)["safety_pass"] is True
        assert mc.active().current("director") == mc.FLASH
        assert mc.active().current("judge") == mc.FLASH
    assert [c["model"] for c in calls] == [mc.HYDROGEN, mc.FLASH]
    assert all("temperature" not in c["config"] for c in calls)
    assert rm.fallback_fired and "fallback=1" in mc.model_choice_line(rm, 0)


@_LEAVES
def test_judge_post_dispatch_block_propagates_without_flash_call(monkeypatch, leaf):
    calls = _critic_client(monkeypatch, refuse=False)
    rm = mc.resolve_run_models({}, {mc.JUDGE_ENV: mc.HYDROGEN})
    with mc.activate(rm), pytest.raises(SpendBlocked):
        leaf(mc.HYDROGEN)
    assert [c["model"] for c in calls] == [mc.HYDROGEN]
    assert not rm.fallback_fired and rm.current("judge") == mc.HYDROGEN


@_LEAVES
def test_outside_activate_the_critic_uses_model_name_as_before(monkeypatch, leaf):
    calls = _critic_client(monkeypatch)
    assert mc.active() is None
    leaf("gemini-2.5-flash")
    assert calls == [{"model": "gemini-2.5-flash", "config": {"response_mime_type": "application/json", "temperature": 0.1}}]


def _render_judges(tmp_path, monkeypatch, replies, *, director=None, judge=None):
    judges = []

    def render(**kwargs):
        judges.append(kwargs["judge_model"])
        raise Stop()

    def recover(commission, *, render, **kw):
        render(commission, photo_only=True)

    monkeypatch.setattr(pipeline, "select_and_render_pulse_assets", render)
    _live(tmp_path, monkeypatch, replies, director=director, judge=judge, recover=recover)
    return judges


def test_pipeline_judge_variable_reaches_the_renderer(tmp_path, monkeypatch):
    assert _render_judges(tmp_path, monkeypatch, [_reply()], judge=mc.HYDROGEN) == [mc.HYDROGEN]


def test_pipeline_renderer_judge_follows_director_fallback(tmp_path, monkeypatch):
    judges = _render_judges(tmp_path, monkeypatch, [_refuse_hydrogen(), _reply()], director=mc.HYDROGEN, judge=mc.HYDROGEN)
    assert judges == [mc.FLASH]


def test_pipeline_renderer_judge_unset_is_flash_as_before(tmp_path, monkeypatch):
    assert _render_judges(tmp_path, monkeypatch, [_reply()]) == [mc.FLASH]
