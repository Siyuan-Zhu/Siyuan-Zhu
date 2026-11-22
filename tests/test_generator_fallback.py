"""An unavailable second-choice image generator cannot stop an edition; synthetic data only."""
import io
import logging
from types import SimpleNamespace

import pytest
from PIL import Image

from src.zeitgeist import visual_router as vr
from src.zeitgeist.spend_guard import SpendBlocked


def png_bytes():
    stream = io.BytesIO()
    Image.new('RGB', (1200, 800), (40, 60, 80)).save(stream, format='PNG')
    return stream.getvalue()


def refused():
    error = SpendBlocked('Verified model pricing and request limits required')
    error.refused_before_dispatch = True
    return error


def image_response(raw):
    part = SimpleNamespace(inline_data=SimpleNamespace(data=raw))
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


class FakeGuardedModels:
    """Answers each model by name from a plan: an exception to raise or a response to return."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = []

    def _answer(self, model):
        self.calls.append(model)
        outcome = self.plan[model]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def generate_content(self, *, model, **kwargs):
        return self._answer(model)

    def generate_images(self, *, model, **kwargs):
        return self._answer(model)


@pytest.fixture
def live(monkeypatch):
    from google import genai
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM', '1')
    monkeypatch.setenv('GEMINI_API_KEY', 'synthetic-key')
    monkeypatch.delenv('BFL_API_KEY', raising=False)
    monkeypatch.setattr(genai, 'Client', lambda **kwargs: object())

    def install(plan):
        models = FakeGuardedModels(plan)
        monkeypatch.setattr(vr, 'guard_genai_client', lambda client, **kwargs: SimpleNamespace(models=models))
        return models
    return install


ALL_MODELS = ('gemini-2.5-flash-image', 'gemini-3.1-flash-image', 'imagen-3.0-generate-002')


def test_every_model_refused_before_dispatch_returns_none(live):
    models = live({name: refused() for name in ALL_MODELS})
    assert vr._generate_live_imagen3_bytes('A quiet harbour at dawn') is None
    assert models.calls == list(ALL_MODELS)


def test_every_model_refused_before_dispatch_lets_the_edition_fall_back(live, tmp_path):
    models = live({name: refused() for name in ALL_MODELS})
    pulse = vr.select_and_render_pulse_assets(
        commission={'word': 'Counterpoise', 'gloss': 'A steady balance.'},
        medium_history_30d=['PHOTOGRAPHY'] * 4, repo_root=tmp_path, date_str='2026-10-29', candidates=[])
    assert pulse.visual_medium == 'BESPOKE_SVG'
    assert models.calls == list(ALL_MODELS)


def test_each_skipped_refusal_logs_only_the_public_model_name(live, caplog):
    live({name: refused() for name in ALL_MODELS})
    prompt = 'Synthetic marker prompt zq7 harbour'
    with caplog.at_level(logging.INFO, logger=vr.logger.name):
        assert vr._generate_live_imagen3_bytes(prompt) is None
    lines = [record.getMessage() for record in caplog.records]
    assert lines == [f'Second-choice generator {name} refused before dispatch; skipping.' for name in ALL_MODELS]
    assert 'zq7' not in caplog.text and 'harbour' not in caplog.text


@pytest.mark.parametrize('failing_model', ALL_MODELS)
def test_refusal_after_dispatch_propagates(live, failing_model):
    plan = {name: refused() for name in ALL_MODELS}
    plan[failing_model] = SpendBlocked('Output policy changed after request validation')
    live(plan)
    with pytest.raises(SpendBlocked, match='Output policy changed'):
        vr._generate_live_imagen3_bytes('A quiet harbour at dawn')


def test_first_model_refused_second_returns_its_bytes(live):
    raw = png_bytes()
    models = live({'gemini-2.5-flash-image': refused(), 'gemini-3.1-flash-image': image_response(raw),
                   'imagen-3.0-generate-002': refused()})
    assert vr._generate_live_imagen3_bytes('A quiet harbour at dawn') == raw
    assert models.calls == ['gemini-2.5-flash-image', 'gemini-3.1-flash-image']


def test_live_llm_unset_returns_none_without_a_client(live, monkeypatch):
    models = live({name: refused() for name in ALL_MODELS})
    monkeypatch.delenv('ZEITGEIST_LIVE_LLM')
    assert vr._generate_live_imagen3_bytes('A quiet harbour at dawn') is None
    assert models.calls == []
