"""Neon as a chosen Director and judge: model choice, slate, calibration, reflector, and an offline staged run."""
from __future__ import annotations

import itertools
import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from google.genai import types

from scripts import private_run
from scripts import reflect
from src.zeitgeist import adk_pipeline as pipeline
from src.zeitgeist import judge_calibration as jc
from src.zeitgeist import model_choice as mc
from src.zeitgeist import spend_guard
from src.zeitgeist import visual_router as vr
from src.zeitgeist.spend_guard import NEON_OPERATION, SpendBlocked
from tests.test_judge_calibration import _document, _pipeline_marks
from tests.test_model_choice_pipeline import (
    SLATE_LINE, Stop, _lines, _live, _packet_run, _refusal, _slate, _slate_reply, _stage0_packet_prompt)
from tests.test_neon_compatibility import NEON_VERSION, neon_guard
from tests.test_reflect_script import Client as ReflectClient, Store as ReflectStore, _record, _reflect, _stored
from tests.test_understanding_pipeline import _reply

NEON = 'gemini-4-neon'


# ---- model choice ----


def test_neon_is_an_allowed_director_and_judge():
    assert mc.NEON == NEON and mc.NEON in mc.GEMINI_4 and mc.HYDROGEN in mc.GEMINI_4
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: NEON, mc.JUDGE_ENV: NEON})
    assert (rm.director, rm.judge, rm.unrecognized) == (NEON, NEON, False)


def test_neon_is_an_allowed_config_default():
    rm = mc.resolve_run_models({'budget_guardrails': {'primary_text_model': NEON, 'multimodal_judge_model': NEON}}, {})
    assert (rm.director, rm.judge, rm.unrecognized) == (NEON, NEON, False)


def test_a_neon_refusal_before_dispatch_moves_both_roles_to_flash():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: NEON, mc.JUDGE_ENV: NEON})
    seen = []

    def call(model):
        seen.append(model)
        if model == NEON:
            raise _refusal()
        return 'ok'
    assert rm.call('judge', call) == 'ok'
    assert seen == [NEON, mc.FLASH] and (rm.director, rm.judge, rm.fallback_fired) == (mc.FLASH, mc.FLASH, True)


def test_a_neon_block_after_dispatch_propagates_without_flash():
    rm = mc.resolve_run_models({}, {mc.DIRECTOR_ENV: NEON})
    with pytest.raises(SpendBlocked):
        rm.call('director', lambda model: (_ for _ in ()).throw(_refusal(pre=False)))
    assert rm.director == NEON and not rm.fallback_fired


def test_neon_configs_carry_no_temperature():
    assert pipeline._director_config(NEON, 0.3) == {'response_mime_type': 'application/json'}
    assert vr.critic_config(NEON) == {'response_mime_type': 'application/json'}


# ---- the candidate slate ----


def test_neon_packet_prompt_ends_with_the_slate_line():
    commission, calls, _ = _packet_run([_slate_reply(_slate(9))], director=NEON)
    assert [c['model'] for c in calls] == [NEON]
    assert calls[0]['contents'] == _stage0_packet_prompt() + '\n' + SLATE_LINE
    assert 'temperature' not in calls[0]['config']
    assert commission.slate_count == 9 and commission.packet_used


def test_no_slate_after_a_neon_refusal_moves_the_director_to_flash():
    refuse = lambda model: _refusal() if model == NEON else _reply()
    commission, calls, rm = _packet_run([refuse, _reply()], director=NEON)
    assert [c['model'] for c in calls] == [NEON, mc.FLASH]
    assert calls[1]['contents'] == _stage0_packet_prompt()
    assert commission.slate_count == 0 and rm.fallback_fired


# ---- the judge calibration ----


def test_pipeline_neon_judge_uses_the_stored_neon_marks(tmp_path, monkeypatch):
    seen, loaded = _pipeline_marks(tmp_path, monkeypatch, judge=NEON, document=_document(model=NEON))
    assert seen == [(0.60, 0.58)] and loaded == ['judge_calibration']


def test_pipeline_neon_judge_with_the_hydrogen_document_uses_today_marks(tmp_path, monkeypatch):
    seen, loaded = _pipeline_marks(tmp_path, monkeypatch, judge=NEON, document=_document(model=mc.HYDROGEN))
    assert seen == [jc.DEFAULT_MARKS] and loaded == ['judge_calibration']


class CalibrationStore:
    def __init__(self):
        self.saved = []

    def save_calibration(self, name, document):
        self.saved.append((name, document))


# Flash passes five photos on meaning and six on composite; the judge scores every photo higher.
FLASH_SCORES = [(0.90, 0.90), (0.85, 0.85), (0.80, 0.80), (0.75, 0.75), (0.70, 0.70), (0.69, 0.795), (0.50, 0.50),
                (0.40, 0.40), (0.30, 0.30)]
JUDGE_SCORES = [0.95, 0.92, 0.88, 0.84, 0.81, 0.78, 0.60, 0.50, 0.40]


def _measured(monkeypatch, judge):
    monkeypatch.setattr(private_run, '_flash_scores', lambda scored, commission: list(FLASH_SCORES))
    result = private_run.PrivateRun([], 0, commission=object(),
                                     scored_candidates=[{'semantic_alignment': m, 'composite_score': m} for m in JUDGE_SCORES])
    store = CalibrationStore()
    line = private_run.measure_judges(store, result, mc.RunModels(mc.FLASH, judge, mc.EMBED_2), '2026-11-25')
    return store, line


def test_a_neon_judged_private_run_writes_a_calibration_naming_neon(monkeypatch):
    store, line = _measured(monkeypatch, NEON)
    assert line.endswith('meaning_min=0.81 composite_min=0.78 written=1')
    assert [(name, doc['model']) for name, doc in store.saved] == [('judge_calibration', NEON)]
    document = store.saved[0][1]
    assert jc.marks_for(NEON, document) == (0.81, 0.78)
    # One slot: Neon's document leaves Hydrogen on today's marks until Hydrogen is measured again.
    assert jc.marks_for(mc.HYDROGEN, document) == jc.DEFAULT_MARKS


def test_a_hydrogen_judged_private_run_still_writes_hydrogen(monkeypatch):
    store, _ = _measured(monkeypatch, mc.HYDROGEN)
    assert store.saved[0][1]['model'] == mc.HYDROGEN


def test_a_flash_judged_private_run_does_not_measure(monkeypatch):
    store, line = _measured(monkeypatch, mc.FLASH)
    assert line is None and store.saved == []


# ---- the reflector ----


def test_the_reflector_runs_on_neon_without_temperature(tmp_path, monkeypatch):
    assert reflect.MODELS == (mc.FLASH, mc.HYDROGEN, NEON)
    store, client = ReflectStore(tmp_path), ReflectClient()
    run_id = store.put(_record(director=mc.FLASH))
    monkeypatch.setenv('ZEITGEIST_REFLECTOR_MODEL', NEON)
    assert _reflect(store, client)['status'] == 'pointer'
    assert {model for model, _, _ in client.generates} == {NEON}
    assert _stored(store, run_id)['model'] == NEON


# ---- an offline staged run on Neon under the real guard ----


def _thumb():
    from PIL import Image

    encoded = BytesIO()
    Image.new('RGB', (8, 8), (120, 130, 140)).save(encoded, format='JPEG')
    return encoded.getvalue()


_SCORES = {'semantic_alignment': 0.9, 'compositional_clarity': 0.9, 'editorial_memorability': 0.9, 'safety_pass': True}
IMAGES = 3
TEXT_COUNTED, DIRECTOR_COUNTED = 216, 1060


def _has_image(value):
    if isinstance(value, types.Part):
        return value.inline_data is not None
    if isinstance(value, dict):
        return 'inline_data' in value or any(_has_image(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_image(item) for item in value)
    return False


class NeonTransport:
    """The provider behind the real guard: records what it was sent, answers as the Neon probe measured."""

    def __init__(self):
        self.calls = []
        self.ids = itertools.count()
        self.models = self

    def count_tokens(self, *, model, contents, config=None):
        return SimpleNamespace(total_tokens=500 if _has_image(contents) else
                               TEXT_COUNTED if isinstance(contents, list) else DIRECTOR_COUNTED)

    def generate_content(self, *, model, contents, config):
        config = config.model_dump(exclude_none=True) if hasattr(config, 'model_dump') else dict(config)
        images = sum(1 for part in contents if _has_image(part)) if isinstance(contents, list) else 0
        self.calls.append({'model': model, 'config': config, 'images': images})
        text = json.dumps({'candidates': [dict(_SCORES, index=i) for i in range(images)]}) if images else _reply()
        prompt = TEXT_COUNTED + 67 + images * 1089 if images else DIRECTOR_COUNTED + 67
        return types.GenerateContentResponse(
            response_id=f'synthetic-neon-{next(self.ids)}', model_version=NEON_VERSION,
            candidates=[types.Candidate(content=types.Content(role='model', parts=[types.Part(text=text)]),
                                        finish_reason='STOP')],
            usage_metadata={'prompt_token_count': prompt, 'candidates_token_count': 400, 'thoughts_token_count': 1200,
                            'total_token_count': prompt + 1600})


def _staged_on_neon(tmp_path, monkeypatch):
    """The staged edition run to the renderer, both roles on Neon, every call through the real guard."""
    from src.state_store import InMemoryStateStore
    from src.zeitgeist import photo_recovery
    from tests.test_understanding_pipeline import _clusters

    guard, transport = neon_guard(), NeonTransport()
    real_guard = spend_guard.guard_genai_client
    guarded = lambda client, **kw: real_guard(transport, guard)
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config/orbit_sources.json').write_text(json.dumps({
        'budget_guardrails': {'primary_text_model': mc.FLASH, 'multimodal_judge_model': mc.FLASH},
        'active_interjection': {'enabled': False}}))
    for name in (mc.EMBEDDING_ENV, 'ZEITGEIST_UNDERSTANDING_V1', 'ZEITGEIST_OFFLINE_DEMO'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(mc.DIRECTOR_ENV, NEON)
    monkeypatch.setenv(mc.JUDGE_ENV, NEON)
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM', '1')
    monkeypatch.setenv('GEMINI_API_KEY', 'synthetic-key')
    monkeypatch.setattr('google.genai.Client', lambda **kw: transport)
    monkeypatch.setattr(spend_guard, 'guard_genai_client', guarded)
    monkeypatch.setattr(vr, 'guard_genai_client', guarded)
    clusters = _clusters()
    for cluster in clusters:
        cluster.silhouette_score = 0.5
    monkeypatch.setattr(pipeline, 'collect_96h_signals', lambda *a, **kw: [])
    monkeypatch.setattr(pipeline, 'score_and_cluster_signals', lambda rows, **kw: clusters)
    judged = []

    def render(**kwargs):
        judged.append(kwargs['judge_model'])
        vr._score_photo_batch([{'thumb_bytes': _thumb()} for _ in range(IMAGES)], word='ANCHOR', gloss='Rooted craft',
                              thesis='Physical stability', model_name=kwargs['judge_model'], strict_errors=True)
        raise Stop()

    def recover(commission, *, render, **kw):
        render(commission, photo_only=True)

    monkeypatch.setattr(pipeline, 'select_and_render_pulse_assets', render)
    monkeypatch.setattr(photo_recovery, 'recover_visual', recover)
    with pytest.raises(Stop):
        pipeline.ZeitgeistADKOrchestrator(tmp_path, InMemoryStateStore())._run_staged('2026-11-21', now_epoch=1795262400.0)
    return guard, transport, judged


def test_a_staged_run_on_neon_makes_guarded_neon_calls_at_medium(tmp_path, monkeypatch, capsys):
    guard, transport, judged = _staged_on_neon(tmp_path, monkeypatch)
    assert judged == [NEON]
    assert [(call['model'], call['images']) for call in transport.calls] == [(NEON, 0), (NEON, IMAGES)]
    for call in transport.calls:
        config = call['config']
        assert config['thinking_config'] == {'thinking_level': 'medium'}
        assert config['media_resolution'] == 'MEDIA_RESOLUTION_HIGH' and config['max_output_tokens'] == 6000
        assert 'temperature' not in config and 'candidate_count' not in config
    records = sorted(guard.ledger.document['reservations'].values(),
                     key=lambda record: record['request_context']['admitted_input_tokens'])
    assert [record['operation'] for record in records] == [NEON_OPERATION, NEON_OPERATION]
    assert [record['request_context']['admitted_input_tokens'] for record in records] == [
        DIRECTOR_COUNTED + 256, TEXT_COUNTED + IMAGES * 1091 + 256]
    assert all(record['status'] == 'settled' for record in records)
    assert guard.ledger.document.get('reconciliation_required') is not True
    lines = _lines(capsys)
    assert len(lines) == 1
    assert lines[0].startswith('[model_choice] director=gemini-4-neon judge=gemini-4-neon '
                               'embedding=gemini-embedding-001 fallback=0 unrecognized=0 slate=')
