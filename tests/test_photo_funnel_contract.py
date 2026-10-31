"""Official photo metadata, bounded batch judgments and winner-only downloads."""
import io
import json
import threading
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from PIL import Image

from src.zeitgeist import visual_router as vr


def jpeg():
    stream = io.BytesIO()
    Image.new('RGB', (1200, 800), (90, 110, 120)).save(stream, format='JPEG')
    return stream.getvalue()


def photo(photo_id):
    return {'id': photo_id, 'urls': {'small': f'https://images.unsplash.com/{photo_id}_small',
        'regular': f'https://images.unsplash.com/{photo_id}_regular', 'raw': f'https://images.unsplash.com/{photo_id}_raw?ixid=keep&w=400'},
        'links': {'html': f'https://unsplash.com/photos/{photo_id}',
            'download_location': f'https://api.unsplash.com/photos/{photo_id}/download'},
        'user': {'name': 'Actual Photographer', 'links': {'html': 'https://unsplash.com/@actual'}},
        'exif': {'make': 'Actual Camera', 'model': 'Real metadata'}}


SCORES = {'semantic_alignment': .9, 'compositional_clarity': .9, 'editorial_memorability': .9, 'safety_pass': True}


def test_parallel_small_thumbnails_dedup_history_before_one_batch_judgment(monkeypatch):
    requests, batches = [], []
    active, maximum = 0, 0
    lock = threading.Lock()
    def get(client, url, *, params=None, **kwargs):
        nonlocal active, maximum
        requests.append(url)
        if url.endswith('/search/photos'):
            return SimpleNamespace(status_code=200, json=lambda: {'results': [photo(params['query'] + str(i)) for i in range(3)]})
        assert '_small' in url, 'No full-size candidate should be downloaded during review'
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(.01)
        with lock:
            active -= 1
        return SimpleNamespace(status_code=200, content=jpeg())
    monkeypatch.setattr('httpx.Client.get', get)
    monkeypatch.setattr(vr, '_score_photo_batch', lambda candidates, **kwargs:
        batches.append(list(candidates)) or [dict(SCORES) for _ in candidates])
    candidates = vr._fetch_live_unsplash_candidates(['a', 'b', 'c'], access_key='fixture', excluded_photo_ids={'a0'})
    assert len(candidates) == 8 and len(batches) == 1
    assert all('image_bytes' not in candidate for candidate in candidates)
    assert maximum > 1
    assert not any('a0_small' in url for url in requests)


def test_real_batch_critic_uses_one_nine_image_request_and_strict_index_mapping(monkeypatch):
    calls = []
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=json.dumps({'candidates': [dict(SCORES, index=i) for i in reversed(range(9))]}), candidates=[])
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM', '1')
    monkeypatch.setenv('GEMINI_API_KEY', 'fixture')
    monkeypatch.setattr('google.genai.Client', lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, 'guard_genai_client', lambda client, **kwargs: client)
    results = vr._score_photo_batch([{'thumb_bytes': jpeg()} for _ in range(9)], word='ANCHOR', gloss='Rooted craft', thesis='A concrete pier')
    assert len(calls) == 1 and len(results) == 9 and all(row['safety_pass'] for row in results)
    images = [part for part in calls[0]['contents'] if hasattr(part, 'inline_data')]
    assert len(images) == 9
    for part in images:
        with Image.open(io.BytesIO(part.inline_data.data)) as thumb:
            assert max(thumb.size) <= 384


@pytest.mark.parametrize('bad', ['false', 'true', 1, None])
def test_batch_critic_never_coerces_malformed_safety_verdict(monkeypatch, bad):
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM', '1')
    monkeypatch.setenv('GEMINI_API_KEY', 'fixture')
    response = SimpleNamespace(text=json.dumps({'candidates': [dict(SCORES, index=0, safety_pass=bad)]}), candidates=[])
    monkeypatch.setattr('google.genai.Client', lambda **kwargs: SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs: response)))
    monkeypatch.setattr(vr, 'guard_genai_client', lambda client, **kwargs: client)
    assert vr._score_photo_batch([{'thumb_bytes': jpeg()}], word='A', gloss='B', thesis='C')[0]['safety_pass'] is False


def test_only_accepted_winner_downloads_crop_metadata_and_tracking_is_deferred(monkeypatch, tmp_path):
    requests = []
    def get(client, url, **kwargs):
        requests.append(url)
        if url.endswith('/download'):
            return SimpleNamespace(status_code=200)
        if 'api.unsplash.com/photos/' in url:
            assert url.endswith('/winner')
            return SimpleNamespace(status_code=200, json=lambda: photo('winner'))
        assert 'winner_raw' in url
        params = parse_qs(urlsplit(url).query)
        assert params['w'] == ['1800'] and params['h'] == ['1200']
        assert params['crop'] == ['entropy,edges'] and params['ixid'] == ['keep']
        return SimpleNamespace(status_code=200, content=jpeg())
    monkeypatch.setattr('httpx.Client.get', get)
    monkeypatch.setenv('UNSPLASH_ACCESS_KEY', 'fixture')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '1')
    candidate = dict(vr._candidate_from_unsplash(photo('winner'), jpeg()), **SCORES, _vision_verified=True)
    pulse = vr.select_and_render_pulse_assets({'word': 'ANCHOR', 'gloss': 'Rooted craft'}, repo_root=tmp_path,
        date_str='2026-10-30', candidates=[candidate])
    assert pulse.visual_medium == 'PHOTOGRAPHY'
    assert pulse.attribution['photographer_name'] == 'Actual Photographer'
    assert pulse.attribution['exif']['make'] == 'Actual Camera'
    assert not any(url.endswith('/download') for url in requests)
    vr.confirm_published_photo_download(pulse)
    vr.confirm_published_photo_download(pulse)
    assert sum(url.endswith('/download') for url in requests) == 1


def test_curated_candidates_cannot_supply_unverified_critic_scores(monkeypatch, tmp_path):
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '1')
    monkeypatch.setattr(vr, 'fetch_curated_library_candidates', lambda **kwargs:
        [dict(SCORES, id='curated', image_bytes=jpeg(), _requires_critic=True)])
    batches = []
    monkeypatch.setattr(vr, '_score_photo_batch', lambda candidates, **kwargs:
        batches.append(candidates) or [dict(vr._FAIL_CLOSED_VISION_SCORES) for _ in candidates])
    with pytest.raises(ValueError, match='Photography cadence'):
        vr.select_and_render_pulse_assets({'word': 'ANCHOR', 'gloss': 'Rooted craft'}, repo_root=tmp_path,
            date_str='2026-10-30', candidates=[], medium_history_30d=6)
    assert len(batches) == 1


def test_history_photo_id_is_rejected_before_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(vr, '_generate_live_imagen3_bytes', lambda *args, **kwargs: None)
    pulse = vr.select_and_render_pulse_assets({'word': 'ANCHOR', 'gloss': 'Rooted craft'}, repo_root=tmp_path,
        date_str='2026-10-30', candidates=[dict(SCORES, id='used', image_bytes=jpeg())],
        photo_history_30d=[{'image_url': 'https://unsplash.com/photos/used'}])
    assert pulse.visual_medium == 'BESPOKE_SVG'


def test_curated_library_resolves_real_identity_without_fabricated_scores(monkeypatch):
    monkeypatch.setenv('UNSPLASH_ACCESS_KEY', 'fixture')
    monkeypatch.setattr(vr, 'CURATED_PHOTO_LIBRARY', [{'photo_id': 'real', 'candidate_id': 'legacy_fake',
        'image_url': 'https://images.unsplash.com/legacy_regular', 'credit_label': 'Fake prose credit', 'keywords': ()}])
    def get(client, url, **kwargs):
        if url.startswith('https://api.unsplash.com/'):
            return SimpleNamespace(status_code=200, json=lambda: photo('real'))
        return SimpleNamespace(status_code=200, content=jpeg())
    monkeypatch.setattr('httpx.Client.get', get)
    candidates = vr.fetch_curated_library_candidates('ANCHOR')
    assert len(candidates) == 1
    assert candidates[0]['photographer_name'] == 'Actual Photographer'
    assert 'semantic_alignment' not in candidates[0]
    assert candidates[0]['photo_id'] == 'real'


@pytest.mark.parametrize('allowed_images', [1, 9])
def test_nine_image_critic_requires_explicit_verified_budget_policy(monkeypatch, allowed_images):
    from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked, guard_genai_client
    from tests.test_spend_hardening import document, NOW
    policy = document()
    policy['providers']['gemini']['token_pricing_required'] = True
    policy['providers']['gemini']['request_limits']['generate_content:gemini-2.5-flash'].update(
        max_input_bytes=200000, max_input_tokens=5000, max_input_images=allowed_images,
        input_token_usd_per_million='.30', output_token_usd_per_million='2.50',
        count_tokens_reference='official-token-metadata-fixture')
    guard = BudgetGuard(MemoryLedger(policy), lambda: NOW)
    dispatches = []
    def generate(**kwargs):
        dispatches.append(kwargs)
        return SimpleNamespace(text=json.dumps({'candidates': [dict(SCORES, index=i) for i in range(9)]}), candidates=[])
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate,
        count_tokens=lambda **kwargs: SimpleNamespace(total_tokens=100)))
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM', '1')
    monkeypatch.setenv('GEMINI_API_KEY', 'fixture')
    monkeypatch.setattr('google.genai.Client', lambda **kwargs: client)
    monkeypatch.setattr(vr, 'guard_genai_client', lambda client, **kwargs: guard_genai_client(client, guard))
    call = lambda: vr._score_photo_batch([{'thumb_bytes': jpeg()} for _ in range(9)],
        word='ANCHOR', gloss='Rooted craft', thesis='A concrete pier', model_name='gemini-2.5-flash')
    if allowed_images == 1:
        with pytest.raises(SpendBlocked, match='image count'):
            call()
        assert dispatches == [] and guard.snapshot()['reservation_count'] == 0
    else:
        assert all(result['safety_pass'] for result in call())
        assert len(dispatches) == 1 and guard.snapshot()['reservation_count'] == 1


@pytest.mark.parametrize('tracking_success', [True, False])
def test_orchestrator_tracks_only_after_commit_and_install_and_retains_failed_followup(monkeypatch, tmp_path, tracking_success):
    from tests.test_publication_atomic_commit import _publication_fixture
    from src.zeitgeist import adk_pipeline as pipeline
    from src.zeitgeist.publication_commit import PublicationMirrorError
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    original = vr.select_and_render_pulse_assets
    def render(*args, **kwargs):
        kwargs.update(candidates=[dict(SCORES, id='winner', image_bytes=jpeg(),
            photographer_name='Actual Photographer', photographer_url='https://unsplash.com/@actual',
            unsplash_url='https://unsplash.com/photos/winner',
            download_location='https://api.unsplash.com/photos/winner/download')],
            requested_medium='photo', require_semantic_rescore=False)
        return original(*args, **kwargs)
    monkeypatch.setattr(pipeline, 'select_and_render_pulse_assets', render)
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '1')
    acknowledgments = []
    def acknowledge(url, **kwargs):
        assert wire.docs['staging_profile_state/daily_pulse']['date'] == '2026-10-30'
        assert json.loads((tmp_path / 'data/zeitgeist_state.json').read_text())['date'] == '2026-10-30'
        acknowledgments.append(url)
        return tracking_success
    monkeypatch.setattr(vr, '_ping_unsplash_download_location', acknowledge)
    orchestrator = pipeline.ZeitgeistADKOrchestrator(tmp_path, store)
    if tracking_success:
        orchestrator.run('2026-10-30', offline_demo=True)
        assert not (tmp_path / 'data/publication_reconcile.json').exists()
    else:
        with pytest.raises(PublicationMirrorError):
            orchestrator.run('2026-10-30', offline_demo=True)
        marker = json.loads((tmp_path / 'data/publication_reconcile.json').read_text())
        assert marker['status'] == 'canonical_committed_download_tracking_pending'
        assert marker['download_location'].endswith('/winner/download')
        with pytest.raises(RuntimeError, match='pending canonical publication reconciliation'):
            orchestrator.run('2026-10-31', offline_demo=True)
    assert len(acknowledgments) == 1


def test_live_critic_top_level_list_preserves_strict_verified_photo_scores(monkeypatch):
    """Replay the exact top-level shape observed in real diagnostic 41230454680."""
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM','1');monkeypatch.setenv('GEMINI_API_KEY','fixture')
    response=SimpleNamespace(text=json.dumps([dict(SCORES,index=i) for i in range(7)]),candidates=[])
    monkeypatch.setattr('google.genai.Client',lambda **kwargs:SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs:response)))
    monkeypatch.setattr(vr,'guard_genai_client',lambda client,**kwargs:client)
    result=vr._score_photo_batch([{'thumb_bytes':jpeg()} for _ in range(7)],word='fixture',gloss='fixture',thesis='fixture')
    assert all(r['safety_pass'] is True and r['semantic_alignment']==.9 for r in result)


@pytest.mark.parametrize('rows',[
    [dict(SCORES,index=0),dict(SCORES,index=0)],
    [dict(SCORES,index=True),dict(SCORES,index=1)],
    [dict(SCORES,index=0),dict(SCORES,index=1,safety_pass='true')],
])
def test_array_critic_shape_keeps_index_and_safety_validation(monkeypatch,rows):
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM','1');monkeypatch.setenv('GEMINI_API_KEY','fixture')
    response=SimpleNamespace(text=json.dumps(rows),candidates=[])
    monkeypatch.setattr('google.genai.Client',lambda **kwargs:SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kwargs:response)))
    monkeypatch.setattr(vr,'guard_genai_client',lambda client,**kwargs:client)
    result=vr._score_photo_batch([{'thumb_bytes':jpeg()}]*2,word='A',gloss='B',thesis='C')
    assert result[1]['safety_pass'] is False


def test_strict_live_critic_error_is_not_a_false_unsafe_photo(monkeypatch):
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM','1');monkeypatch.setenv('GEMINI_API_KEY','fixture')
    def fail(**kwargs):raise RuntimeError('provider unavailable')
    monkeypatch.setattr('google.genai.Client',lambda **kwargs:SimpleNamespace(models=SimpleNamespace(generate_content=fail)))
    monkeypatch.setattr(vr,'guard_genai_client',lambda client,**kwargs:client)
    with pytest.raises(vr.PhotoCriticUnavailable):
        vr._score_photo_batch([{'thumb_bytes':jpeg()}],word='A',gloss='B',thesis='C',strict_errors=True)


def test_strict_thumbnail_failure_does_not_become_semantic_rejection(monkeypatch):
    def get(client,url,**kwargs):
        return SimpleNamespace(status_code=200,json=lambda:{'results':[photo('p')]}) if url.endswith('/search/photos') else SimpleNamespace(status_code=503)
    monkeypatch.setattr('httpx.Client.get',get)
    with pytest.raises(vr.PhotoSearchUnavailable):
        vr._fetch_live_unsplash_candidates(['fixture'],access_key='fixture',strict_errors=True)
