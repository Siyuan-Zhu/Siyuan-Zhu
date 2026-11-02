"""Selected source evidence stays truthful through rendered publication output."""
import json
from types import SimpleNamespace

import pytest

from src.zeitgeist import visual_router as vr
from src.zeitgeist.publication import verify_publication_bundle
from tests.test_photo_funnel_contract import jpeg, photo, SCORES


def publish_source(tmp_path, monkeypatch, *, detail, history=None, requested=None):
    monkeypatch.setenv('UNSPLASH_ACCESS_KEY', 'fixture')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '1')
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM', '0')
    def get(client, url, **kwargs):
        if url.startswith('https://api.unsplash.com/photos/'):
            return SimpleNamespace(status_code=200, json=lambda: detail)
        return SimpleNamespace(status_code=200, content=jpeg())
    monkeypatch.setattr('httpx.Client.get', get)
    candidate = {'id': 'winner', 'photo_id': 'winner', **SCORES, '_vision_verified': True}
    return vr.select_and_render_pulse_assets(
        {'word': 'Anchor', 'gloss': 'A shift in trusted foundations.'},
        repo_root=tmp_path, date_str='2026-11-02', candidates=[candidate],
        medium_history_30d=history or ['PHOTOGRAPHY'] * 4,
        photo_history_30d=[], requested_medium=requested,
        curatorial_archetype_override='STUDIO_MONOGRAPH')


def test_declared_render_keeps_source_credit_and_consumes_shared_allowance(tmp_path, monkeypatch):
    detail = photo('winner')
    detail.pop('exif')
    detail['description'] = 'The artwork is a 3D particle system.'
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    assert pulse.visual_medium == 'DIGITAL_ARTWORK'
    entry = json.loads((tmp_path / 'data/pulse_feed.json').read_text())['entries'][0]
    assert entry['artwork_source'] == 'Unsplash'
    assert entry['media_provenance']['basis'] == 'source_declared_digital'
    assert entry['media_provenance']['source_url'] == 'https://unsplash.com/photos/winner'
    assert verify_publication_bundle(tmp_path)['medium'] == 'DIGITAL_ARTWORK'


@pytest.mark.parametrize('evidence,kind,basis,credit', [
    ({'exif': {'make': 'Panasonic', 'model': 'DMC-G7'}}, 'PHOTOGRAPHY', 'camera_reported', 'Photo by'),
    ({}, 'UNVERIFIED_IMAGE', 'insufficient_evidence', 'Image by'),
    ({'description': 'The artwork is a 3D particle system.', 'exif': {'model': 'DMC-G7'}},
     'UNVERIFIED_IMAGE', 'conflicting_evidence', 'Image by'),
])
def test_camera_and_uncertainty_survive_feed_export_and_native_credit(tmp_path, monkeypatch, evidence, kind, basis, credit):
    detail = photo('winner')
    detail.pop('exif')
    detail.update(evidence)
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    assert pulse.visual_medium == kind
    entry = pulse.to_feed_entry()
    assert entry['media_provenance']['basis'] == basis
    expected_claim = ({'kind': 'creation_declaration', 'process': '3d_particle_system',
        'statement': 'The artwork is a 3D particle system'} if basis == 'conflicting_evidence' else
        {'kind': 'none', 'process': 'unknown', 'statement': ''})
    assert entry['media_provenance']['source_claim'] == expected_claim
    from src.zeitgeist.public_schema import validate_editorial_entry
    validate_editorial_entry(entry)
    from scripts.export_public_release import _editorial
    assert _editorial(entry)['media_provenance'] == entry['media_provenance']
    from scripts.build_readme import _visual_credit_html
    assert _visual_credit_html({}, entry).startswith(credit)
    assert entry['media_provenance']['allowance_exception'] is (kind != 'PHOTOGRAPHY')


@pytest.mark.parametrize('description,exif', [
    ('The artwork is a 3D particle system.', {}), ('', {}),
])
def test_exception_source_cannot_publish_after_exception_or_without_ratio_capacity(tmp_path, monkeypatch, description, exif):
    detail = photo('winner')
    detail.update(description=description, exif=exif)
    with pytest.raises(vr.PhotoSelectionUnavailable):
        publish_source(tmp_path, monkeypatch, detail=detail,
            history=['DIGITAL_ARTWORK'] + ['PHOTOGRAPHY'] * 28)
    assert not (tmp_path / 'assets/pulse_manifest.json').exists()


def test_camera_requirement_does_not_silently_accept_an_unknown_source(tmp_path, monkeypatch):
    detail = photo('winner')
    detail.pop('exif')
    with pytest.raises(vr.PhotoSelectionUnavailable):
        publish_source(tmp_path, monkeypatch, detail=detail, requested='photo')
    assert not (tmp_path / 'data/pulse_feed.json').exists()


def test_unsplash_acknowledgment_is_source_based_even_for_uncertain_image(tmp_path, monkeypatch):
    detail = photo('winner')
    detail.pop('exif')
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    calls = []
    vr.confirm_published_photo_download(pulse, callback=lambda url: calls.append(url))
    vr.confirm_published_photo_download(pulse, callback=lambda url: calls.append(url))
    assert calls == ['https://api.unsplash.com/photos/winner/download']


def test_bounded_detail_lookup_can_choose_camera_photo_behind_higher_scoring_render(tmp_path, monkeypatch):
    monkeypatch.setenv('UNSPLASH_ACCESS_KEY', 'fixture')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '1')
    details = []
    def get(client, url, **kwargs):
        if url.startswith('https://api.unsplash.com/photos/'):
            photo_id = url.rsplit('/', 1)[-1]
            details.append(photo_id)
            item = photo(photo_id)
            if photo_id != 'camera':
                item.update(description='The artwork is a 3D particle system.', exif={})
            return SimpleNamespace(status_code=200, json=lambda: item)
        return SimpleNamespace(status_code=200, content=jpeg())
    monkeypatch.setattr('httpx.Client.get', get)
    candidates = [dict(SCORES, id=name, photo_id=name, _vision_verified=True,
        semantic_alignment=score) for name, score in [('render', .99), ('camera', .9), ('third', .85), ('fourth', .8)]]
    pulse = vr.select_and_render_pulse_assets({'word': 'Anchor', 'gloss': 'Shift in trust'},
        repo_root=tmp_path, date_str='2026-11-02', requested_medium='photo',
        candidates=candidates, photo_history_30d=[], medium_history_30d=[])
    assert pulse.visual_medium == 'PHOTOGRAPHY'
    assert pulse.attribution['photo_id'] == 'camera'
    assert len(details) <= 3


def test_direct_live_generation_uses_shared_allowance_before_any_provider_call(tmp_path, monkeypatch):
    monkeypatch.setenv('ZEITGEIST_OFFLINE_DEMO', '0')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '0')
    calls = []
    with pytest.raises(vr.PhotoSelectionUnavailable):
        vr.select_and_render_pulse_assets({'word': 'Anchor'}, repo_root=tmp_path,
            candidates=[], medium_history_30d=['UNVERIFIED_IMAGE'] + ['PHOTOGRAPHY'] * 28,
            imagen_generator=lambda prompt: calls.append(prompt))
    assert calls == []
    assert not (tmp_path / 'assets').exists()


def test_known_generator_is_not_labeled_as_human_or_camera(tmp_path, monkeypatch):
    monkeypatch.setenv('ZEITGEIST_OFFLINE_DEMO', '0')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '0')
    pulse = vr.select_and_render_pulse_assets({'word': 'Anchor'}, repo_root=tmp_path,
        candidates=[], date_str='2026-11-02', medium_history_30d=['PHOTOGRAPHY'] * 4,
        imagen_bytes=jpeg())
    entry = pulse.to_feed_entry()
    assert pulse.visual_medium == 'IMAGEN_3'
    assert entry['media_provenance']['classification'] == 'GENERATED_ARTWORK'
    assert entry['media_provenance']['basis'] == 'generator_known'


def test_historical_photo_selection_evidence_survives_typed_medium_correction(tmp_path, monkeypatch):
    detail = photo('winner')
    detail.update(description='The artwork is a 3D particle system.', exif={})
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    entry = pulse.to_feed_entry()
    entry.update(photo_selection_policy='owner-approved-best-safe-photo',
        photo_selection_policy_date='2026-11-02',
        media_correction={'kind': 'source_medium_correction',
            'original_publication_id': '1234567890abcdefabcd',
            'original_visual_medium': 'PHOTOGRAPHY'})
    from src.zeitgeist.public_schema import validate_editorial_entry
    from scripts.export_public_release import _editorial
    validate_editorial_entry(entry)
    assert _editorial(entry)['photo_selection_policy_date'] == '2026-11-02'
    assert _editorial(entry)['media_correction'] == entry['media_correction']
    del entry['media_correction']
    with pytest.raises(ValueError):
        validate_editorial_entry(entry)


def test_manifest_binds_reported_provenance_not_only_image_medium(tmp_path, monkeypatch):
    publish_source(tmp_path, monkeypatch, detail=photo('winner'))
    path = tmp_path / 'data/pulse_feed.json'
    feed = json.loads(path.read_text())
    feed['entries'][0]['media_provenance']['camera_make'] = 'Different reported source'
    path.write_text(json.dumps(feed))
    with pytest.raises(ValueError, match='provenance'):
        verify_publication_bundle(tmp_path)


@pytest.mark.parametrize('description', [
    'A photograph of a 3D-printed sculpture at a gallery.',
    'An AI conference with researchers and cameras.',
    'A gallery of digital artwork photographed through the window.',
])
def test_subject_words_do_not_override_reported_camera_evidence(tmp_path, monkeypatch, description):
    detail = photo('winner')
    detail['description'] = description
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    assert pulse.visual_medium == 'PHOTOGRAPHY'
    assert pulse.to_feed_entry()['media_provenance']['basis'] == 'camera_reported'


def test_same_date_replacement_counts_once_from_accepted_history(tmp_path, monkeypatch):
    monkeypatch.setenv('UNSPLASH_ACCESS_KEY', 'fixture')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '1')
    detail = photo('winner')
    detail['exif'] = {}
    def get(client, url, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: detail, content=jpeg())
    monkeypatch.setattr('httpx.Client.get', get)
    history = [{'date': '2026-11-02', 'visual_medium': 'UNVERIFIED_IMAGE'}] + [
        {'date': day, 'visual_medium': 'PHOTOGRAPHY'}
        for day in ('2026-11-01', '2026-10-31', '2026-10-30', '2026-10-29')]
    pulse = vr.select_and_render_pulse_assets({'word': 'Anchor'}, repo_root=tmp_path,
        date_str='2026-11-02', candidates=[dict(SCORES, id='winner', photo_id='winner', _vision_verified=True)],
        medium_history_30d=['UNVERIFIED_IMAGE'] + ['PHOTOGRAPHY'] * 4,
        photo_history_30d=history)
    assert pulse.visual_medium == 'UNVERIFIED_IMAGE'


def test_source_detail_outage_is_not_uncertainty_or_a_generation_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv('UNSPLASH_ACCESS_KEY', 'fixture')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '1')
    monkeypatch.setattr('httpx.Client.get', lambda *args, **kwargs: SimpleNamespace(status_code=500))
    calls = []
    with pytest.raises(vr.PhotoSearchUnavailable):
        vr.select_and_render_pulse_assets({'word': 'Anchor'}, repo_root=tmp_path,
            candidates=[dict(SCORES, id='winner', photo_id='winner', _vision_verified=True)],
            medium_history_30d=['PHOTOGRAPHY'] * 4, photo_history_30d=[],
            imagen_generator=lambda prompt: calls.append(prompt))
    assert calls == [] and not (tmp_path / 'assets').exists()


def test_missing_selected_photo_bytes_cannot_bypass_exception_cadence(tmp_path, monkeypatch):
    monkeypatch.setenv('ZEITGEIST_OFFLINE_DEMO', '0')
    monkeypatch.setenv('ZEITGEIST_LIVE_UNSPLASH', '0')
    with pytest.raises(ValueError, match='decodable image'):
        vr.select_and_render_pulse_assets({'word': 'Anchor'}, repo_root=tmp_path,
            candidates=[dict(SCORES, id='missing', exif={'make': 'Reported Camera'})],
            medium_history_30d=[], photo_history_30d=[])
    assert not (tmp_path / 'assets').exists()


@pytest.mark.parametrize('description,exif,expected,basis', [
    ('This image is a digital render created in Blender.', {'make': 'Canon', 'model': 'EOS'},
     'UNVERIFIED_IMAGE', 'conflicting_evidence'),
    ('This image is a digital render created in Blender.', {},
     'DIGITAL_ARTWORK', 'source_declared_digital'),
    ('A digital artwork exhibition photographed through a gallery window.', {},
     'UNVERIFIED_IMAGE', 'insufficient_evidence'),
    ('A 3D render on a computer screen.', {},
     'UNVERIFIED_IMAGE', 'insufficient_evidence'),
])
def test_creation_claim_and_photographed_subject_are_not_interchangeable(tmp_path, monkeypatch, description, exif, expected, basis):
    detail = photo('winner')
    detail.update(description=description, exif=exif)
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    assert pulse.visual_medium == expected
    assert pulse.to_feed_entry()['media_provenance']['basis'] == basis


@pytest.mark.parametrize('description,process', [
    ('The artwork is a 3D particle system.', '3d_particle_system'),
    ('This image is an AI-generated artwork.', 'ai_generated'),
])
def test_bounded_process_claim_remains_inspectable_and_distinguishes_reported_ai(tmp_path, monkeypatch, description, process):
    import hashlib
    detail = photo('winner')
    detail.update(description=description, exif={})
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    entry = pulse.to_feed_entry()
    claim = entry['media_provenance']['source_claim']
    statement = description.rstrip('.')
    assert claim == {'kind': 'creation_declaration', 'process': process, 'statement': statement}
    assert entry['media_provenance']['source_statement_sha256'] == hashlib.sha256(statement.encode()).hexdigest()
    from scripts.export_public_release import _editorial
    assert _editorial(entry)['media_provenance']['source_claim'] == claim
    assert verify_publication_bundle(tmp_path)['media_provenance']['source_claim'] == claim


def test_public_export_rejects_downgraded_explicit_creation_claim_as_photography(tmp_path, monkeypatch):
    detail = photo('winner')
    detail['description'] = 'This image is an AI-generated artwork.'
    pulse = publish_source(tmp_path, monkeypatch, detail=detail)
    entry = pulse.to_feed_entry()
    entry.update(visual_medium='PHOTOGRAPHY', visual_tier_used='PHOTOGRAPHY')
    entry['media_provenance'].update(classification='PHOTOGRAPHY', basis='camera_reported', allowance_exception=False)
    entry['media_provenance']['source_claim']['kind'] = 'process_mention'
    from scripts.export_public_release import _editorial
    with pytest.raises(ValueError):
        _editorial(entry)
