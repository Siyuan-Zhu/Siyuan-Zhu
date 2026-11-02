import copy

import pytest


def current_entry():
    # This is a complete, deterministic photography DTO. Never inherit the
    # current edition's medium/evidence and then relabel it as a photograph.
    return {
        'schema_version': '2.0', 'date': '2026-11-01',
        'publication_id': '0123456789abcdef0123', 'word': 'Threshold',
        'gloss': 'Pressure gathers at a familiar edge.',
        'commissioned_thesis': 'Independent signals meet at a shared boundary.',
        'visual_medium': 'PHOTOGRAPHY',
        'image_url': 'https://unsplash.com/photos/fixture-photo',
        'fallback_webp_path': 'assets/pulse/2026-11-01.webp',
        'unsplash_photographer': 'Fixture Photographer',
        'unsplash_profile_url': 'https://unsplash.com/@fixture',
        'flux_model': None, 'cluster_keywords': ['boundary'],
        'orbit_contribution_pct': 60.0, 'macro_contribution_pct': 40.0,
        'orbit_sources_matched': ['fixture-source'], 'runner_up_clusters': [],
        'mood_votes': {'spot_on': 0, 'too_sunny': 0, 'too_heavy': 0},
        'critic_scores': {'semantic_alignment': 0.9, 'composite': 0.9},
        'published_at': '2026-11-01T00:00:00Z', 'updated_at': '2026-11-01T00:00:00Z',
        'image_data_uri': '', 'photo_selection_policy': 'owner-approved-best-safe-photo',
        'photo_selection_policy_date': '2026-11-01',
        'media_provenance': {
            'version': 1, 'classification': 'PHOTOGRAPHY', 'basis': 'camera_reported',
            'provider': 'unsplash', 'asset_id': 'fixture-photo',
            'source_url': 'https://unsplash.com/photos/fixture-photo',
            'camera_make': 'SONY', 'camera_model': 'ILCE-6300',
            'allowance_exception': False, 'source_statement_sha256': '',
            'source_claim': {'kind': 'none', 'process': 'unknown', 'statement': ''},
        },
    }


def test_current_exception_is_valid_and_export_preserves_its_provenance():
    from src.zeitgeist.public_schema import validate_editorial_entry
    from scripts.export_public_release import _editorial
    entry = current_entry()
    validate_editorial_entry(entry)
    exported = _editorial(entry)
    for key in ('photo_selection_policy', 'photo_selection_policy_date'):
        assert exported[key] == entry[key]


@pytest.mark.parametrize('change', [
    {'photo_selection_policy': 'arbitrary'},
    {'photo_selection_policy_date': '2026-99-99'},
    {'photo_selection_policy_date': '1900-01-01'},
    {'visual_medium': 'FLUX_PRO'},
    {'private_note': 'must never become public'},
    {'image_data_uri': 'data:image/png;base64,forbidden'},
])
def test_invalid_or_unrecognized_provenance_is_rejected(change):
    from src.zeitgeist.public_schema import validate_editorial_entry
    with pytest.raises(ValueError):
        validate_editorial_entry(current_entry() | change)


@pytest.mark.parametrize('missing', ['photo_selection_policy', 'photo_selection_policy_date'])
def test_partial_policy_pair_is_rejected(missing):
    from src.zeitgeist.public_schema import validate_editorial_entry
    entry = current_entry()
    del entry[missing]
    with pytest.raises(ValueError):
        validate_editorial_entry(entry)


def test_unrecognized_generated_field_blocks_before_canonical_commit(tmp_path, monkeypatch):
    from tests.test_publication_atomic_commit import _publication_fixture, _published_files
    from src.zeitgeist import adk_pipeline, visual_router
    wire, store = _publication_fixture(tmp_path, monkeypatch)
    before_remote = copy.deepcopy(wire.docs)
    before_files = _published_files(tmp_path)
    original = visual_router.PublishedPulse.to_feed_entry
    def contaminated(self, *args, **kwargs):
        return original(self, *args, **kwargs) | {'private_note': 'unrecognized output'}
    monkeypatch.setattr(visual_router.PublishedPulse, 'to_feed_entry', contaminated)
    with pytest.raises(ValueError, match='Disallowed public editorial'):
        adk_pipeline.ZeitgeistADKOrchestrator(tmp_path, store).run(
            '2026-10-30', prefer_visual_mode='blueprint', offline_demo=True)
    assert wire.docs == before_remote and wire.commits == []
    assert _published_files(tmp_path) == before_files


@pytest.mark.parametrize('mutation', ['none', 'medium_mismatch', 'extra_evidence', 'invalid_correction'])
def test_corrected_render_retains_strict_typed_public_contract(mutation):
    from src.zeitgeist.media_correction import evidence
    from src.zeitgeist.public_schema import validate_editorial_entry
    entry = current_entry()
    entry.update(visual_medium='DIGITAL_ARTWORK', media_provenance=evidence(),
                 media_correction={'kind': 'source_medium_correction',
                     'original_publication_id': '11111111111111111111',
                     'original_visual_medium': 'PHOTOGRAPHY'})
    if mutation == 'none':
        validate_editorial_entry(entry)
        return
    if mutation == 'medium_mismatch':
        entry['visual_medium'] = 'PHOTOGRAPHY'
    elif mutation == 'extra_evidence':
        entry['media_provenance']['private_note'] = 'not public'
    else:
        entry['media_correction']['kind'] = 'arbitrary'
    with pytest.raises(ValueError):
        validate_editorial_entry(entry)
