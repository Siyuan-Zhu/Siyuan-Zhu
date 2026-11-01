import copy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def current_entry():
    entry = json.loads((ROOT / 'data/zeitgeist_state.json').read_text())['published_pulse']
    # Keep synthetic policy cases stable after a later ordinary edition. The
    # checked-in feed itself is validated separately by the runtime-schema test.
    entry.update(visual_medium='PHOTOGRAPHY', photo_selection_policy='owner-approved-best-safe-photo',
                 photo_selection_policy_date=str(entry['date'])[:10])
    return entry


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
