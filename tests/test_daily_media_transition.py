"""Edition metadata must survive real Firestore merge semantics between days."""
import copy

import pytest

from scripts.export_public_release import public_editorial_state
from src.state_store import FirestoreStateStore
from src.zeitgeist.media_provenance import unsplash_evidence
from src.zeitgeist.public_schema import validate_editorial_entry
from src.zeitgeist.publication_commit import publish_bundle
from tests.test_firestore_store import _RecordingFirestoreClient
from tests.test_public_editorial_contract import current_entry


@pytest.mark.parametrize('old_medium,new_medium', [
    ('DIGITAL_ARTWORK', 'PHOTOGRAPHY'),
    ('PHOTOGRAPHY', 'PHOTOGRAPHY'),
    ('PHOTOGRAPHY', 'UNVERIFIED_IMAGE'),
])
def test_new_edition_replaces_scoped_metadata_after_firestore_merge(old_medium, new_medium):
    old = current_entry()
    if old_medium == 'DIGITAL_ARTWORK':
        old.update(visual_medium=old_medium,
            media_provenance=unsplash_evidence({'id':'fixture-render','description':'3D render'}),
            media_correction={'kind':'source_medium_correction',
                'original_publication_id':'11111111111111111111',
                'original_visual_medium':'PHOTOGRAPHY'})
    validate_editorial_entry(old)
    previous = dict(copy.deepcopy(old), published_pulse=copy.deepcopy(old),
        visitor_mood_tally=copy.deepcopy(old['mood_votes']),
        retained_private_evidence={'keep':'private evidence is not an edition label'})
    client = _RecordingFirestoreClient()
    client.docs.update({'profile_state/daily_pulse':previous,
        'pulse_history/2026-11-01':copy.deepcopy(old),
        'pulse_feed/v1':{'entries':[copy.deepcopy(old)]}})
    store = FirestoreStateStore(client=client, canonical_only=True)
    entry = current_entry()
    for key in ('photo_selection_policy','photo_selection_policy_date'):
        entry.pop(key)
    entry.update(date='2026-11-02', publication_id='22222222222222222222',
        visual_medium=new_medium)
    entry['media_provenance'] = unsplash_evidence({'id':'fixture-next',
        'exif':{'make':'Canon','model':'EOS R5'}} if new_medium=='PHOTOGRAPHY' else {'id':'fixture-next'})
    validate_editorial_entry(entry)
    payload = {'date_iso':entry['date'], 'publication_id':entry['publication_id'],
        'published_pulse':copy.deepcopy(entry)}
    prepared=[]
    publish_bundle(store,payload,entry,expected=previous,local_feed={'entries':[]},
        prepare=lambda state,*rest: prepared.append(public_editorial_state(state)))
    actual=store.load_pulse_state()
    projected=public_editorial_state(actual)
    assert projected['media_provenance']==entry['media_provenance']
    assert projected['published_pulse']==entry
    for key in ('media_correction','photo_selection_policy','photo_selection_policy_date','release_republication'):
        assert key not in projected
    assert actual['retained_private_evidence']==previous['retained_private_evidence']
    assert 'retained_private_evidence' not in projected
    assert client.docs['pulse_history/2026-11-01']==old
    assert prepared[0]['media_provenance']==entry['media_provenance']
    assert client.docs['pulse_feed/v1']['entries'][1]==old


def test_nullable_clear_is_not_allowed_in_a_public_entry():
    entry=current_entry()
    entry['media_provenance']=None
    with pytest.raises(ValueError):
        validate_editorial_entry(entry)


def test_same_date_metadata_removal_stops_before_any_write():
    old=current_entry()
    client=_RecordingFirestoreClient()
    root=dict(copy.deepcopy(old),published_pulse=copy.deepcopy(old))
    client.docs.update({'profile_state/daily_pulse':root,
        'pulse_history/2026-11-01':copy.deepcopy(old),
        'pulse_feed/v1':{'entries':[copy.deepcopy(old)]}})
    before=copy.deepcopy(client.docs)
    entry=copy.deepcopy(old)
    entry.pop('photo_selection_policy')
    entry.pop('photo_selection_policy_date')
    store=FirestoreStateStore(client=client,canonical_only=True)
    with pytest.raises(ValueError,match='Same-date metadata removal'):
        publish_bundle(store,root,entry,expected=root,local_feed={'entries':[]},
            prepare=lambda *args: pytest.fail('Invalid transition must stop before staging'))
    assert client.docs==before
