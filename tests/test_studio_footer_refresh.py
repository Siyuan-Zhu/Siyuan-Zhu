"""The narrow display correction must not replay providers or accepted activity."""
import base64
import copy
import json
from pathlib import Path
import shutil

from PIL import Image, ImageChops, ImageDraw
import pytest

from scripts import refresh_studio_footer as footer
from src import state_store as stores
from src.zeitgeist import curatorial_compositor as compositor
from src.zeitgeist.publication import write_publication_manifest, verify_publication_bundle
from tests.test_publication_atomic_commit import WireFirestore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    monkeypatch.setenv('STATE_STORE_BACKEND', 'json')
    for name in ('assets', 'data', 'config', 'variants'):
        shutil.copytree(ROOT / name, tmp_path / name)
    for name in ('README.md', 'PROJECTS.md'):
        shutil.copy2(ROOT / name, tmp_path / name)
    entry = json.loads((ROOT / 'data/zeitgeist_state.json').read_text())['published_pulse']
    entry.update(date='2026-11-01', word='Still', gloss='A quieter signal.',
                 visual_medium='PHOTOGRAPHY', curatorial_archetype='STUDIO_MONOGRAPH',
                 fallback_webp_path='assets/pulse/2026-11-01.webp',
                 unsplash_photographer='Fixture', artwork_artist='Fixture',
                 photo_selection_policy='owner-approved-best-safe-photo',
                 photo_selection_policy_date='2026-11-01')
    for theme in ('light', 'dark'):
        image = compositor._render_studio_monograph_plate(Image.new('RGB', (1200, 800), 'blue'),
            word=entry['word'], gloss=entry['gloss'], date_str=entry['date'],
            attribution={'photographer_name': 'Fixture'}, visual_medium='PHOTOGRAPHY', dark_mode=theme == 'dark')
        # Model the deployed legacy prefix, including lossy compression.
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 716, 1199, 747), fill=(15,20,25) if theme == 'dark' else (251,249,245))
        font = compositor._load_font(compositor._MONO_FONT_CANDIDATES, 11)
        credit = 'PLATE 2026-11-01  ·  PHOTOGRAPH · FIXTURE / UNSPLASH'
        box = draw.textbbox((0,0), credit, font=font)
        draw.text((600-(box[2]-box[0])//2,724), credit, font=font, fill=(120,113,108))
        image.save(tmp_path / f'assets/daily_pulse_{theme}.webp', format='WEBP', quality=85)
    encoded = base64.b64encode((tmp_path / 'assets/daily_pulse_dark.webp').read_bytes()).decode()
    (tmp_path / 'assets/daily_pulse.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" data-publication-layout="curatorial"><metadata id="mood-votes">{}</metadata><image href="data:image/webp;base64,'+encoded+'"/></svg>')
    (tmp_path / 'data/pulse_feed.json').write_text(json.dumps({'entries':[entry]}))
    manifest = write_publication_manifest(tmp_path, date=entry['date'], word=entry['word'], gloss=entry['gloss'],
        archetype='STUDIO_MONOGRAPH', medium='PHOTOGRAPHY', archive_path=entry['fallback_webp_path'])
    entry['publication_id'] = manifest['publication_id']
    pulse = {'date':entry['date'], 'word':entry['word'], 'gloss':entry['gloss'],
             'visual_medium':'PHOTOGRAPHY', 'curatorial_archetype':'STUDIO_MONOGRAPH',
             'published_pulse':entry, 'publication_id':entry['publication_id'],
             'state_version':3, 'visitor_mood_tally':{'spot_on':2,'too_sunny':0,'too_heavy':0}}
    (tmp_path / 'data/zeitgeist_state.json').write_text(json.dumps(pulse))
    (tmp_path / 'data/pulse_feed.json').write_text(json.dumps({'entries':[entry]}))
    docs = {'staging_profile_state/daily_pulse':pulse,
        'staging_pulse_history/2026-11-01':entry,
        'staging_pulse_feed/v1':{'entries':[entry]},
        'staging_pulse_votes/2026-11-01':pulse['visitor_mood_tally']}
    for doc, name in [('active_game','game_state'), ('elo_ledger','hall_of_records'),
                      ('bot_memory','bot_brain'), ('orbit_roster','orbit_roster')]:
        docs['staging_profile_state/'+doc] = json.loads((tmp_path / f'data/{name}.json').read_text())
    wire = WireFirestore(monkeypatch, docs)
    store = stores.get_state_store(tmp_path, backend='firestore', firestore_client=wire.client, collection_prefix='staging_')
    monkeypatch.setattr(footer, 'git_commit_and_push_with_retry', lambda *a, **kw: True)
    return tmp_path, pulse, wire, store


def test_preview_preserves_all_pixels_outside_footer_and_archive(accepted):
    root, pulse, wire, store = accepted
    old = {t: Image.open(root / f'assets/daily_pulse_{t}.webp').convert('RGB') for t in ('light','dark')}
    original = {p.relative_to(root):p.read_bytes() for p in (root/'assets').rglob('*') if p.is_file()}
    docs = copy.deepcopy(wire.docs)
    result = footer.refresh(root, pulse['publication_id'])
    assert result['publication_id'] != pulse['publication_id']
    assert wire.docs == docs and not wire.commits
    for theme in ('light','dark'):
        with Image.open(root / f'.footer-preview/daily_pulse_{theme}.webp') as new:
            diff = ImageChops.difference(old[theme], new.convert('RGB'))
            assert diff.getbbox() and not diff.crop((0,0,1200,716)).getbbox()
            assert not diff.crop((0,748,1200,800)).getbbox()
    assert all((root / p).read_bytes() == raw for p,raw in original.items())


def test_apply_preserves_game_ratings_history_dates_and_concurrent_votes(accepted):
    root, pulse, wire, store = accepted
    before = copy.deepcopy(wire.docs)
    def vote(server):
        server.docs['staging_pulse_votes/2026-11-01']['spot_on'] = 3
        server.versions['staging_pulse_votes/2026-11-01'] += 1
    wire.before_commit = vote
    result = footer.refresh(root, pulse['publication_id'], store=store, apply=True)
    assert len(wire.commits) == 2
    assert result['publication_id'] == verify_publication_bundle(root, require_current_pulse=True)['publication_id']
    state = wire.docs['staging_profile_state/daily_pulse']
    assert state['visitor_mood_tally']['spot_on'] == 3
    assert state['published_pulse']['mood_votes']['spot_on'] == 3
    assert state['state_version'] == pulse['state_version'] + 1
    for key,value in before.items():
        if key.startswith('staging_profile_state/') and key != 'staging_profile_state/daily_pulse':
            assert wire.docs[key] == value
    old_entry = before['staging_pulse_history/2026-11-01']
    new_entry = wire.docs['staging_pulse_history/2026-11-01']
    for key in old_entry.keys() - {'publication_id', 'mood_votes', 'schema_version'}:
        assert new_entry[key] == old_entry[key]
    assert not (root / 'data/publication_reconcile.json').exists()
    with pytest.raises(ValueError, match='Canonical publication changed'):
        footer.refresh(root, pulse['publication_id'], store=store, apply=True)


def test_new_editorial_revision_blocks_before_any_write(accepted):
    root, pulse, wire, store = accepted
    wire.docs['staging_profile_state/daily_pulse']['publication_id'] = 'a'*20
    with pytest.raises(ValueError, match='Canonical publication changed'):
        footer.refresh(root, pulse['publication_id'], store=store, apply=True)
    assert not wire.commits


def test_git_failure_keeps_new_canonical_and_recovery_journal(accepted, monkeypatch):
    root, pulse, wire, store = accepted
    monkeypatch.setattr(footer, 'git_commit_and_push_with_retry', lambda *a, **kw: False)
    with pytest.raises(RuntimeError, match='Git reconciliation required'):
        footer.refresh(root, pulse['publication_id'], store=store, apply=True)
    new = wire.docs['staging_profile_state/daily_pulse']['publication_id']
    assert new != pulse['publication_id']
    marker = json.loads((root / 'data/publication_reconcile.json').read_text())
    assert marker['status'] == 'canonical_committed_git_pending'
    assert (root / '.publication-recovery' / new / 'assets/pulse_manifest.json').exists()
    with pytest.raises(RuntimeError, match='Resolve pending publication'):
        footer.refresh(root, pulse['publication_id'], store=store, apply=True)
    assert len(wire.commits) == 1


def test_ambiguous_commit_retains_journal_without_replay(accepted):
    root, pulse, wire, store = accepted
    wire.response_error = TimeoutError('response unavailable')
    with pytest.raises(stores.PublicationOutcomeUnknown):
        footer.refresh(root, pulse['publication_id'], store=store, apply=True)
    assert len(wire.commits) == 1
    assert json.loads((root / 'data/publication_reconcile.json').read_text())['status'] == 'canonical_outcome_unknown'


def test_rehearsal_creates_only_unique_fixture_namespace_and_preserves_sandbox(accepted, monkeypatch):
    from tests.test_profile_readiness import environment
    root, pulse, wire, store = accepted
    env = environment() | {'GITHUB_RUN_ID':'123456', 'GITHUB_RUN_ATTEMPT':'1'}
    before = copy.deepcopy(wire.docs)
    monkeypatch.setattr(footer, 'get_state_store', lambda root, backend, collection_prefix:
        stores.get_state_store(root, backend=backend, firestore_client=wire.client,
                               collection_prefix=collection_prefix))
    result = footer.rehearse(root, pulse['publication_id'], env)
    assert result['status'] == 'sandbox_rehearsed'
    prefix = result['fixture_namespace']
    assert prefix == 'staging_footer_rehearsal_123456_1_'
    assert all(wire.docs[key] == value for key,value in before.items())
    assert all(key.startswith(prefix) for key in wire.docs.keys() - before.keys())
    assert len(wire.commits) == 2  # One create-only seed, one conditional correction.
    assert all(write['currentDocument'] == {'exists':False} for write in wire.commits[0]['writes'])
    with pytest.raises(ValueError, match='Exact accepted fixture'):
        footer.rehearse(root, pulse['publication_id'], env)
    assert len(wire.commits) == 2
