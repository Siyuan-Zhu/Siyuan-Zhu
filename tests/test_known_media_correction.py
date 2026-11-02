"""Provider-free source correction preserves an accepted edition and activity."""
import base64
import copy
import json
from pathlib import Path
import shutil

from PIL import Image, ImageChops
import pytest

from scripts import refresh_studio_footer as footer
from src import state_store as stores
from src.zeitgeist import curatorial_compositor as compositor
from src.zeitgeist.publication import verify_publication_bundle, write_publication_manifest
from tests.test_publication_atomic_commit import WireFirestore

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    for name in ('assets', 'data', 'config', 'variants'):
        shutil.copytree(ROOT / name, tmp_path / name)
    for name in ('README.md', 'PROJECTS.md','docs/resonance.md'):
        (tmp_path/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT / name, tmp_path / name)
    entry = json.loads((ROOT / 'data/zeitgeist_state.json').read_text())['published_pulse']
    for name in ('media_provenance', 'media_correction', 'release_republication'):
        entry.pop(name, None)
    entry.update(date='2026-11-01', word='Insinuation', gloss='Silent infiltration of trusted synthetic minds.',
        visual_medium='PHOTOGRAPHY', visual_tier_used='PHOTOGRAPHY',
        fallback_webp_path='assets/pulse/2026-11-01.webp', artwork_source='Unsplash',
        unsplash_photographer='Richard Horvath', artwork_artist='Richard Horvath',
        unsplash_profile_url='https://unsplash.com/@ricvath',
        artwork_url='https://unsplash.com/photos/WOA3QKFjlo8', image_url='https://unsplash.com/photos/WOA3QKFjlo8',
        photo_selection_policy='owner-approved-best-safe-photo', photo_selection_policy_date='2026-11-01')
    for theme in ('light', 'dark'):
        plate = compositor._render_studio_monograph_plate(Image.new('RGB', (1200, 800), 'blue'),
            word=entry['word'], gloss=entry['gloss'], date_str=entry['date'],
            attribution={'photographer_name':'Richard Horvath'}, visual_medium='PHOTOGRAPHY', dark_mode=theme=='dark')
        plate.save(tmp_path/f'assets/daily_pulse_{theme}.webp', format='WEBP', quality=85)
    encoded = base64.b64encode((tmp_path/'assets/daily_pulse_dark.webp').read_bytes()).decode()
    (tmp_path/'assets/daily_pulse.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" data-publication-layout="curatorial"><metadata id="mood-votes">{}</metadata><image href="data:image/webp;base64,'+encoded+'"/></svg>')
    feed = {'version':'2.0','schema_version':'2.0','entries':[entry], 'updated_at':entry['updated_at']}
    (tmp_path/'data/pulse_feed.json').write_text(json.dumps(feed))
    manifest = write_publication_manifest(tmp_path, date=entry['date'],word=entry['word'],gloss=entry['gloss'],archetype='STUDIO_MONOGRAPH',medium='PHOTOGRAPHY',archive_path=entry['fallback_webp_path'])
    entry['publication_id'] = manifest['publication_id']
    pulse = {'date':entry['date'],'date_iso':entry['date'],'word':entry['word'],'gloss':entry['gloss'],
        'visual_medium':'PHOTOGRAPHY','curatorial_archetype':'STUDIO_MONOGRAPH',
        'visual_decision':{'selected_mode':'CURATED_PHOTO'},'recent_words':['INSINUATION'],
        'published_pulse':entry,'publication_id':entry['publication_id'], 'schema_version':3,
        'published_at':entry['published_at'],'updated_at':entry['updated_at'],
        'state_version':4,'visitor_mood_tally':{'spot_on':2,'too_sunny':0,'too_heavy':0},
        'mood_votes':{'spot_on':2,'too_sunny':0,'too_heavy':0}}
    (tmp_path/'data/zeitgeist_state.json').write_text(json.dumps(pulse))
    (tmp_path/'data/pulse_feed.json').write_text(json.dumps(feed))
    docs = {'staging_profile_state/daily_pulse':pulse,'staging_pulse_history/2026-11-01':entry,
        'staging_pulse_feed/v1':feed,'staging_pulse_votes/2026-11-01':pulse['visitor_mood_tally']}
    for doc, name in [('active_game','game_state'),('elo_ledger','hall_of_records'),('bot_memory','bot_brain'),('orbit_roster','orbit_roster')]:
        docs['staging_profile_state/'+doc] = json.loads((tmp_path/f'data/{name}.json').read_text())
    wire = WireFirestore(monkeypatch, docs)
    store = stores.get_state_store(tmp_path, backend='firestore', firestore_client=wire.client, collection_prefix='staging_')
    monkeypatch.setattr(footer,'git_commit_and_push_with_retry',lambda *a,**kw:True)
    return tmp_path,pulse,wire,store


def test_reviewable_preview_preserves_pixels_archive_date_and_original_selection(accepted):
    root, pulse, wire, store = accepted
    before = copy.deepcopy(wire.docs)
    assets = {p.relative_to(root):p.read_bytes() for p in (root/'assets').rglob('*') if p.is_file()}
    plan = footer.prepare_media_correction(root, pulse['publication_id'])
    result = footer.refresh(root,pulse['publication_id'],media_plan=plan)
    preview = Path(result['preview_path'])
    manifest = verify_publication_bundle(preview,require_current_pulse=True)
    assert manifest['medium']=='DIGITAL_ARTWORK' and manifest['publication_id']!=pulse['publication_id']
    entry = json.loads((preview/'data/zeitgeist_state.json').read_text())['published_pulse']
    assert entry['media_correction']['original_publication_id']==pulse['publication_id']
    assert entry['media_provenance']['source_claim']['process']=='3d_particle_system'
    for field in ('word','gloss','date','published_at','updated_at','photo_selection_policy','photo_selection_policy_date'):
        assert entry[field]==pulse['published_pulse'][field]
    assert (preview/'assets/pulse/2026-11-01.webp').read_bytes()==assets[Path('assets/pulse/2026-11-01.webp')]
    for theme in ('light','dark'):
        old = Image.open(root/f'assets/daily_pulse_{theme}.webp').convert('RGB')
        new = Image.open(preview/f'assets/daily_pulse_{theme}.webp').convert('RGB')
        diff = ImageChops.difference(old,new)
        assert diff.getbbox() and not diff.crop((0,0,1200,716)).getbbox() and not diff.crop((0,748,1200,800)).getbbox()
    assert wire.docs==before and not wire.commits
    assert all((root/path).read_bytes()==raw for path,raw in assets.items())


def test_conditional_apply_updates_effective_metadata_and_preserves_concurrent_votes(accepted):
    root,pulse,wire,store=accepted
    before=copy.deepcopy(wire.docs)
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    def vote(server):
        server.docs['staging_pulse_votes/2026-11-01']['spot_on']=3
        server.versions['staging_pulse_votes/2026-11-01']+=1
    wire.before_commit=vote
    result=footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    state=wire.docs['staging_profile_state/daily_pulse']
    assert state['visual_medium']==state['published_pulse']['visual_medium']=='DIGITAL_ARTWORK'
    assert state['visual_decision']['selected_mode']=='DIGITAL_ARTWORK'
    assert state['visitor_mood_tally']['spot_on']==state['published_pulse']['mood_votes']['spot_on']==3
    assert state['published_pulse']['photo_selection_policy_date']=='2026-11-01'
    assert state['media_correction']==state['published_pulse']['media_correction']
    assert result['publication_id']==verify_publication_bundle(root,require_current_pulse=True)['publication_id']
    assert len(wire.commits)==2
    for key,value in before.items():
        if key.startswith('staging_profile_state/') and key!='staging_profile_state/daily_pulse':
            assert wire.docs[key]==value


@pytest.mark.parametrize('changed', ['history','profile'])
def test_same_id_nested_metadata_race_is_not_overwritten_on_retry(accepted, changed):
    root,pulse,wire,store=accepted
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    key = 'staging_pulse_history/2026-11-01' if changed=='history' else 'staging_profile_state/daily_pulse'
    def race(server):
        target=server.docs[key] if changed=='history' else server.docs[key]['published_pulse']
        target['artwork_artist']='Concurrent source correction'
        server.versions[key]+=1
    wire.before_commit=race
    with pytest.raises(ValueError):
        footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    assert len(wire.commits)==1
    assert wire.docs['staging_profile_state/daily_pulse']['publication_id']==pulse['publication_id']
    assert not (root/'data/publication_reconcile.json').exists()


@pytest.mark.parametrize('failure', ['commit_response','install','git'])
def test_unknown_or_committed_failure_retains_complete_self_validating_successor(accepted,monkeypatch,failure):
    root,pulse,wire,store=accepted
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    error=RuntimeError
    if failure=='commit_response':
        wire.response_error=TimeoutError('response lost')
        error=stores.PublicationOutcomeUnknown
    elif failure=='git':
        monkeypatch.setattr(footer,'git_commit_and_push_with_retry',lambda *a,**kw:False)
    else:
        original=Path.replace
        def replace(path,target):
            if Path(target)==root/'assets/daily_pulse_dark.webp':
                raise OSError('local install failed')
            return original(path,target)
        monkeypatch.setattr(Path,'replace',replace)
        error=OSError
    with pytest.raises(error):
        footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    marker=json.loads((root/'data/publication_reconcile.json').read_text())
    from src.zeitgeist.media_correction import verify_retained
    receipt=verify_retained(marker['recovery_path'])
    assert receipt['publication_id']==wire.docs['staging_profile_state/daily_pulse']['publication_id']
    assert receipt['previous_publication_id']==pulse['publication_id']
    assert (Path(marker['recovery_path'])/'data/zeitgeist_state.json').exists()
    assert (Path(marker['recovery_path'])/'data/pulse_feed.json').exists()
    assert len(wire.commits)==1
    with pytest.raises(RuntimeError,match='Resolve pending publication'):
        footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    assert len(wire.commits)==1


@pytest.mark.parametrize('change',['history','asset','plan'])
def test_changed_reviewed_precondition_rejects_before_any_commit(accepted,change):
    root,pulse,wire,store=accepted
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    if change=='history':
        wire.docs['staging_pulse_history/2026-11-01']['critic_scores']['composite']=.123
    elif change=='asset':
        (root/'assets/daily_pulse_light.webp').write_bytes(b'changed source asset')
    else:
        plan['evidence']['source_claim']['process']='ai_generated'
    with pytest.raises(ValueError):
        footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    assert not wire.commits
    assert not (root/'data/publication_reconcile.json').exists()


def test_recovery_rejects_missing_public_json_and_preview_has_no_private_config(accepted):
    root,pulse,wire,store=accepted
    plan=footer.prepare_media_correction(root,pulse['publication_id'])
    result=footer.refresh(root,pulse['publication_id'],media_plan=plan)
    preview=Path(result['preview_path'])
    assert not (preview/'config').exists() and not (preview/'.env').exists()
    from src.zeitgeist.media_correction import verify_retained
    verify_retained(preview)
    (preview/'data/pulse_feed.json').unlink()
    with pytest.raises(ValueError):
        verify_retained(preview)


def test_exact_successor_reconciliation_uses_retained_art_latest_votes_and_new_remote_head(accepted,monkeypatch):
    from tests.test_generated_publication_retry import git
    root,pulse,wire,store=accepted
    monkeypatch.setenv('GITHUB_REF_NAME','main')
    monkeypatch.setenv('TARGET_REPO_SLUG','Siyuan-Zhu/readme-live-sandbox')
    shutil.copy2(ROOT/'.gitignore',root/'.gitignore')
    origin=root.parent/(root.name+'-origin.git')
    git(root,'init','--bare',str(origin))
    git(root,'init','-b','main')
    git(root,'config','user.name','Fixture');git(root,'config','user.email','fixture@example.invalid')
    git(root,'remote','add','origin',str(origin));git(root,'add','.');git(root,'commit','-m','accepted parent')
    git(root,'push','origin','HEAD:main')
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    monkeypatch.setattr(footer,'git_commit_and_push_with_retry',lambda *a,**kw:False)
    with pytest.raises(RuntimeError):
        footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    successor=wire.docs['staging_profile_state/daily_pulse']['publication_id']
    accepted_vote,tally=store.record_mood_vote('2026-11-01','spot_on','a'*64)
    assert accepted_vote and tally['spot_on']==3
    writes=len(wire.commits)
    # Model a fresh worker with the old public checkout and retained successor.
    git(root,'restore','--source','HEAD','--','assets','data','README.md','variants')
    remote=root.parent/(root.name+'-remote')
    git(root,'clone',str(origin),str(remote));git(remote,'checkout','main')
    git(remote,'config','user.name','Other');git(remote,'config','user.email','other@example.invalid')
    (remote/'newest-source.txt').write_text('preserve latest remote source')
    git(remote,'add','newest-source.txt');git(remote,'commit','-m','independent queued update');git(remote,'push','origin','HEAD:main')
    monkeypatch.setattr(footer,'git_commit_and_push_with_retry',stores.git_commit_and_push_with_retry)
    result=footer.reconcile_media(root,pulse['publication_id'],plan['plan_sha256'],store=store)
    assert result['status']=='reconciled' and result['publication_id']==successor
    git(remote,'pull','--ff-only')
    assert verify_publication_bundle(remote,require_current_pulse=True)['publication_id']==successor
    assert (remote/'newest-source.txt').read_text()=='preserve latest remote source'
    assert 'Digital artwork by' in (remote/'README.md').read_text() and 'Spot On 3' in (remote/'README.md').read_text()
    assert len(wire.commits)==writes and not (root/'data/publication_reconcile.json').exists()


from tests.test_generated_publication_retry import queued


def test_exact_publication_expectation_blocks_publisher_advancement_without_push(queued):
    from tests.test_generated_publication_retry import git
    root,remote,origin,store,client,_=queued
    before=git(remote,'ls-remote','origin','refs/heads/main')
    authority=copy.deepcopy(client.docs)
    assert not stores.git_commit_and_push_with_retry('exact correction only',repo_root=root,
        store=store,sleep_fn=lambda _:None,expected_publication_id='f'*20)
    assert git(remote,'ls-remote','origin','refs/heads/main')==before
    assert client.docs==authority


@pytest.mark.parametrize('outcome',['all-old','mixed','newer'])
def test_reconciliation_rejects_unproven_or_superseded_canonical_outcome(accepted,monkeypatch,outcome):
    root,pulse,wire,store=accepted
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    if outcome=='all-old':
        def lost_before_acceptance(server):
            raise TimeoutError('outcome unknown before acceptance')
        wire.before_commit=lost_before_acceptance
    else:
        wire.response_error=TimeoutError('outcome unknown after acceptance')
    with pytest.raises(stores.PublicationOutcomeUnknown):
        footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    if outcome=='mixed':
        wire.docs['staging_pulse_history/2026-11-01']['artwork_artist']='Unrecognized concurrent correction'
    elif outcome=='newer':
        wire.docs['staging_profile_state/daily_pulse']['publication_id']='f'*20
    monkeypatch.setattr(footer,'git_commit_and_push_with_retry',lambda *a,**kw:pytest.fail('Unproven successor must not publish'))
    with pytest.raises(ValueError):
        footer.reconcile_media(root,pulse['publication_id'],plan['plan_sha256'],store=store)
    assert len(wire.commits)==1 and (root/'data/publication_reconcile.json').exists()


def test_declared_workflow_recovery_upload_is_self_validating_after_download(accepted,monkeypatch):
    root,pulse,wire,store=accepted
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    monkeypatch.setattr(footer,'git_commit_and_push_with_retry',lambda *a,**kw:False)
    with pytest.raises(RuntimeError):
        footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    marker=json.loads((root/'data/publication_reconcile.json').read_text())
    workflow=(ROOT/'.github/workflows/daily_zeitgeist.yml').read_text()
    job=workflow.split('  studio-footer:',1)[1].split('  synthesize-pulse:',1)[0]
    patterns=job.split('          path: |\n',1)[1].splitlines()
    downloaded=root.parent/(root.name+'-downloaded')
    for pattern in patterns:
        for match in root.glob(pattern.strip()):
            for source in match.rglob('*') if match.is_dir() else (match,):
                if source.is_file():
                    target=downloaded/source.relative_to(root)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(source,target)
    from src.zeitgeist.media_correction import verify_retained
    verify_retained(downloaded/'.publication-recovery'/marker['publication_id'])
    assert not list(downloaded.rglob('canonical-parent.json'))


def test_read_only_canonical_preview_returns_the_apply_plan_identity(accepted,monkeypatch,capsys):
    from tests.test_profile_readiness import environment
    root,pulse,wire,store=accepted
    for key,value in environment().items():
        monkeypatch.setenv(key,str(value))
    monkeypatch.setenv('GITHUB_REF_NAME','codex/private-correction-preview')
    monkeypatch.setattr(footer,'ROOT',root)
    monkeypatch.setattr(footer,'get_state_store',lambda *a,**kw:store)
    assert footer.main(['--correct-media','--canonical-preview','--expected-publication-id',pulse['publication_id']])==0
    result=json.loads(capsys.readouterr().out.splitlines()[-1])
    expected=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    assert result['plan_sha256']==expected['plan_sha256']
    assert not wire.commits


def test_raw_canonical_private_metadata_is_preserved_but_never_retained_or_installed_publicly(accepted):
    root,pulse,wire,store=accepted
    private='CANONICAL_PRIVATE_SENTINEL'
    canonical=wire.docs['staging_profile_state/daily_pulse']
    canonical['commission']={'private_prompt':private}
    canonical['telemetry']={'private_trace':private}
    canonical['analysis']={'private_note':private}
    # Display projection of an absent/empty analysis is the same; raw authority
    # remains fully pinned and is never copied into the public workspace.
    local=json.loads((root/'data/zeitgeist_state.json').read_text());local['analysis']={}
    (root/'data/zeitgeist_state.json').write_text(json.dumps(local))
    for entry in (canonical['published_pulse'],wire.docs['staging_pulse_history/2026-11-01'],
                  wire.docs['staging_pulse_feed/v1']['entries'][0]):
        entry['private_source_note']=private
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    result=footer.refresh(root,pulse['publication_id'],store=store,media_plan=plan)
    preview=Path(result['preview_path'])
    assert all(private not in p.read_text() for p in preview.rglob('*.json'))
    assert not wire.commits
    footer.refresh(root,pulse['publication_id'],store=store,apply=True,media_plan=plan)
    assert wire.docs['staging_profile_state/daily_pulse']['commission']['private_prompt']==private
    assert wire.docs['staging_pulse_history/2026-11-01']['private_source_note']==private
    assert private not in (root/'data/zeitgeist_state.json').read_text()
    assert private not in (root/'data/pulse_feed.json').read_text()


def test_normal_canonical_publisher_uses_closed_public_projection_without_mutating_authority(queued):
    from tests.test_generated_publication_retry import git
    root,remote,origin,store,client,_=queued
    private='NORMAL_PUBLISH_PRIVATE_SENTINEL'
    client.docs['profile_state/daily_pulse']['commission']={'private_prompt':private}
    client.docs['profile_state/daily_pulse']['analysis']={'private_note':private}
    client.docs['pulse_history/2026-10-30']['private_note']=private
    client.docs['pulse_feed/v1']={'entries':[copy.deepcopy(client.docs['pulse_history/2026-10-30'])], 'private_trace':private}
    client.docs['profile_state/orbit_roster']={'handles':{'example':{'tier':'O3','status':'active','weight':.6,'private_discovery':private}}}
    before=copy.deepcopy(client.docs)
    assert stores.git_commit_and_push_with_retry('closed public display',repo_root=root,store=store,sleep_fn=lambda _:None)
    git(remote,'pull','--ff-only')
    assert all(private not in p.read_text() for p in (remote/'data').glob('*.json'))
    assert client.docs==before


def test_vote_canonical_mirror_does_not_write_private_editorial_fields(accepted):
    root,pulse,wire,store=accepted
    private='VOTE_MIRROR_PRIVATE_SENTINEL'
    wire.docs['staging_profile_state/daily_pulse']['commission']={'private_prompt':private}
    accepted_vote,_=store.record_mood_vote('2026-11-01','spot_on','b'*64)
    assert accepted_vote
    assert wire.docs['staging_profile_state/daily_pulse']['commission']['private_prompt']==private
    assert private not in (root/'data/zeitgeist_state.json').read_text()


def test_public_workflow_preview_never_writes_raw_original_backup(accepted,monkeypatch):
    root,pulse,wire,store=accepted
    monkeypatch.setenv('GITHUB_ACTIONS','true')
    plan=footer.prepare_media_correction(root,pulse['publication_id'],store=store)
    result=footer.refresh(root,pulse['publication_id'],store=store,media_plan=plan)
    assert not (root/'.release-backups').exists()
    assert not list(Path(result['preview_path']).rglob('canonical-parent.json'))
    assert not wire.commits
