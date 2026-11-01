"""Real local Git integration; all canonical authority is an injected fake."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import pytest
from src import state_store as state
from tests.test_firestore_store import _RecordingFirestoreClient
from src.game_engine import create_initial_game_state
from src.zeitgeist.publication import write_publication_manifest

ROOT=Path(__file__).resolve().parents[1]


def git(root,*args):
    return subprocess.run(['git',*args],cwd=root,check=True,capture_output=True,text=True).stdout.strip()


@pytest.fixture
def queued(tmp_path,monkeypatch):
    monkeypatch.setenv('GITHUB_REF_NAME','main')
    monkeypatch.setenv('TARGET_REPO_SLUG','Siyuan-Zhu/readme-live-sandbox')
    origin=tmp_path/'origin.git'; git(tmp_path,'init','--bare',str(origin))
    root=tmp_path/'event'; git(tmp_path,'clone',str(origin),str(root)); git(root,'checkout','-b','main')
    git(root,'config','user.name','Fixture'); git(root,'config','user.email','fixture@example.invalid')
    for name in ('assets','config'):
        shutil.copytree(ROOT/name,root/name)
    # Model the clean public export, independently of operational files a real
    # private sandbox run may have left in this checkout's data directory.
    (root/'data').mkdir()
    for name in ('game_state.json','zeitgeist_state.json','hall_of_records.json',
                 'bot_brain.json','pulse_feed.json','orbit_roster.json'):
        shutil.copy2(ROOT/'data'/name,root/'data'/name)
    shutil.copy2(ROOT/'PROJECTS.md',root/'PROJECTS.md')
    pulse={'date':'2026-10-30','word':'REAL','gloss':'A real accepted publication',
      'visual_medium':'PHOTOGRAPHY','curatorial_archetype':'POETIC_CINEMA',
      'visitor_mood_tally':{'spot_on':0,'too_sunny':0,'too_heavy':0},'state_version':1}
    (root/'assets/daily_pulse.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" data-publication-layout="curatorial"><metadata id="mood-votes">{}</metadata><rect width="10" height="10"/></svg>')
    (root/'assets/pulse/2026-10-30.webp').write_bytes((root/'assets/daily_pulse_light.webp').read_bytes())
    manifest=write_publication_manifest(root,date=pulse['date'],word=pulse['word'],gloss=pulse['gloss'],archetype=pulse['curatorial_archetype'],medium=pulse['visual_medium'],archive_path='assets/pulse/2026-10-30.webp')
    pulse['publication_id']=manifest['publication_id']
    # Public entries omit canonical state bookkeeping; keep the fixture on the
    # same DTO contract as PublishedPulse.to_feed_entry().
    entry={key: value for key, value in pulse.items()
           if key not in {'visitor_mood_tally', 'state_version'}}
    entry['mood_votes']=copy.deepcopy(pulse['visitor_mood_tally'])
    (root/'data/zeitgeist_state.json').write_text(json.dumps(pulse)); (root/'data/pulse_feed.json').write_text(json.dumps({'entries':[entry]}))
    game=create_initial_game_state(); game['state_version']=1
    (root/'data/game_state.json').write_text(json.dumps(game))
    git(root,'add','.'); git(root,'commit','-m','initial accepted publication'); git(root,'push','origin','HEAD:main')
    before=git(root,'rev-parse','HEAD')
    remote=tmp_path/'other'; git(tmp_path,'clone','-b','main',str(origin),str(remote)); git(remote,'config','user.name','Other'); git(remote,'config','user.email','other@example.invalid')
    (remote/'data/game_state.json').write_text('{"earlier_queue":"game"}')
    (remote/'data/zeitgeist_state.json').write_text('{"earlier_queue":"pulse"}')
    (remote/'data/bot_brain.json').write_text('{"earlier_queue":"brain"}')
    (remote/'data/hall_of_records.json').write_text('{"earlier_queue":"elo"}')
    (remote/'remote-source.txt').write_text('retain newest remote source')
    git(remote,'add','.'); git(remote,'commit','-m','earlier queued writer'); git(remote,'push','origin','HEAD:main')
    client=_RecordingFirestoreClient()
    client.docs.update({'profile_state/active_game':game,'profile_state/daily_pulse':pulse,
      'profile_state/elo_ledger':{'players':{},'bot_elo':1200},'profile_state/bot_memory':{'states':{},'games_learned':0},
      'profile_state/orbit_roster':{'handles':{}},'pulse_history/2026-10-30':entry})
    store=state.get_state_store(root,backend='firestore',firestore_client=client)
    return root,remote,origin,store,client,before


def test_queued_vote_rebuilds_remote_head_from_canonical_without_rebase_or_private_leak(queued,monkeypatch):
    root,remote,origin,store,client,before=queued
    pulse=client.docs['profile_state/daily_pulse']; pulse['visitor_mood_tally']['spot_on']=1; pulse['mood_votes']=dict(pulse['visitor_mood_tally'])
    client.docs['profile_state/active_game']['turn']=3
    (root/'data/zeitgeist_state.json').write_text(json.dumps(pulse))
    (root/'data/game_rate_limits.json').write_text('{"users":{"private-name":{}}}')
    (root/'data/private.json').write_text('{"private":"never-stage"}')
    client.docs['pulse_feed/v1']={'entries':[copy.deepcopy(client.docs['pulse_history/2026-10-30'])], 'updated_at':'2026-10-30T04:00:00Z'}
    authority=copy.deepcopy(client.docs)
    assert state.git_commit_and_push_with_retry('queued vote',repo_root=root,store=store,sleep_fn=lambda _:None)
    assert client.docs==authority
    assert git(root,'rev-parse','HEAD')==before  # Original event checkout remains recoverable.
    git(remote,'pull','--ff-only')
    assert (remote/'remote-source.txt').read_text()=='retain newest remote source'
    assert json.loads((remote/'data/zeitgeist_state.json').read_text())['visitor_mood_tally']['spot_on']==1
    assert json.loads((remote/'data/game_state.json').read_text())['turn']==3
    assert json.loads((remote/'data/pulse_feed.json').read_text())==client.docs['pulse_feed/v1']
    assert 'Spot On 1' in (remote/'README.md').read_text()
    assert 'turn=3:round=1' in (remote/'README.md').read_text()
    assert not (remote/'data/game_rate_limits.json').exists()
    assert not (remote/'data/private.json').exists()
    assert (remote/'assets/daily_pulse_light.webp').read_bytes()==(root/'assets/daily_pulse_light.webp').read_bytes()


def test_retry_preserves_inherited_remote_operational_file_without_publishing_local_changes(queued):
    root,remote,origin,store,client,before=queued
    inherited=remote/'data/game_rate_limits.json'
    inherited.write_text('{"legacy":"retained for private sandbox recovery"}')
    expected=inherited.read_bytes()
    git(remote,'add','data/game_rate_limits.json')
    git(remote,'commit','-m','inherited private sandbox state')
    git(remote,'push','origin','HEAD:main')
    (root/'data/game_rate_limits.json').write_text('{"users":{"new-private-state":{}}}')
    assert state.git_commit_and_push_with_retry('public projection only',repo_root=root,
        store=store,sleep_fn=lambda _:None)
    git(remote,'pull','--ff-only')
    assert inherited.read_bytes()==expected
    assert git(root,'rev-parse','HEAD')==before


def test_default_noncanonical_stage_excludes_private_operational_files(tmp_path,monkeypatch):
    monkeypatch.setenv('GITHUB_REF_NAME','main')
    origin=tmp_path/'origin.git'; git(tmp_path,'init','--bare',str(origin))
    root=tmp_path/'repo'; git(tmp_path,'clone',str(origin),str(root)); git(root,'checkout','-b','main')
    git(root,'config','user.name','Test'); git(root,'config','user.email','fixture@example.invalid')
    (root/'README.md').write_text('before'); git(root,'add','.'); git(root,'commit','-m','before'); git(root,'push','origin','HEAD:main')
    (root/'README.md').write_text('after'); (root/'data').mkdir(); (root/'data/private.json').write_text('secret'); (root/'data/game_rate_limits.json').write_text('private')
    assert state.git_commit_and_push_with_retry('public only',repo_root=root,store=state.InMemoryStateStore(),sleep_fn=lambda _:None)
    assert 'data/private.json' not in git(root,'ls-tree','-r','--name-only','HEAD')
    assert 'data/game_rate_limits.json' not in git(root,'ls-tree','-r','--name-only','HEAD')


def test_rejected_push_fetches_and_renders_again_without_force_or_replaying_vote(queued,monkeypatch):
    root,remote,origin,store,client,_=queued
    real_run=subprocess.run; pushes=[]
    def run(command,**kwargs):
        if command[:3]==['git','push','origin'] and str(kwargs.get('cwd'))!=str(remote):
            pushes.append(command)
            if len(pushes)==1:
                (remote/'concurrent-source.txt').write_text('newest queued change')
                real_run(['git','add','concurrent-source.txt'],cwd=remote,check=True,capture_output=True)
                real_run(['git','commit','-m','concurrent remote writer'],cwd=remote,check=True,capture_output=True)
                real_run(['git','push','origin','HEAD:main'],cwd=remote,check=True,capture_output=True)
                client.docs['profile_state/daily_pulse']['visitor_mood_tally']['spot_on']=2
        return real_run(command,**kwargs)
    monkeypatch.setattr(state.subprocess,'run',run)
    assert state.git_commit_and_push_with_retry('rebuild race',repo_root=root,store=store,sleep_fn=lambda _:None)
    git(remote,'pull','--ff-only')
    assert len(pushes)==2 and all('--force' not in command for command in pushes)
    assert 'Spot On 2' in (remote/'README.md').read_text()
    assert (remote/'concurrent-source.txt').read_text()=='newest queued change'


def test_missing_matching_art_fails_without_pushing_or_inference(queued,monkeypatch):
    root,remote,origin,store,client,_=queued
    (root/'assets/pulse_manifest.json').unlink()
    git(remote,'rm','assets/pulse_manifest.json'); git(remote,'commit','-m','art unavailable'); git(remote,'push','origin','HEAD:main')
    before=git(remote,'rev-parse','HEAD'); authority=copy.deepcopy(client.docs)
    from src.zeitgeist import adk_pipeline
    monkeypatch.setattr(adk_pipeline,'run_daily_zeitgeist',lambda *a,**kw:pytest.fail('Publication retry called providers'))
    assert not state.git_commit_and_push_with_retry('must fail closed',repo_root=root,store=store,sleep_fn=lambda _:None)
    assert git(remote,'ls-remote','origin','refs/heads/main').split()[0]==before
    assert client.docs==authority and store.needs_git_reconcile


def test_conflicting_local_owner_configuration_does_not_overwrite_remote(queued):
    root,remote,origin,store,client,_=queued
    (root/'config/orbit_sources.json').write_text('{"owner":"local"}')
    (remote/'config/orbit_sources.json').write_text('{"owner":"remote"}')
    git(remote,'add','config/orbit_sources.json'); git(remote,'commit','-m','newer owner config'); git(remote,'push','origin','HEAD:main')
    before=git(remote,'rev-parse','HEAD')
    assert not state.git_commit_and_push_with_retry('conflicting config',repo_root=root,store=store,sleep_fn=lambda _:None)
    assert git(remote,'ls-remote','origin','refs/heads/main').split()[0]==before
    assert (remote/'config/orbit_sources.json').read_text()=='{"owner":"remote"}'


def test_newer_canonical_publication_uses_its_remote_art_not_stale_event_art(queued):
    root,remote,origin,store,client,_=queued
    from PIL import Image
    Image.new('RGB',(20,20),'red').save(remote/'assets/daily_pulse_light.webp','WEBP')
    (remote/'assets/pulse/2026-10-30.webp').write_bytes((remote/'assets/daily_pulse_light.webp').read_bytes())
    pulse=client.docs['profile_state/daily_pulse']; pulse['word']='LATEST'
    manifest=write_publication_manifest(remote,date=pulse['date'],word=pulse['word'],gloss=pulse['gloss'],archetype=pulse['curatorial_archetype'],medium=pulse['visual_medium'],archive_path='assets/pulse/2026-10-30.webp')
    pulse['publication_id']=manifest['publication_id']
    entry={key: value for key, value in pulse.items()
           if key not in {'visitor_mood_tally', 'state_version'}}
    entry['mood_votes']=dict(pulse['visitor_mood_tally'])
    client.docs['pulse_history/2026-10-30']=entry
    (remote/'data/zeitgeist_state.json').write_text(json.dumps(pulse)); (remote/'data/pulse_feed.json').write_text(json.dumps({'entries':[entry]}))
    expected=(remote/'assets/daily_pulse_light.webp').read_bytes()
    git(remote,'add','.'); git(remote,'commit','-m','new canonical editorial art'); git(remote,'push','origin','HEAD:main')
    assert state.git_commit_and_push_with_retry('visitor projection',repo_root=root,store=store,sleep_fn=lambda _:None)
    git(remote,'pull','--ff-only')
    assert (remote/'assets/daily_pulse_light.webp').read_bytes()==expected
    assert 'LATEST' in (remote/'README.md').read_text()


def test_runtime_source_change_blocks_old_renderer_without_overwriting_remote(queued):
    root,remote,origin,store,client,_=queued
    (remote/'src').mkdir(); (remote/'src/new_runtime.py').write_text('VERSION = 2\n')
    git(remote,'add','.'); git(remote,'commit','-m','new runtime requires fresh worker'); git(remote,'push','origin','HEAD:main')
    before=git(remote,'rev-parse','HEAD')
    assert not state.git_commit_and_push_with_retry('old runtime must not render',repo_root=root,store=store,sleep_fn=lambda _:None)
    assert git(remote,'ls-remote','origin','refs/heads/main').split()[0]==before


def test_default_stage_rejects_accidentally_pre_staged_private_file(queued):
    root,remote,origin,store,client,_=queued
    (root/'data/private.json').write_text('private operational data')
    git(root,'add','data/private.json'); (root/'README.md').write_text('new public README')
    before=git(remote,'rev-parse','HEAD')
    assert not state.git_commit_and_push_with_retry('must not leak index',repo_root=root,store=state.InMemoryStateStore(),sleep_fn=lambda _:None)
    assert git(remote,'ls-remote','origin','refs/heads/main').split()[0]==before


def test_intentional_local_owner_config_edit_survives_fresh_remote_base(queued):
    root,remote,origin,store,client,_=queued
    path=root/'config/orbit_sources.json'; config=json.loads(path.read_text()); config['fixture_owner_note']='approved local owner update'
    path.write_text(json.dumps(config,indent=2)+'\n')
    assert state.git_commit_and_push_with_retry('owner edit',repo_root=root,store=store,sleep_fn=lambda _:None)
    git(remote,'pull','--ff-only')
    assert json.loads((remote/'config/orbit_sources.json').read_text())['fixture_owner_note']=='approved local owner update'
    assert (remote/'remote-source.txt').read_text()=='retain newest remote source'
