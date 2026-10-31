"""Offline acceptance for publication promotion; fake canonical snapshots only."""
import copy
import json
from pathlib import Path
import pytest
from scripts import promote_release_state as p
from src.state_store import PublicationConflict, PublicationOutcomeUnknown


def snapshot(prefix='', documents=None):
    return {'schema_version':1, 'target':dict(p.TARGET, collection_prefix=prefix),
            'documents':{k:{'data':copy.deepcopy(v),'update_time':'2026-10-30T04:00:00Z'}
                         for k,v in (documents or {}).items()}}


@pytest.fixture
def inputs(tmp_path):
    from src.zeitgeist.publication import write_publication_manifest
    root=tmp_path/'bundle'; (root/'assets/pulse').mkdir(parents=True); (root/'data').mkdir()
    date='2026-10-30'
    for name in ('daily_pulse_light.webp','daily_pulse_dark.webp','daily_pulse.svg','pulse/'+date+'.webp'):
        (root/'assets'/name).write_bytes(b'accepted asset '+name.encode())
    entry={'date':date,'word':'REAL','gloss':'Actual publication','visual_medium':'PHOTOGRAPHY',
           'curatorial_archetype':'POETIC_CINEMA','credit_url':'https://unsplash.com/photos/accepted',
           'mood_votes':{'spot_on':8,'too_sunny':1,'too_heavy':0}}
    manifest=write_publication_manifest(root,date=date,word='REAL',gloss=entry['gloss'],archetype=entry['curatorial_archetype'],medium=entry['visual_medium'],archive_path='assets/pulse/'+date+'.webp')
    entry['publication_id']=manifest['publication_id']
    pulse=dict(entry,visitor_mood_tally=entry['mood_votes'],state_version=5)
    (root/'data/zeitgeist_state.json').write_text(json.dumps(pulse))
    (root/'data/pulse_feed.json').write_text(json.dumps({'entries':[entry]}))
    source=snapshot('staging_persistent_',{'profile_state/daily_pulse':pulse,
         'pulse_history/'+date:entry,'pulse_feed/v1':{'entries':[entry]},'profile_state/orbit_roster':{'handles':{}}})
    target=snapshot(documents={'profile_state/daily_pulse':{'word':'TEST'},
          'profile_state/active_game':{'round_id':6,'turn':3,'board':[['C']*7]*6},
          'profile_state/elo_ledger':{'players':{'showcase':{'wins':99}}},
          'profile_state/bot_memory':{'states':{'fake':1}},'profile_state/game_rate_limits':{'users':{}},
          'processed_events/old-test':{'accepted':True},'vote_rate_limits/old-test':{'count':1}})
    return root,source,target,manifest


def prepare(inputs, **kw):
    root,source,target,manifest=inputs
    return p.prepare_plan(source,target,bundle_root=root,source_sha='a'*40,source_run='41000000000',
        source_snapshot_sha=p.digest(source),target_snapshot_sha=p.digest(target),
        manifest_sha=p.digest(manifest),game_policy=kw.get('policy','fresh'),operation_id='b'*32,source_backup_sha='c'*64,target_backup_sha='d'*64,journal_path=inputs[0].parent/'journal.json')


def test_fresh_keeps_real_publication_and_assets_zeros_votes_and_game(inputs):
    plan=prepare(inputs); after=plan['plan']['after']; pulse=after['profile_state/daily_pulse']
    assert pulse['publication_id']==inputs[3]['publication_id']
    assert pulse['credit_url']=='https://unsplash.com/photos/accepted'
    assert pulse['mood_votes']=={'spot_on':0,'too_sunny':0,'too_heavy':0}
    assert after['profile_state/active_game']['turn']==1
    assert after['profile_state/active_game']['moves_this_round']==0
    assert after['profile_state/elo_ledger']['players']=={}
    assert after['profile_state/bot_memory']['states']=={}
    assert 'processed_events/old-test' not in after
    assert 'vote_rate_limits/old-test' not in after
    assert len(plan['plan']['assets'])==4


def test_preserve_game_retains_exact_game_learning_and_receipts(inputs):
    after=prepare(inputs,policy='preserve-target-game')['plan']['after']
    for path in ('profile_state/active_game','profile_state/elo_ledger','profile_state/bot_memory','processed_events/old-test'):
        assert after[path]==inputs[2]['documents'][path]['data']


def test_real_history_dto_uses_layout_bound_by_daily_pulse_and_manifest_without_rewriting_source(inputs):
    root,source,target,manifest=inputs
    entry=source['documents']['pulse_history/2026-10-30']['data']
    entry.pop('curatorial_archetype')
    source['documents']['pulse_feed/v1']['data']['entries']=[copy.deepcopy(entry)]
    pulse=source['documents']['profile_state/daily_pulse']['data']
    pulse['published_pulse']=copy.deepcopy(entry)
    (root/'data/zeitgeist_state.json').write_text(json.dumps(pulse))
    (root/'data/pulse_feed.json').write_text(json.dumps({'entries':[entry]}))
    baseline=copy.deepcopy(source)
    accepted_bytes=(root/'data/pulse_feed.json').read_bytes()
    after=prepare(inputs)['plan']['after']
    assert source==baseline and (root/'data/pulse_feed.json').read_bytes()==accepted_bytes
    assert 'curatorial_archetype' not in after['pulse_history/2026-10-30']
    assert after['profile_state/daily_pulse']['curatorial_archetype']==manifest['archetype']


def test_explicit_history_layout_conflict_is_rejected_even_with_matching_daily_layout(inputs):
    root,source,target,manifest=inputs
    entry=source['documents']['pulse_history/2026-10-30']['data']
    entry['curatorial_archetype']='WRONG_LAYOUT'
    source['documents']['pulse_feed/v1']['data']['entries']=[copy.deepcopy(entry)]
    (root/'data/pulse_feed.json').write_text(json.dumps({'entries':[entry]}))
    with pytest.raises(ValueError,match='archetype'):
        prepare(inputs)


@pytest.mark.parametrize('change', ['source','target','manifest','missing_manifest','asset','extra_collection','revision'])
def test_prepare_rejects_substitution(inputs,change):
    if change=='source': inputs[1]['target']['collection_prefix']=''
    elif change=='target': inputs[2]['target']['project']='other'
    elif change=='manifest': inputs[3]['word']='forged'
    elif change=='missing_manifest': (inputs[0]/'assets/pulse_manifest.json').unlink()
    elif change=='asset': (inputs[0]/'assets/daily_pulse.svg').write_text('tampered')
    elif change=='extra_collection': inputs[2]['documents']['spend_control/2026-10']={'data':{},'update_time':'x'}
    elif change=='revision': inputs[2]['documents']['profile_state/daily_pulse']['update_time']=None
    with pytest.raises((ValueError,FileNotFoundError)): prepare(inputs)


class Fake:
    def __init__(self,snapshot): self.state=copy.deepcopy(snapshot); self.calls=0; self.mode=None
    def snapshot(self): return copy.deepcopy(self.state)
    def commit_exact(self,writes):
        self.calls+=1
        for path,before,after in writes:
            assert self.state['documents'].get(path)==before
        if self.mode=='conflict': raise PublicationConflict('changed')
        if self.mode=='unknown_before': raise PublicationOutcomeUnknown('secret')
        for path,before,after in writes:
            if after is None: self.state['documents'].pop(path,None)
            else: self.state['documents'][path]={'data':copy.deepcopy(after),'update_time':'2026-10-30T05:00:00Z'}
        if self.mode=='unknown_after': raise PublicationOutcomeUnknown('secret')


def test_apply_exact_pin_and_reconcile_then_rollback(inputs,tmp_path):
    plan=prepare(inputs); fake=Fake(inputs[2]); journal=tmp_path/'journal.json'
    with pytest.raises(PermissionError): p.execute(plan,fake,journal,expected_sha='0'*64,confirmation=p.CONFIRM)
    assert fake.calls==0
    result=p.execute(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert result['status']=='complete'
    assert fake.state['documents']['profile_state/elo_ledger']['data']['players']=={}
    assert p.rollback(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM,dry_run=True)['status']=='rollback_ready'
    p.rollback(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert {k:v['data'] for k,v in fake.state['documents'].items()}=={k:v['data'] for k,v in inputs[2]['documents'].items()}


@pytest.mark.parametrize('mode',['unknown_before','unknown_after','conflict'])
def test_unknown_and_rejected_outcomes_do_not_blindly_redispatch(inputs,tmp_path,mode):
    plan=prepare(inputs); fake=Fake(inputs[2]); fake.mode=mode; journal=tmp_path/'journal.json'
    with pytest.raises((PublicationConflict,PublicationOutcomeUnknown)):
        p.execute(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert json.loads(journal.read_text())['status']=='partial_or_unknown'
    with pytest.raises(ValueError): p.execute(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert fake.calls==1
    if mode=='unknown_after':
        assert p.reconcile(plan,fake,journal)['status']=='complete'
    else:
        with pytest.raises(ValueError): p.reconcile(plan,fake,journal)


@pytest.mark.parametrize('change',['value','same_value_new_revision','new_vote_record'])
def test_rollback_refuses_any_new_visitor_state(inputs,tmp_path,change):
    plan=prepare(inputs); fake=Fake(inputs[2]); journal=tmp_path/'journal.json'
    p.execute(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    if change=='value': fake.state['documents']['profile_state/active_game']['data']['turn']=2
    elif change=='same_value_new_revision': fake.state['documents']['profile_state/active_game']['update_time']='new'
    else: fake.state['documents']['pulse_votes/new']={'data':{'spot_on':1},'update_time':'new'}
    with pytest.raises(ValueError): p.rollback(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert fake.calls==1


def test_apply_refuses_target_identity_substitution_and_alternate_journal(inputs,tmp_path):
    plan=prepare(inputs); fake=Fake(inputs[2]); fake.state['target']['project']='wrong-project'
    with pytest.raises(ValueError): p.execute(plan,fake,tmp_path/'journal.json',expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert fake.calls==0
    fake=Fake(inputs[2])
    with pytest.raises(ValueError): p.execute(plan,fake,tmp_path/'other.json',expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert fake.calls==0


def test_backup_binding_reuses_existing_verified_backup_checks(inputs,tmp_path):
    import hashlib
    root,source,target,_=inputs
    backup=tmp_path/'backup'; backup.mkdir()
    collections={col:{} for col in p.COLLECTIONS}
    for path,rec in target['documents'].items():
        col,doc=path.split('/'); collections[col][doc]=rec['data']
    raw=json.dumps({'collection_prefix':'','collections':collections}).encode()
    (backup/'firestore_export.json').write_bytes(raw)
    manifest={'files':{},'target':p.TARGET,'firestore_export':'firestore_export.json','firestore_export_sha256':hashlib.sha256(raw).hexdigest()}
    (backup/'manifest.json').write_text(json.dumps(manifest))
    assert p._backup_binding(backup,target)==p._sha256_file(backup/'manifest.json')
    (backup/'firestore_export.json').write_text('{}')
    with pytest.raises(ValueError): p._backup_binding(backup,target)


def test_exact_rest_wire_replaces_and_deletes_in_single_conditional_commit(monkeypatch):
    adapter=p.CanonicalAdapter(); adapter.client._resolve_token=lambda:'synthetic'
    captured=[]
    class Response:
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def read(self): return json.dumps({'commitTime':'2026-10-30T05:00:00Z','writeResults':[{'updateTime':'2026-10-30T05:00:00Z'},{}, {'updateTime':'2026-10-30T05:00:00Z'}]}).encode()
    def dispatch(request,**kw): captured.append(json.loads(request.data)); return Response()
    monkeypatch.setattr(p.urllib.request,'urlopen',dispatch)
    old={'data':{'old_field':'remove'},'update_time':'2026-10-30T04:00:00Z'}
    adapter.commit_exact([('profile_state/daily_pulse',old,{'new_field':'keep'}),('processed_events/old',old,None),('pulse_votes/2026-10-30',None,p.ZERO)])
    assert len(captured)==1
    writes=captured[0]['writes']
    assert all('updateMask' not in write for write in writes)
    assert writes[0]['currentDocument']=={'updateTime':old['update_time']}
    assert list(writes[0]['update']['fields'])==['new_field']
    assert writes[1]['delete'].endswith('/processed_events/old')
    assert writes[2]['currentDocument']=={'exists':False}


def test_actual_rest_unknown_ack_is_not_reported_as_success(monkeypatch):
    adapter=p.CanonicalAdapter(); adapter.client._resolve_token=lambda:'synthetic'
    class Response:
        def __enter__(self): return self
        def __exit__(self,*_): pass
        def read(self): return b'{"commitTime":"yes","writeResults":[]}'
    monkeypatch.setattr(p.urllib.request,'urlopen',lambda *a,**k:Response())
    with pytest.raises(PublicationOutcomeUnknown): adapter.commit_exact([('pulse_votes/2026-10-30',None,p.ZERO)])


def test_preserved_noop_update_times_do_not_prevent_confirmed_readback(inputs,tmp_path):
    plan=prepare(inputs,policy='preserve-target-game'); fake=Fake(inputs[2]); journal=tmp_path/'journal.json'
    original=fake.commit_exact
    def noop_preserving_commit(writes):
        original(writes)
        for path,before,after in writes:
            if before and before['data']==after: fake.state['documents'][path]=copy.deepcopy(before)
    fake.commit_exact=noop_preserving_commit
    assert p.execute(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)['status']=='complete'


def test_partial_or_divergent_backend_state_never_becomes_complete(inputs,tmp_path):
    plan=prepare(inputs); fake=Fake(inputs[2]); journal=tmp_path/'journal.json'
    def partial(writes):
        path,before,after=next(w for w in writes if w[2] is not None)
        fake.state['documents'][path]={'data':after,'update_time':'2026-10-30T05:00:00Z'}
        raise PublicationOutcomeUnknown('secret-bearing provider body')
    fake.commit_exact=partial
    with pytest.raises(PublicationOutcomeUnknown): p.execute(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    with pytest.raises(ValueError): p.reconcile(plan,fake,journal)
    with pytest.raises(ValueError): p.rollback(plan,fake,journal,expected_sha=plan['plan_sha256'],confirmation=p.CONFIRM)
    assert json.loads(journal.read_text())['status']=='partial_or_unknown'


def test_prepare_cli_is_offline_private_and_backup_bound(inputs,tmp_path,monkeypatch,capsys):
    import hashlib
    root,source,target,manifest=inputs
    for role,snapshot_value in (('source',source),('target',target)):
        (tmp_path/(role+'.json')).write_text(json.dumps(snapshot_value))
        backup=tmp_path/(role+'-backup'); backup.mkdir()
        collections={col:{} for col in p.COLLECTIONS}
        for path,rec in snapshot_value['documents'].items():
            col,doc=path.split('/'); collections[col][doc]=rec['data']
        raw=json.dumps({'collection_prefix':snapshot_value['target']['collection_prefix'],'collections':collections}).encode()
        (backup/'firestore_export.json').write_bytes(raw)
        (backup/'manifest.json').write_text(json.dumps({'files':{},'target':snapshot_value['target'],
            'firestore_export':'firestore_export.json','firestore_export_sha256':hashlib.sha256(raw).hexdigest()}))
    def forbidden(*args,**kw): raise AssertionError('Offline preparation must never instantiate cloud adapter')
    monkeypatch.setattr(p,'CanonicalAdapter',forbidden)
    monkeypatch.setattr(p,'_durable_location',lambda path:Path(path).resolve())
    args=['prepare','--source-snapshot',str(tmp_path/'source.json'),'--target-snapshot',str(tmp_path/'target.json'),
       '--source-backup',str(tmp_path/'source-backup'),'--target-backup',str(tmp_path/'target-backup'),
       '--bundle-root',str(root),'--output',str(tmp_path/'plan.json'), '--source-snapshot-sha',p.digest(source),
       '--target-snapshot-sha',p.digest(target),'--manifest-sha',p.digest(manifest),'--source-sha','a'*40,
       '--source-run','41000000000','--operation-id','b'*32,'--game-policy','fresh','--journal',str(tmp_path/'journal.json')]
    assert p.main(args)==0
    assert (tmp_path/'plan.json').stat().st_mode & 0o777==0o600
    output=json.loads(capsys.readouterr().out)
    assert set(output)=={'status','plan_sha256'}
    assert output['status']=='prepared_no_writes'


def test_invalid_capture_prefix_does_not_create_client(monkeypatch):
    def forbidden(*a,**k): raise AssertionError('Must reject prefix before cloud setup')
    monkeypatch.setattr(p,'_FirestoreRestClient',forbidden)
    with pytest.raises(ValueError): p.CanonicalAdapter('spend_control')


def test_secret_bearing_403_reports_only_typed_cli_failure(monkeypatch,tmp_path,capsys):
    import urllib.error
    import io
    class Broken:
        def __init__(self,*a): pass
        def snapshot(self): raise urllib.error.HTTPError('https://firestore.googleapis.com/?token=SECRET',403,'SECRET',{},io.BytesIO(b'SECRET'))
    monkeypatch.setattr(p,'CanonicalAdapter',Broken)
    assert p.main(['capture','--collection-prefix','','--output',str(tmp_path/'unused')])==1
    assert json.loads(capsys.readouterr().out)=={'status':'blocked','error_type':'HTTPError'}


@pytest.mark.parametrize('relative',['data/zeitgeist_state.json','data/pulse_feed.json'])
@pytest.mark.parametrize('changed_field',['credit_url','operator_note'])
def test_approved_public_json_bytes_cannot_change_after_prepare(inputs,relative,changed_field):
    plan=prepare(inputs)
    path=inputs[0]/relative
    payload=json.loads(path.read_text())
    item=payload['entries'][0] if relative.endswith('pulse_feed.json') else payload
    item[changed_field]='substituted after approval'
    path.write_text(json.dumps(payload))
    # Artwork/publication ID remain unchanged, but the accepted package differs.
    with pytest.raises(ValueError): p._bundle_bound(plan,inputs[0])


def test_unchanged_full_approved_bundle_still_passes_with_fresh_policy(inputs):
    plan=prepare(inputs)
    p._bundle_bound(plan,inputs[0])
    assert plan['plan']['game_policy']=='fresh'
    assert plan['plan']['after']['profile_state/daily_pulse']['mood_votes']==p.ZERO


@pytest.mark.parametrize('command',['apply','reconcile'])
def test_cli_checks_accepted_json_hashes_before_cloud_setup(inputs,tmp_path,monkeypatch,capsys,command):
    envelope=prepare(inputs)
    plan_path=tmp_path/'plan.json'; plan_path.write_text(json.dumps(envelope))
    path=inputs[0]/'data/zeitgeist_state.json'
    payload=json.loads(path.read_text()); payload['credit_url']='changed after approval'
    path.write_text(json.dumps(payload))
    def forbidden(*args,**kw): raise AssertionError('Substituted bundle must block before any cloud setup')
    monkeypatch.setattr(p,'CanonicalAdapter',forbidden)
    args=[command,'--plan',str(plan_path),'--journal',str(tmp_path/'journal.json'),
      '--expected-plan-sha',envelope['plan_sha256'],'--confirm-target',p.CONFIRM,'--bundle-root',str(inputs[0])]
    for name in ('source-snapshot','target-snapshot','source-backup','target-backup'):
        args.extend(['--'+name,str(tmp_path/name)])
    assert p.main(args)==1
    assert json.loads(capsys.readouterr().out)=={'status':'blocked','error_type':'ValueError'}


def test_fresh_feed_retains_accepted_timestamp_without_copying_private_envelope_fields(inputs):
    feed=inputs[1]['documents']['pulse_feed/v1']['data']
    feed.update(updated_at='2026-10-30T14:15:16Z',private_operator_note='must stay private')
    after=prepare(inputs)['plan']['after']['pulse_feed/v1']
    assert after['updated_at']==feed['updated_at']
    assert 'private_operator_note' not in after
    assert len(after['entries'])==1 and after['entries'][0]['mood_votes']=={'spot_on':0,'too_sunny':0,'too_heavy':0}
