"""Owner-only readiness reads real factory state without spending or publishing."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from scripts import probe_profile_readiness as probe
from tests.test_firestore_store import _RecordingFirestoreClient
from tests.test_automatic_spend_runtime import setup, NOW

ROOT = Path(__file__).resolve().parents[1]


def environment(sha='a'*64):
    return dict(GITHUB_EVENT_NAME='workflow_dispatch', GITHUB_ACTOR='Siyuan-Zhu',
        GITHUB_REPOSITORY='Siyuan-Zhu/readme-live-sandbox',
        TARGET_REPO_SLUG='Siyuan-Zhu/readme-live-sandbox', STATE_STORE_BACKEND='firestore',
        GCP_PROJECT_ID='git-readme-features', FIRESTORE_DATABASE_ID='(default)',
        FIRESTORE_COLLECTION_PREFIX='staging_readiness_',
        GCP_WIF_PROVIDER='//iam.googleapis.com/projects/123/locations/global/workloadIdentityPools/p/providers/p',
        SPEND_POLICY_SHA256=sha, PULSE_VOTE_SALT='synthetic-private-salt',
        GEMINI_API_KEY='synthetic-gemini', BFL_API_KEY='synthetic-bfl',
        APIFY_TOKEN='synthetic-apify', UNSPLASH_ACCESS_KEY='synthetic-unsplash',
        APIFY_INSTAGRAM_BUILD='0.0.648', ZEITGEIST_LIVE_COLLECT='1',
        ZEITGEIST_LIVE_LLM='1', ZEITGEIST_LIVE_UNSPLASH='1')


def fixture(tmp_path, monkeypatch):
    for name in ('assets','data'):
        shutil.copytree(ROOT/name,tmp_path/name)
    client=_RecordingFirestoreClient()
    from tests.test_firestore_store import _RecordingDocRef
    def forbidden(*args,**kwargs):
        pytest.fail('readiness attempted canonical application write')
    monkeypatch.setattr(_RecordingDocRef,'set',forbidden)
    monkeypatch.setattr(_RecordingDocRef,'delete',forbidden)
    monkeypatch.setattr(client,'commit_documents',forbidden)
    for doc,filename in [('active_game','game_state'),('daily_pulse','zeitgeist_state'),
                         ('elo_ledger','hall_of_records'),('bot_memory','bot_brain')]:
        client.docs['staging_readiness_profile_state/'+doc]=json.loads(
            (tmp_path/'data'/f'{filename}.json').read_text())
    original=probe.get_state_store
    def factory(**kwargs):
        assert kwargs['backend']=='firestore'
        return original(**kwargs,firestore_client=client)
    monkeypatch.setattr(probe,'get_state_store',factory)
    wire,sha,observer,obs=setup(monkeypatch)
    monkeypatch.setattr(probe,'PrivateFirestoreStore',lambda:wire.store)
    def metadata(policy,env,now):
        result=observer(policy,now)
        for provider in ('apify','gemini','bfl'):
            result[provider]['status']='metadata_verified'
        return result
    monkeypatch.setattr(probe,'observe_provider_metadata',metadata)
    monkeypatch.setattr(probe,'inspect_unsplash',lambda client,key:{'status':'metadata_verified'})
    return client,wire,environment(sha)


def files(root):
    return {str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}


def test_actual_factory_and_controller_plan_leave_all_state_and_bytes_unchanged(tmp_path,monkeypatch):
    client,wire,env=fixture(tmp_path,monkeypatch)
    before=(copy.deepcopy(client.docs),copy.deepcopy(wire.docs),files(tmp_path))
    probe.probe(env,root=tmp_path,now=NOW)
    assert (client.docs,wire.docs,files(tmp_path))==before
    assert not wire.posts


@pytest.mark.parametrize('field,value',[
 ('GITHUB_ACTOR','someone-else'),('GITHUB_EVENT_NAME','schedule'),
 ('GITHUB_REPOSITORY','Siyuan-Zhu/other'),('SPEND_POLICY_SHA256',''),
 ('PULSE_VOTE_SALT',''),('BFL_API_KEY',''),('STATE_STORE_BACKEND','json'),
])
def test_bad_configuration_stops_before_any_factory_or_provider(field,value,monkeypatch):
    env=environment();env[field]=value
    monkeypatch.setattr(probe,'get_state_store',lambda **kw:pytest.fail('read before configuration check'))
    with pytest.raises(RuntimeError):probe.probe(env)


CANARY_REF='Siyuan-Zhu/Siyuan-Zhu/.github/workflows/provider_canary.yml@refs/heads/main'


class PastGuard(Exception):
    """Sentinel: the owner guard accepted the environment."""


def reaches_past_guard(env,monkeypatch):
    def sentinel(*args,**kwargs):raise PastGuard
    monkeypatch.setattr(probe,'configuration_errors',sentinel)
    try:probe.probe(env)
    except PastGuard:return True
    except RuntimeError as error:
        assert str(error)=='Owner manual dispatch required'
        return False
    raise AssertionError('probe neither refused nor reached the sentinel')


@pytest.mark.parametrize('repo,ref',[
 ('Siyuan-Zhu/Siyuan-Zhu',CANARY_REF),
 ('Siyuan-Zhu/readme-live-sandbox',
  'Siyuan-Zhu/readme-live-sandbox/.github/workflows/provider_canary.yml@refs/heads/main')])
def test_canary_schedule_event_is_accepted(monkeypatch,repo,ref):
    env=environment();env.update(GITHUB_EVENT_NAME='schedule',GITHUB_REPOSITORY=repo,
        GITHUB_WORKFLOW_REF=ref,GITHUB_ACTOR='github-actions[bot]')
    assert reaches_past_guard(env,monkeypatch)


@pytest.mark.parametrize('ref',[
 None,'',
 'Siyuan-Zhu/Siyuan-Zhu/.github/workflows/daily_zeitgeist.yml@refs/heads/main',
 'Siyuan-Zhu/Siyuan-Zhu/.github/workflows/provider_canary.yml.evil@refs/heads/main',
 'Siyuan-Zhu/other/.github/workflows/provider_canary.yml@refs/heads/main',
 'Siyuan-Zhu/readme-live-sandbox/.github/workflows/provider_canary.yml@refs/heads/main'])
def test_schedule_from_other_workflow_refused(monkeypatch,ref):
    env=environment();env.update(GITHUB_EVENT_NAME='schedule',GITHUB_REPOSITORY='Siyuan-Zhu/Siyuan-Zhu')
    if ref is not None:env['GITHUB_WORKFLOW_REF']=ref
    assert not reaches_past_guard(env,monkeypatch)


def test_canary_schedule_from_wrong_repository_refused(monkeypatch):
    env=environment();env.update(GITHUB_EVENT_NAME='schedule',GITHUB_REPOSITORY='Siyuan-Zhu/other',
        GITHUB_WORKFLOW_REF='Siyuan-Zhu/other/.github/workflows/provider_canary.yml@refs/heads/main')
    assert not reaches_past_guard(env,monkeypatch)


@pytest.mark.parametrize('event',['workflow_dispatch','push','pull_request'])
def test_non_owner_dispatch_still_refused(monkeypatch,event):
    env=environment();env.update(GITHUB_EVENT_NAME=event,GITHUB_ACTOR='someone-else',
        GITHUB_WORKFLOW_REF=CANARY_REF)
    assert not reaches_past_guard(env,monkeypatch)
    owner=environment()
    assert reaches_past_guard(owner,monkeypatch)


def test_canary_workflow_mirrors_preflight_job_without_writes():
    text=(ROOT/'.github/workflows/provider_canary.yml').read_text()
    daily=(ROOT/'.github/workflows/daily_zeitgeist.yml').read_text()
    template=daily.split('  profile-preflight:\n',1)[1].split('  studio-footer:',1)[0]
    steps=lambda block:block.split('    runs-on: ubuntu-latest\n',1)[1]
    assert steps(text.split('\njobs:\n',1)[1].split('\n',1)[1])==steps(template)
    assert 'cron: "0 20 * * *"' in text and 'timezone: "America/New_York"' in text
    assert 'workflow_dispatch:' in text and 'group: provider-canary' in text
    assert 'contents: read' in text and 'id-token: write' in text
    assert 'e664ccce9873b17fedaa7eda48e7c928be7540fd3db6c7d99168a125045a661f' in text
    for forbidden in ('inputs.','upload-artifact','--apply','run_zeitgeist.py','build_readme.py','contents: write'):
        assert forbidden not in text


@pytest.mark.parametrize('change',['missing','identity','word','local_asset','unsplash','policy'])
def test_missing_or_divergent_authority_and_provider_metadata_fail_closed(tmp_path,monkeypatch,change):
    client,wire,env=fixture(tmp_path,monkeypatch)
    key='staging_readiness_profile_state/daily_pulse'
    if change=='missing':del client.docs[key]
    elif change=='identity':client.docs[key]['publication_id']='0'*20
    elif change=='word':client.docs[key]['word']='Divergence'
    elif change=='local_asset':(tmp_path/'assets/daily_pulse.svg').write_text('corrupted')
    elif change=='unsplash':monkeypatch.setattr(probe,'inspect_unsplash',lambda *a:{'status':'unavailable'})
    elif change=='policy':del wire.docs['policy_'+env['SPEND_POLICY_SHA256']]
    before=(copy.deepcopy(client.docs),copy.deepcopy(wire.docs),files(tmp_path))
    with pytest.raises(Exception):probe.probe(env,root=tmp_path,now=NOW)
    assert (client.docs,wire.docs,files(tmp_path))==before
    assert not wire.posts


def test_read_only_store_refuses_commit_even_if_future_controller_attempts_it():
    class Store:
        def commit(self,writes):pytest.fail('must not reach mutating store')
    with pytest.raises(RuntimeError):probe.ReadOnlySpendStore(Store()).commit([])


@pytest.mark.parametrize('failure',[False,True])
def test_public_output_is_fixed_even_when_dependency_prints_private_data(monkeypatch,capsys,failure):
    def run(env):
        print('synthetic-private-account 999 secret balance')
        import sys
        print('synthetic-private-stderr',file=sys.stderr)
        if failure:raise RuntimeError('synthetic-private-error')
    monkeypatch.setattr(probe,'probe',run)
    assert probe.main()==(2 if failure else 0)
    out=capsys.readouterr()
    assert out.err=='' and 'synthetic' not in out.out
    assert json.loads(out.out)==dict(status='blocked' if failure else 'ready',
        mode='read_only',paid_calls=0,ledger_writes=0)


@pytest.mark.parametrize('event,preflight,writer,schedule,actor,repo,expected',[
 ('workflow_dispatch',True,True,True,'Siyuan-Zhu','Siyuan-Zhu/Siyuan-Zhu',(True,False)),
 ('workflow_dispatch',True,False,False,'Siyuan-Zhu','Siyuan-Zhu/readme-live-sandbox',(True,False)),
 ('workflow_dispatch',True,True,True,'someone-else','Siyuan-Zhu/Siyuan-Zhu',(False,False)),
 ('workflow_dispatch',True,True,True,'Siyuan-Zhu','Siyuan-Zhu/other',(False,False)),
 ('workflow_dispatch',False,True,False,'Siyuan-Zhu','Siyuan-Zhu/Siyuan-Zhu',(False,True)),
 ('workflow_dispatch',False,False,True,'Siyuan-Zhu','Siyuan-Zhu/Siyuan-Zhu',(False,False)),
 ('schedule',False,True,True,'Siyuan-Zhu','Siyuan-Zhu/Siyuan-Zhu',(False,True)),
 ('schedule',False,True,False,'Siyuan-Zhu','Siyuan-Zhu/Siyuan-Zhu',(False,False)),
])
def test_workflow_job_truth_table(event,preflight,writer,schedule,actor,repo,expected):
    text=(ROOT/'.github/workflows/daily_zeitgeist.yml').read_text()
    def condition(job):
        block=text.split('  '+job+':\n',1)[1].split('    runs-on:',1)[0]
        expression=block.split('    if: >-\n',1)[1].split('    permissions:',1)[0]
        expression=' '.join(expression.split()).replace('&&',' and ').replace('||',' or ')
        expression=expression.replace('true','True')
        # Quoted workflow variables remain strings after Boolean normalization.
        return eval(expression,{'__builtins__':{}},{
            'github':SimpleNamespace(event_name=event,actor=actor,repository=repo),
            'inputs':SimpleNamespace(preflight_only=preflight,studio_footer_action="none"),
            'vars':SimpleNamespace(ZEITGEIST_WRITER_ENABLED=str(writer),ZEITGEIST_SCHEDULE_ENABLED=str(schedule))})
    assert (condition('profile-preflight'),condition('synthesize-pulse'))==expected
    preflight_job=text.split('  profile-preflight:\n',1)[1].split('  studio-footer:',1)[0]
    assert 'contents: read' in preflight_job and 'id-token: write' in preflight_job
    assert 'probe_profile_readiness.py' in preflight_job
    assert 'upload-artifact' not in preflight_job and '--apply' not in preflight_job
    assert 'run_zeitgeist.py' not in preflight_job and 'build_readme.py' not in preflight_job
    assert 'cron: "0 2 * * *"' in text and 'timezone: "America/New_York"' in text
    assert 'default: false' in text
