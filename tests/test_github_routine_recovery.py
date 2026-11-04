"""Existing GitHub refresh/CLI/claim boundaries, without providers or cloud writes."""
import copy
from datetime import datetime, timedelta, timezone
from contextlib import redirect_stdout
from io import StringIO
import json
import subprocess

import httpx
import pytest

from scripts import refresh_daily_digest as daily, maintain_spend_control as maintenance
from src.zeitgeist import edition_execution, public_tariff_http as reads
from src.zeitgeist.collectors import IngestionUnavailable
from src.zeitgeist.daily_outcomes import new_outcome, record_generation_failure, save_outcome
from tests.test_daily_digest_freshness import bundle
from tests.test_edition_execution import Memory

NOW=datetime(2026,11,5,7,tzinfo=timezone.utc)


def test_source_failure_receipt_survives_completion_without_private_data(tmp_path):
    from src.zeitgeist.daily_outcomes import complete_outcome
    path=tmp_path/'outcome.json'
    env={'DAILY_OUTCOME_PATH':str(path),'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1','GITHUB_SHA':'a'*40}
    outcome=new_outcome(env,NOW,NOW.date());outcome['stage']='generation'
    save_outcome(path,outcome)
    error=IngestionUnavailable('PRIVATE_TOKEN_AND_FEED',public_diagnostics={'code':'required-source-coverage-unavailable'},collection_health={
        'instagram':'empty','instagram_item_count':0,'publishable':False,'private':'PRIVATE_TOKEN_AND_FEED'})
    record_generation_failure(tmp_path,error,env)
    result=complete_outcome(tmp_path,env|{'DAILY_DIGEST_OUTCOME':'failure','DAILY_PUBLISH_OUTCOME':'skipped'},now=NOW)
    assert result['failure_code']=='required-source-coverage-unavailable'
    assert result['collection_health']=={'instagram':'empty','instagram_item_count':0,'publishable':False}
    assert result['failure_category']=='source_collection' and result['canonical_commit']=='not_committed'
    assert 'PRIVATE' not in path.read_text()


@pytest.mark.parametrize('kind,code',[('missing','embedding-provider-unavailable'),
    ('timeout','embedding-request-outcome-unavailable'),('invalid','embedding-response-invalid')])
def test_real_embedding_failure_is_not_reported_as_missing_sources(tmp_path,kind,code):
    from types import SimpleNamespace
    from src.zeitgeist.collectors import embed_missing_items
    def embed(**kwargs):
        if kind=='timeout':raise TimeoutError('PRIVATE_PAID_RESPONSE')
        return SimpleNamespace(embeddings=[])
    client=None if kind=='missing' else SimpleNamespace(models=SimpleNamespace(embed_content=embed))
    with pytest.raises(IngestionUnavailable) as raised:
        embed_missing_items([{'url_hash':'x','title':'Real boundary'}],genai_client=client,require_live=True)
    assert raised.value.public_diagnostics=={'code':code}
    path=tmp_path/'outcome.json';env={'DAILY_OUTCOME_PATH':str(path)}
    outcome=new_outcome(env,NOW,NOW.date());outcome['stage']='generation';save_outcome(path,outcome)
    record_generation_failure(tmp_path,raised.value,env)
    result=json.loads(path.read_text())
    assert result['failure_code']==code and result['failure_category']=='embedding_provider'
    assert 'PRIVATE' not in path.read_text()


@pytest.mark.parametrize('kind',['429','503','timeout'])
def test_real_maintenance_cli_defers_only_zero_write_anonymous_failure(monkeypatch,capsys,kind):
    class Frozen(datetime):
        @classmethod
        def now(cls,tz=None):return NOW
    monkeypatch.setattr(maintenance,'datetime',Frozen)
    monkeypatch.setattr(reads,'utc_now',lambda:NOW)
    monkeypatch.setattr(reads.time,'sleep',lambda delay:None)
    calls=[]
    class Store:
        def commit(self,values):pytest.fail('No accounting write permitted')
    monkeypatch.setattr(maintenance,'PrivateFirestoreStore',Store)
    def handler(request):
        calls.append(request)
        if kind=='timeout':raise httpx.ReadTimeout('PRIVATE_PROVIDER_TEXT',request=request)
        return httpx.Response(int(kind),headers={'retry-after':'120'})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        monkeypatch.setattr(maintenance,'observe_provider_metadata',lambda *a:reads.public_get(client,'https://bfl.ai/pricing'))
        monkeypatch.setattr(maintenance,'maintain_spend_control',lambda store,**kwargs:kwargs['observer']({},NOW))
        assert maintenance.main(['--apply'])==75
    result=json.loads(capsys.readouterr().out)
    assert result['retryable'] is True and 1<=len(calls)<=3
    assert 'PRIVATE' not in json.dumps(result)


@pytest.mark.parametrize('written',[False,True])
def test_refresh_allows_later_catchup_only_for_proven_preflight_and_preserves_attempt_bound(bundle,monkeypatch,written):
    root,pulse,store,published=bundle
    clock=[datetime.combine(published+timedelta(days=1),datetime.min.time(),tzinfo=timezone.utc).replace(hour=7)]
    class Frozen(datetime):
        @classmethod
        def now(cls,tz=None):return clock[0]
    monkeypatch.setattr(daily,'datetime',Frozen);monkeypatch.setattr(edition_execution,'datetime',Frozen)
    memory=Memory();monkeypatch.setattr(edition_execution,'_commit_documents',memory.commit)
    env={'SPEND_POLICY_SHA256':'pinned','GITHUB_RUN_ID':'first'}
    monkeypatch.setattr(edition_execution,'for_store',lambda *args:edition_execution.EditionExecution(
        memory,edition=str(clock[0].date()),owner=env['GITHUB_RUN_ID']))
    commands=[]
    def runner(command,**kwargs):
        commands.append(command[1])
        if len(commands)==1:
            report={'status':'blocked','operation':'maintenance','month':clock[0].strftime('%Y-%m'),
                'code':'transient-anonymous-price-read','retryable':True,
                'retry_not_before':(clock[0]+timedelta(minutes=5)).isoformat()}
            raise subprocess.CalledProcessError(2 if written else 75,command,output=json.dumps(report))
    with pytest.raises(subprocess.CalledProcessError):daily.refresh(root,store=store,env=env,now=clock[0],runner=runner)
    assert memory.value['stage']==('blocked' if written else 'preflight_retryable')
    assert commands==['scripts/maintain_spend_control.py']
    clock[0]+=timedelta(minutes=37);env['GITHUB_RUN_ID']='catchup'
    if written:
        with pytest.raises(RuntimeError,match='Deterministic'):daily.refresh(root,store=store,env=env,now=clock[0],runner=runner)
        assert len(commands)==1
    else:
        daily.refresh(root,store=store,env=env,now=clock[0],runner=runner)
        assert commands.count('scripts/run_zeitgeist.py')==1 and memory.value['attempt']==2
    assert store.load_pulse_state()==pulse


@pytest.mark.parametrize('stage',['blocked','held_for_review','requires_reconciliation'])
def test_failed_worker_cannot_revive_generation(monkeypatch,stage):
    from tests.test_edition_execution import claim,NOW as CLAIM_NOW
    memory,worker,_=claim(monkeypatch)
    worker.transition(stage,now=CLAIM_NOW)
    before=copy.deepcopy(memory.value)
    with pytest.raises(RuntimeError,match='terminal'):worker.transition('generation',now=CLAIM_NOW)
    assert memory.value==before


@pytest.mark.parametrize('mutation',['wrong_month','extra','long_hint','duplicate','wrong_exit','untyped'])
def test_unknown_maintenance_output_never_authorizes_retry(mutation):
    value={'status':'blocked','operation':'maintenance','month':'2026-11','code':'transient-anonymous-price-read',
        'retryable':True,'retry_not_before':(NOW+timedelta(minutes=5)).isoformat()}
    if mutation=='wrong_month':value['month']='2026-12'
    if mutation=='extra':value['private']='PRIVATE'
    if mutation=='long_hint':value['retry_not_before']=(NOW+timedelta(days=2)).isoformat()
    raw=json.dumps(value)
    if mutation=='duplicate':raw=raw.replace('"status": "blocked"','"status": "blocked", "status": "blocked"')
    error=RuntimeError('PRIVATE') if mutation=='untyped' else subprocess.CalledProcessError(2 if mutation=='wrong_exit' else 75,[],output=raw)
    assert daily.retryable_maintenance_failure(error,now=NOW) is None
