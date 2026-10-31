"""CLI privacy and read-only planning; no live providers or cloud state."""
import json
from pathlib import Path
import stat

import pytest

from scripts import maintain_spend_control as cli
from src.state_store import PublicationConflict
from src.zeitgeist.spend_guard import SpendBlocked


def failure(monkeypatch,error):
    def fail(*args,**kwargs):raise error
    monkeypatch.setattr(cli,'maintain_spend_control',fail)


def test_known_spend_code_only_reaches_public_log_not_dynamic_diagnostics(monkeypatch,capsys):
    error=SpendBlocked('Primary price snapshot changed, stale or missing')
    error.public_diagnostics.update(status=403,provider='secret-account',error_type='secret-token')
    failure(monkeypatch,error)
    assert cli.main([])==2
    report=json.loads(capsys.readouterr().out)
    assert report['code']=='automatic-primary-price-snapshot-changed-stale-or-missing'
    assert set(report)=={'status','operation','month','code'}


def test_unknown_spend_reason_and_tampered_code_never_become_public(monkeypatch,capsys):
    error=SpendBlocked('private key=secret-token balance=9.73')
    error.public_diagnostics['code']='secret-token'
    failure(monkeypatch,error)
    assert cli.main([])==2
    raw=capsys.readouterr().out
    assert json.loads(raw)['code']=='budget-admission-rejected'
    assert 'secret' not in raw and '9.73' not in raw


@pytest.mark.parametrize('error,expected',[(PublicationConflict('raw provider body'),'PublicationConflict'),
    (RuntimeError('secret-token billing response'),'RuntimeError'),
    (type('SecretTokenError',(Exception,),{})('raw private body'),'UnexpectedError')])
def test_generic_exceptions_only_emit_fixed_allowlisted_type(monkeypatch,capsys,error,expected):
    failure(monkeypatch,error)
    assert cli.main([])==2
    report=json.loads(capsys.readouterr().out)
    assert report['code']=='maintenance-unavailable' and report['error_type']==expected
    assert 'raw' not in json.dumps(report) and 'secret' not in json.dumps(report).lower()


def test_cli_real_controller_plan_is_read_only_and_private_report_is_secure(tmp_path,monkeypatch,capsys):
    from tests.test_automatic_spend_runtime import setup
    from tests.test_automatic_spend import NOW
    wire,sha,observe,observation=setup(monkeypatch)
    monkeypatch.setenv('SPEND_POLICY_SHA256',sha)
    monkeypatch.setattr(cli,'PrivateFirestoreStore',lambda:wire.store)
    monkeypatch.setattr(cli,'observe_provider_metadata',lambda *args:observation)
    actual_controller=cli.maintain_spend_control
    monkeypatch.setattr(cli,'maintain_spend_control',lambda store,**kwargs:actual_controller(store,now=NOW,**kwargs))
    output=tmp_path/'private'/'evidence.json'
    assert cli.main(['--private-report',str(output)])==0
    assert json.loads(capsys.readouterr().out)['status']=='planned' and wire.posts==[]
    report=json.loads(output.read_text())
    assert report['mode']=='plan_only' and report['stage']=='planning_or_commit'
    assert report['status']=='planned' and 'combined' in report['observations']
    assert 'policy' not in report and 'reservations' not in output.read_text()
    assert stat.S_IMODE(output.stat().st_mode)==0o600
    assert stat.S_IMODE(output.parent.stat().st_mode)==0o700


def test_partial_observer_evidence_survives_later_failure_without_raw_data(tmp_path,monkeypatch,capsys):
    from scripts import provider_preflight
    from src.zeitgeist import bfl_tariff,gemini_tariff
    secret='provider-secret-token'
    monkeypatch.setenv('BFL_API_KEY',secret)
    raw={'observed_at':'2026-10-30T20:00:00Z','bfl':{'status':'metadata_verified','balance_credits':'925',
        'headers':{'x-key':secret},'error':'raw HTTP '+secret,'billing':{'error':'private errorbody',
        'note':'echo '+secret}},'env':{'BFL_API_KEY':secret},'policy':{'private':'do not save'}}
    monkeypatch.setattr(provider_preflight,'run_preflight',lambda *a,**k:raw)
    monkeypatch.setattr(bfl_tariff,'observe_bfl_flux3_tariff',lambda *a,**k:{'source_url':'https://bfl.ai/pricing',
        'sha256':'a'*64,'contract':{'unit_price_usd':'.07'},'raw_body':'do not save'})
    def unavailable(*args,**kwargs):raise SpendBlocked('Public Gemini tariff observation unavailable; no price inferred')
    monkeypatch.setattr(gemini_tariff,'observe_gemini_tariff',unavailable)
    def observation(*args):
        print(secret)
        provider_preflight.run_preflight(None,{})
        bfl_tariff.observe_bfl_flux3_tariff(None,now=None)
        gemini_tariff.observe_gemini_tariff(None,now=None)
    monkeypatch.setattr(cli,'observe_provider_metadata',observation)
    def maintain(store,*,observer,**kwargs):
        assert kwargs['apply'] is False
        return observer({'never':'record raw policy'},None)
    monkeypatch.setattr(cli,'maintain_spend_control',maintain)
    output=tmp_path/'evidence.json'
    assert cli.main(['--private-report',str(output)])==2
    public=capsys.readouterr()
    assert secret not in public.out and not public.err
    assert json.loads(public.out)['code']=='automatic-public-gemini-tariff-observation-unavailable-no-price-inferred'
    private=output.read_text();report=json.loads(private)
    assert report['stage']=='observing'
    assert report['observations']['provider_preflight']['bfl']['balance_credits']=='925'
    assert report['observations']['price_snapshots']['bfl']['contract']=={'unit_price_usd':'.07'}
    assert secret not in private and 'raw HTTP' not in private and 'private errorbody' not in private
    assert 'do not save' not in private and 'never' not in private


def test_planning_or_commit_failure_has_distinct_stage_and_no_mutation_retry(tmp_path,monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr(cli,'observe_provider_metadata',lambda *args:{'observed_at':'fixed'})
    def rejected(store,*,observer,**kwargs):
        calls.append(kwargs)
        observer({},None)
        raise PublicationConflict('private firestore wire body')
    monkeypatch.setattr(cli,'maintain_spend_control',rejected)
    output=tmp_path/'evidence.json'
    assert cli.main(['--apply','--private-report',str(output)])==2
    assert len(calls)==1 and calls[0]['apply'] is True
    assert json.loads(output.read_text())['stage']=='planning_or_commit'
    assert json.loads(capsys.readouterr().out)['error_type']=='PublicationConflict'


@pytest.mark.parametrize('kind',['inside_checkout','existing','unwritable'])
def test_private_prewrite_failure_never_dispatches_controller(tmp_path,monkeypatch,capsys,kind):
    def no_reads(*args,**kwargs):pytest.fail('must not call controller or observer')
    monkeypatch.setattr(cli,'maintain_spend_control',no_reads)
    output=tmp_path/'evidence.json'
    if kind=='inside_checkout':monkeypatch.setattr(cli,'ROOT',tmp_path)
    if kind=='existing':output.write_text('preserve private evidence')
    if kind=='unwritable':
        def no_storage(*args,**kwargs):raise OSError('secret private path')
        monkeypatch.setattr(cli,'_private_write',no_storage)
    assert cli.main(['--private-report',str(output)])==2
    public=capsys.readouterr().out
    assert 'secret private path' not in public
    if kind=='existing':assert output.read_text()=='preserve private evidence'


def test_private_metadata_job_runs_plan_only_and_retains_private_evidence():
    path=Path(__file__).resolve().parents[1]/'.github/workflows/staging_zeitgeist_dryrun.yml'
    workflow=path.read_text()
    job=workflow.split('  provider-metadata:\n',1)[1].split('  accounting-maintenance:',1)[0]
    step=job.split('      - name: Plan installed automatic maintenance without ledger writes\n',1)[1].split('      - name:',1)[0]
    assert "if: always() && vars.SPEND_POLICY_SHA256 != ''" in step
    assert '--apply' not in step and '--private-report /tmp/private-spend-maintenance/evidence.json' in step
    for key in ('APIFY_TOKEN','BFL_API_KEY','GEMINI_API_KEY','GCP_PROJECT_ID','GCP_WIF_PROVIDER','SPEND_POLICY_SHA256'):
        assert key+':' in step
    upload=job.split('      - name: Retain private maintenance planning evidence\n',1)[1].split('      - name:',1)[0]
    assert 'retention-days: 7' in upload and 'always()' in upload
    assert "inputs.preflight_only == true" in job
    assert 'default: false' in workflow.split('      bfl_contract_probe:\n',1)[1].split('      live_collect:',1)[0]


def test_maintenance_dispatch_is_exclusive_private_and_has_no_generation():
    path=Path(__file__).resolve().parents[1]/'.github/workflows/staging_zeitgeist_dryrun.yml'
    workflow=path.read_text()
    assert 'default: false' in workflow.split('      maintenance_only:\n',1)[1].split('      legacy_apify_audit:',1)[0]
    job=workflow.split('  accounting-maintenance:\n',1)[1].split('  bfl-contract-probe:',1)[0]
    assert "github.event.repository.private == true" in job
    assert "github.repository == 'Siyuan-Zhu/readme-live-sandbox'" in job
    assert 'inputs.maintenance_only == true' in job
    assert job.count('scripts/maintain_spend_control.py --apply')==1
    assert '--private-report /tmp/private-spend-maintenance/evidence.json' in job
    assert 'retention-days: 7' in job and 'if: always()' in job
    for entry in ('run_daily_zeitgeist','rehearse_staging_application','probe_bfl_contract','--live'):
        assert entry not in job
    for name in ('provider-metadata','bfl-contract-probe','isolated-staging-dryrun'):
        condition=workflow.split('  '+name+':\n',1)[1].split('\n',1)[0]
        assert 'inputs.maintenance_only != true' in condition
