"""Unbilled admission diagnostics cannot reach inference or mutable ledgers."""
import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
import pytest
from scripts import probe_director_admission as probe
from scripts.prepare_spend_ledger import prepare_candidate
from src.zeitgeist.spend_guard import BudgetGuard, FirestoreLedger
from src.state_store import _py_to_firestore_value
from tests.test_spend_provisioning import gemini_metadata, NOW

KEY = 'synthetic-director-key'
SECRET = 'synthetic-private-value-must-not-be-serialized'


def document():
    metadata = gemini_metadata(); acct = metadata['providers']['gemini']
    acct['credential_fingerprint'] = hashlib.sha256(KEY.encode()).hexdigest()[:16]
    first = 'generate_content:gemini-2.5-flash'; second = 'generate_content:gemini-3-flash-preview'
    acct['operation_caps_usd'][second] = '.027288'
    acct['price_evidence'][second] = dict(acct['price_evidence'][first], maximum_gross_usd='.027288')
    acct['request_limits'][second] = dict(acct['request_limits'][first], input_token_usd_per_million='.50', output_token_usd_per_million='3.00', thinking_level='minimal')
    return prepare_candidate(metadata, now=NOW)


class Models:
    def __init__(self, failures=None):
        self.calls = []; self.failures = failures or {}
    def count_tokens(self, **kwargs):
        self.calls.append(kwargs)
        error = self.failures.get(kwargs['model'])
        if error: raise error
        return SimpleNamespace(total_tokens=100)
    def __getattr__(self, name):
        pytest.fail('Only count_tokens may be accessed, attempted: ' + name)


def forbid_mutation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('No reservation, guard.call or mutable Firestore ledger is allowed')
    monkeypatch.setattr(BudgetGuard, 'reserve', forbidden)
    monkeypatch.setattr(BudgetGuard, 'call', forbidden)
    monkeypatch.setattr(FirestoreLedger, 'mutate', forbidden)


def test_both_approved_models_probe_real_budget_policy_without_paid_calls_or_reservations(monkeypatch):
    forbid_mutation(monkeypatch)
    doc = document(); original = copy.deepcopy(doc); models = Models()
    report = probe.run_probe({'GEMINI_API_KEY': KEY}, now=NOW,
        ledger_loader=lambda env, now: doc, client_factory=lambda key: SimpleNamespace(models=models))
    assert report['admitted'] is True and report['admitted_models'] == list(probe.MODELS)
    assert report['paid_calls'] == report['ledger_writes'] == 0
    assert report['reservation_created'] is False
    assert report['metadata_calls'] == 5
    assert report['inference_eligibility_verified'] is False
    assert [call['model'] for call in models.calls[:2]] == list(probe.MODELS)
    assert all(call['model']=='gemini-3.8-flash' for call in models.calls[2:])
    assert all(call['contents'] == probe.PROMPT for call in models.calls[:3])
    assert all(call['config']['http_options']['retry_options']['attempts'] == 1 for call in models.calls)
    assert all(result['counted_input_tokens'] == 100 for result in report['models'])
    assert doc == original and not doc['reservations']
    encoded = json.dumps(report)
    assert KEY not in encoded and probe.PROMPT not in encoded
    assert doc['providers']['gemini']['account_id'] not in encoded


def test_secret_bearing_403_count_error_is_typed_sanitized_and_does_not_hide_other_model(monkeypatch):
    forbid_mutation(monkeypatch)
    failure = HTTPError('https://generativelanguage.googleapis.com/private?key='+SECRET,403,SECRET,{'private':SECRET},BytesIO(SECRET.encode()))
    models = Models({'gemini-2.5-flash': failure})
    report = probe.run_probe({'GEMINI_API_KEY': KEY}, now=NOW,
      ledger_loader=lambda env, now: document(), client_factory=lambda key: SimpleNamespace(models=models))
    rejected, admitted = report['models']
    assert rejected['status'] == 'blocked' and rejected['blocked_phase'] == 'token_metadata'
    assert rejected['diagnostics'] == {'code':'token-metadata-unavailable','status':403,'provider':'gemini','error_type':'HTTPError'}
    assert admitted['admitted'] is True
    assert report['metadata_calls'] == 5 and report['paid_calls'] == 0
    assert SECRET not in json.dumps(report)
    assert failure.fp.tell() == 0


def test_wrong_credential_binding_blocks_before_any_token_request(monkeypatch):
    forbid_mutation(monkeypatch); models = Models()
    report = probe.run_probe({'GEMINI_API_KEY':'different-synthetic-key'},now=NOW,
      ledger_loader=lambda env,now:document(),client_factory=lambda key:SimpleNamespace(models=models))
    assert report['admitted'] is False and report['metadata_calls'] == 0
    assert all(result['blocked_phase'] == 'request_bound' for result in report['models'])
    assert not models.calls


def test_ledger_loader_uses_one_fixed_get_and_never_writes(monkeypatch):
    forbid_mutation(monkeypatch); requests = []
    doc = document()
    name = 'projects/git-readme-features/databases/(default)/documents/spend_control/'+NOW.strftime('%Y-%m')
    class Transport:
        def __init__(self, **kwargs): assert kwargs == {'timeout':15, 'follow_redirects':False}
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self,url,**kwargs):
            requests.append((url,kwargs))
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'name':name,'fields':{k:_py_to_firestore_value(v) for k,v in doc.items()}})
        def __getattr__(self,name): pytest.fail('No transport write allowed: '+name)
    monkeypatch.setattr(probe.httpx,'Client',Transport)
    monkeypatch.setattr(probe._FirestoreRestClient,'_resolve_token',lambda self:'synthetic-access-token')
    read = probe._read_shared_ledger({'GCP_PROJECT_ID':probe.PROJECT,'FIRESTORE_DATABASE_ID':probe.DATABASE},NOW)
    assert read == doc
    assert len(requests) == 1 and requests[0][0] == 'https://firestore.googleapis.com/v1/'+name


def test_mismatched_ledger_target_fails_before_auth_or_request(monkeypatch):
    monkeypatch.setattr(probe._FirestoreRestClient,'_resolve_token',lambda self:pytest.fail('No auth for wrong target'))
    with pytest.raises(RuntimeError): probe._read_shared_ledger({'GCP_PROJECT_ID':'other'},NOW)


def test_cli_suppresses_raw_client_output_and_writes_sanitized_failure_artifact(tmp_path,monkeypatch,capsys):
    def run(env):
        print(SECRET)
        return {'status':'blocked','admitted':False,'models':[],'paid_calls':0,'diagnostics':{'code':'probe-unavailable'}}
    monkeypatch.setattr(probe,'run_probe',run)
    path=tmp_path/'director-admission.json'
    assert probe.main(['--output-json',str(path)]) == 2
    assert SECRET not in path.read_text()+capsys.readouterr().out
    assert json.loads(path.read_text())['paid_calls'] == 0


def test_existing_metadata_job_installs_locked_runtime_and_retains_probe_even_after_failure():
    workflow=(Path(__file__).resolve().parents[1]/'.github/workflows/staging_zeitgeist_dryrun.yml').read_text()
    metadata=workflow.split('  provider-metadata:',1)[1].split('  isolated-staging-dryrun:',1)[0]
    assert 'python -m pip install -r requirements.txt' in metadata
    assert 'scripts/probe_director_admission.py' in metadata
    assert 'name: Probe unbilled Director token admission without a reservation\n        if: always()' in metadata
    assert 'GCP_PROJECT_ID: "git-readme-features"' in metadata
    assert 'FIRESTORE_DATABASE_ID: "(default)"' in metadata
    assert '/tmp/director-admission.json' in metadata.split('name: Retain sanitized metadata',1)[1]
    for job, condition in (('provider-metadata', '=='), ('isolated-staging-dryrun', '!=')):
        header = workflow.split('  '+job+':',1)[1].split('runs-on:',1)[0]
        expected = ("if: github.repository == 'Siyuan-Zhu/readme-live-sandbox' && "
                    "github.event.repository.private == true && inputs.preflight_only " + condition + " true")
        assert expected in header


def automatic_document():
    from tests.test_automatic_guard import budget
    from src.zeitgeist import automatic_spend as accounting
    guard=budget();doc=copy.deepcopy(guard.ledger.document);policy=copy.deepcopy(guard.automatic_policy)
    fingerprint=hashlib.sha256(KEY.encode()).hexdigest()[:16]
    doc['providers']['gemini']['credential_fingerprint']=fingerprint
    policy['providers']['gemini']['account']['credential_fingerprint']=fingerprint
    sha=accounting.digest(policy);doc['automatic_policy_sha256']=sha
    env={'GEMINI_API_KEY':KEY,'GCP_PROJECT_ID':probe.PROJECT,'FIRESTORE_DATABASE_ID':probe.DATABASE,
         'SPEND_POLICY_SHA256':sha}
    return doc,policy,env,guard.now()


def test_schema3_probes_only_active_pinned_generation_models_and_both_real_shapes(monkeypatch):
    forbid_mutation(monkeypatch)
    doc,policy,env,now=automatic_document();original=copy.deepcopy(doc);original_policy=copy.deepcopy(policy)
    models=Models()
    report=probe.run_probe(env,now=now,ledger_loader=lambda env,now:doc,
        policy_loader=lambda env,sha:policy,client_factory=lambda key:SimpleNamespace(models=models))
    assert report['admitted'] is True
    assert report['admitted_models']==['gemini-3.8-flash']
    assert {r['request_shape'] for r in report['models']}=={'text','nine_thumbnails'}
    assert all(r['model']=='gemini-3.8-flash' and r['policy_source']=='active_pinned_automatic_policy' for r in report['models'])
    assert report['policy_source']=='active_pinned_automatic_policy'
    assert report['proposed_policy_results']==[]
    assert report['metadata_calls']==3 and len(models.calls)==3
    assert report['paid_calls']==report['ledger_writes']==0 and report['reservation_created'] is False
    assert doc==original and policy==original_policy
    assert all(call['model']=='gemini-3.8-flash' for call in models.calls)


@pytest.mark.parametrize('change',['missing_sha','mismatched_sha','ledger_hash','policy_tamper','missing_policy','wrong_target','credential','missing_key','legacy_downgrade'])
def test_schema3_invalid_policy_blocks_before_sdk_or_metadata_without_legacy_fallback(monkeypatch,change):
    forbid_mutation(monkeypatch)
    doc,policy,env,now=automatic_document()
    if change=='missing_sha':env.pop('SPEND_POLICY_SHA256')
    if change=='mismatched_sha':env['SPEND_POLICY_SHA256']='0'*64
    if change=='ledger_hash':doc['automatic_policy_sha256']='0'*64
    if change=='policy_tamper':policy['cash_ceiling_usd']='999'
    if change=='missing_policy':policy=None
    if change=='wrong_target':env['GCP_PROJECT_ID']='wrong-project'
    if change=='credential':env['GEMINI_API_KEY']='unapproved-credential'
    if change=='missing_key':env.pop('GEMINI_API_KEY')
    if change=='legacy_downgrade':doc['schema_version']=2
    before=copy.deepcopy(doc)
    def forbidden(key):pytest.fail('SDK construction must follow policy/credential approval')
    report=probe.run_probe(env,now=now,ledger_loader=lambda env,now:doc,
        policy_loader=lambda env,sha:policy,client_factory=forbidden)
    assert report['admitted'] is False and report['models']==[] and report['metadata_calls']==0
    assert report['paid_calls']==report['ledger_writes']==0 and report['reservation_created'] is False
    assert report['blocked_phase']=='policy_read'
    assert doc==before


def test_schema3_policy_loader_reads_only_exact_fixed_target_policy_document(monkeypatch):
    forbid_mutation(monkeypatch)
    doc,policy,env,now=automatic_document();reads=[]
    from src.zeitgeist import automatic_spend_runtime
    class Store:
        def read(self,key):
            reads.append(key)
            return SimpleNamespace(exists=True,to_dict=lambda:copy.deepcopy(policy))
        def __getattr__(self,name):pytest.fail('Only fixed policy GET is allowed: '+name)
    monkeypatch.setattr(automatic_spend_runtime,'PrivateFirestoreStore',Store)
    assert probe._read_approved_policy(env,env['SPEND_POLICY_SHA256'])==policy
    assert reads==['policy_'+env['SPEND_POLICY_SHA256']]
    wrong=dict(env,GCP_PROJECT_ID='other')
    with pytest.raises(RuntimeError):probe._read_approved_policy(wrong,env['SPEND_POLICY_SHA256'])
    assert len(reads)==1


def test_schema3_nine_thumbnail_probe_obeys_existing_limit_without_injecting_proposal(monkeypatch):
    forbid_mutation(monkeypatch)
    from src.zeitgeist import automatic_spend as accounting
    doc,policy,env,now=automatic_document()
    operation='generate_content:gemini-3.8-flash'
    for account in (doc['providers']['gemini'],policy['providers']['gemini']['account']):
        account['request_limits'][operation]['max_input_images']=1
    sha=accounting.digest(policy);env['SPEND_POLICY_SHA256']=sha;doc['automatic_policy_sha256']=sha
    original=copy.deepcopy(doc);models=Models()
    report=probe.run_probe(env,now=now,ledger_loader=lambda env,now:doc,
        policy_loader=lambda env,sha:policy,client_factory=lambda key:SimpleNamespace(models=models))
    assert report['admitted'] is False and report['admitted_models']==[]
    text,photos=report['models']
    assert text['admitted'] is True and photos['admitted'] is False
    assert photos['request_shape']=='nine_thumbnails'
    assert report['proposed_policy_results']==[]
    assert doc==original
