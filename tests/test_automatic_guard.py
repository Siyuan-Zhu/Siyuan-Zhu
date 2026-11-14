"""Runtime policy admission and real SDK result shapes; no provider calls."""
import copy
from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from google.genai import types

from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.automatic_guard import AutomaticBudgetGuard, request_digest
from src.zeitgeist.spend_guard import MemoryLedger, SpendBlocked, guard_genai_client
from tests.test_automatic_spend import contract_for, fixtures, observe_contract, NOW
from tests.test_gemini38_contract import limits

OP = 'generate_content:gemini-3.8-flash'


def budget(now=NOW, rates=None):
    """rates=(input, output) refreshes with that published 3.8 rate instead of the pinned page."""
    ledger, policy, observation = fixtures(now)
    if rates is not None:
        observe_contract(observation, contract_for(rates=rates, op=OP))
    settings = limits()
    settings['allowed_response_model_versions'] = ['gemini-3.8-flash']
    account = policy['providers']['gemini']['account']
    account['request_limits'] = {OP: settings}
    account['operation_caps_usd'] = {OP: '0.054777'}
    ledger['providers']['gemini'].update(copy.deepcopy(account))
    ledger.update(automatic_policy_sha256=auto.digest(policy),
                  credential_binding_required=True, budget_scope='new_requests_after_cutoff')
    observation['gemini']['available_required_models'] = {'gemini-3.8-flash':['generateContent']}
    refreshed = auto.prepare_refresh(ledger, policy, observation,
        expected_policy_sha=auto.digest(policy), now=now)['after']
    # Fresh API observations replace no original manual evidence.
    for item in refreshed['providers'].values():
        item['verified_at'] = '2026-10-27T00:00:00Z'
    store = MemoryLedger(refreshed)
    return AutomaticBudgetGuard(store, policy, auto.digest(policy), lambda:now,
        credential_fingerprints={'gemini':'a'*16,'bfl':'b'*16,'apify':'c'*16},
        require_credential_binding=True)


def sdk_result(*, valid_usage=True):
    return types.GenerateContentResponse(response_id='synthetic-server-response',
        model_version='gemini-3.8-flash', usage_metadata={
            'prompt_token_count':10, 'candidates_token_count':5,
            'thoughts_token_count':2, 'total_token_count':17 if valid_usage else 999})


def client(guard, result=None, failure=None):
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        if failure: raise failure
        return result if result is not None else sdk_result()
    raw=SimpleNamespace(models=SimpleNamespace(generate_content=generate,
        count_tokens=lambda **kw:SimpleNamespace(total_tokens=10)))
    return guard_genai_client(raw, guard), calls


def generate(api):
    return api.models.generate_content(model='gemini-3.8-flash',contents='Synthetic bounded request')


def test_fresh_automatic_metadata_admits_without_restamping_old_manual_policy():
    guard=budget(); api,calls=client(guard)
    assert generate(api).response_id=='synthetic-server-response'
    assert len(calls)==1
    record=next(iter(guard.ledger.document['reservations'].values()))
    assert record['status']=='settled'
    assert Decimal(record['actual_gross_usd'])==(Decimal(10)*Decimal('.75')+Decimal(7)*Decimal('3.75'))/1000000
    assert record['provider_request_id']=='synthetic-server-response'
    assert record['usage_receipt']['accounting_basis']=='conservative_response_usage'
    assert guard.ledger.document['providers']['gemini']['verified_at']=='2026-10-27T00:00:00Z'
    assert 'Synthetic bounded request' not in str(guard.ledger.document)


def test_transport_failure_never_retries_and_retains_the_full_hold():
    guard=budget(); api,calls=client(guard,failure=TimeoutError())
    with pytest.raises(TimeoutError):generate(api)
    assert len(calls)==1
    record=next(iter(guard.ledger.document['reservations'].values()))
    assert record['status']=='reserved' and 'actual_gross_usd' not in record
    assert Decimal(guard.snapshot()['cash_exposure_usd'])==Decimal(record['max_gross_usd'])


def test_missing_or_inconsistent_usage_keeps_known_server_identity_and_full_hold():
    guard=budget(); api,calls=client(guard,result=sdk_result(valid_usage=False))
    generate(api)
    record=next(iter(guard.ledger.document['reservations'].values()))
    assert record['status']=='reserved' and 'actual_gross_usd' not in record
    assert record['provider_request_id']=='synthetic-server-response'
    assert len(calls)==1


@pytest.mark.parametrize('mutation',[
    lambda g:g.ledger.document.update(schema_version=2),
    lambda g:g.ledger.document.update(automatic_policy_sha256='0'*64),
    lambda g:g.ledger.document['providers']['gemini'].update(automatic_observed_at='2026-10-27T00:00:00Z'),
    lambda g:g.ledger.document['providers']['gemini'].update(credential_fingerprint='d'*16),
    lambda g:g.ledger.document['providers']['apify'].update(credit_baseline_usd='50'),
])
def test_invalid_accounting_or_metadata_blocks_before_provider_dispatch(mutation):
    guard=budget();mutation(guard);api,calls=client(guard)
    if guard.ledger.document['providers']['apify']['credit_baseline_usd']=='50':
        with pytest.raises(SpendBlocked):guard.begin_provider_request('apify','batch','.20','a'*64)
    else:
        with pytest.raises(SpendBlocked):generate(api)
    assert not calls and not guard.ledger.document['reservations']


def test_request_cannot_cross_the_utc_month_boundary():
    when=NOW.replace(day=31,hour=23,minute=59,second=0)
    guard=budget(when);api,calls=client(guard)
    with pytest.raises(SpendBlocked,match='month boundary'):generate(api)
    assert not calls


def test_provider_handle_binds_id_and_rejects_untyped_direct_settlement():
    guard=budget()
    handle=guard.begin_provider_request('bfl','image','.20',request_digest({'synthetic':True}))
    handle.submitted('task-one',metadata={'polling_host':'api.bfl.ai','resolution':'1.5k','aspect_ratio':'3:2'})
    with pytest.raises(SpendBlocked):handle.submitted('task-two')
    with pytest.raises(SpendBlocked):guard.settle(handle.reservation_id,'0')
    assert guard.ledger.document['reservations'][handle.reservation_id]['status']=='reserved'
    with pytest.raises(SpendBlocked,match='different request handle'):
        handle.receipt({'reservation_id':'different'})


def test_embedding_success_books_full_cap_without_inventing_response_token_usage():
    guard=budget(); operation='embed_content:gemini-embedding-001'
    account=guard.automatic_policy['providers']['gemini']['account']
    settings={'input_token_usd_per_million':'.15','output_token_usd_per_million':'0',
        'max_input_bytes':10000,'max_input_tokens':2040,'max_output_tokens':0,
        'count_tokens_reference':'https://ai.google.dev/gemini-api/docs/billing#is-gettokens-billed',
        'pricing_reference':'https://ai.google.dev/gemini-api/docs/pricing'}
    account['operation_caps_usd'][operation]='.000306';account['request_limits'][operation]=settings
    guard.expected_policy_sha=auto.digest(guard.automatic_policy)
    guard.ledger.document['automatic_policy_sha256']=guard.expected_policy_sha
    guard.ledger.document['providers']['gemini'].update(copy.deepcopy(account))
    raw=SimpleNamespace(models=SimpleNamespace(
        count_tokens=lambda **kw:SimpleNamespace(total_tokens=10),
        embed_content=lambda **kw:types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[.1,.2])])) )
    result=guard_genai_client(raw,guard).models.embed_content(model='gemini-embedding-001',contents=['Synthetic text'])
    assert len(result.embeddings)==1
    record=next(iter(guard.ledger.document['reservations'].values()))
    assert record['status']=='settled' and record['actual_gross_usd']==record['max_gross_usd']
    assert record['usage_receipt']['request_id_kind']=='local_response_digest'
    assert 'usageMetadata' not in record['usage_receipt']


def test_completion_near_month_end_does_not_require_room_to_start_another_request():
    clock=[NOW.replace(day=31,hour=23,minute=54,second=0)]
    guard=budget(clock[0]);guard.now=lambda:clock[0]
    handle=guard.begin_provider_request('bfl','image','.20','a'*64)
    started=clock[0].isoformat();clock[0]+=timedelta(seconds=120)
    handle.submitted('completed-task',submitted_at=started)
    receipt={'schema_version':1,'accounting_policy':'provider_usage_v1',
        'reservation_id':handle.reservation_id,'provider':'bfl','operation':'image',
        'credential_fingerprint':handle.credential_fingerprint,'provider_request_id':'completed-task',
        'request_sha256':'a'*64,'observed_at':clock[0].isoformat(),
        'accounted_gross_usd':'.07','accounting_basis':'provider_settled_cost','evidence_sha256':'e'*64}
    handle.receipt(receipt)
    assert guard.ledger.document['reservations'][handle.reservation_id]['actual_gross_usd']=='0.07'
    with pytest.raises(SpendBlocked,match='month boundary'):
        guard.begin_provider_request('bfl','image','.20','b'*64)


def test_authenticated_gemini_usage_outside_admitted_shape_freezes_before_next_call():
    guard=budget();result=types.GenerateContentResponse(response_id='mismatch',model_version='gemini-3.8-flash',
        usage_metadata={'prompt_token_count':11,'candidates_token_count':5,'total_token_count':16})
    api,calls=client(guard,result=result)
    with pytest.raises(SpendBlocked,match='Authenticated Gemini usage'):generate(api)
    assert guard.ledger.document['reconciliation_required'] is True
    assert len(calls)==1
    with pytest.raises(SpendBlocked):generate(api)
    assert len(calls)==1


def test_completion_near_policy_expiry_uses_original_admission_window():
    clock=[NOW];guard=budget();guard.now=lambda:clock[0]
    guard.automatic_policy['valid_until']=(NOW+timedelta(minutes=6)).isoformat()
    guard.expected_policy_sha=auto.digest(guard.automatic_policy)
    guard.ledger.document['automatic_policy_sha256']=guard.expected_policy_sha
    handle=guard.begin_provider_request('bfl','image','.20','a'*64)
    started=clock[0].isoformat();clock[0]+=timedelta(minutes=2)
    handle.submitted('policy-end-task',submitted_at=started)
    handle.receipt({'schema_version':1,'accounting_policy':'provider_usage_v1',
        'reservation_id':handle.reservation_id,'provider':'bfl','operation':'image',
        'credential_fingerprint':handle.credential_fingerprint,'provider_request_id':'policy-end-task',
        'request_sha256':'a'*64,'observed_at':clock[0].isoformat(),
        'accounted_gross_usd':'.07','accounting_basis':'provider_settled_cost','evidence_sha256':'e'*64})
    assert guard.ledger.document['reservations'][handle.reservation_id]['status']=='settled'
    with pytest.raises(SpendBlocked):guard.begin_provider_request('bfl','image','.20','b'*64)


def test_ten_cash_reload_policy_does_not_authorize_peer_provider_cash_overage():
    from tests.test_automatic_spend import with_approved_reload
    guard=budget();guard.automatic_policy=with_approved_reload(guard.automatic_policy)
    guard.expected_policy_sha=auto.digest(guard.automatic_policy)
    guard.ledger.document.update(automatic_policy_sha256=guard.expected_policy_sha,
        cash_ceiling_usd='10.00',credit_only_providers=['apify','bfl'])
    guard.ledger.document['providers']['bfl']['credit_baseline_usd']='.10'
    with pytest.raises(SpendBlocked,match='Credit-only'):
        guard.begin_provider_request('bfl','image','.20','a'*64)
    assert not guard.ledger.document['reservations']
    api,calls=client(guard);assert generate(api).response_id=='synthetic-server-response'
    assert len(calls)==1 and guard.ledger.document['cash_ceiling_usd']=='10.00'


def test_independent_reload_cap_cannot_share_cash_with_an_expired_peer_hold():
    from tests.test_automatic_spend import with_approved_reload
    guard=budget();guard.automatic_policy=with_approved_reload(guard.automatic_policy)
    guard.expected_policy_sha=auto.digest(guard.automatic_policy)
    ledger=guard.ledger.document
    ledger.update(automatic_policy_sha256=guard.expected_policy_sha,
        cash_ceiling_usd='10.00',credit_only_providers=['apify','bfl'])
    # At 9.73 USD starting balance with a 5 USD threshold/reload, 9.74 USD usage triggers
    # two reloads (10 USD cash). A separate .25 USD peer liability cannot share that cap.
    for number in range(180):
        ledger['reservations'][str(number)]={'provider':'gemini','operation':OP,
            'status':'settled','max_gross_usd':'.054','actual_gross_usd':'.054',
            'billing_reference':'provider_usage_v1:private-'+str(number)}
    ledger['reservations']['expired-apify']={'provider':'apify','operation':'batch',
        'status':'reserved','max_gross_usd':'.25','credit_excluded':True}
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked,match='uncollateralized retained liability'):
        guard.begin_provider_request('gemini',OP,'.02',request_digest({'next':'bounded'}))
    assert ledger==before


@pytest.mark.parametrize('inactive_period',['expired','future'])
def test_direct_gemini_admission_cannot_trust_unresolved_apify_collateral_outside_its_cycle(inactive_period):
    from tests.test_automatic_spend import with_approved_reload
    guard=budget();guard.automatic_policy=with_approved_reload(guard.automatic_policy)
    guard.expected_policy_sha=auto.digest(guard.automatic_policy)
    ledger=guard.ledger.document
    ledger.update(automatic_policy_sha256=guard.expected_policy_sha,
        cash_ceiling_usd='10.00',credit_only_providers=['apify','bfl'])
    account=ledger['providers']['apify']
    if inactive_period=='expired': account['allowance_expires_at']=(NOW-timedelta(seconds=1)).isoformat()
    else: account['allowance_starts_at']=(NOW+timedelta(seconds=1)).isoformat()
    ledger['reservations']['peer-unknown']={'provider':'apify','operation':'batch',
        'status':'reserved','max_gross_usd':'.25'}
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked,match='outside its verified allowance period'):
        guard.begin_provider_request('gemini',OP,'.02',request_digest({'next':'bounded'}))
    assert ledger==before


def test_reserved_request_links_to_execution_without_exposing_prompt(monkeypatch):
    monkeypatch.setenv('GITHUB_RUN_ID','12345');monkeypatch.setenv('GITHUB_RUN_ATTEMPT','2')
    guard=budget();api,_=client(guard);generate(api)
    record=next(iter(guard.ledger.document['reservations'].values()))
    assert record['execution_owner']=='github-12345-2'
    assert 'Synthetic bounded request' not in str(record)


def test_runtime_admits_apify_only_with_the_rebound_token():
    guard=budget()
    guard.begin_provider_request('apify','batch','.01',request_digest({'control':'unchanged token'}))
    guard=budget()
    guard.ledger.document['providers']['apify']['credential_fingerprint']='f'*16
    with pytest.raises(SpendBlocked,match='Runtime credential'):
        guard.begin_provider_request('apify','batch','.01',request_digest({'old':'token'}))
    guard.credential_fingerprints['apify']='f'*16
    guard.begin_provider_request('apify','batch','.01',request_digest({'new':'token'}))


# Holds and receipts at the charged rate, with rates matched at admission.
IN_BAND=('0.825','4.125')
BFL_OP='generate_image:flux-3-image'


def fake(total):
    return lambda **kw:SimpleNamespace(total_tokens=total)


class Target:
    def __init__(self):
        self.calls=0
    def __call__(self,**kwargs):
        self.calls+=1
        return sdk_result()


def bounded(guard):
    return guard.bounded_genai_request(OP,{'model':'gemini-3.8-flash','contents':'Synthetic bounded request'})


def test_lower_observed_rate_keeps_the_approved_hold():
    guard=budget(rates=('0.70','3.50'))
    assert 'price_band_audit' not in guard.ledger.document['providers']['gemini']
    c=guard.counted_genai_admission(OP,bounded(guard),count_tokens=fake(10000))
    assert c['input_token_usd_per_million']=='0.75' and c['output_token_usd_per_million']=='3.75'
    assert Decimal(c['maximum'])==(Decimal(10000)*Decimal('0.75')+Decimal(1500)*Decimal('3.75'))/10**6


def test_in_band_gemini_hold_uses_the_observed_rate():
    guard=budget(rates=IN_BAND)
    c=guard.counted_genai_admission(OP,bounded(guard),count_tokens=fake(10000))
    assert Decimal(c['maximum'])==Decimal('0.0144375')
    assert c=={'maximum':c['maximum'],'reference':'old-observation','admitted_input_tokens':10000,
        'max_output_tokens':1500,'input_token_usd_per_million':'0.825','output_token_usd_per_million':'4.125',
        'approved_input_token_usd_per_million':'0.75','approved_output_token_usd_per_million':'3.75'}
    assert guard.counted_genai_cap(OP,bounded(guard),fake(10000))==(Decimal('0.0144375'),'old-observation')


def test_hold_records_charged_and_approved_rates_in_request_context():
    guard=budget(rates=IN_BAND);api,calls=client(guard);generate(api)
    ledger=guard.ledger;rid=next(iter(ledger.document['reservations']))
    ctx=ledger.document['reservations'][rid]['request_context']
    assert ctx['input_token_usd_per_million']=='0.825' and ctx['approved_input_token_usd_per_million']=='0.75'
    assert ctx=={'admitted_input_tokens':10,'max_output_tokens':1500,
        'input_token_usd_per_million':'0.825','output_token_usd_per_million':'4.125',
        'approved_input_token_usd_per_million':'0.75','approved_output_token_usd_per_million':'3.75'}


def test_in_band_receipt_uses_the_rate_recorded_on_the_hold():
    guard=budget(rates=IN_BAND)
    def generate_after_audit_clears(**kwargs):
        guard.ledger.document['providers']['gemini'].pop('price_band_audit')
        return sdk_result()
    raw=SimpleNamespace(models=SimpleNamespace(generate_content=generate_after_audit_clears,count_tokens=fake(10)))
    generate(guard_genai_client(raw,guard))
    record=next(iter(guard.ledger.document['reservations'].values()))
    assert 'price_band_audit' not in guard.ledger.document['providers']['gemini']
    ctx_in_tokens,out=record['request_context']['admitted_input_tokens'],7
    receipt=record['usage_receipt']
    assert Decimal(receipt['accounted_gross_usd'])==(Decimal(ctx_in_tokens)*Decimal('0.825')+Decimal(out)*Decimal('4.125'))/10**6
    assert record['actual_gross_usd']==receipt['accounted_gross_usd']


def test_pre_band_hold_without_recorded_rates_receipts_at_approved_rate():
    guard=budget(rates=IN_BAND)
    record={'operation':OP,'request_context':{'admitted_input_tokens':10,'max_output_tokens':1500}}
    response=sdk_result().model_dump(by_alias=True,exclude_none=True)
    receipt=auto.gemini_usage_receipt('pre-band',record,response,credential_fingerprint='a'*16,
        observed_at=NOW.isoformat(),request_sha256='f'*64,policy=guard.automatic_policy)
    assert Decimal(receipt['accounted_gross_usd'])==(Decimal(10)*Decimal('0.75')+Decimal(7)*Decimal('3.75'))/10**6


def test_audit_outside_the_band_blocks_admission():
    guard=budget(rates=IN_BAND);ledger=guard.ledger
    ledger.document['providers']['gemini']['price_band_audit']['above_approval'][0]['observed']='0.9'
    with pytest.raises(SpendBlocked,match='Primary price snapshot'):
        guard.counted_genai_admission(OP,bounded(guard),count_tokens=fake(10000))
    assert ledger.document['reservations']=={}


def test_audit_cleared_between_count_and_admission_blocks():
    guard=budget(rates=IN_BAND);ledger=guard.ledger;request=bounded(guard)
    c=guard.counted_genai_admission(OP,request,count_tokens=fake(10000))
    ledger.document['providers']['gemini'].pop('price_band_audit')
    target=Target()
    with pytest.raises(SpendBlocked,match='Billing policy changed after token counting'):
        guard.invoke_prepared_genai(OP,c,request,target)
    assert target.calls==0 and ledger.document['reservations']=={}


def test_audit_added_between_count_and_admission_blocks():
    guard=budget(rates=IN_BAND);ledger=guard.ledger;request=bounded(guard)
    audit=ledger.document['providers']['gemini'].pop('price_band_audit')
    c=guard.counted_genai_admission(OP,request,count_tokens=fake(10000))
    assert c['input_token_usd_per_million']=='0.75'
    ledger.document['providers']['gemini']['price_band_audit']=audit
    target=Target()
    with pytest.raises(SpendBlocked,match='Billing policy changed after token counting'):
        guard.invoke_prepared_genai(OP,c,request,target)
    assert target.calls==0 and ledger.document['reservations']=={}


def test_tampered_counted_maximum_blocks():
    guard=budget(rates=IN_BAND);ledger=guard.ledger;request=bounded(guard)
    c=guard.counted_genai_admission(OP,request,count_tokens=fake(10000))
    c['maximum']=str(Decimal(c['maximum'])-Decimal('0.000001'))
    target=Target()
    with pytest.raises(SpendBlocked,match='Counted admission identity is inconsistent'):
        guard.invoke_prepared_genai(OP,c,request,target)
    assert target.calls==0 and ledger.document['reservations']=={}


@pytest.mark.parametrize('field',['max_output_tokens','admitted_input_tokens'])
def test_consistently_shrunk_counted_shape_still_blocks(field):
    guard=budget(rates=IN_BAND);ledger=guard.ledger;request=bounded(guard)
    c=guard.counted_genai_admission(OP,request,count_tokens=fake(10000))
    if field=='max_output_tokens': c['max_output_tokens']=0
    else: c['admitted_input_tokens']=-1
    c['maximum']=str((Decimal(c['admitted_input_tokens'])*Decimal('0.825')+Decimal(c['max_output_tokens'])*Decimal('4.125'))/10**6)
    target=Target()
    with pytest.raises(SpendBlocked,match='Counted admission identity is inconsistent'):
        guard.invoke_prepared_genai(OP,c,request,target)
    assert target.calls==0 and ledger.document['reservations']=={}


def test_bfl_hold_records_charged_and_approved_unit_rate():
    guard=budget();ledger=guard.ledger
    account=guard.automatic_policy['providers']['bfl']['account']
    account['operation_caps_usd']={BFL_OP:'0.07'}
    guard.expected_policy_sha=auto.digest(guard.automatic_policy)
    ledger.document['automatic_policy_sha256']=guard.expected_policy_sha
    ledger.document['providers']['bfl'].update(copy.deepcopy(account))
    h=guard.begin_provider_request('bfl',BFL_OP,None,'a'*64)
    r=ledger.document['reservations'][h.reservation_id]
    assert r['max_gross_usd']=='0.07' and r['request_context']=={'unit_price_usd':'0.07','approved_unit_price_usd':'0.07'}
