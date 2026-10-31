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
from tests.test_automatic_spend import fixtures, NOW
from tests.test_gemini38_contract import limits

OP = 'generate_content:gemini-3.8-flash'


def budget(now=NOW):
    ledger, policy, observation = fixtures(now)
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
