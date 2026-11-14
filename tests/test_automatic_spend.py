"""Offline automatic policy planning never refreshes manual facts or loses holds."""
import copy
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import pytest
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import SpendBlocked

NOW = datetime(2026,10,31,12,tzinfo=timezone.utc)
OP = 'generate_content:gemini-2.5-flash'
START='2026-10-26T00:00:00Z'; END='2026-11-25T23:59:59.999Z'


def fixtures(now=NOW):
    policy={'schema_version':1,'target':auto.TARGET,'approval_reference':'candidate-needs-human-approval',
        'approved_at':'2026-10-30T16:00:00Z','valid_until':'2026-12-31T00:00:00Z',
        'cash_ceiling_usd':'2.50','usage_accounting':'provider_usage_v1','providers':{}}
    providers={}
    for name,mode in auto.MODES.items():
        op=OP if name=='gemini' else 'batch' if name=='apify' else 'image'
        account={'account_id':name+'-account','plan_id':'FREE' if name=='apify' else 'pinned-plan',
            'credential_fingerprint':{'gemini':'a','bfl':'b','apify':'c'}[name]*16,
            'credit_kind':{'gemini':'none','bfl':'prepaid','apify':'free'}[name],
            'operation_caps_usd':{op:'.20'}, 'request_limits':{}, 'max_request_duration_seconds':300}
        if name=='gemini':
            account['request_limits'][OP]={'input_token_usd_per_million':'.30','output_token_usd_per_million':'2.50',
                'allowed_response_model_versions':['gemini-2.5-flash']}
        pinned={'mode':mode,'price_evidence_mode':'dynamic_snapshot','account':copy.deepcopy(account),'credit_limit_usd':{'gemini':'0','bfl':'9.25','apify':'5'}[name],
            'price_snapshot_sha256':auto.digest(name),'price_source_url':{'gemini':'https://ai.google.dev/gemini-api/docs/pricing','bfl':'https://bfl.ai/pricing','apify':'https://api.apify.com/v2/actors/pinned'}[name], 'price_valid_until':'2026-12-31T00:00:00Z'}
        if name in ('gemini','bfl'):
            pinned['top_up_disabled_attestation']={'automatic_top_up_enabled':False,
                'observed_at':'2026-10-30T15:30:00Z','reference':'one-time-reviewed-evidence'}
        if name=='bfl': pinned['credit_unit_usd']='.01'
        if name=='apify':
            pinned.update(account_fingerprint='account-fingerprint',actor_build='0.0.648',actor_pricing_sha256=auto.digest({'post':'.0017'}),
                actor_build_id='immutable-test-build',actor_image_digest='sha256:'+'d'*64,actor_schema_sha256='e'*64)
        policy['providers'][name]=pinned
        amount={'gemini':'0','bfl':'9.25','apify':'4.50'}[name]
        account.update(credit_baseline_usd=amount, remaining_credit_usd=amount,
            verified_at='2026-10-30T15:30:00Z',policy_verified_at='2026-10-30T15:30:00Z',
            policy_reference='original-manual-evidence',verification_reference='old-observation')
        if name=='apify': account.update(allowance_period_id=START+'/'+END,allowance_starts_at=START,allowance_expires_at=END)
        providers[name]=account
    observation={'observed_at':now.isoformat(),'price_snapshots':{}}
    for name in providers:
        observation[name]={'status':'metadata_verified','credential_fingerprint':providers[name]['credential_fingerprint']}
        observation['price_snapshots'][name]={'sha256':policy['providers'][name]['price_snapshot_sha256'],
            'source_url':policy['providers'][name]['price_source_url'],'observed_at':now.isoformat()}
    observation['gemini']['available_required_models']={'gemini-2.5-flash':['generateContent']}
    observation['bfl']['balance_credits']='925'
    observation['apify'].update(account_fingerprint='account-fingerprint',configured_build='0.0.648',build_pin_verified=True,
        build_id='immutable-test-build',image_digest='sha256:'+'d'*64,input_schema_sha256='e'*64,
        pricing={'post':'.0017'},billing={'status':'verified_included_credit_snapshot',
            'plan':{'id':'FREE','isEnabled':True,'monthlyBasePriceUsd':'0','monthlyUsageCreditsUsd':'5'},
            'active_actor_job_count':0,'cycle_matches_limits':True,'usage_matches_limits':True,
            'usage_cycle':{'startAt':START,'endAt':END},'remaining_included_credit_usd':'4.50'})
    ledger={'schema_version':3,'automatic_accounting_version':1,'automatic_policy_sha256':auto.digest(policy),
        'month':'2026-10','budget_window_start':'2026-10-30T04:09:08Z','cash_ceiling_usd':'2.50',
        'cash_spent_before_ledger_usd':'0','providers':providers,'reservations':{}}
    return ledger,policy,observation


def refresh(ledger,policy,observation,now=NOW):
    return auto.prepare_refresh(ledger,policy,observation,expected_policy_sha=auto.digest(policy),now=now)['after']


def with_approved_reload(policy):
    policy=copy.deepcopy(policy);policy.update(cash_ceiling_usd='10.00',credit_only_providers=['apify','bfl'])
    gemini=policy['providers']['gemini'];gemini.pop('top_up_disabled_attestation',None)
    gemini['billing_account_id']='AAAAAA-BBBBBB-CCCCCC'
    gemini['bounded_auto_reload_attestation']={'automatic_top_up_enabled':True,
        'billing_account_id':gemini['billing_account_id'],'trigger_balance_usd':'5',
        'reload_amount_usd':'5','monthly_auto_charge_limit_usd':'10',
        'observed_at':'2026-10-30T15:30:00Z','reference':'explicit-owner-approved-current-reload-settings'}
    return policy


def test_explicit_approved_ten_cash_policy_keeps_bounded_reload_on():
    ledger,policy,obs=fixtures();policy=with_approved_reload(policy)
    ledger.update(cash_ceiling_usd='10.00',credit_only_providers=['apify','bfl'],automatic_policy_sha256=auto.digest(policy))
    assert auto.validate_policy(policy,auto.digest(policy),NOW)==policy
    assert refresh(ledger,policy,obs)['cash_ceiling_usd']=='10.00'
    policy['providers']['gemini']['bounded_auto_reload_attestation']['monthly_auto_charge_limit_usd']='20'
    with pytest.raises(SpendBlocked):auto.validate_policy(policy,auto.digest(policy),NOW)


def test_reload_on_cannot_fit_the_previous_two_fifty_policy_or_allow_peer_cash():
    ledger,policy,obs=fixtures();policy=with_approved_reload(policy);policy['cash_ceiling_usd']='2.50'
    with pytest.raises(SpendBlocked):auto.validate_policy(policy,auto.digest(policy),NOW)
    policy=with_approved_reload(fixtures()[1]);policy['credit_only_providers']=[]
    with pytest.raises(SpendBlocked):auto.validate_policy(policy,auto.digest(policy),NOW)


def hold(provider,amount,**kw):
    return dict(provider=provider,operation=OP if provider=='gemini' else 'batch',max_gross_usd=amount,
                created_at=NOW.isoformat(),status='reserved',**kw)


@pytest.mark.parametrize('change', ['reload_amount', 'threshold', 'monthly_limit', 'policy_ceiling'])
def test_bounded_reload_requires_the_exact_user_approved_five_five_ten_shape(change):
    policy=with_approved_reload(fixtures()[1])
    reload=policy['providers']['gemini']['bounded_auto_reload_attestation']
    if change=='reload_amount': reload['reload_amount_usd']='10'
    if change=='threshold': reload['trigger_balance_usd']='100'
    if change=='monthly_limit': reload['monthly_auto_charge_limit_usd']='5'
    if change=='policy_ceiling': policy['cash_ceiling_usd']='9'
    with pytest.raises(SpendBlocked,match='bounded Gemini reload'):
        auto.validate_policy(policy,auto.digest(policy),NOW)


def reload_fixtures(now=NOW):
    ledger,policy,obs=fixtures(now);policy=with_approved_reload(policy)
    ledger.update(cash_ceiling_usd='10.00',credit_only_providers=['apify','bfl'],
        automatic_policy_sha256=auto.digest(policy))
    return ledger,policy,obs


@pytest.mark.parametrize('liability',['excluded_peer','unattributed_cash','overdrawn_credit'])
def test_refresh_cannot_share_gemini_reload_cash_with_retained_peer_liability(liability):
    ledger,policy,obs=reload_fixtures()
    if liability=='excluded_peer':ledger['reservations']['old']=hold('apify','.25',credit_excluded=True)
    if liability=='unattributed_cash':ledger['cash_spent_before_ledger_usd']='.25'
    if liability=='overdrawn_credit':ledger['reservations']['old']=hold('bfl','10')
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked,match='Credit-only'):
        refresh(ledger,policy,obs)
    assert ledger==before


def test_new_apify_cycle_cannot_convert_unknown_holds_into_cash_alongside_gemini_reload():
    now=datetime(2026,11,26,0,1,tzinfo=timezone.utc)
    ledger,policy,obs=reload_fixtures(now);ledger['month']='2026-11'
    ledger['reservations']['old-unknown']=hold('apify','.25')
    obs['apify']['billing'].update(usage_cycle={'startAt':'2026-11-26T00:00:00Z','endAt':'2026-12-25T23:59:59Z'},
        remaining_included_credit_usd='5')
    before=copy.deepcopy(ledger)
    auto.assert_credit_only_funding(ledger, now=NOW)
    with pytest.raises(SpendBlocked,match='outside its verified allowance period'):
        refresh(ledger,policy,obs,now)
    assert ledger==before and 'old-unknown' in ledger['reservations']


def test_month_handoff_keeps_unknowns_and_refuses_peer_cash_collateral_gap():
    now=datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    ledger,policy,obs=reload_fixtures(now)
    ledger['reservations']['peer-unknown']=hold('apify','.25',credit_excluded=True)
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked,match='uncollateralized retained liability'):
        auto.prepare_month_handoff(ledger,policy,obs,expected_policy_sha=auto.digest(policy),
            source_identity=dict(auto.TARGET,document_id='2026-10',update_time=now.isoformat()),now=now)
    assert ledger==before


def test_zero_settled_excluded_peer_liability_is_safe_without_discarding_evidence():
    ledger,policy,obs=reload_fixtures()
    record=hold('apify','.25',credit_excluded=True)
    record.update(status='settled',actual_gross_usd='0',billing_reference='provider_usage_v1:zero')
    ledger['reservations']['zero-cost']=record
    auto.assert_credit_only_funding(ledger, now=NOW)
    assert refresh(ledger,policy,obs)['reservations']['zero-cost']==record


def test_credit_only_funding_requires_current_time_and_blocks_newly_observed_collateral_loss():
    ledger,policy,obs=reload_fixtures()
    with pytest.raises(SpendBlocked,match='current time'):
        auto.assert_credit_only_funding(ledger)
    ledger['reservations']['bfl-unknown']=hold('bfl','.25')
    obs['bfl']['balance_credits']='0'
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked,match='remaining verified collateral'):
        refresh(ledger,policy,obs)
    assert ledger==before


def test_settled_prior_apify_cycle_can_renew_without_any_unresolved_peer_hold():
    now=datetime(2026,11,26,0,1,tzinfo=timezone.utc)
    ledger,policy,obs=reload_fixtures(now);ledger['month']='2026-11'
    record=hold('apify','.25');record.update(status='settled',actual_gross_usd='.10',
        billing_reference='provider_usage_v1:known-prior-cycle')
    ledger['reservations']['known']=record
    obs['apify']['billing'].update(usage_cycle={'startAt':'2026-11-26T00:00:00Z','endAt':'2026-12-25T23:59:59Z'},
        remaining_included_credit_usd='5')
    after=refresh(ledger,policy,obs,now)
    assert after['providers']['apify']['credit_baseline_usd']=='5'
    assert after['cash_spent_before_ledger_usd']=='0'
    assert 'known' not in after['reservations']
    assert after['retired_settlement_groups'][-1]['reservation_ids']==['known']


def test_fresh_api_observations_preserve_old_manual_evidence_and_all_unknown_holds():
    ledger,policy,observation=fixtures();ledger['reservations']['lost']=hold('gemini','.05')
    before=copy.deepcopy(ledger); candidate=refresh(ledger,policy,observation)
    assert ledger==before
    assert candidate['reservations']==before['reservations']
    for name in ledger['providers']:
        assert candidate['providers'][name]['policy_verified_at']=='2026-10-30T15:30:00Z'
        assert candidate['providers'][name]['verified_at']=='2026-10-30T15:30:00Z'
        assert candidate['providers'][name]['automatic_observed_at']==NOW.isoformat()
    assert auto.cash_exposure(candidate)==auto.money('.05')


@pytest.mark.parametrize('change',['topup_on','expired_policy','stale_metadata','price_change','credential','new_grant','legacy'])
def test_automatic_observations_cannot_waive_approval_identity_price_or_freshness(change):
    ledger,policy,obs=fixtures()
    if change=='topup_on': policy['providers']['gemini']['top_up_disabled_attestation']['automatic_top_up_enabled']=True
    if change=='expired_policy': policy['valid_until']=NOW.isoformat()
    if change=='stale_metadata': obs['observed_at']=(NOW-timedelta(minutes=6)).isoformat()
    if change=='price_change': obs['price_snapshots']['gemini']['sha256']='different'
    if change=='credential': obs['bfl']['credential_fingerprint']='d'*16
    if change=='new_grant': obs['apify']['billing']['plan']['monthlyUsageCreditsUsd']='50'
    if change=='legacy': ledger['schema_version']=2
    with pytest.raises(SpendBlocked): refresh(ledger,policy,obs)


def test_prepaid_tranche_only_shrinks_and_credit_get_cannot_mint_a_topup():
    ledger,policy,obs=fixtures();obs['bfl']['balance_credits']='2000'
    assert refresh(ledger,policy,obs)['providers']['bfl']['credit_baseline_usd']=='9.25'
    obs['bfl']['balance_credits']='700'
    first=refresh(ledger,policy,obs)
    assert first['providers']['bfl']['credit_baseline_usd']=='7.00'
    obs['bfl']['balance_credits']='925'
    assert refresh(first,policy,obs)['providers']['bfl']['credit_baseline_usd']=='7.00'


def test_atomic_november_handoff_has_no_new_apify_grant_and_carries_unknown_ids():
    now=datetime(2026,11,1,0,1,tzinfo=timezone.utc)
    ledger,policy,obs=fixtures(now);ledger['reservations']['unknown']=hold('gemini','.10')
    identity=dict(auto.TARGET,document_id='2026-10',update_time='2026-11-01T00:00:05Z')
    plan=auto.prepare_month_handoff(ledger,policy,obs,expected_policy_sha=auto.digest(policy),source_identity=identity,now=now)
    new=plan['destination_after']
    assert len(plan['writes'])==2
    assert plan['writes'][0]['precondition']=={'update_time':identity['update_time']}
    assert plan['writes'][1]['precondition']=={'exists':False}
    assert plan['source_after']['closed_for_rollover'] is True
    assert new['providers']['apify']['credit_baseline_usd']=='4.50'
    assert new['providers']['apify']['allowance_period_id']==START+'/'+END
    assert set(new['reservations'])=={'unknown'}
    assert new['reservations']['unknown']['origin_month']=='2026-10'
    assert new['cash_ceiling_usd']=='2.50' and auto.cash_exposure(new)==auto.money('.10')
    assert auto.reconcile_month_handoff(plan,plan['source_after'],new)['status']=='known_committed'
    changed=copy.deepcopy(new);changed['reservations']['new-visitor']=hold('gemini','.01')
    with pytest.raises(SpendBlocked): auto.reconcile_month_handoff(plan,plan['source_after'],changed)
    assert ledger.get('closed_for_rollover') is None


def test_authenticated_actual_apify_cycle_can_grant_once_without_funding_old_unknowns():
    now=datetime(2026,11,26,0,1,tzinfo=timezone.utc)
    ledger,policy,obs=fixtures(now);ledger['month']='2026-11'
    ledger['reservations']['old-unknown']=hold('apify','.20')
    obs['apify']['billing'].update(usage_cycle={'startAt':'2026-11-26T00:00:00Z','endAt':'2026-12-25T23:59:59Z'},remaining_included_credit_usd='5')
    after=refresh(ledger,policy,obs,now)
    assert after['providers']['apify']['credit_baseline_usd']=='5'
    assert after['reservations']['old-unknown']['credit_excluded'] is True
    assert auto.cash_exposure(after)==auto.money('.20')
    again=refresh(after,policy,obs,now)
    assert len(again['providers']['apify']['automatic_grants'])==1
    assert auto.cash_exposure(again)==auto.money('.20')


def test_full_unknown_cash_carry_eventually_stops_instead_of_resetting_each_month():
    now=datetime(2026,11,1,0,1,tzinfo=timezone.utc);ledger,policy,obs=fixtures(now)
    ledger['reservations']={'unknown':hold('gemini','2.51')}
    with pytest.raises(SpendBlocked,match='liabilit'):
        auto.prepare_month_handoff(ledger,policy,obs,expected_policy_sha=auto.digest(policy),
            source_identity=dict(auto.TARGET,document_id='2026-10',update_time=now.isoformat()),now=now)


def request_fixture():
    ledger,policy,obs=fixtures()
    record=hold('gemini','.00155',credential_fingerprint='a'*16,provider_request_id='response-1',request_sha256='f'*64,
        request_context={'admitted_input_tokens':1000,'max_output_tokens':500})
    ledger['reservations']['reservation-1']=record
    response={'responseId':'response-1','modelVersion':'gemini-2.5-flash','usageMetadata':{
        'promptTokenCount':800,'candidatesTokenCount':40,'thoughtsTokenCount':60,'totalTokenCount':900,
        'cachedContentTokenCount':100,'candidatesTokensDetails':[{'modality':'TEXT','tokenCount':40}]}}
    receipt=auto.gemini_usage_receipt('reservation-1',record,response,credential_fingerprint='a'*16,
        observed_at=NOW.isoformat(),request_sha256='f'*64,policy=policy)
    return ledger,policy,response,receipt


def test_authenticated_request_usage_counts_thoughts_and_keeps_input_floor_without_invoice_claim():
    ledger,policy,response,receipt=request_fixture()
    # 1000 input at USD0.30/M + (40 visible+60 thoughts) at USD2.50/M.
    assert receipt['accounted_gross_usd']=='0.00055'
    assert receipt['accounting_basis']=='conservative_response_usage'
    after=auto.admit_usage_receipt(ledger,policy,receipt,expected_policy_sha=auto.digest(policy),now=NOW)
    assert auto.cash_exposure(after)==auto.money('.00055')
    assert ledger['reservations']['reservation-1']['status']=='reserved'
    assert auto.admit_usage_receipt(after,policy,receipt,expected_policy_sha=auto.digest(policy),now=NOW)==after


@pytest.mark.parametrize('change',['id','credential','request','duplicate','legacy','timestamp'])
def test_wrong_or_unowned_usage_receipts_never_release_a_hold(change):
    ledger,policy,response,receipt=request_fixture()
    if change=='id': receipt['provider_request_id']='another'
    if change=='credential': receipt['credential_fingerprint']='d'*16
    if change=='request': receipt['request_sha256']='e'*64
    if change=='duplicate': ledger['reservations']['other']=copy.deepcopy(ledger['reservations']['reservation-1'])
    if change=='legacy': ledger.pop('automatic_accounting_version')
    if change=='timestamp': receipt['observed_at']=(NOW-timedelta(days=1)).isoformat()
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked): auto.admit_usage_receipt(ledger,policy,receipt,expected_policy_sha=auto.digest(policy),now=NOW)
    assert ledger==before


@pytest.mark.parametrize('change',['missing','thoughts','model','tool','output','image'])
def test_unsafe_gemini_usage_leaves_maximum_hold(change):
    ledger,policy,response,receipt=request_fixture()
    if change=='missing': response['usageMetadata'].pop('totalTokenCount')
    if change=='thoughts': response['usageMetadata']['thoughtsTokenCount']=1000
    if change=='model': response['modelVersion']='unpriced-version'
    if change=='tool': response['usageMetadata']['toolUsePromptTokenCount']=1
    if change=='output': response['usageMetadata'].update(candidatesTokenCount=600,totalTokenCount=1460)
    if change=='image': response['usageMetadata']['candidatesTokensDetails'][0]['modality']='IMAGE'
    with pytest.raises(SpendBlocked): auto.gemini_usage_receipt('reservation-1',ledger['reservations']['reservation-1'],response,
        credential_fingerprint='a'*16,observed_at=NOW.isoformat(),request_sha256='f'*64,policy=policy)
    assert ledger['reservations']['reservation-1']['status']=='reserved'


def test_confirmed_usage_retires_at_month_boundary_but_unknown_cash_does_not():
    now=datetime(2026,11,1,0,1,tzinfo=timezone.utc);ledger,policy,obs=fixtures(now)
    ledger['reservations']={'done':hold('gemini','.10'), 'unknown':hold('gemini','.20')}
    ledger['reservations']['done'].update(status='settled',actual_gross_usd='.01',billing_reference='provider_usage_v1:evidence')
    before=copy.deepcopy(ledger)
    plan=auto.prepare_month_handoff(ledger,policy,obs,expected_policy_sha=auto.digest(policy),
        source_identity=dict(auto.TARGET,document_id='2026-10',update_time=now.isoformat()),now=now)
    assert set(plan['destination_after']['reservations'])=={'unknown'}
    assert auto.cash_exposure(plan['destination_after'])==auto.money('.20')
    assert plan['source_after']['reservations']==before['reservations']


def test_successful_embedding_completion_accounts_full_bound_without_invented_token_usage():
    ledger,policy,obs=fixtures()
    operation='embed_content:gemini-embedding-001'
    for account in (ledger['providers']['gemini'],policy['providers']['gemini']['account']):
        account['operation_caps_usd'][operation]='.003'
        account['request_limits'][operation]={}
    ledger['automatic_policy_sha256']=auto.digest(policy)
    record=hold('gemini','.003',credential_fingerprint='a'*16,request_sha256='f'*64,
        request_context={'embedding_input_count':2,'counted_input_tokens':20000})
    record['operation']=operation
    response={'embeddings':[{'values':[.1,.2]},{'values':[.3,.4]}]}
    receipt=auto.gemini_embedding_receipt('embedding-1',record,response,credential_fingerprint='a'*16,
        observed_at=NOW.isoformat(),request_sha256='f'*64)
    assert receipt['accounted_gross_usd']=='.003' or receipt['accounted_gross_usd']=='0.003'
    assert receipt['request_id_kind']=='local_response_digest'
    record['provider_request_id']=receipt['provider_request_id'];ledger['reservations']['embedding-1']=record
    after=auto.admit_usage_receipt(ledger,policy,receipt,expected_policy_sha=auto.digest(policy),now=NOW)
    assert auto.cash_exposure(after)==auto.money('.003')
    assert after['reservations']['embedding-1']['status']=='settled'
    assert record['status']=='reserved'
    with pytest.raises(SpendBlocked): auto.gemini_embedding_receipt('embedding-1',record,
        {'embeddings':[{'values':[float('nan')]}]},credential_fingerprint='a'*16,
        observed_at=NOW.isoformat(),request_sha256='f'*64)


def test_repeated_same_usage_evidence_can_be_observed_later_without_reaccounting():
    ledger,policy,response,receipt=request_fixture()
    first=auto.admit_usage_receipt(ledger,policy,receipt,expected_policy_sha=auto.digest(policy),now=NOW)
    later=copy.deepcopy(receipt);later['observed_at']=(NOW+timedelta(seconds=30)).isoformat()
    again=auto.admit_usage_receipt(first,policy,later,expected_policy_sha=auto.digest(policy),now=NOW+timedelta(seconds=30))
    assert again['reservations']['reservation-1']['billing_reference']==first['reservations']['reservation-1']['billing_reference']
    assert auto.cash_exposure(again)==auto.cash_exposure(first)


def test_price_window_covers_the_entire_bounded_request_and_unknown_provider_is_not_ignored():
    ledger,policy,obs=fixtures()
    policy['providers']['gemini']['price_valid_until']=(NOW+timedelta(seconds=299)).isoformat()
    ledger['automatic_policy_sha256']=auto.digest(policy)
    with pytest.raises(SpendBlocked,match='expires before'): refresh(ledger,policy,obs)
    ledger,policy,obs=fixtures();ledger['reservations']['unknown-provider']=hold('unpriced-provider','.50')
    with pytest.raises(SpendBlocked,match='Unknown provider liability'): refresh(ledger,policy,obs)


def test_explicit_published_validity_uses_original_contract_without_restamping_it():
    ledger,policy,obs=fixtures()
    for name,pinned in policy['providers'].items():
        pinned['price_evidence_mode']='published_validity'
        pinned['published_contract']={'reference':pinned['price_source_url'], 'tariff_sha256':pinned['price_snapshot_sha256'],
            'observed_at':'2026-10-30T15:00:00Z','valid_from':'2026-10-30T00:00:00Z','valid_until':'2026-12-31T00:00:00Z'}
    ledger['automatic_policy_sha256']=auto.digest(policy)
    obs.pop('price_snapshots')
    original=copy.deepcopy(policy)
    assert refresh(ledger,policy,obs)['providers']['gemini']['automatic_observed_at']==NOW.isoformat()
    assert policy==original
    policy['providers']['gemini']['published_contract']['valid_until']='2026-11-01T00:00:00Z'
    ledger['automatic_policy_sha256']=auto.digest(policy)
    with pytest.raises(SpendBlocked,match='validity commitment'): refresh(ledger,policy,obs)


# 2026-11-13 incident: Apify invalidated the token value; a replacement token for the same
# verified account could never be used, because the ledger bound the exact token hash.
def test_new_apify_token_for_the_same_verified_account_rebinds_without_policy_change():
    ledger,policy,obs=fixtures()
    obs['apify']['credential_fingerprint']='f'*16
    policy_before=copy.deepcopy(policy)
    after=refresh(ledger,policy,obs)
    apify=after['providers']['apify']
    assert apify['credential_fingerprint']=='f'*16
    assert apify['credential_rebinds']==[{'from':'c'*16,'to':'f'*16,'at':obs['observed_at']}]
    assert policy==policy_before
    assert after['providers']['bfl']['credential_fingerprint']=='b'*16
    assert refresh(after,policy,copy.deepcopy(obs))['providers']['apify']['credential_fingerprint']=='f'*16


@pytest.mark.parametrize('change',['other_account','unverified','malformed','open_hold','no_account_identity'])
def test_apify_rebind_requires_same_verified_account_wellformed_token_and_no_open_hold(change):
    ledger,policy,obs=fixtures()
    obs['apify']['credential_fingerprint']='f'*16
    if change=='other_account': obs['apify']['account_fingerprint']='another-account'
    if change=='unverified': obs['apify']['status']='incomplete'
    if change=='malformed': obs['apify']['credential_fingerprint']='F'*16
    if change=='open_hold': ledger['reservations']['open-apify']=hold('apify','.10',credential_fingerprint='c'*16)
    if change=='no_account_identity':
        policy['providers']['apify'].pop('account_fingerprint'); obs['apify']['account_fingerprint']=None
        ledger['automatic_policy_sha256']=auto.digest(policy)
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked): refresh(ledger,policy,obs)
    assert ledger==before


@pytest.mark.parametrize('name',['gemini','bfl'])
def test_gemini_and_bfl_keys_stay_bound_to_the_exact_token(name):
    ledger,policy,obs=fixtures()
    obs[name]['credential_fingerprint']='f'*16
    with pytest.raises(SpendBlocked,match='credential binding'): refresh(ledger,policy,obs)


def test_settled_apify_history_does_not_block_a_rebind_and_both_rebinds_are_kept():
    ledger,policy,obs=fixtures()
    ledger['reservations']['done']=dict(hold('apify','.10'),status='settled',actual_gross_usd='.05',
        billing_reference='apify-run-1',credential_fingerprint='c'*16)
    obs['apify']['credential_fingerprint']='f'*16
    first=refresh(ledger,policy,obs)
    obs2=copy.deepcopy(obs); obs2['apify']['credential_fingerprint']='9'*16
    second=refresh(first,policy,obs2)
    assert [r['to'] for r in second['providers']['apify']['credential_rebinds']]==['f'*16,'9'*16]


def stale_apify_hold(**changes):
    record=dict(hold('apify','.16'),created_at=(NOW-timedelta(days=11)).isoformat(),
        provider_request_id='stale-run',credential_fingerprint='c'*16)
    record.update(changes)
    return record


def test_a_stale_unsettled_apify_hold_stays_held_at_full_bound_and_does_not_block_a_rebind():
    """2026-11-13: a never-settled Nov 2 request (cannot still be in flight) blocked the re-bind."""
    ledger,policy,obs=fixtures()
    ledger['reservations']['stale']=stale_apify_hold()
    obs['apify']['credential_fingerprint']='f'*16
    after=refresh(ledger,policy,obs)
    assert after['providers']['apify']['credential_fingerprint']=='f'*16
    assert after['reservations']['stale']==ledger['reservations']['stale']


@pytest.mark.parametrize('created',['recent','missing'])
def test_a_possibly_in_flight_apify_hold_still_blocks_a_rebind(created):
    ledger,policy,obs=fixtures()
    record=stale_apify_hold(created_at=(NOW-timedelta(seconds=299)).isoformat())
    if created=='missing': record.pop('created_at')
    ledger['reservations']['open']=record
    obs['apify']['credential_fingerprint']='f'*16
    with pytest.raises(SpendBlocked,match='open requests'): refresh(ledger,policy,obs)


@pytest.mark.parametrize('submitted_ago,blocked',[(310,True),(329,True),(331,False)])
def test_hold_submitted_late_is_still_in_flight_for_rebind(submitted_ago,blocked):
    """The window runs from the later of creation and submission, plus the start-request margin."""
    from src.zeitgeist import provider_lifecycle as lifecycle
    ledger,policy,obs=fixtures()
    duration=int(ledger['providers']['apify']['max_request_duration_seconds'])
    assert lifecycle.APIFY_START_REQUEST_SECONDS==30.0
    ledger['reservations']['late']=stale_apify_hold(
        created_at=(NOW-timedelta(days=11)).isoformat(),submitted_at=(NOW-timedelta(seconds=submitted_ago)).isoformat())
    assert duration==300
    obs['apify']['credential_fingerprint']='f'*16
    if blocked:
        with pytest.raises(SpendBlocked,match='open requests'): refresh(ledger,policy,obs)
    else:
        assert refresh(ledger,policy,obs)['providers']['apify']['credential_fingerprint']=='f'*16


def test_rebind_window_ignores_a_submission_time_older_than_creation_and_blocks_a_malformed_one():
    ledger,policy,obs=fixtures()
    obs['apify']['credential_fingerprint']='f'*16
    ledger['reservations']['open']=stale_apify_hold(
        created_at=(NOW-timedelta(seconds=299)).isoformat(),submitted_at=(NOW-timedelta(days=11)).isoformat())
    with pytest.raises(SpendBlocked,match='open requests'): refresh(ledger,policy,obs)
    ledger['reservations']['open']=stale_apify_hold(submitted_at='not a time')
    with pytest.raises(SpendBlocked): refresh(ledger,policy,obs)


# Gemini price check: compare the rates in effect, not the hash of the whole pricing page.
PRICE_BLOCK='Primary price snapshot changed, stale or missing'


def contract_for(*, through='2026-12-31', rates=('0.30','2.50'), op=OP, **extra):
    operation={'input_token_usd_per_million':rates[0],'output_token_usd_per_million':rates[1],
               'published_current_rate_through_date':through}
    operation.update(extra)
    return {'schema_version':2,'provider':'gemini','api':'gemini_developer_api','currency':'USD',
            'service_tier':'paid_standard','unit':'1000000_tokens','operations':{op:operation}}


def observe_contract(obs, contract, **overrides):
    snapshot=obs['price_snapshots']['gemini']
    snapshot.update(sha256=auto.digest(contract),contract=contract)
    snapshot.update(overrides)


def test_future_rate_announcement_passes_and_is_recorded():
    ledger,policy,obs=fixtures()
    contract=contract_for(future_input_token_usd_per_million='0.50',future_output_token_usd_per_million='4.00',
                          future_rate_effective_date='2027-01-01')
    observe_contract(obs,contract)
    assert obs['price_snapshots']['gemini']['sha256']!=policy['providers']['gemini']['price_snapshot_sha256']
    account=refresh(ledger,policy,obs)['providers']['gemini']
    assert account['price_observed_sha256']==auto.digest(contract)
    assert account['price_observed_at']==NOW.isoformat()


def test_unchanged_pinned_hash_does_not_record_a_rate_observation():
    ledger,policy,obs=fixtures()
    account=refresh(ledger,policy,obs)['providers']['gemini']
    assert 'price_observed_sha256' not in account and 'price_observed_at' not in account


def test_lower_current_rate_passes():
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=('0.25','2.00')))
    assert refresh(ledger,policy,obs)['providers']['gemini']['price_observed_at']==NOW.isoformat()


def test_generation_operation_without_published_output_rate_blocks():
    ledger,policy,obs=fixtures()
    contract=contract_for()
    contract['operations'][OP].pop('output_token_usd_per_million')
    observe_contract(obs,contract)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


@pytest.mark.parametrize('rates',[('0.34','2.50'),('0.30','2.76')])
def test_higher_current_rate_blocks(rates):
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=rates))
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_renamed_operation_blocks():
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(op='generate_content:gemini-2.5-flash-renamed'))
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_rate_through_date_before_validity_blocks():
    ledger,policy,obs=fixtures()
    ledger['providers']['gemini']['request_limits'][OP]['pricing_valid_until']='2026-12-31T00:00:00Z'
    policy['providers']['gemini']['account']['request_limits'][OP]['pricing_valid_until']='2026-12-31T00:00:00Z'
    ledger['automatic_policy_sha256']=auto.digest(policy)
    observe_contract(obs,contract_for(through='2026-12-30'))
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)
    observe_contract(obs,contract_for(through='2026-12-31'))
    refresh(ledger,policy,obs)


def test_contract_digest_mismatch_blocks():
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(),sha256='f'*64)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


@pytest.mark.parametrize('change',['source_url','stale'])
def test_rate_comparison_still_requires_the_pinned_source_and_fresh_observation(change):
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for())
    if change=='source_url': obs['price_snapshots']['gemini']['source_url']='https://ai.google.dev/other'
    if change=='stale': obs['price_snapshots']['gemini']['observed_at']=(NOW-timedelta(hours=25)).isoformat()
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_bfl_tariff_change_still_blocks():
    ledger,policy,obs=fixtures()
    contract=contract_for()
    obs['price_snapshots']['bfl'].update(sha256=auto.digest(contract),contract=contract)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


# BFL: a changed hash is compared on the per-image rate within the +15% band; wording is ignored.
BFL_OP='generate_image:flux-3-image'
BFL_TERMS=[
    'pricing is per image generated. charges are based on the resolution tier selected in the request, regardless of the resolution at which the image is processed internally, and apply equally to text-to-image and image-to-image generations.',
    'reference images, whether provided by the user or retrieved by the model, and prompts are included at no extra charge. the per-image rate does not vary with the number of reference images provided, up to the maximum set out in the documentation.',
    'the resolution tier is selected in the request; all supported aspect ratios within a tier are charged the same.',
]


def bfl_contract(price='0.07', **changes):
    contract={'schema_version':1,'provider':'bfl','operation':BFL_OP,'model':'FLUX 3 Image',
        'endpoint':'/v1/flux-3-image','resolution':'1.5k','currency':'USD','unit':'generated_image','quantity':1,
        'unit_price_usd':price,
        'request_shape':{'resolution':'1.5k','aspect_ratio':'3:2','grounding':True,'image_count':1,
                         'supplied_reference_images':0,'explicit_width_height':False},
        'pricing_flags':{'per_selected_resolution_tier':True,'all_supported_aspect_ratios_same_price':True,
                         'user_reference_images_included':True,'model_retrieved_reference_images_included':True},
        'pricing_terms':list(BFL_TERMS)}
    contract.update(changes)
    return contract


def bfl_fixtures(observed=None):
    ledger,policy,obs=fixtures()
    pinned_contract=bfl_contract()
    for target in (ledger['providers']['bfl'],policy['providers']['bfl']['account']):
        target['operation_caps_usd']={BFL_OP:'0.07'}
    policy['providers']['bfl']['price_snapshot_sha256']=auto.digest(pinned_contract)
    observed=pinned_contract if observed is None else observed
    obs['price_snapshots']['bfl'].update(sha256=auto.digest(observed),contract=observed)
    ledger['automatic_policy_sha256']=auto.digest(policy)
    return ledger,policy,obs


def bfl_account(price='0.07', **changes):
    return refresh(*bfl_fixtures(bfl_contract(price,**changes)))['providers']['bfl']


def test_bfl_unchanged_hash_passes_without_audit():
    acct=refresh(*bfl_fixtures())['providers']['bfl']
    assert 'price_band_audit' not in acct and 'price_observed_sha256' not in acct


def test_bfl_wording_change_at_unchanged_price_passes():
    c=bfl_contract(pricing_terms=BFL_TERMS+['an unrelated footnote about invoices.'])
    acct=bfl_account(pricing_terms=c['pricing_terms'])
    assert acct['price_observed_sha256']==auto.digest(c) and 'price_band_audit' not in acct


def test_bfl_lower_price_passes_on_rate_path():
    acct=bfl_account('0.06')
    assert acct['price_observed_at']==NOW.isoformat() and 'price_band_audit' not in acct


def test_bfl_price_exactly_fifteen_percent_above_passes_and_is_audited():
    acct=bfl_account('0.0805')
    assert acct['price_band_audit']['tolerance']=='0.15' and acct['price_band_audit']['above_approval'][0]['observed']=='0.0805'
    assert acct['price_band_audit']['above_approval']==[{'operation':BFL_OP,'rate':'unit_price_usd','approved':'0.07','observed':'0.0805'}]
    assert auto.verified_price_band_audit(acct,'bfl')


def test_bfl_price_just_above_fifteen_percent_blocks():
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): bfl_account('0.080507')


def test_bfl_one_credit_rise_passes():
    assert bfl_account('0.08')['price_band_audit']['above_approval'][0]['observed']=='0.08'


def test_bfl_two_credit_rise_blocks():
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): bfl_account('0.09')


@pytest.mark.parametrize('changes',[{'currency':'EUR'},{'unit':'megapixel'},{'quantity':2},{'resolution':'2k'},
    {'endpoint':'/v1/flux-pro'},{'operation':'generate_image:other'},{'schema_version':2}])
def test_bfl_contract_identity_change_blocks(changes):
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): bfl_account('0.075',**changes)


@pytest.mark.parametrize('case',['flag_false','image_count','clause_removed'])
def test_bfl_terms_change_blocks(case):
    c=bfl_contract('0.075')
    if case=='flag_false': c['pricing_flags']['user_reference_images_included']=False
    if case=='image_count': c['request_shape']['image_count']=2
    if case=='clause_removed': c['pricing_terms']=[t for t in BFL_TERMS if 'at no extra charge' not in t]
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(*bfl_fixtures(c))


@pytest.mark.parametrize('damage',['none','digest','source_url','stale'])
def test_bfl_rate_path_requires_digest_source_and_freshness(damage):
    ledger,policy,obs=bfl_fixtures(bfl_contract('0.075'))
    snapshot=obs['price_snapshots']['bfl']
    if damage=='digest': snapshot['sha256']='f'*64
    if damage=='source_url': snapshot['source_url']='https://bfl.ai/other'
    if damage=='stale': snapshot['observed_at']=(NOW-timedelta(hours=25)).isoformat()
    if damage=='none':
        assert refresh(ledger,policy,obs)['providers']['bfl']['price_band_audit']['above_approval'][0]['observed']=='0.075'
    else:
        with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_bfl_band_never_compounds():
    ledger,policy,obs=bfl_fixtures(bfl_contract('0.0798'))
    a=refresh(ledger,policy,obs)
    b=refresh(a,policy,obs)
    assert b['providers']['bfl']['price_band_audit']==a['providers']['bfl']['price_band_audit']
    obs2=bfl_fixtures(bfl_contract('0.090972'))[2]
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(b,policy,obs2)


@pytest.mark.parametrize('price',['missing','NaN','-0.07',''])
def test_bfl_missing_or_non_numeric_price_blocks(price):
    c=bfl_contract(price)
    if price=='missing': c.pop('unit_price_usd')
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(*bfl_fixtures(c))


def test_bfl_second_budgeted_operation_blocks():
    ledger,policy,obs=bfl_fixtures(bfl_contract(pricing_terms=BFL_TERMS+['a wording change.']))
    for target in (ledger['providers']['bfl'],policy['providers']['bfl']['account']):
        target['operation_caps_usd']['generate_image:other']='0.07'
    ledger['automatic_policy_sha256']=auto.digest(policy)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_bfl_constants_match_the_normalizer():
    from src.zeitgeist.bfl_tariff import normalize_bfl_flux3_tariff
    from tests.test_bfl_tariff import html as bfl_page
    c=normalize_bfl_flux3_tariff(bfl_page())
    assert all(c[k]==v for k,v in auto.BFL_CONTRACT_CONSTANTS.items())
    assert c['request_shape']==auto.BFL_REVIEWED_SHAPE and c['pricing_flags']==auto.BFL_REVIEWED_FLAGS
    assert all(x in ' '.join(c['pricing_terms']) for x in auto.BFL_REQUIRED_CLAUSES)
    assert c==bfl_contract()


EMBED='embed_content:gemini-embedding-001'


def add_caps_only_embedding(ledger,policy,obs):
    """An approved, budgeted operation that carries no approved rate at all."""
    ledger['providers']['gemini']['operation_caps_usd'][EMBED]='.05'
    policy['providers']['gemini']['account']['operation_caps_usd'][EMBED]='.05'
    ledger['automatic_policy_sha256']=auto.digest(policy)
    obs['gemini']['available_required_models']['gemini-embedding-001']=['embedContent']


def test_caps_only_operation_without_approved_rate_blocks():
    ledger,policy,obs=fixtures();add_caps_only_embedding(ledger,policy,obs)
    contract=contract_for()
    contract['operations'][EMBED]={'input_token_usd_per_million':'9.99','billing_basis':'input_tokens'}
    observe_contract(obs,contract)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


@pytest.mark.parametrize('approved',[{'input_token_usd_per_million':'0.15'},
                                     {'input_token_usd_per_million':'0.15','output_token_usd_per_million':'0'}])
def test_input_only_operation_passes_with_input_rate_and_absent_or_zero_output(approved):
    ledger,policy,obs=fixtures();add_caps_only_embedding(ledger,policy,obs)
    for target in (ledger['providers']['gemini'],policy['providers']['gemini']['account']):
        target['request_limits'][EMBED]=dict(approved)
    ledger['automatic_policy_sha256']=auto.digest(policy)
    contract=contract_for()
    contract['operations'][EMBED]={'input_token_usd_per_million':'0.15','billing_basis':'input_tokens'}
    observe_contract(obs,contract)
    refresh(ledger,policy,obs)


def test_input_only_operation_with_nonzero_approved_output_or_without_input_blocks():
    for approved in ({'input_token_usd_per_million':'0.15','output_token_usd_per_million':'1'},
                     {'output_token_usd_per_million':'0'}):
        ledger,policy,obs=fixtures();add_caps_only_embedding(ledger,policy,obs)
        for target in (ledger['providers']['gemini'],policy['providers']['gemini']['account']):
            target['request_limits'][EMBED]=dict(approved)
        ledger['automatic_policy_sha256']=auto.digest(policy)
        contract=contract_for()
        contract['operations'][EMBED]={'input_token_usd_per_million':'0.15','billing_basis':'input_tokens'}
        observe_contract(obs,contract)
        with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_generation_operation_without_approved_output_rate_blocks():
    ledger,policy,obs=fixtures()
    for target in (ledger['providers']['gemini'],policy['providers']['gemini']['account']):
        target['request_limits'][OP].pop('output_token_usd_per_million')
    ledger['automatic_policy_sha256']=auto.digest(policy)
    observe_contract(obs,contract_for())
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


@pytest.mark.parametrize('field,value',[('unit','1000_tokens'),('unit',None),('currency','EUR'),
    ('service_tier','paid_priority'),('api','vertex'),('schema_version',3),('provider','other')])
def test_contract_unit_or_currency_change_blocks(field,value):
    ledger,policy,obs=fixtures()
    contract=contract_for()
    if value is None: contract.pop(field)
    else: contract[field]=value
    observe_contract(obs,contract)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_hash_match_clears_stale_rate_audit_fields():
    ledger,policy,obs=fixtures()
    ledger['providers']['gemini'].update(price_observed_sha256='f'*64,price_observed_at='2026-11-01T00:00:00+00:00',
                                         price_band_audit={'schema_version':1})
    account=refresh(ledger,policy,obs)['providers']['gemini']
    assert 'price_observed_sha256' not in account and 'price_observed_at' not in account
    assert 'price_band_audit' not in account


# Price tolerance band: a current rate up to +10% of the approved rate passes and is audited privately.
def band_audit(ledger,policy,obs,rates,**extra):
    observe_contract(obs,contract_for(rates=rates,**extra))
    return refresh(ledger,policy,obs)['providers']['gemini']['price_band_audit']


def test_price_tolerance_is_the_owner_approved_per_provider_band():
    assert auto.PRICE_TOLERANCE=={'gemini':Decimal('0.10'),'bfl':Decimal('0.15')}


@pytest.mark.parametrize('rates,raised',[
    (('0.33','2.50'),{('input_token_usd_per_million','0.33')}),
    (('0.30','2.75'),{('output_token_usd_per_million','2.75')}),
    (('0.33','2.75'),{('input_token_usd_per_million','0.33'),('output_token_usd_per_million','2.75')})])
def test_rate_exactly_ten_percent_above_approval_passes(rates,raised):
    ledger,policy,obs=fixtures()
    audit=band_audit(ledger,policy,obs,rates)
    assert {(e['rate'],e['observed']) for e in audit['above_approval']}==raised


@pytest.mark.parametrize('rates',[('0.33003','2.50'),('0.30','2.75025')])
def test_rate_just_above_ten_percent_blocks(rates):
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=rates))
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_small_rise_within_band_passes():
    ledger,policy,obs=fixtures()
    assert band_audit(ledger,policy,obs,('0.31','2.50'))['tolerance']=='0.10'


def test_band_is_measured_from_the_approved_rate_never_the_last_observation():
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=('0.327','2.725')))
    first=refresh(ledger,policy,obs)
    obs2=copy.deepcopy(obs)
    observe_contract(obs2,contract_for(rates=('0.327','2.725'),future_rate_effective_date='2027-01-01'))
    second=refresh(first,policy,obs2)
    assert (second['providers']['gemini']['price_band_audit']['above_approval']
            ==first['providers']['gemini']['price_band_audit']['above_approval'])
    obs3=copy.deepcopy(obs)
    observe_contract(obs3,contract_for(rates=('0.35643','2.97025')))
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(second,policy,obs3)


@pytest.mark.parametrize('case',['missing_input_rate','missing_output_rate','renamed_operation'])
def test_in_band_values_never_excuse_a_missing_rate_or_operation(case):
    ledger,policy,obs=fixtures()
    contract=contract_for(rates=('0.315','2.625'),op=OP+'-renamed' if case=='renamed_operation' else OP)
    if case=='missing_input_rate': contract['operations'][OP].pop('input_token_usd_per_million')
    if case=='missing_output_rate': contract['operations'][OP].pop('output_token_usd_per_million')
    observe_contract(obs,contract)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_zero_approved_output_rate_has_no_band():
    ledger,policy,obs=fixtures()
    for target in (ledger['providers']['gemini'],policy['providers']['gemini']['account']):
        target['request_limits'][OP]['output_token_usd_per_million']='0'
    ledger['automatic_policy_sha256']=auto.digest(policy)
    observe_contract(obs,contract_for(rates=('0.30','0.000001'),billing_basis='input_tokens'))
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_2027_rates_never_pass_under_the_2026_approval():
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=('0.60','5.00')))
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


def test_in_band_pass_records_private_audit():
    ledger,policy,obs=fixtures()
    contract=contract_for(rates=('0.33','2.50'))
    observe_contract(obs,contract)
    audit=refresh(ledger,policy,obs)['providers']['gemini']['price_band_audit']
    assert audit=={'schema_version':1,'tolerance':'0.10','basis':'policy_approved_rate',
        'observed_sha256':auto.digest(contract),'observed_at':NOW.isoformat(),
        'above_approval':[{'operation':OP,'rate':'input_token_usd_per_million','approved':'0.3','observed':'0.33'}]}


def test_verified_price_band_audit_rejects_out_of_band_or_malformed():
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=('0.33','2.50')))
    account=refresh(ledger,policy,obs)['providers']['gemini']
    assert auto.verified_price_band_audit(account,'gemini')
    bad=copy.deepcopy(account);bad['price_band_audit']['above_approval'][0]['observed']='0.34'
    assert auto.verified_price_band_audit(bad,'gemini') is False
    bad=copy.deepcopy(account);bad['price_band_audit'].pop('basis')
    assert auto.verified_price_band_audit(bad,'gemini') is False


def test_rate_at_or_below_approval_clears_band_audit():
    ledger,policy,obs=fixtures()
    ledger['providers']['gemini']['price_band_audit']={'schema_version':1}
    observe_contract(obs,contract_for(rates=('0.30','2.50'),future_rate_effective_date='2027-01-01'))
    account=refresh(ledger,policy,obs)['providers']['gemini']
    assert 'price_band_audit' not in account


# Holds and receipts at the charged rate: max(approved, verified in-band observed).
OP38='generate_content:gemini-3.8-flash'


def acct_38(**observed):
    account={'request_limits':{OP38:{'input_token_usd_per_million':'0.75','output_token_usd_per_million':'3.75'}}}
    if observed:
        account['price_band_audit']={'schema_version':1,'tolerance':'0.10','basis':'policy_approved_rate',
            'observed_sha256':'f'*64,'observed_at':NOW.isoformat(),
            'above_approval':[{'operation':OP38,'rate':key,'approved':account['request_limits'][OP38][key],'observed':rate}
                              for key,rate in sorted(observed.items())]}
        account.update(price_observed_sha256='f'*64,price_observed_at=NOW.isoformat())
    return account


def test_charged_rate_is_the_in_band_observed_rate_or_the_approved_rate():
    acct_in_band=acct_38(input_token_usd_per_million='0.825')
    assert auto.charged_rate(acct_in_band,'gemini',OP38,'input_token_usd_per_million')==Decimal('0.825')
    assert auto.charged_rate(acct_in_band,'gemini',OP38,'output_token_usd_per_million')==Decimal('3.75')
    assert auto.charged_rate(acct_38(),'gemini',OP38,'input_token_usd_per_million')==Decimal('0.75')
    outside=acct_38(input_token_usd_per_million='0.9')
    with pytest.raises(SpendBlocked,match='Primary price snapshot'):
        auto.charged_rate(outside,'gemini',OP38,'input_token_usd_per_million')


@pytest.mark.parametrize('rate',['not-a-number',None,'NaN','Infinity',''])
def test_non_numeric_or_missing_current_rate_blocks(rate):
    ledger,policy,obs=fixtures()
    contract=contract_for()
    if rate is None: contract['operations'][OP].pop('input_token_usd_per_million')
    else: contract['operations'][OP]['input_token_usd_per_million']=rate
    observe_contract(obs,contract)
    with pytest.raises(SpendBlocked,match=PRICE_BLOCK): refresh(ledger,policy,obs)


# Month handoff with tolerated provider overruns (P5, I2). The fixture lives with the overrun tests.
HANDOFF_NOW=datetime(2026,11,1,0,1,tzinfo=timezone.utc)


def tolerated_handoff(prior):
    from tests.test_usage_receipt_overrun import tolerance_fixture
    ledger,policy,obs=tolerance_fixture(prior=prior,now=HANDOFF_NOW)
    plan=auto.prepare_month_handoff(ledger,policy,obs,expected_policy_sha=auto.digest(policy),
        source_identity=dict(auto.TARGET,document_id='2026-10',update_time=HANDOFF_NOW.isoformat()),now=HANDOFF_NOW)
    return ledger,policy,plan


def test_handoff_succeeds_with_tolerated_records_and_frozen_copy_passes():
    ledger,policy,plan=tolerated_handoff(['1.00','2.00'])
    assert plan['source_after']['closed_for_rollover'] is True
    assert auto.cash_exposure(plan['source_after'])<=Decimal(ledger['cash_ceiling_usd'])
    assert auto.month_spend_increase_usd(plan['source_after'])==Decimal('3.00')


def test_new_month_starts_with_the_full_increase_allowance():
    ledger,policy,plan=tolerated_handoff(['1.00','2.00'])
    destination=plan['destination_after']
    assert set(destination['reservations'])=={'open-hold'}
    assert auto.month_spend_increase_usd(destination)==Decimal('0')


def test_carried_hold_overrun_counts_against_the_new_month():
    from tests.test_usage_receipt_overrun import open_receipt, admit
    ledger,policy,plan=tolerated_handoff(['1.00','2.00'])
    receipt=open_receipt('0.08',observed_at=HANDOFF_NOW)
    c=admit(plan['destination_after'],policy,receipt,HANDOFF_NOW)
    assert c['reservations']['open-hold']['overrun_tolerated']['month_increase_usd']=='0.01'
    assert not c.get('reconciliation_required')


def test_handoff_baseline_drops_by_the_actual_amount_including_overrun():
    ledger,policy,plan=tolerated_handoff(['0.01'])
    assert ledger['reservations']['prior-0']['actual_gross_usd']=='0.08'
    src_baseline=ledger['providers']['bfl']['credit_baseline_usd']
    dest=plan['destination_after']
    assert Decimal(dest['providers']['bfl']['credit_baseline_usd'])==Decimal(src_baseline)-Decimal('0.08')


# Private owner-notice hook (5b): a notice carries only what the ledger carries; nothing calls it yet.
def notice_ledger(rates=('0.33','2.50'),with_overrun=True):
    from tests.test_usage_receipt_overrun import tolerated_record
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=rates))
    ledger=refresh(ledger,policy,obs)
    if with_overrun: ledger['reservations']['prior-0']=tolerated_record(0,'0.01')
    return ledger,policy,obs


def test_owner_spend_notices_lists_band_and_overrun_kinds():
    ledger,policy,obs=notice_ledger()
    n=auto.owner_spend_notices(ledger)
    rid='prior-0'
    assert [x['kind'] for x in n]==['price_band','tolerated_overrun'] and n[0]['provider']=='gemini' and n[1]['reservation_id']==rid
    audit=ledger['providers']['gemini']['price_band_audit']
    mark=ledger['reservations'][rid]['overrun_tolerated']
    assert n[0]=={'kind':'price_band','notice_id':auto.digest({'kind':'price_band','provider':'gemini',
        'observed_sha256':audit['observed_sha256'],'above_approval':audit['above_approval']}),'provider':'gemini',**audit}
    assert n[1]=={'kind':'tolerated_overrun','notice_id':auto.digest({'kind':'tolerated_overrun','reservation_id':rid,
        'billing_reference':ledger['reservations'][rid]['billing_reference']}),'reservation_id':rid,**mark}


def test_owner_spend_notices_is_pure_and_empty_without_audit_or_mark():
    ledger,policy,obs=fixtures()
    assert auto.owner_spend_notices(ledger)==[]
    ledger,policy,obs=notice_ledger()
    before=copy.deepcopy(ledger)
    n=auto.owner_spend_notices(ledger)
    n[0]['above_approval'].clear();n[1]['overrun_usd']='99'
    assert ledger==before


def test_notice_ids_are_stable_across_refreshes():
    ledger,policy,obs=notice_ledger()
    ids=lambda l:[x['notice_id'] for x in auto.owner_spend_notices(l)]
    assert ids(refresh(ledger,policy,obs))==ids(ledger)
    changed=copy.deepcopy(obs)
    observe_contract(changed,contract_for(rates=('0.32','2.50')))
    assert ids(refresh(ledger,policy,changed))!=ids(ledger)


# The audit is proof only for the refresh that wrote it (F5).
def band_account():
    ledger,policy,obs=fixtures()
    observe_contract(obs,contract_for(rates=('0.33','2.50')))
    account=refresh(ledger,policy,obs)['providers']['gemini']
    assert auto.verified_price_band_audit(account,'gemini')
    return account


def test_band_audit_rejects_a_gemini_rate_key_that_is_not_a_token_rate():
    account=band_account()
    account['request_limits'][OP]['max_input_tokens']=1000
    account['price_band_audit']['above_approval'].append(
        {'operation':OP,'rate':'max_input_tokens','approved':'1000','observed':'1050'})
    assert auto.verified_price_band_audit(account,'gemini') is False
    with pytest.raises(KeyError):
        auto.approved_rate(account,'gemini',OP,'max_input_tokens')


@pytest.mark.parametrize('version',[True,1.0,'1',2])
def test_band_audit_schema_version_must_be_the_integer_1(version):
    account=band_account();account['price_band_audit']['schema_version']=version
    assert auto.verified_price_band_audit(account,'gemini') is False


@pytest.mark.parametrize('field,value',[
    ('price_observed_sha256','0'*64),('price_observed_at','2026-10-30T15:00:00+00:00'),
    ('price_observed_sha256',None),('price_observed_at',None)])
def test_band_audit_must_match_the_observation_it_records(field,value):
    account=band_account()
    if value is None: account.pop(field)
    else: account[field]=value
    assert auto.verified_price_band_audit(account,'gemini') is False


def test_band_audit_rejects_a_duplicate_operation_and_rate():
    account=band_account()
    entries=account['price_band_audit']['above_approval']
    entries.append(dict(entries[0],observed='0.32'))
    assert auto.verified_price_band_audit(account,'gemini') is False
    entries[-1]=dict(entries[0])
    assert auto.verified_price_band_audit(account,'gemini') is False
