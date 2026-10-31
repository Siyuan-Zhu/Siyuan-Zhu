"""Offline automatic policy planning never refreshes manual facts or loses holds."""
import copy
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
