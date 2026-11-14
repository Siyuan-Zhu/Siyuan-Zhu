"""Authenticated excess liability must persist before runtime stops paid work."""
import copy
from datetime import timedelta
import pytest
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import MemoryLedger, SpendBlocked
from tests.test_automatic_spend import request_fixture, NOW


def admit(ledger,policy,receipt,now=NOW):
    return auto.admit_usage_receipt(ledger,policy,receipt,
        expected_policy_sha=auto.digest(policy),now=now)


def test_fully_bound_overrun_returns_exact_higher_liability_and_evidence_without_mutating_source():
    ledger,policy,_,receipt=request_fixture()
    receipt['accounted_gross_usd']='0.25'
    before=copy.deepcopy(ledger)
    candidate=admit(ledger,policy,receipt)
    record=candidate['reservations']['reservation-1']
    assert ledger==before
    assert record['max_gross_usd']==before['reservations']['reservation-1']['max_gross_usd']
    assert record['actual_gross_usd']=='0.25'
    assert record['usage_receipt']==receipt
    assert record['billing_reference'].startswith('provider_usage_v1:')
    assert record['status']=='settled'
    assert candidate['reconciliation_required'] is True
    assert record['reconciliation_reasons']==['usage_receipt_overrun']
    assert candidate['providers']==before['providers']
    with pytest.raises(SpendBlocked,match='overrun'):auto.cash_exposure(candidate)


@pytest.mark.parametrize('reobserved', ['0.00040','0.00070','0.25'])
def test_conflicting_bound_receipts_preserve_larger_liability_and_both_private_receipts(reobserved):
    ledger,policy,_,first=request_fixture()
    prior=admit(ledger,policy,first)
    second=copy.deepcopy(first)
    second.update(accounted_gross_usd=reobserved,evidence_sha256='e'*64,
        observed_at=(NOW+timedelta(seconds=30)).isoformat())
    candidate=admit(prior,policy,second,NOW+timedelta(seconds=30))
    record=candidate['reservations']['reservation-1']
    expected=max(auto.money(first['accounted_gross_usd']),auto.money(reobserved))
    assert auto.money(record['actual_gross_usd'])==expected
    assert candidate['reconciliation_required'] is True
    assert 'usage_receipt_conflict' in record['reconciliation_reasons']
    assert ('usage_receipt_overrun' in record['reconciliation_reasons']) is (expected>auto.money(record['max_gross_usd']))
    evidence=record['usage_receipt_conflict']
    assert evidence['first_receipt']==first and evidence['latest_receipt']==second
    highest=second if auto.money(reobserved)>auto.money(first['accounted_gross_usd']) else first
    assert record['usage_receipt']==highest
    assert prior['reservations']['reservation-1']['usage_receipt']==first
    assert not prior.get('reconciliation_required')


def test_conflicting_same_amount_with_different_evidence_is_frozen_without_releasing_more_credit():
    ledger,policy,_,first=request_fixture()
    prior=admit(ledger,policy,first)
    second=dict(first,evidence_sha256='e'*64)
    candidate=admit(prior,policy,second)
    record=candidate['reservations']['reservation-1']
    assert record['actual_gross_usd']==first['accounted_gross_usd']
    assert record['usage_receipt']==first
    assert candidate['reconciliation_required'] is True


@pytest.mark.parametrize('change',['request','credential','operation','id','timestamp','missing_cost','bad_cost','missing_evidence','duplicate'])
def test_untrusted_overrun_evidence_never_changes_or_freezes_an_unrelated_hold(change):
    ledger,policy,_,receipt=request_fixture();receipt['accounted_gross_usd']='0.25'
    if change=='request':receipt['request_sha256']='e'*64
    if change=='credential':receipt['credential_fingerprint']='e'*16
    if change=='operation':receipt['operation']='unpriced:other'
    if change=='id':receipt['provider_request_id']='unowned'
    if change=='timestamp':receipt['observed_at']=(NOW+timedelta(seconds=1)).isoformat()
    if change=='missing_cost':receipt.pop('accounted_gross_usd')
    if change=='bad_cost':receipt['accounted_gross_usd']='NaN'
    if change=='missing_evidence':receipt.pop('evidence_sha256')
    if change=='duplicate':ledger['reservations']['duplicate']=copy.deepcopy(ledger['reservations']['reservation-1'])
    before=copy.deepcopy(ledger)
    with pytest.raises(SpendBlocked):admit(ledger,policy,receipt)
    assert ledger==before


def test_persist_then_raise_transaction_contract_retains_overrun_on_the_real_memory_ledger():
    ledger,policy,_,receipt=request_fixture();receipt['accounted_gross_usd']='0.25'
    memory=MemoryLedger(ledger)
    def transaction(document):
        candidate=admit(document,policy,receipt)
        document.clear();document.update(candidate)
        return candidate['reconciliation_required']
    # The runtime deliberately raises AFTER mutate commits its returned candidate.
    with pytest.raises(SpendBlocked):
        frozen=memory.mutate(transaction)
        if frozen:raise SpendBlocked('Billing reconciliation required')
    assert memory.document['reconciliation_required'] is True
    assert memory.document['reservations']['reservation-1']['actual_gross_usd']=='0.25'


def test_reobserved_identical_valid_receipt_remains_idempotent_and_unfrozen():
    ledger,policy,_,receipt=request_fixture()
    first=admit(ledger,policy,receipt)
    assert admit(first,policy,receipt)==first
    assert not first.get('reconciliation_required')


def test_later_smaller_conflicts_cannot_overwrite_highest_liability_or_first_evidence():
    ledger,policy,_,first=request_fixture()
    candidate=admit(ledger,policy,first)
    high=dict(first,accounted_gross_usd='0.25',evidence_sha256='e'*64)
    candidate=admit(candidate,policy,high)
    smaller=dict(first,accounted_gross_usd='0.01',evidence_sha256='d'*64)
    final=admit(candidate,policy,smaller)
    record=final['reservations']['reservation-1']
    assert record['actual_gross_usd']=='0.25' and record['usage_receipt']==high
    assert record['usage_receipt_conflict']['first_receipt']==first
    assert record['usage_receipt_conflict']['latest_receipt']==smaller
    assert record['reconciliation_reasons']==['usage_receipt_conflict','usage_receipt_overrun']


# --- Small provider overruns settle while the month's spend increase stays at or below 5 USD (P5) ---
from decimal import Decimal
from src.zeitgeist.automatic_guard import AutomaticBudgetGuard, ProviderRequest
from tests.test_automatic_spend import fixtures, with_approved_reload

OPEN = 'open-hold'
BFL_REQUEST = 'f' * 64


def settled_cost_receipt(reservation_id, amount, task, observed_at=NOW, **changes):
    """A provider-settled-cost receipt as provider_lifecycle.usage_receipt emits it."""
    receipt = {'schema_version': 1, 'accounting_policy': 'provider_usage_v1', 'reservation_id': reservation_id,
        'provider': 'bfl', 'operation': 'image', 'credential_fingerprint': 'b' * 16,
        'provider_request_id': task, 'request_sha256': BFL_REQUEST, 'observed_at': observed_at.isoformat(),
        'accounted_gross_usd': str(amount), 'accounting_basis': 'provider_settled_cost', 'evidence_sha256': 'd' * 64}
    receipt.update(changes)
    return receipt


def tolerated_record(index, overrun, max_gross='0.07'):
    """A valid settled BFL record whose overrun was tolerated: max_gross + overrun_usd == actual."""
    held, over = Decimal(max_gross), Decimal(overrun)
    actual = held + over
    receipt = settled_cost_receipt('prior-%d' % index, actual, 'task-prior-%d' % index)
    return {'provider': 'bfl', 'operation': 'image', 'max_gross_usd': str(held), 'actual_gross_usd': str(actual),
        'status': 'settled', 'created_at': NOW.isoformat(), 'credential_fingerprint': 'b' * 16,
        'provider_request_id': 'task-prior-%d' % index, 'request_sha256': BFL_REQUEST,
        'billing_reference': 'provider_usage_v1:' + 'c' * 64,
        'usage_receipt': receipt,
        'overrun_tolerated': {'schema_version': 1, 'held_usd': str(held), 'actual_usd': str(actual),
            'overrun_usd': str(over), 'month_increase_usd': str(over), 'limit_usd': '5',
            'basis': 'overruns_only', 'observed_at': NOW.isoformat()}}


def tolerance_fixture(prior='0', ceiling='10.00', now=NOW):
    """A reload-policy ledger (ceiling 10) with prior tolerated overruns and one open BFL hold of 0.07.

    prior is one amount or a list of amounts, one tolerated record each; their overruns total
    the sum. Any other ceiling gives the plain (non-reload) policy at that ceiling. The ledger is
    ready for AutomaticBudgetGuard without a refresh. Returns (ledger, policy, observation).
    """
    ledger, policy, observation = fixtures(now)
    if ceiling == '10.00':
        policy = with_approved_reload(policy)
        ledger.update(credit_only_providers=['apify', 'bfl'])
    else:
        policy['cash_ceiling_usd'] = ceiling
    ledger.update(cash_ceiling_usd=ceiling, automatic_policy_sha256=auto.digest(policy),
        credential_binding_required=True, budget_scope='new_requests_after_cutoff')
    for account in ledger['providers'].values():
        account.update(automatic_observed_at=now.isoformat(), automatic_observation_sha256='a' * 64)
    amounts = [a for a in (prior if isinstance(prior, (list, tuple)) else [prior]) if Decimal(a) > 0]
    for index, amount in enumerate(amounts):
        ledger['reservations']['prior-%d' % index] = tolerated_record(index, amount)
    ledger['reservations'][OPEN] = {'provider': 'bfl', 'operation': 'image', 'max_gross_usd': '0.07',
        'status': 'reserved', 'created_at': NOW.isoformat(), 'submitted_at': NOW.isoformat(),
        'credential_fingerprint': 'b' * 16, 'provider_request_id': 'task-open', 'request_sha256': BFL_REQUEST,
        'automatic_policy_sha256': auto.digest(policy)}
    return ledger, policy, observation


def open_receipt(amount, **changes):
    return settled_cost_receipt(OPEN, amount, 'task-open', **changes)


def guard_for(ledger, policy, now=NOW):
    return AutomaticBudgetGuard(MemoryLedger(ledger), policy, auto.digest(policy), lambda: now,
        credential_fingerprints={'gemini': 'a' * 16, 'bfl': 'b' * 16, 'apify': 'c' * 16},
        require_credential_binding=True)


def in_band_hold(excess, **changes):
    """A Gemini hold whose recorded in-band excess is exactly `excess` (input tokens at +0.10 per million)."""
    tokens = int(Decimal(excess) * 10)  # tokens in millions x 0.10 USD per million
    record = {'provider': 'gemini', 'operation': 'generate_content:gemini-2.5-flash', 'max_gross_usd': '0.01',
        'status': 'reserved', 'created_at': NOW.isoformat(),
        'request_context': {'admitted_input_tokens': tokens * 1_000_000, 'max_output_tokens': 0,
            'input_token_usd_per_million': '0.85', 'output_token_usd_per_million': '3.75',
            'approved_input_token_usd_per_million': '0.75', 'approved_output_token_usd_per_million': '3.75'}}
    record.update(changes)
    return record


def mark_of(candidate, key=OPEN):
    return candidate['reservations'][key].get('overrun_tolerated')


def test_monthly_increase_constants():
    assert auto.TOLERATED_MONTHLY_INCREASE_USD == Decimal('5') and auto.MONTHLY_INCREASE_BASIS == 'overruns_only'


def test_spend_increase_of_4_99_settles_and_continues():
    ledger, policy, observation = tolerance_fixture(prior='4.00')
    c = admit(ledger, policy, open_receipt('1.06'))
    assert c['reservations'][OPEN]['overrun_tolerated']['month_increase_usd'] == '4.99'
    assert not c.get('reconciliation_required') and not c['reservations'][OPEN].get('reconciliation_required')
    after = auto.prepare_refresh(c, policy, observation, expected_policy_sha=auto.digest(policy), now=NOW)['after']
    assert auto.cash_exposure(after) == 0
    guard = guard_for(c, policy)
    guard.reserve('bfl', 'image', '0.07')
    assert len(guard.ledger.document['reservations']) == 3  # the prior record, the settled hold, the new hold


def test_spend_increase_of_exactly_5_settles():
    ledger, policy, _ = tolerance_fixture(prior='4.00')
    c = admit(ledger, policy, open_receipt('1.07'))
    assert c['reservations'][OPEN]['overrun_tolerated']['month_increase_usd'] == '5.00'
    assert not c.get('reconciliation_required')


def test_spend_increase_of_5_01_freezes():
    ledger, policy, _ = tolerance_fixture(prior='4.00')
    c = admit(ledger, policy, open_receipt('1.08'))
    record = c['reservations'][OPEN]
    assert c['reconciliation_required'] is True and 'overrun_tolerated' not in record
    assert record['reconciliation_reasons'] == ['usage_receipt_overrun']
    guard = guard_for(ledger, policy)
    with pytest.raises(SpendBlocked, match='Authenticated usage requires spending reconciliation'):
        ProviderRequest(guard, OPEN, 'bfl', 'image', BFL_REQUEST).receipt(open_receipt('1.08'))
    assert guard.ledger.document['reconciliation_required'] is True


def test_record_is_counted_before_it_is_marked(monkeypatch):
    seen, real = [], auto.month_spend_increase_usd
    def spy(ledger):
        seen.append(copy.deepcopy(ledger))
        return real(ledger)
    monkeypatch.setattr(auto, 'month_spend_increase_usd', spy)
    ledger, policy, _ = tolerance_fixture(prior='1.00')
    c = admit(ledger, policy, open_receipt('0.08'))
    first = seen[0]['reservations'][OPEN]
    assert first.get('overrun_tolerated') is None and first['status'] == 'settled' and first['actual_gross_usd'] == '0.08'
    assert mark_of(c)['month_increase_usd'] == '1.01'


def test_readmitting_the_same_receipt_is_idempotent_and_counts_the_record_once():
    ledger, policy, _ = tolerance_fixture(prior='1.00')
    first = admit(ledger, policy, open_receipt('0.08'))
    again = admit(first, policy, open_receipt('0.08'))
    assert again == first and mark_of(again)['month_increase_usd'] == '1.01'


def test_small_overrun_that_breaks_the_ceiling_freezes():
    ledger, policy, _ = tolerance_fixture(prior='0', ceiling='2.50')
    # The BFL credit covers only the 0.07 hold, so any overrun above it is cash.
    ledger['providers']['bfl'].update(credit_baseline_usd='0.07', remaining_credit_usd='0.07')
    ledger['reservations']['gemini-cash'] = in_band_hold('0', max_gross_usd='2.495', request_context={})
    assert auto.cash_exposure(ledger) == Decimal('2.495')
    # Control: 0.005 of cash reaches the ceiling exactly and is tolerated.
    ok = admit(ledger, policy, open_receipt('0.075'))
    assert not ok.get('reconciliation_required') and mark_of(ok)['overrun_usd'] == '0.005'
    c = admit(ledger, policy, open_receipt('0.08'))
    assert c['reconciliation_required'] is True and 'overrun_tolerated' not in c['reservations'][OPEN]
    assert c['reservations'][OPEN]['reconciliation_reasons'] == ['usage_receipt_overrun']


def test_small_overrun_that_breaks_the_bfl_credit_bound_freezes():
    ledger, policy, _ = tolerance_fixture(prior='0')
    ledger['providers']['bfl'].update(credit_baseline_usd='0.07', remaining_credit_usd='0.07')
    c = admit(ledger, policy, open_receipt('0.08'))
    assert c['reconciliation_required'] is True and 'overrun_tolerated' not in c['reservations'][OPEN]
    assert c['reservations'][OPEN]['reconciliation_reasons'] == ['usage_receipt_overrun']


def test_conservative_basis_overrun_still_freezes():
    ledger, policy, _ = tolerance_fixture(prior='0')
    c = admit(ledger, policy, open_receipt('0.08', accounting_basis='conservative_response_usage'))
    assert c['reconciliation_required'] is True and 'overrun_tolerated' not in c['reservations'][OPEN]
    assert c['reservations'][OPEN]['reconciliation_reasons'] == ['usage_receipt_overrun']


def test_conflicting_receipt_still_freezes_even_when_small():
    ledger, policy, _ = tolerance_fixture(prior='0')
    first = admit(ledger, policy, open_receipt('0.07'))
    second = open_receipt('0.08', evidence_sha256='e' * 64)
    c = admit(first, policy, second)
    record = c['reservations'][OPEN]
    assert c['reconciliation_required'] is True and 'usage_receipt_conflict' in record['reconciliation_reasons']
    assert 'overrun_tolerated' not in record


def tamper_overrun(ledger, change):
    record = ledger['reservations']['prior-0']
    if change == 'overrun_mismatch':
        record['overrun_tolerated']['overrun_usd'] = '3.99'
    if change == 'max_gross_inflated':
        record['max_gross_usd'] = '0.50'
    if change == 'sum_above_5':
        ledger['reservations']['prior-1'] = tolerated_record(1, '1.01')


@pytest.mark.parametrize('change', ['overrun_mismatch', 'max_gross_inflated', 'sum_above_5'])
def test_tampered_overrun_record_blocks_cash_exposure(change):
    ledger, policy, _ = tolerance_fixture(prior='4.00')
    assert auto.cash_exposure(ledger) == 0  # the untampered control passes
    tamper_overrun(ledger, change)
    with pytest.raises(SpendBlocked, match='Confirmed overrun requires reconciliation'):
        auto.cash_exposure(ledger)


@pytest.mark.parametrize('flag', ['reconciliation_required', 'reconciliation_reasons', 'usage_receipt_conflict'])
def test_flagged_tolerated_record_is_not_exempt(flag):
    ledger, policy, _ = tolerance_fixture(prior='1.00')
    ledger['reservations']['prior-0'][flag] = True if flag == 'reconciliation_required' else ['x'] if flag == 'reconciliation_reasons' else {}
    with pytest.raises(SpendBlocked, match='Confirmed overrun requires reconciliation'):
        auto.cash_exposure(ledger)


def test_tolerated_record_on_a_conservative_receipt_is_not_exempt():
    ledger, policy, _ = tolerance_fixture(prior='1.00')
    ledger['reservations']['prior-0']['usage_receipt']['accounting_basis'] = 'conservative_response_usage'
    with pytest.raises(SpendBlocked, match='Confirmed overrun requires reconciliation'):
        auto.cash_exposure(ledger)


def test_untolerated_overrun_still_blocks_cash_exposure():
    ledger, policy, _ = tolerance_fixture(prior='1.00')
    del ledger['reservations']['prior-0']['overrun_tolerated']
    with pytest.raises(SpendBlocked, match='Confirmed overrun requires reconciliation'):
        auto.cash_exposure(ledger)


def test_both_together_counts_in_band_excess(monkeypatch):
    ledger, policy, _ = tolerance_fixture(prior='0')
    ledger['reservations']['inband'] = in_band_hold('4.50')
    assert auto.in_band_excess_usd(ledger['reservations']['inband']) == Decimal('4.50')
    tolerated = admit(ledger, policy, open_receipt('0.67'))
    assert mark_of(tolerated)['month_increase_usd'] == '0.60' and not tolerated.get('reconciliation_required')
    monkeypatch.setattr(auto, 'MONTHLY_INCREASE_BASIS', 'both_together')
    assert auto.month_spend_increase_usd(ledger) == Decimal('4.50')
    c = admit(ledger, policy, open_receipt('0.67'))
    assert c['reconciliation_required'] is True and 'overrun_tolerated' not in c['reservations'][OPEN]


def test_unknown_increase_basis_blocks(monkeypatch):
    ledger, policy, _ = tolerance_fixture(prior='1.00')
    monkeypatch.setattr(auto, 'MONTHLY_INCREASE_BASIS', 'other')
    with pytest.raises(SpendBlocked, match='Explicit approved automatic policy required'):
        auto.month_spend_increase_usd(ledger)
    with pytest.raises(SpendBlocked, match='Explicit approved automatic policy required'):
        auto.cash_exposure(ledger)


@pytest.mark.parametrize('context,expected', [
    (None, '0'), ({}, '0'), ({'admitted_input_tokens': 10, 'max_output_tokens': 5}, '0'),
    ({'unit_price_usd': '0.08', 'approved_unit_price_usd': '0.07'}, '0.01'),
    ({'unit_price_usd': '0.07', 'approved_unit_price_usd': '0.07'}, '0'),
    ({'admitted_input_tokens': 65536, 'max_output_tokens': 1500,
      'input_token_usd_per_million': '0.825', 'output_token_usd_per_million': '4.125',
      'approved_input_token_usd_per_million': '0.75', 'approved_output_token_usd_per_million': '3.75'}, '0.0054777')])
def test_in_band_excess_reads_the_recorded_rates(context, expected):
    record = {'provider': 'x', 'operation': 'y'}
    if context is not None:
        record['request_context'] = context
    assert auto.in_band_excess_usd(record) == Decimal(expected)


@pytest.mark.parametrize('context', [
    {'unit_price_usd': 'NaN', 'approved_unit_price_usd': '0.07'},
    {'unit_price_usd': '0.06', 'approved_unit_price_usd': '0.07'},
    {'unit_price_usd': '0.08'}, 'not-a-dict'])
def test_in_band_excess_blocks_malformed_recorded_rates(context):
    with pytest.raises(SpendBlocked):
        auto.in_band_excess_usd({'request_context': context})


def test_released_reservations_do_not_count_in_both_together(monkeypatch):
    ledger, policy, _ = tolerance_fixture(prior='0')
    ledger['reservations']['inband'] = in_band_hold('1.50')
    ledger['reservations']['gone'] = in_band_hold('2.00', status='released')
    monkeypatch.setattr(auto, 'MONTHLY_INCREASE_BASIS', 'both_together')
    assert auto.month_spend_increase_usd(ledger) == Decimal('1.50')


def assert_frozen_unmarked(candidate, key=OPEN):
    record = candidate['reservations'][key]
    assert candidate['reconciliation_required'] is True and record['reconciliation_required'] is True
    assert 'usage_receipt_overrun' in record['reconciliation_reasons'] and 'overrun_tolerated' not in record


@pytest.mark.parametrize('failure', ['unknown_basis', 'malformed_context'])
def test_a_failing_month_count_freezes_with_the_evidence_instead_of_raising(monkeypatch, failure):
    ledger, policy, _ = tolerance_fixture(prior='0')
    if failure == 'unknown_basis':
        monkeypatch.setattr(auto, 'MONTHLY_INCREASE_BASIS', 'other')
    else:
        monkeypatch.setattr(auto, 'MONTHLY_INCREASE_BASIS', 'both_together')
        ledger['reservations']['malformed'] = in_band_hold('1', request_context='not-a-dict')
    candidate = admit(ledger, policy, open_receipt('0.08'))
    assert_frozen_unmarked(candidate)
    assert candidate['reservations'][OPEN]['actual_gross_usd'] == '0.08'


def test_a_conflicting_higher_receipt_removes_the_earlier_tolerated_mark():
    ledger, policy, _ = tolerance_fixture(prior='0')
    first = admit(ledger, policy, open_receipt('0.08'))
    assert mark_of(first)['overrun_usd'] == '0.01' and not first.get('reconciliation_required')
    candidate = admit(first, policy, open_receipt('0.09', evidence_sha256='e' * 64))
    assert_frozen_unmarked(candidate)
    assert 'usage_receipt_conflict' in candidate['reservations'][OPEN]['reconciliation_reasons']
    assert candidate['reservations'][OPEN]['actual_gross_usd'] == '0.09'
    assert not [n for n in auto.owner_spend_notices(candidate) if n['kind'] == 'tolerated_overrun']


@pytest.mark.parametrize('flag', ['reconciliation_required', 'reconciliation_reasons'])
def test_an_already_flagged_record_is_never_newly_marked(flag):
    ledger, policy, _ = tolerance_fixture(prior='0')
    ledger['reservations'][OPEN][flag] = True if flag == 'reconciliation_required' else ['usage_receipt_overrun']
    candidate = admit(ledger, policy, open_receipt('0.08'))
    assert_frozen_unmarked(candidate)
