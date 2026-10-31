"""Admission-bound and verification lifecycle acceptance at the spend boundary."""
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock
import copy
import pytest
from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked, guard_genai_client

NOW = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)

def account(credit='0'):
    return {'account_id': 'sha256:account', 'plan_id': 'free',
            'verification_reference': 'snapshot-1', 'verified_at': NOW.isoformat(),
            'credit_baseline_usd': credit, 'remaining_credit_usd': credit,
            'allowance_period_id': '2026-10', 'allowance_starts_at': '2026-10-01T00:00:00Z',
            'allowance_expires_at': '2026-11-01T00:00:00Z',
            'max_request_duration_seconds': 300,
            'operation_caps_usd': {'generate_content:gemini-2.5-flash': '.1', 'generate_content:gemini-2.5-pro': '.1', 'generate_content:gemini-3-flash-preview': '.1'},
            'request_limits': {op: {'pricing_reference': 'price-1', 'max_input_bytes': 100, 'max_output_tokens': 512,
                                   'max_thinking_tokens': 128, 'thinking_level': 'minimal'}
                               for op in ('generate_content:gemini-2.5-flash', 'generate_content:gemini-2.5-pro', 'generate_content:gemini-3-flash-preview')}}

def document(credit='0'):
    return {'month': '2026-10', 'cash_ceiling_usd': '2.50', 'cash_spent_before_ledger_usd': '0',
            'providers': {'gemini': account(credit), 'apify': account(credit)}}

def wrapper(doc=None):
    client = Mock()
    guard = BudgetGuard(MemoryLedger(doc or document()), lambda: NOW)
    return client, guard, guard_genai_client(client, guard)

@pytest.mark.parametrize('config', [
    {'system_instruction': 'x'*1000}, {'candidate_count': 8},
    {'tools': [{'google_search': {}}]}, {'automatic_function_calling': {'maximum_remote_calls': 100}},
    {'http_options': {'retry_options': {'attempts': 5}}},
])
def test_unpriced_sdk_options_cannot_reach_provider(config):
    client, guard, wrapped = wrapper()
    with pytest.raises(SpendBlocked):
        wrapped.models.generate_content(model='gemini-2.5-flash', contents='hi', config=config)
    client.models.generate_content.assert_not_called()
    assert guard.snapshot()['reservation_count'] == 0


def test_positional_arguments_cannot_bypass_measured_input():
    client, guard, wrapped = wrapper()
    with pytest.raises(SpendBlocked):
        wrapped.models.generate_content('x'*1000, model='gemini-2.5-flash', contents='hi')
    client.models.generate_content.assert_not_called()


def test_verified_transport_and_model_aware_thinking_are_applied():
    client, _, wrapped = wrapper()
    wrapped.models.generate_content(model='gemini-2.5-flash', contents='hi')
    cfg = client.models.generate_content.call_args.kwargs['config']
    assert cfg['candidate_count'] == 1
    assert cfg['http_options'] == {'timeout': 300000, 'retry_options': {'attempts': 1}}
    assert cfg['thinking_config'] == {'thinking_budget': 0}
    wrapped.models.generate_content(model='gemini-2.5-pro', contents='hi')
    assert client.models.generate_content.call_args.kwargs['config']['thinking_config'] == {'thinking_budget': 128}
    wrapped.models.generate_content(model='gemini-3-flash-preview', contents='hi')
    assert client.models.generate_content.call_args.kwargs['config']['thinking_config'] == {'thinking_level': 'minimal'}


def test_credit_or_free_quota_expiring_during_a_request_blocks_before_provider():
    doc = document('1')
    doc['providers']['apify']['allowance_expires_at'] = (NOW + timedelta(seconds=60)).isoformat()
    guard = BudgetGuard(MemoryLedger(doc), lambda: NOW)
    fn = Mock()
    with pytest.raises(SpendBlocked):
        guard.call('apify', 'run', '.1', fn)
    fn.assert_not_called()


def test_missing_provider_identity_cannot_claim_verified_credit():
    doc = document('1')
    doc['providers']['apify'].pop('account_id')
    with pytest.raises(SpendBlocked):
        BudgetGuard(MemoryLedger(doc), lambda: NOW).reserve('apify', 'run', '.1')


def snapshot(now, credit='0'):
    data = account(credit)
    data['verified_at'] = now.isoformat()
    data['verification_reference'] = 'snapshot-2'
    return data


def test_day_two_refresh_preserves_credit_baseline_and_unknown_reservations():
    ledger = MemoryLedger(document('5'))
    guard = BudgetGuard(ledger, lambda: NOW)
    known, _ = guard.reserve('apify', 'known', '2')
    unknown, _ = guard.reserve('apify', 'unknown', '1')
    guard.settle(known, '2')
    tomorrow = NOW + timedelta(days=1)
    day_two = BudgetGuard(ledger, lambda: tomorrow)
    with pytest.raises(SpendBlocked):
        day_two.reserve('apify', 'next', '.1')
    day_two.refresh_verification('apify', snapshot(tomorrow, '3'), expected_reference='snapshot-1')
    assert ledger.document['providers']['apify']['credit_baseline_usd'] == '5'
    assert unknown in ledger.document['reservations']
    assert ledger.document['reservations'][unknown]['status'] == 'reserved'
    assert day_two.snapshot()['cash_exposure_usd'] == '0'
    day_two.reserve('apify', 'next', '.1')


def test_billing_reconciliation_does_not_count_consumed_free_credit_twice():
    ledger = MemoryLedger(document('5'))
    guard = BudgetGuard(ledger, lambda: NOW)
    receipt, _ = guard.reserve('apify', 'run', '5')
    guard.refresh_verification('apify', snapshot(NOW, '0'), expected_reference='snapshot-1', settlements={receipt: '5'})
    assert ledger.document['providers']['apify']['credit_baseline_usd'] == '5'
    assert guard.snapshot()['cash_exposure_usd'] == '0'
    guard.reserve('apify', 'next', '2.50')
    with pytest.raises(SpendBlocked):
        guard.reserve('apify', 'extra', '.01')


def test_refresh_cannot_increase_baseline_credit_or_replace_account_identity():
    ledger = MemoryLedger(document('5'))
    guard = BudgetGuard(ledger, lambda: NOW)
    guard.refresh_verification('apify', snapshot(NOW, '99'), expected_reference='snapshot-1')
    assert ledger.document['providers']['apify']['credit_baseline_usd'] == '5'
    before = copy.deepcopy(ledger.document)
    changed = snapshot(NOW, '5')
    changed['account_id'] = 'different-account'
    with pytest.raises(SpendBlocked):
        guard.refresh_verification('apify', changed, expected_reference='snapshot-2')
    assert ledger.document == before


def test_refresh_rejects_a_stale_compare_and_swap_reference():
    guard = BudgetGuard(MemoryLedger(document('5')), lambda: NOW)
    with pytest.raises(SpendBlocked):
        guard.refresh_verification('apify', snapshot(NOW, '5'), expected_reference='not-current')


def test_remote_file_parts_cannot_hide_unbounded_model_input():
    client, _, wrapped = wrapper()
    with pytest.raises(SpendBlocked):
        wrapped.models.generate_content(model='gemini-2.5-flash', contents=[{'file_data': {'file_uri': 'gs://unbounded-object'}}])
    client.models.generate_content.assert_not_called()


def test_daily_verified_free_quota_can_roll_without_reusing_yesterdays_slots():
    doc = document()
    provider = doc['providers']['gemini']
    operation = 'generate_content:gemini-2.5-flash'
    provider.update(billing_mode='verified_free_tier', verified_free_operations=[operation],
                    free_request_limit=1, allowance_period_id='day-29',
                    allowance_starts_at='2026-10-29T00:00:00Z', allowance_expires_at='2026-10-30T00:00:00Z')
    provider['operation_caps_usd'][operation] = '0'
    ledger = MemoryLedger(doc)
    guard = BudgetGuard(ledger, lambda: NOW)
    old_receipt, _ = guard.reserve('gemini', operation)
    tomorrow = NOW + timedelta(days=1)
    refreshed = copy.deepcopy(provider)
    refreshed.update(verification_reference='snapshot-2', verified_at=tomorrow.isoformat(),
                     allowance_period_id='day-30', allowance_starts_at='2026-10-30T00:00:00Z',
                     allowance_expires_at='2026-10-31T00:00:00Z')
    day_two = BudgetGuard(ledger, lambda: tomorrow)
    day_two.refresh_verification('gemini', refreshed, expected_reference='snapshot-1')
    day_two.reserve('gemini', operation)
    with pytest.raises(SpendBlocked):
        day_two.reserve('gemini', operation)
    assert old_receipt in ledger.document['reservations']


def test_unknown_charge_is_not_settled_by_a_zero_balance_refresh():
    ledger = MemoryLedger(document('5'))
    guard = BudgetGuard(ledger, lambda: NOW)
    receipt, _ = guard.reserve('apify', 'run', '5')
    report = guard.refresh_verification('apify', snapshot(NOW, '0'), expected_reference='snapshot-1')
    assert report['reconciliation_required'] is True
    assert guard.snapshot()['cash_exposure_usd'] == '5'
    assert 'actual_gross_usd' not in ledger.document['reservations'][receipt]
    with pytest.raises(SpendBlocked):
        guard.reserve('apify', 'next', '.01')
