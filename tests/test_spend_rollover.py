"""Offline rollover keeps one credit pool and all unresolved cash liabilities."""
from datetime import datetime, timezone
import copy
import pytest
from scripts.prepare_spend_ledger import prepare_candidate
from scripts.prepare_spend_rollover import prepare_rollover
from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked
from tests.test_spend_provisioning import metadata, NOW as OCTOBER

NOW = datetime(2026, 11, 1, 0, 1, tzinfo=timezone.utc)
IDENTITY = {'project_id': 'test-project', 'database_id': '(default)', 'collection': 'spend_control',
            'document_id': '2026-10', 'frozen_update_time': '2026-11-01T00:00:05.000001Z'}

def fixtures():
    source = prepare_candidate(metadata(), now=OCTOBER)
    source.update(closed_for_rollover=True, rollover_target_month='2026-11')
    fresh = metadata(); fresh['budget_window_start'] = '2026-11-01T00:00:00Z'
    account = fresh['providers']['bfl']; account['verified_at'] = NOW.isoformat()
    for price in account['price_evidence'].values(): price['verified_at'] = NOW.isoformat()
    return source, fresh


def record(amount, **kwargs):
    return {'provider': 'bfl', 'operation': 'generate_image:flux-3-image',
      'max_gross_usd': amount, 'status': 'reserved', 'created_at': OCTOBER.isoformat(), **kwargs}


def test_unknown_holds_survive_and_only_unreserved_credit_can_fund_new_month():
    source, fresh = fixtures()
    source['reservations'] = {'known': record('3', status='settled', actual_gross_usd='3', billing_reference='invoice-1'),
                              'unknown': record('1')}
    fresh['providers']['bfl']['remaining_credit_usd'] = '6.81'
    candidate = prepare_rollover(source, fresh, source_identity=IDENTITY, now=NOW)
    transfer = candidate['rollover']['transfers']['bfl']
    assert transfer['new_unreserved_credit_usd'] == '5.81'
    assert transfer['carried_unknown_credit_collateral_usd'] == '1'
    assert set(candidate['reservations']) == {'unknown'}
    assert candidate['reservations']['unknown']['origin_month'] == '2026-10'
    assert candidate['providers']['bfl']['credit_baseline_usd'] == '6.81'
    assert candidate['budget_policy_origin_start'] == '2026-10-30T04:09:08Z'
    assert BudgetGuard._cash_exposure(candidate) == 0
    # Preparation is pure; source remains a frozen historical accounting record.
    assert set(source['reservations']) == {'known', 'unknown'}


def test_observed_balance_shrink_does_not_grant_credit_to_unknown_calls_twice():
    source, fresh = fixtures(); source['reservations'] = {'lost-response': record('1')}
    fresh['providers']['bfl']['remaining_credit_usd'] = '0'
    candidate = prepare_rollover(source, fresh, source_identity=IDENTITY, now=NOW)
    assert candidate['providers']['bfl']['credit_baseline_usd'] == '0'
    assert str(BudgetGuard._cash_exposure(candidate)) == '1'
    assert candidate['rollover']['transfers']['bfl']['new_unreserved_credit_usd'] == '0'


def test_unverified_actual_amount_is_carried_at_full_bound():
    source, fresh = fixtures(); source['reservations'] = {'pending': record('1', status='settled', actual_gross_usd='.01')}
    candidate = prepare_rollover(source, fresh, source_identity=IDENTITY, now=NOW)
    carried = candidate['reservations']['pending']
    assert 'actual_gross_usd' not in carried
    assert carried['unreconciled_observed_gross_usd'] == '0.01'
    assert carried['max_gross_usd'] == '1'


@pytest.mark.parametrize('mutation', [
    lambda source, fresh, identity: source.update(closed_for_rollover=False),
    lambda source, fresh, identity: source.update(rollover_target_month='2026-12'),
    lambda source, fresh, identity: source.update(reconciliation_required=True),
    lambda source, fresh, identity: identity.pop('frozen_update_time'),
    lambda source, fresh, identity: fresh['providers']['bfl'].update(credential_fingerprint='d'*16),
    lambda source, fresh, identity: fresh['providers']['bfl'].update(allowance_period_id='new-free-grant'),
    lambda source, fresh, identity: fresh.update(budget_window_start='2026-11-01T00:00:01Z'),
])
def test_rollover_refuses_unfrozen_unbound_or_changed_accounting(mutation):
    source, fresh = fixtures(); identity = copy.deepcopy(IDENTITY); mutation(source, fresh, identity)
    with pytest.raises(SpendBlocked):
        prepare_rollover(source, fresh, source_identity=identity, now=NOW)


def test_carried_cash_above_new_month_ceiling_blocks_instead_of_resetting_holds():
    source, fresh = fixtures(); source['reservations'] = {'pending': record('3')}
    fresh['providers']['bfl']['remaining_credit_usd'] = '0'
    with pytest.raises(SpendBlocked, match='carried cash'):
        prepare_rollover(source, fresh, source_identity=IDENTITY, now=NOW)


def test_frozen_source_stops_all_new_reservations():
    source, _ = fixtures()
    guard = BudgetGuard(MemoryLedger(source), lambda: OCTOBER, credential_fingerprints={'bfl': 'a'*16})
    with pytest.raises(SpendBlocked, match='frozen'):
        guard.reserve('bfl', 'generate_image:flux-3-image')


def test_fresh_balance_cannot_renew_stale_manual_policy():
    from datetime import timedelta
    source, fresh = fixtures(); acct = fresh['providers']['bfl']
    acct.update(balance_verified_at=NOW.isoformat(), policy_verified_at=(NOW-timedelta(hours=25)).isoformat(),
      policy_valid_until=(NOW+timedelta(hours=1)).isoformat(), policy_reference='old-ui')
    with pytest.raises(SpendBlocked, match='oldest required evidence'):
        prepare_rollover(source, fresh, source_identity=IDENTITY, now=NOW)


def test_separate_policy_approval_does_not_silently_extend_24h_gate():
    from datetime import timedelta
    source, fresh = fixtures(); acct = fresh['providers']['bfl']
    acct.update(balance_verified_at=NOW.isoformat(), policy_verified_at=NOW.isoformat(),
      policy_valid_until=(NOW+timedelta(days=7)).isoformat(), policy_reference='approved-ui')
    with pytest.raises(SpendBlocked, match='24h gate'):
        prepare_rollover(source, fresh, source_identity=IDENTITY, now=NOW)


def test_late_settlement_cannot_mutate_frozen_historical_source():
    source, _ = fixtures(); source['reservations'] = {'pending': record('.07')}
    ledger = MemoryLedger(source)
    guard = BudgetGuard(ledger, lambda: OCTOBER, credential_fingerprints={'bfl': 'a'*16})
    with pytest.raises(SpendBlocked, match='active carried reservation'):
        guard.settle('pending', '.02')
    assert 'actual_gross_usd' not in ledger.document['reservations']['pending']
