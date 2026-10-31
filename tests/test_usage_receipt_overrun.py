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
