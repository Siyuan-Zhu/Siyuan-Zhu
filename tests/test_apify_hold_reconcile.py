"""One-time settlement of the stale 2026-11-02 Apify hold from Apify's own run record.

Synthetic accounting values only: the reviewed constants are the sole real identifiers.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts import reconcile_apify_hold_20261102 as hold
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import provider_lifecycle as lifecycle
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_automatic_spend import fixtures
from tests.test_automatic_spend_runtime import PrivateWire

NOW = datetime(2026, 11, 13, 18, tzinfo=timezone.utc)
USER = 'SyntheticUser01'
OLD_TOKEN, NEW_TOKEN = 'c' * 16, 'd' * 16
OPERATION = 'instagram-post-scraper'
FINISHED = '2026-11-02T07:18:00.000Z'


def run_body(**changes):
    body = {'id': hold.RUN_ID, 'userId': USER, 'actId': lifecycle.APIFY_ACTOR_ID,
            'status': 'SUCCEEDED', 'finishedAt': FINISHED, 'usageTotalUsd': 0.0034,
            'chargedEventCounts': {'post': 2}, 'accountedChargedEventCounts': {'post': 2}}
    body.update(changes)
    return body


def reservation(**changes):
    """Mirrors the real record's exact key set with synthetic values."""
    record = {
        'operation': OPERATION, 'status': 'reserved',
        'request_sha256': hashlib.sha256(b'synthetic request').hexdigest(),
        'provider': 'apify',
        'submission_metadata': {'actor_id': lifecycle.APIFY_ACTOR_ID, 'build': '0.0.648', 'user_id': USER},
        'credential_fingerprint': OLD_TOKEN,
        'allowance_period_id': '2026-10-26T00:00:00.000Z/2026-11-25T23:59:59.999Z',
        'automatic_policy_sha256': auto.digest('older approved policy'),
        'max_gross_usd': hold.MAX, 'submitted_at': '2026-11-02T07:16:33.660484Z',
        'created_at': hold.CREATED_AT, 'request_context': {},
        'provider_request_id': hold.RUN_ID}
    record.update(changes)
    return record


class FakeRuns:
    def __init__(self, *bodies):
        self.bodies = list(bodies)
        self.calls = []

    def __call__(self, run_id):
        self.calls.append(run_id)
        if not self.bodies:
            raise AssertionError('The run was read again')
        return copy.deepcopy(self.bodies.pop(0))


def setup(monkeypatch, **record_changes):
    ledger, policy, _ = fixtures(NOW)
    for item in (policy['providers']['apify']['account'], ledger['providers']['apify']):
        item['operation_caps_usd'] = {OPERATION: '.20'}
    policy['providers']['apify']['account_fingerprint'] = hashlib.sha256(USER.encode()).hexdigest()[:16]
    sha = auto.digest(policy)
    account = ledger['providers']['apify']
    account['credential_fingerprint'] = NEW_TOKEN
    account['credential_rebinds'] = [{'from': OLD_TOKEN, 'to': NEW_TOKEN, 'at': '2026-11-10T00:00:00Z'}]
    ledger.update(month='2026-11', automatic_policy_sha256=sha,
                  reservations={hold.RESERVATION_ID: reservation(**record_changes)})
    wire = PrivateWire(monkeypatch, {'2026-11': ledger, 'policy_' + sha: policy})
    return wire, sha, ledger, policy


def stable():
    return FakeRuns(run_body(), run_body())


@pytest.mark.parametrize('unknown', [False, True])
def test_terminal_stable_run_settles_once_and_repeat_reconciles(monkeypatch, unknown):
    wire, sha, ledger, policy = setup(monkeypatch)
    wire.unknown_post = 1 if unknown else None
    runs = stable()
    result = hold.reconcile(wire.store, sha, fetch_run=runs, apply=True, now=NOW)
    assert result == {'status': 'applied_or_reconciled', 'actual_gross_usd': '0.0034'}
    assert runs.calls == [hold.RUN_ID, hold.RUN_ID]
    after = wire.docs['2026-11']
    settled = after['reservations'][hold.RESERVATION_ID]
    assert settled['status'] == 'settled' and settled['actual_gross_usd'] == '0.0034'
    assert settled['billing_reference'].startswith('provider_usage_v1:')
    assert 'reconciliation_required' not in after
    # Only the one reservation changes; its admission facts (older policy and token) stay as created.
    expected = copy.deepcopy(ledger)
    expected['reservations'][hold.RESERVATION_ID] = {
        **reservation(), **{k: settled[k] for k in ('status', 'actual_gross_usd', 'billing_reference', 'usage_receipt')}}
    assert after == expected
    assert settled['credential_fingerprint'] == OLD_TOKEN
    assert settled['automatic_policy_sha256'] != sha
    receipt = settled['usage_receipt']
    assert receipt['credential_fingerprint'] == NEW_TOKEN
    normal = lifecycle.usage_receipt(
        SimpleNamespace(reservation_id=hold.RESERVATION_ID, credential_fingerprint=NEW_TOKEN),
        'apify', OPERATION, settled['request_sha256'], hold.RUN_ID, '0.0034', {})
    assert set(receipt) == set(normal)
    assert receipt['accounting_policy'] == 'provider_usage_v1'
    assert receipt['evidence_sha256'] == lifecycle.request_digest(
        {'id': hold.RUN_ID, 'status': 'SUCCEEDED', 'usageTotalUsd': 0.0034, 'finishedAt': FINISHED})
    journal = wire.docs[hold.RECEIPT_ID]
    assert journal['usage_receipt'] == receipt and journal['reservation_id'] == hold.RESERVATION_ID
    # Repeat or uncertain dispatch reconciles from the durable receipt: no new read, no new write.
    again = hold.reconcile(wire.store, sha, fetch_run=FakeRuns(), apply=True, now=NOW)
    assert again == result and len(wire.posts) == 1


def test_plan_reads_the_run_but_writes_nothing(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch)
    result = hold.reconcile(wire.store, sha, fetch_run=stable(), now=NOW)
    assert result == {'status': 'planned', 'actual_gross_usd': '0.0034'}
    assert not wire.posts and wire.docs['2026-11'] == ledger and hold.RECEIPT_ID not in wire.docs


@pytest.mark.parametrize('second', [
    run_body(usageTotalUsd=0.0035),
    run_body(chargedEventCounts={'post': 2}, accountedChargedEventCounts={'post': 1}),
    run_body(status='RUNNING'),
    run_body(finishedAt='2999-01-01T00:00:00Z')])
def test_preliminary_cost_does_not_settle(monkeypatch, second):
    wire, sha, ledger, _ = setup(monkeypatch)
    with pytest.raises(RuntimeError, match='cost|terminal'):
        hold.reconcile(wire.store, sha, fetch_run=FakeRuns(run_body(), second), apply=True, now=NOW)
    assert not wire.posts and wire.docs['2026-11'] == ledger


@pytest.mark.parametrize('bodies', [
    (run_body(userId='SomeoneElse9'), run_body(userId='SomeoneElse9')),
    (run_body(), run_body(userId='SomeoneElse9')),
    (run_body(actId='SomeOtherActor1'), run_body(actId='SomeOtherActor1')),
    (run_body(id='OtherRun0001'), run_body(id='OtherRun0001'))])
def test_other_account_run_blocks(monkeypatch, bodies):
    wire, sha, ledger, _ = setup(monkeypatch)
    with pytest.raises((RuntimeError, ValueError)):
        hold.reconcile(wire.store, sha, fetch_run=FakeRuns(*bodies), apply=True, now=NOW)
    assert not wire.posts and wire.docs['2026-11'] == ledger


def test_policy_account_fingerprint_must_match_the_run_owner(monkeypatch):
    wire, sha, _, policy = setup(monkeypatch)
    policy['providers']['apify']['account_fingerprint'] = 'f' * 16
    changed = auto.digest(policy)
    wire.docs['policy_' + changed] = policy
    wire.versions['policy_' + changed] = 1
    wire.docs['2026-11']['automatic_policy_sha256'] = changed
    with pytest.raises(RuntimeError, match='account'):
        hold.reconcile(wire.store, changed, fetch_run=stable(), apply=True, now=NOW)
    assert not wire.posts


@pytest.mark.parametrize('change', [
    {'status': 'settled'}, {'max_gross_usd': '0.20'}, {'provider_request_id': 'OtherRun0001'},
    {'created_at': '2026-11-02T07:16:32.713320+00:00'}, {'provider': 'bfl'}, {'credit_excluded': True}])
def test_changed_reservation_blocks(monkeypatch, change):
    wire, sha, ledger, _ = setup(monkeypatch, **change)
    runs = stable()
    with pytest.raises((RuntimeError, SpendBlocked)):
        hold.reconcile(wire.store, sha, fetch_run=runs, apply=True, now=NOW)
    assert not wire.posts and not runs.calls


def test_missing_reservation_blocks(monkeypatch):
    wire, sha, _, _ = setup(monkeypatch)
    wire.docs['2026-11']['reservations'] = {}
    with pytest.raises(RuntimeError):
        hold.reconcile(wire.store, sha, fetch_run=stable(), apply=True, now=NOW)
    assert not wire.posts


def test_cost_above_bound_blocks(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch)
    runs = FakeRuns(run_body(usageTotalUsd=0.17), run_body(usageTotalUsd=0.17))
    with pytest.raises(RuntimeError, match='bound'):
        hold.reconcile(wire.store, sha, fetch_run=runs, apply=True, now=NOW)
    assert not wire.posts and wire.docs['2026-11'] == ledger


def test_cost_exactly_at_the_bound_settles(monkeypatch):
    wire, sha, _, _ = setup(monkeypatch)
    runs = FakeRuns(run_body(usageTotalUsd='0.16'), run_body(usageTotalUsd='0.16'))
    assert hold.reconcile(wire.store, sha, fetch_run=runs, apply=True, now=NOW)['actual_gross_usd'] == '0.16'
    assert 'reconciliation_required' not in wire.docs['2026-11']


def test_dispatch_that_never_landed_is_unresolved_then_retries_cleanly(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch)
    wire.unknown_before_post = 1
    with pytest.raises(RuntimeError, match='unresolved'):
        hold.reconcile(wire.store, sha, fetch_run=stable(), apply=True, now=NOW)
    assert wire.docs['2026-11'] == ledger and hold.RECEIPT_ID not in wire.docs
    assert hold.reconcile(wire.store, sha, fetch_run=stable(), apply=True, now=NOW)['actual_gross_usd'] == '0.0034'


def test_ledger_changed_between_read_and_commit_is_refused(monkeypatch):
    wire, sha, _, _ = setup(monkeypatch)
    def touch(w):
        w.versions['2026-11'] += 1
    wire.before_post = touch
    with pytest.raises(Exception):
        hold.reconcile(wire.store, sha, fetch_run=stable(), apply=True, now=NOW)
    assert wire.docs['2026-11']['reservations'][hold.RESERVATION_ID]['status'] == 'reserved'


def test_receipt_that_disagrees_with_the_ledger_blocks_instead_of_reporting_success(monkeypatch):
    wire, sha, _, _ = setup(monkeypatch)
    hold.reconcile(wire.store, sha, fetch_run=stable(), apply=True, now=NOW)
    wire.docs['2026-11']['reservations'][hold.RESERVATION_ID]['actual_gross_usd'] = '0.16'
    with pytest.raises(RuntimeError, match='reconciliation'):
        hold.reconcile(wire.store, sha, fetch_run=FakeRuns(), apply=True, now=NOW)


def test_cli_prints_only_status_and_amount(monkeypatch, capsys):
    wire, sha, _, _ = setup(monkeypatch)
    monkeypatch.setenv('SPEND_POLICY_SHA256', sha)
    assert hold.main(['--apply'], store=wire.store, fetch_run=stable()) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed) == {'status': 'applied_or_reconciled', 'actual_gross_usd': '0.0034'}
    for private in (USER, OLD_TOKEN, NEW_TOKEN, hold.RUN_ID):
        assert private not in printed


def test_no_literal_dollar_sign_in_the_tool_or_its_tests():
    from pathlib import Path
    for path in (Path(hold.__file__), Path(__file__)):
        assert chr(36) not in path.read_text(), path
