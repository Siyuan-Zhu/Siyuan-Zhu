"""One-time reconcile of the frozen 2026-11-22 Hydrogen hold from the reply's own usage record.

Synthetic accounting values only: the reviewed constants are the sole real identifiers.
"""
import copy
from datetime import datetime, timezone
import hashlib
import json

import pytest

from scripts import reconcile_hydrogen_hold_20261122 as hold
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.automatic_guard import request_digest
from src.zeitgeist.spend_successor import build_successor, successor_ledger
from tests.test_automatic_spend_runtime import PrivateWire
from tests.test_spend_successor import KW, installed

NOW = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)
GEMINI_FINGERPRINT = 'a' * 16
REQUEST_SHA = hashlib.sha256(b'synthetic hydrogen request').hexdigest()
USAGE = {'promptTokenCount': 1500, 'candidatesTokenCount': 2000, 'thoughtsTokenCount': 1800,
         'totalTokenCount': 5300, 'promptTokensDetails': [{'modality': 'TEXT', 'tokenCount': 1500}]}
# (1500 x 1.50 + 3800 x 7.50) / 1,000,000
AMOUNT = '0.03075'
CONTEXT = {'admitted_input_tokens': 720, 'max_output_tokens': 6000,
           'input_token_usd_per_million': '1.50', 'output_token_usd_per_million': '7.50',
           'approved_input_token_usd_per_million': '1.50', 'approved_output_token_usd_per_million': '7.50'}
REPORT_KEYS = {'mode', 'reservation_status_before', 'reservation_status_after', 'billed_prompt_tokens',
               'billed_output_tokens', 'accounted_gross_usd', 'max_gross_usd',
               'ledger_reconciliation_required_before', 'ledger_reconciliation_required_after', 'plan_sha256'}


def frozen(policy_sha, usage=USAGE, **changes):
    """The frozen hold as the guard leaves it: reserved, submitted, then frozen by the usage check."""
    record = {
        'provider': 'gemini', 'operation': hold.OPERATION, 'max_gross_usd': hold.MAX, 'status': 'reserved',
        'created_at': hold.CREATED_AT, 'allowance_period_id': None, 'request_context': copy.deepcopy(CONTEXT),
        'request_sha256': REQUEST_SHA, 'credential_fingerprint': GEMINI_FINGERPRINT,
        'automatic_policy_sha256': policy_sha, 'execution_owner': 'github-1-1',
        'provider_request_id': hold.PROVIDER_REQUEST_ID, 'submitted_at': '2026-11-22T11:22:55.100000+00:00',
        'submission_metadata': {}, 'reconciliation_required': True,
        'unreconciled_usage': {'reason': hold.FREEZE_REASON, 'evidence_sha256': request_digest(usage)}}
    record.update(changes)
    return record


def settled_flash():
    return {'provider': 'gemini', 'operation': 'generate_content:gemini-3.8-flash', 'max_gross_usd': '0.05',
            'status': 'settled', 'created_at': '2026-11-21T09:00:00+00:00', 'actual_gross_usd': '0.01',
            'billing_reference': 'provider_usage_v1:' + 'b' * 64}


def setup(monkeypatch, usage=USAGE, ledger_changes=None, **record_changes):
    ledger, p0 = installed()
    policy = build_successor(p0, **dict(KW, hydrogen_max_output_tokens=6000))
    sha = auto.digest(policy)
    ledger = successor_ledger(ledger, policy, sha)
    ledger.update(month='2026-11', credit_only_providers=list(policy['credit_only_providers']),
                  reconciliation_required=True,
                  reservations={hold.RESERVATION_ID: frozen(sha, usage, **record_changes), 'done': settled_flash()})
    ledger.update(ledger_changes or {})
    # The real digest pins the real reply; synthetic usage gets a synthetic pin.
    monkeypatch.setattr(hold, 'EVIDENCE_SHA256', request_digest(usage))
    wire = PrivateWire(monkeypatch, {'2026-11': ledger, 'policy_' + sha: policy})
    return wire, sha, copy.deepcopy(ledger), policy


def reconcile(wire, sha, usage=USAGE, **kw):
    return hold.reconcile(wire.store, sha, copy.deepcopy(usage), now=NOW, **kw)


def assert_untouched(wire, ledger):
    assert not wire.posts and wire.docs['2026-11'] == ledger and hold.RECEIPT_ID not in wire.docs


def test_constants_are_the_reviewed_facts():
    import importlib
    fresh = importlib.reload(hold)
    assert fresh.MONTH == '2026-11'
    assert fresh.RESERVATION_ID == '7e885b47955a4c6b83d311676a6f9a7c'
    assert fresh.OPERATION == 'generate_content:gemini-4-hydrogen'
    assert fresh.CREATED_AT == '2026-11-22T11:22:54.431743+00:00'
    assert fresh.PROVIDER_REQUEST_ID == 'kNACa_WzAbXniLMPidXE0QE'
    assert fresh.MAX == '0.04608'
    assert fresh.ADMITTED_INPUT_TOKENS == 720 and fresh.MAX_OUTPUT_TOKENS == 6000
    assert fresh.FREEZE_REASON == 'Gemini usage exceeds admitted token shape'
    assert fresh.EVIDENCE_SHA256 == '8e77e41af7f68901e9467f6827090c460aa5d635736e726c03011a3d0b9c140b'
    assert fresh.RECEIPT_ID == 'hydrogen_hold_reconcile_20261122'


def test_plan_reports_the_settlement_and_writes_nothing(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch)
    report = reconcile(wire, sha)
    assert set(report) == REPORT_KEYS
    assert report['mode'] == 'plan'
    assert report['reservation_status_before'] == 'reserved' and report['reservation_status_after'] == 'settled'
    assert report['billed_prompt_tokens'] == 1500 and report['billed_output_tokens'] == 3800
    assert report['accounted_gross_usd'] == AMOUNT and report['max_gross_usd'] == hold.MAX
    assert report['ledger_reconciliation_required_before'] is True
    assert report['ledger_reconciliation_required_after'] is False
    assert len(report['plan_sha256']) == 64
    assert_untouched(wire, ledger)


@pytest.mark.parametrize('unknown', [False, True])
def test_apply_settles_the_hold_clears_the_freeze_and_repeat_reconciles(monkeypatch, unknown):
    wire, sha, ledger, _ = setup(monkeypatch)
    plan = reconcile(wire, sha)
    wire.unknown_post = 1 if unknown else None
    report = reconcile(wire, sha, apply=True, expected_plan_sha=plan['plan_sha256'])
    assert report == dict(plan, mode='applied')
    assert len(wire.posts) == 1 and len(wire.posts[0]['writes']) == 2
    after = wire.docs['2026-11']
    assert after['reconciliation_required'] is False
    assert ({k: v for k, v in after.items() if k not in ('reservations', 'reconciliation_required')}
            == {k: v for k, v in ledger.items() if k not in ('reservations', 'reconciliation_required')})
    assert after['reservations']['done'] == ledger['reservations']['done']
    record = after['reservations'][hold.RESERVATION_ID]
    assert record['status'] == 'settled' and record['actual_gross_usd'] == AMOUNT
    assert 'reconciliation_required' not in record and 'unreconciled_usage' not in record
    assert record['reconciled_usage'] == {
        'reason': hold.FREEZE_REASON, 'evidence_sha256': request_digest(USAGE),
        'reconciled_at': NOW.isoformat(), 'receipt_id': hold.RECEIPT_ID,
        'billed_prompt_tokens': 1500, 'billed_output_tokens': 3800}
    receipt = record['usage_receipt']
    assert receipt == {
        'schema_version': 1, 'accounting_policy': 'provider_usage_v1', 'reservation_id': hold.RESERVATION_ID,
        'provider': 'gemini', 'operation': hold.OPERATION, 'credential_fingerprint': GEMINI_FINGERPRINT,
        'provider_request_id': hold.PROVIDER_REQUEST_ID, 'request_sha256': REQUEST_SHA,
        'observed_at': NOW.isoformat(), 'accounted_gross_usd': AMOUNT,
        'accounting_basis': 'conservative_response_usage',
        'evidence_sha256': auto.digest({'response_id': hold.PROVIDER_REQUEST_ID, 'usage': USAGE,
                                        'request_context': CONTEXT})}
    assert record['billing_reference'] == 'provider_usage_v1:' + auto.digest(
        {k: v for k, v in receipt.items() if k != 'observed_at'})
    # The hold's own creation facts stay as created.
    expected = {k: v for k, v in frozen(sha).items() if k not in ('reconciliation_required', 'unreconciled_usage')}
    assert {k: v for k, v in record.items()
            if k not in ('status', 'actual_gross_usd', 'billing_reference', 'usage_receipt', 'reconciled_usage')} == {
        k: v for k, v in expected.items() if k != 'status'}
    journal = wire.docs[hold.RECEIPT_ID]
    assert journal['usage_receipt'] == receipt and journal['reservation_id'] == hold.RESERVATION_ID
    assert journal['plan_sha256'] == plan['plan_sha256'] and journal['actual_gross_usd'] == AMOUNT
    assert journal['before_sha256'] == auto.digest(ledger) and journal['after_sha256'] == auto.digest(after)
    # A repeat or uncertain dispatch reconciles from the durable receipt: no new write.
    docs = copy.deepcopy(wire.docs)
    again = reconcile(wire, sha, apply=True, expected_plan_sha=plan['plan_sha256'])
    assert again == dict(plan, mode='already_reconciled', reservation_status_before='settled',
                         ledger_reconciliation_required_before=False)
    assert len(wire.posts) == 1 and wire.docs == docs
    assert reconcile(wire, sha)['mode'] == 'already_reconciled'


def test_receipt_that_no_longer_describes_the_ledger_refuses(monkeypatch):
    wire, sha, _, _ = setup(monkeypatch)
    reconcile(wire, sha, apply=True, expected_plan_sha=reconcile(wire, sha)['plan_sha256'])
    wire.docs['2026-11']['reservations'][hold.RESERVATION_ID]['actual_gross_usd'] = hold.MAX
    with pytest.raises(RuntimeError, match='reconciliation'):
        reconcile(wire, sha, apply=True)
    assert len(wire.posts) == 1


def test_usage_that_is_not_the_refused_reply_refuses(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch)
    other = dict(USAGE, candidatesTokenCount=1999, totalTokenCount=5299)
    with pytest.raises(RuntimeError, match='usage'):
        reconcile(wire, sha, other, apply=True, expected_plan_sha='0' * 64)
    assert_untouched(wire, ledger)


def test_stored_evidence_that_differs_from_the_pin_refuses(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch, unreconciled_usage={'reason': hold.FREEZE_REASON,
                                                                  'evidence_sha256': 'c' * 64})
    with pytest.raises(RuntimeError):
        reconcile(wire, sha)
    assert_untouched(wire, ledger)


@pytest.mark.parametrize('change', [
    {'status': 'settled'}, {'status': 'released'}, {'provider': 'bfl'},
    {'operation': 'generate_content:gemini-3.8-flash'},
    {'created_at': '2026-11-22T11:22:54.431744+00:00'}, {'provider_request_id': 'OtherResponse01'},
    {'max_gross_usd': '0.04609'},
    {'request_context': dict(CONTEXT, admitted_input_tokens=721)},
    {'request_context': dict(CONTEXT, max_output_tokens=8000)},
    {'request_context': dict(CONTEXT, input_token_usd_per_million='2.00')},
    {'request_context': dict(CONTEXT, output_token_usd_per_million='7.00')},
    {'request_context': dict(CONTEXT, approved_output_token_usd_per_million='7.00')},
    {'request_context': {k: v for k, v in CONTEXT.items() if k != 'approved_input_token_usd_per_million'}},
    {'request_context': dict(CONTEXT, counted_input_tokens=720)},
    {'reconciliation_required': False},
    {'unreconciled_usage': {'reason': 'Gemini response model version not pinned',
                            'evidence_sha256': request_digest(USAGE)}},
    {'usage_receipt': {}}, {'actual_gross_usd': '0.01'}, {'billing_reference': 'provider_usage_v1:x'},
    {'reconciliation_reasons': ['usage_receipt_overrun']}, {'usage_receipt_conflict': {}}])
def test_reservation_that_differs_from_the_reviewed_hold_refuses(monkeypatch, change):
    wire, sha, ledger, _ = setup(monkeypatch, **change)
    with pytest.raises(RuntimeError):
        reconcile(wire, sha, apply=True, expected_plan_sha='0' * 64)
    assert_untouched(wire, ledger)


def test_missing_reservation_refuses(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch)
    del wire.docs['2026-11']['reservations'][hold.RESERVATION_ID]
    with pytest.raises(RuntimeError):
        reconcile(wire, sha)
    assert not wire.posts and hold.RECEIPT_ID not in wire.docs


@pytest.mark.parametrize('usage', [
    dict(USAGE, promptTokenCount=1500.0),
    dict(USAGE, candidatesTokenCount=-1, totalTokenCount=3299),
    {k: v for k, v in USAGE.items() if k != 'totalTokenCount'},
    dict(USAGE, promptTokenCount=True),
    dict(USAGE, toolUsePromptTokenCount=10),
    dict(USAGE, cachedContentTokenCount=1501),
    dict(USAGE, totalTokenCount=5301),
    dict(USAGE, thoughtsTokenCount='1800')])
def test_incomplete_or_inconsistent_usage_refuses(monkeypatch, usage):
    wire, sha, ledger, _ = setup(monkeypatch, usage=usage)
    with pytest.raises(RuntimeError):
        reconcile(wire, sha, usage, apply=True, expected_plan_sha='0' * 64)
    assert_untouched(wire, ledger)


def test_missing_thoughts_count_counts_as_zero(monkeypatch):
    usage = {'promptTokenCount': 1500, 'candidatesTokenCount': 2000, 'totalTokenCount': 3500}
    wire, sha, _, _ = setup(monkeypatch, usage=usage)
    report = reconcile(wire, sha, usage)
    assert report['billed_output_tokens'] == 2000 and report['accounted_gross_usd'] == '0.01725'


def test_amount_above_the_hold_maximum_refuses(monkeypatch):
    usage = {'promptTokenCount': 2000, 'candidatesTokenCount': 3000, 'thoughtsTokenCount': 2900,
             'totalTokenCount': 7900}
    # (2000 x 1.50 + 5900 x 7.50) / 1,000,000 = 0.04725, above 0.04608
    wire, sha, ledger, _ = setup(monkeypatch, usage=usage)
    with pytest.raises(RuntimeError, match='maximum'):
        reconcile(wire, sha, usage, apply=True, expected_plan_sha='0' * 64)
    assert_untouched(wire, ledger)


@pytest.mark.parametrize('marker', [
    {'reconciliation_required': True}, {'reconciliation_reasons': ['usage_receipt_overrun']},
    {'unreconciled_usage': {'reason': 'synthetic', 'evidence_sha256': 'd' * 64}},
    {'usage_receipt_conflict': {}}])
def test_another_frozen_reservation_keeps_the_ledger_flag_and_refuses(monkeypatch, marker):
    wire, sha, ledger, _ = setup(monkeypatch)
    other = dict(settled_flash(), **marker)
    wire.docs['2026-11']['reservations']['other'] = other
    ledger = copy.deepcopy(wire.docs['2026-11'])
    with pytest.raises(RuntimeError):
        reconcile(wire, sha, apply=True, expected_plan_sha='0' * 64)
    assert_untouched(wire, ledger)
    assert wire.docs['2026-11']['reconciliation_required'] is True


def test_ledger_under_another_policy_refuses(monkeypatch):
    wire, sha, ledger, policy = setup(monkeypatch)
    other = copy.deepcopy(policy)
    other['approval_reference'] = 'another synthetic approval'
    other_sha = auto.digest(other)
    wire.docs['policy_' + other_sha] = other
    wire.versions['policy_' + other_sha] = 1
    with pytest.raises(RuntimeError, match='policy'):
        reconcile(wire, other_sha, apply=True, expected_plan_sha='0' * 64)
    assert_untouched(wire, ledger)


def test_cash_exposure_above_the_ceiling_refuses(monkeypatch):
    wire, sha, ledger, _ = setup(monkeypatch, ledger_changes={'cash_spent_before_ledger_usd': '10.00'})
    with pytest.raises(RuntimeError):
        reconcile(wire, sha, apply=True, expected_plan_sha='0' * 64)
    assert_untouched(wire, ledger)


@pytest.mark.parametrize('expected', [None, '0' * 64])
def test_expected_plan_sha_mismatch_refuses(monkeypatch, expected):
    wire, sha, ledger, _ = setup(monkeypatch)
    with pytest.raises(RuntimeError, match='plan digest'):
        reconcile(wire, sha, apply=True, expected_plan_sha=expected)
    assert_untouched(wire, ledger)


def test_plan_digest_binds_the_ledger_it_was_planned_against(monkeypatch):
    wire, sha, _, _ = setup(monkeypatch)
    planned = reconcile(wire, sha)['plan_sha256']
    wire.docs['2026-11']['reservations']['late'] = settled_flash()
    ledger = copy.deepcopy(wire.docs['2026-11'])
    with pytest.raises(RuntimeError, match='plan digest'):
        reconcile(wire, sha, apply=True, expected_plan_sha=planned)
    assert_untouched(wire, ledger)


def test_cli_prints_one_line_without_private_identifiers(monkeypatch, capsys):
    wire, sha, _, _ = setup(monkeypatch)
    monkeypatch.setenv('SPEND_POLICY_SHA256', sha)
    usage = json.dumps(USAGE)
    assert hold.main(['--usage-json', usage], store=wire.store, now=NOW) == 0
    planned = json.loads(capsys.readouterr().out)
    assert set(planned) == REPORT_KEYS and planned['mode'] == 'plan' and not wire.posts
    assert hold.main(['--usage-json', usage, '--apply', '--expected-plan-sha', planned['plan_sha256']],
                     store=wire.store, now=NOW) == 0
    printed = capsys.readouterr().out
    assert len(printed.strip().splitlines()) == 1 and json.loads(printed)['mode'] == 'applied'
    record = wire.docs['2026-11']['reservations'][hold.RESERVATION_ID]
    for private in (GEMINI_FINGERPRINT, REQUEST_SHA, hold.PROVIDER_REQUEST_ID, record['billing_reference'],
                    record['usage_receipt']['evidence_sha256']):
        assert private not in printed


def test_cli_refusal_prints_the_error_and_writes_nothing(monkeypatch, capsys):
    wire, sha, ledger, _ = setup(monkeypatch)
    monkeypatch.setenv('SPEND_POLICY_SHA256', sha)
    assert hold.main(['--usage-json', '[1, 2]', '--apply'], store=wire.store, now=NOW) == 1
    assert set(json.loads(capsys.readouterr().out)) == {'error'}
    assert_untouched(wire, ledger)


def test_no_literal_dollar_sign_in_the_tool_or_its_tests():
    from pathlib import Path
    for path in (Path(hold.__file__), Path(__file__)):
        assert chr(36) not in path.read_text(), path
