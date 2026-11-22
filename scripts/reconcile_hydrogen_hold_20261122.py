#!/usr/bin/env python3
"""One-time reconcile of the frozen 2026-11-22 Hydrogen hold from the reply's own usage record.

The first private run's Director call on Hydrogen was answered, but the reply billed more prompt
tokens than the guard had admitted, so the guard froze the month on that reservation. The reply's
usageMetadata, given as --usage-json, must hash to the digest the freeze stored: it is proven to be
the refused reply's own record, never an estimate. The charge is the billed tokens at the rates the
hold recorded, at or below the hold's maximum, settled through the unchanged shared admission rule
(admit_usage_receipt). The freeze is then lifted on the record and, when nothing else is frozen and
the cash exposure fits the ceiling, on the ledger. Plan by default; --apply with the planned digest
makes one conditional commit of the ledger and a durable receipt. A repeated or uncertain dispatch
reconciles from that receipt.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.state_store import _commit_documents, PublicationConflict, PublicationOutcomeUnknown
from src.zeitgeist.automatic_guard import request_digest
from src.zeitgeist.automatic_spend import (
    admit_usage_receipt, cash_exposure, digest, validate_policy)
from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore
from src.zeitgeist.spend_guard import GENAI_RATE_KEYS, money

MONTH = '2026-11'
RESERVATION_ID = '7e885b47955a4c6b83d311676a6f9a7c'
OPERATION = 'generate_content:gemini-4-hydrogen'
CREATED_AT = '2026-11-22T11:22:54.431743+00:00'
PROVIDER_REQUEST_ID = 'kNACa_WzAbXniLMPidXE0QE'
MAX = '0.04608'
ADMITTED_INPUT_TOKENS = 720
MAX_OUTPUT_TOKENS = 6000
FREEZE_REASON = 'Gemini usage exceeds admitted token shape'
EVIDENCE_SHA256 = '8e77e41af7f68901e9467f6827090c460aa5d635736e726c03011a3d0b9c140b'
RECEIPT_ID = 'hydrogen_hold_reconcile_20261122'

_FROZEN_MARKERS = ('reconciliation_required', 'reconciliation_reasons', 'unreconciled_usage',
                   'usage_receipt_conflict')


def _billed(usage):
    """Billed prompt and output tokens, checked as gemini_usage_receipt checks them."""
    if not isinstance(usage, dict):
        raise RuntimeError('Usage must be the reply usage record object')
    if request_digest(usage) != EVIDENCE_SHA256:
        raise RuntimeError('Usage is not the refused reply usage record')
    counts = {}
    for field in ('promptTokenCount', 'candidatesTokenCount', 'totalTokenCount'):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise RuntimeError('Complete Gemini usage counts required')
        counts[field] = usage[field]
    for field in ('thoughtsTokenCount', 'toolUsePromptTokenCount', 'cachedContentTokenCount'):
        value = usage.get(field, 0)
        if type(value) is not int or value < 0:
            raise RuntimeError('Invalid Gemini usage count')
        counts[field] = value
    if (counts['toolUsePromptTokenCount'] or counts['cachedContentTokenCount'] > counts['promptTokenCount']
            or counts['totalTokenCount'] != (counts['promptTokenCount'] + counts['candidatesTokenCount']
                                             + counts['thoughtsTokenCount'])):
        raise RuntimeError('Tool use or inconsistent Gemini usage')
    return counts['promptTokenCount'], counts['candidatesTokenCount'] + counts['thoughtsTokenCount']


def _check_reservation(record, policy):
    if (not isinstance(record, dict) or record.get('status') != 'reserved' or record.get('provider') != 'gemini'
            or record.get('operation') != OPERATION or record.get('created_at') != CREATED_AT
            or record.get('provider_request_id') != PROVIDER_REQUEST_ID
            or not isinstance(record.get('max_gross_usd'), str) or money(record['max_gross_usd']) != money(MAX)
            or record.get('reconciliation_required') is not True
            or any(key in record for key in ('usage_receipt', 'actual_gross_usd', 'billing_reference',
                                             'reconciliation_reasons', 'usage_receipt_conflict',
                                             'reconciled_usage'))):
        raise RuntimeError('Reservation differs from the reviewed frozen hold')
    if record.get('unreconciled_usage') != {'reason': FREEZE_REASON, 'evidence_sha256': EVIDENCE_SHA256}:
        raise RuntimeError('Reservation freeze differs from the reviewed usage record')
    context = record.get('request_context')
    limits = policy['providers']['gemini']['account']['request_limits'].get(OPERATION, {})
    if (not isinstance(context, dict)
            or set(context) != {'admitted_input_tokens', 'max_output_tokens', *GENAI_RATE_KEYS}
            or type(context['admitted_input_tokens']) is not int
            or context['admitted_input_tokens'] != ADMITTED_INPUT_TOKENS
            or type(context['max_output_tokens']) is not int or context['max_output_tokens'] != MAX_OUTPUT_TOKENS):
        raise RuntimeError('Reservation token bounds differ from the reviewed hold')
    for side in ('input', 'output'):
        rate = money(limits.get(side + '_token_usd_per_million'))
        if (money(context[side + '_token_usd_per_million']) != rate
                or money(context['approved_' + side + '_token_usd_per_million']) != rate):
            raise RuntimeError('Reservation rates differ from the reviewed hold')
    bound = (money(ADMITTED_INPUT_TOKENS) * money(context['input_token_usd_per_million'])
             + money(MAX_OUTPUT_TOKENS) * money(context['output_token_usd_per_million'])) / 1000000
    if bound != money(MAX):
        raise RuntimeError('Reservation rates differ from the reviewed hold')
    return context


def _report(mode, before, after, prompt, output, amount, plan_sha):
    return {'mode': mode,
            'reservation_status_before': before['reservations'][RESERVATION_ID]['status'],
            'reservation_status_after': after['reservations'][RESERVATION_ID]['status'],
            'billed_prompt_tokens': prompt, 'billed_output_tokens': output,
            'accounted_gross_usd': amount, 'max_gross_usd': MAX,
            'ledger_reconciliation_required_before': bool(before.get('reconciliation_required')),
            'ledger_reconciliation_required_after': bool(after.get('reconciliation_required')),
            'plan_sha256': plan_sha}


def _reconciled(prior, ledger):
    """A durable receipt exists: it must still describe the ledger's settled reservation."""
    record = ledger.get('reservations', {}).get(RESERVATION_ID, {})
    receipt = prior.get('usage_receipt')
    if (prior.get('schema_version') != 1 or prior.get('reservation_id') != RESERVATION_ID
            or not isinstance(receipt, dict) or record.get('status') != 'settled'
            or record.get('usage_receipt') != receipt
            or record.get('billing_reference') != 'provider_usage_v1:' + digest(
                {k: v for k, v in receipt.items() if k != 'observed_at'})
            or record.get('actual_gross_usd') != prior.get('actual_gross_usd')
            or receipt.get('accounted_gross_usd') != prior.get('actual_gross_usd')
            or record.get('reconciled_usage') != prior.get('reconciled_usage')
            or any(key in record for key in _FROZEN_MARKERS)):
        raise RuntimeError('Hold receipt requires reconciliation')
    return _report('already_reconciled', ledger, ledger, prior['billed_prompt_tokens'],
                   prior['billed_output_tokens'], prior['actual_gross_usd'], prior['plan_sha256'])


def reconcile(store, policy_sha, usage, *, apply=False, expected_plan_sha=None, now=None):
    now = now or datetime.now(timezone.utc)
    prompt, output = _billed(usage)
    policy = store.read('policy_' + policy_sha).to_dict()
    validate_policy(policy, policy_sha, now)
    ledger_snapshot = store.read(MONTH)
    ledger = ledger_snapshot.to_dict()
    if ledger.get('automatic_policy_sha256') != policy_sha:
        raise RuntimeError('Ledger is under another spend policy')
    journal = store.client.collection('spend_control').document(RECEIPT_ID)
    prior = journal.get()
    if prior.exists:
        return _reconciled(prior.to_dict(), ledger)
    record = ledger.get('reservations', {}).get(RESERVATION_ID)
    context = _check_reservation(record, policy)
    for ident, other in ledger['reservations'].items():
        if ident != RESERVATION_ID and any(key in other for key in _FROZEN_MARKERS):
            raise RuntimeError('Another reservation requires reconciliation; the ledger stays frozen')
    amount = (money(prompt) * money(context['input_token_usd_per_million'])
              + money(output) * money(context['output_token_usd_per_million'])) / 1000000
    if amount > money(MAX):
        raise RuntimeError('Billed usage exceeds the hold maximum; the owner reconciles an overrun')
    receipt = {
        'schema_version': 1, 'accounting_policy': 'provider_usage_v1', 'reservation_id': RESERVATION_ID,
        'provider': 'gemini', 'operation': OPERATION,
        'credential_fingerprint': ledger['providers']['gemini']['credential_fingerprint'],
        'provider_request_id': record['provider_request_id'], 'request_sha256': record['request_sha256'],
        'observed_at': now.isoformat(), 'accounted_gross_usd': str(amount),
        'accounting_basis': 'conservative_response_usage',
        'evidence_sha256': digest({'response_id': record['provider_request_id'], 'usage': usage,
                                   'request_context': context})}
    candidate = admit_usage_receipt(ledger, policy, receipt, expected_policy_sha=policy_sha, now=now)
    settled = candidate['reservations'][RESERVATION_ID]
    untouched = {key: value for key, value in candidate.items() if key != 'reservations'}
    others = {key: value for key, value in candidate['reservations'].items() if key != RESERVATION_ID}
    if (settled.get('status') != 'settled' or 'reconciliation_reasons' in settled
            or 'usage_receipt_conflict' in settled
            or untouched != {key: value for key, value in ledger.items() if key != 'reservations'}
            or others != {key: value for key, value in ledger['reservations'].items() if key != RESERVATION_ID}):
        raise RuntimeError('Settlement would change more than the reviewed hold')
    after = copy.deepcopy(candidate)
    record_after = after['reservations'][RESERVATION_ID]
    del record_after['reconciliation_required']
    record_after['reconciled_usage'] = dict(record_after.pop('unreconciled_usage'),
                                            reconciled_at=now.isoformat(), receipt_id=RECEIPT_ID,
                                            billed_prompt_tokens=prompt, billed_output_tokens=output)
    after['reconciliation_required'] = False
    if cash_exposure(after) > money(policy['cash_ceiling_usd']):
        raise RuntimeError('Cash exposure exceeds the ceiling; the ledger stays frozen')
    amount_text = record_after['actual_gross_usd']
    # The plan binds the ledger read and the usage; only the observation time differs at apply.
    plan_sha = digest({'before_sha256': digest(ledger), 'usage': usage, 'accounted_gross_usd': amount_text,
                       'receipt_id': RECEIPT_ID})
    if not apply:
        return _report('plan', ledger, after, prompt, output, amount_text, plan_sha)
    if expected_plan_sha != plan_sha:
        raise RuntimeError('Apply requires the planned plan digest')
    document = {'schema_version': 1, 'reservation_id': RESERVATION_ID, 'month': MONTH,
                'policy_sha256': policy_sha, 'observed_at': now.isoformat(), 'usage': copy.deepcopy(usage),
                'billed_prompt_tokens': prompt, 'billed_output_tokens': output,
                'actual_gross_usd': amount_text, 'usage_receipt': receipt,
                'reconciled_usage': copy.deepcopy(record_after['reconciled_usage']),
                'plan_sha256': plan_sha, 'before_sha256': digest(ledger), 'after_sha256': digest(after)}
    try:
        _commit_documents(store.client, [('spend_control', MONTH, after, ledger_snapshot),
                                         ('spend_control', RECEIPT_ID, document, prior)])
    except PublicationOutcomeUnknown:
        pass
    except PublicationConflict:
        raise RuntimeError('Reconcile rejected by a precondition; replan') from None
    if journal.get().to_dict() != document or store.read(MONTH).to_dict() != after:
        raise RuntimeError('Hold outcome unresolved; read before any further action')
    return _report('applied', ledger, after, prompt, output, amount_text, plan_sha)


def main(argv=None, *, store=None, now=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--usage-json', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-plan-sha')
    args = parser.parse_args(argv)
    try:
        try:
            usage = json.loads(args.usage_json)
        except ValueError:
            raise RuntimeError('Usage must be the reply usage record object') from None
        result = reconcile(store or PrivateFirestoreStore(), os.environ.get('SPEND_POLICY_SHA256', ''), usage,
                           apply=args.apply, expected_plan_sha=args.expected_plan_sha, now=now)
    except RuntimeError as exc:
        print(json.dumps({'error': str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
