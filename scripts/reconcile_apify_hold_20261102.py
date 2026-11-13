#!/usr/bin/env python3
"""One-time settlement of the stale 2026-11-02 Apify hold from Apify's own run record.

The hold was reserved under an earlier policy and the previous credential and never
settled, so it is counted at its full bound and becomes credit-excluded when the
usage cycle rolls over. This reads the same authenticated run twice, requires a
terminal, stable cost inside the reserved bound for the verified account, and
settles through the unchanged shared admission rule (admit_usage_receipt), which
binds the receipt to the ledger account's current credential. The reservation's
own creation facts stay as created. If Apify has purged the run (404 and the account's oldest
retained run began after the hold was created) the cost can never be proven, so the full
reserved bound is booked with conservative accounting. One conditional commit writes the ledger and a
durable receipt; a repeated or uncertain dispatch reconciles from that receipt.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from src.state_store import _commit_documents, PublicationOutcomeUnknown
from src.zeitgeist import provider_lifecycle as lifecycle
from src.zeitgeist.automatic_spend import _time, admit_usage_receipt, digest, validate_policy
from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore

RESERVATION_ID = 'c14e610a08cb4b15b0ef19bc2e667849'
RUN_ID = 'Q3u3NuwtmLHJqN7g5'
CREATED_AT = '2026-11-02T07:16:32.713319+00:00'
MAX = '0.16'
RECEIPT_ID = 'apify_hold_reconcile_20261102'
MONTH = '2026-11'
OPERATION = 'instagram-post-scraper'
PURGED_BASIS = 'provider_record_purged_full_reserved_bound'
PURGED_STATUS = 'purged_by_provider_retention'


class RunNotFound(Exception):
    """Apify answered 404 for the run itself (not any other HTTP failure)."""


def _reference(receipt):
    return 'provider_usage_v1:' + digest({k: v for k, v in receipt.items() if k != 'observed_at'})


def _result(status, amount, receipt):
    basis = 'full_bound_purged' if receipt.get('completion_basis') == PURGED_BASIS else 'run_usage'
    return {'status': status, 'actual_gross_usd': amount, 'basis': basis}


def _reconciled(prior, ledger, status):
    """A durable receipt exists: it must still describe the ledger's settled reservation."""
    record = ledger.get('reservations', {}).get(RESERVATION_ID, {})
    receipt = prior.get('usage_receipt')
    if (prior.get('reservation_id') != RESERVATION_ID or prior.get('run_id') != RUN_ID
            or not isinstance(receipt, dict) or record.get('status') != 'settled'
            or record.get('usage_receipt') != receipt
            or record.get('billing_reference') != _reference(receipt)
            or record.get('actual_gross_usd') != prior.get('actual_gross_usd')
            or receipt.get('accounted_gross_usd') != prior.get('actual_gross_usd')):
        raise RuntimeError('Hold receipt requires reconciliation')
    return _result(status, prior['actual_gross_usd'], receipt)


def _check_reservation(record):
    if (not isinstance(record, dict) or record.get('status') != 'reserved' or record.get('provider') != 'apify'
            or record.get('operation') != OPERATION or record.get('provider_request_id') != RUN_ID
            or record.get('created_at') != CREATED_AT or record.get('max_gross_usd') != MAX
            or record.get('credit_excluded') or record.get('billing_reference')
            or 'usage_receipt' in record or 'actual_gross_usd' in record):
        raise RuntimeError('Reservation differs from the reviewed stale hold')
    # The submission's own account and build are the pins; an absent pin is never a skipped check.
    metadata = record.get('submission_metadata') or {}
    if not metadata.get('user_id') or not metadata.get('build'):
        raise RuntimeError('Reservation submission metadata lacks the account or build pin')


def _purge_proof(oldest_retained_run_started_at):
    """The account's oldest retained run began after this hold was created."""
    if oldest_retained_run_started_at is None:
        raise RuntimeError('Run is missing and retention cannot be proven')
    oldest = oldest_retained_run_started_at()
    if not isinstance(oldest, str) or not oldest or _time(oldest) <= _time(CREATED_AT):
        raise RuntimeError('Run is missing but retention does not cover the hold')
    return oldest


def _confirmed_cost(policy, record, first, second):
    metadata = record.get('submission_metadata') or {}
    for run in (first, second):
        lifecycle._run_identity(run, run_id=RUN_ID, user_id=metadata['user_id'], build=metadata['build'])
        if run.get('status') not in lifecycle.APIFY_TERMINAL:
            raise RuntimeError('Apify run is not terminal')
        owner = hashlib.sha256(run['userId'].encode()).hexdigest()[:16]
        if owner != policy['providers']['apify'].get('account_fingerprint'):
            raise RuntimeError('Run belongs to a different Apify account')
    if first['status'] != second['status']:
        raise RuntimeError('Apify run status is not stable')
    cost = lifecycle._stable_apify_cost(first, second)
    if cost is None:
        raise RuntimeError('Apify cost is not stable and final')
    if cost > Decimal(MAX):
        raise RuntimeError('Apify cost exceeds the reserved bound')
    return cost


def reconcile(store, policy_sha, *, fetch_run, reader_fingerprint=None, oldest_retained_run_started_at=None,
              apply=False, now=None):
    now = now or datetime.now(timezone.utc)
    policy = store.read('policy_' + policy_sha).to_dict()
    validate_policy(policy, policy_sha, now)
    journal = store.client.collection('spend_control').document(RECEIPT_ID)
    prior = journal.get()
    ledger_snapshot = store.read(MONTH)
    ledger = ledger_snapshot.to_dict()
    status = 'applied_or_reconciled' if apply else 'planned'
    if prior.exists:
        return _reconciled(prior.to_dict(), ledger, status)
    record = ledger.get('reservations', {}).get(RESERVATION_ID)
    _check_reservation(record)
    # The receipt claims the ledger's current credential; the token that reads the run must be it.
    if (not reader_fingerprint
            or reader_fingerprint != ledger['providers']['apify'].get('credential_fingerprint')):
        raise RuntimeError('Reading credential is not the ledger account credential')
    try:
        first = fetch_run(RUN_ID)
    except RunNotFound:
        # Provider retention deleted the run: the cost can never be proven, so book the full bound.
        oldest = _purge_proof(oldest_retained_run_started_at)
        cost = Decimal(MAX)
        evidence = {'id': RUN_ID, 'status': PURGED_STATUS, 'oldest_retained_run_started_at': oldest,
                    'reserved_gross_usd': MAX}
        basis = {'accounting_basis': 'conservative_response_usage', 'completion_basis': PURGED_BASIS}
        facts = {'run_status': PURGED_STATUS, 'oldest_retained_run_started_at': oldest}
    else:
        second = fetch_run(RUN_ID)
        cost = _confirmed_cost(policy, record, first, second)
        evidence = {key: second.get(key) for key in ('id', 'status', 'usageTotalUsd', 'finishedAt')}
        basis = {'accounting_basis': 'provider_settled_cost'}
        facts = {'run_status': str(second['status']), 'finished_at': str(second['finishedAt'])}
    receipt = {
        'schema_version': 1, 'accounting_policy': 'provider_usage_v1',
        'reservation_id': RESERVATION_ID, 'provider': 'apify', 'operation': record['operation'],
        # Admission binds a receipt to the ledger account's CURRENT credential; the reservation keeps
        # the credential and policy it was created under.
        'credential_fingerprint': ledger['providers']['apify']['credential_fingerprint'],
        'provider_request_id': RUN_ID, 'request_sha256': record['request_sha256'],
        'observed_at': now.isoformat(), 'accounted_gross_usd': str(cost),
        'evidence_sha256': lifecycle.request_digest(evidence), **basis}
    after = admit_usage_receipt(ledger, policy, receipt, expected_policy_sha=policy_sha, now=now)
    untouched = {key: value for key, value in after.items() if key != 'reservations'}
    others = {key: value for key, value in after['reservations'].items() if key != RESERVATION_ID}
    if (untouched != {key: value for key, value in ledger.items() if key != 'reservations'}
            or others != {key: value for key, value in ledger['reservations'].items() if key != RESERVATION_ID}
            or after['reservations'][RESERVATION_ID].get('reconciliation_required')):
        raise RuntimeError('Settlement would change more than the reviewed hold')
    amount = after['reservations'][RESERVATION_ID]['actual_gross_usd']
    document = {'schema_version': 1, 'reservation_id': RESERVATION_ID, 'run_id': RUN_ID, 'month': MONTH,
                'policy_sha256': policy_sha, 'observed_at': receipt['observed_at'],
                'reservation_credential_fingerprint': record['credential_fingerprint'], **facts,
                'actual_gross_usd': amount, 'usage_receipt': receipt,
                'before_sha256': digest(ledger), 'after_sha256': digest(after)}
    if apply:
        try:
            _commit_documents(store.client, [('spend_control', MONTH, after, ledger_snapshot),
                                             ('spend_control', RECEIPT_ID, document, prior)])
        except PublicationOutcomeUnknown:
            pass
        if journal.get().to_dict() != document or store.read(MONTH).to_dict() != after:
            raise RuntimeError('Hold outcome unresolved; read before any further action')
    return _result(status, amount, receipt)


def apify_fetch(token):
    def fetch_run(run_id):
        with httpx.Client(follow_redirects=False, timeout=20) as client:
            response = client.get('https://api.apify.com/v2/actor-runs/' + run_id,
                                  headers={'Authorization': 'Bearer ' + token})
            if response.status_code == 404:
                raise RunNotFound()
            return lifecycle._run_body(response)
    return fetch_run


def apify_oldest_started_at(token):
    def oldest():
        with httpx.Client(follow_redirects=False, timeout=20) as client:
            items = lifecycle._run_body(client.get(
                'https://api.apify.com/v2/actor-runs', params={'desc': '0', 'limit': '1'},
                headers={'Authorization': 'Bearer ' + token})).get('items')
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
            raise RuntimeError('No retained Apify run to prove retention')
        return items[0].get('startedAt')
    return oldest


def main(argv=None, *, store=None, fetch_run=None, reader_fingerprint=None, oldest_retained_run_started_at=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if fetch_run is None:
        token = os.environ['APIFY_TOKEN']
        fetch_run = apify_fetch(token)
        oldest_retained_run_started_at = apify_oldest_started_at(token)
        reader_fingerprint = hashlib.sha256(token.encode()).hexdigest()[:16]
    result = reconcile(store or PrivateFirestoreStore(), os.environ['SPEND_POLICY_SHA256'],
                       fetch_run=fetch_run, reader_fingerprint=reader_fingerprint,
                       oldest_retained_run_started_at=oldest_retained_run_started_at, apply=args.apply)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    sys.exit(main())
