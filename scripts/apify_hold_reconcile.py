"""Settlement of one stale Apify hold from Apify's own run record (shared by the dated scripts).

A hold reserved and never settled is counted at its full bound. This reads the same
authenticated run twice, requires a terminal, stable cost inside the reserved bound for the
verified account, and settles through the unchanged shared admission rule (admit_usage_receipt),
which binds the receipt to the ledger account's current credential. The reservation's own
creation facts stay as created. If Apify has purged the run (404 and the account's oldest
retained run began after the hold was created) the cost can never be proven, so the full
reserved bound is booked with conservative accounting. One conditional commit writes the ledger and a
durable receipt; a repeated or uncertain dispatch reconciles from that receipt.

Each dated script names its one reviewed hold as a Hold; nothing here chooses a hold.
"""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import os
import httpx
from src.state_store import _commit_documents, PublicationOutcomeUnknown
from src.zeitgeist import provider_lifecycle as lifecycle
from src.zeitgeist.automatic_spend import _time, admit_usage_receipt, digest, validate_policy
from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore

OPERATION = 'instagram-post-scraper'
PURGED_BASIS = 'provider_record_purged_full_reserved_bound'
PURGED_STATUS = 'purged_by_provider_retention'


@dataclass(frozen=True)
class Hold:
    """The reviewed facts of one stale hold and where its durable receipt lives."""
    reservation_id: str
    run_id: str
    created_at: str
    max_gross_usd: str
    receipt_id: str
    month: str


class RunNotFound(Exception):
    """Apify answered 404 for the run itself (not any other HTTP failure)."""


def _reference(receipt):
    return 'provider_usage_v1:' + digest({k: v for k, v in receipt.items() if k != 'observed_at'})


def _result(status, amount, receipt):
    basis = 'full_bound_purged' if receipt.get('completion_basis') == PURGED_BASIS else 'run_usage'
    return {'status': status, 'actual_gross_usd': amount, 'basis': basis}


def _reconciled(hold, prior, ledger, status):
    """A durable receipt exists: it must still describe the ledger's settled reservation."""
    record = ledger.get('reservations', {}).get(hold.reservation_id, {})
    receipt = prior.get('usage_receipt')
    if (prior.get('reservation_id') != hold.reservation_id or prior.get('run_id') != hold.run_id
            or not isinstance(receipt, dict) or record.get('status') != 'settled'
            or record.get('usage_receipt') != receipt
            or record.get('billing_reference') != _reference(receipt)
            or record.get('actual_gross_usd') != prior.get('actual_gross_usd')
            or receipt.get('accounted_gross_usd') != prior.get('actual_gross_usd')):
        raise RuntimeError('Hold receipt requires reconciliation')
    return _result(status, prior['actual_gross_usd'], receipt)


def _check_reservation(hold, record):
    if (not isinstance(record, dict) or record.get('status') != 'reserved' or record.get('provider') != 'apify'
            or record.get('operation') != OPERATION or record.get('provider_request_id') != hold.run_id
            or record.get('created_at') != hold.created_at or record.get('max_gross_usd') != hold.max_gross_usd
            or record.get('credit_excluded') or record.get('billing_reference')
            or 'usage_receipt' in record or 'actual_gross_usd' in record):
        raise RuntimeError('Reservation differs from the reviewed stale hold')
    # The submission's own account and build are the pins; an absent pin is never a skipped check.
    metadata = record.get('submission_metadata') or {}
    if not metadata.get('user_id') or not metadata.get('build'):
        raise RuntimeError('Reservation submission metadata lacks the account or build pin')


def _purge_proof(hold, oldest_retained_run, policy):
    """The account's oldest retained run (same account) began after this hold was created."""
    if oldest_retained_run is None:
        raise RuntimeError('Run is missing and retention cannot be proven')
    oldest = oldest_retained_run()
    if not isinstance(oldest, dict):
        raise RuntimeError('Run is missing and retention cannot be proven')
    user = oldest.get('userId')
    if (not isinstance(user, str) or not user or hashlib.sha256(user.encode()).hexdigest()[:16]
            != policy['providers']['apify'].get('account_fingerprint')):
        raise RuntimeError('Retained run belongs to a different Apify account')
    started = oldest.get('startedAt')
    if not isinstance(started, str) or not started or _time(started) <= _time(hold.created_at):
        raise RuntimeError('Run is missing but retention does not cover the hold')
    return started


def _confirmed_cost(hold, policy, record, first, second):
    metadata = record.get('submission_metadata') or {}
    for run in (first, second):
        lifecycle._run_identity(run, run_id=hold.run_id, user_id=metadata['user_id'], build=metadata['build'])
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
    if cost > Decimal(hold.max_gross_usd):
        raise RuntimeError('Apify cost exceeds the reserved bound')
    return cost


def reconcile(hold, store, policy_sha, *, fetch_run, reader_fingerprint=None, oldest_retained_run_started_at=None,
              apply=False, now=None):
    now = now or datetime.now(timezone.utc)
    policy = store.read('policy_' + policy_sha).to_dict()
    validate_policy(policy, policy_sha, now)
    journal = store.client.collection('spend_control').document(hold.receipt_id)
    prior = journal.get()
    ledger_snapshot = store.read(hold.month)
    ledger = ledger_snapshot.to_dict()
    status = 'applied_or_reconciled' if apply else 'planned'
    if prior.exists:
        return _reconciled(hold, prior.to_dict(), ledger, status)
    record = ledger.get('reservations', {}).get(hold.reservation_id)
    _check_reservation(hold, record)
    # The receipt claims the ledger's current credential; the token that reads the run must be it.
    if (not reader_fingerprint
            or reader_fingerprint != ledger['providers']['apify'].get('credential_fingerprint')):
        raise RuntimeError('Reading credential is not the ledger account credential')
    try:
        first = fetch_run(hold.run_id)
    except RunNotFound:
        # Provider retention deleted the run: the cost can never be proven, so book the full bound.
        # A second read must agree; any other answer, including an error, leaves the hold untouched.
        try:
            fetch_run(hold.run_id)
        except RunNotFound:
            pass
        else:
            raise RuntimeError('Run presence is inconsistent; not treated as purged')
        oldest = _purge_proof(hold, oldest_retained_run_started_at, policy)
        cost = Decimal(hold.max_gross_usd)
        evidence = {'id': hold.run_id, 'status': PURGED_STATUS, 'oldest_retained_run_started_at': oldest,
                    'reserved_gross_usd': hold.max_gross_usd}
        basis = {'accounting_basis': 'conservative_response_usage', 'completion_basis': PURGED_BASIS}
        facts = {'run_status': PURGED_STATUS, 'oldest_retained_run_started_at': oldest}
    else:
        second = fetch_run(hold.run_id)
        cost = _confirmed_cost(hold, policy, record, first, second)
        evidence = {key: second.get(key) for key in ('id', 'status', 'usageTotalUsd', 'finishedAt')}
        basis = {'accounting_basis': 'provider_settled_cost'}
        facts = {'run_status': str(second['status']), 'finished_at': str(second['finishedAt'])}
    receipt = {
        'schema_version': 1, 'accounting_policy': 'provider_usage_v1',
        'reservation_id': hold.reservation_id, 'provider': 'apify', 'operation': record['operation'],
        # Admission binds a receipt to the ledger account's CURRENT credential; the reservation keeps
        # the credential and policy it was created under.
        'credential_fingerprint': ledger['providers']['apify']['credential_fingerprint'],
        'provider_request_id': hold.run_id, 'request_sha256': record['request_sha256'],
        'observed_at': now.isoformat(), 'accounted_gross_usd': str(cost),
        'evidence_sha256': lifecycle.request_digest(evidence), **basis}
    after = admit_usage_receipt(ledger, policy, receipt, expected_policy_sha=policy_sha, now=now)
    untouched = {key: value for key, value in after.items() if key != 'reservations'}
    others = {key: value for key, value in after['reservations'].items() if key != hold.reservation_id}
    if (untouched != {key: value for key, value in ledger.items() if key != 'reservations'}
            or others != {key: value for key, value in ledger['reservations'].items() if key != hold.reservation_id}
            or after['reservations'][hold.reservation_id].get('reconciliation_required')):
        raise RuntimeError('Settlement would change more than the reviewed hold')
    amount = after['reservations'][hold.reservation_id]['actual_gross_usd']
    document = {'schema_version': 1, 'reservation_id': hold.reservation_id, 'run_id': hold.run_id,
                'month': hold.month, 'policy_sha256': policy_sha, 'observed_at': receipt['observed_at'],
                'reservation_credential_fingerprint': record['credential_fingerprint'], **facts,
                'actual_gross_usd': amount, 'usage_receipt': receipt,
                'before_sha256': digest(ledger), 'after_sha256': digest(after)}
    if apply:
        try:
            _commit_documents(store.client, [('spend_control', hold.month, after, ledger_snapshot),
                                             ('spend_control', hold.receipt_id, document, prior)])
        except PublicationOutcomeUnknown:
            pass
        if journal.get().to_dict() != document or store.read(hold.month).to_dict() != after:
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
    """The account's oldest retained run: its start time and owner, for the purge proof."""
    def oldest():
        with httpx.Client(follow_redirects=False, timeout=20) as client:
            items = lifecycle._run_body(client.get(
                'https://api.apify.com/v2/actor-runs', params={'desc': '0', 'limit': '1'},
                headers={'Authorization': 'Bearer ' + token})).get('items')
        if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
            raise RuntimeError('No retained Apify run to prove retention')
        return {'startedAt': items[0].get('startedAt'), 'userId': items[0].get('userId')}
    return oldest


def main(hold, argv=None, *, store=None, fetch_run=None, reader_fingerprint=None,
         oldest_retained_run_started_at=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    if fetch_run is None:
        token = os.environ['APIFY_TOKEN']
        fetch_run = apify_fetch(token)
        oldest_retained_run_started_at = apify_oldest_started_at(token)
        reader_fingerprint = hashlib.sha256(token.encode()).hexdigest()[:16]
    result = reconcile(hold, store or PrivateFirestoreStore(), os.environ['SPEND_POLICY_SHA256'],
                       fetch_run=fetch_run, reader_fingerprint=reader_fingerprint,
                       oldest_retained_run_started_at=oldest_retained_run_started_at, apply=args.apply)
    print(json.dumps(result))
    return 0
