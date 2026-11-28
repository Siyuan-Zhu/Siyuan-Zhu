#!/usr/bin/env python3
"""Roll the Apify newest-safe-build successor back to the Neon successor (plan by default, --apply to write).

Moves the month ledger's policy pointer back to the policy named by the install receipt and records a durable
rollback receipt, in ONE conditional commit. Policy documents and the install receipt are never touched and
reservations are never rewritten. A repeated or uncertain dispatch reconciles by reading back, never by replaying.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.state_store import PublicationConflict, PublicationOutcomeUnknown
from src.zeitgeist.automatic_spend import digest, validate_policy
from src.zeitgeist.automatic_spend_runtime import (
    APIFY_ROLLBACK_RECEIPT_ID, APIFY_SUCCESSOR_RECEIPT_ID, PrivateFirestoreStore)
from src.zeitgeist.provider_lifecycle import APIFY_START_REQUEST_SECONDS
from src.zeitgeist.spend_successor import (
    APIFY_BASE_POLICY_SHA256, assert_no_hold_in_flight, field_diff, pointer_ledger)

_SHA = re.compile(r'[a-f0-9]{64}')


def _request_window_seconds(policy):
    """The longest any provider request can stay in flight, including the Apify start POST."""
    longest = max(int(pinned['account']['max_request_duration_seconds'])
                  for pinned in policy['providers'].values())
    return longest + int(APIFY_START_REQUEST_SECONDS)


def _open_count(ledger):
    return sum(1 for record in (ledger.get('reservations') or {}).values()
               if record.get('status') != 'settled')


def _plan_sha(before_sha, after_sha):
    return digest({'before_sha256': before_sha, 'after_sha256': after_sha})


def _report(mode, old_sha, new_sha, plan_sha, ledger_diff, ledger, reservations_unchanged):
    return {'mode': mode, 'from_policy_sha256': new_sha, 'to_policy_sha256': old_sha, 'plan_sha256': plan_sha,
            'ledger_diff': ledger_diff, 'open_reservation_count': _open_count(ledger),
            'reservations_unchanged': reservations_unchanged}


def _install_identity(store, month, required_old_sha):
    snapshot = store.read(APIFY_SUCCESSOR_RECEIPT_ID)
    if not snapshot.exists:
        raise RuntimeError('No install receipt; there is nothing to roll back')
    receipt = snapshot.to_dict()
    old_sha, new_sha = receipt.get('old_policy_sha256'), receipt.get('new_policy_sha256')
    if (receipt.get('schema_version') != 1 or receipt.get('month') != month
            or not _SHA.fullmatch(str(old_sha)) or not _SHA.fullmatch(str(new_sha)) or old_sha == new_sha):
        raise RuntimeError("Install receipt is not this month's successor install")
    if required_old_sha is not None and old_sha != required_old_sha:
        raise RuntimeError('Install receipt does not name the reviewed installed policy')
    return old_sha, new_sha


def rollback(store, *, apply=False, expected_plan_sha=None, now=None, required_old_sha=None):
    now = now or datetime.now(timezone.utc)
    month = now.strftime('%Y-%m')
    old_sha, new_sha = _install_identity(store, month, required_old_sha)
    receipt_snapshot = store.read(APIFY_ROLLBACK_RECEIPT_ID)
    if receipt_snapshot.exists:
        receipt = receipt_snapshot.to_dict()
        if (receipt.get('schema_version') != 1 or receipt.get('kind') != 'rollback'
                or receipt.get('old_policy_sha256') != new_sha or receipt.get('new_policy_sha256') != old_sha
                or receipt.get('install_receipt_id') != APIFY_SUCCESSOR_RECEIPT_ID or receipt.get('month') != month):
            raise RuntimeError('Rollback receipt identity mismatch')
        current = store.read(month).to_dict()
        if current.get('automatic_policy_sha256') != old_sha or not store.read('policy_' + old_sha).exists:
            raise RuntimeError('Rollback receipt requires reconciliation')
        return _report('already_applied', old_sha, new_sha,
                       _plan_sha(receipt.get('before_sha256'), receipt.get('after_sha256')),
                       [], current, digest(current) == receipt.get('after_sha256'))
    ledger_snapshot = store.read(month)
    ledger = ledger_snapshot.to_dict()
    if (ledger.get('automatic_policy_sha256') != new_sha or ledger.get('reconciliation_required')
            or ledger.get('closed_for_rollover') or ledger.get('month') != month):
        raise RuntimeError('Ledger changed or frozen; rollback must be replanned')
    old_snapshot, new_snapshot = store.read('policy_' + old_sha), store.read('policy_' + new_sha)
    if not old_snapshot.exists or not new_snapshot.exists:
        raise RuntimeError('Policy documents named by the install receipt are missing')
    old, new = old_snapshot.to_dict(), new_snapshot.to_dict()
    if digest(old) != old_sha or digest(new) != new_sha:
        raise RuntimeError('Policy documents do not match the install receipt digests')
    validate_policy(old, old_sha, now)
    window = _request_window_seconds(old)
    if (now + timedelta(seconds=window)).strftime('%Y-%m') != month:
        raise RuntimeError('Too close to the month boundary to roll back; retry in the new month')
    assert_no_hold_in_flight(ledger, now=now, duration_seconds=window)
    after = pointer_ledger(ledger, old_sha)
    plan_sha = _plan_sha(digest(ledger), digest(after))
    unchanged = digest(after['reservations']) == digest(ledger['reservations'])
    ledger_diff = field_diff(ledger, after)
    if not apply:
        return _report('plan', old_sha, new_sha, plan_sha, ledger_diff, ledger, unchanged)
    if expected_plan_sha != plan_sha:
        raise RuntimeError('Apply requires the planned plan digest')
    receipt = {'schema_version': 1, 'kind': 'rollback', 'old_policy_sha256': new_sha, 'new_policy_sha256': old_sha,
               'install_receipt_id': APIFY_SUCCESSOR_RECEIPT_ID, 'month': month, 'observed_at': now.isoformat(),
               'before_sha256': digest(ledger), 'after_sha256': digest(after)}
    try:
        store.commit([(month, after, ledger_snapshot), (APIFY_ROLLBACK_RECEIPT_ID, receipt, receipt_snapshot)])
    except PublicationOutcomeUnknown:
        pass
    except PublicationConflict:
        raise RuntimeError('Rollback rejected by a precondition; replan') from None
    final = store.read(month).to_dict()
    if store.read(APIFY_ROLLBACK_RECEIPT_ID).to_dict() != receipt or final != after:
        raise RuntimeError('Rollback outcome unresolved; read before any further action')
    return _report('applied', old_sha, new_sha, plan_sha, ledger_diff, final,
                   digest(final['reservations']) == digest(ledger['reservations']))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-plan-sha')
    args = parser.parse_args(argv)
    try:
        # As for the Neon rollback: the workflow pins move back first, so the run names the base policy.
        if os.environ.get('SPEND_POLICY_SHA256') != APIFY_BASE_POLICY_SHA256:
            raise RuntimeError('SPEND_POLICY_SHA256 is not the reviewed installed policy')
        result = rollback(PrivateFirestoreStore(), apply=args.apply, expected_plan_sha=args.expected_plan_sha,
                          required_old_sha=APIFY_BASE_POLICY_SHA256)
    except RuntimeError as exc:
        print(json.dumps({'error': str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
