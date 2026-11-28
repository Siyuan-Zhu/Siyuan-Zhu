#!/usr/bin/env python3
"""One-time install of the Apify price-cushion successor spend policy (plan by default, --apply to write). SIY-291.

Adds exactly providers.apify.actor_pricing_terms (the approved price terms, whose digest is the existing pricing pin)
to the installed newest-safe-build successor (eeecd6d8), moves the month ledger's policy pointer, and records a
durable receipt, in ONE conditional commit. Prices, limits, the cash ceiling and every reservation stay
byte-identical. A repeated or uncertain dispatch reconciles by reading back, never by replaying.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.state_store import PublicationConflict, PublicationOutcomeUnknown
from src.zeitgeist.automatic_spend import digest, validate_policy
from src.zeitgeist.automatic_spend_runtime import APIFY_CUSHION_RECEIPT_ID, PrivateFirestoreStore
from src.zeitgeist.provider_lifecycle import APIFY_START_REQUEST_SECONDS
from src.zeitgeist.spend_successor import (
    APIFY_CUSHION_BASE_POLICY_SHA256, assert_no_hold_in_flight, build_apify_price_cushion_successor, field_diff,
    pointer_ledger)

CHANGE = 'providers.apify: actor_pricing_terms approved (30% price band)'


def _request_window_seconds(policy):
    """The longest any provider request can stay in flight, including the Apify start POST."""
    longest = max(int(pinned['account']['max_request_duration_seconds'])
                  for pinned in policy['providers'].values())
    return longest + int(APIFY_START_REQUEST_SECONDS)


def _open_count(ledger):
    return sum(1 for record in (ledger.get('reservations') or {}).values()
               if record.get('status') != 'settled')


def _report(mode, old_sha, new_sha, policy_diff, ledger_diff, ledger, reservations_unchanged):
    return {'mode': mode, 'old_policy_sha256': old_sha, 'new_policy_sha256': new_sha,
            'receipt_id': APIFY_CUSHION_RECEIPT_ID, 'policy_diff': policy_diff, 'ledger_diff': ledger_diff,
            'open_reservation_count': _open_count(ledger), 'reservations_unchanged': reservations_unchanged}


def install(store, *, old_sha, apply=False, expected_new_sha=None, now=None):
    now = now or datetime.now(timezone.utc)
    old = store.read('policy_' + old_sha).to_dict()
    validate_policy(old, old_sha, now)
    new = build_apify_price_cushion_successor(old)
    new_sha = digest(new)
    validate_policy(new, new_sha, now)
    month = now.strftime('%Y-%m')
    policy_diff = field_diff(old, new)

    receipt_snapshot = store.read(APIFY_CUSHION_RECEIPT_ID)
    if receipt_snapshot.exists:
        # Already installed: prove it from the durable receipt, without writing.
        receipt = receipt_snapshot.to_dict()
        if (receipt.get('old_policy_sha256') != old_sha or receipt.get('new_policy_sha256') != new_sha
                or receipt.get('month') != month or receipt.get('change') != CHANGE
                or receipt.get('schema_version') != 1):
            raise RuntimeError('Install receipt identity mismatch')
        current = store.read(month).to_dict()
        if (store.read('policy_' + new_sha).to_dict() != new
                or current.get('automatic_policy_sha256') != new_sha):
            raise RuntimeError('Install receipt requires reconciliation')
        return _report('already_applied', old_sha, new_sha, policy_diff,
                       field_diff(pointer_ledger(current, old_sha), current), current,
                       digest(current) == receipt.get('after_sha256'))

    ledger_snapshot = store.read(month)
    ledger = ledger_snapshot.to_dict()
    if (ledger.get('automatic_policy_sha256') != old_sha or ledger.get('reconciliation_required')
            or ledger.get('closed_for_rollover') or ledger.get('month') != month):
        raise RuntimeError('Ledger changed or frozen; install must be replanned')
    window = _request_window_seconds(old)
    if (now + timedelta(seconds=window)).strftime('%Y-%m') != month:
        raise RuntimeError('Too close to the month boundary to install; retry in the new month')
    assert_no_hold_in_flight(ledger, now=now, duration_seconds=window)
    if store.read('policy_' + new_sha).exists:
        raise RuntimeError('Unexpected successor policy without an install receipt')
    after = pointer_ledger(ledger, new_sha)
    unchanged = digest(after['reservations']) == digest(ledger['reservations'])
    receipt = {'schema_version': 1, 'old_policy_sha256': old_sha, 'new_policy_sha256': new_sha,
               'month': month, 'observed_at': now.isoformat(),
               'before_sha256': digest(ledger), 'after_sha256': digest(after), 'change': CHANGE}
    ledger_diff = field_diff(ledger, after)
    if not apply:
        return _report('plan', old_sha, new_sha, policy_diff, ledger_diff, ledger, unchanged)
    if expected_new_sha != new_sha:
        raise RuntimeError('Apply requires the planned new policy digest')
    try:
        store.commit([('policy_' + new_sha, new, store.read('policy_' + new_sha)),
                      (month, after, ledger_snapshot),
                      (APIFY_CUSHION_RECEIPT_ID, receipt, receipt_snapshot)])
    except PublicationOutcomeUnknown:
        pass
    except PublicationConflict:
        raise RuntimeError('Install rejected by a precondition; replan') from None
    final = store.read(month).to_dict()
    if (store.read(APIFY_CUSHION_RECEIPT_ID).to_dict() != receipt or final != after
            or store.read('policy_' + new_sha).to_dict() != new):
        raise RuntimeError('Install outcome unresolved; read before any further action')
    return _report('applied', old_sha, new_sha, policy_diff, ledger_diff, final,
                   digest(final['reservations']) == digest(ledger['reservations']))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-new-sha')
    args = parser.parse_args(argv)
    try:
        if os.environ.get('SPEND_POLICY_SHA256') != APIFY_CUSHION_BASE_POLICY_SHA256:
            raise RuntimeError('SPEND_POLICY_SHA256 is not the reviewed installed policy')
        result = install(PrivateFirestoreStore(), old_sha=APIFY_CUSHION_BASE_POLICY_SHA256, apply=args.apply,
                         expected_new_sha=args.expected_new_sha)
    except RuntimeError as exc:
        print(json.dumps({'error': str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
