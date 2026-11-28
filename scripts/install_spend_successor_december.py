#!/usr/bin/env python3
"""One-time install of the December successor spend policy, SIY-198 (plan by default, --apply to write).

Raises the cash ceiling to 30 USD with a fresh reload attestation of Google's 30 USD monthly
auto-charge limit, and prices 3.8 Flash at its 2027 rate through 2027-12-31, on top of the
installed Apify price-cushion successor (46c2c5c1). Moves the month ledger's policy pointer,
cash ceiling and Gemini limits, and records a durable receipt, in ONE conditional commit.
Reservations are never touched, and every hold must be settled first. A repeated or uncertain
dispatch reconciles by reading back, never by replaying.

The installed policy is checked by digest only: this release's validator refuses its 10 USD
reload shape, which is the point of the change.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from src.state_store import PublicationConflict, PublicationOutcomeUnknown
from src.zeitgeist.automatic_spend import digest, gemini_rate_excess, validate_policy
from src.zeitgeist.automatic_spend_runtime import DECEMBER_SUCCESSOR_RECEIPT_ID, PrivateFirestoreStore
from src.zeitgeist.gemini_tariff import observe_gemini_tariff
from src.zeitgeist.provider_lifecycle import APIFY_START_REQUEST_SECONDS
from src.zeitgeist.spend_guard import SpendBlocked
from src.zeitgeist.spend_successor import (
    DECEMBER_BASE_POLICY_SHA256, FLASH_OPERATION, assert_no_open_reservation, build_december_successor,
    ceiling_ledger, effective_until, field_diff)

CHANGE = ('cash ceiling 30 USD with a fresh 30 USD reload attestation; 3.8 Flash at its 2027 rate; '
          'policy and Gemini pricing approved through 2027-12-31')


def _observe_full(now):
    with httpx.Client(follow_redirects=False) as client:
        return observe_gemini_tariff(client, now=now)


def observe(*, now=None, observer=_observe_full):
    """The live public tariff digest and the 3.8 Flash rows it read; public numbers only."""
    observation = observer(now or datetime.now(timezone.utc))
    return {'tariff_sha256': observation['sha256'],
            'flash': observation['contract']['operations'][FLASH_OPERATION]}


def _request_window_seconds(policy):
    """The longest any provider request can stay in flight, including the Apify start POST."""
    longest = max(int(pinned['account']['max_request_duration_seconds'])
                  for pinned in policy['providers'].values())
    return longest + int(APIFY_START_REQUEST_SECONDS)


def _summary(policy):
    gemini = policy['providers']['gemini']
    flash = gemini['account']['request_limits'][FLASH_OPERATION]
    return {'cash_ceiling_usd': policy['cash_ceiling_usd'], 'valid_until': policy['valid_until'],
            'monthly_auto_charge_limit_usd': gemini['bounded_auto_reload_attestation']['monthly_auto_charge_limit_usd'],
            'flash_usd_per_million': [flash['input_token_usd_per_million'], flash['output_token_usd_per_million']],
            'flash_pricing_valid_until': flash['pricing_valid_until'],
            'flash_cap_usd': gemini['account']['operation_caps_usd'][FLASH_OPERATION],
            'effective_until': effective_until(policy)}


def _report(mode, old_sha, new_sha, policy_diff, ledger_diff, new, reservations_unchanged):
    return {'mode': mode, 'old_policy_sha256': old_sha, 'new_policy_sha256': new_sha,
            'receipt_id': DECEMBER_SUCCESSOR_RECEIPT_ID, 'policy_diff': policy_diff, 'ledger_diff': ledger_diff,
            'reservations_unchanged': reservations_unchanged, 'successor': _summary(new)}


def install(store, *, old_sha, tariff_sha256, reload_observation, approved_at, apply=False,
            expected_new_sha=None, now=None, observer=_observe_full):
    now = now or datetime.now(timezone.utc)
    old = store.read('policy_' + old_sha).to_dict()
    if digest(old) != old_sha:
        raise RuntimeError('Installed policy does not match its digest')
    inputs = {'tariff_sha256': tariff_sha256, 'reload_observation': reload_observation, 'approved_at': approved_at}
    new = build_december_successor(old, **inputs)
    new_sha = digest(new)
    validate_policy(new, new_sha, now)
    month = now.strftime('%Y-%m')
    policy_diff = field_diff(old, new)

    receipt_snapshot = store.read(DECEMBER_SUCCESSOR_RECEIPT_ID)
    if receipt_snapshot.exists:
        # Already installed: prove it from the durable receipt, without observing or writing.
        receipt = receipt_snapshot.to_dict()
        if (receipt.get('old_policy_sha256') != old_sha or receipt.get('new_policy_sha256') != new_sha
                or receipt.get('month') != month or receipt.get('inputs') != inputs
                or receipt.get('change') != CHANGE or receipt.get('schema_version') != 1):
            raise RuntimeError('Install receipt identity mismatch')
        current = store.read(month).to_dict()
        if (store.read('policy_' + new_sha).to_dict() != new
                or current.get('automatic_policy_sha256') != new_sha):
            raise RuntimeError('Install receipt requires reconciliation')
        return _report('already_applied', old_sha, new_sha, policy_diff,
                       field_diff(ceiling_ledger(current, old, old_sha), current), new,
                       digest(current) == receipt.get('after_sha256'))

    ledger_snapshot = store.read(month)
    ledger = ledger_snapshot.to_dict()
    if (ledger.get('automatic_policy_sha256') != old_sha or ledger.get('reconciliation_required')
            or ledger.get('closed_for_rollover') or ledger.get('month') != month):
        raise RuntimeError('Ledger changed or frozen; install must be replanned')
    if (now + timedelta(seconds=_request_window_seconds(old))).strftime('%Y-%m') != month:
        raise RuntimeError('Too close to the month boundary to install; retry in the new month')
    assert_no_open_reservation(ledger)

    observation = observer(now)
    if observation.get('sha256') != tariff_sha256:
        raise RuntimeError('Live price contract differs from the supplied tariff digest')
    # The daily refresh's own rate check: every approved operation is published and none above approval.
    gemini = new['providers']['gemini']
    if gemini_rate_excess(gemini['account'], gemini, observation, now) != []:
        raise RuntimeError('Published Gemini rates do not fit the successor policy')

    if store.read('policy_' + new_sha).exists:
        raise RuntimeError('Unexpected successor policy without an install receipt')
    after = ceiling_ledger(ledger, new, new_sha)
    unchanged = digest(after['reservations']) == digest(ledger['reservations'])
    receipt = {'schema_version': 1, 'old_policy_sha256': old_sha, 'new_policy_sha256': new_sha,
               'month': month, 'observed_at': now.isoformat(), 'observation': observation,
               'before_sha256': digest(ledger), 'after_sha256': digest(after),
               'change': CHANGE, 'inputs': inputs}
    ledger_diff = field_diff(ledger, after)
    if not apply:
        return _report('plan', old_sha, new_sha, policy_diff, ledger_diff, new, unchanged)
    if expected_new_sha != new_sha:
        raise RuntimeError('Apply requires the planned new policy digest')

    try:
        store.commit([('policy_' + new_sha, new, store.read('policy_' + new_sha)),
                      (month, after, ledger_snapshot),
                      (DECEMBER_SUCCESSOR_RECEIPT_ID, receipt, receipt_snapshot)])
    except PublicationOutcomeUnknown:
        pass
    except PublicationConflict:
        raise RuntimeError('Install rejected by a precondition; replan') from None
    final = store.read(month).to_dict()
    if (store.read(DECEMBER_SUCCESSOR_RECEIPT_ID).to_dict() != receipt or final != after
            or store.read('policy_' + new_sha).to_dict() != new):
        raise RuntimeError('Install outcome unresolved; read before any further action')
    return _report('applied', old_sha, new_sha, policy_diff, ledger_diff, new,
                   digest(final['reservations']) == digest(ledger['reservations']))


def readback(store, *, now=None):
    """Read-only proof the installed successor is active: receipt, policy, ledger and the validator at now."""
    now = now or datetime.now(timezone.utc)
    snapshot = store.read(DECEMBER_SUCCESSOR_RECEIPT_ID)
    if not snapshot.exists:
        raise RuntimeError('No December install receipt')
    new_sha = snapshot.to_dict().get('new_policy_sha256')
    policy = store.read('policy_' + str(new_sha)).to_dict()
    validate_policy(policy, new_sha, now)
    ledger = store.read(now.strftime('%Y-%m')).to_dict()
    gemini = ledger['providers']['gemini']
    active = (ledger.get('automatic_policy_sha256') == new_sha
              and ledger.get('cash_ceiling_usd') == policy['cash_ceiling_usd']
              and gemini.get('request_limits') == policy['providers']['gemini']['account']['request_limits'])
    return {'mode': 'readback', 'policy_sha256': new_sha, 'active': active, 'successor': _summary(policy)}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--observe-tariff', action='store_true')
    parser.add_argument('--readback', action='store_true')
    parser.add_argument('--tariff-sha256')
    parser.add_argument('--approved-at')
    parser.add_argument('--reload-observed-at')
    parser.add_argument('--reload-reference')
    parser.add_argument('--monthly-auto-charge-limit-usd')
    parser.add_argument('--reload-amount-usd')
    parser.add_argument('--trigger-balance-usd')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-new-sha')
    args = parser.parse_args(argv)
    try:
        if args.observe_tariff:
            result = observe()
        elif args.readback:
            result = readback(PrivateFirestoreStore())
        else:
            if os.environ.get('SPEND_POLICY_SHA256') != DECEMBER_BASE_POLICY_SHA256:
                raise RuntimeError('SPEND_POLICY_SHA256 is not the reviewed installed policy')
            reload_observation = {
                'observed_at': args.reload_observed_at, 'reference': args.reload_reference,
                'monthly_auto_charge_limit_usd': args.monthly_auto_charge_limit_usd,
                'reload_amount_usd': args.reload_amount_usd, 'trigger_balance_usd': args.trigger_balance_usd}
            if None in reload_observation.values() or args.tariff_sha256 is None or args.approved_at is None:
                raise RuntimeError('Tariff digest, approval time and every reload observation field are required')
            result = install(PrivateFirestoreStore(), old_sha=DECEMBER_BASE_POLICY_SHA256,
                             tariff_sha256=args.tariff_sha256, reload_observation=reload_observation,
                             approved_at=args.approved_at, apply=args.apply,
                             expected_new_sha=args.expected_new_sha)
    except (RuntimeError, SpendBlocked) as exc:
        print(json.dumps({'error': str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
