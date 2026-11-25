#!/usr/bin/env python3
"""One-time install of the Neon successor spend policy (plan by default, --apply to write).

Adds exactly the gemini-4-neon operation and the observed tariff digest to the installed
Hydrogen successor (6657e657), moves the month ledger's policy pointer and static Gemini limits,
and records a durable receipt, in ONE conditional commit. Every existing operation, the cash
ceiling, the approval reference and the reload attestation stay byte-identical. Reservations are
never touched. A repeated or uncertain dispatch reconciles by reading back, never by replaying.
"""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx
from scripts.repair_gemini_price_record import check_current_rates
from src.state_store import PublicationConflict, PublicationOutcomeUnknown
from src.zeitgeist.automatic_spend import digest, validate_policy
from src.zeitgeist.automatic_spend_runtime import NEON_SUCCESSOR_RECEIPT_ID, PrivateFirestoreStore
from src.zeitgeist.gemini_tariff import observe_gemini_tariff
from src.zeitgeist.provider_lifecycle import APIFY_START_REQUEST_SECONDS
from src.zeitgeist.spend_guard import NEON_OUTPUT_TOKENS
from src.zeitgeist.spend_successor import (
    NEON_BASE_POLICY_SHA256, NEON_BUILD_THINKING_LEVELS, assert_no_hold_in_flight, build_neon_successor,
    field_diff, rollback_ledger, successor_ledger)

CHANGE = 'added operations only: generate_content:gemini-4-neon'


def _observe_full(now):
    with httpx.Client(follow_redirects=False) as client:
        return observe_gemini_tariff(client, now=now)


def observe(*, now=None):
    """The live public tariff digest and per-operation rates; public numbers only."""
    now = now or datetime.now(timezone.utc)
    observation = _observe_full(now)
    rates = {}
    for operation, row in sorted(((observation.get('contract') or {}).get('operations') or {}).items()):
        output = row.get('output_token_usd_per_million')
        if output is None and row.get('billing_basis') == 'input_tokens':
            output = '0'
        rates[operation] = {'input': row.get('input_token_usd_per_million'), 'output': output}
    return {'tariff_sha256': observation['sha256'], 'rates': rates}


def _request_window_seconds(policy):
    """The longest any provider request can stay in flight, including the Apify start POST."""
    longest = max(int(pinned['account']['max_request_duration_seconds'])
                  for pinned in policy['providers'].values())
    return longest + int(APIFY_START_REQUEST_SECONDS)


def _open_count(ledger):
    return sum(1 for record in (ledger.get('reservations') or {}).values()
               if record.get('status') != 'settled')


def _report(mode, old_sha, new_sha, policy_diff, ledger_diff, ledger, reservations_unchanged,
            neon_max_output_tokens, neon_thinking_level):
    return {'mode': mode, 'old_policy_sha256': old_sha, 'new_policy_sha256': new_sha,
            'receipt_id': NEON_SUCCESSOR_RECEIPT_ID, 'policy_diff': policy_diff,
            'ledger_diff': ledger_diff, 'open_reservation_count': _open_count(ledger),
            'reservations_unchanged': reservations_unchanged,
            'neon_max_output_tokens': neon_max_output_tokens,
            'neon_thinking_level': neon_thinking_level}


def install(store, *, old_sha, neon_model_version, neon_tokens_per_image, neon_max_output_tokens,
            neon_thinking_level, tariff_sha256, apply=False, expected_new_sha=None, now=None):
    now = now or datetime.now(timezone.utc)
    old = store.read('policy_' + old_sha).to_dict()
    validate_policy(old, old_sha, now)
    inputs = {'neon_model_version': neon_model_version, 'neon_tokens_per_image': neon_tokens_per_image,
              'neon_max_output_tokens': neon_max_output_tokens, 'neon_thinking_level': neon_thinking_level,
              'tariff_sha256': tariff_sha256}
    new = build_neon_successor(old, **inputs)
    new_sha = digest(new)
    validate_policy(new, new_sha, now)
    month = now.strftime('%Y-%m')
    policy_diff = field_diff(old, new)

    receipt_snapshot = store.read(NEON_SUCCESSOR_RECEIPT_ID)
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
        # True only while the ledger is still exactly the install's after-image.
        return _report('already_applied', old_sha, new_sha, policy_diff,
                       field_diff(rollback_ledger(current, old, old_sha), current), current,
                       digest(current) == receipt.get('after_sha256'),
                       neon_max_output_tokens, neon_thinking_level)

    ledger_snapshot = store.read(month)
    ledger = ledger_snapshot.to_dict()
    if (ledger.get('automatic_policy_sha256') != old_sha or ledger.get('reconciliation_required')
            or ledger.get('closed_for_rollover') or ledger.get('month') != month):
        raise RuntimeError('Ledger changed or frozen; install must be replanned')
    window = _request_window_seconds(old)
    if (now + timedelta(seconds=window)).strftime('%Y-%m') != month:
        raise RuntimeError('Too close to the month boundary to install; retry in the new month')
    assert_no_hold_in_flight(ledger, now=now, duration_seconds=window)

    observation = _observe_full(now)
    if observation.get('sha256') != tariff_sha256:
        raise RuntimeError('Live price contract differs from the supplied tariff digest')
    check_current_rates(new, observation.get('contract'))

    if store.read('policy_' + new_sha).exists:
        raise RuntimeError('Unexpected successor policy without an install receipt')
    after = successor_ledger(ledger, new, new_sha)
    unchanged = digest(after['reservations']) == digest(ledger['reservations'])
    receipt = {'schema_version': 1, 'old_policy_sha256': old_sha, 'new_policy_sha256': new_sha,
               'month': month, 'observed_at': now.isoformat(), 'observation': observation,
               'before_sha256': digest(ledger), 'after_sha256': digest(after),
               'change': CHANGE, 'inputs': inputs}
    ledger_diff = field_diff(ledger, after)
    if not apply:
        return _report('plan', old_sha, new_sha, policy_diff, ledger_diff, ledger, unchanged,
                       neon_max_output_tokens, neon_thinking_level)
    if expected_new_sha != new_sha:
        raise RuntimeError('Apply requires the planned new policy digest')

    try:
        store.commit([('policy_' + new_sha, new, store.read('policy_' + new_sha)),
                      (month, after, ledger_snapshot),
                      (NEON_SUCCESSOR_RECEIPT_ID, receipt, receipt_snapshot)])
    except PublicationOutcomeUnknown:
        pass
    except PublicationConflict:
        raise RuntimeError('Install rejected by a precondition; replan') from None
    final = store.read(month).to_dict()
    if (store.read(NEON_SUCCESSOR_RECEIPT_ID).to_dict() != receipt or final != after
            or store.read('policy_' + new_sha).to_dict() != new):
        raise RuntimeError('Install outcome unresolved; read before any further action')
    return _report('applied', old_sha, new_sha, policy_diff, ledger_diff, final,
                   digest(final['reservations']) == digest(ledger['reservations']),
                   neon_max_output_tokens, neon_thinking_level)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--observe-tariff', action='store_true')
    parser.add_argument('--neon-model-version')
    parser.add_argument('--neon-tokens-per-image', type=int)
    parser.add_argument('--neon-max-output-tokens', type=int)
    parser.add_argument('--neon-thinking-level')
    parser.add_argument('--tariff-sha256')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--expected-new-sha')
    args = parser.parse_args(argv)
    try:
        if args.observe_tariff:
            result = observe()
        else:
            if os.environ.get('SPEND_POLICY_SHA256') != NEON_BASE_POLICY_SHA256:
                raise RuntimeError('SPEND_POLICY_SHA256 is not the reviewed installed policy')
            if (args.neon_model_version is None or args.neon_tokens_per_image is None
                    or args.neon_max_output_tokens is None or args.neon_thinking_level is None
                    or args.tariff_sha256 is None):
                raise RuntimeError('Model version, tokens per image, max output tokens, thinking level and tariff digest are required')
            if args.neon_max_output_tokens not in NEON_OUTPUT_TOKENS:
                raise RuntimeError('Neon max output tokens must be one of 4000, 6000, 8000, 10000, 15000')
            if args.neon_thinking_level not in NEON_BUILD_THINKING_LEVELS:
                raise RuntimeError('Neon thinking level must be high or medium')
            result = install(PrivateFirestoreStore(), old_sha=NEON_BASE_POLICY_SHA256,
                             neon_model_version=args.neon_model_version,
                             neon_tokens_per_image=args.neon_tokens_per_image,
                             neon_max_output_tokens=args.neon_max_output_tokens,
                             neon_thinking_level=args.neon_thinking_level,
                             tariff_sha256=args.tariff_sha256, apply=args.apply,
                             expected_new_sha=args.expected_new_sha)
    except RuntimeError as exc:
        print(json.dumps({'error': str(exc)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
