#!/usr/bin/env python3
"""Offline month handoff from a frozen canonical ledger and fresh private evidence.

No provider calls, ledger writes, implicit source freeze or deployment approval.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_spend_ledger import prepare_candidate
from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked, money


def prepare_rollover(source, metadata, *, source_identity, now=None):
    """Retain unresolved holds; transfer one conservatively funded allowance.

    Caller must prove this exact source snapshot is frozen using source updateTime
    CAS. The destination must be created only if absent and source remains frozen.
    """
    now = now or datetime.now(timezone.utc)
    month_start = now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    previous_month = (month_start-timedelta(seconds=1)).strftime('%Y-%m')
    required_identity = ('project_id', 'database_id', 'collection', 'document_id', 'frozen_update_time')
    if not isinstance(source_identity, dict) or any(not source_identity.get(key) for key in required_identity):
        raise SpendBlocked('Exact frozen source target and updateTime required')
    BudgetGuard._timestamp(source_identity['frozen_update_time'])
    if source.get('month') != previous_month or source_identity['document_id'] != previous_month:
        raise SpendBlocked('Rollover requires the immediately preceding canonical month')
    if source.get('schema_version') != 2 or source.get('credential_binding_required') is not True:
        raise SpendBlocked('Unsupported source accounting epoch')
    if source.get('closed_for_rollover') is not True or source.get('reconciliation_required'):
        raise SpendBlocked('Source must be explicitly frozen without unresolved overrun')
    if source.get('rollover_target_month') != month_start.strftime('%Y-%m'):
        raise SpendBlocked('Frozen source is bound to a different destination month')
    cutoff = BudgetGuard._timestamp(metadata.get('budget_window_start'))
    if cutoff != month_start:
        raise SpendBlocked('Rollover cutoff must be the UTC month start')
    candidate = prepare_candidate(metadata, now=now)
    if set(candidate['providers']) != set(source.get('providers', {})):
        raise SpendBlocked('Provider set cannot change during month rollover')
    pending = {}; confirmed = {}; unknown = {}
    for reservation_id, raw in source.get('reservations', {}).items():
        provider = raw.get('provider')
        if provider not in candidate['providers']:
            raise SpendBlocked('Reservation references an unknown provider')
        maximum = money(raw.get('max_gross_usd'))
        actual = money(raw['actual_gross_usd']) if 'actual_gross_usd' in raw else None
        if actual is not None and actual > maximum:
            raise SpendBlocked('Confirmed overrun must be reconciled before rollover')
        if raw.get('status') == 'settled' and actual is not None and raw.get('billing_reference'):
            confirmed[provider] = confirmed.get(provider, money(0)) + actual
            continue
        record = copy.deepcopy(raw)
        if actual is not None:
            record['unreconciled_observed_gross_usd'] = str(actual)
            record.pop('actual_gross_usd')
        record.update(status='reserved', origin_month=raw.get('origin_month', source['month']),
                      carried_from_month=source['month'])
        pending[reservation_id] = record
        unknown[provider] = unknown.get(provider, money(0)) + maximum
    transfers = {}
    for provider, fresh in candidate['providers'].items():
        old = source['providers'][provider]
        for key in ('account_id', 'plan_id', 'credential_fingerprint', 'credit_kind'):
            if not old.get(key) or old[key] != fresh.get(key):
                raise SpendBlocked('Provider identity/plan/credit kind changed during rollover')
        if old.get('allowance_period_id') != fresh.get('allowance_period_id'):
            raise SpendBlocked('New allowance grant requires separate verified reconciliation')
        observed = money(fresh['remaining_credit_usd'])
        baseline = BudgetGuard._credit_baseline(old)
        settled = confirmed.get(provider, money(0))
        held = unknown.get(provider, money(0))
        # New baseline funds carried unknown reservations first. Only the balance
        # left after ALL old gross bounds can fund new-month requests. Fresh
        # provider observations can shrink this transfer but never grant credit.
        funded = min(observed, max(money(0), baseline-settled))
        held_credit = min(held, funded)
        available = max(money(0), funded-held)
        old_unreserved = max(money(0), baseline-settled-held)
        if available > min(observed, old_unreserved):
            raise SpendBlocked('Rollover would duplicate spendable credit')
        fresh['credit_baseline_usd'] = str(funded)
        transfers[provider] = {'prior_credit_baseline_usd': str(baseline),
          'confirmed_prior_gross_usd': str(settled), 'carried_unknown_gross_usd': str(held),
          'carried_unknown_credit_collateral_usd': str(held_credit),
          'new_unreserved_credit_usd': str(available)}
    candidate['reservations'] = pending
    candidate['budget_policy_origin_start'] = source.get('budget_policy_origin_start', source.get('budget_window_start'))
    candidate['rollover'] = {'source': copy.deepcopy(source_identity),
      'source_sha256': hashlib.sha256(json.dumps(source, sort_keys=True, allow_nan=False).encode()).hexdigest(),
      'prepared_at': now.isoformat(), 'transfers': transfers,
      'requires_source_frozen_and_destination_absent': True}
    # Cash-backed unknown requests also retain exposure in the new month. This
    # can conservatively count a delayed charge in both months until reconciled.
    if BudgetGuard._cash_exposure(candidate) > money(candidate['cash_ceiling_usd']):
        raise SpendBlocked('Unresolved carried cash liabilities exceed new-month ceiling')
    guard = BudgetGuard(MemoryLedger(candidate), lambda: now,
       credential_fingerprints={p:a['credential_fingerprint'] for p,a in candidate['providers'].items()})
    for provider in candidate['providers']:
        guard._validate(candidate, provider)
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen-source-snapshot', type=Path, required=True)
    parser.add_argument('--source-identity', type=Path, required=True)
    parser.add_argument('--verified-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        candidate = prepare_rollover(json.loads(args.frozen_source_snapshot.read_text()),
          json.loads(args.verified_metadata.read_text()), source_identity=json.loads(args.source_identity.read_text()))
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(candidate, handle, sort_keys=True, indent=2, allow_nan=False); handle.write('\n')
        print('Private rollover candidate created. Source was not frozen and no ledger was written by this command.')
    except (SpendBlocked, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Rollover rejected: {exc}\n')

if __name__ == '__main__':
    main()
