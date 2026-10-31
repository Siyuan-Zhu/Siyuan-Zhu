#!/usr/bin/env python3
"""Prepare a PRIVATE, create-only candidate; never connect to a provider or Firestore."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked, money

OFFICIAL_HOSTS = {'gemini': {'ai.google.dev'}, 'bfl': {'bfl.ai', 'docs.bfl.ai', 'help.bfl.ai'},
                  'apify': {'apify.com', 'docs.apify.com', 'console.apify.com'}}
# Exact standard Developer API tariffs observed on the official pricing page.
# A new model/rate is a reviewed code change, not an inferred alias.
TOKEN_RATES = {
    'generate_content:gemini-3.8-flash': ('0.75', '3.75'),
    'generate_content:gemini-2.5-flash': ('0.30', '2.50'),
    'generate_content:gemini-3-flash-preview': ('0.50', '3.00'),
    'embed_content:gemini-embedding-001': ('0.15', '0'),
}

def _evidence(provider, evidence, now):
    if not isinstance(evidence, dict) or not evidence.get('reference'):
        raise SpendBlocked('Verified pricing evidence reference required')
    parsed = urlparse(str(evidence.get('source_url', '')))
    if parsed.scheme != 'https' or parsed.hostname not in OFFICIAL_HOSTS[provider] or parsed.username:
        raise SpendBlocked('Primary provider pricing URL required')
    observed = BudgetGuard._timestamp(evidence.get('verified_at'))
    if not timedelta(0) <= now-observed < timedelta(hours=24):
        raise SpendBlocked('Pricing evidence expired or future-dated')

def prepare_candidate(metadata, *, now=None):
    """Validate attested private evidence. Authenticity remains a release gate.

    Output is ledger contents only: approval and create-if-missing CAS are separate.
    This function cannot refresh, merge, overwrite or reset an existing ledger.
    """
    now = now or datetime.now(timezone.utc)
    if metadata.get('schema_version') != 1:
        raise SpendBlocked('Unsupported metadata schema')
    cutoff = BudgetGuard._timestamp(metadata.get('budget_window_start'))
    if cutoff > now or cutoff.strftime('%Y-%m') != now.strftime('%Y-%m'):
        raise SpendBlocked('Cutoff must be an explicit past/current UTC-month timestamp')
    for field in ('uncontrolled_post_cutoff_usage', 'pre_cutoff_requests_in_flight'):
        if metadata.get(field) is not False:
            raise SpendBlocked('Uncontrolled or in-flight usage requires reconciliation before initial provisioning')
    providers = metadata.get('providers')
    if not isinstance(providers, dict) or not providers or set(providers)-set(OFFICIAL_HOSTS):
        raise SpendBlocked('Explicit known provider evidence required')
    document = {'schema_version': 2, 'month': now.strftime('%Y-%m'),
      'budget_scope': 'new_requests_after_cutoff', 'budget_window_start': cutoff.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z'),
      'cash_ceiling_usd': '2.50', 'cash_spent_before_ledger_usd': '0',
      'credential_binding_required': True, 'reservations': {}, 'providers': {},
      'initial_evidence_sha256': hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest(),
      'excluded_historical_observations': copy.deepcopy(metadata.get('excluded_historical_observations', []))}
    for provider, raw in providers.items():
        account = copy.deepcopy(raw)
        if not re.fullmatch('[0-9a-f]{16}', str(account.get('credential_fingerprint', ''))):
            raise SpendBlocked('Verified runtime credential fingerprint required')
        if account.get('credit_kind') not in ('none', 'free', 'prepaid'):
            raise SpendBlocked('Explicit none/free/prepaid credit kind required')
        account['credit_baseline_usd'] = str(money(account.get('remaining_credit_usd')))
        if account['credit_kind'] == 'none' and money(account['credit_baseline_usd']) != 0:
            raise SpendBlocked('A positive balance requires an explicit verified credit kind')
        if account.get('credit_kind') == 'prepaid':
            if account.get('automatic_top_up_enabled') is not False:
                raise SpendBlocked('Prepaid automatic top-up must be verified disabled')
            _evidence(provider, {'source_url': account.get('credit_terms_reference'),
                'reference': account.get('verification_reference'), 'verified_at': account.get('verified_at')}, now)
        caps = account.get('operation_caps_usd')
        evidence = account.get('price_evidence')
        if not isinstance(caps, dict) or not caps or not isinstance(evidence, dict) or set(caps) != set(evidence):
            raise SpendBlocked('Every admitted operation requires matching primary price evidence')
        for operation, cap in caps.items():
            price = evidence[operation]; _evidence(provider, price, now)
            if money(cap) != money(price.get('maximum_gross_usd')) or money(cap) <= 0:
                raise SpendBlocked('Operation cap must match verified positive maximum gross charge')
            if provider == 'gemini':
                if operation not in TOKEN_RATES:
                    raise SpendBlocked('Unverified model tariff; operation remains blocked')
                limits = account.get('request_limits', {}).get(operation)
                if not isinstance(limits, dict):
                    raise SpendBlocked('Bounded Gemini request policy required')
                rates = TOKEN_RATES[operation]
                for key, rate in zip(('input_token_usd_per_million', 'output_token_usd_per_million'), rates):
                    if money(limits.get(key)) != money(rate):
                        raise SpendBlocked('Token tariff differs from reviewed official standard price')
                if not limits.get('count_tokens_reference') or not limits.get('pricing_reference'):
                    raise SpendBlocked('Token counting and tariff evidence required')
                count = limits.get('max_input_tokens')
                if type(count) is not int or count <= 0 or int(limits.get('max_input_bytes', 0)) <= 0:
                    raise SpendBlocked('Positive token and serialized-byte ceilings required')
                output = int(limits.get('max_output_tokens', 0))
                if operation.startswith('generate_content:') and output <= 0:
                    raise SpendBlocked('Positive bounded output required')
                maximum = (money(count)*money(rates[0])+money(output)*money(rates[1]))/1000000
                if money(cap) < maximum:
                    raise SpendBlocked('Declared operation cap does not cover full token policy')
                if operation == 'generate_content:gemini-3.8-flash':
                    BudgetGuard(MemoryLedger({}), lambda: now)._validate_gemini38_policy(
                        limits, int(account.get('max_request_duration_seconds', 300)))
                account['token_pricing_required'] = True
        document['providers'][provider] = account
    guard = BudgetGuard(MemoryLedger(document), lambda: now,
        credential_fingerprints={p:a['credential_fingerprint'] for p,a in document['providers'].items()})
    for provider in providers:
        guard._validate(document, provider)
    return document

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--budget-window-start', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        metadata = json.loads(args.metadata.read_text())
        if metadata.get('budget_window_start') != args.budget_window_start:
            raise SpendBlocked('CLI cutoff must exactly match reviewed metadata cutoff')
        document = prepare_candidate(metadata)
        # Exclusive/private local creation prevents accidental replacement of a reviewed candidate.
        fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as handle:
            json.dump(document, handle, indent=2, sort_keys=True); handle.write('\n')
        print('Private candidate created. No ledger writes or provider calls performed.')
    except (SpendBlocked, OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'Candidate rejected: {exc}\n')

if __name__ == '__main__':
    main()
