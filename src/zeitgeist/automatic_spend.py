"""Pure candidates for approved unattended spend controls; no network or writes.

Schema 3 is deliberately not accepted by the legacy schema-2 guard. Installing
this policy, its runtime admission path and the conditional handoff is a separate
reviewed operation. API observation times never replace manual evidence times.
"""
from __future__ import annotations
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import re
from urllib.parse import urlparse

from .spend_guard import BudgetGuard, SpendBlocked, money

STATIC = ('account_id', 'plan_id', 'credential_fingerprint', 'credit_kind',
          'operation_caps_usd', 'request_limits', 'max_request_duration_seconds')
TARGET = {'project_id': 'git-readme-features', 'database_id': '(default)', 'collection': 'spend_control'}
MODES = {'gemini': 'cash_only', 'bfl': 'nonrenewable_prepaid', 'apify': 'recurring_free'}
PRICE_HOSTS = {'gemini': {'ai.google.dev'}, 'bfl': {'bfl.ai', 'docs.bfl.ai', 'help.bfl.ai'}, 'apify': {'apify.com', 'api.apify.com', 'docs.apify.com'}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _time(value):
    return BudgetGuard._timestamp(value)


def validate_policy(policy, expected_policy_sha, now):
    if digest(policy) != expected_policy_sha or not re.fullmatch('[a-f0-9]{64}', expected_policy_sha):
        raise SpendBlocked('Automatic policy digest mismatch')
    if policy.get('schema_version') != 1 or policy.get('target') != TARGET or not policy.get('approval_reference'):
        raise SpendBlocked('Explicit approved automatic policy required')
    if not _time(policy['approved_at']) <= now < _time(policy['valid_until']):
        raise SpendBlocked('Automatic policy expired or future dated')
    ceiling = money(policy.get('cash_ceiling_usd'))
    if not 0 < ceiling <= money('10'):
        raise SpendBlocked('Automatic policy exceeds the approved 10 USD cash ceiling')
    if not isinstance(policy.get('credit_only_providers', []),list) or set(policy.get('credit_only_providers', []))-{'apify','bfl'}:
        raise SpendBlocked('Unapproved credit-only funding configuration')
    if policy.get('usage_accounting') != 'provider_usage_v1' or set(policy.get('providers', {})) != set(MODES):
        raise SpendBlocked('Explicit provider usage-accounting policy required')
    for provider, item in policy['providers'].items():
        if item.get('mode') != MODES[provider] or any(key not in item.get('account', {}) for key in STATIC):
            raise SpendBlocked('Incomplete pinned provider policy')
        account = item['account']
        url = urlparse(item.get('price_source_url', ''))
        if url.scheme != 'https' or url.hostname not in PRICE_HOSTS[provider] or url.username or not re.fullmatch('[a-f0-9]{64}', str(item.get('price_snapshot_sha256', ''))):
            raise SpendBlocked('Pinned primary-provider price evidence required')
        duration = account['max_request_duration_seconds']
        if type(duration) is not int or not 1 <= duration <= 3600:
            raise SpendBlocked('Bounded provider request duration required')
        if not now+timedelta(seconds=duration) < min(_time(item['price_valid_until']), _time(policy['valid_until'])):
            raise SpendBlocked('Pinned provider pricing or policy expires before request completion')
        mode = item.get('price_evidence_mode')
        if mode not in ('dynamic_snapshot', 'published_validity'):
            raise SpendBlocked('Explicit primary tariff evidence mode required')
        if mode == 'published_validity':
            contract = item.get('published_contract', {})
            if (contract.get('tariff_sha256') != item['price_snapshot_sha256'] or
                contract.get('reference') != item['price_source_url'] or
                not _time(contract.get('observed_at')) <= _time(policy['approved_at']) or
                not _time(contract.get('valid_from')) <= now or
                _time(contract.get('valid_until')) < _time(item['price_valid_until'])):
                raise SpendBlocked('Original published tariff validity commitment required')
        if not re.fullmatch('[a-f0-9]{16}', str(account.get('credential_fingerprint', ''))):
            raise SpendBlocked('Pinned credential fingerprint required')
        if provider in ('gemini', 'bfl'):
            evidence = item.get('top_up_disabled_attestation', {})
            reload = item.get('bounded_auto_reload_attestation')
            if provider == 'gemini' and reload is not None:
                if (reload.get('automatic_top_up_enabled') is not True or not reload.get('reference')
                    or reload.get('billing_account_id') != item.get('billing_account_id')
                    or not re.fullmatch(r'[A-F0-9]{6}-[A-F0-9]{6}-[A-F0-9]{6}',str(item.get('billing_account_id','')))
                    or not _time(reload.get('observed_at')) <= _time(policy['approved_at'])
                    or ceiling != money('10')
                    or money(reload.get('reload_amount_usd')) != money('5')
                    or money(reload.get('trigger_balance_usd')) != money('5')
                    or money(reload.get('monthly_auto_charge_limit_usd')) != money('10')
                    or policy.get('credit_only_providers') != ['apify','bfl']):
                    raise SpendBlocked('Verified bounded Gemini reload and credit-only peer providers required')
            elif evidence.get('automatic_top_up_enabled') is not False or not evidence.get('reference') or not _time(evidence.get('observed_at')) <= _time(policy['approved_at']):
                raise SpendBlocked('One-time top-up OFF approval is required; never inferred from metadata')
        if provider == 'gemini' and (account['credit_kind'] != 'none' or money(item.get('credit_limit_usd')) != 0):
            raise SpendBlocked('Gemini automation has no allocated prepaid credit')
        if provider == 'bfl' and (account['credit_kind'] != 'prepaid' or money(item.get('credit_unit_usd')) != money('.01')):
            raise SpendBlocked('Reviewed BFL prepaid conversion required')
        if provider == 'apify' and (account['credit_kind'] != 'free' or account['plan_id'] != 'FREE' or money(item.get('credit_limit_usd')) != money('5')):
            raise SpendBlocked('Only the verified FREE five-dollar recurring grant is supported')
        if provider == 'apify' and (not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', str(item.get('actor_build_id','')))
            or not re.fullmatch(r'(?:sha256:)?[a-f0-9]{64}',str(item.get('actor_image_digest','')))
            or not re.fullmatch(r'[a-f0-9]{64}',str(item.get('actor_schema_sha256','')))):
            raise SpendBlocked('Approved immutable actor build image and schema required')
    return policy


def _known(record):
    return record.get('status') == 'settled' and 'actual_gross_usd' in record and bool(record.get('billing_reference'))


def cash_exposure(ledger):
    """Expired-cycle unknowns cannot consume a new cycle's free allowance."""
    total = money(ledger.get('cash_spent_before_ledger_usd', '0'))
    for record in ledger.get('reservations', {}).values():
        if record.get('provider') not in ledger['providers']:
            raise SpendBlocked('Unknown provider liability cannot be discarded')
        if _known(record) and money(record['actual_gross_usd']) > money(record['max_gross_usd']):
            raise SpendBlocked('Confirmed overrun requires reconciliation')
    for provider, account in ledger['providers'].items():
        eligible = money(0)
        for record in ledger.get('reservations', {}).values():
            if record['provider'] != provider:
                continue
            gross = money(record['actual_gross_usd'] if _known(record) else record['max_gross_usd'])
            if record.get('credit_excluded'):
                total += gross
            else:
                eligible += gross
        total += max(money(0), eligible - money(account['credit_baseline_usd']))
    return total


def _confirmed(ledger, provider, *, excluded=False):
    return sum((money(r['actual_gross_usd']) for r in ledger.get('reservations', {}).values()
                if r['provider'] == provider and _known(r) and bool(r.get('credit_excluded')) == excluded), money(0))


def assert_credit_only_funding(ledger, *, now=None):
    """Keep independent reload cash separate from every peer cash liability.

    A credit-only peer cannot borrow the Gemini reload allocation. Retain all
    unknown holds and stop when their verified collateral no longer covers them.
    Unattributed prior cash likewise cannot be assumed to belong to that cap.
    """
    peers = ledger.get('credit_only_providers', [])
    if not peers:
        return
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise SpendBlocked('Credit-only funding requires a timezone-aware current time')
    if money(ledger.get('cash_spent_before_ledger_usd', '0')) > 0:
        raise SpendBlocked('Credit-only funding has unattributed prior cash liability')
    for provider in peers:
        eligible, unresolved = money(0), money(0)
        for record in ledger.get('reservations', {}).values():
            if record.get('provider') != provider:
                continue
            gross = money(record['actual_gross_usd'] if _known(record) else record['max_gross_usd'])
            if record.get('credit_excluded'):
                if gross > 0:
                    raise SpendBlocked('Credit-only provider has an uncollateralized retained liability')
            else:
                eligible += gross
                if not _known(record):
                    unresolved += gross
        if eligible > money(ledger['providers'][provider]['credit_baseline_usd']):
            raise SpendBlocked('Credit-only provider has no remaining verified collateral')
        if provider == 'apify' and unresolved > 0:
            account = ledger['providers'][provider]
            if not _time(account['allowance_starts_at']) <= now < _time(account['allowance_expires_at']):
                raise SpendBlocked('Credit-only provider collateral is outside its verified allowance period')


def prepare_refresh(ledger, policy, observation, *, expected_policy_sha, now):
    """Fresh authenticated metadata + unchanged approved policy -> pure candidate.

    `price_snapshots` are normalized primary-provider tariff observations collected
    by the read-only driver. A changed digest blocks; it never invents a new rate.
    """
    validate_policy(policy, expected_policy_sha, now)
    if ledger.get('automatic_policy_sha256') != expected_policy_sha or ledger.get('automatic_accounting_version') != 1 or ledger.get('schema_version') != 3:
        raise SpendBlocked('Legacy ledger requires separately approved policy installation')
    if money(ledger.get('cash_ceiling_usd')) != money(policy['cash_ceiling_usd']):
        raise SpendBlocked('Ledger cash ceiling differs from approved policy')
    if ledger.get('credit_only_providers', []) != policy.get('credit_only_providers', []):
        raise SpendBlocked('Ledger credit-only funding differs from approved policy')
    if ledger.get('closed_for_rollover') or ledger.get('reconciliation_required'):
        raise SpendBlocked('Frozen or unreconciled ledger cannot refresh')
    assert_credit_only_funding(ledger, now=now)
    observed = _time(observation.get('observed_at'))
    if not timedelta(0) <= now-observed <= timedelta(minutes=5):
        raise SpendBlocked('New authenticated metadata observation required')
    candidate = copy.deepcopy(ledger)
    for provider, pinned in policy['providers'].items():
        account = candidate['providers'][provider]
        if any(account.get(key) != pinned['account'][key] for key in STATIC):
            raise SpendBlocked('Pinned provider identity, tariff or request policy changed')
        fresh = observation.get(provider, {})
        if fresh.get('status') != 'metadata_verified' or fresh.get('credential_fingerprint') != account['credential_fingerprint']:
            raise SpendBlocked('Provider metadata or credential binding failed')
        if pinned['price_evidence_mode'] == 'dynamic_snapshot':
            price = observation.get('price_snapshots', {}).get(provider, {})
            if price.get('sha256') != pinned['price_snapshot_sha256'] or price.get('source_url') != pinned['price_source_url'] or not timedelta(0) <= now-_time(price.get('observed_at')) <= timedelta(hours=24):
                raise SpendBlocked('Primary price snapshot changed, stale or missing')
        baseline = money(account['credit_baseline_usd'])
        confirmed = _confirmed(candidate, provider)
        if provider == 'gemini':
            methods = fresh.get('available_required_models', {})
            for op in account['operation_caps_usd']:
                method, model = op.split(':', 1)
                required_method = {'generate_content': 'generateContent', 'embed_content': 'embedContent'}.get(method)
                if not required_method or required_method not in methods.get(model, []):
                    raise SpendBlocked('Required exact Gemini model/method unavailable')
            observed_credit = money(0)
        elif provider == 'bfl':
            observed_credit = money(fresh.get('balance_credits')) * money(pinned['credit_unit_usd'])
            if baseline > money(pinned['credit_limit_usd']):
                raise SpendBlocked('Immutable prepaid tranche was increased')
        else:
            billing = fresh.get('billing', {}); plan = billing.get('plan', {})
            if (fresh.get('account_fingerprint') != pinned['account_fingerprint'] or
                billing.get('status') != 'verified_included_credit_snapshot' or plan.get('id') != 'FREE' or
                plan.get('isEnabled') is not True or money(plan.get('monthlyBasePriceUsd')) != 0 or
                money(plan.get('monthlyUsageCreditsUsd')) != 5 or billing.get('active_actor_job_count') != 0 or
                billing.get('cycle_matches_limits') is not True or billing.get('usage_matches_limits') is not True or
                fresh.get('configured_build') != pinned['actor_build'] or fresh.get('build_pin_verified') is not True or
                fresh.get('build_id') != pinned['actor_build_id'] or
                fresh.get('image_digest') != pinned['actor_image_digest'] or
                fresh.get('input_schema_sha256') != pinned['actor_schema_sha256'] or
                digest(fresh.get('pricing')) != pinned['actor_pricing_sha256']):
                raise SpendBlocked('Apify account, free grant, usage cycle, build or pricing changed')
            cycle = billing['usage_cycle']; start, end = _time(cycle['startAt']), _time(cycle['endAt'])
            if not start <= now < end:
                raise SpendBlocked('Apify cycle not active')
            epoch = cycle['startAt'] + '/' + cycle['endAt']
            observed_credit = money(billing['remaining_included_credit_usd'])
            if observed_credit > 5:
                raise SpendBlocked('Observed free credit exceeds the approved grant')
            if epoch != account.get('allowance_period_id'):
                if start < _time(account['allowance_expires_at']) or start <= _time(account['allowance_starts_at']) or epoch in account.get('automatic_grants', {}):
                    raise SpendBlocked('Apify cycle grant repeated or overlaps prior allowance')
                # Retire only evidence-backed settled rows. Old unknown requests
                # retain their full bound as cash; new credit cannot hide them.
                candidate['cash_spent_before_ledger_usd'] = str(money(candidate.get('cash_spent_before_ledger_usd', '0')) + max(money(0), confirmed-baseline) + _confirmed(candidate, provider, excluded=True))
                removed = []
                for key, record in list(candidate['reservations'].items()):
                    if record['provider'] != provider:
                        continue
                    if _known(record):
                        removed.append(key); del candidate['reservations'][key]
                    else:
                        record['credit_excluded'] = True
                candidate.setdefault('retired_settlement_groups', []).append({'provider': provider, 'period': account.get('allowance_period_id'), 'reservation_ids': removed})
                account.setdefault('automatic_grants', {})[epoch] = str(observed_credit)
                account.update(allowance_period_id=epoch, allowance_starts_at=cycle['startAt'], allowance_expires_at=cycle['endAt'])
                baseline, confirmed = observed_credit, money(0)
        account['remaining_credit_usd'] = str(observed_credit)
        account['credit_baseline_usd'] = str(min(baseline, observed_credit+confirmed))
        account['automatic_observed_at'] = observation['observed_at']
        account['automatic_observation_sha256'] = digest(observation)
        # Preserve real manual observation timestamps and references exactly.
    assert_credit_only_funding(candidate, now=now)
    if cash_exposure(candidate) > money(policy['cash_ceiling_usd']):
        raise SpendBlocked('Retained liabilities exhaust the approved monthly cash ceiling')
    return {'before_sha256': digest(ledger), 'policy_sha256': expected_policy_sha, 'after': candidate}


def prepare_month_handoff(source, policy, observation, *, expected_policy_sha, source_identity, now):
    """One CAS batch freezes old source and creates the absent UTC-month ledger."""
    month_start = now.astimezone(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    prior = (month_start-timedelta(seconds=1)).strftime('%Y-%m'); month = month_start.strftime('%Y-%m')
    if any(source_identity.get(k) != v for k,v in TARGET.items()) or source_identity.get('document_id') != prior or not source_identity.get('update_time') or source.get('month') != prior:
        raise SpendBlocked('Immediately preceding exact canonical source revision required')
    _time(source_identity['update_time'])
    destination = prepare_refresh(source, policy, observation, expected_policy_sha=expected_policy_sha, now=now)['after']
    pending = {}
    for key, record in destination.get('reservations', {}).items():
        if _known(record):
            continue
        held = copy.deepcopy(record)
        held.pop('actual_gross_usd', None)
        held.update(status='reserved', origin_month=held.get('origin_month', prior), carried_from_month=prior)
        pending[key] = held
    for provider, account in destination['providers'].items():
        account['credit_baseline_usd'] = str(max(money(0), money(account['credit_baseline_usd'])-_confirmed(destination, provider)))
    destination.update(month=month, budget_window_start=month_start.isoformat(), cash_spent_before_ledger_usd='0', reservations=pending)
    destination['budget_policy_origin_start'] = source.get('budget_policy_origin_start', source['budget_window_start'])
    assert_credit_only_funding(destination, now=now)
    if cash_exposure(destination) > money(policy['cash_ceiling_usd']):
        raise SpendBlocked('Carried unknown liabilities exceed the new-month ceiling')
    identity = digest({'source': source_identity, 'before_sha256': digest(source), 'destination': destination})
    frozen = copy.deepcopy(source)
    frozen.update(closed_for_rollover=True, rollover_target_month=month, automatic_handoff_id=identity)
    destination['automatic_handoff_id'] = identity
    plan = {'schema_version': 1, 'handoff_id': identity, 'source_identity': copy.deepcopy(source_identity),
        'source_before_sha256': digest(source), 'source_after': frozen, 'destination_after': destination,
        'writes': [{'document_id': prior, 'precondition': {'update_time': source_identity['update_time']}, 'after': frozen},
                   {'document_id': month, 'precondition': {'exists': False}, 'after': destination}]}
    plan['plan_sha256'] = digest(plan)
    return plan


def reconcile_month_handoff(plan, source_readback, destination_readback):
    """Known committed only for both exact owned payloads; never dispatch or retry."""
    if plan.get('plan_sha256') != digest({k:v for k,v in plan.items() if k != 'plan_sha256'}):
        raise SpendBlocked('Month handoff plan digest mismatch')
    if source_readback != plan['source_after'] or destination_readback != plan['destination_after']:
        raise SpendBlocked('Month handoff unknown or divergent; do not redispatch')
    return {'status': 'known_committed', 'handoff_id': plan['handoff_id']}


def admit_usage_receipt(ledger, policy, receipt, *, expected_policy_sha, now):
    """Return an exact candidate for a typed receipt; this is not an invoice.

    Invalid/unowned evidence raises without changing the source. Fully bound
    overruns or conflicting observations instead return the higher known
    liability and reconciliation_required=True. The caller MUST persist that
    candidate before raising outside its ledger transaction; raising inside
    mutate would discard the very evidence that must freeze subsequent work.
    """
    if ledger.get('schema_version') != 3 or ledger.get('automatic_policy_sha256') != expected_policy_sha or ledger.get('automatic_accounting_version') != 1:
        raise SpendBlocked('Legacy holds require explicit usage-accounting policy activation')
    if ledger.get('closed_for_rollover'):
        raise SpendBlocked('Settle only the active carried reservation, never the frozen source')
    if receipt.get('schema_version') != 1 or receipt.get('accounting_policy') != 'provider_usage_v1':
        raise SpendBlocked('Unknown usage receipt policy')
    key = receipt.get('reservation_id'); original = ledger.get('reservations', {}).get(key)
    if not original:
        raise SpendBlocked('Usage receipt must identify an existing reservation')
    # Completion uses the policy that admitted this bounded request, rather than
    # requiring enough lifetime to start another full request at completion.
    validate_policy(policy, expected_policy_sha, _time(original['created_at']))
    provider = original['provider']; account = ledger['providers'][provider]
    if any(account.get(k) != policy['providers'][provider]['account'][k] for k in STATIC) or original['operation'] not in account['operation_caps_usd']:
        raise SpendBlocked('Usage receipt policy changed after reservation')
    for field in ('provider', 'operation', 'credential_fingerprint', 'provider_request_id', 'request_sha256'):
        expected = account['credential_fingerprint'] if field == 'credential_fingerprint' else original.get(field)
        if not expected or receipt.get(field) != expected:
            raise SpendBlocked('Usage receipt request or credential binding mismatch')
    if receipt.get('accounting_basis') not in ('provider_settled_cost', 'conservative_response_usage') or not re.fullmatch('[a-f0-9]{64}', str(receipt.get('evidence_sha256', ''))):
        raise SpendBlocked('Typed authenticated usage evidence required')
    observed = _time(receipt.get('observed_at'))
    if not _time(original.get('submitted_at', original['created_at'])) <= observed <= now or now-observed > timedelta(hours=24):
        raise SpendBlocked('Usage receipt timestamp outside request window')
    amount = money(receipt.get('accounted_gross_usd'))
    for other, record in ledger['reservations'].items():
        if other != key and record.get('provider') == provider and record.get('provider_request_id') == receipt['provider_request_id']:
            raise SpendBlocked('Provider response already belongs to another reservation')
    candidate = copy.deepcopy(ledger); record = candidate['reservations'][key]
    reference = 'provider_usage_v1:' + digest({k:v for k,v in receipt.items() if k != 'observed_at'})
    conflict = _known(record) and (record.get('billing_reference') != reference or money(record['actual_gross_usd']) != amount)
    prior_amount = money(record['actual_gross_usd']) if _known(record) else None
    if conflict:
        earlier = record.get('usage_receipt_conflict', {})
        record['usage_receipt_conflict'] = {
            'first_receipt': copy.deepcopy(earlier.get('first_receipt', record.get('usage_receipt'))),
            'first_billing_reference': earlier.get('first_billing_reference', record['billing_reference']),
            'first_accounted_gross_usd': earlier.get('first_accounted_gross_usd', str(prior_amount)),
            'latest_receipt': copy.deepcopy(receipt),
        }
    # Never replace stronger known liability/evidence with a smaller or merely
    # different re-observation. Keep the original admission cap for inspection.
    if prior_amount is None or amount > prior_amount or not conflict:
        record.update(status='settled', actual_gross_usd=str(amount), billing_reference=reference,
                      usage_receipt=copy.deepcopy(receipt))
    overrun = money(record['actual_gross_usd']) > money(record['max_gross_usd'])
    if conflict or overrun:
        candidate['reconciliation_required'] = True
        record['reconciliation_required'] = True
        reasons = set(record.get('reconciliation_reasons', []))
        if conflict:
            reasons.add('usage_receipt_conflict')
        if overrun:
            reasons.add('usage_receipt_overrun')
        record['reconciliation_reasons'] = sorted(reasons)
    return candidate


def gemini_usage_receipt(reservation_id, reservation, response, *, credential_fingerprint, observed_at, request_sha256, policy):
    """Conservative standard-token charge from the SAME authenticated SDK response.

    The caller binds responseId before admission. The admitted input bound is
    retained (including image floors); only provably unused output is released.
    Missing/extra modality or tool usage cannot produce a receipt.
    """
    operation = reservation['operation']; method, model = operation.split(':', 1)
    limits = policy['providers']['gemini']['account']['request_limits'][operation]
    if method != 'generate_content' or response.get('modelVersion') not in limits['allowed_response_model_versions']:
        raise SpendBlocked('Gemini response model version not pinned')
    usage = response.get('usageMetadata', {})
    counts = {}
    for field in ('promptTokenCount','candidatesTokenCount','totalTokenCount'):
        if type(usage.get(field)) is not int or usage[field] < 0:
            raise SpendBlocked('Complete authenticated Gemini usage counts required')
        counts[field] = usage[field]
    for field in ('thoughtsTokenCount','toolUsePromptTokenCount','cachedContentTokenCount'):
        value = usage.get(field, 0)
        if type(value) is not int or value < 0:
            raise SpendBlocked('Invalid Gemini usage count')
        counts[field] = value
    if counts['toolUsePromptTokenCount'] or counts['cachedContentTokenCount'] > counts['promptTokenCount'] or counts['totalTokenCount'] != counts['promptTokenCount']+counts['candidatesTokenCount']+counts['thoughtsTokenCount']:
        raise SpendBlocked('Unpriced tool use or inconsistent Gemini usage')
    output = counts['candidatesTokenCount']+counts['thoughtsTokenCount']
    context = reservation['request_context']
    if counts['promptTokenCount'] > context['admitted_input_tokens'] or output > context['max_output_tokens']:
        raise SpendBlocked('Gemini usage exceeds admitted token shape')
    if not response.get('responseId') or any(item.get('modality') != 'TEXT' for item in usage.get('candidatesTokensDetails', [])):
        raise SpendBlocked('Typed text-only Gemini response identity required')
    amount = (money(context['admitted_input_tokens'])*money(limits['input_token_usd_per_million']) + money(output)*money(limits['output_token_usd_per_million']))/1000000
    return {'schema_version': 1, 'accounting_policy': 'provider_usage_v1', 'reservation_id': reservation_id,
        'provider': 'gemini', 'operation': operation, 'credential_fingerprint': credential_fingerprint,
        'provider_request_id': response['responseId'], 'request_sha256': request_sha256, 'observed_at': observed_at,
        'accounted_gross_usd': str(amount), 'accounting_basis': 'conservative_response_usage',
        'evidence_sha256': digest({'response_id': response['responseId'], 'model': response['modelVersion'], 'usage': usage, 'request_context': context})}


def gemini_embedding_receipt(reservation_id, reservation, response, *, credential_fingerprint, observed_at, request_sha256):
    """Known completed embedding: book the FULL admitted cap, not invented usage.

    The pinned SDK drops REST token usage. Counted-input evidence and a valid
    typed result can therefore retire a fulfilled conservative cap only under
    the separately approved provider_usage_v1 policy; no in-month credit is
    released. A local response digest is explicitly not a provider request ID.
    """
    if reservation.get('operation') != 'embed_content:gemini-embedding-001':
        raise SpendBlocked('Embedding completion policy covers only embedding-001')
    context = reservation.get('request_context', {})
    if type(context.get('counted_input_tokens')) is not int or context['counted_input_tokens'] < 0:
        raise SpendBlocked('Bounded embedding input-count evidence required')
    embeddings = response.get('embeddings')
    if not isinstance(embeddings, list) or not embeddings or len(embeddings) != context.get('embedding_input_count'):
        raise SpendBlocked('Complete typed embedding response required')
    for embedding in embeddings:
        values = embedding.get('values') if isinstance(embedding, dict) else None
        if not isinstance(values, list) or not values or any(type(v) not in (int,float) or not math.isfinite(v) for v in values):
            raise SpendBlocked('Finite typed embedding vectors required')
    evidence = digest({'reservation_id': reservation_id, 'request_sha256': request_sha256, 'context': context, 'response': response})
    response_identity = 'local-response-sha256:' + evidence
    return {'schema_version': 1, 'accounting_policy': 'provider_usage_v1', 'reservation_id': reservation_id,
        'provider': 'gemini', 'operation': reservation['operation'], 'credential_fingerprint': credential_fingerprint,
        'provider_request_id': response_identity, 'request_id_kind': 'local_response_digest',
        'request_sha256': request_sha256, 'observed_at': observed_at,
        'accounted_gross_usd': str(money(reservation['max_gross_usd'])),
        'accounting_basis': 'conservative_response_usage', 'evidence_sha256': evidence}
