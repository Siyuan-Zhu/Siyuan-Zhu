#!/usr/bin/env python3
"""Read the shared ledger and probe unbilled Director token admission only.

No reservation, inference, actors, canonical writes or private source payloads.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
from io import StringIO
import json
import logging
import os
import re
from pathlib import Path
import sys
import urllib.error

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.state_store import _FirestoreRestClient, _firestore_value_to_py
from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked, money

PROJECT = 'git-readme-features'
DATABASE = '(default)'
MODELS = ('gemini-2.5-flash', 'gemini-3-flash-preview')
PROMPT = (
    'You are the Editorial Director for a daily public technology and design digest. '
    'Synthetic cluster: thoughtful architecture, resilient software and collaborative craft. '
    'Commission one concise Latin-script word, a short gloss, a physical visual anchor and '
    'three photographic search queries. Return JSON with word, gloss, commissioned_thesis, '
    'visual_anchor, unsplash_queries, imagen_commission_prompt and palette_hex. '
    'This is synthetic metadata-admission input; do not include personal information.'
)


def proposed_gemini38_policy():
    """Exact reviewed contract for an offline proposal, never an active grant."""
    return {'operation': 'generate_content:gemini-3.8-flash', 'maximum_gross_usd': '0.054777',
      'request_limits': {'pricing_reference': 'https://ai.google.dev/gemini-api/docs/pricing',
        'count_tokens_reference': 'https://ai.google.dev/gemini-api/docs/billing#is-gettokens-billed',
        'pricing_valid_until': '2026-12-31T23:59:59Z',
        'max_input_bytes': 2000000, 'max_input_tokens': 65536, 'max_output_tokens': 1500,
        'input_token_usd_per_million': '0.75', 'output_token_usd_per_million': '3.75',
        'thinking_level': 'low', 'max_input_images': 9, 'image_tokens_per_image': 1120,
        'media_resolution': 'MEDIA_RESOLUTION_HIGH'}}


def _synthetic_photo_batch():
    from io import BytesIO
    from PIL import Image
    from google.genai import types
    stream = BytesIO()
    Image.new('RGB', (32, 32), (40, 70, 90)).save(stream, format='JPEG')
    return [PROMPT] + [types.Part.from_bytes(data=stream.getvalue(), mime_type='image/jpeg') for _ in range(9)]


def _read_shared_ledger(env, now):
    if env.get('GCP_PROJECT_ID') != PROJECT or env.get('FIRESTORE_DATABASE_ID') != DATABASE:
        raise RuntimeError('Fixed canonical read target required')
    client = _FirestoreRestClient(PROJECT, DATABASE)
    name = 'projects/' + PROJECT + '/databases/' + DATABASE + '/documents/spend_control/' + now.strftime('%Y-%m')
    url = 'https://firestore.googleapis.com/v1/' + name
    headers = {'Authorization': 'Bearer ' + client._resolve_token()}
    # GET only. Never use FirestoreLedger.mutate, which writes even for read callbacks.
    with httpx.Client(timeout=15, follow_redirects=False) as transport:
        response = transport.get(url, headers=headers)
        response.raise_for_status()
        body = response.json()
    if body.get('name') != name or not isinstance(body.get('fields'), dict):
        raise RuntimeError('Unexpected canonical ledger response')
    return {key: _firestore_value_to_py(value) for key, value in body['fields'].items()}


def _read_approved_policy(env, policy_sha):
    """Read only the exact pinned private policy document; never install/refresh."""
    if env.get('GCP_PROJECT_ID') != PROJECT or env.get('FIRESTORE_DATABASE_ID') != DATABASE:
        raise RuntimeError('Fixed canonical read target required')
    if not re.fullmatch('[a-f0-9]{64}', policy_sha or ''):
        raise SpendBlocked('A pinned approved policy digest and UTC-compatible clock are required')
    from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore
    snapshot = PrivateFirestoreStore().read('policy_' + policy_sha)
    if not snapshot.exists:
        raise SpendBlocked('Approved automatic policy is not installed')
    return snapshot.to_dict()


def _snapshot_guard(document, key, now, *, policy=None, policy_sha=None):
    options = {'require_credential_binding': True,
        'credential_fingerprints': {'gemini': hashlib.sha256(key.encode()).hexdigest()[:16]}}
    memory = MemoryLedger(copy.deepcopy(document))
    if policy is not None:
        from src.zeitgeist.automatic_guard import AutomaticBudgetGuard
        return AutomaticBudgetGuard(memory, policy, policy_sha, lambda:now, **options)
    return BudgetGuard(memory, lambda:now, **options)


def _client(key):
    from google import genai
    return genai.Client(api_key=key, http_options={'timeout': 300000, 'retry_options': {'attempts': 1}})


def _public_failure(error, phase):
    if isinstance(error, SpendBlocked):
        raw = error.public_diagnostics
        known = set(SpendBlocked._PUBLIC_REASONS.values()) | {'budget-admission-rejected'}
        result = {'code': raw.get('code') if raw.get('code') in known else 'budget-admission-rejected'}
        if type(raw.get('status')) is int and 100 <= raw['status'] <= 599:
            result['status'] = raw['status']
        if raw.get('provider') == 'gemini': result['provider'] = 'gemini'
        if raw.get('error_type') in {'ClientError', 'ServerError', 'APIError', 'HTTPError', 'HTTPStatusError', 'ReadTimeout', 'ConnectError', 'TimeoutError'}:
            result['error_type'] = raw['error_type']
        return result
    result = {'code': {'ledger_read': 'ledger-read-unavailable', 'policy_read': 'policy-read-unavailable', 'sdk_client': 'sdk-client-unavailable'}.get(phase, 'probe-unavailable')}
    if isinstance(error, urllib.error.HTTPError):
        status = error.code
    elif isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
    else:
        status = None
    if type(status) is int and 100 <= status <= 599:
        result['status'] = status
    return result


def _probe_model(document, client, key, model, now, *, contents=None, request_shape="text", policy=None, policy_sha=None):
    report = {'status': 'blocked', 'admitted': False, 'model': model,
              'metadata_calls': 0, 'paid_calls': 0, 'ledger_writes': 0,
              'reservation_created': False, 'request_shape': request_shape}
    operation = 'generate_content:' + model
    phase = 'request_bound'
    try:
        guard = _snapshot_guard(document, key, now, policy=policy, policy_sha=policy_sha)
        phase = 'request_bound'
        config = {'response_mime_type': 'application/json'}
        if not model.startswith('gemini-3'):
            config['temperature'] = 0.3
        bounded = guard.bounded_genai_request(operation, {
            'model': model, 'contents': PROMPT if contents is None else contents, 'config': config,
        })
        def count_tokens(**kwargs):
            report['metadata_calls'] += 1
            response = client.models.count_tokens(**kwargs)
            if type(response.total_tokens) is int and 0 <= response.total_tokens <= 10**9:
                report['counted_input_tokens'] = max(report.get('counted_input_tokens', 0), response.total_tokens)
            return response
        phase = 'token_metadata'
        maximum, _reference = guard.counted_genai_cap(operation, bounded, count_tokens)
        phase = 'snapshot_admission'
        account = document['providers']['gemini']
        approved = account.get('operation_caps_usd', {}).get(operation)
        if approved is None or maximum > money(approved):
            raise SpendBlocked('Request exceeds verified operation tariff')
        # Pure hypothetical arithmetic, never a reserve() or ledger mutation.
        if policy is not None:
            hypothetical = copy.deepcopy(document)
            marker = 'synthetic_admission_probe'
            while marker in hypothetical.get('reservations', {}):
                marker += '_'
            hypothetical.setdefault('reservations', {})[marker] = {
                'provider': 'gemini', 'max_gross_usd': str(maximum), 'status': 'reserved'}
            exposure = guard._cash_exposure(hypothetical)
        else:
            gross = sum((money(record.get('actual_gross_usd', record['max_gross_usd']))
                         for record in document.get('reservations', {}).values()
                         if record.get('provider') == 'gemini'), Decimal('0'))
            credit = guard._credit_baseline(account)
            incremental_cash = max(Decimal('0'), gross + maximum - credit) - max(Decimal('0'), gross - credit)
            exposure = guard._cash_exposure(document) + incremental_cash
        if exposure > money(document['cash_ceiling_usd']):
            raise SpendBlocked('Call would exceed approved monthly cash ceiling')
        report.update(status='admitted', admitted=True, maximum_gross_usd=str(maximum))
    except Exception as error:
        report.update(blocked_phase=phase, diagnostics=_public_failure(error, phase))
    return report


def run_probe(env, *, now=None, ledger_loader=None, policy_loader=None, client_factory=None):
    now = now or datetime.now(timezone.utc)
    report = {'status': 'blocked', 'admitted': False, 'models': [],
              'metadata_calls': 0, 'paid_calls': 0, 'ledger_writes': 0,
              'reservation_created': False, 'inference_eligibility_verified': False,
              'admission_is_snapshot_only': True,
              'scope': 'synthetic_director_token_admission_only'}
    phase = 'ledger_read'
    try:
        document = copy.deepcopy((ledger_loader or _read_shared_ledger)(env, now))
        key = str(env.get('GEMINI_API_KEY') or '').strip()
        policy, policy_sha = None, str(env.get('SPEND_POLICY_SHA256') or '').strip()
        automatic = document.get('schema_version') == 3 or bool(policy_sha)
        if automatic:
            phase = 'policy_read'
            if (env.get('GCP_PROJECT_ID') != PROJECT or env.get('FIRESTORE_DATABASE_ID') != DATABASE
                    or document.get('schema_version') != 3 or document.get('automatic_accounting_version') != 1):
                raise SpendBlocked('Explicit approved automatic accounting epoch required')
            if not re.fullmatch('[a-f0-9]{64}', policy_sha):
                raise SpendBlocked('A pinned approved policy digest and UTC-compatible clock are required')
            if document.get('automatic_policy_sha256') != policy_sha:
                raise SpendBlocked('Automatic policy digest mismatch')
            policy = copy.deepcopy((policy_loader or _read_approved_policy)(env, policy_sha))
            if not isinstance(policy, dict):
                raise SpendBlocked('Approved automatic policy is not installed')
            if not key:
                raise SpendBlocked('Runtime credential does not match verified provider account')
            # Validate the actual credential, epoch and static pinned account
            # before constructing any provider SDK or performing token metadata.
            guard = _snapshot_guard(document, key, now, policy=policy, policy_sha=policy_sha)
            guard._validate(document, 'gemini')
            models = sorted(operation.split(':', 1)[1]
                for operation in policy['providers']['gemini']['account']['operation_caps_usd']
                if operation.startswith('generate_content:'))
            if not models:
                raise SpendBlocked('Required exact Gemini model/method unavailable')
            report['policy_source'] = 'active_pinned_automatic_policy'
        else:
            if document.get('schema_version') != 2:
                raise SpendBlocked('Runtime requires an approved credential-bound budget epoch')
            if not key:
                raise SpendBlocked('Runtime credential does not match verified provider account')
            models = list(MODELS)
            report['policy_source'] = 'active_legacy_schema2_snapshot'
        phase = 'sdk_client'
        client = (client_factory or _client)(key)
        if automatic:
            results = []
            for model in models:
                for shape, contents in (('text', None), ('nine_thumbnails', _synthetic_photo_batch())):
                    result = _probe_model(document, client, key, model, now, contents=contents,
                        request_shape=shape, policy=policy, policy_sha=policy_sha)
                    result.update(policy_source='active_pinned_automatic_policy', active_ledger_changed=False)
                    results.append(result)
            report['models'] = results
            report['proposed_policy_results'] = []
            report['metadata_calls'] = sum(result['metadata_calls'] for result in results)
            report['admitted_models'] = [model for model in models
                if all(result['admitted'] for result in results if result['model'] == model)]
            report['admitted'] = bool(results) and all(result['admitted'] for result in results)
        else:
            report['models'] = [_probe_model(document, client, key, model, now) for model in models]
            phase = 'proposed_policy'
            proposal = proposed_gemini38_policy()
            proposed_document = copy.deepcopy(document)
            account = proposed_document['providers']['gemini']
            account.setdefault('operation_caps_usd', {})[proposal['operation']] = proposal['maximum_gross_usd']
            account.setdefault('request_limits', {})[proposal['operation']] = proposal['request_limits']
            proposed = [
                _probe_model(proposed_document, client, key, 'gemini-3.8-flash', now),
                _probe_model(proposed_document, client, key, 'gemini-3.8-flash', now,
                             contents=_synthetic_photo_batch(), request_shape='nine_thumbnails'),
            ]
            for result in proposed:
                result.update(policy_source='proposed_memory_only_not_active_authority',
                              active_ledger_changed=False,
                              active_policy_present=proposal['operation'] in document['providers']['gemini'].get('request_limits', {}))
            report['proposed_policy_results'] = proposed
            report['metadata_calls'] = sum(result['metadata_calls'] for result in report['models'] + proposed)
            report['admitted_models'] = [result['model'] for result in report['models'] if result['admitted']]
            report['admitted'] = bool(report['admitted_models'])
        report['status'] = 'admitted' if report['admitted'] else 'blocked'
    except Exception as error:
        report.update(blocked_phase=phase, diagnostics=_public_failure(error, phase))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, required=True)
    args = parser.parse_args(argv)
    previous_logging = logging.root.manager.disable
    try:
        # Print only our sanitized report, including when a third-party client
        # logs raw exception content. No private captured stream is serialized.
        logging.disable(logging.CRITICAL)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            report = run_probe(os.environ)
    finally:
        logging.disable(previous_logging)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2) + '\n'
    args.output_json.write_text(encoded)
    print(encoded, end='')
    return 0 if report['admitted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
