"""Runtime admission for an explicitly installed automatic accounting policy.

The legacy guard remains unchanged when no policy is configured. This module
never installs a policy, converts a ledger, refreshes manual facts or buys credit.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import hashlib
import json
import re

from . import automatic_spend as accounting
from .spend_guard import BudgetGuard, SpendBlocked, money


def request_digest(value):
    def encode(item):
        if isinstance(item, bytes):
            return {'bytes_sha256': hashlib.sha256(item).hexdigest(), 'length': len(item)}
        if hasattr(item, 'model_dump'):
            return item.model_dump(exclude_none=True)
        raise SpendBlocked('Unbounded automatic request identity')
    return hashlib.sha256(json.dumps(value, default=encode, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class ProviderRequest:
    def __init__(self, guard, reservation_id, provider, operation, request_sha256):
        self.guard = guard
        self.reservation_id = reservation_id
        self.provider = provider
        self.operation = operation
        self.request_sha256 = request_sha256
        self.credential_fingerprint = guard._credential_fingerprint(provider)

    def submitted(self, provider_request_id, *, submitted_at=None, metadata=None):
        if not isinstance(provider_request_id, str) or not provider_request_id or len(provider_request_id) > 256:
            raise SpendBlocked('Bounded provider response identity required')
        when = submitted_at or self.guard.now().isoformat()
        when = when.isoformat() if hasattr(when, 'isoformat') else when
        metadata = copy.deepcopy(metadata or {})
        allowed = {'actor_id', 'user_id', 'build', 'polling_host', 'resolution', 'aspect_ratio'}
        if set(metadata) - allowed or len(json.dumps(metadata, allow_nan=False)) > 2048:
            raise SpendBlocked('Unapproved private submission metadata')
        def update(document):
            self.guard._validate_completion(document, self.provider, self.reservation_id)
            record = document['reservations'][self.reservation_id]
            if (record.get('provider') != self.provider or record.get('operation') != self.operation
                or record.get('request_sha256') != self.request_sha256
                or record.get('credential_fingerprint') != self.credential_fingerprint):
                raise SpendBlocked('Provider handle no longer matches its reservation')
            if record.get('provider_request_id') not in (None, provider_request_id):
                raise SpendBlocked('Provider reservation already has another request identity')
            if not self.guard._timestamp(record['created_at']) <= self.guard._timestamp(when) <= self.guard.now():
                raise SpendBlocked('Provider submission timestamp outside reservation window')
            record.update(provider_request_id=provider_request_id, submitted_at=when,
                          submission_metadata=metadata)
        self.guard.ledger.mutate(update)

    def receipt(self, receipt):
        expected = {'reservation_id': self.reservation_id, 'provider': self.provider,
                    'operation': self.operation, 'request_sha256': self.request_sha256,
                    'credential_fingerprint': self.credential_fingerprint}
        if not isinstance(receipt, dict) or any(receipt.get(k) != v for k,v in expected.items()):
            raise SpendBlocked('Usage receipt belongs to a different request handle')
        def update(document):
            self.guard._validate_completion(document, self.provider, self.reservation_id)
            candidate = accounting.admit_usage_receipt(document, self.guard.automatic_policy,
                receipt, expected_policy_sha=self.guard.expected_policy_sha, now=self.guard.now())
            document.clear()
            document.update(candidate)
            return candidate.get('reconciliation_required') is True
        # Persist authenticated higher liability before stopping further work.
        if self.guard.ledger.mutate(update):
            raise SpendBlocked('Authenticated usage requires spending reconciliation')


class AutomaticBudgetGuard(BudgetGuard):
    automatic_accounting_enabled = True

    def __init__(self, ledger, policy, expected_policy_sha, *args, **kwargs):
        super().__init__(ledger, *args, **kwargs)
        self.automatic_policy = copy.deepcopy(policy)
        self.expected_policy_sha = expected_policy_sha
        accounting.validate_policy(self.automatic_policy, expected_policy_sha, self.now())

    def _cash_exposure(self, document):
        # With Gemini's independent reload cap using the cash allocation, other
        # providers may consume only their verified prepaid/free collateral.
        accounting.assert_credit_only_funding(document, now=self.now())
        return accounting.cash_exposure(document)

    def _validate_completion(self, document, provider, reservation_id):
        """Existing work may finish without room to START another full request."""
        record = document.get('reservations', {}).get(reservation_id)
        if (document.get('schema_version') != 3 or document.get('automatic_accounting_version') != 1
            or document.get('automatic_policy_sha256') != self.expected_policy_sha
            or document.get('closed_for_rollover') or not isinstance(record, dict)
            or record.get('provider') != provider
            or record.get('automatic_policy_sha256') != self.expected_policy_sha):
            raise SpendBlocked('Completion requires the active owned automatic reservation')
        # Approval and pricing were required for the full bounded request at its
        # admission time. Completion never reauthorizes or dispatches paid work.
        created = self._timestamp(record['created_at'])
        if created > self.now():
            raise SpendBlocked('Completion cannot precede reservation creation')
        accounting.validate_policy(self.automatic_policy, self.expected_policy_sha, created)
        account = document['providers'][provider]
        pinned = self.automatic_policy['providers'][provider]['account']
        if (any(account.get(k) != pinned.get(k) for k in accounting.STATIC)
            or record.get('credential_fingerprint') != self._credential_fingerprint(provider)):
            raise SpendBlocked('Completion account or credential binding changed')
        return record

    def _freeze_usage_mismatch(self, reservation_id, response, reason):
        def update(document):
            record = self._validate_completion(document, 'gemini', reservation_id)
            document['reconciliation_required'] = True
            record['reconciliation_required'] = True
            record['unreconciled_usage'] = {'reason': reason,
                'evidence_sha256': request_digest(response.get('usageMetadata', {}))}
        self.ledger.mutate(update)

    def _validate(self, document, provider, *, ignore_reconciliation=False):
        now = self.now()
        policy = accounting.validate_policy(self.automatic_policy, self.expected_policy_sha, now)
        if (document.get('schema_version') != 3 or document.get('automatic_accounting_version') != 1
            or document.get('automatic_policy_sha256') != self.expected_policy_sha
            or document.get('credential_binding_required') is not True
            or document.get('budget_scope') != 'new_requests_after_cutoff'):
            raise SpendBlocked('Explicit approved automatic accounting epoch required')
        if document.get('closed_for_rollover') or document.get('month') != now.strftime('%Y-%m'):
            raise SpendBlocked('Automatic current-month ledger unavailable or frozen')
        cutoff = self._timestamp(document.get('budget_window_start'))
        if (cutoff > now or cutoff.strftime('%Y-%m') != document['month']
            or money(document.get('cash_ceiling_usd')) != money(policy['cash_ceiling_usd'])
            or document.get('credit_only_providers', []) != policy.get('credit_only_providers', [])):
            raise SpendBlocked('Automatic ledger epoch or cash ceiling mismatch')
        account = document.get('providers', {}).get(provider)
        pinned = policy['providers'].get(provider, {}).get('account', {})
        if not isinstance(account, dict) or any(account.get(k) != pinned.get(k) for k in accounting.STATIC):
            raise SpendBlocked('Automatic provider policy or identity changed')
        import hmac
        if not hmac.compare_digest(str(account.get('credential_fingerprint', '')), self._credential_fingerprint(provider)):
            raise SpendBlocked('Runtime credential does not match verified provider account')
        duration = account['max_request_duration_seconds']
        observed = self._timestamp(account.get('automatic_observed_at'))
        if not observed <= now or not now + timedelta(seconds=duration) < observed + timedelta(hours=24):
            raise SpendBlocked('Automatic provider observation expired')
        if not re.fullmatch('[a-f0-9]{64}', str(account.get('automatic_observation_sha256', ''))):
            raise SpendBlocked('Automatic observation identity required')
        # Do not begin a request that can cross the ledger's UTC accounting month.
        if (now + timedelta(seconds=duration)).strftime('%Y-%m') != document['month']:
            raise SpendBlocked('Provider request would cross the accounting month boundary')
        if provider == 'apify':
            starts, ends = self._timestamp(account['allowance_starts_at']), self._timestamp(account['allowance_expires_at'])
            if not starts <= now or not now + timedelta(seconds=duration) < ends:
                raise SpendBlocked('Verified allowance expires before the bounded request can finish')
        baseline = self._credit_baseline(account)
        if baseline > money(policy['providers'][provider]['credit_limit_usd']):
            raise SpendBlocked('Approved provider credit tranche was increased')
        money(account.get('remaining_credit_usd'))
        if document.get('reconciliation_required') and not ignore_reconciliation:
            raise SpendBlocked('Billing reconciliation required')
        return account

    def begin_provider_request(self, provider, operation, maximum, request_sha256):
        return self._begin(provider, operation, maximum, request_sha256)

    def _begin(self, provider, operation, maximum, request_sha256, *, context=None, reference=None):
        if not re.fullmatch('[a-f0-9]{64}', request_sha256 or ''):
            raise SpendBlocked('Exact automatic request digest required')
        reservation_id, _ = self.reserve(provider, operation, maximum, expected_reference=reference)
        def bind(document):
            self._validate(document, provider)
            document['reservations'][reservation_id].update(request_sha256=request_sha256,
                credential_fingerprint=self._credential_fingerprint(provider),
                automatic_policy_sha256=self.expected_policy_sha,
                request_context=copy.deepcopy(context or {}))
        self.ledger.mutate(bind)
        return ProviderRequest(self, reservation_id, provider, operation, request_sha256)

    def invoke_prepared_genai(self, operation, maximum, reference, bounded, target):
        def admission(document):
            account = self._validate(document, 'gemini')
            if account.get('verification_reference') != reference:
                raise SpendBlocked('Billing policy changed after token counting; retry admission')
            return copy.deepcopy(account['request_limits'][operation])
        limits = self.ledger.mutate(admission)
        method = operation.split(':', 1)[0]
        output = int(bounded['config'].get('max_output_tokens', 0)) if method == 'generate_content' else 0
        tokens = (money(maximum)*1000000 - money(output)*money(limits['output_token_usd_per_million'])) / money(limits['input_token_usd_per_million'])
        if tokens < 0 or tokens != tokens.to_integral_value():
            raise SpendBlocked('Counted admission identity is inconsistent')
        context = {'admitted_input_tokens': int(tokens), 'max_output_tokens': output}
        if method == 'embed_content':
            items = bounded['contents']
            context.update(counted_input_tokens=int(tokens), embedding_input_count=len(items) if isinstance(items, list) else 1)
        fingerprint = request_digest({'operation': operation, 'request': bounded})
        handle = self._begin('gemini', operation, maximum, fingerprint, context=context, reference=reference)
        started = self.now().isoformat()
        result = target(**bounded)  # One dispatch; failures leave the complete hold.
        try:
            response = result.model_dump(by_alias=True, exclude_none=True) if hasattr(result, 'model_dump') else copy.deepcopy(result)
        except (ValueError, TypeError, AttributeError):
            return result
        if not isinstance(response, dict):
            return result
        request_id = response.get('responseId')
        if request_id:
            # Preserve known provider identity even when its usage is unavailable.
            handle.submitted(request_id, submitted_at=started)
        try:
            def read(document):
                return copy.deepcopy(document['reservations'][handle.reservation_id])
            record = self.ledger.mutate(read)
            kwargs = dict(credential_fingerprint=handle.credential_fingerprint,
                observed_at=self.now().isoformat(), request_sha256=fingerprint)
            if method == 'generate_content':
                receipt = accounting.gemini_usage_receipt(handle.reservation_id, record, response,
                    policy=self.automatic_policy, **kwargs)
            else:
                receipt = accounting.gemini_embedding_receipt(handle.reservation_id, record, response, **kwargs)
        except SpendBlocked as error:
            if str(error) in {'Gemini usage exceeds admitted token shape', 'Gemini response model version not pinned'}:
                self._freeze_usage_mismatch(handle.reservation_id, response, str(error))
                raise SpendBlocked('Authenticated Gemini usage exceeds the admitted request policy') from None
            return result
        except (ValueError, TypeError, KeyError, AttributeError):
            # An unusable usage response does not invent a charge or release funds.
            return result
        if not request_id:
            handle.submitted(receipt['provider_request_id'], submitted_at=started)
        handle.receipt(receipt)
        return result

    def settle(self, *args, **kwargs):
        raise SpendBlocked('Automatic settlement requires a typed request-linked usage receipt')

    def refresh_verification(self, *args, **kwargs):
        raise SpendBlocked('Automatic refresh requires the approved metadata controller')
