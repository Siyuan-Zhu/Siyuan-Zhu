"""Fail-closed cash reservations shared by staging and production.

The operator provisions one private ``spend_control/YYYY-MM`` document from
verified provider billing snapshots. No allowances or free credits are inferred.
Unknown call outcomes retain their full reservation until billing reconciliation.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
import os
import threading
import random
import time
from typing import Any, Callable
import uuid

import httpx

EMBEDDING_001_MAX_COUNTED_TOKENS_PER_TEXT = 2040


class SpendBlocked(RuntimeError):
    """A paid call cannot be proven to fit the approved allowance."""

    # Only these fixed source-defined reasons may leave the private runner.
    # Unknown exception text is never transformed into a public identifier.
    _PUBLIC_REASONS = {
        "Unpriced SDK request option": "unpriced-sdk-request",
        "Unbounded model generation configuration": "unbounded-configuration",
        "Unpriced SDK configuration option": "unpriced-sdk-configuration",
        "Unbounded SDK request payload type": "unbounded-payload",
        "Request exceeds verified model input bound": "input-byte-bound",
        "Verified model pricing and request limits required": "unverified-model-policy",
        "Verified countTokens tariff required for paid Gemini call": "unverified-token-tariff",
        "Positive verified input tariff required": "invalid-input-tariff",
        "Token metadata unavailable; no paid call made": "token-metadata-unavailable",
        "Invalid countTokens metadata": "invalid-token-metadata",
        "Counted input exceeds verified token ceiling": "input-token-bound",
        "Requested output exceeds verified monetary bound": "output-token-bound",
        "Unregistered paid SDK method": "unregistered-sdk-method",
        "Request exceeds verified operation tariff": "operation-tariff-bound",
        "Call would exceed approved monthly cash ceiling": "monthly-cash-bound",
        "Billing policy changed after token counting; retry admission": "billing-policy-changed",
        "Output policy changed after request validation": "output-policy-changed",
        "Unverified Gemini 3.8 pricing or model policy": "unverified-gemini38-policy",
        "Gemini 3.8 pricing expires before bounded request completion": "pricing-validity-bound",
        "Gemini 3 sampling controls are not supported": "unsupported-sampling-controls",
        "Allowance period cannot be reset before verified expiry": "allowance-period-cannot-be-reset-before-verified-expiry",
        "Billing policy approval expired or exceeds current 24h gate": "billing-policy-approval-expired-or-exceeds-current-24h-gate",
        "Billing reconciliation required": "billing-reconciliation-required",
        "Billing verification reference changed; reload before refreshing": "billing-verification-reference-changed-reload-before-refreshing",
        "Concurrent reservation contention; no provider call made": "shared-ledger-contention",
        "Conflicting settled billing record": "conflicting-settled-billing-record",
        "Conflicting settlement; reconciliation required": "conflicting-settlement-reconciliation-required",
        "Explicit GCP_PROJECT_ID required for shared spend ledger": "explicit-gcp-project-id-required-for-shared-spend-ledger",
        "Explicit approved billing policy reference required": "explicit-approved-billing-policy-reference-required",
        "Explicit immutable credit_baseline_usd required": "explicit-immutable-credit-baseline-usd-required",
        "Frozen source settlements must follow the active carried reservation": "frozen-source-settlements-must-follow-the-active-carried-reservation",
        "Full verification cannot be newer than its oldest required evidence": "full-verification-cannot-be-newer-than-its-oldest-required-evidence",
        "Image count exceeds verified monetary bound": "image-count-exceeds-verified-monetary-bound",
        "Inline image count exceeds verified policy": "inline-image-count-exceeds-verified-policy",
        "Inline vision input must be a single raster <=384px": "inline-vision-input-must-be-a-single-raster-384px",
        "Inline vision tariff is not verified for this model": "inline-vision-tariff-is-not-verified-for-this-model",
        "Invalid budget liability cutoff": "invalid-budget-liability-cutoff",
        "Invalid free request quota": "invalid-free-request-quota",
        "Invalid inline raster": "invalid-inline-raster",
        "Invalid monetary bound": "invalid-monetary-bound",
        "Invalid timezone-aware billing timestamp": "invalid-timezone-aware-billing-timestamp",
        "Missing or invalid monetary bound": "missing-or-invalid-monetary-bound",
        "Only bounded inline raster images are priced": "only-bounded-inline-raster-images-are-priced",
        "Only bounded text embedding batches are priced": "only-bounded-text-embedding-batches-are-priced",
        "Only one priced generation candidate is permitted": "only-one-priced-generation-candidate-is-permitted",
        "Only text output is covered by the verified token tariff": "only-text-output-is-covered-by-the-verified-token-tariff",
        "Positional SDK payloads are not admitted": "positional-sdk-payloads-are-not-admitted",
        "Prepaid credit expires before bounded request can finish": "prepaid-credit-expires-before-bounded-request-can-finish",
        "Provider billing verification expired or invalid for request duration": "provider-billing-verification-expired-or-invalid-for-request-duration",
        "Provider credit/usage verification missing": "provider-credit-usage-verification-missing",
        "Refresh requires a new verification evidence reference": "refresh-requires-a-new-verification-evidence-reference",
        "Refresh requires a newly observed billing snapshot": "refresh-requires-a-newly-observed-billing-snapshot",
        "Remote file inputs require separate verified token bounds": "remote-file-inputs-require-separate-verified-token-bounds",
        "Remote file or executable SDK parts require separate verified bounds": "remote-file-or-executable-sdk-parts-require-separate-verified-bounds",
        "Runtime credential does not match verified provider account": "runtime-credential-does-not-match-verified-provider-account",
        "Runtime requires an approved credential-bound budget epoch": "runtime-requires-an-approved-credential-bound-budget-epoch",
        "Settlement must identify an existing reservation for this provider": "settlement-must-identify-an-existing-reservation-for-this-provider",
        "Shared spend ledger unavailable or unprovisioned": "shared-ledger-unavailable",
        "Shared spend ledger write outcome unknown; reconciliation required": "shared-ledger-write-outcome-unknown",
        "Spend ceiling exceeds approved 2.50 USD": "spend-ceiling-exceeds-approved-2-50-usd",
        "Spend ledger frozen for month rollover": "spend-ledger-frozen-for-month-rollover",
        "Spend ledger month does not match current billing month": "spend-ledger-month-does-not-match-current-billing-month",
        "Unpriced SDK input part": "unpriced-sdk-input-part",
        "Unpriced SDK input type": "unpriced-sdk-input-type",
        "Unpriced image aspect ratio": "unpriced-image-aspect-ratio",
        "Unpriced response modality": "unpriced-response-modality",
        "Verified 2.5 Pro reasoning allowance required": "verified-2-5-pro-reasoning-allowance-required",
        "Verified allowance expires before the bounded request can finish": "verified-allowance-expires-before-the-bounded-request-can-finish",
        "Verified model-specific thinking level required": "verified-model-specific-thinking-level-required",
        "Verified prepaid terms and disabled automatic top-up required": "verified-prepaid-terms-and-disabled-automatic-top-up-required",
        "Verified provider account and plan identity required": "verified-provider-account-and-plan-identity-required",
        "Verified request duration must be between 1 and 3600 seconds": "verified-request-duration-must-be-between-1-and-3600-seconds",
        "Verified single-image generation pricing required": "verified-single-image-generation-pricing-required",
        "Zero-cost request requires verified free tier and remaining quota": "zero-cost-request-requires-verified-free-tier-and-remaining-quota",
        "gemini-embedding-001 allows at most 2040 counted tokens per text; no paid call made": "embedding-text-token-bound",
        "A positive balance requires an explicit verified credit kind": "a-positive-balance-requires-an-explicit-verified-credit-kind",
        "Bounded Gemini request policy required": "bounded-gemini-request-policy-required",
        "CLI cutoff must exactly match reviewed metadata cutoff": "cli-cutoff-must-exactly-match-reviewed-metadata-cutoff",
        "Confirmed overrun must be reconciled before rollover": "confirmed-overrun-must-be-reconciled-before-rollover",
        "Cutoff must be an explicit past/current UTC-month timestamp": "cutoff-must-be-an-explicit-past-current-utc-month-timestamp",
        "Declared operation cap does not cover full token policy": "declared-operation-cap-does-not-cover-full-token-policy",
        "Every admitted operation requires matching primary price evidence": "every-admitted-operation-requires-matching-primary-price-evidence",
        "Exact frozen source target and updateTime required": "exact-frozen-source-target-and-updatetime-required",
        "Explicit known provider evidence required": "explicit-known-provider-evidence-required",
        "Explicit none/free/prepaid credit kind required": "explicit-none-free-prepaid-credit-kind-required",
        "Frozen source is bound to a different destination month": "frozen-source-is-bound-to-a-different-destination-month",
        "New allowance grant requires separate verified reconciliation": "new-allowance-grant-requires-separate-verified-reconciliation",
        "Operation cap must match verified positive maximum gross charge": "operation-cap-must-match-verified-positive-maximum-gross-charge",
        "Positive bounded output required": "positive-bounded-output-required",
        "Positive token and serialized-byte ceilings required": "positive-token-and-serialized-byte-ceilings-required",
        "Prepaid automatic top-up must be verified disabled": "prepaid-automatic-top-up-must-be-verified-disabled",
        "Pricing evidence expired or future-dated": "pricing-evidence-expired-or-future-dated",
        "Primary provider pricing URL required": "primary-provider-pricing-url-required",
        "Provider identity/plan/credit kind changed during rollover": "provider-identity-plan-credit-kind-changed-during-rollover",
        "Provider set cannot change during month rollover": "provider-set-cannot-change-during-month-rollover",
        "Reservation references an unknown provider": "reservation-references-an-unknown-provider",
        "Rollover cutoff must be the UTC month start": "rollover-cutoff-must-be-the-utc-month-start",
        "Rollover requires the immediately preceding canonical month": "rollover-requires-the-immediately-preceding-canonical-month",
        "Rollover would duplicate spendable credit": "rollover-would-duplicate-spendable-credit",
        "Source must be explicitly frozen without unresolved overrun": "source-must-be-explicitly-frozen-without-unresolved-overrun",
        "Token counting and tariff evidence required": "token-counting-and-tariff-evidence-required",
        "Token tariff differs from reviewed official standard price": "token-tariff-differs-from-reviewed-official-standard-price",
        "Uncontrolled or in-flight usage requires reconciliation before initial provisioning": "uncontrolled-or-in-flight-usage-requires-reconciliation-before-initial-provisio",
        "Unpriced FLUX dimensions; the verified request shape is 1200x800 at 1.5k": "unpriced-flux-dimensions-the-verified-request-shape-is-1200x800-at-1-5k",
        "Unresolved carried cash liabilities exceed new-month ceiling": "unresolved-carried-cash-liabilities-exceed-new-month-ceiling",
        "Unsupported metadata schema": "unsupported-metadata-schema",
        "Unsupported source accounting epoch": "unsupported-source-accounting-epoch",
        "Unverified model tariff; operation remains blocked": "unverified-model-tariff-operation-remains-blocked",
        "Verified pricing evidence reference required": "verified-pricing-evidence-reference-required",
        "Verified runtime credential fingerprint required": "verified-runtime-credential-fingerprint-required",
        "A pinned approved policy digest and UTC-compatible clock are required": "automatic-a-pinned-approved-policy-digest-and-utc-compatible-clock-are-required",
        "Apify account, free grant, usage cycle, build or pricing changed": "automatic-apify-account-free-grant-usage-cycle-build-or-pricing-changed",
        "Apify cycle grant repeated or overlaps prior allowance": "automatic-apify-cycle-grant-repeated-or-overlaps-prior-allowance",
        "Apify cycle not active": "automatic-apify-cycle-not-active",
        "Approved automatic policy is not installed": "automatic-approved-automatic-policy-is-not-installed",
        "Approved prior ledger is missing; initial provisioning is separate": "automatic-approved-prior-ledger-is-missing-initial-provisioning-is-separate",
        "Approved provider credit tranche was increased": "automatic-approved-provider-credit-tranche-was-increased",
        "Authenticated actor tariff observation unavailable": "automatic-authenticated-actor-tariff-observation-unavailable",
        "Automatic current-month ledger unavailable or frozen": "automatic-automatic-current-month-ledger-unavailable-or-frozen",
        "Automatic ledger epoch or cash ceiling mismatch": "automatic-automatic-ledger-epoch-or-cash-ceiling-mismatch",
        "Automatic observation identity required": "automatic-automatic-observation-identity-required",
        "Automatic policy digest mismatch": "automatic-automatic-policy-digest-mismatch",
        "Automatic policy expired or future dated": "automatic-automatic-policy-expired-or-future-dated",
        "Automatic policy must retain the 2.50 USD ceiling": "automatic-automatic-policy-must-retain-the-2-50-usd-ceiling",
        "Automatic policy requires the fixed approved private target": "automatic-automatic-policy-requires-the-fixed-approved-private-target",
        "Automatic provider observation expired": "automatic-automatic-provider-observation-expired",
        "Automatic provider policy or identity changed": "automatic-automatic-provider-policy-or-identity-changed",
        "Automatic refresh requires the approved metadata controller": "automatic-automatic-refresh-requires-the-approved-metadata-controller",
        "Automatic settlement requires a typed request-linked usage receipt": "automatic-automatic-settlement-requires-a-typed-request-linked-usage-receipt",
        "Bounded embedding input-count evidence required": "automatic-bounded-embedding-input-count-evidence-required",
        "Bounded provider request duration required": "automatic-bounded-provider-request-duration-required",
        "Bounded provider response identity required": "automatic-bounded-provider-response-identity-required",
        "Carried unknown liabilities exceed the new-month ceiling": "automatic-carried-unknown-liabilities-exceed-the-new-month-ceiling",
        "Complete authenticated Gemini usage counts required": "automatic-complete-authenticated-gemini-usage-counts-required",
        "Complete typed embedding response required": "automatic-complete-typed-embedding-response-required",
        "Confirmed overrun requires reconciliation": "automatic-confirmed-overrun-requires-reconciliation",
        "Conflicting usage receipt cannot release a hold": "automatic-conflicting-usage-receipt-cannot-release-a-hold",
        "Counted admission identity is inconsistent": "automatic-counted-admission-identity-is-inconsistent",
        "Current ledger does not match its completed handoff": "automatic-current-ledger-does-not-match-its-completed-handoff",
        "Current ledger month mismatch": "automatic-current-ledger-month-mismatch",
        "Dynamic provider price observation is unavailable": "automatic-dynamic-provider-price-observation-is-unavailable",
        "Embedding completion policy covers only embedding-001": "automatic-embedding-completion-policy-covers-only-embedding-001",
        "Exact automatic request digest required": "automatic-exact-automatic-request-digest-required",
        "Exact source revision required": "automatic-exact-source-revision-required",
        "Explicit approved automatic accounting epoch required": "automatic-explicit-approved-automatic-accounting-epoch-required",
        "Explicit approved automatic policy required": "automatic-explicit-approved-automatic-policy-required",
        "Explicit primary tariff evidence mode required": "automatic-explicit-primary-tariff-evidence-mode-required",
        "Explicit provider usage-accounting policy required": "automatic-explicit-provider-usage-accounting-policy-required",
        "Finite typed embedding vectors required": "automatic-finite-typed-embedding-vectors-required",
        "Frozen or unreconciled ledger cannot refresh": "automatic-frozen-or-unreconciled-ledger-cannot-refresh",
        "Gemini automation has no allocated prepaid credit": "automatic-gemini-automation-has-no-allocated-prepaid-credit",
        "Gemini response model version not pinned": "automatic-gemini-response-model-version-not-pinned",
        "Gemini usage exceeds admitted token shape": "automatic-gemini-usage-exceeds-admitted-token-shape",
        "Handoff belongs to another approved policy": "automatic-handoff-belongs-to-another-approved-policy",
        "Handoff intent changed before completion": "automatic-handoff-intent-changed-before-completion",
        "Handoff state requires reconciliation": "automatic-handoff-state-requires-reconciliation",
        "Immediately preceding exact canonical source revision required": "automatic-immediately-preceding-exact-canonical-source-revision-required",
        "Immutable prepaid tranche was increased": "automatic-immutable-prepaid-tranche-was-increased",
        "Incomplete pinned provider policy": "automatic-incomplete-pinned-provider-policy",
        "Invalid Gemini usage count": "automatic-invalid-gemini-usage-count",
        "Invalid private handoff intent": "automatic-invalid-private-handoff-intent",
        "Ledger cash ceiling differs from approved policy": "automatic-ledger-cash-ceiling-differs-from-approved-policy",
        "Legacy holds require explicit usage-accounting policy activation": "automatic-legacy-holds-require-explicit-usage-accounting-policy-activation",
        "Legacy ledger requires separately approved policy installation": "automatic-legacy-ledger-requires-separately-approved-policy-installation",
        "Legacy, frozen, or unapproved ledger requires separate review": "automatic-legacy-frozen-or-unapproved-ledger-requires-separate-review",
        "Month handoff plan digest mismatch": "automatic-month-handoff-plan-digest-mismatch",
        "Month handoff unknown or divergent; do not redispatch": "automatic-month-handoff-unknown-or-divergent-do-not-redispatch",
        "New authenticated metadata observation required": "automatic-new-authenticated-metadata-observation-required",
        "Observed free credit exceeds the approved grant": "automatic-observed-free-credit-exceeds-the-approved-grant",
        "One-time top-up OFF approval is required; never inferred from metadata": "automatic-one-time-top-up-off-approval-is-required-never-inferred-from-metadata",
        "Only the verified FREE five-dollar recurring grant is supported": "automatic-only-the-verified-free-five-dollar-recurring-grant-is-supported",
        "Original published tariff validity commitment required": "automatic-original-published-tariff-validity-commitment-required",
        "Pinned credential fingerprint required": "automatic-pinned-credential-fingerprint-required",
        "Pinned primary-provider price evidence required": "automatic-pinned-primary-provider-price-evidence-required",
        "Pinned provider identity, tariff or request policy changed": "automatic-pinned-provider-identity-tariff-or-request-policy-changed",
        "Pinned provider pricing or policy expires before request completion": "automatic-pinned-provider-pricing-or-policy-expires-before-request-completion",
        "Primary price snapshot changed, stale or missing": "automatic-primary-price-snapshot-changed-stale-or-missing",
        "Private maintenance outcome unknown; reconciliation required": "automatic-private-maintenance-outcome-unknown-reconciliation-required",
        "Provider handle no longer matches its reservation": "automatic-provider-handle-no-longer-matches-its-reservation",
        "Provider metadata or credential binding failed": "automatic-provider-metadata-or-credential-binding-failed",
        "Provider request would cross the accounting month boundary": "automatic-provider-request-would-cross-the-accounting-month-boundary",
        "Provider reservation already has another request identity": "automatic-provider-reservation-already-has-another-request-identity",
        "Provider response already belongs to another reservation": "automatic-provider-response-already-belongs-to-another-reservation",
        "Provider submission timestamp outside reservation window": "automatic-provider-submission-timestamp-outside-reservation-window",
        "Public BFL tariff is unavailable, ambiguous or changed": "automatic-public-bfl-tariff-is-unavailable-ambiguous-or-changed",
        "Public BFL tariff observation unavailable; no price inferred": "automatic-public-bfl-tariff-observation-unavailable-no-price-inferred",
        "Refresh plan lost its source revision": "automatic-refresh-plan-lost-its-source-revision",
        "Required exact Gemini model/method unavailable": "automatic-required-exact-gemini-model-method-unavailable",
        "Retained liabilities exhaust the 2.50 USD monthly ceiling": "automatic-retained-liabilities-exhaust-the-2-50-usd-monthly-ceiling",
        "Reviewed BFL prepaid conversion required": "automatic-reviewed-bfl-prepaid-conversion-required",
        "Settle only the active carried reservation, never the frozen source": "automatic-settle-only-the-active-carried-reservation-never-the-frozen-source",
        "Typed authenticated usage evidence required": "automatic-typed-authenticated-usage-evidence-required",
        "Typed text-only Gemini response identity required": "automatic-typed-text-only-gemini-response-identity-required",
        "Unapproved private spend document": "automatic-unapproved-private-spend-document",
        "Unapproved private submission metadata": "automatic-unapproved-private-submission-metadata",
        "Unbounded automatic request identity": "automatic-unbounded-automatic-request-identity",
        "Unknown provider liability cannot be discarded": "automatic-unknown-provider-liability-cannot-be-discarded",
        "Unknown usage receipt policy": "automatic-unknown-usage-receipt-policy",
        "Unpriced tool use or inconsistent Gemini usage": "automatic-unpriced-tool-use-or-inconsistent-gemini-usage",
        "Usage receipt belongs to a different request handle": "automatic-usage-receipt-belongs-to-a-different-request-handle",
        "Usage receipt exceeds reservation; reconciliation required": "automatic-usage-receipt-exceeds-reservation-reconciliation-required",
        "Usage receipt must identify an existing reservation": "automatic-usage-receipt-must-identify-an-existing-reservation",
        "Usage receipt policy changed after reservation": "automatic-usage-receipt-policy-changed-after-reservation",
        "Usage receipt request or credential binding mismatch": "automatic-usage-receipt-request-or-credential-binding-mismatch",
        "Usage receipt timestamp outside request window": "automatic-usage-receipt-timestamp-outside-request-window",
        "Authenticated Gemini usage exceeds the admitted request policy": "automatic-authenticated-gemini-usage-exceeds-the-admitted-request-policy",
        "Authenticated usage requires spending reconciliation": "automatic-authenticated-usage-requires-spending-reconciliation",
        "Completion account or credential binding changed": "automatic-completion-account-or-credential-binding-changed",
        "Completion cannot precede reservation creation": "automatic-completion-cannot-precede-reservation-creation",
        "Completion requires the active owned automatic reservation": "automatic-completion-requires-the-active-owned-automatic-reservation",
        "Approved immutable actor build image and schema required": "automatic-approved-immutable-actor-build-image-and-schema-required",
        "Handoff intent changed before rejection was recorded": "automatic-handoff-intent-changed-before-rejection-was-recorded",
        "Handoff rejection evidence is incomplete": "automatic-handoff-rejection-evidence-is-incomplete",
        "Handoff rejection history requires reconciliation": "automatic-handoff-rejection-history-requires-reconciliation",
        "Rejected handoff state requires reconciliation": "automatic-rejected-handoff-state-requires-reconciliation",
        'Automatic policy exceeds the approved 10 USD cash ceiling': 'automatic-automatic-policy-exceeds-the-approved-10-usd-cash-ceiling',
        'Credit-only provider has no remaining verified collateral': 'automatic-credit-only-provider-has-no-remaining-verified-collateral',
        'Retained liabilities exhaust the approved monthly cash ceiling': 'automatic-retained-liabilities-exhaust-the-approved-monthly-cash-ceiling',
        'Unapproved credit-only funding configuration': 'automatic-unapproved-credit-only-funding-configuration',
        'Verified bounded Gemini reload and credit-only peer providers required': 'automatic-verified-bounded-gemini-reload-and-credit-only-peer-providers-required',
        'Credit-only funding has unattributed prior cash liability': 'automatic-credit-only-funding-has-unattributed-prior-cash-liability',
        'Credit-only funding requires a timezone-aware current time': 'automatic-credit-only-funding-requires-a-timezone-aware-current-time',
        'Credit-only provider collateral is outside its verified allowance period': 'automatic-credit-only-provider-collateral-is-outside-its-verified-allowance-period',
        'Credit-only provider has an uncollateralized retained liability': 'automatic-credit-only-provider-has-an-uncollateralized-retained-liability',
        'Ledger credit-only funding differs from approved policy': 'automatic-ledger-credit-only-funding-differs-from-approved-policy',
        'Public Gemini tariff is unavailable, ambiguous or changed': 'automatic-public-gemini-tariff-is-unavailable-ambiguous-or-changed',
        'Public Gemini tariff observation unavailable; no price inferred': 'automatic-public-gemini-tariff-observation-unavailable-no-price-inferred',
    }

    def __init__(self, message, *, provider_status=None, provider_error_type=None):
        super().__init__(message)
        self.public_diagnostics = {
            "code": self._PUBLIC_REASONS.get(message, "budget-admission-rejected")
        }
        if message == "Token metadata unavailable; no paid call made" or message in {
            "Shared spend ledger unavailable or unprovisioned",
            "Shared spend ledger write outcome unknown; reconciliation required",
            "Concurrent reservation contention; no provider call made",
        }:
            self.public_diagnostics["provider"] = (
                "gemini"
                if message == "Token metadata unavailable; no paid call made"
                else "firestore"
            )
            if type(provider_status) is int and 100 <= provider_status <= 599:
                self.public_diagnostics["status"] = provider_status
            if isinstance(provider_error_type, str) and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.-]{0,79}", provider_error_type
            ):
                self.public_diagnostics["error_type"] = provider_error_type

def money(value: Any) -> Decimal:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise SpendBlocked("Missing or invalid monetary bound") from None
    if not amount.is_finite() or amount < 0:
        raise SpendBlocked("Invalid monetary bound")
    return amount


class MemoryLedger:
    """Atomic injectable ledger for offline tests; never selected by runtime."""

    def __init__(self, document: dict):
        self.document = copy.deepcopy(document)
        self.lock = threading.Lock()

    def mutate(self, fn):
        with self.lock:
            candidate = copy.deepcopy(self.document)
            result = fn(candidate)
            self.document = candidate
            return result


class FirestoreLedger:
    """Private Firestore document updated using optimistic updateTime CAS.

    Collection is deliberately NOT FIRESTORE_COLLECTION_PREFIX-dependent: paid
    calls from staging and production consume the same provider accounts.
    """

    def __init__(self, project_id: str, month: str):
        if not project_id:
            raise SpendBlocked(
                "Explicit GCP_PROJECT_ID required for shared spend ledger"
            )
        from src.state_store import _FirestoreRestClient

        self.client = _FirestoreRestClient(
            project_id, os.environ.get("FIRESTORE_DATABASE_ID", "(default)")
        )
        self.url = self.client._base_url() + "/spend_control/" + month

    def mutate(self, fn):
        from src.state_store import (
            _firestore_value_to_py,
            _py_to_firestore_value,
            _firestore_error_status,
        )

        # Only a definitive failed condition can repeat a write. A missing reply
        # may already have reserved money and must stop provider dispatch.
        status = None
        error_type = None
        try:
            headers = {"Authorization": "Bearer " + self.client._resolve_token()}
            with httpx.Client(timeout=15) as client:
                for attempt in range(5):
                    status, error_type = None, None
                    response = client.get(self.url, headers=headers)
                    response.raise_for_status()
                    body = response.json()
                    update_time = body["updateTime"]
                    candidate = {
                        k: _firestore_value_to_py(v) for k, v in body["fields"].items()
                    }
                    result = fn(candidate)
                    fields = {k: _py_to_firestore_value(v) for k, v in candidate.items()}
                    try:
                        written = client.patch(
                            self.url,
                            headers=headers,
                            params={"currentDocument.updateTime": update_time},
                            json={"fields": fields},
                        )
                    except BaseException as exc:
                        raise SpendBlocked(
                            "Shared spend ledger write outcome unknown; reconciliation required",
                            provider_error_type=type(exc).__name__,
                        ) from None
                    status = written.status_code
                    error_type = None
                    if status >= 400:
                        try:
                            error_type = _firestore_error_status(written.json())
                        except Exception:
                            pass
                    if status in (409, 412) or (
                        status == 400 and error_type in {"FAILED_PRECONDITION", "ABORTED"}
                    ):
                        if attempt < 4:
                            time.sleep(random.uniform(0.01, 0.05) * (attempt + 1))
                        continue
                    if status >= 500:
                        raise SpendBlocked(
                            "Shared spend ledger write outcome unknown; reconciliation required",
                            provider_status=status,
                            provider_error_type=error_type or "HTTPStatusError",
                        )
                    written.raise_for_status()
                    try:
                        acknowledgement = written.json()
                        if (
                            not isinstance(acknowledgement, dict)
                            or not isinstance(acknowledgement.get("updateTime"), str)
                            or not acknowledgement["updateTime"]
                            or not isinstance(acknowledgement.get("fields"), dict)
                            or {
                                k: _firestore_value_to_py(v)
                                for k, v in acknowledgement["fields"].items()
                            }
                            != candidate
                        ):
                            raise ValueError("Incomplete ledger acknowledgement")
                    except BaseException as exc:
                        raise SpendBlocked(
                            "Shared spend ledger write outcome unknown; reconciliation required",
                            provider_status=status,
                            provider_error_type=type(exc).__name__,
                        ) from None
                    return result
        except SpendBlocked:
            raise
        except Exception as exc:
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                try:
                    error_type = _firestore_error_status(exc.response.json())
                except Exception:
                    error_type = None
            raise SpendBlocked(
                "Shared spend ledger unavailable or unprovisioned",
                provider_status=status,
                provider_error_type=error_type or type(exc).__name__,
            ) from None
        raise SpendBlocked(
            "Concurrent reservation contention; no provider call made",
            provider_status=status,
            provider_error_type=error_type or "HTTPStatusError",
        )


class BudgetGuard:
    def __init__(self, ledger, now: Callable[[], datetime] | None = None, *,
                 credential_fingerprints=None, require_credential_binding=False):
        self.ledger = ledger
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.credential_fingerprints = credential_fingerprints
        self.require_credential_binding = require_credential_binding

    def _credential_fingerprint(self, provider):
        if self.credential_fingerprints is not None:
            return self.credential_fingerprints.get(provider, "")
        names = {"gemini": ("GEMINI_API_KEY",), "bfl": ("BFL_API_KEY",),
                 "apify": ("APIFY_API_TOKEN", "APIFY_TOKEN")}
        key = next((os.environ.get(n, "").strip() for n in names.get(provider, ()) if os.environ.get(n, "").strip()), "")
        return hashlib.sha256(key.encode()).hexdigest()[:16] if key else ""

    @staticmethod
    def _timestamp(value):
        try:
            result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if result.tzinfo is None:
                raise ValueError()
            return result
        except (ValueError, TypeError):
            raise SpendBlocked("Invalid timezone-aware billing timestamp") from None

    @staticmethod
    def _credit_baseline(account):
        if "credit_baseline_usd" in account:
            return money(account["credit_baseline_usd"])
        # Old zero-credit documents cannot grant an inferred allowance.
        if money(account.get("remaining_credit_usd")) != 0:
            raise SpendBlocked("Explicit immutable credit_baseline_usd required")
        return Decimal("0")

    def _validate(self, doc, provider, *, ignore_reconciliation=False):
        now = self.now()
        if doc.get("closed_for_rollover"):
            raise SpendBlocked("Spend ledger frozen for month rollover")
        if self.require_credential_binding and (
            doc.get("schema_version") != 2 or doc.get("credential_binding_required") is not True
            or doc.get("budget_scope") != "new_requests_after_cutoff"
        ):
            raise SpendBlocked("Runtime requires an approved credential-bound budget epoch")
        if doc.get("month") != now.strftime("%Y-%m"):
            raise SpendBlocked(
                "Spend ledger month does not match current billing month"
            )
        if money(doc.get("cash_ceiling_usd")) > Decimal("2.50"):
            raise SpendBlocked("Spend ceiling exceeds approved 2.50 USD")
        if doc.get("budget_scope") == "new_requests_after_cutoff":
            cutoff = self._timestamp(doc.get("budget_window_start"))
            if not cutoff <= now or cutoff.strftime("%Y-%m") != doc["month"]:
                raise SpendBlocked("Invalid budget liability cutoff")
        account = doc.get("providers", {}).get(provider)
        if not isinstance(account, dict) or not account.get("verification_reference"):
            raise SpendBlocked("Provider credit/usage verification missing")
        if not account.get("account_id") or not account.get("plan_id"):
            raise SpendBlocked("Verified provider account and plan identity required")
        if self.require_credential_binding or doc.get("credential_binding_required"):
            expected = str(account.get("credential_fingerprint", ""))
            actual = self._credential_fingerprint(provider)
            if not re.fullmatch("[0-9a-f]{16}", expected) or not hmac.compare_digest(expected, actual):
                raise SpendBlocked("Runtime credential does not match verified provider account")
        duration = int(account.get("max_request_duration_seconds", 300))
        if not 1 <= duration <= 3600:
            raise SpendBlocked("Verified request duration must be between 1 and 3600 seconds")
        verified = self._timestamp(account.get("verified_at"))
        if not (verified <= now and now + timedelta(seconds=duration) < verified + timedelta(hours=24)):
            raise SpendBlocked("Provider billing verification expired or invalid for request duration")
        # A balance GET cannot renew separately observed manual plan, tariff or
        # top-up evidence. These optional explicit timestamps retain the existing
        # full-verification 24h gate; they never extend it.
        split_fields = ("balance_verified_at", "policy_verified_at", "policy_valid_until", "policy_reference")
        if any(key in account for key in split_fields):
            if not account.get("policy_reference"):
                raise SpendBlocked("Explicit approved billing policy reference required")
            balance_at = self._timestamp(account.get("balance_verified_at"))
            policy_at = self._timestamp(account.get("policy_verified_at"))
            policy_until = self._timestamp(account.get("policy_valid_until"))
            if verified > min(balance_at, policy_at) or not policy_at <= now or not balance_at <= now:
                raise SpendBlocked("Full verification cannot be newer than its oldest required evidence")
            if not now + timedelta(seconds=duration) < policy_until <= policy_at + timedelta(hours=24):
                raise SpendBlocked("Billing policy approval expired or exceeds current 24h gate")
        baseline = self._credit_baseline(account)
        prepaid = account.get("credit_kind") == "prepaid"
        if prepaid:
            if account.get("automatic_top_up_enabled") is not False or not account.get("credit_terms_reference"):
                raise SpendBlocked("Verified prepaid terms and disabled automatic top-up required")
            if account.get("prepaid_credits_do_not_expire") is not True:
                expires = self._timestamp(account.get("allowance_expires_at"))
                if now + timedelta(seconds=duration) >= expires:
                    raise SpendBlocked("Prepaid credit expires before bounded request can finish")
        if (baseline > 0 and not prepaid) or account.get("billing_mode") == "verified_free_tier":
            starts = self._timestamp(account.get("allowance_starts_at"))
            expires = self._timestamp(account.get("allowance_expires_at"))
            if not account.get("allowance_period_id") or not (starts <= now and now + timedelta(seconds=duration) < expires):
                raise SpendBlocked("Verified allowance expires before the bounded request can finish")
        money(account.get("remaining_credit_usd"))
        if doc.get("reconciliation_required") and not ignore_reconciliation:
            raise SpendBlocked("Billing reconciliation required")
        return account

    @staticmethod
    def _cash_exposure(doc):
        gross = {}
        for reservation in doc.get("reservations", {}).values():
            provider = reservation["provider"]
            amount = money(
                reservation.get("actual_gross_usd", reservation["max_gross_usd"])
            )
            gross[provider] = gross.get(provider, Decimal("0")) + amount
        total = money(doc.get("cash_spent_before_ledger_usd"))
        for provider, amount in gross.items():
            credit = BudgetGuard._credit_baseline(doc["providers"][provider])
            total += max(Decimal("0"), amount - credit)
        return total

    def reserve(self, provider: str, operation: str, maximum=None, *, expected_reference=None):
        reservation_id = uuid.uuid4().hex

        def update(doc):
            account = self._validate(doc, provider)
            if expected_reference is not None and account.get("verification_reference") != expected_reference:
                raise SpendBlocked("Billing policy changed after token counting; retry admission")
            cap = money(
                maximum
                if maximum is not None
                else account.get("operation_caps_usd", {}).get(operation)
            )
            if self.require_credential_binding or doc.get("credential_binding_required"):
                approved = account.get("operation_caps_usd", {}).get(operation)
                if approved is None or cap > money(approved):
                    raise SpendBlocked("Request exceeds verified operation tariff")
            if cap == 0:
                free_ops = account.get("verified_free_operations", [])
                remaining = int(account.get("free_request_limit", 0))
                used = sum(
                    1
                    for r in doc.get("reservations", {}).values()
                    if r["provider"] == provider and money(r["max_gross_usd"]) == 0
                    and r.get("allowance_period_id") == account.get("allowance_period_id")
                )
                if (
                    account.get("billing_mode") != "verified_free_tier"
                    or operation not in free_ops
                    or used >= remaining
                ):
                    raise SpendBlocked(
                        "Zero-cost request requires verified free tier and remaining quota"
                    )
            doc.setdefault("reservations", {})[reservation_id] = {
                "provider": provider,
                "operation": operation,
                "max_gross_usd": str(cap),
                "status": "reserved",
                "created_at": self.now().isoformat(),
                "allowance_period_id": account.get("allowance_period_id"),
            }
            if self._cash_exposure(doc) > money(doc["cash_ceiling_usd"]):
                raise SpendBlocked("Call would exceed approved monthly cash ceiling")
            return cap

        cap = self.ledger.mutate(update)
        return reservation_id, cap

    def settle(self, reservation_id, actual_gross_usd):
        actual = money(actual_gross_usd)

        def update(doc):
            if doc.get("closed_for_rollover"):
                raise SpendBlocked("Frozen source settlements must follow the active carried reservation")
            record = doc["reservations"][reservation_id]
            if "actual_gross_usd" in record:
                if money(record["actual_gross_usd"]) != actual:
                    raise SpendBlocked(
                        "Conflicting settlement; reconciliation required"
                    )
                return
            record.update(actual_gross_usd=str(actual), status="settled")
            if actual > money(record["max_gross_usd"]):
                doc["reconciliation_required"] = True

        self.ledger.mutate(update)

    def refresh_verification(self, provider, snapshot, *, expected_reference, settlements=None):
        """Atomically admit a fresh billing snapshot without spending old credits twice.

        Snapshot authenticity is an operator/metadata-workflow responsibility. The
        caller supplies an evidence hash, stable account identity and current plan.
        Unknown reservations remain at their full bound until explicitly settled.
        """
        fresh = copy.deepcopy(snapshot)
        settlements = settlements or {}

        def update(doc):
            account = doc.get("providers", {}).get(provider)
            if not isinstance(account, dict) or account.get("verification_reference") != expected_reference:
                raise SpendBlocked("Billing verification reference changed; reload before refreshing")
            now = self.now()
            verified = self._timestamp(fresh.get("verified_at"))
            if not timedelta(0) <= now - verified <= timedelta(minutes=5):
                raise SpendBlocked("Refresh requires a newly observed billing snapshot")
            if not fresh.get("verification_reference") or fresh["verification_reference"] == expected_reference:
                raise SpendBlocked("Refresh requires a new verification evidence reference")
            fields = ["account_id", "plan_id"]
            if doc.get("credential_binding_required") or self.require_credential_binding:
                fields.extend(["credential_fingerprint", "credit_kind"])
            for field in fields:
                if not fresh.get(field) or (account.get(field) and fresh[field] != account[field]):
                    raise SpendBlocked(f"Provider {field} changed; explicit reprovisioning required")
            old_period = account.get("allowance_period_id")
            new_period = fresh.get("allowance_period_id")
            if old_period and new_period != old_period:
                # Daily zero-dollar quotas can roll forward, but a monetary
                # credit grant must never silently reset within a monthly ledger.
                if self._credit_baseline(account) > 0 or now < self._timestamp(account.get("allowance_expires_at")):
                    raise SpendBlocked("Allowance period cannot be reset before verified expiry")
            reservations = doc.get("reservations", {})
            for reservation_id, amount in settlements.items():
                record = reservations.get(reservation_id)
                if not record or record.get("provider") != provider:
                    raise SpendBlocked("Settlement must identify an existing reservation for this provider")
                actual = money(amount)
                if "actual_gross_usd" in record and money(record["actual_gross_usd"]) != actual:
                    raise SpendBlocked("Conflicting settled billing record")
                record.update(actual_gross_usd=str(actual), status="settled",
                              billing_reference=fresh["verification_reference"])
            confirmed = sum((money(r["actual_gross_usd"]) for r in reservations.values()
                             if r.get("provider") == provider and "actual_gross_usd" in r), Decimal("0"))
            observed = money(fresh.get("remaining_credit_usd"))
            baseline = self._credit_baseline(account)
            # A billing refresh can shrink an allowance consumed elsewhere, but
            # cannot grant additional credit or treat pending requests as settled.
            replacement = copy.deepcopy(fresh)
            replacement["credit_baseline_usd"] = str(min(baseline, observed + confirmed))
            old_free_baseline = int(account.get("free_request_limit", 0))
            observed_free = int(fresh.get("free_request_limit", 0))
            if observed_free < 0:
                raise SpendBlocked("Invalid free request quota")
            if new_period == old_period:
                confirmed_free = sum(1 for r in reservations.values()
                    if r.get("provider") == provider and r.get("allowance_period_id") == old_period
                    and money(r.get("actual_gross_usd", r["max_gross_usd"])) == 0
                    and "actual_gross_usd" in r)
                replacement["free_request_limit"] = min(old_free_baseline, observed_free + confirmed_free)
            doc["providers"][provider] = replacement
            self._validate(doc, provider, ignore_reconciliation=True)
            overrun_unreconciled = any("actual_gross_usd" in r
                and money(r["actual_gross_usd"]) > money(r["max_gross_usd"])
                and not r.get("billing_reference") for r in reservations.values())
            doc["reconciliation_required"] = bool(overrun_unreconciled or
                self._cash_exposure(doc) > money(doc["cash_ceiling_usd"]))
            return {"verification_reference": fresh["verification_reference"],
                    "credit_baseline_usd": replacement["credit_baseline_usd"],
                    "remaining_credit_usd": str(observed),
                    "cash_exposure_usd": str(self._cash_exposure(doc)),
                    "reconciliation_required": doc["reconciliation_required"]}

        return self.ledger.mutate(update)

    def call(self, provider, operation, maximum, fn, *, expected_reference=None):
        reservation_id, _ = self.reserve(provider, operation, maximum, expected_reference=expected_reference)
        # Provider exceptions, lost responses and successful calls with unknown
        # billing all retain the conservative bound. Only reconciled billing may
        # release unused credit/cash using settle().
        return fn()

    def _validate_gemini38_policy(self, limits, duration):
        """Introductory tariff and bounded multimodal shape, never inferred credits."""
        if (money(limits.get("input_token_usd_per_million")) != Decimal("0.75")
                or money(limits.get("output_token_usd_per_million")) != Decimal("3.75")
                or limits.get("thinking_level") not in ("low", "medium", "high")
                or limits.get("image_tokens_per_image") != 1120
                or limits.get("media_resolution") != "MEDIA_RESOLUTION_HIGH"):
            raise SpendBlocked("Unverified Gemini 3.8 pricing or model policy")
        for field, ceiling in (("max_input_tokens", 65536), ("max_output_tokens", 1500),
                               ("max_input_bytes", 2000000)):
            value = limits.get(field)
            if type(value) is not int or not 1 <= value <= ceiling:
                raise SpendBlocked("Unverified Gemini 3.8 pricing or model policy")
        images = limits.get("max_input_images")
        if type(images) is not int or not 0 <= images <= 9:
            raise SpendBlocked("Unverified Gemini 3.8 pricing or model policy")
        expires = self._timestamp(limits.get("pricing_valid_until"))
        announced = self._timestamp("2026-12-31T23:59:59Z")
        if not self.now() + timedelta(seconds=duration) < expires <= announced:
            raise SpendBlocked("Gemini 3.8 pricing expires before bounded request completion")

    def bounded_genai_request(self, operation, kwargs):
        """Admit only the SDK request shapes whose costs were verified."""
        method, model = operation.split(":", 1)
        if set(kwargs) - {"model", "contents", "prompt", "config"}:
            raise SpendBlocked("Unpriced SDK request option")
        def read(doc):
            account = self._validate(doc, "gemini")
            limits = account.get("request_limits", {}).get(operation)
            if not isinstance(limits, dict) or not limits.get("pricing_reference"):
                raise SpendBlocked("Verified model pricing and request limits required")
            duration = int(account.get("max_request_duration_seconds", 300))
            if model == "gemini-3.8-flash":
                self._validate_gemini38_policy(limits, duration)
            return copy.deepcopy(limits), duration
        limits, duration = self.ledger.mutate(read)
        config = kwargs.get("config") or {}
        if hasattr(config, "model_dump"):
            config = config.model_dump(exclude_none=True)
        if not isinstance(config, dict):
            raise SpendBlocked("Unbounded model generation configuration")
        config = dict(config)
        allowed = {
            "generate_content": {"temperature", "response_mime_type", "max_output_tokens", "response_modalities", "candidate_count"},
            "generate_images": {"number_of_images", "aspect_ratio"},
            "embed_content": {"output_dimensionality"},
        }.get(method, set())
        if set(config) - allowed:
            raise SpendBlocked("Unpriced SDK configuration option")
        # No system instructions, tools, automatic function calling, file URIs
        # or caller-provided transports can bypass the verified request shape.
        import json
        def encode(obj):
            if isinstance(obj, bytes):
                import base64
                return base64.b64encode(obj).decode("ascii")
            if hasattr(obj, "model_dump"):
                payload = obj.model_dump(exclude_none=True)
                if isinstance(payload, dict) and payload.get("file_data"):
                    raise SpendBlocked("Remote file inputs require separate verified token bounds")
                return payload
            raise SpendBlocked("Unbounded SDK request payload type")
        serialized = json.dumps(kwargs.get("contents", kwargs.get("prompt", "")), default=encode).encode()
        def reject_unbounded_parts(value):
            if isinstance(value, dict):
                if set(value) & {"file_data", "file_uri", "function_call", "function_response", "executable_code", "code_execution_result"}:
                    raise SpendBlocked("Remote file or executable SDK parts require separate verified bounds")
                for child in value.values():
                    reject_unbounded_parts(child)
            elif isinstance(value, list):
                for child in value:
                    reject_unbounded_parts(child)
        reject_unbounded_parts(json.loads(serialized))
        max_bytes = int(limits.get("max_input_bytes", 0))
        if max_bytes <= 0 or len(serialized) > max_bytes:
            raise SpendBlocked("Request exceeds verified model input bound")
        if method == "generate_content":
            if int(config.get("candidate_count", 1)) != 1:
                raise SpendBlocked("Only one priced generation candidate is permitted")
            if model.startswith("gemini-3"):
                if "temperature" in config or "candidate_count" in config:
                    raise SpendBlocked("Gemini 3 sampling controls are not supported")
            else:
                config["candidate_count"] = 1
            output = int(limits.get("max_output_tokens", 0))
            requested = int(config.get("max_output_tokens", output))
            if output <= 0 or requested <= 0 or requested > output:
                raise SpendBlocked("Requested output exceeds verified monetary bound")
            config["max_output_tokens"] = requested
            modalities = config.get("response_modalities", ["TEXT"])
            if not isinstance(modalities, list) or any(m not in ("TEXT", "IMAGE") for m in modalities):
                raise SpendBlocked("Unpriced response modality")
            if "IMAGE" in modalities:
                if int(limits.get("max_images", 0)) != 1 or "IMAGE" not in limits.get("response_modalities", []):
                    raise SpendBlocked("Verified single-image generation pricing required")
            elif model.startswith("gemini-2.5-flash"):
                config["thinking_config"] = {"thinking_budget": 0}
            elif model.startswith("gemini-2.5-pro"):
                thinking = int(limits.get("max_thinking_tokens", 0))
                if not 128 <= thinking <= output:
                    raise SpendBlocked("Verified 2.5 Pro reasoning allowance required")
                config["thinking_config"] = {"thinking_budget": thinking}
            elif model.startswith("gemini-3"):
                level = limits.get("thinking_level")
                allowed_levels = (["low", "medium", "high"] if model == "gemini-3.8-flash" else
                    ["low", "high"] if "pro" in model else ["minimal", "low", "medium", "high"])
                if level not in allowed_levels:
                    raise SpendBlocked("Verified model-specific thinking level required")
                config["thinking_config"] = {"thinking_level": level}
                if model == "gemini-3.8-flash":
                    from google.genai import types
                    # Older SDKs can use the documented default (also 1120/image).
                    if "media_resolution" in types.GenerateContentConfig.model_fields:
                        config["media_resolution"] = "MEDIA_RESOLUTION_HIGH"
        elif method == "generate_images":
            count = int(limits.get("max_images", 0))
            requested = int(config.get("number_of_images", 1))
            if count <= 0 or not 1 <= requested <= count:
                raise SpendBlocked("Image count exceeds verified monetary bound")
            config["number_of_images"] = requested
            if config.get("aspect_ratio", "3:2") not in limits.get("aspect_ratios", ["3:2"]):
                raise SpendBlocked("Unpriced image aspect ratio")
        # Timeout and retries are controlled here, never by unpriced caller config.
        config["http_options"] = {"timeout": duration * 1000, "retry_options": {"attempts": 1}}
        return dict(kwargs, config=config)

    def counted_genai_cap(self, operation, bounded, count_tokens):
        """Count the immutable admitted input, then reserve its verified tariff.

        countTokens is an unbilled metadata endpoint. Inline vision uses the
        documented <=384px image charge as a floor instead of trusting a media
        estimate as an exact billing measurement.
        """
        def policy(doc):
            account = self._validate(doc, 'gemini')
            required = account.get('token_pricing_required') or doc.get('credential_binding_required') or self.require_credential_binding
            limits = account.get('request_limits', {}).get(operation)
            if operation == 'generate_content:gemini-3.8-flash':
                if not isinstance(limits, dict):
                    raise SpendBlocked('Verified model pricing and request limits required')
                self._validate_gemini38_policy(limits, int(account.get('max_request_duration_seconds', 300)))
            return (copy.deepcopy(limits), required, account['verification_reference'])
        limits, required, reference = self.ledger.mutate(policy)
        if not required:
            return None, None  # Explicit injected legacy test seam; runtime requires binding.
        method, model = operation.split(':', 1)
        if method not in ('generate_content', 'embed_content') or not isinstance(limits, dict) or not limits.get('count_tokens_reference'):
            raise SpendBlocked('Verified countTokens tariff required for paid Gemini call')
        input_rate = money(limits.get('input_token_usd_per_million'))
        output_rate = money(limits.get('output_token_usd_per_million'))
        if input_rate <= 0:
            raise SpendBlocked('Positive verified input tariff required')
        contents = bounded.get('contents')
        transport = {'http_options': copy.deepcopy(bounded['config']['http_options'])}
        # Developer API CountTokens rejects generation_config. Its documented
        # DEFAULT allocation is the same 1120/image as explicit HIGH generation.
        def count(value):
            try:
                result = count_tokens(model=model, contents=value, config=copy.deepcopy(transport))
                total = result.total_tokens
            except Exception as error:
                raise SpendBlocked('Token metadata unavailable; no paid call made',
                    provider_status=getattr(error, 'code', None),
                    provider_error_type=type(error).__name__) from None
            if type(total) is not int or total < 0:
                raise SpendBlocked('Invalid countTokens metadata')
            return total
        if method == 'embed_content':
            batch = contents if isinstance(contents, list) else [contents]
            if not batch or len(batch) > 100 or any(not isinstance(item, str) for item in batch):
                raise SpendBlocked('Only bounded text embedding batches are priced')
            # EmbedContent treats strings as distinct requests. Count each string
            # independently; a joined prompt can undercount per-item boundaries.
            total = 0
            for item in batch:
                item_tokens = count(item)
                # embedding-001 can silently truncate oversized individual inputs.
                # Keep eight tokens of its 2048-token window for model framing.
                if model == 'gemini-embedding-001' and item_tokens > EMBEDDING_001_MAX_COUNTED_TOKENS_PER_TEXT:
                    raise SpendBlocked('gemini-embedding-001 allows at most 2040 counted tokens per text; no paid call made')
                total += item_tokens
            output = 0
        else:
            from io import BytesIO
            from PIL import Image
            image_count = 0
            def inspect(value):
                nonlocal image_count
                if hasattr(value, 'model_dump'):
                    value = value.model_dump(exclude_none=True)
                if isinstance(value, str):
                    return value
                elif isinstance(value, list):
                    return [inspect(item) for item in value]
                elif isinstance(value, dict):
                    if 'parts' in value and set(value) <= {'role', 'parts'}:
                        return dict(value, parts=inspect(value['parts']))
                    elif set(value) == {'text'} and isinstance(value['text'], str):
                        return value
                    elif set(value) == {'inline_data'}:
                        blob = value['inline_data']
                        if not isinstance(blob, dict) or set(blob) != {'data', 'mime_type'} or blob['mime_type'] not in ('image/jpeg', 'image/png', 'image/webp') or not isinstance(blob['data'], bytes):
                            raise SpendBlocked('Only bounded inline raster images are priced')
                        try:
                            with Image.open(BytesIO(blob['data'])) as img:
                                if max(img.size) > 384 or min(img.size) <= 0 or getattr(img, 'n_frames', 1) != 1:
                                    raise SpendBlocked('Inline vision input must be a single raster <=384px')
                                img.verify()
                        except SpendBlocked:
                            raise
                        except Exception:
                            raise SpendBlocked('Invalid inline raster') from None
                        image_count += 1
                        return {"text": ""}
                    else:
                        raise SpendBlocked('Unpriced SDK input part')
                else:
                    raise SpendBlocked('Unpriced SDK input type')
            text_contents = inspect(contents)
            if bounded['config'].get('response_modalities', ['TEXT']) != ['TEXT']:
                raise SpendBlocked('Only text output is covered by the verified token tariff')
            if image_count and model not in ('gemini-2.5-flash', 'gemini-3.8-flash'):
                raise SpendBlocked('Inline vision tariff is not verified for this model')
            if image_count > int(limits.get('max_input_images', 1)):
                raise SpendBlocked('Inline image count exceeds verified policy')
            total = count(contents)
            if image_count:
                # Additional text-only count avoids deriving text usage by
                # subtracting an estimated image component from the total.
                total = max(total, count(text_contents) + image_count * (1120 if model == 'gemini-3.8-flash' else 258))
            output = int(bounded['config']['max_output_tokens'])
            if output > int(limits.get('max_output_tokens', 0)):
                raise SpendBlocked('Output policy changed after request validation')
        maximum = limits.get('max_input_tokens')
        if type(maximum) is not int or not 0 <= total <= maximum:
            raise SpendBlocked('Counted input exceeds verified token ceiling')
        return (money(total)*input_rate + money(output)*output_rate) / Decimal('1000000'), reference

    def snapshot(self):
        def read(doc):
            return {
                "status": "reserved_upper_bound",
                "cash_exposure_usd": str(self._cash_exposure(doc)),
                "cash_ceiling_usd": str(money(doc.get("cash_ceiling_usd"))),
                "month": doc.get("month"),
                "reservation_count": len(doc.get("reservations", {})),
            }

        return self.ledger.mutate(read)


def get_spend_guard(*, credential_keys=None):
    fingerprints = None if credential_keys is None else {
        provider: hashlib.sha256(key.strip().encode()).hexdigest()[:16]
        for provider, key in credential_keys.items() if key and key.strip()
    }
    ledger = FirestoreLedger(os.environ.get("GCP_PROJECT_ID", ""), datetime.now(timezone.utc).strftime("%Y-%m"))
    policy_sha = os.environ.get('SPEND_POLICY_SHA256', '').strip()
    if policy_sha:
        from .automatic_spend import TARGET
        from .automatic_spend_runtime import PrivateFirestoreStore
        from .automatic_guard import AutomaticBudgetGuard
        if os.environ.get('GCP_PROJECT_ID') != TARGET['project_id'] or os.environ.get('FIRESTORE_DATABASE_ID', '(default)') != TARGET['database_id'] or not re.fullmatch('[a-f0-9]{64}', policy_sha):
            raise SpendBlocked('Automatic policy requires the fixed approved private target')
        snapshot = PrivateFirestoreStore().read('policy_' + policy_sha)
        if not snapshot.exists:
            raise SpendBlocked('Approved automatic policy is not installed')
        return AutomaticBudgetGuard(ledger, snapshot.to_dict(), policy_sha,
            credential_fingerprints=fingerprints, require_credential_binding=True)
    return BudgetGuard(ledger, credential_fingerprints=fingerprints, require_credential_binding=True)


def guard_genai_client(client, guard=None, *, credential_key=None):
    """Wrap every SDK request in a shared reservation; unknown tariffs block.

    Bounds are operator-verified maxima, not claims of measured usage. The caller
    must configure provider quota/token caps consistent with those maxima.
    """
    guard = guard or get_spend_guard(credential_keys={"gemini": credential_key} if credential_key else None)

    class Models:
        def __getattr__(self, name):
            target = getattr(client.models, name)
            if name not in {"generate_content", "embed_content", "generate_images"}:
                raise SpendBlocked("Unregistered paid SDK method")

            def invoke(*args, **kwargs):
                if args:
                    raise SpendBlocked("Positional SDK payloads are not admitted")
                model = str(kwargs.get("model", ""))
                operation = name + ":" + model
                # Freeze inputs before counting so another caller cannot mutate
                # the paid request after its admission count.
                bounded = guard.bounded_genai_request(operation, copy.deepcopy(kwargs))
                maximum, reference = guard.counted_genai_cap(operation, bounded, client.models.count_tokens)
                automatic_invoke = getattr(guard, 'invoke_prepared_genai', None)
                if getattr(guard, 'automatic_accounting_enabled', False) is True and callable(automatic_invoke):
                    return automatic_invoke(operation, maximum, reference, bounded, target)
                return guard.call(
                    "gemini", operation, maximum, lambda: target(*args, **bounded), expected_reference=reference,
                )

            return invoke

    class Client:
        models = Models()

    return Client()
