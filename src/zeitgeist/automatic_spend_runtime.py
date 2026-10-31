"""Private, conditional orchestration for an already approved spend policy.

This module cannot install policies or convert legacy ledgers. Provider observation
is metadata-only. Unknown handoffs are journaled privately and reconciled by reads.
Only a durably recorded definitive rejection permits a later, newly planned attempt.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import re
from typing import Callable, Mapping

import httpx

from src.state_store import (
    PublicationConflict, PublicationOutcomeUnknown, _FirestoreRestClient,
    _commit_documents,
)
from . import automatic_spend as accounting
from .spend_guard import SpendBlocked


@dataclass
class MaintenanceResult:
    status: str
    operation: str
    month: str
    plan: dict = field(default_factory=dict, repr=False)

    def public_status(self):
        """Never return provider balances, account identities, holds, or errors."""
        return {"status": self.status, "operation": self.operation, "month": self.month}


class PrivateFirestoreStore:
    """Fixed reviewed private target; no staging namespace or environment override."""
    def __init__(self, client=None):
        self.client = client or _FirestoreRestClient(
            accounting.TARGET["project_id"], accounting.TARGET["database_id"]
        )

    def read(self, document_id):
        if not re.fullmatch(r"(?:\d{4}-\d{2}|policy_[a-f0-9]{64}|handoff_\d{4}-\d{2})", document_id):
            raise SpendBlocked("Unapproved private spend document")
        return self.client.collection("spend_control").document(document_id).get()

    def commit(self, writes):
        _commit_documents(self.client, [("spend_control", key, value, snapshot)
                                       for key, value, snapshot in writes])


def observe_provider_metadata(policy, env: Mapping[str, str], now):
    """Authenticate metadata reads; never call inference, Actors, or image generation.

    Published-validity contracts remain the original approved policy facts. Dynamic
    Apify prices come from actor metadata. BFL's exact public structured tariff is
    independently normalized. Unsupported modes fail closed without restamping
    the original policy or inferring a future price guarantee.
    """
    from scripts.provider_preflight import APIFY_ACTOR, APIFY_BASE, run_preflight

    models = sorted({operation.split(":", 1)[1]
                     for operation in policy["providers"]["gemini"]["account"]["operation_caps_usd"]})
    private_env = {name: env[name] for name in (
        "GEMINI_API_KEY", "BFL_API_KEY", "APIFY_TOKEN", "APIFY_API_TOKEN", "APIFY_INSTAGRAM_BUILD"
    ) if name in env}
    # The immutable ID comes from the reviewed policy, never an ambient override.
    private_env['APIFY_INSTAGRAM_BUILD_ID'] = policy['providers']['apify']['actor_build_id']
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        report = run_preflight(client, private_env, now=now, required_models=models)
    report["price_snapshots"] = {}
    for name, pinned in policy["providers"].items():
        mode = pinned.get("price_evidence_mode", "dynamic_snapshot")
        if mode == "published_validity":
            continue  # The pure policy validator checks the original contract.
        if name == "bfl":
            from .bfl_tariff import observe_bfl_flux3_tariff
            with httpx.Client(timeout=20, follow_redirects=False) as public_client:
                report["price_snapshots"][name] = observe_bfl_flux3_tariff(public_client, now=now)
            continue
        if name == "gemini":
            from .gemini_tariff import observe_gemini_tariff
            with httpx.Client(timeout=20, follow_redirects=False) as public_client:
                report["price_snapshots"][name] = observe_gemini_tariff(public_client, now=now)
            continue
        if name != "apify":
            raise SpendBlocked("Dynamic provider price observation is unavailable")
        url = APIFY_BASE + "/actors/" + APIFY_ACTOR
        if pinned["price_source_url"] != url or report.get(name, {}).get("status") != "metadata_verified":
            raise SpendBlocked("Authenticated actor tariff observation unavailable")
        report["price_snapshots"][name] = {
            "source_url": url, "sha256": accounting.digest(report[name]["pricing"]),
            "observed_at": report["observed_at"],
        }
    return report


def _data(snapshot):
    return snapshot.to_dict() if snapshot.exists else None


def _known_write(store, document_id, after, snapshot):
    """One dispatch, then readback on unknown acknowledgement; never retry write."""
    try:
        store.commit([(document_id, after, snapshot)])
    except PublicationOutcomeUnknown:
        if _data(store.read(document_id)) != after:
            raise SpendBlocked("Private maintenance outcome unknown; reconciliation required") from None


def _complete_intent(store, intent_id, snapshot, intent):
    if _data(snapshot) != intent:
        raise SpendBlocked("Handoff intent changed before completion")
    completed = dict(intent, status="complete")
    _known_write(store, intent_id, completed, snapshot)


def _rejected_history(intent):
    history = copy.deepcopy(intent.get("rejected_plan_history", []))
    has_history = "rejected_plan_history" in intent or "rejected_plan_history_sha256" in intent
    if not isinstance(history, list) or (has_history and
            intent.get("rejected_plan_history_sha256") != accounting.digest(history)):
        raise SpendBlocked("Handoff rejection history requires reconciliation")
    return history


def _record_definitive_rejection(store, intent_id, intent, now):
    """Persist a proved noncommit before allowing any later fresh plan."""
    snapshot = store.read(intent_id)
    if _data(snapshot) != intent:
        raise SpendBlocked("Handoff intent changed before rejection was recorded")
    plan = intent["plan"]
    history = _rejected_history(intent)
    history.append({"handoff_id": plan["handoff_id"], "plan_sha256": plan["plan_sha256"],
        "source_identity": copy.deepcopy(plan["source_identity"]),
        "source_before_sha256": plan["source_before_sha256"],
        "rejected_at": now.isoformat(), "reason": "canonical_precondition_conflict"})
    rejected = dict(intent, status="known_rejected", rejected_plan_history=history,
                    rejected_plan_history_sha256=accounting.digest(history))
    _known_write(store, intent_id, rejected, snapshot)


def _reconcile_pending(store, intent_id, intent_snapshot, *, apply):
    intent = _data(intent_snapshot)
    plan = intent.get("plan") if isinstance(intent, dict) else None
    if not isinstance(plan, dict) or intent.get("status") != "pending":
        raise SpendBlocked("Invalid private handoff intent")
    source = store.read(plan["source_identity"]["document_id"])
    destination = store.read(plan["destination_after"]["month"])
    accounting.reconcile_month_handoff(plan, _data(source), _data(destination))
    if apply:
        _complete_intent(store, intent_id, intent_snapshot, intent)
    return MaintenanceResult("reconciled" if apply else "reconciliation_ready", "handoff", plan["destination_after"]["month"], plan)


def maintain_spend_control(store, *, expected_policy_sha, observer: Callable,
                           now=None, apply=False):
    """Plan or execute one refresh/handoff of preinstalled schema-3 accounting.

    `observer(policy, now)` is an authenticated metadata-only dependency. It is
    invoked only after exact policy and ledger approval checks. Conflicts reject
    this invocation; a caller must not loop/redispatch an ambiguous transaction.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or not re.fullmatch(r"[a-f0-9]{64}", expected_policy_sha or ""):
        raise SpendBlocked("A pinned approved policy digest and UTC-compatible clock are required")
    now = now.astimezone(timezone.utc)
    month = now.strftime("%Y-%m")
    policy_snapshot = store.read("policy_" + expected_policy_sha)
    policy = _data(policy_snapshot)
    if not isinstance(policy, dict):
        raise SpendBlocked("Approved automatic policy is not installed")
    accounting.validate_policy(policy, expected_policy_sha, now)
    current = store.read(month)
    intent_id = "handoff_" + month
    intent_snapshot = store.read(intent_id)
    intent = _data(intent_snapshot)
    rejection_history = []
    if intent is not None:
        if intent.get("policy_sha256") != expected_policy_sha:
            raise SpendBlocked("Handoff belongs to another approved policy")
        if intent.get("status") == "pending":
            return _reconcile_pending(store, intent_id, intent_snapshot, apply=apply)
        if intent.get("status") == "known_rejected":
            plan = intent.get("plan")
            if not isinstance(plan, dict) or current.exists:
                raise SpendBlocked("Rejected handoff state requires reconciliation")
            # Check the recorded plan digest, without treating it as a committed
            # canonical state. The replacement below uses fresh reads/metadata.
            accounting.reconcile_month_handoff(plan, plan["source_after"], plan["destination_after"])
            rejection_history = _rejected_history(intent)
            if not rejection_history or rejection_history[-1].get("plan_sha256") != plan["plan_sha256"]:
                raise SpendBlocked("Handoff rejection evidence is incomplete")
        elif intent.get("status") != "complete" or not current.exists:
            raise SpendBlocked("Handoff state requires reconciliation")
        elif (_data(current).get("automatic_handoff_id") != intent.get("plan", {}).get("handoff_id")):
            raise SpendBlocked("Current ledger does not match its completed handoff")
    if current.exists:
        ledger = _data(current)
        if ledger.get("month") != month:
            raise SpendBlocked("Current ledger month mismatch")
        _require_active_approval(ledger, expected_policy_sha)
        observation = observer(copy.deepcopy(policy), now)
        plan = accounting.prepare_refresh(ledger, policy, observation,
                expected_policy_sha=expected_policy_sha, now=now)
        if plan["before_sha256"] != accounting.digest(ledger):
            raise SpendBlocked("Refresh plan lost its source revision")
        if apply:
            _known_write(store, month, plan["after"], current)
        return MaintenanceResult("refreshed" if apply else "planned", "refresh", month, plan)
    previous = (now.replace(day=1)-timedelta(days=1)).strftime("%Y-%m")
    source_snapshot = store.read(previous)
    source = _data(source_snapshot)
    if not isinstance(source, dict):
        raise SpendBlocked("Approved prior ledger is missing; initial provisioning is separate")
    _require_active_approval(source, expected_policy_sha)
    revision = getattr(source_snapshot, "update_time", None)
    if not revision:
        raise SpendBlocked("Exact source revision required")
    identity = dict(accounting.TARGET, document_id=previous,
                    update_time=revision.isoformat() if hasattr(revision, "isoformat") else str(revision))
    observation = observer(copy.deepcopy(policy), now)
    plan = accounting.prepare_month_handoff(source, policy, observation,
            expected_policy_sha=expected_policy_sha, source_identity=identity, now=now)
    if not apply:
        return MaintenanceResult("planned", "handoff", month, plan)
    intent = {"schema_version": 1, "status": "pending", "policy_sha256": expected_policy_sha,
              "plan": plan}
    if rejection_history:
        intent.update(rejected_plan_history=rejection_history,
                      rejected_plan_history_sha256=accounting.digest(rejection_history))
    _known_write(store, intent_id, intent, intent_snapshot)
    # A pending intent is never redispatched. A definitive rejection is recorded
    # separately and permits only a later fresh-source/fresh-metadata attempt.
    try:
        store.commit([(previous, plan["source_after"], source_snapshot),
                      (month, plan["destination_after"], current)])
    except PublicationOutcomeUnknown:
        pass  # Determine ownership only by exact canonical readback below.
    except PublicationConflict:
        _record_definitive_rejection(store, intent_id, intent, now)
        raise  # No blind retry of this batch, even after a proved rejection.
    accounting.reconcile_month_handoff(plan, _data(store.read(previous)), _data(store.read(month)))
    _complete_intent(store, intent_id, store.read(intent_id), intent)
    return MaintenanceResult("handed_off", "handoff", month, plan)


def _require_active_approval(ledger, expected_policy_sha):
    if (ledger.get("schema_version") != 3 or ledger.get("automatic_accounting_version") != 1
            or ledger.get("automatic_policy_sha256") != expected_policy_sha
            or ledger.get("closed_for_rollover") or ledger.get("reconciliation_required")):
        raise SpendBlocked("Legacy, frozen, or unapproved ledger requires separate review")
