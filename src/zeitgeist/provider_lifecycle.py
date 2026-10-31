"""Request-linked provider evidence; ledger policy alone authorizes settlement.

No response body, prompt, API key or signed image URL is retained in receipts.
Legacy guards keep their original call path and full reservations.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from urllib.parse import parse_qsl, urlsplit

APIFY_START_URL = "https://api.apify.com/v2/actors/apify~instagram-post-scraper/runs"
APIFY_ACTOR_ID = "nH2AHrwxeTRJoN5hX"  # Public Actor identifier. gitleaks:allow
APIFY_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT"})
BFL_API_HOST = re.compile(r"api\.(?:(?:isr|(?:eu|us)(?:[1-9][0-9]*)?)\.)?bfl\.ai\Z")
BFL_TERMINAL = frozenset(
    {"Ready", "Error", "Request Moderated", "Content Moderated", "No Card Generated"}
)


def request_digest(payload):
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def begin_lifecycle(guard, provider, operation, maximum, digest):
    begin = getattr(guard, "begin_provider_request", None)
    if begin is None:
        return None
    handle = begin(provider, operation, maximum, digest)
    if handle is not None:
        if (
            not isinstance(handle.reservation_id, str)
            or not handle.reservation_id
            or not isinstance(handle.credential_fingerprint, str)
            or not re.fullmatch(r"[a-f0-9]{16}", handle.credential_fingerprint)
        ):
            raise ValueError("Invalid lifecycle reservation binding")
    return handle


def decimal_cost(value):
    if type(value) not in {str, int, float, Decimal}:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return result if result.is_finite() and result >= 0 else None


def usage_receipt(
    handle,
    provider,
    operation,
    digest,
    request_id,
    amount,
    evidence,
    *,
    observed_at=None,
):
    return {
        "schema_version": 1,
        "accounting_policy": "provider_usage_v1",
        "reservation_id": handle.reservation_id,
        "provider": provider,
        "operation": operation,
        "credential_fingerprint": handle.credential_fingerprint,
        "provider_request_id": request_id,
        "request_sha256": digest,
        "observed_at": observed_at or utc_now(),
        "accounted_gross_usd": str(amount),
        "accounting_basis": "provider_settled_cost",
        "evidence_sha256": request_digest(evidence),
    }


def validated_bfl_poll_url(value, task_id):
    """Validate the authenticated submission's global/regional service URL.

    Includes the api.isr.bfl.ai status router captured from a real authenticated
    FLUX 3 submission, plus the global/eu/us service family. Arbitrary subdomains
    remain excluded, and this never authorizes forwarding credentials.
    """
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", task_id):
        raise ValueError("Invalid BFL task identity")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not BFL_API_HOST.fullmatch(parsed.hostname or "")
        or parsed.username
        or parsed.password
        or parsed.port not in {None, 443}
        or parsed.path != "/v1/get_result"
        or parsed.fragment
    ):
        raise ValueError("Unverified BFL polling destination")
    query = parse_qsl(parsed.query, keep_blank_values=True)
    if query and query != [("id", task_id)]:
        raise ValueError("BFL polling identity differs from submitted task")
    return value


def read_bfl_result(client, polling_url, task_id, *, timeout=15.0):
    """Anonymous exact-task GET; submission credentials never enter this call."""
    validated_bfl_poll_url(polling_url, task_id)
    sensitive = {"authorization", "proxy-authorization", "x-key", "x-goog-api-key", "cookie"}
    if (getattr(client, "auth", None) is not None
        or any(str(key).lower() in sensitive for key in getattr(client, "headers", {}))
        or bool(getattr(client, "params", {}))):
        raise ValueError("BFL result polling requires an anonymous client")
    # A previous response must not cause subsequent requests to carry cookies.
    cookies = getattr(client, "cookies", None)
    if cookies is not None:
        cookies.clear()
    query = urlsplit(polling_url).query
    return client.get(polling_url, params=None if query else {"id": task_id},
        headers={"accept": "application/json"}, auth=None, follow_redirects=False,
        timeout=timeout)


def _ready_full_bound_receipt(handle, digest, task_id, result, *, observed_at=None):
    """A completed priced image books its full bound, never an invoice estimate."""
    guard = getattr(handle, "guard", None)
    if guard is None or not getattr(guard, "automatic_accounting_enabled", False):
        return False
    generated = result.get("result")
    if not isinstance(generated, dict):
        return False
    sample = generated.get("sample")
    if not isinstance(sample, str):
        return False
    image_url = urlsplit(sample)
    if image_url.scheme != "https" or not image_url.hostname or image_url.username or image_url.password:
        return False
    operation = "generate_image:flux-3-image"
    def original(document):
        guard._validate_completion(document, "bfl", handle.reservation_id)
        return dict(document["reservations"][handle.reservation_id])
    record = guard.ledger.mutate(original)
    policy = guard.automatic_policy["providers"]["bfl"]
    metadata = record.get("submission_metadata", {})
    cap = decimal_cost(record.get("max_gross_usd"))
    if (record.get("provider") != "bfl" or record.get("operation") != operation
        or record.get("provider_request_id") != task_id or record.get("request_sha256") != digest
        or record.get("credential_fingerprint") != handle.credential_fingerprint
        or record.get("automatic_policy_sha256") != guard.expected_policy_sha
        or policy.get("price_source_url") != "https://bfl.ai/pricing"
        or not re.fullmatch(r"[a-f0-9]{64}", str(policy.get("price_snapshot_sha256", "")))
        or cap != Decimal("0.07") or decimal_cost(policy["account"]["operation_caps_usd"].get(operation)) != cap
        or metadata.get("resolution") != "1.5k" or metadata.get("aspect_ratio") != "3:2"):
        return False
    evidence = {"id": task_id, "status": "Ready", "request_sha256": digest,
        "policy_sha256": guard.expected_policy_sha, "tariff_sha256": policy["price_snapshot_sha256"],
        "request_shape": {"resolution":"1.5k", "aspect_ratio":"3:2", "image_count":1,
                          "grounding":True, "supplied_reference_images":0},
        "completion_basis":"ready_full_reserved_bound", "reserved_gross_usd":str(cap)}
    receipt = usage_receipt(handle, "bfl", operation, digest, task_id, cap, evidence,
                            observed_at=observed_at)
    receipt.update(accounting_basis="conservative_response_usage", completion_basis="ready_full_reserved_bound")
    handle.receipt(receipt)
    return True


def submit_bfl_receipt(handle, digest, task_id, result, *, observed_at=None):
    """Use settled cost, or book the entire approved bound for a completed image."""
    if not isinstance(result, dict) or result.get("id") != task_id:
        return False
    if result.get("status") not in BFL_TERMINAL:
        return False
    if "cost" not in result:
        return result.get("status") == "Ready" and _ready_full_bound_receipt(
            handle, digest, task_id, result, observed_at=observed_at)
    cost = decimal_cost(result.get("cost"))
    if cost is None:
        return False
    evidence = {key: result[key] for key in ("id", "status", "cost")}
    handle.receipt(
        usage_receipt(
            handle,
            "bfl",
            "generate_image:flux-3-image",
            digest,
            task_id,
            cost * Decimal("0.01"),
            evidence,
            observed_at=observed_at,
        )
    )
    return True


def _run_body(response):
    response.raise_for_status()
    body = response.json()
    if not isinstance(body, dict) or not isinstance(body.get("data"), dict):
        raise ValueError("Invalid Apify run envelope")
    return body["data"]


def _run_identity(run, *, run_id=None, user_id=None, build=None):
    for key in ("id", "userId", "actId"):
        if not isinstance(run.get(key), str) or not re.fullmatch(
            r"[A-Za-z0-9]{1,128}", run[key]
        ):
            raise ValueError("Missing Apify run correlation")
    if run["actId"] != APIFY_ACTOR_ID or run_id is not None and run["id"] != run_id:
        raise ValueError("Apify actor/run identity changed")
    if user_id is not None and run["userId"] != user_id:
        raise ValueError("Apify account identity changed")
    if build is not None and run.get("buildNumber") != build:
        if run.get("status") in APIFY_TERMINAL or run.get("buildNumber") not in {
            None,
            "",
        }:
            raise ValueError("Apify run used an unverified build")


def _stable_apify_cost(first, second):
    """The provider warns initial terminal costs can be preliminary."""
    cost = decimal_cost(second.get("usageTotalUsd"))
    if cost is None or decimal_cost(first.get("usageTotalUsd")) != cost:
        return None
    for run in (first, second):
        try:
            finished = datetime.fromisoformat(
                str(run["finishedAt"]).replace("Z", "+00:00")
            )
            if finished.tzinfo is None or finished > datetime.now(timezone.utc):
                return None
        except (KeyError, TypeError, ValueError):
            return None
        charged, accounted = run.get("chargedEventCounts"), run.get(
            "accountedChargedEventCounts"
        )
        if (
            not isinstance(charged, dict)
            or not isinstance(accounted, dict)
            or charged != accounted
            or any(type(value) is not int or value < 0 for value in charged.values())
        ):
            return None
    return cost


def fetch_apify_lifecycle(
    client, token, payload, build, maximum, timeout_s, handle, digest
):
    """Exactly one start POST, then ID-pinned reads. Never query the latest run."""
    headers = {"Authorization": "Bearer " + token}
    deadline = time.monotonic() + float(timeout_s)
    started = _run_body(
        client.post(
            APIFY_START_URL,
            headers=headers,
            params={
                "build": build,
                "maxTotalChargeUsd": maximum,
                "restartOnError": "false",
                "timeout": 300,
            },
            json=payload,
            timeout=min(float(timeout_s), 30.0),
        )
    )
    run_id = started.get("id")
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9]{1,128}", run_id):
        raise ValueError("Missing Apify submission identity")
    user_id = started.get("userId")
    # This durable private write must complete before any poll. Its unknown
    # outcome cannot authorize another start request.
    metadata = {"build": build}
    for key, source_key in (("actor_id", "actId"), ("user_id", "userId")):
        value = started.get(source_key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9]{1,128}", value):
            metadata[key] = value
    handle.submitted(run_id, submitted_at=utc_now(), metadata=metadata)
    _run_identity(started)
    url = "https://api.apify.com/v2/actor-runs/" + run_id
    run = started
    while run.get("status") not in APIFY_TERMINAL:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Apify lifecycle deadline exceeded")
        run = _run_body(
            client.get(
                url,
                headers=headers,
                params={"waitForFinish": min(60, max(0, remaining - 1))},
                timeout=min(75.0, remaining),
            )
        )
        _run_identity(run, run_id=run_id, user_id=user_id, build=build)
    _run_identity(run, run_id=run_id, user_id=user_id, build=build)
    # Terminal completion is not immediate billing finality. Wait at least the
    # documented ten seconds, then confirm the SAME authenticated run again.
    remaining = deadline - time.monotonic()
    if remaining > 11:
        time.sleep(10.0)
        final = _run_body(
            client.get(
                url, headers=headers, timeout=min(15.0, deadline - time.monotonic())
            )
        )
        _run_identity(final, run_id=run_id, user_id=user_id, build=build)
        if final.get("status") in APIFY_TERMINAL and deadline - time.monotonic() > 1:
            confirmation = _run_body(
                client.get(
                    url, headers=headers, timeout=min(15.0, deadline - time.monotonic())
                )
            )
            _run_identity(confirmation, run_id=run_id, user_id=user_id, build=build)
            if confirmation.get("status") == final.get("status"):
                cost = _stable_apify_cost(final, confirmation)
                if cost is not None:
                    evidence = {
                        key: confirmation.get(key)
                        for key in (
                            "id",
                            "actId",
                            "userId",
                            "buildNumber",
                            "status",
                            "finishedAt",
                            "usageTotalUsd",
                            "chargedEventCounts",
                            "accountedChargedEventCounts",
                        )
                    }
                    handle.receipt(
                        usage_receipt(
                            handle,
                            "apify",
                            "instagram-post-scraper",
                            digest,
                            run_id,
                            cost,
                            evidence,
                        )
                    )
            final = confirmation
        run = final
    if run.get("status") != "SUCCEEDED":
        raise ValueError("Apify actor did not succeed")
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Apify dataset read deadline exceeded")
    response = client.get(
        url + "/dataset/items", headers=headers, timeout=min(30.0, remaining)
    )
    response.raise_for_status()
    return response.json()
