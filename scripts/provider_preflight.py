#!/usr/bin/env python3
"""Inspect provider metadata and validate input, without actors or inference.

Only GET requests and Apify's schema-validation POST are permitted. This command
never provisions a spend ledger and cannot issue production launch approval.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping

import httpx

APIFY_ACTOR = "apify~instagram-post-scraper"
APIFY_ACTOR_ID = "nH2AHrwxeTRJoN5hX"  # Public Actor identifier. gitleaks:allow
MAX_BUILD_INVENTORY_PAGES = 10
APIFY_BASE = "https://api.apify.com/v2"
MODEL_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class MetadataUnavailable(RuntimeError):
    def __init__(self, status=None):
        self.status = status


def decimal_amount(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        amount = Decimal(str(value))
        return amount if amount.is_finite() and amount >= 0 else None
    except (InvalidOperation, ValueError):
        return None


def iso_time(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else None
    except ValueError:
        return None


def safe_json(value, secrets):
    """Final defense after field allowlisting; no raw exception/body is printed."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(k): safe_json(v, secrets) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
        value = re.sub(
            r"https?://[^\s\"<>]+\?[^\s\"<>]+", "[query-url-redacted]", value
        )
    return value


def request_json(client, method, url, **kwargs):
    """A fixed operation allowlist prevents accidental billable calls."""
    if method != "GET" and not (
        method == "POST" and url == f"{APIFY_BASE}/actors/{APIFY_ACTOR}/validate-input"
    ):
        raise MetadataUnavailable()
    try:
        response = client.request(method, url, **kwargs)
        if response.status_code >= 400:
            raise MetadataUnavailable(response.status_code)
        return json.loads(response.text, parse_float=Decimal)
    except MetadataUnavailable:
        raise
    except Exception:
        raise MetadataUnavailable() from None


def cycle_pair(value):
    if not isinstance(value, dict):
        return None
    start, end = iso_time(value.get("startAt")), iso_time(value.get("endAt"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def inspect_apify_billing(user, usage, limits, now):
    plan = user.get("plan") or {}
    base_price = decimal_amount(plan.get("monthlyBasePriceUsd"))
    included = decimal_amount(plan.get("monthlyUsageCreditsUsd"))
    consumed = decimal_amount(usage.get("totalUsageCreditsUsdAfterVolumeDiscount"))
    current = limits.get("current") or {}
    limit_usage = decimal_amount(current.get("monthlyUsageUsd"))
    usage_cycle = cycle_pair(usage.get("usageCycle"))
    limit_cycle = cycle_pair(limits.get("monthlyUsageCycle"))
    cycle_matches = usage_cycle is not None and usage_cycle == limit_cycle
    cycle_active = cycle_matches and usage_cycle[0] <= now <= usage_cycle[1]
    usage_matches = consumed is not None and consumed == limit_usage
    known = bool(
        cycle_active
        and usage_matches
        and included is not None
        and base_price is not None
        and plan.get("isEnabled") is True
        and current.get("activeActorJobCount") == 0
    )
    return {
        "status": "verified_included_credit_snapshot" if known else "unknown",
        "plan": {
            key: plan.get(key)
            for key in (
                "id",
                "tier",
                "isEnabled",
                "monthlyBasePriceUsd",
                "monthlyUsageCreditsUsd",
                "usageDiscountPercent",
            )
        },
        "usage_cycle": {
            k: (usage.get("usageCycle") or {}).get(k) for k in ("startAt", "endAt")
        },
        "cycle_matches_limits": cycle_matches,
        "usage_matches_limits": usage_matches,
        "consumed_service_usage_usd": consumed,
        "remaining_included_credit_usd": (
            max(Decimal("0"), included - consumed) if known else None
        ),
        "credit_expires_at": usage["usageCycle"]["endAt"] if known else None,
        "modeled_usage_overage_usd": (
            max(Decimal("0"), consumed - included) if known else None
        ),
        "subscription_base_price_usd": base_price,
        "account_cash_invoice_verified": False,
        "active_actor_job_count": current.get("activeActorJobCount"),
        "account_usage_limit_usd": (limits.get("limits") or {}).get(
            "maxMonthlyUsageUsd"
        ),
        "note": "Included-credit snapshot only; excludes unverified promotions, tax, invoices and in-flight charges.",
    }


def _exact_apify_build(get, actor, configured_build, configured_build_id=None):
    """Resolve a numeric pin using documented IDs, never a mutable default tag.

    APIs: /actors/{actor}/builds (limit/offset/desc) lists buildNumber+ID;
    /actor-builds/{ID} returns its immutable actor/schema/image snapshot.
    https://docs.apify.com/api/v2/actors-builds-get
    https://docs.apify.com/api/v2/actor-build-get
    """

    def identifier(value):
        return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value)

    if actor.get("id") != APIFY_ACTOR_ID:
        raise MetadataUnavailable()
    try:
        default = get("/actors/" + APIFY_ACTOR + "/builds/default")
    except MetadataUnavailable:
        default = {}  # Optional diagnostics cannot invalidate a known exact ID.
    default_number = default.get("buildNumber")
    if not isinstance(default_number, str) or not re.fullmatch(
        r"\d+\.\d+\.\d+", default_number
    ):
        default_number = None
    candidates = set()
    if configured_build_id is not None:
        if not identifier(configured_build_id):
            raise MetadataUnavailable()
        candidates.add(configured_build_id)
    else:
        hints = [default] + list((actor.get("taggedBuilds") or {}).values())
        for hint in hints:
            if isinstance(hint, dict) and hint.get("buildNumber") == configured_build:
                value = hint.get("buildId", hint.get("id"))
                if identifier(value):
                    candidates.add(value)
    if len(candidates) > 1:
        raise MetadataUnavailable()
    if not candidates:
        offset = 0
        for _ in range(MAX_BUILD_INVENTORY_PAGES):
            page = get(
                "/actors/" + APIFY_ACTOR + "/builds",
                params={"limit": 1000, "offset": offset, "desc": "true"},
            )
            items = page.get("items")
            if (
                not isinstance(items, list)
                or type(page.get("offset")) is not int
                or page["offset"] != offset
                or type(page.get("count")) is not int
                or page["count"] != len(items)
                or type(page.get("total")) is not int
                or page["total"] < offset + len(items)
            ):
                raise MetadataUnavailable()
            for item in items:
                if (
                    isinstance(item, dict)
                    and item.get("buildNumber") == configured_build
                ):
                    if not identifier(item.get("id")) or item.get(
                        "actId", APIFY_ACTOR_ID
                    ) not in {APIFY_ACTOR_ID, APIFY_ACTOR}:
                        raise MetadataUnavailable()
                    candidates.add(item["id"])
            if candidates:
                break
            if not items or offset + len(items) >= page["total"]:
                break
            offset += len(items)
    if len(candidates) != 1:
        raise MetadataUnavailable()
    immutable_id = candidates.pop()
    build = get("/actor-builds/" + immutable_id)
    if (
        build.get("id") != immutable_id
        or build.get("actId") not in {APIFY_ACTOR_ID, APIFY_ACTOR}
        or build.get("buildNumber") != configured_build
    ):
        raise MetadataUnavailable()
    image = build.get("imageDigest")
    if not isinstance(image, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", image):
        raise MetadataUnavailable()
    return build, default_number


def inspect_apify(client, token, configured_build, now, *, configured_build_id=None):
    if not isinstance(configured_build, str) or not re.fullmatch(
        r"\d+\.\d+\.\d+", configured_build
    ):
        raise MetadataUnavailable()
    headers = {"Authorization": "Bearer " + token}

    def get(path, *, params=None):
        body = request_json(
            client, "GET", APIFY_BASE + path, headers=headers, params=params
        )
        data = body.get("data")
        if not isinstance(data, dict):
            raise MetadataUnavailable()
        return data

    user = get("/users/me")
    usage = get("/users/me/usage/monthly")
    limits = get("/users/me/limits")
    actor = get("/actors/" + APIFY_ACTOR)
    build, observed_default = _exact_apify_build(
        get, actor, configured_build, configured_build_id
    )
    observed = build["buildNumber"]
    definition = build.get("actorDefinition")
    schema = definition.get("input") if isinstance(definition, dict) else None
    if not isinstance(schema, dict):
        schema = build.get("inputSchema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except ValueError:
            schema = {}
    schema = schema if isinstance(schema, dict) else {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    contract_ok = (
        "username" in required and (props.get("username") or {}).get("type") == "array"
    )
    schema_sha = hashlib.sha256(
        json.dumps(schema, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    # Validation supports build numbers/tags; use the exact number whose
    # immutable ID, actor, image and schema were independently resolved above.
    validation = {
        "build": observed,
        "legacy_plural_status": None,
        "correct_singular_status": None,
    }
    common = {
        "resultsLimit": 1,
        "dataDetailLevel": "basicData",
        "includeMusicData": False,
        "onlyPostsNewerThan": "14 days",
    }
    for field, label in [
        ("usernames", "legacy_plural_status"),
        ("username", "correct_singular_status"),
    ]:
        try:
            response = client.post(
                APIFY_BASE + "/actors/" + APIFY_ACTOR + "/validate-input",
                headers=headers,
                params={"build": observed},
                json=dict(common, **{field: ["natgeo"]}),
            )
            validation[label] = response.status_code
            if field == "username" and response.status_code == 200:
                body = json.loads(response.text)
                validation["correct_input_valid"] = body.get("valid") is True
        except Exception:
            validation[label] = None
    validation_ok = (
        validation["legacy_plural_status"] == 400
        and validation.get("correct_input_valid") is True
    )
    active_prices = [
        p
        for p in actor.get("pricingInfos", [])
        if isinstance(p, dict)
        and iso_time(p.get("startedAt"))
        and iso_time(p["startedAt"]) <= now
    ]
    pricing = (
        max(active_prices, key=lambda p: iso_time(p["startedAt"]))
        if active_prices
        else {}
    )
    events = (pricing.get("pricingPerEvent") or {}).get("actorChargeEvents") or {}
    event_prices = {
        str(event): {
            str(tier): decimal_amount((entry or {}).get("tieredEventPriceUsd"))
            for tier, entry in (detail.get("eventTieredPricingUsd") or {}).items()
        }
        for event, detail in events.items()
        if isinstance(detail, dict)
    }
    billing = inspect_apify_billing(user, usage, limits, now)
    pin_ok = configured_build == observed and build.get("status") == "SUCCEEDED"
    ready = bool(
        pin_ok
        and contract_ok
        and validation_ok
        and billing["status"] != "unknown"
        and event_prices
    )
    account_id = str(user.get("id") or "")
    return {
        "status": "metadata_verified" if ready else "incomplete",
        "credential_fingerprint": hashlib.sha256(token.encode()).hexdigest()[:16],
        "account_fingerprint": (
            hashlib.sha256(account_id.encode()).hexdigest()[:16] if account_id else None
        ),
        "billing": billing,
        "observed_default_build": observed_default,
        "configured_build": configured_build,
        "actor_id": APIFY_ACTOR_ID,
        "input_schema_sha256": schema_sha,
        "build_pin_verified": pin_ok,
        "build_id": build.get("id"),
        "image_digest": build.get("imageDigest"),
        "required_input_fields": required,
        "username_contract_verified": contract_ok,
        "input_validation": validation,
        "pricing": {
            "started_at": pricing.get("startedAt"),
            "model": pricing.get("pricingModel"),
            "minimum_run_charge_cap_usd": decimal_amount(
                pricing.get("minimalMaxTotalChargeUsd")
            ),
            "event_prices_usd": event_prices,
        },
        "actor_started": False,
    }


def credential_identity(key):
    """Expose a stable correlation hash and a masked suffix, never a whole key."""
    return {
        "credential_fingerprint": hashlib.sha256(key.encode()).hexdigest()[:16],
        "masked_last4": "..." + key[-4:] if len(key) > 8 else "[redacted]",
    }


def inspect_gemini(client, key, required_models):
    found = {}
    token = None
    seen_tokens = set()
    for _ in range(20):
        body = request_json(
            client,
            "GET",
            MODEL_BASE,
            headers={"x-goog-api-key": key},
            params={"pageToken": token} if token else {},
        )
        for model in body.get("models", []):
            name = str(model.get("name", "")).removeprefix("models/")
            if name in required_models:
                found[name] = model.get("supportedGenerationMethods") or []
        token = body.get("nextPageToken")
        if not token:
            break
        if token in seen_tokens:
            raise MetadataUnavailable()
        seen_tokens.add(token)
    else:
        raise MetadataUnavailable()
    missing = sorted(set(required_models) - set(found))
    return {
        "status": "metadata_verified" if not missing else "incomplete",
        "required_models": list(required_models),
        "available_required_models": found,
        "missing_models": missing,
        **credential_identity(key),
        "billing_status": "unknown",
        "free_tier_verified": False,
        "inference_performed": False,
    }


def inspect_unsplash(client, key):
    try:
        response = client.get(
            "https://api.unsplash.com/photos",
            headers={"Authorization": "Client-ID " + key},
            params={"per_page": 1},
        )
        if response.status_code != 200 or not isinstance(
            json.loads(response.text), list
        ):
            raise MetadataUnavailable(response.status_code)
        limit = response.headers.get("X-Ratelimit-Limit")
        remaining = response.headers.get("X-Ratelimit-Remaining")
        return {
            "status": "metadata_verified",
            "rate_limit": int(limit) if limit and limit.isdigit() else None,
            "rate_limit_remaining": (
                int(remaining) if remaining and remaining.isdigit() else None
            ),
            "image_downloaded": False,
            "download_tracking_triggered": False,
        }
    except MetadataUnavailable:
        raise
    except Exception:
        raise MetadataUnavailable() from None


def inspect_bfl(client, key):
    """Official read-only credits endpoint; balance is not a verified free grant."""
    body = request_json(
        client, "GET", "https://api.bfl.ai/v1/credits", headers={"x-key": key}
    )
    raw = body.get("credits")
    if isinstance(raw, bool) or not isinstance(raw, (int, float, Decimal)):
        raise MetadataUnavailable()
    credits = decimal_amount(raw)
    if credits is None:
        raise MetadataUnavailable()
    return {
        "status": "metadata_verified",
        "balance_credits": credits,
        "credential_fingerprint": hashlib.sha256(key.encode()).hexdigest()[:16],
        "billing_status": "unknown",
        "cash_allowance_verified": False,
        "auto_topup_status": "unknown",
        "credit_expiry_status": "unknown",
        "tariff_status": "unknown",
        "generation_performed": False,
        "request_made": True,
    }


def run_preflight(client, env: Mapping[str, str], *, now=None, required_models=None):
    now = now or datetime.now(timezone.utc)
    required_models = list(
        required_models or ["gemini-3.8-flash", "gemini-embedding-001"]
    )
    report = {
        "schema_version": 1,
        "observed_at": now.isoformat(),
        "mode": "metadata_only",
        "release_decision": "NO_GO",
        "metadata_ok": False,
        "remaining_launch_gates": [
            "Live ingestion and rendered output not exercised",
            "Gemini billing/free-tier eligibility unknown",
            "BFL pricing and account allowance unknown",
            "Shared spend ledger not provisioned or reconciled",
        ],
    }
    keys = {
        "apify": env.get("APIFY_TOKEN") or env.get("APIFY_API_TOKEN"),
        "gemini": env.get("GEMINI_API_KEY"),
        "unsplash": env.get("UNSPLASH_ACCESS_KEY"),
        "bfl": env.get("BFL_API_KEY"),
    }
    for name, key in keys.items():
        if not key:
            report[name] = {"status": "missing_credentials"}
            continue
        try:
            if name == "apify":
                result = inspect_apify(
                    client,
                    key,
                    env.get("APIFY_INSTAGRAM_BUILD"),
                    now,
                    configured_build_id=env.get("APIFY_INSTAGRAM_BUILD_ID"),
                )
            elif name == "gemini":
                result = inspect_gemini(client, key, required_models)
            elif name == "unsplash":
                result = inspect_unsplash(client, key)
            else:
                result = inspect_bfl(client, key)
            report[name] = result
        except MetadataUnavailable as exc:
            report[name] = {
                "status": "unavailable",
                "http_status": exc.status,
                "error": "Provider metadata request failed; raw error omitted",
            }
        except Exception:
            report[name] = {
                "status": "unavailable",
                "http_status": None,
                "error": "Provider metadata invalid or unavailable; raw error omitted",
            }
    if keys.get("gemini"):
        report["gemini"].update(credential_identity(keys["gemini"]))
    report["metadata_ok"] = all(
        report[name]["status"] == "metadata_verified" for name in keys
    )
    return safe_json(report, list(keys.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--required-model", action="append", dest="required_models")
    args = parser.parse_args(argv)
    # Disable redirects so credentials cannot follow a provider redirect.
    with httpx.Client(timeout=20, follow_redirects=False) as client:
        report = run_preflight(client, os.environ, required_models=args.required_models)
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["metadata_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
