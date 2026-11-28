"""Pure builders for the successor spend policy and its ledger projection.

No I/O and no printing: callers install, print and roll back. Nothing here deletes a
field, because the Firestore commit only writes the keys of the document it is given.
"""
import copy
import json
import re
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from .automatic_spend import _time
from .spend_guard import (EMBEDDING_2_DIMENSIONS, EMBEDDING_2_OPERATION, GEMINI_4, HYDROGEN_MODEL,
                          HYDROGEN_OPERATION, HYDROGEN_OUTPUT_TOKENS, NEON_MODEL, NEON_OPERATION,
                          NEON_OUTPUT_TOKENS)

INSTALLED_POLICY_SHA256 = "e664ccce9873b17fedaa7eda48e7c928be7540fd3db6c7d99168a125045a661f"
# The Hydrogen successor installed on 22 Nov: the base the Neon successor adds to.
NEON_BASE_POLICY_SHA256 = "6657e657bd261d5a76ffd939d16f0dc1b817e847bfa50a3ce1f39881d102cc96"
FLASH_OPERATION = "generate_content:gemini-3.8-flash"
EMBEDDING_001_OPERATION = "embed_content:gemini-embedding-001"

REDACTED_KEYS = frozenset({"billing_account_id", "credential_fingerprint", "account_id",
                           "account_fingerprint", "approval_reference", "reference",
                           "verification_reference"})


def _gemini4_cap_usd(model: str, max_output_tokens: int) -> str:
    """The exact per-call cap: a full 65536-token input window plus the whole output allowance."""
    terms = GEMINI_4[model]
    total = (Decimal(65536) * Decimal(terms.input_usd_per_million)
             + Decimal(max_output_tokens) * Decimal(terms.output_usd_per_million))
    return format(total / Decimal(1000000), "f")


def hydrogen_cap_usd(max_output_tokens: int) -> str:
    return _gemini4_cap_usd(HYDROGEN_MODEL, max_output_tokens)


def neon_cap_usd(max_output_tokens: int) -> str:
    return _gemini4_cap_usd(NEON_MODEL, max_output_tokens)


HYDROGEN_BUILD_THINKING_LEVELS = ("high", "medium")
NEON_BUILD_THINKING_LEVELS = ("high", "medium")
EMBEDDING_2_INPUT_USD_PER_MILLION = "0.20"


def embedding2_operation(embedding_001_limits: dict) -> tuple[str, str, dict]:
    """Embedding-001's installed limits at the embedding-2 text rate; cap is the full input window."""
    limits = copy.deepcopy(embedding_001_limits)
    limits.update(input_token_usd_per_million=EMBEDDING_2_INPUT_USD_PER_MILLION,
                  output_dimensionality=EMBEDDING_2_DIMENSIONS)
    cap = Decimal(limits["max_input_tokens"]) * Decimal(EMBEDDING_2_INPUT_USD_PER_MILLION) / 10**6
    return EMBEDDING_2_OPERATION, format(cap, "f"), limits


def _check_gemini4_inputs(name, ladder, levels, model_version, tokens_per_image, max_output_tokens, thinking_level):
    if not isinstance(model_version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", model_version):
        raise RuntimeError(f"{name} model version is malformed")
    if type(tokens_per_image) is not int or not 1 <= tokens_per_image <= 4096:
        raise RuntimeError(f"{name} tokens per image must be an integer from 1 to 4096")
    if type(max_output_tokens) is not int or max_output_tokens not in ladder:
        raise RuntimeError(f"{name} max output tokens must be one of 4000, 6000, 8000, 10000, 15000")
    if type(thinking_level) is not str or thinking_level not in levels:
        raise RuntimeError(f"{name} thinking level must be high or medium")


def _check_inputs(model_version, tokens_per_image, max_output_tokens, thinking_level):
    _check_gemini4_inputs("Hydrogen", HYDROGEN_OUTPUT_TOKENS, HYDROGEN_BUILD_THINKING_LEVELS,
                          model_version, tokens_per_image, max_output_tokens, thinking_level)


def _check_neon_inputs(model_version, tokens_per_image, max_output_tokens, thinking_level):
    _check_gemini4_inputs("Neon", NEON_OUTPUT_TOKENS, NEON_BUILD_THINKING_LEVELS,
                          model_version, tokens_per_image, max_output_tokens, thinking_level)


def _gemini4_operation(model, flash_limits, model_version, tokens_per_image, max_output_tokens, thinking_level):
    """Flash's installed limits at the model's exact rates, output cap, thinking level and pinned version."""
    terms = GEMINI_4[model]
    limits = copy.deepcopy(flash_limits)
    limits.update(input_token_usd_per_million=terms.input_usd_per_million,
                  output_token_usd_per_million=terms.output_usd_per_million,
                  max_output_tokens=max_output_tokens, thinking_level=thinking_level,
                  image_tokens_per_image=tokens_per_image,
                  allowed_response_model_versions=[model_version])
    return "generate_content:" + model, _gemini4_cap_usd(model, max_output_tokens), limits


def hydrogen_operation(flash_limits: dict, *, model_version: str, tokens_per_image: int,
                       max_output_tokens: int, thinking_level: str) -> tuple[str, str, dict]:
    _check_inputs(model_version, tokens_per_image, max_output_tokens, thinking_level)
    return _gemini4_operation(HYDROGEN_MODEL, flash_limits, model_version, tokens_per_image,
                              max_output_tokens, thinking_level)


def neon_operation(flash_limits: dict, *, model_version: str, tokens_per_image: int,
                   max_output_tokens: int, thinking_level: str) -> tuple[str, str, dict]:
    _check_neon_inputs(model_version, tokens_per_image, max_output_tokens, thinking_level)
    return _gemini4_operation(NEON_MODEL, flash_limits, model_version, tokens_per_image,
                              max_output_tokens, thinking_level)


def build_successor(installed: dict, *, hydrogen_model_version: str, hydrogen_tokens_per_image: int,
                    hydrogen_max_output_tokens: int, hydrogen_thinking_level: str,
                    tariff_sha256: str, include_embedding_2: bool = False) -> dict:
    _check_inputs(hydrogen_model_version, hydrogen_tokens_per_image, hydrogen_max_output_tokens,
                  hydrogen_thinking_level)
    if not isinstance(tariff_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", tariff_sha256):
        raise RuntimeError("Tariff digest must be 64 lowercase hex characters")
    gemini = installed["providers"]["gemini"]
    account = gemini["account"]
    if (HYDROGEN_OPERATION in account.get("request_limits", {})
            or HYDROGEN_OPERATION in account.get("operation_caps_usd", {})):
        raise RuntimeError("Installed policy already has the Hydrogen operation")
    flash = account.get("request_limits", {}).get(FLASH_OPERATION)
    if not isinstance(flash, dict):
        raise RuntimeError("Installed policy has no Flash limits to derive Hydrogen from")
    added = [hydrogen_operation(flash, model_version=hydrogen_model_version,
                                tokens_per_image=hydrogen_tokens_per_image,
                                max_output_tokens=hydrogen_max_output_tokens,
                                thinking_level=hydrogen_thinking_level)]
    if include_embedding_2:
        if (EMBEDDING_2_OPERATION in account.get("request_limits", {})
                or EMBEDDING_2_OPERATION in account.get("operation_caps_usd", {})):
            raise RuntimeError("Installed policy already has the embedding-2 operation")
        embedding_001 = account.get("request_limits", {}).get(EMBEDDING_001_OPERATION)
        if not isinstance(embedding_001, dict):
            raise RuntimeError("Installed policy has no embedding-001 limits to derive embedding-2 from")
        added.append(embedding2_operation(embedding_001))
    successor = copy.deepcopy(installed)
    new = successor["providers"]["gemini"]
    for operation, cap, limits in added:
        new["account"]["operation_caps_usd"][operation] = cap
        new["account"]["request_limits"][operation] = limits
    new["price_snapshot_sha256"] = tariff_sha256
    return successor


def build_neon_successor(installed: dict, *, neon_model_version: str, neon_tokens_per_image: int,
                         neon_max_output_tokens: int, neon_thinking_level: str, tariff_sha256: str) -> dict:
    """The installed policy plus generate_content:gemini-4-neon and the observed tariff digest; nothing else."""
    _check_neon_inputs(neon_model_version, neon_tokens_per_image, neon_max_output_tokens, neon_thinking_level)
    if not isinstance(tariff_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", tariff_sha256):
        raise RuntimeError("Tariff digest must be 64 lowercase hex characters")
    account = installed["providers"]["gemini"]["account"]
    if (NEON_OPERATION in account.get("request_limits", {})
            or NEON_OPERATION in account.get("operation_caps_usd", {})):
        raise RuntimeError("Installed policy already has the Neon operation")
    flash = account.get("request_limits", {}).get(FLASH_OPERATION)
    if not isinstance(flash, dict):
        raise RuntimeError("Installed policy has no Flash limits to derive Neon from")
    operation, cap, limits = neon_operation(flash, model_version=neon_model_version,
                                            tokens_per_image=neon_tokens_per_image,
                                            max_output_tokens=neon_max_output_tokens,
                                            thinking_level=neon_thinking_level)
    successor = copy.deepcopy(installed)
    new = successor["providers"]["gemini"]
    new["account"]["operation_caps_usd"][operation] = cap
    new["account"]["request_limits"][operation] = limits
    new["price_snapshot_sha256"] = tariff_sha256
    return successor


def _project_ledger(ledger: dict, policy: dict, policy_sha: str) -> dict:
    result = copy.deepcopy(ledger)
    account = policy["providers"]["gemini"]["account"]
    result["automatic_policy_sha256"] = policy_sha
    target = result["providers"]["gemini"]
    target["operation_caps_usd"] = copy.deepcopy(account["operation_caps_usd"])
    target["request_limits"] = copy.deepcopy(account["request_limits"])
    return result


def successor_ledger(ledger: dict, policy: dict, policy_sha: str) -> dict:
    return _project_ledger(ledger, policy, policy_sha)


def rollback_ledger(ledger: dict, old_policy: dict, old_sha: str) -> dict:
    return _project_ledger(ledger, old_policy, old_sha)


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: "[redacted]" if k in REDACTED_KEYS else _redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _show(path: str, value: Any) -> str:
    if REDACTED_KEYS.intersection(path.split(".")):
        return "[redacted]"
    return json.dumps(_redact(value), sort_keys=True, separators=(",", ":"))


def _walk(before: Any, after: Any, path: str, out: list) -> None:
    if before == after:
        return
    last = path.rsplit(".", 1)[-1] if path else ""
    if last == "reservations":
        count = len(after) if isinstance(after, (dict, list)) else 0
        out.append(f"~ {path} ({count} records)")
        return
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in after:
                out.append(f"- {sub}")
            elif key not in before:
                if key == "reservations":
                    out.append(f"~ {sub} ({len(after[key]) if isinstance(after[key], (dict, list)) else 0} records)")
                else:
                    out.append(f"+ {sub} = {_show(sub, after[key])}")
            else:
                _walk(before[key], after[key], sub, out)
        return
    out.append(f"~ {path}: {_show(path, before)} -> {_show(path, after)}")


def field_diff(before: Any, after: Any) -> list[str]:
    out: list[str] = []
    _walk(before, after, "", out)
    return sorted(out)


def assert_no_hold_in_flight(ledger: dict, *, now: datetime, duration_seconds: int) -> None:
    cutoff = now - timedelta(seconds=duration_seconds)
    for record in (ledger.get("reservations") or {}).values():
        if record.get("status") == "settled":
            continue
        began = _time(record["created_at"])
        if record.get("submitted_at"):
            began = max(began, _time(record["submitted_at"]))
        if began > cutoff:
            raise RuntimeError("A reservation may still be in flight; retry after the request duration")


# The Neon successor installed on 25 Nov: the base the Apify newest-safe-build successor changes.
APIFY_BASE_POLICY_SHA256 = "25f5685bfd2b53eac7af18b521765ca558e8cda421ae0d3b254c81e4518366b8"


def build_apify_newest_safe_successor(installed: dict) -> dict:
    """The installed policy with the Apify build in newest_safe mode and no image-digest pin; nothing else.

    Apify stopped reporting build image digests on 2026-11-27; the build ID, schema and pricing pins stay.
    """
    apify = installed.get("providers", {}).get("apify")
    if not isinstance(apify, dict) or "actor_build" not in apify:
        raise RuntimeError("Installed policy has no Apify build pin")
    if apify.get("actor_build_mode", "pinned") != "pinned":
        raise RuntimeError("Installed policy already follows the newest safe Apify build")
    successor = copy.deepcopy(installed)
    successor["providers"]["apify"]["actor_build_mode"] = "newest_safe"
    successor["providers"]["apify"]["actor_image_digest"] = None
    return successor


def pointer_ledger(ledger: dict, policy_sha: str) -> dict:
    """The ledger with only its policy pointer moved (the Apify build fields live in the policy, not the ledger)."""
    result = copy.deepcopy(ledger)
    result["automatic_policy_sha256"] = policy_sha
    return result


# The newest-safe-build successor installed on 27 Nov: the base the Apify price-cushion successor changes (SIY-291).
APIFY_CUSHION_BASE_POLICY_SHA256 = "eeecd6d8b21b65c0ddcf4b0e66a6c8be2e1fd053f83f1fbbfb318875dfa9d483"
# The Apify actor's price terms David approved, exactly as the provider check records them (read 27 Nov). Their
# digest is the installed actor_pricing_sha256; the successor refuses any other.
APIFY_APPROVED_PRICING_TERMS = {
    "event_prices_usd": {
        "music-data": {
            "BRONZE": "0.006",
            "DIAMOND": "0.002",
            "FREE": "0.007",
            "GOLD": "0.004",
            "PLATINUM": "0.003",
            "SILVER": "0.005"
        },
        "post": {
            "BRONZE": "0.0015",
            "DIAMOND": "0.0004",
            "FREE": "0.0017",
            "GOLD": "0.001",
            "PLATINUM": "0.0008",
            "SILVER": "0.0013"
        },
        "post-details": {
            "BRONZE": "0.0008",
            "DIAMOND": "0.0001",
            "FREE": "0.001",
            "GOLD": "0.0006",
            "PLATINUM": "0.0002",
            "SILVER": "0.0007"
        }
    },
    "minimum_run_charge_cap_usd": "0.005",
    "model": "PAY_PER_EVENT",
    "started_at": "2026-10-07T13:39:36.280Z"
}


def build_apify_price_cushion_successor(installed: dict, terms: dict = APIFY_APPROVED_PRICING_TERMS) -> dict:
    """The installed policy with the approved Apify price terms recorded beside their pin; nothing else.

    With the terms recorded, a refresh accepts a price up to 30% above them (automatic_spend.apify_price_review).
    """
    apify = installed.get("providers", {}).get("apify")
    if not isinstance(apify, dict) or "actor_pricing_sha256" not in apify:
        raise RuntimeError("Installed policy has no Apify pricing pin")
    if "actor_pricing_terms" in apify:
        raise RuntimeError("Installed policy already records approved Apify price terms")
    from .automatic_spend import digest
    if digest(terms) != apify["actor_pricing_sha256"]:
        raise RuntimeError("Approved Apify price terms do not match the installed pricing pin")
    successor = copy.deepcopy(installed)
    successor["providers"]["apify"]["actor_pricing_terms"] = copy.deepcopy(terms)
    return successor
