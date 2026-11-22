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
from .spend_guard import HYDROGEN_OPERATION, HYDROGEN_OUTPUT_TOKENS

INSTALLED_POLICY_SHA256 = "e664ccce9873b17fedaa7eda48e7c928be7540fd3db6c7d99168a125045a661f"
FLASH_OPERATION = "generate_content:gemini-3.8-flash"
EMBEDDING_001_OPERATION = "embed_content:gemini-embedding-001"

REDACTED_KEYS = frozenset({"billing_account_id", "credential_fingerprint", "account_id",
                           "account_fingerprint", "approval_reference", "reference",
                           "verification_reference"})


def hydrogen_cap_usd(max_output_tokens: int) -> str:
    """The exact per-call cap: a full 65536-token input window plus the whole output allowance."""
    total = Decimal(65536) * Decimal("1.50") + Decimal(max_output_tokens) * Decimal("7.50")
    return format(total / Decimal(1000000), "f")


HYDROGEN_BUILD_THINKING_LEVELS = ("high", "medium")


def _check_inputs(model_version, tokens_per_image, max_output_tokens, thinking_level):
    if not isinstance(model_version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", model_version):
        raise RuntimeError("Hydrogen model version is malformed")
    if type(tokens_per_image) is not int or not 1 <= tokens_per_image <= 4096:
        raise RuntimeError("Hydrogen tokens per image must be an integer from 1 to 4096")
    if type(max_output_tokens) is not int or max_output_tokens not in HYDROGEN_OUTPUT_TOKENS:
        raise RuntimeError("Hydrogen max output tokens must be one of 4000, 6000, 8000, 10000, 15000")
    if type(thinking_level) is not str or thinking_level not in HYDROGEN_BUILD_THINKING_LEVELS:
        raise RuntimeError("Hydrogen thinking level must be high or medium")


def hydrogen_operation(flash_limits: dict, *, model_version: str, tokens_per_image: int,
                       max_output_tokens: int, thinking_level: str) -> tuple[str, str, dict]:
    _check_inputs(model_version, tokens_per_image, max_output_tokens, thinking_level)
    limits = copy.deepcopy(flash_limits)
    limits.update(input_token_usd_per_million="1.50", output_token_usd_per_million="7.50",
                  max_output_tokens=max_output_tokens, thinking_level=thinking_level,
                  image_tokens_per_image=tokens_per_image,
                  allowed_response_model_versions=[model_version])
    return HYDROGEN_OPERATION, hydrogen_cap_usd(max_output_tokens), limits


def build_successor(installed: dict, *, hydrogen_model_version: str, hydrogen_tokens_per_image: int,
                    hydrogen_max_output_tokens: int, hydrogen_thinking_level: str,
                    tariff_sha256: str) -> dict:
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
    operation, cap, limits = hydrogen_operation(flash, model_version=hydrogen_model_version,
                                                tokens_per_image=hydrogen_tokens_per_image,
                                                max_output_tokens=hydrogen_max_output_tokens,
                                                thinking_level=hydrogen_thinking_level)
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
