from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from unittest.mock import Mock
import pytest
from src.zeitgeist.spend_guard import (
    BudgetGuard,
    MemoryLedger,
    SpendBlocked,
    guard_genai_client,
)

NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def document(credit="0"):
    return {
        "month": "2026-10",
        "cash_ceiling_usd": "2.50",
        "cash_spent_before_ledger_usd": "0",
        "providers": {
            "apify": {
                "account_id": "sha256:apify-account", "plan_id": "free",
                "credit_baseline_usd": credit,
                "allowance_period_id": "2026-10", "allowance_starts_at": "2026-10-01T00:00:00Z",
                "allowance_expires_at": "2026-11-01T00:00:00Z",
                "remaining_credit_usd": credit,
                "verified_at": NOW.isoformat(),
                "verification_reference": "billing-snapshot-sha",
                "operation_caps_usd": {},
            },
            "gemini": {
                "account_id": "sha256:gemini-account", "plan_id": "verified-plan",
                "credit_baseline_usd": "0",
                "allowance_period_id": "2026-10", "allowance_starts_at": "2026-10-01T00:00:00Z",
                "allowance_expires_at": "2026-11-01T00:00:00Z",
                "remaining_credit_usd": "0",
                "verified_at": NOW.isoformat(),
                "verification_reference": "billing-snapshot-sha",
                "operation_caps_usd": {"generate_content:test": "0.1"},
                "request_limits": {
                    "generate_content:test": {
                        "pricing_reference": "verified-price-snapshot",
                        "max_input_bytes": 100,
                        "max_output_tokens": 32,
                    }
                },
            },
        },
    }


def test_combined_production_staging_calls_cannot_overreserve():
    ledger = MemoryLedger(document())
    guards = [BudgetGuard(ledger, lambda: NOW), BudgetGuard(ledger, lambda: NOW)]

    def reserve(i):
        try:
            guards[i % 2].reserve("apify", "run", "0.50")
            return True
        except SpendBlocked:
            return False

    with ThreadPoolExecutor(max_workers=10) as pool:
        accepted = list(pool.map(reserve, range(20)))
    assert sum(accepted) == 5
    assert guards[0].snapshot()["cash_exposure_usd"] == "2.50"


def test_only_verified_credits_offset_cash_and_timeout_retains_reservation():
    guard = BudgetGuard(MemoryLedger(document("5.00")), lambda: NOW)

    def timeout():
        raise TimeoutError()

    with pytest.raises(TimeoutError):
        guard.call("apify", "run", "5.00", timeout)
    assert guard.snapshot()["cash_exposure_usd"] == "0"
    guard.reserve("apify", "another", "2.50")
    with pytest.raises(SpendBlocked):
        guard.reserve("apify", "extra", "0.01")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["providers"]["apify"].pop("verification_reference"),
        lambda d: d["providers"]["apify"].update(verified_at="2026-10-28T00:00:00Z"),
        lambda d: d.update(month="2026-09"),
    ],
)
def test_unknown_or_stale_allowance_blocks_provider_before_call(mutate):
    doc = document()
    mutate(doc)
    fn = Mock()
    with pytest.raises(SpendBlocked):
        BudgetGuard(MemoryLedger(doc), lambda: NOW).call("apify", "run", "0.01", fn)
    fn.assert_not_called()


def test_genai_wrapper_reserves_per_request_and_unknown_models_block():
    client = Mock()
    guard = BudgetGuard(MemoryLedger(document()), lambda: NOW)
    wrapper = guard_genai_client(client, guard)
    wrapper.models.generate_content(model="test", contents="hi")
    assert guard.snapshot()["reservation_count"] == 1
    with pytest.raises(SpendBlocked):
        wrapper.models.generate_content(model="unknown")
    assert client.models.generate_content.call_count == 1


def test_free_tier_is_explicit_and_bounded_without_assuming_infinite_credit():
    doc = document()
    account = doc["providers"]["gemini"]
    account["operation_caps_usd"]["generate_content:test"] = "0"
    guard = BudgetGuard(MemoryLedger(doc), lambda: NOW)
    with pytest.raises(SpendBlocked):
        guard.reserve("gemini", "generate_content:test")
    account.update(
        billing_mode="verified_free_tier",
        verified_free_operations=["generate_content:test"],
        free_request_limit=1,
    )
    guard = BudgetGuard(MemoryLedger(doc), lambda: NOW)
    guard.reserve("gemini", "generate_content:test")
    with pytest.raises(SpendBlocked):
        guard.reserve("gemini", "generate_content:test")
    assert guard.snapshot()["cash_exposure_usd"] == "0"


def test_oversized_sdk_requests_block_before_paid_method():
    client = Mock()
    guard = BudgetGuard(MemoryLedger(document()), lambda: NOW)
    wrapper = guard_genai_client(client, guard)
    with pytest.raises(SpendBlocked):
        wrapper.models.generate_content(model="test", contents="a" * 101)
    with pytest.raises(SpendBlocked):
        wrapper.models.generate_content(
            model="test", contents="a", config={"max_output_tokens": 33}
        )
    client.models.generate_content.assert_not_called()
