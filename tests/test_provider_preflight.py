"""Read-only provider discovery must never start an actor or disclose credentials."""

from datetime import datetime, timezone
import json

import httpx
import pytest

from scripts.provider_preflight import run_preflight

NOW = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
CYCLE = {"startAt": "2026-10-01T00:00:00Z", "endAt": "2026-10-31T23:59:59Z"}


def provider_client(secret="DO_NOT_DISCLOSE", build="0.0.648", *, default_build=None):
    default_build = default_build or build
    requests = []

    def handler(req):
        requests.append(req)
        assert not any(x in str(req.url) for x in [secret, "token=", "key="])
        if req.method == "POST":
            assert req.url.path.endswith("/validate-input")
            assert req.url.params["build"] == build
            body = json.loads(req.content)
            return (
                httpx.Response(
                    400, json={"error": {"type": "invalid-input", "message": secret}}
                )
                if "usernames" in body
                else httpx.Response(200, json={"valid": True})
            )
        if req.url.host == "generativelanguage.googleapis.com":
            return httpx.Response(
                200,
                json={
                    "models": [
                        {
                            "name": "models/gemini-test",
                            "supportedGenerationMethods": ["generateContent"],
                        }
                    ]
                },
            )
        if req.url.host == "api.bfl.ai":
            return httpx.Response(200, json={"credits": 123.45})
        if req.url.host == "api.unsplash.com":
            return httpx.Response(
                200,
                json=[{"private_field": secret}],
                headers={"X-Ratelimit-Limit": "50", "X-Ratelimit-Remaining": "49"},
            )
        p = req.url.path
        if p.endswith("/users/me"):
            body = {
                "plan": {
                    "id": "FREE",
                    "tier": "FREE",
                    "isEnabled": True,
                    "monthlyBasePriceUsd": 0,
                    "monthlyUsageCreditsUsd": 5,
                },
                "email": secret,
                "proxy": {"password": secret},
            }
        elif p.endswith("/usage/monthly"):
            body = {
                "usageCycle": CYCLE,
                "totalUsageCreditsUsdAfterVolumeDiscount": 0.10000000000000002,
            }
        elif p.endswith("/limits"):
            body = {
                "monthlyUsageCycle": CYCLE,
                "limits": {"maxMonthlyUsageUsd": 5},
                "current": {
                    "monthlyUsageUsd": 0.10000000000000002,
                    "activeActorJobCount": 0,
                },
            }
        elif p.endswith("/builds/default") or p.startswith("/v2/actor-builds/"):
            is_default = p.endswith("/builds/default")
            body = {
                "id": "default-build-id" if is_default else p.rsplit("/", 1)[-1],
                "actId": "nH2AHrwxeTRJoN5hX",
                "buildNumber": default_build if is_default else build,
                "status": "SUCCEEDED",
                "inputSchema": {
                    "required": ["username"],
                    "properties": {
                        "username": {"type": "array"},
                        "dataDetailLevel": {"enum": ["basicData", "detailedData"]},
                    },
                },
                "imageDigest": "d" * 64,
            }
        elif p.endswith("/builds"):
            body = {
                "items": [
                    {
                        "id": "build-id",
                        "actId": "nH2AHrwxeTRJoN5hX",
                        "buildNumber": build,
                    }
                ],
                "total": 1,
                "count": 1,
                "offset": 0,
                "limit": 1000,
            }
        else:
            body = {
                "id": "nH2AHrwxeTRJoN5hX",
                "taggedBuilds": {
                    "latest": {
                        "buildId": "default-build-id",
                        "buildNumber": default_build,
                    }
                },
                "defaultRunOptions": {"build": "latest"},
                "pricingInfos": [
                    {
                        "startedAt": "2026-10-01T00:00:00Z",
                        "pricingModel": "PAY_PER_EVENT",
                        "minimalMaxTotalChargeUsd": 0.005,
                        "pricingPerEvent": {
                            "actorChargeEvents": {
                                "post": {
                                    "eventTieredPricingUsd": {
                                        "FREE": {"tieredEventPriceUsd": 0.0017}
                                    }
                                }
                            }
                        },
                    }
                ],
            }
        return httpx.Response(200, json={"data": body})

    return httpx.Client(transport=httpx.MockTransport(handler)), requests


def test_missing_credentials_skip_network_and_report_no_go():
    with httpx.Client(
        transport=httpx.MockTransport(lambda req: pytest.fail("unexpected network"))
    ) as client:
        report = run_preflight(client, {}, now=NOW, required_models=["gemini-test"])
    assert report["release_decision"] == "NO_GO"
    assert not report["metadata_ok"]
    assert all(
        report[p]["status"] == "missing_credentials"
        for p in ["apify", "gemini", "unsplash"]
    )


def test_metadata_contract_credit_math_and_no_paid_requests():
    client, requests = provider_client()
    env = {
        "APIFY_TOKEN": "DO_NOT_DISCLOSE",
        "GEMINI_API_KEY": "DO_NOT_DISCLOSE",
        "UNSPLASH_ACCESS_KEY": "DO_NOT_DISCLOSE",
        "BFL_API_KEY": "DO_NOT_DISCLOSE",
        "APIFY_INSTAGRAM_BUILD": "0.0.648",
    }
    report = run_preflight(client, env, now=NOW, required_models=["gemini-test"])
    assert report["metadata_ok"]
    assert (
        report["apify"]["billing"]["remaining_included_credit_usd"]
        == "4.89999999999999998"
    )
    assert report["apify"]["billing"]["credit_expires_at"] == CYCLE["endAt"]
    assert report["apify"]["input_validation"]["legacy_plural_status"] == 400
    assert report["apify"]["input_validation"]["correct_singular_status"] == 200
    assert (
        report["release_decision"] == "NO_GO"
    )  # Metadata does not prove live integration or all billing.
    assert "DO_NOT_DISCLOSE" not in json.dumps(report)
    assert all(
        req.method == "GET" or req.url.path.endswith("/validate-input")
        for req in requests
    )


def test_changed_build_does_not_certify_configured_pin():
    client, _ = provider_client(build="0.0.649")
    report = run_preflight(
        client,
        {"APIFY_TOKEN": "DO_NOT_DISCLOSE", "APIFY_INSTAGRAM_BUILD": "0.0.648"},
        now=NOW,
    )
    assert not report["apify"].get("build_pin_verified")
    assert not report["metadata_ok"]


def test_raw_provider_errors_and_query_urls_are_never_echoed():
    def handler(req):
        return httpx.Response(
            403,
            json={
                "error": {
                    "type": "DO_NOT_DISCLOSE",
                    "message": "https://x.invalid?key=DO_NOT_DISCLOSE",
                }
            },
            request=req,
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = run_preflight(client, {"GEMINI_API_KEY": "DO_NOT_DISCLOSE"}, now=NOW)
    output = json.dumps(report)
    assert "DO_NOT_DISCLOSE" not in output
    assert "?key=" not in output
    assert report["gemini"]["status"] == "unavailable"


def test_mismatched_cycles_or_inflight_actor_cannot_verify_credit():
    from scripts.provider_preflight import inspect_apify_billing

    user = {
        "plan": {
            "isEnabled": True,
            "monthlyBasePriceUsd": "0",
            "monthlyUsageCreditsUsd": "5",
        }
    }
    usage = {"usageCycle": CYCLE, "totalUsageCreditsUsdAfterVolumeDiscount": "1"}
    limits = {
        "monthlyUsageCycle": dict(CYCLE, endAt="2026-11-30T23:59:59Z"),
        "current": {"monthlyUsageUsd": "1", "activeActorJobCount": 0},
    }
    assert (
        inspect_apify_billing(user, usage, limits, NOW)["remaining_included_credit_usd"]
        is None
    )
    limits["monthlyUsageCycle"] = CYCLE
    limits["current"]["activeActorJobCount"] = 1
    assert (
        inspect_apify_billing(user, usage, limits, NOW)["remaining_included_credit_usd"]
        is None
    )


def test_unsplash_failure_is_reported_without_echoing_photo_data():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(403, text="DO_NOT_DISCLOSE")
        )
    )
    report = run_preflight(client, {"UNSPLASH_ACCESS_KEY": "DO_NOT_DISCLOSE"}, now=NOW)
    assert report["unsplash"]["http_status"] == 403
    assert "DO_NOT_DISCLOSE" not in json.dumps(report)


def test_gemini_reports_only_fingerprint_and_masked_suffix():
    import hashlib

    client, _ = provider_client(secret="example-google-key-Yccg")
    report = run_preflight(
        client,
        {"GEMINI_API_KEY": "example-google-key-Yccg"},
        now=NOW,
        required_models=["gemini-test"],
    )
    assert (
        report["gemini"]["credential_fingerprint"]
        == hashlib.sha256(b"example-google-key-Yccg").hexdigest()[:16]
    )
    assert report["gemini"]["masked_last4"] == "...Yccg"
    assert "example-google-key-Yccg" not in json.dumps(report)


def test_bfl_reads_only_credits_and_never_claims_cash_allowance():
    def handler(req):
        assert req.method == "GET"
        assert str(req.url) == "https://api.bfl.ai/v1/credits"
        assert req.headers["x-key"] == "do-not-disclose-bfl-secret"
        return httpx.Response(
            200, json={"credits": 123.45, "private_field": "do-not-disclose-bfl-secret"}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = run_preflight(
        client, {"BFL_API_KEY": "do-not-disclose-bfl-secret"}, now=NOW
    )
    assert report["bfl"]["balance_credits"] == "123.45"
    assert report["bfl"]["billing_status"] == "unknown"
    assert report["bfl"]["cash_allowance_verified"] is False
    assert report["bfl"]["auto_topup_status"] == "unknown"
    assert report["bfl"]["credit_expiry_status"] == "unknown"
    assert "do-not-disclose-bfl-secret" not in json.dumps(report)


@pytest.mark.parametrize("balance", [-1, None, True, "100"])
def test_bfl_invalid_credit_metadata_is_not_a_verified_balance(balance):
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"credits": balance})
        )
    )
    report = run_preflight(client, {"BFL_API_KEY": "do-not-disclose"}, now=NOW)
    assert report["bfl"]["status"] == "unavailable"


def test_changed_provider_default_does_not_block_the_available_exact_configured_build():
    client, requests = provider_client(default_build="0.0.649")
    report = run_preflight(
        client,
        {"APIFY_TOKEN": "DO_NOT_DISCLOSE", "APIFY_INSTAGRAM_BUILD": "0.0.648"},
        now=NOW,
    )
    apify = report["apify"]
    assert apify["status"] == "metadata_verified"
    assert apify["observed_default_build"] == "0.0.649"
    assert apify["configured_build"] == "0.0.648"
    assert apify["build_pin_verified"] is True
    assert apify["build_id"] == "build-id"
    assert any(request.url.path == "/v2/actor-builds/build-id" for request in requests)
    assert all(
        request.url.params["build"] == "0.0.648"
        for request in requests
        if request.method == "POST"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("actId", "foreign-actor"),
        ("buildNumber", "0.0.999"),
        ("imageDigest", None),
        ("id", "foreign-build"),
    ],
)
def test_exact_immutable_build_rejects_wrong_actor_number_image_or_returned_id(
    field, value
):
    base, requests = provider_client(default_build="0.0.649")

    def handler(request):
        response = base._transport.handle_request(request)
        if request.url.path.startswith("/v2/actor-builds/"):
            body = response.json()
            body["data"][field] = value
            return httpx.Response(200, json=body)
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = run_preflight(
        client,
        {"APIFY_TOKEN": "DO_NOT_DISCLOSE", "APIFY_INSTAGRAM_BUILD": "0.0.648"},
        now=NOW,
    )
    assert report["apify"]["status"] == "unavailable"
    assert all(request.method == "GET" for request in requests)
    assert "DO_NOT_DISCLOSE" not in json.dumps(report)


def test_explicit_build_id_survives_unavailable_default_and_inventory():
    base, requests = provider_client(default_build="0.0.649")

    def handler(request):
        if request.url.path.endswith("/builds/default") or request.url.path.endswith(
            "/builds"
        ):
            return httpx.Response(403, json={"error": {"message": "DO_NOT_DISCLOSE"}})
        return base._transport.handle_request(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = run_preflight(
        client,
        {
            "APIFY_TOKEN": "DO_NOT_DISCLOSE",
            "APIFY_INSTAGRAM_BUILD": "0.0.648",
            "APIFY_INSTAGRAM_BUILD_ID": "build-id",
        },
        now=NOW,
    )
    assert report["apify"]["status"] == "metadata_verified"
    assert report["apify"]["build_id"] == "build-id"
    assert report["apify"]["observed_default_build"] is None
    assert any(request.url.path == "/v2/actor-builds/build-id" for request in requests)


def test_exact_build_inventory_resolution_uses_bounded_documented_pagination():
    from scripts.provider_preflight import _exact_apify_build, APIFY_ACTOR_ID

    calls = []

    def get(path, *, params=None):
        calls.append((path, params))
        if path.endswith("/builds/default"):
            return {"id": "newest-build", "buildNumber": "0.0.999"}
        if path.endswith("/builds"):
            offset = params["offset"]
            assert params["limit"] == 1000 and params["desc"] == "true"
            rows = (
                [{"id": "unrelated-build", "buildNumber": "0.0.999"}]
                if offset == 0
                else [{"id": "pinned-build", "buildNumber": "0.0.648"}]
            )
            return {"items": rows, "offset": offset, "count": 1, "total": 2}
        assert path == "/actor-builds/pinned-build"
        return {
            "id": "pinned-build",
            "actId": APIFY_ACTOR_ID,
            "buildNumber": "0.0.648",
            "imageDigest": "d" * 64,
        }

    build, diagnostic = _exact_apify_build(get, {"id": APIFY_ACTOR_ID}, "0.0.648")
    assert build["id"] == "pinned-build" and diagnostic == "0.0.999"
    assert [
        options["offset"] for path, options in calls if path.endswith("/builds")
    ] == [0, 1]


def test_selected_build_schema_is_bound_not_the_new_default_schema():
    import hashlib

    base, requests = provider_client(default_build="0.0.649")
    schema = {"required": ["username"], "properties": {"username": {"type": "array"}}}

    def handler(request):
        response = base._transport.handle_request(request)
        if request.url.path.startswith("/v2/actor-builds/"):
            body = response.json()
            body["data"]["actorDefinition"] = {"input": schema}
            body["data"]["inputSchema"] = None
            return httpx.Response(200, json=body)
        return response

    client = httpx.Client(transport=httpx.MockTransport(handler))
    report = run_preflight(
        client,
        {"APIFY_TOKEN": "DO_NOT_DISCLOSE", "APIFY_INSTAGRAM_BUILD": "0.0.648"},
        now=NOW,
    )
    assert report["apify"]["status"] == "metadata_verified"
    assert (
        report["apify"]["input_schema_sha256"]
        == hashlib.sha256(
            json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert report["apify"]["actor_id"] == "nH2AHrwxeTRJoN5hX"
    assert report["apify"]["image_digest"] == "d" * 64


@pytest.mark.parametrize("pin", [None, "latest", "0.0.648/path", ""])
def test_non_numeric_pin_cannot_trigger_metadata_or_validation_requests(pin):
    client, requests = provider_client()
    report = run_preflight(
        client,
        {"APIFY_TOKEN": "DO_NOT_DISCLOSE", "APIFY_INSTAGRAM_BUILD": pin},
        now=NOW,
    )
    assert report["apify"]["status"] == "unavailable"
    assert requests == []
