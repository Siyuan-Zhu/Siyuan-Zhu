"""Typed provider metadata survives failure without exporting raw diagnostics."""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from scripts import rehearse_staging_application as application
from src.zeitgeist import adk_pipeline as pipeline, collectors
from tests.test_ingestion_release_safety import ApprovedGuard
from tests.test_staging_application_rehearsal import PREFIX, setup


def _live_fixture(tmp_path, monkeypatch, *, response):
    root, wire, factory = setup(tmp_path, monkeypatch)
    config = {
        "active_interjection": {"enabled": False},
        "curated_o3_instagram_handles": ["a24"],
        "coverage_requirements": {
            "min_orbit_sources": 6,
            "min_macro_sources": 6,
            "required_orbit_tiers": ["O1", "O2", "O3"],
            "required_macro_tiers": ["M1", "M2", "M3"],
        },
    }
    (root / "config/orbit_sources.json").write_text(json.dumps(config))
    for key in (
        "ZEITGEIST_LIVE_COLLECT",
        "ZEITGEIST_LIVE_LLM",
        "ZEITGEIST_LIVE_UNSPLASH",
    ):
        monkeypatch.setenv(key, "1")
    monkeypatch.setenv("ZEITGEIST_OFFLINE_DEMO", "0")
    monkeypatch.setenv("APIFY_TOKEN", "DO_NOT_LOG")
    monkeypatch.setenv("APIFY_API_TOKEN", "DO_NOT_LOG")
    monkeypatch.setenv("APIFY_INSTAGRAM_BUILD", "0.0.648")
    monkeypatch.setattr(pipeline, "fetch_all_curated_feeds", lambda *a, **k: [])
    monkeypatch.setattr(
        "src.zeitgeist.spend_guard.get_spend_guard", lambda **kwargs: ApprovedGuard()
    )
    client = MagicMock()
    client.__enter__.return_value.post.return_value = response
    monkeypatch.setattr(collectors.httpx, "Client", lambda **kwargs: client)
    return root, wire, factory, client


def test_known_400_has_only_typed_status_code_build_and_health_counts_in_public_report(
    tmp_path, monkeypatch
):
    response = httpx.Response(
        400,
        json={
            "error": {
                "type": "invalid-input",
                "message": "DO_NOT_LOG https://private.invalid?token=DO_NOT_LOG",
            }
        },
        request=httpx.Request("POST", "https://private.invalid?token=DO_NOT_LOG"),
    )
    root, wire, factory, client = _live_fixture(
        tmp_path, monkeypatch, response=response
    )
    with pytest.raises(application.ApplicationRehearsalError) as failure:
        application.rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
        )
    report = failure.value.report
    assert report["provider_diagnostics"] == {
        "provider": "apify",
        "error_type": "HTTPStatusError",
        "status": 400,
        "code": "invalid-input",
        "build": "0.0.648",
    }
    assert report["collection"]["instagram"] == "unavailable"
    assert report["collection"]["cache_item_count"] == 0
    assert report["collection"]["curated_item_count"] == 0
    assert "DO_NOT_LOG" not in json.dumps(
        report
    ) and "private.invalid" not in json.dumps(report)
    assert client.__enter__.return_value.post.call_count == 1
    assert wire.docs and not wire.delete_batches


def test_http_success_with_dataset_error_reports_normalization_failure_not_http_failure(
    tmp_path, monkeypatch
):
    response = httpx.Response(
        200,
        json=[{"error": "no_items", "errorDescription": "DO_NOT_LOG"}],
        request=httpx.Request("POST", "https://private.invalid?token=DO_NOT_LOG"),
    )
    root, wire, factory, _ = _live_fixture(tmp_path, monkeypatch, response=response)
    with pytest.raises(application.ApplicationRehearsalError) as failure:
        application.rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
        )
    detail = failure.value.report["provider_diagnostics"]
    assert detail == {
        "provider": "apify",
        "status": 200,
        "error_type": "ProfileResultError",
        "code": "no_items",
        "build": "0.0.648",
    }
    assert "DO_NOT_LOG" not in json.dumps(failure.value.report)


def test_secret_containing_unknown_exception_cannot_inject_public_metadata(
    tmp_path, monkeypatch
):
    root, wire, factory = setup(tmp_path, monkeypatch)

    def fail(*args, **kwargs):
        error = RuntimeError("DO_NOT_LOG https://private.invalid?token=DO_NOT_LOG")
        error.public_diagnostics = {"provider": "DO_NOT_LOG", "message": "DO_NOT_LOG"}
        raise error

    monkeypatch.setattr(pipeline, "collect_96h_signals", fail)
    with pytest.raises(application.ApplicationRehearsalError) as failure:
        application.rehearse(
            root, PREFIX, store_factory=factory, private_root=tmp_path / "private"
        )
    report = failure.value.report
    assert report["error_type"] == "RuntimeError"
    assert "provider_diagnostics" not in report
    assert "DO_NOT_LOG" not in json.dumps(
        report
    ) and "private.invalid" not in json.dumps(report)


def test_public_health_and_metadata_drop_unapproved_fields_values_and_body_text():
    error = collectors.IngestionUnavailable(
        "DO_NOT_LOG",
        public_diagnostics={
            "provider": "apify",
            "status": "400?token=DO_NOT_LOG",
            "error_type": "HTTPStatusError",
            "code": "invalid input DO_NOT_LOG",
            "build": "0.0.648?token=DO_NOT_LOG",
            "message": "DO_NOT_LOG",
            "url": "DO_NOT_LOG",
        },
        collection_health={
            "mode": "cache",
            "cache_item_count": 2,
            "instagram_error": "DO_NOT_LOG",
            "secret_count": "DO_NOT_LOG",
            "fresh_items_by_tier": {"O1": 3, "DO_NOT_LOG": 1},
        },
    )
    assert error.public_diagnostics == {
        "provider": "apify",
        "error_type": "HTTPStatusError",
    }
    assert error.public_collection_health == {
        "mode": "cache",
        "cache_item_count": 2,
        "fresh_items_by_tier": {"O1": 3},
    }
    assert "DO_NOT_LOG" not in json.dumps(
        error.public_diagnostics | error.public_collection_health
    )
