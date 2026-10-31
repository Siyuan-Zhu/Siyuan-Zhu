"""Verified per-profile absence and collaborative attribution in Apify output."""

import copy
import json
from unittest.mock import MagicMock

import httpx
import pytest

from src.zeitgeist import collectors, adk_pipeline as pipeline
from tests.test_ingestion_release_safety import ApprovedGuard


def _post(owner, code, **extra):
    return {
        "ownerUsername": owner,
        "shortCode": code,
        "url": "https://www.instagram.com/p/" + code + "/",
        "caption": "Public craft and community.",
        "timestamp": "2026-10-29T17:00:00Z",
        **extra,
    }


def _missing(handle):
    return {
        "error": "not_found",
        "errorDescription": "Profile does not exist",
        "username": handle,
        "url": "https://www.instagram.com/" + handle,
    }


def _run(monkeypatch, rows, handles, health=None):
    response = httpx.Response(
        200, json=rows, request=httpx.Request("POST", "https://api.apify.com")
    )
    client = MagicMock()
    client.__enter__.return_value.post.return_value = response
    monkeypatch.setattr(collectors.httpx, "Client", lambda **kwargs: client)
    result = collectors.fetch_instagram_batch(
        handles,
        "not-a-real-token",
        spend_guard=ApprovedGuard(),
        actor_build="0.0.648",
        health=health,
    )
    assert client.__enter__.return_value.post.call_count == 1
    return result


def test_mixed_observed_shapes_keep_valid_posts_skip_only_exact_missing_profiles_and_dedup(
    monkeypatch,
):
    handles = ["missing_seed_a", "missing_seed_b", "requested_a", "requested_b"]
    collaborative = _post(
        "outside_owner",
        "shared",
        coauthorProducers=[{"username": "requested_b"}, {"username": "requested_a"}],
    )
    rows = [
        _missing(handles[0]),
        _missing(handles[1]),
        collaborative,
        _post("requested_b", "unique"),
        copy.deepcopy(collaborative),
    ]
    health = {}
    items = _run(monkeypatch, rows, handles, health)
    assert len(items) == 2 and len({item["url_hash"] for item in items}) == 2
    shared = next(item for item in items if item["url"].endswith("/shared"))
    assert shared["source_id"] == "o3_ig_requested_a"
    assert shared["instagram_provenance"] == {
        "owner_username": "outside_owner",
        "requested_username": "requested_a",
        "association": "coauthor",
        "verified_by": "coauthorProducers.username",
    }
    assert health["instagram_unavailable_handle_count"] == 2
    assert health["instagram_skipped_record_count"] == 2
    assert health["instagram_unavailable_reasons"] == {"not_found": 2}
    assert health["instagram_degraded"] is True
    assert health["instagram_normalized_post_count"] == 2
    public = collectors.public_collection_health(health)
    assert public["instagram_unavailable_reasons"] == {"not_found": 2}
    assert all(handle not in json.dumps(public) for handle in handles)


@pytest.mark.parametrize(
    "change",
    [
        {"error": "rate_limited"},
        {"username": "unrequested"},
        {"username": None},
        {"url": "https://www.instagram.com/other"},
    ],
)
def test_unknown_or_unverifiable_profile_errors_still_block_whole_batch(
    monkeypatch, change
):
    rows = [_post("requested_a", "valid"), _missing("requested_a") | change]
    with pytest.raises(collectors.IngestionUnavailable):
        _run(monkeypatch, rows, ["requested_a"], {})


@pytest.mark.parametrize(
    "coauthors",
    [[{"username": "unrequested"}], [{"handle": "requested_a"}], ["requested_a"], None],
)
def test_foreign_owner_without_verified_requested_coauthor_blocks(
    monkeypatch, coauthors
):
    with pytest.raises(collectors.IngestionUnavailable):
        _run(
            monkeypatch,
            [_post("outside_owner", "foreign", coauthorProducers=coauthors)],
            ["requested_a"],
            {},
        )


def test_known_missing_profiles_never_create_fake_posts(monkeypatch):
    health = {}
    assert _run(monkeypatch, [_missing("requested_a")], ["requested_a"], health) == []
    assert health["instagram_unavailable_handle_count"] == 1
    assert health["instagram_normalized_post_count"] == 0


def test_missing_profile_does_not_waive_live_fresh_instagram_coverage(
    tmp_path, monkeypatch
):
    from tests.test_pipeline_launch_acceptance import _live_fixture
    from datetime import datetime, timezone

    store = _live_fixture(monkeypatch, tmp_path)
    cache = store.load_ingestion_cache()
    cache.pop("ig")
    store.load_ingestion_cache = lambda **kwargs: dict(cache)
    monkeypatch.setattr(
        pipeline, "fetch_instagram_batch", collectors.fetch_instagram_batch
    )
    monkeypatch.setenv("APIFY_INSTAGRAM_BUILD", "0.0.648")
    monkeypatch.setattr(
        "src.zeitgeist.spend_guard.get_spend_guard", lambda **kwargs: ApprovedGuard()
    )
    client = MagicMock()
    client.__enter__.return_value.post.return_value = httpx.Response(
        200,
        json=[_missing("demo")],
        request=httpx.Request("POST", "https://api.apify.com"),
    )
    monkeypatch.setattr(collectors.httpx, "Client", lambda **kwargs: client)
    health = {}
    with pytest.raises(collectors.IngestionUnavailable, match="diverse"):
        pipeline.collect_96h_signals(
            tmp_path,
            store=store,
            live_collect=True,
            health=health,
            now_epoch=datetime(2026, 10, 29, 17, 30, tzinfo=timezone.utc).timestamp(),
        )
    assert health["publishable"] is False
    assert health["instagram_unavailable_reasons"] == {"not_found": 1}
    assert health["instagram_item_count"] == 0


def test_verified_coauthor_is_selected_in_requested_order_not_provider_array_order(
    monkeypatch,
):
    rows = [
        _post(
            "outside_owner",
            "shared",
            coauthorProducers=[
                {"username": "requested_b"},
                {"username": "requested_a"},
            ],
        )
    ]
    items = _run(monkeypatch, rows, ["requested_a", "requested_b"], {})
    assert items[0]["source_id"] == "o3_ig_requested_a"
    assert items[0]["instagram_provenance"]["association"] == "coauthor"


def test_sync_actor_read_timeout_covers_server_window_without_increasing_charge_cap(
    monkeypatch,
):
    response = httpx.Response(
        200, json=[], request=httpx.Request("POST", "https://api.apify.com")
    )
    client = MagicMock()
    client.__enter__.return_value.post.return_value = response
    monkeypatch.setattr(collectors.httpx, "Client", lambda **kwargs: client)
    collectors.fetch_instagram_batch(
        ["requested_a"],
        "not-a-real-token",
        spend_guard=ApprovedGuard(),
        actor_build="0.0.648",
    )
    kwargs = client.__enter__.return_value.post.call_args.kwargs
    assert isinstance(kwargs["timeout"], httpx.Timeout)
    assert kwargs["timeout"].read == 330.0 and kwargs["timeout"].connect == 15.0
    assert kwargs["params"]["timeout"] == 300
    assert kwargs["params"]["maxTotalChargeUsd"] == pytest.approx(0.01)
    assert client.__enter__.return_value.post.call_count == 1
