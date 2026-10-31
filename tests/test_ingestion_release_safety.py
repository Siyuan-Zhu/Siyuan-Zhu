from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import json
import pytest
import httpx

from src.zeitgeist import collectors


class ApprovedGuard:
    def __init__(self):
        self.calls = []

    def call(self, provider, operation, maximum, fn):
        self.calls.append((provider, operation, maximum))
        return fn()


def test_instagram_uses_official_contract_and_server_charge_cap():
    response = httpx.Response(
        200, json=[], request=httpx.Request("POST", "https://api.apify.com")
    )
    with patch.object(collectors.httpx, "Client") as cls:
        cls.return_value.__enter__.return_value.post.return_value = response
        collectors.fetch_instagram_batch(
            ["a24"], "test-token", spend_guard=ApprovedGuard(), actor_build="1.2.3"
        )
        kw = cls.return_value.__enter__.return_value.post.call_args.kwargs
        assert kw["json"]["username"] == ["a24"]
        assert "usernames" not in kw["json"]
        assert kw["json"]["dataDetailLevel"] == "basicData"
        assert kw["params"]["build"] == "1.2.3"
        assert kw["params"]["maxTotalChargeUsd"] > 0
        assert "token" not in kw["params"]


def test_instagram_requests_five_posts_but_retains_three_per_source_day():
    rows = [{"ownerUsername": "a24", "shortCode": f"post{i}", "caption": "Public post",
             "timestamp": "2026-10-30T12:00:00Z"} for i in range(5)]
    response = httpx.Response(200, json=rows, request=httpx.Request("POST", "https://api.apify.com"))
    with patch.object(collectors.httpx, "Client") as cls:
        cls.return_value.__enter__.return_value.post.return_value = response
        guard = ApprovedGuard()
        items = collectors.fetch_instagram_batch(["a24"], "test-token", spend_guard=guard, actor_build="1.2.3")
        kw = cls.return_value.__enter__.return_value.post.call_args.kwargs
        assert kw["json"]["resultsLimit"] == 5
        assert kw["params"]["maxTotalChargeUsd"] == pytest.approx(.01)
        assert guard.calls[0][2] == pytest.approx(.01)
        assert len(items) == 3


@pytest.mark.parametrize("rows", [[{"error": "not-found", "username": "a24"}], [{"unexpected": "record"}]])
def test_http_success_with_error_or_unknown_dataset_is_not_successful_collection(rows):
    response = httpx.Response(200, json=rows, request=httpx.Request("POST", "https://api.apify.com"))
    with patch.object(collectors.httpx, "Client") as cls:
        cls.return_value.__enter__.return_value.post.return_value = response
        with pytest.raises(collectors.IngestionUnavailable):
            collectors.fetch_instagram_batch(["a24"], "test-token", spend_guard=ApprovedGuard(), actor_build="1.2.3")


@pytest.mark.parametrize("failure", ["400", "timeout"])
def test_instagram_does_not_retry_bad_input_or_unknown_running_actor(failure, caplog):
    with patch.object(collectors.httpx, "Client") as cls:
        client = cls.return_value.__enter__.return_value
        if failure == "400":
            client.post.return_value = httpx.Response(
                400,
                json={
                    "error": {
                        "type": "invalid-input",
                        "message": "Field username required",
                    }
                },
                request=httpx.Request("POST", "https://api.apify.com?token=DO_NOT_LOG"),
            )
        else:
            client.post.side_effect = httpx.ReadTimeout("DO_NOT_LOG")
        with pytest.raises(collectors.IngestionUnavailable):
            collectors.fetch_instagram_batch(
                ["a24"], "DO_NOT_LOG", spend_guard=ApprovedGuard(), actor_build="1.2.3"
            )
        assert client.post.call_count == 1
        assert "DO_NOT_LOG" not in caplog.text


def test_no_actor_call_without_verified_build_or_spending_allowance(monkeypatch):
    monkeypatch.delenv("APIFY_INSTAGRAM_BUILD", raising=False)
    with patch.object(collectors.httpx, "Client") as client:
        with pytest.raises(collectors.IngestionUnavailable):
            collectors.fetch_instagram_batch(["a24"], "token")
        client.assert_not_called()


def test_live_collection_never_publishes_seeded_fresh_signals(tmp_path, monkeypatch):
    from src.zeitgeist.adk_pipeline import ZeitgeistADKOrchestrator
    from src.state_store import InMemoryStateStore

    monkeypatch.setenv("ZEITGEIST_LIVE_COLLECT", "1")
    monkeypatch.delenv("APIFY_TOKEN", raising=False)
    monkeypatch.delenv("APIFY_API_TOKEN", raising=False)
    store = InMemoryStateStore()
    original = {"word": "KEEP_ME", "state_version": 77}
    store.save_pulse_state(original)
    with patch(
        "src.zeitgeist.adk_pipeline.fetch_all_curated_feeds", return_value=[]
    ), patch("src.zeitgeist.adk_pipeline.fetch_macro_scout_signals", return_value=[]):
        with pytest.raises(collectors.IngestionUnavailable):
            ZeitgeistADKOrchestrator(tmp_path, store).run(date_iso="2026-10-30")
    assert store.load_pulse_state()["word"] == "KEEP_ME"
    assert not (tmp_path / "assets" / "daily_pulse.svg").exists()
