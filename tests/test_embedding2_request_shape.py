"""Embedding-2 request shape: one Content per text, so the service returns one vector per text."""
import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google import genai
from google.genai import types

from src.zeitgeist import collectors
from src.zeitgeist.spend_guard import SpendBlocked, guard_genai_client
from tests.test_automatic_guard import budget
from tests.test_embedding2_guard import OP, embedding2_limits, guard
from src.zeitgeist import automatic_spend as auto

E2 = "gemini-embedding-2"
E1 = "gemini-embedding-001"
OP1 = "embed_content:" + E1
PREFIX = "task: clustering | query: "
TEXTS = ["Alpha\nSynthetic systems summary.", "Beta\nSynthetic systems summary.",
         "Gamma\nSynthetic systems summary."]


def one_text_content(text):
    return types.Content(parts=[types.Part(text=text)])


def _is_content(item):
    return isinstance(item, types.Content) or (isinstance(item, dict) and "parts" in item)


class ServiceLikeEmbedder:
    """Behaves like the service: bare strings are aggregated into ONE vector; each Content gets its own."""

    def __init__(self):
        self.calls = []

    def embed_content(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        contents = kwargs["contents"]
        count = len(contents) if all(_is_content(item) for item in contents) else 1
        return types.EmbedContentResponse(embeddings=[
            types.ContentEmbedding(values=[0.25 + i / 100] * 768) for i in range(count)])


def service_client():
    models = ServiceLikeEmbedder()
    return SimpleNamespace(models=models), models


def rows():
    return [dict(url_hash=f"k{i}", title=t.split("\n")[0], summary="Synthetic systems summary.")
            for i, t in enumerate(TEXTS)]


# ---- what the installed SDK puts on the wire ----

def sdk_requests(monkeypatch, contents, model=E2):
    client = genai.Client(api_key="synthetic-not-a-key")
    sent = []
    def capture(method, path, request_dict, http_options=None):
        sent.append((path, request_dict))
        raise RuntimeError("captured; no network")
    monkeypatch.setattr(client.models._api_client, "request", capture)
    with pytest.raises(RuntimeError, match="captured"):
        client.models.embed_content(model=model, contents=contents, config={"output_dimensionality": 768})
    return sent[0]


def test_sdk_sends_each_single_text_content_as_its_own_request(monkeypatch):
    contents = collectors.embedding_request_contents(TEXTS, E2)
    path, body = sdk_requests(monkeypatch, contents)
    assert path.endswith(":batchEmbedContents")
    assert [r["content"]["parts"] for r in body["requests"]] == [[{"text": PREFIX + t}] for t in TEXTS]


def test_sdk_folds_bare_strings_into_one_content_for_embedding_2(monkeypatch):
    # Why the old request returned ONE vector: the SDK groups consecutive strings into one Content.
    path, body = sdk_requests(monkeypatch, [PREFIX + t for t in TEXTS])
    assert len(body["requests"]) == 1
    assert len(body["requests"][0]["content"]["parts"]) == len(TEXTS)


# ---- the embedder ----

def test_request_contents_for_embedding_2_are_single_text_contents():
    assert collectors.embedding_request_contents(TEXTS, E2) == [one_text_content(PREFIX + t) for t in TEXTS]


def test_request_contents_for_001_are_bare_strings():
    assert collectors.embedding_request_contents(TEXTS, E1) == TEXTS


def test_embedding_2_gets_one_vector_per_text_from_a_service_like_client():
    client, models = service_client()
    result = collectors.embed_missing_items(rows(), {}, client, model=E2, require_live=True)
    assert models.calls == [{"model": E2, "contents": [one_text_content(PREFIX + t) for t in TEXTS],
                             "config": {"output_dimensionality": 768}}]
    assert [r["embedding_source"] for r in result] == ["gemini"] * 3
    assert [r["embedding_model"] for r in result] == [E2] * 3


def test_embedding_001_still_sends_bare_strings():
    client, models = service_client()
    collectors.embed_missing_items(rows()[:1], {}, client, model=E1, require_live=True)
    assert models.calls == [{"model": E1, "contents": [TEXTS[0]], "config": {"output_dimensionality": 768}}]


@pytest.mark.parametrize("count", [1, 4])
def test_embedding_2_wrong_vector_count_is_a_failure(count):
    def embed(**kwargs):
        return types.EmbedContentResponse(embeddings=[
            types.ContentEmbedding(values=[0.25] * 768) for _ in range(count)])
    client = SimpleNamespace(models=SimpleNamespace(embed_content=embed))
    with pytest.raises(collectors.IngestionUnavailable) as caught:
        collectors.embed_missing_items(rows(), {}, client, model=E2, require_live=True)
    assert caught.value.public_diagnostics["code"] == "embedding-response-invalid"


# ---- the spend guard ----

def guard_with_001():
    budget_guard = guard()
    account = budget_guard.ledger.document["providers"]["gemini"]
    account["operation_caps_usd"][OP1] = ".1"
    account["request_limits"][OP1] = {"pricing_reference": "p", "max_input_bytes": 1000,
        "max_input_tokens": 10000, "count_tokens_reference": "official-no-charge",
        "input_token_usd_per_million": ".15", "output_token_usd_per_million": "0"}
    return budget_guard


def admit(contents, model=E2, budget_guard=None, tokens=30):
    budget_guard = budget_guard or (guard_with_001() if model == E1 else guard())
    operation = "embed_content:" + model
    bounded = budget_guard.bounded_genai_request(operation, {
        "model": model, "contents": contents, "config": {"output_dimensionality": 768}})
    counted_with = []
    def count_tokens(**kwargs):
        counted_with.append(kwargs["contents"])
        return SimpleNamespace(total_tokens=tokens)
    maximum, _ = budget_guard.counted_genai_cap(operation, bounded, count_tokens)
    return maximum, counted_with


@pytest.mark.parametrize("form", [
    lambda t: one_text_content(t),
    lambda t: {"parts": [{"text": t}]},
])
def test_guard_admits_single_text_contents_and_counts_the_texts(form):
    maximum, counted = admit([form("synthetic text one"), form("synthetic text two")])
    assert counted == ["synthetic text one", "synthetic text two"]
    assert maximum > 0


@pytest.mark.parametrize("contents", [
    ["synthetic text one", "synthetic text two"],
    [types.Content(parts=[types.Part(text="one"), types.Part(text="two")])],
    [types.Content(parts=[types.Part(inline_data=types.Blob(data=b"x", mime_type="image/png"))])],
    [types.Content(parts=[types.Part(text="")])],
    [{"parts": [{"text": "one"}, {"text": "two"}]}],
    [{"parts": []}],
    [{"parts": [{"text": "one"}], "extra": 1}],
], ids=["bare-strings", "two-parts", "image-part", "empty-text", "dict-two-parts", "no-parts", "extra-key"])
def test_guard_refuses_other_embedding_2_shapes_before_any_call(contents):
    budget_guard = guard()
    client = Mock()
    client.models.count_tokens.return_value = SimpleNamespace(total_tokens=5)
    with pytest.raises(SpendBlocked, match="Only bounded text embedding batches are priced"):
        guard_genai_client(client, budget_guard).models.embed_content(
            model=E2, contents=contents, config={"output_dimensionality": 768})
    client.models.embed_content.assert_not_called()
    assert budget_guard.snapshot()["reservation_count"] == 0


def test_guard_admits_bare_strings_for_001_as_today():
    maximum, counted = admit(["synthetic text one", "synthetic text two"], model=E1)
    assert counted == ["synthetic text one", "synthetic text two"]
    assert maximum > 0


def test_guard_refuses_contents_for_001_as_today():
    with pytest.raises(SpendBlocked, match="Only bounded text embedding batches are priced"):
        admit([one_text_content("synthetic text")], model=E1)


# ---- settlement ----

def test_embedding_2_hold_settles_from_a_reply_with_no_statistics():
    budget_guard = budget()
    account = budget_guard.automatic_policy["providers"]["gemini"]["account"]
    account["request_limits"][OP] = embedding2_limits()
    account["operation_caps_usd"][OP] = "0.04096"
    budget_guard.expected_policy_sha = auto.digest(budget_guard.automatic_policy)
    budget_guard.ledger.document["automatic_policy_sha256"] = budget_guard.expected_policy_sha
    budget_guard.ledger.document["providers"]["gemini"].update(copy.deepcopy(account))
    client = Mock()
    client.models.count_tokens.side_effect = [SimpleNamespace(total_tokens=30), SimpleNamespace(total_tokens=30)]
    client.models.embed_content.return_value = types.EmbedContentResponse(embeddings=[
        types.ContentEmbedding(values=[.1, .2]), types.ContentEmbedding(values=[.3, .4])])
    guard_genai_client(client, budget_guard).models.embed_content(
        model=E2, contents=collectors.embedding_request_contents(["one", "two"], E2),
        config={"output_dimensionality": 768})
    client.models.embed_content.assert_called_once()
    record = next(iter(budget_guard.ledger.document["reservations"].values()))
    assert record["operation"] == OP
    assert record["status"] == "settled"
    assert record["actual_gross_usd"] == record["max_gross_usd"]
    assert record["request_context"]["embedding_input_count"] == 2
