"""Exercise the pinned SDK locally; every HTTP request stays in MockTransport."""
import copy
from decimal import Decimal

import httpx
import pytest
from google import genai
from google.genai import types

from src.zeitgeist.spend_guard import SpendBlocked, guard_genai_client
from tests.test_gemini38_contract import guard, request


def test_actual_sdk_continuation_cannot_send_a_second_paid_request():
    budget = guard()
    paid = []
    payload = request()
    original = copy.deepcopy(payload)

    def handler(req):
        assert req.method == 'POST'
        if req.url.path.endswith(':countTokens'):
            assert budget.snapshot()['reservation_count'] == 0
            return httpx.Response(200, json={'totalTokens': 10})
        assert req.url.path.endswith(':generateContent')
        paid.append(req)
        assert budget.snapshot()['reservation_count'] == 1
        # A second response terminates the unpatched SDK loop so the regression
        # fails with two observed requests, rather than hanging indefinitely.
        candidate = {'index': 0, 'finishReason': 'STOP',
                     'content': {'role': 'model', 'parts': [{'text': 'local fixture'}]}}
        if len(paid) == 1:
            candidate.update(finishReason='CONTINUATION', continuationToken='YWJj')
        return httpx.Response(200, json={'candidates': [candidate]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        with genai.Client(api_key='offline-fake-key', enterprise=False,
                          http_options=types.HttpOptions(httpx_client=transport)) as client:
            result = guard_genai_client(client, budget).models.generate_content(**payload)

    assert len(paid) == 1
    assert budget.snapshot()['reservation_count'] == 1
    assert result.candidates[0].finish_reason == 'CONTINUATION'
    assert payload == original


def test_actual_sdk_uncertain_billing_retains_reservation_without_retry():
    budget = guard()
    paid = []

    def handler(req):
        if req.url.path.endswith(':countTokens'):
            return httpx.Response(200, json={'totalTokens': 10})
        assert req.url.path.endswith(':generateContent')
        paid.append(req)
        assert budget.snapshot()['reservation_count'] == 1
        raise httpx.ReadTimeout('local fixture: response unavailable', request=req)

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        with genai.Client(api_key='offline-fake-key', enterprise=False,
                          http_options=types.HttpOptions(httpx_client=transport)) as client:
            with pytest.raises(httpx.ReadTimeout):
                guard_genai_client(client, budget).models.generate_content(**request())

    assert len(paid) == budget.snapshot()['reservation_count'] == 1
    reservation, = budget.ledger.document['reservations'].values()
    assert reservation['status'] == 'reserved'
    assert 'actual_gross_usd' not in reservation
    assert Decimal(budget.snapshot()['cash_exposure_usd']) == Decimal(reservation['max_gross_usd']) > 0


@pytest.mark.parametrize('config', [
    {'automatic_continuation': True},
    {'automatic_continuation': False},
    {'http_options': {'retry_options': {'attempts': 2}}},
    {'tools': []},
    {'system_instruction': 'local fixture'},
    {'thinking_config': {'thinking_level': 'minimal'}},
    {'candidate_count': 2},
    {'temperature': .1},
])
def test_rejected_caller_options_still_block_before_any_sdk_request(config):
    budget = guard()
    payload = request(config=config)
    original = copy.deepcopy(payload)

    def forbidden(req):
        pytest.fail('rejected caller options reached the SDK transport')

    with httpx.Client(transport=httpx.MockTransport(forbidden)) as transport:
        with genai.Client(api_key='offline-fake-key', enterprise=False,
                          http_options=types.HttpOptions(httpx_client=transport)) as client:
            with pytest.raises(SpendBlocked):
                guard_genai_client(client, budget).models.generate_content(**payload)

    assert budget.snapshot()['reservation_count'] == 0
    assert payload == original
