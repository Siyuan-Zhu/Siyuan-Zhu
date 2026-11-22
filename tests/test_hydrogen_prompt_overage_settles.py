"""A Hydrogen reply whose prompt tokens exceed the admitted count settles when its cost is inside the hold.

Synthetic data only. The first private run (2026-11-22) billed 787 prompt tokens against 720 admitted,
at a cost of half its hold, and froze the month. The hold bounds money; only a cost above it freezes.
"""
from decimal import Decimal
from types import SimpleNamespace

import pytest
from google.genai import types

from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import HYDROGEN_MODEL, HYDROGEN_OPERATION, SpendBlocked, guard_genai_client
from src.zeitgeist.spend_successor import FLASH_OPERATION
from tests.test_hydrogen_admission_allowance import INPUT_RATE, OUTPUT_CAP, OUTPUT_RATE, counter, successor_guard
from tests.test_successor_compatibility import HYDROGEN_VERSION

ADMITTED = 720
COUNTED = ADMITTED - 256  # the guard adds the Hydrogen allowance to the count
HOLD = Decimal('0.04608')
FREEZE = 'Gemini usage exceeds admitted token shape'
# The receipts the code before this change returned for the synthetic reply below, pinned whole.
EVIDENCE = {720: 'b2fc67f66b6c4d7cd58cedb507b4193b7010d48b519afd52dcad5115fcd5304e',
            719: 'ffb7ca4dee3932b12e29c85bc02d29657e8cd9953f2d3286d9bb379aefc20643'}


def response(prompt, candidates, thoughts, *, version=HYDROGEN_VERSION, ident='synthetic-response-1', modality=None):
    usage = {'prompt_token_count': prompt, 'candidates_token_count': candidates, 'thoughts_token_count': thoughts,
             'total_token_count': prompt + candidates + thoughts}
    if modality:
        usage['candidates_tokens_details'] = [{'modality': modality, 'token_count': candidates}]
    return types.GenerateContentResponse(response_id=ident, model_version=version, usage_metadata=usage)


def client(guard, replies):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return replies[len(calls) - 1]
    raw = SimpleNamespace(models=SimpleNamespace(generate_content=generate, count_tokens=counter(COUNTED)))
    return guard_genai_client(raw, guard), calls


def director(api, model=HYDROGEN_MODEL):
    return api.models.generate_content(model=model, contents='A synthetic editorial commission',
                                       config={'response_mime_type': 'application/json'})


def records(guard):
    return list(guard.ledger.document['reservations'].values())


def assert_frozen_as_today(guard, record):
    assert guard.ledger.document['reconciliation_required'] is True
    assert record['reconciliation_required'] is True
    assert record['unreconciled_usage']['reason'] == FREEZE
    assert record['status'] == 'reserved'
    assert 'usage_receipt' not in record and 'actual_gross_usd' not in record


def flash_version(guard):
    limits = guard.automatic_policy['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]
    return limits['allowed_response_model_versions'][0]


def test_the_frozen_call_settles_at_its_billed_cost_and_the_next_call_is_admitted():
    guard = successor_guard()
    api, calls = client(guard, [response(787, 778, 2096),
                                response(700, 500, 1500, ident='synthetic-response-2')])
    director(api)
    record = records(guard)[0]
    assert record['request_context']['admitted_input_tokens'] == ADMITTED
    assert record['request_context']['max_output_tokens'] == OUTPUT_CAP
    assert Decimal(record['max_gross_usd']) == HOLD
    assert record['status'] == 'settled'
    assert Decimal(record['actual_gross_usd']) == Decimal('0.0227355')
    assert Decimal(record['usage_receipt']['accounted_gross_usd']) == Decimal('0.0227355')
    assert record['usage_receipt']['prompt_tokens_over_admitted'] == 67
    assert guard.ledger.document.get('reconciliation_required') is not True
    assert 'reconciliation_required' not in record and 'unreconciled_usage' not in record
    director(api)
    assert len(calls) == 2
    assert [r['status'] for r in records(guard)] == ['settled', 'settled']


def test_a_prompt_overage_whose_cost_is_above_the_hold_freezes_as_today():
    guard = successor_guard()
    api, calls = client(guard, [response(787, 1000, OUTPUT_CAP - 1000)])
    assert (Decimal(787) * INPUT_RATE + Decimal(OUTPUT_CAP) * OUTPUT_RATE) / 1000000 > HOLD
    with pytest.raises(SpendBlocked, match='Authenticated Gemini usage exceeds the admitted request policy'):
        director(api)
    assert len(calls) == 1
    assert_frozen_as_today(guard, records(guard)[0])
    with pytest.raises(SpendBlocked):
        director(api)
    assert len(calls) == 1


def test_output_above_the_cap_freezes_as_today_even_when_the_cost_is_under_the_hold():
    guard = successor_guard()
    api, calls = client(guard, [response(700, 1000, OUTPUT_CAP - 999)])
    assert (Decimal(700) * INPUT_RATE + Decimal(OUTPUT_CAP + 1) * OUTPUT_RATE) / 1000000 < HOLD
    with pytest.raises(SpendBlocked, match='Authenticated Gemini usage exceeds the admitted request policy'):
        director(api)
    assert_frozen_as_today(guard, records(guard)[0])


def test_a_prompt_at_or_below_the_admitted_figure_returns_the_unchanged_receipt():
    policy = {'providers': {'gemini': {'account': {'request_limits': {HYDROGEN_OPERATION: {
        'allowed_response_model_versions': [HYDROGEN_VERSION],
        'input_token_usd_per_million': '1.50', 'output_token_usd_per_million': '7.50'}}}}}}
    context = {'admitted_input_tokens': ADMITTED, 'max_output_tokens': OUTPUT_CAP,
               'input_token_usd_per_million': '1.50', 'output_token_usd_per_million': '7.50'}
    reservation = {'operation': HYDROGEN_OPERATION, 'request_context': context, 'max_gross_usd': str(HOLD)}
    for prompt in (ADMITTED, ADMITTED - 1):
        reply = {'responseId': 'synthetic-response-1', 'modelVersion': HYDROGEN_VERSION,
                 'usageMetadata': {'promptTokenCount': prompt, 'candidatesTokenCount': 778,
                                   'thoughtsTokenCount': 2096, 'totalTokenCount': prompt + 2874}}
        receipt = auto.gemini_usage_receipt('r' * 32, reservation, reply, credential_fingerprint='f' * 64,
            observed_at='2026-11-22T11:23:00+00:00', request_sha256='a' * 64, policy=policy)
        assert receipt == {
            'schema_version': 1, 'accounting_policy': 'provider_usage_v1', 'reservation_id': 'r' * 32,
            'provider': 'gemini', 'operation': HYDROGEN_OPERATION, 'credential_fingerprint': 'f' * 64,
            'provider_request_id': 'synthetic-response-1', 'request_sha256': 'a' * 64,
            'observed_at': '2026-11-22T11:23:00+00:00', 'accounted_gross_usd': '0.022635',
            'accounting_basis': 'conservative_response_usage',
            'evidence_sha256': EVIDENCE[prompt]}


def test_the_same_overage_on_flash_still_freezes():
    guard = successor_guard()
    version = flash_version(guard)
    api, calls = client(guard, [response(ADMITTED + 67, 778, 2096, version=version)])
    with pytest.raises(SpendBlocked, match='Authenticated Gemini usage exceeds the admitted request policy'):
        director(api, model='gemini-3.8-flash')
    record = records(guard)[0]
    assert record['operation'] == FLASH_OPERATION
    assert record['request_context']['admitted_input_tokens'] == COUNTED
    assert_frozen_as_today(guard, record)


def test_a_non_text_reply_with_a_prompt_overage_produces_no_receipt():
    guard = successor_guard()
    api, calls = client(guard, [response(787, 778, 2096, modality='IMAGE')])
    director(api)
    record = records(guard)[0]
    assert record['status'] == 'reserved'
    assert 'usage_receipt' not in record and 'actual_gross_usd' not in record
    assert 'reconciliation_required' not in record and 'unreconciled_usage' not in record
    assert guard.ledger.document.get('reconciliation_required') is not True


def test_a_re_observation_of_the_settled_reply_adds_no_flags():
    guard = successor_guard()
    reply = response(787, 778, 2096)
    api, calls = client(guard, [reply])
    director(api)
    ledger = guard.ledger.document
    ident, record = next(iter(ledger['reservations'].items()))
    again = auto.gemini_usage_receipt(ident, record, reply.model_dump(by_alias=True, exclude_none=True),
        credential_fingerprint=record['usage_receipt']['credential_fingerprint'],
        observed_at=guard.now().isoformat(), request_sha256=record['request_sha256'], policy=guard.automatic_policy)
    assert again['prompt_tokens_over_admitted'] == 67
    candidate = auto.admit_usage_receipt(ledger, guard.automatic_policy, again,
                                         expected_policy_sha=guard.expected_policy_sha, now=guard.now())
    settled = candidate['reservations'][ident]
    assert candidate.get('reconciliation_required') is not True
    assert not {'reconciliation_required', 'reconciliation_reasons', 'usage_receipt_conflict'} & set(settled)
    assert settled['status'] == 'settled' and settled['actual_gross_usd'] == record['actual_gross_usd']
    assert settled['billing_reference'] == record['billing_reference']
