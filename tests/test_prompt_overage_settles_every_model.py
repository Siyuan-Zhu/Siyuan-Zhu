"""A Flash reply whose prompt tokens exceed the admitted count settles when its cost is inside the hold.

Synthetic data only. The rule that settles a Hydrogen prompt overage inside its hold covers every priced
generate_content model (owner, 2026-11-22). Every other freeze stays: a cost above the hold, output above
the cap, a reply that is not text-only or has no responseId, a billed prompt above the input ceiling or an
operation without an integer ceiling, and an unpinned model version.
"""
from decimal import Decimal

import pytest

from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import SpendBlocked
from src.zeitgeist.spend_successor import FLASH_OPERATION
from tests.test_hydrogen_admission_allowance import successor_guard
from tests.test_hydrogen_prompt_overage_settles import (
    FREEZE, assert_frozen_as_today, client, director, flash_version, records, response)

FLASH_MODEL = 'gemini-3.8-flash'
COUNTED = 464
REFUSED = 'Authenticated Gemini usage exceeds the admitted request policy'


def flash(api):
    return director(api, model=FLASH_MODEL)


def cost(record, prompt, output):
    context = record['request_context']
    return (Decimal(prompt) * Decimal(context['input_token_usd_per_million'])
            + Decimal(output) * Decimal(context['output_token_usd_per_million'])) / 1000000


def freezes_and_refuses_the_next_call(guard, api, calls, reason=FREEZE):
    with pytest.raises(SpendBlocked, match=REFUSED):
        flash(api)
    assert len(calls) == 1
    record = records(guard)[0]
    assert record['operation'] == FLASH_OPERATION
    if reason == FREEZE:
        assert_frozen_as_today(guard, record)
    else:
        assert guard.ledger.document['reconciliation_required'] is True
        assert record['reconciliation_required'] is True
        assert record['unreconciled_usage']['reason'] == reason
        assert record['status'] == 'reserved'
        assert 'usage_receipt' not in record and 'actual_gross_usd' not in record
    with pytest.raises(SpendBlocked):
        flash(api)
    assert len(calls) == 1
    return record


def test_a_flash_reply_one_token_above_the_admitted_count_settles_at_its_billed_cost():
    guard = successor_guard()
    version = flash_version(guard)
    api, calls = client(guard, [response(COUNTED + 1, 300, 200, version=version),
                                response(COUNTED, 300, 200, version=version, ident='synthetic-response-2')])
    flash(api)
    record = records(guard)[0]
    assert record['operation'] == FLASH_OPERATION
    assert record['request_context']['admitted_input_tokens'] == COUNTED
    assert cost(record, COUNTED + 1, 500) <= Decimal(record['max_gross_usd'])
    assert record['status'] == 'settled'
    # Booked at the billed prompt tokens, which are above the admitted count.
    assert Decimal(record['actual_gross_usd']) == cost(record, COUNTED + 1, 500)
    assert record['usage_receipt']['prompt_tokens_over_admitted'] == 1
    assert guard.ledger.document.get('reconciliation_required') is not True
    assert not {'reconciliation_required', 'reconciliation_reasons', 'unreconciled_usage',
                'overrun_tolerated'} & set(record)
    flash(api)
    assert len(calls) == 2
    assert [r['status'] for r in records(guard)] == ['settled', 'settled']


def test_the_flash_receipt_carries_the_overage_and_the_billed_cost():
    guard = successor_guard()
    version = flash_version(guard)
    api, calls = client(guard, [response(COUNTED, 300, 200, version=version)])
    flash(api)
    record = records(guard)[0]
    reservation = {key: record[key] for key in ('operation', 'request_context', 'max_gross_usd')}
    reply = {'responseId': 'synthetic-response-9', 'modelVersion': version,
             'usageMetadata': {'promptTokenCount': COUNTED + 1, 'candidatesTokenCount': 300,
                               'thoughtsTokenCount': 200, 'totalTokenCount': COUNTED + 501,
                               'candidatesTokensDetails': [{'modality': 'TEXT', 'tokenCount': 300}]}}
    receipt = auto.gemini_usage_receipt('r' * 32, reservation, reply, credential_fingerprint='f' * 64,
        observed_at='2026-11-22T11:23:00+00:00', request_sha256='a' * 64, policy=guard.automatic_policy)
    assert receipt['operation'] == FLASH_OPERATION
    assert receipt['prompt_tokens_over_admitted'] == 1
    assert Decimal(receipt['accounted_gross_usd']) == cost(record, COUNTED + 1, 500)


def test_a_flash_prompt_overage_whose_cost_is_above_the_hold_freezes():
    guard = successor_guard()
    version = flash_version(guard)
    cap = guard.automatic_policy['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]['max_output_tokens']
    api, calls = client(guard, [response(COUNTED + 1, 100, cap - 100, version=version),
                                response(COUNTED, 10, 10, version=version, ident='synthetic-response-2')])
    record = freezes_and_refuses_the_next_call(guard, api, calls)
    assert record['request_context']['max_output_tokens'] == cap
    assert cost(record, COUNTED + 1, cap) > Decimal(record['max_gross_usd'])


def test_a_flash_reply_that_is_not_text_only_with_an_overage_freezes():
    guard = successor_guard()
    version = flash_version(guard)
    api, calls = client(guard, [response(COUNTED + 1, 300, 200, version=version, modality='IMAGE'),
                                response(COUNTED, 10, 10, version=version)])
    record = freezes_and_refuses_the_next_call(guard, api, calls)
    assert record['provider_request_id'] == 'synthetic-response-1'


def test_a_flash_reply_without_a_response_id_with_an_overage_freezes():
    guard = successor_guard()
    version = flash_version(guard)
    api, calls = client(guard, [response(COUNTED + 1, 300, 200, version=version, ident=None),
                                response(COUNTED, 10, 10, version=version)])
    record = freezes_and_refuses_the_next_call(guard, api, calls)
    assert 'provider_request_id' not in record and 'submitted_at' not in record


def test_a_flash_billed_prompt_above_the_input_ceiling_freezes_even_inside_the_hold():
    guard = successor_guard()
    version = flash_version(guard)
    ceiling = guard.automatic_policy['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]['max_input_tokens']
    admitted = 60000
    assert ceiling + 1 > admitted
    api, calls = client(guard, [response(ceiling + 1, 5, 5, version=version),
                                response(COUNTED, 10, 10, version=version)], counted=admitted)
    record = freezes_and_refuses_the_next_call(guard, api, calls)
    assert record['request_context']['admitted_input_tokens'] == admitted
    assert cost(record, ceiling + 1, 10) <= Decimal(record['max_gross_usd'])


def flash_policy(**ceiling):
    limits = {'allowed_response_model_versions': ['synthetic-flash-version'],
              'input_token_usd_per_million': '0.75', 'output_token_usd_per_million': '3.75'}
    limits.update(ceiling)
    return {'providers': {'gemini': {'account': {'request_limits': {FLASH_OPERATION: limits}}}}}


def flash_overage_receipt(policy):
    context = {'admitted_input_tokens': COUNTED, 'max_output_tokens': 1500,
               'input_token_usd_per_million': '0.75', 'output_token_usd_per_million': '3.75'}
    reservation = {'operation': FLASH_OPERATION, 'request_context': context, 'max_gross_usd': '0.005973'}
    reply = {'responseId': 'synthetic-response-1', 'modelVersion': 'synthetic-flash-version',
             'usageMetadata': {'promptTokenCount': COUNTED + 1, 'candidatesTokenCount': 300,
                               'thoughtsTokenCount': 200, 'totalTokenCount': COUNTED + 501}}
    return auto.gemini_usage_receipt('r' * 32, reservation, reply, credential_fingerprint='f' * 64,
        observed_at='2026-11-22T11:23:00+00:00', request_sha256='a' * 64, policy=policy)


def test_the_same_flash_overage_settles_when_the_operation_has_an_integer_input_ceiling():
    assert flash_overage_receipt(flash_policy(max_input_tokens=65536))['prompt_tokens_over_admitted'] == 1


@pytest.mark.parametrize('ceiling', [{}, {'max_input_tokens': None}, {'max_input_tokens': '65536'},
                                     {'max_input_tokens': 65536.0}, {'max_input_tokens': True}])
def test_a_flash_overage_freezes_when_the_operation_has_no_integer_input_ceiling(ceiling):
    with pytest.raises(SpendBlocked, match=FREEZE):
        flash_overage_receipt(flash_policy(**ceiling))


@pytest.mark.parametrize('prompt', [COUNTED, COUNTED + 1])
def test_an_unpinned_flash_model_version_freezes_with_or_without_an_overage(prompt):
    guard = successor_guard()
    api, calls = client(guard, [response(prompt, 300, 200, version='synthetic-unpinned-version'),
                                response(COUNTED, 10, 10, version=flash_version(guard))])
    freezes_and_refuses_the_next_call(guard, api, calls, reason='Gemini response model version not pinned')
