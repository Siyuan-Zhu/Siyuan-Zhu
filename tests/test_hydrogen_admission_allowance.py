"""Hydrogen bills a fixed prompt overhead that countTokens cannot see; synthetic data only.

Measured on production (2026-11-22): JSON-mode Hydrogen calls bill 67 prompt tokens above
countTokens, plus 2 per image. The guard admits the counted figure plus a fixed allowance.
"""
from decimal import Decimal
from types import SimpleNamespace

import pytest
from google.genai import types

from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import spend_guard
from src.zeitgeist.spend_guard import (
    EMBEDDING_2_OPERATION, HYDROGEN_MODEL, HYDROGEN_OPERATION, SpendBlocked, guard_genai_client,
    refused_before_dispatch)
from src.zeitgeist.spend_successor import EMBEDDING_001_OPERATION, FLASH_OPERATION, build_successor, successor_ledger
from tests import test_gemini38_contract as base
from tests.test_successor_compatibility import HYDROGEN_VERSION, OLD_PIN, T, observation, refreshed
from tests.test_optional_operations import guard_for
from tests.test_spend_successor import installed

INPUT_RATE, OUTPUT_RATE, OUTPUT_CAP = Decimal('1.50'), Decimal('7.50'), 6000
DIRECTOR_COUNTED = 1057
CRITIC_TEXT_COUNTED = 213


def successor_guard(include_embedding_2=False):
    """The successor as installed: 1091 tokens per image, a 6000-token output cap."""
    ledger0, p0 = installed()
    p0['providers']['gemini']['price_snapshot_sha256'] = OLD_PIN
    ledger0.update(credit_only_providers=p0['credit_only_providers'], automatic_policy_sha256=auto.digest(p0))
    p1 = build_successor(p0, hydrogen_model_version=HYDROGEN_VERSION, hydrogen_tokens_per_image=1091,
                         hydrogen_max_output_tokens=OUTPUT_CAP, hydrogen_thinking_level='high', tariff_sha256=T,
                         include_embedding_2=include_embedding_2)
    ledger1 = dict(successor_ledger(ledger0, p1, auto.digest(p1)), credential_binding_required=True,
                   budget_scope='new_requests_after_cutoff')
    obs = observation(p1, embedding_2_methods=['embedContent'] if include_embedding_2 else None)
    return guard_for(refreshed(ledger1, p1, obs), p1)


def has_image(value):
    if isinstance(value, types.Part):
        return value.inline_data is not None
    if isinstance(value, dict):
        return 'inline_data' in value or any(has_image(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(has_image(item) for item in value)
    return False


def counter(contents_tokens, text_tokens=None):
    """countTokens as measured: the full contents, or the contents with every image emptied."""
    def count(**kwargs):
        return SimpleNamespace(total_tokens=contents_tokens if has_image(kwargs['contents']) or text_tokens is None
                               else text_tokens)
    return count


def reply(prompt_tokens, candidates=700, thoughts=2100):
    return types.GenerateContentResponse(response_id='synthetic-hydrogen-response', model_version=HYDROGEN_VERSION,
        usage_metadata={'prompt_token_count': prompt_tokens, 'candidates_token_count': candidates,
                        'thoughts_token_count': thoughts,
                        'total_token_count': prompt_tokens + candidates + thoughts})


def client(guard, count, prompt_tokens=None, **usage):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return reply(prompt_tokens, **usage)
    raw = SimpleNamespace(models=SimpleNamespace(generate_content=generate, count_tokens=count))
    return guard_genai_client(raw, guard), calls


def director(api, model=HYDROGEN_MODEL):
    return api.models.generate_content(model=model, contents='A synthetic editorial commission',
                                       config={'response_mime_type': 'application/json'})


def critic(api, images):
    return api.models.generate_content(model=HYDROGEN_MODEL,
                                       contents=['A synthetic photo judgement'] + [base.thumbnail() for _ in range(images)],
                                       config={'response_mime_type': 'application/json'})


def only_record(guard):
    records = list(guard.ledger.document['reservations'].values())
    assert len(records) == 1
    return records[0]


def hold(input_tokens):
    return (Decimal(input_tokens) * INPUT_RATE + Decimal(OUTPUT_CAP) * OUTPUT_RATE) / 1000000


def test_the_allowance_is_256_tokens():
    assert spend_guard.HYDROGEN_PROMPT_ALLOWANCE_TOKENS == 256


def test_director_admits_the_counted_figure_plus_the_allowance_and_settles_the_measured_bill():
    guard = successor_guard()
    api, calls = client(guard, counter(DIRECTOR_COUNTED), prompt_tokens=1124)
    assert director(api).model_version == HYDROGEN_VERSION
    assert len(calls) == 1
    record = only_record(guard)
    assert record['request_context']['admitted_input_tokens'] == 1313
    assert Decimal(record['max_gross_usd']) == hold(1313)
    assert record['status'] == 'settled'
    # Booked at the admitted figure, not at the billed 1124.
    assert Decimal(record['actual_gross_usd']) == (Decimal(1313) * INPUT_RATE + Decimal(2800) * OUTPUT_RATE) / 1000000
    assert guard.ledger.document.get('reconciliation_required') is not True


def test_director_reply_billing_above_the_allowance_is_refused_and_freezes():
    guard = successor_guard()
    # Output at the cap puts the billed cost above the hold; a prompt overage inside the hold settles (C2).
    api, calls = client(guard, counter(DIRECTOR_COUNTED), prompt_tokens=1314, candidates=700, thoughts=OUTPUT_CAP - 700)
    assert (Decimal(1314) * INPUT_RATE + Decimal(OUTPUT_CAP) * OUTPUT_RATE) / 1000000 > hold(1313)
    with pytest.raises(SpendBlocked, match='Authenticated Gemini usage exceeds the admitted request policy') as caught:
        director(api)
    assert refused_before_dispatch(caught.value) is False
    assert len(calls) == 1
    record = only_record(guard)
    assert record['request_context']['admitted_input_tokens'] == 1313
    assert guard.ledger.document['reconciliation_required'] is True
    assert record['reconciliation_required'] is True
    assert record['unreconciled_usage']['reason'] == 'Gemini usage exceeds admitted token shape'
    assert record['status'] == 'reserved'
    with pytest.raises(SpendBlocked):
        director(api)
    assert len(calls) == 1


def test_critic_with_nine_images_settles_the_measured_bill():
    guard = successor_guard()
    api, calls = client(guard, counter(10293, CRITIC_TEXT_COUNTED), prompt_tokens=10099)
    critic(api, 9)
    assert len(calls) == 1
    record = only_record(guard)
    assert record['request_context']['admitted_input_tokens'] == 10549
    assert Decimal(record['max_gross_usd']) == hold(10549)
    assert record['status'] == 'settled'


def test_critic_with_one_image_needs_the_allowance_to_settle():
    billed = CRITIC_TEXT_COUNTED + 67 + 1091
    assert billed == 1371
    guard = successor_guard()
    without = guard.counted_genai_admission(
        HYDROGEN_OPERATION,
        guard.bounded_genai_request(HYDROGEN_OPERATION, {'model': HYDROGEN_MODEL,
            'contents': ['A synthetic photo judgement', base.thumbnail()],
            'config': {'response_mime_type': 'application/json'}}),
        counter(1333, CRITIC_TEXT_COUNTED))['admitted_input_tokens'] - 256
    # Without the allowance the admitted figure is the count, which the measured bill exceeds.
    assert without == 1333 < billed
    api, calls = client(guard, counter(1333, CRITIC_TEXT_COUNTED), prompt_tokens=billed)
    critic(api, 1)
    assert len(calls) == 1
    record = only_record(guard)
    assert record['request_context']['admitted_input_tokens'] == 1589
    assert record['status'] == 'settled'


def test_flash_admits_the_counted_figure_unchanged():
    guard = successor_guard()
    bounded = guard.bounded_genai_request(FLASH_OPERATION, {'model': 'gemini-3.8-flash',
        'contents': 'A synthetic editorial commission', 'config': {'response_mime_type': 'application/json'}})
    counted = guard.counted_genai_admission(FLASH_OPERATION, bounded, counter(DIRECTOR_COUNTED))
    assert counted['admitted_input_tokens'] == DIRECTOR_COUNTED


@pytest.mark.parametrize('operation', [EMBEDDING_001_OPERATION, EMBEDDING_2_OPERATION])
def test_embedding_admissions_are_unchanged(operation):
    guard = successor_guard(include_embedding_2=True)
    model = operation.split(':', 1)[1]
    text = 'task: clustering | query: a synthetic headline'
    contents = ([types.Content(parts=[types.Part(text=text)]) for _ in range(3)] if operation == EMBEDDING_2_OPERATION
                else [text] * 3)
    bounded = guard.bounded_genai_request(operation, {'model': model, 'contents': contents,
                                                      'config': {'output_dimensionality': 768}})
    counted = guard.counted_genai_admission(operation, bounded, counter(40))
    assert counted['admitted_input_tokens'] == 120


@pytest.mark.parametrize('counted, admitted', [(65536 - 255, False), (65536 - 256, True)])
def test_the_ceiling_applies_to_the_count_plus_the_allowance(counted, admitted):
    guard = successor_guard()
    api, calls = client(guard, counter(counted), prompt_tokens=counted)
    if admitted:
        director(api)
        assert len(calls) == 1
        assert only_record(guard)['request_context']['admitted_input_tokens'] == 65536
    else:
        with pytest.raises(SpendBlocked, match='Counted input exceeds verified token ceiling') as caught:
            director(api)
        assert refused_before_dispatch(caught.value) is True
        assert calls == []
        assert guard.ledger.document['reservations'] == {}
