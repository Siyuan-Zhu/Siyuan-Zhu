"""Neon admission: exact policy at 3.00/15.00, no sampling controls, policy-driven image floor."""
from decimal import Decimal
from types import SimpleNamespace
import pytest
from google.genai import types
from src.zeitgeist import owner_reasons
from src.zeitgeist import spend_guard
from src.zeitgeist.spend_guard import (
    NEON_MODEL, NEON_OPERATION, NEON_PROMPT_ALLOWANCE_TOKENS, SpendBlocked, refuse_disabled_operation,
    sampling_controls_supported)
from tests import test_gemini38_contract as base
from tests.test_spend_provisioning import NOW

MODEL = 'gemini-4-neon'
OP = 'generate_content:' + MODEL
REASON = 'Unverified Gemini 4 Neon pricing or model policy'


def neon_cap(tokens):
    return (Decimal(65536) * Decimal('3.00') + Decimal(tokens) * Decimal('15.00')) / 1000000


def neon_limits(output_tokens=6000):
    policy = base.limits()
    policy.update(input_token_usd_per_million='3.00', output_token_usd_per_million='15.00',
                  max_output_tokens=output_tokens, thinking_level='medium', image_tokens_per_image=1091,
                  allowed_response_model_versions=['synthetic-neon-version'])
    return policy


def guard(now=NOW, mutate=None, cap='0.286608', output_tokens=6000):
    budget = base.guard(now)
    account = budget.ledger.document['providers']['gemini']
    policy = neon_limits(output_tokens)
    if mutate:
        mutate(policy)
    account['request_limits'][OP] = policy
    if cap is None:
        account['operation_caps_usd'].pop(OP, None)
    else:
        account['operation_caps_usd'][OP] = cap
    return budget


def request(contents='A synthetic editorial commission', config=None):
    return {'model': MODEL, 'contents': contents, 'config': config or {'response_mime_type': 'application/json'}}


def counter(first, second=None):
    calls = []
    def count(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(total_tokens=first if len(calls) == 1 or second is None else second)
    return count


def test_constants_name_the_neon_operation():
    assert (NEON_MODEL, NEON_OPERATION) == (MODEL, OP)
    assert NEON_PROMPT_ALLOWANCE_TOKENS == 256


def test_neon_request_has_policy_thinking_medium_and_no_sampling():
    bounded = guard().bounded_genai_request(OP, request())
    assert bounded['config']['thinking_config'] == {'thinking_level': 'medium'}
    assert bounded['config']['max_output_tokens'] == 6000
    assert bounded['config']['media_resolution'] == 'MEDIA_RESOLUTION_HIGH'
    assert 'temperature' not in bounded['config'] and 'candidate_count' not in bounded['config']
    types.GenerateContentConfig(**bounded['config'])


@pytest.mark.parametrize('config', [{'temperature': .1}, {'candidate_count': 1}])
def test_neon_rejects_sampling_controls(config):
    budget = guard()
    with pytest.raises(SpendBlocked) as caught:
        budget.bounded_genai_request(OP, request(config=config))
    assert str(caught.value) == 'Gemini 4 Neon sampling controls are not supported'
    assert caught.value.public_diagnostics['code'] == 'unsupported-sampling-controls'
    assert budget.snapshot()['reservation_count'] == 0


def test_sampling_controls_are_not_supported_on_neon():
    assert sampling_controls_supported('gemini-4-neon') is False
    assert sampling_controls_supported('gemini-4-hydrogen') is False
    assert sampling_controls_supported('gemini-2.5-flash') is True


def test_neon_text_admission_adds_the_prompt_allowance_at_neon_rates():
    budget = guard()
    bounded = budget.bounded_genai_request(OP, request())
    counted = budget.counted_genai_admission(OP, bounded, counter(1060))
    assert counted['admitted_input_tokens'] == 1060 + 256
    assert Decimal(counted['maximum']) == (Decimal(1316) * Decimal('3.00') + Decimal(6000) * Decimal('15.00')) / 1000000
    assert budget.snapshot()['reservation_count'] == 0


def test_neon_nine_images_use_policy_tokens_per_image():
    budget = guard()
    bounded = budget.bounded_genai_request(OP, request(['Synthetic batch'] + [base.thumbnail() for _ in range(9)]))
    maximum, _ = budget.counted_genai_cap(OP, bounded, counter(1, 216))
    assert maximum == (Decimal(216 + 9 * 1091 + 256) * Decimal('3.00') + Decimal(6000) * Decimal('15.00')) / 1000000


def test_neon_images_counted_above_the_floor_use_the_count():
    budget = guard()
    bounded = budget.bounded_genai_request(OP, request(['Synthetic batch'] + [base.thumbnail() for _ in range(9)]))
    counted = budget.counted_genai_admission(OP, bounded, counter(20000, 216))
    assert counted['admitted_input_tokens'] == 20000 + 256


LADDER = [(4000, '0.256608'), (6000, '0.286608'), (8000, '0.316608'), (10000, '0.346608'), (15000, '0.421608')]


@pytest.mark.parametrize('tokens, cap', LADDER)
def test_neon_ladder_caps_are_derived_and_admitted(tokens, cap):
    assert str(neon_cap(tokens)) == cap
    bounded = guard(cap=cap, output_tokens=tokens).bounded_genai_request(OP, request())
    assert bounded['config']['max_output_tokens'] == tokens


@pytest.mark.parametrize('index', range(len(LADDER)))
def test_neon_neighbouring_caps_are_refused(index):
    tokens = LADDER[index][0]
    for other in (index - 1, index + 1):
        if 0 <= other < len(LADDER):
            with pytest.raises(SpendBlocked, match=REASON):
                guard(cap=LADDER[other][1], output_tokens=tokens).bounded_genai_request(OP, request())


def test_neon_at_the_hydrogen_cap_is_refused():
    with pytest.raises(SpendBlocked, match=REASON):
        guard(cap='0.143304').bounded_genai_request(OP, request())


def test_neon_request_above_policy_output_is_refused():
    with pytest.raises(SpendBlocked, match='monetary bound'):
        guard(cap='0.256608', output_tokens=4000).bounded_genai_request(OP, request(config={'max_output_tokens': 6000}))


def _delete_versions(policy):
    del policy['allowed_response_model_versions']


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(input_token_usd_per_million='1.50'),
    lambda p: p.update(output_token_usd_per_million='7.50'),
    lambda p: p.update(input_token_usd_per_million='2.99'),
    lambda p: p.update(output_token_usd_per_million='15.01'),
    lambda p: p.update(thinking_level='minimal'),
    lambda p: p.update(thinking_level=None),
    lambda p: p.update(max_output_tokens=5999),
    lambda p: p.update(max_output_tokens='6000'),
    lambda p: p.update(max_input_tokens=65537),
    lambda p: p.update(max_input_bytes=2000001),
    lambda p: p.update(max_input_images=10),
    lambda p: p.update(image_tokens_per_image=0),
    lambda p: p.update(image_tokens_per_image=4097),
    lambda p: p.update(image_tokens_per_image='1091'),
    lambda p: p.update(pricing_valid_until='2027-01-01T00:00:00Z'),
    lambda p: p.update(allowed_response_model_versions=[]),
    _delete_versions,
    lambda p: p.update(media_resolution='MEDIA_RESOLUTION_LOW'),
])
def test_neon_rejects_unverified_policy(mutation):
    budget = guard(mutate=mutation)
    with pytest.raises(SpendBlocked) as caught:
        budget.bounded_genai_request(OP, request())
    assert 'Hydrogen' not in str(caught.value)
    assert budget.snapshot()['reservation_count'] == 0


def test_neon_missing_cap_is_refused_with_the_neon_reason():
    with pytest.raises(SpendBlocked, match=REASON) as caught:
        guard(cap=None).bounded_genai_request(OP, request())
    assert caught.value.public_diagnostics['code'] == 'unverified-neon-policy'


def test_neon_pricing_must_cover_request_duration():
    now = NOW.replace(month=12, day=31, hour=23, minute=58)
    with pytest.raises(SpendBlocked) as caught:
        guard(now).bounded_genai_request(OP, request())
    assert str(caught.value) == 'Gemini 4 Neon pricing expires before bounded request completion'


def test_counted_admission_validates_the_neon_policy_again():
    budget = guard()
    bounded = budget.bounded_genai_request(OP, request())
    budget.ledger.document['providers']['gemini']['request_limits'][OP]['input_token_usd_per_million'] = '1.50'
    with pytest.raises(SpendBlocked, match=REASON):
        budget.counted_genai_admission(OP, bounded, counter(1060))


def test_disabled_neon_is_refused_with_its_own_reason_before_dispatch():
    with pytest.raises(SpendBlocked, match=REASON) as caught:
        refuse_disabled_operation({'disabled_operations': [OP]}, OP)
    assert caught.value.public_diagnostics['code'] == 'unverified-neon-policy'
    budget = guard()
    budget.ledger.document['providers']['gemini']['disabled_operations'] = [OP]
    with pytest.raises(SpendBlocked, match=REASON):
        budget.bounded_genai_request(OP, request())


def test_neon_reason_codes_resolve():
    reasons = spend_guard.SpendBlocked._PUBLIC_REASONS
    assert reasons[REASON] == 'unverified-neon-policy'
    assert reasons['Gemini 4 Neon pricing expires before bounded request completion'] == 'pricing-validity-bound'
    assert reasons['Gemini 4 Neon sampling controls are not supported'] == 'unsupported-sampling-controls'
    assert owner_reasons.resolve(stage='spend_maintenance', code='unverified-neon-policy')[0] == 'unverified-neon-policy'
