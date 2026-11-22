"""Hydrogen admission: exact policy, no sampling controls, policy-driven image floor."""
from decimal import Decimal
from types import SimpleNamespace
import pytest
from google.genai import types
from src.zeitgeist import owner_reasons
from src.zeitgeist.spend_guard import SpendBlocked, sampling_controls_supported
from tests import test_gemini38_contract as base
from tests.test_spend_provisioning import NOW

MODEL = 'gemini-4-hydrogen'
OP = 'generate_content:' + MODEL


def hydrogen_limits():
    policy = base.limits()
    policy.update(input_token_usd_per_million='1.50', output_token_usd_per_million='7.50',
                  max_output_tokens=4000, thinking_level='high', image_tokens_per_image=1088,
                  allowed_response_model_versions=['synthetic-hydrogen-version'])
    return policy


def guard(now=NOW, mutate=None):
    budget = base.guard(now)
    account = budget.ledger.document['providers']['gemini']
    policy = hydrogen_limits()
    if mutate:
        mutate(policy)
    account['request_limits'][OP] = policy
    account['operation_caps_usd'][OP] = '0.128304'
    return budget


def request(contents='A synthetic editorial commission', config=None):
    return {'model': MODEL, 'contents': contents, 'config': config or {'response_mime_type': 'application/json'}}


def test_hydrogen_request_has_policy_thinking_and_no_sampling():
    bounded = guard().bounded_genai_request(OP, request())
    assert bounded['config']['thinking_config'] == {'thinking_level': 'high'}
    assert bounded['config']['max_output_tokens'] == 4000
    assert bounded['config']['media_resolution'] == 'MEDIA_RESOLUTION_HIGH'
    assert 'temperature' not in bounded['config'] and 'candidate_count' not in bounded['config']
    types.GenerateContentConfig(**bounded['config'])


@pytest.mark.parametrize('config', [{'temperature': .1}, {'candidate_count': 1}])
def test_hydrogen_rejects_sampling_controls(config):
    budget = guard()
    with pytest.raises(SpendBlocked, match='sampling controls'):
        budget.bounded_genai_request(OP, request(config=config))
    assert budget.snapshot()['reservation_count'] == 0


def test_hydrogen_nine_images_use_policy_tokens_per_image():
    budget = guard()
    bounded = budget.bounded_genai_request(OP, request(['Synthetic batch'] + [base.thumbnail() for _ in range(9)]))
    calls = []
    def count(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(total_tokens=1 if len(calls) == 1 else 11)
    maximum, _ = budget.counted_genai_cap(OP, bounded, count)
    assert maximum == (Decimal(11 + 9 * 1088) * Decimal('1.50') + Decimal(4000) * Decimal('7.50')) / 1000000
    assert budget.snapshot()['reservation_count'] == 0


def test_hydrogen_full_bound_equals_cap():
    assert (Decimal(65536) * Decimal('1.50') + Decimal(4000) * Decimal('7.50')) / 1000000 == Decimal('0.128304')


def _delete_versions(policy):
    del policy['allowed_response_model_versions']


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(thinking_level='minimal'),
    lambda p: p.update(input_token_usd_per_million='1.49'),
    lambda p: p.update(output_token_usd_per_million='7.51'),
    lambda p: p.update(max_output_tokens=4001),
    lambda p: p.update(max_input_tokens=65537),
    lambda p: p.update(max_input_images=10),
    lambda p: p.update(image_tokens_per_image=0),
    lambda p: p.update(image_tokens_per_image=4097),
    lambda p: p.update(image_tokens_per_image='1088'),
    lambda p: p.update(image_tokens_per_image=True),
    lambda p: p.update(pricing_valid_until='2027-01-01T00:00:00Z'),
    lambda p: p.update(allowed_response_model_versions=[]),
    _delete_versions,
    lambda p: p.update(media_resolution='MEDIA_RESOLUTION_LOW'),
])
def test_hydrogen_rejects_unverified_policy(mutation):
    budget = guard(mutate=mutation)
    with pytest.raises(SpendBlocked):
        budget.bounded_genai_request(OP, request())


def test_hydrogen_pricing_must_cover_request_duration():
    now = NOW.replace(month=12, day=31, hour=23, minute=58)
    with pytest.raises(SpendBlocked, match='pricing'):
        guard(now).bounded_genai_request(OP, request())


def test_hydrogen_xhigh_never_raises_a_non_spend_error():
    budget = guard(mutate=lambda p: p.update(thinking_level='xhigh'))
    try:
        bounded = budget.bounded_genai_request(OP, request())
    except SpendBlocked as error:
        assert str(error) == 'Verified model-specific thinking level required'
    else:
        types.GenerateContentConfig(**bounded['config'])


def test_sampling_controls_supported():
    assert sampling_controls_supported('gemini-3.8-flash') is False
    assert sampling_controls_supported('gemini-4-hydrogen') is False
    assert sampling_controls_supported('gemini-2.5-flash') is True


def test_new_reason_codes_resolve():
    assert owner_reasons.resolve(stage='spend_maintenance', code='unverified-hydrogen-policy')[0] == 'unverified-hydrogen-policy'
