"""Hydrogen admission: exact policy, no sampling controls, policy-driven image floor."""
from decimal import Decimal
from types import SimpleNamespace
import pytest
from google.genai import types
from src.zeitgeist import owner_reasons
from src.zeitgeist.spend_guard import HYDROGEN_PROMPT_ALLOWANCE_TOKENS, SpendBlocked, sampling_controls_supported
from tests import test_gemini38_contract as base
from tests.test_spend_provisioning import NOW

MODEL = 'gemini-4-hydrogen'
OP = 'generate_content:' + MODEL


def hydrogen_limits(output_tokens=4000):
    policy = base.limits()
    policy.update(input_token_usd_per_million='1.50', output_token_usd_per_million='7.50',
                  max_output_tokens=output_tokens, thinking_level='high', image_tokens_per_image=1088,
                  allowed_response_model_versions=['synthetic-hydrogen-version'])
    return policy


def guard(now=NOW, mutate=None, cap='0.128304', output_tokens=4000):
    budget = base.guard(now)
    account = budget.ledger.document['providers']['gemini']
    policy = hydrogen_limits(output_tokens)
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
    assert maximum == (Decimal(11 + 9 * 1088 + HYDROGEN_PROMPT_ALLOWANCE_TOKENS) * Decimal('1.50') + Decimal(4000) * Decimal('7.50')) / 1000000
    assert budget.snapshot()['reservation_count'] == 0


def test_hydrogen_full_bound_equals_cap():
    assert (Decimal(65536) * Decimal('1.50') + Decimal(4000) * Decimal('7.50')) / 1000000 == Decimal('0.128304')


def test_hydrogen_full_bound_equals_cap_for_6000():
    assert (Decimal(65536) * Decimal('1.50') + Decimal(6000) * Decimal('7.50')) / 1000000 == Decimal('0.143304')


LADDER = [(4000, '0.128304'), (6000, '0.143304'), (8000, '0.158304'), (10000, '0.173304'), (15000, '0.210804')]


@pytest.mark.parametrize('tokens, cap', LADDER)
def test_hydrogen_ladder_caps_are_derived_and_admitted(tokens, cap):
    assert str((Decimal(65536) * Decimal('1.50') + Decimal(tokens) * Decimal('7.50')) / 1000000) == cap
    bounded = guard(cap=cap, output_tokens=tokens).bounded_genai_request(OP, request())
    assert bounded['config']['max_output_tokens'] == tokens


@pytest.mark.parametrize('index', range(len(LADDER)))
def test_hydrogen_neighbouring_caps_are_refused(index):
    tokens = LADDER[index][0]
    for other in (index - 1, index + 1):
        if 0 <= other < len(LADDER):
            with pytest.raises(SpendBlocked, match='Unverified Gemini 4 Hydrogen pricing or model policy'):
                guard(cap=LADDER[other][1], output_tokens=tokens).bounded_genai_request(OP, request())


def test_hydrogen_6000_tokens_admitted_with_matching_cap():
    budget = guard(cap='0.143304', output_tokens=6000)
    bounded = budget.bounded_genai_request(OP, request(config={'response_mime_type': 'application/json',
                                                               'max_output_tokens': 6000}))
    assert bounded['config']['max_output_tokens'] == 6000
    calls = []
    def count(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(total_tokens=1 if len(calls) == 1 else 11)
    maximum, _ = budget.counted_genai_cap(OP, bounded, count)
    assert maximum == (Decimal(1 + HYDROGEN_PROMPT_ALLOWANCE_TOKENS) * Decimal('1.50') + Decimal(6000) * Decimal('7.50')) / 1000000
    assert budget.bounded_genai_request(OP, request())['config']['max_output_tokens'] == 6000


def test_hydrogen_request_above_policy_output_is_refused():
    with pytest.raises(SpendBlocked, match='monetary bound'):
        guard().bounded_genai_request(OP, request(config={'max_output_tokens': 6000}))


@pytest.mark.parametrize('tokens', [1, 3999, 4001, 5000, 5999, 6001, 7000, 10001, 12000, 15001, 65536])
def test_hydrogen_output_must_be_exactly_4000_or_6000_even_with_a_matching_cap(tokens):
    cap = str((Decimal(65536) * Decimal('1.50') + Decimal(tokens) * Decimal('7.50')) / 1000000)
    with pytest.raises(SpendBlocked, match='Unverified Gemini 4 Hydrogen pricing or model policy'):
        guard(cap=cap, output_tokens=tokens).bounded_genai_request(OP, request())


@pytest.mark.parametrize('cap, tokens', [
    ('0.128304', 6000), ('0.143304', 4000), ('0.128303', 4000), ('0.143305', 6000),
    ('0.2', 4000), ('0', 4000), ('not-money', 4000), (None, 4000), (None, 6000)])
def test_hydrogen_cap_must_match_output_tokens(cap, tokens):
    with pytest.raises(SpendBlocked, match='Unverified Gemini 4 Hydrogen pricing or model policy'):
        guard(cap=cap, output_tokens=tokens).bounded_genai_request(OP, request())
    budget = guard(cap=cap, output_tokens=tokens)
    with pytest.raises(SpendBlocked, match='Unverified Gemini 4 Hydrogen pricing or model policy'):
        budget.counted_genai_cap(OP, {'contents': 'x', 'config': {'http_options': {}}}, lambda **k: None)


def _delete_versions(policy):
    del policy['allowed_response_model_versions']


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(thinking_level='minimal'),
    lambda p: p.update(input_token_usd_per_million='1.49'),
    lambda p: p.update(output_token_usd_per_million='7.51'),
    lambda p: p.update(max_output_tokens=4001),
    lambda p: p.update(max_output_tokens=5999),
    lambda p: p.update(max_output_tokens=6001),
    lambda p: p.update(max_output_tokens=5000),
    lambda p: p.update(max_output_tokens=7000),
    lambda p: p.update(max_output_tokens=10001),
    lambda p: p.update(max_output_tokens=12000),
    lambda p: p.update(max_output_tokens=15001),
    lambda p: p.update(max_output_tokens=3999),
    lambda p: p.update(max_output_tokens=True),
    lambda p: p.update(max_output_tokens='4000'),
    lambda p: p.update(max_output_tokens=4000.0),
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
