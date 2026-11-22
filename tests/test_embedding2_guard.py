"""Embedding-2 admission: exact text-only policy, 768 dimensions, per-text token bound."""
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from src.zeitgeist import owner_reasons, spend_guard
from src.zeitgeist.spend_guard import SpendBlocked, guard_genai_client
from tests import test_gemini38_contract as base
from tests.test_spend_provisioning import NOW

MODEL = 'gemini-embedding-2'
OP = 'embed_content:' + MODEL
REASON = 'Unverified Gemini Embedding 2 pricing or model policy'


def embedding2_limits():
    return {'pricing_reference': 'https://ai.google.dev/gemini-api/docs/pricing',
            'count_tokens_reference': 'https://ai.google.dev/gemini-api/docs/billing#is-gettokens-billed',
            'max_input_bytes': 2000000, 'max_input_tokens': 204800,
            'input_token_usd_per_million': '0.20', 'output_token_usd_per_million': '0',
            'output_dimensionality': 768}


def guard(mutate=None, cap='0.04096'):
    budget = base.guard(NOW)
    account = budget.ledger.document['providers']['gemini']
    policy = embedding2_limits()
    if mutate:
        mutate(policy)
    account['request_limits'][OP] = policy
    account['operation_caps_usd'][OP] = cap
    return budget


def request(contents=('synthetic text one', 'synthetic text two'), config=None):
    return {'model': MODEL, 'contents': list(contents),
            'config': {'output_dimensionality': 768} if config is None else config}


def counts(*values):
    client = Mock()
    client.models.count_tokens.side_effect = [SimpleNamespace(total_tokens=v) for v in values]
    return client


def test_embedding2_constants():
    assert spend_guard.EMBEDDING_2_MODEL == MODEL
    assert spend_guard.EMBEDDING_2_OPERATION == OP
    assert spend_guard.EMBEDDING_2_DIMENSIONS == 768
    assert spend_guard.EMBEDDING_2_MAX_COUNTED_TOKENS_PER_TEXT == 2040


def test_embedding2_batch_of_two_texts_is_admitted_at_counted_text_rate():
    budget = guard()
    bounded = budget.bounded_genai_request(OP, request())
    assert bounded['config']['output_dimensionality'] == 768
    maximum, _ = budget.counted_genai_cap(OP, bounded, lambda **kwargs: SimpleNamespace(total_tokens=30))
    assert maximum == Decimal(60) * Decimal('0.20') / 1000000
    assert budget.snapshot()['reservation_count'] == 0


def test_embedding2_paid_batch_reserves_counted_bound():
    budget = guard(); client = counts(30, 30)
    guard_genai_client(client, budget).models.embed_content(**request())
    assert [c.kwargs['contents'] for c in client.models.count_tokens.call_args_list] == ['synthetic text one', 'synthetic text two']
    assert client.models.embed_content.call_args.kwargs['config']['output_dimensionality'] == 768
    assert budget.snapshot()['cash_exposure_usd'] == '0.000012'


def test_embedding2_text_at_2040_tokens_is_admitted():
    budget = guard()
    maximum, _ = budget.counted_genai_cap(OP, budget.bounded_genai_request(OP, request()),
                                          counts(10, 2040).models.count_tokens)
    assert maximum == Decimal(2050) * Decimal('0.20') / 1000000


def test_embedding2_text_counted_above_2040_tokens_is_refused_before_any_paid_call():
    budget = guard(); client = counts(10, 2041)
    with pytest.raises(SpendBlocked, match='2040') as raised:
        guard_genai_client(client, budget).models.embed_content(**request())
    assert str(raised.value) == 'gemini-embedding-2 allows at most 2040 counted tokens per text; no paid call made'
    client.models.embed_content.assert_not_called()
    assert budget.snapshot()['reservation_count'] == 0


@pytest.mark.parametrize('config', [{'output_dimensionality': 1536}, {}])
def test_embedding2_requires_768_dimensions(config):
    budget = guard()
    with pytest.raises(SpendBlocked, match=REASON):
        budget.bounded_genai_request(OP, request(config=config))
    assert budget.snapshot()['reservation_count'] == 0


def test_embedding2_without_config_is_refused():
    budget = guard()
    with pytest.raises(SpendBlocked, match=REASON):
        budget.bounded_genai_request(OP, {'model': MODEL, 'contents': ['synthetic text']})


@pytest.mark.parametrize('mutation', [
    lambda p: p.update(input_token_usd_per_million='0.21'),
    lambda p: p.update(output_dimensionality=3072),
    lambda p: p.update(output_token_usd_per_million='0.10'),
    lambda p: p.update(max_input_tokens=204801),
])
def test_embedding2_rejects_unverified_policy(mutation):
    budget = guard(mutate=mutation)
    with pytest.raises(SpendBlocked, match=REASON):
        budget.bounded_genai_request(OP, request())
    assert budget.snapshot()['reservation_count'] == 0


def test_embedding2_non_string_item_is_refused():
    budget = guard()
    bounded = budget.bounded_genai_request(OP, request(contents=['synthetic text', 7]))
    with pytest.raises(SpendBlocked, match='Only bounded text embedding batches are priced'):
        budget.counted_genai_cap(OP, bounded, lambda **kwargs: SimpleNamespace(total_tokens=1))
    assert budget.snapshot()['reservation_count'] == 0


def test_embedding2_full_bound_equals_cap():
    assert (Decimal(204800) * Decimal('0.20')) / 1000000 == Decimal('0.04096')


@pytest.mark.parametrize('code', ['unverified-embedding2-policy', 'embedding-text-token-bound'])
def test_new_reason_codes_resolve(code):
    assert owner_reasons.resolve(stage='spend_maintenance', code=code)[0] == code
