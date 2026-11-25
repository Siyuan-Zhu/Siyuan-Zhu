"""The Neon successor refreshes, disables only Neon on trouble, admits and settles Neon calls; synthetic data only."""
from decimal import Decimal
from types import SimpleNamespace
import pytest
from google.genai import types
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import (
    HYDROGEN_MODEL, HYDROGEN_OPERATION, NEON_MODEL, NEON_OPERATION, SpendBlocked, guard_genai_client,
    refused_before_dispatch)
from src.zeitgeist.spend_successor import build_neon_successor, build_successor, successor_ledger
from tests import test_gemini38_contract as base
from tests.test_hydrogen_admission_allowance import counter
from tests.test_optional_operations import above_approval, absent, guard_for, published, reported_missing, unlisted
from tests.test_spend_successor import installed
from tests.test_successor_compatibility import CONTRACT, HYDROGEN_VERSION, OLD_PIN, T, observation, refreshed

NEON_VERSION = 'gemini-4-neon'
INPUT_RATE, OUTPUT_RATE, OUTPUT_CAP = Decimal('3.00'), Decimal('15.00'), 6000
# The Neon probe (2026-11-25): Director counted 1060 and billed 1127; critic of nine images counted 10296
# (216 for its text) and billed 10102.
DIRECTOR_COUNTED, DIRECTOR_BILLED = 1060, 1127
CRITIC_COUNTED, CRITIC_TEXT_COUNTED, CRITIC_BILLED = 10296, 216, 10102


def neon_successor():
    """Hydrogen (high, 6000, 1091) and embedding-2 as on 22 Nov, then Neon (medium, 6000, 1091) on the page's digest."""
    ledger0, p0 = installed()
    p0['providers']['gemini']['price_snapshot_sha256'] = OLD_PIN
    ledger0.update(credit_only_providers=p0['credit_only_providers'], automatic_policy_sha256=auto.digest(p0))
    p1 = build_successor(p0, hydrogen_model_version=HYDROGEN_VERSION, hydrogen_tokens_per_image=1091,
                         hydrogen_max_output_tokens=6000, hydrogen_thinking_level='high', tariff_sha256='c' * 64,
                         include_embedding_2=True)
    p2 = build_neon_successor(p1, neon_model_version=NEON_VERSION, neon_tokens_per_image=1091,
                              neon_max_output_tokens=OUTPUT_CAP, neon_thinking_level='medium', tariff_sha256=T)
    ledger = dict(successor_ledger(ledger0, p2, auto.digest(p2)), credential_binding_required=True,
                  budget_scope='new_requests_after_cutoff')
    return ledger, p2


def listed(policy):
    obs = observation(policy, embedding_2_methods=['embedContent'])
    obs['gemini']['available_required_models'][NEON_MODEL] = ['generateContent']
    return obs


def neon_guard():
    ledger, policy = neon_successor()
    return guard_for(refreshed(ledger, policy, listed(policy)), policy)


def test_the_page_digest_covers_neon_at_its_approved_rates():
    assert CONTRACT['operations'][NEON_OPERATION]['input_token_usd_per_million'] == '3'
    assert CONTRACT['operations'][NEON_OPERATION]['output_token_usd_per_million'] == '15'


def test_neon_successor_refreshes_on_the_hash_path_with_nothing_disabled():
    ledger, policy = neon_successor()
    account = refreshed(ledger, policy, listed(policy))['providers']['gemini']
    assert 'disabled_operations' not in account and 'price_observed_sha256' not in account


CASES = {
    'neon_unlisted': unlisted(NEON_MODEL),
    'neon_reported_missing': reported_missing(NEON_MODEL),
    'neon_absent_from_contract': absent(NEON_OPERATION),
    'neon_rate_above_approval_in_band': above_approval(NEON_OPERATION, '1.05'),
    'neon_rate_above_the_band': above_approval(NEON_OPERATION, '3'),
    'neon_price_unreadable': published(lambda policy, operations: operations[NEON_OPERATION].update(
        input_token_usd_per_million=None)),
}


@pytest.mark.parametrize('name', sorted(CASES))
def test_trouble_with_neon_disables_neon_only(name):
    ledger, policy = neon_successor()
    account = refreshed(ledger, policy, CASES[name](policy, listed(policy)))['providers']['gemini']
    assert account['disabled_operations'] == [NEON_OPERATION]


def client(guard, count, prompt_tokens, version=NEON_VERSION, candidates=700, thoughts=1900):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return types.GenerateContentResponse(response_id='synthetic-response-' + version, model_version=version,
            usage_metadata={'prompt_token_count': prompt_tokens, 'candidates_token_count': candidates,
                            'thoughts_token_count': thoughts,
                            'total_token_count': prompt_tokens + candidates + thoughts})
    raw = SimpleNamespace(models=SimpleNamespace(generate_content=generate, count_tokens=count))
    return guard_genai_client(raw, guard), calls


def director(api, model=NEON_MODEL):
    return api.models.generate_content(model=model, contents='A synthetic editorial commission',
                                       config={'response_mime_type': 'application/json'})


def critic(api, images):
    return api.models.generate_content(model=NEON_MODEL,
                                       contents=['A synthetic photo judgement'] + [base.thumbnail() for _ in range(images)],
                                       config={'response_mime_type': 'application/json'})


def only_record(guard):
    records = list(guard.ledger.document['reservations'].values())
    assert len(records) == 1
    return records[0]


def hold(input_tokens, input_rate=INPUT_RATE, output_rate=OUTPUT_RATE):
    return (Decimal(input_tokens) * input_rate + Decimal(OUTPUT_CAP) * output_rate) / 1000000


@pytest.mark.parametrize('name', sorted(CASES))
def test_a_disabled_neon_is_refused_before_dispatch_and_flash_and_hydrogen_still_run(name):
    ledger, policy = neon_successor()
    guard = guard_for(refreshed(ledger, policy, CASES[name](policy, listed(policy))), policy)
    api, calls = client(guard, counter(DIRECTOR_COUNTED), DIRECTOR_BILLED)
    with pytest.raises(SpendBlocked) as caught:
        director(api)
    assert refused_before_dispatch(caught.value) is True and calls == []
    assert caught.value.public_diagnostics['code'] == 'unverified-neon-policy'
    for model, version in (('gemini-3.8-flash', 'gemini-3.8-flash'), (HYDROGEN_MODEL, HYDROGEN_VERSION)):
        api, calls = client(guard, counter(10), 10, version=version, candidates=5, thoughts=2)
        assert director(api, model).model_version == version and len(calls) == 1


def test_director_sends_medium_without_sampling_and_settles_the_probe_bill():
    guard = neon_guard()
    api, calls = client(guard, counter(DIRECTOR_COUNTED), DIRECTOR_BILLED)
    assert director(api).model_version == NEON_VERSION
    config = calls[0]['config']
    config = config.model_dump(exclude_none=True) if hasattr(config, 'model_dump') else config
    assert config['thinking_config'] == {'thinking_level': 'medium'} and config['max_output_tokens'] == OUTPUT_CAP
    assert 'temperature' not in config and 'candidate_count' not in config
    record = only_record(guard)
    assert record['operation'] == NEON_OPERATION
    assert record['request_context']['admitted_input_tokens'] == DIRECTOR_COUNTED + 256
    assert Decimal(record['max_gross_usd']) == hold(DIRECTOR_COUNTED + 256)
    assert record['status'] == 'settled'
    assert guard.ledger.document.get('reconciliation_required') is not True


def test_critic_with_nine_images_is_admitted_at_1091_per_image_and_settles():
    guard = neon_guard()
    api, calls = client(guard, counter(CRITIC_COUNTED, CRITIC_TEXT_COUNTED), CRITIC_BILLED)
    critic(api, 9)
    assert len(calls) == 1
    record = only_record(guard)
    assert record['request_context']['admitted_input_tokens'] == max(CRITIC_COUNTED, CRITIC_TEXT_COUNTED + 9 * 1091) + 256
    assert record['status'] == 'settled'


def test_one_image_below_the_count_floor_is_admitted_at_1091():
    guard = neon_guard()
    api, calls = client(guard, counter(300, CRITIC_TEXT_COUNTED), CRITIC_TEXT_COUNTED + 67 + 1091)
    critic(api, 1)
    record = only_record(guard)
    assert record['request_context']['admitted_input_tokens'] == CRITIC_TEXT_COUNTED + 1091 + 256
    assert record['status'] == 'settled'


def test_a_prompt_overage_inside_the_hold_settles_at_the_billed_figure():
    guard = neon_guard()
    admitted = DIRECTOR_COUNTED + 256
    api, calls = client(guard, counter(DIRECTOR_COUNTED), admitted + 1)
    director(api)
    record = only_record(guard)
    assert record['status'] == 'settled'
    assert record['usage_receipt']['prompt_tokens_over_admitted'] == 1
    assert Decimal(record['actual_gross_usd']) == (Decimal(admitted + 1) * INPUT_RATE + Decimal(2600) * OUTPUT_RATE) / 1000000


def test_a_reply_from_an_unpinned_version_freezes_the_month():
    guard = neon_guard()
    api, calls = client(guard, counter(DIRECTOR_COUNTED), DIRECTOR_BILLED, version='gemini-4-neon-002')
    with pytest.raises(SpendBlocked) as caught:
        director(api)
    assert refused_before_dispatch(caught.value) is False and len(calls) == 1
    assert guard.ledger.document['reconciliation_required'] is True
    assert only_record(guard)['status'] == 'reserved'


def test_hydrogen_on_the_neon_successor_is_held_at_hydrogen_rates():
    guard = neon_guard()
    api, calls = client(guard, counter(DIRECTOR_COUNTED), DIRECTOR_BILLED, version=HYDROGEN_VERSION)
    director(api, HYDROGEN_MODEL)
    record = only_record(guard)
    assert record['operation'] == HYDROGEN_OPERATION
    assert Decimal(record['max_gross_usd']) == hold(DIRECTOR_COUNTED + 256, Decimal('1.50'), Decimal('7.50'))
    assert record['status'] == 'settled'
