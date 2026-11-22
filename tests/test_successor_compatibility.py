"""The installed policy and its Hydrogen successor both refresh and admit; synthetic data only."""
import copy
from types import SimpleNamespace
import pytest
from google.genai import types
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.automatic_guard import AutomaticBudgetGuard
from src.zeitgeist.gemini_tariff import normalize_gemini_tariff
from src.zeitgeist.spend_guard import (
    HYDROGEN_MODEL, HYDROGEN_OPERATION, MemoryLedger, SpendBlocked, guard_genai_client, refused_before_dispatch)
from src.zeitgeist.spend_successor import EMBEDDING_001_OPERATION, FLASH_OPERATION, build_successor, successor_ledger
from tests.test_automatic_spend import NOW, fixtures
from tests.test_gemini_tariff import markdoc
from tests.test_spend_successor import installed

HYDROGEN_VERSION = 'synthetic-hydrogen-version'
CONTRACT = normalize_gemini_tariff(markdoc())
T = auto.digest(CONTRACT)
OLD_PIN = 'd' * 64


def policies():
    ledger0, p0 = installed()
    p0['providers']['gemini']['price_snapshot_sha256'] = OLD_PIN
    ledger0.update(credit_only_providers=p0['credit_only_providers'], automatic_policy_sha256=auto.digest(p0))
    p1 = build_successor(p0, hydrogen_model_version=HYDROGEN_VERSION, hydrogen_tokens_per_image=1088, tariff_sha256=T)
    return ledger0, p0, p1


def observation(policy, hydrogen_listed=True):
    obs = fixtures()[2]
    obs['price_snapshots']['gemini'] = {'sha256': T, 'contract': copy.deepcopy(CONTRACT), 'observed_at': NOW.isoformat(),
        'source_url': policy['providers']['gemini']['price_source_url']}
    models = {'gemini-3.8-flash': ['generateContent'], 'gemini-embedding-001': ['embedContent']}
    if hydrogen_listed:
        models[HYDROGEN_MODEL] = ['generateContent']
    obs['gemini']['available_required_models'] = models
    return obs


def refreshed(ledger, policy, obs):
    return auto.prepare_refresh(ledger, policy, obs, expected_policy_sha=auto.digest(policy), now=NOW)['after']


def test_installed_policy_refreshes_on_the_rate_path():
    ledger0, p0, _ = policies()
    assert p0['providers']['gemini']['price_snapshot_sha256'] != T
    account = refreshed(ledger0, p0, observation(p0))['providers']['gemini']
    assert account['price_observed_sha256'] == T
    assert account['price_observed_at'] == NOW.isoformat()
    assert 'price_band_audit' not in account


def test_successor_refreshes_on_the_hash_path():
    ledger0, _, p1 = policies()
    ledger1 = successor_ledger(ledger0, p1, auto.digest(p1))
    account = refreshed(ledger1, p1, observation(p1))['providers']['gemini']
    assert 'price_observed_sha256' not in account and 'price_observed_at' not in account
    assert 'price_band_audit' not in account
    assert HYDROGEN_OPERATION in account['operation_caps_usd']


def test_successor_requires_hydrogen_listed():
    ledger0, _, p1 = policies()
    ledger1 = successor_ledger(ledger0, p1, auto.digest(p1))
    with pytest.raises(SpendBlocked, match='Required exact Gemini model/method unavailable'):
        refreshed(ledger1, p1, observation(p1, hydrogen_listed=False))
    # The installed policy does not budget Hydrogen, so it does not need it listed.
    ledger0_, p0, _ = policies()
    refreshed(ledger0_, p0, observation(p0, hydrogen_listed=False))


def guard_for(ledger, policy):
    ledger = dict(copy.deepcopy(ledger), credential_binding_required=True, budget_scope='new_requests_after_cutoff')
    refresh = refreshed(ledger, policy, observation(policy))
    for item in refresh['providers'].values():
        item['verified_at'] = '2026-10-27T00:00:00Z'
    return AutomaticBudgetGuard(MemoryLedger(refresh), policy, auto.digest(policy), lambda: NOW,
        credential_fingerprints={'gemini': 'a' * 16, 'bfl': 'b' * 16, 'apify': 'c' * 16},
        require_credential_binding=True)


def fake_client(guard, version):
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return types.GenerateContentResponse(response_id='synthetic-server-response', model_version=version,
            usage_metadata={'prompt_token_count': 10, 'candidates_token_count': 5,
                            'thoughts_token_count': 2, 'total_token_count': 17})
    raw = SimpleNamespace(models=SimpleNamespace(generate_content=generate,
        count_tokens=lambda **kw: SimpleNamespace(total_tokens=10)))
    return guard_genai_client(raw, guard), calls


def call(api, model):
    return api.models.generate_content(model=model, contents='A synthetic editorial commission',
                                       config={'response_mime_type': 'application/json'})


def admitted_guards():
    ledger0, p0, p1 = policies()
    ledger1 = successor_ledger(ledger0, p1, auto.digest(p1))
    return guard_for(ledger0, p0), guard_for(ledger1, p1)


def test_flash_call_admitted_under_both():
    for guard in admitted_guards():
        api, calls = fake_client(guard, 'gemini-3.8-flash')
        assert call(api, 'gemini-3.8-flash').model_version == 'gemini-3.8-flash'
        assert len(calls) == 1


def test_hydrogen_call_admitted_under_successor_only():
    old, new = admitted_guards()
    api, calls = fake_client(old, HYDROGEN_VERSION)
    with pytest.raises(SpendBlocked) as caught:
        call(api, HYDROGEN_MODEL)
    assert refused_before_dispatch(caught.value) is True
    assert calls == []
    api, calls = fake_client(new, HYDROGEN_VERSION)
    assert call(api, HYDROGEN_MODEL).model_version == HYDROGEN_VERSION
    assert len(calls) == 1
