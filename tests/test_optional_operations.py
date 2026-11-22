"""A budgeted new-model operation that is not listed or not published is disabled, not fatal; synthetic data only."""
import copy
from decimal import Decimal
from types import SimpleNamespace
import pytest
from google.genai import types
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.automatic_guard import AutomaticBudgetGuard
from src.zeitgeist.spend_guard import (
    EMBEDDING_2_MODEL, EMBEDDING_2_OPERATION, HYDROGEN_MODEL, HYDROGEN_OPERATION, MemoryLedger, SpendBlocked,
    guard_genai_client, refused_before_dispatch)
from src.zeitgeist.spend_successor import EMBEDDING_001_OPERATION, FLASH_OPERATION, successor_ledger
from tests.test_automatic_spend import NOW
from tests.test_successor_compatibility import CONTRACT, HYDROGEN_VERSION, observation, policies, refreshed

FINGERPRINTS = {'gemini': 'a' * 16, 'bfl': 'b' * 16, 'apify': 'c' * 16}


def test_the_optional_operations_are_exactly_the_two_new_models():
    assert auto.OPTIONAL_OPERATIONS == frozenset({'generate_content:gemini-4-hydrogen', 'embed_content:gemini-embedding-2'})


def successor(include_embedding_2):
    ledger0, _, p1 = policies(include_embedding_2=include_embedding_2)
    ledger1 = successor_ledger(ledger0, p1, auto.digest(p1))
    return dict(ledger1, credential_binding_required=True, budget_scope='new_requests_after_cutoff'), p1


def listed(policy, include_embedding_2):
    return observation(policy, embedding_2_methods=['embedContent'] if include_embedding_2 else None)


def unlisted(model):
    def change(policy, obs):
        del obs['gemini']['available_required_models'][model]
        return obs
    return change


def reported_missing(model):
    """The preflight's own shape when a required model is missing from ListModels."""
    def change(policy, obs):
        del obs['gemini']['available_required_models'][model]
        obs['gemini'].update(status='incomplete', missing_models=[model])
        return obs
    return change


def published(edit):
    def change(policy, obs):
        contract = copy.deepcopy(CONTRACT)
        edit(policy, contract['operations'])
        obs['price_snapshots']['gemini'] = dict(obs['price_snapshots']['gemini'], contract=contract, sha256=auto.digest(contract))
        return obs
    return change


def absent(operation):
    return published(lambda policy, operations: operations.pop(operation))


def above_approval(operation, factor):
    def edit(policy, operations):
        approved = policy['providers']['gemini']['account']['request_limits'][operation]['input_token_usd_per_million']
        operations[operation]['input_token_usd_per_million'] = format((Decimal(approved) * Decimal(factor)).normalize(), 'f')
    return published(edit)


CASES = {
    'hydrogen_unlisted': (False, HYDROGEN_OPERATION, unlisted(HYDROGEN_MODEL)),
    'hydrogen_reported_missing': (False, HYDROGEN_OPERATION, reported_missing(HYDROGEN_MODEL)),
    'hydrogen_absent_from_contract': (False, HYDROGEN_OPERATION, absent(HYDROGEN_OPERATION)),
    'hydrogen_rate_above_approval_in_band': (False, HYDROGEN_OPERATION, above_approval(HYDROGEN_OPERATION, '1.05')),
    'hydrogen_rate_above_the_band': (False, HYDROGEN_OPERATION, above_approval(HYDROGEN_OPERATION, '3')),
    'embedding_2_unlisted': (True, EMBEDDING_2_OPERATION, unlisted(EMBEDDING_2_MODEL)),
    'embedding_2_reported_missing': (True, EMBEDDING_2_OPERATION, reported_missing(EMBEDDING_2_MODEL)),
    'embedding_2_absent_from_contract': (True, EMBEDDING_2_OPERATION, absent(EMBEDDING_2_OPERATION)),
    'embedding_2_rate_above_approval_in_band': (True, EMBEDDING_2_OPERATION, above_approval(EMBEDDING_2_OPERATION, '1.05')),
    'embedding_2_rate_above_the_band': (True, EMBEDDING_2_OPERATION, above_approval(EMBEDDING_2_OPERATION, '3')),
}


def guard_for(refresh, policy):
    refresh = copy.deepcopy(refresh)
    for item in refresh['providers'].values():
        item['verified_at'] = '2026-10-27T00:00:00Z'
    return AutomaticBudgetGuard(MemoryLedger(refresh), policy, auto.digest(policy), lambda: NOW,
        credential_fingerprints=FINGERPRINTS, require_credential_binding=True)


def fake_client(guard, version):
    calls = []

    def generate(**kwargs):
        calls.append(('generate_content', kwargs['model']))
        return types.GenerateContentResponse(response_id='synthetic-server-response', model_version=version,
            usage_metadata={'prompt_token_count': 10, 'candidates_token_count': 5,
                            'thoughts_token_count': 2, 'total_token_count': 17})

    def embed(**kwargs):
        calls.append(('embed_content', kwargs['model']))
        return types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[0.5] * 768)])

    def count(**kwargs):
        calls.append(('count_tokens', kwargs['model']))
        return SimpleNamespace(total_tokens=10)

    raw = SimpleNamespace(models=SimpleNamespace(generate_content=generate, embed_content=embed, count_tokens=count))
    return guard_genai_client(raw, guard), calls


def generate(api, model):
    return api.models.generate_content(model=model, contents='A synthetic editorial commission',
                                       config={'response_mime_type': 'application/json'})


def embed(api, model):
    text = 'task: clustering | query: a synthetic headline'
    # Embedding-2 is admitted only as one-text Contents (bare strings would be aggregated).
    contents = [types.Content(parts=[types.Part(text=text)])] if model == EMBEDDING_2_MODEL else [text]
    return api.models.embed_content(model=model, contents=contents, config={'output_dimensionality': 768})


@pytest.mark.parametrize('name', sorted(CASES))
def test_maintenance_disables_only_the_troubled_optional_operation(name):
    include_embedding_2, operation, change = CASES[name]
    ledger, policy = successor(include_embedding_2)
    account = refreshed(ledger, policy, change(policy, listed(policy, include_embedding_2)))['providers']['gemini']
    assert account['disabled_operations'] == [operation]
    assert operation in account['operation_caps_usd']
    assert all(entry['operation'] != operation for entry in account.get('price_band_audit', {}).get('above_approval', []))


@pytest.mark.parametrize('name', sorted(CASES))
def test_a_disabled_operation_is_refused_before_any_provider_call_and_flash_still_runs(name):
    include_embedding_2, operation, change = CASES[name]
    ledger, policy = successor(include_embedding_2)
    guard = guard_for(refreshed(ledger, policy, change(policy, listed(policy, include_embedding_2))), policy)
    api, calls = fake_client(guard, 'gemini-3.8-flash')
    assert generate(api, 'gemini-3.8-flash').model_version == 'gemini-3.8-flash'
    assert calls[-1] == ('generate_content', 'gemini-3.8-flash')
    del calls[:]
    api, calls = fake_client(guard, HYDROGEN_VERSION)
    model = operation.split(':', 1)[1]
    with pytest.raises(SpendBlocked) as caught:
        (generate if operation == HYDROGEN_OPERATION else embed)(api, model)
    assert refused_before_dispatch(caught.value) is True
    assert caught.value.public_diagnostics['code'] in set(SpendBlocked._PUBLIC_REASONS.values())
    assert calls == []


def test_a_refresh_that_disables_hydrogen_after_counting_still_stops_the_dispatch():
    ledger, policy = successor(False)
    guard = guard_for(refreshed(ledger, policy, listed(policy, False)), policy)
    api, calls = fake_client(guard, HYDROGEN_VERSION)

    def disable_during_count(document):
        document['providers']['gemini']['disabled_operations'] = [HYDROGEN_OPERATION]
    original = guard.counted_genai_admission

    def counted(operation, bounded, count_tokens):
        result = original(operation, bounded, count_tokens)
        guard.ledger.mutate(disable_during_count)
        return result
    guard.counted_genai_admission = counted
    with pytest.raises(SpendBlocked) as caught:
        generate(api, HYDROGEN_MODEL)
    assert refused_before_dispatch(caught.value) is True
    assert ('generate_content', HYDROGEN_MODEL) not in calls


@pytest.mark.parametrize('when', ['before_reserve', 'between_reserve_and_bind'])
def test_a_disable_that_lands_after_admission_is_refused_at_bind_before_any_provider_call(when):
    ledger, policy = successor(False)
    guard = guard_for(refreshed(ledger, policy, listed(policy, False)), policy)
    api, calls = fake_client(guard, HYDROGEN_VERSION)

    def disable(document):
        document['providers']['gemini']['disabled_operations'] = [HYDROGEN_OPERATION]
    original = guard.reserve

    def reserve(*args, **kwargs):
        if when == 'before_reserve':
            guard.ledger.mutate(disable)
        result = original(*args, **kwargs)
        if when == 'between_reserve_and_bind':
            guard.ledger.mutate(disable)
        return result
    guard.reserve = reserve
    with pytest.raises(SpendBlocked) as caught:
        generate(api, HYDROGEN_MODEL)
    assert refused_before_dispatch(caught.value) is True
    assert caught.value.public_diagnostics['code'] in set(SpendBlocked._PUBLIC_REASONS.values())
    assert ('generate_content', HYDROGEN_MODEL) not in calls


def listed_but_reported_missing(model, status):
    """A report whose missing list names the model while its method map still lists it."""
    def change(policy, obs):
        obs['gemini'].update(status=status, missing_models=[model])
        return obs
    return change


@pytest.mark.parametrize('status', ['incomplete', 'metadata_verified'])
@pytest.mark.parametrize('include_embedding_2, model, operation', [
    (False, HYDROGEN_MODEL, HYDROGEN_OPERATION),
    (True, EMBEDDING_2_MODEL, EMBEDDING_2_OPERATION),
], ids=['hydrogen', 'embedding_2'])
def test_an_optional_model_on_the_missing_list_is_disabled_even_when_its_method_is_listed(
        include_embedding_2, model, operation, status):
    ledger, policy = successor(include_embedding_2)
    obs = listed(policy, include_embedding_2)
    assert obs['gemini']['available_required_models'][model]
    refresh = refreshed(ledger, policy, listed_but_reported_missing(model, status)(policy, obs))
    assert refresh['providers']['gemini']['disabled_operations'] == [operation]
    api, calls = fake_client(guard_for(refresh, policy), HYDROGEN_VERSION)
    with pytest.raises(SpendBlocked) as caught:
        (generate if operation == HYDROGEN_OPERATION else embed)(api, model)
    assert refused_before_dispatch(caught.value) is True and calls == []


def test_the_other_optional_operation_keeps_working_when_one_is_disabled():
    ledger, policy = successor(True)
    obs = unlisted(HYDROGEN_MODEL)(policy, listed(policy, True))
    refresh = refreshed(ledger, policy, obs)
    assert refresh['providers']['gemini']['disabled_operations'] == [HYDROGEN_OPERATION]
    api, calls = fake_client(guard_for(refresh, policy), 'gemini-embedding-2')
    embed(api, EMBEDDING_2_MODEL)
    assert ('embed_content', EMBEDDING_2_MODEL) in calls


def test_both_optional_operations_can_be_disabled_together():
    ledger, policy = successor(True)
    obs = unlisted(EMBEDDING_2_MODEL)(policy, absent(HYDROGEN_OPERATION)(policy, listed(policy, True)))
    account = refreshed(ledger, policy, obs)['providers']['gemini']
    assert account['disabled_operations'] == sorted([HYDROGEN_OPERATION, EMBEDDING_2_OPERATION])


def test_a_healthy_refresh_records_nothing_and_clears_an_earlier_disable():
    ledger, policy = successor(True)
    first = refreshed(ledger, policy, unlisted(HYDROGEN_MODEL)(policy, listed(policy, True)))
    assert first['providers']['gemini']['disabled_operations'] == [HYDROGEN_OPERATION]
    again = refreshed(first, policy, listed(policy, True))
    assert 'disabled_operations' not in again['providers']['gemini']
    api, calls = fake_client(guard_for(again, policy), HYDROGEN_VERSION)
    assert generate(api, HYDROGEN_MODEL).model_version == HYDROGEN_VERSION


@pytest.mark.parametrize('change', [
    unlisted('gemini-3.8-flash'),
    reported_missing('gemini-3.8-flash'),
    unlisted('gemini-embedding-001'),
    absent(FLASH_OPERATION),
    absent(EMBEDDING_001_OPERATION),
    above_approval(FLASH_OPERATION, '3'),
    above_approval(EMBEDDING_001_OPERATION, '3'),
])
@pytest.mark.parametrize('include_embedding_2', [False, True])
def test_trouble_with_a_non_optional_operation_still_blocks_everything(change, include_embedding_2):
    ledger, policy = successor(include_embedding_2)
    with pytest.raises(SpendBlocked):
        refreshed(ledger, policy, change(policy, listed(policy, include_embedding_2)))


def test_a_missing_optional_model_does_not_hide_a_missing_flash():
    ledger, policy = successor(False)
    obs = listed(policy, False)
    del obs['gemini']['available_required_models'][HYDROGEN_MODEL]
    del obs['gemini']['available_required_models']['gemini-3.8-flash']
    obs['gemini'].update(status='incomplete', missing_models=['gemini-3.8-flash', HYDROGEN_MODEL])
    with pytest.raises(SpendBlocked):
        refreshed(ledger, policy, obs)


def installed_policy():
    ledger0, p0, _ = policies()
    return dict(ledger0, credential_binding_required=True, budget_scope='new_requests_after_cutoff'), p0


def test_the_installed_policy_is_unaffected():
    ledger, policy = installed_policy()
    healthy = refreshed(ledger, policy, observation(policy))
    without_hydrogen = refreshed(ledger, policy, observation(policy, hydrogen_listed=False))
    for document in (healthy, without_hydrogen):
        assert 'disabled_operations' not in document['providers']['gemini']
        for account in document['providers'].values():
            account.pop('automatic_observation_sha256')
    assert healthy == without_hydrogen
    # No optional operation is budgeted, so a missing Hydrogen report is not tolerated either.
    obs = reported_missing(HYDROGEN_MODEL)(policy, observation(policy))
    obs['gemini']['missing_models'] = ['gemini-3.8-flash']
    with pytest.raises(SpendBlocked):
        refreshed(ledger, policy, obs)


class _Client:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *exc): return False


def observe(monkeypatch, policy, report):
    from scripts import provider_preflight
    from src.zeitgeist import gemini_tariff
    for provider in ('apify', 'bfl'):
        policy['providers'][provider]['price_evidence_mode'] = 'published_validity'
    monkeypatch.setattr(runtime.httpx, 'Client', _Client)
    monkeypatch.setattr(provider_preflight, 'run_preflight', lambda *args, **kwargs: copy.deepcopy(report))
    monkeypatch.setattr(gemini_tariff, 'observe_gemini_tariff', lambda client, *, now: copy.deepcopy(report['price_snapshots']['gemini']))
    return runtime.observe_provider_metadata(policy, {'GEMINI_API_KEY': 'synthetic-private-key'}, NOW)


def test_the_metadata_observer_tolerates_a_missing_optional_model(monkeypatch):
    _, policy = successor(False)
    report = reported_missing(HYDROGEN_MODEL)(policy, listed(policy, False))
    mapped = observe(monkeypatch, copy.deepcopy(policy), report)
    assert mapped['gemini']['missing_models'] == [HYDROGEN_MODEL]


@pytest.mark.parametrize('missing', [['gemini-3.8-flash'], ['gemini-3.8-flash', HYDROGEN_MODEL], []])
def test_the_metadata_observer_still_blocks_a_missing_required_model(monkeypatch, missing):
    _, policy = successor(False)
    report = listed(policy, False)
    report['gemini'].update(status='incomplete', missing_models=missing)
    with pytest.raises(SpendBlocked):
        observe(monkeypatch, copy.deepcopy(policy), report)
