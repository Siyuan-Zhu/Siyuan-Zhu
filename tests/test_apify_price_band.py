"""Apify price band (SIY-291): up to 30% above the policy-approved per-event price, never above it compounding."""
import copy
from decimal import Decimal

import pytest

from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_automatic_spend import NOW, fixtures, refresh

OP = 'instagram-post-scraper'
TERMS = {"event_prices_usd":{"music-data":{"BRONZE":"0.006","DIAMOND":"0.002","FREE":"0.007","GOLD":"0.004","PLATINUM":"0.003","SILVER":"0.005"},"post":{"BRONZE":"0.0015","DIAMOND":"0.0004","FREE":"0.0017","GOLD":"0.001","PLATINUM":"0.0008","SILVER":"0.0013"},"post-details":{"BRONZE":"0.0008","DIAMOND":"0.0001","FREE":"0.001","GOLD":"0.0006","PLATINUM":"0.0002","SILVER":"0.0007"}},"minimum_run_charge_cap_usd":"0.005","model":"PAY_PER_EVENT","started_at":"2026-10-07T13:39:36.280Z"}
CHANGED = 'Apify account, free grant, usage cycle, build or pricing changed'


def priced(changes, base=TERMS):
    """TERMS with 'event:tier' prices replaced (None removes the tier)."""
    terms = copy.deepcopy(base)
    for key, value in changes.items():
        event, tier = key.split(':')
        if value is None:
            terms['event_prices_usd'][event].pop(tier)
        else:
            terms['event_prices_usd'][event][tier] = value
    return terms


def setup(observed=None, *, with_terms=True):
    ledger, policy, observation = fixtures()
    for account in (ledger['providers']['apify'], policy['providers']['apify']['account']):
        account['operation_caps_usd'] = {OP: '.20'}
    apify = policy['providers']['apify']
    # Production shape: the policy's Apify price snapshot is the pricing digest, and the live observer derives the
    # snapshot from the pricing it read (automatic_spend_runtime.observe_provider_metadata).
    apify['actor_pricing_sha256'] = apify['price_snapshot_sha256'] = auto.digest(TERMS)
    if with_terms:
        apify['actor_pricing_terms'] = copy.deepcopy(TERMS)
    ledger['automatic_policy_sha256'] = auto.digest(policy)
    pricing = copy.deepcopy(TERMS if observed is None else observed)
    observation['apify']['pricing'] = pricing
    observation['price_snapshots']['apify']['sha256'] = auto.digest(pricing)
    return ledger, policy, observation


def test_same_terms_new_start_date_is_no_rise():
    assert auto.apify_price_review(TERMS, {**TERMS, 'started_at': '2026-12-01T00:00:00.000Z'}) == []


def test_rise_at_exactly_thirty_percent_passes():
    assert auto.apify_price_review(TERMS, priced({'post:FREE': '0.00221'})) == [
        {'operation': OP, 'rate': 'event_price_usd:post:FREE', 'approved': '0.0017', 'observed': '0.00221'}]


def test_rise_above_thirty_percent_blocks():
    assert auto.apify_price_review(TERMS, priced({'post:FREE': '0.0022101'})) is None


def test_fall_passes_and_is_not_audited():
    review = auto.apify_price_review(TERMS, priced({'post:FREE': '0.0010', 'post-details:FREE': '0.0012'}))
    assert review == [{'operation': OP, 'rate': 'event_price_usd:post-details:FREE', 'approved': '0.001',
                       'observed': '0.0012'}]


def _added_event():
    terms = copy.deepcopy(TERMS)
    terms['event_prices_usd']['actor-start'] = {'FREE': '0.001'}
    return terms


def _removed_event():
    terms = copy.deepcopy(TERMS)
    terms['event_prices_usd'].pop('music-data')
    return terms


@pytest.mark.parametrize('observed', [
    {**TERMS, 'model': 'PAY_PER_RESULT'},
    _added_event(),
    _removed_event(),
    priced({'post:GOLD': None}),
    priced({'post:FREE': 'abc'}),
    priced({'post:FREE': '-0.001'}),
    priced({'post:FREE': None}) | {},
    {**TERMS, 'minimum_run_charge_cap_usd': '0.0066'},
    None,
    'not-a-dict',
])
def test_structural_change_blocks(observed):
    assert auto.apify_price_review(TERMS, observed) is None


def test_refresh_writes_audit_for_in_band_rise():
    observed = priced({'post:FREE': '0.002'})
    after = refresh(*setup(observed))
    account = after['providers']['apify']
    audit = account['price_band_audit']
    assert audit['tolerance'] == '0.30' and audit['basis'] == 'policy_approved_rate'
    assert audit['observed_sha256'] == auto.digest(observed) == account['price_observed_sha256']
    assert audit['above_approval'] == [{'operation': OP, 'rate': 'event_price_usd:post:FREE',
                                        'approved': '0.0017', 'observed': '0.002'}]
    assert account['approved_pricing_terms'] == TERMS
    assert auto.verified_price_band_audit(account, 'apify') is True
    assert auto.charged_rate(account, 'apify', OP, auto.APIFY_POST_RATE) == Decimal('0.002')


def test_refresh_exact_terms_clears_audit():
    ledger, policy, observation = setup()
    ledger['providers']['apify'].update(price_observed_sha256='x', price_observed_at='y', price_band_audit={'stale': 1})
    account = refresh(ledger, policy, observation)['providers']['apify']
    assert not {'price_observed_sha256', 'price_observed_at', 'price_band_audit'} & set(account)
    assert account['approved_pricing_terms'] == TERMS


def test_refresh_redated_terms_is_no_rise():
    account = refresh(*setup({**TERMS, 'started_at': '2026-12-01T00:00:00.000Z'}))['providers']['apify']
    assert 'price_band_audit' not in account


def test_refresh_without_terms_keeps_exact_compare():
    with pytest.raises(SpendBlocked, match='Primary price snapshot changed'):
        refresh(*setup({**TERMS, 'started_at': '2026-12-01T00:00:00.000Z'}, with_terms=False))


def test_refresh_above_band_blocks():
    with pytest.raises(SpendBlocked, match=CHANGED):
        refresh(*setup(priced({'post:FREE': '0.0023'})))


def test_a_snapshot_that_is_not_the_observed_pricing_still_blocks():
    ledger, policy, observation = setup(priced({'post:FREE': '0.002'}))
    observation['price_snapshots']['apify']['sha256'] = 'f' * 64
    with pytest.raises(SpendBlocked, match='Primary price snapshot changed'):
        refresh(ledger, policy, observation)


def test_a_stale_snapshot_still_blocks():
    ledger, policy, observation = setup(priced({'post:FREE': '0.002'}))
    observation['price_snapshots']['apify']['observed_at'] = '2026-10-29T00:00:00+00:00'
    with pytest.raises(SpendBlocked, match='Primary price snapshot changed'):
        refresh(ledger, policy, observation)


def test_a_rise_in_the_minimum_run_charge_alone_is_audited():
    assert auto.apify_price_review(TERMS, {**TERMS, 'minimum_run_charge_cap_usd': '0.006'}) == [
        {'operation': OP, 'rate': 'minimum_run_charge_cap_usd', 'approved': '0.005', 'observed': '0.006'}]
    account = refresh(*setup({**TERMS, 'minimum_run_charge_cap_usd': '0.006'}))['providers']['apify']
    assert auto.verified_price_band_audit(account, 'apify') is True


def test_validate_policy_refuses_terms_that_do_not_match_the_pin():
    ledger, policy, observation = setup()
    policy['providers']['apify']['actor_pricing_terms'] = priced({'post:FREE': '0.0018'})
    with pytest.raises(SpendBlocked):
        auto.validate_policy(policy, auto.digest(policy), NOW)


def _in_band_account():
    return refresh(*setup(priced({'post:FREE': '0.002'})))['providers']['apify']


def _record(context):
    return {'provider': 'apify', 'operation': OP, 'request_context': context}


RATES = {auto.APIFY_POST_RATE: '0.002', 'approved_' + auto.APIFY_POST_RATE: '0.0017', 'max_posts': 60}


def test_hold_cap_grows_by_band_when_charged_above_approved():
    account = _in_band_account()
    assert auto.hold_cap(account, 'apify', _record(dict(RATES))) == Decimal('0.26')
    assert auto.hold_cap(account, 'apify', _record({})) == Decimal('0.20')
    assert auto.effective_cap(account, 'apify', OP) == Decimal('0.26')


def test_in_band_excess_counts_posts():
    assert auto.in_band_excess_usd(_record(dict(RATES))) == Decimal('0.018')
    for posts in (None, '60', -1):
        context = dict(RATES)
        if posts is None:
            context.pop('max_posts')
        else:
            context['max_posts'] = posts
        with pytest.raises(SpendBlocked):
            auto.in_band_excess_usd(_record(context))


# --- Task 2: the hold records its post rates; the run's charge limit follows the charged rate ---

import httpx

from src.zeitgeist import collectors
from src.zeitgeist.automatic_guard import AutomaticBudgetGuard
from src.zeitgeist.spend_guard import MemoryLedger
from tests import test_provider_lifecycle_receipts as lifecycle_tests

DIGEST = 'a' * 64


def apify_guard(observed=None):
    ledger, policy, observation = setup(observed)
    ledger.update(credential_binding_required=True, budget_scope='new_requests_after_cutoff')
    ledger['automatic_policy_sha256'] = auto.digest(policy)
    refreshed = auto.prepare_refresh(ledger, policy, observation, expected_policy_sha=auto.digest(policy), now=NOW)['after']
    for item in refreshed['providers'].values():
        item['verified_at'] = '2026-10-27T00:00:00Z'
    store = MemoryLedger(refreshed)
    guard = AutomaticBudgetGuard(store, policy, auto.digest(policy), lambda: NOW,
                                 credential_fingerprints={'gemini': 'a' * 16, 'bfl': 'b' * 16, 'apify': 'c' * 16},
                                 require_credential_binding=True)
    return guard, store


def context(charged, approved='0.0017', posts=60):
    return {auto.APIFY_POST_RATE: charged, 'approved_' + auto.APIFY_POST_RATE: approved, 'max_posts': posts}


def held(store):
    (record,) = [r for r in store.mutate(lambda d: copy.deepcopy(d))['reservations'].values() if r['provider'] == 'apify']
    return record


def test_guard_reads_post_rates_and_run_floor():
    guard, _ = apify_guard(priced({'post:FREE': '0.002'}))
    assert guard.apify_post_rates() == (Decimal('0.002'), Decimal('0.0017'), Decimal('0.00650'))
    guard, _ = apify_guard()
    assert guard.apify_post_rates()[0] == Decimal('0.0017')


def test_hold_records_post_rates():
    guard, store = apify_guard(priced({'post:FREE': '0.002'}))
    guard.begin_provider_request('apify', OP, 0.12, DIGEST, context=context('0.002'))
    assert held(store)['request_context'] == context('0.002')


def test_twenty_handles_in_band_fit_the_grown_cap():
    guard, _ = apify_guard(priced({'post:FREE': '0.00221'}))
    guard.begin_provider_request('apify', OP, 0.221, DIGEST, context=context('0.00221', posts=100))
    guard, _ = apify_guard()
    guard.begin_provider_request('apify', OP, 0.17, DIGEST, context=context('0.0017', posts=100))


def test_a_limit_above_posts_at_the_charged_rate_is_refused():
    guard, _ = apify_guard(priced({'post:FREE': '0.002'}))
    with pytest.raises(SpendBlocked, match='tariff'):
        guard.begin_provider_request('apify', OP, 0.1221, DIGEST, context=context('0.002'))


def test_stale_rate_context_is_refused():
    guard, _ = apify_guard(priced({'post:FREE': '0.002'}))
    with pytest.raises(SpendBlocked, match='retry admission'):
        guard.begin_provider_request('apify', OP, 0.102, DIGEST, context=context('0.0017'))


class RatedGuard(lifecycle_tests.Guard):
    def __init__(self, events, rates):
        super().__init__(events)
        self.rates, self.contexts = rates, []

    def apify_post_rates(self):
        return self.rates

    def begin_provider_request(self, provider, operation, maximum, digest, *, context=None):
        self.contexts.append(context)
        return super().begin_provider_request(provider, operation, maximum, digest)


def collect(monkeypatch, guard, handles):
    limits = []

    def handle(request):
        if request.method == 'POST':
            limits.append(request.url.params['maxTotalChargeUsd'])
            return httpx.Response(201, json={'data': lifecycle_tests.run_body('RUNNING')})
        if request.url.path.endswith('/dataset/items'):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={'data': lifecycle_tests.run_body()})

    lifecycle_tests.wire(monkeypatch, handle)
    collectors.fetch_instagram_batch(handles, 'fixture-private-key', spend_guard=guard, actor_build='0.0.648')
    return limits


HANDLES = ['handle%02d' % i for i in range(12)]


# One post of headroom above posts at the charged rate, and never below today's fixed 0.002 a post.
@pytest.mark.parametrize('charged,limit', [('0.0017', '0.12'), ('0.002', '0.122'), ('0.00221', '0.13481')])
def test_collector_limit_follows_charged_rate(monkeypatch, charged, limit):
    guard = RatedGuard([], (Decimal(charged), Decimal('0.0017'), Decimal('0.0065')))
    assert collect(monkeypatch, guard, HANDLES) == [limit]
    assert guard.contexts == [context(charged)]


def test_collector_floor_for_a_single_handle(monkeypatch):
    guard = RatedGuard([], (Decimal('0.0017'), Decimal('0.0017'), Decimal('0.0065')))
    assert collect(monkeypatch, guard, ['solo']) == ['0.0102']


def test_collector_never_sends_a_limit_below_the_floor(monkeypatch):
    # The fixture run charges 0.0068, so the floor here sits above it.
    guard = RatedGuard([], (Decimal('0.0001'), Decimal('0.0017'), Decimal('0.011')))
    assert collect(monkeypatch, guard, ['solo']) == ['0.011']


def test_a_guard_without_rates_keeps_the_fixed_limit(monkeypatch):
    guard = lifecycle_tests.Guard([])
    assert collect(monkeypatch, guard, HANDLES) == ['0.12']



# --- End to end: the real collector through the real guard to settlement (production-shaped policy) ---

def run_through(monkeypatch, observed, handles):
    guard, store = apify_guard(observed)
    sent = []

    def handle(request):
        if request.method == 'POST':
            sent.append(request.url.params['maxTotalChargeUsd'])
            return httpx.Response(201, json={'data': lifecycle_tests.run_body('RUNNING')})
        if request.url.path.endswith('/dataset/items'):
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={'data': lifecycle_tests.run_body()})

    lifecycle_tests.wire(monkeypatch, handle)
    # The guard's clock is the fixture's NOW; the lifecycle stamps the submission with its own clock.
    from src.zeitgeist import provider_lifecycle
    monkeypatch.setattr(provider_lifecycle, 'utc_now', lambda: NOW.isoformat().replace('+00:00', 'Z'))
    collectors.fetch_instagram_batch(handles, 'fixture-private-key', spend_guard=guard, actor_build='0.0.648')
    return sent, held(store), store


@pytest.mark.parametrize('observed,charged,limit', [
    (None, '0.0017', '0.12'),
    (priced({'post:FREE': '0.00221'}), '0.00221', '0.13481'),
])
def test_a_collection_runs_through_the_real_guard_and_settles(monkeypatch, observed, charged, limit):
    sent, record, store = run_through(monkeypatch, observed, HANDLES)
    assert sent == [limit]
    assert record['request_context'] == context(charged)
    assert record['max_gross_usd'] == limit
    assert record['status'] == 'settled' and record['actual_gross_usd'] == '0.0068'


def test_after_a_rollback_a_hold_with_rates_still_counts_without_the_terms(monkeypatch):
    sent, record, store = run_through(monkeypatch, priced({'post:FREE': '0.002'}), HANDLES)
    ledger = store.mutate(lambda d: copy.deepcopy(d))
    ledger['providers']['apify'].pop('approved_pricing_terms')
    assert auto.month_spend_increase_usd(ledger) == Decimal('0.018')
