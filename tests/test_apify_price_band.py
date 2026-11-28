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
    apify['actor_pricing_sha256'] = auto.digest(TERMS)
    if with_terms:
        apify['actor_pricing_terms'] = copy.deepcopy(TERMS)
    ledger['automatic_policy_sha256'] = auto.digest(policy)
    observation['apify']['pricing'] = copy.deepcopy(TERMS if observed is None else observed)
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
    with pytest.raises(SpendBlocked, match=CHANGED):
        refresh(*setup({**TERMS, 'started_at': '2026-12-01T00:00:00.000Z'}, with_terms=False))


def test_refresh_above_band_blocks():
    with pytest.raises(SpendBlocked, match=CHANGED):
        refresh(*setup(priced({'post:FREE': '0.0023'})))


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
