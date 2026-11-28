"""The guard admits 3.8 Flash at its 2027 Standard tariff, approved through 2027 (SIY-198)."""
from decimal import Decimal

import pytest

from src.zeitgeist.spend_guard import SpendBlocked
from tests.test_gemini38_contract import NOW, OP, guard, request, thumbnail

JUNE_2027 = NOW.replace(year=2027, month=6, day=1)


def priced(budget, input_rate, output_rate, valid_until):
    limits = budget.ledger.document['providers']['gemini']['request_limits'][OP]
    limits.update(input_token_usd_per_million=input_rate, output_token_usd_per_million=output_rate,
                  pricing_valid_until=valid_until)
    return budget


def counted_cap(budget):
    bounded = budget.bounded_genai_request(OP, request(['Synthetic batch', thumbnail()]))
    return budget.counted_genai_cap(OP, bounded, lambda **_: type('Count', (), {'total_tokens': 11})())[0]


@pytest.mark.parametrize('now', [NOW, JUNE_2027])
def test_the_2027_tariff_admits_flash_through_2027_at_its_own_rates(now):
    budget = priced(guard(now), '1.50', '7.50', '2027-12-31T23:59:59Z')
    assert counted_cap(budget) == (Decimal(11 + 1120) * Decimal('1.50') + Decimal(1500) * Decimal('7.50')) / 1000000


def test_the_2026_tariff_stops_at_its_published_end():
    budget = priced(guard(JUNE_2027), '0.75', '3.75', '2026-12-31T23:59:59Z')
    with pytest.raises(SpendBlocked, match='pricing'):
        budget.bounded_genai_request(OP, request())


@pytest.mark.parametrize('rates,valid_until', [
    (('1.50', '7.50'), '2028-01-01T00:00:00Z'),
    (('0.75', '3.75'), '2027-01-01T00:00:00Z'),
    (('0.75', '7.50'), '2026-12-31T23:59:59Z'),
    (('1.50', '3.75'), '2026-12-31T23:59:59Z'),
    (('1.65', '8.25'), '2027-12-31T23:59:59Z'),
])
def test_a_rate_pair_or_validity_outside_an_approved_tariff_is_refused(rates, valid_until):
    budget = priced(guard(), *rates, valid_until)
    with pytest.raises(SpendBlocked):
        budget.bounded_genai_request(OP, request())
