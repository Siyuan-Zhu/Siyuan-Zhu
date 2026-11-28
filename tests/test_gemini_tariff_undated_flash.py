"""3.8 Flash after Google drops the dated wording: one current rate per row (SIY-198).

Every page here is a recorded page. The undated page is the 2026-11-22 recording with only the
3.8 Flash Standard rows rewritten to the single 2027 rate; no page is fetched.
"""
from pathlib import Path

import pytest

from src.zeitgeist.automatic_spend import digest
from src.zeitgeist.gemini_tariff import normalize_gemini_tariff
from src.zeitgeist.spend_guard import SpendBlocked

FLASH = 'generate_content:gemini-3.8-flash'
FIXTURES = Path(__file__).parent / 'fixtures'
DATED_INPUT = '* \u00240.75 through December 31, 2026.   \n  \u00241.50 starting January 1, 2027.\n'
DATED_OUTPUT = '* \u00243.75 through December 31, 2026.   \n  \u00247.50 starting January 1, 2027.\n'


def recorded(name='gemini-pricing-20261122.txt'):
    return (FIXTURES / name).read_text()


def in_flash_standard(page, old, new):
    """Replace the first `old` after the 3.8 Flash heading: its Standard tab comes first."""
    head, heading, tail = page.partition('## Gemini 3.8 Flash  {% id="gemini-3.8-flash" %}')
    assert heading and old in tail
    return head + heading + tail.replace(old, new, 1)


def undated(page=None, input_row='* \u00241.50\n', output_row='* \u00247.50\n'):
    page = in_flash_standard(page or recorded(), DATED_INPUT, input_row)
    return in_flash_standard(page, DATED_OUTPUT, output_row)


# The contract digests of the two recorded dated pages before SIY-198. The installed policy's tariff
# pin is such a digest, so reading the dated page must not change by a byte.
@pytest.mark.parametrize('name,expected', [
    ('gemini-pricing-20261122.txt', '40b869e255e541afaae9918a3346e90d4071c3f7b64fd71c0cbec14d56c45583'),
    ('gemini-pricing-markdoc.txt', '35df15420dc97e3064ee2a669a1b89dff49d142a79a09bd20375a41d923c63f2'),
])
def test_the_dated_pages_still_read_to_the_same_contract(name, expected):
    assert digest(normalize_gemini_tariff(recorded(name))) == expected


def test_the_undated_page_reads_flash_at_its_single_current_rate():
    dated = normalize_gemini_tariff(recorded())
    contract = normalize_gemini_tariff(undated())
    assert contract['operations'][FLASH] == {
        'model': 'gemini-3.8-flash', 'input_token_usd_per_million': '1.5',
        'output_token_usd_per_million': '7.5', 'output_includes_thinking': True,
        'published_rate_schedule': 'current_only'}
    for operation, terms in dated['operations'].items():
        if operation != FLASH:
            assert contract['operations'][operation] == terms


@pytest.mark.parametrize('rows', [
    {'input_row': '* \u00241.50\n'},
    {'output_row': '* \u00247.50\n'},
    {'input_row': '* \u00241.50, prompts <= 200k tokens\n'},
    {'output_row': '* \u00247.50 starting January 1, 2027.\n'},
    {'input_row': '* Free of charge\n'},
    {'input_row': '* \u00240\n'},
])
def test_a_mixed_conditional_or_free_flash_row_is_refused(rows):
    page = recorded()
    if 'input_row' in rows:
        page = in_flash_standard(page, DATED_INPUT, rows['input_row'])
    if 'output_row' in rows:
        page = in_flash_standard(page, DATED_OUTPUT, rows['output_row'])
    with pytest.raises(SpendBlocked):
        normalize_gemini_tariff(page)
