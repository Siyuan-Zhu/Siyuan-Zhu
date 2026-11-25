"""A new model's price section that does not read cleanly is left out; the old models still parse."""
from pathlib import Path
import pytest
from src.zeitgeist.automatic_spend import digest
from src.zeitgeist.gemini_tariff import normalize_gemini_tariff
from src.zeitgeist.spend_guard import SpendBlocked

FLASH = 'generate_content:gemini-3.8-flash'
EMBEDDING_001 = 'embed_content:gemini-embedding-001'
HYDROGEN = 'generate_content:gemini-4-hydrogen'
EMBEDDING_2 = 'embed_content:gemini-embedding-2'
HYDROGEN_HEADING = '## Gemini 4 Hydrogen  '
FLASH_HEADING = '## Gemini 3.8 Flash  '
EMBEDDING_2_HEADING = '## Gemini Embedding 2  '
USD = chr(36)
ROW_BREAK = '\n\n  ---\n\n  '


def page():
    return (Path(__file__).parent / 'fixtures/gemini-pricing-20261122.txt').read_text()


def in_section(heading, change):
    """Apply change only inside the named model section of the saved page."""
    body = page()
    start = body.index(heading)
    end = body.index('\n## ', start + 1)
    section = change(body[start:end])
    assert section != body[start:end]
    return body[:start] + section + body[end:]


def once(old, new):
    def change(section):
        assert old in section
        return section.replace(old, new, 1)
    return change


EDITS = {
    'hydrogen_dated_price': (HYDROGEN, HYDROGEN_HEADING, once(
        '* ' + USD + '1.50\n', '* ' + USD + '1.50 through December 31, 2026.   \n  ' + USD + '3.00 starting January 1, 2027.\n')),
    'hydrogen_tiered_input': (HYDROGEN, HYDROGEN_HEADING, once(
        '* ' + USD + '1.50\n', '* ' + USD + '1.50, prompts &lt;= 200k tokens   \n  ' + USD + '3.00, prompts &gt; 200k tokens\n')),
    'hydrogen_new_price_row': (HYDROGEN, HYDROGEN_HEADING, once(
        '* Used to improve our products', '* Long context surcharge\n  * Not available\n  * ' + USD + '0.50' + ROW_BREAK + '* Used to improve our products')),
    'hydrogen_row_label_footnote': (HYDROGEN, HYDROGEN_HEADING, once(
        '* Context caching price', '* Context caching price{% sup %}*{% /sup %}')),
    'hydrogen_header_footnote': (HYDROGEN, HYDROGEN_HEADING, once(
        '* Paid Tier, per 1M tokens in USD\n', '* Paid Tier, per 1M tokens in USD{% sup %}*{% /sup %}\n')),
    'hydrogen_preview_code': (HYDROGEN, HYDROGEN_HEADING, once(
        '`gemini-4-hydrogen`', '`gemini-4-hydrogen-preview`')),
    'hydrogen_second_code': (HYDROGEN, HYDROGEN_HEADING, once(
        '*[`gemini-4-hydrogen`](/docs/models/gemini-4-hydrogen)*',
        '*[`gemini-4-hydrogen`](/docs/models/gemini-4-hydrogen)* *[`gemini-4-hydrogen-001`](/docs/models/gemini-4-hydrogen-001)*')),
    'hydrogen_free_promotion': (HYDROGEN, HYDROGEN_HEADING, once(
        '* ' + USD + '1.50\n', '* Free of charge\n')),
    'embedding_2_pdf_row': (EMBEDDING_2, EMBEDDING_2_HEADING, once(
        '* Used to improve our products', '* PDF input price\n  * Free of charge\n  * ' + USD + '0.30 (' + USD + '0.0001 per page)' + ROW_BREAK + '* Used to improve our products')),
    'embedding_2_per_unit_note': (EMBEDDING_2, EMBEDDING_2_HEADING, once(
        '* ' + USD + '0.20\n', '* ' + USD + '0.20 (' + USD + '0.00002 per 1,000 characters)\n')),
}


def assert_old_models_at_todays_rates(contract):
    operations = contract['operations']
    assert operations[FLASH]['input_token_usd_per_million'] == '0.75'
    assert operations[FLASH]['output_token_usd_per_million'] == '3.75'
    assert operations[FLASH]['future_input_token_usd_per_million'] == '1.5'
    assert operations[FLASH]['future_output_token_usd_per_million'] == '7.5'
    assert operations[EMBEDDING_001]['input_token_usd_per_million'] == '0.15'
    assert operations[FLASH] == normalize_gemini_tariff(page())['operations'][FLASH]
    assert operations[EMBEDDING_001] == normalize_gemini_tariff(page())['operations'][EMBEDDING_001]


@pytest.mark.parametrize('name', sorted(EDITS))
def test_an_edit_in_a_new_model_section_leaves_only_that_operation_out(name):
    operation, heading, change = EDITS[name]
    contract = normalize_gemini_tariff(in_section(heading, change))
    assert_old_models_at_todays_rates(contract)
    assert operation not in contract['operations']
    assert contract['operations'][({HYDROGEN, EMBEDDING_2} - {operation}).pop()] == normalize_gemini_tariff(page())['operations'][({HYDROGEN, EMBEDDING_2} - {operation}).pop()]


@pytest.mark.parametrize('name', sorted(EDITS))
def test_a_left_out_section_does_not_reach_the_digest(name):
    operation, heading, change = EDITS[name]
    body = page()
    start = body.index(heading)
    end = body.index('\n## ', start + 1)
    without_section = body[:start] + body[end + 1:]
    assert digest(normalize_gemini_tariff(in_section(heading, change))) == digest(normalize_gemini_tariff(without_section))


def test_the_unmodified_page_yields_all_five_operations():
    contract = normalize_gemini_tariff(page())
    assert set(contract['operations']) == {FLASH, HYDROGEN, 'generate_content:gemini-4-neon', EMBEDDING_2, EMBEDDING_001}
    assert_old_models_at_todays_rates(contract)


@pytest.mark.parametrize('change', [
    once('* Used to improve our products', '* Long context surcharge\n  * Not available\n  * ' + USD + '0.50' + ROW_BREAK + '* Used to improve our products'),
    once('* Context caching price', '* Context caching price{% sup %}*{% /sup %}'),
    once('`gemini-3.8-flash`', '`gemini-3.8-flash-preview`'),
    once('* ' + USD + '0.75 through December 31, 2026.', '* ' + USD + '0.75'),
])
def test_an_edit_inside_the_flash_section_still_raises(change):
    with pytest.raises(SpendBlocked):
        normalize_gemini_tariff(in_section(FLASH_HEADING, change))


def test_an_edit_inside_the_embedding_001_section_still_raises():
    with pytest.raises(SpendBlocked):
        normalize_gemini_tariff(in_section('## Gemini Embedding  ', once('* ' + USD + '0.15\n', '* ' + USD + '0.15 per character\n')))
