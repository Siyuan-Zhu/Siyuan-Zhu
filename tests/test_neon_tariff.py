"""Neon's price section on the saved page: parsed at 3.00/15.00, optional, and harmless to the installed policy."""
from pathlib import Path
import pytest
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.automatic_spend import digest
from src.zeitgeist.gemini_tariff import MODELS, NEON_SECTION, OPTIONAL_SECTIONS, normalize_gemini_tariff
from src.zeitgeist.spend_guard import NEON_OPERATION
from src.zeitgeist.spend_successor import build_successor, successor_ledger
from tests.test_automatic_spend import NOW
from tests.test_gemini_tariff_optional_sections import (
    EMBEDDING_001, EMBEDDING_2, FLASH, HYDROGEN, ROW_BREAK, USD, assert_old_models_at_todays_rates, in_section,
    once, page)
from tests.test_spend_successor import installed
from tests.test_successor_compatibility import observation, refreshed

NEON = 'generate_content:gemini-4-neon'
NEON_HEADING = '## Gemini 4 Neon  '
# The tariff digest the 22 Nov Hydrogen install pinned (price_snapshot_sha256 of policy 6657e657): this saved page.
HYDROGEN_INSTALL_TARIFF = '204ae20efcf48bc500fcb25c6b0fabb0979bc7566cfcd16b512f1e3df0e1f4c2'


def without_neon():
    body = page()
    start = body.index(NEON_HEADING)
    end = body.index('\n## ', start + 1)
    return body[:start] + body[end + 1:]


def test_neon_is_an_optional_section():
    assert NEON_SECTION == 'gemini-4-neon' and MODELS[NEON_SECTION] == 'gemini-4-neon'
    assert NEON_SECTION in OPTIONAL_SECTIONS
    assert NEON_OPERATION in auto.OPTIONAL_OPERATIONS


def test_the_optional_operations_add_only_neon():
    assert auto.OPTIONAL_OPERATIONS == frozenset({'generate_content:gemini-4-hydrogen', 'embed_content:gemini-embedding-2',
                                                  NEON})


def test_neon_standard_rates_from_the_saved_page():
    contract = normalize_gemini_tariff(page())
    assert contract['operations'][NEON] == {'model': 'gemini-4-neon', 'input_token_usd_per_million': '3',
        'output_token_usd_per_million': '15', 'output_includes_thinking': True,
        'published_rate_schedule': 'current_only'}
    assert 'gemini-4-neon' in contract['standard_conditions']
    assert contract['operations'][HYDROGEN]['input_token_usd_per_million'] == '1.5'
    assert contract['operations'][HYDROGEN]['output_token_usd_per_million'] == '7.5'
    assert set(contract['operations']) == {FLASH, HYDROGEN, NEON, EMBEDDING_2, EMBEDDING_001}
    assert_old_models_at_todays_rates(contract)


def test_a_page_without_neon_gives_the_contract_the_hydrogen_install_pinned():
    contract = normalize_gemini_tariff(without_neon())
    assert digest(contract) == HYDROGEN_INSTALL_TARIFF
    assert NEON not in contract['operations'] and 'gemini-4-neon' not in contract['standard_conditions']
    assert digest(normalize_gemini_tariff(page())) != HYDROGEN_INSTALL_TARIFF


EDITS = {
    'neon_dated_price': once('* ' + USD + '3.00\n', '* ' + USD + '3.00 through December 31, 2026.   \n  ' + USD + '6.00 starting January 1, 2027.\n'),
    'neon_tiered_input': once('* ' + USD + '3.00\n', '* ' + USD + '3.00, prompts &amp;lt;= 200k tokens   \n  ' + USD + '6.00, prompts &amp;gt; 200k tokens\n'),
    'neon_new_price_row': once('* Used to improve our products', '* Long context surcharge\n  * Not available\n  * ' + USD + '0.50' + ROW_BREAK + '* Used to improve our products'),
    'neon_row_label_footnote': once('* Context caching price', '* Context caching price{% sup %}*{% /sup %}'),
    'neon_preview_code': once('`gemini-4-neon`', '`gemini-4-neon-preview`'),
    'neon_free_promotion': once('* ' + USD + '3.00\n', '* Free of charge\n'),
    'neon_price_missing': once('* ' + USD + '3.00\n', '* \n'),
}


@pytest.mark.parametrize('name', sorted(EDITS))
def test_an_edit_in_the_neon_section_leaves_only_neon_out(name):
    contract = normalize_gemini_tariff(in_section(NEON_HEADING, EDITS[name]))
    full = normalize_gemini_tariff(page())
    assert NEON not in contract['operations']
    assert_old_models_at_todays_rates(contract)
    for operation in (HYDROGEN, EMBEDDING_2):
        assert contract['operations'][operation] == full['operations'][operation]
    assert digest(contract) == HYDROGEN_INSTALL_TARIFF


def _hydrogen_successor_as_installed():
    """The 22 Nov shape: Hydrogen and embedding-2 added, the page without Neon pinned."""
    ledger0, p0 = installed()
    ledger0.update(credit_only_providers=p0['credit_only_providers'], automatic_policy_sha256=auto.digest(p0))
    p1 = build_successor(p0, hydrogen_model_version='gemini-4-hydrogen', hydrogen_tokens_per_image=1091,
                         hydrogen_max_output_tokens=6000, hydrogen_thinking_level='high',
                         tariff_sha256=HYDROGEN_INSTALL_TARIFF, include_embedding_2=True)
    return successor_ledger(ledger0, p1, auto.digest(p1)), p1


def test_the_installed_hydrogen_policy_refreshes_on_the_rate_path_with_neon_published():
    ledger1, p1 = _hydrogen_successor_as_installed()
    obs = observation(p1, embedding_2_methods=['embedContent'])
    contract = normalize_gemini_tariff(page())
    obs['price_snapshots']['gemini'].update(contract=contract, sha256=digest(contract))
    account = refreshed(ledger1, p1, obs)['providers']['gemini']
    assert account['price_observed_sha256'] == digest(contract) != HYDROGEN_INSTALL_TARIFF
    assert 'disabled_operations' not in account and 'price_band_audit' not in account


def test_the_installed_hydrogen_policy_takes_the_hash_path_on_the_page_without_neon():
    ledger1, p1 = _hydrogen_successor_as_installed()
    obs = observation(p1, embedding_2_methods=['embedContent'])
    contract = normalize_gemini_tariff(without_neon())
    obs['price_snapshots']['gemini'].update(contract=contract, sha256=digest(contract))
    account = refreshed(ledger1, p1, obs)['providers']['gemini']
    assert 'price_observed_sha256' not in account and 'disabled_operations' not in account
