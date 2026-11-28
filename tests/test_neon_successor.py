"""Pure Neon successor builder on the installed Hydrogen policy, and the receipt chain; synthetic data only."""
import copy
import pytest
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore, SpendBlocked
from src.zeitgeist.spend_guard import EMBEDDING_2_OPERATION, HYDROGEN_OPERATION, NEON_OPERATION
from src.zeitgeist.spend_successor import (
    FLASH_OPERATION, NEON_BASE_POLICY_SHA256, build_neon_successor, build_successor, field_diff, neon_cap_usd,
    neon_operation, successor_ledger)
from tests.test_apify_build_repair import receipts
from tests.test_automatic_spend import NOW
from tests.test_spend_successor import KW as HYDROGEN_KW, hold, installed
from tests.test_spend_successor_chain import Client, OLD, SUCCESSOR

TARIFF = 'e' * 64
NEON_KW = dict(neon_model_version='synthetic-neon-version', neon_tokens_per_image=1091, neon_max_output_tokens=6000,
               neon_thinking_level='medium', tariff_sha256=TARIFF)


def hydrogen_installed():
    """The 22 Nov successor: Hydrogen and embedding-2 added to the synthetic installed policy."""
    ledger0, p0 = installed()
    p1 = build_successor(p0, **HYDROGEN_KW, include_embedding_2=True)
    return successor_ledger(ledger0, p1, auto.digest(p1)), p1


def test_the_base_is_the_installed_hydrogen_successor():
    assert NEON_BASE_POLICY_SHA256 == '6657e657bd261d5a76ffd939d16f0dc1b817e847bfa50a3ce1f39881d102cc96'


@pytest.mark.parametrize('tokens, cap', [(4000, '0.256608'), (6000, '0.286608'), (8000, '0.316608'),
                                         (10000, '0.346608'), (15000, '0.421608')])
def test_neon_cap_is_the_full_window_at_neon_rates(tokens, cap):
    assert neon_cap_usd(tokens) == cap


def test_neon_successor_adds_only_neon_and_the_tariff_digest():
    _, p1 = hydrogen_installed()
    p2 = build_neon_successor(p1, **NEON_KW)
    assert [line.split(' ')[1].rstrip(':') for line in field_diff(p1, p2)] == [
        'providers.gemini.account.operation_caps_usd.generate_content:gemini-4-neon',
        'providers.gemini.account.request_limits.generate_content:gemini-4-neon',
        'providers.gemini.price_snapshot_sha256']
    old, new = p1['providers']['gemini'], p2['providers']['gemini']
    for operation in (FLASH_OPERATION, HYDROGEN_OPERATION, EMBEDDING_2_OPERATION):
        assert new['account']['request_limits'][operation] == old['account']['request_limits'][operation]
        assert new['account']['operation_caps_usd'][operation] == old['account']['operation_caps_usd'][operation]
    assert new['price_snapshot_sha256'] == TARIFF
    for key in ('cash_ceiling_usd', 'approval_reference', 'approved_at', 'valid_until'):
        assert p2[key] == p1[key]
    assert p2['cash_ceiling_usd'] == '10.00'
    assert new['bounded_auto_reload_attestation'] == old['bounded_auto_reload_attestation']
    assert auto.validate_policy(p2, auto.digest(p2), NOW) == p2


def test_neon_limits_exact():
    _, p1 = hydrogen_installed()
    flash = p1['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]
    operation, cap, limits = neon_operation(flash, model_version='synthetic-neon-version', tokens_per_image=1091,
                                            max_output_tokens=6000, thinking_level='medium')
    expected = dict(flash, input_token_usd_per_million='3.00', output_token_usd_per_million='15.00',
                    max_output_tokens=6000, thinking_level='medium', image_tokens_per_image=1091,
                    allowed_response_model_versions=['synthetic-neon-version'])
    assert (operation, cap) == (NEON_OPERATION, '0.286608') and limits == expected
    account = build_neon_successor(p1, **NEON_KW)['providers']['gemini']['account']
    assert account['request_limits'][NEON_OPERATION] == expected
    assert account['operation_caps_usd'][NEON_OPERATION] == '0.286608'


def test_neon_successor_is_deterministic_and_leaves_its_input_alone():
    _, p1 = hydrogen_installed()
    before = copy.deepcopy(p1)
    assert auto.digest(build_neon_successor(p1, **NEON_KW)) == auto.digest(build_neon_successor(p1, **NEON_KW))
    assert auto.digest(build_neon_successor(p1, **dict(NEON_KW, neon_model_version='other'))) != auto.digest(
        build_neon_successor(p1, **NEON_KW))
    assert p1 == before


@pytest.mark.parametrize('tokens', [4000, 6000, 8000, 10000, 15000])
@pytest.mark.parametrize('level', ['high', 'medium'])
def test_every_ladder_value_and_level_builds_a_valid_policy(tokens, level):
    _, p1 = hydrogen_installed()
    p2 = build_neon_successor(p1, **dict(NEON_KW, neon_max_output_tokens=tokens, neon_thinking_level=level))
    assert auto.validate_policy(p2, auto.digest(p2), NOW) == p2
    assert p2['providers']['gemini']['account']['operation_caps_usd'][NEON_OPERATION] == neon_cap_usd(tokens)


@pytest.mark.parametrize('change', [
    dict(neon_model_version='Bad Version'), dict(neon_model_version=''), dict(neon_model_version='a' * 65),
    dict(neon_tokens_per_image=0), dict(neon_tokens_per_image=4097), dict(neon_tokens_per_image=True),
    dict(neon_tokens_per_image='1091'), dict(neon_max_output_tokens=5000), dict(neon_max_output_tokens='6000'),
    dict(neon_max_output_tokens=6000.0), dict(neon_max_output_tokens=None), dict(neon_thinking_level='low'),
    dict(neon_thinking_level='xhigh'), dict(neon_thinking_level=None), dict(tariff_sha256='E' * 64),
    dict(tariff_sha256='e' * 63)])
def test_invalid_inputs_raise_naming_neon(change):
    _, p1 = hydrogen_installed()
    with pytest.raises(RuntimeError) as caught:
        build_neon_successor(p1, **dict(NEON_KW, **change))
    assert 'Hydrogen' not in str(caught.value)


def test_neon_preconditions():
    _, p1 = hydrogen_installed()
    with pytest.raises(RuntimeError, match='already has the Neon operation'):
        build_neon_successor(build_neon_successor(p1, **NEON_KW), **NEON_KW)
    del p1['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]
    with pytest.raises(RuntimeError, match='no Flash limits'):
        build_neon_successor(p1, **NEON_KW)


def test_neon_successor_ledger_adds_neon_and_keeps_holds():
    ledger1, p1 = hydrogen_installed()
    ledger1['reservations'] = {'open': hold(NOW), 'h': hold(NOW, operation=HYDROGEN_OPERATION)}
    p2 = build_neon_successor(p1, **NEON_KW)
    after = successor_ledger(ledger1, p2, auto.digest(p2))
    assert after['reservations'] == ledger1['reservations']
    assert sorted(line.split(' ')[1].rstrip(':') for line in field_diff(ledger1, after)) == [
        'automatic_policy_sha256',
        'providers.gemini.operation_caps_usd.generate_content:gemini-4-neon',
        'providers.gemini.request_limits.generate_content:gemini-4-neon']


NEON_SUCCESSOR = {'month': '2026-11', 'old_policy_sha256': 'e', 'new_policy_sha256': 'f'}
NEON_ROLLBACK = {'month': '2026-11', 'old_policy_sha256': 'f', 'new_policy_sha256': 'e'}


def test_neon_receipt_ids_follow_the_hydrogen_pair():
    assert runtime.NEON_SUCCESSOR_RECEIPT_ID == 'spend_successor_20261125'
    assert runtime.NEON_ROLLBACK_RECEIPT_ID == 'spend_rollback_20261125'
    assert runtime.POLICY_REPAIR_RECEIPTS == (
        'price_repair_20261103', 'apify_build_repair_20261106', 'price_repair_20261113',
        'spend_successor_20261122', 'spend_rollback_20261122', 'spend_successor_20261125', 'spend_rollback_20261125',
        'spend_successor_20261128', 'spend_rollback_20261128',
        'spend_successor_20261128_cushion', 'spend_rollback_20261128_cushion')


def test_neon_receipt_extends_the_chain_after_the_hydrogen_install(monkeypatch):
    store = receipts(monkeypatch, {**OLD, 'spend_successor_20261122': SUCCESSOR,
                                   'spend_successor_20261125': NEON_SUCCESSOR})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'f'


def test_neon_rollback_returns_to_the_hydrogen_successor(monkeypatch):
    store = receipts(monkeypatch, {**OLD, 'spend_successor_20261122': SUCCESSOR,
                                   'spend_successor_20261125': NEON_SUCCESSOR,
                                   'spend_rollback_20261125': NEON_ROLLBACK})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'e'


def test_absent_neon_receipts_leave_the_hydrogen_chain(monkeypatch):
    store = receipts(monkeypatch, {**OLD, 'spend_successor_20261122': SUCCESSOR})
    assert runtime._successor_policy(store, 'a', '2026-11') == 'e'


@pytest.mark.parametrize('document_id', ['spend_successor_20261125', 'spend_rollback_20261125'])
def test_store_reads_the_neon_receipt_ids(document_id):
    PrivateFirestoreStore(Client()).read(document_id)


@pytest.mark.parametrize('document_id', ['spend_successor_20261124', 'spend_rollback_20261126', 'spend_successor_2026112'])
def test_store_still_refuses_other_receipt_ids(document_id):
    with pytest.raises(SpendBlocked, match='Unapproved private spend document'):
        PrivateFirestoreStore(Client()).read(document_id)
