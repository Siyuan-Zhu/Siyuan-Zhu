"""Pure successor builder: exact Hydrogen additions, redacted diffs, untouched holds."""
import copy
from datetime import timedelta
import pytest
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import HYDROGEN_OPERATION
from src.zeitgeist.spend_successor import (
    EMBEDDING_001_OPERATION, FLASH_OPERATION, assert_no_hold_in_flight, build_successor,
    field_diff, hydrogen_operation, rollback_ledger, successor_ledger)
from tests import test_gemini38_contract
from tests.test_automatic_spend import NOW, fixtures, with_approved_reload

TARIFF = 'f' * 64
KW = dict(hydrogen_model_version='synthetic-hydrogen-version', hydrogen_tokens_per_image=1088, hydrogen_max_output_tokens=4000,
          hydrogen_thinking_level='high', tariff_sha256=TARIFF)


def installed():
    ledger, policy, _ = fixtures()
    policy = with_approved_reload(policy)
    policy.update(valid_until='2026-12-31T23:59:59Z')
    gemini = policy['providers']['gemini']
    gemini.update(price_evidence_mode='dynamic_snapshot', price_valid_until='2026-12-31T23:59:59Z')
    account = gemini['account']
    account['request_limits'] = {
        FLASH_OPERATION: dict(test_gemini38_contract.limits(), allowed_response_model_versions=['gemini-3.8-flash']),
        EMBEDDING_001_OPERATION: {'pricing_reference': 'https://ai.google.dev/gemini-api/docs/pricing',
            'count_tokens_reference': 'https://ai.google.dev/gemini-api/docs/billing#is-gettokens-billed',
            'max_input_bytes': 2000000, 'max_input_tokens': 204800,
            'input_token_usd_per_million': '.15', 'output_token_usd_per_million': '0'}}
    account['operation_caps_usd'] = {FLASH_OPERATION: '0.054777', EMBEDDING_001_OPERATION: '0.03072'}
    ledger['providers']['gemini']['operation_caps_usd'] = copy.deepcopy(account['operation_caps_usd'])
    ledger['providers']['gemini']['request_limits'] = copy.deepcopy(account['request_limits'])
    ledger.update(cash_ceiling_usd='10.00', automatic_policy_sha256=auto.digest(policy))
    return ledger, policy


def test_successor_adds_only_hydrogen():
    _, p0 = installed()
    p1 = build_successor(p0, **KW)
    assert [line.split(' ')[1].rstrip(':') for line in field_diff(p0, p1)] == [
        'providers.gemini.account.operation_caps_usd.generate_content:gemini-4-hydrogen',
        'providers.gemini.account.request_limits.generate_content:gemini-4-hydrogen',
        'providers.gemini.price_snapshot_sha256']
    assert p1['cash_ceiling_usd'] == '10.00'
    assert (p1['providers']['gemini']['bounded_auto_reload_attestation']
            == p0['providers']['gemini']['bounded_auto_reload_attestation'])


def test_hydrogen_limits_exact():
    _, p0 = installed()
    flash = p0['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]
    operation, cap, limits = hydrogen_operation(flash, model_version='synthetic-hydrogen-version', tokens_per_image=1088,
                                              max_output_tokens=4000, thinking_level='high')
    expected = dict(flash, input_token_usd_per_million='1.50', output_token_usd_per_million='7.50',
                    max_output_tokens=4000, thinking_level='high', image_tokens_per_image=1088,
                    allowed_response_model_versions=['synthetic-hydrogen-version'])
    assert (operation, cap) == (HYDROGEN_OPERATION, '0.128304')
    assert limits == expected
    assert flash['input_token_usd_per_million'] == '0.75'  # input untouched
    p1 = build_successor(p0, **KW)
    assert p1['providers']['gemini']['account']['request_limits'][HYDROGEN_OPERATION] == expected
    assert p1['providers']['gemini']['account']['operation_caps_usd'][HYDROGEN_OPERATION] == '0.128304'


def test_successor_is_deterministic_and_valid():
    _, p0 = installed()
    before = copy.deepcopy(p0)
    p1, again = build_successor(p0, **KW), build_successor(p0, **KW)
    assert auto.digest(p1) == auto.digest(again)
    assert auto.digest(build_successor(p0, **dict(KW, hydrogen_model_version='other-version'))) != auto.digest(p1)
    assert auto.validate_policy(p1, auto.digest(p1), NOW) == p1
    assert p0 == before


@pytest.mark.parametrize('change', [
    dict(hydrogen_model_version='Bad Version'), dict(hydrogen_model_version=''),
    dict(hydrogen_model_version='-lead'), dict(hydrogen_model_version='a' * 65),
    dict(hydrogen_tokens_per_image=0), dict(hydrogen_tokens_per_image=4097),
    dict(hydrogen_tokens_per_image=True), dict(hydrogen_tokens_per_image='1088'),
    dict(hydrogen_max_output_tokens=5000), dict(hydrogen_max_output_tokens=None),
    dict(hydrogen_max_output_tokens=True), dict(hydrogen_max_output_tokens=3999),
    dict(hydrogen_max_output_tokens=6001), dict(hydrogen_max_output_tokens='6000'),
    dict(hydrogen_max_output_tokens=6000.0), dict(hydrogen_max_output_tokens=7000),
    dict(hydrogen_max_output_tokens=15001), dict(hydrogen_max_output_tokens=12000),
    dict(hydrogen_thinking_level='low'), dict(hydrogen_thinking_level='xhigh'),
    dict(hydrogen_thinking_level=None), dict(hydrogen_thinking_level=True), dict(hydrogen_thinking_level=''),
    dict(tariff_sha256='F' * 64), dict(tariff_sha256='f' * 63), dict(tariff_sha256='g' * 64)])
def test_invalid_inputs_raise(change):
    _, p0 = installed()
    with pytest.raises(RuntimeError):
        build_successor(p0, **dict(KW, **change))


def test_hydrogen_6000_derives_cap_and_limits():
    _, p0 = installed()
    flash = p0['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]
    operation, cap, limits = hydrogen_operation(flash, model_version='synthetic-hydrogen-version',
                                                tokens_per_image=1088, max_output_tokens=6000,
                                                thinking_level='high')
    assert (operation, cap) == (HYDROGEN_OPERATION, '0.143304') and limits['max_output_tokens'] == 6000
    p1 = build_successor(p0, **dict(KW, hydrogen_max_output_tokens=6000))
    account = p1['providers']['gemini']['account']
    assert account['operation_caps_usd'][HYDROGEN_OPERATION] == '0.143304'
    assert account['request_limits'][HYDROGEN_OPERATION]['max_output_tokens'] == 6000
    assert auto.validate_policy(p1, auto.digest(p1), NOW) == p1
    assert auto.digest(p1) != auto.digest(build_successor(p0, **KW))


@pytest.mark.parametrize('tokens, cap', [(4000, '0.128304'), (6000, '0.143304'), (8000, '0.158304'),
                                         (10000, '0.173304'), (15000, '0.210804')])
@pytest.mark.parametrize('level', ['high', 'medium'])
def test_every_ladder_value_and_level_builds_a_valid_policy(tokens, cap, level):
    _, p0 = installed()
    p1 = build_successor(p0, **dict(KW, hydrogen_max_output_tokens=tokens, hydrogen_thinking_level=level))
    account = p1['providers']['gemini']['account']
    assert account['operation_caps_usd'][HYDROGEN_OPERATION] == cap
    assert account['request_limits'][HYDROGEN_OPERATION]['thinking_level'] == level
    assert account['request_limits'][HYDROGEN_OPERATION]['max_output_tokens'] == tokens
    assert auto.validate_policy(p1, auto.digest(p1), NOW) == p1


@pytest.mark.parametrize('missing', ['hydrogen_max_output_tokens', 'hydrogen_thinking_level'])
def test_builder_requires_output_tokens_and_level_keywords(missing):
    _, p0 = installed()
    with pytest.raises(TypeError):
        build_successor(p0, **{k: v for k, v in KW.items() if k != missing})


def test_installed_policy_preconditions():
    _, p0 = installed()
    once = build_successor(p0, **KW)
    with pytest.raises(RuntimeError):
        build_successor(once, **KW)
    del p0['providers']['gemini']['account']['request_limits'][FLASH_OPERATION]
    with pytest.raises(RuntimeError):
        build_successor(p0, **KW)


def hold(created, status='reserved', **kw):
    return dict(dict(provider='gemini', operation=FLASH_OPERATION, max_gross_usd='0.05',
                     created_at=created.isoformat(), status=status), **kw)


def test_successor_ledger_keeps_reservations_byte_identical():
    ledger, p0 = installed()
    ledger['reservations'] = {'open': hold(NOW), 'done': hold(NOW, 'settled', actual_gross_usd='0.01', billing_reference='r')}
    p1 = build_successor(p0, **KW)
    after = successor_ledger(ledger, p1, auto.digest(p1))
    assert auto.digest(after['reservations']) == auto.digest(ledger['reservations'])
    paths = [line.split(' ')[1].rstrip(':') for line in field_diff(ledger, after)]
    assert sorted(paths) == ['automatic_policy_sha256',
        'providers.gemini.operation_caps_usd.generate_content:gemini-4-hydrogen',
        'providers.gemini.request_limits.generate_content:gemini-4-hydrogen']
    assert after['providers']['bfl'] == ledger['providers']['bfl']
    assert after['providers']['apify'] == ledger['providers']['apify']


def test_field_diff_redacts():
    for key, new in (('billing_account_id', 'AAAAAA-BBBBBB-CCCCCC'), ('credential_fingerprint', 'a' * 16)):
        lines = field_diff({key: 'old-value-zzz'}, {key: new})
        assert len(lines) == 1 and '[redacted]' in lines[0]
        assert new not in lines[0] and 'old-value-zzz' not in lines[0]
    add = field_diff({}, {'x': {'reference': 'secret-ref'}})
    assert add == ['+ x = {"reference":"[redacted]"}']


def test_field_diff_summarizes_reservations_and_marks_removals():
    lines = field_diff({'reservations': {}, 'a': 1, 'b': 2}, {'reservations': {'k': {'handle': 'zzz'}}, 'a': 3})
    assert lines == ['- b', '~ a: 1 -> 3', '~ reservations (1 records)']


def test_hold_in_flight():
    ledger, _ = installed()
    def check(**records):
        ledger['reservations'] = records
        assert_no_hold_in_flight(ledger, now=NOW, duration_seconds=300)
    check(a=hold(NOW - timedelta(seconds=400)))
    check(a=hold(NOW - timedelta(seconds=10), 'settled'))
    with pytest.raises(RuntimeError):
        check(a=hold(NOW - timedelta(seconds=100)))
    with pytest.raises(RuntimeError):
        check(a=hold(NOW - timedelta(seconds=400), submitted_at=(NOW - timedelta(seconds=100)).isoformat()))


def test_rollback_ledger_restores_old_static_and_keeps_holds():
    ledger, p0 = installed()
    p1 = build_successor(p0, **KW)
    ledger['reservations'] = {'done': hold(NOW, 'settled', actual_gross_usd='0.1', billing_reference='r', operation=HYDROGEN_OPERATION)}
    forward = successor_ledger(ledger, p1, auto.digest(p1))
    back = rollback_ledger(forward, p0, auto.digest(p0))
    assert back['automatic_policy_sha256'] == auto.digest(p0)
    gem = back['providers']['gemini']
    assert gem['operation_caps_usd'] == p0['providers']['gemini']['account']['operation_caps_usd']
    assert gem['request_limits'] == p0['providers']['gemini']['account']['request_limits']
    assert back['reservations'] == ledger['reservations']
