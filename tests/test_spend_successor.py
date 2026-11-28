"""Pure successor builder: exact Hydrogen additions, redacted diffs, untouched holds."""
import copy
from datetime import timedelta
import pytest
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist.spend_guard import EMBEDDING_2_OPERATION, HYDROGEN_OPERATION
from src.zeitgeist.spend_successor import (
    EMBEDDING_001_OPERATION, FLASH_OPERATION, assert_no_hold_in_flight, build_successor,
    embedding2_operation, field_diff, hydrogen_operation, rollback_ledger, successor_ledger)
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
    ledger.update(cash_ceiling_usd=policy['cash_ceiling_usd'], automatic_policy_sha256=auto.digest(policy))
    return ledger, policy


def test_successor_adds_only_hydrogen():
    _, p0 = installed()
    p1 = build_successor(p0, **KW)
    assert [line.split(' ')[1].rstrip(':') for line in field_diff(p0, p1)] == [
        'providers.gemini.account.operation_caps_usd.generate_content:gemini-4-hydrogen',
        'providers.gemini.account.request_limits.generate_content:gemini-4-hydrogen',
        'providers.gemini.price_snapshot_sha256']
    assert p1['cash_ceiling_usd'] == p0['cash_ceiling_usd']
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


VALIDITY_KEYS = {'pricing_valid_until', 'price_valid_until'}


def test_embedding2_limits_copy_embedding_001_with_exact_rate_and_cap():
    _, p0 = installed()
    e001 = p0['providers']['gemini']['account']['request_limits'][EMBEDDING_001_OPERATION]
    before = copy.deepcopy(e001)
    operation, cap, limits = embedding2_operation(e001)
    assert (operation, cap) == (EMBEDDING_2_OPERATION, '0.04096')
    assert limits == dict(e001, input_token_usd_per_million='0.20', output_dimensionality=768)
    assert type(limits['output_dimensionality']) is int
    # The tariff operation has no published rate schedule, so a validity field would block refresh.
    assert not VALIDITY_KEYS & set(limits)
    assert e001 == before and limits is not e001


def test_successor_with_embedding_2_adds_both_operations():
    _, p0 = installed()
    p1 = build_successor(p0, **KW, include_embedding_2=True)
    assert [line.split(' ')[1].rstrip(':') for line in field_diff(p0, p1)] == [
        'providers.gemini.account.operation_caps_usd.embed_content:gemini-embedding-2',
        'providers.gemini.account.operation_caps_usd.generate_content:gemini-4-hydrogen',
        'providers.gemini.account.request_limits.embed_content:gemini-embedding-2',
        'providers.gemini.account.request_limits.generate_content:gemini-4-hydrogen',
        'providers.gemini.price_snapshot_sha256']
    account = p1['providers']['gemini']['account']
    assert account['operation_caps_usd'][EMBEDDING_2_OPERATION] == '0.04096'
    assert account['operation_caps_usd'][HYDROGEN_OPERATION] == '0.128304'
    assert not VALIDITY_KEYS & set(account['request_limits'][EMBEDDING_2_OPERATION])
    assert auto.validate_policy(p1, auto.digest(p1), NOW) == p1
    assert p1['cash_ceiling_usd'] == p0['cash_ceiling_usd']


def test_without_embedding_2_the_successor_is_the_hydrogen_only_one():
    _, p0 = installed()
    plain = build_successor(p0, **KW)
    assert auto.digest(build_successor(p0, **KW, include_embedding_2=False)) == auto.digest(plain)
    assert EMBEDDING_2_OPERATION not in plain['providers']['gemini']['account']['operation_caps_usd']
    assert EMBEDDING_2_OPERATION not in plain['providers']['gemini']['account']['request_limits']
    assert auto.digest(build_successor(p0, **KW, include_embedding_2=True)) != auto.digest(plain)


def test_embedding_2_preconditions():
    _, p0 = installed()
    present = copy.deepcopy(p0)
    present['providers']['gemini']['account']['operation_caps_usd'][EMBEDDING_2_OPERATION] = '0.04096'
    with pytest.raises(RuntimeError):
        build_successor(present, **KW, include_embedding_2=True)
    missing = copy.deepcopy(p0)
    del missing['providers']['gemini']['account']['request_limits'][EMBEDDING_001_OPERATION]
    with pytest.raises(RuntimeError):
        build_successor(missing, **KW, include_embedding_2=True)
    build_successor(missing, **KW)  # Hydrogen-only does not need embedding-001 limits.


def test_successor_ledger_with_embedding_2_adds_both_operations_and_keeps_holds():
    ledger, p0 = installed()
    ledger['reservations'] = {'open': hold(NOW)}
    p1 = build_successor(p0, **KW, include_embedding_2=True)
    after = successor_ledger(ledger, p1, auto.digest(p1))
    assert after['reservations'] == ledger['reservations']
    assert sorted(line.split(' ')[1].rstrip(':') for line in field_diff(ledger, after)) == [
        'automatic_policy_sha256',
        'providers.gemini.operation_caps_usd.embed_content:gemini-embedding-2',
        'providers.gemini.operation_caps_usd.generate_content:gemini-4-hydrogen',
        'providers.gemini.request_limits.embed_content:gemini-embedding-2',
        'providers.gemini.request_limits.generate_content:gemini-4-hydrogen']


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
