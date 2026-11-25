"""One-time Neon successor install on the Hydrogen successor: plan, atomic apply, reconciliation, refusals."""
from datetime import datetime, timedelta, timezone
import json
import pytest
from scripts import install_spend_successor_20261125 as install_module
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.spend_guard import EMBEDDING_2_OPERATION, HYDROGEN_OPERATION, NEON_OPERATION
from src.zeitgeist.spend_successor import (
    EMBEDDING_001_OPERATION, FLASH_OPERATION, NEON_BASE_POLICY_SHA256, build_neon_successor, field_diff,
    successor_ledger)
from tests.test_automatic_spend import fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from tests.test_neon_successor import NEON_KW, hydrogen_installed
from tests.test_spend_successor import hold

NOW = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
T = NEON_KW['tariff_sha256']
INPUTS = dict(NEON_KW)
CHANGE = 'added operations only: generate_content:gemini-4-neon'


def contract(neon=('3', '15'), hydrogen=('1.5', '7.5')):
    operations = {FLASH_OPERATION: {'input_token_usd_per_million': '0.75', 'output_token_usd_per_million': '3.75'},
                  EMBEDDING_001_OPERATION: {'input_token_usd_per_million': '0.15', 'billing_basis': 'input_tokens'},
                  EMBEDDING_2_OPERATION: {'input_token_usd_per_million': '0.2', 'billing_basis': 'input_tokens'}}
    if hydrogen:
        operations[HYDROGEN_OPERATION] = {'input_token_usd_per_million': hydrogen[0],
                                          'output_token_usd_per_million': hydrogen[1]}
    if neon:
        operations[NEON_OPERATION] = {'input_token_usd_per_million': neon[0], 'output_token_usd_per_million': neon[1]}
    return {'operations': operations}


def setup(monkeypatch, published=None, sha=T):
    ledger, p1 = hydrogen_installed()
    old_sha = auto.digest(p1)
    ledger.update(month='2026-11', automatic_policy_sha256=old_sha,
                  credit_only_providers=list(p1['credit_only_providers']),
                  reservations={'old': hold(NOW - timedelta(hours=2))})
    wire = PrivateWire(monkeypatch, {'policy_' + old_sha: p1, '2026-11': ledger})
    monkeypatch.setattr(install_module, 'observe_gemini_tariff',
                        lambda *a, **k: {'sha256': sha, 'contract': published or contract()})
    return wire, old_sha, ledger, p1


def run(wire, old_sha, **kw):
    kw.setdefault('apply', False)
    return install_module.install(wire.store, old_sha=old_sha, now=NOW, **dict(INPUTS, **kw))


def applied(monkeypatch):
    wire, old_sha, ledger, p1 = setup(monkeypatch)
    p2 = build_neon_successor(p1, **NEON_KW)
    new_sha = auto.digest(p2)
    run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    return wire, old_sha, new_sha, p2


def test_plan_writes_nothing_and_prints_the_digest_and_the_three_line_diff(monkeypatch):
    wire, old_sha, ledger, p1 = setup(monkeypatch)
    report = run(wire, old_sha)
    p2 = build_neon_successor(p1, **NEON_KW)
    assert wire.posts == []
    assert set(report) == {'mode', 'old_policy_sha256', 'new_policy_sha256', 'receipt_id', 'policy_diff',
                           'ledger_diff', 'open_reservation_count', 'reservations_unchanged',
                           'neon_max_output_tokens', 'neon_thinking_level'}
    assert report['mode'] == 'plan' and report['receipt_id'] == 'spend_successor_20261125'
    assert report['neon_max_output_tokens'] == 6000 and report['neon_thinking_level'] == 'medium'
    assert report['new_policy_sha256'] == auto.digest(p2) and report['old_policy_sha256'] == old_sha
    assert report['policy_diff'] == field_diff(p1, p2) and len(report['policy_diff']) == 3
    assert any(line.endswith(NEON_OPERATION + ' = "0.286608"') for line in report['policy_diff'])
    assert report['open_reservation_count'] == 1 and report['reservations_unchanged'] is True


@pytest.mark.parametrize('unknown', [False, True])
def test_apply_commits_three_documents_once_and_reads_back(monkeypatch, unknown):
    wire, old_sha, ledger, p1 = setup(monkeypatch)
    if unknown:
        wire.unknown_post = 1
    p2 = build_neon_successor(p1, **NEON_KW)
    new_sha = auto.digest(p2)
    report = run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert report['mode'] == 'applied' and report['reservations_unchanged'] is True
    assert len(wire.posts) == 1 and len(wire.posts[0]['writes']) == 3
    assert wire.docs['policy_' + new_sha] == p2
    assert wire.docs['2026-11'] == successor_ledger(ledger, p2, new_sha)
    receipt = wire.docs[runtime.NEON_SUCCESSOR_RECEIPT_ID]
    assert receipt['change'] == CHANGE and receipt['inputs'] == INPUTS and receipt['schema_version'] == 1
    assert receipt['old_policy_sha256'] == old_sha and receipt['new_policy_sha256'] == new_sha
    assert runtime.SUCCESSOR_RECEIPT_ID not in wire.docs
    again = run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert again['mode'] == 'already_applied' and len(wire.posts) == 1


def test_apply_requires_the_planned_digest(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch)
    for expected in ('0' * 64, None):
        with pytest.raises(RuntimeError, match='planned new policy digest'):
            run(wire, old_sha, apply=True, expected_new_sha=expected)
    assert wire.posts == []


@pytest.mark.parametrize('neon', [('3.3', '15'), ('3', '15.5'), ('1.5', '7.5'), None])
def test_published_neon_rate_must_equal_3_and_15(monkeypatch, neon):
    wire, old_sha, _, p1 = setup(monkeypatch, published=contract(neon=neon))
    with pytest.raises(RuntimeError, match='rate'):
        run(wire, old_sha, apply=True, expected_new_sha=auto.digest(build_neon_successor(p1, **NEON_KW)))
    assert wire.posts == []


def test_a_moved_hydrogen_rate_still_blocks(monkeypatch):
    wire, old_sha, _, p1 = setup(monkeypatch, published=contract(hydrogen=('1.65', '7.5')))
    with pytest.raises(RuntimeError, match='rate'):
        run(wire, old_sha, apply=True, expected_new_sha=auto.digest(build_neon_successor(p1, **NEON_KW)))
    assert wire.posts == []


def test_tariff_input_must_match_live_observation(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch, sha='d' * 64)
    with pytest.raises(RuntimeError, match='tariff digest'):
        run(wire, old_sha)
    assert wire.posts == []


def test_in_flight_hold_blocks(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch)
    wire.docs['2026-11']['reservations']['fresh'] = hold(NOW - timedelta(seconds=10))
    with pytest.raises(RuntimeError, match='in flight'):
        run(wire, old_sha)
    assert wire.posts == []


@pytest.mark.parametrize('change', [{'reconciliation_required': True}, {'closed_for_rollover': True},
                                    {'automatic_policy_sha256': 'f' * 64}, {'month': '2026-10'}])
def test_frozen_or_moved_ledger_blocks(monkeypatch, change):
    wire, old_sha, _, _ = setup(monkeypatch)
    wire.docs['2026-11'].update(change)
    with pytest.raises(RuntimeError):
        run(wire, old_sha)
    assert wire.posts == []


def test_month_boundary_blocks_install(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch)
    late = datetime(2026, 11, 30, 23, 59, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match='month boundary'):
        install_module.install(wire.store, old_sha=old_sha, now=late, **INPUTS)
    assert wire.posts == []


@pytest.mark.parametrize('tamper', [lambda r: r.update(inputs=dict(r['inputs'], neon_thinking_level='high')),
                                    lambda r: r.update(inputs=dict(r['inputs'], neon_tokens_per_image=1)),
                                    lambda r: r.update(change='added operations only: generate_content:gemini-4-hydrogen'),
                                    lambda r: r.update(month='2026-10')])
def test_receipt_that_does_not_match_the_request_is_refused(monkeypatch, tamper):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    tamper(wire.docs[runtime.NEON_SUCCESSOR_RECEIPT_ID])
    with pytest.raises(RuntimeError, match='identity mismatch'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert len(wire.posts) == 1


def test_after_apply_the_chain_reaches_the_neon_successor(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    assert runtime._successor_policy(wire.store, old_sha, '2026-11') == new_sha


def test_after_apply_refresh_passes_on_the_hash_path_with_neon_listed(monkeypatch):
    wire, _, new_sha, p2 = applied(monkeypatch)
    _, _, observation = fixtures(NOW)
    observation['price_snapshots']['gemini'] = {
        'sha256': T, 'source_url': p2['providers']['gemini']['price_source_url'], 'observed_at': NOW.isoformat()}
    observation['gemini']['available_required_models'] = {
        'gemini-3.8-flash': ['generateContent'], 'gemini-embedding-001': ['embedContent'],
        'gemini-4-hydrogen': ['generateContent'], 'gemini-4-neon': ['generateContent'],
        'gemini-embedding-2': ['embedContent']}
    after = auto.prepare_refresh(wire.docs['2026-11'], p2, observation, expected_policy_sha=new_sha, now=NOW)['after']
    assert 'disabled_operations' not in after['providers']['gemini']


ARGS = ['--neon-model-version', 'gemini-4-neon', '--neon-tokens-per-image', '1091',
        '--neon-max-output-tokens', '6000', '--neon-thinking-level', 'medium', '--tariff-sha256', T]


def test_main_refuses_another_installed_sha(monkeypatch, capsys):
    reads = []
    monkeypatch.setattr(install_module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', 'e664ccce9873b17fedaa7eda48e7c928be7540fd3db6c7d99168a125045a661f')
    code = install_module.main(ARGS)
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and reads == [] and len(out) == 1 and 'reviewed installed policy' in json.loads(out[0])['error']


def test_main_installs_on_the_hydrogen_successor_with_the_neon_flags(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(install_module, 'PrivateFirestoreStore', lambda *a, **k: 'store')
    monkeypatch.setattr(install_module, 'install', lambda store, **kw: seen.append(kw) or {'mode': 'plan'})
    monkeypatch.setenv('SPEND_POLICY_SHA256', NEON_BASE_POLICY_SHA256)
    assert install_module.main(ARGS + ['--apply', '--expected-new-sha', 'a' * 64]) == 0
    assert seen == [dict(old_sha=NEON_BASE_POLICY_SHA256, neon_model_version='gemini-4-neon', neon_tokens_per_image=1091,
                         neon_max_output_tokens=6000, neon_thinking_level='medium', tariff_sha256=T, apply=True,
                         expected_new_sha='a' * 64)]
    assert json.loads(capsys.readouterr().out.strip()) == {'mode': 'plan'}


@pytest.mark.parametrize('flag, value', [('--neon-max-output-tokens', '5000'), ('--neon-thinking-level', 'low'),
                                         ('--neon-thinking-level', 'xhigh')])
def test_main_rejects_values_outside_the_build_ladder(monkeypatch, capsys, flag, value):
    monkeypatch.setattr(install_module, 'PrivateFirestoreStore', lambda *a, **k: pytest.fail('store opened'))
    monkeypatch.setenv('SPEND_POLICY_SHA256', NEON_BASE_POLICY_SHA256)
    args = list(ARGS)
    args[args.index(flag) + 1] = value
    assert install_module.main(args) == 1
    assert 'Neon' in json.loads(capsys.readouterr().out.strip())['error']


def test_observe_tariff_prints_the_neon_rate(monkeypatch, capsys):
    monkeypatch.setattr(install_module, 'observe_gemini_tariff', lambda *a, **k: {'sha256': T, 'contract': contract()})
    assert install_module.main(['--observe-tariff']) == 0
    seen = json.loads(capsys.readouterr().out.strip())
    assert seen['tariff_sha256'] == T
    assert seen['rates'][NEON_OPERATION] == {'input': '3', 'output': '15'}
    assert seen['rates'][HYDROGEN_OPERATION] == {'input': '1.5', 'output': '7.5'}
