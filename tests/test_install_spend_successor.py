"""One-time successor install: plan, atomic apply, reconciliation and every refusal."""
import copy
from datetime import datetime, timedelta, timezone
import json
import pytest
from scripts import install_spend_successor_20261122 as install_module
from src.zeitgeist import automatic_spend as auto
from src.zeitgeist import automatic_spend_runtime as runtime
from src.zeitgeist.spend_guard import EMBEDDING_2_OPERATION, HYDROGEN_OPERATION
from src.zeitgeist.spend_successor import (
    EMBEDDING_001_OPERATION, FLASH_OPERATION, build_successor, field_diff, successor_ledger)
from tests.test_automatic_spend import fixtures
from tests.test_automatic_spend_runtime import PrivateWire
from tests.test_spend_successor import KW, hold, installed

NOW = datetime(2026, 11, 22, 12, tzinfo=timezone.utc)
T = KW['tariff_sha256']
INPUTS = dict(hydrogen_model_version=KW['hydrogen_model_version'],
              hydrogen_tokens_per_image=KW['hydrogen_tokens_per_image'],
              hydrogen_max_output_tokens=KW['hydrogen_max_output_tokens'],
              hydrogen_thinking_level=KW['hydrogen_thinking_level'], tariff_sha256=T)


def contract(hydrogen=('1.5', '7.5'), embedding_2=None):
    operations = {FLASH_OPERATION: {'input_token_usd_per_million': '0.75', 'output_token_usd_per_million': '3.75'},
                  EMBEDDING_001_OPERATION: {'input_token_usd_per_million': '0.15', 'billing_basis': 'input_tokens'}}
    if hydrogen:
        operations[HYDROGEN_OPERATION] = {'input_token_usd_per_million': hydrogen[0],
                                          'output_token_usd_per_million': hydrogen[1]}
    if embedding_2:
        operations[EMBEDDING_2_OPERATION] = dict({'billing_basis': 'input_tokens'}, **embedding_2)
    return {'operations': operations}


E2_PUBLISHED = {'input_token_usd_per_million': '0.2'}


def setup(monkeypatch, published=None, sha=T):
    ledger, p0 = installed()
    old_sha = auto.digest(p0)
    ledger.update(month='2026-11', automatic_policy_sha256=old_sha,
                  credit_only_providers=list(p0['credit_only_providers']),
                  reservations={'old': hold(NOW - timedelta(hours=2))})
    wire = PrivateWire(monkeypatch, {'policy_' + old_sha: p0, '2026-11': ledger})
    monkeypatch.setattr(install_module, 'observe_gemini_tariff',
                        lambda *a, **k: {'sha256': sha, 'contract': published or contract()})
    return wire, old_sha, ledger, p0


def run(wire, old_sha, **kw):
    kw.setdefault('apply', False)
    return install_module.install(wire.store, old_sha=old_sha, now=NOW, **dict(INPUTS, **kw))


def test_plan_writes_nothing_and_prints_digest_and_diff(monkeypatch):
    wire, old_sha, ledger, p0 = setup(monkeypatch)
    report = run(wire, old_sha)
    p1 = build_successor(p0, **KW)
    assert wire.posts == []
    assert set(report) == {'mode', 'old_policy_sha256', 'new_policy_sha256', 'receipt_id', 'policy_diff',
                           'ledger_diff', 'open_reservation_count', 'reservations_unchanged',
                           'hydrogen_max_output_tokens', 'hydrogen_thinking_level'}
    assert report['hydrogen_max_output_tokens'] == 4000 and report['hydrogen_thinking_level'] == 'high'
    assert report['mode'] == 'plan' and report['receipt_id'] == runtime.SUCCESSOR_RECEIPT_ID
    assert report['new_policy_sha256'] == auto.digest(p1)
    assert report['policy_diff'] == field_diff(p0, p1)
    assert report['open_reservation_count'] == 1 and report['reservations_unchanged'] is True
    text = json.dumps(report)
    assert 'AAAAAA' not in text and 'a' * 16 not in text


def test_plan_with_6000_shows_value_cap_and_policy_diff(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    report = install_module.install(wire.store, old_sha=old_sha, now=NOW, **dict(INPUTS, hydrogen_max_output_tokens=6000))
    p1 = build_successor(p0, **dict(KW, hydrogen_max_output_tokens=6000))
    assert wire.posts == [] and report['hydrogen_max_output_tokens'] == 6000
    assert report['policy_diff'] == field_diff(p0, p1)
    assert any('0.143304' in line for line in report['policy_diff'])
    assert not any('0.128304' in line for line in report['policy_diff'])


@pytest.mark.parametrize('bad', [5000, None, True, 3999, 6001, 7000, 12000, 15001])
def test_install_rejects_other_output_tokens(monkeypatch, bad):
    wire, old_sha, _, _ = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=NOW, **dict(INPUTS, hydrogen_max_output_tokens=bad))
    assert wire.posts == []


def test_apply_with_6000_records_it_and_rerun_with_4000_is_refused(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    inputs = dict(INPUTS, hydrogen_max_output_tokens=6000)
    p1 = build_successor(p0, **dict(KW, hydrogen_max_output_tokens=6000))
    new_sha = auto.digest(p1)
    report = install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True,
                                    expected_new_sha=new_sha, **inputs)
    assert report['mode'] == 'applied'
    assert wire.docs[runtime.SUCCESSOR_RECEIPT_ID]['inputs'] == inputs
    assert install_module.install(wire.store, old_sha=old_sha, now=NOW, **inputs)['mode'] == 'already_applied'
    posts = len(wire.posts)
    with pytest.raises(RuntimeError, match='identity mismatch'):
        run(wire, old_sha)
    assert len(wire.posts) == posts


@pytest.mark.parametrize('unknown', [False, True])
def test_apply_commits_three_documents_once_and_reads_back(monkeypatch, unknown):
    wire, old_sha, ledger, p0 = setup(monkeypatch)
    if unknown:
        wire.unknown_post = 1
    p1 = build_successor(p0, **KW)
    new_sha = auto.digest(p1)
    report = run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert report['mode'] == 'applied' and report['reservations_unchanged'] is True
    assert len(wire.posts) == 1 and len(wire.posts[0]['writes']) == 3
    after = successor_ledger(ledger, p1, new_sha)
    assert wire.docs['policy_' + new_sha] == p1
    assert wire.docs['2026-11'] == after
    assert wire.docs['2026-11']['reservations'] == ledger['reservations']
    receipt = wire.docs[runtime.SUCCESSOR_RECEIPT_ID]
    assert set(receipt) == {'schema_version', 'old_policy_sha256', 'new_policy_sha256', 'month', 'observed_at',
                            'observation', 'before_sha256', 'after_sha256', 'change', 'inputs'}
    assert receipt['change'] == 'added operations only: generate_content:gemini-4-hydrogen'
    assert receipt['inputs'] == INPUTS and receipt['schema_version'] == 1
    assert receipt['before_sha256'] == auto.digest(ledger) and receipt['after_sha256'] == auto.digest(after)
    again = run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert again['mode'] == 'already_applied' and len(wire.posts) == 1


def test_apply_requires_the_planned_digest(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        run(wire, old_sha, apply=True, expected_new_sha='0' * 64)
    with pytest.raises(RuntimeError):
        run(wire, old_sha, apply=True)
    assert wire.posts == []


def test_precondition_conflict_is_not_retried(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    new_sha = auto.digest(build_successor(p0, **KW))
    def bump(server):
        server.versions['2026-11'] += 1
    wire.before_post = bump
    before = copy.deepcopy(wire.docs)
    with pytest.raises(RuntimeError, match='replan'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert len(wire.posts) == 1 and wire.docs == before


def test_unresolved_readback_is_refused_without_a_second_post(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    new_sha = auto.digest(build_successor(p0, **KW))
    wire.unknown_before_post = 1
    before = copy.deepcopy(wire.docs)
    with pytest.raises(RuntimeError, match='unresolved'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert len(wire.posts) == 1 and wire.docs == before


def test_hold_created_after_the_ledger_read_forces_a_replan(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    new_sha = auto.digest(build_successor(p0, **KW))
    def observe(*a, **k):
        wire.docs['2026-11']['reservations']['late'] = hold(NOW - timedelta(seconds=5))
        wire.versions['2026-11'] += 1
        return {'sha256': T, 'contract': contract()}
    monkeypatch.setattr(install_module, 'observe_gemini_tariff', observe)
    with pytest.raises(RuntimeError, match='replan'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert len(wire.posts) == 1 and 'late' in wire.docs['2026-11']['reservations']
    assert wire.docs['2026-11']['automatic_policy_sha256'] == old_sha
    assert set(wire.docs) == {'policy_' + old_sha, '2026-11'}


def test_tariff_input_must_match_live_observation(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch, sha='e' * 64)
    with pytest.raises(RuntimeError):
        run(wire, old_sha)
    assert wire.posts == []


@pytest.mark.parametrize('hydrogen', [('1.65', '7.5'), ('1.5', '7.6'), None])
def test_published_hydrogen_rate_must_equal_policy(monkeypatch, hydrogen):
    wire, old_sha, _, p0 = setup(monkeypatch, published=contract(hydrogen))
    new_sha = auto.digest(build_successor(p0, **KW))
    with pytest.raises(RuntimeError, match='rate'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert wire.posts == []


def test_in_flight_hold_blocks(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch)
    wire.docs['2026-11']['reservations']['fresh'] = hold(NOW - timedelta(seconds=10))
    with pytest.raises(RuntimeError, match='in flight'):
        run(wire, old_sha)
    assert wire.posts == []


def test_hold_in_flight_window_covers_the_longest_provider(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    policy = wire.docs['policy_' + old_sha]
    assert max(p['account']['max_request_duration_seconds'] for p in policy['providers'].values()) == 300
    wire.docs['2026-11']['reservations']['edge'] = dict(
        hold(NOW - timedelta(seconds=320)), provider='apify')
    with pytest.raises(RuntimeError, match='in flight'):
        run(wire, old_sha)


@pytest.mark.parametrize('change', [{'reconciliation_required': True}, {'closed_for_rollover': True},
                                    {'automatic_policy_sha256': 'f' * 64}, {'month': '2026-10'}])
def test_frozen_or_moved_ledger_blocks(monkeypatch, change):
    wire, old_sha, _, _ = setup(monkeypatch)
    wire.docs['2026-11'].update(change)
    with pytest.raises(RuntimeError):
        run(wire, old_sha)
    assert wire.posts == []


def test_main_refuses_other_installed_sha(monkeypatch, capsys):
    reads = []
    monkeypatch.setattr(install_module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', 'f' * 64)
    code = install_module.main(['--hydrogen-model-version', 'synthetic-hydrogen-version',
                                '--hydrogen-tokens-per-image', '1088', '--hydrogen-max-output-tokens', '4000',
                                '--hydrogen-thinking-level', 'high', '--tariff-sha256', T])
    out = capsys.readouterr().out.strip().splitlines()
    assert code != 0 and reads == [] and len(out) == 1 and 'error' in json.loads(out[0])


@pytest.mark.parametrize('bad', ['low', 'xhigh', None, True, ''])
def test_install_rejects_other_thinking_levels(monkeypatch, bad):
    wire, old_sha, _, _ = setup(monkeypatch)
    with pytest.raises(RuntimeError):
        install_module.install(wire.store, old_sha=old_sha, now=NOW, **dict(INPUTS, hydrogen_thinking_level=bad))
    assert wire.posts == []


@pytest.mark.parametrize('tokens, cap', [(8000, '0.158304'), (10000, '0.173304'), (15000, '0.210804')])
def test_plan_and_apply_for_other_ladder_values_with_medium(monkeypatch, tokens, cap):
    wire, old_sha, _, p0 = setup(monkeypatch)
    inputs = dict(INPUTS, hydrogen_max_output_tokens=tokens, hydrogen_thinking_level='medium')
    plan = install_module.install(wire.store, old_sha=old_sha, now=NOW, **inputs)
    assert any(cap in line for line in plan['policy_diff']) and plan['hydrogen_thinking_level'] == 'medium'
    done = install_module.install(wire.store, old_sha=old_sha, now=NOW, apply=True,
                                  expected_new_sha=plan['new_policy_sha256'], **inputs)
    assert done['mode'] == 'applied'
    assert wire.docs[runtime.SUCCESSOR_RECEIPT_ID]['inputs'] == inputs


def test_main_requires_output_tokens_flag(monkeypatch, capsys):
    reads = []
    monkeypatch.setattr(install_module, 'PrivateFirestoreStore', lambda *a, **k: reads.append(1))
    monkeypatch.setenv('SPEND_POLICY_SHA256', install_module.INSTALLED_POLICY_SHA256)
    code = install_module.main(['--hydrogen-model-version', 'synthetic-hydrogen-version',
                                '--hydrogen-tokens-per-image', '1088', '--tariff-sha256', T])
    out = capsys.readouterr().out.strip().splitlines()
    assert code == 1 and reads == [] and len(out) == 1 and 'are required' in json.loads(out[0])['error']
    code = install_module.main(['--hydrogen-model-version', 'synthetic-hydrogen-version',
                                '--hydrogen-tokens-per-image', '1088', '--hydrogen-max-output-tokens', '4000',
                                '--tariff-sha256', T])
    assert code == 1 and reads == [] and 'are required' in json.loads(capsys.readouterr().out.strip())['error']
    code = install_module.main(['--hydrogen-model-version', 'synthetic-hydrogen-version',
                                '--hydrogen-tokens-per-image', '1088', '--hydrogen-max-output-tokens', '5000',
                                '--hydrogen-thinking-level', 'high', '--tariff-sha256', T])
    assert code == 1 and reads == [] and 'error' in json.loads(capsys.readouterr().out.strip())
    code = install_module.main(['--hydrogen-model-version', 'synthetic-hydrogen-version',
                                '--hydrogen-tokens-per-image', '1088', '--hydrogen-max-output-tokens', '4000',
                                '--hydrogen-thinking-level', 'low', '--tariff-sha256', T])
    assert code == 1 and reads == [] and 'error' in json.loads(capsys.readouterr().out.strip())


def test_observe_tariff_prints_one_json_line_of_public_numbers(monkeypatch, capsys):
    monkeypatch.setattr(install_module, 'observe_gemini_tariff',
                        lambda *a, **k: {'sha256': T, 'contract': contract()})
    assert install_module.main(['--observe-tariff']) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1
    seen = json.loads(out[0])
    assert seen['tariff_sha256'] == T
    assert seen['rates'][HYDROGEN_OPERATION] == {'input': '1.5', 'output': '7.5'}
    assert seen['rates'][EMBEDDING_001_OPERATION] == {'input': '0.15', 'output': '0'}


def applied(monkeypatch, output_tokens=4000, level='high'):
    wire, old_sha, ledger, p0 = setup(monkeypatch)
    p1 = build_successor(p0, **dict(KW, hydrogen_max_output_tokens=output_tokens, hydrogen_thinking_level=level))
    new_sha = auto.digest(p1)
    run(wire, old_sha, apply=True, expected_new_sha=new_sha, hydrogen_max_output_tokens=output_tokens,
        hydrogen_thinking_level=level)
    return wire, old_sha, new_sha, p1


def test_month_boundary_blocks_install(monkeypatch):
    wire, old_sha, _, _ = setup(monkeypatch)
    late = datetime(2026, 11, 30, 23, 59, tzinfo=timezone.utc)
    with pytest.raises(RuntimeError, match='month boundary'):
        install_module.install(wire.store, old_sha=old_sha, now=late, **INPUTS)
    assert wire.posts == []


def test_successor_policy_without_receipt_is_refused(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    p1 = build_successor(p0, **KW)
    key = 'policy_' + auto.digest(p1)
    wire.docs[key] = p1
    wire.versions[key] = 1
    with pytest.raises(RuntimeError, match='Unexpected successor policy'):
        run(wire, old_sha)
    assert wire.posts == []


def other_tariff(receipt):
    receipt['inputs'] = dict(receipt['inputs'], tariff_sha256='d' * 64)


def other_output(receipt):
    receipt['inputs'] = dict(receipt['inputs'], hydrogen_max_output_tokens=6000)


def other_level(receipt):
    receipt['inputs'] = dict(receipt['inputs'], hydrogen_thinking_level='medium')


def other_tokens(receipt):
    receipt['inputs'] = dict(receipt['inputs'], hydrogen_tokens_per_image=1)


@pytest.mark.parametrize('tamper', [other_tariff, other_tokens, other_output, other_level,
                                    lambda r: r.update(month='2026-10'),
                                    lambda r: r.update(old_policy_sha256='c' * 64)])
def test_receipt_that_does_not_match_the_request_is_refused(monkeypatch, tamper):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    tamper(wire.docs[runtime.SUCCESSOR_RECEIPT_ID])
    with pytest.raises(RuntimeError, match='identity mismatch'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert len(wire.posts) == 1


def test_receipt_with_a_ledger_that_moved_off_the_successor_needs_reconciliation(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    wire.docs['2026-11']['automatic_policy_sha256'] = old_sha
    with pytest.raises(RuntimeError, match='reconciliation'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert len(wire.posts) == 1


def test_already_applied_reports_reservations_unchanged_only_when_the_ledger_is_the_after_image(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    assert run(wire, old_sha, apply=True, expected_new_sha=new_sha)['reservations_unchanged'] is True
    wire.docs['2026-11']['reservations']['late'] = hold(NOW)
    again = run(wire, old_sha, apply=True, expected_new_sha=new_sha)
    assert again['mode'] == 'already_applied' and again['reservations_unchanged'] is False


def test_after_apply_refresh_passes_on_the_hash_path(monkeypatch):
    wire, old_sha, new_sha, p1 = applied(monkeypatch)
    _, _, observation = fixtures(NOW)
    observation['price_snapshots']['gemini'] = {
        'sha256': T, 'source_url': p1['providers']['gemini']['price_source_url'], 'observed_at': NOW.isoformat()}
    observation['gemini']['available_required_models'] = {
        'gemini-3.8-flash': ['generateContent'], 'gemini-embedding-001': ['embedContent'],
        'gemini-4-hydrogen': ['generateContent']}
    after = wire.docs['2026-11']
    result = auto.prepare_refresh(after, p1, observation, expected_policy_sha=new_sha, now=NOW)
    assert 'price_band_audit' not in result['after']['providers']['gemini']


def test_after_apply_chain_reaches_successor(monkeypatch):
    wire, old_sha, new_sha, _ = applied(monkeypatch)
    assert runtime._successor_policy(wire.store, old_sha, '2026-11') == new_sha


E2_CHANGE = 'added operations only: generate_content:gemini-4-hydrogen, embed_content:gemini-embedding-2'


def test_plan_with_embedding_2_lists_both_operations(monkeypatch):
    wire, old_sha, ledger, p0 = setup(monkeypatch, published=contract(embedding_2=E2_PUBLISHED))
    report = run(wire, old_sha, include_embedding_2=True)
    p1 = build_successor(p0, **KW, include_embedding_2=True)
    assert wire.posts == [] and report['mode'] == 'plan'
    assert report['new_policy_sha256'] == auto.digest(p1)
    assert report['policy_diff'] == field_diff(p0, p1)
    for operation in (HYDROGEN_OPERATION, EMBEDDING_2_OPERATION):
        for table in ('operation_caps_usd', 'request_limits'):
            assert any(line.startswith('+ providers.gemini.account.' + table + '.' + operation + ' = ')
                       for line in report['policy_diff'])
            assert any(line.startswith('+ providers.gemini.' + table + '.' + operation + ' = ')
                       for line in report['ledger_diff'])
    assert any(line.endswith(EMBEDDING_2_OPERATION + ' = "0.04096"') for line in report['policy_diff'])
    assert report['reservations_unchanged'] is True


def test_apply_with_embedding_2_records_the_change_and_a_hydrogen_only_rerun_is_refused(monkeypatch):
    wire, old_sha, ledger, p0 = setup(monkeypatch, published=contract(embedding_2=E2_PUBLISHED))
    p1 = build_successor(p0, **KW, include_embedding_2=True)
    new_sha = auto.digest(p1)
    report = run(wire, old_sha, apply=True, expected_new_sha=new_sha, include_embedding_2=True)
    assert report['mode'] == 'applied' and len(wire.posts) == 1
    assert wire.docs['policy_' + new_sha] == p1
    assert wire.docs['2026-11'] == successor_ledger(ledger, p1, new_sha)
    receipt = wire.docs[runtime.SUCCESSOR_RECEIPT_ID]
    assert receipt['change'] == E2_CHANGE and receipt['inputs'] == INPUTS
    assert run(wire, old_sha, include_embedding_2=True)['mode'] == 'already_applied'
    with pytest.raises(RuntimeError, match='identity mismatch'):
        run(wire, old_sha)
    assert len(wire.posts) == 1


def test_receipt_with_a_hydrogen_only_change_does_not_prove_an_embedding_2_install(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch, published=contract(embedding_2=E2_PUBLISHED))
    new_sha = auto.digest(build_successor(p0, **KW, include_embedding_2=True))
    run(wire, old_sha, apply=True, expected_new_sha=new_sha, include_embedding_2=True)
    wire.docs[runtime.SUCCESSOR_RECEIPT_ID]['change'] = 'added operations only: generate_content:gemini-4-hydrogen'
    with pytest.raises(RuntimeError, match='identity mismatch'):
        run(wire, old_sha, include_embedding_2=True)
    assert len(wire.posts) == 1


@pytest.mark.parametrize('published', [None, {'input_token_usd_per_million': '0.25'},
                                       {'input_token_usd_per_million': '0.19'},
                                       {'input_token_usd_per_million': '0.2000001'},
                                       {'input_token_usd_per_million': '0.2', 'output_token_usd_per_million': '0.01'}])
def test_published_embedding_2_rate_must_equal_0_20(monkeypatch, published):
    wire, old_sha, _, p0 = setup(monkeypatch, published=contract(embedding_2=published))
    new_sha = auto.digest(build_successor(p0, **KW, include_embedding_2=True))
    with pytest.raises(RuntimeError, match='rate'):
        run(wire, old_sha, apply=True, expected_new_sha=new_sha, include_embedding_2=True)
    assert wire.posts == []


def test_hydrogen_only_install_does_not_need_embedding_2_published(monkeypatch):
    wire, old_sha, _, p0 = setup(monkeypatch)
    new_sha = auto.digest(build_successor(p0, **KW))
    assert run(wire, old_sha, apply=True, expected_new_sha=new_sha)['mode'] == 'applied'
    assert wire.docs[runtime.SUCCESSOR_RECEIPT_ID]['change'] == 'added operations only: generate_content:gemini-4-hydrogen'


@pytest.mark.parametrize('flag, expected', [([], False), (['--include-embedding-2'], True)])
def test_main_passes_the_embedding_2_flag(monkeypatch, capsys, flag, expected):
    seen = []
    monkeypatch.setattr(install_module, 'PrivateFirestoreStore', lambda *a, **k: 'store')
    monkeypatch.setattr(install_module, 'install', lambda store, **kw: seen.append(kw) or {'mode': 'plan'})
    monkeypatch.setenv('SPEND_POLICY_SHA256', install_module.INSTALLED_POLICY_SHA256)
    code = install_module.main(['--hydrogen-model-version', 'synthetic-hydrogen-version',
                                '--hydrogen-tokens-per-image', '1088', '--hydrogen-max-output-tokens', '4000',
                                '--hydrogen-thinking-level', 'high', '--tariff-sha256', T] + flag)
    assert code == 0 and len(seen) == 1 and seen[0]['include_embedding_2'] is expected
    assert json.loads(capsys.readouterr().out.strip()) == {'mode': 'plan'}


def test_observe_tariff_shows_the_embedding_2_rate(monkeypatch, capsys):
    monkeypatch.setattr(install_module, 'observe_gemini_tariff',
                        lambda *a, **k: {'sha256': T, 'contract': contract(embedding_2=E2_PUBLISHED)})
    assert install_module.main(['--observe-tariff']) == 0
    assert json.loads(capsys.readouterr().out.strip())['rates'][EMBEDDING_2_OPERATION] == {'input': '0.2', 'output': '0'}
