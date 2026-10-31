"""Runtime factory, actual visitor handler and renderer use canonical authority."""
import builtins
import copy
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
import pytest
from src import state_store as state
from src.game_engine import create_initial_game_state, render_board_svg
from tests.test_firestore_store import _RecordingFirestoreClient

ZERO = {'spot_on': 0, 'too_sunny': 0, 'too_heavy': 0}


def fixtures(tmp_path):
    root = tmp_path/'repo'; local = state.JsonFileStateStore(root)
    game = create_initial_game_state(); game.update(state_version=5, round_id=8)
    local.save_game_state(dict(game, state_version=99, round_id=99))
    local.save_pulse_state({'state_version': 99, 'word': 'CHECKOUT', 'visitor_mood_tally': {'spot_on': 90, 'too_sunny': 80, 'too_heavy': 70}})
    local.save_elo_ledger({'state_version': 99, 'players': {'invented-local': {'elo': 9999}}})
    client = _RecordingFirestoreClient()
    client.docs.update({'profile_state/active_game': copy.deepcopy(game),
      'profile_state/daily_pulse': {'state_version': 5, 'date': '2026-10-30', 'word': 'CANONICAL', 'visitor_mood_tally': dict(ZERO)},
      'profile_state/elo_ledger': {'state_version': 5, 'bot_elo':1200, 'players': {}, 'recent_rounds': []},
      'profile_state/bot_memory': {'state_version':0,'states':{},'games_learned':0},
      'profile_state/game_rate_limits': {'state_version':0,'users':{}}})
    return root, local, game, client


def files(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('local_version', [1, 5, 99])
def test_factory_never_promotes_checkout_over_canonical_versions(tmp_path, local_version):
    root, local, game, client = fixtures(tmp_path)
    local.save_game_state(dict(game, state_version=local_version, round_id=99))
    local.save_pulse_state({'state_version': local_version, 'word': 'CHECKOUT'})
    local.save_elo_ledger({'state_version': local_version, 'players': {'invented-local': {'elo': 9999}}})
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    assert store.load_game_state()['round_id'] == 8
    assert store.load_pulse_state()['word'] == 'CANONICAL'
    assert store.load_elo_ledger()['players'] == {}
    assert store.canonical_only is True
    assert store._fallback_store is None


@pytest.mark.parametrize('method,key', [('load_game_state','active_game'), ('load_pulse_state','daily_pulse')])
def test_missing_canonical_state_never_bootstraps_from_checkout(tmp_path, method, key):
    root, local, game, client = fixtures(tmp_path); del client.docs['profile_state/'+key]
    baseline = files(root); canonical = copy.deepcopy(client.docs)
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    with pytest.raises(RuntimeError, match='Canonical'):
        getattr(store, method)()
    assert client.docs == canonical
    assert files(root) == baseline


def deny_auth(monkeypatch):
    monkeypatch.setenv('FIRESTORE_ACCESS_TOKEN', 'synthetic-token')
    def fail(*args, **kwargs):
        raise HTTPError('https://sts.googleapis.com/private?token=synthetic-secret',403,'synthetic-secret',{},BytesIO(b'synthetic-secret'))
    monkeypatch.setattr(state._FirestoreRestClient, '_resolve_token', fail)
    original = builtins.__import__
    def imports(name, *args, **kwargs):
        if name == 'google.cloud': raise ImportError('offline SDK unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', imports)


@pytest.mark.parametrize('title', ['game:move:connect4:col:4:turn=1', 'pulse:vote:spot_on'])
def test_real_visitor_factory_auth_failure_never_succeeds_or_mutates_local_files(tmp_path, monkeypatch, title):
    from scripts import process_visitor_event as visitor
    root, local, game, client = fixtures(tmp_path); baseline = files(root)
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore'); deny_auth(monkeypatch)
    with pytest.raises(HTTPError):
        visitor.process_event(root, 'test-visitor', title, timestamp_epoch=1793350000)
    assert files(root) == baseline


def test_real_readme_factory_auth_failure_does_not_render_or_mutate(tmp_path, monkeypatch):
    from scripts import build_readme
    root, local, game, client = fixtures(tmp_path); baseline = files(root)
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore'); deny_auth(monkeypatch)
    with pytest.raises(HTTPError):
        build_readme.render_profile_markdown(root)
    assert files(root) == baseline


def test_absent_canonical_elo_has_honest_empty_players(tmp_path):
    root, local, game, client = fixtures(tmp_path); del client.docs['profile_state/elo_ledger']
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    assert store.load_elo_ledger()['players'] == {}
    assert 'profile_state/elo_ledger' not in client.docs


def test_actual_move_projects_canonical_game_elo_and_rendered_board(tmp_path, monkeypatch):
    from scripts import process_visitor_event as visitor, build_readme
    root, local, game, client = fixtures(tmp_path)
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore')
    factory = state.get_state_store
    def runtime_factory(**kwargs):
        return factory(**kwargs, firestore_client=client)
    monkeypatch.setattr(visitor, 'get_state_store', runtime_factory)
    monkeypatch.setattr(build_readme, 'get_state_store', runtime_factory)
    result = visitor.process_event(root, 'test-visitor', 'game:move:connect4:col:4:turn=1', issue_number=701, timestamp_epoch=1793350000)
    assert result['accepted'] is True
    canonical_store = factory(root, backend='firestore', firestore_client=client)
    canonical_game = canonical_store.load_game_state()
    mirror_game = json.loads((root/'data/game_state.json').read_text())
    mirror_elo = json.loads((root/'data/hall_of_records.json').read_text())
    assert mirror_game['board'] == canonical_game['board']
    assert mirror_game['turn'] == canonical_game['turn']
    assert mirror_game['round_id'] == 8
    assert mirror_elo['players'] == canonical_store.load_elo_ledger()['players'] == {}
    readme = build_readme.render_profile_markdown(root)
    assert 'game_board.svg' in readme
    assert (root/'assets/game_board.svg').read_text() == render_board_svg(canonical_game, elo_ledger=canonical_store.load_elo_ledger())


def test_successful_real_vote_updates_only_canonical_tally_and_its_json_mirror(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from scripts import process_visitor_event as visitor
    from tests.test_staging_application_rehearsal import ApplicationWire
    root, local, game, initial = fixtures(tmp_path)
    wire = ApplicationWire(monkeypatch)
    wire.docs.update(copy.deepcopy(initial.docs)); wire.versions.update({k:1 for k in wire.docs})
    wire.docs['profile_state/daily_pulse']['date'] = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore')
    monkeypatch.setenv('PULSE_VOTE_SALT', 'synthetic-vote-salt')
    monkeypatch.setattr(visitor, 'get_state_store', lambda **kwargs: state.get_state_store(**kwargs, firestore_client=wire.client))
    accepted = visitor.process_event(root, 'test-voter', 'pulse:vote:spot_on')
    assert accepted['status'] == 'recorded_vote'
    expected = dict(ZERO, spot_on=1)
    assert wire.docs['profile_state/daily_pulse']['visitor_mood_tally'] == expected
    assert json.loads((root/'data/zeitgeist_state.json').read_text())['visitor_mood_tally'] == expected
    duplicate = visitor.process_event(root, 'test-voter', 'pulse:vote:spot_on')
    assert duplicate['status'] == 'ignored'
    assert wire.docs['profile_state/daily_pulse']['visitor_mood_tally'] == expected


def test_storage_error_during_reconcile_cannot_partially_overwrite_mirrors(tmp_path):
    root, local, game, client = fixtures(tmp_path); baseline = files(root)
    original = client.collection
    def collection(name):
        col = original(name); document = col.document
        def reference(key):
            ref = document(key)
            if key == 'daily_pulse':
                def denied():
                    raise HTTPError('https://firestore.googleapis.com/private', 403, 'synthetic', {}, BytesIO())
                ref.get = denied
            return ref
        col.document = reference
        return col
    client.collection = collection
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    with pytest.raises(HTTPError):
        store.reconcile_git_from_firestore_if_needed()
    assert files(root) == baseline


def test_incomplete_canonical_tally_blocks_without_synthesizing_votes(tmp_path):
    root, local, game, client = fixtures(tmp_path)
    del client.docs['profile_state/daily_pulse']['visitor_mood_tally']
    baseline = files(root)
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    with pytest.raises(RuntimeError, match='Canonical.*tally'):
        store.load_pulse_state()
    assert files(root) == baseline


def test_renderer_missing_local_pulse_projects_authority_without_starting_pipeline(tmp_path, monkeypatch):
    from scripts import build_readme
    from src.zeitgeist import adk_pipeline
    root, local, game, client = fixtures(tmp_path)
    (root/'data/zeitgeist_state.json').unlink()
    canonical_before = copy.deepcopy(client.docs)
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore')
    monkeypatch.setattr(build_readme, 'get_state_store', lambda **kwargs: state.get_state_store(**kwargs, firestore_client=client))
    monkeypatch.setattr(adk_pipeline, 'run_daily_zeitgeist', lambda **kwargs: pytest.fail('Rendering cannot run providers or bootstrap canonical state'))
    readme = build_readme.render_profile_markdown(root)
    assert 'CANONICAL' in readme
    assert 'CHECKOUT' not in readme
    assert json.loads((root/'data/zeitgeist_state.json').read_text()) == client.docs['profile_state/daily_pulse']
    assert client.docs == canonical_before


@pytest.mark.parametrize('missing', ['active_game', 'daily_pulse'])
def test_renderer_missing_canonical_document_does_not_bootstrap_or_touch_checkout(tmp_path, monkeypatch, missing):
    from scripts import build_readme
    from src.zeitgeist import adk_pipeline
    root, local, game, client = fixtures(tmp_path)
    (root/'data/zeitgeist_state.json').unlink()
    del client.docs['profile_state/'+missing]
    baseline = files(root); canonical_before = copy.deepcopy(client.docs)
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore')
    monkeypatch.setattr(build_readme, 'get_state_store', lambda **kwargs: state.get_state_store(**kwargs, firestore_client=client))
    monkeypatch.setattr(adk_pipeline, 'run_daily_zeitgeist', lambda **kwargs: pytest.fail('Missing authority cannot trigger provider calls'))
    with pytest.raises(RuntimeError, match='Canonical'):
        build_readme.render_profile_markdown(root)
    assert files(root) == baseline
    assert client.docs == canonical_before


def test_projection_drops_stale_checkout_vote_and_cache_fields_without_merging(tmp_path):
    root, local, game, client = fixtures(tmp_path)
    local_pulse = json.loads(local.pulse_state_path.read_text())
    local_pulse.update(vote_rate_limits={'private-voter': True}, cached_results={'untrusted': True}, recent_words=['UNTRUSTED'])
    local.pulse_state_path.write_text(json.dumps(local_pulse))
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    store.canonical_projections(write_mirrors=True)
    mirror = json.loads(local.pulse_state_path.read_text())
    assert mirror == client.docs['profile_state/daily_pulse']
    assert 'vote_rate_limits' not in mirror and 'cached_results' not in mirror and 'recent_words' not in mirror


def test_readonly_renderer_does_not_write_missing_mirrors_or_canonical_documents(tmp_path, monkeypatch):
    from scripts import build_readme
    root, local, game, client = fixtures(tmp_path)
    (root/'data/zeitgeist_state.json').unlink()
    baseline = files(root); canonical_before = copy.deepcopy(client.docs)
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore')
    monkeypatch.setattr(build_readme, 'get_state_store', lambda **kwargs: state.get_state_store(**kwargs, firestore_client=client))
    assert 'CANONICAL' in build_readme.render_profile_markdown(root, write_assets=False)
    assert files(root) == baseline
    assert client.docs == canonical_before
