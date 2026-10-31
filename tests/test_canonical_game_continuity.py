"""Persist actual game-engine domain state across independent runtime handlers."""
import copy
from datetime import datetime, timezone
import json
import pytest
from src import game_engine as game
from src import state_store as stores
from scripts import process_visitor_event as visitor
from tests.test_canonical_runtime_factory import fixtures

BASE = 1793350000
PAST = [{'round_id': 6, 'game_key': 'connect4', 'winner': 'draw', 'turns_played': 21, 'winning_player': 'previous-player'}]


def runtime(monkeypatch, root, client):
    instances = []
    def factory(**kwargs):
        store = stores.get_state_store(**kwargs, firestore_client=client)
        instances.append(store)
        return store
    monkeypatch.setenv('STATE_STORE_BACKEND', 'firestore')
    monkeypatch.setattr(visitor, 'get_state_store', factory)
    return instances


def load(root, client):
    return stores.get_state_store(root, backend='firestore', firestore_client=client).load_game_state()


def near_complete(root, client):
    state = game.create_initial_game_state(round_id=7)
    board = [[0]*7 for _ in range(6)]; trace = []
    for column, token in [(0,1),(4,2),(1,1),(5,2),(2,1),(6,2)]:
        board[5][column] = token
        trace.append(game.canonical_board_key(board))
    state.update(board=board, turn=4, moves_this_round=6, episode_trace=trace,
                 red_contributors={'alice': 3}, past_rounds=copy.deepcopy(PAST))
    store = stores.get_state_store(root, backend='firestore', firestore_client=client)
    store.save_game_state(state)
    return trace


def test_two_independent_handler_invocations_keep_trace_counts_history_and_rate_limits(tmp_path, monkeypatch):
    root, local, initial, client = fixtures(tmp_path)
    initial['past_rounds'] = copy.deepcopy(PAST)
    client.docs['profile_state/active_game'] = initial
    instances = runtime(monkeypatch, root, client)
    first = visitor.process_event(root, 'alice', 'game:move:connect4:col:4:turn=1', issue_number=11, timestamp_epoch=BASE)
    first_state = load(root, client)
    assert first['accepted'] is True and first_state['moves_this_round'] == 2
    first_trace = list(first_state['episode_trace'])
    second = visitor.process_event(root, 'alice', 'game:move:connect4:col:3:turn=2', issue_number=12, timestamp_epoch=BASE+60)
    final = load(root, client)
    assert second['accepted'] is True
    assert len(instances) == 2 and instances[0] is not instances[1]
    assert final['moves_this_round'] == 4
    assert final['episode_trace'][:2] == first_trace and len(final['episode_trace']) == 4
    assert final['past_rounds'] == PAST
    assert final['last_move_epoch'] == final['last_move_timestamp'] == BASE+60
    assert final['last_challenger_move'] == 'col3' and final['last_bot_move']
    rate = client.docs['profile_state/game_rate_limits']['users']['alice']
    assert rate['daily_games_count'] == 1
    assert rate['hourly_timestamps'] == [BASE, BASE+60]
    mirror = json.loads((root/'data/game_state.json').read_text())
    assert mirror['episode_trace'] == final['episode_trace']
    assert mirror['moves_this_round'] == 4
    assert game.encode_bitboard(final['board']) == (final['bitboard_p1'], final['bitboard_p2'])


def test_completed_round_learning_rewards_and_new_round_survive_fresh_factory_reads(tmp_path, monkeypatch):
    root, local, initial, client = fixtures(tmp_path)
    prior_trace = near_complete(root, client)
    runtime(monkeypatch, root, client)
    title = 'game:move:connect4:col:4:turn=4'
    result = visitor.process_event(root, 'alice', title, issue_number=21, timestamp_epoch=BASE)
    completed = load(root, client)
    assert result['accepted'] and result['round_over'] and result['winner'] == 'red'
    assert completed['status'] == 'COMPLETED' and completed['round_over'] is True
    assert completed['round_id'] == 7 and completed['turn'] == 5
    assert completed['finisher_handle'] == 'alice' and completed['last_elo_delta'] == 37
    assert completed['moves_this_round'] == 7
    assert completed['episode_trace'][:6] == prior_trace
    assert len(completed['episode_trace']) == 7
    assert completed['past_rounds'] == PAST
    ledger = client.docs['profile_state/elo_ledger']
    memory = client.docs['profile_state/bot_memory']
    assert ledger['players']['alice']['elo'] == 837
    assert ledger['players']['alice']['wins'] == ledger['players']['alice']['finishes'] == 1
    assert ledger['recent_rounds'][0]['moves_count'] == 7
    assert memory['games_learned'] == 1
    assert all(memory['states'][key]['n'] == 1 for key in completed['episode_trace'])
    assert all(memory['states'][key]['v'] < 0 for key in completed['episode_trace'])
    before_retry = copy.deepcopy(client.docs)
    stale = visitor.process_event(root, 'alice', title, issue_number=21, timestamp_epoch=BASE+60)
    assert stale['accepted'] is False and stale['status'] == 'already_processed'
    assert client.docs == before_retry
    new = visitor.process_event(root, 'bob', 'game:move:connect4:col:1:turn=5', issue_number=22, timestamp_epoch=BASE+120)
    current = load(root, client)
    assert new['accepted'] is True and current['round_id'] == 8
    assert current['round_over'] is False and current['status'] == 'ACTIVE'
    assert current['moves_this_round'] == 2 and len(current['episode_trace']) == 2
    assert current['past_rounds'][:-1] == PAST
    assert current['past_rounds'][-1]['round_id'] == 7
    assert current['past_rounds'][-1]['board'] == completed['board']
    assert client.docs['profile_state/elo_ledger'] == ledger
    assert client.docs['profile_state/bot_memory']['games_learned'] == 1


def test_serialized_game_covers_actual_normalized_domain_without_operational_baggage(tmp_path):
    root, local, initial, client = fixtures(tmp_path)
    initial.update(_needs_git_reconcile=True, operational_metadata={'private': 'synthetic-only'})
    store = stores.get_state_store(root, backend='firestore', firestore_client=client)
    store.save_game_state(initial)
    stored = client.docs['profile_state/active_game']
    domain_keys = set(game.normalize_game_state({})) - {'board_state'}
    assert domain_keys <= set(stored)
    assert 'operational_metadata' not in stored and '_needs_git_reconcile' not in stored
    # board_state is a derived mirror reconstructed by the actual normalizer.
    assert load(root, client)['board_state']['past_rounds'] == initial['past_rounds']
