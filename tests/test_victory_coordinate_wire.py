"""Completed boards survive the real REST codec and exact legacy tuple strings."""
import copy
from io import BytesIO
import json

import pytest

from src.game_engine import find_winning_cells, normalize_game_state, render_board_svg
from src import state_store


def completed_state():
    # Board and coordinate shape from the failed owner terminal projection;
    # no account, receipt, reward, or private event metadata is needed to replay.
    board = [[0, 0, 0, 0, 0, 0, 0], [0, 0, 2, 1, 0, 0, 0],
             [0, 0, 2, 2, 0, 1, 0], [0, 0, 1, 2, 2, 2, 2],
             [0, 1, 1, 2, 2, 1, 1], [0, 1, 2, 1, 1, 1, 2]]
    return dict(board=board, winning_cells=find_winning_cells(board, player=2),
                winner='bot', status='COMPLETED', round_over=True,
                turn=12, mode='VS_BOT', round_id=1003, active_game='connect4')


def test_victory_roundtrips_real_rest_set_get_without_string_coordinates(monkeypatch):
    payload = {}
    def wire(request, **kwargs):
        if request.get_method() == 'PATCH':
            payload.update(json.loads(request.data))
        else:
            assert request.get_method() == 'GET'
        return BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(state_store.urllib.request, 'urlopen', wire)
    client = state_store._FirestoreRestClient('synthetic', access_token='synthetic')
    state = completed_state()
    client.set_document('profile_state', 'active_game',
                        state_store.FirestoreStateStore.game_state_document(state))
    restored = client.get_document('profile_state', 'active_game').to_dict()
    assert restored['winning_cells'] == [[3, 3], [3, 4], [3, 5], [3, 6]]
    svg = render_board_svg(restored)
    assert svg == render_board_svg(state)
    assert svg.count('stroke-width="3.5" filter="url(#win-glow)"') == 4
    assert 'x1="290" y1="210" x2="485" y2="210"' in svg


def test_exact_existing_legacy_tuple_strings_render_without_mutating_state(monkeypatch):
    state = completed_state()
    expected = render_board_svg(state)
    state['winning_cells'] = ['(3, 3)', '(3, 4)', '(3, 5)', '(3, 6)']
    before = copy.deepcopy(state)
    assert render_board_svg(state) == expected
    assert state == before
    projection = normalize_game_state(state)
    assert projection['winning_cells'] == [[3, 3], [3, 4], [3, 5], [3, 6]]
    assert projection['board'] == before['board']
    assert projection['winner'] == before['winner']
    assert state == before
    payload = {'fields': {key: state_store._py_to_firestore_value(value)
                          for key, value in state.items()}}
    def read_only_wire(request, **kwargs):
        assert request.get_method() == 'GET'
        return BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(state_store.urllib.request, 'urlopen', read_only_wire)
    client = state_store._FirestoreRestClient('synthetic', access_token='synthetic')
    store = state_store.FirestoreStateStore(client=client, canonical_only=True)
    assert store.load_game_state()['winning_cells'] == projection['winning_cells']
    assert state == before


@pytest.mark.parametrize('cells', [
    ['(6, 0)'], ['(0, 7)'], ['(-1, 0)'], ['(03, 3)'], ['(3,3)'],
    ['(3, 3) extra'], ['__import__("os").system("false")'],
    [[True, 3]], [[3.0, 3]], [[3, '3']], [[3]], [[3, 3, 3]],
    [[-1, 0]], [[0, 7]], [None], '[(3, 3)]', [[3, 3]] * 17,
])
def test_invalid_or_unbounded_coordinates_fail_closed(cells):
    state = completed_state()
    state['winning_cells'] = cells
    with pytest.raises(ValueError, match='winning coordinates'):
        render_board_svg(state)
