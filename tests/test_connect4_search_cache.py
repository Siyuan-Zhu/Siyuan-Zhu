"""A per-search pure-function cache preserves search work and TD decisions."""
import copy

import pytest

from src import game_engine as game


def midgame(mirrored):
    board = [[0 for _ in range(7)] for _ in range(6)]
    board[5][3] = board[5][2] = 1
    board[4][3] = board[5][4] = 2
    return [list(reversed(row)) for row in board] if mirrored else board


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("player", [1, 2])
@pytest.mark.parametrize("learned", [False, True])
def test_bounded_win_cache_preserves_move_work_and_td_memory(monkeypatch, mirrored, player, learned):
    board = midgame(mirrored)
    memory = {"states": {}, "games_learned": 0}
    if learned:
        # Give each available next position a different learned prior, so the
        # comparison covers TD-guided decisions as well as the tactical score.
        for column in range(7):
            child = copy.deepcopy(board)
            row = max(row for row in range(6) if not child[row][column])
            child[row][column] = player
            memory["states"][game.canonical_board_key(child)] = {"v": (column - 3) / 4, "n": 5}
    original_board, original_memory = copy.deepcopy(board), copy.deepcopy(memory)
    cached_stats, uncached_stats = {}, {}
    cached = game.choose_bot_move(board, memory, depth=8, player=player,
                                  time_budget_seconds=5, stats=cached_stats)
    # Remove memoization only, preserving the same rules, search, ordering,
    # depth and node budget as the production call above.
    # The uncached adapter retains the cache lifecycle interface while calling
    # the real win detector on every lookup.
    def uncached(*, maxsize):
        def decorate(function):
            def check(value):
                return function(value)
            check.cache_clear = lambda: None
            return check
        return decorate
    monkeypatch.setattr(game, "lru_cache", uncached)
    uncached_move = game.choose_bot_move(board, memory, depth=8, player=player,
                                         time_budget_seconds=5, stats=uncached_stats)
    assert cached == uncached_move
    for field in ("nodes", "completed_depth", "budget_exhausted", "tactical"):
        assert cached_stats[field] == uncached_stats[field]
    assert board == original_board and memory == original_memory
