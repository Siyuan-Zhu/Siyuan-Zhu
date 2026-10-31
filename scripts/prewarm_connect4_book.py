#!/usr/bin/env python3
"""Offline Self-Play Prewarming Script for Connect 4 Bot Memory (data/bot_brain.json).

Runs 500 self-play episodes at depth=4 with epsilon=0.18 exploration to seed
~1,200 canonical opening/midgame states in < 5 seconds.
"""

from __future__ import annotations

import json
import pathlib
import random
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game_engine import (
    CENTER_ORDER,
    canonical_board_key,
    choose_bot_move,
    encode_bitboard,
    evaluate_tactical_window,
    has_won_bitboard,
    update_td_lambda_memory,
)


def _lowest_empty_row(board: list[list[int]], col: int) -> int | None:
    for r in range(5, -1, -1):
        if board[r][col] == 0:
            return r
    return None


def prewarm_connect4_book(
    episodes: int = 500,
    depth: int = 4,
    epsilon: float = 0.18,
    seed: int = 42,
) -> dict[str, Any]:
    """Run self-play episodes at `depth=4` (`epsilon=0.18`) to seed Tabular TD(lambda) memory."""
    rng = random.Random(seed)
    bot_memory: dict[str, Any] = {"version": 2, "games_learned": 0, "states": {}}
    move_cache: dict[tuple[str, int, int], int] = {}

    for _ep in range(episodes):
        board = [[0] * 7 for _ in range(6)]
        trace: list[str] = []
        player = 1
        winner = 0

        for _ply in range(1, 9):
            valid_cols = [c for c in CENTER_ORDER if board[0][c] == 0]
            if not valid_cols:
                break

            if rng.random() < epsilon:
                col = rng.choice(valid_cols)
            else:
                b_key = canonical_board_key(board)
                cache_key = (b_key, player, depth)
                if cache_key in move_cache:
                    col = move_cache[cache_key]
                else:
                    col = choose_bot_move(board, bot_memory, depth=depth, player=player)
                    move_cache[cache_key] = col

            row = _lowest_empty_row(board, col)
            if row is None:
                col = valid_cols[0]
                row = _lowest_empty_row(board, col)
            assert row is not None
            board[row][col] = player

            key = canonical_board_key(board)
            if len(bot_memory["states"]) < 1200 or key in bot_memory["states"]:
                trace.append(key)

            b1, b2 = encode_bitboard(board)
            pos = b1 if player == 1 else b2
            if has_won_bitboard(pos):
                winner = player
                break

            player = 2 if player == 1 else 1

        if winner == 2:
            reward = 1.0
        elif winner == 1:
            reward = -1.0
        else:
            reward = max(-1.0, min(1.0, evaluate_tactical_window(board, bot_memory) / 100.0))
        bot_memory = update_td_lambda_memory(bot_memory, trace, reward)

    return bot_memory


def main() -> int:
    out_path = REPO_ROOT / "data" / "bot_brain.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    memory = prewarm_connect4_book()
    out_path.write_text(json.dumps(memory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Prewarmed {len(memory['states'])} states across {memory['games_learned']} self-play games -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
