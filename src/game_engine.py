r"""Deep Module for the Dual-Mode Connect 4 Community Game & Anti-Abuse Guard.

Implements `src/game_engine.py` following `/codebase-design`:
  - Small, deep interface:
    - `MoveRequest` & `MoveResult` (immutable `@dataclass(frozen=True)`)
    - `create_initial_game_state(game_key, round_id)`
    - `build_midgame_showcase_state(game_key, round_id)`
    - `process_move(state, request, repo_slug)`
    - `render_board_svg(state)`
    - `render_readme_game_section(state, repo_slug)`
  - Dedicated Dual-Mode Connect 4 (`VALID_GAMES = ("connect4",)`).
  - Five deterministic Anti-Abuse Guardrails inside `process_move`:
    1. `invalid_login`: GitHub handle regex validation (`\A[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z`)
    2. `wrong_game`: Active game key verification
    3. `stale_turn`: Optimistic concurrency turn lock (`expected_turn == state["turn"]`)
    4. `cooldown_active` / `rate_limited`: 300s same-player consecutive-move cooldown
       (0s wait for a different visitor) + 6 moves/hour per-user rolling cap
    5. `invalid_move`: Strict move whitelist (zero free-text injection possible)
"""

from __future__ import annotations

import copy
import time
from src.game_records import outcome_text, display_bot_copy
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Protocol
import urllib.parse
import xml.etree.ElementTree as ET


VALID_GAMES: tuple[str, ...] = ("connect4",)
DEFAULT_GAME_KEY: str = "connect4"

COOLDOWN_SECONDS: int = 300
MAX_MOVES_PER_USER_PER_HOUR: int = 6
RATE_LIMIT_WINDOW_SECONDS: float = 3600.0

GITHUB_LOGIN_RE = re.compile(r"\A[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")


def xml_escape(value: Any) -> str:
    """Escape text for safe inclusion inside SVG XML elements and attributes."""
    raw = str(value if value is not None else "")
    return (
        raw.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def verify_svg(svg_text: str) -> str:
    """Validate that `svg_text` parses cleanly via `xml.etree.ElementTree`."""
    cleaned = svg_text.strip()
    ET.fromstring(cleaned)
    return cleaned


@dataclass(frozen=True)
class MoveRequest:
    """Immutable visitor move request across the Community Game seam."""

    game_key: str
    move_id: str
    player_login: str
    expected_turn: int | None = None
    timestamp_epoch: float = 0.0
    issue_number: int = 0


@dataclass(frozen=True)
class MoveResult:
    """Immutable result returned by `process_move` across the Community Game seam."""

    accepted: bool
    status_code: str
    message: str
    player_move_summary: str
    bot_move_summary: str
    round_over: bool
    winner: str | None
    new_state: dict[str, Any]
    svg_board: str
    readme_section: str


@dataclass(frozen=True)
class _StepOutcome:
    """Internal mechanics outcome before shared move-log and SVG/Markdown assembly."""

    valid: bool
    canonical_move_id: str
    player_move_summary: str
    bot_move_summary: str
    short_status_line: str
    round_over: bool
    winner: str | None
    updated_state: dict[str, Any]
    error_message: str = ""


class _GameMechanics(Protocol):
    """Internal strategy interface for each rotatable game (avoids Repeated Switches)."""

    game_key: str
    heading_title: str
    board_alt: str
    how_it_works: str

    def initial_state(self, round_id: int) -> dict[str, Any]:
        ...

    def showcase_state(self, round_id: int) -> dict[str, Any]:
        ...

    def apply_move(
        self,
        state: dict[str, Any],
        raw_move_id: str,
        player_login: str,
    ) -> _StepOutcome:
        ...

    def render_svg(self, state: dict[str, Any], elo_ledger: dict[str, Any] | None = None) -> str:
        ...

    def render_controls_md(self, state: dict[str, Any], repo_slug: str) -> str:
        ...


# ============================================================================
# 1. Connect 4 Bitboard, 8-Ply Negamax, Tabular TD(lambda) & Dual-Mode Engine
# ============================================================================

_C4_ROWS = 6
_C4_COLS = 7
_C4_COL_ORDER = (3, 2, 4, 1, 5, 0, 6)
CENTER_ORDER = _C4_COL_ORDER
_C4_DEPTH = 8

MAX_TD_STATES = 5000
PVP_STALL_TIMEOUT_SECONDS = 900
STALE_TURN_TIMEOUT_SECONDS = 900
_TT_MAX_ENTRIES = 32768
_TT_EXACT = 0
_TT_LOWER = 1
_TT_UPPER = 2

_COL_MASK_7 = 0x7F
_CENTER_COL_MASK = 0x3F << 21


def _build_window_start_masks() -> tuple[tuple[int, int], ...]:
    """Precompute valid 4-in-a-row starting bitmasks for all 4 directions in 7x7 bitboard."""
    w_vert = 0
    for c in range(7):
        for h in range(3):
            w_vert |= 1 << (c * 7 + h)

    w_horiz = 0
    for c in range(4):
        for h in range(6):
            w_horiz |= 1 << (c * 7 + h)

    w_diag_down = 0  # d = 6 (c+1, h-1)
    for c in range(4):
        for h in range(3, 6):
            w_diag_down |= 1 << (c * 7 + h)

    w_diag_up = 0  # d = 8 (c+1, h+1)
    for c in range(4):
        for h in range(3):
            w_diag_up |= 1 << (c * 7 + h)

    return (
        (1, w_vert),
        (7, w_horiz),
        (6, w_diag_down),
        (8, w_diag_up),
    )


_WINDOW_DIR_MASKS: tuple[tuple[int, int], ...] = _build_window_start_masks()


def _cell_to_player(token: Any) -> int:
    """Normalize any board cell representation into 0 (empty), 1 (Red/Challenger), or 2 (Yellow/Bot)."""
    if token in (1, "1", "C", "R", "V", "c", "r", "v"):
        return 1
    if token in (2, "2", "B", "Y", "b", "y"):
        return 2
    return 0


def _c4_empty_board() -> list[list[str]]:
    return [["." for _ in range(_C4_COLS)] for _ in range(_C4_ROWS)]


def _c4_norm_cell(token: Any) -> str:
    p = _cell_to_player(token)
    if p == 1:
        return "C"
    if p == 2:
        return "B"
    return "."


def encode_bitboard(
    board: list[list[Any]],
    player: int | str | None = None,
) -> tuple[int, int]:
    """Encode a 6x7 Connect 4 board into 64-bit bitboards (7-bit column stride with sentinel row).

    Bit index = `col * 7 + (5 - row)` where `row=5` is bottom (`h=0`) and `h=6` is top sentinel `0`.
    If `player` is omitted, returns `(bitboard_p1, bitboard_p2)`.
    If `player` is specified (`1`/`2` or `'C'`/`'B'`), returns `(pos_player, mask_all)`.
    """
    b1 = 0
    b2 = 0
    for r in range(_C4_ROWS):
        h = 5 - r
        row = board[r]
        for c in range(_C4_COLS):
            p = _cell_to_player(row[c])
            if p == 1:
                b1 |= 1 << (c * 7 + h)
            elif p == 2:
                b2 |= 1 << (c * 7 + h)
    if player is None:
        return b1, b2
    p_idx = _cell_to_player(player)
    pos = b1 if p_idx == 1 else b2
    return pos, (b1 | b2)


def has_won_bitboard(bitboard: int) -> bool:
    """O(1) 4-direction shift-and-mask win check (`d in (1, 7, 6, 8)`)."""
    b = int(bitboard)
    for d in (1, 7, 6, 8):
        m = b & (b >> d)
        if m & (m >> (2 * d)):
            return True
    return False


def find_winning_cells(
    board_or_bitboard: list[list[Any]] | int,
    player: int | str | None = None,
) -> list[tuple[int, int]]:
    """Return the 4 `(row, col)` coordinates of a winning Connect 4 alignment, or `[]` if none."""
    if isinstance(board_or_bitboard, int):
        b = board_or_bitboard
    else:
        b1, b2 = encode_bitboard(board_or_bitboard)
        if player is not None:
            p_idx = _cell_to_player(player)
            b = b1 if p_idx == 1 else b2
        else:
            b = b1 if has_won_bitboard(b1) else (b2 if has_won_bitboard(b2) else 0)

    if not b:
        return []

    for d in (1, 7, 6, 8):
        w = b & (b >> d) & (b >> (2 * d)) & (b >> (3 * d))
        if w:
            lowest_bit = (w & -w).bit_length() - 1
            coords = [
                (5 - ((lowest_bit + i * d) % 7), (lowest_bit + i * d) // 7)
                for i in range(4)
            ]
            if d == 1:
                coords.sort(key=lambda rc: (rc[0], rc[1]))
            else:
                coords.sort(key=lambda rc: (rc[1], rc[0]))
            return coords
    return []


def _winning_coordinates(raw: Any) -> list[tuple[int, int]]:
    """Read typed cells or the exact bounded tuple strings written by old REST.

    This is a domain projection boundary, not a state migration. It never
    recomputes a winner or changes persisted coordinates, boards, or rewards.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) > 16:
        raise ValueError("Invalid winning coordinates")
    result = []
    for pair in raw:
        if isinstance(pair, str):
            legacy = re.fullmatch(r"\(([0-5]), ([0-6])\)", pair)
            if legacy is None:
                raise ValueError("Invalid winning coordinates")
            row, col = (int(value) for value in legacy.groups())
        elif (isinstance(pair, (list, tuple)) and len(pair) == 2
              and all(type(value) is int for value in pair)):
            row, col = pair
        else:
            raise ValueError("Invalid winning coordinates")
        if not (0 <= row < _C4_ROWS and 0 <= col < _C4_COLS):
            raise ValueError("Invalid winning coordinates")
        result.append((row, col))
    return result


def _mirror_bitboard(b: int) -> int:
    """Mirror a 7-column Connect 4 bitboard horizontally across column 3 in O(1)."""
    return (
        ((b & _COL_MASK_7) << 42)
        | (((b >> 7) & _COL_MASK_7) << 35)
        | (((b >> 14) & _COL_MASK_7) << 28)
        | (b & (_COL_MASK_7 << 21))
        | (((b >> 28) & _COL_MASK_7) << 14)
        | (((b >> 35) & _COL_MASK_7) << 7)
        | ((b >> 42) & _COL_MASK_7)
    )


def canonical_board_key(
    board_or_b1: list[list[Any]] | int,
    b2: int | None = None,
) -> str:
    """Compute a 26-char canonical hex key invariant under horizontal reflection (`min(key, mirror_key)`)."""
    if isinstance(board_or_b1, int):
        b1_int = int(board_or_b1)
        b2_int = int(b2 or 0)
    else:
        b1_int, b2_int = encode_bitboard(board_or_b1)
    k1 = f"{b1_int:013x}{b2_int:013x}"
    m1 = _mirror_bitboard(b1_int)
    m2 = _mirror_bitboard(b2_int)
    k2 = f"{m1:013x}{m2:013x}"
    return k1 if k1 <= k2 else k2


def _eval_bitboard_windows(b_me: int, b_opp: int) -> float:
    """Bitwise-parallel 69-window tactical evaluation + center column weight."""
    center_diff = (b_me & _CENTER_COL_MASK).bit_count() - (
        b_opp & _CENTER_COL_MASK
    ).bit_count()
    me_3 = 0
    me_2 = 0
    opp_3 = 0
    opp_2 = 0

    for d, w_mask in _WINDOW_DIR_MASKS:
        x0 = b_me & w_mask
        x1 = (b_me >> d) & w_mask
        x2 = (b_me >> (2 * d)) & w_mask
        x3 = (b_me >> (3 * d)) & w_mask

        y0 = b_opp & w_mask
        y1 = (b_opp >> d) & w_mask
        y2 = (b_opp >> (2 * d)) & w_mask
        y3 = (b_opp >> (3 * d)) & w_mask

        open_me = w_mask & ~(y0 | y1 | y2 | y3)
        open_opp = w_mask & ~(x0 | x1 | x2 | x3)

        if open_me:
            s01 = x0 ^ x1
            c01 = x0 & x1
            s23 = x2 ^ x3
            c23 = x2 & x3
            ex3_me = (s01 ^ s23) & (c01 | c23) & open_me
            ex2_me = (((c01 ^ c23) & ~(s01 ^ s23)) | (s01 & s23)) & open_me
            if ex3_me:
                me_3 += ex3_me.bit_count()
            if ex2_me:
                me_2 += ex2_me.bit_count()

        if open_opp:
            os01 = y0 ^ y1
            oc01 = y0 & y1
            os23 = y2 ^ y3
            oc23 = y2 & y3
            ex3_opp = (os01 ^ os23) & (oc01 | oc23) & open_opp
            ex2_opp = (((oc01 ^ oc23) & ~(os01 ^ os23)) | (os01 & os23)) & open_opp
            if ex3_opp:
                opp_3 += ex3_opp.bit_count()
            if ex2_opp:
                opp_2 += ex2_opp.bit_count()

    return (
        center_diff * 6.0
        + me_3 * 50.0
        + me_2 * 10.0
        - opp_3 * 80.0
        - opp_2 * 12.0
    )


def evaluate_tactical_window(
    board_or_b_me: list[list[Any]] | int,
    bot_memory_or_b_opp: dict[str, Any] | int | None = None,
    bot_memory: dict[str, Any] | None = None,
    perspective_player: int = 2,
) -> float:
    """Evaluate tactical 4-cell windows plus Tabular TD(lambda) value prior `120.0 * V(s)`."""
    if isinstance(board_or_b_me, int):
        b_me = int(board_or_b_me)
        b_opp = int(bot_memory_or_b_opp) if isinstance(bot_memory_or_b_opp, int) else 0
        mem = bot_memory if isinstance(bot_memory, dict) else (
            bot_memory_or_b_opp if isinstance(bot_memory_or_b_opp, dict) else None
        )
        base = _eval_bitboard_windows(b_me, b_opp)
        if mem and mem.get("states"):
            b1 = b_opp if perspective_player == 2 else b_me
            b2 = b_me if perspective_player == 2 else b_opp
            key = canonical_board_key(b1, b2)
            entry = mem["states"].get(key)
            if isinstance(entry, dict):
                v_val = float(entry.get("v", entry.get("V", 0.0)))
                sign = 1.0 if perspective_player == 2 else -1.0
                base += sign * 120.0 * v_val
        return base

    b1, b2 = encode_bitboard(board_or_b_me)
    b_me = b2 if perspective_player == 2 else b1
    b_opp = b1 if perspective_player == 2 else b2
    mem = (
        bot_memory_or_b_opp
        if isinstance(bot_memory_or_b_opp, dict)
        else bot_memory
    )
    base = _eval_bitboard_windows(b_me, b_opp)
    if mem and isinstance(mem.get("states"), dict):
        key = canonical_board_key(b1, b2)
        entry = mem["states"].get(key)
        if isinstance(entry, dict):
            v_val = float(entry.get("v", entry.get("V", 0.0)))
            sign = 1.0 if perspective_player == 2 else -1.0
            base += sign * 120.0 * v_val
    return base


def _column_heights_from_bitboards(b1: int, b2: int) -> list[int]:
    """Return a 7-element list of the next free bit index `c * 7 + h` for each column."""
    mask = b1 | b2
    heights: list[int] = []
    for c in range(7):
        base = c * 7
        col_bits = (mask >> base) & 0x3F
        heights.append(base + col_bits.bit_length())
    return heights


def choose_bot_move(
    board: list[list[Any]],
    bot_memory: dict[str, Any] | None = None,
    depth: int = _C4_DEPTH,
    player: int | str = 2,
    *, max_nodes: int = 50000, time_budget_seconds: float = 1.5,
    stats: dict[str, Any] | None = None,
) -> int:
    """Select the strongest Connect 4 column (`0..6`) via 8-ply Negamax + Alpha-Beta + TT + TD(lambda)."""
    started = time.monotonic()
    deadline = started + max(0.0,min(float(time_budget_seconds),5.0))
    node_limit = max(0,min(int(max_nodes),200000))
    nodes = 0
    completed_depth = 0
    exhausted = False
    def finish(column, tactical=False):
        if stats is not None:
            stats.update(nodes=nodes,completed_depth=completed_depth,budget_exhausted=exhausted,
                         tactical=tactical,elapsed_seconds=time.monotonic()-started)
        return column
    class SearchBudgetExhausted(Exception):
        pass
    b1, b2 = encode_bitboard(board)
    p_idx = _cell_to_player(player) or 2
    b_root_me = b2 if p_idx == 2 else b1
    b_root_opp = b1 if p_idx == 2 else b2

    heights = _column_heights_from_bitboards(b1, b2)
    legal_cols = [c for c in _C4_COL_ORDER if heights[c] < c * 7 + 6]
    if not legal_cols:
        return finish(3)

    # 1. Immediate 1-move win check
    for c in legal_cols:
        if has_won_bitboard(b_root_me | (1 << heights[c])):
            return finish(c,True)

    # 2. Immediate 1-move opponent threat block check
    for c in legal_cols:
        if has_won_bitboard(b_root_opp | (1 << heights[c])):
            return finish(c,True)

    states_dict: dict[str, Any] = (
        bot_memory.get("states", {})
        if isinstance(bot_memory, dict) and isinstance(bot_memory.get("states"), dict)
        else {}
    )
    has_td_states = bool(states_dict)
    tt: dict[tuple[int, int], tuple[int, int, float, int]] = {}

    def _negamax(
        b_me: int,
        b_opp: int,
        rem_depth: int,
        alpha: float,
        beta: float,
        me_is_p2: bool,
    ) -> float:
        nonlocal nodes
        if nodes >= node_limit or (nodes % 64 == 0 and time.monotonic() >= deadline):
            raise SearchBudgetExhausted()
        nodes += 1
        alpha_orig = alpha
        tt_entry = tt.get((b_me, b_opp))
        tt_best_col = -1
        if tt_entry is not None:
            entry_depth, entry_flag, entry_val, tt_best_col = tt_entry
            if entry_depth >= rem_depth:
                if entry_flag == _TT_EXACT:
                    return entry_val
                if entry_flag == _TT_LOWER:
                    if entry_val > alpha:
                        alpha = entry_val
                elif entry_flag == _TT_UPPER:
                    if entry_val < beta:
                        beta = entry_val
                if alpha >= beta:
                    return entry_val

        # Collect playable columns
        playable: list[int] = []
        for c in _C4_COL_ORDER:
            if heights[c] < c * 7 + 6:
                playable.append(c)
        if not playable:
            return 0.0

        # Check immediate 1-ply win for active player
        for c in playable:
            if has_won_bitboard(b_me | (1 << heights[c])):
                return 100000.0 + rem_depth * 1000.0

        if rem_depth <= 0:
            score = _eval_bitboard_windows(b_me, b_opp)
            if has_td_states:
                cur_b1 = b_opp if me_is_p2 else b_me
                cur_b2 = b_me if me_is_p2 else b_opp
                entry = states_dict.get(canonical_board_key(cur_b1, cur_b2))
                if isinstance(entry, dict):
                    v_val = float(entry.get("v", entry.get("V", 0.0)))
                    score += (120.0 * v_val) if me_is_p2 else (-120.0 * v_val)
            return score

        # Forced-block check: if opponent threatens an immediate win on their next turn
        forced_col = -1
        for c in playable:
            if has_won_bitboard(b_opp | (1 << heights[c])):
                if forced_col != -1:
                    # Double threat: opponent has >=2 immediate winning columns -> unavoidable loss next ply
                    return -(100000.0 + (rem_depth - 1) * 1000.0)
                forced_col = c

        if forced_col != -1:
            ordered_cols = [forced_col]
        elif tt_best_col in playable and playable[0] != tt_best_col:
            ordered_cols = [tt_best_col] + [c for c in playable if c != tt_best_col]
        else:
            ordered_cols = playable

        best_val = -1e12
        best_col = ordered_cols[0]
        for c in ordered_cols:
            bit_idx = heights[c]
            heights[c] = bit_idx + 1
            val = -_negamax(
                b_opp,
                b_me | (1 << bit_idx),
                rem_depth - 1,
                -beta,
                -alpha,
                not me_is_p2,
            )
            heights[c] = bit_idx
            if val > best_val:
                best_val = val
                best_col = c
            if best_val > alpha:
                alpha = best_val
            if alpha >= beta:
                break

        if len(tt) < _TT_MAX_ENTRIES:
            if best_val <= alpha_orig:
                flag = _TT_UPPER
            elif best_val >= beta:
                flag = _TT_LOWER
            else:
                flag = _TT_EXACT
            tt[(b_me, b_opp)] = (rem_depth, flag, best_val, best_col)

        return best_val

    best_root_col = legal_cols[0]  # Legal center-first fallback; tactics checked above.
    root_is_p2 = p_idx == 2
    for search_depth in range(1,min(max(1,int(depth)),8)+1):
        iteration_col = legal_cols[0]
        best_root_val = -1e12
        alpha, beta = -1e12, 1e12
        try:
            for c in legal_cols:
                bit_idx = heights[c]
                b_next_me = b_root_me | (1 << bit_idx)
                heights[c] = bit_idx + 1
                try:
                    val = -_negamax(b_root_opp,b_next_me,search_depth-1,
                                    -beta,-alpha,not root_is_p2)
                finally:
                    heights[c] = bit_idx
                if has_td_states:
                    child_b1 = b_root_opp if root_is_p2 else b_next_me
                    child_b2 = b_next_me if root_is_p2 else b_root_opp
                    entry = states_dict.get(canonical_board_key(child_b1,child_b2))
                    if isinstance(entry,dict):
                        val += (120.0 if root_is_p2 else -120.0)*float(entry.get('v',entry.get('V',0.0)))
                if val > best_root_val:
                    best_root_val, iteration_col = val, c
                alpha = max(alpha,best_root_val)
        except SearchBudgetExhausted:
            exhausted = True
            break
        best_root_col = iteration_col
        completed_depth = search_depth
    return finish(best_root_col)


def update_td_lambda_memory(
    bot_memory: dict[str, Any],
    episode_trace: list[str],
    outcome_for_bot: float,
    alpha: float = 0.08,
    gamma: float = 0.95,
    lam: float = 0.7,
    round_id: int | None = None,
) -> dict[str, Any]:
    """Apply backward eligibility-trace Tabular TD(lambda) update to `bot_memory` (capped at 5,000 states)."""
    if not isinstance(bot_memory, dict):
        bot_memory = {"version": 2, "games_learned": 0, "states": {}}

    bot_memory.setdefault("version", 2)
    states: dict[str, dict[str, Any]] = bot_memory.setdefault("states", {})
    clean_trace = [str(k) for k in (episode_trace or []) if k]
    if not clean_trace:
        return bot_memory

    games_learned = int(bot_memory.get("games_learned", 0)) + 1
    bot_memory["games_learned"] = games_learned
    cur_round = int(round_id if round_id is not None else games_learned)
    reward = max(-1.0, min(1.0, float(outcome_for_bot)))

    # Ensure all visited states exist in `states` before computing temporal differences
    for s_key in clean_trace:
        if s_key not in states or not isinstance(states[s_key], dict):
            states[s_key] = {"v": 0.0, "n": 0, "round": cur_round}

    eligibility: dict[str, float] = {}
    t_len = len(clean_trace)
    for t in range(t_len):
        s_t = clean_trace[t]
        v_t = float(states[s_t].get("v", states[s_t].get("V", 0.0)))
        if t < t_len - 1:
            s_next = clean_trace[t + 1]
            v_next = float(states[s_next].get("v", states[s_next].get("V", 0.0)))
            delta_t = gamma * v_next - v_t
        else:
            delta_t = reward - v_t

        for k in list(eligibility.keys()):
            eligibility[k] *= gamma * lam
        eligibility[s_t] = eligibility.get(s_t, 0.0) + 1.0

        for k, e_val in eligibility.items():
            entry = states[k]
            old_v = float(entry.get("v", entry.get("V", 0.0)))
            new_v = max(-1.0, min(1.0, old_v + alpha * delta_t * e_val))
            rounded_v = round(new_v, 6)
            entry["v"] = rounded_v
            entry.pop("V", None)

    seen_in_episode: set[str] = set()
    for s_key in clean_trace:
        if s_key not in seen_in_episode:
            seen_in_episode.add(s_key)
            entry = states[s_key]
            entry["n"] = int(entry.get("n", 0)) + 1
            entry["round"] = cur_round

    if len(states) > MAX_TD_STATES:
        excess = len(states) - MAX_TD_STATES
        prune_order = sorted(
            states.keys(),
            key=lambda k: (
                int(states[k].get("n", 0)),
                int(states[k].get("round", 0)),
                k,
            ),
        )
        for k in prune_order[:excess]:
            del states[k]

    return bot_memory


def check_and_heal_stale_turn(
    state: dict[str, Any],
    requested_turn: int | None,
) -> tuple[bool, str]:
    """Validate whether `requested_turn` matches the active board turn or a completed round's reset link."""
    if requested_turn is None:
        return (True, "ok")
    status = str(state.get("status", "ACTIVE")).upper()
    cur_turn = int(state.get("turn", 1))
    req_turn = int(requested_turn)
    if status == "COMPLETED" or bool(state.get("round_over", False)):
        if req_turn in (1, cur_turn):
            return (True, "ok")
        return (False, "stale_turn")
    if req_turn == cur_turn:
        return (True, "ok")
    return (False, "stale_turn")


def _c4_available_row(board: list[list[Any]], col_idx: int) -> int | None:
    for r in range(_C4_ROWS - 1, -1, -1):
        if _cell_to_player(board[r][col_idx]) == 0:
            return r
    return None


def _c4_drop_disc(
    board: list[list[Any]], col_idx: int, token: str
) -> tuple[list[list[str]], int]:
    row_idx = _c4_available_row(board, col_idx)
    if row_idx is None:
        raise ValueError(f"Column {col_idx + 1} is full")
    next_board = [[_c4_norm_cell(cell) for cell in row] for row in board]
    next_board[row_idx][col_idx] = token
    return next_board, row_idx


def _c4_find_winning_line(
    board: list[list[Any]], token: str
) -> list[tuple[int, int]] | None:
    cells = find_winning_cells(board, player=token)
    return cells if cells else None


def _c4_is_full(board: list[list[Any]]) -> bool:
    return all(_cell_to_player(board[0][c]) != 0 for c in range(_C4_COLS))


def _c4_select_bot_col(
    board: list[list[Any]],
    bot_memory: dict[str, Any] | None = None,
) -> int | None:
    legal_cols = [c for c in _C4_COL_ORDER if _c4_available_row(board, c) is not None]
    if not legal_cols:
        return None
    return choose_bot_move(board, bot_memory=bot_memory, depth=_C4_DEPTH, player=2)


class _Connect4Mechanics:
    game_key = "connect4"
    heading_title = "Connect Four"
    board_alt = "Connect Four board"
    how_it_works = (
        "Each column link opens a pre-filled 1-click GitHub Issue with the current turn number. "
        "When submitted, GitHub Actions validates the turn lock and per-player cooldown, drops "
        "your Terracotta disc, lets House - Bot answer in Charcoal, updates the "
        "SVG board, and closes the issue automatically."
    )

    def initial_state(self, round_id: int = 1) -> dict[str, Any]:
        return {
            "active_game": "connect4",
            "game_key": "connect4",
            "round_id": int(round_id),
            "turn": 1,
            "mode": "VS_BOT",
            "status": "ACTIVE",
            "current_turn_color": "RED",
            "round_over": False,
            "winner": None,
            "finisher_handle": None,
            "last_elo_delta": 0,
            "moves_this_round": 0,
            "board": _c4_empty_board(),
            "red_contributors": {},
            "yellow_contributors": {},
            "episode_trace": [],
            "last_challenger_move": None,
            "last_bot_move": None,
            "winning_cells": [],
            "last_mover": None,
            "last_move_epoch": 0.0,
            "last_move_timestamp": 0.0,
            "last_move_summary": "New round ready — click any column below to drop the first Terracotta disc.",
            "move_log": [],
            "past_rounds": [],
            "rate_limits": {},
        }

    def showcase_state(self, round_id: int = 1) -> dict[str, Any]:
        state = self.initial_state(round_id=round_id)
        board = _c4_empty_board()
        # Match the exact mid-game showcase board locked in Ticket #14
        board[5][1] = "C"
        board[5][2] = "C"
        board[5][3] = "B"
        board[5][4] = "B"
        board[5][5] = "C"
        board[4][2] = "B"
        board[4][3] = "C"
        board[4][4] = "B"
        board[3][3] = "C"
        state["board"] = board
        state["turn"] = 4
        state["moves_this_round"] = 9
        state["red_contributors"] = {"sam-patel": 1, "maya-lin": 1, "alex-chen": 1}
        state["last_challenger_move"] = "col4"
        state["last_bot_move"] = "col5"
        state["last_mover"] = "alex-chen"
        state["last_move_epoch"] = 1729700000.0
        state["last_move_timestamp"] = 1729700000.0
        state["last_move_summary"] = (
            "<code>@alex-chen</code> dropped in col 4 (bot answered col 5) · turn 4"
        )
        state["move_log"] = [
            {
                "turn": 1,
                "game_key": "connect4",
                "round_id": int(round_id),
                "player": "sam-patel",
                "move_id": "col3",
                "player_move": "Dropped in col 3",
                "bot_reply": "Answered col 4",
                "timestamp_epoch": 1729699200.0,
                "issue_number": 12,
            },
            {
                "turn": 2,
                "game_key": "connect4",
                "round_id": int(round_id),
                "player": "maya-lin",
                "move_id": "col4",
                "player_move": "Dropped in col 4",
                "bot_reply": "Answered col 3",
                "timestamp_epoch": 1729699600.0,
                "issue_number": 15,
            },
            {
                "turn": 3,
                "game_key": "connect4",
                "round_id": int(round_id),
                "player": "alex-chen",
                "move_id": "col4",
                "player_move": "Dropped in col 4",
                "bot_reply": "Answered col 5",
                "timestamp_epoch": 1729700000.0,
                "issue_number": 19,
            },
        ]
        return state

    def apply_move(
        self,
        state: dict[str, Any],
        raw_move_id: str,
        player_login: str,
    ) -> _StepOutcome:
        working = copy.deepcopy(state)
        board: list[list[Any]] = working.get("board") or _c4_empty_board()

        norm = raw_move_id.strip().lower()
        if norm.startswith("drop "):
            norm = f"col{norm.split('drop ', 1)[1].strip()}"
        if norm.startswith("col:") and norm[4:].isdigit():
            norm = f"col{norm[4:]}"
        if not re.fullmatch(r"col[1-7]", norm):
            return _StepOutcome(
                valid=False,
                canonical_move_id=norm,
                player_move_summary="",
                bot_move_summary="",
                short_status_line="",
                round_over=False,
                winner=None,
                updated_state=working,
                error_message=f"Invalid Connect 4 move '{raw_move_id}'. Choose col1 through col7.",
            )

        col_idx = int(norm[3:]) - 1
        if _c4_available_row(board, col_idx) is None:
            return _StepOutcome(
                valid=False,
                canonical_move_id=norm,
                player_move_summary="",
                bot_move_summary="",
                short_status_line="",
                round_over=False,
                winner=None,
                updated_state=working,
                error_message=f"Column {col_idx + 1} is already full.",
            )

        board_after_p, _ = _c4_drop_disc(board, col_idx, "C")
        player_summary = f"Dropped in col {col_idx + 1}"

        p_win = _c4_find_winning_line(board_after_p, "C")
        round_over = False
        winner: str | None = None
        winning_cells: list[list[int]] = []
        bot_summary = "Bot stood down (round complete)"
        bot_short = "round complete"
        bot_move_id: str | None = None

        if p_win is not None:
            round_over = True
            winner = "community"
            winning_cells = [[r, c] for r, c in p_win]
            final_board = board_after_p
            bot_summary = "Community connected 4!"
            bot_short = "community won!"
        elif _c4_is_full(board_after_p):
            round_over = True
            winner = "draw"
            final_board = board_after_p
            bot_summary = "Board full (draw)"
            bot_short = "board full — draw"
        else:
            bot_col = _c4_select_bot_col(
                board_after_p, bot_memory=working.get("bot_memory")
            )
            if bot_col is not None:
                final_board, _ = _c4_drop_disc(board_after_p, bot_col, "B")
                bot_move_id = f"col{bot_col + 1}"
                bot_summary = f"Answered col {bot_col + 1}"
                bot_short = f"bot answered col {bot_col + 1}"
                b_win = _c4_find_winning_line(final_board, "B")
                if b_win is not None:
                    round_over = True
                    winner = "house_bot"
                    winning_cells = [[r, c] for r, c in b_win]
                elif _c4_is_full(final_board):
                    round_over = True
                    winner = "draw"
            else:
                final_board = board_after_p
                round_over = True
                winner = "draw"

        next_turn = int(working.get("turn", 1)) + 1
        working["board"] = final_board
        working["last_challenger_move"] = norm
        working["last_bot_move"] = bot_move_id
        working["winning_cells"] = winning_cells
        working["turn"] = next_turn
        working["round_over"] = round_over
        working["winner"] = winner
        working["status"] = "COMPLETED" if round_over else "ACTIVE"

        short_line = (
            f"<code>@{xml_escape(player_login)}</code> dropped in col {col_idx + 1} "
            f"({bot_short}) · turn {next_turn}"
        )
        return _StepOutcome(
            valid=True,
            canonical_move_id=norm,
            player_move_summary=player_summary,
            bot_move_summary=display_bot_copy(bot_summary),
            short_status_line=short_line,
            round_over=round_over,
            winner=winner,
            updated_state=working,
        )

    def render_svg(self, state: dict[str, Any], elo_ledger: dict[str, Any] | None = None) -> str:
        board: list[list[Any]] = state.get("board") or _c4_empty_board()
        raw_win_cells = state.get("winning_cells", [])
        win_coords = _winning_coordinates(raw_win_cells)
        win_set = set(win_coords)
        winner = state.get("winner")
        status_upper = str(state.get("status", "ACTIVE")).upper()
        is_completed_win = (
            status_upper == "COMPLETED" or bool(state.get("round_over", False))
        ) and bool(win_coords)

        defs_svg = (
            "  <defs>\n"
            '    <filter id="win-glow" x="-40%" y="-40%" width="180%" height="180%">\n'
            '      <feGaussianBlur stdDeviation="4" result="blur"/>\n'
            "      <feMerge>\n"
            '        <feMergeNode in="blur"/>\n'
            '        <feMergeNode in="SourceGraphic"/>\n'
            "      </feMerge>\n"
            "    </filter>\n"
            '    <linearGradient id="victory-beam" x1="0%" y1="0%" x2="100%" y2="0%">\n'
            '      <stop offset="0%" stop-color="#C85A32"/>\n'
            '      <stop offset="50%" stop-color="#9B482D"/>\n'
            '      <stop offset="100%" stop-color="#D88667"/>\n'
            "    </linearGradient>\n"
            "  </defs>\n"
        )

        col_headers = "".join(
            f'<text x="{95 + c * 65}" y="47" text-anchor="middle" fill="#716A62" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
            f'font-size="12" font-weight="600">{("I", "II", "III", "IV", "V", "VI", "VII")[c]}</text>'
            for c in range(_C4_COLS)
        )

        circles: list[str] = []
        outer_rings: list[str] = []
        for r in range(_C4_ROWS):
            cy = 66 + r * 48
            for c in range(_C4_COLS):
                cx = 95 + c * 65
                cell = _c4_norm_cell(board[r][c])
                is_win = (r, c) in win_set
                if cell == "C":
                    if is_win:
                        circles.append(
                            f'<circle cx="{cx}" cy="{cy}" r="18" fill="#C85A32" '
                            f'stroke="#9B482D" stroke-width="3.5" filter="url(#win-glow)"/>'
                        )
                        outer_rings.append(
                            f'<circle cx="{cx}" cy="{cy}" r="23" fill="none" '
                            f'stroke="#C85A32" stroke-width="2" opacity="0.85" stroke-dasharray="4 3"/>'
                        )
                    elif is_completed_win:
                        circles.append(
                            f'<circle cx="{cx}" cy="{cy}" r="18" fill="#C85A32" '
                            f'stroke="#B64B26" stroke-width="1.5" opacity="0.35"/>'
                        )
                    else:
                        circles.append(
                            f'<circle cx="{cx}" cy="{cy}" r="18" fill="#C85A32" '
                            f'stroke="#B64B26" stroke-width="1.5"/>'
                        )
                elif cell == "B":
                    if is_win:
                        circles.append(
                            f'<circle cx="{cx}" cy="{cy}" r="18" fill="#1C1917" '
                            f'stroke="#9B482D" stroke-width="3.5" filter="url(#win-glow)"/>'
                        )
                        outer_rings.append(
                            f'<circle cx="{cx}" cy="{cy}" r="23" fill="none" '
                            f'stroke="#C85A32" stroke-width="2" opacity="0.85" stroke-dasharray="4 3"/>'
                        )
                    elif is_completed_win:
                        circles.append(
                            f'<circle cx="{cx}" cy="{cy}" r="18" fill="#1C1917" '
                            f'stroke="#4A4540" stroke-width="1.5" opacity="0.35"/>'
                        )
                    else:
                        circles.append(
                            f'<circle cx="{cx}" cy="{cy}" r="18" fill="#1C1917" '
                            f'stroke="#4A4540" stroke-width="1.5"/>'
                        )
                else:
                    circles.append(
                        f'<circle cx="{cx}" cy="{cy}" r="18" fill="#FAF8F5" '
                        f'stroke="#DED8CF" stroke-width="1.2"/>'
                    )

        victory_beam_svg = ""
        header_banner_svg = ""
        if is_completed_win and len(win_coords) >= 2:
            r_start, c_start = win_coords[0]
            r_end, c_end = win_coords[-1]
            cx_start = 95 + c_start * 65
            cy_start = 66 + r_start * 48
            cx_end = 95 + c_end * 65
            cy_end = 66 + r_end * 48
            victory_beam_svg = (
                f'  <line x1="{cx_start}" y1="{cy_start}" x2="{cx_end}" y2="{cy_end}" '
                f'stroke="url(#victory-beam)" stroke-width="6" stroke-linecap="round" filter="url(#win-glow)"/>\n'
            )

        charcoal_role = "Visitors" if str(state.get("mode", "VS_BOT")).upper() == "PVP_DUEL" else "House - Bot"
        footer_svg = (
            '  <circle cx="80" cy="364" r="3.6" fill="#C85A32"/>\n'
            '  <text x="92" y="368" fill="#514A43" font-size="13">Terracotta · Visitors</text>\n'
            '  <circle cx="355" cy="364" r="3.6" fill="#1C1917"/>\n'
            f'  <text x="366" y="368" fill="#716A62" font-size="13">Charcoal · {charcoal_role}</text>\n'
        )

        grid_path = " ".join(
            [f"M {127.5 + c * 65} 58 V 356" for c in range(6)]
            + [f"M 62 {110 + r * 48} H 518" for r in range(5)]
        )
        theme_css = (
            '<style>text { font-family: "IBM Plex Mono", ui-monospace, monospace; }\n'
            '.editorial-heading { font-family: "Cormorant Garamond", Georgia, serif; }\n'
            '@media (prefers-color-scheme: dark) {\n'
            '[fill="#FAF8F5"] { fill: #141312; }\n'
            '[fill="#F1EEEA"] { fill: #1C1917; }\n'
            '[fill="#1C1917"] { fill: #E7E2DA; }\n'
            '[fill="#514A43"], [fill="#716A62"] { fill: #C8C1B8; }\n'
            '[fill="#9B482D"] { fill: #D88667; }\n'
            '[stroke="#DED8CF"], .board-grid { stroke: #2A2926; }\n'
            '} </style>\n'
        )
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 580 360" width="100%" role="img" aria-label="Connect Four board — terracotta and charcoal discs">\n'
            f"{theme_css}"
            f"{defs_svg}"
            '  <rect width="580" height="360" rx="4" fill="#FAF8F5" stroke="#DED8CF" stroke-width="1"/>\n'
            '  <g transform="translate(0 -28)">\n'
            '  <rect x="52" y="58" width="476" height="298" fill="#F1EEEA" stroke="#DED8CF" stroke-width="1"/>\n'
            f'  <path class="board-grid" d="{grid_path}" fill="none" stroke="#D6D0C4" stroke-width="1"/>\n'
            f"  {col_headers}\n"
            f"{header_banner_svg}"
            '  <g transform="translate(0 20)">\n'
            f"  {''.join(circles)}\n"
            f"  {''.join(outer_rings)}\n"
            f"{victory_beam_svg}"
            "  </g>\n"
            f'<g transform="translate(0 5)">{footer_svg}</g>\n'
            "  </g>\n"
            "</svg>"
        )
        return verify_svg(svg)

    def render_controls_md(self, state: dict[str, Any], repo_slug: str) -> str:
        board: list[list[str]] = state.get("board") or _c4_empty_board()
        turn = int(state.get("turn", 1))
        round_id = int(state.get("round_id", 1))
        mode = str(state.get("mode", "VS_BOT")).upper()
        status_upper = str(state.get("status", "ACTIVE")).upper()
        cells: list[str] = []
        for c in range(_C4_COLS):
            col_num = c + 1
            if status_upper != "COMPLETED" and _c4_available_row(board, c) is None:
                cells.append("—")
            else:
                title_param = urllib.parse.quote(
                    f"game:move:connect4:col:{col_num}:turn={turn}:round={round_id}", safe=":="
                )
                disc_color = "Charcoal" if status_upper != "COMPLETED" and mode == "PVP_DUEL" and turn % 2 == 0 else "Terracotta"
                instruction = (
                    f"Just click 'Submit new issue' to start Round {round_id + 1} with a Terracotta disc in Column {col_num}."
                    if status_upper == "COMPLETED" else
                    f"Just click 'Submit new issue' to place a {disc_color} disc in Column {col_num} (Turn {turn})."
                )
                instruction += " Processing is asynchronous. " + (
                    "Both sides are visitors on this shared board. This is not a private room." if mode == "PVP_DUEL" and status_upper != "COMPLETED" else "House - Bot is automated."
                )
                body_param = urllib.parse.quote_plus(instruction)
                url = f"https://github.com/{repo_slug}/issues/new?title={title_param}&body={body_param}"
                cells.append(f'[{("I", "II", "III", "IV", "V", "VI", "VII")[c]}]({url} "Place a disc in column {col_num}")')
        header_row = "| " + " | ".join(cells) + " |"
        align_row = "| :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        mode_line = ""
        if mode == "PVP_DUEL":
            m_title = urllib.parse.quote(
                f"game:mode:connect4:new_vs_bot:turn={turn}:round={round_id}", safe=":="
            )
            m_body = urllib.parse.quote_plus(
                "Just click 'Submit new issue' to let House - Bot take the next turn."
            )
            m_url = f"https://github.com/{repo_slug}/issues/new?title={m_title}&body={m_body}"
            takeover_note = " (after 15 minutes idle)" if turn > 1 and status_upper != "COMPLETED" else ""
            mode_line = f"\n\n<sub><a href=\"{m_url}\">Let House - Bot play</a>{takeover_note}</sub>"
        elif turn <= 2 or status_upper == "COMPLETED":
            m_title = urllib.parse.quote(
                f"game:mode:connect4:new_pvp:turn={turn}:round={round_id}", safe=":="
            )
            m_body = urllib.parse.quote_plus(
                "Just click 'Submit new issue' to switch this shared board to PvP Duel (Terracotta vs. Charcoal). This is not a private room."
            )
            m_url = f"https://github.com/{repo_slug}/issues/new?title={m_title}&body={m_body}"
            mode_line = f"\n\n<sub><a href=\"{m_url}\">Open a duel</a></sub>"
        else:
            mode_line = ""
        return f"{header_row}\n{align_row}{mode_line}"


_GAME_MECHANICS: dict[str, _GameMechanics] = {
    "connect4": _Connect4Mechanics(),
}


def _resolve_game_key(game_key: str) -> str:
    norm = (game_key or DEFAULT_GAME_KEY).strip().lower()
    if norm in _GAME_MECHANICS:
        return norm
    valid = ", ".join(VALID_GAMES)
    raise ValueError(f"Unknown game_key '{game_key}'. Valid games: {valid}")


def normalize_game_state(raw_state: dict[str, Any]) -> dict[str, Any]:
    """Normalize persisted game state into the canonical Connect 4 Community Game schema."""
    if not isinstance(raw_state, dict):
        return build_midgame_showcase_state(DEFAULT_GAME_KEY, round_id=1)

    state = copy.deepcopy(raw_state)
    nested_board = state.get("board_state")
    if isinstance(nested_board, dict) and "turn" not in state:
        merged = copy.deepcopy(nested_board)
        merged["active_game"] = DEFAULT_GAME_KEY
        merged["game_key"] = DEFAULT_GAME_KEY
        state = merged

    if "move_log" not in state and isinstance(state.get("move_history"), list):
        state["move_log"] = copy.deepcopy(state["move_history"])
    if "last_mover" not in state and state.get("last_voter") is not None:
        state["last_mover"] = state["last_voter"]
    if "winning_cells" not in state and isinstance(state.get("winning_line"), list):
        state["winning_cells"] = copy.deepcopy(state["winning_line"])
    if "red_contributors" not in state and isinstance(state.get("round_participants"), dict):
        rp = state["round_participants"]
        if isinstance(rp.get("red"), dict) or isinstance(rp.get("yellow"), dict):
            state["red_contributors"] = copy.deepcopy(rp.get("red") or {})
            state["yellow_contributors"] = copy.deepcopy(rp.get("yellow") or {})
        else:
            state["red_contributors"] = copy.deepcopy(rp)

    active_key = DEFAULT_GAME_KEY
    defaults = _GAME_MECHANICS[active_key].initial_state(
        round_id=int(state.get("round_id", 1))
    )
    for k, v in defaults.items():
        state.setdefault(k, copy.deepcopy(v))

    # Old REST writes stringified tuple coordinates. Decode only that exact
    # bounded syntax on our deep copy; future JSON/Firestore projections use
    # typed list pairs supported by the existing nested-array wire codec.
    state["winning_cells"] = [list(pair) for pair in _winning_coordinates(state["winning_cells"])]

    state["active_game"] = active_key
    state["game_key"] = active_key

    board_mirror = {
        k: copy.deepcopy(v)
        for k, v in state.items()
        if k != "board_state"
    }
    state["board_state"] = board_mirror
    return state


def create_initial_game_state(
    game_key: str = DEFAULT_GAME_KEY,
    round_id: int = 1,
) -> dict[str, Any]:
    """Create a fresh initial game state for `connect4`."""
    resolved = _resolve_game_key(game_key)
    state = _GAME_MECHANICS[resolved].initial_state(round_id=int(round_id))
    return normalize_game_state(state)


build_initial_game_state = create_initial_game_state


def build_midgame_showcase_state(
    game_key: str = DEFAULT_GAME_KEY,
    round_id: int = 1,
) -> dict[str, Any]:
    """Build the curated mid-game showcase state with `@alex-chen`, `@maya-lin`, `@sam-patel`."""
    resolved = _resolve_game_key(game_key)
    state = _GAME_MECHANICS[resolved].showcase_state(round_id=int(round_id))
    return normalize_game_state(state)



def render_board_svg(state: dict[str, Any], *, elo_ledger: dict[str, Any] | None = None) -> str:
    """Render and XML-verify the editorial GitHub-native 580x410 SVG board."""
    normalized = normalize_game_state(state)
    active_key = normalized["active_game"]
    return _GAME_MECHANICS[active_key].render_svg(normalized, elo_ledger=elo_ledger)


def render_readme_game_section(
    state: dict[str, Any],
    repo_slug: str = "Siyuan-Zhu/Siyuan-Zhu",
    asset_prefix: str = "./assets",
    elo_ledger: dict[str, Any] | None = None,
    enabled: bool = True,
) -> str:
    """Render the minimalist Community Game README section with 1-click controls and collapsible move log."""
    normalized = normalize_game_state(state)
    active_key = normalized["active_game"]
    mechanics = _GAME_MECHANICS[active_key]

    from src.editorial_typography import picture
    from src.game_records import outcome_text, is_complete, history_markdown, leaderboard_markdown, display_bot_copy
    controls_md = mechanics.render_controls_md(normalized, repo_slug=repo_slug)
    next_action = "<sub>Next round: choose a column.</sub>\n\n" if is_complete(normalized) and enabled else ""
    status_text = xml_escape(outcome_text(normalized))
    if not enabled:
        status_text = "Game paused. New moves are temporarily unavailable; the recorded board is preserved."
        controls_md = ""
    last_summary = str(
        normalized.get("last_move_summary")
        or f"turn {normalized.get('turn', 1)} · ready for next move"
    )

    last_summary = display_bot_copy(last_summary)

    move_log: list[dict[str, Any]] = list(normalized.get("move_log", []))
    recent_entries = list(reversed(move_log[-5:]))
    if recent_entries:
        log_rows = [
            "| Turn | Player | Move | Bot Reply |",
            "| :---: | :--- | :--- | :--- |",
        ]
        for entry in recent_entries:
            t_num = int(entry.get("turn", 1))
            player = xml_escape(str(entry.get("player", "visitor")).lstrip("@"))
            p_move = xml_escape(
                str(entry.get("player_move") or entry.get("summary") or entry.get("move_id") or "")
            )
            b_reply = "—" if player == "House-Bot" else xml_escape(
                str(entry.get("bot_reply") or entry.get("side") or "")
            )
            player_label = "House - Bot" if player == "House-Bot" else f"`@{player}`"
            log_rows.append(f"| {t_num} | {player_label} | {display_bot_copy(p_move)} | {display_bot_copy(b_reply)} |")
        history_table = "\n".join(log_rows)
    else:
        history_table = "_No community moves recorded in this round yet._"

    cache_seed = f"{active_key}:{normalized.get('round_id', 1)}:{normalized.get('turn', 1)}:{normalized.get('status', 'ACTIVE')}"
    sha8 = hashlib.sha256(cache_seed.encode("utf-8")).hexdigest()[:8]
    cache_tag = f"r{normalized.get('round_id', 1)}t{normalized.get('turn', 1)}-{sha8}"

    return (
        '<div align="center">\n\n'
        ' <a id="connect-four"></a>\n'
        f'<h4>{picture("connect_four_title", asset_prefix, mechanics.heading_title)}</h4>\n\n'
        f'<img src="{asset_prefix}/game_board.svg?v={cache_tag}" width="580" alt="{mechanics.board_alt}">\n\n'
        "<br>\n\n"
        f"<sub>{status_text}</sub>\n\n"
        f"{next_action}{controls_md}\n\n"
        "</div>\n\n"
        f"<sub>last move: {last_summary}</sub>\n\n"
        "<details>\n"
        "<summary><sub>Recent moves</sub></summary>\n\n"
        f"{history_table}\n\n"
        "</details>\n\n---\n\n"
        f"{history_markdown(normalized)}"
        "---\n\n"
        f"{leaderboard_markdown(elo_ledger)}"

    )


def _rejected_result(
    state: dict[str, Any],
    status_code: str,
    message: str,
    repo_slug: str,
) -> MoveResult:
    normalized = normalize_game_state(state)
    return MoveResult(
        accepted=False,
        status_code=status_code,
        message=message,
        player_move_summary="",
        bot_move_summary="",
        round_over=bool(normalized.get("round_over", False)),
        winner=normalized.get("winner"),
        new_state=normalized,
        svg_board=render_board_svg(normalized),
        readme_section=render_readme_game_section(normalized, repo_slug=repo_slug),
    )


def process_move(
    state: dict[str, Any],
    request: MoveRequest,
    repo_slug: str = "Siyuan-Zhu/Siyuan-Zhu",
) -> MoveResult:
    """Validate all 5 Anti-Abuse Guardrails, apply visitor move + bot reply, and render outputs."""
    working = normalize_game_state(state)

    # Guardrail 1: Sanitized GitHub Login (`invalid_login`)
    login = (request.player_login or "").strip()
    if not GITHUB_LOGIN_RE.fullmatch(login) or login.lower().endswith("-bot"):
        return _rejected_result(
            working,
            status_code="invalid_login",
            message=f"Rejected malformed or automated GitHub login '{request.player_login}'.",
            repo_slug=repo_slug,
        )

    # Guardrail 2: Active Game Match (`wrong_game`)
    active_game = working["active_game"]
    req_game = (request.game_key or "").strip().lower()
    if req_game not in _GAME_MECHANICS or req_game != active_game:
        return _rejected_result(
            working,
            status_code="wrong_game",
            message=f"Move targets '{req_game}', but active game is '{active_game}'.",
            repo_slug=repo_slug,
        )

    # Guardrail 3: Stale-Turn Lock (`stale_turn`)
    current_turn = int(working.get("turn", 1))
    if request.expected_turn is not None and int(request.expected_turn) != current_turn:
        return _rejected_result(
            working,
            status_code="stale_turn",
            message=(
                f"Board is already on turn {current_turn} "
                f"(move link was for turn {request.expected_turn})."
            ),
            repo_slug=repo_slug,
        )

    # Guardrail 4: Same-Player Cooldown & Rolling Hourly Rate Limit (`cooldown_active` / `rate_limited`)
    now_ts = float(request.timestamp_epoch or 0.0)
    login_key = login.lower()
    rate_limits: dict[str, list[float]] = working.setdefault("rate_limits", {})
    user_history = [
        float(ts)
        for ts in rate_limits.get(login_key, [])
        if now_ts <= 0.0 or (now_ts - float(ts)) < RATE_LIMIT_WINDOW_SECONDS
    ]

    if now_ts > 0.0 and len(user_history) >= MAX_MOVES_PER_USER_PER_HOUR:
        return _rejected_result(
            working,
            status_code="rate_limited",
            message=(
                f"Hourly move cap reached for @{login} "
                f"(max {MAX_MOVES_PER_USER_PER_HOUR} moves per hour)."
            ),
            repo_slug=repo_slug,
        )

    last_mover = str(working.get("last_mover") or "").strip().lower()
    last_move_ts = float(working.get("last_move_timestamp") or 0.0)
    if last_mover and login_key == last_mover and now_ts > 0.0 and last_move_ts > 0.0:
        elapsed = now_ts - last_move_ts
        if 0.0 <= elapsed < COOLDOWN_SECONDS:
            rem = int(COOLDOWN_SECONDS - elapsed)
            return _rejected_result(
                working,
                status_code="cooldown_active",
                message=(
                    f"Same-player cooldown active for @{login} ({rem}s remaining; "
                    "another visitor can play immediately)."
                ),
                repo_slug=repo_slug,
            )

    # Guardrail 5: Strict Move Whitelist & Game Mechanics (`invalid_move`)
    mechanics = _GAME_MECHANICS[active_game]
    step = mechanics.apply_move(working, request.move_id, player_login=login)
    if not step.valid:
        return _rejected_result(
            working,
            status_code="invalid_move",
            message=step.error_message or f"Invalid move '{request.move_id}'.",
            repo_slug=repo_slug,
        )

    next_state = step.updated_state
    next_state["last_mover"] = login
    next_state["last_move_timestamp"] = now_ts
    next_state["last_move_summary"] = step.short_status_line

    if now_ts > 0.0:
        user_history.append(now_ts)
    next_state.setdefault("rate_limits", {})[login_key] = user_history

    move_log: list[dict[str, Any]] = list(next_state.get("move_log", []))
    move_log.append(
        {
            "turn": current_turn,
            "game_key": active_game,
            "round_id": int(working.get("round_id", 1)),
            "player": login,
            "move_id": step.canonical_move_id,
            "player_move": step.player_move_summary,
            "bot_reply": step.bot_move_summary,
            "timestamp_epoch": now_ts,
            "issue_number": int(request.issue_number or 0),
        }
    )
    next_state["move_log"] = move_log[-25:]

    if step.round_over:
        past_rounds: list[dict[str, Any]] = list(next_state.get("past_rounds", []))
        completed_round_id = int(working.get("round_id", 1))
        past_rounds.append(
            {
                "round_id": completed_round_id,
                "game_key": active_game,
                "winner": step.winner,
                "turns_played": current_turn,
                "winning_player": login,
            }
        )
        fresh_round = _GAME_MECHANICS[active_game].initial_state(
            round_id=completed_round_id + 1
        )
        fresh_round["past_rounds"] = past_rounds[-20:]
        fresh_round["move_log"] = next_state["move_log"]
        fresh_round["rate_limits"] = next_state["rate_limits"]
        fresh_round["last_mover"] = login
        fresh_round["last_move_timestamp"] = now_ts
        fresh_round["last_move_summary"] = (
            f"{step.short_status_line} · Round {completed_round_id} won by "
            f"{step.winner} (Round {completed_round_id + 1} started)"
        )
        next_state = fresh_round

    normalized_next = normalize_game_state(next_state)
    svg_board = render_board_svg(normalized_next)
    readme_section = render_readme_game_section(normalized_next, repo_slug=repo_slug)

    return MoveResult(
        accepted=True,
        status_code="accepted",
        message=f"{step.player_move_summary}. {step.bot_move_summary}.",
        player_move_summary=step.player_move_summary,
        bot_move_summary=step.bot_move_summary,
        round_over=step.round_over,
        winner=step.winner,
        new_state=normalized_next,
        svg_board=svg_board,
        readme_section=readme_section,
    )


def _settle_elo_if_provided(
    elo_ledger: dict[str, Any] | None,
    mode: str,
    winner: str,
    red_contributors: dict[str, int],
    yellow_contributors: dict[str, int],
    finisher_handle: str | None,
    moves_count: int,
    now_epoch: float,
    round_id: int | None = None,
) -> int:
    """Invoke `settle_round` on `elo_ledger` when provided and return finisher's ELO delta."""
    if not isinstance(elo_ledger, dict):
        return 0
    from datetime import datetime, timezone
    from src.leaderboard.elo_ledger import settle_round
    now_iso = datetime.fromtimestamp(float(now_epoch or 0.0),tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if round_id is not None and any(row.get("round_id")==round_id for row in elo_ledger.get("recent_rounds",[])):
        raise RuntimeError("Round settlement already recorded")
    before_bot = int(elo_ledger.get("bot_elo",1200))
    result = settle_round(elo_ledger,mode=mode,winner=winner,
        red_contributors=red_contributors,yellow_contributors=yellow_contributors,
        finisher_handle=finisher_handle,moves_count=moves_count,now_iso=now_iso)
    if not (isinstance(result,tuple) and len(result)==2 and isinstance(result[1],dict)):
        raise RuntimeError("Round settlement did not return a verified result")
    if round_id is not None:
        elo_ledger["recent_rounds"][0]["round_id"] = round_id
    if mode == "VS_BOT":
        elo_ledger["recent_rounds"][0]["house_delta"] = int(elo_ledger["bot_elo"]) - before_bot
    return int(result[1].get(finisher_handle,0))



def _record_move_log_entries(
    state: dict[str, Any],
    *entries: dict[str, Any],
) -> None:
    """Append move entries to `move_log` (capped at 12) and `move_history` (capped at 42)."""
    move_log = list(state.get("move_log") or [])
    move_history = list(state.get("move_history") or move_log)
    for entry in entries:
        move_log.append(entry)
        move_history.append(entry)
    state["move_log"] = move_log[-12:]
    state["move_history"] = move_history[-42:]


def apply_visitor_move(
    state: dict[str, Any],
    actor_handle: str = "",
    action: str = "",
    now_epoch: float = 0.0,
    bot_memory: dict[str, Any] | None = None,
    elo_ledger: dict[str, Any] | None = None,
    *,
    actor: str | None = None,
    expected_turn: int | None = None,
    repo_slug: str = "Siyuan-Zhu/Siyuan-Zhu",
) -> MoveResult:
    """Execute a Dual-Mode Connect 4 visitor action (`col:1`..`col:7`, `mode:new_pvp`, `mode:new_vs_bot`).

    Mutates `state` in-place (preserving integer cell convention `0, 1, 2`) and returns a `MoveResult`.
    """
    handle = (actor or actor_handle or "").strip().lstrip("@")
    if not GITHUB_LOGIN_RE.fullmatch(handle) or handle.lower().endswith("-bot"):
        return _rejected_result(
            state,
            status_code="invalid_login",
            message=f"Rejected malformed or automated GitHub login '{handle}'.",
            repo_slug=repo_slug,
        )

    ok_turn, turn_reason = check_and_heal_stale_turn(state, expected_turn)
    if not ok_turn:
        return _rejected_result(
            state,
            status_code=turn_reason,
            message=f"Stale turn lock (expected {expected_turn}, active {state.get('turn', 1)}).",
            repo_slug=repo_slug,
        )

    now_ts = float(now_epoch or 0.0)

    # Preserve the completed position before the first explicit next-round action.
    from src.game_records import archive_completed_round

    # Auto-reset completed round on first action of the next round
    if (
        str(state.get("status", "ACTIVE")).upper() == "COMPLETED"
        or bool(state.get("round_over", False))
    ):
        archive_completed_round(state)
        state["round_id"] = int(state.get("round_id", 1)) + 1
        state["status"] = "ACTIVE"
        state["round_over"] = False
        state["turn"] = 1
        state["mode"] = (
            state.get("mode")
            if state.get("mode") in ("VS_BOT", "PVP_DUEL")
            else "VS_BOT"
        )
        state["current_turn_color"] = "RED"
        state["board"] = [[0 for _ in range(_C4_COLS)] for _ in range(_C4_ROWS)]
        state["winning_cells"] = []
        state["winner"] = None
        state["finisher_handle"] = None
        state["last_elo_delta"] = 0
        state["moves_this_round"] = 0
        state["red_contributors"] = {}
        state["yellow_contributors"] = {}
        state["episode_trace"] = []
        state["move_log"] = []
        state["move_history"] = []

    # Ensure state has v2 fields and integer board cells (0, 1, 2)
    raw_board = state.get("board") or [[0 for _ in range(_C4_COLS)] for _ in range(_C4_ROWS)]
    int_board = [[_cell_to_player(cell) for cell in row] for row in raw_board]
    state["board"] = int_board
    state.setdefault("active_game", "connect4")
    state.setdefault("game_key", "connect4")
    state.setdefault("round_id", 1)
    state.setdefault("turn", 1)
    state.setdefault("mode", "VS_BOT")
    state["status"] = "ACTIVE"
    state.setdefault(
        "current_turn_color",
        "RED" if int(state["turn"]) % 2 == 1 else "YELLOW",
    )
    red_contrib: dict[str, int] = state.setdefault("red_contributors", {})
    yellow_contrib: dict[str, int] = state.setdefault("yellow_contributors", {})
    trace: list[str] = state.setdefault("episode_trace", [])
    state.setdefault("winning_cells", [])
    state.setdefault("moves_this_round", 0)
    state.setdefault("last_move_epoch", 0.0)

    act = (action or "").strip().lower()

    # 1. Mode switch: `mode:new_pvp`
    if act in ("mode:new_pvp", "new_pvp"):
        if int(state.get("turn", 1)) > 2:
            return _rejected_result(
                state,
                status_code="invalid_mode_switch",
                message="A duel can only start on turns 1 or 2. Continue this round against House - Bot, or start a duel early in the next round.",
                repo_slug=repo_slug,
            )
        state["mode"] = "PVP_DUEL"
        state["current_turn_color"] = (
            "RED" if int(state["turn"]) % 2 == 1 else "YELLOW"
        )
        state["last_move_epoch"] = now_ts
        state["last_move_timestamp"] = now_ts
        state["last_mover"] = handle
        state["last_move_summary"] = (
            f"@{handle} switched Round #{state['round_id']} to PvP Duel mode."
        )
        _record_move_log_entries(
            state,
            {
                "turn": int(state.get("turn", 1)),
                "player": f"@{handle}",
                "side": "MODE",
                "col": None,
                "summary": state["last_move_summary"],
                "player_move": "Switched to PVP_DUEL",
                "bot_reply": "Awaiting a Charcoal player",
            },
        )
        svg_board = render_board_svg(state)
        readme_sec = render_readme_game_section(state, repo_slug=repo_slug)
        return MoveResult(
            accepted=True,
            status_code="accepted",
            message=f"@{handle} switched Round #{state['round_id']} to PVP_DUEL mode.",
            player_move_summary="Switched to PVP_DUEL",
            bot_move_summary="Awaiting a Charcoal player",
            round_over=False,
            winner=None,
            new_state=state,
            svg_board=svg_board,
            readme_section=readme_sec,
        )

    # 2. Mode switch / stall takeover: `mode:new_vs_bot`
    if act in ("mode:new_vs_bot", "new_vs_bot"):
        cur_mode = str(state.get("mode", "VS_BOT")).upper()
        cur_turn = int(state.get("turn", 1))
        if cur_mode == "PVP_DUEL" and cur_turn > 1:
            last_epoch = float(
                state.get("last_move_epoch")
                or state.get("last_move_timestamp")
                or 0.0
            )
            elapsed = now_ts - last_epoch
            if elapsed < PVP_STALL_TIMEOUT_SECONDS:
                return _rejected_result(
                    state,
                    status_code="stall_timeout_not_reached",
                    message=(
                        f"PvP Duel stall takeover requires {PVP_STALL_TIMEOUT_SECONDS}s "
                        f"inactivity ({int(max(0.0, PVP_STALL_TIMEOUT_SECONDS - elapsed))}s remaining)."
                    ),
                    repo_slug=repo_slug,
                )

        state["mode"] = "VS_BOT"
        bot_summary = "Switched to VS_BOT"
        takeover_col: int | None = None
        # If it is currently Yellow's turn, House Bot immediately takes over Yellow's move
        if int(state.get("turn", 1)) % 2 == 0 or state.get("current_turn_color") == "YELLOW":
            bot_col = choose_bot_move(
                int_board, bot_memory=bot_memory, depth=_C4_DEPTH, player=2
            )
            b_row = _c4_available_row(int_board, bot_col)
            if b_row is not None:
                int_board[b_row][bot_col] = 2
                takeover_col = bot_col + 1
                state["last_bot_move"] = f"col{bot_col + 1}"
                state["moves_this_round"] = int(state.get("moves_this_round", 0)) + 1
                trace.append(canonical_board_key(int_board))
                state["turn"] = int(state["turn"]) + 1
                state["current_turn_color"] = "RED"
                bot_summary = f"House took over Charcoal and dropped in col {bot_col + 1}"

                _, b2 = encode_bitboard(int_board)
                if has_won_bitboard(b2):
                    state["status"] = "COMPLETED"
                    state["round_over"] = True
                    state["winner"] = "bot"
                    state["finisher_handle"] = "House-Bot"
                    state["winning_cells"] = find_winning_cells(int_board, player=2)
                    if bot_memory is not None:
                        update_td_lambda_memory(
                            bot_memory,
                            trace,
                            outcome_for_bot=1.0,
                            round_id=int(state["round_id"]),
                        )
                    state["last_elo_delta"] = _settle_elo_if_provided(
                        elo_ledger,
                        mode="VS_BOT",
                        winner="bot",
                        red_contributors=red_contrib,
                        yellow_contributors=yellow_contrib,
                        finisher_handle="House-Bot",
                        moves_count=int(state["moves_this_round"]),
                        now_epoch=now_ts,
                        round_id=int(state["round_id"]),
                    )
                elif _c4_is_full(int_board):
                    state.update(status="COMPLETED",round_over=True,winner="draw",finisher_handle=None)
                    if bot_memory is not None:
                        update_td_lambda_memory(bot_memory,trace,outcome_for_bot=0.0,round_id=int(state["round_id"]))
                    _settle_elo_if_provided(elo_ledger,mode="VS_BOT",winner="draw",
                        red_contributors=red_contrib,yellow_contributors=yellow_contrib,
                        finisher_handle=None,moves_count=int(state["moves_this_round"]),
                        now_epoch=now_ts,round_id=int(state["round_id"]))

        state["last_move_epoch"] = now_ts
        state["last_move_timestamp"] = now_ts
        state["last_mover"] = handle
        state["last_move_summary"] = bot_summary
        _record_move_log_entries(
            state,
            {
                "turn": cur_turn,
                "player": f"@{handle}",
                "side": "MODE",
                "col": takeover_col,
                "summary": bot_summary,
                "player_move": "Triggered VS_BOT takeover",
                "bot_reply": bot_summary,
            },
        )
        svg_board = render_board_svg(state)
        readme_sec = render_readme_game_section(state, repo_slug=repo_slug)
        return MoveResult(
            accepted=True,
            status_code="accepted",
            message=display_bot_copy(bot_summary) + ' ' + outcome_text(state),
            player_move_summary="Triggered VS_BOT takeover",
            bot_move_summary=display_bot_copy(bot_summary),
            round_over=bool(state.get("round_over", False)),
            winner=state.get("winner"),
            new_state=state,
            svg_board=svg_board,
            readme_section=readme_sec,
        )

    # 3. Column Drop (`col:1`..`col:7` or `col1`..`col7` or `drop 1`..`drop 7`)
    col_token = act
    if col_token.startswith("drop "):
        col_token = f"col{col_token.split('drop ', 1)[1].strip()}"
    elif col_token.startswith("col:"):
        col_token = f"col{col_token[4:].strip()}"

    if not re.fullmatch(r"col[1-7]", col_token):
        return _rejected_result(
            state,
            status_code="invalid_move",
            message=f"Invalid Connect 4 action '{action}'.",
            repo_slug=repo_slug,
        )

    col_idx = int(col_token[3:]) - 1
    p_row = _c4_available_row(int_board, col_idx)
    if p_row is None:
        return _rejected_result(
            state,
            status_code="column_full",
            message=f"Column {col_idx + 1} is already full.",
            repo_slug=repo_slug,
        )

    cur_mode = str(state.get("mode", "VS_BOT")).upper()
    cur_turn = int(state.get("turn", 1))

    if cur_mode == "PVP_DUEL":
        active_player = 1 if (cur_turn % 2 == 1) else 2
        if active_player == 1 and handle in yellow_contrib:
            return _rejected_result(
                state,
                status_code="team_affinity_violation",
                message=f"@{handle} already played Charcoal in Round #{state['round_id']}.",
                repo_slug=repo_slug,
            )
        if active_player == 2 and handle in red_contrib:
            return _rejected_result(
                state,
                status_code="team_affinity_violation",
                message=f"@{handle} already played Terracotta in Round #{state['round_id']}.",
                repo_slug=repo_slug,
            )

        int_board[p_row][col_idx] = active_player
        if active_player == 1:
            red_contrib[handle] = int(red_contrib.get(handle, 0)) + 1
            state["last_challenger_move"] = f"col{col_idx + 1}"
            state["current_turn_color"] = "YELLOW"
        else:
            yellow_contrib[handle] = int(yellow_contrib.get(handle, 0)) + 1
            state["last_bot_move"] = f"col{col_idx + 1}"
            state["current_turn_color"] = "RED"

        trace.append(canonical_board_key(int_board))
        state["moves_this_round"] = int(state.get("moves_this_round", 0)) + 1
        state["turn"] = cur_turn + 1
        state["last_mover"] = handle
        state["last_move_epoch"] = now_ts
        state["last_move_timestamp"] = now_ts

        b1, b2 = encode_bitboard(int_board)
        b_active = b1 if active_player == 1 else b2
        if has_won_bitboard(b_active):
            win_side = "red" if active_player == 1 else "yellow"
            state["status"] = "COMPLETED"
            state["round_over"] = True
            state["winner"] = win_side
            state["finisher_handle"] = handle
            state["winning_cells"] = find_winning_cells(int_board, player=active_player)
            if bot_memory is not None:
                update_td_lambda_memory(
                    bot_memory,
                    trace,
                    outcome_for_bot=(-0.5 if active_player == 1 else 0.5),
                    round_id=int(state["round_id"]),
                )
            state["last_elo_delta"] = _settle_elo_if_provided(
                elo_ledger,
                mode="PVP_DUEL",
                winner=win_side,
                red_contributors=red_contrib,
                yellow_contributors=yellow_contrib,
                finisher_handle=handle,
                moves_count=int(state["moves_this_round"]),
                now_epoch=now_ts,
                        round_id=int(state["round_id"]),
            )
        elif _c4_is_full(int_board):
            state["status"] = "COMPLETED"
            state["round_over"] = True
            state["winner"] = "draw"
            if bot_memory is not None:
                update_td_lambda_memory(
                    bot_memory,
                    trace,
                    outcome_for_bot=0.0,
                    round_id=int(state["round_id"]),
                )
            _settle_elo_if_provided(
                elo_ledger,
                mode="PVP_DUEL",
                winner="draw",
                red_contributors=red_contrib,
                yellow_contributors=yellow_contrib,
                finisher_handle=None,
                moves_count=int(state["moves_this_round"]),
                now_epoch=now_ts,
                        round_id=int(state["round_id"]),
            )

        p_summary = f"Dropped {'Terracotta' if active_player == 1 else 'Charcoal'} in col {col_idx + 1}"
        b_summary = (
            f"PvP Duel — awaiting {'Charcoal' if active_player == 1 else 'Terracotta'}"
            if not state["round_over"]
            else outcome_text(state)
        )
        state["last_move_summary"] = (
            f"@{handle} ({'Terracotta' if active_player == 1 else 'Charcoal'}) dropped in col {col_idx + 1} · {b_summary}"
        )
        _record_move_log_entries(
            state,
            {
                "turn": cur_turn,
                "player": f"@{handle}",
                "side": "RED" if active_player == 1 else "YELLOW",
                "col": col_idx + 1,
                "summary": p_summary,
                "move_id": f"col{col_idx + 1}",
                "player_move": p_summary,
                "bot_reply": b_summary,
            },
        )
        svg_board = render_board_svg(state)
        readme_sec = render_readme_game_section(state, repo_slug=repo_slug)
        return MoveResult(
            accepted=True,
            status_code="accepted",
            message=f"{p_summary}. {b_summary}. " + outcome_text(state),
            player_move_summary=p_summary,
            bot_move_summary=b_summary,
            round_over=bool(state["round_over"]),
            winner=state.get("winner"),
            new_state=state,
            svg_board=svg_board,
            readme_section=readme_sec,
        )

    # Default Mode: VS_BOT (Human plays Red = 1, House Bot replies Yellow = 2)
    int_board[p_row][col_idx] = 1
    red_contrib[handle] = int(red_contrib.get(handle, 0)) + 1
    state["last_challenger_move"] = f"col{col_idx + 1}"
    state["moves_this_round"] = int(state.get("moves_this_round", 0)) + 1
    trace.append(canonical_board_key(int_board))
    state["last_mover"] = handle
    state["last_move_epoch"] = now_ts
    state["last_move_timestamp"] = now_ts

    p_summary = f"Dropped in col {col_idx + 1}"
    b_summary = "Bot stood down (round complete)"
    bot_replied_col: int | None = None

    b1, _ = encode_bitboard(int_board)
    if has_won_bitboard(b1):
        state["status"] = "COMPLETED"
        state["round_over"] = True
        state["winner"] = "red"
        state["finisher_handle"] = handle
        state["winning_cells"] = find_winning_cells(int_board, player=1)
        state["turn"] = cur_turn + 1
        if bot_memory is not None:
            update_td_lambda_memory(
                bot_memory,
                trace,
                outcome_for_bot=-1.0,
                round_id=int(state["round_id"]),
            )
        state["last_elo_delta"] = _settle_elo_if_provided(
            elo_ledger,
            mode="VS_BOT",
            winner="red",
            red_contributors=red_contrib,
            yellow_contributors=yellow_contrib,
            finisher_handle=handle,
            moves_count=int(state["moves_this_round"]),
            now_epoch=now_ts,
                        round_id=int(state["round_id"]),
        )
    elif _c4_is_full(int_board):
        state["status"] = "COMPLETED"
        state["round_over"] = True
        state["winner"] = "draw"
        state["turn"] = cur_turn + 1
        if bot_memory is not None:
            update_td_lambda_memory(
                bot_memory,
                trace,
                outcome_for_bot=0.0,
                round_id=int(state["round_id"]),
            )
        _settle_elo_if_provided(
            elo_ledger,
            mode="VS_BOT",
            winner="draw",
            red_contributors=red_contrib,
            yellow_contributors=yellow_contrib,
            finisher_handle=None,
            moves_count=int(state["moves_this_round"]),
            now_epoch=now_ts,
                        round_id=int(state["round_id"]),
        )
    else:
        bot_col = choose_bot_move(
            int_board, bot_memory=bot_memory, depth=_C4_DEPTH, player=2
        )
        b_row = _c4_available_row(int_board, bot_col)
        if b_row is not None:
            int_board[b_row][bot_col] = 2
            bot_replied_col = bot_col
            state["last_bot_move"] = f"col{bot_col + 1}"
            state["moves_this_round"] = int(state.get("moves_this_round", 0)) + 1
            trace.append(canonical_board_key(int_board))
            b_summary = f"Answered col {bot_col + 1}"

        state["turn"] = cur_turn + 1
        state["current_turn_color"] = "RED"

        _, b2 = encode_bitboard(int_board)
        if has_won_bitboard(b2):
            state["status"] = "COMPLETED"
            state["round_over"] = True
            state["winner"] = "bot"
            state["finisher_handle"] = "House-Bot"
            state["winning_cells"] = find_winning_cells(int_board, player=2)
            if bot_memory is not None:
                update_td_lambda_memory(
                    bot_memory,
                    trace,
                    outcome_for_bot=1.0,
                    round_id=int(state["round_id"]),
                )
            state["last_elo_delta"] = _settle_elo_if_provided(
                elo_ledger,
                mode="VS_BOT",
                winner="bot",
                red_contributors=red_contrib,
                yellow_contributors=yellow_contrib,
                finisher_handle="House-Bot",
                moves_count=int(state["moves_this_round"]),
                now_epoch=now_ts,
                        round_id=int(state["round_id"]),
            )
        elif _c4_is_full(int_board):
            state["status"] = "COMPLETED"
            state["round_over"] = True
            state["winner"] = "draw"
            if bot_memory is not None:
                update_td_lambda_memory(
                    bot_memory,
                    trace,
                    outcome_for_bot=0.0,
                    round_id=int(state["round_id"]),
                )
            _settle_elo_if_provided(
                elo_ledger,
                mode="VS_BOT",
                winner="draw",
                red_contributors=red_contrib,
                yellow_contributors=yellow_contrib,
                finisher_handle=None,
                moves_count=int(state["moves_this_round"]),
                now_epoch=now_ts,
                        round_id=int(state["round_id"]),
            )

    state["last_move_summary"] = (
        f"@{handle} dropped in col {col_idx + 1} ({b_summary.lower()}) · turn {state['turn']}"
    )
    vs_bot_entries: list[dict[str, Any]] = [
        {
            "turn": cur_turn,
            "player": f"@{handle}",
            "side": "RED",
            "col": col_idx + 1,
            "summary": p_summary,
            "move_id": f"col{col_idx + 1}",
            "player_move": p_summary,
            "bot_reply": b_summary,
        }
    ]
    if bot_replied_col is not None:
        vs_bot_entries.append(
            {
                "turn": cur_turn,
                "player": "House-Bot",
                "side": "YELLOW",
                "col": bot_replied_col + 1,
                "summary": b_summary,
                "move_id": f"col{bot_replied_col + 1}",
                "player_move": b_summary,
                "bot_reply": "",
            }
        )
    _record_move_log_entries(state, *vs_bot_entries)

    svg_board = render_board_svg(state)
    readme_sec = render_readme_game_section(state, repo_slug=repo_slug)
    return MoveResult(
        accepted=True,
        status_code="accepted",
        message=f"{p_summary}. {b_summary}. " + outcome_text(state),
        player_move_summary=p_summary,
        bot_move_summary=b_summary,
        round_over=bool(state.get("round_over", False)),
        winner=state.get("winner"),
        new_state=state,
        svg_board=svg_board,
        readme_section=readme_sec,
    )


def parse_issue_round_lock(issue_title: str) -> int | None:
    """Read a canonical round suffix; old unlocked manual commands remain valid."""
    raw = (issue_title or '').strip()
    parts = [part.strip() for part in raw.split(':')]
    locks = [part for part in parts if part.lower().startswith('round=')]
    if not locks:
        return None
    if (len(locks) != 1 or parts[-1] != locks[0]
            or not re.fullmatch(r'round=[1-9][0-9]*', locks[0], re.IGNORECASE)
            or not raw.lower().startswith(('game:move:', 'game:mode:'))):
        raise ValueError('Invalid canonical round lock')
    return int(locks[0].split('=', 1)[1])


def check_issue_round_lock(state: dict[str, Any], issue_title: str) -> tuple[bool, str]:
    """Bind newly generated buttons to the visible board, including completed rounds."""
    try:
        expected = parse_issue_round_lock(issue_title)
    except ValueError:
        return False, 'invalid_round_lock'
    if expected is not None and expected != int(state.get('round_id', 1)):
        return False, 'stale_round'
    return True, 'ok'


def parse_move_issue_title(
    issue_title: str,
    active_game_key: str = DEFAULT_GAME_KEY,
) -> tuple[str, str, int | None] | None:
    """Parse canonical (`game:move:<game>:<move>:turn=<N>`) or legacy issue titles."""
    raw = (issue_title or "").strip()
    if len(raw) > 256:
        return None
    try:
        round_lock = parse_issue_round_lock(raw)
    except ValueError:
        return None
    if round_lock is not None:
        raw = raw.rsplit(':', 1)[0]
    lower = raw.lower()

    if lower.startswith("connect4: drop "):
        col_part = lower.split("connect4: drop ", 1)[1].strip()
        return ("connect4", f"col{col_part}", None)

    if lower.startswith("game:mode:"):
        rest_mode = raw[len("game:mode:") :].strip()
        parts_mode = [p.strip() for p in rest_mode.split(":") if p.strip()]
        expected_turn_mode: int | None = None
        if parts_mode and parts_mode[-1].lower().startswith("turn="):
            turn_str = parts_mode.pop().split("=", 1)[1].strip()
            if turn_str.lstrip("-").isdigit():
                expected_turn_mode = int(turn_str)
        game_cand = parts_mode[0].lower() if len(parts_mode) >= 2 else active_game_key
        raw_mode = parts_mode[-1].lower() if parts_mode else "pvp"
        if raw_mode in ("pvp", "new_pvp", "pvp_duel"):
            return (game_cand, "mode:new_pvp", expected_turn_mode)
        if raw_mode in ("vs_bot", "new_vs_bot", "bot"):
            return (game_cand, "mode:new_vs_bot", expected_turn_mode)
        return (game_cand, f"mode:{raw_mode}", expected_turn_mode)

    if not lower.startswith("game:move:"):
        return None

    rest = raw[len("game:move:") :].strip()
    if not rest:
        return (active_game_key, "", None)

    parts = [p.strip() for p in rest.split(":")]
    expected_turn: int | None = None
    if parts and parts[-1].lower().startswith("turn="):
        turn_str = parts.pop().split("=", 1)[1].strip()
        if turn_str.lstrip("-").isdigit():
            expected_turn = int(turn_str)

    # Legacy 2-segment format: `game:move:<move_id>:<stake_tier>`
    if len(parts) == 2 and parts[1].lower() in ("standard", "conviction"):
        return (active_game_key, parts[0], expected_turn)

    if len(parts) == 1:
        return (active_game_key, parts[0], expected_turn)

    game_candidate = parts[0].lower()
    if game_candidate in ("mode", "col"):
        return (active_game_key, ":".join(parts), expected_turn)
    move_id = ":".join(parts[1:])
    return (game_candidate, move_id, expected_turn)
