"""Identity-Staked Reputation (ELO) Leaderboard & Hall of Records Engine."""

from __future__ import annotations

from src.leaderboard.elo_ledger import (
    DEFAULT_LEDGER_PATH,
    INITIAL_ELO,
    STAKE_TIERS,
    build_seeded_ledger,
    compute_reputation_rank,
    create_empty_ledger,
    ensure_player,
    get_active_stakes,
    get_reputation_rank,
    get_sorted_leaderboard,
    load_ledger,
    record_move,
    record_staked_move,
    resolve_stake_tier,
    save_ledger,
    settle_round,
)
from src.leaderboard.identity_verifier import PlayerIdentity, verify_github_player
from src.leaderboard.render_markdown import (
    render_game_section_md,
    render_hall_of_records_md,
    render_move_table_md,
)

__all__ = [
    "DEFAULT_LEDGER_PATH",
    "INITIAL_ELO",
    "PlayerIdentity",
    "STAKE_TIERS",
    "build_seeded_ledger",
    "compute_reputation_rank",
    "create_empty_ledger",
    "ensure_player",
    "get_active_stakes",
    "get_reputation_rank",
    "get_sorted_leaderboard",
    "load_ledger",
    "record_move",
    "record_staked_move",
    "render_game_section_md",
    "render_hall_of_records_md",
    "render_move_table_md",
    "resolve_stake_tier",
    "save_ledger",
    "settle_round",
    "verify_github_player",
]
