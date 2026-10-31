"""Markdown renderer for the Identity-Staked README Game & Hall of Records."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import urllib.parse

from src.leaderboard.elo_ledger import (
    INITIAL_ELO,
    STAKE_TIERS,
    compute_reputation_rank,
    get_active_stakes,
    get_sorted_leaderboard,
    load_ledger,
)

DEFAULT_REPO = "Siyuan-Zhu/Siyuan-Zhu"

DEFAULT_CONNECT4_MOVES: list[dict[str, str]] = [
    {"id": str(col), "label": f"Column {col}", "note": "Center file" if col == 4 else f"Drop disc in Col {col}"}
    for col in range(1, 8)
]


def _format_player_badge(github_login: str, avatar_url: str | None = None) -> str:
    """Format a GitHub player link with a 16px inline avatar."""
    clean_login = str(github_login).lstrip("@").strip() or "visitor"
    resolved_avatar = avatar_url or f"https://github.com/{clean_login}.png?size=32"
    return (
        f'<img src="{resolved_avatar}" width="16" height="16" alt="@{clean_login}" /> '
        f"[@{clean_login}](https://github.com/{clean_login})"
    )


def _normalize_moves(
    raw_moves: Sequence[Mapping[str, Any] | tuple[Any, ...] | str | int] | None,
) -> list[dict[str, str]]:
    if raw_moves is None:
        return list(DEFAULT_CONNECT4_MOVES)

    normalized: list[dict[str, str]] = []
    for item in raw_moves:
        if isinstance(item, Mapping):
            move_id = str(
                item.get("id")
                or item.get("move_id")
                or item.get("col")
                or item.get("key")
                or item.get("code")
                or ""
            ).strip()
            label = str(
                item.get("label")
                or item.get("move_label")
                or item.get("name")
                or item.get("title")
                or f"Move {move_id}"
            ).strip()
            note = str(
                item.get("note")
                or item.get("description")
                or item.get("hint")
                or ""
            ).strip()
            if move_id:
                normalized.append({"id": move_id, "label": label, "note": note})
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            move_id = str(item[0]).strip()
            label = str(item[1]).strip()
            note = str(item[2]).strip() if len(item) >= 3 else ""
            if move_id:
                normalized.append({"id": move_id, "label": label, "note": note})
        else:
            move_id = str(item).strip()
            if move_id:
                label = f"Column {move_id}" if move_id.isdigit() else f"Move {move_id}"
                normalized.append({"id": move_id, "label": label, "note": ""})

    return normalized


def _build_issue_url(repo: str, move_id: str, stake_tier: str, move_label: str) -> str:
    elo_val = STAKE_TIERS.get(stake_tier, 25)
    body_text = (
        f"Submitting Identity-Staked Move '{move_label}' ({stake_tier.upper()}: {elo_val} ELO at risk). "
        "Click 'Submit new issue' below to authenticate your GitHub handle and execute the turn."
    )
    encoded_body = urllib.parse.quote_plus(body_text)
    return (
        f"https://github.com/{repo}/issues/new"
        f"?title=game:move:{move_id}:{stake_tier}&body={encoded_body}"
    )


def render_move_table_md(
    valid_moves: Sequence[Mapping[str, Any] | tuple[Any, ...] | str | int] | None = None,
    *,
    repo: str = DEFAULT_REPO,
) -> str:
    """Render the 1-Click Move Table with Standard (25 ELO) and Conviction (50 ELO) links."""
    moves = _normalize_moves(valid_moves)
    if not moves:
        return "_Round completed — awaiting board reset._"

    lines = [
        "| Valid Move | Standard Stake (`25 ELO`) | High-Conviction Stake (`50 ELO`) |",
        "| :--- | :--- | :--- |",
    ]

    for mv in moves:
        move_id = mv["id"]
        label = mv["label"]
        note = f" — _{mv['note']}_" if mv.get("note") else ""
        std_url = _build_issue_url(repo, move_id, "standard", label)
        conv_url = _build_issue_url(repo, move_id, "conviction", label)
        lines.append(
            f"| **{label}**{note} "
            f"| [Play Move (Stake 25 ELO)]({std_url}) "
            f"| [🔥 Double Down (Stake 50 ELO)]({conv_url}) |"
        )

    return "\n".join(lines)


def render_hall_of_records_md(ledger: Mapping[str, Any]) -> str:
    """Render the collapsible Hall of Records with Active Stakes, Move Log, and ELO Leaderboard."""
    active_stakes = get_active_stakes(ledger)
    round_log = list(ledger.get("current_round_log", []))
    leaderboard = get_sorted_leaderboard(ledger)
    round_id = int(ledger.get("round_id", 1))

    lines: list[str] = [
        "<details><summary><b>📜 Hall of Records — Active Stakes, Move Log & All-Time ELO Leaderboard</b></summary>",
        "<br>",
        "",
        f"#### 🎯 Active Stakes in Current Round (Round #{round_id})",
        "",
    ]

    if active_stakes:
        lines.extend([
            "| Player | Moves in Round | Latest Tier | ELO Staked (At Risk) | Current ELO | Projected Win / Loss |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for item in active_stakes:
            login = str(item["github_login"])
            badge = _format_player_badge(login, item.get("avatar_url"))
            moves_cnt = int(item.get("moves_in_round", 1))
            tier = str(item.get("last_stake_tier", "standard"))
            tier_label = "🔥 Conviction" if tier == "conviction" else "Standard"
            staked = int(item.get("elo_staked", 25))
            cur_elo = int(item.get("current_elo", INITIAL_ELO))
            win_elo = int(item.get("potential_win_elo", cur_elo + staked))
            loss_elo = int(item.get("potential_loss_elo", cur_elo - staked))
            lines.append(
                f"| {badge} | `{moves_cnt}` | {tier_label} | **`{staked} ELO`** | `{cur_elo}` | `{win_elo}` / `{loss_elo}` |"
            )
    else:
        lines.append("_No active stakes in the current round yet — make the opening move above to stake your ELO!_")

    lines.extend([
        "",
        "#### 🕒 Recent Move Log",
        "",
    ])

    # Combine current_round_log or fallback to most recent past_rounds moves if current round is empty
    display_log = list(round_log)
    if not display_log:
        for past in reversed(list(ledger.get("past_rounds", []))):
            if isinstance(past, Mapping) and past.get("moves"):
                display_log = list(past["moves"])
                break

    if display_log:
        lines.extend([
            "| Turn | Timestamp (UTC) | Player | Move | Stake | Bot Response |",
            "| :---: | :--- | :--- | :--- | :---: | :--- |",
        ])
        for entry in reversed(display_log[-10:]):
            turn_num = int(entry.get("turn_number", entry.get("turn", 1)))
            ts = str(entry.get("timestamp_utc", entry.get("timestamp", "")))
            login = str(entry.get("github_login", "visitor"))
            badge = _format_player_badge(login, entry.get("avatar_url"))
            move_label = str(entry.get("move_label", ""))
            tier = str(entry.get("stake_tier", "standard"))
            staked = int(entry.get("elo_staked", STAKE_TIERS.get(tier, 25)))
            stake_cell = f"🔥 `{staked} ELO` (Conviction)" if tier == "conviction" else f"`{staked} ELO` (Standard)"
            bot_reply = str(entry.get("bot_reply", "Awaiting counter-move"))
            lines.append(
                f"| `#{turn_num}` | `{ts}` | {badge} | **{move_label}** | {stake_cell} | {bot_reply} |"
            )
    else:
        lines.append("_No moves recorded yet._")

    lines.extend([
        "",
        "#### 🏆 All-Time Verified GitHub Leaderboard",
        "",
        "| Rank | Player | ELO Rating | Title | Moves | Win-Loss Record |",
        "| :---: | :--- | :---: | :--- | :---: | :---: |",
    ])

    if leaderboard:
        for row in leaderboard:
            rank_num = int(row["rank"])
            login = str(row["github_login"])
            badge = _format_player_badge(login, row.get("avatar_url"))
            elo = int(row["elo"])
            title = compute_reputation_rank(elo)
            moves_cnt = int(row.get("moves_played", 0))
            wins = int(row.get("wins", 0))
            losses = int(row.get("losses", 0))
            draws = int(row.get("draws", 0))
            wl_str = f"`{wins}W - {losses}L`" if draws == 0 else f"`{wins}W - {losses}L - {draws}D`"
            lines.append(
                f"| **`#{rank_num}`** | {badge} | **`{elo}`** | _{title}_ | `{moves_cnt}` | {wl_str} |"
            )
    else:
        lines.append("| `#1` | _Awaiting first challenger_ | `1000` | _Verified Challenger_ | `0` | `0W - 0L` |")

    lines.extend([
        "",
        "</details>",
    ])
    return "\n".join(lines)


def render_game_section_md(
    ledger: Mapping[str, Any] | Sequence[Any] | None = None,
    valid_moves: Sequence[Mapping[str, Any] | tuple[Any, ...] | str | int] | Mapping[str, Any] | None = None,
    *,
    moves: Sequence[Mapping[str, Any] | tuple[Any, ...] | str | int] | None = None,
    repo: str = DEFAULT_REPO,
    game_title: str = "README Strategy Arena — Visitors vs. Minimax Bot",
    board_svg_path: str | None = "./assets/connect4_board.svg",
    status_text: str | None = None,
) -> str:
    """Render the complete minimalist README game section with 1-click stakes and Hall of Records."""
    # Support flexible positional arguments if caller passes (valid_moves, ledger)
    resolved_ledger: Mapping[str, Any] | None = None
    resolved_moves: Sequence[Any] | None = moves

    if isinstance(ledger, (list, tuple)):
        resolved_moves = ledger
        if isinstance(valid_moves, Mapping):
            resolved_ledger = valid_moves
    elif isinstance(ledger, Mapping):
        if "players" in ledger or "current_round_log" in ledger or "past_rounds" in ledger:
            resolved_ledger = ledger
            if isinstance(valid_moves, (list, tuple)):
                resolved_moves = valid_moves
        else:
            # Game state dict passed as first argument
            if resolved_moves is None and isinstance(ledger.get("valid_moves"), (list, tuple)):
                resolved_moves = ledger["valid_moves"]
            if isinstance(valid_moves, Mapping):
                resolved_ledger = valid_moves
    elif isinstance(valid_moves, (list, tuple)) and resolved_moves is None:
        resolved_moves = valid_moves

    active_ledger = resolved_ledger if resolved_ledger is not None else load_ledger()

    sections: list[str] = [
        f"### 🎮 {game_title}",
        "",
        "> *Identity-Staked Play: Submitting a move authenticates your GitHub profile (`@username` + avatar) and stakes your ELO on the Hall of Records.*",
        "",
    ]

    if board_svg_path:
        sections.extend([
            '<div align="center">',
            f'  <img src="{board_svg_path}" width="100%" alt="{game_title}" />',
            "</div>",
            "",
        ])

    if status_text:
        sections.extend([
            f"**Board Status:** {status_text}",
            "",
        ])

    sections.extend([
        render_move_table_md(resolved_moves, repo=repo),
        "",
        render_hall_of_records_md(active_ledger),
    ])

    return "\n".join(sections)
