"""Reputation Stake & ELO Ledger for Identity-Staked README Games."""

from __future__ import annotations

import copy
import datetime as dt
import json
import pathlib
from typing import Any, Mapping

from src.leaderboard.identity_verifier import PlayerIdentity, verify_github_player


INITIAL_ELO: int = 800
BOT_INITIAL_ELO: int = 1200
DEFAULT_BOT_ELO: int = BOT_INITIAL_ELO
K_FACTOR: int = 32
FINISHER_BONUS: int = 8
MIN_ELO_FLOOR: int = 100

STAKE_TIERS: dict[str, int] = {
    "standard": 25,
    "conviction": 50,
}

DEFAULT_LEDGER_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "data" / "hall_of_records.json"
)

_VISITOR_WIN_LABELS = frozenset({
    "visitor",
    "visitors",
    "player",
    "players",
    "challenger",
    "challengers",
    "human",
    "community",
    "win",
    "1",
})

_BOT_WIN_LABELS = frozenset({
    "bot",
    "ai",
    "house",
    "engine",
    "minimax",
    "opponent",
    "loss",
    "2",
})

_DRAW_LABELS = frozenset({
    "draw",
    "tie",
    "stalemate",
    "push",
    "none",
    "0",
})


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_reputation_rank(elo: int) -> str:
    """Compute dynamic Reputation Rank from current ELO rating."""
    if elo >= 1100:
        return "Grandmaster Strategist"
    if elo >= 1030:
        return "Principal Tactician"
    if elo >= 980:
        return "Verified Challenger"
    return "Underdog Contender (Staked & Exposed)"


get_reputation_rank = compute_reputation_rank


def resolve_stake_tier(stake_tier: str | None) -> tuple[str, int]:
    """Normalize stake tier name and return `(canonical_tier, elo_staked)`."""
    raw = (stake_tier or "standard").strip().lower()
    if raw in ("conviction", "double", "doubledown", "double_down", "high", "50"):
        return "conviction", STAKE_TIERS["conviction"]
    if raw in ("standard", "default", "normal", "25"):
        return "standard", STAKE_TIERS["standard"]
    raise ValueError(f"Unsupported stake_tier '{stake_tier}'. Expected 'standard' or 'conviction'.")


def create_empty_ledger(game_key: str = "connect4") -> dict[str, Any]:
    """Return an initialized empty Hall of Records ELO ledger dictionary."""
    return {
        "schema_version": "1.0",
        "round_id": 1,
        "active_game_key": game_key,
        "active_stakes": {},
        "current_round_log": [],
        "players": {},
        "past_rounds": [],
        "bot_elo": BOT_INITIAL_ELO,
        "recent_rounds": [],
    }


def _coerce_identity(
    player: PlayerIdentity | str | None,
    github_login: str | None = None,
    fetch_online: bool = False,
) -> PlayerIdentity:
    if isinstance(player, PlayerIdentity):
        return player
    login_str = str(player if player is not None else (github_login or "")).strip()
    return verify_github_player(login_str, fetch_online=fetch_online)


def ensure_player(
    ledger: dict[str, Any],
    player: PlayerIdentity | str,
    *,
    timestamp_utc: str | None = None,
    fetch_online: bool = False,
) -> dict[str, Any]:
    """Ensure a verified GitHub player is registered in `ledger['players']` at INITIAL_ELO."""
    identity = _coerce_identity(player, fetch_online=fetch_online)
    if not identity.verified:
        raise ValueError(
            f"Cannot register unverified GitHub player '{identity.github_login}': "
            f"{identity.rejection_reason}"
        )

    players = ledger.setdefault("players", {})
    now_iso = timestamp_utc or _utc_now_iso()
    login = identity.github_login

    if login not in players:
        rank_title = compute_reputation_rank(INITIAL_ELO)
        players[login] = {
            "github_login": login,
            "avatar_url": identity.avatar_url,
            "elo": INITIAL_ELO,
            "peak_elo": INITIAL_ELO,
            "rank_title": rank_title,
            "title": rank_title,
            "moves_played": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "total_elo_staked": 0,
            "joined_utc": now_iso,
            "last_active_utc": now_iso,
        }
    else:
        record = players[login]
        record["avatar_url"] = identity.avatar_url
        record["rank_title"] = compute_reputation_rank(int(record.get("elo", INITIAL_ELO)))
        record["title"] = record["rank_title"]

    return players[login]


def _rebuild_active_stakes(ledger: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Synchronize `ledger['active_stakes']` from `ledger['current_round_log']`."""
    players = ledger.setdefault("players", {})
    aggregated: dict[str, dict[str, Any]] = {}

    for entry in ledger.get("current_round_log", []):
        login = str(entry.get("github_login", "")).lstrip("@")
        if not login:
            continue
        avatar = str(entry.get("avatar_url") or f"https://github.com/{login}.png?size=32")
        staked = int(entry.get("elo_staked", STAKE_TIERS["standard"]))
        tier = str(entry.get("stake_tier", "standard"))
        player_rec = players.get(login, {})
        current_elo = int(player_rec.get("elo", INITIAL_ELO))

        if login not in aggregated:
            aggregated[login] = {
                "github_login": login,
                "avatar_url": avatar,
                "elo_staked": 0,
                "total_elo_staked": 0,
                "moves_in_round": 0,
                "last_stake_tier": tier,
                "current_elo": current_elo,
                "rank_title": compute_reputation_rank(current_elo),
            }

        item = aggregated[login]
        item["elo_staked"] += staked
        item["total_elo_staked"] = item["elo_staked"]
        item["moves_in_round"] += 1
        item["last_stake_tier"] = tier
        item["potential_win_elo"] = current_elo + item["elo_staked"]
        item["potential_loss_elo"] = current_elo - item["elo_staked"]

    ledger["active_stakes"] = aggregated
    return aggregated


def record_move(
    ledger: dict[str, Any],
    player: PlayerIdentity | str | None = None,
    game_key: str = "connect4",
    move_label: str = "",
    stake_tier: str = "standard",
    bot_reply: str = "",
    issue_number: int = 0,
    *,
    github_login: str | None = None,
    identity: PlayerIdentity | None = None,
    timestamp_utc: str | None = None,
    fetch_online: bool = False,
) -> dict[str, Any]:
    """Record a verified player's staked move in `current_round_log`."""
    resolved_identity = _coerce_identity(
        identity if identity is not None else player,
        github_login=github_login,
        fetch_online=fetch_online,
    )
    now_iso = timestamp_utc or _utc_now_iso()
    player_rec = ensure_player(ledger, resolved_identity, timestamp_utc=now_iso)
    canonical_tier, elo_staked = resolve_stake_tier(stake_tier)

    round_log = ledger.setdefault("current_round_log", [])
    turn_number = len(round_log) + 1

    entry: dict[str, Any] = {
        "timestamp_utc": now_iso,
        "turn_number": turn_number,
        "turn": turn_number,
        "github_login": resolved_identity.github_login,
        "avatar_url": resolved_identity.avatar_url,
        "game_key": game_key,
        "move_label": move_label,
        "stake_tier": canonical_tier,
        "elo_staked": elo_staked,
        "bot_reply": bot_reply,
        "issue_number": int(issue_number),
    }
    round_log.append(entry)

    player_rec["last_active_utc"] = now_iso
    player_rec["total_elo_staked"] = int(player_rec.get("total_elo_staked", 0)) + elo_staked
    ledger["active_game_key"] = game_key
    _rebuild_active_stakes(ledger)
    return entry


record_staked_move = record_move


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _ensure_player_ws4(players: dict[str, dict[str, Any]], handle: str, now_iso: str) -> dict[str, Any]:
    if handle not in players:
        players[handle] = {
            "elo": int(INITIAL_ELO),
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "finishes": 0,
            "last_played_at": now_iso,
        }
    return players[handle]


def _settle_round_ws4(
    ledger: dict[str, Any],
    mode: str,
    winner: str,
    red_contributors: dict[str, int],
    yellow_contributors: dict[str, int],
    finisher_handle: str | None,
    moves_count: int,
    now_iso: str,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Settle ELO deltas across contributors and finisher in VS_BOT or PVP_DUEL."""
    players = ledger.setdefault("players", {})
    bot_elo = int(ledger.get("bot_elo", BOT_INITIAL_ELO))
    deltas: dict[str, int] = {}

    for h in list(red_contributors.keys()) + list(yellow_contributors.keys()):
        _ensure_player_ws4(players, h, now_iso)

    red_total = sum(red_contributors.values()) or 1
    red_weighted_elo = (
        sum(players[h]["elo"] * cnt for h, cnt in red_contributors.items()) / red_total
        if red_contributors
        else float(int(INITIAL_ELO))
    )

    if mode == "VS_BOT":
        opp_elo = float(bot_elo)
    else:
        yellow_total = sum(yellow_contributors.values()) or 1
        opp_elo = (
            sum(players[h]["elo"] * cnt for h, cnt in yellow_contributors.items()) / yellow_total
            if yellow_contributors
            else float(int(INITIAL_ELO))
        )

    norm_winner = (winner or "").strip().lower()
    if norm_winner in ("red", "challengers", "community", "c"):
        actual_red = 1.0
    elif norm_winner in ("yellow", "bot", "house_bot", "b"):
        actual_red = 0.0
    else:
        actual_red = 0.5

    pool_delta_red = K_FACTOR * (actual_red - exp_red if (exp_red := expected_score(red_weighted_elo, opp_elo)) is not None else 0.0)
    pool_delta_yellow = -pool_delta_red

    for h, cnt in red_contributors.items():
        share = cnt / red_total
        d = int(round(pool_delta_red * share))
        if norm_winner in ("red", "challengers", "community", "c") and h == finisher_handle:
            d += FINISHER_BONUS
            players[h]["finishes"] = int(players[h].get("finishes", 0)) + 1
        players[h]["elo"] = max(MIN_ELO_FLOOR, int(players[h]["elo"]) + d)
        players[h]["last_played_at"] = now_iso
        if norm_winner in ("red", "challengers", "community", "c"):
            players[h]["wins"] = int(players[h].get("wins", 0)) + 1
        elif norm_winner in ("yellow", "bot", "house_bot", "b"):
            players[h]["losses"] = int(players[h].get("losses", 0)) + 1
        else:
            players[h]["draws"] = int(players[h].get("draws", 0)) + 1
        deltas[h] = d

    if mode == "VS_BOT":
        bot_delta = int(round(pool_delta_yellow))
        ledger["bot_elo"] = max(MIN_ELO_FLOOR, bot_elo + bot_delta)
    else:
        yellow_total = sum(yellow_contributors.values()) or 1
        for h, cnt in yellow_contributors.items():
            share = cnt / yellow_total
            d = int(round(pool_delta_yellow * share))
            if norm_winner in ("yellow", "bot", "house_bot", "b") and h == finisher_handle:
                d += FINISHER_BONUS
                players[h]["finishes"] = int(players[h].get("finishes", 0)) + 1
            players[h]["elo"] = max(MIN_ELO_FLOOR, int(players[h]["elo"]) + d)
            players[h]["last_played_at"] = now_iso
            if norm_winner in ("yellow", "bot", "house_bot", "b"):
                players[h]["wins"] = int(players[h].get("wins", 0)) + 1
            elif norm_winner in ("red", "challengers", "community", "c"):
                players[h]["losses"] = int(players[h].get("losses", 0)) + 1
            else:
                players[h]["draws"] = int(players[h].get("draws", 0)) + 1
            deltas[h] = deltas.get(h, 0) + d

    recent = ledger.setdefault("recent_rounds", [])
    recent.insert(
        0,
        {
            "settled_at": now_iso,
            "mode": mode,
            "winner": winner,
            "finisher": finisher_handle,
            "moves_count": moves_count,
            "deltas": deltas,
        },
    )
    del recent[20:]
    return ledger, deltas


def settle_round(
    ledger: dict[str, Any],
    mode: str | None = None,
    winner: str | None = None,
    red_contributors: dict[str, int] | None = None,
    yellow_contributors: dict[str, int] | None = None,
    finisher_handle: str | None = None,
    moves_count: int = 0,
    now_iso: str = "2026-10-25T12:00:00Z",
    *,
    settled_at_utc: str | None = None,
) -> Any:
    """Settle the active round across participants (supports both WS4 and legacy signatures)."""
    if mode in ("VS_BOT", "PVP_DUEL") or red_contributors is not None:
        return _settle_round_ws4(
            ledger=ledger,
            mode=mode or "VS_BOT",
            winner=winner or "draw",
            red_contributors=red_contributors or {},
            yellow_contributors=yellow_contributors or {},
            finisher_handle=finisher_handle,
            moves_count=moves_count,
            now_iso=settled_at_utc or now_iso,
        )

    legacy_winner = winner if winner is not None else (mode or "")
    resolved_now = settled_at_utc or _utc_now_iso()
    round_log: list[dict[str, Any]] = list(ledger.get("current_round_log", []))
    raw_winner = str(legacy_winner or "").strip().lstrip("@")
    lower_winner = raw_winner.lower()

    participants_in_round = {
        str(entry.get("github_login", "")).lstrip("@")
        for entry in round_log
        if entry.get("github_login")
    }

    if lower_winner in _VISITOR_WIN_LABELS or raw_winner in participants_in_round:
        outcome = "visitors"
        multiplier = 1
    elif lower_winner in _BOT_WIN_LABELS:
        outcome = "bot"
        multiplier = -1
    elif lower_winner in _DRAW_LABELS:
        outcome = "draw"
        multiplier = 0
    else:
        raise ValueError(
            f"Unrecognized round winner '{legacy_winner}'. Expected 'visitors', 'bot', or 'draw'."
        )

    # Aggregate stakes and moves per participant in the round
    round_stakes: dict[str, dict[str, Any]] = {}
    for entry in round_log:
        login = str(entry.get("github_login", "")).lstrip("@")
        if not login:
            continue
        avatar = str(entry.get("avatar_url") or f"https://github.com/{login}.png?size=32")
        staked = int(entry.get("elo_staked", STAKE_TIERS["standard"]))
        if login not in round_stakes:
            round_stakes[login] = {
                "github_login": login,
                "avatar_url": avatar,
                "elo_staked": 0,
                "moves_in_round": 0,
            }
        round_stakes[login]["elo_staked"] += staked
        round_stakes[login]["moves_in_round"] += 1

    players = ledger.setdefault("players", {})
    settlement_summary: dict[str, dict[str, Any]] = {}

    for login, summary in round_stakes.items():
        if login not in players:
            ensure_player(ledger, login, timestamp_utc=resolved_now)
        player_rec = players[login]
        elo_before = int(player_rec.get("elo", INITIAL_ELO))
        elo_delta = summary["elo_staked"] * multiplier
        elo_after = elo_before + elo_delta
        new_rank = compute_reputation_rank(elo_after)

        player_rec["elo"] = elo_after
        player_rec["peak_elo"] = max(int(player_rec.get("peak_elo", elo_before)), elo_after)
        player_rec["rank_title"] = new_rank
        player_rec["title"] = new_rank
        player_rec["moves_played"] = int(player_rec.get("moves_played", 0)) + int(summary["moves_in_round"])
        player_rec["last_active_utc"] = resolved_now

        if multiplier > 0:
            player_rec["wins"] = int(player_rec.get("wins", 0)) + 1
        elif multiplier < 0:
            player_rec["losses"] = int(player_rec.get("losses", 0)) + 1
        else:
            player_rec["draws"] = int(player_rec.get("draws", 0)) + 1

        settlement_summary[login] = {
            "github_login": login,
            "elo_before": elo_before,
            "elo_staked": summary["elo_staked"],
            "elo_delta": elo_delta,
            "elo_after": elo_after,
            "rank_title": new_rank,
            "moves_in_round": summary["moves_in_round"],
        }

    round_id = int(ledger.get("round_id", len(ledger.get("past_rounds", [])) + 1))
    game_key = str(
        (round_log[-1].get("game_key") if round_log else None)
        or ledger.get("active_game_key")
        or "connect4"
    )

    archived_round = {
        "round_id": round_id,
        "game_key": game_key,
        "winner": outcome,
        "settled_at_utc": resolved_now,
        "turns_count": len(round_log),
        "participants": settlement_summary,
        "moves": copy.deepcopy(round_log),
    }

    ledger.setdefault("past_rounds", []).append(archived_round)
    ledger["current_round_log"] = []
    ledger["active_stakes"] = {}
    ledger["round_id"] = round_id + 1
    return ledger


def get_active_stakes(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return active stakes in the current round sorted by ELO staked descending."""
    working = dict(ledger)
    stakes_map = _rebuild_active_stakes(working)
    items = list(stakes_map.values())
    items.sort(
        key=lambda item: (
            -int(item.get("elo_staked", 0)),
            -int(item.get("moves_in_round", 0)),
            str(item.get("github_login", "")).lower(),
        )
    )
    return items


def get_sorted_leaderboard(ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return all verified players sorted by ELO rating descending."""
    players_raw = ledger.get("players", {})
    rows: list[dict[str, Any]] = []

    if isinstance(players_raw, Mapping):
        pairs = list(players_raw.items())
    elif isinstance(players_raw, list):
        pairs = [
            (str(r.get("github_login", "")), r)
            for r in players_raw
            if isinstance(r, Mapping)
        ]
    else:
        pairs = []

    for key_login, rec in pairs:
        if not isinstance(rec, Mapping):
            continue
        login = str(rec.get("github_login") or key_login or "").lstrip("@")
        if not login:
            continue
        elo = int(rec.get("elo", INITIAL_ELO))
        rank_title = compute_reputation_rank(elo)
        avatar = str(rec.get("avatar_url") or f"https://github.com/{login}.png?size=32")
        wins = int(rec.get("wins", 0))
        losses = int(rec.get("losses", 0))
        draws = int(rec.get("draws", 0))
        moves_played = int(rec.get("moves_played", 0))

        rows.append({
            **dict(rec),
            "github_login": login,
            "avatar_url": avatar,
            "elo": elo,
            "rank_title": rank_title,
            "title": rank_title,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "moves_played": moves_played,
        })

    rows.sort(
        key=lambda r: (
            -int(r["elo"]),
            -int(r["wins"]),
            -int(r["moves_played"]),
            str(r["github_login"]).lower(),
        )
    )

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return rows


def load_ledger(path: str | pathlib.Path | None = None) -> dict[str, Any]:
    """Load the Hall of Records ledger from JSON disk storage."""
    target = pathlib.Path(path) if path is not None else DEFAULT_LEDGER_PATH
    if not target.exists():
        return create_empty_ledger()
    data = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return create_empty_ledger()
    _rebuild_active_stakes(data)
    return data


def save_ledger(ledger: Mapping[str, Any], path: str | pathlib.Path | None = None) -> pathlib.Path:
    """Persist the Hall of Records ledger to JSON disk storage."""
    target = pathlib.Path(path) if path is not None else DEFAULT_LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    working = dict(ledger)
    _rebuild_active_stakes(working)
    target.write_text(json.dumps(working, indent=2) + "\n", encoding="utf-8")
    return target


def load_elo_ledger(path: pathlib.Path | None = None) -> dict[str, Any]:
    """Load the WS4 ELO ledger from disk, initializing bot_elo and recent_rounds if needed."""
    target = pathlib.Path(path) if path is not None else DEFAULT_LEDGER_PATH
    if target.exists():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "players" in data:
                data.setdefault("bot_elo", BOT_INITIAL_ELO)
                data.setdefault("recent_rounds", [])
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"bot_elo": BOT_INITIAL_ELO, "players": {}, "recent_rounds": []}


def save_elo_ledger(path: pathlib.Path, ledger: dict[str, Any]) -> None:
    """Persist the WS4 ELO ledger to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_seeded_ledger() -> dict[str, Any]:
    """Construct a realistic multi-round Hall of Records ledger covering all 4 ELO rank tiers."""
    ledger = create_empty_ledger(game_key="connect4")

    # Round 1: Visitors win (Siyuan-Zhu & karpathy stake conviction 50 ELO each)
    record_move(
        ledger,
        player="Siyuan-Zhu",
        game_key="connect4",
        move_label="Column 4 (Center Opening)",
        stake_tier="conviction",
        bot_reply="Minimax 4-ply countered in Column 4 (eval +8)",
        issue_number=11,
        timestamp_utc="2026-10-19T15:10:00Z",
    )
    record_move(
        ledger,
        player="karpathy",
        game_key="connect4",
        move_label="Column 3 (Diagonal Threat)",
        stake_tier="conviction",
        bot_reply="Minimax 4-ply blocked in Column 5 (eval -4)",
        issue_number=12,
        timestamp_utc="2026-10-19T16:42:00Z",
    )
    record_move(
        ledger,
        player="Siyuan-Zhu",
        game_key="connect4",
        move_label="Column 2 (Double Fork)",
        stake_tier="conviction",
        bot_reply="Visitors connected four on diagonal!",
        issue_number=14,
        timestamp_utc="2026-10-19T17:05:00Z",
    )
    settle_round(ledger, winner="visitors", settled_at_utc="2026-10-19T17:05:12Z")

    # Round 2: Visitors win (Siyuan-Zhu stakes standard 25 ELO -> 1125 Grandmaster; octocat stakes 25 ELO -> 1025)
    record_move(
        ledger,
        player="Siyuan-Zhu",
        game_key="connect4",
        move_label="Column 4 (Center Control)",
        stake_tier="standard",
        bot_reply="Minimax 4-ply countered in Column 3",
        issue_number=18,
        timestamp_utc="2026-10-21T13:20:00Z",
    )
    record_move(
        ledger,
        player="octocat",
        game_key="connect4",
        move_label="Column 5 (Flank Pressure)",
        stake_tier="standard",
        bot_reply="Visitors connected four along Row 6!",
        issue_number=19,
        timestamp_utc="2026-10-21T14:11:00Z",
    )
    settle_round(ledger, winner="visitors", settled_at_utc="2026-10-21T14:11:10Z")

    # Round 3: Bot wins (tiangolo stakes conviction 50 ELO -> 950 Underdog Contender)
    record_move(
        ledger,
        player="tiangolo",
        game_key="connect4",
        move_label="Column 1 (Edge Gambit)",
        stake_tier="conviction",
        bot_reply="Minimax 4-ply seized center trap and connected four in Column 4",
        issue_number=23,
        timestamp_utc="2026-10-22T20:15:00Z",
    )
    settle_round(ledger, winner="bot", settled_at_utc="2026-10-22T20:15:09Z")

    # Active Round 4: Three moves currently in flight with active ELO on the line
    record_move(
        ledger,
        player="karpathy",
        game_key="connect4",
        move_label="Column 4 (Center Tempo)",
        stake_tier="conviction",
        bot_reply="Minimax 4-ply mirrored in Column 4 (depth=4, alpha-beta pruned 148 nodes)",
        issue_number=27,
        timestamp_utc="2026-10-23T14:08:22Z",
    )
    record_move(
        ledger,
        player="octocat",
        game_key="connect4",
        move_label="Column 3 (Left Center Split)",
        stake_tier="standard",
        bot_reply="Minimax 4-ply countered in Column 5 (keeping center parity)",
        issue_number=28,
        timestamp_utc="2026-10-23T16:34:10Z",
    )
    record_move(
        ledger,
        player="tiangolo",
        game_key="connect4",
        move_label="Column 3 (Vertical Stack)",
        stake_tier="conviction",
        bot_reply="Minimax 4-ply blocked vertical threat in Column 3",
        issue_number=29,
        timestamp_utc="2026-10-23T18:50:45Z",
    )

    return ledger

