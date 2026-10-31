#!/usr/bin/env python3
"""Process visitor 1-click GitHub Issue moves (`game:move:*`, `connect4: drop *`) and mood votes (`pulse:vote:*`)."""

from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_readme import resolve_repo_slug
from src.game_engine import (
    DEFAULT_GAME_KEY,
    GITHUB_LOGIN_RE,
    MoveRequest,
    apply_visitor_move,
    check_and_heal_stale_turn,
    check_issue_round_lock,
    parse_move_issue_title,
    process_move,
)
from src.state_store import FirestoreStateStore, StateStore, compute_voter_hmac, get_state_store

_LEGACY_MOOD_KEY_MAP: dict[str, str] = {
    "resonant": "spot_on",
    "reflective": "spot_on",
    "electric": "too_sunny",
    "wistful": "too_sunny",
    "skeptical": "too_heavy",
}


def _resolve_github_token(repo_root: Path | None = None) -> str:
    """Resolve GH_TOKEN from env or `actions/checkout@v4`'s persisted `.git/config` extraheader."""
    env_tok = (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or "").strip()
    if env_tok:
        return env_tok
    root = repo_root or REPO_ROOT
    git_cfg = root / ".git" / "config"
    if git_cfg.is_file():
        try:
            cfg_text = git_cfg.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"AUTHORIZATION:\s*basic\s+([A-Za-z0-9+/=]+)", cfg_text, re.IGNORECASE)
            if m:
                decoded = base64.b64decode(m.group(1)).decode("utf-8", errors="ignore")
                if ":" in decoded:
                    return decoded.split(":", 1)[1].strip()
        except Exception:
            pass
    return ""


def _auto_close_and_lock_github_issue(
    issue_number: int,
    comment_body: str,
    repo_root: Path | None = None,
    repo_slug: str | None = None,
) -> None:
    """When running inside GitHub Actions, comment on, close, and lock the visitor's issue."""
    if os.environ.get("GITHUB_ACTIONS") != "true" or issue_number <= 0:
        return
    repo = (os.environ.get("GITHUB_REPOSITORY") or "").strip() or resolve_repo_slug(repo_slug)
    issue_str = str(issue_number)
    gh_env = dict(os.environ)
    tok = _resolve_github_token(repo_root)
    if tok and not gh_env.get("GH_TOKEN"):
        gh_env["GH_TOKEN"] = tok
    commands = [
        ["gh", "issue", "comment", issue_str, "--repo", repo, "--edit-last", "--create-if-none", "--body", comment_body],
        ["gh", "issue", "close", issue_str, "--repo", repo, "--reason", "completed"],
        ["gh", "api", "-X", "PUT", f"repos/{repo}/issues/{issue_str}/lock", "-f", "lock_reason=resolved"],
    ]
    for phase, command in zip(("comment", "close", "lock"), commands):
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True,
                                    env=gh_env, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError(f"GitHub Issue feedback failed during {phase}") from None
        if result.returncode != 0:
            raise RuntimeError(f"GitHub Issue feedback failed during {phase}")



def process_event(
    repo_root: Path,
    visitor_login: str,
    issue_title: str,
    issue_number: int = 0,
    fetch_online: bool = False,
    timestamp_epoch: float | None = None,
    store: StateStore | None = None,
    repo_slug: str | None = None,
) -> dict:
    """Validate visitor interaction against Anti-Abuse Guardrails, apply game move or mood vote, and persist state."""
    title = (issue_title or "").strip()
    state_store = store if store is not None else get_state_store(repo_root=repo_root)
    if hasattr(state_store, "reconcile_git_from_firestore_if_needed"):
        state_store.reconcile_git_from_firestore_if_needed()

    # 0. Repo-Owner Mobile Issue Interjection (`orbit:interject:<WORD>:<PHRASE>`, `orbit:reset`)
    if title.lower().startswith("orbit:"):
        resolved_slug = resolve_repo_slug(repo_slug)
        repo_owner = resolved_slug.split("/", 1)[0].strip() if "/" in resolved_slug else "Siyuan-Zhu"
        if (visitor_login or "").strip().lower() != repo_owner.lower():
            _auto_close_and_lock_github_issue(
                issue_number,
                f"Rejected `orbit:` command from non-owner `@{visitor_login}`.",
                repo_root=repo_root,
                repo_slug=resolved_slug,
            )
            return {
                "status": "rejected_unauthorized",
                "reason": "Only the repository owner can issue orbit:* commands.",
                "player": visitor_login,
            }
        from scripts.orbit import cmd_clear_interjection, cmd_interject

        parts = title.split(":", 3)
        subcmd = parts[1].strip().lower() if len(parts) >= 2 else ""
        if subcmd == "reset":
            cmd_clear_interjection(repo_root)
            _auto_close_and_lock_github_issue(
                issue_number,
                "Cleared manual interjection and restored autonomous ADK synthesis.",
                repo_root=repo_root,
            )
            return {"status": "orbit_reset"}
        elif subcmd == "interject" and len(parts) >= 3:
            word_arg = parts[2].strip().upper()
            phrase_arg = parts[3].strip() if len(parts) >= 4 else ""
            ns = argparse.Namespace(
                word=word_arg,
                phrase=phrase_arg or None,
                note=None,
                mode="auto",
                image_url=None,
                channel="github_owner_issue",
            )
            cmd_interject(repo_root, ns)
            _auto_close_and_lock_github_issue(
                issue_number,
                f"Applied owner interjection (`{word_arg}`). Closing and locking issue.",
                repo_root=repo_root,
            )
            return {"status": "orbit_interjected", "word": word_arg}
        elif (
            subcmd in ("add", "remove")
            and len(parts) >= 4
            and parts[2].strip().lower() == "instagram"
        ):
            handle_arg = parts[3].strip().lstrip("@").lower()
            sources_path = repo_root / "config" / "orbit_sources.json"
            if sources_path.is_file() and handle_arg:
                cfg_data = json.loads(sources_path.read_text(encoding="utf-8"))
                blocklist = {
                    str(b).strip().lstrip("@").lower()
                    for b in cfg_data.get("handle_blocklist", [])
                }
                if subcmd == "add" and handle_arg in blocklist:
                    _auto_close_and_lock_github_issue(
                        issue_number,
                        f"Rejected blocked Instagram handle `@{handle_arg}`. Closing and locking issue.",
                        repo_root=repo_root,
                    )
                    return {
                        "status": "ignored",
                        "reason": f"Handle '@{handle_arg}' is in handle_blocklist",
                        "handle": handle_arg,
                    }
                handles_list = list(cfg_data.get("curated_o3_instagram_handles", []))
                if subcmd == "add" and handle_arg not in handles_list:
                    handles_list.append(handle_arg)
                elif subcmd == "remove" and handle_arg in handles_list:
                    handles_list = [h for h in handles_list if h != handle_arg]
                cfg_data["curated_o3_instagram_handles"] = handles_list
                sources_path.write_text(json.dumps(cfg_data, indent=2) + "\n", encoding="utf-8")

            if (
                handle_arg
                and hasattr(state_store, "load_orbit_roster")
                and hasattr(state_store, "save_orbit_roster")
            ):
                roster = state_store.load_orbit_roster()
                handles_map = dict(roster.get("handles") or {})
                now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if subcmd == "add":
                    existing_entry = dict(handles_map.get(handle_arg) or {})
                    handles_map[handle_arg] = {
                        "handle": handle_arg,
                        "decayed_score": max(float(existing_entry.get("decayed_score", 0.0)), 2.0),
                        "vitality_score": min(1.0, max(float(existing_entry.get("vitality_score", 0.0)), 1.0)),
                        "mention_count": max(int(existing_entry.get("mention_count", 0)), 2),
                        "promoted": True,
                        "distinct_parent_sources": list(
                            existing_entry.get("distinct_parent_sources")
                            or ["owner_curated_o3", "owner_issue_dispatch"]
                        ),
                        "parent_sources": list(
                            existing_entry.get("parent_sources")
                            or ["owner_curated_o3", "owner_issue_dispatch"]
                        ),
                        "first_seen_utc": str(existing_entry.get("first_seen_utc") or now_iso),
                        "last_seen_utc": now_iso,
                        "last_mentioned_at": now_iso,
                        "last_updated_utc": now_iso,
                        "status": "ACTIVE_O4_SCOUT",
                        "tier": "O3_CURATED",
                    }
                elif subcmd == "remove":
                    handles_map.pop(handle_arg, None)
                roster["handles"] = handles_map
                roster["updated_at"] = now_iso
                roster["state_version"] = int(roster.get("state_version", 0)) + 1
                state_store.save_orbit_roster(roster)

            _auto_close_and_lock_github_issue(
                issue_number,
                f"Applied owner Instagram seed `{subcmd}` (`@{handle_arg}`). Closing and locking issue.",
                repo_root=repo_root,
            )
            return {"status": f"orbit_{subcmd}_instagram", "handle": handle_arg}

    # 1. Visitor Mood Vote (`pulse:vote:spot_on` | `pulse:vote:too_sunny` | `pulse:vote:too_heavy`)
    if title.lower().startswith("pulse:vote:"):
        raw_vote_key = title.split(":", 2)[2].strip().lower()
        vote_key = _LEGACY_MOOD_KEY_MAP.get(raw_vote_key, raw_vote_key)
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        voter_hmac = compute_voter_hmac(
            client_ip=visitor_login or "github-issue-voter",
            user_agent="github-issue",
            date_str=today_str,
        )
        accepted, tally = state_store.record_mood_vote(
            date_str=today_str,
            mood_key=vote_key,
            voter_hmac=voter_hmac,
        )
        if accepted:
            daily_svg_path = repo_root / "assets" / "daily_pulse.svg"
            if daily_svg_path.exists():
                try:
                    from src.zeitgeist.visual_router import update_svg_mood_vote_bar

                    update_svg_mood_vote_bar(daily_svg_path, tally)
                except Exception:
                    pass
            _auto_close_and_lock_github_issue(
                issue_number,
                f"Recorded mood vote `{vote_key}` from `@{visitor_login}`. Closing and locking issue.",
            )
            return {"status": "recorded_vote", "vote": vote_key, "tally": tally}
        _auto_close_and_lock_github_issue(
            issue_number, "No additional vote counted. This account may already have voted today, or the vote command is not current.",
            repo_root=repo_root, repo_slug=resolve_repo_slug(repo_slug))
        return {"status": "ignored", "reason": f"Duplicate or unknown vote key '{raw_vote_key}'", "tally": tally}

    # 2. Reactive Community Game Move (`game:move:<game>:<move_id>:turn=<N>` or `connect4: drop <N>`)
    if isinstance(state_store, FirestoreStateStore):
        from src.game_events import apply_issue
        from src.game_engine import render_board_svg

        response, canonical_game, ledger = apply_issue(
            state_store, repository=resolve_repo_slug(repo_slug), issue_number=issue_number,
            actor=visitor_login, title=title,
            now_epoch=float(timestamp_epoch) if timestamp_epoch is not None else time.time())
        if canonical_game is not None:
            svg_path = repo_root / 'assets' / 'game_board.svg'
            svg_path.parent.mkdir(parents=True, exist_ok=True)
            svg_path.write_text(render_board_svg(canonical_game, elo_ledger=ledger), encoding='utf-8')
        # Final rejections need feedback even though there is no board to render.
        # Storage conflicts/unknown outcomes remain open for explicit recovery.
        status = response.get('status', 'unknown')
        if status not in {'missing_event_identity', 'ignored'}:
            message = response.get('message') or (
                'Issue already accepted; current canonical board rebuilt.' if response.get('previously_accepted') else
                f'Move not applied (`{status}`). Open the current profile README for the latest round and turn links.')
            if response.get('accepted') is not True and not response.get('previously_accepted'):
                message = f'Move not applied (`{status}`): {message} Open the current profile README for updated controls.'
            current_url = f'https://github.com/{resolve_repo_slug(repo_slug)}#connect-four'
            if response.get('accepted') or response.get('previously_accepted'):
                message = f'Saved to the canonical game. {message} README publication is pending; this acknowledgment does not confirm a board commit. Do not resubmit this move.'
            message += f'\n\n[Current board]({current_url}).'
            _auto_close_and_lock_github_issue(issue_number, message,
                                             repo_root=repo_root, repo_slug=resolve_repo_slug(repo_slug))
        return response
    current_state = state_store.load_game_state()
    round_matches, round_reason = check_issue_round_lock(current_state, title)
    if not round_matches:
        _auto_close_and_lock_github_issue(issue_number,
            f'Move not applied (`{round_reason}`). Open the current profile README for current round links.',
            repo_root=repo_root, repo_slug=resolve_repo_slug(repo_slug))
        return {'status': round_reason, 'accepted': False, 'turn': current_state.get('turn', 1)}
    active_game_key = str(current_state.get("active_game", DEFAULT_GAME_KEY)).strip().lower()
    parsed = parse_move_issue_title(title, active_game_key=active_game_key)
    if parsed is not None:
        req_game, move_id, expected_turn = parsed
        now_ts = float(timestamp_epoch) if timestamp_epoch is not None else time.time()
        use_dual_mode_c4 = req_game == "connect4" and active_game_key == "connect4"
        if use_dual_mode_c4:
            clean_handle = (visitor_login or "").strip().lstrip("@")
            ok_turn, _ = check_and_heal_stale_turn(current_state, expected_turn)
            rate_limits_before = (
                copy.deepcopy(state_store.load_game_rate_limits())
                if hasattr(state_store, "load_game_rate_limits")
                else None
            )
            is_new_round = (
                not str(move_id).strip().lower().startswith("mode:")
                and (
                    int(current_state.get("turn", 1)) <= 1
                    or int(current_state.get("moves_this_round", 0) or 0) == 0
                    or str(current_state.get("status", "ACTIVE")).upper() == "COMPLETED"
                    or bool(current_state.get("round_over", False))
                )
            )
            if (
                GITHUB_LOGIN_RE.fullmatch(clean_handle)
                and not clean_handle.lower().endswith("-bot")
                and ok_turn
                and hasattr(state_store, "check_and_increment_game_rate_limit")
            ):
                rl_ok, rl_reason = state_store.check_and_increment_game_rate_limit(
                    actor_handle=visitor_login,
                    now_epoch=int(now_ts),
                    mode=str(current_state.get("mode", "VS_BOT")),
                    is_new_round=is_new_round,
                )
                if not rl_ok:
                    _auto_close_and_lock_github_issue(
                        issue_number,
                        f"Move not applied (`{rl_reason}`): Rate limit or cooldown active.",
                    )
                    return {
                        "status": rl_reason,
                        "accepted": False,
                        "message": f"Rate limit or cooldown active ({rl_reason}).",
                        "turn": current_state.get("turn", 1),
                    }
            bot_memory = (
                state_store.load_bot_memory()
                if hasattr(state_store, "load_bot_memory")
                else {"states": {}, "games_learned": 0}
            )
            elo_ledger = (
                state_store.load_elo_ledger()
                if hasattr(state_store, "load_elo_ledger")
                else {"bot_elo": 1200, "players": {}, "recent_rounds": []}
            )
            result = apply_visitor_move(
                current_state,
                actor_handle=visitor_login,
                action=move_id,
                now_epoch=now_ts,
                bot_memory=bot_memory,
                elo_ledger=elo_ledger,
                expected_turn=expected_turn,
            )
            if result.accepted:
                if hasattr(state_store, "save_bot_memory"):
                    state_store.save_bot_memory(bot_memory)
                if hasattr(state_store, "save_elo_ledger"):
                    state_store.save_elo_ledger(elo_ledger)
            elif (
                rate_limits_before is not None
                and hasattr(state_store, "save_game_rate_limits")
            ):
                state_store.save_game_rate_limits(rate_limits_before)
        else:
            request = MoveRequest(
                game_key=req_game,
                move_id=move_id,
                player_login=visitor_login,
                expected_turn=expected_turn,
                timestamp_epoch=now_ts,
                issue_number=issue_number,
            )
            result = process_move(current_state, request)

        if not result.accepted:
            _auto_close_and_lock_github_issue(
                issue_number,
                f"Move not applied (`{result.status_code}`): {result.message}",
            )
            return {
                "status": result.status_code,
                "accepted": False,
                "message": result.message,
                "turn": result.new_state.get("turn", 1),
            }

        state_store.save_game_state(result.new_state)
        svg_path = repo_root / "assets" / "game_board.svg"
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(result.svg_board, encoding="utf-8")

        _auto_close_and_lock_github_issue(
            issue_number,
            f"Applied move by `@{visitor_login}`: {result.message}",
        )
        return {
            "status": "move_applied",
            "status_code": result.status_code,
            "accepted": True,
            "player": visitor_login,
            "game_key": result.new_state.get("active_game"),
            "turn": result.new_state.get("turn"),
            "player_move": result.player_move_summary,
            "bot_move": result.bot_move_summary,
            "round_over": result.round_over,
            "winner": result.winner,
            "message": result.message,
        }

    return {"status": "ignored", "reason": "Issue title does not match game:move:* or pulse:vote:*"}


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (`--login`, `--title`, `--issue-number`, `--event-name`) with env fallback."""
    parser = argparse.ArgumentParser(
        description="Process visitor 1-click GitHub Issue moves and mood votes."
    )
    parser.add_argument(
        "--login",
        "--actor",
        dest="login",
        default=os.environ.get("VISITOR_LOGIN", ""),
        help="GitHub login of the visitor.",
    )
    parser.add_argument("--title", default=os.environ.get("ISSUE_TITLE", ""), help="Issue title command string.")
    parser.add_argument(
        "--issue-number",
        type=int,
        default=int(os.environ.get("ISSUE_NUMBER", "0") or "0"),
        help="GitHub Issue number.",
    )
    parser.add_argument("--event-name", default="", help="GitHub Actions event name.")
    parser.add_argument("--result-file", type=Path, help="Private runner result for publication feedback.")
    parser.add_argument(
        "--require-accepted", action="store_true",
        help="Fail a staging success-path check when the action was not accepted.",
    )
    parser.add_argument("--repo-root", default="", help="Optional repository root override.")
    parser.add_argument(
        "--repo-slug",
        default=None,
        help="Optional target GitHub repository slug (e.g., Siyuan-Zhu/Siyuan-Zhu).",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def main(argv: list[str] | None = None, *, store: StateStore | None = None) -> int:
    args = parse_cli_args(argv)
    repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT
    resolved_slug = resolve_repo_slug(args.repo_slug)
    login = (args.login or os.environ.get("VISITOR_LOGIN", "")).strip()
    title = (args.title or os.environ.get("ISSUE_TITLE", "")).strip()
    num = int(args.issue_number or int(os.environ.get("ISSUE_NUMBER", "0") or "0"))
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if event_path and Path(event_path).exists():
        ev = json.loads(Path(event_path).read_text(encoding="utf-8"))
        issue = ev.get("issue") or {}
        login = login or (issue.get("user") or {}).get("login", "octocat")
        title = title or issue.get("title", "")
        num = num or int(issue.get("number", 0) or 0)
    res = process_event(
        repo_root,
        login or "octocat",
        title,
        issue_number=num,
        fetch_online=False,
        repo_slug=resolved_slug,
        store=store,
    )
    print(json.dumps(res, indent=2))
    if args.result_file:
        args.result_file.write_text(json.dumps(res))
        args.result_file.chmod(0o600)
    from scripts.build_readme import build_all_readmes

    build_all_readmes(repo_root, repo_slug=resolved_slug, state_store=store)
    if args.require_accepted and not (
        res.get("accepted") is True or res.get("status") == "recorded_vote"
    ):
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
