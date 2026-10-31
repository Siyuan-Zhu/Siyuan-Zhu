#!/usr/bin/env python3
"""Exercise the visitor CLI on a disposable JSON copy, never canonical state."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.game_engine import create_initial_game_state
from src.state_store import JsonFileStateStore


def verify(source: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="egm-interactions-") as temp:
        root = Path(temp) / "repo"
        shutil.copytree(source, root, ignore=shutil.ignore_patterns(".git", ".venv", "work", "__pycache__"))
        store = JsonFileStateStore(root)
        store.save_game_state(create_initial_game_state())
        store.save_game_rate_limits({})
        before = store.load_game_state()
        turn = int(before["turn"])
        env = dict(os.environ, STATE_STORE_BACKEND="json", ZEITGEIST_LIVE_LLM="0",
                   ZEITGEIST_LIVE_COLLECT="0", ZEITGEIST_LIVE_UNSPLASH="0",
                   PULSE_VOTE_SALT="disposable-staging-fixture")
        for key in ("GITHUB_ACTIONS", "GITHUB_EVENT_PATH", "ISSUE_NUMBER", "ISSUE_TITLE", "VISITOR_LOGIN"):
            env.pop(key, None)

        def event(login, title, accepted=False):
            command = [sys.executable, str(root / "scripts/process_visitor_event.py"),
                       "--repo-root", str(root), "--login", login, "--title", title]
            if accepted:
                command.append("--require-accepted")
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"Visitor CLI failed ({result.returncode}): {result.stderr[-1000:]}")
            return json.loads(result.stdout)

        title = f"game:move:connect4:col:4:turn={turn}"
        move = event("staging-player", title, accepted=True)
        after = store.load_game_state()
        if not (move.get("accepted") and int(after["turn"]) > turn and move.get("bot_move")):
            raise AssertionError("Move must advance the turn and include a bot response")
        pieces = [cell for row in after["board"] for cell in row]
        if sum(cell in (1, "C") for cell in pieces) != 1 or sum(cell in (2, "B") for cell in pieces) != 1:
            raise AssertionError("Fresh board must contain exactly one visitor and one bot disc")
        stale = event("staging-second", title)
        if stale.get("accepted") is not False or store.load_game_state() != after:
            raise AssertionError("Stale move must be rejected without changing the board")
        vote = event("staging-voter", "pulse:vote:spot_on", accepted=True)
        duplicate = event("staging-voter", "pulse:vote:spot_on")
        if vote.get("status") != "recorded_vote" or duplicate.get("status") != "ignored" or vote["tally"] != duplicate["tally"]:
            raise AssertionError("Vote must count once; duplicate must leave tally unchanged")
        return {"backend": "disposable_json", "move": move["status"], "bot_response": True,
                "stale_move": stale["status"], "vote": vote["status"],
                "duplicate_vote": duplicate["status"], "isolation": "temporary_copy"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    report = verify(args.repo_root.resolve())
    text = json.dumps(report, indent=2) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
