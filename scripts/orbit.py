#!/usr/bin/env python3
"""Orbit Control Center CLI & Mobile Interjection Harness.

Allows instant interjection from:
  1. Terminal / Antigravity Chat:
     - `python3 scripts/orbit.py status`
     - `python3 scripts/orbit.py interject --word "ANCHORED" --phrase "Morning coffee in Williamsburg." --mode photo`
     - `python3 scripts/orbit.py clear-interjection`
     - `python3 scripts/orbit.py add-source --lens personal_orbit --id "new_creator" --name "Creator" --uri "https://..."`
  2. Phone (via GitHub Mobile App `workflow_dispatch` or Cloud Run webhook):
     - Automatically maps mobile form inputs into `config/orbit_sources.json` and regenerates
       `assets/daily_pulse.svg`, `assets/game_board.svg`, and `README.md`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game_engine import (
    VALID_GAMES,
    build_initial_game_state,
    render_board_svg,
)
from src.state_store import get_state_store
from src.zeitgeist.adk_pipeline import run_daily_zeitgeist
from src.zeitgeist.methodology import load_orbit_config, validate_owner_interjection
from contextlib import contextmanager


_MODE_ALIASES: dict[str, str] = {
    "ai": "VS_BOT",
    "vs_bot": "VS_BOT",
    "coop": "VS_BOT",
    "pvp": "PVP_DUEL",
    "pvp_duel": "PVP_DUEL",
}


def save_orbit_config(config_path: Path, data: dict) -> None:
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def cmd_status(repo_root: Path) -> int:
    config_path = repo_root / "config" / "orbit_sources.json"
    store = get_state_store(repo_root=repo_root)

    cfg = load_orbit_config(config_path)
    interj = cfg.get("active_interjection", {})

    print("=== ORBIT CONTROL CENTER STATUS ===")
    for lens_id, spec in cfg.get("lenses", {}).items():
        w = float(spec.get("weight", 0.0)) * 100.0
        sources = spec.get("sources", [])
        print(f"  • {lens_id:<16} ({w:4.0f}% weight) : {len(sources)} public feeds")

    print(f"  • Active Interjection : {'ENABLED' if interj.get('enabled') else 'off (autonomous ADK mode)'}")
    if interj.get("enabled"):
        print(f"    - Override Word     : {interj.get('override_word')}")
        print(f"    - Override Phrase   : {interj.get('override_phrase')}")
        print(f"    - Personal Note     : {interj.get('personal_note')}")
        print(f"    - Visual Mode       : {interj.get('override_visual_mode')}")

    pulse_state = store.load_pulse_state()
    analysis = pulse_state.get("analysis", {})
    visual = pulse_state.get("visual_decision", {})
    if analysis:
        print(
            f"  • Today's Pulse       : {analysis.get('word_of_the_day')} "
            f"(Score {analysis.get('composite_score')} / 100 · {visual.get('selected_mode')})"
        )

    gstate = store.load_game_state()
    print(f"  • Active Profile Game : {gstate.get('active_game', 'connect4')} (turn {gstate.get('turn', 1)})")

    return 0


@contextmanager
def _preserve_local_publication_on_failure(root: Path):
    patterns = ("config/orbit_sources.json", "README.md", "variants/*.md", "data/*.json", "state/*.json", "assets/**/*")
    def files():
        return {p for pattern in patterns for p in root.glob(pattern) if p.is_file()}
    snapshot = {p: p.read_bytes() for p in files()}
    try:
        yield
    except BaseException as exc:
        from src.zeitgeist.publication_commit import PublicationMirrorError
        if isinstance(exc, PublicationMirrorError):
            raise  # Canonical state may have committed; never restore stale local files.
        for path in files() - snapshot.keys():
            path.unlink()
        for path, data in snapshot.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        raise


def cmd_interject(repo_root: Path, args: argparse.Namespace) -> int:
    config_path = repo_root / "config" / "orbit_sources.json"
    cfg = load_orbit_config(config_path)

    interjection = validate_owner_interjection({
        "enabled": True, "override_word": args.word, "override_phrase": args.phrase,
        "override_visual_mode": args.mode, "override_image_url": args.image_url,
        "personal_note": args.note,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_channel": args.channel,
    })
    with _preserve_local_publication_on_failure(repo_root):
        cfg["active_interjection"] = interjection
        save_orbit_config(config_path, cfg)
        result = run_daily_zeitgeist(repo_root=repo_root, prefer_visual_mode=interjection["override_visual_mode"])
    print(
        f"[Orbit Interject] Updated Pulse -> Word: {result['analysis']['word_of_the_day']} | "
        f"Visual: {result['visual_decision']['selected_mode']}"
    )
    return 0


def cmd_clear_interjection(repo_root: Path) -> int:
    config_path = repo_root / "config" / "orbit_sources.json"
    cfg = load_orbit_config(config_path)
    cfg["active_interjection"] = {
        "enabled": False,
        "override_word": None,
        "override_phrase": None,
        "override_visual_mode": None,
        "override_image_url": None,
        "personal_note": None,
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_channel": "cli_reset",
    }
    with _preserve_local_publication_on_failure(repo_root):
        save_orbit_config(config_path, cfg)
        result = run_daily_zeitgeist(repo_root=repo_root)
    print(
        f"[Orbit Reset] Restored autonomous ADK synthesis -> Word: {result['analysis']['word_of_the_day']}"
    )
    return 0


def cmd_add_source(repo_root: Path, args: argparse.Namespace) -> int:
    config_path = repo_root / "config" / "orbit_sources.json"
    cfg = load_orbit_config(config_path)
    lenses = cfg.get("lenses", {})
    if args.lens not in lenses:
        raise SystemExit(f"Unknown lens '{args.lens}'. Choose from: {', '.join(lenses.keys())}")

    tags = [t.strip().lower() for t in (args.tags or "").split(",") if t.strip()]
    sources = lenses[args.lens].setdefault("sources", [])
    sources = [s for s in sources if s.get("id") != args.id]
    sources.append(
        {
            "id": args.id,
            "name": args.name,
            "category": args.category,
            "feed_type": "rss_or_public_web",
            "uri": args.uri,
            "tags": tags,
        }
    )
    lenses[args.lens]["sources"] = sources
    save_orbit_config(config_path, cfg)
    print(f"[Orbit Sources] Added '{args.name}' ({args.id}) to lens '{args.lens}'.")
    return 0


def cmd_switch_game(repo_root: Path, args: argparse.Namespace) -> int:
    raw_game = str(
        getattr(args, "game_key", None) or getattr(args, "game", None) or ""
    ).strip().lower()
    if raw_game not in VALID_GAMES:
        raise SystemExit(
            f"Unsupported game '{raw_game}'. Only {', '.join(VALID_GAMES)} is supported."
        )

    raw_mode = getattr(args, "mode", None)
    target_mode = "VS_BOT"
    if raw_mode is not None and str(raw_mode).strip():
        mode_key = str(raw_mode).strip().lower()
        if mode_key not in _MODE_ALIASES:
            raise SystemExit(
                f"Unsupported game mode '{raw_mode}'. Choose from: ai, pvp, coop."
            )
        target_mode = _MODE_ALIASES[mode_key]

    store = get_state_store(repo_root=repo_root)
    current = store.load_game_state()
    round_id = max(1, int(current.get("round_id", 1) or 1))
    next_version = int(current.get("state_version", 0) or 0) + 1

    state = build_initial_game_state("connect4", round_id=round_id)
    state["mode"] = target_mode
    state["state_version"] = next_version
    store.save_game_state(state)

    svg_path = repo_root / "assets" / "game_board.svg"
    if svg_path.parent.exists():
        svg_path.write_text(render_board_svg(state), encoding="utf-8")

    print(
        f"[Orbit Game] Active game set to {raw_game} "
        f"(mode={target_mode}, round={round_id}, state_version={next_version})."
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orbit Control Center CLI & Mobile Harness")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show current orbit weights, pulse word, and active game")

    p_inter = sub.add_parser("interject", help="Interject a word, phrase, note, or visual mode")
    p_inter.add_argument("--word", help="Override Word of the Day (e.g. ANCHORED)")
    p_inter.add_argument("--phrase", help="Override 1-line explanation phrase")
    p_inter.add_argument("--note", help="Short personal field note")
    p_inter.add_argument("--mode", choices=["photo", "blueprint", "auto"], default="auto")
    p_inter.add_argument("--image-url", help="Optional custom image URL")
    p_inter.add_argument("--channel", default="cli_or_mobile", help="Source channel label")

    sub.add_parser("clear-interjection", help="Clear manual override and return to autonomous ADK mode")

    p_add = sub.add_parser("add-source", help="Add a creator, region, or community to an orbit lens")
    p_add.add_argument("--lens", default="personal_orbit")
    p_add.add_argument("--id", required=True)
    p_add.add_argument("--name", required=True)
    p_add.add_argument("--uri", required=True)
    p_add.add_argument("--category", default="builder_creator")
    p_add.add_argument("--tags", default="craft,architecture")

    p_switch = sub.add_parser(
        "switch-game",
        help="Reset or switch the active Connect 4 game mode (ai | pvp | coop)",
    )
    p_switch.add_argument(
        "game_key",
        nargs="?",
        default="connect4",
        help="Game key to activate (must be connect4)",
    )
    p_switch.add_argument(
        "--mode",
        choices=["ai", "pvp", "coop", "vs_bot", "pvp_duel"],
        default=None,
        help="Optional Connect 4 mode (ai | pvp | coop)",
    )

    sub.add_parser("github-event", help="Process GitHub Actions workflow_dispatch or schedule event payload")

    return parser


def cmd_github_event(repo_root: Path) -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    inputs: dict = {}
    if event_path and Path(event_path).exists():
        payload = json.loads(Path(event_path).read_text(encoding="utf-8"))
        inputs = payload.get("inputs") or {}

    if str(inputs.get("clear_override", "")).lower() == "true":
        cmd_clear_interjection(repo_root)
    elif any(inputs.get(k) for k in ("override_word", "override_phrase", "personal_note")) or (
        inputs.get("visual_mode") in ("photo", "blueprint")
    ):
        ns = argparse.Namespace(
            word=inputs.get("override_word"),
            phrase=inputs.get("override_phrase"),
            note=inputs.get("personal_note"),
            mode=inputs.get("visual_mode", "auto"),
            image_url=None,
            channel="github_mobile_dispatch",
        )
        cmd_interject(repo_root, ns)
    else:
        run_daily_zeitgeist(repo_root=repo_root)
    return 0


def _normalize_argv(argv: list[str]) -> list[str]:
    """Support both subcommands and top-level flags (`--status`, `--interject`, `--clear-interjection`)."""
    if not argv:
        return argv
    first = argv[0]
    if first == "--clear-interjection":
        return ["clear-interjection", *argv[1:]]
    if first == "--interject":
        return ["interject", *argv[1:]]
    if first == "--status":
        return ["status", *argv[1:]]
    if first == "--switch-game":
        return ["switch-game", *argv[1:]]
    return argv


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    norm = _normalize_argv(raw)
    parser = build_parser()
    args = parser.parse_args(norm)
    if args.command == "status":
        return cmd_status(REPO_ROOT)
    if args.command == "interject":
        return cmd_interject(REPO_ROOT, args)
    if args.command == "clear-interjection":
        return cmd_clear_interjection(REPO_ROOT)
    if args.command == "add-source":
        return cmd_add_source(REPO_ROOT, args)
    if args.command == "switch-game":
        return cmd_switch_game(REPO_ROOT, args)
    if args.command == "github-event":
        return cmd_github_event(REPO_ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
