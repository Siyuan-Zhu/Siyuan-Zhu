#!/usr/bin/env python3
"""CLI entrypoint for the Daily Zeitgeist v2.0 Multi-Agent Pipeline.

Executes the 4-stage pipeline:
  1. 96h Macro + 336h Orbit Signal Ingestion & Stratified Windowing
  2. Average-Linkage Clustering & Disjoint Surge / Orbit Resonance Scoring
  3. Editorial Director Synthesis (with Latin-Script & 30-Day Deduplication Gate)
  4. Multimodal Visual Critic Funnel, Dual-Res WebP + SVG Rendering, and README Rebuild
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_readme import build_all_readmes
from src.state_store import get_state_store
from src.zeitgeist.adk_pipeline import run_daily_zeitgeist
from src.zeitgeist.daily_outcomes import record_generation_failure


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the Daily Zeitgeist v2.0 synthesis and rebuild README surfaces."
    )
    parser.add_argument(
        "--date",
        default=os.environ.get("ZEITGEIST_DATE", ""),
        help="ISO date string (YYYY-MM-DD). Defaults to current UTC date.",
    )
    parser.add_argument(
        "--prefer-visual-mode",
        default=None,
        help="Optional visual mode override.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root directory. Defaults to workspace root.",
    )
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Deprecated compatibility flag; publications always validate README surfaces.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    target_root = Path(args.repo_root).resolve()
    resolved_date = (
        str(args.date).strip()[:10]
        if str(args.date).strip()
        else datetime.now(timezone.utc).strftime("%Y-%m-%d")
    )

    try:
        store = get_state_store(repo_root=target_root)
        if hasattr(store, "reconcile_to_git"):
            store.reconcile_to_git()
        result = run_daily_zeitgeist(
            repo_root=target_root,
            date_iso=resolved_date,
            prefer_visual_mode=args.prefer_visual_mode,
        )
    except BaseException as error:
        try:
            record_generation_failure(target_root, error, os.environ)
        except Exception:
            pass  # Preserve the actual generation failure if observation fails.
        raise

    # The orchestrator validates all README surfaces before canonical persistence.

    commission = result.get("commission") or {}
    pulse = result.get("published_pulse") or {}
    word = commission.get("word") or result.get("analysis", {}).get("word_of_the_day", "Equilibrium")
    medium = pulse.get("visual_tier_used") or result.get("visual_decision", {}).get("selected_mode", "BESPOKE_SVG")
    print(f"Zeitgeist synthesis complete for {resolved_date}: word={word} medium={medium}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
