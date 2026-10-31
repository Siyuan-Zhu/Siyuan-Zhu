#!/usr/bin/env python3
"""Automated Release Link & Asset Validator CLI for the Siyuan-Zhu Profile README bundle.

Audits `README.md`, `PROJECTS.md`, and `variants/*.md` before staging or production cutover:
  1. Every relative asset link (`./assets/...`, `../assets/...`, stripping `?v=...`) resolves to an
     existing, non-empty file on disk.
  2. `assets/daily_pulse.svg` and `assets/game_board.svg` parse cleanly as XML via `xml.etree.ElementTree`.
  3. `assets/daily_pulse_dark.webp` and `assets/daily_pulse_light.webp` have valid RIFF/WEBP headers.
  4. All currently available interactive GitHub Issue links (votes, legal columns and mode switches)
     in `README.md` (and variants) target `https://github.com/{expected_repo_slug}/issues/new` and match
     the active turn `<T>` in `data/connect4_state.json` (or `data/game_state.json`).
  5. Zero dead `gl-variance-and-run-rate-pipeline` links appear anywhere, and all 5 Flagship repository
     URLs are present in `PROJECTS.md` and every role/architectural variant in `variants/*.md`.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.zeitgeist.publication import verify_publication_bundle

EXPECTED_VARIANT_FILENAMES: tuple[str, ...] = (
    "README-ai-engineering.md",
    "README-fpa-analytics.md",
    "README-bi-data.md",
    "README.A-minimal-editorial.md",
    "README.B-interactive-arcade.md",
    "README.C-hardware-bento.md",
    "README.D-live-telemetry.md",
    "README.E-hybrid-flagship.md",
)

EXPECTED_FLAGSHIP_URLS: tuple[str, ...] = (
    "https://github.com/Siyuan-Zhu/multi-agent-fpa-workspace",
    "https://github.com/Siyuan-Zhu/finance-analytics-warehouse",
    "https://github.com/Siyuan-Zhu/recurring-lifecycle-and-churn-pipeline",
    "https://github.com/Siyuan-Zhu/oecd-aid-finance-llm-classifier",
    "https://github.com/Siyuan-Zhu/term-deposit-conversion-roi",
)

# Offline allowlist only. Live reachability and visibility need separate API evidence.
ALLOWED_SIYUAN_ZHU_REPOS: frozenset[str] = frozenset(
    {
        "Siyuan-Zhu/Siyuan-Zhu",
        "Siyuan-Zhu/readme-live-sandbox",
        "Siyuan-Zhu/multi-agent-fpa-workspace",
        "Siyuan-Zhu/finance-analytics-warehouse",
        "Siyuan-Zhu/recurring-lifecycle-and-churn-pipeline",
        "Siyuan-Zhu/oecd-aid-finance-llm-classifier",
        "Siyuan-Zhu/term-deposit-conversion-roi",
        "Siyuan-Zhu/computer-vision-document-motion-convnet",
        "Siyuan-Zhu/lowess-boosting-from-scratch",
        "Siyuan-Zhu/heart-disease-risk-model",
    }
)

SIYUAN_ZHU_REPO_URL_RE = re.compile(
    r"""https://github\.com/(Siyuan-Zhu/[A-Za-z0-9_.-]+)"""
)

DEAD_REPO_SLUGS: tuple[str, ...] = (
    "gl-variance-and-run-rate-pipeline",
)

RELATIVE_ASSET_RE = re.compile(
    r"""(?:srcset|src|href)=["'](\.\.?/assets/[^"'\s>]+)["']|\((\.\.?/assets/[^)\s]+)\)"""
)

ISSUE_NEW_URL_RE = re.compile(
    r"""https://github\.com/([^/\s"'>)]+/[^/\s"'>)]+)/issues/new\?([^\s"'>)]+)"""
)


def _load_active_game_lock(repo_root: Path, errors: list[str]) -> tuple[int, int]:
    """Load the active Connect 4 turn and round from the same state snapshot."""
    c4_path = repo_root / "data" / "connect4_state.json"
    game_path = repo_root / "data" / "game_state.json"
    state_path = c4_path if c4_path.exists() else game_path
    if not state_path.exists():
        errors.append("Missing game state file: expected data/connect4_state.json or data/game_state.json")
        return 1, 1
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return int(payload.get("turn", 1)), int(payload.get("round_id", 1))
    except Exception as exc:
        errors.append(f"Failed to parse {state_path.name}: {exc}")
        return 1, 1


def _validate_svg_and_webp_assets(repo_root: Path, errors: list[str]) -> None:
    """Validate XML well-formedness for SVGs and RIFF/WEBP magic bytes for WebP assets."""
    for svg_rel in ("assets/daily_pulse.svg", "assets/game_board.svg"):
        svg_path = repo_root / svg_rel
        if not svg_path.exists():
            errors.append(f"Missing required SVG asset: {svg_rel}")
            continue
        if svg_path.stat().st_size <= 0:
            errors.append(f"Empty SVG asset: {svg_rel}")
            continue
        try:
            ET.fromstring(svg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Malformed XML in {svg_rel}: {exc}")

    for webp_rel in ("assets/daily_pulse_dark.webp", "assets/daily_pulse_light.webp"):
        webp_path = repo_root / webp_rel
        if not webp_path.exists():
            errors.append(f"Missing required WebP asset: {webp_rel}")
            continue
        raw = webp_path.read_bytes()
        if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
            errors.append(f"Invalid RIFF/WEBP header in {webp_rel}")


def _validate_issue_links_in_markdown(
    rel_label: str,
    text: str,
    expected_repo_slug: str,
    expected_turn: int,
    expected_round: int,
    errors: list[str],
    expected_game: dict | None = None,
) -> int:
    """Verify exactly the controls available for this canonical game snapshot."""
    matches = ISSUE_NEW_URL_RE.findall(text)
    found_titles: list[str] = []
    for slug, raw_query in matches:
        if slug != expected_repo_slug:
            errors.append(
                f"{rel_label}: issues/new link targets slug '{slug}' instead of expected '{expected_repo_slug}'"
            )
        unescaped_query = html.unescape(raw_query)
        parsed_qs = urllib.parse.parse_qs(unescaped_query)
        titles = parsed_qs.get("title", [])
        if titles:
            found_titles.append(titles[0])

    game = expected_game or {}
    completed = str(game.get("status", "ACTIVE")).upper() == "COMPLETED"
    board = game.get("board")
    available = range(1, 8)
    if isinstance(board, list) and len(board) == 6 and all(isinstance(row, list) and len(row) == 7 for row in board) and not completed:
        available = [col + 1 for col in range(7) if any(row[col] in (0, ".", "", None, "0") for row in board)]
    expected = {f"pulse:vote:{mood}" for mood in ("spot_on", "too_sunny", "too_heavy")}
    expected.update(f"game:move:connect4:col:{col}:turn={expected_turn}:round={expected_round}" for col in available)
    mode = str(game.get("mode", "VS_BOT")).upper()
    if mode == "PVP_DUEL":
        expected.add(f"game:mode:connect4:new_vs_bot:turn={expected_turn}:round={expected_round}")
    elif expected_turn <= 2 or completed:
        expected.add(f"game:mode:connect4:new_pvp:turn={expected_turn}:round={expected_round}")
    if "<sub>Game paused." in text:
        expected = {title for title in expected if title.startswith("pulse:vote:")}
    for title in sorted(expected):
        if title not in found_titles:
            errors.append(f"{rel_label}: missing or turn/round-mismatched issue link '{title}'")
    for title in found_titles:
        if title not in expected:
            errors.append(f"{rel_label}: unavailable or turn/round-mismatched issue link '{title}'")
    if len(matches) != len(expected):
        errors.append(f"{rel_label}: expected {len(expected)} available interactive issue links, found {len(matches)}")

    return len(matches)


def validate_release_bundle(
    repo_root: Path,
    expected_repo_slug: str = "Siyuan-Zhu/Siyuan-Zhu",
    *,
    require_publication_manifest: bool = False,
) -> dict:
    """Validate README.md, PROJECTS.md, variants/*.md, and assets/ for release readiness."""
    repo_root = Path(repo_root).resolve()
    errors: list[str] = []
    checked_files: list[str] = []
    resolved_assets: set[str] = set()
    publication = None
    if require_publication_manifest:
        try:
            publication = verify_publication_bundle(repo_root, require_manifest=True, require_current_pulse=True)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            errors.append(f"Publication verification failed: {exc}")

    expected_turn, expected_round = _load_active_game_lock(repo_root, errors)
    game_path = repo_root / "data" / "connect4_state.json"
    if not game_path.exists():
        game_path = repo_root / "data" / "game_state.json"
    try:
        expected_game = json.loads(game_path.read_text(encoding="utf-8"))
        if not isinstance(expected_game, dict):
            expected_game = {}
    except (OSError, ValueError):
        expected_game = {}
    _validate_svg_and_webp_assets(repo_root, errors)

    readme_path = repo_root / "README.md"
    projects_path = repo_root / "PROJECTS.md"
    variants_dir = repo_root / "variants"

    target_md_files: list[Path] = [readme_path, projects_path]
    for vf in EXPECTED_VARIANT_FILENAMES:
        target_md_files.append(variants_dir / vf)
    if variants_dir.is_dir():
        for extra_vf in sorted(variants_dir.glob("*.md")):
            if extra_vf not in target_md_files:
                target_md_files.append(extra_vf)

    readme_issue_links_count = 0

    for md_path in target_md_files:
        rel_label = str(md_path.relative_to(repo_root)) if md_path.is_relative_to(repo_root) else md_path.name
        if not md_path.exists():
            errors.append(f"Missing markdown file: {rel_label}")
            continue
        if md_path.stat().st_size <= 0:
            errors.append(f"Empty markdown file: {rel_label}")
            continue
        checked_files.append(rel_label)
        content = md_path.read_text(encoding="utf-8")

        # 1. Dead repository link check
        for dead_slug in DEAD_REPO_SLUGS:
            if dead_slug in content:
                errors.append(f"{rel_label}: contains dead repository reference '{dead_slug}'")

        # 2. Relative asset link check
        for m in RELATIVE_ASSET_RE.finditer(content):
            raw_asset_ref = m.group(1) or m.group(2)
            if not raw_asset_ref:
                continue
            clean_ref = raw_asset_ref.split("?", 1)[0].split("#", 1)[0]
            resolved_path = (md_path.parent / clean_ref).resolve()
            if not resolved_path.exists():
                errors.append(f"{rel_label}: broken relative asset link '{raw_asset_ref}'")
            elif resolved_path.stat().st_size <= 0:
                errors.append(f"{rel_label}: empty relative asset target '{raw_asset_ref}'")
            else:
                resolved_assets.add(str(resolved_path.relative_to(repo_root)))

        # 3. Interactive issue links check for README.md and variants/*.md
        if md_path == readme_path or md_path.parent == variants_dir:
            cnt = _validate_issue_links_in_markdown(
                rel_label=rel_label,
                text=content,
                expected_repo_slug=expected_repo_slug,
                expected_turn=expected_turn,
                expected_round=expected_round,
                errors=errors,
                expected_game=expected_game,
            )
            if md_path == readme_path:
                readme_issue_links_count = cnt

        # 4. Flagship project URLs check for PROJECTS.md and variants/*.md
        if md_path == projects_path or md_path.parent == variants_dir:
            for flagship_url in EXPECTED_FLAGSHIP_URLS:
                if flagship_url not in content:
                    errors.append(f"{rel_label}: missing flagship repository URL '{flagship_url}'")
            for repo_match in SIYUAN_ZHU_REPO_URL_RE.findall(content):
                if repo_match not in ALLOWED_SIYUAN_ZHU_REPOS and repo_match != expected_repo_slug:
                    errors.append(
                        f"{rel_label}: repository URL outside the reviewed allowlist 'https://github.com/{repo_match}'"
                    )

    status_label = "PASS" if len(errors) == 0 else "FAIL"
    markdown_summary = (
        f"### Release Link & Asset Audit: **{status_label}**\n"
        f"- **Expected Repo Slug:** `{expected_repo_slug}`\n"
        f"- **Active Connect 4 Turn:** `{expected_turn}`\n"
        f"- **Checked Markdown Files:** `{len(checked_files)}`\n"
        f"- **Interactive Issue Links (`README.md`):** `{readme_issue_links_count}`\n"
        f"- **Resolved Assets:** `{len(resolved_assets)}`\n"
        f"- **Errors:** `{len(errors)}`"
    )

    return {
        "ok": len(errors) == 0,
        "expected_repo_slug": expected_repo_slug,
        "external_links_verified": False,
        "publication_manifest_required": require_publication_manifest,
        "publication_manifest_verified": publication is not None,
        "publication_id": publication.get("publication_id") if publication else None,
        "expected_turn": expected_turn,
        "expected_round": expected_round,
        "checked_files_count": len(checked_files),
        "checked_files": checked_files,
        "resolved_assets": sorted(resolved_assets),
        "issue_links_count": readme_issue_links_count,
        "markdown_summary": markdown_summary,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate README.md, PROJECTS.md, variants/*.md, and assets/ for release readiness."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory.",
    )
    parser.add_argument(
        "--expected-repo-slug",
        type=str,
        default="Siyuan-Zhu/Siyuan-Zhu",
        help="Expected GitHub repository slug for interactive issue links.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit validation report as JSON.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Emit validation report as a Markdown summary.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path to write the JSON validation report.",
    )
    parser.add_argument("--require-publication-manifest", action="store_true", help="Require a coherent manifest, all theme/SVG/archive bytes, feed and current pulse before release.")
    args = parser.parse_args(argv)
    report = validate_release_bundle(
        repo_root=args.repo_root,
        expected_repo_slug=args.expected_repo_slug,
        require_publication_manifest=args.require_publication_manifest,
    )
    if args.output_json is not None:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, indent=2))
    elif args.markdown:
        print(report["markdown_summary"])
        for err in report["errors"]:
            print(f"  - ERROR: {err}", file=sys.stderr)
    else:
        status_str = "PASS" if report["ok"] else "FAIL"
        print(
            f"[Release Validator: {status_str}] "
            f"files={report['checked_files_count']} "
            f"issue_links={report['issue_links_count']} "
            f"turn={report['expected_turn']} "
            f"slug={report['expected_repo_slug']}"
        )
        for err in report["errors"]:
            print(f"  - ERROR: {err}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
