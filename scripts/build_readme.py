#!/usr/bin/env python3
"""Assemble the Minimalist `Siyuan-Zhu/Siyuan-Zhu` Profile README and Role Variants.

Design principles (aligned with Ticket #14 & Ticket #17):
  1. Compact approved introduction and subordinate public-feed attribution
     with zero duplicate sidebar chrome or unbuilt website links.
  2. GitHub-themed daily WebP artwork paired with a concise `<= 50`-character
     caption, platform-level photo credit (or `<= 50`-character SVG explanation), and 1-click Mood poll.
  3. Reactive, rotatable Community Game section (`Connect 4`, `Word Guess`, `20 Questions`) rendered by
     `src/game_engine.py` (`render_readme_game_section`), with turn-locked 1-click move links,
     collapsible `<details>` recent moves log, and `##### How it works`.
"""

from __future__ import annotations

import argparse
import hashlib
from html import escape
from urllib.parse import urlsplit, parse_qsl, urlencode
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.game_engine import (
    render_board_svg,
    render_readme_game_section,
)
from src.editorial_typography import OPENING_NAME, OPENING_PERSPECTIVE, OPENING_GAME, picture, write_typography
from src.state_store import get_state_store


FLAGSHIP_PROJECTS = [
    {
        "id": "gl-variance",
        "name": "finance-analytics-warehouse",
        "url": "https://github.com/Siyuan-Zhu/finance-analytics-warehouse",
        "domain": "Financial Data Engineering · DuckDB & dbt Star Schema",
        "summary": (
            "Runs 34 monthly closes for a fictional five-store retailer (100% synthetic NIU data), "
            "deduplicating multi-export payables (13.32M to 7.33M NIU), isolating year-end stock-count "
            "shrinkage distortions, and passing 116 automated SQL and Excel checks per close."
        ),
        "stack": "Python · DuckDB · dbt · SQL · openpyxl",
    },
    {
        "id": "fpa-workspace",
        "name": "multi-agent-fpa-workspace",
        "url": "https://github.com/Siyuan-Zhu/multi-agent-fpa-workspace",
        "domain": "Agentic Systems · Governed FP&A Automation",
        "summary": (
            "Three-tier context architecture and four-stage rolling budget pipeline where AI agents "
            "collaborate with deterministic Python rules while preserving 100% of human-locked analyst "
            "overrides and ring-fenced reserve boundaries across re-runs."
        ),
        "stack": "Python · Multi-Agent Context Contracts · Architecture Decision Records · openpyxl",
    },
    {
        "id": "lifecycle-churn",
        "name": "recurring-lifecycle-and-churn-pipeline",
        "url": "https://github.com/Siyuan-Zhu/recurring-lifecycle-and-churn-pipeline",
        "domain": "Entity Resolution · Multi-Rail Payment Telemetry",
        "summary": (
            "Reconciles card, ACH, and subledger payment rails across UTC/Eastern date boundaries, "
            "links corporate/household accounts via Union-Find entity resolution, and suppresses "
            "75.0% of false-positive churn alerts while enforcing zero-leakage role redaction."
        ),
        "stack": "Python · SQLite · Union-Find Disjoint Set · Role-Redacted Briefs",
    },
    {
        "id": "oecd-classifier",
        "name": "oecd-aid-finance-llm-classifier",
        "url": "https://github.com/Siyuan-Zhu/oecd-aid-finance-llm-classifier",
        "domain": "Applied NLP · Hierarchical Taxonomy Calibration",
        "summary": (
            "Benchmarks flat lexical, top-down cascade, and retrieval-augmented hierarchical classification "
            "across 960 multilateral development records and 24 OECD CRS 5-digit purpose codes—reaching "
            "0.935 Macro-F1 and 100% parent-child taxonomy consistency."
        ),
        "stack": "Python · Retrieval-Augmented Few-Shot · Confidence Routing · OECD CRS",
    },
    {
        "id": "conversion-roi",
        "name": "term-deposit-conversion-roi",
        "url": "https://github.com/Siyuan-Zhu/term-deposit-conversion-roi",
        "domain": "Decision Science · Calibrated Unit Economics",
        "summary": (
            "Strips post-call duration leakage via a schema guard, calibrates pre-call propensity "
            "probabilities (ECE 0.0124), and optimizes outbound call thresholds against explicit unit "
            "economics—capturing 89.0% of conversions in 43.3% of calls (+62.6% Net Value Index)."
        ),
        "stack": "Python · Isotonic Calibration · Cost-Sensitive Threshold Optimization",
    },
]

VARIANT_ORDERS = {
    "default": ["fpa-workspace", "gl-variance", "lifecycle-churn", "oecd-classifier", "conversion-roi"],
    "ai-engineering": ["fpa-workspace", "oecd-classifier", "lifecycle-churn", "gl-variance", "conversion-roi"],
    "fpa-analytics": ["gl-variance", "fpa-workspace", "lifecycle-churn", "conversion-roi", "oecd-classifier"],
    "bi-data": ["lifecycle-churn", "gl-variance", "conversion-roi", "oecd-classifier", "fpa-workspace"],
    "A-minimal-editorial": ["fpa-workspace", "gl-variance", "lifecycle-churn", "oecd-classifier", "conversion-roi"],
    "B-interactive-arcade": ["fpa-workspace", "oecd-classifier", "lifecycle-churn", "conversion-roi", "gl-variance"],
    "C-hardware-bento": ["gl-variance", "fpa-workspace", "lifecycle-churn", "oecd-classifier", "conversion-roi"],
    "D-live-telemetry": ["lifecycle-churn", "fpa-workspace", "gl-variance", "conversion-roi", "oecd-classifier"],
    "E-hybrid-flagship": ["fpa-workspace", "gl-variance", "lifecycle-churn", "oecd-classifier", "conversion-roi"],
}

VARIANT_FILES = {
    "A": "README.A-minimal-editorial.md",
    "B": "README.B-interactive-arcade.md",
    "C": "README.C-hardware-bento.md",
    "D": "README.D-live-telemetry.md",
    "E": "README.E-hybrid-flagship.md",
}

CONCISE_WORD_PHRASES = {
    "CONVERGENCE": "Separate threads locking into shared form.",
    "ANCHORED": "Steady execution under shifting macro currents.",
    "EQUILIBRIUM": "Balanced tension across systems and markets.",
    "RESONANCE": "Quiet craft aligning with broader signals.",
    "CALIBRATION": "Measuring twice before committing the cut.",
    "MOMENTUM": "Compounding small wins into durable velocity.",
}


def _concise_phrase(word: str, raw_phrase: str) -> str:
    """Guarantee a <= 50 character explanation phrase for the minimalist centerpiece."""
    w_upper = (word or "").strip().upper()
    if raw_phrase and len(raw_phrase.strip()) <= 50:
        return raw_phrase.strip()
    if w_upper in CONCISE_WORD_PHRASES:
        return CONCISE_WORD_PHRASES[w_upper]
    cleaned = (raw_phrase or "Quiet synthesis across code and markets.").strip()
    if len(cleaned) <= 50:
        return cleaned
    return cleaned[:47].rstrip(" ,;:-") + "..."


def _visual_credit_html(visual_decision: dict, published_pulse: dict | None = None) -> str:
    """Describe the actual medium and supplied attribution without inventing it."""
    published = published_pulse or {}
    attribution = published.get("attribution") or visual_decision.get("attribution") or {}
    mode = str(published.get("visual_medium") or visual_decision.get("selected_mode") or "").upper()

    def unsplash_url(value):
        value = str(value or "").strip()
        try:
            parsed = urlsplit(value)
            if parsed.scheme == "https" and parsed.netloc.lower() in {"unsplash.com", "www.unsplash.com"}:
                query = dict(parse_qsl(parsed.query, keep_blank_values=True))
                query.update(utm_source="siyuan_zhu_profile", utm_medium="referral")
                return parsed._replace(query=urlencode(query)).geturl()
        except ValueError:
            pass
        return ""

    def link(url, label):
        return f'<a href="{escape(url, quote=True)}">{escape(label)}</a>'

    if mode in {"CURATED_PHOTO", "CURATED_PHOTO_LIBRARY", "PHOTOGRAPHY", "UNSPLASH_PHOTO"}:
        source = next((url for value in (
            attribution.get("unsplash_url"), published.get("artwork_url"),
            visual_decision.get("source_page_url"), visual_decision.get("credit_url")
        ) if (url := unsplash_url(value))), unsplash_url("https://unsplash.com"))
        name = str(published.get("unsplash_photographer") or attribution.get("photographer_name") or "").strip()
        profile = unsplash_url(published.get("unsplash_profile_url") or attribution.get("photographer_url"))
        if profile and not urlsplit(profile).path.startswith("/@"):
            profile = ""
        if profile and (not name or name in {"Unsplash Contributor", "Bespoke Geometric Study"}):
            return f'Photo by {link(profile, "Photographer")} on {link(source, "Unsplash")}'
        if name and name not in {"Unsplash Contributor", "Bespoke Geometric Study"}:
            photographer = link(profile, name) if profile else escape(name)
            return f'Photo by {photographer} on {link(source, "Unsplash")}'
        return link(source, "Photo via Unsplash")
    flux_model = str(published.get("flux_model") or attribution.get("flux_model") or "").lower()
    if mode == "FLUX_3" or (mode == "FLUX_PRO" and "flux-3" in flux_model):
        return "Visual · FLUX 3 editorial synthesis"
    if mode in {"FLUX_PRO", "FLUX_FAL", "FLUX_1_PRO"}:
        return "Visual · FLUX editorial synthesis"
    if mode == "IMAGEN_3":
        return "Visual · Imagen 3 editorial synthesis"
    if mode == "GEMINI_IMAGE":
        return "Visual · Gemini image synthesis"
    if mode in {"BESPOKE_SVG", "SVG_FALLBACK", "GENERATIVE_SVG", "ARCHITECTURAL_BLUEPRINT_SVG"}:
        return "SVG · Procedural editorial composition"
    if mode in {"FINE_ART", "ILLUSTRATION"}:
        return "Visual · " + ("Fine art" if mode == "FINE_ART" else "Illustration")
    return "Visual · Source unverified"


def _render_selected_work_section(variant: str) -> str:
    """Render a compact role-ordered Selected Work section for non-default README variants."""
    if variant == "default" or variant not in VARIANT_ORDERS:
        return ""
    proj_by_id = {p["id"]: p for p in FLAGSHIP_PROJECTS}
    ordered_ids = VARIANT_ORDERS[variant]
    lines = [
        "",
        "---",
        "",
        f"#### Selected Work (Role-Tailored) · `{variant}`",
        "",
    ]
    for idx, pid in enumerate(ordered_ids, start=1):
        proj = proj_by_id.get(pid)
        if not proj:
            continue
        lines.append(
            f"{idx}. **[{proj['name']}]({proj['url']})** — _{proj['domain']}_  "
        )
        lines.append(f"   {proj['summary']}  ")
        lines.append(f"   <sub>Stack: {proj['stack']}</sub>")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def resolve_repo_slug(repo_slug: str | None = None) -> str:
    """Resolve repository slug in precedence order:
    1. Explicit `repo_slug` argument (if non-empty)
    2. `TARGET_REPO_SLUG` environment variable (if non-empty)
    3. `GITHUB_REPOSITORY` environment variable (if non-empty)
    4. Default `"Siyuan-Zhu/Siyuan-Zhu"`
    """
    for candidate in (
        repo_slug,
        os.environ.get("TARGET_REPO_SLUG"),
        os.environ.get("GITHUB_REPOSITORY"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return "Siyuan-Zhu/Siyuan-Zhu"


def render_profile_markdown(
    repo_root: Path,
    variant: str = "default",
    repo_slug: str | None = None,
    asset_prefix: str = "./assets",
    write_assets: bool = True,
    state_store=None,
    _canonical_projection=None,
) -> str:
    if variant not in VARIANT_ORDERS:
        raise ValueError(f"Unknown README variant: {variant}")
    resolved_slug = resolve_repo_slug(repo_slug)
    from src.zeitgeist.publication import verify_publication_bundle
    publication = verify_publication_bundle(repo_root, require_manifest=False)
    store = state_store if state_store is not None else get_state_store(repo_root=repo_root)
    canonical = bool(getattr(store, "canonical_only", False))
    if canonical:
        # A missing checkout mirror is not permission to run the publishing
        # pipeline. Validate authority first, then project it without inference.
        projections = _canonical_projection if _canonical_projection is not None else store.canonical_projections(write_mirrors=write_assets)
        store_pulse = projections["zeitgeist_state.json"]
        local_zstate = store_pulse
        game_state = projections["game_state.json"]
        elo_ledger = projections["hall_of_records.json"]
    else:
        state_path = repo_root / "data" / "zeitgeist_state.json"
        if not state_path.exists():
            from src.zeitgeist.adk_pipeline import run_daily_zeitgeist

            run_daily_zeitgeist(repo_root=repo_root)
        local_zstate = json.loads(state_path.read_text(encoding="utf-8"))
        store_pulse = store.load_pulse_state()
        game_state = store.load_game_state()
        elo_ledger = store.load_elo_ledger()
    analysis = store_pulse.get("analysis") or local_zstate.get("analysis", {})
    visual = store_pulse.get("visual_decision") or local_zstate.get("visual_decision", {})
    tally = (
        store_pulse.get("visitor_mood_tally")
        or store_pulse.get("mood_votes")
        or local_zstate.get("visitor_mood_tally")
        or local_zstate.get("mood_votes")
        or {"spot_on": 20, "too_sunny": 3, "too_heavy": 4}
    )

    if write_assets:
        write_typography(repo_root)
        svg_board = render_board_svg(game_state, elo_ledger=elo_ledger)
        svg_path = repo_root / "assets" / "game_board.svg"
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        svg_path.write_text(svg_board, encoding="utf-8")

    word = str(
        store_pulse.get("word")
        or analysis.get("word_of_the_day")
        or local_zstate.get("word")
        or "CONVERGENCE"
    )
    raw_gloss = str(
        store_pulse.get("gloss")
        or analysis.get("explanation_phrase")
        or local_zstate.get("gloss")
        or ""
    )
    phrase = _concise_phrase(word, raw_gloss)
    published_metadata = store_pulse.get("published_pulse") or local_zstate.get("published_pulse") or store_pulse
    credit_html = _visual_credit_html(visual, published_metadata)
    pulse_date = str(
        store_pulse.get("date")
        or store_pulse.get("date_iso")
        or local_zstate.get("date")
        or local_zstate.get("date_iso")
        or "2026-10-27"
    )[:10]
    if publication and (
        publication["word"].casefold() != word.casefold() or publication["date"] != pulse_date
    ):
        raise ValueError("README state does not match the generated publication; regenerate before publishing")
    publication_id = publication["publication_id"] if publication else ""
    pulse_sha8 = hashlib.sha256(f"{pulse_date}:{word}:{phrase}:{tally}:{publication_id}".encode("utf-8")).hexdigest()[:8]
    pulse_v = f"{pulse_date}-{pulse_sha8}"

    game_section = render_readme_game_section(
        game_state,
        elo_ledger=elo_ledger,
        enabled=os.environ.get("COMMUNITY_GAME_ENABLED","true") == "true",
        repo_slug=resolved_slug,
        asset_prefix=asset_prefix,
    )
    vote_base = f"https://github.com/{resolved_slug}/issues/new"
    vote_body = "body=Click+%27Submit+new+issue%27+to+cast+your+mood+vote."
    spot_on_cnt = int(tally.get("spot_on", tally.get("resonant", 20)))
    too_sunny_cnt = int(tally.get("too_sunny", tally.get("electric", 3)))
    too_heavy_cnt = int(tally.get("too_heavy", tally.get("reflective", 4)))
    selected_work_section = _render_selected_work_section(variant)

    return f"""<div align="center">

<h2>{picture("editorial_opening", asset_prefix, OPENING_NAME + " — " + OPENING_PERSPECTIVE + " " + OPENING_GAME)}</h2>

{picture("daily_pulse", asset_prefix, word + " — " + phrase, width="100%", version=pulse_v)}

</div>

<p align="left"><sub>{credit_html} · <a href="{("../" if asset_prefix.startswith("../") else "./")}docs/resonance.md">RESONANCE</a>: <a href="{vote_base}?title=pulse:vote:spot_on&amp;{vote_body}">Spot On {spot_on_cnt}</a> / <a href="{vote_base}?title=pulse:vote:too_sunny&amp;{vote_body}">Too Sunny {too_sunny_cnt}</a> / <a href="{vote_base}?title=pulse:vote:too_heavy&amp;{vote_body}">Too Heavy {too_heavy_cnt}</a></sub></p>

---

{game_section}
{selected_work_section}""".rstrip() + "\n"


def check_all_readmes(repo_root: Path, repo_slug: str | None = None, *, state_store=None) -> bool:
    """Return True if README.md and all variants match current state without writing files."""
    expected_files: list[tuple[Path, str]] = [
        (
            repo_root / "README.md",
            render_profile_markdown(
                repo_root,
                variant="default",
                repo_slug=repo_slug,
                asset_prefix="./assets",
                write_assets=False, state_store=state_store,
            ),
        )
    ]
    variants_dir = repo_root / "variants"
    for var_key in ("ai-engineering", "fpa-analytics", "bi-data"):
        expected_files.append(
            (
                variants_dir / f"README-{var_key}.md",
                render_profile_markdown(
                    repo_root,
                    variant=var_key,
                    repo_slug=repo_slug,
                    asset_prefix="../assets",
                    write_assets=False, state_store=state_store,
                ),
            )
        )
    for _short_key, filename in VARIANT_FILES.items():
        order_key = filename.replace("README.", "").replace(".md", "")
        expected_files.append(
            (
                variants_dir / filename,
                render_profile_markdown(
                    repo_root,
                    variant=order_key,
                    repo_slug=repo_slug,
                    asset_prefix="../assets",
                    write_assets=False, state_store=state_store,
                ),
            )
        )

    for path, expected_content in expected_files:
        if not path.exists():
            return False
        if path.read_text(encoding="utf-8") != expected_content:
            return False
    return True


def build_all_readmes(repo_root: Path, repo_slug: str | None = None, *, state_store=None) -> dict[str, Path]:
    state_store = state_store if state_store is not None else get_state_store(repo_root=repo_root)
    canonical = bool(getattr(state_store, "canonical_only", False))
    # Read authority once before rendering any file. Every variant must describe
    # the same verified snapshot, without repeated network reads or JSON fallback.
    projection = state_store.canonical_projections(write_mirrors=True) if canonical else None
    if not canonical and not (repo_root / "data" / "zeitgeist_state.json").exists():
        from src.zeitgeist.adk_pipeline import run_daily_zeitgeist

        run_daily_zeitgeist(repo_root=repo_root)
    main_md = render_profile_markdown(
        repo_root,
        variant="default",
        repo_slug=repo_slug,
        asset_prefix="./assets", state_store=state_store, _canonical_projection=projection,
    )
    readme_path = repo_root / "README.md"
    readme_path.write_text(main_md, encoding="utf-8")

    variants_dir = repo_root / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {"default": readme_path}
    for var_key in ("ai-engineering", "fpa-analytics", "bi-data"):
        v_md = render_profile_markdown(
            repo_root,
            variant=var_key,
            repo_slug=repo_slug,
            asset_prefix="../assets", state_store=state_store, write_assets=False, _canonical_projection=projection,
        )
        out_p = variants_dir / f"README-{var_key}.md"
        out_p.write_text(v_md, encoding="utf-8")
        written[var_key] = out_p

    for short_key, filename in VARIANT_FILES.items():
        order_key = filename.replace("README.", "").replace(".md", "")
        v_md = render_profile_markdown(
            repo_root,
            variant=order_key,
            repo_slug=repo_slug,
            asset_prefix="../assets", state_store=state_store, write_assets=False, _canonical_projection=projection,
        )
        out_p = variants_dir / filename
        out_p.write_text(v_md, encoding="utf-8")
        written[short_key] = out_p

    return written


build_all = build_all_readmes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify profile README.md and role variants.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify README.md and variants/*.md are synchronized with current state.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root directory.",
    )
    parser.add_argument(
        "--repo-slug",
        type=str,
        default=None,
        help="Target GitHub repository slug (e.g., Siyuan-Zhu/Siyuan-Zhu).",
    )
    args = parser.parse_args(argv)
    if args.check:
        ok = check_all_readmes(args.repo_root, repo_slug=args.repo_slug)
        if not ok:
            print("README.md or variants/*.md are out of sync with current state.", file=sys.stderr)
            return 1
        print("All README and variant files are synchronized.")
        return 0
    build_all_readmes(args.repo_root, repo_slug=args.repo_slug)
    print("Built minimalist README.md, 3 role-emphasis variants, and 5 architectural variants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
