#!/usr/bin/env python3
"""Build an offline, history-free public tree from a pinned source and accepted plan.

Never initializes Git, contacts services, applies canonical state or enables writers.
The private inventory belongs outside the exported directory.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
from functools import lru_cache
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import promote_release_state as promotion
from scripts.build_readme import build_all_readmes, check_all_readmes
from scripts.validate_release_links import validate_release_bundle
from src.editorial_typography import ASSETS as TYPOGRAPHY_ASSETS
from src.state_store import JsonFileStateStore
from src.zeitgeist.publication import verify_publication_bundle
from src.zeitgeist.visual_router import update_svg_mood_vote_bar

PUBLIC_SLUG = "Siyuan-Zhu/Siyuan-Zhu"
STATIC_FILES = {
    ".gitignore",
    "docs/resonance.md",
    "requirements.txt",
    "requirements-ci.lock",
    "config/orbit_sources.json",
    "PROJECTS.md",
    "firebase.json",
    "firestore.rules",
    "firestore.indexes.json",
    "deploy/cloud_run_service.py",
    "deploy/Dockerfile",
    "assets/fonts/CormorantGaramond.ttf",
    "assets/fonts/CormorantGaramond-Italic.ttf",
    "assets/fonts/IBMPlexMono-Regular.ttf",
    "assets/fonts/OFL-CormorantGaramond.txt",
    "assets/fonts/OFL-IBMPlexMono.txt",
    "assets/fonts/README.md",
}
WORKFLOWS = {
    ".github/workflows/ci.yml",
    ".github/workflows/community_game_move.yml",
    ".github/workflows/visitor_workflow_feedback.yml",
    ".github/workflows/daily_zeitgeist.yml",
    ".github/workflows/mobile_interject.yml",
    ".github/workflows/sealed_production_secret_export.yml",
    ".github/workflows/staging_zeitgeist_dryrun.yml",
}
# Public DTO fields only. Source rows, prompts, owner notes, account metadata,
# billing telemetry and operational receipts are deliberately outside this DTO.
EDITORIAL_FIELDS = frozenset("""
publication_id schema_version state_version date date_iso word gloss commissioned_thesis
visual_medium curatorial_archetype text_quadrant quietest_quadrant window_hours
headline summary dominant_category dominant_signal_type orbit_weight_ratio
24h_shift top_orbit_themes top_macro_themes quote quote_attribution palette_hex
artwork_title artwork_artist artwork_source artwork_url image_url fallback_webp_path image_data_uri
unsplash_photographer unsplash_profile_url flux_model credit_url credit_label
cluster_keywords orbit_contribution_pct macro_contribution_pct critic_scores
visual_tier_used orbit_sources_matched runner_up_clusters
published_at updated_at mood_votes visitor_mood_tally
""".split())
ANALYSIS_FIELDS = frozenset("""date_iso composite_score valence arousal dominance
 epistemic_tension word_of_the_day explanation_phrase visual_concrete_anchor
 is_builder_systems_day dominant_keywords""".split())
VISUAL_FIELDS = frozenset(
    """selected_mode word_of_the_day explanation_phrase composite_score
 image_url credit_label credit_url semantic_alignment minimalist_restraint safety_pass""".split()
)
ATTRIBUTION_FIELDS = frozenset("""photographer_name photographer_url unsplash_url
 photo_id source model exif_summary""".split())
PUBLIC_JSON = {
    "data/game_state.json",
    "data/zeitgeist_state.json",
    "data/hall_of_records.json",
    "data/bot_brain.json",
    "data/pulse_feed.json",
    "data/orbit_roster.json",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.PIPE
    )


@lru_cache(maxsize=512)
def _git_blob(root, source_sha, path):
    return _git(root, "show", source_sha + ":" + path)


def selected_source(path):
    """The closed source selectors; generated data/artifacts have separate lists."""
    return (
        path in STATIC_FILES | WORKFLOWS
        or path.endswith(".py")
        and (
            path.startswith("src/")
            or path.startswith("tests/")
            or path.startswith("scripts/")
            and path.count("/") == 1
        )
    )


def source_inventory(root, source_sha):
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("Exact source commit required")
    if _git(root, "rev-parse", "HEAD").decode().strip() != source_sha:
        raise ValueError("Run from the pinned source checkout")
    entries = {}
    for row in _git(root, "ls-tree", "-rz", source_sha).split(b"\0"):
        if not row:
            continue
        header, name = row.split(b"\t", 1)
        mode, kind, _object = header.decode().split()
        path = name.decode("utf-8")
        selected = selected_source(path)
        if not selected:
            continue
        if kind != "blob" or mode not in {"100644", "100755"}:
            raise ValueError("Selected source must contain regular files only")
        if PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts:
            raise ValueError("Invalid source path")
        raw = _git_blob(root, source_sha, path)
        # Imported serializers/renderers must match the pinned tree as well.
        local = root / path
        if local.is_symlink() or not local.is_file() or local.read_bytes() != raw:
            raise ValueError("Selected source has uncommitted changes")
        entries[path] = {"sha256": sha(raw), "mode": mode}
    if not STATIC_FILES | WORKFLOWS <= entries.keys():
        raise ValueError("Required runtime, workflow or font files missing")
    return dict(sorted(entries.items()))


def _select(value, fields):
    if not isinstance(value, dict):
        raise ValueError("Invalid public projection source")
    return {key: copy.deepcopy(value[key]) for key in sorted(fields & value.keys())}


def _public_republication(value):
    fields = {'kind', 'original_publication_id', 'original_publication_date',
              'original_published_at', 'new_inference_performed',
              'artwork_reused_byte_exact'}
    record = _select(value, fields)
    if (set(record) != fields or record['kind'] != 'initial_release_republication'
            or not isinstance(record['original_publication_id'], str)
            or not re.fullmatch(r'[a-f0-9]{20}', record['original_publication_id'])
            or record['new_inference_performed'] is not False
            or record['artwork_reused_byte_exact'] is not True):
        raise ValueError('Invalid public republication provenance')
    from datetime import date, datetime, timedelta
    original_date = record['original_publication_date']
    if not isinstance(original_date, str) or date.fromisoformat(original_date).isoformat() != original_date:
        raise ValueError('Invalid original publication date')
    timestamp = record['original_published_at']
    if timestamp is not None:
        if not isinstance(timestamp, str) or len(timestamp) > 64:
            raise ValueError('Invalid original publication timestamp')
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        if parsed.utcoffset() != timedelta(0) or parsed.date().isoformat() != original_date:
            raise ValueError('Original publication timestamp must match its UTC date')
    return record


def _editorial(value):
    public = _select(value, EDITORIAL_FIELDS)
    if public.get("image_data_uri"):
        raise ValueError("Public editorial JSON cannot contain inline image payloads")
    if "attribution" in value:
        public["attribution"] = _select(value["attribution"], ATTRIBUTION_FIELDS)
    if "release_republication" in value:
        public["release_republication"] = _public_republication(value["release_republication"])
    return public


def public_projections(envelope):
    """Project the approved plan's fresh state; this never changes canonical data."""
    plan = promotion._validate_plan(envelope)
    if plan.get("game_policy") != "fresh":
        raise ValueError("Initial public export requires reviewed fresh policy")
    after = plan["after"]
    pulse = _editorial(after["profile_state/daily_pulse"])
    original = after["profile_state/daily_pulse"]
    if "analysis" in original:
        pulse["analysis"] = _select(original["analysis"], ANALYSIS_FIELDS)
    if "visual_decision" in original:
        pulse["visual_decision"] = _select(original["visual_decision"], VISUAL_FIELDS)
        pulse["visual_decision"]["local_svg_path"] = "assets/daily_pulse.svg"
    pulse["published_pulse"] = _editorial(original["published_pulse"])
    feed = copy.deepcopy(after["pulse_feed/v1"])
    feed["entries"] = [_editorial(item) for item in feed["entries"]]
    if "updated_at" not in feed:
        feed["updated_at"] = feed["entries"][0]["updated_at"]
    # The public roster contains rotation state and public handles, never source
    # payloads or server-only discovery evidence. Full roster remains canonical.
    roster = after["profile_state/orbit_roster"]
    public_roster = _select(
        roster, {"state_version", "curated_o3_cursor", "rss_o3_cursor", "updated_at"}
    )
    public_roster["handles"] = {
        handle: _select(
            record,
            {"tier", "status", "weight", "first_seen", "last_seen", "empty_runs"},
        )
        for handle, record in roster.get("handles", {}).items()
    }
    return {
        "data/game_state.json": copy.deepcopy(after["profile_state/active_game"]),
        "data/zeitgeist_state.json": pulse,
        "data/hall_of_records.json": copy.deepcopy(after["profile_state/elo_ledger"]),
        "data/bot_brain.json": copy.deepcopy(after["profile_state/bot_memory"]),
        "data/pulse_feed.json": feed,
        "data/orbit_roster.json": public_roster,
    }


def _audit_public_text(path, raw):
    if Path(path).suffix not in {".json", ".md", ".svg"}:
        return
    text = raw.decode("utf-8")
    forbidden = (
        r"file://",
        r"/private/tmp/",
        r"/tmp/",
        r"/Users/",
        r"/home/",
        r"[A-Za-z]:\\",
        r"readme-live-sandbox/(?:issues|raw)",
        r"wonsulting-ai-job-search-oct",
        r"(?:apify_api_|gh[pousr]_|AIza|ya29\.)[A-Za-z0-9_-]{20,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    )
    if any(re.search(pattern, text) for pattern in forbidden):
        raise ValueError(
            "Private path, credential or sandbox route in public projection"
        )


def _safe_bundle_bytes(root, relative):
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root)
        or not path.is_file()
    ):
        raise ValueError("Accepted bundle file must be contained and regular")
    return path.read_bytes()


def export_release(
    source_root,
    source_sha,
    bundle_root,
    envelope,
    expected_plan_sha,
    output_dir,
    *,
    dry_run=False,
):
    source_root, bundle_root = Path(source_root).resolve(), Path(bundle_root).resolve()
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Export destination must not exist")
    resolved_output = output_dir.resolve()
    if any(
        resolved_output.is_relative_to(root) or root.is_relative_to(resolved_output)
        for root in (source_root, bundle_root)
    ):
        raise ValueError("Export must be separate from source and immutable bundle")
    if source_root != ROOT:
        raise ValueError("Execute the exporter from its pinned source checkout")
    plan = promotion._validate_plan(envelope)
    if envelope["plan_sha256"] != expected_plan_sha:
        raise ValueError("Approved plan and source commit mismatch")
    promotion._bundle_bound(envelope, bundle_root)
    for relative, expected in plan["public_json_sha256"].items():
        if sha(_safe_bundle_bytes(bundle_root, relative)) != expected:
            raise ValueError("Accepted public JSON changed while checking containment")
    inventory = source_inventory(source_root, source_sha)
    public = public_projections(envelope)
    with tempfile.TemporaryDirectory(prefix="egm-public-export-") as temporary:
        candidate = Path(temporary) / "tree"
        candidate.mkdir()
        for relative, record in inventory.items():
            path = candidate / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = _git_blob(source_root, source_sha, relative)
            if sha(raw) != record["sha256"]:
                raise ValueError("Pinned source changed")
            path.write_bytes(raw)
            path.chmod(0o755 if record["mode"] == "100755" else 0o644)
        bundle_files = {"assets/pulse_manifest.json", *plan["assets"]}
        original_hashes = {}
        for relative in sorted(bundle_files):
            raw = _safe_bundle_bytes(bundle_root, relative)
            if relative in plan["assets"] and sha(raw) != plan["assets"][relative]:
                raise ValueError("Accepted asset bytes changed while copying")
            original_hashes[relative] = sha(raw)
            destination = candidate / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
        for relative, value in public.items():
            path = candidate / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
            )
        store = JsonFileStateStore(candidate)
        store.save_game_state(public["data/game_state.json"])
        svg = candidate / "assets/daily_pulse.svg"
        if 'data-publication-layout="curatorial"' not in svg.read_text():
            raise ValueError("Accepted artwork must use the v3 curatorial layout")
        update_svg_mood_vote_bar(svg, promotion.ZERO)
        build_all_readmes(candidate, repo_slug=PUBLIC_SLUG, state_store=store)
        if not check_all_readmes(candidate, repo_slug=PUBLIC_SLUG, state_store=store):
            raise ValueError("Generated public README does not match projection")
        verification = validate_release_bundle(
            candidate, expected_repo_slug=PUBLIC_SLUG, require_publication_manifest=True
        )
        if not verification["ok"]:
            raise ValueError("Export release link/asset gate failed")
        verify_publication_bundle(candidate, require_current_pulse=True)
        generated = (
            PUBLIC_JSON
            | set(TYPOGRAPHY_ASSETS)
            | bundle_files
            | {"assets/game_board.svg", "README.md"}
            | {
                str(path.relative_to(candidate))
                for path in (candidate / "variants").glob("*.md")
            }
        )
        files = {
            str(path.relative_to(candidate)): sha(path.read_bytes())
            for path in sorted(candidate.rglob("*"))
            if path.is_file()
        }
        if set(files) != set(inventory) | generated:
            raise ValueError("Unexpected export inventory")
        for relative in generated | {"PROJECTS.md", "config/orbit_sources.json"}:
            _audit_public_text(relative, (candidate / relative).read_bytes())
        promotion._bundle_bound(envelope, bundle_root)
        report = {
            "schema_version": 1,
            "status": "validated_dry_run" if dry_run else "exported_offline",
            "release_decision": "NO_GO_PENDING_APPROVAL_AND_LIVE_VISUAL_GATES",
            "source_sha": source_sha,
            "commissioning_source_sha": plan["source_sha"],
            "commissioning_run": plan["source_run"],
            "source_inventory_sha256": sha(("\n".join(inventory) + "\n").encode()),
            "source_files": inventory,
            "plan_sha256": expected_plan_sha,
            "publication_id": plan["publication_id"],
            "public_repo_slug": PUBLIC_SLUG,
            "accepted_bundle_files": original_hashes,
            "export_files": files,
            "git_history_created": False,
            "cloud_state_changed": False,
            "external_links_verified": False,
            "native_visual_review_complete": False,
        }
        if not dry_run:
            # Destination is created exclusively; an interrupted copy is never
            # reported complete and must not be reused by another invocation.
            shutil.copytree(candidate, output_dir)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "bundle-root",
        "plan",
        "output-dir",
        "private-inventory",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--expected-plan-sha", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        report_path = args.private_inventory.absolute()
        if report_path.exists() or any(
            report_path.resolve().is_relative_to(path.resolve())
            for path in (args.output_dir, args.source_root, args.bundle_root)
        ):
            raise ValueError(
                "Private inventory must be new and outside public/source/bundle directories"
            )
        envelope = json.loads(args.plan.read_text())
        report = export_release(
            args.source_root,
            args.source_sha,
            args.bundle_root,
            envelope,
            args.expected_plan_sha,
            args.output_dir,
            dry_run=args.dry_run,
        )
        promotion._private_write(report_path, report, exclusive=True)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "source_sha": args.source_sha,
                    "file_count": len(report["export_files"]),
                    "production_authorized": False,
                }
            )
        )
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "error_type": type(error).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
