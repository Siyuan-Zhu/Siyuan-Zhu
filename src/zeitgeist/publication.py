"""Bind a cultural publication to its rendered bytes; reject mixed/stale assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

MANIFEST_PATH = "assets/pulse_manifest.json"


def _asset_digest(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".svg":
        # Votes change independently of the commissioned artwork.
        raw = re.sub(
            rb'<metadata id="mood-votes">.*?</metadata>', b"", raw, flags=re.DOTALL
        )
    return hashlib.sha256(raw).hexdigest()


def write_publication_manifest(
    root: Path,
    *,
    date: str,
    word: str,
    gloss: str,
    archetype: str,
    medium: str,
    archive_path: str,
) -> dict:
    files = [
        "assets/daily_pulse_light.webp",
        "assets/daily_pulse_dark.webp",
        "assets/daily_pulse.svg",
        archive_path,
    ]
    manifest = {
        "version": 1,
        "date": date[:10],
        "word": word,
        "gloss": gloss,
        "archetype": archetype,
        "medium": medium,
        "files": {p: _asset_digest(root / p) for p in files},
    }
    manifest["publication_id"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:20]
    (root / MANIFEST_PATH).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_publication_bundle(
    root: Path, *, require_manifest: bool = True, require_current_pulse: bool = False
) -> dict | None:
    root = Path(root).resolve()
    path = root / MANIFEST_PATH
    if not path.exists() and not require_manifest:
        return None
    if not path.exists():
        raise ValueError(
            "Missing publication manifest; generate the complete cultural publication first"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("version") != 1:
        raise ValueError("Unsupported publication manifest")
    for field in ("date", "word", "gloss", "archetype", "medium"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"Publication manifest missing {field}")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", manifest["date"]):
        raise ValueError("Invalid publication manifest date")
    expected = dict(manifest)
    identity = expected.pop("publication_id", None)
    if (
        hashlib.sha256(
            json.dumps(expected, sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        != identity
    ):
        raise ValueError("Publication manifest identity mismatch")
    required = {
        "assets/daily_pulse_light.webp",
        "assets/daily_pulse_dark.webp",
        "assets/daily_pulse.svg",
        f"assets/pulse/{manifest['date']}.webp",
    }
    if (
        not isinstance(manifest.get("files"), dict)
        or set(manifest["files"]) != required
    ):
        raise ValueError(
            "Publication manifest must contain both theme plates, SVG, and archive"
        )
    for rel, digest in manifest["files"].items():
        asset = (root / rel).resolve()
        if (
            not asset.is_relative_to(root)
            or not asset.is_file()
            or _asset_digest(asset) != digest
        ):
            raise ValueError(f"Publication asset mismatch: {rel}")
    feed_path = root / "data/pulse_feed.json"
    feed = json.loads(feed_path.read_text(encoding="utf-8"))
    if not isinstance(feed, dict) or not isinstance(feed.get("entries"), list):
        raise ValueError("Invalid publication feed envelope")
    entries = feed["entries"]
    entry = next(
        (
            x
            for x in entries
            if isinstance(x, dict) and str(x.get("date", ""))[:10] == manifest["date"]
        ),
        None,
    )
    if (
        not entry
        or str(entry.get("word", "")).casefold() != manifest["word"].casefold()
    ):
        raise ValueError("Publication feed word/date mismatch")
    for key, manifest_key in (("gloss", "gloss"), ("visual_medium", "medium")):
        if str(entry.get(key, "")).strip() != manifest[manifest_key].strip():
            raise ValueError(f"Publication feed {key} mismatch")
    if (
        entry.get("curatorial_archetype")
        and entry["curatorial_archetype"] != manifest["archetype"]
    ):
        raise ValueError("Publication feed archetype mismatch")
    if require_current_pulse:
        state_path = root / "data/zeitgeist_state.json"
        if not state_path.is_file():
            raise ValueError("Missing current pulse state for publication verification")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not isinstance(state, dict):
            raise ValueError("Invalid current pulse state")
        verify_current_pulse(state, manifest)
    return manifest


def verify_current_pulse(state: dict, manifest: dict) -> None:
    """Reject divergent supplied state projections, not just one preferred copy."""
    projections = [state]
    for key in ("published_pulse", "analysis", "visual_decision"):
        if isinstance(state.get(key), dict):
            projections.append(state[key])
    aliases = {
        "date": ("date", "date_iso"),
        "word": ("word", "word_of_the_day"),
        "gloss": ("gloss", "explanation_phrase"),
        "medium": ("visual_medium", "selected_mode"),
        "archetype": ("curatorial_archetype",),
    }
    medium_aliases = {
        "CURATED_PHOTO": "PHOTOGRAPHY",
        "UNSPLASH_PHOTO": "PHOTOGRAPHY",
        "CURATED_PHOTO_LIBRARY": "PHOTOGRAPHY",
        "SVG_FALLBACK": "BESPOKE_SVG",
        "ARCHITECTURAL_BLUEPRINT_SVG": "BESPOKE_SVG",
    }
    for manifest_key, keys in aliases.items():
        values = [
            str(projection[key]).strip()
            for projection in projections
            for key in keys
            if projection.get(key) is not None and str(projection[key]).strip()
        ]
        if not values:
            raise ValueError(f"Current pulse missing publication {manifest_key}")
        expected = manifest[manifest_key].strip()
        for value in values:
            if manifest_key == "date":
                same = value[:10] == expected
            elif manifest_key == "word":
                same = value.casefold() == expected.casefold()
            elif manifest_key == "medium":
                same = medium_aliases.get(
                    value.upper(), value.upper()
                ) == medium_aliases.get(expected.upper(), expected.upper())
            else:
                same = value == expected
            if not same:
                raise ValueError(
                    f"Current pulse {manifest_key} does not match publication manifest"
                )
