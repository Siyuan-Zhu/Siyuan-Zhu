"""Release publication checks must fail closed, independently of local link checks."""

import json
from pathlib import Path
import shutil

import pytest
from PIL import Image

from scripts import build_readme, validate_release_links
from src.zeitgeist.publication import write_publication_manifest

ROOT = Path(__file__).resolve().parents[1]


def bundle(tmp_path):
    shutil.copytree(ROOT / "data", tmp_path / "data")
    shutil.copytree(ROOT / "assets", tmp_path / "assets")
    shutil.copy2(ROOT / "PROJECTS.md", tmp_path / "PROJECTS.md")
    pulse = {
        "date": "2026-10-30",
        "word": "Counterpoise",
        "gloss": "A steady balance.",
        "visual_medium": "BESPOKE_SVG",
        "curatorial_archetype": "LITERARY_JOURNAL",
    }
    for name in ["daily_pulse_light.webp", "daily_pulse_dark.webp"]:
        Image.new("RGB", (1200, 800), "white").save(tmp_path / "assets" / name, "WEBP")
    (tmp_path / "assets/pulse/2026-10-30.webp").write_bytes(
        (tmp_path / "assets/daily_pulse_light.webp").read_bytes()
    )
    (tmp_path / "assets/daily_pulse.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg"/>'
    )
    (tmp_path / "data/pulse_feed.json").write_text(json.dumps({"entries": [pulse]}))
    (tmp_path / "data/zeitgeist_state.json").write_text(json.dumps(pulse))
    write_publication_manifest(
        tmp_path,
        date=pulse["date"],
        word=pulse["word"],
        gloss=pulse["gloss"],
        medium=pulse["visual_medium"],
        archetype=pulse["curatorial_archetype"],
        archive_path="assets/pulse/2026-10-30.webp",
    )
    build_readme.build_all_readmes(tmp_path)
    return pulse


def test_strict_release_gate_requires_manifest_and_checks_current_pulse(tmp_path):
    bundle(tmp_path)
    assert validate_release_links.validate_release_bundle(
        tmp_path, require_publication_manifest=True
    )["ok"]
    manifest = tmp_path / "assets/pulse_manifest.json"
    manifest.unlink()
    report = validate_release_links.validate_release_bundle(
        tmp_path, require_publication_manifest=True
    )
    assert not report["ok"]
    assert any("Missing publication manifest" in err for err in report["errors"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("date", "2026-10-29"),
        ("word", "STALE"),
        ("gloss", "Stale wording."),
        ("visual_medium", "PHOTOGRAPHY"),
        ("curatorial_archetype", "CONCRETE_POETRY"),
    ],
)
def test_strict_gate_rejects_matching_assets_with_divergent_state(
    tmp_path, field, value
):
    pulse = bundle(tmp_path)
    pulse[field] = value
    (tmp_path / "data/zeitgeist_state.json").write_text(json.dumps(pulse))
    report = validate_release_links.validate_release_bundle(
        tmp_path, require_publication_manifest=True
    )
    assert not report["ok"]
    assert any("current pulse" in err.lower() for err in report["errors"])


def test_cli_forwards_strict_gate_and_never_silently_passes_missing_manifest(
    tmp_path, capsys
):
    bundle(tmp_path)
    (tmp_path / "assets/pulse_manifest.json").unlink()
    rc = validate_release_links.main(
        ["--repo-root", str(tmp_path), "--require-publication-manifest", "--json"]
    )
    assert rc == 1
    assert json.loads(capsys.readouterr().out)["publication_manifest_required"] is True


def test_photo_credit_uses_real_metadata_and_escapes_html():
    credit = build_readme._visual_credit_html(
        {"selected_mode": "PHOTOGRAPHY"},
        {
            "visual_medium": "PHOTOGRAPHY",
            "unsplash_photographer": "A & B",
            "unsplash_profile_url": "https://unsplash.com/@ab",
            "artwork_url": "https://unsplash.com/photos/actual-photo-id",
        },
    )
    assert "https://unsplash.com/@ab" in credit
    assert "A &amp; B" in credit
    assert "actual-photo-id" in credit
    assert "1506744038136" not in credit


@pytest.mark.parametrize(
    "mode,label",
    [
        ("FLUX_3", "FLUX 3"),
        ("FLUX_PRO", "FLUX"),
        ("IMAGEN_3", "Imagen 3"),
        ("GEMINI_IMAGE", "Gemini"),
        ("BESPOKE_SVG", "Procedural"),
        ("SVG_FALLBACK", "Procedural"),
    ],
)
def test_v3_media_are_not_mislabeled_as_legacy_swiss_grid(mode, label):
    credit = build_readme._visual_credit_html({"selected_mode": mode})
    assert label in credit
    assert "Swiss" not in credit
    assert "4-lens" not in credit


def test_unknown_credit_does_not_invent_unsplash_photo_or_name():
    credit = build_readme._visual_credit_html({"selected_mode": "PHOTOGRAPHY"})
    assert "1506744038136" not in credit
    assert "Photo via Unsplash" in credit
    assert "Source unverified" in build_readme._visual_credit_html(
        {"selected_mode": "not-a-known-medium"}
    )


def test_strict_gate_rejects_changed_asset_even_when_headers_remain_valid(tmp_path):
    bundle(tmp_path)
    image = tmp_path / "assets/daily_pulse_light.webp"
    image.write_bytes(image.read_bytes() + b"changed")
    report = validate_release_links.validate_release_bundle(
        tmp_path, require_publication_manifest=True
    )
    assert not report["ok"]
    assert any("asset mismatch" in error for error in report["errors"])


def test_strict_gate_requires_state_and_rejects_feed_gloss_drift(tmp_path):
    bundle(tmp_path)
    state = tmp_path / "data/zeitgeist_state.json"
    original = state.read_bytes()
    state.unlink()
    report = validate_release_links.validate_release_bundle(
        tmp_path, require_publication_manifest=True
    )
    assert not report["ok"]
    assert any("current pulse" in error.lower() for error in report["errors"])
    state.write_bytes(original)
    feed = tmp_path / "data/pulse_feed.json"
    content = json.loads(feed.read_text())
    content["entries"][0]["gloss"] = "Stale caption."
    feed.write_text(json.dumps(content))
    report = validate_release_links.validate_release_bundle(
        tmp_path, require_publication_manifest=True
    )
    assert not report["ok"]
    assert any("feed gloss mismatch" in error for error in report["errors"])


def test_profile_only_credit_is_linked_without_inventing_a_person_name():
    credit = build_readme._visual_credit_html(
        {"selected_mode": "PHOTOGRAPHY"},
        {"unsplash_profile_url": "https://unsplash.com/@actual-profile"},
    )
    assert 'href="https://unsplash.com/@actual-profile?' in credit
    assert '>Photographer</a>' in credit


def test_unsplash_source_and_photographer_links_identify_the_application():
    from html import unescape
    import re
    from urllib.parse import parse_qs, urlsplit
    credit = build_readme._visual_credit_html({"selected_mode": "PHOTOGRAPHY"}, {
        "visual_medium": "PHOTOGRAPHY", "unsplash_photographer": "A & B",
        "unsplash_profile_url": "https://unsplash.com/@ab?existing=kept",
        "artwork_url": "https://unsplash.com/photos/real-photo-id"})
    links = re.findall(r'href="([^"]+)"', credit)
    assert len(links) == 2
    for link in links:
        query = parse_qs(urlsplit(unescape(link)).query)
        assert query["utm_source"] == ["siyuan_zhu_profile"]
        assert query["utm_medium"] == ["referral"]
    assert parse_qs(urlsplit(unescape(links[0])).query)["existing"] == ["kept"]
