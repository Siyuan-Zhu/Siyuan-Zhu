"""The blind comparison page built from private run records. Synthetic records only; no store or network."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from scripts import compare_arms_local as cmp
from src.state_store import InMemoryStateStore

ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64
OTHER = "b" * 64
MODEL_NAMES = ("gemini-3.8-flash", "gemini-4-hydrogen", "gemini-embedding-001", "gemini-embedding-2")
ARM_NAMES = ("previous", "current", "current_links", "current_reflector")
MODELS = {
    "previous": {"director": "gemini-3.8-flash", "judge": "gemini-3.8-flash", "embedding": "gemini-embedding-001",
                 "fallback_fired": False, "unrecognized": False},
    "current": {"director": "gemini-4-hydrogen", "judge": "gemini-4-hydrogen", "embedding": "gemini-embedding-2",
                "fallback_fired": False, "unrecognized": False},
}


def _record(arm, repeat, *, intake_hash=HASH, chosen="cluster_0", written_at="2026-11-22T08:00:00+00:00"):
    return {
        "record_schema": 1, "kind": "private_run", "arm": arm, "repeat": repeat, "intake_hash": intake_hash,
        "written_at": written_at, "word": f"Synthword{arm[:2]}{repeat}", "gloss": f"Synthetic gloss {repeat}",
        "thesis": f"A synthetic thesis {repeat}.", "chosen_cluster_id": chosen,
        "chosen_rank": int(chosen.rsplit("_", 1)[1]) + 1,
        "candidate_words": [{"word": f"Slateword{repeat}{i}", "fit": "synthetic fit"} for i in range(3)],
        "prompt_path": "packet" if arm == "current" else "legacy", "models": MODELS[arm],
    }


def _records(intake_hash=HASH, written_at="2026-11-22T08:00:00+00:00"):
    return [_record(arm, repeat, intake_hash=intake_hash, written_at=written_at,
                    chosen="cluster_0" if arm == "previous" else f"cluster_{repeat}")
            for arm in ("previous", "current") for repeat in (1, 2, 3)]


def test_arms_are_lettered_by_the_seed():
    records = _records()
    seen = set()
    for seed in range(20):
        comparison = cmp.build_comparison(records, seed=seed)
        assert comparison == cmp.build_comparison(records, seed=seed)
        assert sorted(comparison["key"]) == ["A", "B"]
        assert sorted(comparison["key"].values()) == ["current", "previous"]
        seen.add(comparison["key"]["A"])
    assert seen == {"previous", "current"}


def test_each_letter_holds_its_arms_repeats_and_whether_they_agree():
    comparison = cmp.build_comparison(_records(), seed=3)
    assert comparison["intake_hash"] == HASH
    for letter, arm in comparison["key"].items():
        entry = comparison["arms"][letter]
        assert [row["repeat"] for row in entry["repeats"]] == [1, 2, 3]
        first = entry["repeats"][0]
        assert first["word"] == f"Synthword{arm[:2]}1" and first["gloss"] == "Synthetic gloss 1"
        assert first["thesis"] == "A synthetic thesis 1."
        assert first["candidate_words"] == ["Slateword10", "Slateword11", "Slateword12"]
        assert first["prompt_path"] == ("packet" if arm == "current" else "legacy")
        assert first["models"] == MODELS[arm]
        assert entry["same_choice"] is (arm == "previous")
        assert [row["chosen_rank"] for row in entry["repeats"]] == ([1, 1, 1] if arm == "previous" else [2, 3, 4])
    assert not any(name in json.dumps(comparison["arms"]) for name in ARM_NAMES)


def test_the_page_names_no_arm_and_no_model():
    comparison = cmp.build_comparison(_records(), seed=5)
    page = cmp.render_html(comparison)
    assert page.startswith("<!doctype html>")
    for letter in ("A", "B"):
        assert f"Engine {letter}" in page
    for name in ARM_NAMES + MODEL_NAMES + ("flash", "hydrogen", "embedding", "packet", "legacy"):
        assert name not in page.lower(), name
    assert "Synthwordpr1" in page and "Synthwordcu3" in page
    # Candidate words exist only on the packet slate prompt, so they would tell the arms apart.
    assert "Slateword" not in page and "candidate words" not in page.lower()


def test_the_page_escapes_record_text():
    records = _records()
    records[0]["word"] = "<script>Zq</script>"
    page = cmp.render_html(cmp.build_comparison(records, seed=1))
    assert "<script>" not in page and "&lt;script&gt;Zq&lt;/script&gt;" in page


def test_records_of_another_intake_are_ignored():
    older = _records(OTHER, written_at="2026-11-20T08:00:00+00:00")
    for record in older:
        record["word"] = "Oldintakeword"
    comparison = cmp.build_comparison(older + _records(), seed=2)
    assert comparison["intake_hash"] == HASH
    assert "Oldintakeword" not in json.dumps(comparison)
    assert all(len(entry["repeats"]) == 3 for entry in comparison["arms"].values())


def test_records_without_an_arm_are_ignored_and_none_raises():
    single = dict(_record("previous", 1), arm=None, repeat=None, written_at="2026-11-23T08:00:00+00:00")
    comparison = cmp.build_comparison([single] + _records(), seed=0)
    assert comparison["intake_hash"] == HASH
    with pytest.raises(ValueError):
        cmp.build_comparison([single], seed=0)
    with pytest.raises(ValueError):
        cmp.build_comparison([], seed=0)


def test_a_repeated_dispatch_on_the_same_intake_keeps_the_newest_record_per_repeat():
    newer = _records(written_at="2026-11-23T08:00:00+00:00")
    for record in newer:
        record["word"] = "Newerword"
    comparison = cmp.build_comparison(_records() + newer, seed=4)
    for entry in comparison["arms"].values():
        assert [row["word"] for row in entry["repeats"]] == ["Newerword"] * 3


def _store_with(records):
    store = InMemoryStateStore()
    for index, record in enumerate(records):
        store.save_run_record(f"private_run:2026-11-22:local:1:{record['arm']}:{record['repeat']}:{index}", record)
    return store


def test_main_writes_the_page_and_the_key_outside_the_checkout(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cmp, "_store", lambda: _store_with(_records()))
    out_dir = tmp_path / "compare"
    assert cmp.main(["--out-dir", str(out_dir)]) == 0
    page = out_dir / f"comparison-{HASH[:12]}.html"
    key_file = out_dir / f"key-{HASH[:12]}.json"
    assert sorted(path.name for path in out_dir.iterdir()) == sorted([page.name, key_file.name])
    key = json.loads(key_file.read_text(encoding="utf-8"))
    assert key["intake_hash"] == HASH
    assert sorted(key["key"]) == ["A", "B"] and sorted(key["key"].values()) == ["current", "previous"]
    text = page.read_text(encoding="utf-8")
    assert not any(name in text for name in ARM_NAMES + MODEL_NAMES)
    printed = capsys.readouterr().out.splitlines()
    assert printed == [str(page), str(key_file), "[compare] arms=2 records=6"]


def test_main_refuses_an_output_directory_inside_a_git_checkout(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cmp, "_store", lambda: _store_with(_records()))
    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    for target in (checkout, checkout / "deep" / "compare", ROOT / "compare-out"):
        assert cmp.main(["--out-dir", str(target)]) == 2
        assert not (target / f"comparison-{HASH[:12]}.html").exists()
    assert not (ROOT / "compare-out").exists() and not (checkout / "deep").exists()
    assert "Synthword" not in capsys.readouterr().out


def test_main_with_no_arm_records_writes_nothing(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cmp, "_store", lambda: _store_with([]))
    assert cmp.main(["--out-dir", str(tmp_path / "compare")]) == 1
    assert not (tmp_path / "compare").exists()
    assert capsys.readouterr().out.splitlines() == ["[compare] arms=0 records=0"]


def test_the_default_output_directory_is_the_private_compare_folder():
    assert cmp.DEFAULT_OUT_DIR == Path("~/egm-k1-private/compare").expanduser()


FOOT_NOTE = "An engine may show fewer attempts when a call limit or a fallback was reached."


def _columns(page):
    return re.findall(r'<div class="engine">.*?</div>', page)


def _shape(column):
    """A column with every record value blanked: what is left is the kind of thing it shows."""
    column = re.sub(r"<h2>Engine [A-Z]</h2>", "<h2>Engine ?</h2>", column)
    column = re.sub(r'(<p class="(?:word|gloss|thesis)">).*?(</p>)', r"\1?\2", column)
    column = re.sub(r"(rank: )[^<]*", r"\1?", column)
    column = re.sub(r"(same candidate: )(?:yes|no)", r"\1?", column)
    return column


def test_an_arm_without_candidate_words_looks_the_same_in_kind_as_one_with_them():
    records = _records()
    for record in records:
        if record["arm"] == "previous":
            record["candidate_words"] = []
    page = cmp.render_html(cmp.build_comparison(records, seed=7))
    first, second = _columns(page)
    assert _shape(first) == _shape(second)
    assert "Slateword" not in page
    comparison = cmp.build_comparison(records, seed=7)
    letter = next(letter for letter, arm in comparison["key"].items() if arm == "current")
    assert comparison["arms"][letter]["repeats"][0]["candidate_words"] == ["Slateword10", "Slateword11", "Slateword12"]


def test_the_page_ends_with_the_fixed_note_on_fewer_attempts():
    page = cmp.render_html(cmp.build_comparison(_records(), seed=1))
    assert page.count(FOOT_NOTE) == 1
    assert page.index(FOOT_NOTE) > page.rindex('<div class="engine">')
