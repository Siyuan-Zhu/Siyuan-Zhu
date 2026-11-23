#!/usr/bin/env python3
"""Build a blind comparison page from the private run records of the newest comparison dispatch.

Run on the owner's machine only. It reads the private_run records through the state store,
takes the ones that share the newest intake hash and name an arm, letters the arms in a shuffled
order, and writes two files outside any git checkout: an HTML page that names no arm and no model,
and a key file that maps the letters to the arms. It prints only the two paths and the counts.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT_DIR = Path("~/egm-k1-private/compare").expanduser()
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
COUNTS_LINE = "[compare] arms=%d records=%d"
FOOT_NOTE = "An engine may show fewer attempts when a call limit or a fallback was reached."
RECORD_LIMIT = 1000


def _candidate_words(record: Dict[str, Any]) -> List[str]:
    words = []
    for entry in record.get("candidate_words") or ():
        word = entry.get("word") if isinstance(entry, dict) else entry
        if word:
            words.append(str(word))
    return words


def build_comparison(records: Sequence[Any], *, seed: Any) -> Dict[str, Any]:
    """The arm records of the newest intake, lettered by a shuffle fixed by `seed`. Pure; raises ValueError if none."""
    armed = [record for record in records
             if isinstance(record, dict) and record.get("arm") and record.get("intake_hash")]
    if not armed:
        raise ValueError("no comparison records")
    newest = max(armed, key=lambda record: str(record.get("written_at") or ""))["intake_hash"]
    # Per arm and repeat, the newest record: a second dispatch on the same intake replaces the first.
    chosen: Dict[str, Dict[Any, Dict[str, Any]]] = {}
    for record in sorted((r for r in armed if r["intake_hash"] == newest),
                         key=lambda record: str(record.get("written_at") or "")):
        chosen.setdefault(str(record["arm"]), {})[record.get("repeat")] = record
    names = sorted(chosen)
    random.Random(seed).shuffle(names)
    arms: Dict[str, Any] = {}
    key: Dict[str, str] = {}
    for letter, name in zip(LETTERS, names):
        repeats = [chosen[name][repeat] for repeat in sorted(chosen[name], key=lambda value: (value is None, value))]
        arms[letter] = {
            "repeats": [{
                "repeat": record.get("repeat"),
                "word": record.get("word"),
                "gloss": record.get("gloss"),
                "thesis": record.get("thesis"),
                "chosen_rank": record.get("chosen_rank"),
                "candidate_words": _candidate_words(record),
                "prompt_path": record.get("prompt_path"),
                "models": record.get("models"),
            } for record in repeats],
            "same_choice": len({record.get("chosen_cluster_id") for record in repeats}) == 1,
        }
        key[letter] = name
    return {"intake_hash": newest, "arms": arms, "key": key}


def _text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def render_html(comparison: Dict[str, Any]) -> str:
    """One page with the lettered arms side by side. Pure.

    It shows no arm name, model, prompt path or candidate words: candidate words exist only on the
    packet slate prompt, so their presence alone would tell the arms apart. They stay in the key file's
    comparison data, not on the page.
    """
    columns = []
    for letter, entry in comparison["arms"].items():
        blocks = []
        for row in entry["repeats"]:
            blocks.append(
                '<section class="repeat">'
                f"<h3>Repeat {_text(row['repeat'])}</h3>"
                f'<p class="word">{_text(row["word"])}</p>'
                f'<p class="gloss">{_text(row["gloss"])}</p>'
                f'<p class="thesis">{_text(row["thesis"])}</p>'
                f"<p>Chosen candidate rank: {_text(row['chosen_rank'])}</p>"
                "</section>"
            )
        agree = "yes" if entry["same_choice"] else "no"
        columns.append(
            f'<div class="engine"><h2>Engine {_text(letter)}</h2>'
            f"<p>Repeats chose the same candidate: {agree}</p>{''.join(blocks)}</div>"
        )
    return (
        "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<title>Blind comparison</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#1f2933}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(18rem,1fr));gap:1.5rem}"
        ".engine{border:1px solid #cbd2d9;border-radius:6px;padding:1rem}"
        ".word{font-size:1.5rem;font-weight:600;margin:0.25rem 0}"
        ".gloss{font-style:italic}.note{color:#52606d}"
        ".repeat{border-top:1px solid #e4e7eb;padding-top:0.5rem}"
        "</style></head><body>"
        f"<h1>Blind comparison</h1><p>Intake {_text(comparison['intake_hash'][:12])}</p>"
        f'<div class="grid">{"".join(columns)}</div>'
        f'<p class="note">{FOOT_NOTE}</p></body></html>\n'
    )


def _inside_git_checkout(path: Path) -> bool:
    for folder in (path, *path.parents):
        if (folder / ".git").exists():
            return True
    return False


def _store() -> Any:
    from src.state_store import get_state_store

    return get_state_store(repo_root=ROOT)


def _write_private(path: Path, text: str) -> None:
    """Write the file readable by the owner only; a file that already existed keeps no wider mode."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.chmod(path, 0o600)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where the page and key are written")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if _inside_git_checkout(out_dir):
        print("[compare] refused: the output directory is inside a git checkout")
        return 2
    records = _store().load_run_records(kind="private_run", limit=RECORD_LIMIT)
    try:
        comparison = build_comparison(records, seed=random.SystemRandom().getrandbits(64))
    except ValueError:
        print(COUNTS_LINE % (0, 0))
        return 1
    out_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    # mkdir leaves an existing directory's mode as it was.
    os.chmod(out_dir, 0o700)
    stem = comparison["intake_hash"][:12]
    page = out_dir / f"comparison-{stem}.html"
    key_file = out_dir / f"key-{stem}.json"
    _write_private(page, render_html(comparison))
    _write_private(key_file, json.dumps({"intake_hash": comparison["intake_hash"], "key": comparison["key"]},
                                        indent=2) + "\n")
    print(page)
    print(key_file)
    print(COUNTS_LINE % (len(comparison["arms"]),
                         sum(len(entry["repeats"]) for entry in comparison["arms"].values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
