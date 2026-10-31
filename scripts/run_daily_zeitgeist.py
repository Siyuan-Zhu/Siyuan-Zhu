#!/usr/bin/env python3
"""Staging and CLI entrypoint alias for `scripts/run_zeitgeist.py`."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_zeitgeist import main as run_zeitgeist_main, parse_args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Execute `scripts.run_zeitgeist.main` and optionally snapshot state to `ZEITGEIST_STATE_DIR`."""
    args = parse_args(argv)
    target_root = Path(args.repo_root).resolve()
    rc = run_zeitgeist_main(argv)

    state_dir_raw = os.environ.get("ZEITGEIST_STATE_DIR", "").strip()
    if state_dir_raw:
        state_dir = Path(state_dir_raw).resolve()
        state_dir.mkdir(parents=True, exist_ok=True)
        data_dir = (target_root / "data").resolve()
        if data_dir.is_dir() and state_dir != data_dir:
            for json_file in data_dir.glob("*.json"):
                dst = state_dir / json_file.name
                if json_file.resolve() != dst.resolve():
                    shutil.copy2(json_file, dst)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
