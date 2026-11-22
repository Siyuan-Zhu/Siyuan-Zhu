"""Read or clear the private embedding calibration document (operator only, run by hand).

  STATE_STORE_BACKEND=firestore python3 scripts/embedding_calibration.py show
  STATE_STORE_BACKEND=firestore python3 scripts/embedding_calibration.py clear --expected-status rejected

`clear` refuses unless the stored state matches the expected status, then reads back absence.
The script never writes a calibration and prints numbers only. No workflow calls it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.state_store import get_state_store
from src.zeitgeist.embedding_migration import (
    CAL_ACCEPTED,
    CAL_INVALID,
    CAL_REJECTED,
    CALIBRATION_DOCUMENT,
    _joined,
    _share,
    parse_calibration,
)

_EXPECTED_STATES = {"accepted": CAL_ACCEPTED, "rejected": CAL_REJECTED, "invalid": CAL_INVALID}


def _show(store: Any) -> int:
    document = store.load_calibration(CALIBRATION_DOCUMENT)
    state, cal = parse_calibration(document)
    print(
        "[embedding_calibration] present=%d state=%d date=%s items=%d p=%s "
        "multi_source=%d largest_share=%s top5_survivors=%d"
        % (
            0 if document is None else 1,
            state,
            cal.date if cal else "-1",
            cal.item_count if cal else -1,
            _joined(cal.percentiles if cal else None, "%.6f"),
            cal.multi_source if cal else -1,
            _share(cal.largest_share if cal else None),
            cal.top5_survivors if cal else -1,
        )
    )
    return 0


def _clear(store: Any, expected_status: str) -> int:
    document = store.load_calibration(CALIBRATION_DOCUMENT)
    state, _ = parse_calibration(document)
    if document is None or state != _EXPECTED_STATES[expected_status]:
        print("[embedding_calibration] cleared=0 present=%d state=%d" % (0 if document is None else 1, state))
        return 2
    store.clear_calibration(CALIBRATION_DOCUMENT)
    if store.load_calibration(CALIBRATION_DOCUMENT) is not None:
        print("[embedding_calibration] cleared=0 present=1 state=%d" % state)
        return 3
    print("[embedding_calibration] cleared=1")
    return 0


def main(argv: Optional[Sequence[str]] = None, *, store: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("show")
    clear = commands.add_parser("clear")
    clear.add_argument("--expected-status", required=True, choices=sorted(_EXPECTED_STATES))
    args = parser.parse_args(argv)
    if store is None:
        store = get_state_store(
            repo_root=ROOT,
            backend="firestore",
            collection_prefix=os.environ.get("FIRESTORE_COLLECTION_PREFIX", ""),
        )
    if args.command == "show":
        return _show(store)
    return _clear(store, args.expected_status)


if __name__ == "__main__":
    sys.exit(main())
