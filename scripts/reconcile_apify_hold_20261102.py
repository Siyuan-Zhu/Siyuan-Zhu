#!/usr/bin/env python3
"""One-time settlement of the stale 2026-11-02 Apify hold from Apify's own run record.

The hold was reserved under an earlier policy and the previous credential and never
settled, so it is counted at its full bound and becomes credit-excluded when the
usage cycle rolls over. The reviewed settlement logic lives in scripts/apify_hold_reconcile.py;
this script names the one hold it may settle.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import httpx  # noqa: F401 (the shared reads use this module; its tests patch it here)
from scripts import apify_hold_reconcile as shared
from scripts.apify_hold_reconcile import RunNotFound, apify_fetch, apify_oldest_started_at  # noqa: F401 (public names)

RESERVATION_ID = 'c14e610a08cb4b15b0ef19bc2e667849'
RUN_ID = 'Q3u3NuwtmLHJqN7g5'
CREATED_AT = '2026-11-02T07:16:32.713319+00:00'
MAX = '0.16'
RECEIPT_ID = 'apify_hold_reconcile_20261102'
MONTH = '2026-11'
HOLD = shared.Hold(reservation_id=RESERVATION_ID, run_id=RUN_ID, created_at=CREATED_AT, max_gross_usd=MAX,
                   receipt_id=RECEIPT_ID, month=MONTH)


def reconcile(store, policy_sha, **options):
    return shared.reconcile(HOLD, store, policy_sha, **options)


def main(argv=None, **options):
    return shared.main(HOLD, argv, **options)


if __name__ == '__main__':
    sys.exit(main())
