#!/usr/bin/env python3
"""One-time settlement of the stale 2026-11-20 Apify hold from Apify's own run record.

The hold was never settled, so it is counted at its full bound; once its allowance period ended
on 2026-11-25 the credit-only funding guard blocks every paid run while it stays unresolved.
The reviewed settlement logic lives in scripts/apify_hold_reconcile.py (the same logic that
settled the 2 Nov hold); this script names the one hold it may settle. Plan by default,
--apply to write. Prints status, amount and basis only.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import apify_hold_reconcile as shared
from scripts.apify_hold_reconcile import RunNotFound, apify_fetch, apify_oldest_started_at  # noqa: F401 (public names)

RESERVATION_ID = '234b8e56ad494c788e1e2798e88707e8'
RUN_ID = '1XiiuY7eK4Ct98e9a'
CREATED_AT = '2026-11-20T08:17:43.561694+00:00'
MAX = '0.16'
RECEIPT_ID = 'apify_hold_reconcile_20261120'
MONTH = '2026-11'
HOLD = shared.Hold(reservation_id=RESERVATION_ID, run_id=RUN_ID, created_at=CREATED_AT, max_gross_usd=MAX,
                   receipt_id=RECEIPT_ID, month=MONTH)


def reconcile(store, policy_sha, **options):
    return shared.reconcile(HOLD, store, policy_sha, **options)


def main(argv=None, **options):
    return shared.main(HOLD, argv, **options)


if __name__ == '__main__':
    sys.exit(main())
