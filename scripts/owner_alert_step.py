#!/usr/bin/env python3
"""Workflow backstop for the private owner notice. Always exits 0 and prints nothing."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_attempt(root):
    try:
        path = Path(root) / '.daily-outcomes/attempt.json'
        if path.stat().st_size > 65536:
            return None
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _started_at(attempt):
    try:
        value = datetime.fromisoformat(attempt['started_at'])
        return value if value.tzinfo is not None else None
    except Exception:
        return None


def _failure(root, env, notify, marker):
    if (Path(root) / marker).exists():
        return
    attempt = _read_attempt(root)
    if attempt is None:
        notify(stage='startup', code='startup', failure_category='startup', env=env, root=root)
        return
    stage = attempt.get('stage') if isinstance(attempt.get('stage'), str) else 'startup'
    category = attempt.get('failure_category')
    notify(stage=stage, failure_category=category if isinstance(category, str) else None,
           started_at=_started_at(attempt), env=env, root=root)


def _edition_date(attempt):
    value = attempt.get('edition_date')
    try:
        if isinstance(value, str) and date.fromisoformat(value).isoformat() == value:
            return value
    except ValueError:
        pass
    return None


def _success(root, env, notify, store):
    attempt = _read_attempt(root)
    if attempt is None or attempt.get('status') != 'published':
        return
    edition = _edition_date(attempt)
    if edition is None:
        return
    if store is None:
        from src.state_store import get_state_store
        store = get_state_store(repo_root=root, backend='firestore',
                                collection_prefix=env.get('FIRESTORE_COLLECTION_PREFIX', ''))
    snapshot = store._get_client().collection(store._col('editorial_executions')).document(edition).get()
    claim = snapshot.to_dict() if snapshot.exists else None
    attempts = claim.get('attempt') if isinstance(claim, dict) else None
    if type(attempts) is int and attempts > 1:
        notify(stage='publication', cls_override='recovered', code='recovered',
               started_at=_started_at(attempt), env=env,
               now=datetime.now(timezone.utc))


def main(argv=None, *, root=None, env=None, notify=None, store=None):
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        outcome = args[args.index('--outcome') + 1] if '--outcome' in args else None
        root = Path.cwd() if root is None else Path(root)
        env = os.environ if env is None else env
        from src.zeitgeist import owner_alert
        notify = notify or owner_alert.notify
        if outcome == 'failure':
            _failure(root, env, notify, owner_alert.MARKER)
        elif outcome == 'success':
            _success(root, env, notify, store)
    except BaseException:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
