#!/usr/bin/env python3
"""Owner-only repair page: reviewed tools from a fixed list, plan first, apply on a pasted code.

The public run shows only {"status":"ok"} or {"status":"refused"}. Every reason,
plan and plan code goes to the private owner-alert log as fixed-template text.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
from io import StringIO
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib
from types import SimpleNamespace

OWNER = 'Siyuan-Zhu'
REPOSITORY = 'Siyuan-Zhu/Siyuan-Zhu'
WORKFLOW_REF = 'Siyuan-Zhu/Siyuan-Zhu/.github/workflows/owner_repair.yml@refs/heads/main'
ACTIONS = ('none', 'check-credentials', 'plan:retry-today', 'apply:retry-today')
ALPHABET = '0123456789ABCDEFGHJKMNPQRSTVWXYZ'
CODE_LENGTH = 10
CODE_TTL = timedelta(hours=12)
RECORDS = 'owner_repairs'
CLAIMS = 'editorial_executions'
ACTIVE_UNCERTAIN = frozenset({'generation', 'validation', 'publication', 'awaiting_approval',
                              'requires_reconciliation'})
BOT_EMAIL = '41898282+github-actions[bot]@users.noreply.github.com'
_ET = ZoneInfo('America/New_York')
_B1_CODE = 'automatic-provider-metadata-or-credential-binding-failed'
_STAGE = 'owner_repair'


def _deps():
    """Project imports, resolved at call time and only inside main's redirected region.

    Nothing from src/ is imported at module load, so an import-time warning or
    traceback can never reach the public log.
    """
    store = importlib.import_module('src.state_store')
    reasons = importlib.import_module('src.zeitgeist.owner_reasons')
    return SimpleNamespace(
        commit=store._commit_documents, Conflict=store.PublicationConflict,
        Unknown=store.PublicationOutcomeUnknown,
        owner_alert=importlib.import_module('src.zeitgeist.owner_alert'),
        observe=importlib.import_module('src.zeitgeist.automatic_spend_runtime').observe_provider_metadata,
        Reason=reasons.Reason, resolve=reasons.resolve,
        SpendBlocked=importlib.import_module('src.zeitgeist.spend_guard').SpendBlocked)


def _reason(cls, headline, tonight, fix):
    return _deps().Reason(cls, headline, tonight, fix)


class Refusal(Exception):
    """A plain refusal; only its fixed code (and a whole number of minutes) is ever used."""

    def __init__(self, code: str, minutes: int | None = None):
        super().__init__(code)
        self.code = code
        self.minutes = minutes


# --- guard ----------------------------------------------------------------------------

def owner_guard(env, action: str) -> None:
    """Refuse everything except the owner's own dispatch of this file on main."""
    ok = (action in ACTIONS
          and env.get('GITHUB_EVENT_NAME') == 'workflow_dispatch'
          and env.get('GITHUB_ACTOR') == OWNER
          and env.get('GITHUB_TRIGGERING_ACTOR') == OWNER
          and env.get('GITHUB_REPOSITORY') == REPOSITORY
          and env.get('GITHUB_REF') == 'refs/heads/main'
          and env.get('GITHUB_WORKFLOW_REF') == WORKFLOW_REF
          and (not action.startswith('apply:') or env.get('GITHUB_RUN_ATTEMPT') == '1'))
    if not ok:
        raise RuntimeError('Owner repair refused')


# --- plan code ------------------------------------------------------------------------

def canonical_json(plan) -> bytes:
    return json.dumps(plan, sort_keys=True, separators=(',', ':')).encode('utf-8')


def make_plan_code(salt: bytes, plan) -> str:
    digest = hashlib.sha256(bytes(salt) + canonical_json(plan)).digest()
    bits = int.from_bytes(digest[:8], 'big') >> (64 - 5 * CODE_LENGTH)
    return ''.join(ALPHABET[(bits >> (5 * (CODE_LENGTH - 1 - i))) & 31] for i in range(CODE_LENGTH))


def normalise_plan_code(text) -> str | None:
    if not isinstance(text, str) or len(text) > 64:
        return None
    value = re.sub(r'[\s-]', '', text).upper()
    if len(value) != CODE_LENGTH or any(ch not in ALPHABET for ch in value):
        return None
    return value


def _record_id(code: str) -> str:
    return hashlib.sha256(code.encode('ascii')).hexdigest()


# --- retry-today plan -----------------------------------------------------------------

def _aware(raw):
    value = datetime.fromisoformat(raw)
    if value.tzinfo is None:
        raise ValueError('aware time required')
    return value


def _minutes_until(moment, now) -> int:
    return max(1, math.ceil((moment - now).total_seconds() / 60))


def _check_claim(prior, now):
    """Mirror claim()'s refusals (edition_execution.py) without changing them, plus spec 3.5."""
    if not isinstance(prior, dict) or not prior:
        raise Refusal('no-failed-run')
    stage = prior.get('stage')
    if stage == 'published':
        raise Refusal('published')
    if stage in ACTIVE_UNCERTAIN:
        raise Refusal('reconcile')
    if stage == 'held_for_review':
        raise Refusal('held')
    if stage == 'preflight':
        try:
            lease = _aware(prior['lease_until'])
        except Exception:
            raise Refusal('not-retryable') from None
        if lease > now:
            raise Refusal('lease', minutes=_minutes_until(lease, now))
        raise Refusal('not-retryable')
    if stage not in ('blocked', 'preflight_retryable'):
        raise Refusal('not-retryable')
    if prior.get('publication_id'):
        raise Refusal('reconcile')
    # The attempt bound, with claim()'s pre-inference exemption for spend maintenance blocks.
    pre_inference = stage == 'blocked' and prior.get('failure_category') == 'spend_maintenance'
    attempt = prior.get('attempt', 0)
    if type(attempt) is not int or attempt < 0:
        raise Refusal('not-retryable')
    if attempt >= 3 and not pre_inference:
        raise Refusal('attempts')
    try:
        if stage == 'preflight_retryable':
            from src.zeitgeist.public_tariff_http import validate_retry_not_before
            if prior.get('failure_category') != 'transient_anonymous_price_read':
                raise Refusal('not-retryable')
            deferred = validate_retry_not_before(prior.get('retry_not_before'),
                                                 now=_aware(prior['updated_at']))
            if now < deferred:
                raise Refusal('wait', minutes=_minutes_until(deferred, now))
        elif not pre_inference:
            raise Refusal('not-retryable')
        lease = _aware(prior['lease_until'])
    except Refusal:
        raise
    except Exception:
        raise Refusal('not-retryable') from None
    if lease > now:
        raise Refusal('lease', minutes=_minutes_until(lease, now))
    source_revision = prior.get('source_revision')
    if not isinstance(source_revision, str) or not source_revision:
        raise Refusal('not-retryable')
    return stage, attempt, source_revision, prior.get('failure_category')


def _maintenance_planned(env, runner) -> bool:
    """Plan-only spend maintenance (no --apply; it never emits or writes). Output stays captured."""
    try:
        done = runner([sys.executable, 'scripts/maintain_spend_control.py'], cwd=ROOT,
                      capture_output=True, text=True, timeout=300, env=dict(env))
        if done.returncode != 0:
            return False
        lines = [line for line in str(done.stdout or '').splitlines() if line.strip()]
        summary = json.loads(lines[-1]) if lines else None
        return isinstance(summary, dict) and summary.get('status') == 'planned'
    except Exception:
        return False


def retry_today_plan(store, edition_date, env, now, *, runner=subprocess.run) -> dict:
    """Read-only pre-check; the returned plan holds only stable facts (no times)."""
    client = store._get_client()
    snapshot = client.collection(store._col(CLAIMS)).document(str(edition_date)).get()
    prior = snapshot.to_dict() if snapshot.exists else None
    stage, attempt, source_revision, failure_category = _check_claim(prior, now)
    if not _maintenance_planned(env, runner):
        raise Refusal('maintenance')
    return {'action': 'retry-today', 'edition_date': str(edition_date), 'claim_stage': stage,
            'attempt': attempt, 'source_revision': source_revision,
            'failure_category': failure_category, 'publication_id': 'none',
            'maintenance': 'planned'}


# --- fixed-template words (R3) --------------------------------------------------------

_AGAIN = 'Run the owner page again with plan:retry-today for a fresh plan code.'
_NOTHING = 'Nothing was changed and nothing was spent.'
SECRET_NOTE = ('The page cannot change a GitHub secret, so a token fix is a hand step,'
               ' then check-credentials, then retry-today.')


def refusal_reason(refusal: Refusal, edition: str) -> Reason:
    minutes = refusal.minutes if type(refusal.minutes) is int and refusal.minutes > 0 else 1
    words = {
        'no-failed-run': ('There is no stopped run on record for the ' + edition + ' edition, so there is nothing to retry',
                          'Wait for the scheduled run, or start the daily workflow by hand.'),
        'published': ('The ' + edition + ' edition is already published, so there is nothing to retry',
                      'Nothing to do.'),
        'reconcile': ("This needs a reconcile, the page can't retry it",
                      'The last run reached paid work or publication. It needs a reviewed reconcile in a build session.'),
        'held': ("The " + edition + " edition is held for your review, the page can't retry it",
                 'Review the held edition first.'),
        'attempts': ('Three attempts are used, this needs a reconcile',
                     'The page will not start a fourth attempt. Inspect it in a build session.'),
        'lease': ("A run is still holding today's edition, try in " + str(minutes) + ' minutes',
                  'Run plan:retry-today again after that.'),
        'wait': ('The price page asked us to wait, try in ' + str(minutes) + ' minutes',
                 'The catch-up runs may retry on their own. Otherwise run plan:retry-today again after that.'),
        'not-retryable': ("This kind of stop is not one the page can retry",
                          'It needs a reviewed fix first. Look at it in a build session.'),
        'maintenance': ('The spending check plan did not pass, so a retry would stop the same way',
                        'Run check-credentials first. Fix what it reports, then plan the retry again. '
                        + SECRET_NOTE),
        'code': ('That plan code is not valid', _AGAIN),
        'used': ('That plan code was already used', _AGAIN),
        'expired': ('That plan code has expired (codes last 12 hours)', _AGAIN),
        'changed': ("Today's state changed since the plan, so nothing was applied", _AGAIN),
        'store': ('The page could not read or write its private record, so nothing was applied',
                  'Try again later. If it keeps failing, look at it in a build session.'),
    }
    headline, fix = words.get(refusal.code, words['not-retryable'])
    return _reason('repair-result', headline, _NOTHING, fix)


def _plan_reason(plan: dict, code: str) -> Reason:
    if plan['claim_stage'] == 'preflight_retryable':
        cause = 'The last run stopped because a public price page asked us to wait; that wait is over.'
    else:
        cause = 'The last run stopped at the spending check, before any paid call.'
    return _reason(
        'repair-plan',
        'Retry today: plan ready for the ' + plan['edition_date'] + ' edition',
        cause + ' The spending check plan passes now. Nothing has changed yet. If you confirm, the page'
        ' adds one empty commit to main and starts the daily run once. Spend limits and the attempt'
        ' bound are unchanged.',
        'Run the owner page again, choose apply:retry-today and paste the plan code ' + code
        + '. The code works once and expires in 12 hours.')


_STEP_FAILED = {
    'commit': 'The code is used, but the retry commit could not be made, so no run started',
    'push': 'The code is used, but the retry commit could not be pushed, so no run started',
    'dispatch': 'The code is used and the retry commit is on main, but the daily run could not be started',
}


def _step_failed_reason(step: str) -> Reason:
    fix = ('Start the daily workflow by hand, or plan the retry again.' if step == 'dispatch'
           else 'Plan the retry again for a fresh code.')
    return _reason('repair-result', _STEP_FAILED[step], 'Nothing was spent by the page.', fix)


def _applied_reason(edition: str) -> Reason:
    return _reason('repair-result', 'Retry started for the ' + edition + ' edition',
                  'One empty commit is on main and the daily run is queued. You get the usual email if it stops again.',
                  'Nothing to do.')


def _credentials_reason(sub_cause, failed: bool) -> Reason:
    tonight = 'This was a read-only check. ' + _NOTHING + ' ' + SECRET_NOTE
    if sub_cause is not None:
        _, b1 = _deps().resolve(stage='spend_maintenance', code=_B1_CODE, sub_cause=sub_cause)
        return _reason('repair-result', b1.headline, tonight, b1.fix)
    if failed:
        return _reason('repair-result', 'The credential check could not finish', tonight,
                      'Run it again later. If it keeps failing, look at it in a build session.')
    return _reason('repair-result', 'All providers accept our keys', tonight,
                  "If today's edition stopped, run plan:retry-today.")


def _emit(reason: Reason, code: str, env, now, *, details=None, plan_code=None) -> bool:
    try:
        owner_alert = _deps().owner_alert
        entry = owner_alert.build_entry(code=code, reason=reason, stage=_STAGE, env=env, now=now,
                                        details=details, source='owner_repair', plan_code=plan_code)
        return owner_alert.emit(entry, env=env) is True
    except Exception:
        return False


# --- actions --------------------------------------------------------------------------

def _edition(now) -> str:
    return now.astimezone(_ET).strftime('%Y-%m-%d')


def check_credentials(env, now, spend_store=None) -> bool:
    """Read-only: policy read, metadata observation, sub cause. No writes, no plan."""
    from scripts.probe_profile_readiness import ReadOnlySpendStore
    deps = _deps()
    owner_alert = deps.owner_alert
    sha = env.get('SPEND_POLICY_SHA256', '')
    policy = None
    failed = False
    holder = None
    try:
        if not re.fullmatch('[a-f0-9]{64}', sha):
            raise RuntimeError('pinned policy required')
        if spend_store is None:
            from src.zeitgeist.automatic_spend_runtime import PrivateFirestoreStore
            spend_store = PrivateFirestoreStore()
        snapshot = ReadOnlySpendStore(spend_store).read('policy_' + sha)
        policy = snapshot.to_dict() if snapshot.exists else None
        if not isinstance(policy, dict):
            raise RuntimeError('policy missing')
        with owner_alert.captured_preflight() as holder:
            try:
                deps.observe(policy, env, now)
            except deps.SpendBlocked:
                failed = True  # a credential failure raises B1 inside; the holder has the report
    except Exception:
        failed = True
    sub_cause, details = owner_alert.sub_cause_from_observation(
        getattr(holder, 'report', None), policy)
    details = dict(details)
    if sub_cause is not None:
        details['sub_cause'] = sub_cause
    return _emit(_credentials_reason(sub_cause, failed), 'check-credentials', env, now, details=details)


def _write(client, collection, doc_id, value, snapshot) -> None:
    """One conditional write; an unknown outcome is settled by exact readback only."""
    deps = _deps()
    try:
        deps.commit(client, [(collection, doc_id, value, snapshot)])
    except deps.Conflict:
        raise Refusal('used') from None
    except deps.Unknown:
        after = client.collection(collection).document(doc_id).get()
        if not after.exists or after.to_dict() != value:
            raise Refusal('store') from None


def plan_retry_today(store, env, now, runner) -> None:
    edition = _edition(now)
    plan = retry_today_plan(store, edition, env, now, runner=runner)
    salt = secrets.token_bytes(16)
    code = make_plan_code(salt, plan)
    client = store._get_client()
    collection = store._col(RECORDS)
    doc_id = _record_id(code)
    snapshot = client.collection(collection).document(doc_id).get()
    if snapshot.exists:
        raise Refusal('store')
    record = {'action': 'retry-today', 'plan': plan,
              'salt_b64': base64.b64encode(salt).decode('ascii'),
              'created_at': now.isoformat(), 'expires_at': (now + CODE_TTL).isoformat(),
              'used': False}
    try:
        _write(client, collection, doc_id, record, snapshot)
    except Refusal:
        raise Refusal('store') from None
    if not _emit(_plan_reason(plan, code), 'retry-today-plan', env, now, plan_code=code):
        raise _StepFailed()  # no email means no plan was shown; the unused record just expires


def apply_retry_today(store, env, now, runner, raw_code) -> None:
    code = normalise_plan_code(raw_code)
    if code is None:
        raise Refusal('code')
    edition = _edition(now)
    plan = retry_today_plan(store, edition, env, now, runner=runner)
    client = store._get_client()
    collection = store._col(RECORDS)
    doc_id = _record_id(code)
    snapshot = client.collection(collection).document(doc_id).get()
    record = snapshot.to_dict() if snapshot.exists else None
    if not isinstance(record, dict):
        raise Refusal('code')
    if record.get('action') != 'retry-today':
        raise Refusal('code')
    if record.get('used') is not False:
        raise Refusal('used')
    try:
        expires = _aware(record['expires_at'])
        salt = base64.b64decode(record['salt_b64'], validate=True)
    except Exception:
        raise Refusal('code') from None
    if now >= expires:
        raise Refusal('expired')
    if len(salt) != 16:
        raise Refusal('code')
    if record.get('plan') != plan or not hmac.compare_digest(make_plan_code(salt, plan), code):
        raise Refusal('changed')
    marker = 'github-' + str(env.get('GITHUB_RUN_ID', 'local')) + '-' + str(env.get('GITHUB_RUN_ATTEMPT', '1'))
    used = dict(record, used=True, used_at=now.isoformat(), used_by=marker)
    _write(client, collection, doc_id, used, snapshot)  # compare-and-set: the code is spent here
    steps = (
        ('commit', ['git', '-c', 'user.name=' + OWNER, '-c', 'user.email=' + BOT_EMAIL,
                    'commit', '--allow-empty', '-m', 'chore: owner retry [skip ci]']),
        ('push', ['git', 'push', 'origin', 'HEAD:refs/heads/main']),
        ('dispatch', ['gh', 'workflow', 'run', 'daily_zeitgeist.yml', '--ref', 'main']),
    )
    for step, command in steps:
        try:
            runner(command, cwd=ROOT, capture_output=True, text=True, check=True, timeout=120,
                   env=dict(env))
        except Exception:
            _emit(_step_failed_reason(step), 'retry-today-' + step + '-failed', env, now)
            raise _StepFailed() from None
    _emit(_applied_reason(edition), 'retry-today-applied', env, now)


class _StepFailed(Exception):
    """A step after the compare-and-set failed; already reported, never retried."""


def _dispatch(args, env, now, store, runner, spend_store) -> bool:
    action = args.action
    if action == 'none':
        return True
    if action == 'check-credentials':
        return check_credentials(env, now, spend_store) is True
    if store is None:
        from src.state_store import get_state_store
        store = get_state_store(repo_root=ROOT, backend='firestore',
                                collection_prefix=env.get('FIRESTORE_COLLECTION_PREFIX', ''))
    try:
        if action == 'plan:retry-today':
            plan_retry_today(store, env, now, runner)
        else:
            apply_retry_today(store, env, now, runner, args.plan_code)
        return True
    except Refusal as refusal:
        _emit(refusal_reason(refusal, _edition(now)), 'retry-today-' + refusal.code, env, now)
        return False
    except _StepFailed:
        return False
    except Exception:
        _emit(refusal_reason(Refusal('store'), _edition(now)), 'retry-today-store', env, now)
        return False


def main(argv=None, *, env=None, now=None, store=None, runner=None, spend_store=None) -> int:
    ok = False
    previous_logging = logging.root.manager.disable
    try:
        logging.disable(logging.CRITICAL)
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(StringIO()))
            stack.enter_context(redirect_stderr(StringIO()))
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('--action', default='none')
            parser.add_argument('--plan-code', default='')
            args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
            env = os.environ if env is None else env
            owner_guard(env, args.action)  # before any store read, secret use or subprocess
            _deps()  # project imports load here, inside the redirect; a failure is a refusal
            now = now or datetime.now(timezone.utc)
            ok = _dispatch(args, env, now, store, runner or subprocess.run, spend_store)
    except BaseException:
        ok = False
    finally:
        logging.disable(previous_logging)
    print(json.dumps({'status': 'ok' if ok else 'refused'}, separators=(',', ':')))
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
