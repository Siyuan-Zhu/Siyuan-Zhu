"""Private owner-alert emitter: one fixed-schema Cloud Logging entry per failure.

Nothing here prints, logs or raises into the caller. The entry carries only
fixed reason text and a small allow-listed ``details`` block; no provider
text, token, response body or exception text is ever copied in.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import urllib.request
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, time as dtime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from src.zeitgeist.owner_reasons import Reason, resolve

MARKER = '.daily-outcomes/owner_alerted'
SUB_CAUSES = frozenset({'credential_rejected', 'provider_unreachable', 'metadata_incomplete',
                        'credential_missing', 'credential_not_approved'})
CLASSES = frozenset({'owner-act', 'owner-approve', 'watch-on-it', 'self-heal',
                     'repair-plan', 'repair-result', 'recovered', 'heartbeat'})
PROVIDERS = frozenset({'gemini', 'bfl', 'apify', 'unsplash'})

_PROJECT = 'git-readme-features'
_LOG_NAME = f'projects/{_PROJECT}/logs/egm-owner-alert'
_WRITE_URL = 'https://logging.googleapis.com/v2/entries:write'
_ET = ZoneInfo('America/New_York')
_SELF_HEAL_CUTOFF = dtime(3, 30)
_FP6 = re.compile(r'[0-9a-f]{6}')


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _clean_details(details: Mapping | None) -> dict:
    out: dict = {}
    if not isinstance(details, Mapping):
        return out
    provider = details.get('provider')
    if isinstance(provider, str) and provider in PROVIDERS:
        out['provider'] = provider
    status = details.get('http_status')
    if type(status) is int and 100 <= status <= 599:
        out['http_status'] = status
    for key in ('runner_fp6', 'ledger_fp6'):
        value = details.get(key)
        if isinstance(value, str) and _FP6.fullmatch(value):
            out[key] = value
    sub = details.get('sub_cause')
    if isinstance(sub, str) and sub in SUB_CAUSES:
        out['sub_cause'] = sub
    return out


def build_entry(*, code: str, reason: Reason, stage: str, env: Mapping[str, str], now: datetime,
                details: Mapping | None = None, source: str = 'runner',
                plan_code: str | None = None) -> dict:
    run_id = env.get('GITHUB_RUN_ID') or 'local'
    edition = env.get('ZEITGEIST_EDITION_DATE') or _utc(now).astimezone(_ET).strftime('%Y-%m-%d')
    payload = {
        'schema_version': 1,
        'class': reason.cls,
        'code': code,
        'headline': reason.headline,
        'tonight': reason.tonight,
        'fix': reason.fix,
        'ref': f'run {run_id} | edition {edition} | stage {stage}',
        'run_id': run_id,
        'edition_date': edition,
        'repo': env.get('GITHUB_REPOSITORY') or 'local',
        'source': source,
        'details': _clean_details(details),
    }
    if plan_code is not None:
        payload['plan_code'] = plan_code
    return {'logName': _LOG_NAME,
            'resource': {'type': 'global', 'labels': {'project_id': _PROJECT}},
            'jsonPayload': payload}


def promote_self_heal(cls: str, *, started_at: datetime, source: str) -> str:
    if cls != 'self-heal':
        return cls
    if source == 'mac_fallback':
        return 'owner-act'
    return 'owner-act' if _utc(started_at).astimezone(_ET).time() >= _SELF_HEAL_CUTOFF else cls


def default_token(env: Mapping[str, str]) -> str:
    if os.environ.get('ZEITGEIST_OFFLINE_DEMO') == '1':
        raise RuntimeError('owner alerts are disabled in offline demo mode')
    if env.get('ACTIONS_ID_TOKEN_REQUEST_URL') and env.get('ACTIONS_ID_TOKEN_REQUEST_TOKEN'):
        from src import state_store
        return state_store.exchange_github_oidc_token(env)
    done = subprocess.run(['gcloud', 'auth', 'application-default', 'print-access-token'],
                          capture_output=True, text=True, check=True, timeout=10)
    return done.stdout.strip()


def emit(entry: dict, *, env: Mapping[str, str],
         token_provider: Callable[[Mapping[str, str]], str] | None = None,
         opener=urllib.request.urlopen) -> bool:
    try:
        token = (token_provider or default_token)(env)
        request = urllib.request.Request(
            _WRITE_URL, data=json.dumps({'entries': [entry]}).encode('utf-8'),
            headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
            method='POST')
        with opener(request, timeout=10) as response:
            response.read()
        return True
    except Exception:
        return False


def notify(*, stage: str, env: Mapping[str, str] | None = None, now: datetime | None = None,
           started_at: datetime | None = None, code: str | None = None,
           failure_category: str | None = None, error_type: str | None = None,
           message: str | None = None, details: Mapping | None = None,
           sub_cause: str | None = None, source: str = 'runner',
           cls_override: str | None = None, plan_code: str | None = None,
           root: Path | None = None) -> bool:
    try:
        env = os.environ if env is None else env
        now = now or datetime.now(timezone.utc)
        started_at = started_at or now
        public_code, reason = resolve(stage=stage, code=code, failure_category=failure_category,
                                      error_type=error_type, message=message, sub_cause=sub_cause)
        if cls_override is not None:
            reason = dataclasses.replace(reason, cls=cls_override)
            public_code = code if code is not None else public_code
        else:
            reason = dataclasses.replace(reason, cls=promote_self_heal(
                reason.cls, started_at=started_at, source=source))
        merged = dict(details) if isinstance(details, Mapping) else {}
        if sub_cause is not None:
            merged['sub_cause'] = sub_cause
        entry = build_entry(code=public_code, reason=reason, stage=stage, env=env, now=now,
                            details=merged, source=source, plan_code=plan_code)
        if not emit(entry, env=env):
            return False
        if root is not None:
            marker = Path(root) / MARKER
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()
        return True
    except Exception:
        return False


class _Holder:
    report = None


@contextmanager
def captured_preflight():
    """Keep the last raw run_preflight return value in memory only."""
    from scripts import provider_preflight
    original = provider_preflight.run_preflight
    holder = _Holder()

    def capture(*args, **kwargs):
        value = original(*args, **kwargs)
        holder.report = value
        return value

    provider_preflight.run_preflight = capture
    try:
        yield holder
    finally:
        provider_preflight.run_preflight = original


def _fp6(value) -> str | None:
    if isinstance(value, str) and _FP6.fullmatch(value[:6]):
        return value[:6]
    return None


# Mirrors automatic_spend.REBINDABLE_CREDENTIALS (a test pins them equal); the runtime
# skips the runner-vs-policy fingerprint check for these, so the alert must too.
REBINDABLE = frozenset({'apify'})


def sub_cause_from_observation(observation: Mapping | None, policy: Mapping | None) -> tuple[str | None, dict]:
    providers = policy.get('providers') if isinstance(policy, Mapping) else None
    if not isinstance(providers, Mapping) or not isinstance(observation, Mapping):
        return None, {}
    for name in providers:
        seen = observation.get(name)
        if name == 'unsplash' or name not in PROVIDERS or not isinstance(seen, Mapping):
            continue
        status, http = seen.get('status'), seen.get('http_status')
        details: dict = {'provider': name}
        if type(http) is int:
            details['http_status'] = http
        if http in (401, 403):
            return 'credential_rejected', details
        if status == 'unavailable':
            return 'provider_unreachable', details
        if status == 'incomplete':
            return 'metadata_incomplete', details
        if status == 'missing_credentials':
            return 'credential_missing', details
        if status in ('metadata_verified', 'verified'):
            account = providers[name].get('account') if isinstance(providers[name], Mapping) else None
            ledger = _fp6(account.get('credential_fingerprint')) if isinstance(account, Mapping) else None
            runner = _fp6(seen.get('credential_fingerprint'))
            if name not in REBINDABLE and ledger and runner and runner != ledger:
                return 'credential_not_approved', {'provider': name, 'runner_fp6': runner, 'ledger_fp6': ledger}
    return None, {}
