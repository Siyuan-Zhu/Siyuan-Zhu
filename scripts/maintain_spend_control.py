#!/usr/bin/env python3
"""Maintain an installed approved private ledger; default is read-only planning."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
import json
import logging
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.zeitgeist.automatic_spend_runtime import (
    PrivateFirestoreStore, maintain_spend_control, observe_provider_metadata,
)
from src.zeitgeist.spend_guard import SpendBlocked
from scripts.promote_release_state import _private_write

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CODES = frozenset(SpendBlocked._PUBLIC_REASONS.values()) | {'budget-admission-rejected'}
PUBLIC_ERROR_TYPES = frozenset({'PublicationConflict','PublicationOutcomeUnknown','HTTPError',
    'URLError','HTTPStatusError','ReadTimeout','ConnectTimeout','ConnectError','TimeoutError',
    'ConnectionError','ValueError','TypeError','KeyError','JSONDecodeError','RuntimeError',
    'OSError','PermissionError','FileExistsError','FileNotFoundError'})
OBSERVATION_KEYS = frozenset({'schema_version','observed_at','mode','metadata_ok','gemini','bfl','apify','price_snapshots'})
PROVIDER_KEYS = frozenset({'status','http_status','credential_fingerprint','account_fingerprint',
    'billing','billing_status','free_tier_verified','inference_performed','required_models',
    'available_required_models','missing_models','balance_credits','cash_allowance_verified',
    'auto_topup_status','credit_expiry_status','tariff_status','generation_performed','request_made',
    'observed_default_build','configured_build','build_pin_verified','build_id','image_digest',
    'required_input_fields','username_contract_verified','input_validation','pricing','actor_started',
    'input_schema_sha256'})
OMIT_KEYS = frozenset({'error','errors','message','detail','body','body_utf8','raw','raw_body',
    'headers','request','response','authorization','cookie','prompt','contents','policy','env'})


def public_failure(error):
    """No exception message or provider-supplied dynamic field leaves this CLI."""
    if isinstance(error,SpendBlocked):
        raw=error.public_diagnostics
        code=raw.get('code') if isinstance(raw,dict) else None
        return {'code':code if isinstance(code,str) and code in PUBLIC_CODES else 'budget-admission-rejected'}
    name=type(error).__name__
    return {'code':'maintenance-unavailable',
            'error_type':name if name in PUBLIC_ERROR_TYPES else 'UnexpectedError'}


def _secrets(env):
    return sorted({value for name,value in env.items() if isinstance(value,str) and value
        and any(part in name.upper() for part in ('TOKEN','SECRET','API_KEY','PASSWORD','CREDENTIAL','PRIVATE_KEY'))},key=len,reverse=True)


def _safe_value(value,secrets,depth=0):
    if depth>20:return '[omitted]'
    if isinstance(value,dict):
        return {_safe_value(str(key),secrets,depth+1):_safe_value(child,secrets,depth+1)
            for key,child in value.items() if str(key).casefold() not in OMIT_KEYS}
    if isinstance(value,(list,tuple)):
        return [_safe_value(child,secrets,depth+1) for child in value[:4096]]
    if isinstance(value,str):
        for secret in secrets:value=value.replace(secret,'[redacted]')
        return value
    if isinstance(value,Decimal):return str(value)
    if isinstance(value,float) and not math.isfinite(value):return '[omitted]'
    if value is None or type(value) in (bool,int,float):return value
    return '[omitted]'


def safe_observation(value,secrets):
    if not isinstance(value,dict):return {'status':'observation-unavailable'}
    selected={key:child for key,child in value.items() if key in OBSERVATION_KEYS}
    for provider in ('gemini','bfl','apify'):
        if isinstance(selected.get(provider),dict):
            selected[provider]={key:child for key,child in selected[provider].items() if key in PROVIDER_KEYS}
    return _safe_value(selected,secrets)


@contextmanager
def recorded_observations(report,secrets):
    """Capture returned metadata only, never request arguments or policy objects."""
    from scripts import provider_preflight
    from src.zeitgeist import bfl_tariff,gemini_tariff
    preflight=provider_preflight.run_preflight
    bfl=bfl_tariff.observe_bfl_flux3_tariff
    gemini=gemini_tariff.observe_gemini_tariff
    def metadata(*args,**kwargs):
        value=preflight(*args,**kwargs)
        report['observations']['provider_preflight']=safe_observation(value,secrets)
        return value
    def tariff(original,provider):
        def capture(*args,**kwargs):
            value=original(*args,**kwargs)
            report['observations'].setdefault('price_snapshots',{})[provider]=_safe_value(
                {key:value[key] for key in ('source_url','observed_at','sha256','contract','source_body_sha256') if key in value},secrets)
            return value
        return capture
    with patch.object(provider_preflight,'run_preflight',metadata), \
         patch.object(bfl_tariff,'observe_bfl_flux3_tariff',tariff(bfl,'bfl')), \
         patch.object(gemini_tariff,'observe_gemini_tariff',tariff(gemini,'gemini')):
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply only a previously approved installed policy")
    parser.add_argument('--private-report',type=Path,help='Private diagnostics outside the checkout; never prints observation data')
    args = parser.parse_args(argv)
    report={'schema_version':1,'status':'pending','stage':'planning_or_commit',
        'mode':'apply' if args.apply else 'plan_only','observations':{}}
    path=None;created=False;summary=None
    secrets=_secrets(os.environ)
    def observer(policy,now):
        report['stage']='observing'
        if path is None:
            value=observe_provider_metadata(policy,os.environ,now)
        else:
            with recorded_observations(report,secrets):
                value=observe_provider_metadata(policy,os.environ,now)
            report['observations']['combined']=safe_observation(value,secrets)
        report['stage']='planning_or_commit'
        return value
    try:
        if args.private_report is not None:
            candidate=args.private_report.resolve()
            if args.private_report.is_symlink() or candidate.is_relative_to(ROOT):
                raise PermissionError('Private diagnostics require an external private path')
            path=candidate
            path.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            _private_write(path,report,exclusive=True)
            created=True
        previous_logging=logging.root.manager.disable
        try:
            logging.disable(logging.CRITICAL)
            with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
                result = maintain_spend_control(PrivateFirestoreStore(),
                    expected_policy_sha=os.environ.get("SPEND_POLICY_SHA256", ""), apply=args.apply,
                    observer=observer)
        finally:
            logging.disable(previous_logging)
        summary=result.public_status()
    except BaseException as error:
        summary={'status':'blocked','operation':'maintenance',
                 'month':datetime.now(timezone.utc).strftime('%Y-%m'),**public_failure(error)}
    finally:
        if summary is not None:
            report.update(status=summary['status'],public_diagnostics={key:summary[key]
                for key in ('code','error_type') if key in summary})
        if path is not None and created:
            try:_private_write(path,report)
            except BaseException:
                summary={'status':'blocked','operation':'maintenance',
                    'month':datetime.now(timezone.utc).strftime('%Y-%m'),
                    'code':'private-report-unavailable','error_type':'OSError'}
    print(json.dumps(summary,sort_keys=True))
    return 0 if summary['status']!='blocked' else 2


if __name__ == "__main__":
    raise SystemExit(main())
