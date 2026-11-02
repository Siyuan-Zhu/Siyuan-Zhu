"""Sanitized observation of one daily attempt; never performs recovery writes."""
from datetime import date, datetime, timezone
import json
from pathlib import Path
import re
import tempfile


def receipt_path(root, env):
    return Path(env.get('DAILY_OUTCOME_PATH') or Path(root) / '.daily-outcomes/attempt.json')


def _match(value, pattern):
    value = str(value or '')
    return value if re.fullmatch(pattern, value) else None


def new_outcome(env, now, edition_date):
    run_id = _match(env.get('GITHUB_RUN_ID'), r'[0-9]{1,24}')
    attempt = _match(env.get('GITHUB_RUN_ATTEMPT'), r'[0-9]{1,6}')
    return {'schema_version':1, 'edition_date':str(edition_date),
        'started_at':now.astimezone(timezone.utc).isoformat(),
        'run_id':run_id, 'run_attempt':attempt,
        'source_revision':_match(env.get('GITHUB_SHA'), r'[a-f0-9]{40}'),
        'trigger':env.get('GITHUB_EVENT_NAME') if env.get('GITHUB_EVENT_NAME') in {'schedule','workflow_dispatch'} else 'local',
        'trigger_schedule':_match(env.get('DAILY_TRIGGER_SCHEDULE'), r'[0-9*, /-]{1,64}'),
        'stage':'freshness', 'status':'started', 'canonical_commit':'unknown',
        'git_publication':'not_started', 'publication_id':None,
        'previous_publication_id':None, 'failure_category':None,
        'recovery_action':'inspect_outcome_before_retry',
        'artifact_name':f'publication-recovery-{run_id}-{attempt}' if run_id and attempt else None}


_GUIDANCE = {
    'none':'No action is required for this attempt.',
    'inspect_outcome_before_retry':'Inspect this run and its outcome before choosing a recovery. Do not replay moves or votes.',
    'inspect_canonical_outcome':'Inspect the canonical publication identity and retained run artifacts. Do not resubmit an uncertain mutation.',
    'reconcile_retained_publication':'Reconcile the exact committed publication using its retained artwork and latest visitor state. Do not generate another digest or restore stale state.',
    'inspect_download_tracking':'Inspect the committed publication and download-tracking journal before recovery. Do not blindly repeat an uncertain acknowledgment.',
    'inspect_readiness':'Inspect configuration, authentication, canonical/public identity and asset validation in this run before another attempt.',
    'inspect_spend_control':'Inspect the spending-control outcome. Keep existing limits, settings and uncertain charges intact.',
    'retain_last_digest':'Keep the last complete digest and original date; the next configured run may try the existing bounded route.',
    'inspect_provider':'Inspect the failed source/provider stage and its bounded diagnostics before another attempt.'}
_ENUMS = {
    'stage':{'unknown','startup','freshness','spend_maintenance','generation','readme_build','publication'},
    'status':{'started','already_current','awaiting_publication','failed','requires_reconciliation','published','unknown'},
    'canonical_commit':{'unknown','not_attempted','not_required','not_committed','committed'},
    'git_publication':{'not_started','not_required','published','failed','unknown'},
    'failure_category':{'unknown','readiness','spend_control','photo_quality','photo_provider',
                        'source_collection','permission','publication','startup'},
}


def _iso(value, *, day=False):
    if not isinstance(value,str) or len(value)>40:
        return None
    try:
        if day:
            return value if date.fromisoformat(value).isoformat()==value else None
        parsed=datetime.fromisoformat(value)
        return value if parsed.tzinfo is not None else None
    except ValueError:
        return None


def save_outcome(path, outcome):
    # Only fixed fields and bounded identifiers leave this observer. Never copy
    # arbitrary exception text, journal paths or provider response payloads.
    clean = {key:outcome.get(key) if isinstance(outcome.get(key),str) and outcome.get(key) in values else
             'unknown' for key,values in _ENUMS.items()}
    if outcome.get('failure_category') is None:
        clean['failure_category']=None
    for key,pattern in {'run_id':r'[0-9]{1,24}', 'run_attempt':r'[0-9]{1,6}',
        'source_revision':r'[a-f0-9]{40}', 'publication_id':r'[a-f0-9]{20}',
        'previous_publication_id':r'[a-f0-9]{20}',
        'trigger_schedule':r'[0-9*, /-]{1,64}'}.items():
        clean[key]=_match(outcome.get(key),pattern)
    clean.update(schema_version=1, edition_date=_iso(outcome.get('edition_date'),day=True),
        started_at=_iso(outcome.get('started_at')),
        completed_at=_iso(outcome.get('completed_at')),
        updated_at=datetime.now(timezone.utc).isoformat(),
        trigger=outcome.get('trigger') if isinstance(outcome.get('trigger'),str) and outcome.get('trigger') in {'schedule','workflow_dispatch'} else 'local')
    action=outcome.get('recovery_action')
    clean['recovery_action']=action if isinstance(action,str) and action in _GUIDANCE else 'inspect_outcome_before_retry'
    clean['recovery_guidance']=_GUIDANCE[clean['recovery_action']]
    clean['artifact_name']=(f"publication-recovery-{clean['run_id']}-{clean['run_attempt']}"
        if clean['run_id'] and clean['run_attempt'] else None)
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,delete=False) as handle:
        json.dump(clean,handle,indent=2)
        handle.write('\n')
        temporary=Path(handle.name)
    temporary.replace(path)
    return clean


def observe_committed_publication(root, store, outcome):
    """Positive canonical/art identity evidence, never an inference from exit0."""
    from .publication import verify_publication_bundle, verify_current_pulse
    try:
        manifest=verify_publication_bundle(root,require_current_pulse=True)
        pulse=store.load_pulse_state()
        verify_current_pulse(pulse,manifest)
        if (pulse.get('publication_id')==manifest['publication_id']
                and manifest['date']==outcome['edition_date']):
            outcome.update(canonical_commit='committed',publication_id=manifest['publication_id'])
    except Exception:
        pass  # Observation failure grants no authority and hides no real failure.



def observe_journal(root, outcome):
    """A journal is recovery evidence, not permission to retry its mutation."""
    path = Path(root) / 'data/publication_reconcile.json'
    if not path.exists():
        return False
    marker = {}
    try:
        if path.stat().st_size <= 16384:
            marker = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError):
        pass
    if not isinstance(marker, dict):
        marker = {}
    identity = _match(marker.get('publication_id'), r'[a-f0-9]{20}')
    if identity:
        outcome['publication_id'] = identity
    status=marker.get('status')
    known_committed = isinstance(status,str) and status in {
        'canonical_committed_download_tracking_pending',
        'canonical_committed_local_install_failed', 'canonical_committed_git_pending'}
    outcome.update(status='requires_reconciliation',
        canonical_commit='committed' if known_committed else 'unknown',
        recovery_action='reconcile_retained_publication' if known_committed else 'inspect_canonical_outcome')
    if marker.get('status') == 'canonical_committed_download_tracking_pending':
        outcome['recovery_action'] = 'inspect_download_tracking'
    return True


def record_failure(root, path, outcome, error, *, store=None):
    from urllib.error import HTTPError
    category='permission' if isinstance(error,PermissionError) or (
        isinstance(error,HTTPError) and error.code in {401,403}) else 'unknown'
    # A generation subprocess may already have recorded its typed failure.
    try:
        child = json.loads(Path(path).read_text())
        if (child.get('stage') == outcome['stage'] and child.get('status') == 'failed'
                and child.get('run_id') == outcome.get('run_id')
                and child.get('started_at') == outcome.get('started_at')):
            outcome = child
    except (OSError, ValueError, AttributeError):
        pass
    if store is not None and outcome['stage'] in {'generation','readme_build'}:
        observe_committed_publication(root,store,outcome)
    outcome.update(status='failed', failure_category=outcome.get('failure_category') or category)
    if outcome['stage'] in {'freshness', 'spend_maintenance'}:
        outcome['canonical_commit'] = 'not_attempted'
    if not observe_journal(root,outcome) and outcome['stage'] in {'generation','readme_build'}:
        if outcome['canonical_commit']=='committed':
            outcome.update(status='requires_reconciliation',recovery_action='reconcile_retained_publication')
        elif outcome['stage']=='readme_build':
            outcome['recovery_action']='inspect_canonical_outcome'
    if outcome['stage']=='freshness' and outcome['recovery_action']=='inspect_outcome_before_retry':
        outcome.update(failure_category=category if category!='unknown' else 'readiness',
                       recovery_action='inspect_readiness')
    if outcome['stage']=='spend_maintenance':
        outcome.update(failure_category='spend_control',recovery_action='inspect_spend_control')
    save_outcome(path, outcome)


def record_generation_failure(root, error, env):
    """Preserve only known semantic error categories from the actual CLI error."""
    path=receipt_path(root,env)
    if not path.is_file():
        return
    try:
        outcome=json.loads(path.read_text())
    except (OSError,ValueError,RecursionError):
        return
    if not isinstance(outcome,dict) or outcome.get('stage')!='generation':
        return
    from .visual_router import PhotoSelectionUnavailable, PhotoSearchUnavailable, PhotoCriticUnavailable
    from .collectors import IngestionUnavailable
    from .spend_guard import SpendBlocked
    from urllib.error import HTTPError
    category='unknown'
    known_precommit=False
    if isinstance(error,PhotoSelectionUnavailable):
        category='photo_quality';known_precommit=True
    elif isinstance(error,(PhotoSearchUnavailable,PhotoCriticUnavailable)):
        category='photo_provider';known_precommit=True
    elif isinstance(error,IngestionUnavailable):
        category='source_collection';known_precommit=True
    elif isinstance(error,SpendBlocked):
        category='spend_control';known_precommit=True
    elif isinstance(error,PermissionError) or isinstance(error,HTTPError) and error.code in {401,403}:
        category='permission'
    outcome.update(status='failed',failure_category=category)
    journal=observe_journal(root,outcome)
    if not journal and known_precommit:
        outcome.update(canonical_commit='not_committed',
            recovery_action='retain_last_digest' if category=='photo_quality' else
                            'inspect_spend_control' if category=='spend_control' else 'inspect_provider')
    save_outcome(path,outcome)


def complete_outcome(root, env, *, now=None):
    """Finish observed workflow outcomes, without reading credentials or retrying."""
    from zoneinfo import ZoneInfo
    now=now or datetime.now(timezone.utc)
    path=receipt_path(root,env)
    try:
        if path.stat().st_size>65536:
            raise ValueError('Oversized outcome')
        outcome=json.loads(path.read_text())
        if not isinstance(outcome,dict):
            raise ValueError('Invalid outcome')
        context=new_outcome(env,now,now.astimezone(ZoneInfo('America/New_York')).date())
        for key in ('run_id','run_attempt','source_revision'):
            if context[key] is not None and outcome.get(key)!=context[key]:
                raise ValueError('Receipt belongs to another attempt')
    except (OSError,ValueError,RecursionError):
        outcome=new_outcome(env,now,now.astimezone(ZoneInfo('America/New_York')).date())
        outcome.update(stage='startup',status='unknown',failure_category='startup',
                       git_publication='unknown')
        outcome['completed_at']=now.astimezone(timezone.utc).isoformat()
        return save_outcome(path,outcome)
    outcome=save_outcome(path,outcome)
    digest=env.get('DAILY_DIGEST_OUTCOME')
    publication=env.get('DAILY_PUBLISH_OUTCOME')
    journal=observe_journal(root,outcome)
    if publication=='failure' or publication=='cancelled':
        outcome.update(stage='publication',failure_category='publication',
            git_publication='failed' if publication=='failure' else 'unknown',
            status='requires_reconciliation' if outcome.get('canonical_commit')=='committed' else 'unknown',
            recovery_action='reconcile_retained_publication' if outcome.get('canonical_commit')=='committed' else 'inspect_canonical_outcome')
    elif publication=='success' and digest=='success' and not journal:
        # The publisher succeeded, but a green job alone does not prove that
        # the target edition/artifact exists. Validate the actual accepted bundle.
        from .publication import verify_publication_bundle
        try:
            manifest=verify_publication_bundle(root,require_current_pulse=True)
            if (manifest['date']!=outcome.get('edition_date')
                    or manifest['publication_id']!=outcome.get('publication_id')):
                raise ValueError('Published edition differs from this attempt')
        except (OSError,ValueError,KeyError,TypeError,RecursionError):
            outcome.update(status='unknown',stage='publication',git_publication='unknown',
                failure_category='publication',recovery_action='inspect_canonical_outcome')
        else:
            outcome.update(status='published',stage='publication',git_publication='published',
                           failure_category=None,recovery_action='none')
    elif digest=='success' and publication=='skipped' and outcome.get('status')=='already_current' and not journal:
        pass
    elif outcome.get('status') not in {'failed','requires_reconciliation'}:
        outcome.update(status='requires_reconciliation' if outcome.get('canonical_commit')=='committed' else 'unknown',
            git_publication='not_started' if publication=='skipped' else 'unknown',
            recovery_action='reconcile_retained_publication' if outcome.get('canonical_commit')=='committed' else 'inspect_canonical_outcome')
    outcome['completed_at']=now.astimezone(timezone.utc).isoformat()
    return save_outcome(path,outcome)
