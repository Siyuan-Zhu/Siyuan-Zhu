"""Workflow completion must report Git failure separately from generation."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def complete(tmp_path, digest, publication, record=None, **extra_env):
    path = tmp_path / 'receipt.json'
    if record is not None:
        path.write_text(json.dumps(record))
    env = dict(PATH=os.environ.get('PATH',''), LANG='en_US.UTF-8', DAILY_OUTCOME_PATH=str(path), DAILY_DIGEST_OUTCOME=digest,
        DAILY_PUBLISH_OUTCOME=publication, GITHUB_RUN_ID='1234', GITHUB_RUN_ATTEMPT='1',
        GITHUB_SHA='a'*40, **extra_env)
    result = subprocess.run([sys.executable,str(ROOT/'scripts/complete_daily_outcome.py'),
        '--repo-root',str(tmp_path)], env=env, capture_output=True,text=True,check=True)
    return json.loads(path.read_text()),result.stdout


def accepted_record():
    return {'schema_version':1,'edition_date':'2026-11-02',
        'started_at':'2026-11-02T07:00:00+00:00','run_id':'1234','run_attempt':'1',
        'source_revision':'a'*40,'trigger':'schedule','trigger_schedule':'0 2 * * *',
        'stage':'publication','status':'awaiting_publication','canonical_commit':'committed',
        'git_publication':'not_started','publication_id':'b'*20,
        'previous_publication_id':'c'*20, 'failure_category':None,
        'recovery_action':'reconcile_retained_publication'}


def test_successful_generation_with_failed_git_retains_reconciliation_action(tmp_path):
    outcome,stdout=complete(tmp_path,'success','failure',accepted_record())
    assert outcome['status']=='requires_reconciliation'
    assert outcome['canonical_commit']=='committed'
    assert outcome['git_publication']=='failed'
    assert outcome['publication_id']=='b'*20
    assert outcome['recovery_action']=='reconcile_retained_publication'
    assert 'Do not generate another digest' in outcome['recovery_guidance']


def test_malformed_receipt_and_secret_fields_never_escape_completion(tmp_path):
    record=accepted_record() | {'status':{'raw':'private-status-payload'},
        'failure_category':['private-provider-error'], 'publication_id':'private-bad-id',
        'private_note':'never-copy-this', 'recovery_path':'/private/owner-path'}
    outcome,stdout=complete(tmp_path,'cancelled','skipped',record)
    text=json.dumps(outcome)+stdout
    assert outcome['status']=='requires_reconciliation'
    assert outcome['publication_id'] is None
    assert all(value not in text for value in ['private-status-payload','private-provider-error',
        'private-bad-id','never-copy-this','/private/owner-path'])


def test_missing_receipt_is_an_unknown_startup_outcome_with_completion_time(tmp_path):
    outcome,stdout=complete(tmp_path,'failure','skipped')
    assert outcome['status']=='unknown' and outcome['stage']=='startup'
    assert outcome['canonical_commit']=='unknown'
    assert outcome['git_publication']=='unknown'
    assert datetime.fromisoformat(outcome['completed_at']).tzinfo is not None
    assert outcome['recovery_action']=='inspect_outcome_before_retry'


def test_green_workflow_without_matching_accepted_bundle_is_not_publication_success(tmp_path):
    outcome,stdout=complete(tmp_path,'success','success',accepted_record())
    assert outcome['status']=='unknown'
    assert outcome['git_publication']=='unknown'
    assert outcome['recovery_action']=='inspect_canonical_outcome'


def test_matching_accepted_bundle_and_successful_publisher_close_the_attempt(tmp_path):
    import shutil
    for name in ('assets','data'):
        shutil.copytree(ROOT/name,tmp_path/name)
    state=json.loads((tmp_path/'data/zeitgeist_state.json').read_text())
    record=accepted_record() | {'edition_date':state['date'],'publication_id':state['publication_id']}
    outcome,stdout=complete(tmp_path,'success','success',record)
    assert outcome['status']=='published' and outcome['git_publication']=='published'
    assert outcome['publication_id']==state['publication_id']
    assert outcome['recovery_action']=='none'


def test_deep_accepted_json_finalizes_as_unknown_without_leaking_payload(tmp_path):
    import shutil
    for name in ('assets','data'):
        shutil.copytree(ROOT/name,tmp_path/name)
    state=json.loads((tmp_path/'data/zeitgeist_state.json').read_text())
    record=accepted_record() | {'edition_date':state['date'],'publication_id':state['publication_id']}
    (tmp_path/'data/pulse_feed.json').write_bytes(
        b'[' * 50000 + b'"never-export-nested-payload"' + b']' * 50000)
    outcome,stdout=complete(tmp_path,'success','success',record)
    assert outcome['status']=='unknown'
    assert outcome['stage']=='publication'
    assert outcome['git_publication']=='unknown'
    assert outcome['recovery_action']=='inspect_canonical_outcome'
    assert 'never-export-nested-payload' not in json.dumps(outcome)+stdout


def test_cancelled_generation_preserves_uncertain_outcome(tmp_path):
    record=accepted_record() | {'stage':'generation','status':'started','canonical_commit':'unknown'}
    outcome,stdout=complete(tmp_path,'cancelled','skipped',record)
    assert outcome['status']=='unknown' and outcome['canonical_commit']=='unknown'
    assert outcome['recovery_action']=='inspect_canonical_outcome'


def test_stale_attempt_receipt_is_not_attributed_to_the_current_run(tmp_path):
    record=accepted_record() | {'run_attempt':'99','source_revision':'d'*40}
    outcome,stdout=complete(tmp_path,'success','failure',record)
    assert outcome['status']=='unknown' and outcome['stage']=='startup'
    assert outcome['run_attempt']=='1' and outcome['source_revision']=='a'*40
    assert outcome['publication_id'] is None
