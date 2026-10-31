#!/usr/bin/env python3
"""Prepare pinned canonical promotion; explicit production apply is separately gated.

No inference, JSON seeding, provider spend operations, or automatic execution.
All snapshot/plan/journal files are PRIVATE and must stay outside Git.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from scripts.migrate_release_state import PROFILE_STATE_DOC_IDS, _verified_backup, _sha256_file
from src.game_engine import create_initial_game_state
from src.leaderboard.elo_ledger import create_empty_ledger
from src.state_store import (PublicationConflict, PublicationOutcomeUnknown,
    _FirestoreRestClient, _py_to_firestore_value, _firestore_error_status)
from src.zeitgeist.publication import verify_publication_bundle, verify_current_pulse

TARGET={'project':'git-readme-features','database':'(default)','collection_prefix':''}
CONFIRM='Siyuan-Zhu/Siyuan-Zhu:git-readme-features:(default):unprefixed'
COLLECTIONS={'profile_state','pulse_feed','pulse_history','pulse_votes','vote_rate_limits','processed_events'}
ZERO={'spot_on':0,'too_sunny':0,'too_heavy':0}
MAX_DOCUMENTS=200
MAX_BYTES=5_000_000
PUBLIC_JSON_PATHS=('data/zeitgeist_state.json','data/pulse_feed.json')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _validate_snapshot(value, *, source=False):
    expected=dict(TARGET)
    if source:
        prefix=value.get('target',{}).get('collection_prefix','')
        if not re.fullmatch(r'staging_[A-Za-z0-9_]+_',prefix): raise ValueError('Source must be an explicit staging namespace')
        expected['collection_prefix']=prefix
    if value.get('schema_version')!=1 or value.get('target')!=expected: raise ValueError('Snapshot target binding mismatch')
    documents=value.get('documents')
    if not isinstance(documents,dict) or len(documents)>MAX_DOCUMENTS: raise ValueError('Snapshot scope exceeds limit')
    for path,record in documents.items():
        if not isinstance(path,str) or path.count('/')!=1: raise ValueError('Invalid document path')
        collection,doc=path.split('/')
        if collection not in COLLECTIONS or not re.fullmatch(r'[A-Za-z0-9_.:-]+',doc): raise ValueError('Unmanaged document path')
        if collection=='profile_state' and doc not in PROFILE_STATE_DOC_IDS: raise ValueError('Unknown profile document')
        if collection=='pulse_feed' and doc!='v1': raise ValueError('Unknown feed document')
        if (not isinstance(record,dict) or set(record)!={'data','update_time'} or
            not isinstance(record['data'],dict) or not isinstance(record['update_time'],str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?Z',record['update_time'])):
            raise ValueError('Exact canonical payload and update_time required')
    if len(json.dumps(value).encode())>MAX_BYTES: raise ValueError('Snapshot byte ceiling exceeded')
    return documents


def _zero_votes(value):
    result=copy.deepcopy(value)
    if isinstance(result,dict):
        for key,item in list(result.items()):
            result[key]=({} if key=='vote_rate_limits' else copy.deepcopy(ZERO)) if key in {'mood_votes','visitor_mood_tally','vote_rate_limits'} else _zero_votes(item)
    elif isinstance(result,list): result=[_zero_votes(item) for item in result]
    return result


def prepare_plan(source,target,*,bundle_root,source_sha,source_run,source_snapshot_sha,
                 target_snapshot_sha,manifest_sha,game_policy,operation_id,source_backup_sha,target_backup_sha,journal_path):
    src=_validate_snapshot(source,source=True); before=_validate_snapshot(target)
    if any(not re.fullmatch(r'[a-f0-9]{64}',v) for v in (source_backup_sha,target_backup_sha)): raise ValueError('Verified backup digests required')
    if digest(source)!=source_snapshot_sha or digest(target)!=target_snapshot_sha: raise ValueError('Snapshot digest mismatch')
    if not re.fullmatch(r'[a-f0-9]{40}',source_sha) or not re.fullmatch(r'[0-9]+',source_run): raise ValueError('Pinned source SHA and run required')
    if not re.fullmatch(r'[a-f0-9]{32}',operation_id): raise ValueError('Unique operation ID required')
    if game_policy not in {'fresh','preserve-target-game'}: raise ValueError('Explicit game policy required')
    root=Path(bundle_root).resolve()
    manifest=verify_publication_bundle(root,require_current_pulse=True)
    if digest(manifest)!=manifest_sha: raise ValueError('Manifest digest mismatch')
    pulse=copy.deepcopy(src['profile_state/daily_pulse']['data'])
    verify_current_pulse(pulse,manifest)
    if pulse.get('publication_id')!=manifest['publication_id']: raise ValueError('Canonical publication ID mismatch')
    date=manifest['date']; entry=copy.deepcopy(src['pulse_history/'+date]['data'])
    # The real feed/history DTO carries date/word/gloss/medium/identity, while
    # layout is stored on the canonical daily pulse and bound artwork manifest.
    # Validate their composed view without altering the accepted history bytes.
    # An explicit conflicting history layout still fails in the verifier.
    verify_current_pulse({'curatorial_archetype':manifest['archetype'], 'published_pulse':entry},manifest)
    if entry.get('publication_id')!=manifest['publication_id']: raise ValueError('History publication ID mismatch')
    feed=src['pulse_feed/v1']['data']
    if entry not in feed.get('entries',[]): raise ValueError('Canonical feed/history diverge')
    # Compare parsed values and hash the SAME accepted bytes; preserve the source
    # bundle separately from the later fresh-launch public projection.
    public_json_bytes={path:(root/path).read_bytes() for path in PUBLIC_JSON_PATHS}
    local_entry=next((e for e in json.loads(public_json_bytes['data/pulse_feed.json'])['entries'] if e.get('date','')[:10]==date),None)
    if local_entry!=entry: raise ValueError('Bundle credit/content differs from canonical history')
    if json.loads(public_json_bytes['data/zeitgeist_state.json'])!=pulse: raise ValueError('Bundle state differs from canonical source')
    # No canonical state is derived from a checkout game/leaderboard fixture.
    pulse=_zero_votes(pulse); entry=_zero_votes(entry)
    pulse.update(mood_votes=copy.deepcopy(ZERO),visitor_mood_tally=copy.deepcopy(ZERO),published_pulse=entry)
    feed_header={key:copy.deepcopy(feed[key]) for key in ('updated_at',) if key in feed}
    after={'profile_state/daily_pulse':pulse,'pulse_history/'+date:entry,
           'pulse_feed/v1':dict(feed_header,version='2.0',schema_version='2.0',entries=[entry]),
           'pulse_votes/'+date:copy.deepcopy(ZERO),
           'profile_state/orbit_roster':copy.deepcopy(src.get('profile_state/orbit_roster',{'data':{'handles':{}}})['data'])}
    if game_policy=='fresh':
        game=create_initial_game_state(round_id=1); game.pop('board_state',None)
        after.update({'profile_state/active_game':game,'profile_state/elo_ledger':create_empty_ledger(),
           'profile_state/bot_memory':{'schema_version':3,'state_version':0,'games_learned':0,'states':{}},
           'profile_state/game_rate_limits':{'state_version':0,'users':{}}})
    else:
        for key in ('active_game','elo_ledger','bot_memory','game_rate_limits'):
            path='profile_state/'+key
            if path not in before: raise ValueError('Preserve policy requires complete existing game domain')
            after[path]=copy.deepcopy(before[path]['data'])
        after.update({path:copy.deepcopy(value['data']) for path,value in before.items() if path.startswith('processed_events/')})
    intent={'schema_version':1,'target':TARGET,'source_target':source['target'],
      'source_sha':source_sha,'source_run':source_run,'source_snapshot_sha256':source_snapshot_sha,
      'target_snapshot_sha256':target_snapshot_sha,'manifest_sha256':manifest_sha,
      'source_backup_manifest_sha256':source_backup_sha,'target_backup_manifest_sha256':target_backup_sha,
      'publication_id':manifest['publication_id'],'operation_id':operation_id,'game_policy':game_policy,
      'operational_records_policy':'archive_exact_backup_then_remove_vote_records_and_'+('test_game_records' if game_policy=='fresh' else 'preserve_game_receipts'),
      'before':copy.deepcopy(before),'after':after,
      'journal_path':str(Path(journal_path).resolve()),
      'public_json_sha256':{path:hashlib.sha256(raw).hexdigest() for path,raw in public_json_bytes.items()},
      'assets':{path:_sha256_file(root/path) for path in manifest['files']}}
    receipt='processed_events/release_promotion_'+operation_id
    if receipt in before: raise ValueError('Operation receipt already exists')
    after[receipt]={'kind':'release_promotion','intent_sha256':digest(intent),'publication_id':manifest['publication_id']}
    if len(set(before)|set(after))>MAX_DOCUMENTS or len(json.dumps(intent).encode())>MAX_BYTES: raise ValueError('Atomic promotion exceeds bounded scope')
    return {'plan_sha256':digest(intent),'plan':intent}


def _validate_plan(envelope):
    plan=envelope['plan']
    if envelope.get('plan_sha256')!=digest(plan) or plan.get('target')!=TARGET or plan.get('schema_version')!=1: raise ValueError('Plan digest/target mismatch')
    public_json=plan.get('public_json_sha256')
    if not isinstance(public_json,dict) or set(public_json)!=set(PUBLIC_JSON_PATHS) or any(not isinstance(value,str) or not re.fullmatch(r'[a-f0-9]{64}',value) for value in public_json.values()): raise ValueError('Both accepted public JSON byte hashes required')
    _validate_snapshot({'schema_version':1,'target':TARGET,'documents':plan['before']})
    _validate_snapshot({'schema_version':1,'target':TARGET,'documents':{p:{'data':v,'update_time':'2026-10-30T00:00:00Z'} for p,v in plan['after'].items()}})
    if len(set(plan['before'])|set(plan['after']))>MAX_DOCUMENTS or len(json.dumps(plan).encode())>MAX_BYTES: raise ValueError('Plan exceeds bound')
    return plan


def _private_write(path,payload,*,exclusive=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if exclusive:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'wb') as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
    else:
        fd,name=tempfile.mkstemp(dir=path.parent,prefix='.promotion-')
        try:
            with os.fdopen(fd,'wb') as handle: handle.write(raw); handle.flush(); os.fsync(handle.fileno())
            os.replace(name,path)
        finally:
            if os.path.exists(name): os.unlink(name)
    directory=os.open(path.parent,os.O_RDONLY)
    try: os.fsync(directory)
    finally: os.close(directory)


def _check_gate(envelope,expected_sha,confirmation):
    plan=_validate_plan(envelope)
    if confirmation!=CONFIRM or expected_sha!=envelope['plan_sha256']: raise PermissionError('Exact production target and plan digest confirmation required')
    return plan


def _capture_after(envelope,adapter):
    plan=_validate_plan(envelope); snapshot=adapter.snapshot(); docs=_validate_snapshot(snapshot)
    if {p:v['data'] for p,v in docs.items()}!=plan['after']: raise ValueError('Canonical readback differs from owned promotion; keep writers frozen')
    # The random, plan-bound receipt is in the SAME atomic commit, including unknown acknowledgements.
    receipt='processed_events/release_promotion_'+plan['operation_id']
    if receipt not in docs: raise ValueError('No atomic promotion ownership receipt')
    commit_revision=docs[receipt]['update_time']
    for path,record in docs.items():
        original=plan['before'].get(path)
        allowed={commit_revision}
        # Firestore may keep updateTime unchanged for an exact no-op update.
        if original and original['data']==plan['after'][path]: allowed.add(original['update_time'])
        if record['update_time'] not in allowed: raise ValueError('Promotion documents have intervening revisions')
    return snapshot


def execute(envelope,adapter,journal_path,*,expected_sha,confirmation):
    plan=_check_gate(envelope,expected_sha,confirmation)
    if str(Path(journal_path).resolve())!=plan['journal_path']: raise ValueError('Journal path differs from approved plan')
    if Path(journal_path).exists(): raise ValueError('Journal exists; reconcile instead of redispatch')
    snapshot=adapter.snapshot(); _validate_snapshot(snapshot)
    if snapshot['documents']!=plan['before']: raise ValueError('Target changed since approved backup/snapshot')
    journal={'plan_sha256':envelope['plan_sha256'],'status':'partial_or_unknown','direction':'apply'}
    # Never repeat a dispatched operation, including after cancellation or a broken acknowledgement.
    _private_write(journal_path,journal,exclusive=True)
    try:
        adapter.commit_exact([(path,plan['before'].get(path),plan['after'].get(path)) for path in sorted(set(plan['before'])|set(plan['after']))])
        journal.update(status='complete',after_snapshot=_capture_after(envelope,adapter))
        _private_write(journal_path,journal)
    except BaseException:
        # Initial fsynced journal remains unknown even if the process cannot write again.
        raise
    return {'status':journal['status'],'plan_sha256':envelope['plan_sha256']}


def reconcile(envelope,adapter,journal_path):
    plan=_validate_plan(envelope)
    if str(Path(journal_path).resolve())!=plan['journal_path']: raise ValueError('Journal path differs from approved plan')
    journal=json.loads(Path(journal_path).read_text())
    if journal.get('plan_sha256')!=envelope['plan_sha256'] or journal.get('direction')!='apply' or journal.get('status')!='partial_or_unknown': raise ValueError('No matching unknown apply to reconcile')
    journal.update(status='complete',after_snapshot=_capture_after(envelope,adapter))
    _private_write(journal_path,journal)
    return {'status':'complete'}


def rollback(envelope,adapter,journal_path,*,expected_sha,confirmation,dry_run=False):
    plan=_check_gate(envelope,expected_sha,confirmation)
    if str(Path(journal_path).resolve())!=plan['journal_path']: raise ValueError('Journal path differs from approved plan')
    journal=json.loads(Path(journal_path).read_text())
    if journal.get('plan_sha256')!=envelope['plan_sha256'] or journal.get('status')!='complete': raise ValueError('Complete owned promotion journal required')
    current=adapter.snapshot()
    if current!=journal.get('after_snapshot'): raise ValueError('Rollback refuses newer documents or revisions; preserve accepted events')
    if dry_run: return {'status':'rollback_ready'}
    journal.update(status='partial_or_unknown',direction='rollback'); _private_write(journal_path,journal)
    adapter.commit_exact([(path,current['documents'].get(path),plan['before'].get(path,{}).get('data')) for path in sorted(set(current['documents'])|set(plan['before']))])
    readback=adapter.snapshot()
    if {p:v['data'] for p,v in readback['documents'].items()}!={p:v['data'] for p,v in plan['before'].items()}: raise ValueError('Rollback readback mismatch')
    journal.update(status='rolled_back',rollback_snapshot=readback); _private_write(journal_path,journal)
    return {'status':'rolled_back'}


class CanonicalAdapter:
    """REST-only exact commit: no update mask, no merge, retries or sequential fallback."""
    def __init__(self,prefix=''):
        if prefix and not re.fullmatch(r'staging_[A-Za-z0-9_]+_',prefix): raise ValueError('Invalid staging capture prefix')
        self.prefix=prefix; self.client=_FirestoreRestClient(TARGET['project'],TARGET['database'])
    def snapshot(self):
        def read():
            docs={}
            for col in sorted(COLLECTIONS):
                collection=self.client.collection(self.prefix+col)
                if col=='profile_state': snapshots=[collection.document(k).get() for k in PROFILE_STATE_DOC_IDS]
                elif col=='pulse_feed': snapshots=[collection.document('v1').get()]
                else: snapshots=collection.stream()
                for snap in snapshots:
                    if snap.exists:
                        docs[col+'/'+snap.id]={'data':snap.to_dict(),'update_time':str(snap.update_time)}
                        if len(docs)>MAX_DOCUMENTS: raise ValueError('Snapshot scope exceeds ceiling')
            return {'schema_version':1,'target':dict(TARGET,collection_prefix=self.prefix),'documents':docs}
        first=read(); second=read()
        if first!=second: raise ValueError('Canonical snapshot changed while reading; freeze and drain writers')
        _validate_snapshot(second,source=bool(self.prefix)); return second
    def commit_exact(self,writes):
        if self.prefix: raise PermissionError('Promotion target must be unprefixed')
        _validate_snapshot({'schema_version':1,'target':TARGET,'documents':{path:before for path,before,after in writes if before}})
        _validate_snapshot({'schema_version':1,'target':TARGET,'documents':{path:{'data':after,'update_time':'2026-10-30T00:00:00Z'} for path,before,after in writes if after is not None}})
        if len({path for path,_,_ in writes})!=len(writes): raise ValueError('Duplicate atomic document path')
        wire=[]; base=self.client._base_url().split('/v1/',1)[1]
        for path,before,after in writes:
            precondition={'updateTime':before['update_time']} if before else {'exists':False}
            if after is None:
                if not before: raise ValueError('Cannot delete an absent document')
                wire.append({'delete':base+'/'+path,'currentDocument':precondition})
            else:
                wire.append({'update':{'name':base+'/'+path,'fields':{k:_py_to_firestore_value(v) for k,v in after.items()}},'currentDocument':precondition})
        if len(wire)>MAX_DOCUMENTS or len(json.dumps(wire).encode())>MAX_BYTES: raise ValueError('Atomic commit exceeds ceiling')
        request=urllib.request.Request(self.client._base_url()+':commit',data=json.dumps({'writes':wire}).encode(),headers={'Authorization':'Bearer '+self.client._resolve_token(),'Content-Type':'application/json'},method='POST')
        try:
            response=urllib.request.urlopen(request,timeout=15)
        except urllib.error.HTTPError as exc:
            conflict=exc.code in (409,412)
            if exc.code==400:
                try: conflict=_firestore_error_status(json.loads(exc.read(65537))) in {'FAILED_PRECONDITION','ABORTED'}
                except Exception: pass
            if conflict: raise PublicationConflict('Conditional promotion rejected') from None
            if 400<=exc.code<500: raise RuntimeError('Promotion rejected before commit') from None
            raise PublicationOutcomeUnknown('Promotion outcome unknown; reconcile') from None
        except BaseException:
            raise PublicationOutcomeUnknown('Promotion outcome unknown; reconcile') from None
        try:
            with response: acknowledgement=json.loads(response.read())
            results=acknowledgement.get('writeResults')
            # Delete results have no updateTime; commitTime still binds one atomic acknowledgement.
            if not acknowledgement.get('commitTime') or not isinstance(results,list) or len(results)!=len(wire): raise ValueError('Invalid acknowledgement')
            for result,(_,_,after) in zip(results,writes):
                if not isinstance(result,dict) or after is not None and not result.get('updateTime'): raise ValueError('Incomplete acknowledgement')
        except BaseException:
            raise PublicationOutcomeUnknown('Promotion acknowledgement unknown; reconcile') from None


def _backup_binding(directory,snapshot):
    from types import SimpleNamespace
    identity=snapshot['target']
    store=SimpleNamespace(_project_id=identity['project'],_database_id=identity['database'],collection_prefix=identity['collection_prefix'])
    _,export=_verified_backup(Path(directory),store)
    documents={col:{} for col in COLLECTIONS}
    for path,record in snapshot['documents'].items():
        col,doc=path.split('/'); documents[col][doc]=record['data']
    if export['collections']!=documents: raise ValueError('Backup and revision snapshot differ')
    return _sha256_file(Path(directory)/'manifest.json')


def _durable_location(path):
    original=Path(path)
    if original.is_symlink(): raise ValueError('Private output cannot be a symlink')
    path=original.resolve()
    if path.is_relative_to(ROOT) or path.is_relative_to(Path('/tmp').resolve()) or path.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError('Use durable private storage outside checkout and temporary directories')
    if path.is_symlink(): raise ValueError('Private output cannot be a symlink')
    return path


def _bundle_bound(envelope,root):
    plan=_validate_plan(envelope); root=Path(root).resolve()
    manifest=verify_publication_bundle(root,require_current_pulse=True)
    if digest(manifest)!=plan['manifest_sha256'] or any(_sha256_file(root/path)!=value for path,value in {**plan['assets'],**plan['public_json_sha256']}.items()): raise ValueError('Publication bundle changed after approval')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    capture=sub.add_parser('capture',help='Explicit read-only canonical snapshot; never run automatically')
    capture.add_argument('--collection-prefix',required=True)
    capture.add_argument('--output',type=Path,required=True)
    prepare=sub.add_parser('prepare',help='Offline plan only; verified backup and revision snapshots required')
    for name in ('source-snapshot','target-snapshot','source-backup','target-backup','bundle-root','output'):
        prepare.add_argument('--'+name,type=Path,required=True)
    for name in ('source-snapshot-sha','target-snapshot-sha','manifest-sha','source-sha','source-run','operation-id'):
        prepare.add_argument('--'+name,required=True)
    prepare.add_argument('--journal',type=Path,required=True)
    prepare.add_argument('--game-policy',choices=('fresh','preserve-target-game'),required=True)
    for command in ('apply','rollback','reconcile'):
        action=sub.add_parser(command)
        action.add_argument('--plan',type=Path,required=True)
        action.add_argument('--journal',type=Path,required=True)
        action.add_argument('--expected-plan-sha',required=True)
        action.add_argument('--confirm-target',required=True)
        action.add_argument('--bundle-root',type=Path,required=True)
        for name in ('source-snapshot','target-snapshot','source-backup','target-backup'):
            action.add_argument('--'+name,type=Path,required=True)
        if command=='rollback': action.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='capture':
            snapshot=CanonicalAdapter(args.collection_prefix).snapshot()
            _private_write(_durable_location(args.output),snapshot,exclusive=True)
            report={'status':'captured','snapshot_sha256':digest(snapshot)}
        elif args.command=='prepare':
            source=json.loads(args.source_snapshot.read_text()); target=json.loads(args.target_snapshot.read_text())
            source_backup_sha=_backup_binding(_durable_location(args.source_backup),source)
            target_backup_sha=_backup_binding(_durable_location(args.target_backup),target)
            envelope=prepare_plan(source,target,bundle_root=args.bundle_root,source_sha=args.source_sha,source_run=args.source_run,
                source_snapshot_sha=args.source_snapshot_sha,target_snapshot_sha=args.target_snapshot_sha,manifest_sha=args.manifest_sha,
                game_policy=args.game_policy,operation_id=args.operation_id,
                source_backup_sha=source_backup_sha,target_backup_sha=target_backup_sha,journal_path=_durable_location(args.journal))
            _private_write(_durable_location(args.output),envelope,exclusive=True)
            report={'status':'prepared_no_writes','plan_sha256':envelope['plan_sha256']}
        else:
            envelope=json.loads(args.plan.read_text()); _check_gate(envelope,args.expected_plan_sha,args.confirm_target)
            _bundle_bound(envelope,args.bundle_root)
            for role in ('source','target'):
                snapshot=json.loads(getattr(args,role+'_snapshot').read_text())
                if digest(snapshot)!=envelope['plan'][role+'_snapshot_sha256']: raise ValueError('Approved snapshot substituted')
                if _backup_binding(_durable_location(getattr(args,role+'_backup')),snapshot)!=envelope['plan'][role+'_backup_manifest_sha256']: raise ValueError('Approved backup substituted')
            journal=_durable_location(args.journal); adapter=CanonicalAdapter()
            if args.command=='apply': report=execute(envelope,adapter,journal,expected_sha=args.expected_plan_sha,confirmation=args.confirm_target)
            elif args.command=='rollback': report=rollback(envelope,adapter,journal,expected_sha=args.expected_plan_sha,confirmation=args.confirm_target,dry_run=args.dry_run)
            else: report=reconcile(envelope,adapter,journal)
        print(json.dumps(report,sort_keys=True)); return 0
    except BaseException as exc:
        # Never serialize exception text, URLs, auth bodies or canonical state.
        print(json.dumps({'status':'blocked','error_type':type(exc).__name__})); return 1


if __name__=='__main__': raise SystemExit(main())
