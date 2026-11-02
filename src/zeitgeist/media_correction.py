"""Exact, provider-free preparation for the accepted November1 source correction.

This is a reviewed one-edition operation, not image selection or a migration
framework. The retained statement came from the official source page inspection;
no authenticated API response, AI-free authorship or toolchain is fabricated.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from .media_provenance import public_media_provenance
from .publication import verify_publication_bundle

DATE = '2026-11-01'
SOURCE_ID = 'WOA3QKFjlo8'
SOURCE_URL = 'https://unsplash.com/photos/' + SOURCE_ID
STATEMENT = 'The artwork is a 3D particle system'
ROOT = Path(__file__).resolve().parents[2]
CODE_FILES = ('scripts/refresh_studio_footer.py', 'scripts/build_readme.py',
    'src/zeitgeist/media_correction.py', 'src/zeitgeist/media_provenance.py',
    'src/zeitgeist/publication.py', 'src/zeitgeist/publication_commit.py',
    'src/zeitgeist/public_schema.py', 'src/zeitgeist/curatorial_compositor.py',
    'src/state_store.py', '.github/workflows/daily_zeitgeist.yml',
    'requirements.txt', 'requirements-ci.lock',
    'assets/fonts/IBMPlexMono-Regular.ttf')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def evidence():
    record = {'version':1, 'classification':'DIGITAL_ARTWORK',
        'basis':'source_declared_digital','provider':'unsplash','asset_id':SOURCE_ID,
        'source_url':SOURCE_URL,'camera_make':'','camera_model':'','allowance_exception':True,
        'source_claim':{'kind':'creation_declaration','process':'3d_particle_system','statement':STATEMENT},
        'source_statement_sha256':hashlib.sha256(STATEMENT.encode()).hexdigest()}
    return public_media_provenance(record, medium='DIGITAL_ARTWORK')


def semantic(value, kind):
    """Only current-edition vote projections are mutable during this correction."""
    value = copy.deepcopy(value)
    if kind in ('state', 'history'):
        value.pop('mood_votes', None)
        if kind == 'state':
            value.pop('visitor_mood_tally', None)
            value.pop('_needs_git_reconcile', None)
            if isinstance(value.get('published_pulse'), dict):
                value['published_pulse'].pop('mood_votes', None)
    elif kind == 'feed':
        for item in value.get('entries', []):
            if isinstance(item, dict) and str(item.get('date', ''))[:10] == DATE:
                item.pop('mood_votes', None)
    return value


def context(store):
    from scripts.build_readme import resolve_repo_slug
    renderer={'repository':resolve_repo_slug(),'game_controls_enabled':os.environ.get('COMMUNITY_GAME_ENABLED','true')=='true'}
    if store is None:
        return {'authority':'local_preview',**renderer}
    client = store._get_client()
    return {'authority':'canonical', 'project':str(getattr(client,'project_id',None) or getattr(client,'project','')),
        'database':str(getattr(client,'database_id',None) or getattr(store,'_database_id','')),
        'collection_prefix':store.collection_prefix,**renderer}


def documents(root, store=None):
    if store is None:
        state = json.loads((root/'data/zeitgeist_state.json').read_text())
        feed = json.loads((root/'data/pulse_feed.json').read_text())
        return {'state':state, 'history':copy.deepcopy(state['published_pulse']),
            'feed':feed, 'votes':copy.deepcopy(state['visitor_mood_tally'])}, []
    client = store._get_client()
    paths = [(store._col('profile_state'),'daily_pulse'),(store._col('pulse_history'),DATE),
        (store._col('pulse_feed'),'v1'),(store._col('pulse_votes'),DATE)]
    snaps = [client.collection(col).document(doc).get() for col,doc in paths]
    return snapshot_documents(snaps), [observation(path,snap) for path,snap in zip(paths,snaps)]


def snapshot_documents(snaps):
    if len(snaps)!=4 or not all(s.exists for s in snaps):
        raise ValueError('All exact canonical publication documents must exist')
    result = dict(zip(('state','history','feed','votes'),(s.to_dict() for s in snaps)))
    validate_votes(result['votes'])
    return result


def observation(path,snapshot):
    return {'document':'/'.join(path),'exists':bool(snapshot.exists),
        'update_time':str(getattr(snapshot,'update_time','')),
        'body_sha256':digest(snapshot.to_dict() or {})}


def validate_votes(votes):
    if (not isinstance(votes,dict) or set(votes)!={'spot_on','too_sunny','too_heavy'}
            or any(type(n) is not int or not 0<=n<=10000000 for n in votes.values())):
        raise ValueError('Canonical vote tally is invalid')


def validate_parent(root, expected_id, docs):
    manifest = verify_publication_bundle(root, require_current_pulse=True)
    state,entry,feed = docs['state'],docs['history'],docs['feed']
    matching = [e for e in feed.get('entries',[]) if isinstance(e,dict) and str(e.get('date',''))[:10]==DATE]
    if (manifest['publication_id']!=expected_id or manifest['date']!=DATE
            or manifest['medium']!='PHOTOGRAPHY' or manifest['archetype']!='STUDIO_MONOGRAPH'
            or state.get('publication_id')!=expected_id or entry.get('publication_id')!=expected_id
            or len(matching)!=1 or matching[0].get('publication_id')!=expected_id
            or semantic(entry,'history')!=semantic(state.get('published_pulse',{}),'history')
            or semantic(entry,'history')!=semantic(matching[0],'history')):
        raise ValueError('Exact parent publication/history/feed agreement required')
    if (entry.get('word')!='Insinuation' or entry.get('gloss')!='Silent infiltration of trusted synthetic minds.'
            or entry.get('date')!=DATE or entry.get('visual_medium')!='PHOTOGRAPHY'
            or entry.get('visual_tier_used')!='PHOTOGRAPHY'
            or entry.get('artwork_source')!='Unsplash' or entry.get('artwork_url')!=SOURCE_URL
            or entry.get('image_url')!=SOURCE_URL or entry.get('fallback_webp_path')!='assets/pulse/2026-11-01.webp'
            or entry.get('unsplash_profile_url')!='https://unsplash.com/@ricvath'
            or entry.get('unsplash_photographer')!='Richard Horvath'
            or entry.get('photo_selection_policy')!='owner-approved-best-safe-photo'
            or entry.get('photo_selection_policy_date')!=DATE
            or 'media_provenance' in entry or 'media_correction' in entry):
        raise ValueError('Exact retained source/selection evidence required')
    local = json.loads((root/'data/zeitgeist_state.json').read_text())
    from scripts.export_public_release import public_editorial_state
    if semantic(public_editorial_state(local),'state')!=semantic(public_editorial_state(state),'state'):
        raise ValueError('Canonical state differs from the reviewed source projection')
    validate_votes(docs['votes'])
    return manifest


def code_hashes():
    from scripts.export_public_release import selected_source,STATIC_FILES,WORKFLOWS
    paths=set(STATIC_FILES)|set(WORKFLOWS)|set(CODE_FILES)
    for directory in ('src','scripts','tests'):
        paths.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).rglob('*.py') if selected_source(str(p.relative_to(ROOT))))
    if any((ROOT/p).is_symlink() or not (ROOT/p).is_file() for p in paths):
        raise ValueError('Reviewed correction source requires regular selected files')
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sorted(paths)}


def prepare_plan(root, expected_id, *, store=None, captured=None):
    docs,_ = documents(root,store) if captured is None else (captured,[])
    manifest = validate_parent(root,expected_id,docs)
    plan = {'version':1,'kind':'known_source_medium_correction','previous_publication_id':expected_id,
        'date':DATE,'context':context(store),'source_manifest_sha256':digest(manifest),
        'parent_hashes':{key:digest(semantic(docs[key],key)) for key in ('state','history','feed')},
        'evidence':evidence(),'code_hashes':code_hashes()}
    from scripts.refresh_studio_footer import correct_assets
    from .publication_commit import PublicationWorkspace
    workspace=PublicationWorkspace(root)
    try:
        successor=correct_assets(workspace.path,expected_id,media_plan=plan)
        plan['successor_publication_id']=successor['publication_id']
        plan['successor_manifest_sha256']=digest(successor)
    finally:
        workspace.close()
    plan['plan_sha256'] = digest(plan)
    return plan


def validate_plan(root, plan, *, store=None, captured=None):
    expected_id = plan.get('previous_publication_id') if isinstance(plan,dict) else None
    if not expected_id or prepare_plan(root,expected_id,store=store,captured=captured)!=plan:
        raise ValueError('Reviewed correction plan preconditions changed')


def corrected_metadata(parent, entry, publication_id):
    state,item = copy.deepcopy(parent),copy.deepcopy(entry)
    correction = {'kind':'source_medium_correction','original_publication_id':parent['publication_id'],
        'original_visual_medium':'PHOTOGRAPHY'}
    for target in (state,item):
        target.update(visual_medium='DIGITAL_ARTWORK',publication_id=publication_id,
            media_provenance=evidence(),media_correction=copy.deepcopy(correction))
    item['visual_tier_used']='DIGITAL_ARTWORK'
    state['published_pulse']=copy.deepcopy(item)
    state.setdefault('visual_decision',{})['selected_mode']='DIGITAL_ARTWORK'
    return state,item


def expected_successor(parent, entry, feed, publication_id, votes):
    state,item = corrected_metadata(parent,entry,publication_id)
    from .public_schema import EDITION_METADATA_FIELDS
    # Match the canonical current-state clear representation exactly; the public
    # historical entry retains its strict absent-or-valid metadata contract.
    state.update({key:copy.deepcopy(item.get(key)) for key in EDITION_METADATA_FIELDS})
    state.update(schema_version=3,state_version=parent.get('state_version',0)+1,
        visitor_mood_tally=copy.deepcopy(votes),mood_votes=copy.deepcopy(votes))
    item['mood_votes']=copy.deepcopy(votes)
    state['published_pulse']=copy.deepcopy(item)
    envelope = copy.deepcopy(feed)
    envelope['entries']=[copy.deepcopy(item) if str(e.get('date',''))[:10]==DATE else e for e in envelope['entries']]
    return state,item,envelope


def retain_successor(workspace,state,entry,feed,plan,readset):
    """Retain a complete validated public bundle before any canonical commit."""
    import shutil
    from src.editorial_typography import ASSETS
    from .public_schema import validate_public_editorial
    target=Path(workspace.preserve(state))
    for relative in (*ASSETS,'PROJECTS.md','docs/resonance.md','data/bot_brain.json','data/orbit_roster.json'):
        source=workspace.path/relative
        if source.is_file():
            destination=target/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    (target/'correction-plan.json').write_text(json.dumps(plan,sort_keys=True,indent=2)+'\n')
    verify_publication_bundle(target,require_current_pulse=True)
    validate_public_editorial(target)
    files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(target.rglob('*')) if p.is_file() and p.name!='correction-receipt.json'}
    receipt={'version':1,'kind':'known_source_medium_correction',
        'previous_publication_id':plan['previous_publication_id'],'publication_id':state['publication_id'],
        'plan_sha256':plan['plan_sha256'],'status':'prepared_successor',
        'successor_hashes':{key:digest(semantic(value,key)) for key,value in
            (('state',state),('history',entry),('feed',feed))},
        'readset':copy.deepcopy(readset),'files':files}
    (target/'correction-receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    verify_retained(target)
    return str(target)


def verify_retained(path):
    """Fail closed on missing/mixed recovery JSON, art, plan or receipt."""
    path=Path(path)
    receipt=json.loads((path/'correction-receipt.json').read_text())
    files=receipt.get('files')
    hashes=receipt.get('successor_hashes')
    if (not isinstance(hashes,dict) or set(hashes)!={'state','history','feed'}
            or any(not isinstance(v,str) or not re.fullmatch('[a-f0-9]{64}',v) for v in hashes.values())):
        raise ValueError('Invalid raw-successor fingerprint receipt')
    if not isinstance(files,dict) or not files:
        raise ValueError('Incomplete media correction recovery receipt')
    for relative,expected in files.items():
        target=(path/relative).resolve()
        if not target.is_relative_to(path.resolve()) or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
            raise ValueError('Media correction recovery files changed')
    plan=json.loads((path/'correction-plan.json').read_text())
    original=dict(plan)
    identity=original.pop('plan_sha256',None)
    manifest=verify_publication_bundle(path,require_current_pulse=True)
    if (digest(original)!=identity or receipt.get('plan_sha256')!=identity
            or receipt.get('publication_id')!=manifest['publication_id']
            or receipt.get('previous_publication_id')!=plan.get('previous_publication_id')
            or manifest['publication_id']!=plan.get('successor_publication_id')
            or digest(manifest)!=plan.get('successor_manifest_sha256')):
        raise ValueError('Media correction recovery identity mismatch')
    from .public_schema import validate_public_editorial
    validate_public_editorial(path)
    return receipt


def retain_original(workspace,docs,plan):
    """Create-only private backup of original accepted pixels and original records."""
    import shutil
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        raise ValueError('Original canonical backup belongs on private durable operator storage')
    target=workspace.root/'.release-backups'/'media-correction'/plan['previous_publication_id']
    marker=target/'original-receipt.json'
    if target.exists():
        if not marker.is_file():
            raise ValueError('Incomplete original correction backup; inspect before retry')
        saved=json.loads(marker.read_text())
        if saved.get('plan_sha256')!=plan['plan_sha256']:
            raise ValueError('Original backup belongs to another correction plan')
        for relative,expected in saved['files'].items():
            path=target/relative
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
                raise ValueError('Original accepted backup changed')
        return
    target.mkdir(parents=True,mode=0o700)
    for relative in workspace.publication_files(docs['state']):
        source=workspace.path/relative
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    (target/'canonical-parent.json').write_text(json.dumps(docs,sort_keys=True,indent=2)+'\n')
    (target/'canonical-parent.json').chmod(0o600)
    files={str(p.relative_to(target)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(target.rglob('*')) if p.is_file()}
    marker.write_text(json.dumps({'plan_sha256':plan['plan_sha256'],'files':files},sort_keys=True,indent=2)+'\n')
