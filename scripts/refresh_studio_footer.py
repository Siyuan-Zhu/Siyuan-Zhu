"""Owner-dispatched, provider-free correction of an exact accepted publication."""
from __future__ import annotations

import argparse
import base64
import copy
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image, ImageChops
from src.state_store import (
    FirestoreStateStore, PublicationOutcomeUnknown, _commit_documents, get_state_store, git_commit_and_push_with_retry,
)
from PIL import ImageOps
from src.zeitgeist.curatorial_compositor import (
    LITERARY_ART_BOX, _render_literary_journal_plate, replace_studio_monograph_footer,
)
from src.zeitgeist.publication import verify_publication_bundle, write_publication_manifest
from src.zeitgeist.publication_commit import PublicationWorkspace, publish_bundle


CORRECTABLE = ('STUDIO_MONOGRAPH', 'LITERARY_JOURNAL')
COMMIT_MESSAGES = {'STUDIO_MONOGRAPH': 'fix(pulse): remove Studio Monograph footer prefix [skip ci]',
                   'LITERARY_JOURNAL': 'fix(pulse): apply Gallery label layout to the Literary Journal edition [skip ci]'}


def _unchanged_outside(old, saved, boxes):
    from PIL import ImageDraw
    mask = Image.new('L', old.size, 255)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, fill=0)
    diff = ImageChops.difference(old, saved).convert('L')
    return ImageChops.multiply(diff, mask).getbbox() is None


def correct_assets(root: Path, expected_id: str, *, media_plan=None) -> dict:
    """Studio Monograph: footer pixels only. Literary Journal: re-render from the archive. Archive kept."""
    before = verify_publication_bundle(root, require_current_pulse=True)
    archetype = before['archetype']
    if (before['publication_id'] != expected_id or archetype not in CORRECTABLE
            or (media_plan and archetype != 'STUDIO_MONOGRAPH')):
        raise ValueError('Exact accepted correctable publication required')
    pulse = json.loads((root / 'data/zeitgeist_state.json').read_text())
    entry = pulse.get('published_pulse') or next(
        e for e in json.loads((root / 'data/pulse_feed.json').read_text())['entries']
        if str(e.get('date', ''))[:10] == before['date'])
    attribution = {'photographer_name': entry.get('unsplash_photographer') or entry.get('artwork_artist')}
    archive = root / f"assets/pulse/{before['date']}.webp"
    if archetype == 'LITERARY_JOURNAL':
        # Re-render from the accepted edition's own archived source photograph.
        with Image.open(archive) as source:
            hero = source.convert('RGB')
        x, y, w, h = LITERARY_ART_BOX
        expected_plate = ImageOps.fit(hero, (w, h), method=Image.Resampling.LANCZOS)
    for theme in ('light', 'dark'):
        path = root / f'assets/daily_pulse_{theme}.webp'
        with Image.open(path) as source:
            old = source.convert('RGB')
        if archetype == 'LITERARY_JOURNAL':
            new = _render_literary_journal_plate(hero, word=before['word'], gloss=before['gloss'], thesis='',
                date_str=before['date'], attribution=attribution, visual_medium=before['medium'],
                dark_mode=theme == 'dark')
        else:
            new = replace_studio_monograph_footer(old, date_str=before['date'],
                attribution=attribution, visual_medium='DIGITAL_ARTWORK' if media_plan else before['medium'], dark_mode=theme == 'dark')
        new.save(path, format='WEBP', lossless=True, exact=True)
        with Image.open(path) as saved:
            saved = saved.convert('RGB')
            if archetype == 'LITERARY_JOURNAL':
                if ImageChops.difference(saved.crop((x, y, x + w, y + h)), expected_plate).getbbox():
                    raise ValueError('Re-rendered plate is not the archived photograph')
            elif not _unchanged_outside(old, saved, ((0, 716, 1199, 747),)):
                raise ValueError('Correction changed accepted artwork pixels')
    path = root / 'assets/daily_pulse.svg'
    svg = path.read_text()
    encoded = base64.b64encode((root / 'assets/daily_pulse_dark.webp').read_bytes()).decode('ascii')
    svg, count = re.subn(r'data:image/webp;base64,[A-Za-z0-9+/=]+',
                        f'data:image/webp;base64,{encoded}', svg)
    if count != 1:
        raise ValueError('Expected one embedded accepted artwork')
    path.write_text(svg)
    return write_publication_manifest(root, date=before['date'], word=before['word'],
        gloss=before['gloss'], archetype=before['archetype'], medium='DIGITAL_ARTWORK' if media_plan else before['medium'],
        archive_path=f"assets/pulse/{before['date']}.webp",
        media_provenance=media_plan['evidence'] if media_plan else before.get('media_provenance'),
        media_correction=({'kind':'source_medium_correction','original_publication_id':expected_id,
            'original_visual_medium':'PHOTOGRAPHY'} if media_plan else before.get('media_correction')))


def prepare_media_correction(root, expected_id, *, store=None):
    from src.zeitgeist.media_correction import prepare_plan
    return prepare_plan(Path(root), expected_id, store=store)


def refresh(root, expected_id, *, store=None, apply=False, media_plan=None):
    journal = root / 'data/publication_reconcile.json'
    if journal.exists():
        raise RuntimeError('Resolve pending publication before a footer correction')
    verify_publication_bundle(root, require_current_pulse=True)
    if media_plan and apply and not isinstance(store,FirestoreStateStore):
        raise ValueError('Known media correction requires canonical Firestore')
    if apply:
        projection = store.canonical_projections(write_mirrors=False)
        expected = projection['zeitgeist_state.json']
    else:
        expected = json.loads((root / 'data/zeitgeist_state.json').read_text())
    if expected.get('publication_id') != expected_id:
        raise ValueError('Canonical publication changed; do not replay the correction')
    if media_plan:
        from src.zeitgeist.media_correction import documents, validate_plan, digest
        media_docs, media_observation = documents(root, store)
        validate_plan(root, media_plan, store=store, captured=media_docs)
        expected = media_docs['state']
    workspace = PublicationWorkspace(root)
    committed = False
    journal_created = False
    try:
        if media_plan and os.environ.get('GITHUB_ACTIONS') != 'true':
            from src.zeitgeist.media_correction import retain_original
            retain_original(workspace,media_docs,media_plan)
        manifest = correct_assets(workspace.path, expected_id, media_plan=media_plan)
        if media_plan and (manifest['publication_id']!=media_plan['successor_publication_id']
                or digest(manifest)!=media_plan['successor_manifest_sha256']):
            raise ValueError('Rendered correction differs from reviewed successor')
        if manifest['publication_id'] == expected_id:
            raise ValueError('Footer is already corrected; no canonical mutation needed')
        payload = copy.deepcopy(expected)
        payload['publication_id'] = manifest['publication_id']
        entry = copy.deepcopy(expected['published_pulse'])
        entry['publication_id'] = manifest['publication_id']
        payload['published_pulse'] = copy.deepcopy(entry)
        feed = json.loads((root / 'data/pulse_feed.json').read_text())
        if media_plan:
            from src.zeitgeist.media_correction import expected_successor, corrected_metadata
            payload,entry = corrected_metadata(expected,media_docs['history'],manifest['publication_id'])
            feed = media_docs['feed']
        if not apply:
            if media_plan:
                if store is not None:
                    workspace.stage_canonical_game(store)
                payload,entry,feed = expected_successor(expected,media_docs['history'],feed,manifest['publication_id'],media_docs['votes'])
            feed['entries'] = [entry if str(e.get('date', ''))[:10] == manifest['date'] else e
                               for e in feed['entries']]
            workspace.prepare(payload, entry, feed)
            if media_plan:
                from src.zeitgeist.media_correction import retain_successor
                retained=Path(retain_successor(workspace,payload,entry,feed,media_plan,media_observation))
                preview = root / '.media-correction-preview'
                preview.mkdir(exist_ok=True)
                import shutil
                shutil.copytree(retained,preview,dirs_exist_ok=True)
                return {'status':'preview','previous_publication_id':expected_id,
                    'publication_id':manifest['publication_id'],'plan_sha256':media_plan['plan_sha256'],
                    'preview_path':str(preview),'paid_calls':0}
            preview = root / '.footer-preview'
            preview.mkdir(exist_ok=True)
            for name in ('daily_pulse_light.webp', 'daily_pulse_dark.webp', 'pulse_manifest.json'):
                (preview / name).write_bytes((workspace.path / 'assets' / name).read_bytes())
            return {'status': 'preview', 'previous_publication_id': expected_id,
                    'publication_id': manifest['publication_id'], 'paid_calls': 0}
        workspace.stage_canonical_game(store)

        def validate_current(snaps):
            nonlocal media_docs,media_observation
            from src.zeitgeist.media_correction import snapshot_documents,observation
            current=snapshot_documents(snaps)
            validate_plan(root,media_plan,store=store,captured=current)
            media_docs=current
            paths=[(store._col('profile_state'),'daily_pulse'),(store._col('pulse_history'),'2026-11-01'),
                (store._col('pulse_feed'),'v1'),(store._col('pulse_votes'),'2026-11-01')]
            media_observation=[observation(path,snap) for path,snap in zip(paths,snaps)]

        def prepare(state, item, envelope):
            nonlocal journal_created
            if media_plan:
                intended=expected_successor(media_docs['state'],media_docs['history'],media_docs['feed'],
                    manifest['publication_id'],media_docs['votes'])
                if (state,item,envelope)!=intended:
                    raise ValueError('Correction changed fields outside its exact allowed diff')
            workspace.prepare(state, item, envelope)
            if media_plan:
                from src.zeitgeist.media_correction import retain_successor
                recovery=retain_successor(workspace,state,item,envelope,media_plan,media_observation)
            else:
                recovery = workspace.preserve(state)
            journal.write_text(json.dumps({'publication_id': state['publication_id'],
                'previous_publication_id': expected_id, 'status': 'canonical_commit_pending',
                'recovery_path': recovery, **({'correction_plan_sha256':media_plan['plan_sha256']} if media_plan else {})}) + '\n')
            journal_created = True

        state = publish_bundle(store, payload, entry, expected=expected,
                               local_feed=feed, prepare=prepare, validate_current=validate_current if media_plan else None)
        committed = True
        workspace.install(state)
        if not git_commit_and_push_with_retry(COMMIT_MESSAGES[manifest['archetype']],
                                             repo_root=root, store=store,
                                             expected_publication_id=state['publication_id'] if media_plan else None):
            raise RuntimeError('Canonical correction committed; Git reconciliation required')
        journal.unlink()
        return {'status': 'published', 'previous_publication_id': expected_id,
                'publication_id': state['publication_id'], 'paid_calls': 0,
                **({'plan_sha256':media_plan['plan_sha256'],
                    'recovery_path':str(root/'.publication-recovery'/state['publication_id'])} if media_plan else {})}
    except BaseException as exc:
        if journal_created:
            if committed or isinstance(exc, PublicationOutcomeUnknown):
                marker = json.loads(journal.read_text())
                marker['status'] = 'canonical_committed_git_pending' if committed else 'canonical_outcome_unknown'
                journal.write_text(json.dumps(marker) + '\n')
                store.set_needs_git_reconcile(True)
            else:
                journal.unlink(missing_ok=True)
        raise
    finally:
        workspace.close()


def reconcile_media(root,expected_id,expected_plan_sha256,*,store):
    """Mirror only a positively observed exact successor; never replay correction."""
    from src.zeitgeist.media_correction import verify_retained,documents,semantic,context,code_hashes,digest
    if not isinstance(store,FirestoreStateStore):
        raise ValueError('Exact correction reconciliation requires canonical Firestore')
    journal=root/'data/publication_reconcile.json'
    if not journal.is_file():
        raise RuntimeError('No pending correction journal; inspect actual publication')
    marker=json.loads(journal.read_text())
    if (not re.fullmatch('[a-f0-9]{20}',str(marker.get('publication_id','')))
            or marker.get('previous_publication_id')!=expected_id
            or marker.get('status') not in {'canonical_commit_pending','canonical_outcome_unknown','canonical_committed_git_pending'}):
        raise ValueError('Exact pending correction identity required')
    target=root/'.publication-recovery'/marker['publication_id']
    receipt=verify_retained(target)
    plan=json.loads((target/'correction-plan.json').read_text())
    if (plan['previous_publication_id']!=expected_id or plan['plan_sha256']!=expected_plan_sha256
            or marker.get('correction_plan_sha256')!=expected_plan_sha256
            or context(store)!=plan['context'] or code_hashes()!=plan['code_hashes']):
        raise ValueError('Exact reviewed correction source/context required for reconciliation')
    current,_=documents(root,store)
    retained_state=json.loads((target/'data/zeitgeist_state.json').read_text())
    retained={'state':retained_state,'history':retained_state['published_pulse'],
        'feed':json.loads((target/'data/pulse_feed.json').read_text())}
    if (current['state'].get('publication_id')!=receipt['publication_id']
            or any(digest(semantic(current[key],key))!=receipt['successor_hashes'][key] for key in retained)):
        raise ValueError('Canonical successor is unknown, changed or superseded; do not replay')
    if not git_commit_and_push_with_retry('fix(pulse): reconcile exact source-medium correction [skip ci]',repo_root=root,store=store,expected_publication_id=receipt['publication_id']):
        marker['status']='canonical_committed_git_pending'
        journal.write_text(json.dumps(marker)+'\n')
        store.set_needs_git_reconcile(True)
        raise RuntimeError('Exact successor accepted; Git outcome needs inspection')
    after,_=documents(root,store)
    if any(digest(semantic(after[key],key))!=receipt['successor_hashes'][key] for key in retained):
        raise ValueError('Canonical successor advanced during Git publication; inspect before closing journal')
    journal.unlink()
    return {'status':'reconciled','publication_id':receipt['publication_id'],
        'previous_publication_id':expected_id,'plan_sha256':expected_plan_sha256,'paid_calls':0}


def rehearse(root, expected_id, env, *, correct_media=False):
    """Create-only fixture namespace in the private sandbox; never touch its game."""
    from scripts.check_runtime_configuration import configuration_errors
    if (env.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'
            or env.get('GITHUB_ACTOR') != 'Siyuan-Zhu'
            or env.get('GITHUB_REPOSITORY') != 'Siyuan-Zhu/readme-live-sandbox'
            or configuration_errors(env)
            or not re.fullmatch(r'[0-9]+', env.get('GITHUB_RUN_ID', ''))
            or not re.fullmatch(r'[0-9]+', env.get('GITHUB_RUN_ATTEMPT', ''))):
        raise ValueError('Reviewed private sandbox owner dispatch required')
    verify_publication_bundle(root, require_current_pulse=True)
    pulse = json.loads((root / 'data/zeitgeist_state.json').read_text())
    if pulse.get('publication_id') != expected_id:
        raise ValueError('Exact accepted fixture required before create-only seeding')
    if correct_media:
        prepare_media_correction(root,expected_id)  # Validate exact source before create-only seeding.
    prefix = f"staging_footer_rehearsal_{env['GITHUB_RUN_ID']}_{env['GITHUB_RUN_ATTEMPT']}_"
    store = get_state_store(root, backend='firestore', collection_prefix=prefix)
    client = store._get_client()
    fixtures = {('profile_state', 'daily_pulse'): pulse,
        ('pulse_history', pulse['date'][:10]): pulse['published_pulse'],
        ('pulse_feed', 'v1'): json.loads((root / 'data/pulse_feed.json').read_text()),
        ('pulse_votes', pulse['date'][:10]): pulse['visitor_mood_tally']}
    for doc, name in [('active_game','game_state'), ('elo_ledger','hall_of_records'),
                      ('bot_memory','bot_brain'), ('orbit_roster','orbit_roster')]:
        fixtures[('profile_state', doc)] = json.loads((root / f'data/{name}.json').read_text())
    writes = []
    for (collection, doc), payload in fixtures.items():
        collection = prefix + collection
        snapshot = client.collection(collection).document(doc).get()
        if snapshot.exists:
            raise ValueError('Rehearsal namespace already exists; never overwrite or replay')
        writes.append((collection, doc, payload, snapshot))
    _commit_documents(client, writes)
    media_plan=prepare_media_correction(root,expected_id,store=store) if correct_media else None
    result = refresh(root, expected_id, store=store, apply=True, media_plan=media_plan)
    result.update(status='sandbox_rehearsed', fixture_namespace=prefix,
        **({'plan_sha256':media_plan['plan_sha256']} if media_plan else {}))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-publication-id', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rehearse', action='store_true')
    mode.add_argument('--reconcile', action='store_true')
    mode.add_argument('--canonical-preview', action='store_true')
    parser.add_argument('--correct-media',action='store_true',help='Reviewed exact November1 source-medium correction')
    parser.add_argument('--expected-plan-sha256',default='')
    args = parser.parse_args(argv)
    if args.canonical_preview and not args.correct_media:
        parser.error('Canonical preview is scoped to the reviewed media correction')
    if not re.fullmatch('[a-f0-9]{20}', args.expected_publication_id):
        parser.error('An exact publication ID is required')
    if args.rehearse:
        print(json.dumps(rehearse(ROOT, args.expected_publication_id, os.environ, correct_media=args.correct_media), sort_keys=True))
        return 0
    store = None
    if args.apply or args.reconcile or args.canonical_preview:
        from scripts.check_runtime_configuration import configuration_errors
        if (os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'
                or os.environ.get('GITHUB_ACTOR') != 'Siyuan-Zhu'
                or (not args.canonical_preview and os.environ.get('GITHUB_REF_NAME') != 'main')
                or os.environ.get('GITHUB_REPOSITORY') not in {'Siyuan-Zhu/Siyuan-Zhu','Siyuan-Zhu/readme-live-sandbox'}
                or configuration_errors(os.environ)):
            parser.error('Reviewed owner dispatch, main branch and canonical namespace required')
        store = get_state_store(ROOT, backend='firestore')
    if args.reconcile:
        if not args.correct_media or not re.fullmatch('[a-f0-9]{64}',args.expected_plan_sha256):
            parser.error('Exact reviewed media plan required for reconciliation')
        print(json.dumps(reconcile_media(ROOT,args.expected_publication_id,args.expected_plan_sha256,store=store),sort_keys=True))
        return 0
    media_plan=prepare_media_correction(ROOT,args.expected_publication_id,store=store) if args.correct_media else None
    if media_plan and args.apply and (not re.fullmatch('[a-f0-9]{64}',args.expected_plan_sha256)
            or media_plan['plan_sha256']!=args.expected_plan_sha256):
        parser.error('Exact reviewed media correction plan SHA256 required before apply')
    print(json.dumps(refresh(ROOT,args.expected_publication_id,store=store,apply=args.apply,media_plan=media_plan),sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
