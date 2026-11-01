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
    PublicationOutcomeUnknown, _commit_documents, get_state_store, git_commit_and_push_with_retry,
)
from src.zeitgeist.curatorial_compositor import replace_studio_monograph_footer
from src.zeitgeist.publication import verify_publication_bundle, write_publication_manifest
from src.zeitgeist.publication_commit import PublicationWorkspace, publish_bundle


def correct_assets(root: Path, expected_id: str) -> dict:
    """Only change the footer pixels and embedded copy; preserve the archive."""
    before = verify_publication_bundle(root, require_current_pulse=True)
    if before['publication_id'] != expected_id or before['archetype'] != 'STUDIO_MONOGRAPH':
        raise ValueError('Exact accepted Studio Monograph publication required')
    pulse = json.loads((root / 'data/zeitgeist_state.json').read_text())
    entry = pulse.get('published_pulse') or next(
        e for e in json.loads((root / 'data/pulse_feed.json').read_text())['entries']
        if str(e.get('date', ''))[:10] == before['date'])
    attribution = {'photographer_name': entry.get('unsplash_photographer') or entry.get('artwork_artist')}
    for theme in ('light', 'dark'):
        path = root / f'assets/daily_pulse_{theme}.webp'
        with Image.open(path) as source:
            old = source.convert('RGB')
        new = replace_studio_monograph_footer(old, date_str=before['date'],
            attribution=attribution, visual_medium=before['medium'], dark_mode=theme == 'dark')
        new.save(path, format='WEBP', lossless=True, exact=True)
        with Image.open(path) as saved:
            diff = ImageChops.difference(old, saved.convert('RGB'))
            if diff.crop((0, 0, 1200, 716)).getbbox() or diff.crop((0, 748, 1200, 800)).getbbox():
                raise ValueError('Footer correction changed accepted artwork pixels')
    path = root / 'assets/daily_pulse.svg'
    svg = path.read_text()
    encoded = base64.b64encode((root / 'assets/daily_pulse_dark.webp').read_bytes()).decode('ascii')
    svg, count = re.subn(r'data:image/webp;base64,[A-Za-z0-9+/=]+',
                        f'data:image/webp;base64,{encoded}', svg)
    if count != 1:
        raise ValueError('Expected one embedded accepted artwork')
    path.write_text(svg)
    return write_publication_manifest(root, date=before['date'], word=before['word'],
        gloss=before['gloss'], archetype=before['archetype'], medium=before['medium'],
        archive_path=f"assets/pulse/{before['date']}.webp")


def refresh(root, expected_id, *, store=None, apply=False):
    journal = root / 'data/publication_reconcile.json'
    if journal.exists():
        raise RuntimeError('Resolve pending publication before a footer correction')
    verify_publication_bundle(root, require_current_pulse=True)
    if apply:
        projection = store.canonical_projections(write_mirrors=False)
        expected = projection['zeitgeist_state.json']
    else:
        expected = json.loads((root / 'data/zeitgeist_state.json').read_text())
    if expected.get('publication_id') != expected_id:
        raise ValueError('Canonical publication changed; do not replay the correction')
    workspace = PublicationWorkspace(root)
    committed = False
    journal_created = False
    try:
        manifest = correct_assets(workspace.path, expected_id)
        if manifest['publication_id'] == expected_id:
            raise ValueError('Footer is already corrected; no canonical mutation needed')
        payload = copy.deepcopy(expected)
        payload['publication_id'] = manifest['publication_id']
        entry = copy.deepcopy(expected['published_pulse'])
        entry['publication_id'] = manifest['publication_id']
        payload['published_pulse'] = copy.deepcopy(entry)
        feed = json.loads((root / 'data/pulse_feed.json').read_text())
        if not apply:
            feed['entries'] = [entry if str(e.get('date', ''))[:10] == manifest['date'] else e
                               for e in feed['entries']]
            workspace.prepare(payload, entry, feed)
            preview = root / '.footer-preview'
            preview.mkdir(exist_ok=True)
            for name in ('daily_pulse_light.webp', 'daily_pulse_dark.webp', 'pulse_manifest.json'):
                (preview / name).write_bytes((workspace.path / 'assets' / name).read_bytes())
            return {'status': 'preview', 'previous_publication_id': expected_id,
                    'publication_id': manifest['publication_id'], 'paid_calls': 0}
        workspace.stage_canonical_game(store)

        def prepare(state, item, envelope):
            nonlocal journal_created
            workspace.prepare(state, item, envelope)
            recovery = workspace.preserve(state)
            journal.write_text(json.dumps({'publication_id': state['publication_id'],
                'previous_publication_id': expected_id, 'status': 'canonical_commit_pending',
                'recovery_path': recovery}) + '\n')
            journal_created = True

        state = publish_bundle(store, payload, entry, expected=expected,
                               local_feed=feed, prepare=prepare)
        committed = True
        workspace.install(state)
        if not git_commit_and_push_with_retry('fix(pulse): remove Studio Monograph footer prefix [skip ci]',
                                             repo_root=root, store=store):
            raise RuntimeError('Canonical correction committed; Git reconciliation required')
        journal.unlink()
        return {'status': 'published', 'previous_publication_id': expected_id,
                'publication_id': state['publication_id'], 'paid_calls': 0}
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


def rehearse(root, expected_id, env):
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
    result = refresh(root, expected_id, store=store, apply=True)
    result.update(status='sandbox_rehearsed', fixture_namespace=prefix)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-publication-id', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--rehearse', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch('[a-f0-9]{20}', args.expected_publication_id):
        parser.error('An exact publication ID is required')
    if args.rehearse:
        print(json.dumps(rehearse(ROOT, args.expected_publication_id, os.environ), sort_keys=True))
        return 0
    store = None
    if args.apply:
        from scripts.check_runtime_configuration import configuration_errors
        if (os.environ.get('GITHUB_EVENT_NAME') != 'workflow_dispatch'
                or os.environ.get('GITHUB_ACTOR') != 'Siyuan-Zhu'
                or os.environ.get('GITHUB_REF_NAME') != 'main'
                or configuration_errors(os.environ)):
            parser.error('Reviewed owner dispatch, main branch and canonical namespace required')
        store = get_state_store(ROOT, backend='firestore')
    print(json.dumps(refresh(ROOT, args.expected_publication_id, store=store, apply=args.apply), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
