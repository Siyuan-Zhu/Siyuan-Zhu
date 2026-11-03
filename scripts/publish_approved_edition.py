"""Publish one retained, approved edition without calling generation providers."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.state_store import get_state_store, git_commit_and_push_with_retry
from src.zeitgeist.publication import verify_publication_bundle
from src.zeitgeist.publication_commit import PublicationWorkspace, publish_bundle
from src.zeitgeist.edition_approval import require_edition_approval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--expected-publication-id', required=True)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    manifest = verify_publication_bundle(bundle, require_current_pulse=True)
    if manifest['date'] != '2026-11-03' or manifest['publication_id'] != args.expected_publication_id:
        raise ValueError('Exact approved November 3 publication required')
    require_edition_approval(manifest)
    payload = json.loads((bundle/'data/zeitgeist_state.json').read_text())
    feed = json.loads((bundle/'data/pulse_feed.json').read_text())
    entry = next(e for e in feed['entries'] if e.get('publication_id') == manifest['publication_id'])
    from src.zeitgeist.public_schema import validate_editorial_entry
    validate_editorial_entry(entry)
    if args.validate_only:
        print(json.dumps({'status':'validated', 'publication_id':manifest['publication_id'], 'provider_calls':0}))
        return
    if os.environ.get('GITHUB_REPOSITORY') != 'Siyuan-Zhu/Siyuan-Zhu' or os.environ.get('GITHUB_REF_NAME') != 'main':
        raise RuntimeError('Canonical publication requires the public main writer queue')
    store = get_state_store(repo_root=ROOT, backend='firestore', collection_prefix='')
    expected = store.load_pulse_state()
    if expected.get('publication_id') == manifest['publication_id']:
        raise RuntimeError('Canonical edition already committed; reconcile retained bundle instead of replaying')
    if str(expected.get('date',expected.get('date_iso','')))[:10] >= manifest['date']:
        raise RuntimeError('Refuse replacement of a same-day or newer canonical edition')
    if (ROOT/'data/publication_reconcile.json').exists():
        raise RuntimeError('Existing publication requires reconciliation')
    workspace = PublicationWorkspace(ROOT)
    journal = ROOT/'data/publication_reconcile.json'
    try:
        for relative in [*manifest['files'], 'assets/pulse_manifest.json']:
            destination = workspace.path/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(bundle/relative, destination)
        workspace.stage_canonical_game(store)
        def prepare(state, item, envelope):
            workspace.prepare(state,item,envelope)
            recovery = workspace.preserve(state)
            journal.write_text(json.dumps({'publication_id':manifest['publication_id'], 'status':'canonical_commit_pending', 'recovery_path':recovery})+'\n')
        state = publish_bundle(store,payload,entry,expected=expected,local_feed=feed,prepare=prepare)
        workspace.install(state)
        journal.unlink(missing_ok=True)
        if not git_commit_and_push_with_retry('Publish approved November 3 Climacteric edition [skip ci]',repo_root=ROOT,store=store,expected_publication_id=manifest['publication_id']):
            raise RuntimeError('Canonical committed; Git publication requires retained-bundle reconciliation')
        print(json.dumps({'status':'published', 'publication_id':manifest['publication_id'], 'provider_calls':0}))
    finally:
        workspace.close()


if __name__ == '__main__':
    main()
