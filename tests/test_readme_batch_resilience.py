"""One authoritative read per README batch; bounded transport-only GET retries."""
from io import BytesIO
from html.parser import HTMLParser
from pathlib import Path
import shutil
import re
from urllib.error import HTTPError, URLError
import json
import pytest
from scripts import build_readme
from src import state_store as state
from tests.test_canonical_runtime_factory import fixtures, files

def test_all_variants_share_one_canonical_snapshot_even_when_remote_advances(tmp_path, monkeypatch):
    root, local, game, client = fixtures(tmp_path)
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    original = store.canonical_projections
    calls = []

    def once(**kwargs):
        calls.append(kwargs)
        if len(calls) > 1:
            raise TimeoutError('second canonical fetch should not occur')
        result = original(**kwargs)
        client.docs['profile_state/daily_pulse']['word'] = 'NEXT'
        return result
    monkeypatch.setattr(store, 'canonical_projections', once)
    written = build_readme.build_all_readmes(root, repo_slug='Siyuan-Zhu/readme-live-sandbox', state_store=store)
    assert len(written) == 9 and len(calls) == 1
    for path in written.values():
        text = path.read_text()
        assert 'CANONICAL' in text and 'CHECKOUT' not in text and ('NEXT' not in text)

def test_missing_local_mirror_is_not_permission_for_editorial_inference(tmp_path, monkeypatch):
    root, local, game, client = fixtures(tmp_path)
    (root / 'data/zeitgeist_state.json').unlink()
    store = state.get_state_store(root, backend='firestore', firestore_client=client)
    from src.zeitgeist import adk_pipeline
    monkeypatch.setattr(adk_pipeline, 'run_daily_zeitgeist', lambda **kwargs: pytest.fail('No providers during canonical rendering'))
    written = build_readme.build_all_readmes(root, state_store=store)
    assert len(written) == 9 and 'CANONICAL' in (root / 'README.md').read_text()

def test_canonical_auth_failure_cannot_render_from_stale_checkout(tmp_path, monkeypatch):
    root, local, game, client = fixtures(tmp_path)
    before = files(root)
    store = state.get_state_store(root, backend='firestore', firestore_client=client)

    def denied(**kwargs):
        raise HTTPError('https://firestore.googleapis.com/', 403, 'denied', {}, BytesIO())
    monkeypatch.setattr(store, 'canonical_projections', denied)
    with pytest.raises(HTTPError):
        build_readme.build_all_readmes(root, state_store=store)
    assert files(root) == before

def test_firestore_get_retries_a_tls_timeout_without_any_write(monkeypatch):
    requests = []
    sleeps = []

    def fetch(request, **kwargs):
        requests.append(request)
        if len(requests) == 1:
            raise URLError(TimeoutError('TLS handshake timed out'))
        return BytesIO(json.dumps({'fields': {'turn': {'integerValue': '3'}}, 'updateTime': '2026-10-30T20:00:00Z'}).encode())
    monkeypatch.setattr(state.urllib.request, 'urlopen', fetch)
    monkeypatch.setattr(state.time, 'sleep', sleeps.append)
    snapshot = state._FirestoreRestClient('test-project', access_token='synthetic').get_document('profile_state', 'active_game')
    assert snapshot.to_dict() == {'turn': 3}
    assert len(requests) == 2 and all((r.get_method() == 'GET' for r in requests)) and (len(sleeps) == 1)

@pytest.mark.parametrize('status', [401, 403, 404, 500])
def test_firestore_http_status_never_becomes_a_transport_retry(monkeypatch, status):
    calls = []

    def fail(request, **kwargs):
        calls.append(request)
        raise HTTPError(request.full_url, status, 'fixed', {}, BytesIO())
    monkeypatch.setattr(state.urllib.request, 'urlopen', fail)
    monkeypatch.setattr(state.time, 'sleep', lambda seconds: pytest.fail('HTTP failures must not retry'))
    client = state._FirestoreRestClient('test-project', access_token='synthetic')
    if status == 404:
        assert client.get_document('profile_state', 'active_game').exists is False
    else:
        with pytest.raises(HTTPError):
            client.get_document('profile_state', 'active_game')
    assert len(calls) == 1

def test_persistent_firestore_transport_failure_stops_after_three_reads(monkeypatch):
    calls = []
    sleeps = []

    def fail(request, **kwargs):
        calls.append(request)
        raise URLError(TimeoutError('TLS handshake timed out'))
    monkeypatch.setattr(state.urllib.request, 'urlopen', fail)
    monkeypatch.setattr(state.time, 'sleep', sleeps.append)
    with pytest.raises(URLError):
        state._FirestoreRestClient('test-project', access_token='synthetic').get_document('profile_state', 'active_game')
    assert len(calls) == 3 and len(sleeps) == 2 and all((r.get_method() == 'GET' for r in calls))


class _DailyImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img' and '/daily_pulse_' in attrs.get('src', ''):
            self.images.append(attrs)


def _accepted_publication(tmp_path):
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / 'accepted'
    for name in ('assets', 'data', 'config'):
        shutil.copytree(source / name, root / name)
    shutil.copy2(source / 'PROJECTS.md', root / 'PROJECTS.md')
    from src.game_engine import create_initial_game_state
    store = state.JsonFileStateStore(root)
    store.save_game_state(create_initial_game_state())
    return root, store


def test_all_readmes_use_github_theme_images_without_changing_accepted_art_or_state(tmp_path):
    root, store = _accepted_publication(tmp_path)
    protected = {p.relative_to(root): p.read_bytes() for p in (root / 'assets').rglob('*')
                 if p.is_file() and ('daily_pulse' in p.name or p.name == 'pulse_manifest.json' or 'pulse' in p.parts)}
    recorded = {p.name: p.read_bytes() for p in (root / 'data').glob('*.json')}
    written = build_readme.build_all_readmes(root, repo_slug='Siyuan-Zhu/readme-live-sandbox', state_store=store)
    assert len(written) == 9 and build_readme.check_all_readmes(
        root, repo_slug='Siyuan-Zhu/readme-live-sandbox', state_store=store)
    for name, path in written.items():
        page = _DailyImages()
        content = path.read_text()
        page.feed(content)
        assert len(page.images) == 2
        prefix = './assets' if name == 'default' else '../assets'
        for image, theme in zip(page.images, ('light', 'dark')):
            assert image['src'].startswith(f'{prefix}/daily_pulse_{theme}.webp?v=')
            assert image['src'].endswith(f'#gh-{theme}-mode-only')
            assert image['width'] == '100%'
            assert image['alt']
        destinations = re.findall(r'https://github.com/[^/]+/[^/]+/issues/new\?', content)
        assert destinations == ['https://github.com/Siyuan-Zhu/readme-live-sandbox/issues/new?'] * 11
        assert 'title=pulse:vote:spot_on' in content
        assert 'title=game:move:connect4:col:4:turn=1:round=1' in content
    assert all((root / path).read_bytes() == raw for path, raw in protected.items())
    assert {p.name: p.read_bytes() for p in (root / 'data').glob('*.json')} == recorded


def test_same_day_accepted_art_change_refreshes_all_daily_image_urls(tmp_path):
    from PIL import Image
    from src.zeitgeist.publication import write_publication_manifest
    root, store = _accepted_publication(tmp_path)
    written = build_readme.build_all_readmes(root, state_store=store)
    before = {}
    for name, path in written.items():
        page = _DailyImages()
        page.feed(path.read_text())
        before[name] = [image['src'] for image in page.images]
    old = json.loads((root / 'assets/pulse_manifest.json').read_text())
    # A separately accepted same-day artwork correction, without changing copy
    # or votes. Only this test fixture's light asset receives a changed pixel.
    asset = root / 'assets/daily_pulse_light.webp'
    with Image.open(asset) as current:
        corrected = current.convert('RGB')
    corrected.putpixel((0, 0), (255, 0, 0))
    corrected.save(asset, format='WEBP', lossless=True)
    accepted = write_publication_manifest(root, date=old['date'], word=old['word'],
        gloss=old['gloss'], archetype=old['archetype'], medium=old['medium'],
        archive_path=f"assets/pulse/{old['date']}.webp")
    assert accepted['publication_id'] != old['publication_id']
    pulse = store.load_pulse_state()
    pulse['publication_id'] = accepted['publication_id']
    pulse['published_pulse']['publication_id'] = accepted['publication_id']
    store.save_pulse_state(pulse)
    feed_path = root / 'data/pulse_feed.json'
    feed = json.loads(feed_path.read_text())
    for entry in feed['entries']:
        if str(entry['date'])[:10] == accepted['date']:
            entry['publication_id'] = accepted['publication_id']
    feed_path.write_text(json.dumps(feed))
    changed = build_readme.build_all_readmes(root, state_store=store)
    assert build_readme.check_all_readmes(root, state_store=store)
    for name, path in changed.items():
        page = _DailyImages()
        page.feed(path.read_text())
        assert all(image['src'] != old_url for image, old_url in zip(page.images, before[name]))
