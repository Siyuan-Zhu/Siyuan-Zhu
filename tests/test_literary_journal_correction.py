"""Owner decisions 2026-11-06: Literary Journal is a gallery label with no masthead or paragraph."""
import base64
import json
from pathlib import Path
import shutil

from PIL import Image, ImageChops, ImageOps
import pytest

from scripts import refresh_studio_footer as footer
from src.zeitgeist import curatorial_compositor as compositor
from src.zeitgeist.publication import write_publication_manifest

ROOT = Path(__file__).resolve().parents[1]
WORD, GLOSS, DATE = 'Lattice', 'Civic space reborn in steel and monumental light.', '2026-11-06'
THESIS = 'A thesis paragraph that must never be drawn on a Literary Journal plate.'
LEFT, COLUMN = compositor.LITERARY_COLUMN
X, Y, W, H = compositor.LITERARY_ART_BOX


def hero():
    image = Image.new('RGB', (1200, 800), (40, 90, 160))
    image.paste((200, 120, 60), (600, 0, 1200, 800))
    return image


def render(dark=False, word=WORD, gloss=GLOSS):
    return compositor._render_literary_journal_plate(
        hero(), word=word, gloss=gloss, thesis=THESIS, date_str=DATE,
        attribution={'photographer_name': 'Fixture'}, visual_medium='PHOTOGRAPHY', dark_mode=dark)


def ink(image, box):
    bg = image.getpixel((8, 8))
    region = image.crop(box)
    return ImageChops.difference(region, Image.new('RGB', region.size, bg)).getbbox()


@pytest.mark.parametrize('dark', [False, True])
def test_gallery_label_places_untouched_plate_and_draws_no_masthead_or_paragraph(dark):
    image = render(dark)
    expected = ImageOps.fit(hero(), (W, H), method=Image.Resampling.LANCZOS)
    assert ImageChops.difference(image.crop((X, Y, X + W, Y + H)), expected).getbbox() is None
    # The column between the date rule and the word holds nothing: no paragraph, no masthead.
    assert ink(image, (LEFT, Y + 40, LEFT + COLUMN, Y + 280)) is None
    # The top row is the spaced date only.
    date_row = ink(image, (LEFT, Y - 2, LEFT + COLUMN, Y + 20))
    assert date_row is not None and date_row[2] < 150
    # The word and gloss stand on the plate's bottom edge.
    assert ink(image, (LEFT, Y + 300, LEFT + COLUMN, Y + H + 4)) is not None
    assert ink(image, (LEFT, Y + H + 8, LEFT + COLUMN, 800)) is None


def test_long_word_steps_down_to_fit_the_column():
    image = render(word='Stewardship')
    assert ink(image, (LEFT + COLUMN + 2, Y + 300, X - 4, Y + H)) is None


def test_text_that_cannot_fit_the_column_is_refused():
    with pytest.raises(ValueError):
        render(gloss=' '.join(['overflowing'] * 80))


@pytest.fixture
def accepted(tmp_path, monkeypatch):
    monkeypatch.setenv('STATE_STORE_BACKEND', 'json')
    for name in ('assets', 'data', 'config', 'variants'):
        shutil.copytree(ROOT / name, tmp_path / name)
    for name in ('README.md', 'PROJECTS.md'):
        shutil.copy2(ROOT / name, tmp_path / name)
    entry = dict(schema_version='2.0', date=DATE, word=WORD, gloss=GLOSS, visual_medium='PHOTOGRAPHY',
                 visual_tier_used='PHOTOGRAPHY', curatorial_archetype='LITERARY_JOURNAL',
                 fallback_webp_path=f'assets/pulse/{DATE}.webp', unsplash_photographer='Fixture',
                 artwork_artist='Fixture', artwork_source='Unsplash',
                 published_at=f'{DATE}T15:35:00Z', updated_at=f'{DATE}T15:35:00Z',
                 mood_votes={'spot_on': 0, 'too_sunny': 0, 'too_heavy': 0})
    hero().save(tmp_path / entry['fallback_webp_path'], format='WEBP', quality=85)
    for theme in ('light', 'dark'):
        # Any previously accepted layout; the correction re-renders from the archive.
        Image.new('RGB', (1200, 800), (251, 249, 245) if theme == 'light' else (15, 20, 25)).save(
            tmp_path / f'assets/daily_pulse_{theme}.webp', format='WEBP', quality=85)
    encoded = base64.b64encode((tmp_path / 'assets/daily_pulse_dark.webp').read_bytes()).decode()
    (tmp_path / 'assets/daily_pulse.svg').write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" data-publication-layout="curatorial"><metadata id="mood-votes">{}'
        '</metadata><image href="data:image/webp;base64,' + encoded + '"/></svg>')
    (tmp_path / 'data/pulse_feed.json').write_text(json.dumps({'entries': [entry]}))
    manifest = write_publication_manifest(tmp_path, date=DATE, word=WORD, gloss=GLOSS,
        archetype='LITERARY_JOURNAL', medium='PHOTOGRAPHY', archive_path=entry['fallback_webp_path'])
    entry['publication_id'] = manifest['publication_id']
    pulse = {'date': DATE, 'word': WORD, 'gloss': GLOSS, 'visual_medium': 'PHOTOGRAPHY',
             'curatorial_archetype': 'LITERARY_JOURNAL', 'published_pulse': entry,
             'publication_id': entry['publication_id'], 'state_version': 3,
             'visitor_mood_tally': entry['mood_votes']}
    (tmp_path / 'data/zeitgeist_state.json').write_text(json.dumps(pulse))
    (tmp_path / 'data/pulse_feed.json').write_text(json.dumps({'entries': [entry]}))
    return tmp_path, manifest


def test_accepted_edition_is_re_rendered_from_its_archived_photograph(accepted):
    root, manifest = accepted
    archive_path = root / f'assets/pulse/{DATE}.webp'
    archive = archive_path.read_bytes()
    corrected = footer.correct_assets(root, manifest['publication_id'])
    assert corrected['publication_id'] != manifest['publication_id']
    assert corrected['archetype'] == 'LITERARY_JOURNAL' and corrected['word'] == WORD
    assert archive_path.read_bytes() == archive
    with Image.open(archive_path) as source:
        expected = ImageOps.fit(source.convert('RGB'), (W, H), method=Image.Resampling.LANCZOS)
    for theme in ('light', 'dark'):
        new = Image.open(root / f'assets/daily_pulse_{theme}.webp').convert('RGB')
        assert ImageChops.difference(new.crop((X, Y, X + W, Y + H)), expected).getbbox() is None
        assert ink(new, (LEFT, Y + 40, LEFT + COLUMN, Y + 280)) is None
    svg = (root / 'assets/daily_pulse.svg').read_text()
    assert base64.b64encode((root / 'assets/daily_pulse_dark.webp').read_bytes()).decode() in svg
    assert 'Gallery label' in footer.COMMIT_MESSAGES['LITERARY_JOURNAL']


def test_a_different_publication_id_is_refused(accepted):
    root, _ = accepted
    with pytest.raises(ValueError):
        footer.correct_assets(root, '0' * 20)


@pytest.mark.parametrize('channel,delta', [(0, 1), (1, 1), (2, 1), (2, 4)])
def test_pixel_guard_catches_a_one_level_change_in_any_channel(channel, delta):
    old = Image.new('RGB', (1200, 800), (100, 100, 100))
    new = old.copy()
    pixel = [100, 100, 100]
    pixel[channel] += delta
    new.putpixel((10, 10), tuple(pixel))
    assert footer._unchanged_outside(old, new, ((0, 716, 1199, 747),)) is False
    assert footer._unchanged_outside(old, old.copy(), ((0, 716, 1199, 747),)) is True
