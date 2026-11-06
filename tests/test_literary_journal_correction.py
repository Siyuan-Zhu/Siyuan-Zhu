"""Owner decision 2026-11-06: Literary Journal shows no masthead name or paragraph."""
import base64
import json
from pathlib import Path
import shutil

from PIL import Image, ImageChops, ImageDraw
import pytest

from scripts import refresh_studio_footer as footer
from src.zeitgeist import curatorial_compositor as compositor
from src.zeitgeist.publication import write_publication_manifest

ROOT = Path(__file__).resolve().parents[1]
WORD, GLOSS, DATE = 'Lattice', 'Civic space reborn in steel and monumental light.', '2026-11-06'
THESIS = ('The unveiling of a new civic museum marks a profound architectural and institutional '
          'shift, anchoring the city with ambitious engineering and immersive commissions.')


def render(dark=False):
    return compositor._render_literary_journal_plate(
        Image.new('RGB', (1200, 800), 'blue'), word=WORD, gloss=GLOSS, thesis=THESIS, date_str=DATE,
        attribution={'photographer_name': 'Fixture'}, visual_medium='PHOTOGRAPHY', dark_mode=dark)


def legacy(dark=False):
    """The deployed layout before the decision: masthead name plus thesis paragraph."""
    image = render(dark)
    draw = ImageDraw.Draw(image)
    bg = (15, 20, 25) if dark else (251, 249, 245)
    kicker, paragraph = compositor.literary_journal_text_boxes(WORD, GLOSS)
    draw.rectangle(kicker, fill=bg)
    mono = compositor._load_font(compositor._MONO_FONT_CANDIDATES, 12)
    draw.text((56, 168), f'THE DAILY ZEITGEIST  ·  {DATE}', font=mono, fill=(120, 113, 108))
    body = compositor._load_font(compositor._SANS_FONT_CANDIDATES, 14)
    y = paragraph[1] + 4
    for line in ['The unveiling of a new civic museum', 'marks a profound architectural shift']:
        draw.text((56, y), line, font=body, fill=(87, 83, 78))
        y += 22
    return image


def ink_bbox(image, box, dark=False):
    bg = image.getpixel((8, 8))  # encoding may shift the background by a level
    region = image.crop(box)
    return ImageChops.difference(region, Image.new('RGB', region.size, bg)).getbbox()


@pytest.mark.parametrize('dark', [False, True])
def test_new_plates_show_date_only_and_no_paragraph(dark):
    image = render(dark)
    kicker, paragraph = compositor.literary_journal_text_boxes(WORD, GLOSS)
    assert ink_bbox(image, paragraph, dark) is None
    mono = compositor._load_font(compositor._MONO_FONT_CANDIDATES, 12)
    drawn = ink_bbox(image, kicker, dark)
    assert drawn is not None and drawn[2] <= 56 + mono.getlength(DATE) + 2


@pytest.mark.parametrize('dark', [False, True])
def test_correction_removes_text_and_keeps_every_other_pixel(tmp_path, dark):
    path = tmp_path / 'plate.webp'
    legacy(dark).save(path, format='WEBP', quality=85)
    old = Image.open(path).convert('RGB')
    new = compositor.strip_literary_journal_text(old, word=WORD, gloss=GLOSS, date_str=DATE, dark_mode=dark)
    boxes = compositor.literary_journal_text_boxes(WORD, GLOSS)
    assert footer._unchanged_outside(old, new, boxes)
    bg = old.getpixel((8, 8))
    assert ImageChops.difference(new.crop(boxes[1]), Image.new('RGB', new.crop(boxes[1]).size, bg)).getbbox() is None


def test_correction_rejects_unexpected_canvas():
    with pytest.raises(ValueError):
        compositor.strip_literary_journal_text(Image.new('RGB', (100, 100)), word=WORD, gloss=GLOSS,
                                               date_str=DATE, dark_mode=False)


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
    Image.new('RGB', (1200, 800), 'blue').save(tmp_path / entry['fallback_webp_path'], format='WEBP', quality=85)
    for theme in ('light', 'dark'):
        legacy(theme == 'dark').save(tmp_path / f'assets/daily_pulse_{theme}.webp', format='WEBP', quality=85)
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


def test_accepted_literary_journal_correction_is_a_new_exact_publication(accepted):
    root, manifest = accepted
    archive = (root / f'assets/pulse/{DATE}.webp').read_bytes()
    old = {t: Image.open(root / f'assets/daily_pulse_{t}.webp').convert('RGB') for t in ('light', 'dark')}
    corrected = footer.correct_assets(root, manifest['publication_id'])
    assert corrected['publication_id'] != manifest['publication_id']
    assert corrected['archetype'] == 'LITERARY_JOURNAL' and corrected['word'] == WORD
    assert (root / f'assets/pulse/{DATE}.webp').read_bytes() == archive
    boxes = compositor.literary_journal_text_boxes(WORD, GLOSS)
    for theme in ('light', 'dark'):
        new = Image.open(root / f'assets/daily_pulse_{theme}.webp').convert('RGB')
        assert footer._unchanged_outside(old[theme], new, boxes)
        assert ink_bbox(new, boxes[1], theme == 'dark') is None
    svg = (root / 'assets/daily_pulse.svg').read_text()
    assert base64.b64encode((root / 'assets/daily_pulse_dark.webp').read_bytes()).decode() in svg
    assert footer.COMMIT_MESSAGES['LITERARY_JOURNAL'].startswith('fix(pulse): remove Literary Journal')


def test_other_archetypes_are_not_correctable(accepted):
    root, manifest = accepted
    with pytest.raises(ValueError):
        footer.correct_assets(root, '0' * 20)
