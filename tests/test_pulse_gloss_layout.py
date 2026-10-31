"""Editorial admission and real-font plate geometry keep one complete gloss."""
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.zeitgeist.adk_pipeline import (
    DirectorCommission, commission_daily_zeitgeist, validate_director_commission,
)


def commission(gloss):
    return DirectorCommission(word='Dilution', gloss=gloss,
        commissioned_thesis='Capital expenditure spreads ownership and attention.')


@pytest.mark.parametrize('gloss,valid', [
    ('A'*50, True), ('A'*51, False),
    ('a b c d e f g h i j k l', True), ('a b c d e f g h i j k l m', False),
    ('The thinning of ownership, margin, and focus through relentless capital expenditure.', False),
    ('Two lines\nof dilution.', False),
])
def test_director_gloss_has_both_character_and_word_limits(gloss, valid):
    ok, reason = validate_director_commission(commission(gloss))
    assert ok is valid, reason


def test_director_retries_overlong_gloss_without_changing_accepted_text():
    prompts = []
    accepted = 'Capital thins ownership and focus.'
    def director(prompt):
        prompts.append(prompt)
        return commission('A'*51 if len(prompts) == 1 else accepted)
    result = commission_daily_zeitgeist([], llm_callable=director)
    assert result.gloss == accepted and result.attempts_used == 2
    assert '50 characters' in prompts[0] and '12 words' in prompts[0]
    assert '51' in prompts[1] and 'REJECTED' in prompts[1]


def test_exhausted_overlong_director_fails_closed_without_demo_fallback():
    calls = []
    def director(prompt):
        calls.append(prompt)
        return commission('A'*51)
    with pytest.raises(ValueError, match='gloss|Gloss'):
        commission_daily_zeitgeist([], llm_callable=director)
    assert len(calls) == 3


@pytest.mark.parametrize('quadrant', ['upper_left','upper_right','lower_left','lower_right'])
def test_poetic_text_uses_real_font_bounds_and_complete_pixel_wrapping(quadrant):
    from src.zeitgeist.curatorial_compositor import layout_poetic_cinema_text, QUADRANT_BOUNDS_1200X800
    # Wide glyphs expose why character-count wrapping cannot prove a safe fit.
    word, gloss = 'Wide Margins', 'W'*50
    hero = Image.new('RGB', (1200,800), (240,238,234))
    layout = layout_poetic_cinema_text(hero, word=word, gloss=gloss, quadrant=quadrant)
    x0,y0,x1,y1 = QUADRANT_BOUNDS_1200X800[quadrant]
    assert ''.join(r['text'] for r in layout['runs'] if r['italic']) == gloss
    for run in layout['runs']:
        assert run['font_size'] == (21.25 if run['italic'] else 60)
        left,top,right,bottom = run['bounds']
        assert x0+48 <= left < right <= x1-48
        assert y0+48 <= top < bottom <= y1-48


def test_poetic_ink_is_readable_on_local_bright_text_background_not_whole_quadrant():
    from src.zeitgeist.curatorial_compositor import layout_poetic_cinema_text
    hero = Image.new('RGB', (1200,800), (12,12,12))
    ImageDraw.Draw(hero).rectangle((648,448,1152,752), fill=(245,245,245))
    layout = layout_poetic_cinema_text(hero, word='Dilution',
        gloss='Capital thins ownership and focus.', quadrant='lower_right')
    assert max(layout['ink']) < 70
    # Independent WCAG contrast calculation against the actual chosen text region.
    for run in layout['runs']:
        pixels = np.asarray(hero.crop(run['bounds']), dtype=float)/255
        rgb = np.where(pixels <= .04045, pixels/12.92, ((pixels+.055)/1.055)**2.4)
        luminance = rgb @ np.array([.2126,.7152,.0722])
        ink = np.array(layout['ink'])/255
        ink = np.where(ink <= .04045, ink/12.92, ((ink+.055)/1.055)**2.4)
        ink_luminance = ink @ np.array([.2126,.7152,.0722])
        contrast = (np.maximum(luminance,ink_luminance)+.05)/(np.minimum(luminance,ink_luminance)+.05)
        assert np.percentile(contrast,5) >= 4.5
    assert layout['scrim_opacity'] <= .18


@pytest.mark.parametrize('word',['Countermovements','Common Ground'])
def test_enlarged_literary_title_stays_inside_existing_column(word):
    from src.zeitgeist.curatorial_compositor import layout_literary_title,_load_font,_SERIF_FONT_CANDIDATES
    font=_load_font(_SERIF_FONT_CANDIDATES,57.5)
    runs,gloss_y=layout_literary_title(word)
    assert ''.join(text for text,_ in runs).replace(' ','')==word.replace(' ','')
    for text,(x,y) in runs:
        left,top,right,bottom=font.getbbox(text)
        assert x+left>=56 and x+right<=416
        assert y+bottom<gloss_y


def test_enlarged_concrete_title_preserves_full_word_inside_grid():
    from src.zeitgeist.curatorial_compositor import layout_concrete_title,_load_font,_SERIF_FONT_CANDIDATES
    runs,gloss_y,size=layout_concrete_title('Common Commonwealth')
    font=_load_font(_SERIF_FONT_CANDIDATES,size)
    assert ''.join(t for t,_ in runs).replace(' ','')=='COMMONCOMMONWEALTH'
    for text,(x,y) in runs:
        left,top,right,bottom=font.getbbox(text)
        assert x+left>=84 and x+right<=1164 and y+bottom<gloss_y
