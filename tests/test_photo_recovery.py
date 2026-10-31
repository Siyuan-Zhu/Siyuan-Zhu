from types import SimpleNamespace
import pytest
from src.zeitgeist.photo_recovery import ai_exception_allowed,recover_visual
from src.zeitgeist.visual_router import PhotoSelectionUnavailable


def test_mix_counts_proposed_publication_and_never_allows_consecutive_ai():
    assert not ai_exception_allowed([])
    assert not ai_exception_allowed(['PHOTOGRAPHY']*3)
    assert ai_exception_allowed(['PHOTOGRAPHY']*4)
    assert not ai_exception_allowed(['FLUX_PRO']+['PHOTOGRAPHY']*28)
    assert not ai_exception_allowed(['PHOTOGRAPHY']*23+['FLUX_PRO']*6)
    assert ai_exception_allowed(['PHOTOGRAPHY']*24+['FLUX_PRO']*5)


def test_bounded_requery_then_same_story_revision_before_eligible_ai():
    calls=[]
    def render(c,*,photo_only):
        calls.append(('render',c,photo_only))
        if photo_only:raise PhotoSelectionUnavailable('fixture')
        return 'image'
    def revise(c,*,change_word):calls.append(('revise',change_word));return c+' revised'
    word,image,trace=recover_visual('story',render=render,revise=revise,history=['PHOTOGRAPHY']*4)
    assert image=='image' and len(trace)==4
    assert calls==[('render','story',True),('revise',False),('render','story revised',True),('revise',True),('render','story revised revised',True),('render','story revised revised',False)]


def test_photo_required_never_calls_ai_even_when_history_would_allow_it():
    calls=[]
    def render(c,*,photo_only):calls.append(photo_only);raise PhotoSelectionUnavailable('fixture')
    with pytest.raises(PhotoSelectionUnavailable,match='exhausted'):
        recover_visual('story',render=render,revise=lambda c,**kw:c,history=['PHOTOGRAPHY']*20,photo_required=True)
    assert calls==[True]*3


def test_storage_budget_or_unknown_error_is_never_retried():
    calls=[]
    def render(c,**kw):calls.append(c);raise RuntimeError('unknown outcome')
    with pytest.raises(RuntimeError,match='unknown outcome'):
        recover_visual('story',render=render,revise=lambda *a,**k:pytest.fail('must not revise'),history=[])
    assert calls==['story']


def test_owner_locks_and_quoted_generated_context_survive_recovery():
    from src.zeitgeist.photo_recovery import revision_owner_constraints
    from src.zeitgeist.methodology import validate_owner_interjection
    c=SimpleNamespace(word='Other',gloss='Other gloss.',commissioned_thesis='The "reserve" story\nremains grounded.',unsplash_queries=['stone "barrier"'])
    owner={'enabled':True,'override_word':'Anchor','override_phrase':'Locked phrase.','personal_note':'Keep original direction.'}
    constraints=validate_owner_interjection(revision_owner_constraints(owner,c,change_word=True))
    assert constraints['override_word']=='ANCHOR' and constraints['override_phrase']=='Locked phrase.'
    assert 'Keep original direction.' in constraints['personal_note'] and 'reserve' in constraints['personal_note']
