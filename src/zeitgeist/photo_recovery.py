"""Bounded editorial recovery; photographs first and guarded AI exceptions."""
from .visual_router import PhotoSelectionUnavailable

PHOTO_MEDIA = frozenset({'PHOTOGRAPHY','CURATED_PHOTO'})


def ai_exception_allowed(history):
    """Include the proposed publication in the <=20% rolling mix; never consecutive."""
    recent = [str(value).upper() for value in list(history)[:29]]
    if not recent or recent[0] not in PHOTO_MEDIA:
        return False
    nonphoto = sum(value not in PHOTO_MEDIA for value in recent)
    return (nonphoto + 1) * 5 <= len(recent) + 1


def recover_visual(commission, *, render, revise, history, photo_required=False):
    """Initial search, one revised search, one same-story word revision; then eligible AI.

    Only an explicit no-photo verdict permits recovery. Provider authorization,
    budget, storage, decoding and publication failures propagate without replay.
    """
    current = commission
    trace = []
    for phase in ('initial_photo','requery_photo','reword_photo'):
        if phase != 'initial_photo':
            current = revise(current, change_word=phase=='reword_photo')
        try:
            result = render(current, photo_only=True)
            trace.append({'phase':phase,'status':'accepted'})
            return current, result, trace
        except PhotoSelectionUnavailable:
            trace.append({'phase':phase,'status':'no_qualifying_photo'})
    if not photo_required and ai_exception_allowed(history):
        result = render(current, photo_only=False)
        trace.append({'phase':'bounded_ai_exception','status':'accepted'})
        return current, result, trace
    raise PhotoSelectionUnavailable('Bounded photography and same-story revision attempts exhausted; no new publication')


def revision_owner_constraints(owner, current, *, change_word):
    """Preserve owner locks; normalize model text only for the plain-text note seam."""
    constraints = dict(owner)
    note = (str(owner.get('personal_note') or '') + " Keep the same factual story. "
            + ("Choose a subtler word and matching gloss. " if change_word else "Keep word and gloss; broaden concrete photo queries. ")
            + "Original thesis: " + current.commissioned_thesis + ". Previous queries: " + ", ".join(current.unsplash_queries))
    note = ''.join(ch if ord(ch)>=32 and ord(ch)!=127 else ' ' for ch in note)
    note = ' '.join(note.replace('"', "'").replace('<','(').replace('>',')').split())[:950]
    constraints.update(enabled=True, personal_note=note)
    if not change_word:
        constraints.setdefault('override_word', current.word)
        constraints.setdefault('override_phrase', current.gloss)
        # Normalized absent owner fields are None, not missing keys.
        constraints['override_word'] = constraints.get('override_word') or current.word
        constraints['override_phrase'] = constraints.get('override_phrase') or current.gloss
    return constraints
