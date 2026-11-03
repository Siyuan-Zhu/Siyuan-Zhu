"""Edition-specific approval policy; independent of wall clock and scheduler.

Only explicit owner approval may add an exact publication identity below.
A held draft does not become publishable when the calendar advances.
"""
HELD_EDITIONS = {'2026-11-03': frozenset({'ff043f9a96df9c9a42ec'})}


def require_edition_approval(entry):
    edition = str(entry.get('date', ''))[:10]
    allowed = HELD_EDITIONS.get(edition)
    if allowed is not None and entry.get('publication_id') not in allowed:
        raise PermissionError(f'Edition {edition} requires explicit owner approval of this publication identity')
