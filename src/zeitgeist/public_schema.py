"""The public editorial DTO and its narrowly typed provenance extensions."""
import copy
from datetime import date
import json
import re
from pathlib import Path

LEGACY_EDITORIAL_FIELDS = frozenset('''schema_version date word gloss commissioned_thesis
visual_medium image_url fallback_webp_path unsplash_photographer unsplash_profile_url
flux_model cluster_keywords orbit_contribution_pct macro_contribution_pct
orbit_sources_matched runner_up_clusters mood_votes critic_scores published_at updated_at
headline summary 24h_shift dominant_category dominant_signal_type orbit_weight_ratio
top_orbit_themes top_macro_themes visual_tier_used artwork_title artwork_artist
artwork_source artwork_url image_data_uri quote quote_attribution window_hours palette_hex
curatorial_archetype text_quadrant date_iso word_of_the_day explanation_phrase composite_score'''.split())
PHOTO_POLICY_FIELDS = frozenset({'photo_selection_policy', 'photo_selection_policy_date'})
PUBLIC_EDITORIAL_FIELDS = LEGACY_EDITORIAL_FIELDS | PHOTO_POLICY_FIELDS | {
    'publication_id', 'release_republication'}


def public_republication(value):
    fields = {'kind', 'original_publication_id', 'original_publication_date',
              'original_published_at', 'new_inference_performed',
              'artwork_reused_byte_exact'}
    if not isinstance(value, dict):
        raise ValueError('Invalid public republication source')
    record = {key: copy.deepcopy(value[key]) for key in sorted(fields & value.keys())}
    if (set(record) != fields or record['kind'] != 'initial_release_republication'
            or not isinstance(record['original_publication_id'], str)
            or not re.fullmatch(r'[a-f0-9]{20}', record['original_publication_id'])
            or record['new_inference_performed'] is not False
            or record['artwork_reused_byte_exact'] is not True):
        raise ValueError('Invalid public republication provenance')
    from datetime import date, datetime, timedelta
    original_date = record['original_publication_date']
    if not isinstance(original_date, str) or date.fromisoformat(original_date).isoformat() != original_date:
        raise ValueError('Invalid original publication date')
    timestamp = record['original_published_at']
    if timestamp is not None:
        if not isinstance(timestamp, str) or len(timestamp) > 64:
            raise ValueError('Invalid original publication timestamp')
        parsed = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        if parsed.utcoffset() != timedelta(0) or parsed.date().isoformat() != original_date:
            raise ValueError('Original publication timestamp must match its UTC date')
    return record


def photo_policy_projection(payload):
    present = PHOTO_POLICY_FIELDS & payload.keys()
    if not present:
        return {}
    if present != PHOTO_POLICY_FIELDS:
        raise ValueError('Photo provenance requires its policy and date together')
    value = payload['photo_selection_policy_date']
    try:
        valid_date = isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except (TypeError, ValueError):
        valid_date = False
    if (payload['photo_selection_policy'] != 'owner-approved-best-safe-photo'
            or not valid_date or value != str(payload.get('date', ''))[:10]
            or payload.get('visual_medium') != 'PHOTOGRAPHY'):
        raise ValueError('Invalid public photo-selection provenance')
    return {key: payload[key] for key in PHOTO_POLICY_FIELDS}


def validate_editorial_entry(payload):
    if not isinstance(payload, dict):
        raise ValueError('Invalid public editorial object')
    if set(payload) - PUBLIC_EDITORIAL_FIELDS:
        raise ValueError('Disallowed public editorial fields')
    if payload.get('image_data_uri'):
        raise ValueError('Public editorial JSON cannot contain inline image payloads')
    photo_policy_projection(payload)
    if 'release_republication' in payload:
        # Existing release provenance remains its own strict, reviewed contract.
        record = payload['release_republication']
        if public_republication(record) != record:
            raise ValueError('Invalid public republication fields')


def validate_public_editorial(root: Path):
    root = Path(root)
    state = json.loads((root / 'data/zeitgeist_state.json').read_text())
    photo_policy_projection(state)
    if isinstance(state.get('published_pulse'), dict):
        validate_editorial_entry(state['published_pulse'])
    feed = json.loads((root / 'data/pulse_feed.json').read_text())
    if not isinstance(feed, dict) or not isinstance(feed.get('entries'), list):
        raise ValueError('Invalid public editorial feed envelope')
    for entry in feed['entries']:
        validate_editorial_entry(entry)
