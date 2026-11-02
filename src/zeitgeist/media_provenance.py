"""Source-reported medium evidence, independent of provider and editorial fit.

These are bounded source claims, not forensic image-authenticity guarantees.
Unknown physical origin consumes the same conservative exception allowance without
being asserted to be AI-generated or physically non-photographic.
"""
import hashlib
import re
from urllib.parse import urlsplit

PHOTO_MEDIA = frozenset({'PHOTOGRAPHY', 'CURATED_PHOTO', 'CURATED_PHOTO_LIBRARY', 'UNSPLASH_PHOTO'})
SOURCE_MEDIA = frozenset({'PHOTOGRAPHY', 'DIGITAL_ARTWORK', 'UNVERIFIED_IMAGE'})
FIELDS = frozenset({'version', 'classification', 'basis', 'provider', 'asset_id',
    'source_url', 'camera_make', 'camera_model', 'source_statement_sha256',
    'allowance_exception', 'source_claim'})

PROCESS_KINDS = frozenset({'3d_particle_system', '3d_render', 'digital_render',
    'digital_artwork', 'computer_generated', 'cgi_render', 'ai_generated', 'tool_generated'})
CREATION_PREFIX = r'\b(?:this|the|my) (?:artwork|image|work|picture|photo) (?:is|was) (?:a |an )?'


def _source_claim(description, match, *, declaration):
    if match is None:
        return {'kind': 'none', 'process': 'unknown', 'statement': ''}
    text = match.group(0).lower()
    process = next((kind for pattern, kind in (
        (r'ai[- ]generated', 'ai_generated'), (r'3[ -]?d particle', '3d_particle_system'),
        (r'3[ -]?d', '3d_render'), (r'digital render', 'digital_render'),
        (r'digital artwork', 'digital_artwork'), (r'computer[- ]generated', 'computer_generated'),
        (r'cgi', 'cgi_render')) if re.search(pattern, text)), 'tool_generated')
    statement = ' '.join(match.group(0).split())
    return {'kind': 'creation_declaration' if declaration else 'process_mention',
        'process': process, 'statement': statement}


def unsplash_evidence(detail):
    """Classify explicit source claims; absence of EXIF alone is never an AI claim."""
    raw_exif = detail.get('exif')
    exif = raw_exif if isinstance(raw_exif, dict) else {}
    make = exif['make'].strip()[:100] if isinstance(exif.get('make'), str) else ''
    model = exif['model'].strip()[:100] if isinstance(exif.get('model'), str) else ''
    description = detail['description'].strip()[:2000] if isinstance(detail.get('description'), str) else ''
    tool = r'(?:blender|cinema 4d|houdini|unreal engine)'
    process = (r'(?:3[ -]?d (?:render(?:ing)?|particle system|artwork)|digital (?:artwork|render(?:ing)?)|'
        r'computer[- ]generated (?:image|render|artwork)|cgi (?:render|artwork)|'
        r'ai[- ]generated (?:image|artwork)|(?:rendered|created) (?:in|with|using) ' + tool + r')')
    # A self-referential creation statement or a complete process-only caption is
    # evidence. Leading subject nouns followed by exhibition/screen context are
    # not declarations that this asset was rendered.
    declaration = re.search(CREATION_PREFIX + process + r'\b', description, re.I)
    caption = re.fullmatch(r'(?:a |an )?' + process
        + r'(?: (?:created|rendered) (?:in|with|using) ' + tool + r')?[.!]?', description, re.I)
    subject = re.search(r'\b(?:photographed|photograph of|photo of|exhibition|'
        r'on (?:a |the )?(?:computer )?screen)\b', description, re.I)
    digital = bool((declaration or caption) and not subject)
    camera = bool(make or model)
    possible_conflict = bool(camera and (declaration or caption
        or re.search(process, description, re.I) and not subject))
    if digital and not camera:
        kind, basis = 'DIGITAL_ARTWORK', 'source_declared_digital'
    elif camera and not possible_conflict and not digital:
        kind, basis = 'PHOTOGRAPHY', 'camera_reported'
    else:
        kind = 'UNVERIFIED_IMAGE'
        basis = 'conflicting_evidence' if possible_conflict else 'insufficient_evidence'
    claim = _source_claim(description, declaration or caption or re.search(process, description, re.I), declaration=bool(declaration or caption))
    asset_id = str(detail.get('id') or '')[:100]
    return {'version': 1, 'classification': kind, 'basis': basis,
        'provider': 'unsplash', 'asset_id': asset_id,
        'source_url': 'https://unsplash.com/photos/' + asset_id,
        'camera_make': make, 'camera_model': model,
        'source_claim': claim,
        'source_statement_sha256': hashlib.sha256(claim['statement'].encode()).hexdigest() if claim['statement'] else '',
        'allowance_exception': kind != 'PHOTOGRAPHY'}


def generated_evidence(medium):
    procedural = medium == 'BESPOKE_SVG'
    return {'version': 1,
        'classification': 'PROCEDURAL_ARTWORK' if procedural else 'GENERATED_ARTWORK',
        'basis': 'generator_known', 'provider': {'FLUX_PRO': 'bfl', 'IMAGEN_3': 'google'}.get(medium, 'egm'),
        'asset_id': '', 'source_url': '', 'camera_make': '', 'camera_model': '',
        'source_statement_sha256': '', 'allowance_exception': True,
        'source_claim': {'kind': 'none', 'process': 'unknown', 'statement': ''}}


def public_media_provenance(value, *, medium):
    """Validate a closed public record instead of copying raw provider metadata."""
    if not isinstance(value, dict) or set(value) != FIELDS or type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Invalid public media provenance')
    if type(value['allowance_exception']) is not bool:
        raise ValueError('Invalid media allowance evidence')
    kind = value['classification']
    bases = {'PHOTOGRAPHY': {'camera_reported'}, 'DIGITAL_ARTWORK': {'source_declared_digital'},
        'UNVERIFIED_IMAGE': {'insufficient_evidence', 'conflicting_evidence'},
        'GENERATED_ARTWORK': {'generator_known'}, 'PROCEDURAL_ARTWORK': {'generator_known'}}
    if not isinstance(kind, str) or kind not in bases or not isinstance(value['basis'], str) or value['basis'] not in bases[kind] or value['allowance_exception'] != (kind != 'PHOTOGRAPHY'):
        raise ValueError('Inconsistent public media classification')
    if not isinstance(medium, str) or medium not in SOURCE_MEDIA | {'FLUX_PRO', 'IMAGEN_3', 'BESPOKE_SVG'}:
        raise ValueError('Unsupported selected media evidence')
    expected = medium if medium in SOURCE_MEDIA else ('PROCEDURAL_ARTWORK' if medium == 'BESPOKE_SVG' else 'GENERATED_ARTWORK')
    if kind != expected:
        raise ValueError('Media evidence disagrees with selected medium')
    for key, limit in (('asset_id', 100), ('source_url', 500), ('camera_make', 100), ('camera_model', 100), ('source_statement_sha256', 64)):
        if not isinstance(value[key], str) or len(value[key]) > limit or any(ord(c) < 32 for c in value[key]):
            raise ValueError('Invalid public media evidence field')
    digest = value['source_statement_sha256']
    claim = value['source_claim']
    if (not isinstance(claim, dict) or set(claim) != {'kind', 'process', 'statement'}
            or not isinstance(claim['kind'], str) or claim['kind'] not in {'none', 'creation_declaration', 'process_mention'}
            or not isinstance(claim['process'], str) or claim['process'] not in PROCESS_KINDS | {'unknown'}
            or not isinstance(claim['statement'], str) or len(claim['statement']) > 240
            or len(claim['statement'].split()) > 25 or any(ord(c) < 32 for c in claim['statement'])):
        raise ValueError('Invalid retained source claim')
    if ((claim['kind'] == 'none') != (claim['process'] == 'unknown')
            or (claim['kind'] == 'none') != (claim['statement'] == '')
            or digest != (hashlib.sha256(claim['statement'].encode()).hexdigest() if claim['statement'] else '')):
        raise ValueError('Source claim and statement identity disagree')
    if kind == 'DIGITAL_ARTWORK' and claim['kind'] != 'creation_declaration':
        raise ValueError('Digital classification requires a source creation claim')
    if claim['kind'] != 'none':
        derived = unsplash_evidence({'id': value['asset_id'], 'description': claim['statement']})['source_claim']
        if (derived['process'] != claim['process'] or derived['statement'] != claim['statement']
                or claim['kind'] == 'creation_declaration' and derived['kind'] != 'creation_declaration'):
            raise ValueError('Source claim is not a bounded recognized process span')
        if claim['kind'] == 'process_mention' and re.search(CREATION_PREFIX, claim['statement'], re.I):
            raise ValueError('Explicit creation declaration cannot be downgraded to a subject mention')
    camera_present = bool(value['camera_make'] or value['camera_model'])
    if (kind == 'DIGITAL_ARTWORK' and camera_present
            or kind == 'PHOTOGRAPHY' and claim['kind'] == 'creation_declaration'
            or value['basis'] == 'conflicting_evidence' and (not camera_present or claim['kind'] == 'none')):
        raise ValueError('Conflicting physical-origin evidence')
    if kind == 'PHOTOGRAPHY' and not (value['camera_make'] or value['camera_model']):
        raise ValueError('Reported camera evidence is missing')
    if kind == 'DIGITAL_ARTWORK' and not digest:
        raise ValueError('Source process statement identity is missing')
    if digest and not re.fullmatch('[a-f0-9]{64}', digest):
        raise ValueError('Invalid public media statement identity')
    if kind in SOURCE_MEDIA:
        source = urlsplit(value['source_url'])
        if (value['provider'] != 'unsplash' or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value['asset_id'])
                or source.scheme != 'https' or source.netloc != 'unsplash.com'
                or source.path != '/photos/' + value['asset_id'] or source.query or source.fragment):
            raise ValueError('Invalid public media source')
    elif value['camera_make'] or value['camera_model'] or digest:
        raise ValueError('Generated media cannot carry invented camera/process evidence')
    elif value['provider'] != {'FLUX_PRO': 'bfl', 'IMAGEN_3': 'google', 'BESPOKE_SVG': 'egm'}[medium] or value['source_url'] or value['asset_id']:
        raise ValueError('Invalid generated media source')
    return dict(value, source_claim=dict(claim))


def exception_allowed(history):
    """One <=20% including-candidate rule; exceptions cannot be consecutive."""
    recent = [str(value).upper() for value in list(history)[:29]]
    return bool(recent and recent[0] in PHOTO_MEDIA and
                (sum(value not in PHOTO_MEDIA for value in recent) + 1) * 5 <= len(recent) + 1)


def publication_media_history(entries, *, replacing_date):
    """Count accepted dates once, excluding replacement; keep uncertainty conservative."""
    days = {}
    for index, row in enumerate(entries):
        if not isinstance(row, dict):
            continue
        day = str(row.get('date', ''))[:10]
        if day == replacing_date[:10]:
            continue
        medium = str(row.get('visual_medium') or row.get('visual_tier_used') or 'UNKNOWN')
        if not day:
            day, medium = 'undated-' + str(index), 'UNKNOWN'
        if day in days and days[day] != medium:
            days[day] = 'UNKNOWN'
        else:
            days.setdefault(day, medium)
    return list(days.values())[:29]


def public_media_correction(value):
    """Retain the original selected medium/identity without rewriting its event."""
    fields = {'kind', 'original_publication_id', 'original_visual_medium'}
    if (not isinstance(value, dict) or set(value) != fields
            or value['kind'] != 'source_medium_correction'
            or value['original_visual_medium'] != 'PHOTOGRAPHY'
            or not isinstance(value['original_publication_id'], str)
            or not re.fullmatch('[a-f0-9]{20}', value['original_publication_id'])):
        raise ValueError('Invalid original media selection evidence')
    return dict(value)
