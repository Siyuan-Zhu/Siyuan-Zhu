"""Current public FLUX 3 / 1.5k tariff, normalized from two structured sources.

No JavaScript execution, authenticated browser state, provider key or image call.
The public site's JSON-LD price and calculator configuration must agree. This is
an observation, never a future price-validity commitment.
"""
from __future__ import annotations
from decimal import Decimal
from html.parser import HTMLParser
import hashlib
import json
import re

from .automatic_spend import digest
from .spend_guard import SpendBlocked, money

SOURCE_URL = 'https://bfl.ai/pricing'
MAX_HTML_BYTES = 2_000_000
ENDPOINT = '/v1/flux-3-image'


class _Scripts(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.kind = None
        self.parts = []
        self.scripts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'script':
            self.kind = dict(attrs).get('type', 'javascript')
            self.parts = []

    def handle_data(self, data):
        if self.kind is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.kind is not None:
            self.scripts.append((self.kind, ''.join(self.parts)))
            self.kind = None


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('Nonfinite JSON number')
    return json.loads(raw, parse_float=Decimal, parse_constant=constant, object_pairs_hook=pairs)


def _walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk(child)


def _text(value):
    return ' '.join(str(value).split()).casefold()


def normalize_bfl_flux3_tariff(html):
    """Extract the exact model/tier/unit and relevant pricing conditions only."""
    try:
        if not isinstance(html, str) or len(html.encode('utf-8')) > MAX_HTML_BYTES:
            raise ValueError('Invalid public pricing HTML size')
        parser = _Scripts(); parser.feed(html); parser.close()
        products, flight = [], []
        for kind, raw in parser.scripts:
            if kind == 'application/ld+json':
                products.extend(value for value in _walk(_json(raw))
                    if value.get('@type') == 'Product' and value.get('name') == 'FLUX 3 Image')
            else:
                # Next.js flight transports JSON strings. Decode JSON only; never
                # eval JS or execute the public calculator's script bundles.
                match = re.fullmatch(r'\s*self\.__next_f\.push\((\[.*\])\)\s*;?\s*', raw, re.DOTALL)
                if match:
                    record = _json(match[1])
                    if isinstance(record, list) and len(record) == 2 and record[0] == 1 and isinstance(record[1], str):
                        flight.append(record[1])
        cards = []
        for line in ''.join(flight).splitlines():
            prefix, separator, raw = line.partition(':')
            if separator and re.fullmatch('[a-fA-F0-9]+', prefix) and raw.startswith(('[', '{')):
                for value in _walk(_json(raw)):
                    if value.get('apiEndpoint') == ENDPOINT and 'imageTierRates' in value:
                        cards.append(value)
        if len(products) != 1 or len(cards) != 1:
            raise ValueError('Exact unique product and endpoint required')
        product, card = products[0], cards[0]
        if product.get('brand', {}).get('name') != 'Black Forest Labs':
            raise ValueError('Product brand mismatch')
        offer = product['offers']
        if set(offer) - {'@type','priceCurrency','price','priceSpecification','url','seller','availability'}:
            raise ValueError('Unreviewed offer eligibility or pricing conditions')
        if offer.get('@type') != 'Offer' or offer.get('priceCurrency') != 'USD' or offer.get('url') != SOURCE_URL:
            raise ValueError('Product offer identity mismatch')
        seller = offer.get('seller', {})
        if (set(seller) - {'@type','name','url'} or seller.get('name') != 'Black Forest Labs'
            or seller.get('url') != 'https://bfl.ai' or seller.get('@type', 'Organization') != 'Organization'):
            raise ValueError('Offer seller identity or conditions changed')
        known_tiers = {'768 × 768 image','1k image','1.5k image','2k image','4k image'}
        names = [_text(item.get('name')) for item in offer['priceSpecification']]
        if len(names) != len(set(names)) or not set(names) <= known_tiers:
            raise ValueError('Additional or ambiguous price component')
        for item in offer['priceSpecification']:
            if set(item) != {'@type','name','priceCurrency','price','unitText','referenceQuantity'}:
                raise ValueError('Unreviewed price component conditions')
        specs = [s for s in offer['priceSpecification'] if _text(s.get('name')) == '1.5k image']
        if len(specs) != 1:
            raise ValueError('Exact unique resolution price required')
        spec = specs[0]
        if (set(spec) != {'@type','name','priceCurrency','price','unitText','referenceQuantity'} or
            spec['@type'] != 'UnitPriceSpecification' or spec['priceCurrency'] != 'USD' or spec['unitText'] != 'image'):
            raise ValueError('Ambiguous currency, unit or conditional price')
        quantity = spec['referenceQuantity']
        if set(quantity) != {'@type','value','unitText'} or quantity['@type'] != 'QuantitativeValue' or quantity['unitText'] != 'image' or isinstance(quantity['value'], bool) or money(quantity['value']) != 1:
            raise ValueError('Price must be for one generated image')
        title = ''.join(str(span.get('text', '')) for block in card['title'] for span in block.get('children', []))
        if title != 'FLUX 3 Image' or card.get('category', {}).get('name') != 'FLUX 3 Image':
            raise ValueError('Calculator model identity mismatch')
        alternate = ('firstMpPrice','inputMpPrice','mpPrice','mpSecondPrice','priceUnit','secondPrice','videoRates')
        if any(key not in card or card[key] is not None for key in alternate):
            raise ValueError('Additional calculator pricing mode is not reviewed')
        tiers = card['imageTierRates']
        if tiers.get('_type') != 'imageTierRates':
            raise ValueError('Explicit image resolution tier required')
        usd = money(spec['price'])
        if 'price' in offer and money(offer['price']) > usd:
            raise ValueError('Offer starting price contradicts the selected tier')
        # BFL documents one credit = USD0.01. Cross-check the calculator's
        # seven-credit tier with its independently published USD price.
        if usd <= 0 or money(tiers['tier1_5k']) * Decimal('.01') != usd:
            raise ValueError('Public USD and credit tariff disagree')
        terms = []
        for notes in (card['category'].get('footnotes'), card.get('footnotes')):
            if notes is None:
                continue
            if not isinstance(notes, list) or any(not isinstance(note, str) for note in notes):
                raise ValueError('Unrecognized pricing terms')
            terms.extend(_text(note) for note in notes)
        terms = sorted(set(terms))
        all_terms = ' '.join(terms)
        required = (
            'pricing is per image generated.',
            'charges are based on the resolution tier selected in the request',
            'reference images, whether provided by the user or retrieved by the model, and prompts are included at no extra charge.',
            'the per-image rate does not vary with the number of reference images provided',
            'all supported aspect ratios within a tier are charged the same.',
        )
        if any(clause not in all_terms for clause in required):
            raise ValueError('Resolution, retrieval or aspect-ratio conditions missing')
        return {'schema_version': 1, 'provider': 'bfl', 'operation': 'generate_image:flux-3-image',
            'model': 'FLUX 3 Image', 'endpoint': ENDPOINT, 'resolution': '1.5k',
            'currency': 'USD', 'unit': 'generated_image', 'quantity': 1,
            'unit_price_usd': format(usd.normalize(), 'f'),
            'request_shape': {'resolution':'1.5k','aspect_ratio':'3:2','grounding':True,
                              'image_count':1,'supplied_reference_images':0,'explicit_width_height':False},
            'pricing_flags': {'per_selected_resolution_tier':True,'all_supported_aspect_ratios_same_price':True,
                              'user_reference_images_included':True,'model_retrieved_reference_images_included':True},
            # New pricing conditions change the digest even if the four known
            # affirmative clauses remain present; a surcharge cannot be hidden.
            'pricing_terms': terms}
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError) as error:
        if isinstance(error, SpendBlocked):
            raise
        raise SpendBlocked('Public BFL tariff is unavailable, ambiguous or changed') from None


def observe_bfl_flux3_tariff(client, *, now):
    """One fixed unauthenticated public GET; suitable for price_snapshots['bfl']."""
    try:
        sensitive = {'authorization','proxy-authorization','x-key','x-goog-api-key'}
        if getattr(client, 'auth', None) is not None or any(str(key).lower() in sensitive for key in getattr(client, 'headers', {})):
            raise ValueError('Public pricing must use an unauthenticated client')
        response = client.get(SOURCE_URL, follow_redirects=False, timeout=15)
        if response.status_code != 200 or str(response.url) != SOURCE_URL or response.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'text/html':
            raise ValueError('Public pricing response must be direct HTML on exact origin')
        raw = response.content
        if not isinstance(raw, bytes) or len(raw) > MAX_HTML_BYTES:
            raise ValueError('Public pricing byte ceiling exceeded')
        contract = normalize_bfl_flux3_tariff(raw.decode('utf-8'))
        return {'source_url': SOURCE_URL, 'observed_at': now.isoformat(), 'sha256': digest(contract),
                'contract': contract, 'source_body_sha256': hashlib.sha256(raw).hexdigest()}
    except SpendBlocked:
        raise
    except Exception:
        raise SpendBlocked('Public BFL tariff observation unavailable; no price inferred') from None
