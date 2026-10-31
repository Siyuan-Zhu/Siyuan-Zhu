"""Current public standard-paid tariffs for the two approved Gemini operations.

The generation table publishes a December 31 deadline. The embedding table
publishes a current rate only; no validity commitment is invented for it.
"""
from __future__ import annotations
from html.parser import HTMLParser
import hashlib
import re

from .automatic_spend import digest
from .spend_guard import SpendBlocked, money

SOURCE_URL = 'https://ai.google.dev/gemini-api/docs/pricing'
MAX_HTML_BYTES = 2_000_000
CURRENCY_PREFIX = re.escape(chr(36))
MODELS = {'gemini-3.8-flash': 'gemini-3.8-flash', 'gemini-embedding': 'gemini-embedding-001'}


def _text(value):
    return ' '.join(value.split())


def _label(value):
    return _text(value).casefold()


class _PricingTables(HTMLParser):
    """Read semantic headings/model code/standard tables, never execute markup."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.section = None
        self.tier = None
        self.heading = None
        self.code = None
        self.seen = []
        self.codes = {key: [] for key in MODELS}
        self.scoped_prose = {key: [] for key in MODELS}
        self.tables = []
        self.table = None
        self.table_depth = 0
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            self.table_depth += 1
        if tag == 'footer':
            self.section = None
        if tag == 'h2':
            name = dict(attrs).get('id')
            self.section = name if name in MODELS else None
            self.tier = None
            if self.section:
                self.seen.append(self.section)
        elif tag == 'h3':
            self.heading = []
        elif tag == 'code' and self.section and self.tier is None:
            self.code = []
        elif tag == 'table' and self.section and self.tier == 'standard':
            if self.table is not None or 'pricing-table' not in dict(attrs).get('class', '').split():
                raise ValueError('Unexpected standard pricing table')
            self.table = {'section': self.section, 'rows': []}
        elif tag == 'tr' and self.table is not None:
            self.row = []
        elif tag in ('th','td') and self.row is not None:
            if self.cell is not None:
                raise ValueError('Nested pricing cell')
            self.cell = []
        elif tag == 'br' and self.cell is not None:
            self.cell.append(' ')

    def handle_data(self, data):
        if self.section and self.table_depth == 0 and self.heading is None:
            self.scoped_prose[self.section].append(data)
        if self.heading is not None:
            self.heading.append(data)
        if self.code is not None:
            self.code.append(data)
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == 'h3' and self.heading is not None:
            self.tier = _label(''.join(self.heading))
            self.heading = None
        elif tag == 'code' and self.code is not None:
            self.codes[self.section].append(_text(''.join(self.code)))
            self.code = None
        elif tag in ('th','td') and self.cell is not None:
            self.row.append(_text(''.join(self.cell)))
            self.cell = None
        elif tag == 'tr' and self.row is not None:
            self.table['rows'].append(self.row)
            self.row = None
        elif tag == 'table' and self.table is not None:
            self.tables.append(self.table)
            self.table = None
        if tag == 'table':
            self.table_depth -= 1
            if self.table_depth < 0:
                raise ValueError('Unbalanced pricing tables')


def _rows(parser, section, allowed):
    if parser.seen.count(section) != 1 or parser.codes[section] != [MODELS[section]]:
        raise ValueError('Exact unique model heading and code required')
    tables = [table for table in parser.tables if table['section'] == section]
    if len(tables) != 1:
        raise ValueError('Exact unique Standard pricing table required')
    rows = tables[0]['rows']
    if not rows or [_label(c) for c in rows[0]] != ['', 'free tier', 'paid tier, per 1m tokens in usd']:
        raise ValueError('Explicit paid-tier per-million USD header required')
    paid = {}
    for row in rows[1:]:
        if len(row) != 3:
            raise ValueError('Ambiguous pricing table columns')
        label = _label(row[0])
        # Only this existing annotated label is reviewed. Markers on input or
        # output prices must not disappear before the eligibility/caveat gate.
        if label == 'grounding with google search*':
            label = 'grounding with google search'
        if label not in allowed or label in paid:
            raise ValueError('Unreviewed or duplicate price condition')
        paid[label] = row[2]
    if set(paid) != set(allowed):
        raise ValueError('Required standard price rows missing')
    return paid, {'scoped_model_caveats': _text(' '.join(parser.scoped_prose[section])),
                  'standard_rows': rows}


def _dated_rate(value):
    match = re.fullmatch(CURRENCY_PREFIX + r'(\d+(?:\.\d+)?) through December 31, 2026\. ' + CURRENCY_PREFIX + r'(\d+(?:\.\d+)?) starting January 1, 2027\.', value)
    if not match or money(match[1]) <= 0 or money(match[2]) <= 0:
        raise ValueError('Unrecognized published generation pricing schedule')
    return format(money(match[1]).normalize(),'f'), format(money(match[2]).normalize(),'f')


def normalize_gemini_tariff(html):
    """Normalize exact standard generation+embedding rows, excluding other models.

    Batch/Flex/Priority, free-tier values, caching storage and grounding are not
    admitted by this contract or the bounded SDK policy. New paid row labels,
    conditional input/output prices and ambiguous model/table identities reject.
    All model-scoped outside-table caveats through the next model/footer and
    full Standard row terms are bound, including caching/grounding eligibility
    conditions. Prices in other traffic-tier tables remain outside the digest.
    """
    try:
        if not isinstance(html,str) or len(html.encode('utf-8')) > MAX_HTML_BYTES:
            raise ValueError('Invalid public pricing body')
        parser = _PricingTables(); parser.feed(html); parser.close()
        generation, generation_conditions = _rows(parser, 'gemini-3.8-flash', (
            'input price','output price (including thinking tokens)','context caching price',
            'grounding with google search','grounding with google maps','used to improve our products'))
        embedding, embedding_conditions = _rows(parser, 'gemini-embedding', ('input price','used to improve our products'))
        input_rate, future_input = _dated_rate(generation['input price'])
        output_rate, future_output = _dated_rate(generation['output price (including thinking tokens)'])
        match = re.fullmatch(CURRENCY_PREFIX + r'(\d+(?:\.\d+)?)', embedding['input price'])
        if not match or money(match[1]) <= 0:
            raise ValueError('Unrecognized current input-only embedding tariff')
        return {'schema_version':2,'provider':'gemini','api':'gemini_developer_api',
            'currency':'USD','service_tier':'paid_standard','unit':'1000000_tokens',
            'operations':{
                'generate_content:gemini-3.8-flash':{
                    'model':'gemini-3.8-flash','input_token_usd_per_million':input_rate,
                    'output_token_usd_per_million':output_rate,'output_includes_thinking':True,
                    'published_current_rate_through_date':'2026-12-31',
                    'published_future_rate_from_date':'2027-01-01',
                    'future_input_token_usd_per_million':future_input,
                    'future_output_token_usd_per_million':future_output},
                'embed_content:gemini-embedding-001':{
                    'model':'gemini-embedding-001','input_token_usd_per_million':format(money(match[1]).normalize(),'f'),
                    'billing_basis':'input_tokens','input_modalities':['text'],'output_type':'embedding'}},
            'standard_conditions': {'gemini-3.8-flash':generation_conditions, 'gemini-embedding-001':embedding_conditions},
            'excluded_features':['batch','flex','priority','contributor','grounding','explicit_cache_storage']}
    except (ValueError,TypeError,KeyError,AttributeError,RecursionError):
        raise SpendBlocked('Public Gemini tariff is unavailable, ambiguous or changed') from None


def observe_gemini_tariff(client, *, now):
    """One anonymous fixed-origin public GET; usable as price_snapshots['gemini']."""
    try:
        sensitive = {'authorization','proxy-authorization','x-key','x-goog-api-key'}
        if now.tzinfo is None or getattr(client,'auth',None) is not None or any(str(k).lower() in sensitive for k in getattr(client,'headers',{})):
            raise ValueError('Anonymous client and real timezone-aware observation clock required')
        response = client.get(SOURCE_URL,follow_redirects=False,timeout=15)
        if response.status_code != 200 or str(response.url) != SOURCE_URL or response.headers.get('content-type','').split(';',1)[0].strip().lower() != 'text/html':
            raise ValueError('Direct fixed-origin public HTML required')
        raw=response.content
        if not isinstance(raw,bytes) or len(raw)>MAX_HTML_BYTES:
            raise ValueError('Public pricing byte ceiling exceeded')
        contract=normalize_gemini_tariff(raw.decode('utf-8'))
        return {'source_url':SOURCE_URL,'observed_at':now.isoformat(),'sha256':digest(contract),
            'contract':contract,'source_body_sha256':hashlib.sha256(raw).hexdigest()}
    except SpendBlocked:
        raise
    except Exception:
        raise SpendBlocked('Public Gemini tariff observation unavailable; no price inferred') from None
