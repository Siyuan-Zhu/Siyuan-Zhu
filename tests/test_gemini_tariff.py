"""Exact paid-standard model tables; no inferred embedding price commitment."""
from datetime import datetime,timezone
from types import SimpleNamespace
import pytest
from src.zeitgeist.gemini_tariff import EMBEDDING_2_SECTION, HYDROGEN_SECTION, MODELS, normalize_gemini_tariff, observe_gemini_tariff
from src.zeitgeist.spend_guard import SpendBlocked

NOW=datetime(2026,10,30,19,tzinfo=timezone.utc)


def table(rows,header='Paid Tier, per 1M tokens in USD'):
    return '<table class="pricing-table"><thead><tr><th></th><th>Free Tier</th><th>'+header+'</th></tr></thead><tbody>'+''.join(
        '<tr><td>'+label+'</td><td>Free of charge</td><td>'+value+'</td></tr>' for label,value in rows)+'</tbody></table>'


HYDROGEN_ROWS=[('Input price','\u00241.50'),('Output price (including thinking tokens)','\u00247.50'),
    ('Context caching price','\u00240.075 Implicit caching only (explicit caching is not supported).'),
    ('Grounding with Google Search<sup>*</sup>','5,000 free search requests per month (shared across all Gemini 3 and newer models), then \u002414 per 1,000 requests (subject to a daily cap of 1,500 requests per day).'),
    ('Grounding with Google Maps','5,000 requests per month (free, shared across all Gemini 3 and newer models), then \u002414 / 1,000 search queries.'),
    ('Used to improve our products','No')]


def hydrogen_section(rows=None,code='gemini-4-hydrogen'):
    return ('<h2 id="'+HYDROGEN_SECTION+'">Gemini 4 Hydrogen</h2><em><a href="/gemini-api/docs/models/gemini-4-hydrogen"><code>'+code+'</code></a></em>'+
        '<section><h3>Standard</h3>'+table(HYDROGEN_ROWS if rows is None else rows)+'</section><section><h3>Batch</h3>'+table([('Input price','\u00240.75')])+'</section>')


EMBEDDING_2_ROWS=[('Text input price','\u00240.20'),('Image input price','\u00240.45 (\u00240.00012 per image)'),
    ('Audio input price','\u00246.50 (\u00240.00016 per second)'),('Video input price','\u002412.00 (\u00240.00079 per frame)'),
    ('Used to improve our products','No')]


def embedding_2_section(rows=None,code='gemini-embedding-2'):
    return ('<h2 id="'+EMBEDDING_2_SECTION+'">Gemini Embedding 2</h2><em><a href="/gemini-api/docs/models/gemini-embedding-2"><code>'+code+'</code></a></em>'+
        '<section><h3>Standard</h3>'+table(EMBEDDING_2_ROWS if rows is None else rows)+'</section><section><h3>Batch</h3>'+table([('Text input price','\u00240.10')])+'</section>')


def fixture(*,hydrogen=None,embedding_2=None,embedding_price='\u00240.15',generation_input='\u00240.75 through December 31, 2026.<br>\u00241.50 starting January 1, 2027.',extra_row=None):
    generation=[('Input price',generation_input),('Output price (including thinking tokens)','\u00243.75 through December 31, 2026.<br>\u00247.50 starting January 1, 2027.'),
        ('Context caching price','\u00240.075 through December 31, 2026.'),
        ('Grounding with Google Search<sup>*</sup>','5,000 free search requests, then \u002414 per 1,000 requests.'),
        ('Grounding with Google Maps','5,000 free requests, then \u002414 per 1,000 queries.'),('Used to improve our products','No')]
    embedding=[('Input price',embedding_price),('Used to improve our products','No')]
    if extra_row:embedding.append(extra_row)
    # The same semantic heading/code/standard table structure as public HTML.
    return ('<h2 id="gemini-3.8-flash">Gemini 3.8 Flash</h2><em><a href="/gemini-api/docs/models/gemini-3.8-flash"><code>gemini-3.8-flash</code></a></em>'+
        '<section><h3>Standard</h3>'+table(generation)+'</section><section><h3>Batch</h3>'+table([('Input price','\u00240.375')])+'</section>'+(hydrogen_section() if hydrogen is None else hydrogen)+
        (embedding_2_section() if embedding_2 is None else embedding_2)+
        '<h2 id="gemini-embedding">Gemini Embedding</h2><em><a href="/gemini-api/docs/models/gemini-embedding-001"><code>gemini-embedding-001</code></a></em>'+
        '<section><h3>Standard</h3>'+table(embedding)+'</section><section><h3>Batch</h3>'+table([('Input price','\u00240.075')])+'</section>')


def test_exact_standard_paid_tariffs_and_only_generation_has_published_deadline():
    result=normalize_gemini_tariff(fixture())
    assert set(result['operations'])=={'generate_content:gemini-3.8-flash','generate_content:gemini-4-hydrogen',
        'embed_content:gemini-embedding-2','embed_content:gemini-embedding-001'}
    generation=result['operations']['generate_content:gemini-3.8-flash']
    embedding=result['operations']['embed_content:gemini-embedding-001']
    assert generation['input_token_usd_per_million']=='0.75'
    assert generation['output_token_usd_per_million']=='3.75'
    assert generation['output_includes_thinking'] is True
    assert generation['published_current_rate_through_date']=='2026-12-31'
    assert embedding['input_token_usd_per_million']=='0.15'
    assert embedding['billing_basis']=='input_tokens' and embedding['input_modalities']==['text']
    assert not any('valid' in key or 'date' in key for key in embedding)
    assert result['currency']=='USD' and result['service_tier']=='paid_standard'
    assert result==normalize_gemini_tariff('<header>New campaign and build</header>'+fixture()+'<footer>Different copyright</footer>')


@pytest.mark.parametrize('body',[
    lambda:fixture().replace('gemini-embedding-001','embedding-001'),
    lambda:fixture().replace('id="gemini-embedding"','id="different-model"'),
    lambda:fixture(embedding_price='\u00240.15 per character'),
    lambda:fixture(embedding_price='\u00240.15 + \u00240.01 service fee'),
    lambda:fixture(embedding_price='Free of charge'),
    lambda:fixture(extra_row=('Mandatory request fee','\u00240.01')),
    lambda:fixture().replace('Paid Tier, per 1M tokens in USD','Paid Tier, per 1M tokens in EUR'),
    lambda:fixture(generation_input='\u00240.75 through November 30, 2026.'),
    lambda:fixture().replace('Output price (including thinking tokens)','Output price'),
])
def test_wrong_identity_ambiguous_rows_units_currency_or_surcharge_reject(body):
    with pytest.raises(SpendBlocked):normalize_gemini_tariff(body())


def test_changed_plain_rate_changes_normalized_digest_instead_of_renewing_old_policy():
    first=observe_gemini_tariff(Client(fixture()),now=NOW)
    changed=observe_gemini_tariff(Client(fixture(embedding_price='\u00240.16')),now=NOW)
    assert changed['sha256']!=first['sha256']
    assert changed['contract']['operations']['embed_content:gemini-embedding-001']['input_token_usd_per_million']=='0.16'


def test_duplicate_exact_model_or_standard_table_rejects_ambiguity():
    for body in (fixture()+fixture(),fixture().replace('<h3>Standard</h3>','<h3>Standard</h3>'+table([('Input price','\u00240.01')]),1)):
        with pytest.raises(SpendBlocked):normalize_gemini_tariff(body)


class Client:
    def __init__(self,body,*,status=200,url='https://ai.google.dev/gemini-api/docs/pricing',kind='text/html'):
        self.headers={};self.calls=[]
        self.response=SimpleNamespace(status_code=status,url=url,headers={'content-type':kind},content=body.encode())
    def get(self,url,**kwargs):self.calls.append((url,kwargs));return self.response


def test_observer_one_anonymous_read_and_fresh_normalized_contract_identity():
    client=Client(fixture());result=observe_gemini_tariff(client,now=NOW)
    assert client.calls==[('https://ai.google.dev/gemini-api/docs/pricing',{'follow_redirects':False,'timeout':15})]
    assert result['observed_at']==NOW.isoformat() and result['source_url']==client.response.url
    assert len(result['sha256'])==64 and result['sha256']!=result['source_body_sha256']
    assert 'valid_until' not in result


@pytest.mark.parametrize('kwargs',[{'status':302},{'url':'https://untrusted.invalid/pricing'},{'kind':'application/json'}])
def test_redirect_wrong_origin_or_nonhtml_response_rejected(kwargs):
    with pytest.raises(SpendBlocked):observe_gemini_tariff(Client(fixture(),**kwargs),now=NOW)


def test_authenticated_client_and_oversize_body_reject_without_price_inference():
    client=Client(fixture());client.headers['Authorization']='must-not-send'
    with pytest.raises(SpendBlocked):observe_gemini_tariff(client,now=NOW)
    assert not client.calls
    with pytest.raises(SpendBlocked):normalize_gemini_tariff('X'*2_000_001)


def test_unapproved_input_price_marker_and_scoped_surcharge_rejects_review_reproduction():
    body=fixture().replace('Input price','Input price<sup>*</sup>',1)
    body=body.replace('<section><h3>Batch</h3>', '<p>* Mandatory additional fee: '+chr(36)+'0.01 per generation request.</p><section><h3>Batch</h3>',1)
    with pytest.raises(SpendBlocked):normalize_gemini_tariff(body)


@pytest.mark.parametrize('location',['intro','standard_caveat','embedding_intro','paid_cache_condition'])
def test_scoped_price_or_eligibility_conditions_are_bound_into_contract(location):
    original=observe_gemini_tariff(Client(fixture()),now=NOW)
    body=fixture()
    if location=='intro':body=body.replace('<section><h3>Standard</h3>','<p>Standard pricing requires a separate subscription.</p><section><h3>Standard</h3>',1)
    if location=='standard_caveat':body=body.replace('<section><h3>Batch</h3>','<p>Every Standard request incurs an additional service charge.</p><section><h3>Batch</h3>',1)
    if location=='embedding_intro':body=body.replace('<h2 id="gemini-embedding">Gemini Embedding</h2>','<h2 id="gemini-embedding">Gemini Embedding</h2><p>Embedding rate applies only with a paid plan commitment.</p>')
    if location=='paid_cache_condition':body=body.replace('0.075 through December 31','0.076 through December 31',1)
    changed=observe_gemini_tariff(Client(body),now=NOW)
    assert changed['sha256']!=original['sha256']


def test_model_scoped_common_caveat_after_other_tier_is_bound_but_other_tier_prices_are_not():
    original=observe_gemini_tariff(Client(fixture()),now=NOW)
    body=fixture().replace('<h2 id="gemini-embedding-2">', '<p>All generation tiers require an additional platform fee.</p><h2 id="gemini-embedding-2">',1)
    assert observe_gemini_tariff(Client(body),now=NOW)['sha256']!=original['sha256']
    other_tier=fixture().replace('0.375','0.400',1)
    assert observe_gemini_tariff(Client(other_tier),now=NOW)['sha256']==original['sha256']


def markdoc():
    from pathlib import Path
    return (Path(__file__).parent/'fixtures/gemini-pricing-20261122.txt').read_text()


def markdoc_20261113():
    from pathlib import Path
    return (Path(__file__).parent/'fixtures/gemini-pricing-markdoc.txt').read_text()


def test_observed_official_markdoc_retains_exact_standard_rates():
    contract=normalize_gemini_tariff(markdoc())
    assert contract['operations']==normalize_gemini_tariff(fixture())['operations']
    assert 'Discounts and model availability' in contract['standard_conditions']['gemini-4-hydrogen']['scoped_model_caveats']
    # The 3.8 section of the 2026-11-13 page carried this caveat; the newer page moved it to Hydrogen.
    assert 'Discounts and model availability' in normalize_gemini_tariff(markdoc_20261113())['standard_conditions']['gemini-3.8-flash']['scoped_model_caveats']


def _from(page,anchor,change):
    head,sep,tail=page.partition(anchor)
    assert sep
    return head+sep+change(tail)


@pytest.mark.parametrize('change', [
    lambda s:s.replace('per 1M tokens in USD','per 1K characters in USD'),
    lambda s:s.replace('* \u00240.15\n','* Free\n'),
    lambda s:s.replace('Output price (including thinking tokens)','Output price'),
    lambda s:_from(s,'## Gemini 3.8 Flash',lambda t:t.replace('{% /table %}','',1)),
    lambda s:_from(s,'## Gemini 3.8 Flash',lambda t:t.replace('* Input price','* Unexpected fee',1)),
    lambda s:_from(s,'## Gemini 4 Hydrogen',lambda t:t.replace('{% /table %}','',1)),
    lambda s:_from(s,'## Gemini 4 Hydrogen',lambda t:t.replace('* Input price','* Unexpected fee',1)),
    lambda s:_from(s,'## Gemini 4 Hydrogen',lambda t:t.replace('* \u00241.50\n','* \u00241.50, prompts <= 200k tokens\n',1)),
    lambda s:_from(s,'## Gemini 4 Hydrogen',lambda t:t.replace('{% sup %}*{% /sup %}','{% sup %}**{% /sup %}',1)),
    lambda s:_from(s,'## Gemini 4 Hydrogen',lambda t:t.replace('* Free of charge','* Free of charge{% sup %}*{% /sup %}',1)),
    lambda s:_from(s,'## Gemini 4 Hydrogen',lambda t:t.replace('* [No](/docs/terms)','* [No](/docs/terms){% sup %}*{% /sup %}',1)),
    lambda s:_from(s,'## Gemini 4 Hydrogen',lambda t:t.replace('* Context caching price','* Context caching price{% sup %}*{% /sup %}',1)),
    lambda s:_from(s,'## Gemini 3.8 Flash',lambda t:t.replace('* Used to improve our products','* Used to improve our products{% sup %}*{% /sup %}',1)),
    lambda s:s.replace('gemini-embedding-001','wrong-model'),
    lambda s:s+ s,
])
def test_markdoc_unknown_missing_malformed_never_infers_free(change):
    with pytest.raises(SpendBlocked): normalize_gemini_tariff(change(markdoc()))


@pytest.mark.parametrize('kind', ['text/plain; charset=utf-8', 'text/markdown; charset=utf-8'])
def test_only_exact_official_permanent_redirect_is_allowed(kind):
    new='https://aistudio.google.com/docs/pricing'
    client=Client(markdoc(),url=new,kind=kind)
    final=client.response
    redirect=SimpleNamespace(status_code=301,url='https://ai.google.dev/gemini-api/docs/pricing',headers={'location':new})
    responses=iter([redirect,final])
    client.get=lambda *a,**k:next(responses)
    result=observe_gemini_tariff(client,now=NOW)
    assert result['resolved_source_url']==new
    assert result['source_url']=='https://ai.google.dev/gemini-api/docs/pricing'
    assert result['parser_version']=='gemini-tariff-v4'
    for location in ('https://evil.invalid/pricing',new+'?x=1'):
        client=Client('');client.response.status_code=301;client.response.headers['location']=location
        with pytest.raises(SpendBlocked):observe_gemini_tariff(client,now=NOW)


@pytest.mark.parametrize('kind', ['application/json', 'text/html', 'application/octet-stream'])
def test_markdown_redirect_does_not_accept_other_representations(kind):
    new='https://aistudio.google.com/docs/pricing'
    client=Client(markdoc(),url=new,kind=kind)
    responses=iter([SimpleNamespace(status_code=301,url='https://ai.google.dev/gemini-api/docs/pricing',headers={'location':new}),client.response])
    client.get=lambda *a,**k:next(responses)
    with pytest.raises(SpendBlocked):observe_gemini_tariff(client,now=NOW)


def test_redirect_chain_and_malformed_utf8_reject():
    new='https://aistudio.google.com/docs/pricing'
    client=Client('');client.response.status_code=301;client.response.headers['location']=new
    with pytest.raises(SpendBlocked):observe_gemini_tariff(client,now=NOW)
    client=Client('');client.response.content=b'\xff'
    with pytest.raises(SpendBlocked):observe_gemini_tariff(client,now=NOW)


def test_transient_public_read_has_bounded_backoff(monkeypatch):
    from src.zeitgeist import gemini_tariff as tariff
    delays=[];monkeypatch.setattr(tariff.time,'sleep',delays.append)
    client=Client(fixture(),status=503)
    with pytest.raises(SpendBlocked):observe_gemini_tariff(client,now=NOW)
    assert len(client.calls)==3 and len(delays)==2
    assert .5<=delays[0]<=.625 and 1<=delays[1]<=1.25



def test_hydrogen_standard_rates():
    op=normalize_gemini_tariff(fixture())['operations']['generate_content:gemini-4-hydrogen']
    assert MODELS[HYDROGEN_SECTION]=='gemini-4-hydrogen'
    assert op['model']=='gemini-4-hydrogen'
    assert op['input_token_usd_per_million']=='1.5' and op['output_token_usd_per_million']=='7.5'
    assert op['output_includes_thinking'] is True
    assert op['published_rate_schedule']=='current_only'
    assert not any(key.startswith(('published_current','published_future','future_')) for key in op)
    assert normalize_gemini_tariff(markdoc())['operations']['generate_content:gemini-4-hydrogen']==op
    assert 'gemini-4-hydrogen' in normalize_gemini_tariff(fixture())['standard_conditions']


def test_page_without_hydrogen_still_parses_the_old_operations():
    contract=normalize_gemini_tariff(markdoc_20261113())
    assert set(contract['operations'])=={'generate_content:gemini-3.8-flash','embed_content:gemini-embedding-001'}
    assert 'gemini-4-hydrogen' not in contract['standard_conditions']
    assert set(normalize_gemini_tariff(fixture(hydrogen='',embedding_2=''))['operations'])==set(contract['operations'])


def test_flash_never_carries_current_only():
    assert 'published_rate_schedule' not in normalize_gemini_tariff(fixture())['operations']['generate_content:gemini-3.8-flash']


def _hydrogen_rows(**changes):
    return [(label,changes.get(label,value)) for label,value in HYDROGEN_ROWS]


@pytest.mark.parametrize('body',[
    lambda:fixture(hydrogen=hydrogen_section(_hydrogen_rows(**{'Input price':'\u00241.50, prompts <= 200k tokens'}))),
    lambda:fixture(hydrogen=hydrogen_section(HYDROGEN_ROWS+[('Mandatory request fee','\u00240.01')])),
    lambda:fixture(hydrogen=hydrogen_section(code='gemini-4-hydrogen-preview')),
    lambda:fixture(hydrogen=hydrogen_section()*2),
    lambda:fixture(hydrogen=hydrogen_section(_hydrogen_rows(**{'Input price':'Free of charge'}))),
    lambda:fixture(hydrogen=hydrogen_section([('Output price' if label.startswith('Output price') else label,value) for label,value in HYDROGEN_ROWS])),
])
def test_unreviewed_hydrogen_forms_reject(body):
    with pytest.raises(SpendBlocked):normalize_gemini_tariff(body())


def test_embedding_2_text_rate():
    assert EMBEDDING_2_SECTION=='gemini-embedding-2' and MODELS[EMBEDDING_2_SECTION]=='gemini-embedding-2'
    contract=normalize_gemini_tariff(fixture())
    assert contract['operations']['embed_content:gemini-embedding-2']=={'model':'gemini-embedding-2',
        'input_token_usd_per_million':'0.2','billing_basis':'input_tokens','input_modalities':['text'],'output_type':'embedding'}
    assert 'gemini-embedding-2' in contract['standard_conditions']
    assert normalize_gemini_tariff(markdoc())['operations']['embed_content:gemini-embedding-2']==contract['operations']['embed_content:gemini-embedding-2']


def _embedding_2_rows(**changes):
    return [(label,changes.get(label,value)) for label,value in EMBEDDING_2_ROWS]


@pytest.mark.parametrize('body',[
    lambda:fixture(embedding_2=embedding_2_section(_embedding_2_rows(**{'Text input price':'\u00240.20 per character'}))),
    lambda:fixture(embedding_2=embedding_2_section(EMBEDDING_2_ROWS+[('Mandatory request fee','\u00240.01')])),
    lambda:fixture(embedding_2=embedding_2_section(code='gemini-embedding-2-preview')),
])
def test_unreviewed_embedding_2_forms_reject(body):
    with pytest.raises(SpendBlocked):normalize_gemini_tariff(body())
