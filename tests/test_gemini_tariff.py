"""Exact paid-standard model tables; no inferred embedding price commitment."""
from datetime import datetime,timezone
from types import SimpleNamespace
import pytest
from src.zeitgeist.gemini_tariff import normalize_gemini_tariff, observe_gemini_tariff
from src.zeitgeist.spend_guard import SpendBlocked

NOW=datetime(2026,10,30,19,tzinfo=timezone.utc)


def table(rows,header='Paid Tier, per 1M tokens in USD'):
    return '<table class="pricing-table"><thead><tr><th></th><th>Free Tier</th><th>'+header+'</th></tr></thead><tbody>'+''.join(
        '<tr><td>'+label+'</td><td>Free of charge</td><td>'+value+'</td></tr>' for label,value in rows)+'</tbody></table>'


def fixture(*,embedding_price='\u00240.15',generation_input='\u00240.75 through December 31, 2026.<br>\u00241.50 starting January 1, 2027.',extra_row=None):
    generation=[('Input price',generation_input),('Output price (including thinking tokens)','\u00243.75 through December 31, 2026.<br>\u00247.50 starting January 1, 2027.'),
        ('Context caching price','\u00240.075 through December 31, 2026.'),
        ('Grounding with Google Search<sup>*</sup>','5,000 free search requests, then \u002414 per 1,000 requests.'),
        ('Grounding with Google Maps','5,000 free requests, then \u002414 per 1,000 queries.'),('Used to improve our products','No')]
    embedding=[('Input price',embedding_price),('Used to improve our products','No')]
    if extra_row:embedding.append(extra_row)
    # The same semantic heading/code/standard table structure as public HTML.
    return ('<h2 id="gemini-3.8-flash">Gemini 3.8 Flash</h2><em><a href="/gemini-api/docs/models/gemini-3.8-flash"><code>gemini-3.8-flash</code></a></em>'+
        '<section><h3>Standard</h3>'+table(generation)+'</section><section><h3>Batch</h3>'+table([('Input price','\u00240.375')])+'</section>'+
        '<h2 id="gemini-embedding-2">Gemini Embedding 2</h2><code>gemini-embedding-2</code><h3>Standard</h3>'+table([('Input price','\u00240.20')])+
        '<h2 id="gemini-embedding">Gemini Embedding</h2><em><a href="/gemini-api/docs/models/gemini-embedding-001"><code>gemini-embedding-001</code></a></em>'+
        '<section><h3>Standard</h3>'+table(embedding)+'</section><section><h3>Batch</h3>'+table([('Input price','\u00240.075')])+'</section>')


def test_exact_standard_paid_tariffs_and_only_generation_has_published_deadline():
    result=normalize_gemini_tariff(fixture())
    assert set(result['operations'])=={'generate_content:gemini-3.8-flash','embed_content:gemini-embedding-001'}
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
