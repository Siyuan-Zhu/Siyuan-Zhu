"""Public structured pricing is evidence; calculator defaults and UI hashes are not."""
import copy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest
from src.zeitgeist.spend_guard import SpendBlocked
from src.zeitgeist.bfl_tariff import normalize_bfl_flux3_tariff, observe_bfl_flux3_tariff

NOW = datetime(2026,10,30,17,tzinfo=timezone.utc)
NOTES = [
    'Pricing is per image generated. Charges are based on the resolution tier selected in the request, regardless of the resolution at which the image is processed internally, and apply equally to text-to-image and image-to-image generations.',
    'Reference images, whether provided by the user or retrieved by the model, and prompts are included at no extra charge. The per-image rate does not vary with the number of reference images provided, up to the maximum set out in the documentation.',
    'The resolution tier is selected in the request; all supported aspect ratios within a tier are charged the same.',
]


def fixture():
    spec={'@type':'UnitPriceSpecification','name':'1.5K image','priceCurrency':'USD','price':.07,
          'unitText':'image','referenceQuantity':{'@type':'QuantitativeValue','value':1,'unitText':'image'}}
    product={'@type':'Product','name':'FLUX 3 Image','brand':{'@type':'Brand','name':'Black Forest Labs'},
        'offers':{'@type':'Offer','priceCurrency':'USD','price':.041,'priceSpecification':[spec],
            'url':'https://bfl.ai/pricing','seller':{'name':'Black Forest Labs','url':'https://bfl.ai'}}}
    card={'apiEndpoint':'/v1/flux-3-image','category':{'name':'FLUX 3 Image','footnotes':NOTES.copy()},
        'title':[{'children':[{'text':'FLUX 3 Image'}]}], 'footnotes':None,
        'imageTierRates':{'_type':'imageTierRates','tier1_5k':7,'tier1k':4.8},
        **{key:None for key in ('firstMpPrice','inputMpPrice','mpPrice','mpSecondPrice','priceUnit','secondPrice','videoRates')}}
    return product,card


def html(product=None, card=None, *, unrelated=''):
    p,c=fixture(); product=product if product is not None else p; card=card if card is not None else c
    # This is the real public Next.js flight record format, parsed as JSON only.
    flight='53:'+json.dumps([chr(36), chr(36)+'L1', None, {'pricingCards':[card]}])+'\n'
    return ('<html>'+unrelated+'<script type="application/ld+json">'+json.dumps({'@context':'https://schema.org','@graph':[product]})+
        '</script><script>self.__next_f.push('+json.dumps([1,flight])+')</script></html>')


def test_exact_flux3_tier_uses_both_structured_sources_and_no_calculator_default():
    result=normalize_bfl_flux3_tariff(html(unrelated='<p>Rate:USD0.048 / image</p>'))
    assert result['operation']=='generate_image:flux-3-image'
    assert result['endpoint']=='/v1/flux-3-image'
    assert result['resolution']=='1.5k' and result['unit_price_usd']=='0.07'
    assert result['currency']=='USD' and result['unit']=='generated_image' and result['quantity']==1
    assert result['pricing_flags']['all_supported_aspect_ratios_same_price'] is True
    assert result['pricing_flags']['model_retrieved_reference_images_included'] is True
    assert 'valid_until' not in result
    assert result==normalize_bfl_flux3_tariff(html(unrelated='<p>New campaign, build hash, footer and navigation</p>'))


@pytest.mark.parametrize('change',['klein','endpoint','currency','unit','quantity','crosscheck','missing_terms','new_surcharge','extra_price','required_surcharge','offer_eligibility'])
def test_wrong_model_tier_unit_or_ambiguous_pricing_never_admits_a_tariff(change):
    product,card=fixture();spec=product['offers']['priceSpecification'][0]
    if change=='klein': product['name']='FLUX 3 Image Klein'
    if change=='endpoint': card['apiEndpoint']='/v1/flux-3-image-klein'
    if change=='currency': spec['priceCurrency']='EUR'
    if change=='unit': spec['unitText']='megapixel'
    if change=='quantity': spec['referenceQuantity']['value']=10
    if change=='crosscheck': card['imageTierRates']['tier1_5k']=8
    if change=='missing_terms': card['category']['footnotes']=[]
    if change=='new_surcharge': card['inputMpPrice']=1
    if change=='extra_price': spec['priceSpecification']={'price':.02}
    if change=='required_surcharge':
        charge=copy.deepcopy(spec);charge.update(name='Required grounding surcharge',price=.01)
        product['offers']['priceSpecification'].append(charge)
    if change=='offer_eligibility': product['offers']['eligibleCustomerType']='Special contracted customers only'
    with pytest.raises(SpendBlocked): normalize_bfl_flux3_tariff(html(product,card))


def test_real_tariff_or_pricing_conditions_change_the_normalized_digest():
    first=observe_bfl_flux3_tariff(Client(html()),now=NOW)
    p,c=fixture();p['offers']['priceSpecification'][0]['price']=.08;c['imageTierRates']['tier1_5k']=8
    assert observe_bfl_flux3_tariff(Client(html(p,c)),now=NOW)['sha256']!=first['sha256']
    p,c=fixture();c['category']['footnotes'].append('A new optional service has an additional charge.')
    assert observe_bfl_flux3_tariff(Client(html(p,c)),now=NOW)['sha256']!=first['sha256']


class Client:
    def __init__(self,body,status=200,url='https://bfl.ai/pricing',content_type='text/html'):
        self.response=SimpleNamespace(content=body.encode(),status_code=status,url=url,headers={'content-type':content_type})
        self.calls=[]
    def get(self,url,**kwargs): self.calls.append((url,kwargs));return self.response


def test_public_observer_is_unauthed_fixed_origin_and_returns_fresh_normalized_contract_only():
    client=Client(html())
    report=observe_bfl_flux3_tariff(client,now=NOW)
    assert client.calls==[('https://bfl.ai/pricing',{'follow_redirects':False,'timeout':15})]
    assert report['source_url']=='https://bfl.ai/pricing' and report['observed_at']==NOW.isoformat()
    assert len(report['sha256'])==len(report['source_body_sha256'])==64
    assert report['sha256']!=report['source_body_sha256']
    assert 'valid_until' not in report and 'policy_verified_at' not in report


@pytest.mark.parametrize('options',[{'status':302},{'url':'https://untrusted.invalid/pricing'},{'content_type':'application/javascript'}])
def test_redirect_wrong_content_and_untrusted_origin_are_rejected(options):
    with pytest.raises(SpendBlocked): observe_bfl_flux3_tariff(Client(html(),**options),now=NOW)


def test_duplicate_products_or_flight_cards_fail_closed():
    p,c=fixture();body=html(p,c)
    duplicate='<script type="application/ld+json">'+json.dumps(p)+'</script>'
    with pytest.raises(SpendBlocked): normalize_bfl_flux3_tariff(body+duplicate)
    flight='54:'+json.dumps({'pricingCards':[c]})+'\n'
    with pytest.raises(SpendBlocked): normalize_bfl_flux3_tariff(body+'<script>self.__next_f.push('+json.dumps([1,flight])+')</script>')
