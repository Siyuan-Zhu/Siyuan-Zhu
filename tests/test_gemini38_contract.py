"""Stable 3.8 migration keeps the verified cash boundary and real model wiring."""
import copy
from datetime import timedelta
from decimal import Decimal
from io import BytesIO
import json
from types import SimpleNamespace
import pytest
from PIL import Image
from google.genai import types
from src.zeitgeist.spend_guard import BudgetGuard, MemoryLedger, SpendBlocked
from tests.test_spend_provisioning import gemini_metadata, NOW

MODEL = 'gemini-3.8-flash'
OP = 'generate_content:' + MODEL


def limits():
    return {'pricing_reference':'https://ai.google.dev/gemini-api/docs/pricing',
      'count_tokens_reference':'https://ai.google.dev/gemini-api/docs/billing#is-gettokens-billed',
      'pricing_valid_until':'2026-12-31T23:59:59Z',
      'max_input_bytes':2000000,'max_input_tokens':65536,'max_output_tokens':1500,
      'input_token_usd_per_million':'0.75','output_token_usd_per_million':'3.75',
      'thinking_level':'low','max_input_images':9,'image_tokens_per_image':1120,
      'media_resolution':'MEDIA_RESOLUTION_HIGH'}


def guard(now=NOW):
    from scripts.prepare_spend_ledger import prepare_candidate
    metadata=gemini_metadata()
    document=prepare_candidate(metadata,now=NOW)
    account=document['providers']['gemini']; account['verified_at']=now.isoformat()
    document['month']=now.strftime('%Y-%m'); document['budget_window_start']=now.replace(day=1,hour=0,minute=0,second=0).isoformat()
    account['request_limits'][OP]=limits(); account['operation_caps_usd'][OP]='0.054777'
    return BudgetGuard(MemoryLedger(document),lambda:now,credential_fingerprints={'gemini':'a'*16},require_credential_binding=True)


def request(contents='A synthetic editorial commission',config=None):
    return {'model':MODEL,'contents':contents,'config':config or {'response_mime_type':'application/json'}}


def thumbnail():
    stream=BytesIO();Image.new('RGB',(32,32),(30,60,90)).save(stream,format='JPEG')
    return types.Part.from_bytes(data=stream.getvalue(),mime_type='image/jpeg')


def test_38_contract_omits_unsupported_sampling_and_applies_low_thinking_and_high_media():
    bounded=guard().bounded_genai_request(OP,request())
    assert bounded['config']['thinking_config']=={'thinking_level':'low'}
    assert bounded['config']['max_output_tokens']==1500
    assert bounded['config']['media_resolution']=='MEDIA_RESOLUTION_HIGH'
    assert 'candidate_count' not in bounded['config'] and 'temperature' not in bounded['config']
    types.GenerateContentConfig(**bounded['config'])


def test_nine_small_images_use_1120_floor_and_supported_default_media_count_config():
    budget=guard(); bounded=budget.bounded_genai_request(OP,request(['Synthetic batch']+[thumbnail() for _ in range(9)]))
    calls=[]
    def count(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(total_tokens=1 if len(calls)==1 else 11)
    maximum,_=budget.counted_genai_cap(OP,bounded,count)
    assert maximum==(Decimal(11+9*1120)*Decimal('.75')+Decimal(1500)*Decimal('3.75'))/1000000
    from google.genai import models
    assert all('generation_config' not in call['config'] for call in calls)
    assert all(models._CountTokensConfig_to_mldev(call['config']) == {} for call in calls)
    assert budget.snapshot()['reservation_count']==0


@pytest.mark.parametrize('mutation',[
    lambda policy:policy.update(thinking_level='minimal'),
    lambda policy:policy.update(pricing_valid_until='2027-02-01T00:00:00Z'),
    lambda policy:policy.update(input_token_usd_per_million='.01'),
    lambda policy:policy.update(max_output_tokens=1501),
    lambda policy:policy.update(max_input_images=10),
])
def test_38_rejects_unverified_or_unsupported_policy(mutation):
    budget=guard();mutation(budget.ledger.document['providers']['gemini']['request_limits'][OP])
    with pytest.raises(SpendBlocked):budget.bounded_genai_request(OP,request())


def test_38_pricing_must_remain_valid_for_full_request_duration():
    now=NOW.replace(month=12,day=31,hour=23,minute=58,second=0)
    with pytest.raises(SpendBlocked,match='pricing'):
        guard(now).bounded_genai_request(OP,request())


@pytest.mark.parametrize('payload',[
    request(config={'http_options':{'timeout':1}}),
    request(contents=[{'file_data':{'file_uri':'https://example.test/unbounded'}}]),
    request(contents=[{'executable_code':{'code':'unbounded'}}]),
    request(config={'thinking_config':{'thinking_level':'minimal'}}),
    request(config={'candidate_count':2}),
    request(config={'temperature':.1}),
])
def test_38_unpriced_request_controls_and_parts_are_rejected(payload):
    budget=guard()
    with pytest.raises(SpendBlocked):
        bounded=budget.bounded_genai_request(OP,payload)
        budget.counted_genai_cap(OP,bounded,lambda **kw:SimpleNamespace(total_tokens=1))
    assert budget.snapshot()['reservation_count']==0


def test_no_38_policy_remains_blocked_without_active_authorization():
    budget=guard();del budget.ledger.document['providers']['gemini']['request_limits'][OP]
    with pytest.raises(SpendBlocked):budget.bounded_genai_request(OP,request())


def test_probe_proposal_is_exact_pure_contract_not_a_ledger_mutation():
    from scripts.probe_director_admission import proposed_gemini38_policy
    policy=proposed_gemini38_policy()
    assert policy=={'operation':OP,'maximum_gross_usd':'0.054777','request_limits':limits()}
    policy['request_limits']['max_input_tokens']=1
    assert proposed_gemini38_policy()['request_limits']['max_input_tokens']==65536


def test_actual_developer_api_serializer_rejects_vertex_only_count_generation_config():
    from google.genai import models
    # Types accept this field, but the actual Developer API serializer does not.
    config=types.CountTokensConfig(generation_config={'media_resolution':'MEDIA_RESOLUTION_HIGH'})
    with pytest.raises(ValueError,match='generation_config'):
        models._CountTokensConfig_to_mldev(config)
    assert models._CountTokensConfig_to_mldev(types.CountTokensConfig(http_options={'timeout':300000}))=={}


def test_actual_single_and_batch_critic_requests_use_selected_38_model_without_sampling(monkeypatch):
    from src.zeitgeist import visual_router as vr
    calls=[]
    scores={'semantic_alignment':.9,'compositional_clarity':.9,'editorial_memorability':.9,'safety_pass':True}
    def generate(**kwargs):
        calls.append(kwargs)
        is_batch=any(isinstance(part,str) and 'Candidate index' in part for part in kwargs['contents'])
        data={'candidates':[dict(scores,index=0)]} if is_batch else scores
        return SimpleNamespace(text=json.dumps(data),candidates=[])
    monkeypatch.setenv('ZEITGEIST_LIVE_LLM','1')
    monkeypatch.setenv('GEMINI_API_KEY','synthetic-key')
    monkeypatch.setattr('google.genai.Client',lambda **kw:SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr,'guard_genai_client',lambda client,**kw:client)
    raw=thumbnail().inline_data.data
    assert vr._score_candidate_with_gemini_vision(raw,'ANCHOR','Rooted craft','Physical stability',model_name=MODEL)['safety_pass']
    assert vr._score_photo_batch([{'thumb_bytes':raw}],word='ANCHOR',gloss='Rooted craft',thesis='Physical stability',model_name=MODEL)[0]['safety_pass']
    assert len(calls)==2 and all(call['model']==MODEL for call in calls)
    assert all('temperature' not in call['config'] and 'candidate_count' not in call['config'] for call in calls)


def test_real_orchestrator_passes_configured_director_and_judge_models(monkeypatch,tmp_path):
    from src.zeitgeist import adk_pipeline as pipeline
    from src.state_store import InMemoryStateStore
    (tmp_path/'config').mkdir()
    (tmp_path/'config/orbit_sources.json').write_text(json.dumps({'budget_guardrails':{
        'primary_text_model':MODEL,'multimodal_judge_model':'gemini-2.5-flash'},'active_interjection':{'enabled':False}}))
    observed={}; original=pipeline.commission_daily_zeitgeist
    def commission(*args,**kwargs):
        observed['director']=kwargs['model_name']
        return original(*args,**kwargs)
    class ReachedRenderer(Exception):pass
    def render(**kwargs):
        observed['judge']=kwargs['judge_model']
        raise ReachedRenderer()
    monkeypatch.setattr(pipeline,'commission_daily_zeitgeist',commission)
    monkeypatch.setattr(pipeline,'select_and_render_pulse_assets',render)
    with pytest.raises(ReachedRenderer):
        pipeline.ZeitgeistADKOrchestrator(tmp_path,InMemoryStateStore()).run('2026-10-30',offline_demo=True)
    assert observed=={'director':MODEL,'judge':'gemini-2.5-flash'}


def test_proposed_text_and_nine_image_probe_preserve_all_active_ledger_state(monkeypatch):
    from scripts import probe_director_admission as probe
    from tests.test_director_admission_probe import document,KEY,Models,forbid_mutation
    forbid_mutation(monkeypatch)
    active=document();active['reservations']={'held':{'provider':'gemini','operation':'old','max_gross_usd':'.01','status':'reserved'}}
    original=copy.deepcopy(active)
    report=probe.run_probe({'GEMINI_API_KEY':KEY},now=NOW,
        ledger_loader=lambda env,now:active,client_factory=lambda key:SimpleNamespace(models=Models()))
    proposed=report['proposed_policy_results']
    assert len(proposed)==2 and {result['request_shape'] for result in proposed}=={'text','nine_thumbnails'}
    assert all(result['admitted'] and result['model']==MODEL for result in proposed)
    assert all(result['policy_source']=='proposed_memory_only_not_active_authority' for result in proposed)
    assert all(result['active_policy_present'] is False and result['active_ledger_changed'] is False for result in proposed)
    assert report['inference_eligibility_verified'] is False and report['reservation_created'] is False
    assert active==original and OP not in active['providers']['gemini']['request_limits']


def test_initial_candidate_can_preserve_explicit_38_policy_for_monthly_handoff():
    from scripts.prepare_spend_ledger import prepare_candidate
    metadata=gemini_metadata();account=metadata['providers']['gemini']
    account['operation_caps_usd'][OP]='0.054777'
    account['request_limits'][OP]=limits()
    account['price_evidence'][OP]={'source_url':'https://ai.google.dev/gemini-api/docs/pricing',
        'verified_at':NOW.isoformat(),'reference':'verified-38-policy','maximum_gross_usd':'0.054777'}
    candidate=prepare_candidate(metadata,now=NOW)
    assert candidate['providers']['gemini']['request_limits'][OP]['pricing_valid_until']=='2026-12-31T23:59:59Z'
    assert candidate['providers']['gemini']['credit_baseline_usd']=='0'
