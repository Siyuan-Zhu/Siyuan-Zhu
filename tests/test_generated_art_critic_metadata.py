"""Generated artwork retains its existing final critic verdict without extra calls."""
import pytest
from src.zeitgeist import visual_router as router
from tests.test_visual_critic_funnel import _make_sample_commission, _make_test_photo_bytes


def test_generated_art_preserves_numeric_final_verdict_without_another_judgment(tmp_path,monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY','synthetic')
    verdict=dict(semantic_alignment=.82,compositional_clarity=.90,editorial_memorability=.75,safety_pass=True)
    calls=[]
    def judge(**kwargs):
        calls.append(True)
        return verdict
    monkeypatch.setattr(router,'_score_candidate_with_gemini_vision',judge)
    pulse=router.select_and_render_pulse_assets(commission=_make_sample_commission(),
        medium_history_30d=[],repo_root=tmp_path,date_str='2026-10-30',candidates=[],
        flux_generator=lambda prompt:_make_test_photo_bytes())
    assert pulse.visual_medium=='FLUX_PRO' and len(calls)==4
    assert pulse.semantic_alignment_score==.82
    assert pulse.composite_critic_score==pytest.approx(.83)
    assert pulse.to_feed_entry()['critic_scores']==dict(semantic_alignment=.82,composite=.83)
