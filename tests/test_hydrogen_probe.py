"""The Hydrogen probe prints versions and token counts only and refuses unsafe runs."""
from io import BytesIO
import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from google.genai import types
from PIL import Image

from scripts import probe_hydrogen as probe

ROOT = Path(__file__).resolve().parents[1]
KEY = "synthetic-key"
SECRET = "SECRET-RESPONSE-TEXT"


class FakeResponse:
    status_code = 200

    def __init__(self, body):
        self.text = json.dumps(body)


class FakeHttp:
    def __init__(self, models=("gemini-4-hydrogen", "gemini-embedding-2"), fail=None):
        self.models = models
        self.fail = fail
        self.requests = []

    def request(self, method, url, **kw):
        self.requests.append((method, url))
        if self.fail:
            raise self.fail
        return FakeResponse({"models": [
            {"name": "models/" + m, "supportedGenerationMethods": ["generateContent"]}
            for m in self.models
        ]})


def usage(prompt, image=None, total=20):
    details = None
    if image is not None:
        details = [types.ModalityTokenCount(modality=types.MediaModality.IMAGE, token_count=image),
                   types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=12)]
    return types.GenerateContentResponseUsageMetadata(
        prompt_token_count=prompt, candidates_token_count=3, thoughts_token_count=2,
        total_token_count=total, prompt_tokens_details=details)


class FakeModels:
    def __init__(self, usages, version="synthetic-hydrogen-version", fail_at=None,
                 text=SECRET, finish_reason=types.FinishReason.STOP, embed_values=None,
                 embed_fail=False, embed_tokens=7.0):
        self.embed_values = [[0.25] * 768, [0.5] * 768] if embed_values is None else embed_values
        self.embed_fail = embed_fail
        self.embed_tokens = embed_tokens
        self.embed_calls = []
        self.text = text
        self.finish_reason = finish_reason
        self.usages = list(usages)
        self.version = version
        self.fail_at = fail_at
        self.calls = []

    def generate_content(self, **kw):
        self.calls.append(kw)
        if self.fail_at == len(self.calls):
            raise RuntimeError("key=" + KEY)
        index = min(len(self.calls), len(self.usages)) - 1
        response = types.GenerateContentResponse(
            model_version=self.version, response_id="r",
            usage_metadata=self.usages[index],
            candidates=[types.Candidate(
                finish_reason=self.finish_reason,
                content=types.Content(parts=[types.Part(text=self.text)], role="model"))])
        return response

    def embed_content(self, **kw):
        self.embed_calls.append(kw)
        if self.embed_fail:
            raise RuntimeError("key=" + KEY)
        stats = (None if self.embed_tokens is None
                 else types.ContentEmbeddingStatistics(token_count=self.embed_tokens))
        return types.EmbedContentResponse(embeddings=[
            types.ContentEmbedding(values=values, statistics=stats) for values in self.embed_values])


class FakeGenai:
    def __init__(self, models):
        self.models = models


def default_models(**kw):
    return FakeModels([usage(12), usage(1112, image=1100)], **kw)


def run_main(argv=("--confirm-paid-probe",), http=None, models=None, env=None, capsys=None):
    http = http or FakeHttp()
    models = models or default_models()
    made = []

    def genai_factory(key):
        made.append(key)
        return FakeGenai(models)

    code = probe.main(list(argv), env={"GEMINI_API_KEY": KEY} if env is None else env,
                      http_factory=lambda: http, genai_factory=genai_factory)
    full = capsys.readouterr().out
    run_main.full = full
    return code, full.splitlines()[0] if full else full, models, http, made


def test_report_has_only_versions_and_counts(capsys):
    report = probe.run_probe(KEY, http_client=FakeHttp(), genai_client=FakeGenai(default_models()))
    assert set(report) == {"listed", "missing", "text", "image"}
    assert set(report["text"]) == {"model_version", "prompt_token_count", "candidates_token_count",
                                   "thoughts_token_count", "total_token_count", "seconds"}
    assert set(report["image"]) == set(report["text"]) | {"image_prompt_token_count"}
    code, out, *_ = run_main(capsys=capsys)
    assert code == 0
    for banned in (KEY, SECRET, "masked", "fingerprint"):
        assert banned not in run_main.full
    assert json.loads(out)["image"]["model_version"] == "synthetic-hydrogen-version"


def test_two_calls_second_has_384px_image(capsys):
    _, _, models, _, made = run_main(capsys=capsys)
    assert len(models.calls) == 7 and made == [KEY]  # two A8 calls, four thinking-fit calls, one critic call
    for call in models.calls[:2]:
        assert call["model"] == "gemini-4-hydrogen"
        assert call["config"] == {"max_output_tokens": 16,
                                  "thinking_config": {"thinking_level": "low"},
                                  "media_resolution": "MEDIA_RESOLUTION_HIGH"}
    assert models.calls[0]["contents"] == probe.PROBE_TEXT
    part, text = models.calls[1]["contents"]
    assert text == probe.PROBE_TEXT and part.inline_data.mime_type == "image/png"
    assert Image.open(BytesIO(part.inline_data.data)).size == (384, 384)


def test_tokens_per_image(capsys):
    _, out, *_ = run_main(capsys=capsys)
    assert json.loads(out)["tokens_per_image"] == 1100
    models = FakeModels([usage(12), usage(1132)])
    _, out, *_ = run_main(models=models, capsys=capsys)
    assert json.loads(out)["tokens_per_image"] == 1120


def test_unlisted_hydrogen_makes_no_paid_call(capsys):
    code, out, models, http, made = run_main(http=FakeHttp(models=("gemini-embedding-2",)), capsys=capsys)
    assert code == 2 and models.calls == [] and made == []
    assert len(http.requests) == 1


def test_failure_prints_type_only(capsys):
    code, out, models, *_ = run_main(models=default_models(fail_at=2), capsys=capsys)
    assert code == 1
    assert json.loads(out) == {"status": "failed", "phase": "image", "error_type": "RuntimeError"}
    assert KEY not in run_main.full and SECRET not in run_main.full
    code, out, *_ = run_main(models=default_models(fail_at=1), capsys=capsys)
    assert json.loads(out)["phase"] == "text"
    assert KEY not in run_main.full and SECRET not in run_main.full
    code, out, models, *_ = run_main(http=FakeHttp(fail=RuntimeError("key=" + KEY)), capsys=capsys)
    assert code == 1 and json.loads(out)["phase"] == "listing" and models.calls == []
    assert KEY not in run_main.full and SECRET not in run_main.full


def test_requires_confirm_flag(capsys):
    code, out, models, http, made = run_main(argv=(), capsys=capsys)
    assert code == 2 and models.calls == [] and http.requests == [] and made == []


def test_missing_key_makes_no_call(capsys):
    code, out, models, http, made = run_main(env={}, capsys=capsys)
    assert code == 2 and models.calls == [] and http.requests == [] and made == []


def test_unrecognized_version(capsys):
    models = default_models(version="bad version!")
    _, out, *_ = run_main(models=models, capsys=capsys)
    body = json.loads(out)
    assert body["text"]["model_version"] == "unrecognized"
    assert "bad version" not in out


def test_bad_counts_print_null(capsys):
    bad = types.GenerateContentResponseUsageMetadata.model_construct(
        prompt_token_count=True, candidates_token_count=-1, thoughts_token_count="3",
        total_token_count=1.5)
    models = FakeModels([bad, usage(-5, image=1100)])
    _, out, *_ = run_main(models=models, capsys=capsys)
    body = json.loads(out)
    assert body["text"]["prompt_token_count"] is None
    assert body["image"]["prompt_token_count"] is None
    assert body["tokens_per_image"] is None or isinstance(body["tokens_per_image"], int)


def test_probe_workflow_is_minimal():
    text = (ROOT / ".github/workflows/hydrogen_probe.yml").read_text()
    assert "up to seven Hydrogen calls" in text and "up to 15000 output tokens" in text
    assert "one gemini-embedding-2 call" in text
    assert "workflow_dispatch:" in text
    for banned in ("schedule:", "push:", "pull_request", "id-token", "contents: write",
                   "SPEND_POLICY_SHA256", "upload-artifact", "git commit", "git push", "publish_readme"):
        assert banned not in text, banned
    assert set(re.findall(r"secrets\.([A-Z0-9_]+)", text)) == {"GEMINI_API_KEY"}
    assert "permissions:\n  contents: read\njobs:" in text and text.count("permissions:") == 1
    assert "python3 scripts/probe_hydrogen.py --confirm-paid-probe" in text


def _top_level_block(text, key):
    match = re.search(r"^" + key + r":\n(?:  .*\n)+", text, re.MULTILINE)
    return match.group(0) if match else None


def test_probe_workflow_has_the_owner_guard_timeout_and_writer_queue():
    text = (ROOT / ".github/workflows/hydrogen_probe.yml").read_text()
    daily = (ROOT / ".github/workflows/daily_zeitgeist.yml").read_text()
    actor = "github.actor == 'Siyuan-Zhu' &&"
    repository = ("(github.repository == 'Siyuan-Zhu/readme-live-sandbox' || "
                  "github.repository == 'Siyuan-Zhu/Siyuan-Zhu')")
    assert actor in daily and repository in daily
    assert ("  probe:\n    if: >-\n      " + actor + "\n      " + repository + "\n") in text
    assert "\n    timeout-minutes: 15\n" in text and text.count("timeout-minutes:") == 1
    block = "concurrency:\n  group: readme-state-writer\n  cancel-in-progress: false\n  queue: max\n"
    assert _top_level_block(text, "concurrency") == block == _top_level_block(daily, "concurrency")
    assert text.count("concurrency:") == 1


def test_private_run_workflow_comment_names_every_state_write():
    text = (ROOT / ".github/workflows/private_run.yml").read_text()
    comments = " ".join(line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#"))
    assert "only state write" not in comments
    assert "judge_calibration" in comments
    assert "maintain_spend_control.py --apply" in comments and "spend ledger" in comments
    assert "holds" in comments


# ---- Task A8b: thinking fit for the Director call ----

THINK_USAGE = types.GenerateContentResponseUsageMetadata(
    prompt_token_count=2000, candidates_token_count=300, thoughts_token_count=1500,
    total_token_count=3800)


def run_fit(capsys, **kw):
    models = FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE], **kw)
    code, _, models, http, made = run_main(models=models, capsys=capsys)
    lines = run_main.full.splitlines()
    return code, lines, models


FIT_PLAN = (("high", 15000), ("high", 15000), ("high", 15000), ("medium", 15000))


def test_thinking_fit_calls_three_high_then_one_medium_at_15000(capsys):
    code, lines, models = run_fit(capsys, text='{"a": 1}')
    assert code == 0 and len(models.calls) == 7
    for call, (level, tokens) in zip(models.calls[2:], FIT_PLAN):
        assert call["model"] == "gemini-4-hydrogen"
        assert call["config"]["max_output_tokens"] == tokens
        assert call["config"]["response_mime_type"] == "application/json"
        assert call["config"]["thinking_config"] == {"thinking_level": level}
        assert isinstance(call["contents"], str)


def test_thinking_fit_line_format(capsys, monkeypatch):
    fixed_clock(monkeypatch, 1.5)
    code, lines, _ = run_fit(capsys, text='{"a": 1}')
    assert len(lines) == 8
    assert json.loads(lines[1]) == {
        "thinking_level": "high", "max_output_tokens": 15000, "finish_reason": "STOP",
        "prompt_tokens": 2000, "candidates_tokens": 300, "thoughts_tokens": 1500,
        "total_output": 1800, "json_ok": 1, "seconds": 1.5}
    assert [(json.loads(l)["thinking_level"], json.loads(l)["max_output_tokens"])
            for l in lines[1:5]] == list(FIT_PLAN)
    assert json.loads(lines[6]) == {"max_total_output_high": 1800, "max_total_output_medium": 1800,
                                    "max_seconds_high": 1.5, "critic_total_output": 1800,
                                    "critic_seconds": 1.5}


def test_summary_takes_the_max_per_level_and_null_when_none(capsys):
    usages = [usage(12), usage(1112, image=1100)] + [
        types.GenerateContentResponseUsageMetadata(prompt_token_count=1, candidates_token_count=c,
                                                   thoughts_token_count=t_)
        for c, t_ in ((100, 200), (500, 700), (50, 60), (10, 20))]
    run_main(models=FakeModels(usages, text='{"a": 1}'), capsys=capsys)
    lines = run_main.full.splitlines()
    assert [json.loads(l)["total_output"] for l in lines[1:5]] == [300, 1200, 110, 30]
    summary = json.loads(lines[6])
    assert {k: summary[k] for k in ("max_total_output_high", "max_total_output_medium",
                                    "critic_total_output")} == {
        "max_total_output_high": 1200, "max_total_output_medium": 30, "critic_total_output": 30}
    bad = types.GenerateContentResponseUsageMetadata.model_construct(
        prompt_token_count=1, candidates_token_count=5, thoughts_token_count=None)
    run_main(models=FakeModels([usage(12), usage(1112, image=1100), bad], text="{}"), capsys=capsys)
    lines = run_main.full.splitlines()
    assert json.loads(lines[1])["total_output"] is None
    summary = json.loads(lines[6])
    assert {k: summary[k] for k in ("max_total_output_high", "max_total_output_medium",
                                    "critic_total_output")} == {
        "max_total_output_high": None, "max_total_output_medium": None, "critic_total_output": None}


def test_json_ok_rules(capsys):
    _, lines, _ = run_fit(capsys, text='{"a": 1}')
    assert json.loads(lines[1])["json_ok"] == 1
    _, lines, _ = run_fit(capsys, text='{"a": [1, 2')
    assert json.loads(lines[1])["json_ok"] == 0
    _, lines, _ = run_fit(capsys, text='{"a": 1}', finish_reason=types.FinishReason.MAX_TOKENS)
    row = json.loads(lines[1])
    assert row["finish_reason"] == "MAX_TOKENS" and row["json_ok"] == 0


def test_full_output_never_leaks_key_or_secret(capsys):
    code, lines, _ = run_fit(capsys, text=KEY + SECRET)
    assert code == 0 and len(lines) == 8
    for banned in (KEY, SECRET):
        assert banned not in run_main.full
    for row in lines[1:]:
        assert KEY not in row and SECRET not in row


def test_json_ok_requires_object(capsys):
    for text, expected in (('{"a": 1}', 1), ("42", 0), ("[1, 2]", 0), ('"x"', 0), ("null", 0)):
        _, lines, _ = run_fit(capsys, text=text)
        assert json.loads(lines[1])["json_ok"] == expected, text


def test_finish_reason_sanitised_and_counts_null(capsys):
    assert probe.safe_finish("STOP") == "STOP"
    assert probe.safe_finish("bad reason!") == "unrecognized"
    assert probe.safe_finish(None) == "unrecognized"
    assert probe.safe_finish("A" * 33) == "unrecognized"
    bad = types.GenerateContentResponseUsageMetadata.model_construct(
        prompt_token_count=True, candidates_token_count=-1, thoughts_token_count="3",
        total_token_count=1.5)
    models = FakeModels([usage(12), usage(1112, image=1100), bad], text="{}")
    run_main(models=models, capsys=capsys)
    row = json.loads(run_main.full.splitlines()[1])
    assert [row[k] for k in ("prompt_tokens", "candidates_tokens", "thoughts_tokens",
                             "total_output")] == [None] * 4


def test_thinking_fit_never_prints_reply_text(capsys):
    _, lines, _ = run_fit(capsys, text=SECRET)
    assert SECRET not in "\n".join(lines) and json.loads(lines[1])["json_ok"] == 0


def test_no_thinking_calls_when_unlisted_or_unflagged(capsys):
    _, _, models, *_ = run_main(http=FakeHttp(models=("gemini-embedding-2",)), capsys=capsys)
    assert models.calls == []
    _, _, models, *_ = run_main(argv=(), capsys=capsys)
    assert models.calls == []


def test_thinking_failure_is_type_only_and_stops(capsys):
    base = [usage(12), usage(1112, image=1100), THINK_USAGE]
    for fail_at, phase in ((3, "thinking_high_1"), (4, "thinking_high_2"), (5, "thinking_high_3"),
                           (6, "thinking_medium_4")):
        models = FakeModels(base, fail_at=fail_at)
        code, _, models, *_ = run_main(models=models, capsys=capsys)
        lines = run_main.full.splitlines()
        assert code == 1 and len(models.calls) == fail_at and len(lines) == 1 + (fail_at - 3) + 1
        assert json.loads(lines[-1]) == {"status": "failed", "phase": phase, "error_type": "RuntimeError"}
        assert KEY not in run_main.full and SECRET not in run_main.full


def test_director_fit_prompt_is_the_engine_builders_output(capsys):
    from src.zeitgeist import adk_pipeline
    clusters = probe.fit_clusters()
    assert len(clusters) == 5 and all(len(c.items) == 4 for c in clusters)
    expected = adk_pipeline._build_packet_director_prompt(
        clusters, list(probe.FIT_HISTORY), None, None, slate=True)
    prompt = probe.director_fit_prompt()
    assert prompt == expected and probe.director_fit_prompt() == prompt
    assert '[{"word": "...", "fit": "..."}]' in prompt and prompt.count("- Cluster ") == 5
    for cluster in clusters:
        for item in cluster.items:
            assert item["title"] in prompt  # every invented title arrives whole
        for keyword in cluster.cluster_keywords:
            assert keyword in prompt
    for banned in (KEY, "http", "@", "api_key", "AIza"):
        assert banned not in prompt
    _, _, models, *_ = run_main(capsys=capsys)
    assert [call["contents"] for call in models.calls[2:6]] == [prompt] * 4


# ---- Task T4: the critic call, the embedding call, timings ----

HYDROGEN = "gemini-4-hydrogen"


def fixed_clock(monkeypatch, step):
    ticks = iter(range(10 ** 6))
    monkeypatch.setattr(probe, "_clock", lambda: next(ticks) * step)


def engine_critic_request(monkeypatch):
    """The request the engine's own batch critic sends for the probe's synthetic images and commission."""
    from src.zeitgeist import visual_router as vr
    seen = []

    def generate(**kw):
        seen.append(kw)
        rows = [{"index": i, "semantic_alignment": 0.5, "compositional_clarity": 0.5,
                 "editorial_memorability": 0.5, "safety_pass": True} for i in range(9)]
        return SimpleNamespace(text=json.dumps({"candidates": rows}), candidates=[])

    monkeypatch.setenv("ZEITGEIST_LIVE_LLM", "1")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr("google.genai.Client",
                        lambda **kw: SimpleNamespace(models=SimpleNamespace(generate_content=generate)))
    monkeypatch.setattr(vr, "guard_genai_client", lambda client, **kw: client)
    scores = vr._score_photo_batch([{"thumb_bytes": raw} for raw in probe.critic_images()],
                                   word=probe.CRITIC_WORD, gloss=probe.CRITIC_GLOSS,
                                   thesis=probe.CRITIC_THESIS, model_name=HYDROGEN)
    assert len(seen) == 1 and all(s["safety_pass"] for s in scores)
    return seen[0]


def assert_same_parts(mine, theirs):
    assert len(mine) == len(theirs) == 19
    for ours, engine in zip(mine, theirs):
        if isinstance(engine, str):
            assert ours == engine
            continue
        assert ours.inline_data.mime_type == engine.inline_data.mime_type == "image/jpeg"
        assert ours.inline_data.data == engine.inline_data.data
        assert (Image.open(BytesIO(ours.inline_data.data)).size
                == Image.open(BytesIO(engine.inline_data.data)).size == (384, 384))


def test_critic_fit_parts_match_the_engine_batch_request(monkeypatch):
    raws = probe.critic_images()
    assert len(raws) == 9 and len(set(raws)) == 9
    assert all(Image.open(BytesIO(raw)).size == (384, 384) for raw in raws)
    assert_same_parts(probe.critic_fit_parts(), engine_critic_request(monkeypatch)["contents"])


def test_critic_fit_call_carries_nine_images_and_the_engine_prompt(capsys, monkeypatch):
    from src.zeitgeist import visual_router as vr
    fixed_clock(monkeypatch, 1.5)
    code, lines, models = run_fit(capsys, text='{"candidates": []}')
    assert code == 0 and len(models.calls) == 7
    call = models.calls[6]
    assert call["model"] == HYDROGEN
    assert call["config"] == {**vr.critic_config(HYDROGEN), "max_output_tokens": 15000,
                              "thinking_config": {"thinking_level": "high"},
                              "media_resolution": "MEDIA_RESOLUTION_HIGH"}
    assert sum(not isinstance(part, str) for part in call["contents"]) == 9
    assert_same_parts(call["contents"], engine_critic_request(monkeypatch)["contents"])
    assert json.loads(lines[5]) == {
        "call": "critic", "images": 9, "thinking_level": "high", "max_output_tokens": 15000,
        "finish_reason": "STOP", "prompt_tokens": 2000, "candidates_tokens": 300,
        "thoughts_tokens": 1500, "total_output": 1800, "json_ok": 1, "seconds": 1.5}


def test_critic_failure_is_type_only_and_stops(capsys):
    models = FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE], fail_at=7)
    code, _, models, *_ = run_main(models=models, capsys=capsys)
    lines = run_main.full.splitlines()
    assert code == 1 and len(models.calls) == 7 and models.embed_calls == [] and len(lines) == 6
    assert json.loads(lines[-1]) == {"status": "failed", "phase": "critic", "error_type": "RuntimeError"}
    assert KEY not in run_main.full and SECRET not in run_main.full


def test_embedding_step_line(capsys, monkeypatch):
    from src.zeitgeist import collectors
    fixed_clock(monkeypatch, 1.5)
    code, lines, models = run_fit(capsys, text='{"a": 1}')
    assert code == 0 and len(lines) == 8
    assert models.embed_calls == [{
        "model": "gemini-embedding-2",
        "contents": [collectors._request_text(text, "gemini-embedding-2") for text in probe.EMBED_TEXTS],
        "config": {"output_dimensionality": collectors.EMBEDDING_DIM}}]
    assert all(c.startswith("task: clustering | query: ") for c in models.embed_calls[0]["contents"])
    assert len(probe.EMBED_TEXTS) == 2
    assert json.loads(lines[7]) == {"call": "embedding", "model": "gemini-embedding-2", "vectors": 2,
                                    "dimension": 768, "finite": 1, "billed_tokens": 14, "seconds": 1.5}


@pytest.mark.parametrize("values, tokens, expected", [
    ([[0.1] * 768, [float("nan")] + [0.1] * 767], 7.0,
     {"vectors": 2, "dimension": 768, "finite": 0, "billed_tokens": 14}),
    ([[0.1] * 768, [0.1] * 3], None, {"vectors": 2, "dimension": None, "finite": 1, "billed_tokens": None}),
    ([[0.1] * 768], 2.5, {"vectors": 1, "dimension": 768, "finite": 1, "billed_tokens": None}),
    ([], None, {"vectors": 0, "dimension": None, "finite": 0, "billed_tokens": None}),
])
def test_embedding_line_reports_shape_and_finiteness(capsys, values, tokens, expected):
    models = FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE], text="{}",
                        embed_values=values, embed_tokens=tokens)
    code, _, models, *_ = run_main(models=models, capsys=capsys)
    row = json.loads(run_main.full.splitlines()[-1])
    assert code == 0 and row["call"] == "embedding"
    assert {k: row[k] for k in expected} == expected


def test_no_embedding_call_when_embedding_2_unlisted(capsys):
    code, _, models, *_ = run_main(http=FakeHttp(models=("gemini-4-hydrogen",)), capsys=capsys)
    lines = run_main.full.splitlines()
    assert code == 0 and len(models.calls) == 7 and models.embed_calls == [] and len(lines) == 7
    assert "max_seconds_high" in json.loads(lines[-1])
    assert '"embedding"' not in run_main.full
    _, _, models, *_ = run_main(http=FakeHttp(models=("gemini-embedding-2",)), capsys=capsys)
    assert models.calls == [] and models.embed_calls == []


def test_embedding_failure_is_type_only_after_the_summary(capsys):
    models = FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE], text="{}", embed_fail=True)
    code, _, models, *_ = run_main(models=models, capsys=capsys)
    lines = run_main.full.splitlines()
    assert code == 1 and len(models.embed_calls) == 1 and len(lines) == 8
    assert "critic_total_output" in json.loads(lines[6])
    assert json.loads(lines[7]) == {"status": "failed", "phase": "embedding", "error_type": "RuntimeError"}
    assert KEY not in run_main.full and SECRET not in run_main.full


def test_every_paid_call_line_has_seconds(capsys, monkeypatch):
    fixed_clock(monkeypatch, 2.34)
    _, lines, _ = run_fit(capsys, text="{}")
    first = json.loads(lines[0])
    assert first["text"]["seconds"] == first["image"]["seconds"] == 2.3
    assert [json.loads(line)["seconds"] for line in lines[1:6] + lines[7:8]] == [2.3] * 6
    summary = json.loads(lines[6])
    assert summary["max_seconds_high"] == summary["critic_seconds"] == 2.3
    _, lines, _ = run_fit(capsys, text="{}")  # the real clock
    for line in lines[1:6]:
        seconds = json.loads(line)["seconds"]
        assert isinstance(seconds, float) and seconds >= 0 and round(seconds, 1) == seconds


def _strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_leak_scan_over_all_printed_output(capsys):
    models = FakeModels([usage(12), usage(1112, image=1100), THINK_USAGE], text=KEY + SECRET)
    code, _, models, *_ = run_main(models=models, capsys=capsys)
    full = run_main.full
    assert code == 0 and len(full.splitlines()) == 8
    synthetic = [KEY, SECRET, probe.CRITIC_WORD, probe.CRITIC_GLOSS, probe.CRITIC_THESIS,
                 *probe.EMBED_TEXTS, *probe.FIT_HISTORY]
    for cluster in probe.fit_clusters():
        synthetic += [item["title"] for item in cluster.items]
    for banned in synthetic:
        assert banned not in full, banned
    public = {"ok", "STOP", "high", "medium", "critic", "embedding", "generateContent",
              "gemini-4-hydrogen", "gemini-embedding-2", "synthetic-hydrogen-version"}
    for line in full.splitlines():
        assert set(_strings(json.loads(line))) <= public, line


def test_docstring_states_the_cap_rule():
    doc = " ".join(probe.__doc__.split())
    assert "4000/6000/8000/10000/15000" in doc
    assert "at least 1.5 times the LARGER of max_total_output_high and critic_total_output" in doc
