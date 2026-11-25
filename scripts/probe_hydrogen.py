#!/usr/bin/env python3
"""Manual Gemini 4 probe (Hydrogen by default): list models, then make up to nine paid calls.

Prints model versions, token counts and timings only. It never prints the key, response
text, or any credential identity. Each call is one SDK attempt, and nothing paid runs for a
model that is not listed. The calls, in order:

- two tiny Hydrogen calls (16 output tokens), text then one 384-pixel image;
- four Director calls, each with max_output_tokens 15000 on the prompt the engine's own
  Director builder makes for synthetic clusters (three at thinking level high, then one at
  medium);
- one critic call: a synthetic batch of nine 384-pixel images in the engine's batch critic
  request shape, at thinking level high with max_output_tokens 15000;
- one cap_check call: the Director prompt at thinking level high with max_output_tokens 1024,
  to show whether thoughts are bounded by the output cap (a reply cut off by the cap is the
  expected result, not a probe failure);
- one gemini-embedding-2 call, only if that model is listed: two short synthetic texts in the
  engine's clustering form at 768 dimensions (the embedder's own request contents); the probe
  exits 1 after the embedding line when its vector count differs from its text count.

Each call prints one numbers-only line with seconds=<elapsed, one decimal> (total_output =
candidates + thoughts tokens). Every generate line carries usage_metadata, the reply's own
usageMetadata record. The text call, the Director-sized calls and the critic call also print
counted_tokens: the free countTokens figure for the same model and contents (null when it
fails); the critic line adds counted_text_tokens, the count with every image part replaced by an
empty text part (the guard's own second count). A summary line after the cap_check call gives
max_total_output_high, max_total_output_medium, max_seconds_high, critic_total_output,
critic_seconds and max_prompt_minus_counted (the largest billed prompt tokens minus counted
tokens); the embedding line follows it. When the critic or cap_check call fails, the summary is
still printed (critic fields null if the critic failed) before the failed line.

Coordinator rule for choosing the Hydrogen output cap (not enforced by code): take the smallest
cap from 4000/6000/8000/10000/15000 that is at least 1.5 times the LARGER of
max_total_output_high and critic_total_output; only if 15000 is still too small, apply the same
rule at medium to max_total_output_medium.

--model (gemini-4-hydrogen by default, or gemini-4-neon) and --thinking-level (high by default;
low, medium, high or xhigh) choose what is probed. The default pair makes exactly the calls above
and prints exactly the lines above. Any other pair (the Neon probe is --model gemini-4-neon
--thinking-level medium):

- lists only the chosen model (Hydrogen also lists gemini-embedding-2 and keeps its embedding
  call; Neon makes none);
- makes the same two tiny calls, then one tiny temperature_check call carrying temperature 0.1
  and candidate_count 1, as the guard sends them where sampling controls are allowed; its line
  records accepted 1 or 0 with the error class and HTTP status, and it is never fatal;
- runs all four Director calls, the critic call and the cap_check call at the requested level;
- adds model_version to every Director, critic and cap_check line and http_status to a failed
  line; if the first call at the requested level fails, a line with thinking_level_accepted 0,
  the error class and HTTP status comes before the failed line;
- prints a summary with max_total_output_<level>, max_thoughts_<level>, max_seconds_<level>,
  critic_total_output, critic_thoughts, critic_seconds, max_prompt_minus_counted,
  thinking_level_accepted and model_versions (every distinct response version seen). The cap rule
  above then reads max_total_output_<level> in place of max_total_output_high.

They are authorised
outside the spend guard by the spec, so the command refuses to run without --confirm-paid-probe.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.provider_preflight import inspect_gemini  # noqa: E402

HYDROGEN = "gemini-4-hydrogen"
NEON = "gemini-4-neon"
EMBEDDING_2 = "gemini-embedding-2"
PROBE_MODELS = (HYDROGEN, EMBEDDING_2)
PROBE_MODEL_CHOICES = (HYDROGEN, NEON)
NOT_LISTED_STATUS = {HYDROGEN: "hydrogen_not_listed", NEON: "neon_not_listed"}
THINKING_LEVELS = ("low", "medium", "high", "xhigh")
DEFAULT_THINKING_LEVEL = "high"
PROBE_TEXT = "A quiet harbor at dawn with three small boats."
VERSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
CALL_CONFIG = {
    "max_output_tokens": 16,
    "thinking_config": {"thinking_level": "low"},
    "media_resolution": "MEDIA_RESOLUTION_HIGH",
}
# The guard adds candidate_count 1 to every request that carries sampling controls.
TEMPERATURE_CHECK_CONFIG = {**CALL_CONFIG, "temperature": 0.1, "candidate_count": 1}
COUNT_FIELDS = (
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)


FIT_OUTPUT_TOKENS = 15000
CAP_CHECK_OUTPUT_TOKENS = 1024
FIT_PLAN = (("high", FIT_OUTPUT_TOKENS),) * 3 + (("medium", FIT_OUTPUT_TOKENS),)
FINISH_PATTERN = re.compile(r"[A-Z_]{1,32}")
FIT_TOPICS = (
    ("harbor", "tide", "ferry", "lantern", "mooring"),
    ("orchard", "frost", "ladder", "harvest", "cider"),
    ("workshop", "lathe", "timber", "varnish", "chisel"),
    ("observatory", "comet", "lens", "dome", "almanac"),
    ("market", "awning", "ledger", "spice", "scale"),
)
FIT_HISTORY = (
    "tessera", "solstice", "cadence", "ember", "plinth", "verdigris", "lattice", "murmur",
    "tangent", "vellum", "cinder", "aperture", "meridian", "fathom", "saffron", "bastion",
    "quorum", "ripple", "canopy", "ballast", "nocturne", "sextant", "parable", "gossamer",
)
CRITIC_WORD = "Lantern"
CRITIC_GLOSS = "A small light carried through dusk"
CRITIC_THESIS = "Neighbours share one steady light while the harbor tide turns at dusk."
CRITIC_IMAGES = 9
EMBED_TEXTS = (
    "A ferry crosses the quiet harbor at first light.",
    "Volunteers repaint the old market awning before the frost.",
)
_clock = time.monotonic


def fit_clusters():
    """Five synthetic clusters, four invented titles each (whole under the engine's title cut)."""
    clusters = []
    for idx, words in enumerate(FIT_TOPICS):
        items = [
            {"title": f"Local {words[n]} story {n + 1}: residents compare the {words[(n + 1) % 5]} "
                      f"and the {words[(n + 2) % 5]} as the {words[(n + 3) % 5]} changes this season"}
            for n in range(4)
        ]
        clusters.append(SimpleNamespace(
            cluster_id=f"synthetic_{idx}", items=items, cluster_keywords=list(words),
            commission_score=0.9 - idx * 0.1, velocity=1.5 - idx * 0.2, orbit_resonance=1.0))
    return clusters


def director_fit_prompt() -> str:
    """The Director prompt the engine's own builder makes for the synthetic clusters (slate on)."""
    from src.zeitgeist import adk_pipeline

    return adk_pipeline._build_packet_director_prompt(
        fit_clusters(), list(FIT_HISTORY), None, None, slate=True)


def critic_batch_text(word, gloss, thesis) -> str:
    # Copy of the engine's batch critic text (visual_router._score_photo_batch); a test pins it.
    return (f'Evaluate these photographs against word: {word}\nGloss: {gloss}\nThesis: {thesis}\n'
            'Return strict JSON {"candidates": [{"index": 0, "semantic_alignment": 0.0, '
            '"compositional_clarity": 0.0, "editorial_memorability": 0.0, "safety_pass": false}]}. '
            'Return exactly one result per candidate index. Scores must be finite numbers from 0 to 1. '
            'safety_pass must be a JSON boolean, true only when completely safe, nonviolent, non-NSFW, '
            'minimalist, free of watermarks and memes. Judge semantic fit to the actual commission.')


def critic_images():
    return [synthetic_image(variant) for variant in range(CRITIC_IMAGES)]


def critic_fit_parts():
    """The engine's batch critic request for the synthetic images: labels, 384-pixel JPEGs, text."""
    from google.genai import types
    from PIL import Image

    parts = []
    for index, raw in enumerate(critic_images()):
        with Image.open(BytesIO(raw)) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail((384, 384), Image.Resampling.LANCZOS)
            encoded = BytesIO()
            thumb.save(encoded, format="JPEG", quality=85)
        parts.extend([f"Candidate index {index}",
                      types.Part.from_bytes(data=encoded.getvalue(), mime_type="image/jpeg")])
    parts.append(critic_batch_text(CRITIC_WORD, CRITIC_GLOSS, CRITIC_THESIS))
    return parts


def is_default_run(model, level):
    """The pair the probe ran before it took flags; its calls and lines stay exactly as they were."""
    return model == HYDROGEN and level == DEFAULT_THINKING_LEVEL


def fit_plan(model=HYDROGEN, level=DEFAULT_THINKING_LEVEL):
    if is_default_run(model, level):
        return FIT_PLAN
    return ((level, FIT_OUTPUT_TOKENS),) * len(FIT_PLAN)


def counted_tokens(genai_client, contents, model=HYDROGEN):
    """countTokens (a free metadata endpoint) for the same model and contents; null on any failure."""
    try:
        return safe_count(genai_client.models.count_tokens(model=model, contents=contents).total_tokens)
    except Exception:
        return None


def text_only_parts(parts):
    """The guard's second count: every image part replaced by an empty text part."""
    return [part if isinstance(part, str) else {"text": ""} for part in parts]


def usage_record(response):
    """The reply's own usageMetadata record, as the guard stores and hashes it."""
    try:
        return response.model_dump(by_alias=True, exclude_none=True).get("usageMetadata")
    except Exception:
        return None


def max_prompt_minus_counted(measured):
    gaps = [row["prompt_tokens"] - row["counted_tokens"] for row in measured
            if row.get("prompt_tokens") is not None and row.get("counted_tokens") is not None]
    return max(gaps) if gaps else None


def _timed(call):
    start = _clock()
    result = call()
    return result, round(_clock() - start, 1)


def safe_finish(value):
    name = getattr(value, "name", value)
    if isinstance(name, str) and FINISH_PATTERN.fullmatch(name):
        return name
    return "unrecognized"


def _reply_text(response):
    try:
        text = response.text
    except Exception:
        return None
    return text if isinstance(text, str) else None


def fit_row(level, max_output_tokens, response):
    usage = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", None) or [None]
    finish = safe_finish(getattr(candidates[0], "finish_reason", None))
    json_ok = 0
    text = _reply_text(response)
    if text is not None and finish != "MAX_TOKENS":
        try:
            json_ok = 1 if isinstance(json.loads(text), dict) else 0
        except ValueError:
            pass
    produced = safe_count(getattr(usage, "candidates_token_count", None))
    thought = safe_count(getattr(usage, "thoughts_token_count", None))
    total_output = None if produced is None or thought is None else produced + thought
    return {
        "thinking_level": level,
        "max_output_tokens": max_output_tokens,
        "finish_reason": finish,
        "prompt_tokens": safe_count(getattr(usage, "prompt_token_count", None)),
        "candidates_tokens": safe_count(getattr(usage, "candidates_token_count", None)),
        "thoughts_tokens": safe_count(getattr(usage, "thoughts_token_count", None)),
        "total_output": total_output,
        "json_ok": json_ok,
    }


def response_version(response):
    return safe_version(getattr(response, "model_version", None))


def with_version(row, response, model, level):
    """Off the default run every call's line names the reply's model version."""
    if not is_default_run(model, level):
        row["model_version"] = response_version(response)
    return row


def run_thinking_fit(genai_client, model=HYDROGEN, run_level=DEFAULT_THINKING_LEVEL):
    """One call per (level, output cap) in the fit plan, in order; a failure stops before the next call."""
    try:
        prompt = director_fit_prompt()
    except Exception as error:
        raise ProbeFailure.of("director_prompt", error) from None
    for index, (level, max_output_tokens) in enumerate(fit_plan(model, run_level), start=1):
        config = {
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
            "thinking_config": {"thinking_level": level},
        }
        try:
            counted = counted_tokens(genai_client, prompt, model)
            response, seconds = _timed(lambda: genai_client.models.generate_content(
                model=model, contents=prompt, config=config
            ))
            row = fit_row(level, max_output_tokens, response)
        except Exception as error:
            raise ProbeFailure.of(f"thinking_{level}_{index}", error) from None
        yield with_version({**row, "counted_tokens": counted, "usage_metadata": usage_record(response),
                            "seconds": seconds}, response, model, run_level)


def run_critic_fit(genai_client, model=HYDROGEN, level=DEFAULT_THINKING_LEVEL):
    """One critic call in the engine's batch shape at the run's level; config as the guard sends it."""
    config = {
        "response_mime_type": "application/json",
        "max_output_tokens": FIT_OUTPUT_TOKENS,
        "thinking_config": {"thinking_level": level},
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
    }
    try:
        parts = critic_fit_parts()
        counted = counted_tokens(genai_client, parts, model)
        counted_text = counted_tokens(genai_client, text_only_parts(parts), model)
        response, seconds = _timed(lambda: genai_client.models.generate_content(
            model=model, contents=parts, config=config
        ))
        row = fit_row(level, FIT_OUTPUT_TOKENS, response)
    except Exception as error:
        raise ProbeFailure.of("critic", error) from None
    return with_version({"call": "critic", "images": CRITIC_IMAGES, **row, "counted_tokens": counted,
                         "counted_text_tokens": counted_text, "usage_metadata": usage_record(response),
                         "seconds": seconds}, response, model, level)


def run_cap_check(genai_client, model=HYDROGEN, level=DEFAULT_THINKING_LEVEL):
    """One Director-sized call at the run's level with a 1024-token cap: are thoughts bounded by it?"""
    config = {
        "max_output_tokens": CAP_CHECK_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "thinking_config": {"thinking_level": level},
    }
    try:
        prompt = director_fit_prompt()
        counted = counted_tokens(genai_client, prompt, model)
        response, seconds = _timed(lambda: genai_client.models.generate_content(
            model=model, contents=prompt, config=config
        ))
        row = fit_row(level, CAP_CHECK_OUTPUT_TOKENS, response)
    except Exception as error:
        raise ProbeFailure.of("cap_check", error) from None
    return with_version({"call": "cap_check", "max_output_tokens": CAP_CHECK_OUTPUT_TOKENS,
                         "candidates_tokens": row["candidates_tokens"], "thoughts_tokens": row["thoughts_tokens"],
                         "total_output": row["total_output"], "finish_reason": row["finish_reason"],
                         "prompt_tokens": row["prompt_tokens"], "counted_tokens": counted,
                         "usage_metadata": usage_record(response), "seconds": seconds},
                        response, model, level)


def run_temperature_check(genai_client, model):
    """One tiny call carrying temperature; a refusal is recorded (class and HTTP status), never fatal."""
    start = _clock()
    try:
        response = genai_client.models.generate_content(
            model=model, contents=PROBE_TEXT, config=TEMPERATURE_CHECK_CONFIG)
    except Exception as error:
        return {"call": "temperature_check", "accepted": 0, "error_type": type(error).__name__,
                "http_status": http_status(error), "model_version": None,
                "seconds": round(_clock() - start, 1)}
    return {"call": "temperature_check", "accepted": 1, "error_type": None, "http_status": None,
            "model_version": response_version(response), "usage_metadata": usage_record(response),
            "seconds": round(_clock() - start, 1)}


def _billed_tokens(response):
    total = 0
    for embedding in getattr(response, "embeddings", None) or []:
        value = getattr(getattr(embedding, "statistics", None), "token_count", None)
        if isinstance(value, float) and math.isfinite(value) and value.is_integer():
            value = int(value)
        value = safe_count(value)
        if value is None:
            return None
        total += value
    return total if getattr(response, "embeddings", None) else None


def embedding_row(response, seconds, texts):
    vectors = list(getattr(response, "embeddings", None) or [])
    values = [list(getattr(v, "values", None) or []) for v in vectors]
    sizes = {len(v) for v in values}
    finite = bool(values) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        for vector in values for x in vector)
    return {
        "call": "embedding",
        "model": EMBEDDING_2,
        "texts": texts,
        "vectors": len(vectors),
        "dimension": sizes.pop() if len(sizes) == 1 and 0 not in sizes else None,
        "finite": 1 if finite else 0,
        "billed_tokens": _billed_tokens(response),
        "seconds": seconds,
    }


def run_embedding(genai_client):
    """One embed_content call with two synthetic texts in the engine's clustering form."""
    from src.zeitgeist.collectors import EMBEDDING_DIM, embedding_request_contents

    try:
        contents = embedding_request_contents(EMBED_TEXTS, EMBEDDING_2)
        response, seconds = _timed(lambda: genai_client.models.embed_content(
            model=EMBEDDING_2, contents=contents, config={"output_dimensionality": EMBEDDING_DIM}
        ))
        return embedding_row(response, seconds, len(EMBED_TEXTS))
    except Exception as error:
        raise ProbeFailure.of("embedding", error) from None


def max_at_level(rows, level, field):
    values = [r[field] for r in rows if r["thinking_level"] == level and r[field] is not None]
    return max(values) if values else None


def max_total_output(rows, level):
    return max_at_level(rows, level, "total_output")


def http_status(error):
    """The API error's integer HTTP status (google.genai errors carry it as code), else null."""
    return safe_count(getattr(error, "code", None))


class ProbeFailure(Exception):
    def __init__(self, phase, error_type, http_status=None):
        super().__init__(phase)
        self.phase = phase
        self.error_type = error_type
        self.http_status = http_status

    @classmethod
    def of(cls, phase, error):
        return cls(phase, type(error).__name__, http_status(error))


def synthetic_image(variant: int = 0) -> bytes:
    from PIL import Image, ImageDraw

    shift = variant * 23
    image = Image.new("RGB", (384, 384), (235, 240 - shift // 2, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 220, 344, 344), fill=(40 + shift, 90, 140))
    draw.polygon([(192, 60 + variant * 8), (292, 220), (92, 220)], fill=(220, 120, 60 + shift))
    draw.ellipse((280 - variant * 20, 40, 340 - variant * 20, 100), fill=(250, 220, 90))
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def safe_count(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def safe_version(value):
    if isinstance(value, str) and VERSION_PATTERN.fullmatch(value):
        return value
    return "unrecognized"


def summarize(response):
    usage = getattr(response, "usage_metadata", None)
    summary = {"model_version": safe_version(getattr(response, "model_version", None))}
    for field in COUNT_FIELDS:
        summary[field] = safe_count(getattr(usage, field, None))
    return summary, usage


def image_detail_count(usage):
    for detail in getattr(usage, "prompt_tokens_details", None) or []:
        modality = getattr(detail, "modality", None)
        if str(getattr(modality, "value", modality)).upper() == "IMAGE":
            return safe_count(getattr(detail, "token_count", None))
    return None


def tokens_per_image(report):
    text, image = report["text"], report["image"]
    if text is None or image is None:
        return None
    if text["prompt_token_count"] is None or image["prompt_token_count"] is None:
        return None
    return max(
        image["prompt_token_count"] - text["prompt_token_count"],
        image["image_prompt_token_count"] or 0,
    )


def run_probe(key: str, *, http_client, genai_client, model=HYDROGEN) -> dict:
    from google.genai import types

    try:
        found = inspect_gemini(http_client, key, PROBE_MODELS if model == HYDROGEN else (model,))
    except Exception as error:
        raise ProbeFailure.of("listing", error) from None
    report = {
        "listed": found["available_required_models"],
        "missing": found["missing_models"],
        "text": None,
        "image": None,
    }
    if model in report["missing"]:
        return report
    try:
        counted = counted_tokens(genai_client, PROBE_TEXT, model)
        response, seconds = _timed(lambda: genai_client.models.generate_content(
            model=model, contents=PROBE_TEXT, config=CALL_CONFIG
        ))
        report["text"], _ = summarize(response)
        report["text"].update(counted_tokens=counted, usage_metadata=usage_record(response))
        report["text"]["seconds"] = seconds
    except Exception as error:
        raise ProbeFailure.of("text", error) from None
    try:
        part = types.Part.from_bytes(data=synthetic_image(), mime_type="image/png")
        response, seconds = _timed(lambda: genai_client.models.generate_content(
            model=model, contents=[part, PROBE_TEXT], config=CALL_CONFIG
        ))
        report["image"], usage = summarize(response)
        report["image"]["image_prompt_token_count"] = image_detail_count(usage)
        report["image"]["usage_metadata"] = usage_record(response)
        report["image"]["seconds"] = seconds
    except Exception as error:
        raise ProbeFailure.of("image", error) from None
    return report


def _http_client():
    import httpx

    return httpx.Client(timeout=20, follow_redirects=False)


def _genai_client(key):
    from google import genai

    return genai.Client(api_key=key, http_options={"retry_options": {"attempts": 1}})


class _LazyGenai:
    """Build the paid client only when the first paid call is made."""

    def __init__(self, factory, key):
        self._factory, self._key, self._client = factory, key, None

    @property
    def models(self):
        if self._client is None:
            self._client = self._factory(self._key)
        return self._client.models


def failed_line(phase, error_type, http_status, default_run):
    line = {"status": "failed", "phase": phase, "error_type": error_type}
    if not default_run:
        line["http_status"] = http_status
    return line


def default_summary(rows):
    summary = {"max_total_output_" + level: max_total_output(rows, level)
               for level in ("high", "medium")}
    high_seconds = [r["seconds"] for r in rows if r["thinking_level"] == "high"]
    summary.update(max_seconds_high=max(high_seconds) if high_seconds else None,
                   critic_total_output=None, critic_seconds=None)
    return summary


def level_summary(rows, level):
    seconds = [r["seconds"] for r in rows if r["thinking_level"] == level]
    return {"max_total_output_" + level: max_total_output(rows, level),
            "max_thoughts_" + level: max_at_level(rows, level, "thoughts_tokens"),
            "max_seconds_" + level: max(seconds) if seconds else None,
            "critic_total_output": None, "critic_thoughts": None, "critic_seconds": None}


def main(argv=None, *, env=None, http_factory=None, genai_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-paid-probe", action="store_true")
    parser.add_argument("--model", choices=PROBE_MODEL_CHOICES, default=HYDROGEN)
    parser.add_argument("--thinking-level", choices=THINKING_LEVELS, default=DEFAULT_THINKING_LEVEL)
    args = parser.parse_args(argv)
    model, level = args.model, args.thinking_level
    default_run = is_default_run(model, level)
    env = os.environ if env is None else env
    key = env.get("GEMINI_API_KEY", "")
    if not args.confirm_paid_probe or not key:
        print(json.dumps({"status": "refused"}))
        return 2
    genai_client = _LazyGenai(genai_factory or _genai_client, key)
    try:
        http_client = (http_factory or _http_client)()
        try:
            report = run_probe(key, http_client=http_client, genai_client=genai_client, model=model)
        finally:
            close = getattr(http_client, "close", None)
            if close:
                close()
    except ProbeFailure as failure:
        print(json.dumps(failed_line(failure.phase, failure.error_type, failure.http_status, default_run)))
        return 1
    except Exception as error:
        print(json.dumps(failed_line("listing", type(error).__name__, http_status(error), default_run)))
        return 1
    if model in report["missing"]:
        print(json.dumps({"status": NOT_LISTED_STATUS[model], **report}))
        return 2
    print(json.dumps({"status": "ok", **report, "tokens_per_image": tokens_per_image(report)}))
    versions = [report["text"]["model_version"], report["image"]["model_version"]]
    if not default_run:
        temperature = run_temperature_check(genai_client, model)
        print(json.dumps(temperature))
        versions.append(temperature["model_version"])
    try:
        rows = []
        try:
            for row in run_thinking_fit(genai_client, model, level):
                rows.append(row)
                print(json.dumps(row))
        except ProbeFailure as failure:
            if not default_run and not rows and failure.phase != "director_prompt":
                print(json.dumps({"thinking_level": level, "thinking_level_accepted": 0,
                                  "error_type": failure.error_type, "http_status": failure.http_status}))
            raise
        if default_run:
            summary = default_summary(rows)
        else:
            summary = level_summary(rows, level)
            summary["thinking_level_accepted"] = 1
            versions += [row["model_version"] for row in rows]
        measured = [{"prompt_tokens": report["text"]["prompt_token_count"],
                     "counted_tokens": report["text"]["counted_tokens"]}] + rows

        def finish(summary):
            summary["max_prompt_minus_counted"] = max_prompt_minus_counted(measured)
            if not default_run:
                summary["model_versions"] = sorted({v for v in versions if v is not None})
            return summary

        try:
            critic = run_critic_fit(genai_client, model, level)
            print(json.dumps(critic))
            measured.append(critic)
            summary.update(critic_total_output=critic["total_output"], critic_seconds=critic["seconds"])
            if not default_run:
                summary["critic_thoughts"] = critic["thoughts_tokens"]
                versions.append(critic["model_version"])
            cap_check = run_cap_check(genai_client, model, level)
            print(json.dumps(cap_check))
            measured.append(cap_check)
            if not default_run:
                versions.append(cap_check["model_version"])
        except ProbeFailure:
            # The measurements so far still reach the coordinator; the critic fields stay null
            # when the critic call failed.
            print(json.dumps(finish(summary)))
            raise
        print(json.dumps(finish(summary)))
        if model == HYDROGEN and EMBEDDING_2 not in report["missing"]:
            embedding = run_embedding(genai_client)
            print(json.dumps(embedding))
            if embedding["vectors"] != embedding["texts"]:
                return 1
    except ProbeFailure as failure:
        print(json.dumps(failed_line(failure.phase, failure.error_type, failure.http_status, default_run)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
