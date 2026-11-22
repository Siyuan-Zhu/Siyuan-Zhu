#!/usr/bin/env python3
"""Manual Hydrogen probe: list models, then make up to eight paid calls.

Prints model versions, token counts and timings only. It never prints the key, response
text, or any credential identity. Each call is one SDK attempt, and nothing paid runs for a
model that is not listed. The calls, in order:

- two tiny Hydrogen calls (16 output tokens), text then one 384-pixel image;
- four Director calls, each with max_output_tokens 15000 on the prompt the engine's own
  Director builder makes for synthetic clusters (three at thinking level high, then one at
  medium);
- one critic call: a synthetic batch of nine 384-pixel images in the engine's batch critic
  request shape, at thinking level high with max_output_tokens 15000;
- one gemini-embedding-2 call, only if that model is listed: two short synthetic texts in the
  engine's clustering form at 768 dimensions.

Each call prints one numbers-only line with seconds=<elapsed, one decimal> (total_output =
candidates + thoughts tokens). A summary line after the critic call gives
max_total_output_high, max_total_output_medium, max_seconds_high, critic_total_output and
critic_seconds; the embedding line follows it. When the critic call fails, the summary is still
printed (critic fields null) before the failed line.

Coordinator rule for choosing the Hydrogen output cap (not enforced by code): take the smallest
cap from 4000/6000/8000/10000/15000 that is at least 1.5 times the LARGER of
max_total_output_high and critic_total_output; only if 15000 is still too small, apply the same
rule at medium to max_total_output_medium.

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
EMBEDDING_2 = "gemini-embedding-2"
PROBE_MODELS = (HYDROGEN, EMBEDDING_2)
PROBE_TEXT = "A quiet harbor at dawn with three small boats."
VERSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
CALL_CONFIG = {
    "max_output_tokens": 16,
    "thinking_config": {"thinking_level": "low"},
    "media_resolution": "MEDIA_RESOLUTION_HIGH",
}
COUNT_FIELDS = (
    "prompt_token_count",
    "candidates_token_count",
    "thoughts_token_count",
    "total_token_count",
)


FIT_OUTPUT_TOKENS = 15000
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


def run_thinking_fit(genai_client):
    """One call per (level, output cap) in FIT_PLAN, in order; a failure stops before the next call."""
    try:
        prompt = director_fit_prompt()
    except Exception as error:
        raise ProbeFailure("director_prompt", type(error).__name__) from None
    for index, (level, max_output_tokens) in enumerate(FIT_PLAN, start=1):
        config = {
            "max_output_tokens": max_output_tokens,
            "response_mime_type": "application/json",
            "thinking_config": {"thinking_level": level},
        }
        try:
            response, seconds = _timed(lambda: genai_client.models.generate_content(
                model=HYDROGEN, contents=prompt, config=config
            ))
            row = fit_row(level, max_output_tokens, response)
        except Exception as error:
            raise ProbeFailure(f"thinking_{level}_{index}", type(error).__name__) from None
        yield {**row, "seconds": seconds}


def run_critic_fit(genai_client):
    """One critic call in the engine's batch shape at thinking high; config as the guard sends it."""
    config = {
        "response_mime_type": "application/json",
        "max_output_tokens": FIT_OUTPUT_TOKENS,
        "thinking_config": {"thinking_level": "high"},
        "media_resolution": "MEDIA_RESOLUTION_HIGH",
    }
    try:
        parts = critic_fit_parts()
        response, seconds = _timed(lambda: genai_client.models.generate_content(
            model=HYDROGEN, contents=parts, config=config
        ))
        row = fit_row("high", FIT_OUTPUT_TOKENS, response)
    except Exception as error:
        raise ProbeFailure("critic", type(error).__name__) from None
    return {"call": "critic", "images": CRITIC_IMAGES, **row, "seconds": seconds}


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


def embedding_row(response, seconds):
    vectors = list(getattr(response, "embeddings", None) or [])
    values = [list(getattr(v, "values", None) or []) for v in vectors]
    sizes = {len(v) for v in values}
    finite = bool(values) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        for vector in values for x in vector)
    return {
        "call": "embedding",
        "model": EMBEDDING_2,
        "vectors": len(vectors),
        "dimension": sizes.pop() if len(sizes) == 1 and 0 not in sizes else None,
        "finite": 1 if finite else 0,
        "billed_tokens": _billed_tokens(response),
        "seconds": seconds,
    }


def run_embedding(genai_client):
    """One embed_content call with two synthetic texts in the engine's clustering form."""
    from src.zeitgeist.collectors import EMBEDDING_DIM, _request_text

    try:
        contents = [_request_text(text, EMBEDDING_2) for text in EMBED_TEXTS]
        response, seconds = _timed(lambda: genai_client.models.embed_content(
            model=EMBEDDING_2, contents=contents, config={"output_dimensionality": EMBEDDING_DIM}
        ))
        return embedding_row(response, seconds)
    except Exception as error:
        raise ProbeFailure("embedding", type(error).__name__) from None


def max_total_output(rows, level):
    totals = [r["total_output"] for r in rows
              if r["thinking_level"] == level and r["total_output"] is not None]
    return max(totals) if totals else None


class ProbeFailure(Exception):
    def __init__(self, phase, error_type):
        super().__init__(phase)
        self.phase = phase
        self.error_type = error_type


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


def run_probe(key: str, *, http_client, genai_client) -> dict:
    from google.genai import types

    try:
        found = inspect_gemini(http_client, key, PROBE_MODELS)
    except Exception as error:
        raise ProbeFailure("listing", type(error).__name__) from None
    report = {
        "listed": found["available_required_models"],
        "missing": found["missing_models"],
        "text": None,
        "image": None,
    }
    if HYDROGEN in report["missing"]:
        return report
    try:
        response, seconds = _timed(lambda: genai_client.models.generate_content(
            model=HYDROGEN, contents=PROBE_TEXT, config=CALL_CONFIG
        ))
        report["text"], _ = summarize(response)
        report["text"]["seconds"] = seconds
    except Exception as error:
        raise ProbeFailure("text", type(error).__name__) from None
    try:
        part = types.Part.from_bytes(data=synthetic_image(), mime_type="image/png")
        response, seconds = _timed(lambda: genai_client.models.generate_content(
            model=HYDROGEN, contents=[part, PROBE_TEXT], config=CALL_CONFIG
        ))
        report["image"], usage = summarize(response)
        report["image"]["image_prompt_token_count"] = image_detail_count(usage)
        report["image"]["seconds"] = seconds
    except Exception as error:
        raise ProbeFailure("image", type(error).__name__) from None
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


def main(argv=None, *, env=None, http_factory=None, genai_factory=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-paid-probe", action="store_true")
    args = parser.parse_args(argv)
    env = os.environ if env is None else env
    key = env.get("GEMINI_API_KEY", "")
    if not args.confirm_paid_probe or not key:
        print(json.dumps({"status": "refused"}))
        return 2
    genai_client = _LazyGenai(genai_factory or _genai_client, key)
    try:
        http_client = (http_factory or _http_client)()
        try:
            report = run_probe(key, http_client=http_client, genai_client=genai_client)
        finally:
            close = getattr(http_client, "close", None)
            if close:
                close()
    except ProbeFailure as failure:
        print(json.dumps({"status": "failed", "phase": failure.phase,
                          "error_type": failure.error_type}))
        return 1
    except Exception as error:
        print(json.dumps({"status": "failed", "phase": "listing",
                          "error_type": type(error).__name__}))
        return 1
    if HYDROGEN in report["missing"]:
        print(json.dumps({"status": "hydrogen_not_listed", **report}))
        return 2
    print(json.dumps({"status": "ok", **report, "tokens_per_image": tokens_per_image(report)}))
    try:
        rows = []
        for row in run_thinking_fit(genai_client):
            rows.append(row)
            print(json.dumps(row))
        summary = {"max_total_output_" + level: max_total_output(rows, level)
                   for level in ("high", "medium")}
        high_seconds = [r["seconds"] for r in rows if r["thinking_level"] == "high"]
        summary.update(max_seconds_high=max(high_seconds) if high_seconds else None,
                       critic_total_output=None, critic_seconds=None)
        try:
            critic = run_critic_fit(genai_client)
        except ProbeFailure:
            # The Director measurements still reach the coordinator; the critic fields stay null.
            print(json.dumps(summary))
            raise
        print(json.dumps(critic))
        summary.update(critic_total_output=critic["total_output"], critic_seconds=critic["seconds"])
        print(json.dumps(summary))
        if EMBEDDING_2 not in report["missing"]:
            print(json.dumps(run_embedding(genai_client)))
    except ProbeFailure as failure:
        print(json.dumps({"status": "failed", "phase": failure.phase,
                          "error_type": failure.error_type}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
