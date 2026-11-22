"""Run-wide model choice with a refused-before-spend fallback to Flash."""
from __future__ import annotations

import contextlib
import contextvars
import os
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Optional, Tuple, TypeVar

from .spend_guard import SpendBlocked, refused_before_dispatch

T = TypeVar("T")

FLASH = "gemini-3.8-flash"
HYDROGEN = "gemini-4-hydrogen"
EMBED_001 = "gemini-embedding-001"
EMBED_2 = "gemini-embedding-2"

DIRECTOR_ENV = "ZEITGEIST_DIRECTOR_MODEL"
JUDGE_ENV = "ZEITGEIST_JUDGE_MODEL"
EMBEDDING_ENV = "ZEITGEIST_EMBEDDING_MODEL"

_TEXT_MODELS = frozenset({FLASH, HYDROGEN})
_EMBEDDING_MODELS = frozenset({EMBED_001, EMBED_2})
_ROLES = ("director", "judge", "embedding")


@dataclass
class RunModels:
    director: str
    judge: str
    embedding: str
    fallback_fired: bool = False
    unrecognized: bool = False

    def current(self, role: str) -> str:
        if role not in _ROLES:
            raise ValueError("unknown model role")
        return getattr(self, role)

    def call(self, role: str, fn: Callable[[str], T]) -> T:
        """A non-Flash model that fails hands the run to Flash; a block after a provider call propagates."""
        model = self.current(role)
        try:
            return fn(model)
        except Exception as error:
            if role == "embedding" or model == FLASH:
                raise
            if isinstance(error, SpendBlocked) and not refused_before_dispatch(error):
                raise
        self.give_up(role)
        return fn(FLASH)

    def give_up(self, role: str) -> bool:
        """Move the run to Flash when the role is on another model; False when it already is."""
        if role not in ("director", "judge") or self.current(role) == FLASH:
            return False
        self.director = FLASH
        self.judge = FLASH
        self.fallback_fired = True
        return True


def _pick(raw: Optional[str], default: str, allowed: frozenset) -> Tuple[str, bool]:
    value = (raw or "").strip()
    if not value:
        return default, False
    if value in allowed:
        return value, False
    return default, True


def _config_text_model(config: Mapping[str, Any], key: str) -> str:
    guardrails = config.get("budget_guardrails") if isinstance(config, Mapping) else None
    value = guardrails.get(key) if isinstance(guardrails, Mapping) else None
    return value if value in _TEXT_MODELS else FLASH


def resolve_run_models(config: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None) -> RunModels:
    environ = os.environ if environ is None else environ
    director, bad_d = _pick(environ.get(DIRECTOR_ENV), _config_text_model(config, "primary_text_model"), _TEXT_MODELS)
    judge, bad_j = _pick(environ.get(JUDGE_ENV), _config_text_model(config, "multimodal_judge_model"), _TEXT_MODELS)
    embedding, bad_e = _pick(environ.get(EMBEDDING_ENV), EMBED_001, _EMBEDDING_MODELS)
    return RunModels(director, judge, embedding, unrecognized=bad_d or bad_j or bad_e)


_ACTIVE: "contextvars.ContextVar[Optional[RunModels]]" = contextvars.ContextVar("zeitgeist_run_models", default=None)


@contextlib.contextmanager
def activate(run_models: RunModels) -> Iterator[None]:
    token = _ACTIVE.set(run_models)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def active() -> Optional[RunModels]:
    return _ACTIVE.get()


def model_choice_line(run_models: RunModels, slate_count: int) -> str:
    return (
        f"[model_choice] director={run_models.director} judge={run_models.judge} "
        f"embedding={run_models.embedding} fallback={int(run_models.fallback_fired)} "
        f"unrecognized={int(run_models.unrecognized)} slate={slate_count}"
    )
