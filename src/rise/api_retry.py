"""Retry wrapper for DeepSeek (OpenAI-compatible) API calls.

Handles:
- Transient HTTP errors: 429 (rate limit), 5xx (server), connection / timeout
- DeepSeek-specific 'terminated' / partial-response cases (reasoning mid-stream
  is sometimes cut off — the SDK surfaces this as APIError or as a response
  with empty content + suspicious finish_reason)
- Empty content with `finish_reason='stop'` (treated as transient, retried)

Does NOT retry:
- 400 BadRequest (prompt is bad — deterministic)
- 401/403 (auth) — deterministic
- 404 (model not found) — deterministic
- JSON parse errors from the model — the caller should sanitize/fall back

Usage:
    response = chat_with_retry(
        client,
        model="deepseek-v4-pro",
        messages=[...],
        tools=[...],
        max_tokens=8192,
    )
"""
from __future__ import annotations

import random
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from openai import AuthenticationError as OpenAIAuthenticationError


# Errors that are deterministic failures — DO NOT retry.
NON_RETRIABLE_ERRORS = (
    BadRequestError,
    OpenAIAuthenticationError,
    PermissionDeniedError,
    NotFoundError,
)

# Errors that may be transient — retry with backoff.
RETRIABLE_ERRORS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)


class TerminatedResponseError(Exception):
    """Raised when DeepSeek returns a response that looks truncated/terminated."""


# Provider-aware helpers. Both DeepSeek and Xiaomi MiMo are OpenAI-compatible,
# but their token-usage payload differs:
#   - DeepSeek: usage.prompt_cache_hit_tokens (custom top-level field)
#   - MiMo / OpenAI standard: usage.prompt_tokens_details.cached_tokens
# `reasoning_tokens` is consistent across both: completion_tokens_details.reasoning_tokens.

REASONER_MODELS: frozenset[str] = frozenset({
    "deepseek-v4-pro", "deepseek-reasoner",
    "mimo-v2.5-pro", "mimo-v2.5",
})


def is_reasoner_model(name: str) -> bool:
    """Return True if the given model id is a reasoning model (emits
    `reasoning_content` and benefits from higher `max_tokens`)."""
    if not name:
        return False
    if name in REASONER_MODELS:
        return True
    n = name.lower()
    # gpt-5 series + o-series (o3, o4, o5) are reasoning models.
    if n.startswith("gpt-5") or n.startswith("o3") or n.startswith("o4") or n.startswith("o5"):
        return True
    return False


def uses_max_completion_tokens(name: str) -> bool:
    """gpt-5 / o-series models reject `max_tokens` and require `max_completion_tokens`."""
    if not name:
        return False
    n = name.lower()
    return n.startswith("gpt-5") or n.startswith("o3") or n.startswith("o4") or n.startswith("o5")


def is_responses_api_model(name: str) -> bool:
    """Return True for models that should be driven via the Responses API
    (`client.responses.create(...)`) instead of chat-completions.

    For gpt-5 / o-series the Responses API exposes reasoning summary
    text and supports `previous_response_id` for server-side reasoning-state
    continuity across turns — both unavailable via chat.completions.
    """
    if not name:
        return False
    n = name.lower()
    return n.startswith("gpt-5") or n.startswith("o3") or n.startswith("o4") or n.startswith("o5")


def effective_reasoning_effort(default: str = "medium") -> str:
    """Return the effective reasoning effort for gpt/o-series runs.

    The experiment default is medium; callers can still override with
    REASONING_EFFORT for deliberate ablations.
    """
    import os as _os
    return _os.environ.get("REASONING_EFFORT") or default


def extract_cached_tokens(usage: Any) -> int:
    """Read the cached prompt-token count regardless of provider.

    DeepSeek puts it at `usage.prompt_cache_hit_tokens`. OpenAI/MiMo put it
    at `usage.prompt_tokens_details.cached_tokens`. Try both; return 0 if
    neither field is present or readable.
    """
    if usage is None:
        return 0
    n = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    if n:
        return n
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        try:
            return int(getattr(details, "cached_tokens", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def extract_reasoning_tokens(usage: Any) -> int:
    """Read the reasoning-token count (completion_tokens_details.reasoning_tokens),
    consistent across DeepSeek / OpenAI / MiMo."""
    if usage is None:
        return 0
    d = getattr(usage, "completion_tokens_details", None)
    if d is None:
        return 0
    try:
        return int(getattr(d, "reasoning_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _looks_terminated(response: Any) -> bool:
    """Detect a DeepSeek 'terminated' response.

    Indicators (any of):
    - message.content is empty AND no tool_calls AND finish_reason == 'stop'
      (model returned nothing usable mid-stream)
    - finish_reason explicitly 'error' or contains 'terminated'
    """
    if not response.choices:
        return True
    choice = response.choices[0]
    finish = (choice.finish_reason or "").lower()
    if finish in ("error", "terminated") or "terminated" in finish:
        return True
    msg = choice.message
    content = (msg.content or "").strip()
    has_tool_calls = bool(getattr(msg, "tool_calls", None))
    if not content and not has_tool_calls and finish == "stop":
        return True
    return False


def chat_with_retry(
    client: OpenAI,
    *,
    max_retries: int = 5,
    initial_backoff: float = 1.5,
    backoff_factor: float = 2.0,
    backoff_cap: float = 30.0,
    jitter: float = 0.25,
    **kwargs: Any,
) -> Any:
    """Call client.chat.completions.create with retry on transient errors.

    Backoff schedule (with jitter): 1.5s, 3s, 6s, 12s, 24s (capped at 30).

    Returns the response object on first success; raises the last exception (or
    a TerminatedResponseError) after `max_retries` failed attempts.
    """
    # gpt-5 / o-series reject `max_tokens`; translate to `max_completion_tokens`.
    model_name = kwargs.get("model", "") or ""
    if uses_max_completion_tokens(model_name) and "max_tokens" in kwargs:
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
    # gpt-5 / o-series don't accept temperature/top_p tweaks either, and need
    # an explicit `reasoning_effort` — the API default is `minimal` which disables
    # CoT (reasoning_tokens=0). Default to `medium` for these experiments, override
    # with REASONING_EFFORT env var (one of minimal/low/medium/high/xhigh).
    if uses_max_completion_tokens(model_name):
        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)
        kwargs.setdefault("reasoning_effort", effective_reasoning_effort())

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            if _looks_terminated(response):
                last_exc = TerminatedResponseError(
                    f"response looked terminated/empty on attempt {attempt + 1}; "
                    f"finish_reason={response.choices[0].finish_reason!r}"
                )
            else:
                return response
        except NON_RETRIABLE_ERRORS as e:
            # Deterministic failure — propagate immediately
            raise
        except RETRIABLE_ERRORS as e:
            last_exc = e
        except APIError as e:
            # Other API errors — treat as retriable but be cautious
            last_exc = e
        except Exception as e:
            # Unknown error — log and retry once or twice
            last_exc = e

        # backoff with multiplicative + jitter
        if attempt < max_retries - 1:
            backoff = min(backoff_cap, initial_backoff * (backoff_factor ** attempt))
            backoff *= 1.0 + random.uniform(-jitter, jitter)
            time.sleep(max(0.5, backoff))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"chat_with_retry: {max_retries} attempts failed (no exception captured)")


# ----- Responses API path (gpt-5 / o-series, OpenAI) -----


def _looks_terminated_responses(response: Any) -> bool:
    """Detect a Responses-API truncation/empty response. Returns True if the
    `output` list is empty AND there's no message content / function_calls."""
    out = getattr(response, "output", None) or []
    if not out:
        # No output blocks at all → likely an aborted stream
        return True
    # Accept any non-empty output as valid (reasoning-only blocks are OK on
    # turns where the model immediately calls a tool with no visible content).
    return False


def responses_with_retry(
    client: Any,
    *,
    max_retries: int = 5,
    initial_backoff: float = 1.5,
    backoff_factor: float = 2.0,
    backoff_cap: float = 30.0,
    jitter: float = 0.25,
    **kwargs: Any,
) -> Any:
    """Call `client.responses.create(...)` with retry on transient errors.

    Mirrors `chat_with_retry()` but for the Responses API. Forwards all kwargs
    verbatim; caller is responsible for the request shape (`input`, `tools`,
    `reasoning`, `prompt_cache_key`, `previous_response_id`, etc.).
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.responses.create(**kwargs)
            if _looks_terminated_responses(response):
                last_exc = TerminatedResponseError(
                    f"responses output looked empty on attempt {attempt + 1}"
                )
            else:
                return response
        except NON_RETRIABLE_ERRORS:
            raise
        except RETRIABLE_ERRORS as e:
            last_exc = e
        except APIError as e:
            last_exc = e
        except Exception as e:
            last_exc = e

        if attempt < max_retries - 1:
            backoff = min(backoff_cap, initial_backoff * (backoff_factor ** attempt))
            backoff *= 1.0 + random.uniform(-jitter, jitter)
            time.sleep(max(0.5, backoff))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(
        f"responses_with_retry: {max_retries} attempts failed (no exception captured)"
    )


def extract_responses_usage(usage: Any) -> dict[str, int]:
    """Pull token counts out of a Responses-API `usage` object.

    Responses API: `input_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens`,
    `input_tokens_details.cached_tokens`, `total_tokens`. Returns a dict with the
    same keys used elsewhere in the codebase so cost estimation flows
    unchanged (input_tokens=uncached, cached_input_tokens, output_tokens
    [includes reasoning per OpenAI convention], reasoning_tokens).
    """
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
    input_total = int(getattr(usage, "input_tokens", 0) or 0)
    output_total = int(getattr(usage, "output_tokens", 0) or 0)
    cached = 0
    in_details = getattr(usage, "input_tokens_details", None)
    if in_details is not None:
        cached = int(getattr(in_details, "cached_tokens", 0) or 0)
    reasoning = 0
    out_details = getattr(usage, "output_tokens_details", None)
    if out_details is not None:
        reasoning = int(getattr(out_details, "reasoning_tokens", 0) or 0)
    return {
        "input_tokens": max(0, input_total - cached),  # UNCACHED prompt tokens
        "cached_input_tokens": cached,
        "output_tokens": output_total,  # includes reasoning per OpenAI convention
        "reasoning_tokens": reasoning,
    }
