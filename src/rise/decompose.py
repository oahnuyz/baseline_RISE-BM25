"""Query decomposition via DeepSeek (OpenAI-compatible).

Two prompt styles:

- "entity-guess" (default): instructs the LLM to hypothesize the unknown
  entity from indirect clues and write sub-queries naming candidate entities
  directly. Strong when the entity is in LLM's world knowledge, fails hard on
  obscure BrowseComp targets.
- "crossref": instructs the LLM NOT to guess the entity. Each verifiable
  specific clue in the query (date, place, related person, event) becomes one
  sub-query of rare-anchor keywords. Independent of LLM world knowledge —
  should be more robust on the obscure-entity failure mode but may miss the
  long-tail "synthesized" sub-queries that name the entity directly.

Supports both `deepseek-chat` (fast) and `deepseek-reasoner` (R1, slower CoT).
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from openai import OpenAI

from .api_retry import chat_with_retry


DECOMP_ENTITY_GUESS = """You are decomposing an obfuscated multi-hop search query into sub-queries that can be answered by searching a fixed 100k-document corpus with BM25 (a sparse lexical retriever — it only matches on words that appear verbatim).

The original query describes an entity (person, place, event, etc.) INDIRECTLY via attributes — e.g., "the person who founded X company in 1987 and later did Y." The target documents in the corpus contain the entity's REAL NAME and direct factual statements, NOT the indirect description from the query. BM25 on the raw query therefore retrieves almost nothing useful.

Your task: produce 3-6 sub-queries optimized for BM25 lexical retrieval. Each sub-query should contain rare, specific anchors (proper nouns, dates, place names, event titles, numeric specifics) that are LIKELY to appear VERBATIM in the target documents.

Strategies:
1. ENTITY RESOLUTION (most important): from the indirect description, hypothesize WHO or WHAT the entity is using your world knowledge. Then write sub-queries that name that entity directly. List multiple candidates if uncertain.
2. RARE-ANCHOR TARGETING: identify the rarest, most-specific clues (e.g., "1987 Silicon Valley sustainable farming founder") and combine them.
3. CROSS-REFERENCE: identify supporting facts in the query (other people, places, events, dates mentioned) and search for those — these often appear in the same documents as the answer.

CRITICAL CONSTRAINTS:
- Sub-queries are NOT for a chatbot or LLM. They are for BM25 — keyword soup is fine, full sentences are not needed.
- Avoid generic terms ("year", "event", "person"). Prefer specific terms.
- Each sub-query should be 3-15 words.
- 3-6 sub-queries total.

Return JSON exactly:
{
  "candidate_entities": ["primary guess about the entity", "alternative guesses"],
  "sub_queries": ["sub_query_1", "sub_query_2", "sub_query_3", ...],
  "rationale": "1-2 sentences explaining the strategy you used."
}"""


DECOMP_CROSSREF = """You are decomposing an obfuscated multi-hop search query into BM25 sub-queries.

The query describes an UNKNOWN entity through multiple specific clues — dates, places, related people, events, organizations, numeric specifics. The entity itself is intentionally obscure; trying to guess its identity from the indirect description usually fails. Your job is the OPPOSITE: ignore "who is this?" and instead extract each verifiable clue and turn it into a BM25 sub-query targeting that clue's verbatim form in the corpus.

Strategy (do exactly this):
1. Scan the query and list every SPECIFIC clue: a year, a place name, a related person's name, a specific event title, a number, an organization, a date range. Ignore generic descriptors ("a person", "an event").
2. For each clue, write one sub-query containing rare anchor terms from that clue. Combine ~2-3 anchors per sub-query for selectivity.
3. If two clues plausibly co-occur in the same document, combine them in one sub-query.
4. Aim for 4-8 sub-queries, one per major clue.

DO NOT:
- Guess the entity's identity or name candidate entities.
- Use generic words ("year", "person", "event", "country", "founder").
- Write full sentences. Sub-queries are keyword soup for BM25.

Examples of good sub-queries:
- "Edinburgh Robert Louis Stevenson Jekyll Hyde author"
- "2010 war crimes trial Hague indictment"
- "Susan Sontag Sarajevo siege 1993"

Return JSON exactly:
{
  "candidate_entities": [],
  "sub_queries": ["sub_query_1", "sub_query_2", ...],
  "rationale": "1-2 sentences explaining which clues you targeted."
}
Note: candidate_entities MUST be empty in this mode."""


PROMPT_STYLES = {
    "entity-guess": DECOMP_ENTITY_GUESS,
    "crossref": DECOMP_CROSSREF,
}


def make_client(model: str | None = None) -> OpenAI:
    """Build an OpenAI-compatible client, routed by `model` prefix when given.

    - `mimo-*`     → Xiaomi MiMo (XIAOMI_API_KEY / XIAOMI_BASE_URL)
    - `deepseek-*` → DeepSeek (DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL)
    - `gpt-5*` / `o3*`/`o4*`/`o5*` → OpenAI direct (OPENAI_DIRECT_API_KEY,
                      api.openai.com)
    - otherwise    → first available: Xiaomi → DeepSeek → generic OPENAI_*

    Same OpenAI Python SDK works for all; only key + base URL change.
    Callers that don't yet know the model can pass `model=None` and fall
    through to the default (Xiaomi, if configured).
    """
    m = (model or "").strip().lower()

    def _xiaomi_creds() -> tuple[str | None, str | None]:
        return os.getenv("XIAOMI_API_KEY"), os.getenv("XIAOMI_BASE_URL")

    def _deepseek_creds() -> tuple[str | None, str | None]:
        return os.getenv("DEEPSEEK_API_KEY"), os.getenv("DEEPSEEK_BASE_URL")

    if m.startswith("mimo"):
        api_key, base_url = _xiaomi_creds()
        if not api_key:
            raise RuntimeError("XIAOMI_API_KEY not set; required for mimo-* models")
        return OpenAI(api_key=api_key, base_url=base_url or "https://api.xiaomimimo.com/v1")

    if m.startswith("deepseek"):
        api_key, base_url = _deepseek_creds()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set; required for deepseek-* models")
        return OpenAI(api_key=api_key, base_url=base_url or "https://api.deepseek.com/v1")

    if m.startswith("gpt-5") or m.startswith("o3") or m.startswith("o4") or m.startswith("o5"):
        # OpenAI direct (api.openai.com) for gpt-5 / o-series reasoning models.
        api_key = os.getenv("OPENAI_DIRECT_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_DIRECT_API_KEY required for gpt-5/o-series models"
            )
        return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")

    # Unknown / no model hint — fall through to the first configured provider.
    api_key = (
        os.getenv("XIAOMI_API_KEY")
        or os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "No API key set: expected XIAOMI_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY in .env"
        )
    base_url = (
        os.getenv("XIAOMI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.deepseek.com/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url)





def _extract_json(text: str) -> dict[str, Any]:
    """Pull the JSON object out of `text`. Falls back to regex if needed."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # find outermost { ... }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object found in:\n{text[:500]}")
    return json.loads(match.group(0))


def decompose(
    query: str,
    *,
    model: str = "deepseek-chat",
    prompt_style: str = "entity-guess",
    client: OpenAI | None = None,
) -> dict[str, Any]:
    """Return a dict {candidate_entities, sub_queries, rationale, _usage, _raw}."""
    client = client or make_client()
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt_style: {prompt_style}; choose from {list(PROMPT_STYLES)}")
    system_prompt = PROMPT_STYLES[prompt_style]

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ORIGINAL QUERY:\n{query}"},
        ],
    }
    # Reasoning-capable models need a much larger max_tokens budget because the
    # chain-of-thought tokens count against it (they're in reasoning_content,
    # which we ignore for retrieval but still gets billed).
    from .api_retry import is_reasoner_model
    is_reasoner = is_reasoner_model(model)
    if model == "deepseek-reasoner":
        # legacy R1: no response_format, no temperature
        kwargs["max_tokens"] = 8192
    elif model == "deepseek-v4-pro":
        # v4-pro: supports both, just needs bigger budget
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["temperature"] = 0.3
        kwargs["max_tokens"] = 8192
    else:
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["temperature"] = 0.3
        kwargs["max_tokens"] = 1024

    response = chat_with_retry(client, **kwargs)
    raw = response.choices[0].message.content or "{}"
    data = _extract_json(raw)

    return {
        "candidate_entities": data.get("candidate_entities", []) or [],
        "sub_queries": data.get("sub_queries", []) or [],
        "rationale": data.get("rationale", "") or "",
        "_raw": data,
        "_usage": {
            "input_tokens": getattr(response.usage, "prompt_tokens", None),
            "output_tokens": getattr(response.usage, "completion_tokens", None),
            "reasoning_tokens": getattr(getattr(response.usage, "completion_tokens_details", None), "reasoning_tokens", None) if is_reasoner else None,
        },
    }
