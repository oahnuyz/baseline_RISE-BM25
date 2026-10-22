"""Unified trajectory schema for every agent runner in this repo.

Every baseline (BCP retrieval-agent, Pi-Serini, DCI, our single-agent) writes
the same per-query JSON shape so that:
  - Token usage + cost is computed per turn from the live API response
  - The full assistant content + reasoning_content is preserved per turn
  - Tool args are kept (raw JSON + parsed dict)
  - Tool results are stored compactly — retrieved docids + summary + length,
    NOT the full snippet text. The exception is bash, where the (truncated)
    stdout is small enough to keep verbatim.

The runners themselves do not change their prompts or tool specs — this module
adds persistence on top of the existing loops. See docs/PROGRESS.md for the
trace-schema rationale.

Per-query JSON layout:

    {
      "query_id": ..., "question": "...", "gold_answer": "...",
      "gold_doc_ids": [...], "evidence_doc_ids": [...],
      "final_text": "...",
      "terminated_by": "...", "elapsed_seconds": ...,
      "model": "...",
      "turns": [ TurnTrace dict, ... ],
      "totals": { ... },
      "all_retrieved": [...],
      "judge": { ... },           # filled by runner after agent finishes
      "is_correct": bool,
      "total_cost_usd": float
    }
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .api_retry import extract_cached_tokens, extract_reasoning_tokens
from .dci_artifacts import estimate_cost

# Schema version — bump when the layout changes incompatibly.
# 1.1 (2026-05-19): runner rename (`single_agent` → `rise`), new
# `discarded_attempt_audit` block, new `n_committed_natural` /
# `n_committed_coerced` / `n_no_final` aggregate fields. Old 1.0
# rows in `outputs/single_dev100_*` and `outputs/dci_baseline_*`
# stay valid as immutable provenance but are NOT reusable by
# new runners.
SCHEMA_VERSION = "1.1"


# Placeholder judge fields written by agent-runners that defer judging
# to a separate step (see `scripts/judge.py`). Signal of "unjudged":
# `row["judge"]["reasoning"] == ""`. `judge.py` overwrites these in
# place when it judges the row.
PLACEHOLDER_JUDGE_OUT = {
    "extracted_final_answer": "",
    "reasoning": "",
    "correct": "no",
    "confidence": 0,
    "model": "",
}
PLACEHOLDER_JUDGE_USAGE = {
    "input_tokens": 0,
    "cached_input_tokens": 0,
    "output_tokens": 0,
    "reasoning_tokens": 0,
}


def is_judge_pending(row: dict) -> bool:
    """A per-query row is 'judge pending' if its judge block is the
    placeholder shape — empty `reasoning` field. After `scripts/judge.py`
    runs, the field gets a real explanation and this returns False.
    """
    judge = row.get("judge") or {}
    return (judge.get("reasoning") or "") == ""


def compute_run_config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 (first 12 hex chars) of a run-config dict.

    The config should include every knob that can change agent behavior or
    the corpus the agent sees — model, judge_model, agent prompt id, max_turns,
    bm25_k, corpus_path, index_dir, filename_map, sandbox, etc. Anything that
    isn't in the dict is implicitly "matches by default", so be generous.

    Keys are sorted before serialisation so dict-order doesn't affect the hash;
    `Path` values are coerced to absolute string form (so `./outputs/x` and
    `outputs/x` both stringify the same way). We use `.absolute()` (NOT
    `.resolve()`) so that moving a corpus directory behind a symlink doesn't
    invalidate the cache; the symlink path is what the runner CLI was
    invoked with, and that's what we treat as the canonical identity.
    """
    norm: dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, Path):
            try:
                norm[k] = str(v.absolute())
            except Exception:
                norm[k] = str(v)
        elif v is None:
            norm[k] = None
        else:
            norm[k] = v
    blob = json.dumps(norm, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


@dataclass
class ToolCallTrace:
    id: str = ""
    name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    args_raw: str = ""                 # verbatim JSON string from tool_call.arguments
    result_docids: list[str] = field(default_factory=list)
    result_summary: str = ""           # 1-line human-readable summary
    result_chars: int = 0              # length of the actual payload sent to the model
    result_text: str = ""              # only kept for bash; empty for search/read
    # First 200 chars of the payload that went to the model. Always set
    # (even for search/read where `result_text` is empty by design), so
    # trace audits can see what the model actually received without
    # needing to reconstruct from corpus+docids.
    result_text_preview: str = ""
    result_truncated: bool = False
    duration_seconds: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TurnTrace:
    turn: int = 0
    finish_reason: str = ""
    content: str = ""                  # assistant content (final answer or running prose)
    reasoning_content: str = ""        # CoT, when the provider exposes it
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    # Per-turn token counts (uncached input separated from cached input).
    # By OpenAI convention, `completion_tokens` already includes `reasoning_tokens`.
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "finish_reason": self.finish_reason,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tokens": {
                "prompt": self.prompt_tokens,
                "cached_prompt": self.cached_prompt_tokens,
                "completion": self.completion_tokens,
                "reasoning": self.reasoning_tokens,
            },
            "cost_usd": self.cost_usd,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }


def usage_from_response(response: Any) -> dict[str, int]:
    """Extract token counts from an openai-compatible response.usage object.

    Returns {input_tokens (uncached), cached_input_tokens, output_tokens,
    reasoning_tokens}. All four are >= 0.
    """
    usage = getattr(response, "usage", None) if response is not None else None
    if not usage:
        return {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    prompt_total = int(getattr(usage, "prompt_tokens", 0) or 0)
    cached = extract_cached_tokens(usage)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning = extract_reasoning_tokens(usage)
    return {
        "input_tokens": max(0, prompt_total - cached),
        "cached_input_tokens": cached,
        "output_tokens": completion,
        "reasoning_tokens": reasoning,
    }


def make_turn_from_response(
    *,
    turn_idx: int,
    response: Any,
    model: str,
) -> TurnTrace:
    """Build a TurnTrace from a chat-completion response (tool calls added by caller)."""
    msg = response.choices[0].message
    finish = response.choices[0].finish_reason or ""
    u = usage_from_response(response)
    turn_cost = estimate_cost(u, model)
    return TurnTrace(
        turn=turn_idx,
        finish_reason=finish,
        content=msg.content or "",
        reasoning_content=(getattr(msg, "reasoning_content", "") or ""),
        prompt_tokens=u["input_tokens"],
        cached_prompt_tokens=u["cached_input_tokens"],
        completion_tokens=u["output_tokens"],
        reasoning_tokens=u["reasoning_tokens"],
        cost_usd=turn_cost,
    )


def turns_totals(turns: list[TurnTrace]) -> dict[str, Any]:
    """Aggregate per-turn token/cost numbers into a totals dict."""
    if not turns:
        return {
            "n_turns": 0,
            "tool_call_count": 0,
            "tool_call_breakdown": {},
            "prompt_tokens": 0,
            "cached_prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "agent_cost_usd": 0.0,
        }
    breakdown: dict[str, int] = {}
    n_tool_calls = 0
    for t in turns:
        for tc in t.tool_calls:
            breakdown[tc.name] = breakdown.get(tc.name, 0) + 1
            n_tool_calls += 1
    return {
        "n_turns": len(turns),
        "tool_call_count": n_tool_calls,
        "tool_call_breakdown": breakdown,
        "prompt_tokens": sum(t.prompt_tokens for t in turns),
        "cached_prompt_tokens": sum(t.cached_prompt_tokens for t in turns),
        "completion_tokens": sum(t.completion_tokens for t in turns),
        "reasoning_tokens": sum(t.reasoning_tokens for t in turns),
        "agent_cost_usd": sum(t.cost_usd for t in turns),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to `path` via a `.tmp` intermediate so partial files don't poison resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def cached_row_is_reusable(
    cached: dict[str, Any],
    *,
    expected_schema: str = SCHEMA_VERSION,
    expected_runner: str | None = None,
    expected_model: str | None = None,
    expected_run_config_hash: str | None = None,
) -> tuple[bool, str]:
    """Return (ok, reason) for whether `cached` (a per-query JSON row) can be
    resumed instead of re-running.

    A row is reusable only when its schema version, runner identity, agent
    model, and (most importantly) full `run_config_hash` all match the
    current run's. The hash check is the only thing that catches
    corpus/index/filename-map swaps — model + max_turns alone don't
    distinguish 100k from 1M runs.

    Judge model is **not** part of the reuse check: judge is decoupled
    (see `scripts/judge.py`), so an agent-run cache is reusable across
    judge models.
    """
    if not isinstance(cached, dict):
        return False, "cached payload is not a dict"
    schema = cached.get("schema_version")
    if schema != expected_schema:
        return False, f"schema_version {schema!r} != expected {expected_schema!r}"
    if expected_runner is not None:
        runner = cached.get("runner")
        if runner != expected_runner:
            return False, f"runner {runner!r} != expected {expected_runner!r}"
    if expected_model is not None:
        model = cached.get("model")
        if model != expected_model:
            return False, f"model {model!r} != expected {expected_model!r}"
    if expected_run_config_hash is not None:
        actual_hash = cached.get("run_config_hash")
        if actual_hash != expected_run_config_hash:
            return False, f"run_config_hash {actual_hash!r} != expected {expected_run_config_hash!r}"
    return True, ""


# Tool-name → human-readable summary string templates. Used by callers when
# they want to record a compact summary for traces without storing the full
# tool-result payload.

def search_result_summary(query: str, n_returned: int, n_total: int | None = None) -> str:
    """Short summary for a `search`-style tool call."""
    if n_total is not None and n_total != n_returned:
        return f"query={query!r}: returned {n_returned} of {n_total} cached"
    return f"query={query!r}: returned {n_returned}"


def doc_result_summary(docid: str, n_chars: int | None) -> str:
    if n_chars is None:
        return f"docid={docid!r} not found"
    return f"docid={docid}: {n_chars} chars"


# =============================================================================
# Unified per-query row schema (v1.1).
# =============================================================================
#
# `build_per_query_row` is the SINGLE source of truth for what every runner
# script writes to `outputs/<run_dir>/_per_query/qid_<id>.json`. Every
# baseline (DCI native, RISE, BCP retrieval-agent, Pi-Serini) MUST go
# through this builder so that downstream analysis (RESULTS.md, paper
# tables, ablation diffs) sees byte-identical field names + ordering.
#
# The schema has three layers:
#
#   A. **Required, identical across baselines** — protocol provenance,
#      query, judge, terminated_by, elapsed, retries, cost, coverage.
#      Field names and types are FIXED.
#
#   B. **`tool_usage`** — uniform `{tool_name: count}` map; baselines fill
#      whichever tools they expose. No structural difference across runners.
#
#   C. **`runner_specific`** — free-form dict for per-runner extras that
#      genuinely don't translate cross-baseline (e.g. RISE `bm25_queries`,
#      DCI `bash_kinds`, BCP/Pi-Serini `gold_recall`/`evidence_recall`).
#
# The trajectory itself (per-turn data) lives in a SEPARATE file at
# `_traces/qid_<id>/<runner>.json` — referenced here as `trace_path`
# (relative to the run dir). This keeps per_query.json files small and
# parseable in O(seconds) on dev_100.

# Public list of required top-level keys, for the strict-mode validator.
PER_QUERY_REQUIRED_KEYS = (
    # Provenance
    "schema_version", "protocol_version", "runner", "runner_mode", "model",
    "run_config_hash", "run_config",
    # Query identity
    "query_id", "question", "gold_answer", "gold_doc_ids", "evidence_doc_ids",
    "n_gold", "n_evidence",
    # Outcome
    "final_text", "judge", "is_correct",
    # Coverage (agent-surfaced docs vs gold + evidence)
    "n_surfaced", "surfaced_relpaths", "all_retrieved",
    "coverage_any", "coverage_mean", "coverage_all", "evidence_coverage_mean",
    # Loop accounting (proof of budget enforcement)
    "terminated_by", "elapsed_seconds", "n_turns",
    "n_attempts", "max_calls_hit", "wall_clock_hit",
    # Cost
    "agent_usage", "agent_cost_usd",
    "discarded_attempt_audit",
    "judge_usage", "judge_cost_usd",
    "total_cost_usd",
    # Tool usage + runner-specific extras
    "tool_usage", "runner_specific",
    # Trajectory pointer (separate file)
    "trace_path",
)


def _terminated_by_budget_hit(terminated_by: str) -> tuple[bool, bool]:
    """Return (max_calls_hit, wall_clock_hit) from a `terminated_by` string.

    `max_calls_hit` covers Agent class's "max_turns" + BCP/Pi-Serini's
    "max_iterations" — both mean "the per-query LLM-call budget hit zero
    before the agent committed an answer". `wall_clock_hit` covers
    "wall_clock_timeout" emitted by every loop now.
    """
    tb = (terminated_by or "").lower()
    max_calls_hit = ("max_turns" in tb) or ("max_iterations" in tb) or ("coerced_max_turns" in tb)
    wall_clock_hit = "wall_clock_timeout" in tb
    return max_calls_hit, wall_clock_hit


def build_per_query_row(
    *,
    run_config: Any,  # RunConfig (typed loosely to avoid circular import)
    run_config_hash: str,
    query_id: Any,
    question: str,
    gold_answer: str,
    gold_doc_ids: list[str],
    evidence_doc_ids: list[str],
    final_text: str,
    judge_out: dict[str, Any],
    judge_usage: dict[str, int],
    judge_cost_usd: float,
    # Loop result
    terminated_by: str,
    elapsed_seconds: float,
    n_turns: int,
    n_attempts: int = 1,
    # Cost rollup (agent only — judge is separate)
    agent_usage: dict[str, int],
    agent_cost_usd: float,
    discarded_attempt_audit: dict[str, Any] | None = None,
    # Coverage
    surfaced_relpaths: list[str] | set[str],
    surfaced_docids: set[str],
    all_retrieved: list[str],
    # Tool usage map + per-baseline extras
    tool_usage: dict[str, int],
    runner_specific: dict[str, Any] | None = None,
    # Trajectory pointer (relative to run_dir)
    trace_path: str | Path | None,
) -> dict[str, Any]:
    """Build a per-query JSON row in the unified v1.1 schema.

    Computes derived fields (coverage_any/mean/all, evidence_coverage_mean,
    max_calls_hit, wall_clock_hit, total_cost_usd) from inputs. Caller is
    responsible for `surfaced_docids` (mapped from surfaced_relpaths via
    the run's filename↔docid map) since the map isn't on RunConfig.

    Returns the row dict; caller persists via `atomic_write_json`.
    """
    gold = set(gold_doc_ids or [])
    evi = set(evidence_doc_ids or [])

    n_gold = len(gold)
    n_evidence = len(evi)
    cov_any = float(len(surfaced_docids & gold) >= 1) if gold else 0.0
    cov_mean = (len(surfaced_docids & gold) / n_gold) if gold else 0.0
    cov_all = float(gold.issubset(surfaced_docids)) if gold else 0.0
    ev_cov_mean = (len(surfaced_docids & evi) / n_evidence) if evi else 0.0

    max_calls_hit, wall_clock_hit = _terminated_by_budget_hit(terminated_by or "")

    if discarded_attempt_audit is None:
        discarded_attempt_audit = {
            "tokens": 0, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "cost_usd": 0.0,
        }

    judge_payload = {
        **(judge_out or {}),
        "usage": judge_usage,
        "cost_usd": judge_cost_usd,
    }

    surfaced_list = sorted(surfaced_relpaths) if not isinstance(surfaced_relpaths, list) else list(surfaced_relpaths)

    # Building below; final dict is validated before return to catch any
    # caller-side omission BEFORE the file write — fail-fast at construction.
    row: dict[str, Any] = {
        # Provenance
        "schema_version": SCHEMA_VERSION,
        "protocol_version": getattr(run_config, "protocol_version", ""),
        "runner": run_config.runner,
        "runner_mode": run_config.runner_mode,
        "model": run_config.agent_model,
        "run_config_hash": run_config_hash,
        "run_config": run_config.to_hashable_dict() if hasattr(run_config, "to_hashable_dict") else dict(run_config),
        # Query identity
        "query_id": query_id,
        "question": question,
        "gold_answer": gold_answer,
        "gold_doc_ids": sorted(gold),
        "evidence_doc_ids": sorted(evi),
        "n_gold": n_gold,
        "n_evidence": n_evidence,
        # Outcome
        "final_text": final_text or "",
        "judge": judge_payload,
        "is_correct": (judge_out or {}).get("correct") == "yes",
        # Coverage
        "n_surfaced": len(surfaced_docids),
        "surfaced_relpaths": surfaced_list,
        "all_retrieved": list(all_retrieved or []),
        "coverage_any": cov_any,
        "coverage_mean": cov_mean,
        "coverage_all": cov_all,
        "evidence_coverage_mean": ev_cov_mean,
        # Loop accounting
        "terminated_by": terminated_by or "",
        "elapsed_seconds": float(elapsed_seconds or 0.0),
        "n_turns": int(n_turns or 0),
        "n_attempts": int(n_attempts or 1),
        "max_calls_hit": max_calls_hit,
        "wall_clock_hit": wall_clock_hit,
        # Cost (final-attempt-only headline + discarded audit)
        "agent_usage": dict(agent_usage or {}),
        "agent_cost_usd": float(agent_cost_usd or 0.0),
        "discarded_attempt_audit": dict(discarded_attempt_audit),
        "judge_usage": dict(judge_usage or {}),
        "judge_cost_usd": float(judge_cost_usd or 0.0),
        "total_cost_usd": float((agent_cost_usd or 0.0) + (judge_cost_usd or 0.0)),
        # Tool stats
        "tool_usage": dict(tool_usage or {}),
        "runner_specific": dict(runner_specific or {}),
        # Trajectory pointer
        "trace_path": str(trace_path) if trace_path is not None else "",
    }
    # Fail-fast schema check at construction time. Any runner that
    # forgets a required field or smuggles a wrong-typed value gets a
    # ValueError BEFORE the file is written, instead of producing a
    # silent schema-incompatible per_query.json that pollutes downstream
    # analysis.
    ok, errs = validate_per_query_row(row, strict=True)
    if not ok:
        raise ValueError(
            f"build_per_query_row produced a row that fails strict validation "
            f"for runner={row.get('runner')!r} qid={query_id!r}: {errs}"
        )
    return row


def validate_per_query_row(row: dict[str, Any], *, strict: bool = True) -> tuple[bool, list[str]]:
    """Return (ok, errors). Strict mode requires every key in
    PER_QUERY_REQUIRED_KEYS to exist; non-strict only checks types of
    keys that ARE present."""
    errors: list[str] = []
    if not isinstance(row, dict):
        return False, ["row is not a dict"]
    if strict:
        missing = [k for k in PER_QUERY_REQUIRED_KEYS if k not in row]
        if missing:
            errors.append(f"missing keys: {missing}")
    # Type sanity for a handful of high-value fields.
    for k, typ in (
        ("is_correct", bool),
        ("max_calls_hit", bool),
        ("wall_clock_hit", bool),
        ("coverage_any", (int, float)),
        ("coverage_mean", (int, float)),
        ("agent_cost_usd", (int, float)),
        ("total_cost_usd", (int, float)),
    ):
        if k in row and not isinstance(row[k], typ):
            errors.append(f"key {k!r} has type {type(row[k]).__name__}, expected {typ}")
    return (not errors), errors
