"""Shared experiment protocol for all BCP+ baselines + our method.

Includes a `RuntimeContextSettings` re-implementation of Pi's
`RuntimeContextManagementLevel` profiles (`level0` ... `level5`).
The two relevant mechanisms are:

  (A) per-tool-result truncation — each tool's output is clipped to
      `max_tool_result_chars` and a marker string is appended.
  (B) micro-compaction — when accumulated tool-result chars across the
      conversation exceed `micro_compact_min_tool_result_chars`,
      replace the content of every tool_result message older than the
      last `micro_compact_keep_turns` ASSISTANT turns with a
      `[cleared]` placeholder. Zero-LLM-call (in-memory only).

Levels match Pi's spec verbatim. Native DCI uses level3 by default
(paper §4 Implementation Details).

Every headline runner (rise_agent, bcp_retrieval_agent,
piserini_agent, dci_native) constructs a `RunConfig` from its CLI
args + method-specific defaults, then passes that to its run loop.

The contract:

- **`max_model_calls`** counts LLM API calls. In `Agent.run` this maps
  to `max_turns`; in `bcp_retrieval_agent` / `piserini_agent`'s custom
  loops this maps to their `max_iterations`. Different historical
  names; one canonical meaning.

- **Two retry axes** (`api_max_retries`, `run_max_attempts`) address
  two failure modes and compose multiplicatively in the worst case.
  Don't conflate them.

- **`coerce_final_on_max_turns`** is False by default for every
  headline run — fair comparison. Empirically a no-op on RISE dev_100
  (100/100 natural commit), but the principle is explicit. The
  `Agent.__init__` class default stays True (don't break other
  callers); headline runners pass `coerce_final_on_max_turns=False`
  explicitly.

- **Cost attribution** = "final accepted attempt only". Tokens of
  discarded refusal-restarts or transient-error attempts go into
  audit fields (`discarded_attempt_*_tokens`), not the headline
  `agent_cost_usd`. Paper reports headline + a "true API spend ≈
  headline × mean(n_attempts)" caveat.

- **Hash identity bumps** on this refactor: schema 1.0 → 1.1, and the
  `runner` name changes (`single_agent` → `rise`). Old
  `outputs/single_dev100_*` dirs stay as immutable provenance; new
  runs use `outputs/rise_dev100_*`. No automatic cache reuse across
  the cutover.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Bump for this refactor (was "1.0"). See docs/PLAN_refactor_unified_protocol.md.
PROTOCOL_VERSION = "1.1"


@dataclass(frozen=True)
class RunConfig:
    """Shared protocol settings for an experiment run.

    Method-specific defaults are NOT encoded here — they're applied by
    the per-method factory functions below (`for_rise`, `for_dci_native`,
    `for_bcp_retrieval`, `for_piserini`).
    """
    # ---- Method identity --------------------------------------------------
    # Stable identifier used in the per-qid `runner` field and in
    # `compute_run_config_hash`. Renaming a runner bumps cache identity.
    runner: str
    # Optional sub-mode for runners that have multiple tool surfaces
    # (e.g. RISE has doc-mode vs passage-mode).
    runner_mode: str = "default"

    # ---- Agent-loop budget ------------------------------------------------
    # `max_model_calls` counts LLM API calls. Per-method defaults:
    #   DCI native: 300 (paper §4)
    #   RISE headline runs: 100 (passed by scripts/run_rise.py --max-turns)
    #   BCP retrieval-agent: 100 (paper code default)
    #   Pi-Serini: 100
    max_model_calls: int = 300
    # Per-call max output tokens (anti-runaway). 32000 fits ~16k
    # reasoning + ~16k visible output for gpt-5/o-series.
    per_call_max_tokens: int = 32000

    # ---- Wall-clock cap ---------------------------------------------------
    wall_clock_timeout_sec: int = 1 * 60 * 60

    # ---- Coerce policy ----------------------------------------------------
    # Inject "you're out of budget, commit now" turn when
    # max_model_calls hits without a final answer. RunConfig default is
    # False; runners pass this through explicitly. Agent.__init__ class
    # default stays True (other callers may depend on it).
    coerce_final_on_max_turns: bool = False

    # ---- Three independent retry axes -------------------------------------
    # NOT a single number. Each addresses a different failure mode; they
    # compose multiplicatively in the worst case.
    #
    # Inside a single Agent.run(...): how many times to retry a transient
    # API error (5xx, rate-limit, network) on the SAME request. Handled
    # by `api_retry.responses_with_retry`.
    api_max_retries: int = 3
    # Outside Agent.run(...): how many times the runner may re-invoke
    # Agent.run for transient infra failures that survived the two inner
    # retry budgets. Default 1 = no outer retry. Legacy Pi-based DCI had
    # an effective `run_max_attempts` up to 7; native DCI sets this to
    # 1 (paper-faithful).
    run_max_attempts: int = 1

    # ---- Cost attribution -------------------------------------------------
    # Always "final_attempt_only". Tokens of discarded transient-error
    # attempts go into per-qid audit fields (`discarded_attempt_tokens`,
    # `discarded_attempt_input_tokens`, `discarded_attempt_output_tokens`,
    # `discarded_attempt_reasoning_tokens`), recorded but not summed
    # into headline `agent_cost_usd`.
    cost_attribution: str = "final_attempt_only"

    # ---- Model + judge ----------------------------------------------------
    agent_model: str = ""
    judge_model: str = ""
    judge_prompt_id: str = "bcp_appendix_f"

    # ---- Bookkeeping ------------------------------------------------------
    # Per-protocol version. Bump on any incompatible change here.
    protocol_version: str = PROTOCOL_VERSION
    # Per-trajectory schema version (different concept — schema of the
    # per_query.json payload, defined in `trajectory.py`).
    trajectory_schema: str = "1.1"

    # ---- Method-specific knobs (free-form) --------------------------------
    # Anything that affects agent behavior but isn't standardized across
    # methods (e.g. bm25_k for RISE, per_search_k for BCP, search_depth_k
    # for Pi-Serini, context_level for DCI). Always recorded in the
    # hash; runners populate via `extras=...` kwargs.
    extras: dict[str, Any] = field(default_factory=dict)

    def to_hashable_dict(self) -> dict[str, Any]:
        """Stable dict for `compute_run_config_hash`. Sorts deterministically
        downstream. Path objects in `extras` are coerced to str.absolute().

        Judge-related fields (`judge_model`, `judge_prompt_id`) are
        **excluded** from the hashable view: judge is decoupled and runs
        as a separate step via `scripts/judge.py`, so swapping the judge
        model must not invalidate the agent-run cache.
        """
        EXCLUDED = {"judge_model", "judge_prompt_id"}
        out: dict[str, Any] = {}
        for k, v in asdict(self).items():
            if k in EXCLUDED:
                continue
            if k == "extras":
                out[k] = _coerce_paths(v)
            elif isinstance(v, Path):
                out[k] = str(v.absolute())
            else:
                out[k] = v
        return out


# ---- Runtime context-management profiles (Pi-faithful) --------------------
#
# Mirrors `external/dci-agent-lite/pi-mono/packages/coding-agent/src/core/
# settings-manager.ts::RUNTIME_CONTEXT_MANAGEMENT_PROFILES`. Two of the
# fields (`in_loop_compaction`, `in_loop_compaction_failure_limit`) are
# kept for parity but not implemented in the Python Agent class — Pi's
# in-loop compaction triggers a separate LLM call to summarise old
# turns, and level3 has it disabled anyway. We only implement the two
# zero-LLM mechanisms (truncate_tool_results, micro_compact).

@dataclass(frozen=True)
class RuntimeContextSettings:
    """Subset of Pi's RuntimeContextManagementSettings sufficient for
    paper-faithful DCI native + RISE + (later) BCP / Pi-Serini."""
    level: str = "level0"
    truncate_tool_results: bool = False
    max_tool_result_chars: int = 0  # 0 = no truncation
    micro_compact: bool = False
    micro_compact_keep_turns: int = 12
    micro_compact_min_tool_result_chars: int = 240_000


# Pi profile table, verbatim from settings-manager.ts L137-L208.
# Notes: in_loop_compaction (Pi's LLM-based summary) is not implemented
# here; level4 and level5 will degrade silently if used.
RUNTIME_CONTEXT_PROFILES: dict[str, RuntimeContextSettings] = {
    "level0": RuntimeContextSettings(
        level="level0",
        truncate_tool_results=False,
        max_tool_result_chars=0,
        micro_compact=False,
    ),
    "level1": RuntimeContextSettings(
        level="level1",
        truncate_tool_results=True,
        max_tool_result_chars=50_000,
        micro_compact=False,
    ),
    "level2": RuntimeContextSettings(
        level="level2",
        truncate_tool_results=True,
        max_tool_result_chars=20_000,
        micro_compact=False,
    ),
    "level3": RuntimeContextSettings(
        level="level3",
        truncate_tool_results=True,
        max_tool_result_chars=20_000,
        micro_compact=True,
        micro_compact_keep_turns=12,
        micro_compact_min_tool_result_chars=240_000,
    ),
    "level4": RuntimeContextSettings(
        level="level4",
        truncate_tool_results=True,
        max_tool_result_chars=20_000,
        micro_compact=True,
        micro_compact_keep_turns=10,
        micro_compact_min_tool_result_chars=200_000,
    ),
    "level5": RuntimeContextSettings(
        level="level5",
        truncate_tool_results=True,
        max_tool_result_chars=10_000,
        micro_compact=True,
        micro_compact_keep_turns=6,
        micro_compact_min_tool_result_chars=60_000,
    ),
}


def get_runtime_context_settings(level: str = "level0") -> RuntimeContextSettings:
    """Resolve a level name (level0..level5) to its preset."""
    if level not in RUNTIME_CONTEXT_PROFILES:
        raise ValueError(
            f"unknown context-management level {level!r}; "
            f"valid: {sorted(RUNTIME_CONTEXT_PROFILES.keys())}"
        )
    return RUNTIME_CONTEXT_PROFILES[level]


def _coerce_paths(obj: Any) -> Any:
    """Recursively coerce Path → str.absolute() for hash stability."""
    if isinstance(obj, Path):
        return str(obj.absolute())
    if isinstance(obj, dict):
        return {k: _coerce_paths(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_paths(x) for x in obj]
    return obj


# ---- Per-method factory functions -----------------------------------------

def for_rise(
    *,
    agent_model: str,
    judge_model: str = "",
    bm25_k: int = 1000,
    bm25_top_n_preview: int = 10,
    bm25_snippet_chars: int = 100,
    bash_truncate_chars: int = 4000,
    read_default_limit: int = 2000,
    enable_sandbox: bool = False,
    index_dir: Path | str = "",
    filename_map: Path | str = "",
    bc_plus_docs: Path | str = "",
    passage_mode: bool = False,
    passage_index_dir: Path | str = "",
    passage_files_root: Path | str = "",
    passage_map: Path | str = "",
    max_model_calls: int = 300,
    coerce_final_on_max_turns: bool = False,
    reasoning_effort: str = "medium",
    structured_doc_mode: bool = False,
    no_bash: bool = False,
) -> RunConfig:
    """RunConfig for RISE (formerly single-agent)."""
    extras: dict[str, Any] = {
        "bm25_k": bm25_k,
        "bm25_top_n_preview": bm25_top_n_preview,
        "bm25_snippet_chars": bm25_snippet_chars,
        "bash_truncate_chars": bash_truncate_chars,
        "read_default_limit": read_default_limit,
        "enable_sandbox": enable_sandbox,
        "index_dir": str(Path(index_dir).absolute()) if index_dir else "",
        "filename_map": str(Path(filename_map).absolute()) if filename_map else "",
        "bc_plus_docs": str(Path(bc_plus_docs).absolute()) if bc_plus_docs else "",
        "passage_mode": passage_mode,
        "structured_doc_mode": structured_doc_mode,
        "no_bash": no_bash,
        "reasoning_effort": reasoning_effort,
    }
    if passage_mode:
        extras.update({
            "passage_index_dir": str(Path(passage_index_dir).absolute()) if passage_index_dir else "",
            "passage_files_root": str(Path(passage_files_root).absolute()) if passage_files_root else "",
            "passage_map": str(Path(passage_map).absolute()) if passage_map else "",
        })
    return RunConfig(
        runner="rise",
        runner_mode="passage" if passage_mode else "doc",
        max_model_calls=max_model_calls,
        coerce_final_on_max_turns=coerce_final_on_max_turns,
        agent_model=agent_model,
        judge_model=judge_model,
        extras=extras,
    )


def for_dci_native(
    *,
    agent_model: str,
    judge_model: str = "",
    corpus_dir: Path | str,
    filename_map: Path | str,
    # Pi-faithful bash tail-truncate (port of `pi-mono/.../tools/truncate.ts`
    # truncateTail defaults). These limits MUST match Pi's bash tool exactly
    # — agent looping happens with smaller budgets (see qid 376 diagnostic).
    bash_max_output_lines: int = 2000,
    bash_max_output_bytes: int = 50 * 1024,
    read_default_limit: int = 2000,
    max_model_calls: int = 300,
    coerce_final_on_max_turns: bool = False,
    reasoning_effort: str = "medium",
    context_management: str = "none",
    # DCI native gets a longer wall-clock budget (1.5 h vs the 1 h default
    # used by retrieval-agent baselines) because direct-corpus interaction
    # via bash + ripgrep can be I/O-bound on large hits, and the 300-call
    # budget naturally takes longer wall-clock than the 100-call retrieval
    # agents.
    wall_clock_timeout_sec: int = int(1.5 * 60 * 60),
) -> RunConfig:
    """RunConfig for native DCI baseline (Agent class + bash + read,
    no `search`, full corpus root)."""
    return RunConfig(
        runner="dci_native",
        runner_mode="default",
        max_model_calls=max_model_calls,
        coerce_final_on_max_turns=coerce_final_on_max_turns,
        wall_clock_timeout_sec=wall_clock_timeout_sec,
        agent_model=agent_model,
        judge_model=judge_model,
        extras={
            "corpus_dir": str(Path(corpus_dir).absolute()),
            "filename_map": str(Path(filename_map).absolute()),
            "bash_max_output_lines": bash_max_output_lines,
            "bash_max_output_bytes": bash_max_output_bytes,
            "read_default_limit": read_default_limit,
            "reasoning_effort": reasoning_effort,
            # Documents the deviation from Pi's L3 compaction. See
            # docs/PLAN_refactor_unified_protocol.md §A.
            "context_management": context_management,
        },
    )


def for_bcp_retrieval(
    *,
    agent_model: str,
    judge_model: str = "",
    index_dir: Path | str,
    bc_plus_docs: Path | str,
    filename_map: Path | str,
    per_search_k: int = 5,
    snippet_tokens: int = 512,
    enable_get_document: bool = True,
    max_model_calls: int = 100,
    coerce_final_on_max_turns: bool = False,
    reasoning_effort: str = "medium",
) -> RunConfig:
    """RunConfig for BCP retrieval-agent baseline (paper §E).

    Corpus storage: `DocidFsLookup` (disk-on-demand `.txt` reads via
    `bc_plus_docs / relpath`, with `relpath` resolved from `filename_map`).
    Byte-equivalent to the upstream parquet-loaded dict (verified by
    `scripts/test_corpus_storage_equivalence.py` for all 100k docids);
    avoids the ~2-3 GB resident-set hit of holding the whole 100k corpus
    in memory, and makes 1M-corpus runs feasible.
    """
    return RunConfig(
        # Stable identifier — matches historical `runner="bcp_retrieval_agent"`
        # in legacy per_query.json and `expected_runner` cache checks. Don't
        # shorten without updating cache-resume + trace-file `runner` fields.
        runner="bcp_retrieval_agent",
        runner_mode="default",
        max_model_calls=max_model_calls,
        coerce_final_on_max_turns=coerce_final_on_max_turns,
        agent_model=agent_model,
        judge_model=judge_model,
        extras={
            "index_dir": str(Path(index_dir).absolute()),
            "bc_plus_docs": str(Path(bc_plus_docs).absolute()),
            "filename_map": str(Path(filename_map).absolute()),
            "per_search_k": per_search_k,
            "snippet_tokens": snippet_tokens,
            "enable_get_document": enable_get_document,
            "reasoning_effort": reasoning_effort,
        },
    )


def for_piserini(
    *,
    agent_model: str,
    judge_model: str = "",
    index_dir: Path | str,
    bc_plus_docs: Path | str,
    filename_map: Path | str,
    search_depth_k: int = 1000,
    search_first_page: int = 5,
    rsr_default_limit: int = 10,
    read_doc_limit: int = 200,
    max_model_calls: int = 100,
    coerce_final_on_max_turns: bool = False,
    reasoning_effort: str = "medium",
) -> RunConfig:
    """RunConfig for Pi-Serini-style baseline (re-impl).

    Corpus storage: same `DocidFsLookup` swap as `for_bcp_retrieval` — see
    docstring there for rationale and the byte-equivalence test.
    """
    return RunConfig(
        runner="piserini",
        runner_mode="default",
        max_model_calls=max_model_calls,
        coerce_final_on_max_turns=coerce_final_on_max_turns,
        agent_model=agent_model,
        judge_model=judge_model,
        extras={
            "index_dir": str(Path(index_dir).absolute()),
            "bc_plus_docs": str(Path(bc_plus_docs).absolute()),
            "filename_map": str(Path(filename_map).absolute()),
            "search_depth_k": search_depth_k,
            "search_first_page": search_first_page,
            "rsr_default_limit": rsr_default_limit,
            "read_doc_limit": read_doc_limit,
            "reasoning_effort": reasoning_effort,
        },
    )
