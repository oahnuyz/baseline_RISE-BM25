"""Parse per-query artifacts written by `dci-agent-lite` runs.

Each query produces a directory like `qid_540/` containing:
  - events.jsonl              (turn-by-turn events, including tool_execution_*)
  - conversation_full.json    (messages, each assistant msg has token usage)
  - final.txt, state.json, ...

This module pulls out:
  - tool calls (count + per-tool, plus bash-command kind histogram)
  - token usage (input / output / cacheRead / cacheWrite / total)
  - estimated cost (using PRICE_TABLE)
  - turns, elapsed wall time
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

# DeepSeek pricing per 1M tokens, from https://api-docs.deepseek.com/quick_start/pricing
# (read 2026-05-14). v4-pro is currently at 75% off promo; list prices are
# $1.74 / $0.0145 / $3.48 — recorded below for when promo ends.
#
# Note: `deepseek-chat` and `deepseek-reasoner` are deprecated aliases for
# deepseek-v4-flash non-thinking and thinking mode respectively — same price.
PRICE_TABLE: dict[str, dict[str, float]] = {
    "deepseek-v4-pro":      {"input": 0.435,  "cached_input": 0.003625, "output": 0.87},   # promo (75% off)
    "deepseek-v4-pro-list": {"input": 1.74,   "cached_input": 0.0145,   "output": 3.48},   # list price
    "deepseek-v4-flash":    {"input": 0.14,   "cached_input": 0.0028,   "output": 0.28},
    "deepseek-chat":        {"input": 0.14,   "cached_input": 0.0028,   "output": 0.28},   # = v4-flash non-thinking
    "deepseek-reasoner":    {"input": 0.14,   "cached_input": 0.0028,   "output": 0.28},   # = v4-flash thinking
    # Xiaomi MiMo (api.xiaomimimo.com) — pricing as of 2026-05.
    "mimo-v2.5-pro":        {"input": 1.00,   "cached_input": 0.20,     "output": 3.00},   # flagship agent/coding
    "mimo-v2.5":            {"input": 0.40,   "cached_input": 0.08,     "output": 2.00},   # full-modal agent
    "mimo-v2.5-flash":      {"input": 0.10,   "cached_input": 0.01,     "output": 0.30},   # low-cost text
    # gpt-5 series pricing per 1M tokens (OpenAI, 2026-05).
    "gpt-5.4-mini":         {"input": 0.75,   "cached_input": 0.08,     "output": 4.50},
    # gpt-5.4-nano — user-provided 2026-05-21. ~3.6x cheaper than mini on output.
    "gpt-5.4-nano":         {"input": 0.20,   "cached_input": 0.02,     "output": 1.25},
    # Alias for Pi-side runs (Pi only has `gpt-5-mini` registered; we route to
    # the gpt-5.4-mini model on OpenAI direct, but our
    # cost calc keys on the logical name).
    "gpt-5-mini":           {"input": 0.75,   "cached_input": 0.08,     "output": 4.50},
    "gpt-5-nano":           {"input": 0.20,   "cached_input": 0.02,     "output": 1.25},
    # gpt-5.1 (judge) — pricing per 1M tokens (OpenAI, 2026-05).
    "gpt-5.1":              {"input": 1.25,   "cached_input": 0.13,     "output": 10.00},
    # gpt-5.4 (full) — user-provided pricing 2026-05-28.
    "gpt-5.4":              {"input": 2.50,   "cached_input": 0.25,     "output": 15.00},
    "gpt-5.5":              {"input": 2.50,   "cached_input": 0.25,     "output": 20.00},  # placeholder — fix before using
}


def extract_surfaced_paths(qid_dir: Path, map_keys: set[str]) -> set[str]:
    """Scan a per-query events.jsonl for corpus filenames seen by the agent.

    Returns the set of relative paths (matching `map_keys`) that appeared in
    any `tool_execution_{start,end}` event — either as a bash command argument
    or in the text of a tool result. Line-by-line parsing handles spaces in
    filenames.
    """
    ev_path = Path(qid_dir) / "events.jsonl"
    if not ev_path.exists():
        return set()
    surfaced: set[str] = set()
    bc_marker = "bc_plus_docs/"
    for line in ev_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except Exception:
            continue
        if e.get("type") not in ("tool_execution_start", "tool_execution_end"):
            continue
        blobs: list[str] = []
        args = e.get("args") or {}
        if isinstance(args, dict):
            for v in args.values():
                if isinstance(v, str):
                    blobs.append(v)
        result = e.get("result")
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, str):
                blobs.append(content)
            elif isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict):
                        t = blk.get("text") or blk.get("content") or ""
                        if isinstance(t, str):
                            blobs.append(t)
                    elif isinstance(blk, str):
                        blobs.append(blk)
        for blob in blobs:
            if not blob or ".txt" not in blob:
                continue
            for ln in blob.split("\n"):
                ls = ln.strip()
                if ".txt" not in ls:
                    continue
                idx = ls.find(bc_marker)
                if idx >= 0:
                    cand = ls[idx + len(bc_marker):]
                    ti = cand.find(".txt")
                    if ti >= 0:
                        cand = cand[: ti + 4]
                    if cand in map_keys:
                        surfaced.add(cand)
                else:
                    if ls in map_keys:
                        surfaced.add(ls)
                        continue
                    ti = ls.find(".txt")
                    if ti >= 0:
                        cand = ls[: ti + 4]
                        if cand in map_keys:
                            surfaced.add(cand)
    return surfaced


def _classify_bash(cmd: str) -> str:
    """Return a coarse kind for the bash command — drop leading `cd X &&`."""
    if not cmd:
        return "empty"
    s = cmd.strip()
    s = re.sub(r"^cd\s+\S+\s*(?:&&|;)\s*", "", s, count=1)
    s = s.lstrip()
    if not s:
        return "cd-only"
    first = s.split()[0]
    # normalize a few synonyms
    return {
        "rg": "rg",
        "grep": "grep",
        "find": "find",
        "ls": "ls",
        "cat": "cat",
        "head": "head/tail",
        "tail": "head/tail",
        "sed": "sed",
        "awk": "awk",
        "wc": "wc",
    }.get(first, first)


_UNPRICED_MODEL_WARNED: set[str] = set()


def estimate_cost(usage: dict[str, int], model: str) -> float:
    prices = PRICE_TABLE.get(model)
    if not prices:
        if model and model not in _UNPRICED_MODEL_WARNED:
            _UNPRICED_MODEL_WARNED.add(model)
            print(
                f"[estimate_cost] WARNING: no pricing for model {model!r} in PRICE_TABLE; "
                f"cost will be reported as $0. Add an entry to src/rise/dci_artifacts.py.",
                flush=True,
            )
        return 0.0
    in_tok = usage.get("input_tokens", 0)
    cache_tok = usage.get("cached_input_tokens", 0)
    # `output_tokens` already includes reasoning tokens by OpenAI convention
    # (see api_retry.py:336 / trajectory.py:143). `reasoning_tokens` is
    # stored separately for analysis but must NOT be added to the cost
    # formula or it would be double-counted.
    out_tok = usage.get("output_tokens", 0)
    return (
        in_tok / 1_000_000 * prices["input"]
        + cache_tok / 1_000_000 * prices["cached_input"]
        + out_tok / 1_000_000 * prices["output"]
    )


def parse_qid_artifacts(qid_dir: Path) -> dict[str, Any]:
    """Read all parseable info out of a single per-query DCI artifact dir."""
    qid_dir = Path(qid_dir)

    state = {}
    if (qid_dir / "state.json").exists():
        try:
            state = json.loads((qid_dir / "state.json").read_text(encoding="utf-8"))
        except Exception:
            state = {}

    final_text = ""
    if (qid_dir / "final.txt").exists():
        final_text = (qid_dir / "final.txt").read_text(encoding="utf-8").strip()

    # --- token usage from conversation_full.json assistant messages ---
    usage = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "assistant_messages": 0,
    }
    model = ""
    if (qid_dir / "conversation_full.json").exists():
        conv = json.loads((qid_dir / "conversation_full.json").read_text(encoding="utf-8"))
        model = conv.get("model", "")
        for m in conv.get("messages", []):
            u = m.get("usage")
            if not u:
                continue
            usage["input_tokens"] += int(u.get("input", 0) or 0)
            usage["cached_input_tokens"] += int(u.get("cacheRead", 0) or 0)
            usage["output_tokens"] += int(u.get("output", 0) or 0)
            usage["total_tokens"] += int(u.get("totalTokens", 0) or 0)
            usage["assistant_messages"] += 1

    # --- tool calls from events.jsonl ---
    tool_counts: Counter[str] = Counter()
    bash_kinds: Counter[str] = Counter()
    n_turns = 0
    started_at = None
    finished_at = None
    if (qid_dir / "events.jsonl").exists():
        for line in (qid_dir / "events.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            et = e.get("type", "")
            if et == "tool_execution_start":
                name = e.get("toolName") or "?"
                tool_counts[name] += 1
                if name == "bash":
                    args = e.get("args") or {}
                    cmd = args.get("command") or args.get("cmd") or ""
                    bash_kinds[_classify_bash(cmd)] += 1
            elif et == "turn_start":
                n_turns += 1
            elif et == "agent_start":
                started_at = e.get("timestamp")
            elif et == "agent_end":
                finished_at = e.get("timestamp")

    elapsed = 0.0
    if started_at and finished_at:
        from datetime import datetime
        try:
            t0 = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
            elapsed = max(0.0, (t1 - t0).total_seconds())
        except Exception:
            pass

    return {
        "model": model or (state.get("model") if isinstance(state, dict) else ""),
        "status": state.get("last_event_type") or state.get("status") or "",
        "final_answer": final_text,
        "n_turns": n_turns,
        "tool_calls": {
            "total": sum(tool_counts.values()),
            "by_tool": dict(tool_counts),
            "bash_kinds": dict(bash_kinds),
        },
        "tokens": usage,
        "elapsed_seconds": elapsed,
    }


# Markers used to recognise corpus file paths inside Pi tool outputs / args.
_BC_CORPUS_MARKERS = ("bc_plus_docs/", "bcp_plus_fineweb_1m_docs/")


def _docids_from_blobs(blobs: list[str], map_keys: set[str], relpath_to_docid: dict[str, str]) -> list[str]:
    """Extract corpus docids appearing in a bag of free-text strings.

    Walks each blob line-by-line, recognises both bare relpaths (`<domain>/<title>.txt`)
    and absolute paths under a `bc_plus_docs/` or distractor `_docs/` root.
    Returns docids in insertion order (deduped).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for blob in blobs:
        if not blob or ".txt" not in blob:
            continue
        for ln in blob.split("\n"):
            ls = ln.strip()
            if ".txt" not in ls:
                continue
            cand: str | None = None
            for marker in _BC_CORPUS_MARKERS:
                idx = ls.find(marker)
                if idx >= 0:
                    c = ls[idx + len(marker):]
                    ti = c.find(".txt")
                    if ti >= 0:
                        cand = c[: ti + 4]
                    break
            if cand is None:
                if ls in map_keys:
                    cand = ls
                else:
                    ti = ls.find(".txt")
                    if ti >= 0:
                        c = ls[: ti + 4]
                        if c in map_keys:
                            cand = c
            if cand and cand in relpath_to_docid:
                did = relpath_to_docid[cand]
                if did not in seen:
                    seen.add(did)
                    ordered.append(did)
    return ordered


def _summarize_tool_result(tool_name: str, args: dict, result_text: str, result_docids: list[str]) -> str:
    if tool_name == "bash":
        cmd = (args.get("command") or args.get("cmd") or "").strip().replace("\n", " ")[:80]
        return f"bash {cmd!r}: {len(result_text)} chars, {len(result_docids)} doc-hits"
    if tool_name == "read":
        fp = (args.get("path") or args.get("file_path") or "").strip()
        return f"read {fp!r}: {len(result_text)} chars"
    return f"{tool_name}: {len(result_text)} chars, {len(result_docids)} doc-hits"


def pi_artifacts_to_trajectory(
    qid_dir: Path,
    *,
    model: str,
    relpath_to_docid: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Convert a Pi/dci-agent-lite per-query artifact dir to the unified
    trajectory schema (turns[] + totals) used by every other runner.

    Pi's `conversation_full.json` carries the per-message usage; its
    `events.jsonl` carries tool inputs (args) and outputs (result.content
    text blobs). We join the two by sequence: each assistant message is one
    turn; tool messages immediately following an assistant message are its
    tool results.

    `relpath_to_docid` is optional — when provided, we replace verbose tool
    result text with the list of corpus docids that appeared in the result
    (matching what the other runners store).
    """
    qid_dir = Path(qid_dir)
    map_keys = set(relpath_to_docid.keys()) if relpath_to_docid else set()

    # 1. Index Pi events by tool_call_id (start/end pairs).
    starts_by_call: dict[str, dict[str, Any]] = {}
    ends_by_call: dict[str, dict[str, Any]] = {}
    ev_path = qid_dir / "events.jsonl"
    if ev_path.exists():
        for line in ev_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            et = e.get("type", "")
            if et not in ("tool_execution_start", "tool_execution_end"):
                continue
            call_id = e.get("toolCallId") or e.get("tool_call_id") or e.get("id") or ""
            if not call_id:
                continue
            (starts_by_call if et == "tool_execution_start" else ends_by_call)[call_id] = e

    # 2. Walk conversation_full.json, build a turn per assistant message.
    conv_path = qid_dir / "conversation_full.json"
    turns: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    if conv_path.exists():
        try:
            conv = json.loads(conv_path.read_text(encoding="utf-8"))
            messages = list(conv.get("messages") or [])
            if not model:
                model = conv.get("model", "") or model
        except Exception:
            messages = []

    # Index tool messages by tool_call_id for O(1) result lookup.
    # Pi uses role='toolResult' + toolCallId; OpenAI uses role='tool' +
    # tool_call_id. Accept either.
    tool_msg_by_id: dict[str, dict[str, Any]] = {}
    for m in messages:
        if m.get("role") in ("tool", "toolResult"):
            tid = m.get("toolCallId") or m.get("tool_call_id") or ""
            if tid:
                tool_msg_by_id[tid] = m

    def _tool_result_text(call_id: str) -> str:
        """Best-effort recovery of the text Pi handed back to the model for a tool call."""
        # Prefer tool_execution_end.result.content[].text
        e = ends_by_call.get(call_id)
        if e:
            result = e.get("result") or {}
            content = result.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for blk in content:
                    if isinstance(blk, dict):
                        t = blk.get("text") or blk.get("content") or ""
                        if isinstance(t, str):
                            parts.append(t)
                    elif isinstance(blk, str):
                        parts.append(blk)
                return "\n".join(parts)
        # Fall back to the tool-role message body in conversation_full.json
        tm = tool_msg_by_id.get(call_id)
        if tm:
            c = tm.get("content")
            if isinstance(c, str):
                return c
            if isinstance(c, list):
                parts = []
                for blk in c:
                    if isinstance(blk, dict):
                        t = blk.get("text") or blk.get("content") or ""
                        if isinstance(t, str):
                            parts.append(t)
                    elif isinstance(blk, str):
                        parts.append(blk)
                return "\n".join(parts)
        return ""

    final_text = ""
    final_path = qid_dir / "final.txt"
    if final_path.exists():
        final_text = final_path.read_text(encoding="utf-8").strip()

    turn_idx = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        u = m.get("usage") or {}
        input_t = int(u.get("input", 0) or 0)
        cached_t = int(u.get("cacheRead", 0) or 0)
        output_t = int(u.get("output", 0) or 0)
        # Pi's canonical Usage semantics already store `input` as uncached
        # input tokens and `cacheRead` separately. Do not subtract cacheRead
        # again here; OpenAI/Responses-specific total-input normalization
        # happens inside Pi's provider adapter before conversation_full.json
        # is written.
        non_cached_in = input_t
        turn_cost = estimate_cost(
            {"input_tokens": non_cached_in, "cached_input_tokens": cached_t, "output_tokens": output_t},
            model,
        )

        # Pi stores assistant content as a list of typed blocks:
        #   {type: 'text', text: '...'}        — visible content
        #   {type: 'thinking', thinking: '...'} — reasoning text (CoT)
        #   {type: 'toolCall', name, arguments, id, ...}
        # This is NOT the OpenAI shape (m.tool_calls / m.reasoning_content);
        # we have to walk the content blocks to pull each out.
        content_field = m.get("content")
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_call_blocks: list[dict] = []
        if isinstance(content_field, list):
            for blk in content_field:
                if not isinstance(blk, dict):
                    continue
                btype = blk.get("type", "")
                if btype == "text":
                    t = blk.get("text") or ""
                    if isinstance(t, str):
                        text_parts.append(t)
                elif btype == "thinking":
                    th = blk.get("thinking") or ""
                    if isinstance(th, str):
                        thinking_parts.append(th)
                elif btype in ("toolCall", "tool_use", "tool_call"):
                    tool_call_blocks.append(blk)
        elif content_field is not None:
            text_parts.append(str(content_field))
        assistant_text = "\n".join(text_parts)
        # Reasoning text: prefer in-content `thinking` blocks; fall back to
        # legacy top-level fields for safety.
        reasoning_text = "\n".join(thinking_parts) or (
            m.get("reasoningContent") or m.get("reasoning_content") or ""
        )

        tcalls_meta: list[dict[str, Any]] = []
        # Two shapes possible: in-content `toolCall` blocks (Pi default) OR a
        # top-level `tool_calls`/`toolCalls` list (OpenAI / future Pi). Try both.
        oa_tool_calls = m.get("tool_calls") or m.get("toolCalls") or []
        tc_iter = tool_call_blocks if tool_call_blocks else oa_tool_calls
        for tc in tc_iter:
            # Pi's `toolCall` block: name, arguments (dict), id
            # OpenAI's `tool_call`: id, function: {name, arguments (JSON string)}
            call_id = tc.get("id") or tc.get("toolCallId") or tc.get("tool_call_id") or ""
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            args_raw = fn.get("arguments") if fn else tc.get("arguments")
            if args_raw is None:
                args_raw = ""
            if isinstance(args_raw, dict):
                # Pi: arguments is already a parsed dict.
                args = dict(args_raw)
                args_raw_str = json.dumps(args_raw, ensure_ascii=False)
            else:
                args_raw_str = str(args_raw)
                try:
                    args = json.loads(args_raw_str or "{}")
                except Exception:
                    args = {}
            args_raw = args_raw_str
            result_text = _tool_result_text(call_id)
            result_docids: list[str] = []
            if relpath_to_docid:
                blobs: list[str] = [result_text]
                # also scan the tool input args for surfaced paths
                for v in (args or {}).values():
                    if isinstance(v, str):
                        blobs.append(v)
                result_docids = _docids_from_blobs(blobs, map_keys, relpath_to_docid)
            # Decide what to keep verbatim. Pi exposes only bash/read in DCI baseline;
            # bash output is small enough to keep. For other tools, drop the verbose
            # body and rely on result_docids + summary.
            kept_text = result_text if name == "bash" else ""
            tcalls_meta.append({
                "id": call_id,
                "name": name,
                "args": args,
                "args_raw": args_raw if isinstance(args_raw, str) else json.dumps(args_raw, ensure_ascii=False),
                "result_docids": result_docids,
                "result_summary": _summarize_tool_result(name, args, result_text, result_docids),
                "result_chars": len(result_text),
                "result_text": kept_text,
                "result_truncated": False,
                "duration_seconds": 0.0,
                "error": "",
            })

        turns.append({
            "turn": turn_idx,
            "finish_reason": str(m.get("stopReason") or m.get("finish_reason") or ""),
            "content": assistant_text,
            "reasoning_content": reasoning_text,
            "tokens": {
                "prompt": non_cached_in,
                "cached_prompt": cached_t,
                "completion": output_t,
                "reasoning": 0,  # Pi doesn't break this out
            },
            "cost_usd": turn_cost,
            "tool_calls": tcalls_meta,
        })
        turn_idx += 1

    # Rollup
    total_input = sum(t["tokens"]["prompt"] for t in turns)
    total_cached = sum(t["tokens"]["cached_prompt"] for t in turns)
    total_output = sum(t["tokens"]["completion"] for t in turns)
    total_cost = sum(t["cost_usd"] for t in turns)
    breakdown: dict[str, int] = {}
    tool_total = 0
    all_retrieved: list[str] = []
    seen_ret: set[str] = set()
    for t in turns:
        for tc in t["tool_calls"]:
            breakdown[tc["name"]] = breakdown.get(tc["name"], 0) + 1
            tool_total += 1
            for did in tc.get("result_docids", []):
                if did not in seen_ret:
                    seen_ret.add(did)
                    all_retrieved.append(did)

    return {
        "schema_version": "1.0",
        "runner": "dci_baseline",
        "model": model,
        "final_text": final_text,
        "turns": turns,
        "totals": {
            "n_turns": len(turns),
            "tool_call_count": tool_total,
            "tool_call_breakdown": breakdown,
            "prompt_tokens": total_input,
            "cached_prompt_tokens": total_cached,
            "completion_tokens": total_output,
            "reasoning_tokens": 0,  # not tracked separately by Pi
            "agent_cost_usd": total_cost,
        },
        "all_retrieved": all_retrieved,
    }


def aggregate_artifacts(out_root: Path, model: str | None = None) -> dict[str, Any]:
    """Walk an output_root (with qid_*/ subdirs) and aggregate per-query parses."""
    out_root = Path(out_root)
    qids = sorted([d for d in out_root.iterdir() if d.is_dir() and d.name.startswith("qid_")])
    per_query: list[dict[str, Any]] = []
    for qd in qids:
        qid = qd.name.removeprefix("qid_")
        parsed = parse_qid_artifacts(qd)
        if model and parsed["model"]:
            cost = estimate_cost(parsed["tokens"], parsed["model"])
        else:
            cost = estimate_cost(parsed["tokens"], model or parsed.get("model", ""))
        parsed["query_id"] = qid
        parsed["estimated_cost_usd"] = cost
        per_query.append(parsed)

    # aggregate
    n = max(1, len(per_query))
    agg_tokens = {k: sum(r["tokens"].get(k, 0) for r in per_query) for k in [
        "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens",
    ]}
    agg_tools_total = sum(r["tool_calls"]["total"] for r in per_query)
    agg_by_tool: Counter[str] = Counter()
    agg_bash_kinds: Counter[str] = Counter()
    for r in per_query:
        for k, v in r["tool_calls"]["by_tool"].items():
            agg_by_tool[k] += v
        for k, v in r["tool_calls"]["bash_kinds"].items():
            agg_bash_kinds[k] += v
    total_cost = sum(r["estimated_cost_usd"] for r in per_query)

    return {
        "per_query": per_query,
        "aggregate": {
            "n": len(per_query),
            "mean_n_turns": sum(r["n_turns"] for r in per_query) / n,
            "mean_tool_calls": agg_tools_total / n,
            "tool_calls_total": agg_tools_total,
            "tool_calls_by_tool": dict(agg_by_tool),
            "bash_kinds_total": dict(agg_bash_kinds),
            "tokens_total": agg_tokens,
            "tokens_mean_per_query": {k: v / n for k, v in agg_tokens.items()},
            "total_cost_usd": total_cost,
            "mean_cost_per_query_usd": total_cost / n,
        },
    }
