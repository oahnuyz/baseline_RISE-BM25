"""Generic LLM agent loop (OpenAI-compatible, used with DeepSeek).

Tool-calling chat-completions loop with hard turn cap, API retry (5x exp.
backoff via api_retry.chat_with_retry), and full trace logging. The same
Agent class is used for both layers of the hierarchical setup — only the
system prompt and tool registry differ.

The trace JSON written per Agent.run() contains every turn: assistant
content/reasoning, finish_reason, per-turn tokens + cost, and every tool
call with args + result + duration. Totals are computed at the end.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from .api_retry import (
    chat_with_retry,
    extract_cached_tokens,
    extract_reasoning_tokens,
    extract_responses_usage,
    effective_reasoning_effort,
    is_responses_api_model,
    responses_with_retry,
)
from .dci_artifacts import estimate_cost
from .trajectory import SCHEMA_VERSION


Tool = Callable[[dict], str]


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: str
    result_chars: int
    # First 200 chars of `result` — duplicated for easy trace browsing
    # without loading the (potentially KB-sized) full `result` field.
    result_text_preview: str = ""
    truncated: bool = False
    duration_seconds: float = 0.0
    error: str = ""


@dataclass
class TurnRecord:
    turn: int
    content: str = ""
    reasoning_content: str = ""
    finish_reason: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class AgentRun:
    model: str
    system_prompt: str
    user_prompt: str
    final_answer: str = ""
    terminated_reason: str = ""  # done | max_turns | coerced_max_turns | coerce_failed | api_error
    error: str = ""
    turns: list[TurnRecord] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_cached_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_cost_usd: float = 0.0
    tool_call_count: int = 0
    tool_call_breakdown: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    # Pi-faithful L3 instrumentation: how many times micro_compact fired
    # mid-run, and how many tool results got per-call truncated. Used to
    # verify the context-management implementation is actually engaging.
    n_micro_compactions: int = 0
    n_tool_results_truncated: int = 0


class Agent:
    """Single-threaded tool-calling agent.

    Construct with a model, tool registry (name → callable), and tool specs
    (OpenAI function-calling JSON schemas). Call .run(system, user) to get
    a complete AgentRun. The loop terminates when the model emits a message
    with no tool calls (treated as the final answer) or hits max_turns.
    """

    DEFAULT_COERCE_MSG = (
        "Your tool budget has been exhausted. Based ONLY on the information you "
        "have already gathered from previous tool calls, produce your final "
        "answer now. Do NOT call any more tools — output the answer directly as "
        "text. If the evidence is insufficient, say so explicitly."
    )

    def __init__(
        self,
        *,
        client: OpenAI,
        model: str,
        tools: dict[str, Tool],
        tool_specs: list[dict],
        max_turns: int,
        per_call_max_tokens: int = 32000,
        temperature: float = 0.7,
        max_retries: int = 5,
        coerce_final_on_max_turns: bool = True,
        coerce_message: str | None = None,
        wall_clock_timeout_sec: float = 120 * 60,  # 2 hours per agent run
        # Pi-faithful runtime context-management level. "level0" = no
        # truncation, no compaction (default; matches RISE's historical
        # behavior). "level3" = DCI paper's default (truncate tool
        # results to 20k chars per call, micro-compact when total tool
        # result chars > 240k). See src/rise/protocol.py.
        context_level: str = "level0",
    ) -> None:
        self.client = client
        self.model = model
        self.tools = tools
        self.tool_specs = tool_specs
        self.max_turns = max_turns
        self.per_call_max_tokens = per_call_max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.wall_clock_timeout_sec = wall_clock_timeout_sec
        self.coerce_final_on_max_turns = coerce_final_on_max_turns
        self.coerce_message = coerce_message or self.DEFAULT_COERCE_MSG
        from .protocol import get_runtime_context_settings
        self.context_settings = get_runtime_context_settings(context_level)

    def _truncate_tool_result_text(self, text: str, run: AgentRun | None = None) -> str:
        """Pi level3 truncate_tool_results: clip to max_tool_result_chars
        and append a `[...truncated, N chars omitted]` marker. No-op if
        truncate_tool_results is False or text is short enough. When
        truncation fires, optionally bumps `run.n_tool_results_truncated`.
        """
        cs = self.context_settings
        if not cs.truncate_tool_results or cs.max_tool_result_chars <= 0:
            return text
        if not text or len(text) <= cs.max_tool_result_chars:
            return text
        omitted = len(text) - cs.max_tool_result_chars
        if run is not None:
            run.n_tool_results_truncated += 1
        return text[: cs.max_tool_result_chars] + f"\n[...truncated, {omitted} chars omitted]"

    def _maybe_micro_compact_responses(
        self,
        mirror: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Pi level3 micro_compact for Responses API local mirror.

        `mirror` is the conversation built so far (system + user + per-turn
        function_call / function_call_output items). Each
        function_call_output's `output` field carries the tool result text.

        If the sum of those output strings exceeds the threshold, replace
        the `output` of every function_call_output whose function_call lies
        before the (keep_recent_turns)-th-from-end assistant boundary with
        `[cleared]`. Returns (new_mirror, did_compact).

        Zero LLM calls (mirrors Pi's micro_compact.ts:14 verbatim).
        """
        cs = self.context_settings
        if not cs.micro_compact:
            return mirror, False
        # Estimate total tool-result chars.
        total = 0
        for m in mirror:
            if m.get("type") == "function_call_output":
                out = m.get("output", "")
                if isinstance(out, str):
                    total += len(out)
        if total <= cs.micro_compact_min_tool_result_chars:
            return mirror, False
        # Walk backwards counting "assistant turns" (= function_call items
        # in Responses API parlance, since each turn emits one or more
        # function_call items). To match Pi's `messages[i].role === "assistant"`
        # count semantics, we treat each consecutive run of function_call
        # items (within one turn) as ONE assistant turn — collapse them.
        # Simplest approximation: count distinct call_ids as "tool calls"
        # and each turn ends when we transition from function_call to
        # function_call_output (or system→user→function_call boundary).
        # For Pi-faithful behavior we use a simpler proxy: count
        # function_call_output items as one-per-turn (each output is the
        # result for one assistant tool_call, and we keep last
        # keep_recent_turns of them).
        keep = cs.micro_compact_keep_turns
        # Find indices of function_call items (one per agent turn's
        # tool_call); cutoff = position immediately AFTER the
        # (keep+1)-th-from-end function_call.
        fc_indices: list[int] = [i for i, m in enumerate(mirror) if m.get("type") == "function_call"]
        if len(fc_indices) <= keep:
            return mirror, False
        cutoff_fc = fc_indices[-(keep + 1)]
        cutoff_idx = cutoff_fc + 1  # everything before cutoff_idx gets cleared
        # Clear function_call_output content for indices < cutoff_idx.
        changed = False
        new_mirror: list[dict[str, Any]] = []
        for i, m in enumerate(mirror):
            if i < cutoff_idx and m.get("type") == "function_call_output":
                out = m.get("output", "")
                if out != "[cleared]":
                    nm = dict(m)
                    nm["output"] = "[cleared]"
                    new_mirror.append(nm)
                    changed = True
                    continue
            new_mirror.append(m)
        return new_mirror, changed

    def run(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        trace_path: Path | None = None,
        trace_relpath_to_docid: dict[str, str] | None = None,
    ) -> AgentRun:
        # gpt-5 / o-series → Responses API (reasoning summary visible,
        # previous_response_id chains server-side reasoning state across turns,
        # explicit prompt_cache_key). All other providers (DeepSeek, Mimo) stay
        # on chat.completions.
        if is_responses_api_model(self.model):
            # gpt-5 / o-series use the Responses API (reasoning summary visible,
            # previous_response_id chains server-side reasoning state across
            # turns, explicit prompt_cache_key).
            run = AgentRun(
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            t0 = time.time()
            self._run_responses_loop(run, system_prompt, user_prompt, t0)
            run.elapsed_seconds = time.time() - t0
            if trace_path is not None:
                _dump_trace(run, trace_path, relpath_to_docid=trace_relpath_to_docid)
            return run

        run = AgentRun(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        t0 = time.time()

        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        terminated = False
        for turn_idx in range(self.max_turns):
            # Wall-clock cap: bound the worst case (a single query shouldn't
            # ever take longer than `wall_clock_timeout_sec`). If exceeded
            # before the next LLM call, break out and let the coerce path
            # below try to produce a final answer from what we have.
            if (time.time() - t0) > self.wall_clock_timeout_sec:
                run.terminated_reason = "wall_clock_timeout"
                break
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.per_call_max_tokens,
                "temperature": self.temperature,
                "max_retries": self.max_retries,
            }
            if self.tool_specs:
                kwargs["tools"] = self.tool_specs
                kwargs["tool_choice"] = "auto"
            try:
                response = chat_with_retry(self.client, **kwargs)
            except Exception as e:
                run.terminated_reason = "api_error"
                run.error = f"{type(e).__name__}: {str(e)[:300]}"
                terminated = True
                break

            choice = response.choices[0]
            msg = choice.message
            content = msg.content or ""
            tool_calls = list(msg.tool_calls or [])
            reasoning_content = (getattr(msg, "reasoning_content", "") or "")

            usage = response.usage
            prompt_t = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            cache_t = extract_cached_tokens(usage)
            comp_t = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            reason_t = extract_reasoning_tokens(usage)

            non_cached_in = max(0, prompt_t - cache_t)
            turn_cost = estimate_cost(
                {
                    "input_tokens": non_cached_in,
                    "cached_input_tokens": cache_t,
                    "output_tokens": comp_t,
                },
                self.model,
            )

            turn = TurnRecord(
                turn=turn_idx,
                content=content,
                reasoning_content=reasoning_content,
                finish_reason=(choice.finish_reason or ""),
                prompt_tokens=non_cached_in,
                cached_prompt_tokens=cache_t,
                completion_tokens=comp_t,
                reasoning_tokens=reason_t,
                cost_usd=turn_cost,
            )
            run.total_prompt_tokens += turn.prompt_tokens
            run.total_cached_prompt_tokens += turn.cached_prompt_tokens
            run.total_completion_tokens += turn.completion_tokens
            run.total_reasoning_tokens += turn.reasoning_tokens
            run.total_cost_usd += turn_cost

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [tc.model_dump() for tc in tool_calls]
            # Preserve provider reasoning alongside content in later requests.
            if reasoning_content:
                assistant_msg["reasoning_content"] = reasoning_content
            messages.append(assistant_msg)

            if not tool_calls:
                # A length-truncated turn with no content and no tool_calls is
                # NOT a successful finish — the reasoner blew its output budget
                # before emitting anything visible. Falling out of the loop here
                # would set final_answer="" and call it "done". Instead: drop
                # the failed turn from history and continue, so the model gets
                # another shot. (With per_call_max_tokens=32000 default this is
                # rare, but cheap to defend against.)
                if (choice.finish_reason or "").lower() == "length" and not content:
                    messages.pop()
                    turn.content = "(empty — length-truncated, retrying)"
                    run.turns.append(turn)
                    continue
                run.final_answer = content
                run.terminated_reason = "done"
                run.turns.append(turn)
                terminated = True
                break

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError as e:
                    tcr = ToolCallRecord(
                        name=name,
                        args={},
                        result=f"Error: invalid JSON in tool call arguments: {e}",
                        result_chars=0,
                        error="json_decode_error",
                    )
                    tcr.result_chars = len(tcr.result)
                    tcr.result_text_preview = tcr.result[:200]
                    turn.tool_calls.append(tcr)
                    run.tool_call_count += 1
                    run.tool_call_breakdown[name] = run.tool_call_breakdown.get(name, 0) + 1
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": tcr.result})
                    continue

                tool_fn = self.tools.get(name)
                if tool_fn is None:
                    result = f"Error: unknown tool {name!r}"
                    err = "unknown_tool"
                    duration = 0.0
                else:
                    tts = time.time()
                    try:
                        result = tool_fn(args)
                        err = ""
                    except Exception as e:
                        result = f"Error: tool {name!r} raised {type(e).__name__}: {str(e)[:300]}"
                        err = f"{type(e).__name__}"
                    duration = time.time() - tts

                truncated = "...[truncated" in result or "[truncated" in result
                tcr = ToolCallRecord(
                    name=name,
                    args=args,
                    result=result,
                    result_chars=len(result),
                    result_text_preview=result[:200],
                    truncated=truncated,
                    duration_seconds=duration,
                    error=err,
                )
                turn.tool_calls.append(tcr)
                run.tool_call_count += 1
                run.tool_call_breakdown[name] = run.tool_call_breakdown.get(name, 0) + 1
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            run.turns.append(turn)

        if not terminated:
            # If we broke out for wall-clock, keep that reason; otherwise it's max_turns.
            if run.terminated_reason != "wall_clock_timeout":
                run.terminated_reason = "max_turns"
            if self.coerce_final_on_max_turns:
                self._coerce_final_answer(run, messages)

        run.elapsed_seconds = time.time() - t0

        if trace_path is not None:
            _dump_trace(run, trace_path, relpath_to_docid=trace_relpath_to_docid)
        return run

    def _run_responses_loop(
        self,
        run: AgentRun,
        system_prompt: str,
        user_prompt: str,
        t0: float,
    ) -> None:
        """Agent loop driven by the Responses API (gpt-5 / o-series).

        Differences from the chat-completions loop (intentional):
          - `previous_response_id` chains server-side reasoning state across
            turns, so per-turn `input` is only the new `function_call_output`
            items from the tool calls of the previous turn.
          - `reasoning: {effort, summary: "auto"}` returns visible reasoning
            summary text in `output[].type == "reasoning"` blocks (which we
            store in `turn.reasoning_content`).
          - `prompt_cache_key` is a deterministic hash of (model, system_prompt
            prefix) so all queries of the same agent type share a cache key.
        """
        import hashlib
        # Convert tool specs from chat.completions shape to responses shape.
        # chat: {type:'function', function:{name, description, parameters}}
        # responses: {type:'function', name, description, parameters}
        tools_responses: list[dict[str, Any]] = []
        for spec in self.tool_specs:
            fn = spec.get("function") or spec
            tools_responses.append({
                "type": "function",
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
                "strict": False,
            })

        # First turn carries system + user; subsequent turns carry only
        # function_call_output items, with `previous_response_id` providing
        # the rest of the context.
        next_input: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Local mirror of conversation for context-management. Built
        # alongside next_input. When micro_compact fires we drop
        # previous_response_id and re-seed the API with the full
        # compacted mirror.
        mirror: list[dict[str, Any]] = list(next_input)
        # Deterministic cache key per (model, system prompt prefix) so OpenAI's
        # prompt cache hits across queries that share an agent type.
        cache_key = hashlib.sha256(
            (self.model + "::" + system_prompt[:512]).encode("utf-8")
        ).hexdigest()[:48]
        reasoning_effort = effective_reasoning_effort()
        previous_response_id: str | None = None

        terminated = False
        for turn_idx in range(self.max_turns):
            if (time.time() - t0) > self.wall_clock_timeout_sec:
                run.terminated_reason = "wall_clock_timeout"
                break

            # Pi-faithful micro_compact: check accumulated tool-result
            # chars in our local mirror. If over threshold, replace old
            # function_call_output content with `[cleared]` and re-seed
            # the Responses API with the compacted mirror (drop the
            # previous_response_id chain). Zero LLM calls; mirrors
            # micro-compact.ts:14 verbatim.
            compacted_mirror, did_compact = self._maybe_micro_compact_responses(mirror)
            if did_compact:
                mirror = compacted_mirror
                # Re-seed: send the full compacted history instead of just
                # the delta. We drop previous_response_id so the API treats
                # this as a fresh chain.
                next_input = list(compacted_mirror)
                previous_response_id = None
                run.n_micro_compactions += 1

            kwargs: dict[str, Any] = {
                "model": self.model,
                "input": next_input,
                "reasoning": {"effort": reasoning_effort, "summary": "auto"},
                "prompt_cache_key": cache_key,
                "max_output_tokens": self.per_call_max_tokens,
                "max_retries": self.max_retries,
            }
            if tools_responses:
                kwargs["tools"] = tools_responses
                kwargs["tool_choice"] = "auto"
            if previous_response_id is not None:
                kwargs["previous_response_id"] = previous_response_id

            try:
                response = responses_with_retry(self.client, **kwargs)
            except Exception as e:
                run.terminated_reason = "api_error"
                run.error = f"{type(e).__name__}: {str(e)[:300]}"
                terminated = True
                break

            previous_response_id = getattr(response, "id", None)
            status = getattr(response, "status", "") or ""

            # Walk output blocks: reasoning (with summary text), message
            # (with output_text content), and function_call (with name/args).
            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            function_calls: list[Any] = []
            for blk in (getattr(response, "output", None) or []):
                btype = getattr(blk, "type", "")
                if btype == "reasoning":
                    for s in (getattr(blk, "summary", None) or []):
                        st = getattr(s, "text", None) or ""
                        if st:
                            reasoning_parts.append(st)
                elif btype == "message":
                    for c in (getattr(blk, "content", None) or []):
                        if getattr(c, "type", "") == "output_text":
                            ct = getattr(c, "text", None) or ""
                            if ct:
                                text_parts.append(ct)
                elif btype == "function_call":
                    function_calls.append(blk)
            content = "\n".join(text_parts)
            reasoning_content = "\n".join(reasoning_parts)

            u = extract_responses_usage(getattr(response, "usage", None))
            turn_cost = estimate_cost(
                {
                    "input_tokens": u["input_tokens"],
                    "cached_input_tokens": u["cached_input_tokens"],
                    "output_tokens": u["output_tokens"],
                },
                self.model,
            )
            turn = TurnRecord(
                turn=turn_idx,
                content=content,
                reasoning_content=reasoning_content,
                finish_reason=status,
                prompt_tokens=u["input_tokens"],
                cached_prompt_tokens=u["cached_input_tokens"],
                completion_tokens=u["output_tokens"],
                reasoning_tokens=u["reasoning_tokens"],
                cost_usd=turn_cost,
            )
            run.total_prompt_tokens += turn.prompt_tokens
            run.total_cached_prompt_tokens += turn.cached_prompt_tokens
            run.total_completion_tokens += turn.completion_tokens
            run.total_reasoning_tokens += turn.reasoning_tokens
            run.total_cost_usd += turn_cost

            if not function_calls:
                # No tool calls AND no visible content + status=incomplete →
                # transient (output budget blown before anything visible). Drop
                # the turn from the chain (don't advance previous_response_id
                # state interpretation) and try again.
                if not content and status == "incomplete":
                    turn.content = "(empty — incomplete, retrying)"
                    run.turns.append(turn)
                    # next_input stays empty; with previous_response_id the
                    # model will continue from prior state
                    next_input = []
                    continue
                run.final_answer = content
                run.terminated_reason = "done"
                run.turns.append(turn)
                terminated = True
                break

            # Execute each function call; build the next turn's input as a
            # list of function_call_output items (one per call). Each
            # tool's result is post-truncated to context_settings'
            # max_tool_result_chars before being recorded into the
            # local mirror or sent back to the model — matching Pi's
            # afterToolCall hook in agent-session.ts:421.
            next_input = []
            # Track the function_call items in mirror BEFORE we add this
            # turn's outputs, so micro_compact can find turn boundaries.
            for fc in function_calls:
                name = getattr(fc, "name", "") or ""
                call_id = getattr(fc, "call_id", None) or getattr(fc, "id", "") or ""
                args_raw = getattr(fc, "arguments", "") or ""
                # Add the function_call to the mirror first (assistant turn).
                mirror.append({
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": args_raw,
                })
                try:
                    args = json.loads(args_raw or "{}")
                except json.JSONDecodeError as e:
                    err_msg = f"Error: invalid JSON in tool call arguments: {e}"
                    err_msg_trunc = self._truncate_tool_result_text(err_msg, run)
                    tcr = ToolCallRecord(
                        name=name, args={}, result=err_msg_trunc,
                        result_chars=len(err_msg_trunc),
                        result_text_preview=err_msg_trunc[:200],
                        error="json_decode_error",
                    )
                    turn.tool_calls.append(tcr)
                    run.tool_call_count += 1
                    run.tool_call_breakdown[name] = run.tool_call_breakdown.get(name, 0) + 1
                    out_item = {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": err_msg_trunc,
                    }
                    next_input.append(out_item)
                    mirror.append(out_item)
                    continue

                tool_fn = self.tools.get(name)
                if tool_fn is None:
                    result = f"Error: unknown tool {name!r}"
                    err = "unknown_tool"
                    duration = 0.0
                else:
                    tts = time.time()
                    try:
                        result = tool_fn(args)
                        err = ""
                    except Exception as e:
                        result = f"Error: tool {name!r} raised {type(e).__name__}: {str(e)[:300]}"
                        err = f"{type(e).__name__}"
                    duration = time.time() - tts

                # Apply Pi level3 truncate-tool-results (clip to
                # max_tool_result_chars + append marker). No-op when
                # context_level=level0.
                result_truncated = self._truncate_tool_result_text(result, run)

                truncated = "...[truncated" in result_truncated or "[truncated" in result_truncated
                tcr = ToolCallRecord(
                    name=name, args=args, result=result_truncated,
                    result_chars=len(result_truncated),
                    result_text_preview=result_truncated[:200],
                    truncated=truncated,
                    duration_seconds=duration, error=err,
                )
                turn.tool_calls.append(tcr)
                run.tool_call_count += 1
                run.tool_call_breakdown[name] = run.tool_call_breakdown.get(name, 0) + 1
                out_item = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result_truncated,
                }
                next_input.append(out_item)
                mirror.append(out_item)

            run.turns.append(turn)

        if not terminated:
            if run.terminated_reason != "wall_clock_timeout":
                run.terminated_reason = "max_turns"
            if self.coerce_final_on_max_turns:
                self._coerce_final_answer_responses(
                    run,
                    previous_response_id=previous_response_id,
                    cache_key=cache_key,
                    reasoning_effort=reasoning_effort,
                )

    def _coerce_final_answer_responses(
        self,
        run: AgentRun,
        *,
        previous_response_id: str | None,
        cache_key: str,
        reasoning_effort: str,
    ) -> None:
        """Coerce path for the Responses API: send a user message + previous_response_id,
        no tools. Records the coerce turn in run.turns."""
        # If we never produced a response (e.g. wall_clock before turn 0), we
        # don't have a chain — fall back to a fresh request with the original
        # system+user. The caller's `messages`/`next_input` was empty here, so
        # we synthesize from run.system_prompt + run.user_prompt.
        if previous_response_id is None:
            input_items = [
                {"role": "system", "content": run.system_prompt},
                {"role": "user", "content": run.user_prompt},
                {"role": "user", "content": self.coerce_message},
            ]
        else:
            input_items = [{"role": "user", "content": self.coerce_message}]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "reasoning": {"effort": reasoning_effort, "summary": "auto"},
            "prompt_cache_key": cache_key,
            "max_output_tokens": self.per_call_max_tokens,
            "max_retries": self.max_retries,
        }
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id

        try:
            response = responses_with_retry(self.client, **kwargs)
        except Exception as e:
            run.terminated_reason = "coerce_failed"
            run.error = f"coerce: {type(e).__name__}: {str(e)[:300]}"
            return

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        for blk in (getattr(response, "output", None) or []):
            btype = getattr(blk, "type", "")
            if btype == "reasoning":
                for s in (getattr(blk, "summary", None) or []):
                    st = getattr(s, "text", None) or ""
                    if st:
                        reasoning_parts.append(st)
            elif btype == "message":
                for c in (getattr(blk, "content", None) or []):
                    if getattr(c, "type", "") == "output_text":
                        ct = getattr(c, "text", None) or ""
                        if ct:
                            text_parts.append(ct)
        content = "\n".join(text_parts)
        reasoning_content = "\n".join(reasoning_parts)
        u = extract_responses_usage(getattr(response, "usage", None))
        turn_cost = estimate_cost(
            {
                "input_tokens": u["input_tokens"],
                "cached_input_tokens": u["cached_input_tokens"],
                "output_tokens": u["output_tokens"],
            },
            self.model,
        )
        status = getattr(response, "status", "") or ""
        turn = TurnRecord(
            turn=len(run.turns),
            content=content,
            reasoning_content=reasoning_content,
            finish_reason=f"coerced/{status}",
            prompt_tokens=u["input_tokens"],
            cached_prompt_tokens=u["cached_input_tokens"],
            completion_tokens=u["output_tokens"],
            reasoning_tokens=u["reasoning_tokens"],
            cost_usd=turn_cost,
        )
        run.turns.append(turn)
        run.total_prompt_tokens += turn.prompt_tokens
        run.total_cached_prompt_tokens += turn.cached_prompt_tokens
        run.total_completion_tokens += turn.completion_tokens
        run.total_reasoning_tokens += turn.reasoning_tokens
        run.total_cost_usd += turn_cost
        run.final_answer = content
        run.terminated_reason = "coerced_max_turns"

    def _coerce_final_answer(self, run: AgentRun, messages: list[dict]) -> None:
        """Inject a 'budget exhausted, finalize now' user message and run one
        more LLM call WITHOUT tools, so the model is forced to emit a text
        answer based on the evidence it has already gathered.

        Records the coerce turn in run.turns so it appears in the trace, and
        sets terminated_reason to 'coerced_max_turns' on success (or
        'coerce_failed' on API error).
        """
        messages.append({"role": "user", "content": self.coerce_message})
        try:
            response = chat_with_retry(
                self.client,
                model=self.model,
                messages=messages,
                max_tokens=self.per_call_max_tokens,
                temperature=self.temperature,
                max_retries=self.max_retries,
            )
        except Exception as e:
            run.terminated_reason = "coerce_failed"
            run.error = f"coerce: {type(e).__name__}: {str(e)[:300]}"
            return

        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        reasoning_content = (getattr(msg, "reasoning_content", "") or "")
        usage = response.usage
        prompt_t = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
        cache_t = extract_cached_tokens(usage)
        comp_t = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
        reason_t = extract_reasoning_tokens(usage)
        non_cached_in = max(0, prompt_t - cache_t)
        turn_cost = estimate_cost(
            {
                "input_tokens": non_cached_in,
                "cached_input_tokens": cache_t,
                "output_tokens": comp_t,
            },
            self.model,
        )

        turn = TurnRecord(
            turn=len(run.turns),
            content=content,
            reasoning_content=reasoning_content,
            finish_reason=f"coerced/{choice.finish_reason or ''}",
            prompt_tokens=non_cached_in,
            cached_prompt_tokens=cache_t,
            completion_tokens=comp_t,
            reasoning_tokens=reason_t,
            cost_usd=turn_cost,
        )
        run.turns.append(turn)
        run.total_prompt_tokens += turn.prompt_tokens
        run.total_cached_prompt_tokens += turn.cached_prompt_tokens
        run.total_completion_tokens += turn.completion_tokens
        run.total_reasoning_tokens += turn.reasoning_tokens
        run.total_cost_usd += turn_cost

        run.final_answer = content
        run.terminated_reason = "coerced_max_turns"


import re as _re
from .trajectory import SCHEMA_VERSION as _SCHEMA_VERSION

# Match a single line of the bm25_search preview, e.g.:
#   "  1. [12.34] path/to/file.txt"
#   "  2. [9.12] path/to/file.txt [already in wd]"
# We capture the path up to `.txt`, then tolerate an optional `[...]` annotation.
_SEARCH_PREVIEW_RE = _re.compile(
    r"^\s*\d+\.\s+\[[\d\.\-]+\]\s+(.+?\.txt)\s*(?:\[[^\]]*\])?\s*$"
)

# A passage_id like `5412_p0042`. Used by the trace serializer to detect
# read_passage targets that are passage_ids vs relpaths.
_PASSAGE_ID_RE = _re.compile(r"^\d+_p\d+$")


def _extract_search_relpaths(result_text: str) -> list[str]:
    """Pull the relpaths shown in a search-tool preview, preserving rank order
    and de-duplicating across multiple per-query blocks."""
    if not result_text or ".txt" not in result_text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for line in result_text.split("\n"):
        m = _SEARCH_PREVIEW_RE.match(line)
        if not m:
            continue
        rel = m.group(1).strip()
        if rel in seen:
            continue
        seen.add(rel)
        ordered.append(rel)
    return ordered


def _normalize_relpath(p: str) -> str:
    """Normalize a corpus path into the relpath keys used by filename maps.

    Handles plain relpaths, ``./`` prefixes, ``bc_plus_docs/`` prefixes, and
    Pi/DCI-style absolute citations such as
    ``@/abs/path/to/bc_plus_docs/domain/file.txt``.
    """
    if not p:
        return p
    s = p.strip().lstrip("@").strip()
    marker = "/bc_plus_docs/"
    if marker in s:
        s = s.split(marker, 1)[1]
    if s.startswith("bc_plus_docs/"):
        s = s[len("bc_plus_docs/"):]
    while s.startswith("./"):
        s = s[2:]
    while s.startswith("/"):
        s = s[1:]
    return s


def _tool_call_to_trace(
    c: ToolCallRecord,
    relpath_to_docid: dict[str, str] | None = None,
) -> dict:
    """Convert an in-memory ToolCallRecord to a compact trace dict.

    For search/read the visible identifier is a corpus relpath. ``result_paths``
    always holds the raw paths as the agent saw them; ``result_docids`` holds
    the strict corpus docids when ``relpath_to_docid`` is supplied — paths
    that don't resolve (e.g. a malformed `read` arg) are OMITTED from
    ``result_docids`` rather than mixing path strings in alongside docids.
    Bash output is preserved verbatim (already truncated by the bash tool).
    """
    name = c.name
    base = {
        "name": name,
        "args": c.args,
        "args_raw": "",  # the Agent class parses args before storing; raw JSON not available here
        "result_docids": [],     # strict corpus docids only (no path fallback)
        "result_paths": [],      # raw relpaths (always; for inspection/render)
        "result_unresolved_paths": [],  # paths that couldn't be mapped to a docid
        "result_summary": "",
        "result_chars": c.result_chars,
        "result_text": "",
        # First 200 chars of the payload sent to the model — always
        # populated, even when result_text is empty by design (search /
        # read clear result_text since docs are recoverable from docids).
        "result_text_preview": c.result_text_preview,
        "result_truncated": c.truncated,
        "duration_seconds": c.duration_seconds,
        "error": c.error,
    }

    def _to_docids(paths: list[str]) -> tuple[list[str], list[str]]:
        """Return (resolved_docids, unresolved_paths).

        Unresolved paths are NOT mixed into resolved_docids — keeping
        result_docids strict makes downstream cross-runner analysis safe.
        """
        if not relpath_to_docid:
            return [], list(paths)
        resolved: list[str] = []
        unresolved: list[str] = []
        for p in paths:
            d = relpath_to_docid.get(_normalize_relpath(p))
            if d is not None:
                resolved.append(d)
            else:
                unresolved.append(p)
        return resolved, unresolved

    if name == "search":
        relpaths = _extract_search_relpaths(c.result)
        base["result_paths"] = relpaths
        resolved, unresolved = _to_docids(relpaths)
        base["result_docids"] = resolved
        base["result_unresolved_paths"] = unresolved
        # Pull the header line (e.g. "search (2 queries): 12 new docs added; ...")
        first_nonempty = next(
            (ln.strip() for ln in c.result.split("\n") if ln.strip()),
            "",
        )
        base["result_summary"] = first_nonempty[:300]
    elif name in ("read", "read_doc"):
        fp = ""
        if isinstance(c.args, dict):
            # Legacy RISE read uses `file_path`; Pi-faithful DCI read uses `path`.
            fp = (c.args.get("file_path") or c.args.get("path") or "").strip()
        if fp:
            base["result_paths"] = [fp]
            resolved, unresolved = _to_docids([fp])
            base["result_docids"] = resolved
            base["result_unresolved_paths"] = unresolved
        base["result_summary"] = f"{name} {fp!r}: {c.result_chars} chars"
    elif name == "read_passage":
        # Passage-mode: args is {"target": <passage_id|passage_relpath>}.
        # The trace-time map is relpath → passage_id; we resolve both
        # input forms uniformly:
        #   - passage_id (regex `\d+_p\d+`): docid is the target itself;
        #     resolve relpath by reverse-lookup so result_paths is filled.
        #   - passage_relpath: standard `_to_docids` path.
        tgt = ""
        if isinstance(c.args, dict):
            tgt = (c.args.get("target") or c.args.get("passage_id") or c.args.get("file_path") or "").strip()
        if tgt:
            if _PASSAGE_ID_RE.match(tgt):
                # passage_id is the docid; reverse-lookup its relpath if
                # we have the relpath_to_docid map for it.
                base["result_docids"] = [tgt]
                if relpath_to_docid:
                    # Lazy reverse lookup (only run when we need it; cheap
                    # one-pass since traces have few read_passage calls).
                    rel = next((r for r, d in relpath_to_docid.items() if d == tgt), None)
                    if rel:
                        base["result_paths"] = [rel]
                    else:
                        base["result_paths"] = [tgt]  # keep target for inspection
                else:
                    base["result_paths"] = [tgt]
            else:
                base["result_paths"] = [tgt]
                resolved, unresolved = _to_docids([tgt])
                base["result_docids"] = resolved
                base["result_unresolved_paths"] = unresolved
        base["result_summary"] = f"read_passage {tgt!r}: {c.result_chars} chars"
    elif name == "bash":
        # bash output is small (truncated to 2-4k by the tool); keep verbatim.
        base["result_text"] = c.result
        cmd = (c.args.get("command") or "").strip() if isinstance(c.args, dict) else ""
        base["result_summary"] = f"bash {cmd[:80]}: {c.result_chars} chars"
    else:
        # Unknown tool — keep result verbatim so nothing is silently lost.
        base["result_text"] = c.result
        base["result_summary"] = f"{name}: {c.result_chars} chars"
    return base


def _dump_trace(
    run: AgentRun,
    trace_path: Path,
    *,
    relpath_to_docid: dict[str, str] | None = None,
) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "schema_version": _SCHEMA_VERSION,
        "model": run.model,
        "system_prompt": run.system_prompt,
        "user_prompt": run.user_prompt,
        "final_text": run.final_answer,
        "terminated_by": run.terminated_reason,
        "error": run.error,
        "elapsed_seconds": run.elapsed_seconds,
        "totals": {
            "n_turns": len(run.turns),
            "tool_call_count": run.tool_call_count,
            "tool_call_breakdown": dict(run.tool_call_breakdown),
            "prompt_tokens": run.total_prompt_tokens,
            "cached_prompt_tokens": run.total_cached_prompt_tokens,
            "completion_tokens": run.total_completion_tokens,
            "reasoning_tokens": run.total_reasoning_tokens,
            "agent_cost_usd": run.total_cost_usd,
        },
        "turns": [
            {
                "turn": t.turn,
                "finish_reason": t.finish_reason,
                "content": t.content,
                "reasoning_content": t.reasoning_content,
                "tokens": {
                    "prompt": t.prompt_tokens,
                    "cached_prompt": t.cached_prompt_tokens,
                    "completion": t.completion_tokens,
                    "reasoning": t.reasoning_tokens,
                },
                "cost_usd": t.cost_usd,
                "tool_calls": [_tool_call_to_trace(c, relpath_to_docid) for c in t.tool_calls],
            }
            for t in run.turns
        ],
    }
    trace_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
