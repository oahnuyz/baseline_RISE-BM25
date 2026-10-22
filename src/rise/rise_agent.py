"""Single-agent variant: one LLM with three tools (bm25_search, bash, read).

Architecture: a single Agent loop with all the corpus access in one
conversation context — no orchestrator / sub-agent split. Compared to the
hierarchical setup, this preserves full evidence (no prose-truncation
boundary), uses prompt cache more effectively (one monotonically growing
conversation), and matches the BCP retrieval-agent / DCI bash-only baselines
more closely.

Mini-corpus mode is ACCUMULATE: each bm25_search hardlinks the union of its
top-k matches into the working dir without clearing prior entries. The
`bash` and `read` tools are both scoped to the working dir, so the agent
can only inspect docs that some search has retrieved this session.

Output dataclass mirrors the field set used by the runner script (so the
existing aggregation/judging machinery works).
"""
from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import bm25s
from openai import OpenAI

from .agent import Agent, AgentRun
from .dci_artifacts import _classify_bash
from .protocol import RunConfig
from .tools import (
    bash_tool_spec,
    bm25_search_tool_spec,
    make_bash_tool,
    make_bm25_search_tool,
    make_read_doc_tool,
    make_read_passage_tool,
    make_read_tool,
    read_doc_tool_spec,
    read_passage_tool_spec,
    read_tool_spec,
)


RISE_SYSTEM_PROMPT = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have three tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use bash/read to "
    "explore beyond the preview.\n"
    "- bash(command): run a shell command (rg, grep, ls, find, cat, head, "
    "etc.) over your working directory. Paths are relative to the working "
    "directory; absolute paths (anything starting with `/`) won't find "
    "anything — bash can't see outside the working directory. Use rg/grep "
    "to find phrases across retrieved docs. Files are plain text; run "
    "`ls` first to see the directory layout.\n"
    "- read(file_path): read a file by its full path.\n\n"
    "Use the tools iteratively: search to pull evidence into your working "
    "directory, bash/read to inspect it, and repeat until confident. When "
    "confident, output the final answer in this format and stop calling "
    "tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_NOBASH = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have two tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use read to "
    "explore beyond the preview.\n"
    "- read(file_path): read a file by its full path.\n\n"
    "Use the tools iteratively: search to pull candidate docs into your "
    "working directory, read to inspect them, and repeat until confident. "
    "When confident, output the final answer in this format and stop "
    "calling tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_STRUCTURED_NOBASH = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have two tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use read to "
    "explore beyond the preview.\n"
    "- read(file_path, offset?, limit?): read a file by its full path. "
    "Optional 0-indexed `offset` and `limit` (both in LINES, not chars) "
    "let you read a slice instead of the whole file; defaults to lines "
    "0..2000.\n\n"
    "Every document begins with a YAML frontmatter block (title / author "
    "/ date), followed by a `## Table of Contents` listing each section "
    "with a 1-indexed line range and a short description (e.g. `- L48–"
    "65: Early history — Founded 1874 as the Nautical School...`), then "
    "a literal `=== DOCUMENT BODY ===` line, then the body with `## "
    "<heading>` markers at the listed line ranges.\n\n"
    "How to read these docs:\n"
    "(1) Start with `read(path, offset=0, limit=60)` to see the YAML + "
    "TOC + sentinel. If the result does NOT contain the `=== DOCUMENT "
    "BODY ===` line, the TOC is longer — re-read with a larger limit "
    "(e.g. `offset=0, limit=200`).\n"
    "(2) The TOC is a navigation index, NOT an answer source. Each TOC "
    "entry is a short summary line, not the actual text — it can omit "
    "details, drop nuance, or be slightly off. Never extract a specific "
    "fact (number, name, date, quote) from a TOC summary line.\n"
    "(3) Pick the most relevant section(s) from the TOC and read each "
    "one in full: `read(path, offset=<start-1>, limit=<end-start+1>)`. "
    "Confirm the answer in the actual section body before committing. "
    "If the body doesn't contain the expected fact, the TOC summary was "
    "misleading — read a wider chunk or another section.\n"
    "(4) If multiple sections look plausible, read each. If no section "
    "looks right, read more of the doc (larger limit, or a different "
    "offset). Don't commit just because the TOC line sounds relevant.\n"
    "(5) If the TOC says `(no sections — short or single-topic document; "
    "full text follows)`, read the file normally.\n\n"
    "Use the tools iteratively: search to pull candidate docs into your "
    "working directory, read to inspect them, and repeat until confident. "
    "When confident, output the final answer in this format and stop "
    "calling tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_STRUCTURED = (
    "You answer research questions over a large document corpus you can't "
    "see directly.\n\n"
    "You have three tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Each query is matched as a bag of words against document "
    "text, so write queries as natural-language descriptions with several "
    "distinctive terms; pass multiple complementary queries together for "
    "broader coverage. Returns a per-query top-10 preview with file paths "
    "and short snippets — the preview is only a sample; the full match "
    "set per query (hundreds to thousands of docs) is ADDED to your "
    "working directory and accumulates across turns, so use bash/read to "
    "explore beyond the preview.\n"
    "- bash(command): run a shell command (rg, grep, ls, find, cat, head, "
    "etc.) over your working directory. Paths are relative to the working "
    "directory; absolute paths (anything starting with `/`) won't find "
    "anything — bash can't see outside the working directory. Use rg/grep "
    "to find phrases across retrieved docs. Files are plain text; run "
    "`ls` first to see the directory layout.\n"
    "- read(file_path, offset?, limit?): read a file by its full path. "
    "Optional 0-indexed `offset` and `limit` (both in LINES, not chars) "
    "let you read a slice instead of the whole file; defaults to lines "
    "0..2000.\n\n"
    "Each document contains a `## Table of Contents` listing each section "
    "with a 1-indexed line range and a short description (e.g. `- L48–"
    "65: Early history — Founded 1874 as the Nautical School...`), then "
    "a literal `=== DOCUMENT BODY ===` line, then the body with `## "
    "<heading>` markers at the listed line ranges.\n\n"
    "How to read these docs:\n"
    "(1) Start with `read(path, offset=0, limit=60)` to see the "
    "TOC + sentinel. If the result does NOT contain the `=== DOCUMENT "
    "BODY ===` line, the TOC is longer — re-read with a larger limit "
    "(e.g. `offset=0, limit=200`).\n"
    "(2) The TOC is a navigation index, NOT an answer source. Each TOC "
    "entry is a short summary line, not the actual text — it can omit "
    "details, drop nuance, or be slightly off. Never extract a specific "
    "fact (number, name, date, quote) from a TOC summary line.\n"
    "(3) Pick the most relevant section(s) from the TOC and read each "
    "one in full: `read(path, offset=<start-1>, limit=<end-start+1>)`. "
    "Confirm the answer in the actual section body before committing. "
    "If the body doesn't contain the expected fact, the TOC summary was "
    "misleading — read a wider chunk or another section.\n"
    "(4) If multiple sections look plausible, read each. If no section "
    "looks right, read more of the doc (larger limit, or a different "
    "offset). Don't commit just because the TOC line sounds relevant.\n"
    "(5) If the TOC says `(no sections — short or single-topic document; "
    "full text follows)`, read the file normally.\n\n"
    "Use the tools iteratively: search to pull evidence into your working "
    "directory, bash/read to inspect it, and repeat until confident. When "
    "confident, output the final answer in this format and stop calling "
    "tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_SYSTEM_PROMPT_PASSAGE = (
    "You answer research questions over a large document corpus you can't "
    "see directly. The corpus is indexed at the PASSAGE level: each "
    "document is split into ~512-token chunks. Search returns the most "
    "relevant passages, not whole documents.\n\n"
    "You have four tools:\n"
    "- search(queries): search the corpus with one or more queries in a "
    "single call. Returns top-matching passages; the union is ADDED to "
    "your working directory. Each passage file is `<domain>/<title>/"
    "passages/p<NNNN>.txt`, so the directory layout preserves the parent "
    "document for context. Per-query preview shows passage paths and "
    "short snippets. Write queries as natural-language descriptions with "
    "several distinctive terms.\n"
    "- bash(command): run a shell command (rg, grep, ls, find, cat, head, "
    "etc.) over your working directory of passages. `rg -l` returns "
    "passage-level paths. Files are short (~3-5kB each), boilerplate-"
    "free by construction. Use `ls <domain>/<title>/passages` to see how "
    "a document was chunked.\n"
    "- read_passage(target): read ONE passage. Accepts either a "
    "passage_id (`5412_p0042`) or a passage relpath (the path search/rg "
    "shows you). No offset/limit needed — passages are bounded. Prefer "
    "this when one passage carries the evidence you need.\n"
    "- read_doc(file_path): read the FULL parent document, by parent "
    "relpath (`<domain>/<title>.txt`). Use this when one passage isn't "
    "enough and you need surrounding sections (e.g., cross-section "
    "claims). If you pass a passage relpath, it auto-coerces to the "
    "parent with a one-line warning.\n\n"
    "General strategy: search to pull relevant passages, scan them with "
    "read_passage (and bash for cross-passage rg), and only fall back to "
    "read_doc when one passage's context is genuinely insufficient. When "
    "confident, output the final answer in this format and stop calling "
    "tools:\n"
    "Explanation: {your reasoning}\n"
    "Exact Answer: {your succinct final answer}\n"
    "Confidence: {0-100%}"
)


RISE_USER_PROMPT = "QUESTION:\n{question}"


@dataclass
class RISERun:
    query_id: Any
    question: str
    final_text: str = ""
    terminated_by: str = ""
    elapsed_seconds: float = 0.0
    # token / cost rollup
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    # tool stats
    n_turns: int = 0
    n_tool_calls: int = 0
    tool_call_breakdown: dict[str, int] = field(default_factory=dict)
    bash_kinds: dict[str, int] = field(default_factory=dict)
    n_bm25_searches: int = 0     # # of search tool CALLS
    n_bm25_queries: int = 0      # total # of QUERIES across all search calls
    n_bash_calls: int = 0
    n_read_calls: int = 0         # doc-mode `read` only
    n_read_passage_calls: int = 0  # passage-mode `read_passage`
    n_read_doc_calls: int = 0      # passage-mode `read_doc`
    # corpus coverage
    surfaced_relpaths: list[str] = field(default_factory=list)
    bm25_queries: list[str] = field(default_factory=list)  # flat list, all queries
    read_paths: list[str] = field(default_factory=list)


# bm25_search preview line: "  1. [12.34] path/to/file.txt"  (optionally with
# a trailing "[already in wd]" tag).
_BM25_PREVIEW_RE = re.compile(r"\]\s+(.+?\.txt)\s*(?:\[[^\]]*\])?\s*$")


_PASSAGE_ID_RE_SINGLE = re.compile(r"^\d+_p\d+$")


def _surfaced_relpaths_from_run(
    agent_run: AgentRun, map_keys: set[str], working_dir: Path,
    *,
    passage_id_to_relpath: dict[str, str] | None = None,
) -> set[str]:
    """Walk recorded tool-call args + results to find corpus relpaths that
    were touched (read tool target, bm25_search Top-20 preview, or rg/grep
    hits in bash output).

    In passage mode, `read_passage` may be called with a passage_id
    (`5412_p0042`) rather than a relpath. If `passage_id_to_relpath` is
    provided, we reverse-lookup such ids to their relpaths and add
    those, so coverage doesn't miss directly-read passages."""
    surfaced: set[str] = set()
    wd_str = str(working_dir.resolve())

    def _try_add(s: str) -> bool:
        if not s:
            return False
        s = s.strip()
        for prefix in ("./", "bc_plus_docs/"):
            if s.startswith(prefix):
                s = s[len(prefix):]
        if s.startswith(wd_str):
            s = s[len(wd_str):].lstrip("/")
        if s in map_keys:
            surfaced.add(s)
            return True
        # Passage-id reverse lookup (passage mode only)
        if passage_id_to_relpath is not None and _PASSAGE_ID_RE_SINGLE.match(s):
            rel = passage_id_to_relpath.get(s)
            if rel and rel in map_keys:
                surfaced.add(rel)
                return True
        return False

    for t in agent_run.turns:
        for tc in t.tool_calls:
            # `read` tool: file_path arg is a direct corpus relpath
            if tc.name == "read":
                _try_add((tc.args.get("file_path") or ""))
            # `read_doc` (passage-mode): file_path arg may be a parent
            # relpath OR a passage relpath that gets coerced. Record the
            # raw arg; the runner's coverage step coerces passage→parent.
            elif tc.name == "read_doc":
                _try_add((tc.args.get("file_path") or ""))
            # `read_passage`: target is passage_id (`\d+_p\d+`) OR a
            # passage relpath. We add both forms to surfaced so coverage
            # can resolve via passage_id-as-docid or relpath lookups.
            elif tc.name == "read_passage":
                tgt = (tc.args.get("target") or tc.args.get("passage_id") or tc.args.get("file_path") or "").strip()
                if tgt:
                    _try_add(tgt)

            blob = tc.result or ""
            if ".txt" not in blob:
                continue
            for ln in blob.split("\n"):
                ls = ln.strip()
                if ".txt" not in ls:
                    continue
                # 1) direct full-line match (e.g. `ls` output)
                if _try_add(ls):
                    continue
                # 2) bm25_search preview line: "  N. [score] path.txt"
                m = _BM25_PREVIEW_RE.search(ls)
                if m and _try_add(m.group(1)):
                    continue
                # 3) generic: cut at first ".txt" (handles "path.txt:line:..." rg)
                ti = ls.find(".txt")
                if ti >= 0:
                    _try_add(ls[: ti + 4])
    return surfaced


def run_rise_agent(
    question: str,
    query_id: Any,
    *,
    run_config: "RunConfig",
    searcher: bm25s.BM25,
    doc_ids: Sequence[str],
    docid_to_relpath: dict[str, str],
    bc_plus_docs_root: Path,
    working_dir: Path,
    client: OpenAI,
    trace_path: Path | None = None,
    map_keys: set[str] | None = None,
    enable_sandbox: bool = True,
    # Passage-mode plumbing. When `passage_mode=True`, `searcher / doc_ids
    # / docid_to_relpath / bc_plus_docs_root` are expected to be the
    # PASSAGE index, passage relpath map, and the passage-files root.
    # `parent_docs_root` + `relpath_to_parent_docid` are then used by the
    # new read_doc / read_passage tools. Toggled via run_config.runner_mode
    # ("passage" or "doc") — these params carry the runtime resources.
    parent_docs_root: Path | None = None,
    relpath_to_parent_docid: dict[str, str] | None = None,
) -> RISERun:
    """Run RISE on one query.

    All budget/timeout/retry knobs flow from `run_config`. Method-specific
    knobs come from `run_config.extras`:

      - `bm25_k`, `bm25_top_n_preview`, `bm25_snippet_chars`
      - `bash_truncate_chars`, `read_default_limit`
      - `passage_mode` (mirror of `run_config.runner_mode == "passage"`)

    `working_dir` is the per-query mini-corpus path. It will be (re)created
    on each bm25_search call. `searcher / doc_ids / docid_to_relpath /
    bc_plus_docs_root` are runtime resources passed by the caller.
    """
    extras = run_config.extras
    bm25_k = int(extras.get("bm25_k", 100))
    bm25_top_n_preview = int(extras.get("bm25_top_n_preview", 10))
    bm25_snippet_chars = int(extras.get("bm25_snippet_chars", 100))
    bash_truncate_chars = int(extras.get("bash_truncate_chars", 4000))
    read_default_limit = int(extras.get("read_default_limit", 2000))
    passage_mode = (run_config.runner_mode == "passage") or bool(extras.get("passage_mode", False))
    # When set, swap RISE_SYSTEM_PROMPT for RISE_SYSTEM_PROMPT_STRUCTURED — assumes
    # the corpus on disk has the doc_representation pipeline's YAML+TOC+heading
    # layout. Caller is responsible for pointing `bc_plus_docs_root` at the
    # restructured corpus tree.
    structured_doc_mode = bool(extras.get("structured_doc_mode", False))
    # When set, omit the `bash` tool from the registry — agent can only use
    # search and read. Ablation cell for "RISE minus mini-corpus exploration",
    # closest in spirit to a BCP retrieval-agent setup.
    no_bash = bool(extras.get("no_bash", False))

    t0 = time.time()
    out = RISERun(query_id=query_id, question=question)

    # Ensure the working dir exists (empty) before the agent runs — the agent
    # might call bash before its first bm25_search, and we want bash's cwd
    # to be valid.
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    bm25_fn = make_bm25_search_tool(
        searcher=searcher, doc_ids=doc_ids,
        docid_to_relpath=docid_to_relpath,
        bc_plus_docs_root=bc_plus_docs_root,
        working_dir=working_dir, k=bm25_k,
        top_n_preview=bm25_top_n_preview,
        snippet_chars=bm25_snippet_chars,
    )
    bash_fn = make_bash_tool(
        working_dir, truncate_chars=bash_truncate_chars,
        enable_sandbox=enable_sandbox,
    )

    if passage_mode:
        if parent_docs_root is None or relpath_to_parent_docid is None:
            raise ValueError(
                "passage_mode=True requires parent_docs_root and "
                "relpath_to_parent_docid (for read_doc fallback and "
                "[from <parent>] annotations)."
            )
        # In passage mode:
        # - `search` is over passages (caller passed passage index + map).
        # - `read_passage` reads a single passage from `bc_plus_docs_root`
        #   (which is the passage files root in passage mode).
        # - `read_doc` reads a parent doc from `parent_docs_root` (full corpus),
        #   and tolerates passage relpaths by stripping `/passages/p*.txt`.
        read_passage_fn = make_read_passage_tool(
            passage_files_root=bc_plus_docs_root,
            passage_id_to_relpath=docid_to_relpath,
            relpath_to_parent_docid=relpath_to_parent_docid,
        )
        read_doc_fn = make_read_doc_tool(parent_docs_root, default_limit=read_default_limit)
        tool_registry = {
            "search": bm25_fn,
            "bash": bash_fn,
            "read_passage": read_passage_fn,
            "read_doc": read_doc_fn,
        }
        tool_specs = [bm25_search_tool_spec(), bash_tool_spec(), read_passage_tool_spec(), read_doc_tool_spec()]
        system_prompt = RISE_SYSTEM_PROMPT_PASSAGE
    else:
        # read root = working_dir, same scope as bash. Agent can only read
        # docs that some search has retrieved into the working directory.
        read_fn = make_read_tool(working_dir, default_limit=read_default_limit)
        if no_bash:
            tool_registry = {"search": bm25_fn, "read": read_fn}
            tool_specs = [bm25_search_tool_spec(), read_tool_spec()]
            system_prompt = (
                RISE_SYSTEM_PROMPT_STRUCTURED_NOBASH if structured_doc_mode
                else RISE_SYSTEM_PROMPT_NOBASH
            )
        else:
            tool_registry = {"search": bm25_fn, "bash": bash_fn, "read": read_fn}
            tool_specs = [bm25_search_tool_spec(), bash_tool_spec(), read_tool_spec()]
            system_prompt = (
                RISE_SYSTEM_PROMPT_STRUCTURED if structured_doc_mode
                else RISE_SYSTEM_PROMPT
            )

    agent = Agent(
        client=client, model=run_config.agent_model,
        tools=tool_registry, tool_specs=tool_specs,
        max_turns=run_config.max_model_calls,
        per_call_max_tokens=run_config.per_call_max_tokens,
        max_retries=run_config.api_max_retries,
        wall_clock_timeout_sec=run_config.wall_clock_timeout_sec,
        coerce_final_on_max_turns=run_config.coerce_final_on_max_turns,
    )
    # Build relpath→docid map for the trace dump so search/read tool calls
    # store corpus docids (matching BCP / Pi-Serini / DCI) instead of paths.
    # In passage_mode: relpath here is a passage relpath; docid is a
    # passage_id. The runner separately resolves passage_id → parent_docid
    # for gold-doc coverage.
    trace_relpath_to_docid = {rel: did for did, rel in docid_to_relpath.items()}
    agent_run: AgentRun = agent.run(
        system_prompt=system_prompt,
        user_prompt=RISE_USER_PROMPT.format(question=question),
        trace_path=trace_path,
        trace_relpath_to_docid=trace_relpath_to_docid,
    )

    out.final_text = agent_run.final_answer or ""
    out.terminated_by = agent_run.terminated_reason
    if agent_run.error:
        out.terminated_by = f"{agent_run.terminated_reason}: {agent_run.error}"
    out.elapsed_seconds = agent_run.elapsed_seconds
    out.input_tokens = agent_run.total_prompt_tokens
    out.cached_input_tokens = agent_run.total_cached_prompt_tokens
    out.output_tokens = agent_run.total_completion_tokens
    out.reasoning_tokens = agent_run.total_reasoning_tokens
    out.cost_usd = agent_run.total_cost_usd
    out.n_turns = len(agent_run.turns)
    out.n_tool_calls = agent_run.tool_call_count
    out.tool_call_breakdown = dict(agent_run.tool_call_breakdown)
    out.n_bm25_searches = out.tool_call_breakdown.get("search", 0)
    out.n_bash_calls = out.tool_call_breakdown.get("bash", 0)
    out.n_read_calls = out.tool_call_breakdown.get("read", 0)
    out.n_read_passage_calls = out.tool_call_breakdown.get("read_passage", 0)
    out.n_read_doc_calls = out.tool_call_breakdown.get("read_doc", 0)

    bash_kinds: dict[str, int] = {}
    bm25_queries: list[str] = []
    read_paths: list[str] = []
    for turn in agent_run.turns:
        for tc in turn.tool_calls:
            if tc.name == "bash":
                kind = _classify_bash(tc.args.get("command", ""))
                bash_kinds[kind] = bash_kinds.get(kind, 0) + 1
            elif tc.name == "search":
                # tolerant: queries list, or single query string
                qs = tc.args.get("queries")
                if qs is None:
                    qs = tc.args.get("query")
                if isinstance(qs, str):
                    qs = [qs]
                if isinstance(qs, list):
                    for q in qs:
                        if isinstance(q, str) and q.strip():
                            bm25_queries.append(q.strip())
            elif tc.name == "read":
                fp = tc.args.get("file_path", "")
                if fp:
                    read_paths.append(fp)
            elif tc.name == "read_doc":
                fp = tc.args.get("file_path", "")
                if fp:
                    read_paths.append(fp)
            elif tc.name == "read_passage":
                tgt = tc.args.get("target") or tc.args.get("passage_id") or tc.args.get("file_path", "")
                if tgt:
                    read_paths.append(tgt)
    out.bash_kinds = bash_kinds
    out.bm25_queries = bm25_queries
    out.n_bm25_queries = len(bm25_queries)
    out.read_paths = read_paths

    if map_keys is not None:
        # In passage mode, `docid_to_relpath` is `passage_id_to_relpath`;
        # pass it so `_surfaced_relpaths_from_run` can reverse-resolve
        # passage_ids the agent passed to read_passage directly.
        pid_to_rel = docid_to_relpath if passage_mode else None
        out.surfaced_relpaths = sorted(
            _surfaced_relpaths_from_run(
                agent_run, map_keys, working_dir,
                passage_id_to_relpath=pid_to_rel,
            )
        )
    return out
