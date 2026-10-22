"""Single-doc sectioning prototype.

Pipeline:
 1. Pick one BCP doc (default: SUNY Maritime College Wikipedia entry).
 2. Call an LLM (default gpt-5.4-nano) with {sections: [{title, anchor}]}.
 3. Validate each anchor is a verbatim substring of the doc body, in order
    (with whitespace-normalized and word-prefix fallbacks).
 4. Insert ## headings, build a TOC with line numbers, print final doc.

No content rewriting — only headings + TOC are injected. If the LLM
returns sections: [] (the doc doesn't need structuring), we leave the doc
as-is.
"""
from __future__ import annotations

import json
import os
import sys
import time
import textwrap
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "doc_representation"
CORPUS = ROOT / "external/dci-agent-lite/corpus/browsecomp_plus/data.parquet"

# Allow `from rise import ...` when running this script directly.
sys.path.insert(0, str(ROOT / "src"))
from rise.api_retry import chat_with_retry  # noqa: E402
from rise.decompose import _extract_json, make_client  # noqa: E402

# Use the same docid we saw in the sample (SUNY Maritime College Wikipedia entry,
# ~10k chars, decent length for a first inspection).
TARGET_DOCID = os.environ.get("TARGET_DOCID", "68896")
MODEL = os.environ.get("SECTION_MODEL", "gpt-5.4-nano")

load_dotenv(ROOT / ".env")

MAX_LLM_INPUT_CHARS = int(os.environ.get("MAX_LLM_INPUT_CHARS", "200000"))


def truncate_for_llm(text: str, max_chars: int = MAX_LLM_INPUT_CHARS) -> tuple[str, dict]:
    """Truncate text sent to the LLM at `max_chars`, preferring a paragraph
    boundary. The full original doc is preserved for downstream anchor matching
    and heading insertion — only the LLM input is shortened. Returns
    (truncated_text, info) where info records what was cut.
    """
    if len(text) <= max_chars:
        return text, {"truncated": False, "orig_chars": len(text), "kept_chars": len(text)}
    # prefer cutting at the last `\n\n` within the window; fall back to a hard cut
    cut = text.rfind("\n\n", 0, max_chars)
    if cut < int(max_chars * 0.9):  # paragraph boundary too far back → hard cut
        cut = max_chars
    truncated = text[:cut].rstrip() + (
        f"\n\n[NOTE TO MODEL: the document is {len(text):,} chars but only the "
        f"first {cut:,} chars are shown here. Propose sections only for the "
        f"shown portion; remaining content will be left unsectioned.]\n"
    )
    return truncated, {
        "truncated": True,
        "orig_chars": len(text),
        "kept_chars": cut,
        "dropped_chars": len(text) - cut,
    }


SECTION_PROMPT = """You restructure a plain-text document so a search agent (using bash tools `cat`, `sed`, `grep`) can navigate it.

You will NOT rewrite the document. You only propose:
  - section boundaries (where each section begins),
  - a short heading for each section, and
  - a one-sentence description of what each section covers.

A downstream script will use your output to insert `## <title>` headings into the ORIGINAL text and build a table-of-contents (with line numbers + your descriptions). The script locates each section by searching the document for the anchor string you provide. So the anchor is a SHORT LOCATOR — not the section's content.

OUTPUT FORMAT — return exactly this JSON, no commentary, no markdown fences:
{
  "sections": [
    {
      "title": "<short heading, 2-7 words, describing the section's content>",
      "anchor": "<the FIRST 6 to 12 WORDS of the section, copied EXACTLY from the document>",
      "description": "<one sentence, 10-25 words, summarizing what this section covers — designed for an agent skimming the TOC to decide if the section is worth reading>"
    },
    ...
  ]
}

ANCHOR RULES (CRITICAL — the script does exact substring matching):
- An anchor is ONLY a short locator string — typically 6 to 12 words. NEVER more than 12 words. Do not paste the whole paragraph.
- Copy those 6-12 words letter-for-letter from the document: same words, same punctuation, same capitalization, same internal whitespace.
- Each anchor must be a unique substring within the document.
- Anchors must appear in the same order as the sections appear in the document.
- The first section's anchor must be the first 6-12 words of the document BODY (skip the YAML `---` frontmatter at the top — do not anchor inside it).
- NEVER paraphrase, summarize, abbreviate, normalize, or "clean up" the anchor text. If the document says `Master of the SS United States`, the anchor must contain those exact words, not `Master of the` followed by a newline.
- If a section starts with a list, the anchor is the first 6-12 words of the FIRST list item (including any leading `*` or `-`), not multiple items.
- If you cannot locate a verbatim 6-12 word anchor in the document for a section you would like to propose, DO NOT propose that section. Under-sectioning is acceptable; inventing anchors is not.

SECTIONING GUIDELINES:
- Default behavior: ALMOST EVERY DOCUMENT HAS STRUCTURE WORTH MARKING. If the document has more than ~1500 characters of content, it almost certainly has 2+ sections you can identify (e.g., introduction + main content + conclusion, or topic A vs topic B).
- Only return {"sections": []} for genuinely short single-paragraph documents (under ~1000 characters of body text) OR documents that are pure metadata/boilerplate. When in doubt, propose sections — under-sectioning helps no one, but over-sectioning is mildly wasteful at worst.
- Split at natural topic shifts: a new subject, a clear temporal jump, a sub-topic the reader would expect a heading for.
- Default cadence for prose: one section per ~500-2000 words. A 3000-word narrative doc yields 2-6 sections, not 15.
- CATALOG / LIST documents are different and SHOULD be sectioned aggressively — one section per item is often correct:
  * A wildlife-identification guide listing 24 bird species → 24 sections (one per species).
  * A "list of films released in 2020" → either one big "Films" section, or one section per studio/country/letter grouping. Not 0 sections.
  * A Wikipedia franchise/series article with infobox + many distinct sub-topics → many sections.
  * A scientific paper with Abstract/Methods/Results/Discussion → at least those 4 sections.
- Procedural step lists (recipe instructions, how-to numbered steps, tutorials) belong to ONE umbrella section ("Instructions"), not one section per step. This is the OPPOSITE of catalog handling — steps are dependent and sequential, catalog items are independent.
- Section titles describe the content (e.g., "Early career at Bell Labs"), not generic ("Section 1", "Introduction" — those are too vague).
- Do NOT create a section for the YAML frontmatter itself.
- Do NOT create a section for trailing boilerplate (navigation, "Related links", "Follow us") unless it contains substantial content.

DESCRIPTION RULES:
- 10-25 words. Concrete, not generic. Mention key entities (names, dates, places, numbers) when those are the section's focus.
- Bad:  "Discusses the topic in more detail."
- Good: "Erling Persson's 1947 vision, opening of first womenswear store in Vasteras, family expansion through the 1950s."

DOCUMENT:
<<<DOC>>>
"""


def load_doc(docid: str) -> dict:
    df = pd.read_parquet(CORPUS, columns=["docid", "text", "url"])
    row = df[df["docid"] == docid]
    if len(row) == 0:
        raise SystemExit(f"docid={docid} not found")
    r = row.iloc[0]
    return {"docid": r["docid"], "text": r["text"], "url": r["url"]}


def call_llm(doc_text: str, model: str = MODEL) -> tuple[dict, dict]:
    """Call the configured LLM via the project's client factory.

    `make_client(model)` routes by model-name prefix: gpt-5* → OpenAI direct,
    deepseek-* → DeepSeek, etc. `chat_with_retry` handles gpt-5/o-series
    quirks (max_completion_tokens, reasoning_effort, no temperature).
    """
    client = make_client(model)
    llm_input, trunc_info = truncate_for_llm(doc_text)
    prompt = SECTION_PROMPT.replace("<<<DOC>>>", llm_input)
    t0 = time.time()
    resp = chat_with_retry(
        client,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=32768,  # large budget — long docs need ample reasoning + output room; chat_with_retry maps to max_completion_tokens
    )
    elapsed = time.time() - t0
    content = resp.choices[0].message.content or ""
    reasoning_tokens = None
    details = getattr(resp.usage, "completion_tokens_details", None)
    if details is not None:
        reasoning_tokens = getattr(details, "reasoning_tokens", None)
    usage = {
        "model": model,
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "elapsed_s": round(elapsed, 2),
        "truncation": trunc_info,
    }
    # `_extract_json` falls back to regex extraction if the model wraps the
    # object in markdown fences or adds a preamble.
    try:
        data = _extract_json(content)
    except (ValueError, json.JSONDecodeError) as e:
        # Defensive: return an empty sections list rather than crashing.
        return {"sections": [], "_parse_error": str(e), "_raw": content[:400]}, usage
    if not isinstance(data, dict):
        return {"sections": [], "_parse_error": "top-level JSON is not an object", "_raw": content[:400]}, usage
    return data, usage


import re


def _ws_norm(s: str) -> str:
    """Collapse all runs of whitespace to a single space, for fallback matching."""
    return re.sub(r"\s+", " ", s).strip()


def _find_all(haystack: str, needle: str, start: int) -> list[int]:
    """All non-overlapping start positions of `needle` in `haystack[start:]`."""
    if not needle:
        return []
    out: list[int] = []
    i = start
    while True:
        j = haystack.find(needle, i)
        if j == -1:
            return out
        out.append(j)
        i = j + max(1, len(needle))


def _has_para_break_before(doc_text: str, pos: int) -> bool:
    """Is `pos` immediately after a blank-line paragraph break? (`\\n\\n` within
    the preceding ~4 chars, modulo whitespace.) Also true at doc start (pos==0)
    or right after the YAML frontmatter closer."""
    if pos == 0:
        return True
    # Clamp pos to doc bounds — the whitespace-normalized fallback can return
    # an approximate offset that lands just past EOF.
    pos = min(pos, len(doc_text))
    if pos == 0:
        return True
    # walk backwards through whitespace
    j = pos - 1
    nl_count = 0
    while j >= 0 and doc_text[j].isspace():
        if doc_text[j] == "\n":
            nl_count += 1
            if nl_count >= 2:
                return True
        j -= 1
    return False


def _disambiguate(
    doc_text: str,
    candidates: list[int],
    strategy: str,
    section_idx: int,
    n_total: int,
) -> tuple[int, str | None]:
    """Pick one offset from multiple candidates. Returns (chosen_offset, warning).

    Heuristics, applied in order:
      1. Filter to candidates immediately preceded by a paragraph break
         (`\\n\\n`). Section boundaries almost always align with `\\n\\n`.
      2. If multiple candidates survive, pick the one closest to the
         document-relative expected position for this section index: docs with
         internal TOCs (e.g., the Nevada NRS-463) cause each section name to
         appear twice — once early (in the TOC) and once later (real body).
         The expected fraction `(section_idx+1) / (n_total+1)` is a strong
         prior on which copy is the real one.
      3. If still tied (extremely unlikely), pick the earliest. Log a warning
         in any disambiguation case so we can audit.
    """
    para_starts = [p for p in candidates if _has_para_break_before(doc_text, p)]
    pool = para_starts if para_starts else candidates
    pool_label = "para-aligned" if para_starts else "no-para-aligned"

    if len(pool) == 1:
        return pool[0], (
            f"ambiguous@{strategy}: {len(candidates)} matches, picked unique {pool_label} at {pool[0]}"
        )

    # Multiple candidates — use expected-position tie-breaker.
    doc_len = max(1, len(doc_text))
    expected_frac = (section_idx + 1) / (n_total + 1) if n_total > 0 else 0.5
    expected_pos = expected_frac * doc_len
    pool_sorted = sorted(pool, key=lambda p: abs(p - expected_pos))
    chosen = pool_sorted[0]
    return chosen, (
        f"ambiguous@{strategy}: {len(candidates)} matches "
        f"({len(para_starts)} {pool_label}), expected~{int(expected_pos)} "
        f"(idx={section_idx}/{n_total}), picked {chosen} "
        f"(others: {[p for p in pool_sorted[1:]][:3]})"
    )


def _locate_one(
    doc_text: str,
    anchor: str,
    cursor: int,
    section_idx: int,
    n_total: int,
) -> tuple[int, int, str, str | None]:
    """Locate `anchor` in `doc_text[cursor:]` using fallbacks, with uniqueness
    + paragraph-break + expected-position disambiguation at every level.

    Returns (offset, matched_len, strategy, warning). `offset == -1` means total
    failure. `matched_len` is the length of the substring actually located in
    the doc (used to advance the cursor correctly).
    """
    # Strategy 1: exact match on the full anchor
    matches = _find_all(doc_text, anchor, cursor)
    if len(matches) == 1:
        return matches[0], len(anchor), "exact", None
    if len(matches) >= 2:
        chosen, warn = _disambiguate(doc_text, matches, "exact", section_idx, n_total)
        return chosen, len(anchor), "exact-disambig", warn

    # Strategy 2: whitespace-normalized match on the full anchor
    doc_norm = _ws_norm(doc_text)
    anc_norm = _ws_norm(anchor)
    norm_matches = _find_all(doc_norm, anc_norm, 0)
    # Map normalized matches back to doc_text offsets and filter by cursor
    if norm_matches:
        approx_offsets = [
            _norm_offset_to_doc_offset(doc_text, m)
            for m in norm_matches
        ]
        approx_offsets = [o for o in approx_offsets if o >= cursor]
        if len(approx_offsets) == 1:
            return approx_offsets[0], len(anc_norm), "ws-normalized", None
        if len(approx_offsets) >= 2:
            chosen, warn = _disambiguate(
                doc_text, approx_offsets, "ws-normalized", section_idx, n_total
            )
            return chosen, len(anc_norm), "ws-normalized-disambig", warn

    # Strategy 3: progressive prefix shortening (12 → 6 words). Anything shorter
    # than 6 words is too generic and we'd rather skip the section than guess.
    words = anchor.split()
    for n in (12, 10, 8, 6):
        if n >= len(words):
            continue
        prefix = " ".join(words[:n])
        # 3a. exact prefix match
        pmatches = _find_all(doc_text, prefix, cursor)
        if len(pmatches) == 1:
            return pmatches[0], len(prefix), f"prefix-{n}", None
        if len(pmatches) >= 2:
            chosen, warn = _disambiguate(
                doc_text, pmatches, f"prefix-{n}", section_idx, n_total
            )
            return chosen, len(prefix), f"prefix-{n}-disambig", warn
        # 3b. ws-normalized prefix match
        prefix_norm = _ws_norm(prefix)
        pn_matches = _find_all(doc_norm, prefix_norm, 0)
        if pn_matches:
            approx = [
                _norm_offset_to_doc_offset(doc_text, m)
                for m in pn_matches
            ]
            approx = [o for o in approx if o >= cursor]
            if len(approx) == 1:
                return approx[0], len(prefix_norm), f"prefix-{n}-ws", None
            if len(approx) >= 2:
                chosen, warn = _disambiguate(
                    doc_text, approx, f"prefix-{n}-ws", section_idx, n_total
                )
                return chosen, len(prefix_norm), f"prefix-{n}-ws-disambig", warn

    return -1, 0, "no-match", None


def _norm_offset_to_doc_offset(doc_text: str, target_norm_offset: int) -> int:
    """Map a position in the whitespace-normalized version of doc_text back to
    the original doc_text. Walks once. Approximate: returns the original char
    index whose run-of-text-up-to-here has `target_norm_offset` non-collapsed chars.
    """
    norm_len = 0
    prev_was_ws = True
    for i, c in enumerate(doc_text):
        if c.isspace():
            if not prev_was_ws:
                if norm_len == target_norm_offset:
                    return i  # boundary, return position of the whitespace
                norm_len += 1  # the collapsed single space
            prev_was_ws = True
        else:
            if norm_len == target_norm_offset:
                return i
            norm_len += 1
            prev_was_ws = False
    return len(doc_text)


def validate_and_locate(
    doc_text: str, sections: list[dict]
) -> tuple[list[dict], list[dict], list[str]]:
    """For each section, find anchor's offset using fallbacks + disambiguation.

    Returns (located, skipped, warnings). Sections with missing fields, no
    findable anchor, or out-of-order matches are skipped (not fatal). Warnings
    from ambiguous matches are collected so we can audit risky placements.
    """
    located: list[dict] = []
    skipped: list[dict] = []
    warnings: list[str] = []
    cursor = 0
    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            skipped.append({"raw": str(sec)[:80], "reason": "not a dict"})
            continue
        title = sec.get("title")
        anchor = sec.get("anchor")
        if not isinstance(title, str) or not title.strip():
            skipped.append({**sec, "reason": "missing/empty 'title' field"})
            continue
        if not isinstance(anchor, str) or not anchor.strip():
            skipped.append({**sec, "reason": "missing/empty 'anchor' field"})
            continue

        offset, matched_len, strategy, warn = _locate_one(
            doc_text, anchor, cursor, section_idx=i, n_total=len(sections)
        )
        if offset == -1:
            skipped.append({**sec, "reason": f"anchor not found at any fallback level"})
            continue
        if offset < cursor:
            skipped.append({**sec, "reason": f"out-of-order (offset {offset} < cursor {cursor})"})
            continue
        if warn:
            warnings.append(f"section {i} '{title}': {warn}")
        located.append({
            **sec,
            "offset": offset,
            "matched_len": matched_len,
            "strategy": strategy,
        })
        # Advance cursor past the actual matched substring (not just +1) so the
        # next section's search window starts cleanly after this one.
        cursor = offset + matched_len
    return located, skipped, warnings


def build_restructured(doc_text: str, located: list[dict]) -> str:
    """Insert `## <title>` headings before each anchor, leaving content untouched.

    Ensures every inserted heading is on its own line: if the anchor offset
    sits mid-line (the LLM picked a substring not at a paragraph boundary),
    we prefix a `\\n` so `## <title>` starts on a fresh line. Otherwise the
    line-start `## ` scanner in `build_final` misses the heading entirely.
    """
    if not located:
        return doc_text  # no-op
    n = len(doc_text)
    pieces = []
    prev_end = 0
    for sec in located:
        # Clamp the offset to [0, len(doc_text)] — the ws-normalized fallback
        # in validate_and_locate can occasionally return an approximate offset
        # one or two past EOF.
        offset = max(0, min(sec["offset"], n))
        # everything up to the anchor
        pieces.append(doc_text[prev_end:offset])
        # If the char just before the anchor isn't a newline, the inserted
        # heading would land mid-line. Force a newline so `## <title>` is
        # guaranteed to start at line position 0.
        needs_prefix_nl = offset > 0 and doc_text[offset - 1] != "\n"
        prefix = "\n" if needs_prefix_nl else ""
        pieces.append(f"{prefix}## {sec['title']}\n\n")
        prev_end = offset
    pieces.append(doc_text[prev_end:])
    return "".join(pieces)


def _insert_toc_block(restructured: str, toc_block: str) -> str:
    """Insert `toc_block` after YAML frontmatter (if any), else at top.

    `toc_block` should end with a single `\n` and is wrapped with one blank
    line on each side at the seam.
    """
    if restructured.startswith("---\n"):
        idx = restructured.find("\n---\n", 4)
        if idx != -1:
            split = idx + len("\n---\n")
            return restructured[:split] + "\n" + toc_block + "\n" + restructured[split:]
    return toc_block + "\n" + restructured


def build_final(
    doc_text: str, located: list[dict], empty_reason: str | None = None
) -> str:
    """Insert ## headings + a self-consistent TOC.

    If `located` is empty, we still emit a TOC block with a one-line notice so
    every doc has the same top-of-file structure (uniform agent navigation).
    `empty_reason` describes WHY there are no sections:
      - "llm-zero": LLM judged the doc didn't need structuring
      - "all-failed": LLM proposed sections but none survived validation

    Strategy: insert a placeholder TOC of the SAME line count as the real one
    will be (heading + blank + N entries). Compute line numbers on the resulting
    document (so they reflect post-insert positions). Then patch the placeholder
    entries with the real numbers, in place — no rows shift, no offsets needed.
    """
    # Sentinel inserted between TOC and doc body to give the agent an
    # unambiguous "TOC ends here" marker. Plain ASCII so grep/rg can match it
    # cleanly. Two lines added to the TOC block: a blank line + the sentinel.
    BODY_SENTINEL = "=== DOCUMENT BODY ==="

    if not located:
        notice_map = {
            "llm-zero":   "(no sections — short or single-topic document; full text follows)",
            "all-failed": "(sectioning attempted but no anchors validated; full text follows)",
            "llm-empty":  "(sectioning skipped — LLM returned no parseable response; full text follows)",
        }
        notice = notice_map.get(empty_reason or "", notice_map["llm-zero"])
        empty_block = f"## Table of Contents\n\n{notice}\n\n{BODY_SENTINEL}\n"
        return _insert_toc_block(doc_text, empty_block)

    # Track only inserted headings; source Markdown headings are not proposals.
    # Unique temporary single-line headings preserve exactly the final line count.
    import hashlib
    marker = "RISE_HEADING_" + hashlib.sha256(doc_text.encode()).hexdigest() + "_"
    while marker in doc_text or any(marker in str(s) for s in located):
        marker += "X"
    tagged = [{**sec, "title": marker + str(i)} for i, sec in enumerate(located)]
    restructured = build_restructured(doc_text, tagged)

    # One-line TOC entry per section: "- L<start>–<end>: <title> — <description>"
    # The placeholder uses the SAME line count as the real TOC (1 entry / line +
    # the body sentinel), so swapping content in place doesn't shift any
    # line numbers.
    placeholder_entries = [
        f"- L????–????: {sec['title']}"
        + (f" — {sec['description']}" if sec.get("description") else "")
        for sec in located
    ]
    placeholder_block = (
        "## Table of Contents\n\n"
        + "\n".join(placeholder_entries) + "\n"
        + f"\n{BODY_SENTINEL}\n"
    )
    final = _insert_toc_block(restructured, placeholder_block)

    # Compute real line ranges. Ignore the "Table of Contents" heading itself.
    lines = final.split("\n")
    # `wc -l` counts newline characters; if final ends with "\n", split produces
    # a phantom empty last element. Subtract it from total_lines but DON'T mutate
    # `lines` itself (we splice the placeholder back in by index later).
    total_lines = len(lines) - (1 if lines and lines[-1] == "" else 0)

    heading_positions: list[tuple[int, str]] = []  # (1-indexed line, title)
    for i, ln in enumerate(lines):
        if ln.startswith("## " + marker):
            idx = int(ln[len("## " + marker):])
            title = located[idx]["title"]
            if "\n" in title or "\r" in title:
                raise ValueError("Section title must be a single line")
            lines[i] = "## " + title
            heading_positions.append((i + 1, title))
    # Build real entries — pair each heading position with the located dict in
    # order, so we can pull the description through.
    real_entries: list[str] = []
    for idx, (lno, title) in enumerate(heading_positions):
        end_line = (
            heading_positions[idx + 1][0] - 1
            if idx + 1 < len(heading_positions)
            else total_lines
        )
        desc = located[idx].get("description") if idx < len(located) else None
        suffix = f" — {desc}" if desc else ""
        real_entries.append(f"- L{lno}–{end_line}: {title}{suffix}")

    # Replace the placeholder entry lines in place (same count → no row shift).
    # Use the inserted placeholder rather than a similarly named source heading.
    block_start = final.index(placeholder_block)
    toc_start = final[:block_start].count("\n")
    # placeholder layout: [heading, blank, entry_1, ..., entry_N]
    entries_start = toc_start + 2
    entries_end = entries_start + len(located)
    lines[entries_start:entries_end] = real_entries
    return "\n".join(lines)


def main() -> None:
    doc = load_doc(TARGET_DOCID)
    print(f"=== INPUT DOC ===")
    print(f"docid: {doc['docid']}")
    print(f"url:   {doc['url']}")
    print(f"chars: {len(doc['text'])}  lines: {doc['text'].count(chr(10)) + 1}")
    print()
    print(textwrap.indent(doc["text"][:600] + "\n  ...", "  "))
    print(f"  [truncated for preview, full doc is {len(doc['text'])} chars]")
    print()

    print(f"=== LLM CALL ({MODEL}) ===")
    result, usage = call_llm(doc["text"], model=MODEL)
    print(json.dumps(usage, indent=2))
    print()
    print("=== LLM OUTPUT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print()

    sections = result.get("sections", [])

    print("=== VALIDATION (anchor match with fallbacks + disambiguation) ===")
    located, skipped, warnings = validate_and_locate(doc["text"], sections)
    for sec in located:
        print(f"  OK   [{sec['strategy']:30s}] offset={sec['offset']:7d}  '{sec['title']}'")
    for sec in skipped:
        title = sec.get("title", sec.get("raw", "<?>"))
        print(f"  SKIP [{sec['reason']:30s}]                  '{title}'")
    for w in warnings:
        print(f"  WARN {w}")
    print(f"  --> {len(located)} located, {len(skipped)} skipped, {len(warnings)} warnings")
    print()

    # Determine empty_reason for the placeholder TOC in the no-located case.
    # Three sub-cases:
    #   - "llm-empty"   : LLM call returned no parseable JSON (timeout, budget)
    #   - "llm-zero"    : LLM said sections=[] (judged short/single-topic)
    #   - "all-failed"  : LLM proposed sections but none survived validation
    empty_reason: str | None = None
    if not located:
        if "_parse_error" in result:
            empty_reason = "llm-empty"
        elif not sections:
            empty_reason = "llm-zero"
        else:
            empty_reason = "all-failed"
        print(f">>> No sections will be inserted ({empty_reason}).")

    print("=== FINAL RESTRUCTURED DOC ===")
    final = build_final(doc["text"], located, empty_reason=empty_reason)
    out_path = EXP_DIR / f"outputs/sectioning_demo_{doc['docid']}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(final)
    print(f"Wrote {out_path}")
    print(f"Original: {len(doc['text']):,} chars / {doc['text'].count(chr(10))+1} lines")
    print(f"Final:    {len(final):,} chars / {final.count(chr(10))+1} lines")
    print()

    # Sidecar audit log — keeps the doc itself clean while preserving traceability.
    audit = {
        "docid": doc["docid"],
        "url": doc["url"],
        "model": MODEL,
        "usage": usage,
        "llm_section_count": len(sections),
        "located_count": len(located),
        "skipped_count": len(skipped),
        "warning_count": len(warnings),
        "empty_reason": empty_reason,
        "llm_parse_error": result.get("_parse_error"),
        "llm_raw_excerpt": result.get("_raw"),
        "located": [
            {"title": s["title"], "offset": s["offset"], "strategy": s["strategy"]}
            for s in located
        ],
        "skipped": skipped,
        "warnings": warnings,
    }
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"Wrote {audit_path}")
    print()

    print("--- FIRST 80 LINES ---")
    for i, ln in enumerate(final.split("\n")[:80], 1):
        print(f"{i:4d}  {ln}")


if __name__ == "__main__":
    main()
