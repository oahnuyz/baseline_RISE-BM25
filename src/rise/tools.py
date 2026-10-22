"""Agent tools: `bash`, `read`, and `bm25_search`.

`bash` is wrapped with macOS `sandbox-exec` so commands literally cannot read
files outside the mini-corpus directory — the kernel enforces it. macOS-only.

`read` reads files by path relative to a configurable root, with line offset
and limit (DCI / Claude-Code style). Path traversal outside the root is
rejected before opening. The hierarchical sub-agent passes its mini-corpus
as the root (read restricted to the mini-corpus); the single-agent variant
passes the same mini-corpus root (read is restricted to retrieved files).

`bm25_search` runs BM25 over the full corpus and accumulates each
query's top-k matches into the working directory via hardlinks.

Output truncation: bash defaults to ~1000 tokens (≈4000 chars in BPE);
read defaults to 2000 lines.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Sequence

import bm25s

from .retrieval import retrieve


def make_sandbox_profile(mini_corpus_dir: Path) -> str:
    """Build a sandbox-exec profile string.

    Strategy: import macOS's stock `bsd.sb` (sets up mach-lookup, sysctl,
    ipc, etc. correctly for normal POSIX programs), then explicitly:
      - re-allow process-exec* (bsd.sb defaults to restrictive)
      - deny ALL file reads
      - re-allow reads on macOS system dirs (so dyld can load bash, rg, grep)
        and on the mini-corpus subtree (and nothing else)
      - allow writes only to TMPDIR paths (shell scratch) — NOT the mini-corpus,
        which stays read-only

    Result: bash inside the sandbox sees the mini-corpus and the macOS
    system, and gets "Operation not permitted" on any other path including
    /Users/. Network calls (curl, etc.) also fail. Smoke-tested 2026-05-14.
    """
    mc = str(mini_corpus_dir.resolve())
    return f"""(version 1)
(import "bsd.sb")
(allow process-fork)
(allow process-exec*)
(deny file-read*)
(allow file-read-metadata)
(allow file-read* (subpath "/usr"))
(allow file-read* (subpath "/bin"))
(allow file-read* (subpath "/sbin"))
(allow file-read* (subpath "/System"))
(allow file-read* (subpath "/Library"))
(allow file-read* (subpath "/opt"))
(allow file-read* (subpath "/private/var/folders"))
(allow file-read* (subpath "/private/etc"))
(allow file-read* (subpath "/private/tmp"))
(allow file-read* (subpath "{mc}"))
(allow file-write* (subpath "/private/var/folders"))
(allow file-write* (subpath "/private/tmp"))
"""


def _pi_tail_truncate(
    content: str,
    *,
    max_lines: int = 2000,
    max_bytes: int = 50 * 1024,
) -> str:
    """Pi-faithful tail truncation, ported verbatim from
    `pi-mono/.../tools/truncate.ts:truncateTail`.

    Keeps the **last** N lines or M bytes (whichever limit hits first).
    Appends a Pi-style warning line `Truncated: showing X of Y lines`
    (or `Truncated: X lines shown (50.0KB limit)`) when truncation fires.

    The byte counter uses UTF-8 byte length per line + 1 byte per
    newline joiner; lines whose UTF-8 length alone exceeds `max_bytes`
    are partially kept from the END (Pi's edge case).
    """
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return content

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = "lines"
    for i in range(len(lines) - 1, -1, -1):
        if len(out_lines) >= max_lines:
            truncated_by = "lines"
            break
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if out_lines else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            # Edge case: a single line longer than max_bytes. Keep its tail.
            if not out_lines:
                buf = line.encode("utf-8")
                start = max(0, len(buf) - max_bytes)
                # Walk forward to a valid UTF-8 boundary (skip continuation bytes).
                while start < len(buf) and (buf[start] & 0xc0) == 0x80:
                    start += 1
                out_lines.insert(0, buf[start:].decode("utf-8", errors="replace"))
                out_bytes = len(buf) - start
            break
        out_lines.insert(0, line)
        out_bytes += line_bytes

    out = "\n".join(out_lines)
    if truncated_by == "lines":
        warning = f"\n[Truncated: showing {len(out_lines)} of {total_lines} lines]"
    else:
        kb = max_bytes / 1024
        kb_str = f"{kb:.1f}KB" if kb < 1024 else f"{kb/1024:.1f}MB"
        warning = f"\n[Truncated: {len(out_lines)} lines shown ({kb_str} limit)]"
    return out + warning


def make_bash_tool(
    mini_corpus_dir: Path,
    *,
    truncate_chars: int = 4000,
    max_output_lines: int | None = None,
    max_output_bytes: int | None = None,
    enable_sandbox: bool = True,
) -> Callable[[dict], str]:
    """Return a callable(args)->str that runs `bash -c <command>` sandboxed.

    args = {"command": "...", "timeout"?: float (seconds)}

    Timeout handling: the bash_tool_spec advertises an optional `timeout`
    arg. The agent ALWAYS receives this arg description and routinely
    passes e.g. `"timeout": 30`. Earlier versions of this impl silently
    ignored it ("No timeout — user spec") which caused a wedge bug: an
    agent issuing PCRE2 multi-lookahead regexes like
    `rg -P '(?=.*A)(?=.*B)(?=.*C)'` triggers catastrophic backtracking
    and hangs the subprocess for hours, blocking the agent's main thread
    so even wall_clock_timeout can't fire (it checks before the next
    LLM call). Now we enforce:
      - if args["timeout"] is set, use it (capped at HARD_TIMEOUT_S)
      - else, use HARD_TIMEOUT_S as a safety ceiling
    On TimeoutExpired the subprocess is SIGKILL'd and we return an
    error message so the agent can adapt (e.g. simplify the regex).

    Truncation modes:
    - Legacy head-truncate (default): clip output to first `truncate_chars`
      characters. Used by RISE / BCP / hierarchical orchestrator.
    - Pi-faithful tail-truncate (when either `max_output_lines` or
      `max_output_bytes` is set): keep the LAST 2000 lines / 50KB of output,
      verbatim port of Pi's `truncateTail`. Used by `dci_native` to match
      the Pi-based DCI baseline byte-for-byte.
    """
    profile = make_sandbox_profile(mini_corpus_dir) if enable_sandbox else None
    mc_resolved = mini_corpus_dir.resolve()
    pi_tail_mode = max_output_lines is not None or max_output_bytes is not None
    _max_lines = max_output_lines if max_output_lines is not None else 2000
    _max_bytes = max_output_bytes if max_output_bytes is not None else 50 * 1024

    # Server-side safety cap. Agents (esp. small models on hard queries)
    # write expensive PCRE2 multi-lookahead regexes that backtrack for
    # hours. Cap each subprocess; agent's wall_clock then gates total.
    HARD_TIMEOUT_S = 60.0

    def _bash(args: dict) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            return "Error: bash called with empty command."

        raw_to = args.get("timeout")
        try:
            timeout_s = float(raw_to) if raw_to is not None else HARD_TIMEOUT_S
        except (TypeError, ValueError):
            timeout_s = HARD_TIMEOUT_S
        timeout_s = max(1.0, min(timeout_s, HARD_TIMEOUT_S))

        if enable_sandbox:
            cmd = ["/usr/bin/sandbox-exec", "-p", profile, "/bin/bash", "-c", command]
        else:
            cmd = ["/bin/bash", "-c", command]

        # Use Popen + start_new_session + os.killpg instead of
        # subprocess.run(timeout=…). The latter only SIGKILLs the immediate
        # child (bash); descendants like `rg` in a `rg | head` pipeline can
        # outlive bash and get reparented to init (because head closes the
        # pipe before rg notices SIGPIPE). The `finally` killpg sweeps the
        # process group on every exit path. Matches make_pi_bash_tool.
        import signal as _signal
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(mc_resolved),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout_b, stderr_b = proc.communicate(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                return (
                    f"Error: bash subprocess timed out after {timeout_s:.0f}s "
                    f"(SIGKILL'd). Likely catastrophic regex backtracking or an "
                    f"infinite loop. Try a simpler search: avoid PCRE2 multi-"
                    f"lookahead `(?=...)` and `rg --pcre2`; prefer plain regex "
                    f"or multiple narrower searches."
                )
        except FileNotFoundError as e:
            return f"Error: failed to spawn bash: {e}"
        except Exception as e:
            return f"Error: bash subprocess exception: {type(e).__name__}: {e}"
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        stdout = (stdout_b or "").rstrip("\n")
        stderr = (stderr_b or "").rstrip("\n")
        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append("[stderr]\n" + stderr)

        if parts:
            out = "\n".join(parts)
        else:
            # Both empty — annotate by exit code so the agent can tell
            # "command ran, found nothing" (search-style success) from
            # "command crashed".
            if proc.returncode == 0:
                out = "(command succeeded, no output)"
            elif proc.returncode == 1:
                # grep/rg/find convention: exit 1 + empty = no matches.
                out = "(no matches found)"
            else:
                out = f"(no output; exit={proc.returncode})"

        # Non-zero exit WITH output: prefix exit so the agent sees the
        # command's pipeline produced something but also failed.
        if proc.returncode != 0 and parts:
            out = f"[exit={proc.returncode}]\n" + out

        if pi_tail_mode:
            out = _pi_tail_truncate(out, max_lines=_max_lines, max_bytes=_max_bytes)
        elif len(out) > truncate_chars:
            cut = len(out) - truncate_chars
            out = out[:truncate_chars] + f"\n...[truncated; {cut} chars cut from output]"
        return out

    return _bash


def make_read_tool(
    root_dir: Path,
    *,
    default_limit: int = 2000,
    max_line_chars: int = 2000,
) -> Callable[[dict], str]:
    """Return a callable(args)->str that reads files relative to root_dir.

    `root_dir` is the boundary the agent may not escape — pass the
    mini-corpus dir for a corpus-restricted view, or the full corpus root
    to let the agent read any doc by path.

    args = {"file_path": "...", "offset"?: int, "limit"?: int}
    Output is numbered lines (`<line_no>\\t<content>`), like Claude Code's
    read tool.
    """
    root_resolved = root_dir.resolve()

    def _read(args: dict) -> str:
        rel = (args.get("file_path") or "").strip()
        if not rel:
            return "Error: read called with empty file_path."
        # Strip common prefixes agents tend to prepend (bc_plus_docs/, ./).
        for prefix in ("bc_plus_docs/", "./"):
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
        offset = int(args.get("offset") or 0)
        limit = int(args.get("limit") or default_limit)
        offset = max(0, offset)
        limit = max(1, limit)

        candidate = Path(rel)
        target = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root_resolved / candidate).resolve()
        )
        if not target.is_relative_to(root_resolved):
            return f"Error: path {rel!r} escapes the corpus root."
        if not target.exists():
            return (
                f"Error: file not found: {rel!r}. If you expected this file "
                f"to exist, run `search` first with a query that should "
                f"retrieve it — the exact path will appear in the snippet "
                f"preview, then you can read it."
            )
        if not target.is_file():
            return f"Error: not a regular file: {rel!r}"

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error: read failed: {type(e).__name__}: {e}"

        lines = text.splitlines()
        n_total = len(lines)
        selected = lines[offset:offset + limit]
        if not selected:
            return f"(no lines at offset={offset}; total={n_total})"
        formatted: list[str] = []
        for i, line in enumerate(selected, start=offset + 1):
            if len(line) > max_line_chars:
                cut = len(line) - max_line_chars
                line = line[:max_line_chars] + f"...[line truncated; {cut} chars]"
            formatted.append(f"{i:6d}\t{line}")
        out = "\n".join(formatted)
        end = offset + len(selected)
        if end < n_total:
            out += f"\n...[{n_total - end} more lines remain; total {n_total}]"
        return out

    return _read


def bash_tool_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the current working directory. Useful for "
                "searching with `rg`, `grep`, listing with `ls`, `find`, or reading "
                "with `cat`, `head`. Output is truncated; use focused patterns and "
                "flags like `-l` to keep results manageable."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to run.",
                    },
                },
                "required": ["command"],
            },
        },
    }


# ============================================================================
# Pi-faithful tools (DCI baseline). Byte-faithful reproduction of
# `pi-mono/.../tools/{bash,read}.ts`. Do NOT use for our own methods (RISE,
# subagent, orchestrator) — those keep the legacy tools above for backwards
# compat and to avoid changing already-shipped numbers.
# ============================================================================

# Pi-bash spec, verbatim from bash.ts:269-274:
#   description: "Execute a bash command in the current working directory.
#       Returns stdout and stderr. Output is truncated to last 2000 lines or
#       50KB (whichever is hit first). If truncated, full output is saved to
#       a temp file. Optionally provide a timeout in seconds."
#   params: command: string + optional timeout: number
PI_BASH_DEFAULT_MAX_LINES = 2000
PI_BASH_DEFAULT_MAX_BYTES = 50 * 1024


def pi_bash_tool_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                f"Execute a bash command in the current working directory. "
                f"Returns stdout and stderr. Output is truncated to last "
                f"{PI_BASH_DEFAULT_MAX_LINES} lines or "
                f"{PI_BASH_DEFAULT_MAX_BYTES // 1024}KB (whichever is hit "
                f"first). If truncated, full output is saved to a temp file. "
                f"Optionally provide a timeout in seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute",
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Timeout in seconds (optional, no default timeout)"
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    }


def make_pi_bash_tool(
    cwd: Path,
    *,
    max_lines: int = PI_BASH_DEFAULT_MAX_LINES,
    max_bytes: int = PI_BASH_DEFAULT_MAX_BYTES,
    enable_sandbox: bool = False,
) -> Callable[[dict], str]:
    """Pi-faithful bash tool. Mirrors `bash.ts`'s execute() behavior:
    - run command in `cwd` via /bin/bash -c
    - tail-truncate output to last N lines / M bytes (Pi `truncateTail`)
    - save full output to a temp file when truncated; emit notice
      "[Showing lines X-Y of Z. Full output: /tmp/...]"
    - append "Command exited with code N" when exit != 0 (matches Pi's
      reject path which surfaces this string to the agent as an error)
    - optional `timeout` argument (seconds; SIGKILL after timeout, emit
      "Command timed out after N seconds")
    """
    import os as _os
    import tempfile as _tempfile
    cwd_resolved = cwd.resolve()
    profile = make_sandbox_profile(cwd) if enable_sandbox else None

    # 10-minute hard cap: prevents catastrophic-backtracking rg patterns
    # (e.g. multi-lookahead PCRE) from spinning a CPU core indefinitely.
    # Mirrors the pi-mono bash.ts patch in dist/.
    HARD_TIMEOUT_S = 600.0

    # Truncation overflow temp files written by this tool instance. Kept
    # so the runner can `cleanup_temp_files()` at end-of-query and avoid
    # accumulating GBs of pi-bash-*.log across runs. Pi-faithful behavior
    # (writing the full output to /tmp and surfacing the path in the
    # agent-visible string) is preserved during the query.
    _temp_files: list[str] = []

    def _bash(args: dict) -> str:
        command = (args.get("command") or "").strip()
        if not command:
            return "Error: bash called with empty command."
        raw_to = args.get("timeout")
        try:
            timeout = float(raw_to) if raw_to is not None else HARD_TIMEOUT_S
        except (TypeError, ValueError):
            timeout = HARD_TIMEOUT_S
        timeout = max(1.0, min(timeout, HARD_TIMEOUT_S))

        if enable_sandbox:
            cmd = ["/usr/bin/sandbox-exec", "-p", profile, "/bin/bash", "-c", command]
        else:
            cmd = ["/bin/bash", "-c", command]

        # We use Popen + start_new_session + os.killpg on timeout instead of
        # subprocess.run(timeout=…) because the latter only SIGKILLs the
        # immediate child (bash); its descendants (e.g. rg) get reparented to
        # init and keep burning CPU. start_new_session puts bash in its own
        # process group, and killpg propagates SIGKILL to all grandchildren.
        # We also defensively killpg in the `finally` block — bash normally
        # waits for all pipeline children, but for `rg ... | head` style
        # commands head can close the pipe early, and if rg is in a
        # CPU-bound section it may not check for SIGPIPE before bash
        # decides to exit. In that race rg gets reparented to init and
        # keeps scanning. Re-issuing killpg on the (still-active) pgid
        # after the parent exits cleans up that case.
        import signal as _signal
        timed_out = False
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd_resolved),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
                returncode = proc.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                # Kill the entire process group so rg / grep / etc. die too.
                try:
                    _os.killpg(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                # Drain any output already buffered (best effort).
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    stdout = stderr = ""
                full = ((stdout or "").rstrip("\n") + ("\n" + (stderr or "").rstrip("\n") if stderr else "")).rstrip("\n")
                return (full + ("\n\n" if full else "") + f"Command timed out after {int(timeout)} seconds")
        except FileNotFoundError as e:
            return f"Error: failed to spawn bash: {e}"
        except Exception as e:
            return f"Error: bash subprocess exception: {type(e).__name__}: {e}"
        finally:
            # Always sweep the process group, even on normal return. This
            # catches `rg | head` orphans (see header comment); a no-op
            # when bash already reaped its children.
            if proc is not None:
                try:
                    _os.killpg(proc.pid, _signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

        stdout = (stdout or "").rstrip("\n")
        stderr = (stderr or "").rstrip("\n")
        # Pi's bash combines stdout + stderr into a single stream (both
        # are piped to the same `onData` handler in createLocalBashOperations).
        # Match that — DON'T prefix stderr with "[stderr]".
        if stdout and stderr:
            full = stdout + "\n" + stderr
        else:
            full = stdout or stderr

        # Apply Pi's truncateTail. Get back content + truncation metadata.
        total_lines = full.count("\n") + (1 if full else 0)
        total_bytes_full = len(full.encode("utf-8"))
        truncated_text, was_truncated, trunc_meta = _pi_tail_truncate_with_meta(
            full, max_lines=max_lines, max_bytes=max_bytes,
        )

        # If truncated, save full output to temp file (Pi behavior).
        temp_path: str | None = None
        if was_truncated and full:
            try:
                fd, temp_path = _tempfile.mkstemp(prefix="pi-bash-", suffix=".log")
                with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(full)
                _temp_files.append(temp_path)
            except OSError:
                temp_path = None

        out = truncated_text or "(no output)"

        if was_truncated:
            # Pi emits "[Showing lines X-Y of Z. Full output: /tmp/...]"
            # where X = totalLines - outputLines + 1, Y = totalLines.
            output_lines = trunc_meta["output_lines"]
            output_bytes = trunc_meta["output_bytes"]
            start_line = total_lines - output_lines + 1
            end_line = total_lines
            tp_str = temp_path or "(temp file unavailable)"
            if trunc_meta.get("last_line_partial"):
                last_line_bytes = total_bytes_full
                kb_str = (f"{output_bytes/1024:.1f}KB" if output_bytes < 1024*1024
                          else f"{output_bytes/1024/1024:.1f}MB")
                full_line_kb = (f"{last_line_bytes/1024:.1f}KB" if last_line_bytes < 1024*1024
                                else f"{last_line_bytes/1024/1024:.1f}MB")
                out += (
                    f"\n\n[Showing last {kb_str} of line {end_line} "
                    f"(line is {full_line_kb}). Full output: {tp_str}]"
                )
            elif trunc_meta["truncated_by"] == "lines":
                out += (
                    f"\n\n[Showing lines {start_line}-{end_line} of "
                    f"{total_lines}. Full output: {tp_str}]"
                )
            else:
                kb_str = (f"{max_bytes/1024:.1f}KB" if max_bytes < 1024*1024
                          else f"{max_bytes/1024/1024:.1f}MB")
                out += (
                    f"\n\n[Showing lines {start_line}-{end_line} of "
                    f"{total_lines} ({kb_str} limit). Full output: {tp_str}]"
                )

        # Pi appends "Command exited with code N" when exit != 0.
        if not timed_out and proc.returncode != 0:
            out += f"\n\nCommand exited with code {proc.returncode}"

        return out

    def _cleanup_temp_files() -> int:
        """Delete every truncation-overflow temp file this tool instance
        wrote. Called by the per-query runner in a try/finally so the
        cleanup runs whether the agent loop succeeded, errored, or hit
        wall_clock_timeout. Returns the number of files removed.

        Safe to call multiple times; missing files are silently skipped."""
        removed = 0
        for p in _temp_files:
            try:
                _os.unlink(p)
                removed += 1
            except OSError:
                pass
        _temp_files.clear()
        return removed

    _bash.cleanup_temp_files = _cleanup_temp_files  # type: ignore[attr-defined]
    return _bash


def _pi_tail_truncate_with_meta(
    content: str,
    *,
    max_lines: int = 2000,
    max_bytes: int = 50 * 1024,
) -> tuple[str, bool, dict]:
    """Same algorithm as `_pi_tail_truncate` but returns truncation metadata
    (output_lines, output_bytes, truncated_by, last_line_partial) so the
    caller can format Pi's "[Showing lines X-Y of Z. Full output: ...]"
    notice instead of the shorter `[Truncated: ...]` marker."""
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return content, False, {
            "truncated_by": None, "output_lines": total_lines,
            "output_bytes": total_bytes, "last_line_partial": False,
        }

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = "lines"
    last_line_partial = False
    for i in range(len(lines) - 1, -1, -1):
        if len(out_lines) >= max_lines:
            truncated_by = "lines"
            break
        line = lines[i]
        line_bytes = len(line.encode("utf-8")) + (1 if out_lines else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            if not out_lines:
                buf = line.encode("utf-8")
                start = max(0, len(buf) - max_bytes)
                while start < len(buf) and (buf[start] & 0xc0) == 0x80:
                    start += 1
                out_lines.insert(0, buf[start:].decode("utf-8", errors="replace"))
                out_bytes = len(buf) - start
                last_line_partial = True
            break
        out_lines.insert(0, line)
        out_bytes += line_bytes

    return "\n".join(out_lines), True, {
        "truncated_by": truncated_by,
        "output_lines": len(out_lines),
        "output_bytes": out_bytes,
        "last_line_partial": last_line_partial,
    }


# Pi-read spec, verbatim from read.ts:17-21, 121-126:
#   description: "Read the contents of a file. Supports text files and
#       images (jpg, png, gif, webp). Images are sent as attachments. For
#       text files, output is truncated to 2000 lines or 50KB (whichever
#       is hit first). Use offset/limit for large files. When you need the
#       full file, continue with offset until complete."
#   params: path: string, optional offset: number (1-indexed), optional limit: number
PI_READ_DEFAULT_MAX_LINES = 2000
PI_READ_DEFAULT_MAX_BYTES = 50 * 1024


def pi_read_tool_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                f"Read the contents of a file. Supports text files and "
                f"images (jpg, png, gif, webp). Images are sent as "
                f"attachments. For text files, output is truncated to "
                f"{PI_READ_DEFAULT_MAX_LINES} lines or "
                f"{PI_READ_DEFAULT_MAX_BYTES // 1024}KB (whichever is hit "
                f"first). Use offset/limit for large files. When you need "
                f"the full file, continue with offset until complete."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the file to read (relative or absolute)"
                        ),
                    },
                    "offset": {
                        "type": "number",
                        "description": "Line number to start reading from (1-indexed)",
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of lines to read",
                    },
                },
                "required": ["path"],
            },
        },
    }


def _pi_head_truncate(
    content: str,
    *,
    max_lines: int = PI_READ_DEFAULT_MAX_LINES,
    max_bytes: int = PI_READ_DEFAULT_MAX_BYTES,
) -> tuple[str, bool, dict]:
    """Port of `truncate.ts:truncateHead`. Keeps the FIRST N lines / M bytes.

    Returns (content, truncated, {output_lines, truncated_by,
    first_line_exceeds_limit}).
    """
    total_bytes = len(content.encode("utf-8"))
    lines = content.split("\n")
    total_lines = len(lines)
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return content, False, {
            "truncated_by": None, "output_lines": total_lines,
            "first_line_exceeds_limit": False,
        }

    # First-line-too-long edge case (Pi returns empty content).
    if lines:
        first_line_bytes = len(lines[0].encode("utf-8"))
        if first_line_bytes > max_bytes:
            return "", True, {
                "truncated_by": "bytes", "output_lines": 0,
                "first_line_exceeds_limit": True,
            }

    out_lines: list[str] = []
    out_bytes = 0
    truncated_by = "lines"
    for i, line in enumerate(lines):
        if i >= max_lines:
            truncated_by = "lines"
            break
        line_bytes = len(line.encode("utf-8")) + (1 if i > 0 else 0)
        if out_bytes + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        out_lines.append(line)
        out_bytes += line_bytes
    if len(out_lines) >= max_lines and out_bytes <= max_bytes:
        truncated_by = "lines"

    return "\n".join(out_lines), True, {
        "truncated_by": truncated_by, "output_lines": len(out_lines),
        "first_line_exceeds_limit": False,
    }


def make_pi_read_tool(
    cwd: Path,
    *,
    max_lines: int = PI_READ_DEFAULT_MAX_LINES,
    max_bytes: int = PI_READ_DEFAULT_MAX_BYTES,
) -> Callable[[dict], str]:
    """Pi-faithful read tool. Mirrors `read.ts:execute`:
    - accepts `path` (NOT `file_path`); resolves relative-to-cwd or
      absolute (Pi's `resolveReadPath`).
    - 1-indexed `offset`; default = start at line 1.
    - optional `limit`; default = read to EOF (then truncateHead caps).
    - output is **raw lines joined by '\\n'** — NO line-number prefix.
      (Our legacy `make_read_tool` adds line numbers; Pi does NOT.)
    - truncation notice in Pi's exact format:
        "[Showing lines X-Y of Z. Use offset=N to continue.]"
        "[Showing lines X-Y of Z (50.0KB limit). Use offset=N to continue.]"
        "[Line N is XKB, exceeds 50.0KB limit. Use bash: sed ...]"
        "[K more lines in file. Use offset=N to continue.]"

    Note: Pi does NOT restrict reads to cwd (its `resolveReadPath` accepts
    any absolute path). We replicate that for byte-faithfulness; the agent
    is constrained by the corpus prompt, not by tool-level sandboxing.
    """
    cwd_resolved = cwd.resolve()

    def _read(args: dict) -> str:
        # Pi uses `path`. Accept `file_path` as a legacy alias for
        # robustness, but prefer `path`.
        rel = (args.get("path") or args.get("file_path") or "").strip()
        if not rel:
            return "Error: read called with empty path."
        # Resolve like Pi's resolveToCwd: absolute paths kept; relatives
        # joined to cwd. Expand ~.
        from os.path import expanduser, isabs
        rel_expanded = expanduser(rel)
        if isabs(rel_expanded):
            target = Path(rel_expanded).resolve()
        else:
            target = (cwd_resolved / rel_expanded).resolve()
        if not target.exists():
            return f"Error: file not found: {rel!r}"
        if not target.is_file():
            return f"Error: not a regular file: {rel!r}"

        # Optional offset / limit (1-indexed).
        offset = args.get("offset")
        limit = args.get("limit")
        try:
            offset = int(offset) if offset is not None else None
        except (TypeError, ValueError):
            offset = None
        try:
            limit = int(limit) if limit is not None else None
        except (TypeError, ValueError):
            limit = None

        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error: read failed: {type(e).__name__}: {e}"

        all_lines = text.split("\n")
        total_lines = len(all_lines)
        # offset: 1-indexed line number; convert to 0-indexed array start.
        start_line = max(0, (offset - 1)) if offset else 0
        start_line_display = start_line + 1
        if start_line >= total_lines:
            return f"Error: Offset {offset} is beyond end of file ({total_lines} lines total)"

        if limit is not None:
            end_line = min(start_line + limit, total_lines)
            selected = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected = "\n".join(all_lines[start_line:])
            user_limited_lines = None

        truncated, was_truncated, meta = _pi_head_truncate(
            selected, max_lines=max_lines, max_bytes=max_bytes,
        )

        # Pi's exact format-result branches.
        if meta.get("first_line_exceeds_limit"):
            first_line_bytes = len(all_lines[start_line].encode("utf-8"))
            kb = (f"{first_line_bytes/1024:.1f}KB"
                  if first_line_bytes < 1024*1024
                  else f"{first_line_bytes/1024/1024:.1f}MB")
            limit_kb = f"{max_bytes/1024:.1f}KB"
            return (
                f"[Line {start_line_display} is {kb}, exceeds {limit_kb} "
                f"limit. Use bash: sed -n '{start_line_display}p' {rel} "
                f"| head -c {max_bytes}]"
            )
        if was_truncated:
            output_lines = meta["output_lines"]
            end_line_display = start_line_display + output_lines - 1
            next_offset = end_line_display + 1
            if meta["truncated_by"] == "lines":
                return (
                    truncated
                    + f"\n\n[Showing lines {start_line_display}-"
                    f"{end_line_display} of {total_lines}. Use "
                    f"offset={next_offset} to continue.]"
                )
            else:
                kb_str = f"{max_bytes/1024:.1f}KB"
                return (
                    truncated
                    + f"\n\n[Showing lines {start_line_display}-"
                    f"{end_line_display} of {total_lines} ({kb_str} "
                    f"limit). Use offset={next_offset} to continue.]"
                )
        if user_limited_lines is not None and start_line + user_limited_lines < total_lines:
            remaining = total_lines - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            return (
                truncated
                + f"\n\n[{remaining} more lines in file. Use "
                f"offset={next_offset} to continue.]"
            )
        return truncated

    return _read


def read_tool_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read a file by path with optional line offset and limit. "
                "Returns numbered lines. The default limit (2000) is usually "
                "sufficient to read most documents in one call — prefer that "
                "to small limits, since the first 50-100 lines of many "
                "documents are navigation/template chrome, not content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path relative to the working directory.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-indexed). Default 0.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Default 2000; prefer the default unless you have a specific reason to read only a slice.",
                    },
                },
                "required": ["file_path"],
            },
        },
    }


# --- passage-mode tools (read_doc + read_passage) ----------------------
#
# In passage-mode the agent sees passage paths like
# `<domain>/<title>/passages/p0042.txt` in search previews and `rg`
# output. Two read tools wrap this:
#   - `read_doc`: reads the full parent doc. Accepts the parent relpath
#     (`<domain>/<title>.txt`). If given a passage relpath, auto-strips
#     `/passages/pNN.txt` and reads the parent, with a one-line warning
#     so the agent learns the right surface instead of silently failing.
#     Does NOT accept docids directly; if the agent wants to read by
#     docid, it should grep `_meta.json` or use the passage map.
#   - `read_passage`: reads a single passage. Accepts a passage_id
#     (`5412_p0042`) or passage relpath. Prefixes the body with
#     `[from <parent_relpath>]` so isolated passage reads stay grounded.

_PASSAGE_RELPATH_RE = re.compile(r"^(.+?)/passages/p\d+\.txt$")


def _coerce_passage_relpath_to_parent(rel: str) -> tuple[str, str | None]:
    """If `rel` looks like a passage relpath, strip `/passages/pNN.txt`
    and return the parent relpath. Returns (parent_relpath, warning_msg)
    where warning_msg is non-None when coercion happened.
    """
    m = _PASSAGE_RELPATH_RE.match(rel)
    if not m:
        return rel, None
    parent = m.group(1) + ".txt"
    warning = (
        f"Note: {rel!r} is a passage path. read_doc reads the parent document; "
        f"coerced to {parent!r}. Use read_passage if you only want one passage."
    )
    return parent, warning


def make_read_doc_tool(
    root_dir: Path,
    *,
    default_limit: int = 2000,
    max_line_chars: int = 2000,
) -> Callable[[dict], str]:
    """Like make_read_tool, but accepts passage relpaths and silently
    coerces them to the parent doc relpath (with a one-line warning in
    the output). Use this in passage-mode runs as the full-doc read
    surface.
    """
    inner = make_read_tool(root_dir, default_limit=default_limit, max_line_chars=max_line_chars)

    def _read_doc(args: dict) -> str:
        rel = (args.get("file_path") or "").strip()
        if not rel:
            return "Error: read_doc called with empty file_path."
        coerced, warning = _coerce_passage_relpath_to_parent(rel)
        if warning:
            new_args = dict(args)
            new_args["file_path"] = coerced
            body = inner(new_args)
            return warning + "\n" + body
        return inner(args)

    return _read_doc


def read_doc_tool_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": (
                "Read a parent document by path. Useful when one passage's "
                "context is insufficient and you need surrounding sections. "
                "Accepts a parent relpath like `<domain>/<title>.txt`. If "
                "you pass a passage relpath (`<domain>/<title>/passages/"
                "p0042.txt`) it auto-coerces to the parent doc with a "
                "warning. Returns numbered lines; default limit (2000) is "
                "usually enough."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Parent doc relpath (or passage relpath, auto-coerced).",
                    },
                    "offset": {"type": "integer", "description": "0-indexed line offset (default 0)."},
                    "limit": {"type": "integer", "description": "Max lines to read (default 2000)."},
                },
                "required": ["file_path"],
            },
        },
    }


def make_read_passage_tool(
    passage_files_root: Path,
    *,
    passage_id_to_relpath: dict[str, str],
    relpath_to_parent_docid: dict[str, str],
    max_line_chars: int = 2000,
) -> Callable[[dict], str]:
    """Return a callable(args)->str that reads a passage file.

    args = {"target": "5412_p0042"} OR {"target": "<domain>/<title>/passages/p0042.txt"}
    Returns the passage body prefixed with `[from <parent_relpath>]`.
    No offset/limit — passages are bounded by construction.
    """
    root_resolved = passage_files_root.resolve()

    def _read_passage(args: dict) -> str:
        target = (args.get("target") or args.get("passage_id") or args.get("file_path") or "").strip()
        if not target:
            return "Error: read_passage called with empty target."

        # Resolve target → passage relpath.
        if target in passage_id_to_relpath:
            relpath = passage_id_to_relpath[target]
        else:
            # Strip common prefixes.
            t = target
            for prefix in ("bcp_passages_100k_files/", "./"):
                if t.startswith(prefix):
                    t = t[len(prefix):]
            relpath = t

        # Materialize parent relpath (for the `[from …]` annotation) by
        # coercing passage relpath → parent. `relpath_to_parent_docid`
        # keys are PARENT relpaths (`.txt`), so look up after coercion.
        parent_relpath, _ = _coerce_passage_relpath_to_parent(relpath)
        parent_docid = relpath_to_parent_docid.get(parent_relpath, "?")

        candidate = Path(relpath)
        full = (
            candidate.resolve()
            if candidate.is_absolute()
            else (root_resolved / candidate).resolve()
        )
        if not str(full).startswith(str(root_resolved)):
            return f"Error: passage path {relpath!r} escapes the passage root."
        if not full.exists():
            return (
                f"Error: passage not found: {relpath!r}. Pass either a "
                f"passage_id like `<docid>_p0042` or a passage relpath "
                f"surfaced from search/rg."
            )

        try:
            text = full.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error: read_passage failed: {type(e).__name__}: {e}"

        header = f"[from {parent_relpath} (docid={parent_docid})]"
        # Long-line guard
        lines = text.splitlines()
        out: list[str] = []
        for line in lines:
            if len(line) > max_line_chars:
                cut = len(line) - max_line_chars
                line = line[:max_line_chars] + f"...[line truncated; {cut} chars]"
            out.append(line)
        return header + "\n" + "\n".join(out)

    return _read_passage


def read_passage_tool_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "read_passage",
            "description": (
                "Read one passage. Accepts either a passage_id like "
                "`5412_p0042` (canonical) or a passage relpath like "
                "`<domain>/<title>/passages/p0042.txt` (as surfaced by "
                "search and `rg -l`). Returns the bounded passage body "
                "prefixed with the parent doc relpath. Prefer this over "
                "read_doc when one passage suffices — it's bounded and "
                "boilerplate-free by construction. Use read_doc if you "
                "need cross-passage context within the same doc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Passage_id (e.g. `5412_p0042`) or passage relpath.",
                    },
                },
                "required": ["target"],
            },
        },
    }


def _read_snippet(path: Path, max_chars: int = 100) -> str:
    """First ~max_chars of a doc with whitespace collapsed (Google-style snippet)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(max_chars * 4)  # read extra in case of much whitespace
    except Exception:
        return "(unreadable)"
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars] + "…"
    return text


def make_bm25_search_tool(
    *,
    searcher: bm25s.BM25,
    doc_ids: Sequence[str],
    docid_to_relpath: dict[str, str],
    bc_plus_docs_root: Path,
    working_dir: Path,
    k: int = 100,
    top_n_preview: int = 10,
    snippet_chars: int = 100,
) -> Callable[[dict], str]:
    """Return a callable(args)->str that runs BM25 with one or more queries
    and ACCUMULATES the UNION of each query's top-k matches into working_dir
    (hardlinked, deduped). Previously retrieved docs from earlier search
    tool calls remain available — bash/grep can range over the full history
    of retrieved evidence in this session.

    args = {"queries": ["...", "...", ...]}  — also accepts {"queries": "..."}
    or legacy {"query": "..."} for tolerance.

    Each call's preview reports both NEW docs added by this call's queries
    and TOTAL working-dir size, so the agent can see whether a search is
    bringing fresh evidence or restating what's already on disk.
    """
    wd_resolved = working_dir.resolve()
    corpus_resolved = bc_plus_docs_root.resolve()

    def _coerce_queries(args: dict) -> tuple[list[str], str]:
        """Tolerate several malformed shapes the model may emit:
          - dict {"queries": ["a","b"]}    → ["a","b"]
          - dict {"queries": "a"}          → ["a"]
          - dict {"queries": '["a","b"]'}  → ["a","b"] (stringified JSON list)
          - legacy {"query": "a"}          → ["a"]

        Returns (queries, notice). `notice` is a short string the caller can
        prepend to the tool output when the input shape was non-standard, so
        the model learns the correct schema instead of silently relying on
        our tolerance.
        """
        notice = ""
        q = args.get("queries")
        if q is None:
            q = args.get("query")
            if q is not None:
                notice = (
                    "Note: `query` is a legacy single-string parameter. "
                    "Prefer `queries` as a list of strings, e.g. "
                    "{\"queries\": [\"first query\", \"second query\"]}."
                )
        if q is None:
            return [], notice
        if isinstance(q, str):
            s = q.strip()
            if not s:
                return [], notice
            # JSON-list-as-string: agent dumped a list literal into one string.
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        notice = (
                            "Note: `queries` was passed as a stringified JSON "
                            "list. Pass an actual JSON array instead, e.g. "
                            "{\"queries\": [\"q1\", \"q2\"]} (not "
                            "{\"queries\": \"[\\\"q1\\\", \\\"q2\\\"]\"})."
                        )
                        return [str(x).strip() for x in parsed if str(x).strip()], notice
                except json.JSONDecodeError:
                    pass
            return [s], notice
        if isinstance(q, (list, tuple)):
            return [str(x).strip() for x in q if str(x).strip()], notice
        return [], notice

    def _search(args: dict) -> str:
        queries, notice = _coerce_queries(args)
        if not queries:
            err = "Error: search called without any queries."
            return f"{notice}\n{err}" if notice else err

        wd_resolved.mkdir(parents=True, exist_ok=True)

        # Snapshot what's already in the working dir BEFORE this call so we
        # can report per-query new-vs-already numbers.
        pre_existing: set[str] = set()
        for p in wd_resolved.rglob("*.txt"):
            try:
                pre_existing.add(str(p.relative_to(wd_resolved)))
            except ValueError:
                continue

        # Per query, retrieve top-k.
        per_query_results: list[list[tuple[str, float]]] = []  # (relpath, score)
        all_relpaths: set[str] = set()
        for q in queries:
            results = retrieve(searcher, doc_ids, q, k=k)
            top: list[tuple[str, float]] = []
            for docid, score in results:
                rel = docid_to_relpath.get(str(docid))
                if not rel:
                    continue
                top.append((rel, score))
                all_relpaths.add(rel)
            per_query_results.append(top)

        # Hardlink the union.  Skip docs that already exist in working_dir
        # (we accumulate; previous contents are kept).
        n_new = 0
        for rel in all_relpaths:
            src = corpus_resolved / rel
            if not src.exists():
                continue
            dst = wd_resolved / rel
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
                n_new += 1
            except FileExistsError:
                pass

        # Compute new working_dir size (cheap walk; mini-corpus is small).
        total_in_wd = sum(1 for _ in wd_resolved.rglob("*.txt"))

        # Build per-query preview.
        n_queries = len(queries)
        header = (
            f"search ({n_queries} {'query' if n_queries == 1 else 'queries'}): "
            f"{n_new} new docs added; working directory now contains {total_in_wd} docs."
        )
        lines: list[str] = []
        if notice:
            lines.append(notice)
            lines.append("")
        lines.append(header)
        for i, (q, top) in enumerate(zip(queries, per_query_results), 1):
            preview = top[:top_n_preview]
            n_already = sum(1 for rel, _ in top if rel in pre_existing)
            lines.append("")
            lines.append(
                f'Query {i}: "{q}"  — {len(top)} matched '
                f'({len(top) - n_already} new, {n_already} already in working dir); '
                f'showing top {len(preview)}:'
            )
            if not preview:
                lines.append("  (no matches)")
                continue
            for j, (rel, score) in enumerate(preview, 1):
                tag = "" if rel not in pre_existing else " [already in wd]"
                snippet = _read_snippet(corpus_resolved / rel, snippet_chars)
                lines.append(f"  {j:>2}. [{score:.2f}] {rel}{tag}")
                lines.append(f'       "{snippet}"')
        return "\n".join(lines)

    return _search


def bm25_search_tool_spec() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search the corpus with one or more queries simultaneously. "
                "The top matching documents (union across all queries) are "
                "added to your working directory; documents already there "
                "from earlier searches are kept (the working directory "
                "accumulates evidence across this session). Returns a "
                "per-query top-10 preview with file paths and short snippets, "
                "and the new total working-directory size. Pass multiple "
                "complementary queries in one call. Specific distinctive "
                "terms (rare phrases, proper nouns, distinctive number/date "
                "combinations) tend to give the best results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "One or more search queries.",
                    },
                },
                "required": ["queries"],
            },
        },
    }


def copy_files_into(
    src_root: Path,
    relpaths: list[str],
    dst_dir: Path,
) -> tuple[int, list[str]]:
    """COPY (not symlink) selected files from src_root → dst_dir/<relpath>.

    Returns (n_copied, list_of_relpaths_actually_copied).
    """
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    added: list[str] = []
    for rel in relpaths:
        src = src_root / rel
        if not src.exists():
            continue
        dst = dst_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(src, dst)
            n += 1
            added.append(rel)
        except FileExistsError:
            pass
    return n, added
