"""verify: the project's KNOWN gate as a first-class verb.

Council ruling (2026-11-06): a verify that GUESSES the command is a false
green. Discovery rides agent.coding_context.detect_project_facts — the same
single source of truth behind the workspace snapshot's "Verify:" line. A
project with no known gate gets an honest refusal pointing at terminal.
No compaction logic lives here; the result is a bounded envelope (command,
exit code, pass/fail, byte counts, tail + honest truncation).
"""

import os
import signal
import subprocess
import time
from contextlib import suppress
from pathlib import Path

from tools.registry import registry

_OUTPUT_TAIL_CHARS = 6_000
_DEFAULT_TIMEOUT = 300.0
_MAX_TIMEOUT = 600.0
# Read once at import: tests monkeypatch this seam, and patching os.name itself
# poisons pathlib (Path.is_dir() goes False) — never do that.
_IS_POSIX = os.name == "posix"

VERIFY_SCHEMA = {
    "name": "verify",
    "description": (
        "Run the project's KNOWN verify gate and return a structured pass/fail verdict. "
        "Commands are discovered from the repo (Makefile check/test/lint targets, "
        "scripts/run_tests.sh, package.json scripts, pytest config) — the same discovery "
        "behind the workspace snapshot's Verify line. Never guesses: with no known gate "
        "it refuses and you use terminal. Result: command, exit_code, passed, "
        "duration_seconds, stdout/stderr byte counts, bounded output_tail, truncated flag, "
        "and commands_discovered (re-invoke with command_index to run another gate)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project root (default: the session's working directory; falls back to this process's cwd when unrecorded)",
                "default": ".",
            },
            "command_index": {
                "type": "integer",
                "description": "Index into the discovered gate list (default 0 = first gate)",
                "default": 0,
                "minimum": 0,
            },
            "timeout": {
                "type": "number",
                "description": "Seconds before the gate is killed (default 300, max 600)",
                "default": 300,
                "minimum": 1,
                "maximum": 600,
            },
        },
        "required": [],
    },
}


def _handle_verify(args, **kw):
    root_arg = args.get("path")
    root = None
    if root_arg:
        root = Path(root_arg).expanduser()
    else:
        # Default to the SESSION's cwd, not this process's — the gateway process
        # sits somewhere unrelated to the workspace the model is working in. Same
        # precedence read_file uses for relative paths (file_tools_paths): recorded
        # session cwd, registered task override, scoped TERMINAL_CWD, process cwd.
        with suppress(Exception):
            from agent.runtime_cwd import scope_terminal_cwd
            from tools.terminal_tool import get_session_cwd, resolve_task_overrides
            key = kw.get("task_id")
            for candidate in (get_session_cwd(key),
                              resolve_task_overrides(key).get("cwd"),
                              scope_terminal_cwd()):
                text = str(candidate or "").strip()
                if text:
                    root = Path(text).expanduser()
                    break
        if root is None:
            root = Path.cwd()
    if not root.is_dir():
        return {"error": f"verify: project root not found: {root_arg}"}
    from agent.coding_context import detect_project_facts

    commands = detect_project_facts(root).verify_commands
    if not commands:
        return {
            "error": (
                "verify: no known gate for this project (checked Makefile "
                "check/test/lint targets, scripts/run_tests.sh, package.json scripts, "
                "pytest config). Run the project's check explicitly with terminal — "
                "a guessed gate is a false green."
            )
        }
    idx = args.get("command_index", 0)
    if not isinstance(idx, int) or idx < 0 or idx >= len(commands):
        return {
            "error": f"verify: command_index {idx!r} out of range; discovered gates: "
            + "; ".join(f"[{i}] {c}" for i, c in enumerate(commands))
        }
    cmd = commands[idx]
    try:
        timeout = min(float(args.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT
    started = time.monotonic()
    # Own process group: on timeout the whole tree dies (make spawns workers —
    # killing only the shell would orphan them mid-test-run). POSIX only; the
    # Windows gateway degrades to killing the shell itself.
    posix = _IS_POSIX
    proc = subprocess.Popen(
        cmd, shell=True, cwd=str(root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", start_new_session=posix)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        out, err = "", ""  # first communicate() raised before assigning
        if posix:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)  # start_new_session: pgid == pid
        else:
            with suppress(OSError):
                proc.kill()
        # A grandchild that daemonized out of the group can still hold the pipe
        # open — a second bounded wait keeps the verdict itself from hanging.
        with suppress(subprocess.TimeoutExpired, OSError):
            out, err = proc.communicate(timeout=5)
        partial = ((out or "") + (("\n" + err) if err else ""))[-_OUTPUT_TAIL_CHARS:]
        return {
            "command": cmd, "passed": False, "timeout": True,
            "exit_code": proc.returncode,
            "duration_seconds": round(time.monotonic() - started, 2),
            "output_tail": partial,
            "truncated": True,
            "error": f"verify: gate exceeded {timeout:.0f}s and was killed; "
            "re-run with a larger timeout or use terminal.",
            "commands_discovered": commands,
        }
    except OSError as exc:
        return {"error": f"verify: could not execute {cmd!r}: {exc}"}
    duration = round(time.monotonic() - started, 2)
    out, err = out or "", err or ""
    combined = out + ("\n" + err if err else "")
    return {
        "command": cmd,
        "passed": proc.returncode == 0,
        "exit_code": proc.returncode,
        "duration_seconds": duration,
        "stdout_bytes": len(out.encode("utf-8", "replace")),
        "stderr_bytes": len(err.encode("utf-8", "replace")),
        "output_tail": combined[-_OUTPUT_TAIL_CHARS:],
        "truncated": len(combined) > _OUTPUT_TAIL_CHARS,
        "total_output_chars": len(combined),
        "commands_discovered": commands,
    }


registry.register(
    name="verify", toolset="file", schema=VERIFY_SCHEMA, handler=_handle_verify,
    emoji="✅", execution_timeout=_MAX_TIMEOUT + 60)
