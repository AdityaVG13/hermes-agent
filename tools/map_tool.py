"""map: first-contact repo skeleton — one bounded call, everything chained.

Council ruling (2026-11-06): the 8th verb. A first read of a project is ONE
bounded result the model can act on: layout, manifests, package managers, entry
points, gates (same discovery space as verify's command_index), and context
files as read_file refs — not an improvised ls/find/grep scatter. Discovery
rides agent.coding_context.detect_project_facts (the single source of truth
behind verify and the workspace snapshot). Byte-stable: no timestamps, so the
same tree maps to the same result bytes.
"""

import os
from pathlib import Path

from tools.registry import registry

_MAX_DIRS_DEFAULT = 20
_NOISE_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules", "__pycache__",
    ".venv", "venv", ".mypy_cache", ".pytest_cache", ".tox", ".ruff_cache",
    "target", "dist", "build", "out", ".cache", ".coverage",
}
_ENTRY_NAMES = (
    "main.py", "__main__.py", "app.py", "manage.py", "main.ts", "index.ts",
    "index.tsx", "index.js", "main.rs", "main.go", "main.c", "main.cpp",
)

MAP_SCHEMA = {
    "name": "map",
    "description": (
        "First-contact skeleton of a project: layout (top dirs + file counts), "
        "manifests, package managers, entry points, verify gates (same indexes "
        "verify's command_index uses), and context files (AGENTS.md, README...) "
        "as read_file refs. One bounded call instead of an ls/find scatter. "
        "Bounded: layout caps at max_dirs with an honest truncated flag."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Project root (default: the session's working directory; falls back to this process's cwd when unrecorded)",
            },
            "max_dirs": {
                "type": "integer",
                "description": "Maximum layout directories to list (default 20).",
                "default": _MAX_DIRS_DEFAULT, "minimum": 1, "maximum": 100,
            },
        },
        "required": [],
    },
}


def _layout(root: Path, max_dirs: int) -> tuple[list[dict], bool, int]:
    """Top-level dirs (noise skipped) with immediate file counts. Honest truncation."""
    dirs = []
    total = 0
    try:
        entries = sorted(
            (e for e in os.scandir(root) if e.is_dir(follow_symlinks=False)
             and e.name not in _NOISE_DIRS),
            key=lambda e: e.name)
    except OSError:
        return [], False, 0
    for entry in entries:
        total += 1
        if len(dirs) < max_dirs:
            try:
                files = sum(1 for e in os.scandir(entry.path) if e.is_file(follow_symlinks=False))
            except OSError:
                files = 0
            dirs.append({"dir": entry.name, "files_direct": files})
    return dirs, total > len(dirs), total


def _entry_points(root: Path) -> list[str]:
    """Convention-named entry files at root and one level into src/app/cmd."""
    found = []
    for scope in (root, root / "src", root / "app", root / "cmd"):
        if not scope.is_dir():
            continue
        try:
            names = {e.name for e in os.scandir(scope) if e.is_file(follow_symlinks=False)}
        except OSError:
            continue
        found.extend(
            f"{scope.relative_to(root) / name}" if scope != root else name
            for name in _ENTRY_NAMES if name in names)
    return found[:10]


def _handle_map(args, **kw) -> dict:
    from tools.verify_tool import _resolve_project_root

    root = _resolve_project_root(args.get("path"), kw)
    if root is None or not root.is_dir():
        return {"error": f"map: project root not found: {args.get('path') or '(session cwd)'}"}

    from agent.coding_context import detect_project_facts
    facts = detect_project_facts(root)
    # First-contact view: the agent-rules convention (AGENTS/CLAUDE/.cursorrules)
    # plus README when present — the canonical first read of an unknown repo.
    context_files = list(facts.context_files)
    if (root / "README.md").is_file() and "README.md" not in context_files:
        context_files.insert(0, "README.md")

    try:
        max_dirs = max(1, min(int(args.get("max_dirs", _MAX_DIRS_DEFAULT)), 100))
    except (TypeError, ValueError):
        max_dirs = _MAX_DIRS_DEFAULT
    layout, truncated, total_dirs = _layout(root, max_dirs)

    result = {
        "success": True,
        "root": str(root),
        "manifests": facts.manifests,
        "package_managers": facts.package_managers,
        "verify_commands": facts.verify_commands,
        "verify_hint": "Run one with verify(path, command_index=i); indexes match this list",
        "context_files": context_files,
        "entry_points": _entry_points(root),
        "layout": layout,
        "truncated": truncated,
        "total_dirs": total_dirs,
    }
    if not facts.manifests and not layout:
        result["note"] = ("No manifests and no directories found — this may not be a project "
                          "root; use search_files to look around.")
    elif not facts.manifests:
        result["note"] = "No known manifests detected; verify will refuse here (no known gate)."
    return result


registry.register(
    name="map", toolset="file", schema=MAP_SCHEMA, handler=_handle_map,
    emoji="🗺️")
