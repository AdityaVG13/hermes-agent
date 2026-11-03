"""``hermes skills health`` — read-only diagnosis of the skills library.

Surfaces the failure classes that stay silent until a skill "just isn't there":

- **hidden**: same-name duplicates under one tier resolve to nothing (``ambiguous``
  with no load path) — the skill exists on disk but cannot be invoked. The exact
  class that hid ~166 skills fleet-wide via accidental top-level/category twins.
- **orphan locks**: hub lock entries whose recorded ``install_path`` no longer
  exists. Reported from direct directory existence only — the update checker's
  orphan heuristic misfires on live skills, and uninstalling those deletes work.
- **dead links**: directory symlinks in the skills root that dangle or resolve to
  a directory without SKILL.md (old agent-bridge links).
- **husk dirs**: non-dot directories holding no SKILL.md at any depth (emptied
  categories, interrupted moves).
- **cache**: the disk-persisted catalog (``cache/skills-catalog-v1.json``) — a
  corrupt file is a finding (torn-write signal); rows that merely disagree with
  the current library are informational (the file is digest-keyed and never
  served stale; the next scan rewrites it).

Read-only by construction: no scan caches are cleared, no files written.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

_HUSK_WALK_DEPTH = 3


def _skill_root() -> Any:
    from tools import skills_tool as st
    return st.SKILLS_DIR


def _lock_entries() -> list[dict]:
    try:
        from tools.skills_hub import HubLockFile
        return list(HubLockFile().list_installed() or [])
    except Exception:
        return []


def _walk_debris(root) -> tuple[list[dict], list[dict]]:
    """Dead directory symlinks and husk dirs at skill/category level.

    A symlink is healthy only when it resolves to a directory holding SKILL.md
    (the Mac library legitimately carries a few). A real directory with SKILL.md
    is a skill leaf; a real directory whose subtree (bounded) holds no SKILL.md
    anywhere is a husk.
    """
    dead: list[dict] = []
    husks: list[dict] = []

    def has_skill_md(path, depth: int) -> bool:
        try:
            entries = list(os.scandir(path))
        except OSError:
            return False
        if any(e.name == "SKILL.md" for e in entries):
            return True
        if depth <= 0:
            return False
        return any(e.is_dir(follow_symlinks=False) and has_skill_md(e.path, depth - 1)
                   for e in entries if not e.name.startswith("."))

    def visit(path, rel: str, depth: int) -> None:
        try:
            entries = list(os.scandir(path))
        except OSError:
            return
        for entry in entries:
            if entry.name.startswith("."):
                continue
            child_rel = f"{rel}/{entry.name}" if rel else entry.name
            if entry.is_symlink():
                try:
                    resolved = os.path.realpath(entry.path)
                    healthy = os.path.isdir(resolved) and os.path.isfile(os.path.join(resolved, "SKILL.md"))
                    reason = None if healthy else (
                        "dangling" if not os.path.exists(resolved) else "target has no SKILL.md")
                except OSError as exc:
                    resolved, reason = "", f"unresolvable ({exc.__class__.__name__})"
                if reason:
                    dead.append({"name": entry.name, "path": child_rel,
                                 "target": resolved, "reason": reason})
            elif entry.is_dir(follow_symlinks=False):
                if any(e.name == "SKILL.md" for e in _scandir_safe(entry.path)):
                    continue  # a skill is a scan leaf: its subtree is not catalog state
                if has_skill_md(entry.path, _HUSK_WALK_DEPTH - 1):
                    visit(entry.path, child_rel, depth - 1)  # category dir: descend
                else:
                    husks.append({"name": entry.name, "path": child_rel})

    visit(root, "", 1)
    return dead, husks


def _scandir_safe(path):
    try:
        return list(os.scandir(path))
    except OSError:
        return []


def _cache_state(catalog: list[dict]) -> dict:
    """State of the disk-persisted catalog vs the freshly scanned library."""
    from tools import skills_tool as st
    path = st._catalog_cache_path()
    if not path.exists():
        return {"state": "absent", "note": "cold; first scan will write it"}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return {"state": "unreadable", "error": str(exc)}
    if not isinstance(data, dict):
        return {"state": "unreadable", "error": "cache file is not a JSON object"}
    entry = next((v for v in data.values()
                  if isinstance(v, dict) and isinstance(v.get("rows"), list)), None)
    if entry is None:
        return {"state": "unreadable", "error": "no rows entry in cache file"}
    cached = {r.get("load_name") or r.get("name")
              for r in entry["rows"] if isinstance(r, dict)}
    cached.discard(None)
    fresh = {s["load_name"] for s in catalog if s.get("load_name")}
    if cached != fresh:
        return {"state": "rows_disagree",
                "cached_rows": len(cached), "current_rows": len(fresh),
                "only_in_cache": sorted(x for x in cached - fresh if x)[:10],
                "note": "the file is digest-keyed so it will be rewritten; "
                        "disagreement beyond a rewrite indicates a torn write"}
    return {"state": "ok", "rows": len(cached)}


def collect_health() -> dict[str, Any]:
    """All checks, as a JSON-serialisable dict. Read-only: the scan's disk-cache
    persistence is suspended so the catalog file is inspected as found, never
    repaired in passing."""
    from tools import skills_tool as st

    st._SKILLS_CACHE.clear()  # health must see the library, not this process's memo
    _real_store = st._store_disk_catalog
    st._store_disk_catalog = lambda *a, **k: None
    try:
        catalog = st._skill_catalog(include_hidden=True, skip_disabled=False)
    finally:
        st._store_disk_catalog = _real_store

    hidden: list[dict] = []
    shadowed: list[dict] = []
    loadable = 0
    for s in catalog:
        status = s.get("status")
        if status == "unique" and s.get("load_name"):
            loadable += 1
        elif status == "ambiguous" and not s.get("load_name"):
            hidden.append({"name": s.get("name", ""), "path": str(s.get("path", ""))})
        elif status == "shadowed":
            shadowed.append({"name": s.get("name", ""), "path": str(s.get("path", ""))})

    root = _skill_root()
    orphan_locks = [{"name": e.get("name", ""), "install_path": e.get("install_path", "")}
                    for e in _lock_entries()
                    if e.get("install_path") and not (root / e["install_path"]).is_dir()]

    dead_links, husk_dirs = _walk_debris(root)
    cache = _cache_state(catalog)

    findings = (len(hidden) + len(orphan_locks) + len(dead_links) + len(husk_dirs)
                + (1 if cache["state"] == "unreadable" else 0))
    return {
        "ok": findings == 0,
        "findings": findings,
        "hidden": hidden,
        "orphan_locks": orphan_locks,
        "dead_links": dead_links,
        "husk_dirs": husk_dirs,
        "cache": cache,
        "info": {"shadowed": shadowed},
        "totals": {"catalog_entries": len(catalog), "loadable": loadable,
                   "hidden": len(hidden), "shadowed": len(shadowed)},
    }


def do_health(as_json: bool = False, console: Optional[Any] = None) -> Optional[bool]:
    """CLI entry: prints the report, returns False when findings exist (exit 1)."""
    report = collect_health()
    if as_json:
        print(json.dumps(report, indent=1))
    else:
        c = console or _console()
        t = report["totals"]
        c.print(f"[bold]Skills health[/] — {t['loadable']} loadable, "
                f"{t['hidden']} hidden, {t['shadowed']} shadowed "
                f"(by-design tier precedence)\n")

        def section(label: str, rows: list[dict], hint: str, style: str) -> None:
            if not rows:
                return
            c.print(f"[{style}]{label} ({len(rows)}):[/]")
            for r in rows[:20]:
                detail = " ".join(f"{k}={v}" for k, v in r.items() if k != "name" and v)
                c.print(f"  {r.get('name', '?')} [dim]{detail}[/]")
            if len(rows) > 20:
                c.print(f"  [dim]… {len(rows) - 20} more[/]")
            c.print(f"[dim]  {hint}[/]\n")

        section("Hidden (uninvocable duplicates)", report["hidden"],
                "same-name copies resolve to nothing — move the stale twin out of the "
                "skills root (quarantine, do not delete)", "bold red")
        section("Orphaned lock entries", report["orphan_locks"],
                "directory truly missing — `hermes skills uninstall <name>` is safe "
                "ONLY for these; `skills check` may name others whose dirs exist, "
                "never uninstall those", "yellow")
        section("Dead symlinks", report["dead_links"],
                "remove or repoint (quarantine, do not delete)", "yellow")
        section("Husk directories", report["husk_dirs"],
                "no SKILL.md anywhere beneath — leftover category or interrupted move",
                "yellow")
        state = report["cache"]["state"]
        if state in ("unreadable", "rows_disagree"):
            c.print(f"[yellow]Catalog cache:[/] {state} (informational — digest-keyed, never served stale; "
                    f"rewritten on next scan) — {report['cache']}\n")
        c.print("[green]healthy[/]" if report["ok"] else
                f"[bold red]{report['findings']} finding(s)[/]")
    return None if report["ok"] else False


def _console():
    from rich.console import Console
    return Console()
