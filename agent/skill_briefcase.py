"""Pinned hot-skills briefcase: static skill bodies as a byte-stable system block.

Prompt-cache campaign (2026-11-05): skill_view results were 14.3% of all new
content entering codex prompts (32.5M tokens / 14d) — the same static bodies
re-read fresh in every session, sitting mid-conversation where no cross-session
prefix cache can reach them. Pinning them here appends their raw SKILL.md text
to the effective system message (built once per content revision, replayed
verbatim), so the per-account prefix cache serves them across sessions at the
cached rate, and skill_view short-circuits their re-loads.

Byte-stability contract: config order is preserved, content is RAW (no template
preprocessing — rendered output is task-dependent), and each skill carries a
content-hash stamp. A skill edit changes the block (one deliberate prefix bust
for new sessions), never mid-session drift: the block is keyed on file stats
and rebuilt only when they move.

Config (``~/.hermes/config.yaml``, per profile)::

    skills:
      pin:
        - tensorast-grammar-data

Plugin-qualified names (``plugin:skill``) and skills needing live preprocessing
are not pinnable — they fail resolution and are skipped, never fatal.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

_LOCK = threading.Lock()
# (names, stats) -> assembled block; and the resolved-name set for skill_view.
_CACHE: dict[tuple, tuple[str, frozenset[str]]] = {}
_CONFIG_MEMO: dict[str, Any] | None = None  # read once per process: the pin list is
# prompt-shape config, effective from gateway start (same lifecycle as the
# system prompt), and the per-turn assembly path must stay I/O-free.


def _active_config() -> dict[str, Any]:
    """Live config; separate function so tests can monkeypatch it."""
    global _CONFIG_MEMO
    if _CONFIG_MEMO is None:
        from hermes_cli.config import load_config_readonly

        try:
            _CONFIG_MEMO = load_config_readonly() or {}
        except Exception:
            _CONFIG_MEMO = {}
    return _CONFIG_MEMO


def pinned_skill_names(config: dict[str, Any] | None = None) -> list[str]:
    """Configured pin list, order-preserving, deduped, non-empty."""
    cfg = _active_config() if config is None else (config or {})
    skills = cfg.get("skills") if isinstance(cfg, dict) else None
    pin = skills.get("pin") if isinstance(skills, dict) else None
    if not isinstance(pin, list):
        return []
    names: list[str] = []
    for entry in pin:
        name = str(entry).strip() if entry is not None else ""
        if name and name not in names:
            names.append(name)
    return names


def _resolve_pinned(names: list[str]) -> list[tuple[str, str]]:
    """``[(name, raw SKILL.md text)]`` for every resolvable pinned skill.

    Resolution mirrors skill_view's (locate_skill over the tier roots) but skips
    plugin-qualified names and reads RAW text — preprocessing is deliberately
    not applied (byte-stability over per-task rendering). Unresolvable names
    are skipped, never fatal: a typo must not break every session's prefix.
    """
    if not names:
        return []
    from tools.skills_tool import locate_skill, skill_search_dirs
    from tools.skills_tool_plugin import _read_skill_text

    roots, _active = skill_search_dirs()
    resolved: list[tuple[str, str]] = []
    for name in names:
        if ":" in name:  # plugin namespace — not pinnable (v1)
            continue
        try:
            error, _skill_dir, skill_md = locate_skill(name, None, roots)
            if error is not None or skill_md is None:
                continue
            resolved.append((name, _read_skill_text(skill_md)))
        except Exception:
            continue
    return resolved


def _stats_key(resolved: list[tuple[str, str]]) -> tuple:
    return tuple((name, len(text), hashlib.sha256(text.encode("utf-8")).hexdigest()[:16])
                 for name, text in resolved)


def build_briefcase_block(*, config: dict[str, Any] | None = None) -> str | None:
    """The byte-stable pinned-skills block, or ``None`` when nothing is pinned.

    Memoized on (pin names, per-skill content hashes): identical inputs return
    the identical cached string, so every session this process serves carries
    the same bytes.
    """
    names = pinned_skill_names(config)
    if not names:
        return None
    resolved = _resolve_pinned(names)
    if not resolved:
        return None
    key = (tuple(names), _stats_key(resolved))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached[0]

    parts: list[str] = []
    digest = hashlib.sha256()
    for name, text in resolved:
        stamp = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        digest.update(stamp.encode("ascii"))
        parts.append(f"### skill: {name} (v{stamp})\n\n{text.rstrip()}")
    block = (
        f"## Pinned skills (pre-loaded, v{digest.hexdigest()[:12]})\n\n"
        "The following skills are fully loaded above — do not call skill_view on them "
        "to re-read their SKILL.md (it will return a stub). Call skill_view(name, "
        "file_path=...) only for their reference/template files.\n\n"
        + "\n\n".join(parts)
    )
    with _LOCK:
        _CACHE[key] = (block, frozenset(name for name, _ in resolved))
    return block


def is_pinned(name: str, *, config: dict[str, Any] | None = None) -> bool:
    """True when *name*'s SKILL.md body is in the current briefcase."""
    if not name:
        return False
    names = pinned_skill_names(config)
    if name not in names:
        return False
    block_names = _resolved_names(names)
    return name in block_names


def _resolved_names(names: list[str]) -> frozenset[str]:
    """Memoized resolved-name set for the pin list (stats-keyed like the block)."""
    resolved = _resolve_pinned(names)
    if not resolved:
        return frozenset()
    key = (tuple(names), _stats_key(resolved))
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached[1]
    build_briefcase_block(config={"skills": {"pin": list(names)}})
    with _LOCK:
        cached = _CACHE.get(key)
        return cached[1] if cached else frozenset()


def reset_cache() -> None:
    """Test hook: drop memoized blocks and the config memo."""
    global _CONFIG_MEMO
    with _LOCK:
        _CACHE.clear()
    _CONFIG_MEMO = None
