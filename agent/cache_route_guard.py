"""Session-start prompt-cache route guard.

A route whose fleet history shows structurally-uncached input re-bills the whole
prompt on every call (stealth/crouton-alpha via OpenRouter: 406M input tokens at
0% cache over 14 days) and drags every blended cache number down with it. The
guard reads ``session_model_usage`` once per session, fail-open, and warns via
the agent's warning channel:

- ``no_upstream_cache`` — reads AND writes ~zero: the upstream does not appear to
  support prompt caching. Routing advice: prefer a caching route for long work.
- ``write_never_read`` — cache writes are paid but reads are ~zero: misplaced
  breakpoints or a provider that never serves hits. Worth reporting, not routing.

Deliberately NOT a gate: it never blocks or switches a route, it only surfaces
evidence the operator already paid for. Everything fails open — no db, unreadable
db, odd agent shape → silence.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("agent.cache_route_guard")

WINDOW_DAYS = 14
# "Structurally uncached" = the cache-read share of billed input is ~zero. A poor
# but real 40% route is a different (tail-size) problem this guard must not noise on.
MAX_CACHE_READ_FRACTION = 0.05
# Evidence floors: a handful of calls or a few tokens proves nothing about a route.
MIN_CALLS = 100
MIN_UNCACHED_INPUT_TOKENS = 10_000_000
# Separates "upstream never caches" (no writes either) from "writes paid, never read".
MIN_CACHE_WRITE_TOKENS = 1_000_000

# Session starts are frequent; the aggregate does not change within an hour.
_MEMO_TTL_SECONDS = 3600.0
_memo_lock = threading.Lock()
_memo: dict[tuple[str, int], tuple[float, list[dict[str, Any]]]] = {}


def _route_stats(db_path: Any, window_days: int) -> list[dict[str, Any]]:
    """Aggregated per-route cache stats, memoized per (path, window) for the TTL."""
    key = (str(db_path), window_days)
    now = time.time()
    with _memo_lock:
        hit = _memo.get(key)
        if hit and now - hit[0] < _MEMO_TTL_SECONDS:
            return hit[1]

    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(db_path), read_only=True)
    try:
        stats = db.get_route_cache_stats(window_days=window_days)
    finally:
        try:
            db.close()
        except Exception:
            pass
    with _memo_lock:
        _memo[key] = (now, stats)
    return stats


def _classify(stat: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Return a warning dict for a structurally-uncached route, else ``None``."""
    calls = stat.get("calls") or 0
    cache_read = stat.get("cache_read") or 0
    cache_write = stat.get("cache_write") or 0
    input_tokens = stat.get("input_tokens") or 0
    if calls < MIN_CALLS or input_tokens < MIN_UNCACHED_INPUT_TOKENS:
        return None
    billed = cache_read + input_tokens
    if billed <= 0:
        return None
    fraction = cache_read / billed
    if fraction >= MAX_CACHE_READ_FRACTION:
        return None

    model = stat.get("model") or ""
    provider = stat.get("billing_provider") or ""
    percent = round(fraction * 100)
    if cache_write < MIN_CACHE_WRITE_TOKENS:
        message = (
            f"Route {model} via {provider} served {calls:,} calls at {percent}% cached "
            f"input over the last {WINDOW_DAYS} days (~{input_tokens / 1e6:.0f}M tokens "
            f"re-billed on every call). This upstream does not appear to support prompt "
            f"caching — prefer a caching route for long-context work."
        )
        return {"kind": "no_upstream_cache", "message": message, **stat}
    message = (
        f"Route {model} via {provider} paid cache writes (~{cache_write / 1e6:.0f}M tokens) "
        f"but read back only {percent}% of billed input over {calls:,} calls — breakpoints "
        f"may be misplaced or the provider never serves hits. Worth reporting."
    )
    return {"kind": "write_never_read", "message": message, **stat}


def _match_and_classify(
    stats: list[dict[str, Any]], model: str, provider: str, base_url: str,
) -> Optional[dict[str, Any]]:
    """Find this route in the aggregate and classify it, if it is structurally bad."""

    def _norm(value: Any) -> str:
        return (value or "").strip().lower()

    for stat in stats:
        if _norm(stat.get("model")) != _norm(model):
            continue
        if _norm(stat.get("billing_provider")) != _norm(provider):
            continue
        row_url, want_url = _norm(stat.get("billing_base_url")), _norm(base_url)
        if row_url and want_url and row_url != want_url:
            continue
        return _classify(stat)
    return None


def route_cache_warning(
    db_path: Any, model: str, provider: str, base_url: str = "",
    window_days: int = WINDOW_DAYS,
) -> Optional[dict[str, Any]]:
    """Warning dict for the given route if its history is structurally uncached.

    Matches on model + provider, and on ``base_url`` when both sides carry one
    (subscription providers often record '' on both sides — that still matches).
    Fail-open: any error (missing db, locked db) returns ``None``.
    """
    if not model or not provider:
        return None
    try:
        stats = _route_stats(db_path, window_days)
    except Exception:
        logger.debug("cache-route stats unavailable", exc_info=True)
        return None
    return _match_and_classify(stats, model, provider, base_url)


def route_cache_warning_from_db(
    db: Any, model: str, provider: str, base_url: str = "",
    window_days: int = WINDOW_DAYS,
) -> Optional[dict[str, Any]]:
    """Same check over an open (possibly writer) ``SessionDB`` — the agent's own
    store, queried in-process. Fail-open on any error."""
    if not model or not provider:
        return None
    try:
        stats = db.get_route_cache_stats(window_days=window_days)
    except Exception:
        logger.debug("cache-route stats unavailable on session db", exc_info=True)
        return None
    return _match_and_classify(stats, model, provider, base_url)


def check_session_route(agent: Any) -> None:
    """Once per agent: warn when this session's route is structurally uncached.

    Resolution order: the agent's own ``_session_db`` (in-process query, the
    gateway/CLI case), then an explicit ``_cache_route_guard_db`` override
    (tests, standalone tools). No home resolution — an agent with neither is a
    fake or a persistence-isolated fork, where the check does not apply.
    Sets ``agent._cache_route_guard_done`` no matter what happened — a broken
    guard must cost one attempt per session, not one per iteration.
    """
    try:
        try:
            if getattr(agent, "_cache_route_guard_done", False):
                return
            model = getattr(agent, "model", "")
            provider = (getattr(agent, "provider", "") or "").strip()
            base_url = (getattr(agent, "base_url", "") or "").strip()
            if isinstance(model, str) and model and provider:
                db_override = getattr(agent, "_cache_route_guard_db", None)
                session_db = getattr(agent, "_session_db", None)
                if db_override is not None:
                    result = route_cache_warning(db_override, model, provider, base_url)
                elif session_db is not None and hasattr(session_db, "get_route_cache_stats"):
                    result = route_cache_warning_from_db(
                        session_db, model, provider, base_url
                    )
                else:
                    result = None
                if result is not None:
                    emit = getattr(agent, "_emit_warning", None)
                    if callable(emit):
                        emit(result["message"])
        finally:
            try:
                agent._cache_route_guard_done = True
            except Exception:
                pass
    except Exception:
        logger.debug("cache-route guard failed", exc_info=True)
