"""Cache-route guard (lever 1 of the prompt-cache campaign, 2026-11-05).

A route whose fleet history shows structurally-uncached input (crouton-alpha via
OpenRouter served 406M tokens at 0% over 14 days) re-bills every prompt and
tanks every blended cache number. The guard reads ``session_model_usage`` once
per session and warns:

- ``no_upstream_cache``  — reads AND writes ~zero: the upstream does not cache.
- ``write_never_read``   — cache writes are paid but never read back: misplaced
  breakpoints or a provider that never serves hits. Worth reporting, not routing.

Fail-open everywhere: no db, unreadable db, weird agent shape → silence.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _seed_route(db, *, session_id="s1", model="m-zero", provider="nous",
                base_url="https://openrouter.example/v1", calls=500,
                cache_read=0, cache_write=0, input_tokens=60_000_000,
                last_seen=None):
    now = last_seen if last_seen is not None else time.time()
    db.create_session(session_id, "test")
    with db._lock:
        db._conn.execute(
            """INSERT INTO session_model_usage
               (session_id, model, billing_provider, billing_base_url, billing_mode, task,
                api_call_count, input_tokens, output_tokens, cache_read_tokens,
                cache_write_tokens, reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (session_id, model, provider, base_url, "", "", calls, input_tokens, 0,
             cache_read, cache_write, 0, 0.0, 0.0, now - 3600, now),
        )
        db._conn.commit()


class _FakeAgent:
    """Just the surface check_session_route touches."""

    def __init__(self, db_path, model="m-zero", provider="nous",
                 base_url="https://openrouter.example/v1"):
        self._cache_route_guard_db = db_path
        self.model: Any = model
        self.provider = provider
        self.base_url = base_url
        self.warnings: list[str] = []

    def _emit_warning(self, message, **kwargs):
        self.warnings.append(message)


def test_no_cache_route_flagged(db, tmp_path):
    from agent.cache_route_guard import route_cache_warning

    _seed_route(db)
    result = route_cache_warning(
        tmp_path / "state.db", "m-zero", "nous", "https://openrouter.example/v1"
    )
    assert result is not None
    assert result["kind"] == "no_upstream_cache"
    assert "m-zero" in result["message"]
    assert "nous" in result["message"]
    assert "prompt caching" in result["message"]


def test_write_paid_never_read_flagged_differently(db, tmp_path):
    from agent.cache_route_guard import route_cache_warning

    _seed_route(db, cache_write=40_000_000)
    result = route_cache_warning(
        tmp_path / "state.db", "m-zero", "nous", "https://openrouter.example/v1"
    )
    assert result is not None
    assert result["kind"] == "write_never_read"
    assert "m-zero" in result["message"]


def test_healthy_route_silent(db, tmp_path):
    from agent.cache_route_guard import route_cache_warning

    _seed_route(db, cache_read=540_000_000, input_tokens=60_000_000)
    assert route_cache_warning(
        tmp_path / "state.db", "m-zero", "nous", "https://openrouter.example/v1"
    ) is None


def test_mediocre_but_caching_route_silent(db, tmp_path):
    """40% cached is poor but not structural — the guard targets zeros."""
    from agent.cache_route_guard import route_cache_warning

    _seed_route(db, cache_read=40_000_000)
    assert route_cache_warning(
        tmp_path / "state.db", "m-zero", "nous", "https://openrouter.example/v1"
    ) is None


def test_low_volume_route_silent(db, tmp_path):
    from agent.cache_route_guard import route_cache_warning

    _seed_route(db, calls=10)
    assert route_cache_warning(
        tmp_path / "state.db", "m-zero", "nous", "https://openrouter.example/v1"
    ) is None


def test_stale_window_silent(db, tmp_path):
    from agent.cache_route_guard import route_cache_warning

    _seed_route(db, last_seen=time.time() - 30 * 86400)
    assert route_cache_warning(
        tmp_path / "state.db", "m-zero", "nous", "https://openrouter.example/v1"
    ) is None


def test_other_route_silent(db, tmp_path):
    from agent.cache_route_guard import route_cache_warning

    _seed_route(db)
    assert route_cache_warning(
        tmp_path / "state.db", "different-model", "nous", "https://openrouter.example/v1"
    ) is None


def test_missing_db_fail_open(tmp_path):
    from agent.cache_route_guard import route_cache_warning

    assert route_cache_warning(
        tmp_path / "does-not-exist.db", "m-zero", "nous", "https://openrouter.example/v1"
    ) is None


def test_check_session_route_emits_once(db, tmp_path):
    from agent import cache_route_guard

    _seed_route(db)
    agent = _FakeAgent(tmp_path / "state.db")
    cache_route_guard.check_session_route(agent)
    assert len(agent.warnings) == 1
    assert "m-zero" in agent.warnings[0]
    cache_route_guard.check_session_route(agent)
    assert len(agent.warnings) == 1
    assert getattr(agent, "_cache_route_guard_done", False) is True


def test_check_session_route_uses_session_db(db, tmp_path):
    """The gateway case: no path override, but the agent carries its own SessionDB."""
    from agent import cache_route_guard

    _seed_route(db)
    agent = _FakeAgent(tmp_path / "unused.db")
    agent._cache_route_guard_db = None
    agent._session_db = db
    cache_route_guard.check_session_route(agent)
    assert len(agent.warnings) == 1
    assert "m-zero" in agent.warnings[0]


def test_check_session_route_no_db_silent(tmp_path):
    """Fake/isolated agents with no db at all: skip, never resolve a real home."""
    from agent import cache_route_guard

    agent = _FakeAgent(tmp_path / "unused.db")
    agent._cache_route_guard_db = None
    cache_route_guard.check_session_route(agent)
    assert agent.warnings == []
    assert agent._cache_route_guard_done is True


def test_check_session_route_dict_model_fail_open(db, tmp_path):
    from agent import cache_route_guard

    agent = _FakeAgent(tmp_path / "state.db")
    agent.model = {"model": "nested"}
    cache_route_guard.check_session_route(agent)  # must not raise
    assert agent.warnings == []
