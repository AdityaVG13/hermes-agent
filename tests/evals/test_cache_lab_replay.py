"""Cache-lab replay simulator (prefix-cache campaign, calibration phase).

The postmortem's ``forensics/tokens.py`` replays prompt *sizes*; this lab
replays *prefixes*: for each assistant call it reconstructs the uid stream that
was in context, computes the longest-common-prefix token share against the
previous call (that is what an implicit longest-prefix provider serves as a
cache hit), and applies policy terms — TTL decay and match reliability — so
``evals/cache_lab/calibrate.py`` can grid-fit them against the observed
``session_model_usage`` cache rates and gate the model at ±3pts on 80% of
sessions before any counterfactual is trusted.
"""

import json
import sqlite3

import pytest

from evals.cache_lab.replay import (
    CHARS_PER_TOKEN, read_session_calls, simulate, cache_ratio, _TOK)


def _mk_db(tmp_path):
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.executescript("""
        CREATE TABLE sessions (id TEXT PRIMARY KEY, system_prompt_hash TEXT, model TEXT);
        CREATE TABLE system_prompts (hash TEXT PRIMARY KEY, prompt TEXT NOT NULL);
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
            content TEXT, tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL,
            active INTEGER NOT NULL DEFAULT 1, _compressed_summary INTEGER NOT NULL DEFAULT 0,
            message_uid TEXT, absorbed_message_uids TEXT);
    """)
    conn.execute("INSERT INTO sessions VALUES ('s1', 'h1', 'm-test')")
    conn.execute("INSERT INTO system_prompts VALUES ('h1', '')")
    conn.commit()
    return conn


def _tok(n):  # a text blob of exactly n estimated tokens
    return "x" * int(n * CHARS_PER_TOKEN)


def _msg(conn, role, tokens, ts, *, uid=None, summary=False, absorbed=None):
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, _compressed_summary,"
        " message_uid, absorbed_message_uids) VALUES ('s1', ?, ?, ?, ?, ?, ?)",
        (role, _tok(tokens), ts, int(summary), uid,
         json.dumps(absorbed) if absorbed else None))
    conn.commit()
    return cur.lastrowid


def test_pure_append_cache_equals_growth_share(tmp_path):
    db = _mk_db(tmp_path)
    _msg(db, "user", 100, 1.0, uid="u1")
    _msg(db, "assistant", 50, 2.0, uid="a1")
    _msg(db, "user", 10, 3.0, uid="u2")
    _msg(db, "assistant", 50, 4.0, uid="a2")
    _msg(db, "user", 10, 5.0, uid="u3")
    _msg(db, "assistant", 50, 6.0, uid="a3")

    calls = read_session_calls(db, "s1", tool_schema_tokens=0)
    # streams at the three calls: [u1]=100, [u1,a1,u2]=160, [..,a2,u3]=220
    assert [c.prompt_tokens for c in calls] == [100, 160, 220]
    assert [c.cached_prefix_tokens for c in calls] == [0, 100, 160]

    sim = simulate(calls, ttl_seconds=None)
    cached, total = cache_ratio(sim)
    assert cached == 0 + 100 + 160
    assert total == 100 + 160 + 220


def test_ttl_expiry_zeroes_cache_for_slow_gaps(tmp_path):
    db = _mk_db(tmp_path)
    _msg(db, "user", 100, 1.0, uid="u1")
    _msg(db, "assistant", 50, 100.0, uid="a1")   # gap 99s from u1's ts… use assistant ts
    _msg(db, "user", 10, 200.0, uid="u2")        # call gap 100s
    _msg(db, "assistant", 50, 200.0 + 400, uid="a2")  # call gap 400s

    calls = read_session_calls(db, "s1", tool_schema_tokens=0)
    fast = simulate(calls, ttl_seconds=600)
    assert [c.cached_prefix_tokens for c in fast] == [0, 100]
    slow = simulate(calls, ttl_seconds=400)  # 500s call gap > 400s ttl
    assert slow[1].cached_prefix_tokens == 0


def test_compaction_resets_prefix_at_summary(tmp_path):
    db = _mk_db(tmp_path)
    _msg(db, "user", 100, 1.0, uid="u1")
    _msg(db, "assistant", 50, 2.0, uid="a1")
    _msg(db, "user", 10, 3.0, uid="u2")
    _msg(db, "assistant", 50, 4.0, uid="a2")
    # compaction: summary (30 tok) replaces u1,a1,u2 ; tail a2 kept
    _msg(db, "assistant", 30, 5.0, uid="sum1", summary=True,
         absorbed=["u1", "a1", "u2"])
    _msg(db, "user", 10, 6.0, uid="u3")
    _msg(db, "assistant", 50, 7.0, uid="a3")

    calls = read_session_calls(db, "s1", tool_schema_tokens=0)
    # call after compaction: stream = [sum1, a2, u3] = 90; LCP vs previous
    # stream ([u1,a1,u2,a2]) = 0 (no shared leading uid)
    after = calls[2]
    assert after.prompt_tokens == 30 + 50 + 10
    assert after.cached_prefix_tokens == 0


def test_reliability_multiplier_scales_cached(tmp_path):
    db = _mk_db(tmp_path)
    _msg(db, "user", 100, 1.0, uid="u1")
    _msg(db, "assistant", 50, 2.0, uid="a1")
    _msg(db, "user", 10, 3.0, uid="u2")
    _msg(db, "assistant", 50, 4.0, uid="a2")

    calls = read_session_calls(db, "s1", tool_schema_tokens=0)
    half = simulate(calls, ttl_seconds=None, reliability=0.5)
    assert half[1].cached_prefix_tokens == 50


def test_partial_compaction_keeps_surviving_prefix(tmp_path):
    db = _mk_db(tmp_path)
    _msg(db, "user", 100, 1.0, uid="u1")
    _msg(db, "assistant", 50, 2.0, uid="a1")
    # summary absorbs ONLY u1; a1 survives
    _msg(db, "assistant", 30, 3.0, uid="sum1", summary=True, absorbed=["u1"])
    _msg(db, "user", 10, 4.0, uid="u2")
    _msg(db, "assistant", 50, 5.0, uid="a2")

    calls = read_session_calls(db, "s1", tool_schema_tokens=0)
    # stream at call2 = [sum1, a1, u2] = 90; previous stream [u1] shares no
    # leading uid → cached 0. prompt reflects absorption (u1 gone, sum1 in).
    assert calls[1].prompt_tokens == 30 + 50 + 10
    assert calls[1].cached_prefix_tokens == 0


def test_tok_estimation_is_chars_over_ratio():
    assert _TOK("x" * 350) == 100
