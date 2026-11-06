"""Counterfactual policies on the replay lab (Phase 2).

Three policies, labeled by evidence class:
- READ-STUB (REPLAYED): identical re-sent read_file/search_files results
  collapse to a stub, with token overrides fed back through the reader.
- KEEPALIVE (REPLAYED): the cached share lost to TTL expiry — simulate with
  the fitted TTL vs no TTL and diff.
- LANE AFFINITY (ESTIMATED): aux-lane uncached tokens vs a tail-only estimate;
  from usage rows + replay stats, not a transcript replay.
"""

import json
import sqlite3

from evals.cache_lab.counterfactual import find_resends, ttl_loss, affinity_savings
from evals.cache_lab.replay import read_session_calls, simulate, cache_ratio


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


def _msg(conn, role, content, ts, *, uid=None, tool_name=None):
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp, tool_name, message_uid)"
        " VALUES ('s1', ?, ?, ?, ?, ?)", (role, content, ts, tool_name, uid))
    conn.commit()
    return cur.lastrowid


def test_find_resends_flags_later_duplicate_tool_results(tmp_path):
    db = _mk_db(tmp_path)
    big = "x" * 3500  # 1000 tokens
    _msg(db, "user", "r1", 1.0, uid="u1")
    _msg(db, "tool", big, 2.0, uid="t1", tool_name="read_file")
    _msg(db, "assistant", "ok", 3.0, uid="a1")
    _msg(db, "user", "r2", 4.0, uid="u2")
    _msg(db, "tool", big, 5.0, uid="t2", tool_name="read_file")   # identical resend
    _msg(db, "assistant", "ok", 6.0, uid="a2")
    _msg(db, "tool", "y" * 350, 7.0, uid="t3", tool_name="read_file")  # unique

    resends = find_resends(db, "s1")
    assert [r["uid"] for r in resends] == ["t2"]
    assert resends[0]["tokens"] > 900  # ~1000 estimated tokens


def test_find_resends_ignores_non_read_tools(tmp_path):
    db = _mk_db(tmp_path)
    poll = '{"status":"running"}'
    _msg(db, "tool", poll, 1.0, uid="p1", tool_name="process_manage")
    _msg(db, "tool", poll, 2.0, uid="p2", tool_name="process_manage")
    assert find_resends(db, "s1") == []


def test_stub_overrides_shrink_downstream_prompts(tmp_path):
    db = _mk_db(tmp_path)
    big = "x" * 3500
    _msg(db, "user", "go", 1.0, uid="u1")
    _msg(db, "tool", big, 2.0, uid="t1", tool_name="read_file")
    _msg(db, "assistant", "ok", 3.0, uid="a1")
    _msg(db, "user", "again", 4.0, uid="u2")
    _msg(db, "tool", big, 5.0, uid="t2", tool_name="read_file")  # resend -> stub
    _msg(db, "assistant", "ok", 6.0, uid="a2")

    base = read_session_calls(db, "s1", tool_schema_tokens=0)
    resends = find_resends(db, "s1")
    overrides = {r["uid"]: 10.0 for r in resends}  # stub ≈ 10 tokens
    stubbed = read_session_calls(db, "s1", tool_schema_tokens=0, token_overrides=overrides)

    assert stubbed[1].prompt_tokens < base[1].prompt_tokens
    # uncached (total - cached) must drop with the stub policy
    bc, bt = cache_ratio(simulate(base, ttl_seconds=None))
    sc, st = cache_ratio(simulate(stubbed, ttl_seconds=None))
    assert (st - sc) < (bt - bc)


def test_ttl_loss_measures_expiry_damage(tmp_path):
    db = _mk_db(tmp_path)
    _msg(db, "user", "x" * 350, 1.0, uid="u1")
    _msg(db, "assistant", "ok", 2.0, uid="a1")
    _msg(db, "user", "y" * 350, 1000.0, uid="u2")
    _msg(db, "assistant", "ok", 1000.0 + 700, uid="a2")  # 700s gap

    calls = read_session_calls(db, "s1", tool_schema_tokens=0)
    loss = ttl_loss(calls, ttl_seconds=600)
    # with ttl the second call's cached share is zeroed; loss == that share
    with_ttl = cache_ratio(simulate(calls, ttl_seconds=600))
    no_ttl = cache_ratio(simulate(calls, ttl_seconds=None))
    assert loss == no_ttl[0] - with_ttl[0]
    assert loss > 0


def test_affinity_savings_from_usage_rows():
    aux = [
        {"task": "background_review", "calls": 100, "uncached_tokens": 1_000_000},
        {"task": "approval", "calls": 50, "uncached_tokens": 25_000},
    ]
    tail_per_call = 6_000
    savings = affinity_savings(aux, tail_per_call=tail_per_call)
    # aux uncached now: 1,025,000; after affinity: 150 calls * 6,000 tail = 900,000
    assert savings["uncached_now"] == 1_025_000
    assert savings["uncached_after"] == 900_000
    assert savings["saved"] == 125_000
