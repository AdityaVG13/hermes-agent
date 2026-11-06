"""Gap-cause decomposition tests (cache lab, lever-b evidence)."""

import sqlite3

import pytest

from evals.cache_lab.gap_causes import decompose

DDL = """
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT,
    tool_calls TEXT, tool_name TEXT, timestamp REAL, active INTEGER DEFAULT 1,
    compacted INTEGER DEFAULT 0, _compressed_summary INTEGER DEFAULT 0);
CREATE TABLE session_model_usage (
    session_id TEXT, model TEXT, billing_provider TEXT, task TEXT DEFAULT '',
    api_call_count INTEGER DEFAULT 0, last_seen REAL);
"""


def _db(tmp_path):
    db = sqlite3.connect(tmp_path / "s.db")
    db.executescript(DDL)
    return db


def _msg(db, sid, role, chars, ts):
    db.execute("INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
               (sid, role, "x" * chars, ts))


T = 1000.0  # session start


def test_human_paced_gap_dominates_pool(tmp_path):
    db = _db(tmp_path)
    # call1; user idles 20min then types; call2 (gap 1200s > ttl 600)
    _msg(db, "s", "user", 3500, T)          # ~1000 tok
    _msg(db, "s", "assistant", 0, T + 10)
    _msg(db, "s", "user", 700, T + 1210)    # human idle 1200s
    _msg(db, "s", "assistant", 0, T + 1212)
    rep = decompose(db, ["s"], ttl=600.0)
    assert rep["by_cause"]["human"] > 0
    assert rep["by_cause"]["machine"] == 0
    assert rep["detail"][0]["bucket"] == "15-60min"


def test_machine_paced_gap_is_tool_wait(tmp_path):
    db = _db(tmp_path)
    # call1; tool takes 20min; user replies fast; call2 → machine-paced
    _msg(db, "s", "user", 3500, T)
    _msg(db, "s", "assistant", 0, T + 10)
    _msg(db, "s", "tool", 350, T + 1210)    # long tool execution
    _msg(db, "s", "user", 350, T + 1212)
    _msg(db, "s", "assistant", 0, T + 1214)
    rep = decompose(db, ["s"], ttl=600.0)
    assert rep["by_cause"]["machine"] > 0
    assert rep["by_cause"]["human"] == 0


def test_short_gaps_do_not_enter_the_pool(tmp_path):
    db = _db(tmp_path)
    _msg(db, "s", "user", 3500, T)
    _msg(db, "s", "assistant", 0, T + 10)
    _msg(db, "s", "user", 700, T + 100)     # 90s gap < ttl
    _msg(db, "s", "assistant", 0, T + 102)
    rep = decompose(db, ["s"], ttl=600.0)
    assert rep["total_loss_tokens"] == 0


def test_loss_weight_is_prefix_at_previous_call(tmp_path):
    db = _db(tmp_path)
    _msg(db, "s", "user", 3500, T)          # 1000 tok in prefix at call1
    _msg(db, "s", "assistant", 0, T + 10)
    _msg(db, "s", "user", 7000, T + 1300)   # idle 1290s, then 2000 tok
    _msg(db, "s", "assistant", 0, T + 1302)
    rep = decompose(db, ["s"], ttl=600.0)
    # expiry destroys exactly the 1000-token prefix, not the 2000-token tail
    assert 900 < rep["total_loss_tokens"] < 1100
