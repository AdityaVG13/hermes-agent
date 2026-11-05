#!/usr/bin/env python3
"""read_file hash handshake (prompt-cache campaign, lever 4).

The auto-dedup stub only knows what THIS task already read; the handshake gives
the model a portable proof of the bytes it holds: every successful read carries
``content_hash`` (sha256[:16] of the exact rendered content), and passing it back
as ``if_hash`` collapses a byte-identical re-read into an ``unchanged`` stub —
across tasks, generations and sessions. Stale or wrong hashes fail open to full
content; a matching stub never records a read (stubs must not bless writes).
"""

import json
import re

import pytest

from tools.file_tools import read_file_tool
from tools.file_tools_read_tracking import _read_tracker


@pytest.fixture(autouse=True)
def _clean_tracker():
    _read_tracker.clear()
    yield
    _read_tracker.clear()


_HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _read(path, task="t", **kw):
    raw = read_file_tool(str(path), task_id=task, **kw)
    return json.loads(raw)


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def test_result_carries_content_hash(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "alpha\nbeta\n")
    r = _read(f)
    assert _HEX16.match(r["content_hash"]), r.get("content_hash")


def test_hash_stable_across_tasks(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "alpha\nbeta\n")
    h1 = _read(f, task="t1")["content_hash"]
    h2 = _read(f, task="t2")["content_hash"]
    assert h1 == h2


def test_if_hash_match_returns_unchanged_stub(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "alpha\nbeta\n")
    h = _read(f, task="t1")["content_hash"]
    r = _read(f, task="t2", if_hash=h)
    assert r["status"] == "unchanged"
    assert r["content_returned"] is False
    assert "content" not in r
    assert r["content_hash"] == h


def test_stale_if_hash_after_change_returns_content(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "alpha\nbeta\n")
    stale = _read(f, task="t1")["content_hash"]
    _write(f, "alpha\nCHANGED\n")
    r = _read(f, task="t2", if_hash=stale)
    assert "content" in r
    assert r["content_hash"] != stale


def test_garbage_if_hash_fails_open_to_content(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "alpha\nbeta\n")
    r = _read(f, task="t1", if_hash="deadbeefdeadbeef")
    assert "content" in r
    assert r["content_hash"] == "deadbeefdeadbeef" or _HEX16.match(r["content_hash"])


def test_if_hash_of_different_region_returns_content(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "line1\nline2\nline3\nline4\n")
    h_region1 = _read(f, task="t1", offset=1, limit=2)["content_hash"]
    r = _read(f, task="t2", offset=3, limit=2, if_hash=h_region1)
    assert "content" in r


def test_auto_dedup_stub_echoes_hash(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "alpha\nbeta\n")
    first = _read(f, task="t1")
    second = _read(f, task="t1")
    assert second["status"] == "unchanged"
    assert second["content_hash"] == first["content_hash"]


def test_hash_changes_with_content(tmp_path):
    f = tmp_path / "a.txt"
    _write(f, "alpha\n")
    h1 = _read(f, task="t1")["content_hash"]
    _write(f, "alpha\ngamma\n")
    h2 = _read(f, task="t2")["content_hash"]
    assert h1 != h2
