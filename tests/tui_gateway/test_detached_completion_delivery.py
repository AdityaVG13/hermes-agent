"""Diagnostic: disconnected child-result ownership must last through delivery.

No production changes. Controlled reaper timing, real delegation liveness query.
Run with Hermes's Python 3.14 test environment and scripts/run_tests.sh.
"""
from contextlib import nullcontext
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
import importlib
import threading

import pytest


@pytest.mark.parametrize("state,delivery,keep_owner", [
    ("running", "pending", True),
    ("completed", "pending", True),
    ("completed", "delivered", False),
])
def test_detached_owner_lasts_until_child_result_delivery(tmp_path, monkeypatch, state, delivery, keep_owner):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    server = importlib.import_module("tui_gateway.server")
    delegation = importlib.import_module("tools.async_delegation")
    sid, key = "diagnostic-runtime", "diagnostic-durable-parent"
    record = {"delegation_id": "diagnostic-batch", "session_key": key,
              "origin_ui_session_id": sid, "parent_session_id": key,
              "status": state, "delivery_state": delivery}
    monkeypatch.setattr(delegation, "_records", {"diagnostic-batch": record})
    pending_events = Queue()
    if state == "completed" and delivery == "pending":
        pending_events.put({"type": "async_delegation", "delegation_id": "diagnostic-batch",
                            "origin_ui_session_id": sid, "session_key": key,
                            "results": [{"status": "completed", "summary": "bounded probe result"}]})
    session = {"session_key": key, "profile_home": str(home), "running": False,
               "history_lock": threading.Lock(), "transport": server._detached_ws_transport,
               "agent": SimpleNamespace(session_id=key), "history": []}
    timers, closed = [], []

    class Timer:
        def __init__(self, delay, callback):
            self.callback = callback
            timers.append(self)
        def start(self):
            pass
        def cancel(self):
            pass

    def teardown(session, **kwargs):
        if session is not None:
            session["_finalized"] = True
            closed.append(kwargs["end_reason"])

    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_session_db", lambda _session: nullcontext(None))
    monkeypatch.setattr(server, "_session_owns_live_wakeup_schedule", lambda _session: False)
    monkeypatch.setattr(server, "_teardown_popped_session", teardown)

    # No mock of delegation.has_live_for_session or the reaper under test.
    server._schedule_ws_orphan_reap(sid)
    timers[0].callback()
    resident = server._sessions.get(sid) is session
    assert resident is keep_owner, (
        f"state={state}, delivery={delivery}: resident={resident}, "
        f"pending_completion_events={pending_events.qsize()}, teardown={closed}; "
        "the disconnected parent must remain available until its child result is delivered"
    )
