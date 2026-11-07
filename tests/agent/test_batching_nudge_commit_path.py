"""The batching nudge at the real commit choke point (2026-11-07 live lesson).

Unit tests pin the pure functions; this drives the REAL ``_commit_tool_result``
with the two live append orderings the loop actually produces:

* row-present — turn_tool_round appends the assistant row BEFORE dispatch
  (turn_tool_round.py:113), so the streak walker already counts the in-flight
  turn; the commit's own ``tool_call_id`` must NOT count it twice.
* row-absent — approval-segmented turns commit before their row lands; the id
  must top the streak up by one.

Plus the invariants: a 3+-call batch in flight is never nagged (even though
its row already sits in the transcript), and the 2nd/4th single stay silent.
"""
from types import SimpleNamespace

import agent.tool_executor as te
from agent.tool_executor import _ToolCallRef, _budget_for_agent, _commit_tool_result


def _asst(cid):
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "name": "read_file", "arguments": "{}"}]}


def _tool(cid):
    return {"role": "tool", "tool_call_id": cid, "content": "{}"}


def _run(msgs, cid, batch_size):
    agent = SimpleNamespace(
        _current_tool_batch_size=batch_size,
        _read_tracker=SimpleNamespace(record=lambda *a, **k: None),
        _touch_activity=lambda *a, **k: None,
        _subdirectory_hints=SimpleNamespace(check_tool_call=lambda *a, **k: None),
        _tool_result_content_for_active_model=lambda name, result: result,
        tool_progress_callback=None, session_db=None)
    te.note_tool_result = lambda *a, **k: None
    ref = _ToolCallRef(name="read_file", args={"path": "/etc/hostname"}, call_id=cid,
                       task_id="t", trace=[])
    _commit_tool_result(agent, msgs, ref, '{"ok": true}',
                        is_error=False, blocked=False, effect_disposition=None,
                        budget=_budget_for_agent(agent), tool_duration=0.0)
    tm = msgs[-1]
    return "[batching hint]" in str(tm.get("content"))


def test_live_commit_path_both_orderings():
    mA = [_asst("a"), _tool("a"), _asst("b"), _tool("b"), _asst("c")]
    mB = [_asst("a"), _tool("a"), _asst("b"), _tool("b")]
    mC = [_asst("a"), _tool("a"), _asst("b"), _tool("b"),
          {"role": "assistant", "content": "", "tool_calls": [
              {"id": "c1", "name": "read_file", "arguments": "{}"},
              {"id": "c2", "name": "terminal", "arguments": "{}"},
              {"id": "c3", "name": "search_files", "arguments": "{}"}]}]
    assert _run(list(mA), "c", 1) is True, "row-present: 3rd single must fire"
    assert _run(list(mB), "c", 1) is True, "row-absent (approval path): 3rd single must fire"
    assert _run(list(mC), "c2", 3) is False, "batch in flight must never nag"
    assert _run(mA + [_tool("c"), _asst("d")], "d", 1) is False, "4th single stays silent"
    assert _run([_asst("a"), _tool("a"), _asst("b")], "b", 1) is False, "2nd single stays silent"
