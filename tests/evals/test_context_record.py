"""Context Record: shape, determinism, and the fields a fold must carry."""
import json

from evals.postmortem.forensics.context_record import build_record, last_goal, render_record


def _row(role, content="", tool_calls=None, tool_call_id="", tool_name="", summary=False):
    return {
        "role": role,
        "content": content,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "_compressed_summary": summary,
    }


def _call(cid, name, **args):
    return {"id": cid, "function": {"name": name, "arguments": json.dumps(args)}}


def _batch(rows, text, calls, results):
    rows.append(_row("assistant", text, tool_calls=calls))
    for call, result in zip(calls, results):
        rows.append(_row("tool", json.dumps(result), tool_call_id=call["id"], tool_name=call["function"]["name"]))


def test_goal_skips_injected_and_recap_rows():
    rows = [
        _row("user", "[CONTEXT SUMMARY] old recap", summary=True),
        _row("user", "[Earlier conversation] carried context"),
        _row("user", "fix the fold so it stops replaying"),
        _row("user", "[IMPORTANT] system note"),
    ]
    assert last_goal(rows) == "fix the fold so it stops replaying"


def test_one_line_per_batch_and_now_is_the_last():
    rows = [_row("user", "do the thing")]
    _batch(rows, "read the fold", [_call("c1", "read_file", path="evals/x.py", offset=10, limit=30)],
           [{"content_hash": "abc123", "success": True}])
    _batch(rows, "run the lane", [_call("c2", "terminal", command="make check")], [{"output": "ok"}])

    rec = build_record(rows, session_id="s1", first_row=1, last_row=len(rows))
    assert len(rec["done"]) == 1
    assert rec["now"] is not None
    text = render_record(rec)
    assert "DONE (completed; do not repeat):" in text
    assert "1. read the fold — read_file — ok" in text or "1. read the fold" in text
    assert "LAST FOLDED (done): run the lane" in text and "— ok" in text
    assert "evals/x.py#L10-39 (hash abc123)" in text
    assert "NEXT:" in text


def test_recovery_pointer_names_real_session_search_parameters():
    rows = [_row("user", "fix the fold")]
    query_form = build_record(rows, session_id="local-x")["loss"]["recovery"]
    assert "session_id=local-x" in query_form and "query=" in query_form
    assert "session=" not in query_form.replace("session_id=", "")
    row_form = build_record(rows, session_id="local-x", first_row=42, last_row=50)["loss"]["recovery"]
    assert "session_id=local-x" in row_form and "around_message_id=42" in row_form
    assert "around=" not in row_form


def test_record_is_byte_stable_for_the_same_rows():
    rows = [_row("user", "do it")]
    _batch(rows, "read", [_call("c1", "read_file", path="a.py", limit=5)], [{"content_hash": "h"}])
    assert build_record(rows)["text"] == build_record(rows)["text"]


def test_windows_drive_letter_is_not_a_line_suffix():
    from evals.postmortem.forensics.context_record import _basename
    assert _basename(r"C:\repo\a.py") == "a.py"
    assert _basename("src/a.py:12") == "a.py"
    assert _basename("src/a.py#L1-5") == "a.py"


def test_a_carried_edit_keeps_its_marker():
    rows = [_row("user", "go")]
    _batch(rows, "run", [_call("c1", "terminal", command="ls")], [{"output": "ok"}])
    rec = build_record(rows, carried_lines=["- /a.py: edited (1x)"])
    assert "/a.py (edited)" in rec["text"]


def test_older_files_trailer_is_not_a_file_handle():
    rows = [_row("user", "go")]
    _batch(rows, "run", [_call("c1", "terminal", command="ls")], [{"output": "ok"}])
    rec = build_record(rows, carried_lines=["- /a.py: read lines 1-5", "(3 older files not listed)"])
    assert "(3 older files not listed)" not in rec["text"]


def test_non_numeric_read_arguments_do_not_break_the_fold():
    rows = [_row("user", "go")]
    _batch(rows, "read", [_call("c1", "read_file", path="a.py", offset="all", limit="many")], [{}])
    assert "a.py" in build_record(rows)["text"]


def test_failed_result_is_marked_err():
    rows = [_row("user", "try it")]
    _batch(rows, "patch the file", [_call("c1", "patch", path="a.py")], [{"success": False, "error": "no match"}])
    rec = build_record(rows)
    assert "— err" in rec["text"]


def test_plan_and_next_come_from_the_todo_list():
    rows = [_row("user", "work the plan")]
    _batch(rows, "check todos", [_call("c1", "todo_list")], [{"todos": [
        {"id": "1", "content": "done step", "status": "completed"},
        {"id": "2", "content": "wire the record", "status": "in_progress"},
        {"id": "3", "content": "ship it", "status": "pending"},
    ]}])
    rec = build_record(rows)
    assert rec["next"].startswith("2. wire the record")
    assert "PLAN:" in rec["text"]
    assert "done step" not in rec["text"]


def test_plan_is_capped_and_reports_the_remainder():
    rows = [_row("user", "work the plan")]
    todos = [{"id": str(i), "content": f"step {i}", "status": "pending"} for i in range(12)]
    _batch(rows, "check todos", [_call("c1", "todo_list")], [{"todos": todos}])
    rec = build_record(rows)
    assert len(rec["plan"]) == 8
    assert rec["plan_more"] == 4
    assert "4 more open items" in rec["text"]


def test_a_none_goal_does_not_poison_the_next_fold():
    rows = [_row("user", "fix the fold")]
    rec = build_record(rows, carried_goal="(none)")
    assert rec["goal"] == "fix the fold"


def test_an_edited_file_is_marked_edited_not_read():
    rows = [_row("user", "edit it")]
    _batch(rows, "patch", [_call("c1", "patch", path="/x/a.py")], [{"success": True}])
    rec = build_record(rows)
    assert "/x/a.py (edited)" in rec["text"]


def test_non_string_content_does_not_crash_the_fold():
    rows = [{"role": "user", "content": [{"type": "text", "text": "hi"}], "tool_calls": None,
             "tool_call_id": "", "tool_name": "", "_compressed_summary": False}]
    rec = build_record(rows)
    assert rec["goal"] == "(none)"


def test_deterministic_and_bounded():
    rows = [_row("user", "go")]
    for i in range(60):
        _batch(rows, f"step {i} " + "x" * 180, [_call(f"c{i}", "read_file", path=f"f{i}.py")], [{"success": True}])
    a = build_record(rows, max_batches=40, max_tokens=2000)
    b = build_record(rows, max_batches=40, max_tokens=2000)
    assert a["text"] == b["text"]
    assert len(a["done"]) <= 40
    assert "earlier batches omitted" in a["text"]
    assert len(a["text"]) // 4 <= 2000
    assert "LOSS:" in a["text"] and "session_search(" in a["text"]
