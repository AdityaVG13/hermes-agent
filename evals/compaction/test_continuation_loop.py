"""Unit tests for the closed-loop continuation probe — no model calls, no database."""
import json

from evals.compaction import continuation_loop as L


def test_plan_block_names_the_in_progress_item():
    todos = {"todos": [
        {"id": "a", "content": "baseline capture", "status": "completed"},
        {"id": "b", "content": "Cut the tail at the item boundary", "status": "in_progress"},
        {"id": "c", "content": "run the eval", "status": "pending"},
    ]}
    fold = {"anchor": 2, "rows": [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_name": "todo_list", "content": json.dumps(todos)},
    ]}
    block = L.plan_block(fold)
    assert "IN PROGRESS: Cut the tail at the item boundary" in block
    assert "NEXT PENDING: run the eval" in block
    assert "(1/3 completed)" in block


def test_plan_block_without_todos_is_explicit():
    fold = {"anchor": 1, "rows": [{"role": "user", "content": "x"}]}
    assert "no todo list recorded" in L.plan_block(fold)


def test_orientation_patterns():
    assert L.is_orientation({"tool": "read_file", "args": {"path": "/r/AGENTS.md"}})
    assert L.is_orientation({"tool": "terminal", "args": {"command": "git status --short"}})
    assert L.is_orientation({"tool": "terminal", "args": {"command": "cd /r && git status --short"}})
    assert L.is_orientation({"tool": "session_search", "args": {"query": "continue"}})
    assert L.is_orientation({"tool": "delegate_task", "args": {"action": "list"}})
    assert L.is_orientation({"tool": "todo_list", "args": {}})
    assert not L.is_orientation({"tool": "terminal", "args": {"command": "pytest -q"}})
    assert not L.is_orientation({"tool": "read_file", "args": {"path": "/r/src/lexer.py"}})


def test_situation_block_carries_repo_state():
    rows = [
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_name": "terminal",
         "content": "## main...origin/main\n M src/a.py\n?? notes.md"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "tool_name": "terminal", "content": "abc1234def\n"},
    ]
    fold = {"pre": [(0, "terminal", {"command": "cd /repo && git status --short --branch"}, "k0"),
                    (2, "terminal", {"command": "git rev-parse HEAD"}, "k1")],
            "rows": rows}
    block = L.situation_block(fold)
    assert "cwd: /repo" in block
    assert "branch: main" in block
    assert "abc1234def" in block
    assert "dirty (2)" in block


def test_reestablish_beats_orientation():
    fold = {"last_target": {("file", "/r/AGENTS.md"): 1}, "writes": []}
    assert L.classify_loop({"tool": "read_file", "args": {"path": "/r/AGENTS.md"}}, fold) == "reestablish"
    assert L.classify_loop({"tool": "read_file", "args": {"path": "/r/README.md"}}, fold) == "orientation"
    assert L.classify_loop({"tool": "read_file", "args": {"path": "/r/src/a.py"}}, fold) == "fresh"


def test_replay_prefers_recorded_results():
    bank = {("file", "/r/a.py"): "print(1)"}
    assert L.replay(bank, {"tool": "read_file", "args": {"path": "/r/a.py"}}) == "print(1)"
    assert "no recorded read" in L.replay(bank, {"tool": "read_file", "args": {"path": "/r/b.py"}})
    assert "not executed" in L.replay(bank, {"tool": "patch", "args": {"path": "/r/a.py"}})


def test_classify_action_wrapper_tools():
    fold = {"last_target": {}, "writes": []}
    assert L.classify_action({"tool": "patch", "args": {"path": "/a"}}, fold) == "task"
    assert L.classify_action({"tool": "delegate_task", "args": {"action": "list"}}, fold) == "orientation"
    assert L.classify_action({"tool": "execute_code", "args": {"code": "git status"}}, fold) == "orientation"
    assert L.classify_action({"tool": "execute_code", "args": {"code": "patch('/a')"}}, fold) == "task"
    assert L.classify_action({"tool": "terminal", "args": {"command": "pytest -q"}}, fold) == "task"


def test_parse_one_accepts_a_single_object():
    assert L.parse_one('```json\n{"tool": "terminal", "args": {"command": "ls"}}\n```') == {
        "tool": "terminal", "args": {"command": "ls"}}
    assert L.parse_one("no call here") is None
