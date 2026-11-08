"""Unit tests for the continuation eval — no model calls, no database."""
import json

from evals.compaction import continuation as C


def _assistant(call_id: str, name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)}}]}


def make_fold(n_files: int = 2, with_skill: bool = False) -> dict:
    rows, pre, last, writes = [], [], {}, []
    for i in range(n_files):
        path = f"/repo/f{i}.rs"
        index = len(rows)
        rows.append(_assistant(f"c{i}", "read_file", {"path": path}))
        rows.append({"role": "tool", "tool_name": "read_file", "tool_call_id": f"c{i}",
                     "content": f"BODY_{i} " + "x" * 40})
        pre.append((index, "read_file", {"path": path}, f"k{i}"))
        last[("file", path)] = index
    if with_skill:
        index = len(rows)
        rows.append(_assistant("cs", "skill_view", {"name": "deep-investigation"}))
        rows.append({"role": "tool", "tool_name": "skill_view", "tool_call_id": "cs",
                     "content": "the skill body"})
        pre.append((index, "skill_view", {"name": "deep-investigation"}, "ks"))
        last[("skill", "deep-investigation")] = index
    return {"session": "t", "anchor": len(rows), "rows": rows, "window": rows,
            "pre": pre, "last_target": last, "writes": writes, "summary": "SUM"}


def test_state_block_is_bounded():
    block = C.state_block(make_fold(10))
    assert block.count("--- /repo/") == 4


def test_state_block_carries_bodies_and_following():
    block = C.state_block(make_fold(2, with_skill=True))
    assert "BODY_0" in block
    assert "deep-investigation" in block and "unchanged" in block


def test_state_block_stays_under_budget():
    block = C.state_block(make_fold(6, with_skill=True))
    assert len(block) < 40_000


def test_classify_buckets():
    fold = make_fold(2)
    fold["writes"] = [(1, "file", "/repo/f0.rs")]  # edited after it was read
    assert C.classify({"tool": "read_file", "args": {"path": "/repo/f0.rs"}}, fold) == "verification"
    assert C.classify({"tool": "read_file", "args": {"path": "/repo/f1.rs"}}, fold) == "reestablish"
    assert C.classify({"tool": "read_file", "args": {"path": "/repo/new.rs"}}, fold) == "fresh"


def test_classify_command_counts_any_edit_since():
    fold = make_fold(2)
    fold["last_target"][("cmd", "pytest -q")] = 2
    fold["writes"] = [(3, "file", "/repo/f0.rs")]
    assert C.classify({"tool": "terminal", "args": {"command": "pytest -q"}}, fold) == "verification"
    assert C.classify({"tool": "terminal", "args": {"command": "git log"}}, fold) == "fresh"


def test_directive_arm_only():
    fold = make_fold(2)
    assert C.DIRECTIVE not in C.handoff(fold, "record_state")
    assert C.DIRECTIVE in C.handoff(fold, "record_state_directive")
    assert "CONTEXT RECORD" in C.handoff(fold, "record")


def test_proposals_from_tolerates_fences_and_prose():
    text = 'Here you go:\n```json\n[{"tool": "read_file", "args": {"path": "/a.rs"}}]\n```'
    assert C.proposals_from(text) == [{"tool": "read_file", "args": {"path": "/a.rs"}}]
    assert C.proposals_from('{"tool": "terminal"}') == [{"tool": "terminal", "args": {}}]
    assert C.proposals_from("not json at all") == []


def test_adoption_uses_target():
    fold = make_fold(1)
    text = C.handoff(fold, "record_state")
    assert C.adoption({"tool": "read_file", "args": {"path": "/repo/f0.rs"}}, text)
    assert not C.adoption({"tool": "read_file", "args": {"path": "/repo/elsewhere.rs"}}, text)
