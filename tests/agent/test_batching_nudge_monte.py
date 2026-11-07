"""Seeded Monte Carlo property harness for the batching campaign (2026-11-07).

Distilled from scratch/monte_batching.py, which ran the full-size sweep
(12,000 oracle cases, 3,000 crash shapes, 3,000 planner batches, 400×3
prefix mutations — zero mismatches, zero crashes). This file keeps a
fast, deterministic slice of the same properties in the suite:

  A) the REAL ``_commit_tool_result`` fire/silence verdict equals an
     independent SPEC oracle over randomized transcripts (both append
     timings, batches in flight, errors/blocked, execute_code, attr
     present/absent, dict/object rows, mid-batch sibling results)
  B) the streak walker + nudge never raise on garbage message shapes
  C) ``_plan_tool_batch_segments`` preserves call order exactly, keeps
     writers from sharing a path inside any parallel run, keeps barrier
     tools out of parallel runs, and parallelizes pure readers
  D) the verdict is prefix-irrelevant: only the trailing acting turns
     decide it
"""

import json
import random
from types import SimpleNamespace

import agent.tool_executor as te
from agent.tool_executor import (
    _ToolCallRef,
    _batching_nudge_note,
    _budget_for_agent,
    _commit_tool_result,
    _single_call_streak,
)
from agent.tool_dispatch_helpers import _plan_tool_batch_segments

TOOL_POOL = ["read_file", "terminal", "search_files", "execute_code", "skill_view", "web_search"]


def make_row(cid, name="read_file", as_object=False):
    if as_object:
        return SimpleNamespace(role="assistant", content="",
                               tool_calls=[SimpleNamespace(id=cid, name=name, arguments="{}")])
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": cid, "name": name, "arguments": "{}"}]}


def make_multi_row(cids, as_object=False):
    if as_object:
        return SimpleNamespace(role="assistant", content="",
                               tool_calls=[SimpleNamespace(id=c, name="read_file", arguments="{}") for c in cids])
    return {"role": "assistant", "content": "",
            "tool_calls": [{"id": c, "name": "read_file", "arguments": "{}"} for c in cids]}


def make_tool_row(cid, as_object=False):
    if as_object:
        return SimpleNamespace(role="tool", tool_call_id=cid, content="{}")
    return {"role": "tool", "tool_call_id": cid, "content": "{}"}


def build_history(rng, turns):
    rows, uid = [], 0
    for kind in turns:
        uid += 1
        as_obj = rng.random() < 0.3
        if kind == "single":
            cid = f"h{uid}"
            rows.append(make_row(cid, rng.choice(TOOL_POOL), as_obj))
            rows.append(make_tool_row(cid, as_obj))
        elif kind == "multi":
            k = rng.randint(2, 4)
            cids = [f"h{uid}_{j}" for j in range(k)]
            rows.append(make_multi_row(cids, as_obj))
            rows.extend(make_tool_row(c, as_obj) for c in cids)
        elif kind == "text":
            rows.append({"role": "assistant", "content": "note", "tool_calls": None})
        elif kind == "empty":
            rows.append({"role": "assistant", "content": "", "tool_calls": []})
        elif kind == "user":
            rows.append({"role": "user", "content": "hi"})
    return rows


def suffix_singles(turns):
    n = 0
    for kind in reversed(turns):
        if kind == "single":
            n += 1
        else:
            break
    return n


def oracle(turns, batch, name, is_error, blocked, attr):
    """SPEC, written independently of the walker implementation:
    no hint on error/blocked/execute_code or any batch of 2+; otherwise fire
    iff the current single completes a streak of exactly three."""
    if is_error or blocked or name == "execute_code":
        return False
    if attr is not None and attr >= 2:
        return False
    if batch >= 2:
        return False
    return (1 + suffix_singles(turns)) == 3


def run_commit(msgs, cid, name, is_error, blocked, attr, task):
    agent = SimpleNamespace(
        **({"_current_tool_batch_size": attr} if attr is not None else {}),
        _read_tracker=SimpleNamespace(record=lambda *a, **k: None),
        _touch_activity=lambda *a, **k: None,
        _subdirectory_hints=SimpleNamespace(check_tool_call=lambda *a, **k: None),
        _tool_result_content_for_active_model=lambda n, r: r,
        tool_progress_callback=None, session_db=None)
    te.note_tool_result = lambda *a, **k: None
    ref = _ToolCallRef(name=name, args={"path": "/etc/hostname"}, call_id=cid, task_id=task, trace=[])
    _commit_tool_result(agent, msgs, ref, '{"ok": true}',
                        is_error=is_error, blocked=blocked, effect_disposition=None,
                        budget=_budget_for_agent(agent), tool_duration=0.0)
    return "[batching hint]" in str(msgs[-1].get("content"))


def test_oracle_differential_over_random_transcripts():
    rng = random.Random(1337)
    fired = 0
    for i in range(250):
        hist = [rng.choice(["single", "single", "multi", "text", "user", "empty"])
                for _ in range(rng.randint(0, 9))]
        batch = rng.choices([1, 2, 3], weights=[70, 20, 10])[0]
        name = rng.choice(TOOL_POOL)
        is_error, blocked = rng.random() < 0.12, rng.random() < 0.06
        row_present = rng.random() < 0.6
        attr = batch if rng.random() < 0.8 else (None if row_present else batch)
        correctness = not (attr is None and not row_present)
        msgs = build_history(rng, hist)
        cid = f"cur{i}"
        if row_present:
            if batch == 1:
                msgs.append(make_row(cid, name, rng.random() < 0.3))
            else:
                cids = [f"sib{i}_{j}" for j in range(batch - 1)] + [cid]
                rng.shuffle(cids)
                msgs.append(make_multi_row(cids, rng.random() < 0.3))
                for c in cids[:-1]:
                    if rng.random() < 0.4:
                        msgs.append(make_tool_row(c))
        got = run_commit(msgs, cid, name, is_error, blocked, attr, f"mc-{i}")
        fired += got
        if correctness:
            want = oracle(hist, batch, name, is_error, blocked, attr)
            assert got == want, (hist, batch, name, is_error, blocked, row_present, attr, want, got)
    assert 0 < fired < 250, "generator must reach both verdicts"


def test_crash_fuzz_garbage_shapes():
    rng = random.Random(2468)
    for i in range(400):
        garbage = []
        for _ in range(rng.randint(0, 8)):
            r = rng.random()
            if r < 0.15:
                garbage.append(rng.choice(["string-row", None, 42, [], {"no_role": True}]))
            elif r < 0.3:
                garbage.append({"role": rng.choice(["assistant", "tool", "user", None]),
                                "tool_calls": rng.choice([None, [], [{"no_id": 1}], "junk", [None], [{"id": None}]]),
                                "content": None})
            elif r < 0.4:
                garbage.append(SimpleNamespace(role="assistant", tool_calls="junk"))
            else:
                garbage.append(make_row(f"g{i}", "read_file"))
        _single_call_streak(garbage, rng.choice([None, "", f"g{i}", "missing"]))
        _batching_nudge_note(SimpleNamespace(), garbage, "read_file",
                             is_error=bool(rng.getrandbits(1)), blocked=bool(rng.getrandbits(1)),
                             tool_call_id=rng.choice(["", "x"]))


def _fake_call(cid, name, args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def test_planner_properties_over_random_batches():
    rng = random.Random(9001)
    paths = ["/tmp/mc/a.txt", "/tmp/mc/b.txt", "/tmp/mc/sub/c.txt"]
    tools = [("read_file", "reader"), ("write_file", "writer"), ("patch", "writer"),
             ("search_files", "safe"), ("skill_view", "safe"), ("terminal", "barrier")]
    for i in range(400):
        k = rng.randint(2, 8)
        calls, truth = [], []
        for j in range(k):
            name, cls = rng.choice(tools)
            args = {"path": rng.choice(paths)} if name in ("read_file", "write_file", "patch") else \
                   ({"pattern": "x"} if name == "search_files" else {})
            calls.append(_fake_call(f"c{j}", name, args))
            truth.append((f"c{j}", name, cls, args.get("path")))
        segs = _plan_tool_batch_segments(calls)
        flat = [c.id for _, cs in segs for c in cs]
        assert flat == [t[0] for t in truth], f"order not preserved: {flat}"
        for kind, cs in segs:
            meta = {t[0]: t for t in truth}
            if kind == "parallel":
                assert len(cs) >= 2, "parallel run of one call"
                assert all(meta[c.id][1] != "terminal" for c in cs), "terminal inside parallel run"
                for c in cs:
                    _, nm, cls, p = meta[c.id]
                    if cls == "writer":
                        for c2 in cs:
                            if c2.id != c.id and meta[c2.id][3] == p:
                                raise AssertionError(f"writer {nm}({p}) parallel with {meta[c2.id][1]}")
        if all(t[1] == "read_file" and t[3] == paths[0] for t in truth) and k >= 2:
            assert len(segs) == 1 and segs[0][0] == "parallel", "same-file readers must parallelize"


def test_prefix_irrelevance():
    rng = random.Random(31415)
    for i in range(60):
        k_tail = rng.randint(1, 3)
        expect = k_tail == 2
        verdicts = set()
        for _ in range(3):
            head = [rng.choice(["single", "multi", "text", "user"]) for _ in range(rng.randint(0, 4))]
            turns = head + ["user"] + ["single"] * k_tail
            msgs = build_history(rng, turns)
            cid = f"m{i}"
            msgs.append(make_row(cid, "read_file"))
            verdicts.add(run_commit(msgs, cid, "read_file", False, False, 1, f"mm-{i}"))
        assert verdicts == {expect}, (k_tail, verdicts, expect)
