"""Invariants of the compaction-folds forensics lane (evals/postmortem/forensics/compaction_folds.py).

Pure functions only: the lane's DB walk is exercised against a copy, never in CI. The contract
under test is the measurement's honesty — a fold is a flagged summary row, a re-serve is not a
fold, two anchors at one boundary are one interval, and only a target the model already had
counts as a re-buy.
"""
from __future__ import annotations

import json

from evals.postmortem.forensics.compaction_folds import (
    analyze_session,
    cadence_gaps,
    exact_key,
    fold_anchors,
    recap_carriers,
    recovery_tax,
    repeat_tax,
    same_target,
    session_calls,
    session_calls_exact,
    tool_target,
)


def _call(row_id, tool, **args):
    return {
        "id": row_id,
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": f"c{row_id}", "type": "function",
             "function": {"name": tool, "arguments": json.dumps(args)}}
        ],
        "_compressed_summary": 0,
    }


def _fold(row_id, text="The following is a summary of your previous actions:\n- stuff"):
    return {"id": row_id, "role": "user", "content": text, "tool_calls": None, "_compressed_summary": 1}


def _carrier(row_id, text="ask\n\nThe following is a summary of your previous actions:\n- stuff"):
    return {"id": row_id, "role": "user", "content": text, "tool_calls": None, "_compressed_summary": 0}


# ── target identity ───────────────────────────────────────────────────────────────────────────


def test_tool_target_kinds_and_ref_suffix():
    assert tool_target("skill_view", {"name": "deep-investigation"}) == ("skill", "deep-investigation")
    assert tool_target("read_file", {"path": "src/a.rs#L12-L40"}) == ("file", "src/a.rs")
    assert tool_target("read_file", {"path": "./src/a.rs:12"}) == ("file", "src/a.rs")
    assert tool_target("terminal", {"command": "  cargo   test -p x "}) == ("cmd", "cargo test -p x")
    assert tool_target("todo_list", {}) == ("other", "")


def test_same_target_relative_absolute_but_not_bare_basename():
    assert same_target("src/a.rs", "/Users/x/src/a.rs")
    assert same_target("/Users/x/src/a.rs", "src/a.rs")
    assert not same_target("src/a/mod.rs", "src/b/mod.rs")


# ── fold identity ────────────────────────────────────────────────────────────────────────────


def test_a_recap_without_the_flag_is_not_a_fold():
    rows = [_call(0, "read_file", path="a.rs"), _carrier(1), _fold(2)]
    assert fold_anchors(rows) == [2]
    assert recap_carriers(rows) == 1


def test_uninstrumented_session_reports_carriers_not_folds():
    rows = [_call(0, "read_file", path="a.rs"), _carrier(1), _call(2, "read_file", path="a.rs")]
    s = analyze_session(rows)
    assert s["folds"] == 0
    assert s["recap_carriers"] == 1


# ── cadence ─────────────────────────────────────────────────────────────────────────────────


def test_two_anchors_at_one_boundary_are_one_interval():
    rows = [
        _call(0, "read_file", path="a.rs"),
        _fold(1),
        _fold(2),
        _call(3, "read_file", path="b.rs"),
        _fold(4),
    ]
    gaps = cadence_gaps(session_calls(rows), fold_anchors(rows))
    assert gaps == [1]


# ── lookback is measured in calls, not rows ────────────────────────────────────────────────


def _filler(row_id):
    return {"id": row_id, "role": "assistant", "content": "thinking", "tool_calls": None,
            "_compressed_summary": 0}


def test_lookback_counts_calls_not_transcript_rows():
    # One call, then 300 rows with no calls, then the fold. A one-call lookback must still
    # find the pre-fold call; measuring distance in row indices finds nothing.
    rows = ([_call(0, "read_file", path="a.rs")]
            + [_filler(i) for i in range(1, 301)]
            + [_fold(301), _call(302, "read_file", path="a.rs")])
    r = repeat_tax(session_calls_exact(rows), fold_anchors(rows), lookback=1)
    assert r["post"]["repeats"] == 1
    tax = recovery_tax(session_calls(rows), fold_anchors(rows), lookback=1)
    assert tax["totals"]["reacquired"] == 1


def test_zero_lookback_means_no_pre_fold_set():
    rows = [_call(0, "read_file", path="a.rs"), _fold(1), _call(2, "read_file", path="a.rs")]
    r = repeat_tax(session_calls_exact(rows), fold_anchors(rows), lookback=0)
    assert r["post"]["repeats"] == 0


# ── recovery tax ────────────────────────────────────────────────────────────────────────────


def test_recovery_tax_counts_only_pre_existing_targets():
    rows = [
        _call(0, "read_file", path="src/a.rs"),
        _call(1, "skill_view", name="rch"),
        _fold(2),
        _call(3, "read_file", path="/abs/src/a.rs"),   # re-buys a.rs (relative vs absolute)
        _call(4, "skill_view", name="rch"),            # re-buys the skill
        _call(5, "read_file", path="src/new.rs"),      # new work, not a re-buy
        _call(6, "terminal", command="cargo test"),    # never seen before -> not a re-buy
    ]
    tax = recovery_tax(session_calls(rows), fold_anchors(rows))
    assert tax["totals"] == {"windows": 1, "calls": 4, "reacquired": 2}
    assert tax["by_kind"]["file"] == {"calls": 2, "reacquired": 1}
    assert tax["by_kind"]["skill"] == {"calls": 1, "reacquired": 1}
    assert tax["by_kind"]["cmd"] == {"calls": 1, "reacquired": 0}


def test_recovery_window_never_crosses_the_next_fold():
    rows = [
        _call(0, "read_file", path="src/a.rs"),
        _fold(1),
        _call(2, "read_file", path="src/a.rs"),   # fold 1's post window (re-buy)
        _fold(3),
        _call(4, "read_file", path="src/a.rs"),   # fold 3's post window
    ]
    tax = recovery_tax(session_calls(rows), fold_anchors(rows))
    # fold 1 must stop at fold 3's boundary: 1 call each, not 2 then 1
    assert tax["totals"] == {"windows": 2, "calls": 2, "reacquired": 2}


def test_receipt_annuity_is_size_times_gap():
    rows = [
        _call(0, "read_file", path="a.rs"),
        _fold(1, text="x" * 4000),
        _call(2, "read_file", path="b.rs"),
        _call(3, "read_file", path="c.rs"),
        _fold(4, text="y" * 8000),
    ]
    s = analyze_session(rows)
    assert s["folds"] == 2
    assert s["gap_calls_median"] == 2
    assert s["receipt_chars_median"] == 6000
    assert s["receipt_tokens_median"] == 1500
    assert s["receipt_annuity_tokens"] == 3000


# ── replay identity ──────────────────────────────────────────────────────────────────────────


def test_exact_key_normalizes_argument_order():
    a = {"function": {"name": "read_file", "arguments": json.dumps({"path": "a", "limit": 5})}}
    b = {"function": {"name": "read_file", "arguments": json.dumps({"limit": 5, "path": "a"})}}
    assert exact_key(a) == exact_key(b)
    assert exact_key(a) != exact_key(
        {"function": {"name": "read_file", "arguments": json.dumps({"path": "a", "limit": 6})}})


def test_repeat_tax_counts_only_identical_actions():
    rows = [
        _call(0, "read_file", path="src/a.rs"),
        _fold(1),
        _call(2, "read_file", path="src/a.rs"),   # identical action -> replay
        _call(3, "read_file", path="src/a.rs", limit=40),  # same target, different action
    ]
    r = repeat_tax(session_calls_exact(rows), fold_anchors(rows))
    assert r["post"] == {"windows": 1, "calls": 2, "repeats": 1}
    assert r["control"]["calls"] == 0  # one fold -> no mid-cycle control


def test_repeat_tax_control_sits_mid_gap():
    rows = [
        _call(0, "terminal", command="ls"),
        _fold(1),
        _call(2, "terminal", command="pwd"),
        _call(3, "terminal", command="ls"),   # mid-gap replay (control)
        _fold(4),
        _call(5, "terminal", command="ls"),   # post-fold replay
    ]
    r = repeat_tax(session_calls_exact(rows), fold_anchors(rows))
    # post is clipped at the next fold: fold 1 sees rows 2-3 (row 3 repeats), fold 4 sees row 5
    assert r["post"] == {"windows": 2, "calls": 3, "repeats": 2}
    # control at row 2 is clipped at fold 4: only row 3, which repeats the pre-window "ls"
    assert r["control"] == {"windows": 1, "calls": 1, "repeats": 1}


def test_repeat_tax_respects_the_lookback():
    rows = [
        _call(0, "read_file", path="a.rs"),
        _call(1, "read_file", path="z.rs"),
        _fold(2),
        _call(3, "read_file", path="a.rs"),
    ]
    r = repeat_tax(session_calls_exact(rows), fold_anchors(rows), lookback=1)
    assert r["post"]["repeats"] == 0
