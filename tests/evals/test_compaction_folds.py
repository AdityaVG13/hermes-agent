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
    fold_anchors,
    recap_carriers,
    recovery_tax,
    same_target,
    session_calls,
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
