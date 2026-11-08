"""Lane: compaction folds — how often a session folds, what the receipt costs, and what the model
re-buys afterwards. OBSERVED from the persisted transcript.

    python -m evals.postmortem.forensics.compaction_folds --db state_copy.db [--all-sessions]

Anchors are rows the host persisted as a compaction summary (``_compressed_summary``). Both
engines set that flag — the built-in compressor when it writes its handoff, OptChat when it
folds (it stamps the host key deliberately). Rows that merely *carry* a recap without the flag
are counted separately as ``recap_carriers``: they are re-serves or pre-flag generations, not
fold events, and mixing them into the cadence is what made an earlier measurement report a
median of 8 tool calls between folds instead of 34.

Three numbers per session:

* **cadence** — tool calls between consecutive folds, boundaries with no calls between them
  collapsed (a recap row and its summary row are one boundary, not two).
* **recovery tax** — of the first ``--window`` tool calls after a fold, how many name a target
  (skill, file, command) that was already named in the ``--lookback`` calls before it. That is
  the working set the fold threw away and the model had to buy again.
* **receipt** — bytes and tokens (chars/4) of the fold's summary row, with the annuity estimate
  ``receipt_tokens * median_gap_calls``: a receipt is re-read every turn until the next fold,
  so its size is paid per turn, not once.
"""
from __future__ import annotations

import json
import re
import statistics
from itertools import pairwise
from typing import Any, Iterable, Optional

from evals.postmortem.forensics.common import Run

CLIFF_HEADER = "The following is a summary of your previous actions"

_REF_SUFFIX = re.compile(r"[#:][Ll]?\d+(?:-[Ll]?\d+)?$")
_WS = re.compile(r"\s+")

_FILE_TOOLS = {"read_file", "patch", "write_file"}


def _tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    tc = row.get("tool_calls")
    if not tc:
        return []
    if isinstance(tc, str):
        try:
            tc = json.loads(tc)
        except (ValueError, TypeError):
            return []
    return [c for c in tc if isinstance(c, dict)] if isinstance(tc, list) else []


def _args(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except (ValueError, TypeError):
            return {}
    return a if isinstance(a, dict) else {}


def _norm_path(p: Any) -> str:
    p = str(p or "").strip().strip("\"'")
    p = _REF_SUFFIX.sub("", p)
    while p.startswith("./"):
        p = p[2:]
    return p


def _cmd_head(c: Any) -> str:
    return _WS.sub(" ", str(c or "")).strip()[:160]


def tool_target(name: str, args: dict[str, Any]) -> tuple[str, str]:
    """Map a tool call to ``(kind, target)``. ``other`` carries no target and never matches."""
    if name == "skill_view":
        return "skill", str(args.get("name") or "")
    if name in _FILE_TOOLS:
        return "file", _norm_path(args.get("path"))
    if name == "terminal":
        return "cmd", _cmd_head(args.get("command"))
    return "other", ""


def same_target(a: str, b: str) -> bool:
    """Relative/absolute forms of one path are the same target; bare basenames are not
    (``mod.rs`` appears in every crate)."""
    if not a or not b:
        return False
    return a == b or a.endswith("/" + b) or b.endswith("/" + a)


def session_calls(rows: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    """``(row_index, kind, target)`` for every tool call, in transcript order."""
    out: list[tuple[int, str, str]] = []
    for i, row in enumerate(rows):
        if row.get("role") != "assistant":
            continue
        for call in _tool_calls(row):
            fn = call.get("function") or {}
            kind, target = tool_target(str(fn.get("name") or ""), _args(call))
            out.append((i, kind, target))
    return out


def fold_anchors(rows: list[dict[str, Any]]) -> list[int]:
    """Row indices the host persisted as a compaction summary."""
    return [i for i, r in enumerate(rows) if r.get("_compressed_summary")]


def recap_carriers(rows: list[dict[str, Any]]) -> int:
    """Rows carrying a recap without the flag — re-serves or pre-flag generations."""
    return sum(
        1 for r in rows
        if CLIFF_HEADER in (r.get("content") or "") and not r.get("_compressed_summary")
    )


def cadence_gaps(calls: list[tuple[int, str, str]], anchors: list[int]) -> list[int]:
    """Tool calls strictly between consecutive fold boundaries.

    Anchors with no call between them are one boundary (a recap row plus its summary row);
    the later one is dropped rather than recorded as a zero-length interval.
    """
    if len(anchors) < 2:
        return []
    positions = sorted(a for a, _, _ in calls)
    gaps: list[int] = []
    for prev, cur in pairwise(anchors):
        lo = sum(1 for p in positions if p <= prev)
        hi = sum(1 for p in positions if p <= cur)
        n = hi - lo
        if n <= 0:
            continue
        gaps.append(n)
    return gaps


def _pre_window(
    calls: list[tuple[int, str, str]], center: int, lookback: int
) -> list[tuple[int, str, str]]:
    """The last ``lookback`` calls before ``center`` — a call budget, not a row distance.

    Row indices and call counts diverge in tool-heavy transcripts (a fold's own summary
    row, multi-row turns, tool results). Measuring "the last 250 calls" as "calls in the
    last 250 rows" silently shrinks the pre-fold set and undercounts every repeat.
    """
    if lookback <= 0:
        return []  # [-0:] would slice the whole list, not none of it
    return [c for c in calls if c[0] < center][-lookback:]


def recovery_tax(
    calls: list[tuple[int, str, str]], anchors: list[int], *, window: int = 12, lookback: int = 250
) -> dict[str, Any]:
    """Of the first ``window`` calls after each fold, how many re-buy a pre-fold target."""
    tot = {"windows": 0, "calls": 0, "reacquired": 0}
    by_kind: dict[str, dict[str, int]] = {}
    for index, anchor in enumerate(anchors):
        # A window never crosses the next fold: those calls belong to another epoch.
        stop = anchors[index + 1] if index + 1 < len(anchors) else None
        before = [(k, t) for _, k, t in _pre_window(calls, anchor, lookback) if t]
        after = [(k, t) for i, k, t in calls
                 if i > anchor and (stop is None or i < stop)][:window]
        if not after:
            continue
        tot["windows"] += 1
        for kind, target in after:
            tot["calls"] += 1
            k = by_kind.setdefault(kind, {"calls": 0, "reacquired": 0})
            k["calls"] += 1
            if target and any(
                bk == kind and same_target(bt, target) for bk, bt in before
            ):
                tot["reacquired"] += 1
                k["reacquired"] += 1
    return {"totals": tot, "by_kind": by_kind}


def exact_key(call: dict[str, Any]) -> str:
    """Tool + canonical arguments — the identity of an action, not just of its target."""
    fn = call.get("function") or {}
    return str(fn.get("name") or "") + "|" + json.dumps(_args(call), sort_keys=True, default=str)[:400]


def session_calls_exact(rows: list[dict[str, Any]]) -> list[tuple[int, str, str]]:
    """``(row_index, tool, exact_key)`` for every tool call, in transcript order."""
    out: list[tuple[int, str, str]] = []
    for i, row in enumerate(rows):
        if row.get("role") != "assistant":
            continue
        for call in _tool_calls(row):
            tool = str((call.get("function") or {}).get("name") or "")
            key = exact_key(call)
            if tool and key.split("|", 1)[1]:
                out.append((i, tool, key))
    return out


def repeat_tax(
    calls: list[tuple[int, str, str]],
    anchors: list[int],
    *,
    window: int = 12,
    lookback: int = 250,
) -> dict[str, Any]:
    """Exact-argument repeats after a fold, against the same window at mid-cycle.

    A repeat is an identical ``(tool, canonical args)`` call that already appeared in the
    ``lookback`` before the window. The control sits at the midpoint between two folds, so
    workload and transcript position are held roughly constant and only the boundary varies.
    That is the mechanical signature of replay ("trace back and redo"), as opposed to a
    legitimate fresh read of the same file.
    """
    post = {"windows": 0, "calls": 0, "repeats": 0}
    control = {"windows": 0, "calls": 0, "repeats": 0}

    def observe(center: int, bucket: dict[str, int], stop: int | None = None) -> None:
        # A window never crosses the next fold: calls after it belong to another epoch.
        before = {k for _, _, k in _pre_window(calls, center, lookback)}
        after = [
            c for c in calls
            if c[0] > center and (stop is None or c[0] < stop)
        ][:window]
        if not after:
            return
        bucket["windows"] += 1
        for _, _, key in after:
            bucket["calls"] += 1
            if key in before:
                bucket["repeats"] += 1

    for index, anchor in enumerate(anchors):
        stop = anchors[index + 1] if index + 1 < len(anchors) else None
        observe(anchor, post, stop)
    for prev, cur in pairwise(anchors):
        observe(prev + max(0, (cur - prev) // 2), control, cur)
    return {"post": post, "control": control}


def receipt_bytes(rows: list[dict[str, Any]], anchors: list[int]) -> list[int]:
    return [len(rows[i].get("content") or "") for i in anchors]


def analyze_session(
    rows: list[dict[str, Any]], *, window: int = 12, lookback: int = 250
) -> dict[str, Any]:
    calls = session_calls(rows)
    exact = session_calls_exact(rows)
    anchors = fold_anchors(rows)
    gaps = cadence_gaps(calls, anchors)
    tax = recovery_tax(calls, anchors, window=window, lookback=lookback)
    repeats = repeat_tax(exact, anchors, window=window, lookback=lookback)
    receipts = receipt_bytes(rows, anchors)
    median_gap = statistics.median(gaps) if gaps else 0
    median_receipt = statistics.median(receipts) if receipts else 0
    return {
        "folds": len(anchors),
        "recap_carriers": recap_carriers(rows),
        "gap_calls_median": median_gap,
        "gap_calls_p25": (sorted(gaps)[len(gaps) // 4] if gaps else 0),
        "gap_calls_p75": (sorted(gaps)[(3 * len(gaps)) // 4] if gaps else 0),
        "receipt_chars_median": median_receipt,
        "receipt_tokens_median": int(median_receipt // 4),
        "receipt_annuity_tokens": int((median_receipt // 4) * median_gap),
        "recovery": tax,
        "repeats": repeats,
    }


def _aggregate(per_session: dict[str, dict[str, Any]]) -> dict[str, Any]:
    folds = sum(s["folds"] for s in per_session.values())
    gaps, receipts = [], []
    calls = reacq = 0
    by_kind: dict[str, dict[str, int]] = {}
    rep = {"post": {"windows": 0, "calls": 0, "repeats": 0},
           "control": {"windows": 0, "calls": 0, "repeats": 0}}
    for s in per_session.values():
        if s["folds"] >= 2:
            gaps.append(s["gap_calls_median"])
        if s["receipt_chars_median"]:
            receipts.append(s["receipt_chars_median"])
        t = s["recovery"]["totals"]
        calls += t["calls"]
        reacq += t["reacquired"]
        for k, v in s["recovery"]["by_kind"].items():
            agg = by_kind.setdefault(k, {"calls": 0, "reacquired": 0})
            agg["calls"] += v["calls"]
            agg["reacquired"] += v["reacquired"]
        for bucket in ("post", "control"):
            for field in ("windows", "calls", "repeats"):
                rep[bucket][field] += s["repeats"][bucket][field]
    return {
        "sessions": len(per_session),
        "folds": folds,
        "uninstrumented_sessions": sum(1 for s in per_session.values() if s["folds"] == 0 and s["recap_carriers"]),
        "gap_calls_median_of_medians": statistics.median(gaps) if gaps else 0,
        "receipt_chars_median": statistics.median(receipts) if receipts else 0,
        "recovery_calls": calls,
        "recovery_reacquired": reacq,
        "recovery_share": round(reacq / calls, 4) if calls else 0.0,
        "by_kind": {k: {**v, "share": round(v["reacquired"] / v["calls"], 4) if v["calls"] else 0.0}
                    for k, v in sorted(by_kind.items())},
        "repeat_post": rep["post"],
        "repeat_control": rep["control"],
        "repeat_post_share": round(rep["post"]["repeats"] / rep["post"]["calls"], 4)
        if rep["post"]["calls"] else 0.0,
        "repeat_control_share": round(rep["control"]["repeats"] / rep["control"]["calls"], 4)
        if rep["control"]["calls"] else 0.0,
        "repeat_ratio": round(rep["post"]["repeats"] / rep["control"]["repeats"], 2)
        if rep["control"]["repeats"] else 0.0,
    }


def _sessions(run: Run, all_sessions: bool) -> Iterable[str]:
    if all_sessions:
        return sorted(run.sessions)
    return run.in_run


def main(argv=None) -> int:
    ap = Run.parser((__doc__ or "").split("\n\n")[0])
    ap.add_argument("--all-sessions", action="store_true",
                    help="scan every session in the DB, not just the root's run tree")
    ap.add_argument("--window", type=int, default=12, help="tool calls observed after a fold")
    ap.add_argument("--lookback", type=int, default=250, help="tool calls that define the pre-fold set")
    args = ap.parse_args(argv)
    run = Run.open(args.db, root=args.root, out=args.out)
    window, lookback = args.window, args.lookback
    per: dict[str, dict[str, Any]] = {}
    for sid in _sessions(run, args.all_sessions):
        rows = run.messages(sid, "id, role, content, tool_calls, _compressed_summary")
        if not rows:
            continue
        per[sid] = analyze_session(rows, window=window, lookback=lookback)
    agg = _aggregate(per)
    run.write("compaction_folds.json", {"aggregate": agg, "sessions": per})
    lines = [
        "# Compaction folds",
        "",
        f"- sessions: {agg['sessions']}  folds: {agg['folds']}  "
        f"uninstrumented (recap, no fold record): {agg['uninstrumented_sessions']}",
        f"- tool calls between folds (median of session medians): {agg['gap_calls_median_of_medians']}",
        f"- receipt (median): {agg['receipt_chars_median']} chars "
        f"(~{int(agg['receipt_chars_median'] // 4)} tok)",
        f"- recovery tax: {agg['recovery_reacquired']}/{agg['recovery_calls']} "
        f"= {agg['recovery_share']:.1%} of the first {window} calls after a fold",
        f"- replay: {agg['repeat_post']['repeats']}/{agg['repeat_post']['calls']} = "
        f"{agg['repeat_post_share']:.1%} of post-fold calls repeat an exact pre-fold action "
        f"(mid-cycle control {agg['repeat_control_share']:.1%}; {agg['repeat_ratio']:.1f}x)",
        "",
        "| kind | calls | re-acquired | share |",
        "|---|---|---|---|",
    ]
    for kind, v in agg["by_kind"].items():
        lines.append(f"| {kind} | {v['calls']} | {v['reacquired']} | {v['share']:.1%} |")
    run.write("compaction_folds.md", "\n".join(lines) + "\n")
    print(f"wrote {run.out_dir / 'compaction_folds.json'}")
    print("\n".join(lines[2:]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
