"""Coverage test for the Context Record: would the record have named what was replayed?

For every real fold: build the record from the calls the replay test uses as the pre-fold
set (the last ``--lookback`` calls before the fold, not the fold's structural turn window),
then take the calls the model actually replayed afterwards and ask whether the record named
them. The approximation favours the record: a wider window can only add coverage.

  covered  — the record said this file/skill/command; the model had it and replayed anyway
  dropped  — the action was inside the window, but the record's rendering lost it
  outside  — the action predates the window (older knowledge, not the record's job)

Run: python -m evals.postmortem.forensics.record_coverage /path/state.db
"""
import argparse
import collections
import json
import sqlite3
from pathlib import Path

from evals.postmortem.forensics.context_record import _args, _name, _tool_calls, build_record

COLS = "id, role, content, tool_calls, tool_call_id, tool_name, _compressed_summary"


def calls_of(rows):
    out = []
    for index, row in enumerate(rows):
        if row.get("role") != "assistant":
            continue
        for call in _tool_calls(row):
            name = _name(call)
            if not name:
                continue
            args = _args(call)
            key = name + "|" + json.dumps(args, sort_keys=True, default=str)[:400]
            out.append((index, name, args, key))
    return out


def target(name, args):
    if name in ("read_file", "patch", "write_file"):
        return ("file", str(args.get("path") or ""))
    if name == "skill_view":
        return ("skill", str(args.get("name") or args.get("file_path") or ""))
    if name == "search_files":
        return ("query", str(args.get("pattern") or ""))
    if name in ("terminal", "execute_code"):
        raw = " ".join(str(args.get("command") or args.get("code") or "").split())
        return ("cmd", raw[:60])
    return ("tool", name)


def in_text(kind, value, text):
    if not value:
        return True  # nothing to name; do not blame the record
    if kind == "cmd":
        return value[:30] in text
    return value in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--lookback", type=int, default=250)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--max-folds", type=int, default=400)
    args = ap.parse_args()

    path = "/" + Path(args.db).resolve().as_posix().lstrip("/")
    con = sqlite3.connect("file:" + path + "?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sids = [r[0] for r in con.execute(
        "select session_id from messages where _compressed_summary=1 group by session_id"
    )]

    stats = collections.Counter()
    by_regime = {"plan": collections.Counter(), "no_plan": collections.Counter()}
    by_tool = collections.defaultdict(collections.Counter)
    examples = {"dropped": [], "outside": []}
    folds_seen = 0
    sessions = 0

    for sid in sids:
        rows = [dict(r) for r in con.execute(
            f"select {COLS} from messages where session_id=? order by id", (sid,)
        )]
        anchors = [i for i, r in enumerate(rows) if r.get("_compressed_summary")]
        if not anchors:
            continue
        has_plan = bool(con.execute(
            "select 1 from messages where session_id=? and tool_name='todo_list' limit 1", (sid,)
        ).fetchone())
        regime = "plan" if has_plan else "no_plan"
        calls = calls_of(rows)
        if len(calls) < 40:
            continue
        sessions += 1

        for anchor_index, anchor in enumerate(anchors):
            if folds_seen >= args.max_folds:
                break
            folds_seen += 1
            pre = [c for c in calls if c[0] < anchor][-args.lookback:]
            if not pre:
                continue
            start = pre[0][0]
            window = rows[start:anchor]
            rec = build_record(window, session_id=sid, first_row=window[0]["id"] if window else 0,
                               last_row=window[-1]["id"] if window else 0)
            text = rec["text"]
            before = {c[3] for c in pre}
            stop = anchors[anchor_index + 1] if anchor_index + 1 < len(anchors) else None
            post = [c for c in calls
                    if c[0] > anchor and (stop is None or c[0] < stop)][:args.window]
            window_calls = {(c[1], json.dumps(c[2], sort_keys=True, default=str)) for c in pre}

            for _, name, call_args, key in post:
                if key not in before:
                    continue
                kind, value = target(name, call_args)
                stats["repeats"] += 1
                by_regime[regime]["repeats"] += 1
                by_tool[name]["repeats"] += 1
                if in_text(kind, value, text):
                    stats["covered"] += 1
                    by_regime[regime]["covered"] += 1
                    by_tool[name]["covered"] += 1
                    continue
                was_in_window = (name, json.dumps(call_args, sort_keys=True, default=str)) in window_calls
                bucket = "dropped" if was_in_window else "outside"
                stats[bucket] += 1
                by_regime[regime][bucket] += 1
                by_tool[name][bucket] += 1
                if len(examples[bucket]) < 12:
                    examples[bucket].append(f"{name}: {value[:90]}")

    assert stats["covered"] + stats["dropped"] + stats["outside"] == stats["repeats"], (
        "a replayed call was classified into no bucket"
    )
    print(f"sessions: {sessions}  folds: {folds_seen}")
    total = max(1, stats["repeats"])
    print(f"replayed calls: {stats['repeats']}")
    for bucket in ("covered", "dropped", "outside"):
        print(f"  {bucket:8s} {stats[bucket]:5d}  {stats[bucket]/total:6.1%}")
    print("\nby regime:")
    for regime, c in by_regime.items():
        t = max(1, c["repeats"])
        print(f"  {regime:8s} repeats {c['repeats']:5d}  covered {c['covered']/t:6.1%}"
              f"  dropped {c['dropped']/t:6.1%}  outside {c['outside']/t:6.1%}")
    print("\nby tool (repeats / covered / dropped / outside):")
    for name, c in sorted(by_tool.items(), key=lambda kv: -kv[1]["repeats"])[:10]:
        print(f"  {name:14s} {c['repeats']:5d}  {c['covered']:5d}  {c['dropped']:5d}  {c['outside']:5d}")
    for bucket in ("dropped", "outside"):
        if examples[bucket]:
            print(f"\n{bucket} examples:")
            for line in examples[bucket]:
                print("  " + line)


if __name__ == "__main__":
    main()
