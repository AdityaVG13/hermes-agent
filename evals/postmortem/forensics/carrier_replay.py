"""Replay the Context Record across two revisions to measure a carrier change.

  python -m evals.postmortem.forensics.carrier_replay /path/state.db \
      --before b9f97874be^ --after b9f97874be

For every session with at least ``--min-user-rows`` user rows, builds the record's active
window with both revisions and reports how many sessions change GOAL or LATEST ASK, plus the
``extract_constraints`` totals. Read-only: the database is opened with ``mode=ro`` and the
revisions are read with ``git show`` (nothing is checked out).
"""
import argparse
import importlib.util
import sqlite3
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
COLS = "role, content, tool_calls, tool_call_id, display_kind, _compressed_summary, tool_name"


def load_revision(rev: str):
    """Import ``context_record.py`` as it existed at *rev*."""
    source = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{rev}:evals/postmortem/forensics/context_record.py"]
    ).decode()
    path = Path(tempfile.mkdtemp(prefix="carrier-replay-")) / "context_record.py"
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(f"carrier_replay_{rev.replace('^', '_')}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", default="HEAD")
    parser.add_argument("--min-user-rows", type=int, default=8)
    args = parser.parse_args()

    before, after = load_revision(args.before), load_revision(args.after)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sessions = [r[0] for r in con.execute(
        "select session_id from messages where role='user' group by session_id having count(*) >= ?",
        (args.min_user_rows,),
    )]
    changed = 0
    volume_before = volume_after = 0
    for sid in sessions:
        rows = [dict(r) for r in con.execute(
            f"select {COLS} from messages where session_id=? and active=1 order by id", (sid,)
        )]
        old, new = before.build_record(rows), after.build_record(rows)
        volume_before += len(before.extract_constraints(rows))
        volume_after += len(after.extract_constraints(rows))
        if (old.get("goal"), old.get("latest_ask")) != (new.get("goal"), new.get("latest_ask")):
            changed += 1
            print(f"{sid[:24]}  goal {old.get('goal')!r} -> {new.get('goal')!r}")
            print(f"{'':24}  ask  {old.get('latest_ask')!r} -> {new.get('latest_ask')!r}")
    print(f"\nsessions with >= {args.min_user_rows} user rows: {len(sessions)}")
    print(f"GOAL/LATEST ASK changed: {changed}")
    print(f"constraints: {volume_before} -> {volume_after}")


if __name__ == "__main__":
    main()
