"""Continuation eval: does a handoff change what the agent DOES next?

The recall eval (``runner.py``) scores whether facts survive compression. This arm
scores the opposite question: after a context boundary, does the agent continue the
task or re-establish state? Same real folds, same model, same prompt; only the
handoff block changes.

Arms
  summary              the handoff actually written at that fold in ``state.db``
  record               the Context Record built from the fold window
  record_state         record + working-set bodies (newest reads, <=4 files x 8K)
                       + "following skill X, unchanged" assertions
  record_state_directive   record_state + explicit resume rules

Probe
  The model states its next three tool calls. Each proposal is classified against
  the pre-fold window:
    fresh         the target was not in the window
    verification  it was, but the target (or, for commands, any file) was written after
    reestablish   it was, and nothing changed — the pathological re-buy

The bench sanity gate: the ``summary`` arm must show substantial re-establishment
intent. If it does not, the probe is wrong, not the world.

Run:
  python -m evals.compaction.continuation --db ~/.hermes/state.db --folds 8
  python -m evals.compaction.continuation --db ~/.hermes/state.db --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.compaction.runner import EVAL_USAGE, _call, _extract_json
from evals.postmortem.forensics.record_coverage import COLS, calls_of, target
from evals.postmortem.forensics.context_record import build_record

WRITE = {"patch", "write_file", "skill_manage"}
TOOLS = ("read_file", "patch", "write_file", "terminal", "search_files", "skill_view",
         "execute_code", "todo_list", "delegate_task")

PROMPT = """You are an AI agent resuming a work session after a context boundary. Your only \
context is below.

<CONTEXT>
{handoff}
</CONTEXT>

State your next THREE tool calls. Use only these tools: {tools}.
Return strict JSON only, no prose and no markdown fences:
[{{"tool": "<name>", "args": {{<arguments>}}}}]
"""

DIRECTIVE = (
    "RESUME RULES: The working set above is current as of the fold. Do not re-read a file "
    "or reload a skill unless you are about to edit it or its hash changed. Continue from NEXT."
)

DRY_PATH = "/tmp/x"


# ── folds ────────────────────────────────────────────────────────────────────────────────────
def load_folds(db: str, limit: int, min_calls: int = 40, window_calls: int = 250,
               max_per_session: int = 2) -> list[dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sids = [r[0] for r in con.execute(
        "select session_id from messages where _compressed_summary=1 group by session_id")]
    folds = []
    for sid in sids:
        if len(folds) >= limit:
            break
        rows = [dict(r) for r in con.execute(
            f"select {COLS} from messages where session_id=? order by id", (sid,))]
        anchors = [i for i, r in enumerate(rows) if r.get("_compressed_summary")]
        if not anchors:
            continue
        has_plan = bool(con.execute(
            "select 1 from messages where session_id=? and tool_name='todo_list' limit 1", (sid,)
        ).fetchone())
        if not has_plan:
            continue
        calls = calls_of(rows)
        taken = 0
        for anchor in anchors:
            if len(folds) >= limit or taken >= max_per_session:
                break
            pre = [c for c in calls if c[0] < anchor][-window_calls:]
            if len(pre) < min_calls:
                continue
            window = rows[pre[0][0]:anchor]
            last_target, writes = {}, []
            for index, name, args, _key in pre:
                kind, value = target(name, args)
                if value:
                    last_target[(kind, value)] = index
                if name in WRITE:
                    writes.append((index, kind, value))
            folds.append({
                "session": sid, "anchor": anchor, "rows": rows, "window": window,
                "pre": pre, "last_target": last_target, "writes": writes,
                "summary": (rows[anchor].get("content") or "").strip(),
            })
            taken += 1
    return folds


# ── handoffs ─────────────────────────────────────────────────────────────────────────────────
def _result_text(rows: list[dict], call_index: int) -> str:
    for j in range(call_index + 1, len(rows)):
        if rows[j].get("role") == "assistant":
            break
        if rows[j].get("role") == "tool" and isinstance(rows[j].get("content"), str):
            return rows[j]["content"]
    return ""


def _touched(name: str, args: dict) -> str:
    if name in ("read_file", "patch", "write_file"):
        return str(args.get("path") or "")
    if name == "skill_view":
        return str(args.get("name") or "")
    return ""


def state_block(fold: dict, *, max_files: int = 4, max_skills: int = 3,
                per_file: int = 8000) -> str:
    """Newest content for the files the fold window touched, plus following-assertions.

    Files carry content because an agent cannot edit from a pointer. Skills carry only an
    assertion + provenance: their bodies average 39K chars per reload and do not change
    inside a session, so the body is a protocol problem, not a working-set one.
    """
    files: dict[str, tuple[int, str]] = {}
    skills: dict[str, int] = {}
    for index, name, args, _key in reversed(fold["pre"]):
        if name in ("read_file", "patch", "write_file"):
            path = str(args.get("path") or "")
            if path and path not in files and len(files) < max_files:
                body = _result_text(fold["rows"], index)
                if body:
                    files[path] = (index, body[:per_file])
        elif name == "skill_view":
            skill = str(args.get("name") or "")
            if skill and skill not in skills and len(skills) < max_skills:
                skills[skill] = index
        if len(files) >= max_files and len(skills) >= max_skills:
            break

    lines = ["", "WORKING SET (content current as of the fold):"]
    if not files:
        lines.append("  (no file bodies captured)")
    for path, (index, body) in files.items():
        lines.append(f"  --- {path} (read at call {index}) ---")
        lines.append(body)
    lines.append("FOLLOWING (instructions unchanged since load):")
    if not skills:
        lines.append("  (none)")
    for skill, index in list(skills.items())[:max_skills]:
        lines.append(f"  {skill} — loaded at call {index}, unchanged")

    command = ""
    for index, name, args, _key in reversed(fold["pre"]):
        if name == "terminal":
            body = _result_text(fold["rows"], index)
            if body:
                command = (str(args.get("command") or "")[:120] + "\n" + body[:4000]).strip()
                break
    if command:
        lines += ["LAST COMMAND OUTPUT:", command]
    return "\n".join(lines)


def handoff(fold: dict, arm: str) -> str:
    if arm == "summary":
        return fold["summary"]
    record = build_record(fold["window"], session_id=fold["session"])["text"]
    if arm == "record":
        return record
    if arm == "record_state":
        return record + "\n" + state_block(fold)
    if arm == "record_state_directive":
        return record + "\n" + state_block(fold) + "\n" + DIRECTIVE
    raise ValueError(f"unknown arm {arm!r}")


# ── probe + scoring ─────────────────────────────────────────────────────────────────────────
def proposals_from(text: str) -> list[dict]:
    try:
        data = _extract_json(text)
    except (ValueError, json.JSONDecodeError):
        return []
    out = []
    if isinstance(data, dict):
        data = data["calls"] if isinstance(data.get("calls"), list) else [data]
    for item in data if isinstance(data, list) else []:
        if isinstance(item, dict) and isinstance(item.get("tool"), str):
            args = item.get("args")
            out.append({"tool": item["tool"], "args": args if isinstance(args, dict) else {}})
    return out


def classify(proposal: dict, fold: dict) -> str:
    kind, value = target(proposal["tool"], proposal["args"])
    if not value:
        return "fresh"
    last = fold["last_target"].get((kind, value))
    if last is None:
        return "fresh"
    if kind == "cmd":
        changed = any(index > last for index, _k, _v in fold["writes"])
    else:
        changed = any(index > last and k == kind and v == value
                      for index, k, v in fold["writes"])
    return "verification" if changed else "reestablish"


def adoption(proposal: dict, text: str) -> bool:
    value = target(proposal["tool"], proposal["args"])[1]
    if not value:
        return False
    return value[:60] in text


def score_arm(fold: dict, arm: str, proposals: list[dict]) -> dict:
    text = handoff(fold, arm)
    classes = [classify(p, fold) for p in proposals]
    adopted = [adoption(p, text) for p in proposals]
    return {"arm": arm, "proposals": proposals, "classes": classes,
            "adopted": adopted, "handoff_chars": len(text)}


def ground_truth(fold: dict) -> list[dict]:
    """The call the original agent actually made next (target only)."""
    out = []
    after = [c for c in calls_of(fold["rows"]) if c[0] > fold["anchor"]][:1]
    for _, name, args, _key in after:
        kind, value = target(name, args)
        if value:
            out.append({"kind": kind, "value": value})
    return out


# ── main ────────────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--arms", default="summary,record,record_state,record_state_directive")
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--dry-run", action="store_true", help="canned proposals; no model calls")
    ap.add_argument("--out", default="evals/compaction/results/continuation.json")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    folds = load_folds(args.db, args.folds)
    if not folds:
        print("no folds found (need a session with a todo list and >=40 pre-fold calls)")
        return 1

    results = []
    for fold in folds:
        truth = ground_truth(fold)
        for arm in arms:
            if args.dry_run:
                text = handoff(fold, arm)
                path = re.search(r"/[^ \n]+", text)
                proposals = [
                    {"tool": "read_file", "args": {"path": path.group(0) if path else "/tmp/x"}},
                    {"tool": "skill_view", "args": {"name": "x"}},
                    {"tool": "terminal", "args": {"command": "git status"}},
                ]
            else:
                text = _call(PROMPT.format(handoff=handoff(fold, arm), tools=", ".join(TOOLS)),
                             max_tokens=args.max_tokens)
                proposals = proposals_from(text)
            scored = score_arm(fold, arm, proposals)
            scored["session"] = fold["session"]
            scored["ground_truth"] = truth
            results.append(scored)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"usage": EVAL_USAGE, "results": results}, indent=1))
    print(f"folds {len(folds)}  arms {len(arms)}  calls {EVAL_USAGE['calls']}  -> {out}")

    print("\narm                       reestablish  verification  fresh  adopted  handoff chars")
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm and r["classes"]]
        if not rows:
            print(f"{arm:24s} (no proposals)")
            continue
        flat = [c for r in rows for c in r["classes"]]
        n = max(1, len(flat))
        ad = sum(1 for r in rows for x in r["adopted"] if x) / n
        chars = sum(r["handoff_chars"] for r in rows) / len(rows)
        print(f"{arm:24s} {flat.count('reestablish')/n:11.1%}  {flat.count('verification')/n:12.1%}"
              f"  {flat.count('fresh')/n:5.1%}  {ad:6.1%}  {chars:8.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
