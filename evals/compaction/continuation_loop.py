"""Closed-loop continuation probe.

One-shot probes ask "what would you do next?" and get orientation. This runs the model
through N steps against a replay bank seeded from the source session: a proposed call
returns the result the session actually produced for that target, so orientation can
finish and the measurement is how much of it the handoff forces.

Classes, in precedence order:
  reestablish  the target was already in the fold window and unchanged (the re-buy)
  orientation  a new inspection of the environment/process (AGENTS.md, git status,
               delegate_task(list), todo_list read) — bearings, not task work
  fresh        neither: a task action the window did not contain
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from evals.compaction.continuation import (
    DIRECTIVE, REPO_ROOT, _result_text, handoff, load_folds, state_block,
)
from evals.compaction.runner import EVAL_USAGE, _call
from evals.postmortem.forensics.context_record import build_record
from evals.postmortem.forensics.record_coverage import calls_of, target

FRAMING = ("You are mid-task in a long-running session. Continue the work; do not restart "
           "it and do not re-plan it.")
LOOP_PROMPT = """{framing}

<CONTEXT>
{handoff}
</CONTEXT>
{transcript}
State exactly ONE next tool call as strict JSON, no prose and no fences:
{{"tool": "<name>", "args": {{<arguments>}}}}
"""

ORIENT_BASENAMES = {"AGENTS.md", "README.md", "CLAUDE.md"}
ORIENT_CMD = re.compile(
    r"\b(git (status|log|branch|rev-parse|remote|diff --stat)|pwd|ls|find|date|"
    r"cat\s+\S*(AGENTS\.md|README\.md|CLAUDE\.md))\b")


def situation_block(fold: dict) -> str:
    """What the first orientation call would discover — carried instead of re-run."""
    cwd = ""
    for _index, name, args, _key in fold["pre"]:
        if name == "terminal":
            m = re.search(r"\bcd\s+([^\s&;|]+)", str(args.get("command") or ""))
            if m:
                cwd = m.group(1)

    def result_for(pred) -> str:
        for index, name, args, _key in reversed(fold["pre"]):
            if name == "terminal" and pred(" ".join(str(args.get("command") or "").split())):
                return _result_text(fold["rows"], index).strip()
        return ""

    status = result_for(lambda c: "git status" in c)
    branch = result_for(lambda c: "git branch --show-current" in c)
    head = result_for(lambda c: "git rev-parse HEAD" in c)
    if not branch:
        first = next((l for l in status.splitlines() if l.startswith("## ")), "")
        branch = first[3:].strip() if first else ""
    dirty = [l for l in status.splitlines() if re.match(r"^(\s?[MADRCU?!]{1,2} )", l)]

    last_cmd = ""
    for index, name, args, _key in reversed(fold["pre"]):
        if name == "terminal":
            body = _result_text(fold["rows"], index).strip()
            last_cmd = (str(args.get("command") or "")[:160]
                        + (("\n    -> " + body.splitlines()[0][:200]) if body else ""))
            break

    lines = ["", "SITUATION (repo state as of the fold; do not re-discover it):"]
    lines.append(f"  cwd: {cwd or '(unknown)'}")
    lines.append(f"  branch: {branch or '(unknown)'}   HEAD: {head or '(unknown)'}")
    if dirty:
        lines.append(f"  dirty ({len(dirty)}): " + "; ".join(d[:80] for d in dirty[:12]))
    else:
        lines.append("  dirty: (no git status in the folded window)")
    if last_cmd:
        lines.append("  last command: " + last_cmd)
    return "\n".join(lines) + "\n"


def plan_block(fold: dict) -> str:
    """The in-progress todo item verbatim — the step grain the record's NEXT line lacks."""
    todos = None
    for row in reversed(fold["rows"][:fold["anchor"]]):
        if row.get("role") == "tool" and row.get("tool_name") == "todo_list" \
                and isinstance(row.get("content"), str):
            try:
                todos = json.loads(row["content"]).get("todos")
            except (ValueError, TypeError):
                todos = None
            if todos:
                break
    if not todos:
        return "\nPLAN (authoritative): no todo list recorded\n"
    current = next((t for t in todos if t.get("status") == "in_progress"), None)
    pending = next((t for t in todos if t.get("status") not in ("completed", "in_progress")), None)
    lines = ["", "PLAN (authoritative):"]
    if current:
        lines.append(f"  IN PROGRESS: {current.get('content', '')}")
    else:
        lines.append("  IN PROGRESS: (none marked)")
    if pending:
        lines.append(f"  NEXT PENDING: {pending.get('content', '')}")
    done = sum(1 for t in todos if t.get("status") == "completed")
    lines.append(f"  ({done}/{len(todos)} completed)")
    return "\n".join(lines) + "\n"


def make_handoff(fold: dict, arm: str) -> str:
    base = build_record(fold["window"], session_id=fold["session"])["text"]
    if arm == "summary":
        return fold["summary"]
    if arm == "record":
        return base
    if arm == "record_plan":
        return base + plan_block(fold)
    if arm == "record_plan_state":
        return base + plan_block(fold) + state_block(fold)
    if arm == "record_plan_state_directive":
        return base + plan_block(fold) + state_block(fold) + "\n" + DIRECTIVE
    if arm == "record_plan_situation":
        return base + plan_block(fold) + situation_block(fold)
    if arm == "record_plan_situation_state":
        return base + plan_block(fold) + situation_block(fold) + state_block(fold)
    if arm == "record_plan_situation_state_directive":
        return (base + plan_block(fold) + situation_block(fold) + state_block(fold)
                + "\n" + DIRECTIVE)
    if arm == "record_state":
        return base + state_block(fold)
    raise ValueError(f"unknown arm {arm!r}")


def is_orientation(proposal: dict) -> bool:
    tool, args = proposal.get("tool"), proposal.get("args") or {}
    if tool == "delegate_task":
        return args.get("action") == "list"
    if tool == "todo_list":
        return args.get("action") in (None, "", "list", "read")
    if tool == "session_search":
        return True  # searching your own transcript is rebuilding state
    if tool == "read_file":
        return str(args.get("path", "")).rsplit("/", 1)[-1] in ORIENT_BASENAMES
    if tool == "terminal":
        return bool(ORIENT_CMD.search(" ".join(str(args.get("command", "")).split())))
    if tool == "search_files":
        return str(args.get("pattern", "")) in ("AGENTS.md", "README.md", "CLAUDE.md")
    return False


def tool_bank(fold: dict) -> dict[tuple[str, str], str]:
    """What the source session actually returned for each target (last non-empty)."""
    bank = {}
    rows = fold["rows"]
    for index, name, args, _key in calls_of(rows):
        kind, value = target(name, args)
        if not value or index >= fold["anchor"]:
            continue
        body = _result_text(rows, index)
        if body:
            bank[(kind, value)] = body[:6000]
    return bank


def _lookup(bank: dict, kind: str, value: str) -> str | None:
    body = bank.get((kind, value))
    if body is not None:
        return body
    for (k, v), b in bank.items():
        if k == kind and v and (v.endswith(value) or value.endswith(v)):
            return b
    return None


def replay(bank: dict, proposal: dict, fold: dict | None = None) -> str:
    tool = proposal.get("tool")
    args = proposal.get("args") or {}
    _kind, value = target(tool, args)
    if tool in ("patch", "write_file"):
        return "replay: edit acknowledged but not executed (read-only environment)."
    if tool == "delegate_task":
        return "replay: delegation is unavailable in the replay environment."
    body = _lookup(bank, _kind, value)
    if body is not None:
        return body
    if tool == "session_search":
        terms = [t for t in re.findall(r"\w{4,}", str(args.get("query", "")))]
        hits = []
        for row in (fold["rows"][:fold["anchor"]] if fold else []):
            c = row.get("content")
            if isinstance(c, str) and any(t.lower() in c.lower() for t in terms):
                hits.append(c[:300].replace("\n", " "))
                if len(hits) >= 4:
                    break
        return ("replay: matches from the folded transcript:\n" + "\n".join(hits)
                if hits else "replay: no matches in the folded transcript.")
    if tool == "terminal":
        segments = [s.strip() for s in re.split(r"\s*(?:&&|\|\||;|\|)\s*", str(args.get("command", "")))
                    if s.strip()]
        out = []
        for seg in segments:
            if seg.startswith("cd "):
                out.append(f"$ {seg}\n(ok)")
                continue
            hit = _lookup(bank, "cmd", seg)
            if hit is None and seg.startswith("cat "):
                for word in seg[4:].split():
                    hit = _lookup(bank, "file", word)
                    if hit:
                        break
            if hit is None and seg.startswith("git status") and fold is not None:
                hit = situation_block(fold).strip()
            if hit is None and seg in ("pwd", "git branch --show-current", "git rev-parse HEAD"):
                hit = situation_block(fold).strip() if fold is not None else None
            out.append(f"$ {seg}\n{(hit[:1500] if hit else '(replay: no recorded output)')}")
        return "\n".join(out)[:6000]
    if tool == "read_file":
        return "replay: no recorded read of this path in the source session."
    if tool == "skill_view":
        return "replay: this skill was not loaded in the source session."
    if tool == "search_files":
        return "replay: no recorded result for this search."
    return "replay: no recorded result for this call."


def classify_loop(proposal: dict, fold: dict) -> str:
    kind, value = target(proposal.get("tool"), proposal.get("args") or {})
    if value:
        last = fold["last_target"].get((kind, value))
        if last is not None:
            if kind == "cmd":
                changed = any(i > last for i, _k, _v in fold["writes"])
            else:
                changed = any(i > last and k == kind and v == value
                              for i, k, v in fold["writes"])
            return "verification" if changed else "reestablish"
    return "orientation" if is_orientation(proposal) else "fresh"


def parse_one(text: str) -> dict | None:
    from evals.compaction.continuation import proposals_from

    got = proposals_from(text)
    if got:
        return got[0]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        item = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if isinstance(item, dict) and isinstance(item.get("tool"), str):
        return {"tool": item["tool"], "args": item.get("args") if isinstance(item.get("args"), dict) else {}}
    return None


def run_fold(fold: dict, arm: str, bank: dict, steps: int, max_tokens: int) -> dict:
    text = make_handoff(fold, arm)
    transcript = ""
    trace = []
    for _ in range(steps):
        raw = _call(LOOP_PROMPT.format(framing=FRAMING, handoff=text, transcript=transcript),
                    max_tokens=max_tokens)
        proposal = parse_one(raw)
        if proposal is None:
            trace.append({"class": "unparsed", "tool": None, "args": {}, "raw": raw[:200]})
            break
        cls = classify_loop(proposal, fold)
        result = replay(bank, proposal, fold)
        trace.append({"class": cls, "tool": proposal["tool"], "args": proposal["args"],
                      "result_chars": len(result)})
        transcript += (f"\n<CALL>{json.dumps({'tool': proposal['tool'], 'args': proposal['args']})}"
                       f"</CALL>\n<RESULT>{result[:1500]}</RESULT>\n")
        if cls == "fresh":
            break
    first_task = next((i for i, t in enumerate(trace) if t["class"] == "fresh"), None)
    return {"arm": arm, "session": fold["session"], "handoff_chars": len(text),
            "trace": trace, "orientation_before_task": first_task, "task_found": first_task is not None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--arms", default="summary,record,record_plan,record_plan_state,"
                                    "record_plan_state_directive")
    ap.add_argument("--out", default="evals/compaction/results/continuation-loop.json")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    folds = load_folds(args.db, args.folds)
    results = []
    for fold in folds:
        bank = tool_bank(fold)
        for arm in arms:
            results.append(run_fold(fold, arm, bank, args.steps, args.max_tokens))
            r = results[-1]
            print(f"  {r['session'][:16]} {arm:28s} "
                  + " ".join(t["class"][:4] for t in r["trace"]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"usage": EVAL_USAGE, "results": results}, indent=1))
    print(f"\ncalls {EVAL_USAGE['calls']}  prompt {EVAL_USAGE['prompt_tokens']:,} "
          f"completion {EVAL_USAGE['completion_tokens']:,} -> {out}")
    print("\narm                          orientation-before-task  reestablish  orientation  fresh")
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm]
        found = [r["orientation_before_task"] for r in rows if r["task_found"]]
        flat = [t["class"] for r in rows for t in r["trace"]]
        n = max(1, len(flat))
        med = statistics.median(found) if found else None
        print(f"{arm:28s} {med!s:>10s} / {len(found)} of {len(rows)}   "
              f"{flat.count('reestablish')/n:9.1%}  {flat.count('orientation')/n:9.1%}  "
              f"{flat.count('fresh')/n:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
