"""Faithful continuation probe: the runtime's prompt tier, the session's model and tools.

The earlier probes (``continuation.py``, ``continuation_loop.py``) used the auxiliary
compression model with a one-line framing and no tool schemas. They opened with
orientation while the real first post-fold call is a task write in two thirds of folds:
a cold-start agent, not a folded one.

This module closes that gap:
  * the system prompt is the runtime's stable tier (identity, tool-use enforcement,
    execution guidance, completion and skills rules), gated by model family;
  * the model and provider are the session's own, from ``sessions.model`` /
    ``model_config``;
  * tools are real schemas from ``model_tools.get_tool_definitions``, filtered to the
    tools the session actually used;
  * ``full_context`` is the control arm: the verbatim pre-fold window. Its first-call
    distribution is the fidelity gate — it must look like the real DB distribution
    (task write ~2/3), or the instrument is still cold.

Arms: full_context (control), summary, record_plan_situation_state.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agent.auxiliary_client import call_llm
from agent.prompt_builder import (
    DEFAULT_AGENT_IDENTITY, EXECUTION_GUIDANCE_MODELS, OPENAI_MODEL_EXECUTION_GUIDANCE,
    SKILLS_GUIDANCE, TASK_COMPLETION_GUIDANCE, TOOL_USE_ENFORCEMENT_GUIDANCE,
    TOOL_USE_ENFORCEMENT_MODELS,
)
from evals.compaction.continuation import load_folds
from evals.compaction.continuation_loop import (
    classify_action, classify_loop, is_orientation, make_handoff, parse_one,
)
from evals.compaction.runner import serialize_for_exam
from evals.postmortem.forensics.record_coverage import calls_of, target

CORE_TOOLS = ("read_file", "patch", "write_file", "terminal", "search_files", "skill_view",
              "todo_list", "delegate_task", "execute_code")
INSTRUCTION = ("You are resuming a session that was compacted. State exactly ONE next tool call. "
               "Prefer continuing the task over re-discovering state.")


def _matches(model: str, families) -> bool:
    return any(f in (model or "").lower() for f in families)


def runtime_prompt(model: str, cwd: str, branch: str) -> str:
    parts = [DEFAULT_AGENT_IDENTITY]
    if _matches(model, TOOL_USE_ENFORCEMENT_MODELS):
        parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
    if _matches(model, EXECUTION_GUIDANCE_MODELS):
        parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)
    parts += [TASK_COMPLETION_GUIDANCE, SKILLS_GUIDANCE]
    parts.append(f"# Workspace\ncwd: {cwd or '(unknown)'}\nbranch: {branch or '(unknown)'}")
    return "\n\n".join(parts)


def session_row(db: str, sid: str) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute("select model, model_config, cwd, git_branch from sessions where id=?",
                      (sid,)).fetchone()
    out = dict(row) if row else {}
    try:
        out["cfg"] = json.loads(out.get("model_config") or "{}")
    except (ValueError, TypeError):
        out["cfg"] = {}
    return out


def tool_schemas(fold: dict) -> list[dict]:
    from model_tools import get_tool_definitions

    used = {c[1] for c in calls_of(fold["rows"]) if c[0] < fold["anchor"]}
    wanted = used | set(CORE_TOOLS)
    return [d for d in get_tool_definitions(quiet_mode=True)
            if d.get("function", {}).get("name") in wanted]


def native_call(resp) -> dict | None:
    try:
        message = resp.choices[0].message
    except (AttributeError, IndexError):
        return None
    for call in getattr(message, "tool_calls", None) or []:
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", None)
        if not name:
            continue
        raw = getattr(fn, "arguments", "{}") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except json.JSONDecodeError:
            args = {}
        return {"tool": name, "args": args if isinstance(args, dict) else {}}
    return None


def prompt_for(fold: dict, arm: str) -> str:
    if arm == "full_context":
        body = serialize_for_exam(fold["window"], char_cap=30_000)
    else:
        body = make_handoff(fold, arm)
    return f"{INSTRUCTION}\n\n<CONTEXT>\n{body}\n</CONTEXT>"


def probe(fold: dict, arm: str, session: dict, tools: list[dict], max_tokens: int) -> dict:
    cfg = session.get("cfg") or {}
    model = session.get("model") or cfg.get("model") or ""
    system = runtime_prompt(model, session.get("cwd") or "", session.get("git_branch") or "")
    try:
        resp = call_llm(
            provider=cfg.get("provider"), model=model, base_url=cfg.get("base_url"),
            api_mode=cfg.get("api_mode"), reasoning_config=cfg.get("reasoning_config"),
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt_for(fold, arm)}],
            tools=tools, max_tokens=max_tokens, temperature=0,
        )
        proposal = native_call(resp) or parse_one(
            getattr(resp.choices[0].message, "content", "") or "")
    except Exception as exc:  # provider refusal is data, not a crash
        return {"arm": arm, "session": fold["session"], "error": f"{type(exc).__name__}: {exc}"[:300]}
    if proposal is None:
        return {"arm": arm, "session": fold["session"], "error": "no tool call in response"}
    cls = classify_action(proposal, fold)
    truth = [c for c in calls_of(fold["rows"]) if c[0] > fold["anchor"]][:1]
    truth_target = target(truth[0][1], truth[0][2]) if truth else None
    got = target(proposal["tool"], proposal["args"])
    return {"arm": arm, "session": fold["session"], "proposal": proposal, "class": cls,
            "truth": truth_target, "match": bool(truth_target and got == truth_target),
            "prompt_chars": len(system) + len(prompt_for(fold, arm))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--arms", default="full_context,summary,record_plan_situation_state")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--out", default="evals/compaction/results/continuation-faithful.json")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    folds = load_folds(args.db, args.folds)
    results = []
    for fold in folds:
        session = session_row(args.db, fold["session"])
        tools = tool_schemas(fold)
        for arm in arms:
            results.append(probe(fold, arm, session, tools, args.max_tokens))
            r = results[-1]
            mark = "MATCH" if r.get("match") else ""
            print(f"  {fold['session'][:16]} {arm:28s} {r.get('class', r.get('error', '?')):14s} "
                  f"{mark}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, indent=1))
    print("\narm                          task   reestablish  orientation  fresh  verify  match")
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm and "class" in r]
        n = max(1, len(rows))
        c = {k: sum(1 for r in rows if r["class"] == k) for k in
             ("task", "reestablish", "orientation", "fresh", "verification")}
        print(f"{arm:28s} {c['task']/n:5.0%}  {c['reestablish']/n:11.0%}  "
              f"{c['orientation']/n:11.0%}  {c['fresh']/n:5.0%}  {c['verification']/n:6.0%}  "
              f"{sum(1 for r in rows if r.get('match'))}/{len(rows)}")
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n{len(errors)} call(s) failed; first: {errors[0]['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
