"""Context Record: the one artifact a fold writes, an epoch seeds, recovery dereferences.

Prototype over persisted message rows. Pure and deterministic: no model call, no imports
from the runtime, no I/O. The engine will emit exactly this shape once it is approved.

    rec = build_record(rows, session_id="local-…", first_row=1, last_row=880)
    text = render_record(rec)

Rows are dicts with ``role``, ``content``, ``tool_calls`` (list or JSON text),
``tool_call_id`` and ``_compressed_summary``. Everything the record states is either
verbatim (the ask, the todos) or mechanically derived (tool name, args, outcome, hash).
"""
from __future__ import annotations

import json
from typing import Any

RECORD_HEADER = "== CONTEXT RECORD v1 =="
RECORD_FOOTER = "== END RECORD =="
CLIFF_HEADER = "The following is a summary of your previous actions"
INJECTED_PREFIXES = (
    "[CONTEXT",
    "[Earlier conversation",
    "[System",
    "[IMPORTANT",
    "[The user",
    "[OUT-OF-BAND",
    "[ASYNC",
    "[DELEG",
    "[BACKGROUND",
    "[TASK",
    "[NOTICE",
)
DEFAULT_MAX_BATCHES = 40
DEFAULT_GOAL_CHARS = 400
DEFAULT_TOKENS = 2000


def _tool_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
    tc = row.get("tool_calls")
    if not tc:
        return []
    if isinstance(tc, str):
        try:
            tc = json.loads(tc)
        except Exception:
            return []
    return [c for c in tc if isinstance(c, dict)] if isinstance(tc, list) else []


def _args(call: dict[str, Any]) -> dict[str, Any]:
    fn = call.get("function") or {}
    a = fn.get("arguments")
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            a = {}
    return a if isinstance(a, dict) else {}


def _name(call: dict[str, Any]) -> str:
    return str((call.get("function") or {}).get("name") or "")


def _parse(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text.startswith("{"):
        return {}
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def is_notice(text: str) -> bool:
    """A bracketed all-caps preamble is a host notice (async batch, system note), not an ask."""
    head = (text or "").lstrip()
    if not head.startswith("["):
        return False
    end = head.find("]")
    label = head[1:end] if end > 0 else head[1:60]
    letters = [c for c in label if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7


def is_injected(text: str) -> bool:
    """A host-injected row is not a user ask: recaps, carried context, notices."""
    head = (text or "").lstrip()
    return (
        head.startswith(INJECTED_PREFIXES)
        or CLIFF_HEADER in head[:200]
        or is_notice(head)
    )


def _strip_attachment(text: str) -> str:
    """Drop the pasted-file preamble so GOAL starts at the ask, not at the attachment."""
    lines = text.splitlines()
    index = 0
    while index < len(lines) and (
        not lines[index].strip() or lines[index].lstrip().startswith(("@file:", "---", "📄"))
    ):
        index += 1
    body = "\n".join(lines[index:]).strip()
    if body.startswith("```"):
        body = body[3:].lstrip("`").lstrip()
    return body or text


def last_goal(rows: list[dict[str, Any]], *, goal_chars: int = DEFAULT_GOAL_CHARS) -> str:
    for row in reversed(rows):
        if row.get("role") != "user" or row.get("_compressed_summary"):
            continue
        text = (row.get("content") or "").strip()
        if text and not is_injected(text):
            return _clip(_strip_attachment(text), goal_chars)
    return "(none)"


def _basename(path: str) -> str:
    path = path.split("#", 1)[0].split(":", 1)[0]
    return path.rsplit("/", 1)[-1] or path


def _target(call: dict[str, Any]) -> str:
    name, args = _name(call), _args(call)
    if name in ("read_file", "patch", "write_file"):
        path = str(args.get("path") or "")
        return _basename(path) if path else name
    if name == "skill_view":
        return str(args.get("name") or name)
    if name == "search_files":
        pat = str(args.get("pattern") or "")
        return f"search {_clip(pat, 40)}" if pat else name
    if name in ("terminal", "execute_code"):
        cmd = str(args.get("command") or args.get("code") or "")
        return _clip(cmd, 60) if cmd else name
    for value in args.values():
        if isinstance(value, str) and value.strip():
            return f"{name} {_clip(value, 40)}"
    return name


def batches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group tool calls by the assistant row that issued them, with their results attached."""
    out: list[dict[str, Any]] = []
    by_call: dict[str, int] = {}
    for row in rows:
        role = row.get("role")
        if role == "assistant":
            calls = _tool_calls(row)
            if not calls:
                continue
            batch = {"text": (row.get("content") or "").strip(), "calls": [], "results": {}}
            out.append(batch)
            for call in calls:
                cid = str(call.get("id") or "")
                if cid:
                    by_call[cid] = len(out) - 1
                batch["calls"].append(call)
        elif role == "tool":
            cid = str(row.get("tool_call_id") or "")
            if cid in by_call:
                raw = row.get("content") or ""
                out[by_call[cid]]["results"][cid] = {"json": _parse(raw), "text": raw}
    return out


def _outcome(results: dict[str, dict[str, Any]]) -> str:
    for res in results.values():
        parsed = res.get("json") or {}
        if parsed.get("success") is False:
            return "err"
        head = (res.get("text") or "").strip().lower()
        if head.startswith(("error", "traceback", "permission denied")):
            return "err"
    return "ok"


def _handle(call: dict[str, Any], results: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    name = _name(call)
    args = _args(call)
    res = results.get(str(call.get("id") or ""), {})
    parsed = res.get("json") or {}
    if name == "read_file":
        path = str(args.get("path") or "")
        if not path:
            return None
        start = int(args.get("offset") or 1)
        limit = int(args.get("limit") or 0)
        ref = path
        if limit:
            ref = f"{path}#L{start}-{start + limit - 1}"
        elif start > 1:
            ref = f"{path}#L{start}-"
        digest = parsed.get("content_hash")
        return ("f", f"{ref} (hash {digest})" if digest else ref)
    if name == "skill_view":
        ref = str(parsed.get("ref") or args.get("name") or "")
        if not ref:
            return None
        digest = parsed.get("digest")
        return ("s", f"{ref} (digest {digest})" if digest else ref)
    if name in ("terminal", "execute_code", "tool_call"):
        spill = parsed.get("stdout_spill_path")
        if spill:
            return ("a", f"artifact: {spill}")
    return None


def _looks_like_prose(text: str) -> bool:
    """An assistant preamble is an intent only when it reads like one, not like code."""
    t = (text or "").strip()
    if not t or len(t) > 140 or "\n" in t:
        return False
    if any(c in t for c in "{}`$<>"):
        return False
    if t.startswith(("import ", "from ", "def ", "class ", "for ", "if ", "with ", "#!/")):
        return False
    grounded = sum(c.isalpha() or c.isspace() for c in t)
    return grounded / len(t) > 0.7


def _targets(calls: list[dict[str, Any]], *, limit: int = 90) -> str:
    labels: list[str] = []
    counts: list[int] = []
    for call in calls:
        label = _target(call)
        if labels and labels[-1] == label:
            counts[-1] += 1
        else:
            labels.append(label)
            counts.append(1)
    rendered = [f"{label} ×{n}" if n > 1 else label for label, n in zip(labels, counts)]
    return _clip(", ".join(rendered), limit)


def _done_line(index: int, batch: dict[str, Any], *, intent_chars: int = 110) -> str:
    intent = _clip(batch.get("text") or "", intent_chars) if _looks_like_prose(batch.get("text") or "") else ""
    targets = _targets(batch.get("calls") or [])
    label = f"{intent} — {targets}" if intent and targets else (intent or targets or "(no text)")
    return f"  {index}. {_clip(label, 150)} — {_outcome(batch.get('results') or {})}"


def _plan(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    for row in reversed(rows):
        if row.get("role") != "tool" or row.get("tool_name") != "todo_list":
            continue
        items = (_parse(row.get("content") or "") or {}).get("todos")
        if isinstance(items, list):
            open_items = [
                t for t in items
                if isinstance(t, dict) and str(t.get("status") or "") in ("pending", "in_progress")
            ]
            return [
                {
                    "id": str(t.get("id") or "?"),
                    "status": str(t.get("status") or "pending"),
                    "content": _clip(str(t.get("content") or ""), 120),
                }
                for t in open_items
            ]
    return []


def build_record(
    rows: list[dict[str, Any]],
    *,
    session_id: str = "",
    first_row: int | None = None,
    last_row: int | None = None,
    max_batches: int = DEFAULT_MAX_BATCHES,
    goal_chars: int = DEFAULT_GOAL_CHARS,
    max_tokens: int = DEFAULT_TOKENS,
) -> dict[str, Any]:
    """Build the record for the rows being folded. Deterministic for identical input."""
    batches_ = batches(rows)
    plan = _plan(rows)
    goal = last_goal(rows, goal_chars=goal_chars)
    now = batches_[-1] if batches_ else None
    done = batches_[:-1]

    dropped = 0
    records = list(done)
    while len(records) > max_batches:
        records.pop(0)
        dropped += 1

    handles: dict[str, str] = {}
    for batch in batches_:
        for call in batch.get("calls") or []:
            h = _handle(call, batch.get("results") or {})
            if h:
                kind, value = h
                handles[f"{kind}|{value}"] = value
    f_handles = [v for k, v in handles.items() if k.startswith("f|")][-8:]
    s_handles = [v for k, v in handles.items() if k.startswith("s|")][-4:]
    a_handles = [v for k, v in handles.items() if k.startswith("a|")][-4:]

    results = sum(len(b.get("results") or {}) for b in batches_)
    chars = sum(len(str(b.get("text") or "")) for b in batches_)
    chars += sum(len(str(r.get("text") or "")) for b in batches_ for r in (b.get("results") or {}).values())

    next_step = ""
    for item in plan:
        if item["status"] == "in_progress":
            next_step = f"{item['id']}. {item['content']}"
            break
    if not next_step and plan:
        next_step = f"{plan[0]['id']}. {plan[0]['content']}"
    if not next_step:
        # Never derive NEXT from the last action: that instructs the model to redo it.
        next_step = "no plan recorded — infer from GOAL"

    first = first_row if first_row is not None else 0
    last = last_row if last_row is not None else len(rows)
    recovery = (
        f"session_search(session={session_id or '?'}, around={first})"
        f"  # rows {first}-{last}"
    )

    rec: dict[str, Any] = {
        "version": 1,
        "goal": goal,
        "plan": plan,
        "done": records,
        "done_dropped": dropped,
        "now": now,
        "next": next_step or "infer from GOAL",
        "handles": {"f": f_handles, "s": s_handles, "a": a_handles},
        "loss": {"rows": len(rows), "results": results, "chars": chars, "recovery": recovery},
        "batches": len(batches_),
    }
    rec["text"] = render_record(rec)
    # Size guard: drop the oldest completed batches until the record fits.
    while len(rec["text"]) // 4 > max_tokens and rec["done"]:
        rec["done"].pop(0)
        rec["done_dropped"] += 1
        rec["text"] = render_record(rec)
    return rec


def render_record(rec: dict[str, Any]) -> str:
    lines = [
        RECORD_HEADER,
        "Completed work below. Do not repeat it. Resume from NEXT.",
        "",
        f"GOAL: {rec['goal']}",
    ]
    plan = rec.get("plan") or []
    if plan:
        lines.append("PLAN:")
        lines.extend(f"  - [{item['status']}] {item['id']}. {item['content']}" for item in plan)
    else:
        lines.append("PLAN: none — state ambiguous")
    lines.append("DONE (completed; do not repeat):")
    if rec.get("done_dropped"):
        lines.append(f"  … {rec['done_dropped']} earlier batches omitted (see LOSS)")
    start = rec["done_dropped"] + 1
    for offset, batch in enumerate(rec.get("done") or []):
        lines.append(_done_line(start + offset, batch))
    now = rec.get("now")
    if now is not None:
        last = _clip(now.get("text") or "", 110) if _looks_like_prose(now.get("text") or "") else ""
        action = _targets(now.get("calls") or [])
        label = f"{last} — {action}" if last and action else (last or action)
        lines.append(f"LAST (done): {_clip(label, 150)} — {_outcome(now.get('results') or {})}")
    lines.append(f"NEXT: {rec['next']}")
    handles = rec.get("handles") or {}
    entries = (
        [("f", v) for v in handles.get("f") or []]
        + [("s", v) for v in handles.get("s") or []]
        + [("a", v) for v in handles.get("a") or []]
    )
    if entries:
        lines.append("WORKING SET:")
        lines.extend(f"  {kind}  {value}" for kind, value in entries)
    loss = rec.get("loss") or {}
    lines.append(
        f"LOSS: {loss.get('rows', 0)} rows, {loss.get('results', 0)} tool results, "
        f"{loss.get('chars', 0)} chars elided — recover: {loss.get('recovery', '')}"
    )
    lines.append(RECORD_FOOTER)
    return "\n".join(lines)
