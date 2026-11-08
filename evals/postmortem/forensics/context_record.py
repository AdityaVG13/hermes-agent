"""Context Record: the one artifact a fold writes, an epoch seeds, recovery dereferences.

Pure and deterministic over message rows: no model call, no runtime imports, no I/O, so
the same transcript always renders the same bytes. `optchat/context_engine.py` calls it as
the fold body. A byte-identical copy lives at
`optchat/record.py` (plugin) and `evals/postmortem/forensics/context_record.py`
(Hermes checkout) — the two files must stay identical; port changes to both.

    rec = build_record(rows, session_id="local-…", first_row=1, last_row=880)
    text = render_record(rec)

Rows are dicts with ``role``, ``content``, ``tool_calls`` (list or JSON text),
``tool_call_id`` and ``_compressed_summary``. Everything the record states is either
verbatim (the ask, the todos) or mechanically derived (tool name, args, outcome, hash).
"""
from __future__ import annotations

import json
import re
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
DEFAULT_PLAN_ITEMS = 8
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


def bare_ref(value: str) -> str:
    """The path a file handle names, without its span, hash, or edit marker."""
    path = (value or "").split("#L", 1)[0].split(" (hash ", 1)[0]
    return path.removesuffix(" (edited)")


def _carried_ref(line: str) -> str:
    """`- path: read lines 3-9` -> `path#L3-9`; `- path: edited (1x)` -> `path (edited)`."""
    text = (line or "").strip()
    if not text.startswith("- "):
        return ""  # a trailer like "(3 older files not listed)" is not a handle
    text = text.removeprefix("- ")
    path, _, detail = text.partition(": ")
    if "read lines " in detail:
        span = detail.split("read lines ", 1)[1].split(";", 1)[0].strip()
        return f"{path}#L{span}"
    if "edited" in detail:
        return f"{path} (edited)"
    return path


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def is_notice(text: str) -> bool:
    """A bracketed all-caps preamble is a host notice (async batch, system note), not an ask."""
    head = text.lstrip() if isinstance(text, str) else ""
    if not head.startswith("["):
        return False
    end = head.find("]")
    label = head[1:end] if end > 0 else head[1:60]
    letters = [c for c in label if c.isalpha()]
    return bool(letters) and sum(1 for c in letters if c.isupper()) / len(letters) > 0.7


def is_injected(text: str) -> bool:
    """A host-injected row is not a user ask: recaps, carried context, notices."""
    head = text.lstrip() if isinstance(text, str) else ""
    return bool(head) and (
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
        body = body.split("\n", 1)[1] if "\n" in body else body.lstrip("`")
    return body.strip() or text


def last_goal(rows: list[dict[str, Any]], *, goal_chars: int = DEFAULT_GOAL_CHARS) -> str:
    for row in reversed(rows):
        if row.get("role") != "user" or row.get("_compressed_summary"):
            continue
        text = row.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        if not is_injected(text):
            return _clip(_strip_attachment(text), goal_chars)
    return "(none)"


def ask_pair(rows: list[dict[str, Any]], *, goal_chars: int = DEFAULT_GOAL_CHARS) -> tuple[str, str]:
    """(first real ask, last real ask) in the window — objective and latest instruction."""
    asks: list[str] = []
    for row in rows:
        if row.get("role") != "user" or row.get("_compressed_summary"):
            continue
        text = row.get("content")
        if not isinstance(text, str) or not text.strip():
            continue
        if not is_injected(text):
            asks.append(_clip(_strip_attachment(text), goal_chars))
    if not asks:
        return "(none)", ""
    return asks[0], asks[-1]


_REF_SUFFIX = re.compile(r":[0-9]+(?:-[0-9]+)?$")


def _basename(path: str) -> str:
    """The file's own name: strip a line suffix ("a.py:12") and any "#L" span, but never a
    Windows drive letter ("C:\\repo\\a.py" still yields "a.py")."""
    path = _REF_SUFFIX.sub("", path.split("#", 1)[0])
    return path.replace("\\", "/").rsplit("/", 1)[-1] or path


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
            raw_text = row.get("content")
            batch = {"text": raw_text.strip() if isinstance(raw_text, str) else "",
                     "calls": [], "results": {}}
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


def _int(value: Any, default: int = 0) -> int:
    """Model-supplied arguments are untrusted: a non-numeric offset must not break a fold."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _handle(call: dict[str, Any], results: dict[str, dict[str, Any]],
            cmd_chars: int = 80) -> tuple[str, str] | None:
    name = _name(call)
    args = _args(call)
    res = results.get(str(call.get("id") or ""), {})
    parsed = res.get("json") or {}
    if name in ("read_file", "patch", "write_file"):
        path = str(args.get("path") or "")
        if not path:
            return None
        if name != "read_file":
            return ("f", f"{path} (edited)")
        start = _int(args.get("offset") or 1, 1)
        limit = _int(args.get("limit") or 0, 0)
        ref = path
        if limit:
            ref = f"{path}#L{start}-{start + limit - 1}"
        elif start > 1:
            ref = f"{path}#L{start}-"
        digest = parsed.get("content_hash")
        return ("f", f"{ref} (hash {digest})" if digest else ref)
    if name == "search_files":
        pattern = " ".join(str(args.get("pattern") or "").split())
        return ("q", pattern[:80]) if pattern else None
    if name == "skill_view":
        ref = str(parsed.get("ref") or args.get("name") or "")
        if not ref:
            return None
        digest = parsed.get("digest")
        return ("s", f"{ref} (digest {digest})" if digest else ref)
    if name in ("terminal", "execute_code"):
        spill = parsed.get("stdout_spill_path")
        if spill:
            return ("a", f"artifact: {spill}")
        command = " ".join(str(args.get("command") or args.get("code") or "").split())
        return ("c", command[:cmd_chars]) if command else None
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


def _plan(rows: list[dict[str, Any]]) -> tuple[list[dict[str, str]], int, dict[str, str] | None]:
    """(shown open items, hidden count, the item NEXT should name).

    NEXT comes from the full open list, not the shown slice: an in-progress item beyond
    the cap must still direct the next step.
    """
    for row in reversed(rows):
        if row.get("role") != "tool" or row.get("tool_name") != "todo_list":
            continue
        items = (_parse(row.get("content") or "") or {}).get("todos")
        if isinstance(items, list):
            open_items = [
                t for t in items
                if isinstance(t, dict) and str(t.get("status") or "") in ("pending", "in_progress")
            ]
            shown = open_items[:DEFAULT_PLAN_ITEMS]
            active = next(
                (t for t in open_items if str(t.get("status") or "") == "in_progress"),
                open_items[0] if open_items else None,
            )
            nxt = (
                {"id": str(active.get("id") or "?"),
                 "content": _clip(str(active.get("content") or ""), 120)}
                if active is not None else None
            )
            return [
                {
                    "id": str(t.get("id") or "?"),
                    "status": str(t.get("status") or "pending"),
                    "content": _clip(str(t.get("content") or ""), 120),
                }
                for t in shown
            ], len(open_items) - len(shown), nxt
    return [], 0, None


def build_record(
    rows: list[dict[str, Any]],
    *,
    session_id: str = "",
    first_row: int | None = None,
    last_row: int | None = None,
    max_batches: int = DEFAULT_MAX_BATCHES,
    goal_chars: int = DEFAULT_GOAL_CHARS,
    max_tokens: int = DEFAULT_TOKENS,
    carried_lines: list[str] | None = None,
    carried_goal: str = "",
    cmd_chars: int = 80,
) -> dict[str, Any]:
    """Build the record for the rows being folded. Deterministic for identical input."""
    batches_ = batches(rows)
    plan, plan_more, plan_next = _plan(rows)
    first_ask, last_ask = ask_pair(rows, goal_chars=goal_chars)
    carried = ""
    if carried_goal and carried_goal.strip() != "(none)":
        carried = _clip(_strip_attachment(carried_goal), goal_chars)
    # GOAL is the standing objective: the carried one survives a pivot, else the window's first
    # ask. The newest instruction is listed separately only when it is a different thing.
    goal = carried or first_ask
    latest_ask = last_ask if last_ask and last_ask[:80] != goal[:80] else ""
    now = batches_[-1] if batches_ else None
    done = batches_[:-1]

    dropped = 0
    records = list(done)
    while len(records) > max_batches:
        records.pop(0)
        dropped += 1

    seen: dict[str, dict[str, int]] = {"f": {}, "s": {}, "q": {}, "c": {}, "a": {}}
    sequence = 0
    for batch in batches_:
        for call in batch.get("calls") or []:
            h = _handle(call, batch.get("results") or {}, cmd_chars)
            if h:
                kind, value = h
                seen.setdefault(kind, {})[value] = sequence
                sequence += 1
    caps = {"f": 20, "s": 6, "q": 6, "c": 8, "a": 6}
    handles = {
        kind: [value for value, _ in sorted(hits.items(), key=lambda kv: kv[1])][-caps[kind]:]
        for kind, hits in seen.items()
    }
    if carried_lines:
        existing = {bare_ref(value) for value in handles.get("f", [])}
        carried_refs = []
        for line in carried_lines:
            ref = _carried_ref(line)
            bare = bare_ref(ref)
            if not bare or bare in existing:
                continue
            existing.add(bare)
            carried_refs.append(ref)
        handles["f"] = (carried_refs + handles.get("f", []))[-caps["f"]:]

    results = sum(len(b.get("results") or {}) for b in batches_)
    chars = sum(len(str(b.get("text") or "")) for b in batches_)
    chars += sum(len(str(r.get("text") or "")) for b in batches_ for r in (b.get("results") or {}).values())

    # Never derive NEXT from the last action: that instructs the model to redo it.
    next_step = (f"{plan_next['id']}. {plan_next['content']}" if plan_next
                 else "no plan recorded — infer from GOAL")

    # The recovery line must be a call the model can actually make: session_search takes
    # ``session_id`` and ``around_message_id`` (or ``query``), never ``session``/``around``.
    kwargs = [f"session_id={session_id}"] if session_id else []
    if first_row is not None:
        kwargs.append(f"around_message_id={first_row}")
        recovery = f"session_search({', '.join(kwargs)})"
        if last_row is not None:
            recovery += f"  # rows {first_row}-{last_row}"
    else:
        if goal and goal != "(none)":
            kwargs.append(f"query={_clip(goal, 60).replace(chr(34), chr(39))!r}")
        recovery = f"session_search({', '.join(kwargs)})" if kwargs else "session_search()"

    rec: dict[str, Any] = {
        "version": 1,
        "goal": goal,
        "latest_ask": latest_ask,
        "plan": plan,
        "plan_more": plan_more,
        "done": records,
        "done_dropped": dropped,
        "now": now,
        "next": next_step or "infer from GOAL",
        "handles": handles,
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
    if rec.get("latest_ask"):
        lines.append(f"LATEST ASK: {rec['latest_ask']}")
    plan = rec.get("plan") or []
    if plan:
        lines.append("PLAN:")
        lines.extend(f"  - [{item['status']}] {item['id']}. {item['content']}" for item in plan)
        if rec.get("plan_more"):
            lines.append(f"  … {rec['plan_more']} more open items")
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
        lines.append(f"LAST FOLDED (done): {_clip(label, 150)} — {_outcome(now.get('results') or {})}")
    lines.append(f"NEXT: {rec['next']}")
    handles = rec.get("handles") or {}
    entries = (
        [("f", v) for v in handles.get("f") or []]
        + [("s", v) for v in handles.get("s") or []]
        + [("q", v) for v in handles.get("q") or []]
        + [("c", v) for v in handles.get("c") or []]
        + [("a", v) for v in handles.get("a") or []]
    )
    lines.append("WORKING SET:")
    if entries:
        lines.extend(f"  {kind}  {value}" for kind, value in entries)
    else:
        lines.append("  (none)")
    loss = rec.get("loss") or {}
    lines.append(
        f"LOSS: {loss.get('rows', 0)} rows, {loss.get('results', 0)} tool results, "
        f"{loss.get('chars', 0)} chars elided — recover: {loss.get('recovery', '')}"
    )
    lines.append(RECORD_FOOTER)
    return "\n".join(lines)
