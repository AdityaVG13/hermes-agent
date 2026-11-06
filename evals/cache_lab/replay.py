"""Offline prefix-cache replay over real session transcripts.

The postmortem's ``forensics/tokens.py`` reconstructs prompt *sizes*; this
module reconstructs *prefixes*. For each assistant call it rebuilds the
message-uid stream that was in context at that moment (compaction summaries
replace the ranges they absorbed, in place), computes the longest-common-prefix
token share against the previous call — exactly what an implicit
longest-prefix provider serves back as a cache hit — and exposes policy terms
(``ttl_seconds``, ``reliability``) for ``calibrate`` to grid-fit against the
observed ``session_model_usage`` cache rates.

Reader contract: takes an open sqlite connection (read-only is fine) with the
production ``sessions`` / ``system_prompts`` / ``messages`` schema. Token sizes
are estimated (chars/3.5, the same estimator the postmortem used) — cache
RATIOS are dimensionless, so a global estimator bias largely cancels; only the
per-provider calibration gate can say whether that is good enough.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from typing import Any

CHARS_PER_TOKEN = 3.5
TOOL_SCHEMA_TOKENS = 12_000


def _TOK(text: str | None) -> float:
    return (len(text) + 0.0) / CHARS_PER_TOKEN


@dataclass(frozen=True)
class ReplayedCall:
    idx: int
    ts: float
    gap_seconds: float
    prompt_tokens: float          # system + tool schemas + in-context stream
    cached_prefix_tokens: float   # LCP token share vs the previous call


def _row_tokens(row: dict[str, Any]) -> float:
    return _TOK(row["content"] or "") + _TOK(row["tool_calls"] or "")


def _uid(row_id: int, uid: str | None) -> str:
    return uid or f"id:{row_id}"


def _absorbed(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(u) for u in parsed]
    except (ValueError, TypeError):
        return [part for part in raw.split(",") if part]
    return []


def read_session_calls(db: sqlite3.Connection, session_id: str,
                       *, tool_schema_tokens: float = TOOL_SCHEMA_TOKENS) -> list[ReplayedCall]:
    """Reconstruct per-call prompt size and cached-prefix share for one session.

    Streams are uid lists; a summary row replaces the uid range it absorbed at
    the position of the first absorbed row, so the longest common prefix between
    consecutive calls is exactly the byte-prefix an implicit cache would serve.
    The system prompt + tool schemas are byte-identical across a session's
    calls, so they count as cached on every call after the first.
    """
    sys_row = db.execute(
        "SELECT p.prompt FROM sessions s JOIN system_prompts p ON p.hash = s.system_prompt_hash"
        " WHERE s.id = ?", (session_id,)).fetchone()
    system_tokens = (_TOK(sys_row[0]) if sys_row and sys_row[0] else 0.0) + tool_schema_tokens

    rows = db.execute(
        "SELECT id, role, content, tool_calls, timestamp, message_uid,"
        " absorbed_message_uids, _compressed_summary"
        " FROM messages WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()

    uids: list[str] = []
    tok: dict[str, float] = {}
    calls: list[ReplayedCall] = []
    prev_uids: list[str] | None = None
    prev_ts: float | None = None

    for row_id, role, content, tool_calls, ts, uid, absorbed_raw, is_summary in rows:
        row_uid = _uid(row_id, uid)
        tokens = _row_tokens({"content": content, "tool_calls": tool_calls})

        # A compression summary is not a main-lane model call (it runs on the
        # compaction lane); it only rewrites the stream.
        if role == "assistant" and not is_summary:
            # The call happens with the stream as it stands BEFORE this row
            # (this row is the response). Prompt = system + tools + stream.
            prompt = system_tokens + sum(tok.values())
            if prev_uids is None:
                lcp_tokens = 0.0
            else:
                shared = 0.0
                for a, b in zip(prev_uids, uids):
                    if a != b:
                        break
                    shared += tok.get(a, 0.0)
                # system + tool schemas precede the stream and never change
                # inside a session, so they are always part of the served prefix
                lcp_tokens = system_tokens + shared
            calls.append(ReplayedCall(
                idx=len(calls), ts=float(ts),
                gap_seconds=(float(ts) - prev_ts) if prev_ts is not None else 0.0,
                prompt_tokens=prompt, cached_prefix_tokens=lcp_tokens))
            prev_uids = list(uids)
            prev_ts = float(ts)

        if is_summary:
            gone = _absorbed(absorbed_raw)
            insert_at = len(uids)
            for g in gone:
                if g in uids:
                    insert_at = min(insert_at, uids.index(g))
                    uids.remove(g)
                    tok.pop(g, None)
            uids.insert(insert_at, row_uid)
            tok[row_uid] = tokens
        elif role != "assistant" or tokens:
            # Assistant responses join the stream for the NEXT call too.
            uids.append(row_uid)
            tok[row_uid] = tokens

    return calls


def simulate(calls: list[ReplayedCall], *, ttl_seconds: float | None = None,
             reliability: float = 1.0) -> list[ReplayedCall]:
    """Apply policy terms: TTL expiry zeroes a call's cache share; reliability
    scales it (provider-side accounting / partial retention)."""
    out: list[ReplayedCall] = []
    for c in calls:
        cached = c.cached_prefix_tokens * reliability
        if ttl_seconds is not None and c.gap_seconds > ttl_seconds:
            cached = 0.0
        out.append(replace(c, cached_prefix_tokens=cached))
    return out


def cache_ratio(calls: list[ReplayedCall]) -> tuple[float, float]:
    """(cached token sum, total prompt token sum) for simulated calls."""
    return sum(c.cached_prefix_tokens for c in calls), sum(c.prompt_tokens for c in calls)
