"""Gap-cause decomposition: who owns the TTL-expiry loss.

    python -m evals.cache_lab.gap_causes --db ~/.hermes/state.db --days 14 --provider openai-codex

For every call gap that exceeds the fitted TTL, attribute the gap to HUMAN
pace (a user message dominated the idle window) or MACHINE pace (tool/child
execution dominated). The answer decides which levers can reach the pool:
scheduling only touches machine-paced gaps; human-paced ones are reachable
solely by a pre-expiry keepalive. On this fleet (14d, codex): 100% human —
200 expiring gaps worth ~2.9G tokens, zero machine-paced.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

# Fitted cache TTLs (cache_lab.calibrate, 2026-11-05); refresh when refitting.
FITTED_TTL = {
    "openai-codex": 600.0,
    "muse-code": 1800.0,
    "xai-oauth": 300.0,
    "zai": 300.0,
    "nous": 300.0,
}

CHARS_PER_TOKEN = 3.5


def dominant_rows(db: sqlite3.Connection, days: int, min_calls: int, provider: str):
    cutoff = time.time() - days * 86400.0
    rows = db.execute(
        "SELECT session_id, model, billing_provider, api_call_count FROM session_model_usage"
        " WHERE last_seen > ? AND model <> '' AND (task = '' OR task IS NULL)", (cutoff,)).fetchall()
    by_session: dict[str, list[tuple]] = defaultdict(list)
    for r in rows:
        by_session[r[0]].append(r)
    out = []
    for sid, rs in by_session.items():
        total = sum(r[3] for r in rs)
        if total < min_calls:
            continue
        dom = max(rs, key=lambda r: r[3])
        if dom[3] / total >= 0.8 and dom[2] == provider:
            out.append(sid)
    return out


def decompose(db: sqlite3.Connection, session_ids: list[str], ttl: float) -> dict:
    """Token-weighted cause attribution over TTL-expiring gaps. Loss weight for
    a gap = the token share the next call would have cached (the prefix at the
    previous call): exactly what expiry destroys."""
    loss = defaultdict(float)
    counts = defaultdict(int)
    for sid in session_ids:
        msgs = db.execute(
            "SELECT role, timestamp, length(coalesce(content,'')) + length(coalesce(tool_calls,'')) AS chars,"
            " _compressed_summary FROM messages WHERE session_id = ? ORDER BY id", (sid,)).fetchall()
        events: list[tuple[float, str, int]] = []
        for role, ts, chars, is_sum in msgs:
            if role == "assistant" and not is_sum:
                events.append((float(ts), "call", chars or 0))
            elif role == "tool":
                events.append((float(ts), "tool", chars or 0))
            elif role == "user":
                events.append((float(ts), "user", chars or 0))
        calls, stream_chars = [], 0
        for ts, kind, chars in events:
            if kind == "call":
                calls.append((ts, stream_chars))
                stream_chars += chars
            else:
                stream_chars += chars
        for i in range(1, len(calls)):
            ts0, chars_before_prev = calls[i - 1]
            ts1, _ = calls[i]
            gap = ts1 - ts0
            if gap <= ttl:
                continue
            loss_tokens = chars_before_prev / CHARS_PER_TOKEN
            between = [(t, k) for t, k, _ in events if ts0 < t < ts1]
            first_user = next((t for t, k in between if k == "user"), None)
            last_tool = max((t for t, k in between if k == "tool"), default=None)
            if first_user is not None and (last_tool is None or first_user < last_tool):
                human = first_user - ts0
            elif last_tool is not None:
                human = max(0.0, (first_user if first_user is not None else ts1) - last_tool)
            else:
                human = gap
            cause = "human" if human > gap * 0.5 else "machine"
            bucket = "5-15min" if gap < 900 else ("15-60min" if gap < 3600 else ">1h")
            loss[(cause, bucket)] += loss_tokens
            counts[(cause, bucket)] += 1
    return {
        "total_loss_tokens": sum(loss.values()),
        "by_cause": {
            cause: sum(v for (c, _), v in loss.items() if c == cause)
            for cause in ("human", "machine")},
        "detail": [
            {"cause": c, "bucket": b, "loss_tokens": round(v), "gaps": counts[(c, b)]}
            for (c, b), v in sorted(loss.items(), key=lambda kv: -kv[1])],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--db", default=str(Path.home() / ".hermes" / "state.db"))
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--min-calls", type=int, default=20)
    ap.add_argument("--provider", default="openai-codex")
    ap.add_argument("--ttl", type=float, default=None)
    ap.add_argument("--out", default="gap_causes.json")
    a = ap.parse_args(argv)

    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    ttl = a.ttl if a.ttl is not None else FITTED_TTL.get(a.provider, 600.0)
    sids = dominant_rows(db, a.days, a.min_calls, a.provider)
    report = {"provider": a.provider, "ttl_seconds": ttl, "sessions": len(sids),
              **decompose(db, sids, ttl)}
    Path(a.out).write_text(json.dumps(report, indent=2))
    total = report["total_loss_tokens"] or 1
    print(f"[gaps] {a.provider} n={len(sids)} ttl={ttl:.0f}s  pool={total/1e6:.0f}M tokens")
    for cause, v in report["by_cause"].items():
        print(f"[gaps]   {cause:7s}-paced: {v/1e6:8.0f}M ({100*v/total:.0f}%)")
    for d in report["detail"]:
        print(f"[gaps]     {d['cause']:7s} {d['bucket']:8s}: {d['loss_tokens']/1e6:8.0f}M ({d['gaps']} gaps)")
    print(f"[gaps] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
