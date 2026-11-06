"""Counterfactual policies on the calibrated replay lab (Phase 2).

    python -m evals.cache_lab.counterfactual --db ~/.hermes/state.db --days 14

Three policies, labeled by evidence class:
- READ-STUB (REPLAYED): later tool results whose full bytes were already sent
  earlier in the session collapse to a ~10-token stub (the if_hash handshake's
  effect), re-replayed with per-row token overrides.
- KEEPALIVE (REPLAYED): the cached share lost to TTL expiry at the fitted TTL —
  the upper bound an idle keepalive could preserve.
- LANE AFFINITY (ESTIMATED): aux-lane uncached tokens vs a tail-only estimate
  from usage rows plus replayed per-turn growth; not a transcript replay.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from evals.cache_lab.calibrate import _targets
from evals.cache_lab.replay import (
    _TOK, cache_ratio, read_session_calls, simulate)

STUB_TOOLS = {"read_file", "search_files"}
STUB_TOKENS = 10.0


def find_resends(db: sqlite3.Connection, session_id: str) -> list[dict]:
    """Later tool results that re-send bytes an earlier same-tool result already
    delivered — exactly what a portable hash handshake would collapse. Status
    polls and other small tools are excluded: identical small replies are
    legitimate, not waste."""
    rows = db.execute(
        "SELECT id, message_uid, tool_name, content FROM messages"
        " WHERE session_id = ? AND role = 'tool' AND tool_name IN (?, ?)"
        " ORDER BY id", (session_id, *sorted(STUB_TOOLS))).fetchall()
    seen: dict[str, str] = {}
    resends = []
    for row_id, uid, tool, content in rows:
        key = f"{tool}:{content}"
        row_uid = uid or f"id:{row_id}"
        if key in seen:
            resends.append({"uid": row_uid, "tokens": _TOK(content), "tool": tool})
        else:
            seen[key] = row_uid
    return resends


def ttl_loss(calls: list, *, ttl_seconds: float) -> float:
    """Cached tokens destroyed by TTL expiry: the cached sum with no TTL minus
    the cached sum at the fitted TTL (the keepalive lever's addressable pool)."""
    with_ttl = cache_ratio(simulate(calls, ttl_seconds=ttl_seconds))[0]
    no_ttl = cache_ratio(simulate(calls, ttl_seconds=None))[0]
    return no_ttl - with_ttl


def affinity_savings(aux_rows: list[dict], *, tail_per_call: float) -> dict:
    """ESTIMATED: if aux-lane calls rode the main session's cached prefix, their
    uncached cost drops to a per-call tail. aux_rows: [{task, calls,
    uncached_tokens}] from usage rows; tail_per_call from replayed growth."""
    uncached_now = sum(r["uncached_tokens"] for r in aux_rows)
    calls = sum(r["calls"] for r in aux_rows)
    uncached_after = calls * tail_per_call
    return {"uncached_now": uncached_now, "uncached_after": uncached_after,
            "saved": max(0.0, uncached_now - uncached_after), "calls": calls}


def _aux_rows(db: sqlite3.Connection, session_ids: set[str]) -> dict[str, list[dict]]:
    rows = db.execute(
        "SELECT session_id, task, api_call_count, input_tokens FROM session_model_usage"
        " WHERE task NOT IN ('', 'NULL') AND task IS NOT NULL").fetchall()
    out: dict[str, list[dict]] = defaultdict(list)
    for sid, task, calls, input_tokens in rows:
        if sid in session_ids:
            out[sid].append({"task": task, "calls": calls or 0,
                             "uncached_tokens": input_tokens or 0})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--db", default=str(Path.home() / ".hermes" / "state.db"))
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--min-calls", type=int, default=20)
    ap.add_argument("--out", default="cache_counterfactual.json")
    a = ap.parse_args(argv)

    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    targets = _targets(db, a.days, a.min_calls)

    # Fitted policies per (provider, model) — reuse calibrate's grid inline.
    from evals.cache_lab.calibrate import TTL_GRID, RELIABILITY_GRID
    replays: dict[str, list] = {}
    per_group: dict[tuple, list[dict]] = defaultdict(list)
    for t in targets:
        try:
            calls = read_session_calls(db, t["session_id"])
        except sqlite3.Error:
            continue
        if len(calls) >= 2:
            replays[t["session_id"]] = calls
            per_group[(t["provider"], t["model"])].append(t)

    fits: dict[tuple, dict] = {}
    for key, ts in per_group.items():
        best = None
        for ttl in TTL_GRID:
            for rel in RELIABILITY_GRID:
                errs = []
                for t in ts:
                    c, tot = cache_ratio(simulate(replays[t["session_id"]],
                                                  ttl_seconds=ttl, reliability=rel))
                    pred = 100.0 * c / tot if tot else 0.0
                    errs.append(abs(pred - t["observed_pct"]))
                mae = sum(errs) / len(errs)
                if best is None or mae < best[0]:
                    best = (mae, ttl, rel)
        fits[key] = {"ttl": best[1], "rel": best[2], "mae": best[0]} if best else {
            "ttl": None, "rel": 1.0, "mae": 0.0}

    report: dict = {"sessions": len(replays), "providers": {}}
    prov_acc: dict[str, dict] = defaultdict(lambda: {
        "calls": 0, "baseline_uncached": 0.0, "stub_uncached": 0.0,
        "stub_saved": 0.0, "ttl_loss": 0.0, "resend_tokens": 0.0,
        "resends": 0, "affinity_saved": 0.0, "sessions": 0})

    for (provider, model), ts in per_group.items():
        fit = fits[(provider, model)]
        acc = prov_acc[provider or "unknown"]
        for t in ts:
            sid = t["session_id"]
            calls = replays[sid]
            base_cached, base_total = cache_ratio(
                simulate(calls, ttl_seconds=fit["ttl"], reliability=fit["rel"]))
            resends = find_resends(db, sid)
            if resends:
                stub_calls = read_session_calls(
                    db, sid, token_overrides={r["uid"]: STUB_TOKENS for r in resends})
                stub_cached, stub_total = cache_ratio(
                    simulate(stub_calls, ttl_seconds=fit["ttl"], reliability=fit["rel"]))
            else:
                stub_cached, stub_total = base_cached, base_total
            acc["calls"] += len(calls)
            acc["sessions"] += 1
            acc["baseline_uncached"] += base_total - base_cached
            acc["stub_uncached"] += stub_total - stub_cached
            acc["stub_saved"] += (base_total - base_cached) - (stub_total - stub_cached)
            acc["resend_tokens"] += sum(r["tokens"] for r in resends)
            acc["resends"] += len(resends)
            acc["ttl_loss"] += ttl_loss(calls, ttl_seconds=fit["ttl"]) if fit["ttl"] else 0.0

    # LANE AFFINITY (ESTIMATED) — aux-lane uncached tokens vs tail-only cost,
    # tail_per_call from the replayed per-turn growth of the same sessions.
    aux = _aux_rows(db, set(replays))
    growth = []
    for sid, calls in replays.items():
        if len(calls) > 1:
            growth.append(calls[-1].prompt_tokens / len(calls) * 0.3)  # rough tail share
    tail_per_call = (sum(growth) / len(growth)) if growth else 4000.0
    total_affinity = {"uncached_now": 0.0, "uncached_after": 0.0, "saved": 0.0, "calls": 0}
    for sid, rows in aux.items():
        if sid in replays and rows:
            s = affinity_savings(rows, tail_per_call=tail_per_call)
            for k in total_affinity:
                total_affinity[k] += s[k]
    report["lane_affinity_estimated"] = {
        "tail_per_call_tokens": round(tail_per_call), **{
            k: round(v) for k, v in total_affinity.items()}}

    for provider, acc in prov_acc.items():
        report["providers"][provider] = {
            **{k: (round(v) if isinstance(v, float) else v) for k, v in acc.items()},
            "fitted": {f"{k[1]}": v for k, v in fits.items() if k[0] == provider}}
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"[cf] {report['sessions']} sessions replayed; tail/call ≈ {report['lane_affinity_estimated']['tail_per_call_tokens']} tok")
    for provider, acc in report["providers"].items():
        base = acc["baseline_uncached"] or 1
        print(f"[cf] {provider:14s} resends={acc['resends']:4d} ({acc['resend_tokens']/1e6:6.1f}M tok)"
              f"  stub-saves {acc['stub_saved']/1e6:6.1f}M uncached ({100*acc['stub_saved']/base:4.1f}% of uncached)"
              f"  ttl-loss {acc['ttl_loss']/1e6:6.1f}M")
    aff = report["lane_affinity_estimated"]
    print(f"[cf] affinity (ESTIMATED): {aff['saved']/1e6:.1f}M uncached of {aff['uncached_now']/1e6:.1f}M aux"
          f" across {aff['calls']} aux calls")
    print(f"[cf] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
