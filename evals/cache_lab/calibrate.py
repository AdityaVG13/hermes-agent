"""Grid-fit the prefix-replay model against observed cache rates.

    python -m evals.cache_lab.calibrate --db ~/.hermes/state.db --days 14 --min-calls 20

For each session with a dominant main-lane usage row, replay the transcript,
then fit (ttl, reliability) per billing provider by mean absolute error on the
cache share. GATE: at each provider's best fit, >= gate_share of sessions must
land within gate_pts points of the observed rate — only then may counterfactual
replays be trusted for that provider. Everything is read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

from evals.cache_lab.replay import cache_ratio, read_session_calls, simulate

TTL_GRID: list[float | None] = [120, 300, 600, 1800, 3600, None]
# Reliability doubles as a cache-support detector: a provider whose best fit is
# ~0 serves no cache regardless of prefix (the crouton/stealth case), which the
# session-start guard also flags from the same usage rows.
RELIABILITY_GRID: list[float] = [1.0, 0.97, 0.95, 0.9, 0.75, 0.5, 0.25, 0.0]


def _targets(db: sqlite3.Connection, days: int, min_calls: int) -> list[dict]:
    cutoff = time.time() - days * 86400.0
    rows = db.execute(
        "SELECT session_id, model, billing_provider, api_call_count, cache_read_tokens,"
        " input_tokens FROM session_model_usage WHERE last_seen > ? AND model <> ''"
        " AND (task = '' OR task IS NULL)", (cutoff,)).fetchall()
    by_session: dict[str, list[tuple]] = defaultdict(list)
    for r in rows:
        by_session[r[0]].append(r)
    targets = []
    for sid, rows in by_session.items():
        total_calls = sum(r[3] for r in rows)
        if total_calls < min_calls:
            continue
        dom = max(rows, key=lambda r: r[3])
        if dom[3] / total_calls < 0.8:   # mixed-model session: replay can't attribute
            continue
        denom = (dom[4] or 0) + (dom[5] or 0)
        if denom <= 0:
            continue
        targets.append({
            "session_id": sid, "model": dom[1], "provider": dom[2],
            "calls": dom[3], "observed_pct": 100.0 * (dom[4] or 0) / denom})
    return targets


def calibrate(db: sqlite3.Connection, days: int, min_calls: int,
              gate_pts: float = 3.0, gate_share: float = 0.8) -> dict:
    targets = _targets(db, days, min_calls)
    replays: dict[str, list] = {}
    for t in targets:
        try:
            calls = read_session_calls(db, t["session_id"])
        except sqlite3.Error:
            continue
        if len(calls) >= 2:
            replays[t["session_id"]] = calls

    by_provider: dict[tuple, list[dict]] = defaultdict(list)
    for t in targets:
        if t["session_id"] in replays:
            # Cache behavior is per upstream model, not per provider key:
            # one relay can front both a caching model (deepseek ~96%) and a
            # non-caching one (stealth 0%). Group by (provider, model).
            by_provider[(t["provider"] or "unknown", t["model"])].append(t)

    report: dict = {"gate": {"pts": gate_pts, "share": gate_share},
                    "sessions_considered": len(targets), "sessions_replayed": len(replays),
                    "providers": {}}
    for (provider, model), ts in sorted(by_provider.items(), key=lambda kv: -len(kv[1])):
        fits = []
        for ttl in TTL_GRID:
            for rel in RELIABILITY_GRID:
                errs = []
                for t in ts:
                    calls = replays[t["session_id"]]
                    cached, total = cache_ratio(simulate(calls, ttl_seconds=ttl, reliability=rel))
                    pred = 100.0 * cached / total if total else 0.0
                    errs.append(abs(pred - t["observed_pct"]))
                fits.append({"ttl": ttl, "reliability": rel,
                             "mae": sum(errs) / len(errs),
                             "within_gate": sum(1 for e in errs if e <= gate_pts) / len(errs),
                             "max_err": max(errs)})
        fits.sort(key=lambda f: f["mae"])
        best = fits[0]
        report["providers"][f"{provider} / {model}"] = {
            "sessions": len(ts), "best_fit": best, "top_fits": fits[:4],
            "gate_pass": best["within_gate"] >= gate_share}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--db", default=str(Path.home() / ".hermes" / "state.db"))
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--min-calls", type=int, default=20)
    ap.add_argument("--out", default="cache_calibration.json")
    a = ap.parse_args(argv)

    db = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
    report = calibrate(db, a.days, a.min_calls)
    Path(a.out).write_text(json.dumps(report, indent=2))

    print(f"[calibrate] {report['sessions_replayed']}/{report['sessions_considered']} sessions replayed")
    for provider, pr in report["providers"].items():
        b = pr["best_fit"]
        verdict = "PASS" if pr["gate_pass"] else "FAIL"
        print(f"[calibrate] {provider:16s} n={pr['sessions']:4d}  ttl={b['ttl']} rel={b['reliability']}"
              f"  MAE={b['mae']:.1f}pts  within±3={b['within_gate']:.0%}  max={b['max_err']:.0f}pts  gate {verdict}")
    print(f"[calibrate] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
