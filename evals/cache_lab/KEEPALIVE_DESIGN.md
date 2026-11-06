# Keepalive design — reaching the human-paced TTL pool

Status: **DO NOT BUILD — killed by observed bounding (2026-11-05).**

Verdict: the replay-projected pool (2.85G tokens) inflated real prompts on
uncompacted monster sessions. Bounded by *observed* uncached spend instead:
codex 14d has 467M uncached total, of which post-gap calls carry only
**2.33M tokens** (1,027 of 40,697 calls) — ≈52 credits at the published
card (cached = 0.1× input). Pings would cost more than they save; the
meter read 0% with ~190 credits spent since the window reset, confirming
per-gap refills are trivially small. Lab lesson recorded in
counterfactual.py: stream-replay RATIOS are calibrated (gate-passed), but
ABSOLUTE token-savings projections must be bounded by observed
session_model_usage input_tokens.

## The pool (measured, 14d, 268 codex sessions)

`gap_causes.py`: 2.85G uncached tokens lost to idle gaps past the fitted
600s codex cache TTL. **100% human-paced** (15–60min buckets dominate) —
scheduling levers reach zero of it; a pre-expiry keepalive ping is the only
reach. Concentration: 200 gaps, top bucket 100 gaps ≈ 1.3G.

## Mechanics

When a session on an opted-in route goes idle past `ttl − ε`, send one
minimal completion request with the exact assembled prefix (same
`api_content` bytes — any drift makes the ping itself a miss and the whole
point moot), `max_tokens: 1`. The provider re-serves the prefix as cache
read and refreshes the entry; the user's next real call reads cache instead
of paying full input.

## Economics — the one unknown, and how the design kills it

Per idle window of prefix P tokens, cache-read price c, input price i:

    keepalive saves  P·(i − 2c)     (ping pays P·c, next call pays P·c not P·i)

- c = 0.10·i → saves 0.80·P  (≈2.3G/14d on codex) → build it
- c = 0.50·i → saves 0.00    → pointless
- c ≥ 0.50·i → hurts

The codex sub's effective cache-read quota weight is not locally measurable
yet (CodexBar records per-account percents; Hermes rotates accounts per
session, so the fit needs window-aligned attribution). Therefore the feature
ships **self-measuring** and defaults OFF:

1. Every ping logs `prefix_tokens` and the observed `cached_tokens` of the
   ping itself (proves the prefix was still warm) into session_model_usage
   under `task='keepalive'`.
2. The next real call's cached delta vs. the no-keepalive counterfactual is
   reported per week by `evals.cache_lab` (keepalive rows are separable).
3. Kill-switch arithmetic in the report: if median next-call cache-read
   after a ping ≈ 0 (ping missed) or quota spend rose while total tokens
   fell ≤ 0 → the lane's c ≥ 0.5·i → recommend OFF for that provider.

## Guards (fail-closed)

- Per-provider opt-in table (config), default empty; `openai-codex` first.
- Cap: max 6 pings/session, none after 6h idle (a stale session is dead —
  let it expire), none when the gateway is on battery-saver/quit-pending.
- Rate-limit aware: a ping that 429s disables keepalive for that session.
- Never ping during an active turn (lock on turn leases); only true idle.
- Ping uses the same credential/lane as the session's last call (account
  stickiness matters: cache is per-account upstream).
- Observability: one log line per ping, aggregated weekly by the lab —
  no per-ping user-visible noise (standing preference: no idle chatter).

## What would make this obsolete

If the sub is later measured at c ≥ 0.5·i, delete the feature; the report
output is the evidence. If OpenAI lengthens the TTL, shrink the pool first
(refit FITTED_TTL via calibrate.py) before crediting the pinger.
