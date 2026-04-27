# W13 Live Universe Migration — 2026-04-27

**Bundle**: `w12_60d_ridge_swbuf_v2` → `w12_60d_ridge_swbuf_v3`
**Universe mode**: frozen 501 snapshot → dynamic admission policy
**Trigger**: User feedback — HOOD (real S&P 500 member since 2025-09-22) was missing from Signal Feed because the frozen 2025-02-28 snapshot predates HOOD's S&P inclusion.

## What changed

| Aspect | v2 (pre-W13) | v3 (W13) |
|---|---|---|
| Universe source | static JSON file (501 frozen) | `LiveUniverseResolver(as_of)` dynamic |
| Admission criterion | "in 2025-02-28 snapshot" | positive policy (see below) |
| HOOD visibility | ✗ (not in snapshot) | ✓ (admitted) |
| FISV | ✓ (in snapshot but no longer S&P) | ✗ (correctly excluded) |
| MRSH/PSKY | ✗ (manually patched out) | ✗ (naturally rejected by policy) |

## Admission policy (positive criteria, no blacklist)

Source: `src/universe/live_resolver.py` → `LiveUniversePolicy`

| Gate | Default | What it catches |
|---|---|---|
| Active S&P 500 member at `as_of` | required | non-members |
| `min_history_days` distinct calc_dates in feature_store | 90 | very recent IPOs / fresh adds |
| `min_continuous_price_days` in last 365 days | 200 | discontinuous tickers (mergers, ticker changes — PSKY, MRSH) |
| `min_required_features_present` non-null in last 30 days | 26 / 31 | feature pipeline gaps |
| Latest non-null feature within `max_stale_days` | 7 | data freshness regression |
| 20-day rolling ADV | ≥ $1M | micro-caps / illiquid |

Each ticker gets a `TickerDiagnostic` with structured rejection reasons. Diagnostics are persisted into `week_*.json` under `live_universe_diagnostics`.

## Bundle schema v3

`scripts/freeze_champion_bundle.py` now emits:

```json
{
  "live_universe_mode": "dynamic_admission_policy",
  "live_universe_policy": { ... },
  "research_universe_snapshot_path": "data/.../frozen_universe_501.json",
  "eligible_universe_path": "...same..."   // backward-compat for v1/v2 readers
}
```

`run_greyscale_live.py` checks `live_universe_mode == "dynamic_admission_policy"`. If yes → calls `resolve_live_universe()`. Otherwise → legacy frozen-file path. Old bundles still work.

## Sanity check (pragmatic — universe overlap)

Comparing v3 dynamic universe vs v2 frozen-501 at signal_date 2026-04-24:

| Metric | Value |
|---|---|
| v2 universe count | 477 |
| v3 universe count | 498 |
| Common tickers | 476 (95.6% overlap) |
| Added in v3 | 22 (real 2025+ S&P additions) |
| Removed from v2 | 1 (FISV — ticker change) |
| Score mean abs diff (common) | 0.00096 |
| Score max abs diff (common) | 0.00725 |
| Score diff > 0.01 count | 0 / 476 |

**Verdict**: dynamic universe is essentially a correction of the frozen snapshot, not a fundamentally different cohort. Strategy direction preserved. Safe to cut over.

22 new admissions: APP, ARES, CIEN, COHR, COIN, CRH, CVNA, DASH, DDOG, EME, EXE, FIX, **HOOD**, IBKR, LITE, SATS, SNDK, TKO, TTD, VRT, WSM, XYZ.

4 rejections (correctly): MRSH (insufficient_history), PSKY (discontinuous_prices), 2 others (discontinuous_prices in 113/119/180 days range).

## Cutover

- Wrapper `scripts/run_w12_greyscale_once.sh` updated: BUNDLE → v3
- Saturday 5/2 06:15 China = Friday 5/1 18:15 ET DST = first auto greyscale on v3
- 8-week clock NOT reset; week_01 already on v3 (replaces v2 week_01)

## Files changed

- **NEW** `src/universe/live_resolver.py` (LiveUniverseResolver implementation)
- **NEW** `data/models/bundles/w12_60d_ridge_swbuf_v3/` (bundle artifact)
- `scripts/freeze_champion_bundle.py` (v3 schema with policy + research-snapshot split)
- `scripts/run_greyscale_live.py` (dynamic-universe path with backward-compat fallback)
- `scripts/run_w12_greyscale_once.sh` (BUNDLE → v3)
- v2 week_01 / heartbeat backed up as `*.v2_pre_w13.json.bak`

## What's still NOT changed (no scope creep)

- Ridge model.pkl: identical to v2 (same train+val rows=75312, same α=0.001, same test_ic -0.0293)
- Bundle's required_features (62) and feature_fingerprint format: unchanged
- Greyscale wrapper's `--dry-run` flag: still active (paper, not real money)
- 8-week greyscale gate definition: unchanged
- `BundleValidator`: still validates schema + recency the same way (no per-ticker live-policy validation yet — could be a W14 follow-up)

## Open follow-ups

- The `BundleValidator.validate_recency` is bundle-level; live-resolver does per-ticker. Could unify in future if these contracts diverge.
- New S&P entrant policy: currently any ticker that passes the gates is admitted at next signal date. Could add "next weekly rebalance only" as an explicit field, but Friday-aligned greyscale already enforces this implicitly.
- Stale-snapshot fingerprint: bundle.json v3 still has `eligible_universe_path` pointing to a stale 501 file. The dynamic path doesn't use it, but it's there for v1/v2 reader compat. Drop in v4 once no callers reference it.

## Greyscale 5/2 expectations

- Expect HOOD to appear in Signal Feed
- Expect 22 new tickers to appear (vs the v2 baseline)
- Total universe count ~498 ± a few
- Layer 1-4 all PASS expected (no infra changes)
