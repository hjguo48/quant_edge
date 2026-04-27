# W12 Universe Quality Patch — 2026-04-27

**Bundle**: `w12_60d_ridge_swbuf_v1` → `w12_60d_ridge_swbuf_v2`
**Universe**: 503 → 501 tickers
**Trigger**: User-reported MRSH price chart anomaly in frontend SignalDetail (gap 2016 → 2026)

## Issue

`frozen_universe_503.json` contained two phantom identities — `MRSH` and `PSKY` — that:
- Have **zero rows** in `walkforward_v9full9y_fm_60d.parquet` (the W7-W11 research feature matrix)
- Were never represented in W7-W11 IC / walk-forward / truth-table results
- Were re-introduced at live-greyscale time via the universe-membership intersect path
- Appeared as **rank #1 (PSKY, 0.224) and rank #2 (MRSH, 0.220)** in the 4/27 dry-run, with combined ~5.15% paper-portfolio weight

## Root cause

Universe quality bug: `frozen_universe_503.json` was sourced from a snapshot that included tickers without research-grade feature coverage. The cheap fingerprint + distinct-name validator did not catch this because the feature-store had a few intraday-feature rows for these tickers (all-null), satisfying the existence check.

## Fix scope

This patch is **universe-only**. The Ridge model + research IC / Sharpe / DSR / SPA / capacity sweeps are unaffected because MRSH and PSKY were never in the research matrix.

| Artifact | v1 | v2 | Diff |
|---|---|---|---|
| Bundle dir | `data/models/bundles/w12_60d_ridge_swbuf_v1/` | `data/models/bundles/w12_60d_ridge_swbuf_v2/` | new dir |
| Universe file | `frozen_universe_503.json` | `frozen_universe_501.json` | -2 tickers |
| Universe count | 503 | 501 | -2 |
| Reproduced test_ic | -0.0293 | -0.0293 | identical |
| Ridge alpha | 0.001 | 0.001 | identical |
| Train rows | 75312 | 75312 | identical |
| Feature fingerprint | (v1 hash) | (v2 hash) | differs (version field) |

## Verification (dry-run as_of 2026-04-25)

After v2:
- Total tickers in fusion: 477 (down from 479)
- PSKY, MRSH: **absent** ✓
- Top 5: FISV, CFG, BX, COF, NVR — all real S&P 500 names
- Layer 1-4 risk gates: all PASS
- Greyscale week_01.json regenerated cleanly

## What was W7-W11 research validity?

**Unchanged.** Cross-checked: MRSH and PSKY have **0 rows** in:
- `walkforward_v9full9y_fm_60d.parquet`
- 60D feature matrix (501 distinct tickers, not 503)

W10 truth table (24 cells), W10.5 stress/capacity, W11 fusion verdict all stand.

## What changed in the live stack

- `data/features/frozen_universe_501.json` — new
- `data/models/bundles/w12_60d_ridge_swbuf_v2/` — new bundle dir
- `data/models/fusion_model_bundle_60d.json` — legacy pointer refreshed to v2
- `scripts/run_w12_greyscale_once.sh` — `BUNDLE` var → v2
- `scripts/freeze_champion_bundle.py` — universe filename now derived from source path basename (no more hardcoded `frozen_universe_503.json`)

## What changed in the frontend

`Signals.tsx` and `Dashboard.tsx`:
- `direction: prediction.score > 0 ? "long" : "short"` was misleading for the long-only champion. Negative-score positions are **buffer-held LONG** (held by the `sell_buffer_pct=0.25` mechanism), not short positions.
- Renamed in UI: "long / short" → "active / buffer-held"
- `DIRECTIONS` filter: `["All", "Long", "Short"]` → `["All", "Active", "Buffer"]`
- `SignalRow` already had a `"neutral"` path; `direction` now maps `score <= 0 → "neutral"` to surface buffer-held semantics

## Backwards-compat notes

- `w12_60d_ridge_swbuf_v1/` directory is left in place for audit / rollback. No callers should reference it after this commit.
- Contaminated week_01.json moved to `data/reports/greyscale/week_01.contaminated_v1_psky_mrsh.json.bak` (not deleted).

## Follow-up (not in this patch)

- W13 / future: extend `BundleValidator.validate_recency` to also validate `len(retained_features) > 0` per ticker on a sample date — would have caught this earlier.
- W13: surface `strategy_mode: "long_only" | "long_short"` in `/api/predictions/latest` so frontend doesn't have to hardcode buffer-held semantics.

## First scheduled greyscale

Saturday 2026-05-02 06:15 China = Friday 2026-05-01 18:15 ET DST.
Will use v2 bundle (wrapper updated). No reschedule needed.
