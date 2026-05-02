# 2026-05-02 W12 Greyscale Preflight Failure Diagnosis

Generated: 2026-05-02 Asia/Shanghai  
Scope: investigation only; no production code or data fixed.

## Executive Summary

The 2026-05-02 05:00 UTC W12 greyscale wrapper failed before `run_greyscale_live.py`.
The PIT date check passed (`latest_pit_trade_date=2026-04-30`, expected `2026-05-01`, tolerance `2026-04-30`), but `BundleValidator.validate_recency(..., min_coverage_count=100)` failed on two required features:

| feature | latest calc date | coverage | broken since | root cause |
|---|---:|---:|---|---|
| `short_sale_ratio_5d` | 2026-05-01 | 0 | source absent from 2026-04-24 onward; latest wrapper self-heal wrote all-null 2026-05-01 rows | `short_sale_volume_daily` is not refreshed by any daily DAG task and currently contains only synthetic/test-like AAPL/ADF rows. |
| `high_vix_x_beta` | 2026-04-30 | 0 | 2026-04-17 | `stock_beta_252` and `vix` are dense; composite VIX z-score is computed only over the requested output slice, which is too short for `_rolling_date_zscore(..., min_periods=60)`. |

The preflight log confirms 12 identical failures:

```text
OK: PIT trade date=2026-04-30, expected=2026-05-01, tolerance=2026-04-30
FAIL: 2 features sparse (<100 tickers): ['high_vix_x_beta', 'short_sale_ratio_5d']
```

## Evidence

### Wrapper / Validator

- `scripts/run_w12_greyscale_once.sh:205-218` runs `BundleValidator.validate_recency(conn, max_stale_days=7, min_coverage_count=100)` and exits preflight on sparse features.
- Local validator reproduction:

```text
passed False
stale []
sparse ['high_vix_x_beta', 'short_sale_ratio_5d']
high_vix_x_beta {'latest_calc_date': '2026-04-30', 'coverage': 0}
short_sale_ratio_5d {'latest_calc_date': '2026-05-01', 'coverage': 0}
```

### Airflow

There is no `scheduled__2026-05-01T06:00:00+00:00` daily run yet. The latest successful daily run is logical date `2026-04-30T06:00:00+00:00`, started `2026-05-01T06:00:01Z`.

Task states for that run:

```text
fetch_prices success
fetch_vix success
fetch_macro success
fetch_fundamentals success
fetch_alternative_data success
update_features_cache success
check_quality success
```

`fetch_alternative_data` did not fetch FINRA daily short-sale volume. It fetched:

```text
sources: ['earnings', 'insider', 'analyst', 'short-interest']
short-interest incremental complete: 1002 rows since 2026-04-22
```

That `short-interest` source is Polygon biweekly short interest, not the FINRA daily RegSHO table used by `short_sale_ratio_5d`.

## Root Cause 1: `short_sale_ratio_5d`

### Responsible modules

- Source model/ingest implementation: `src/data/finra_short_sale.py:75-155`, `src/data/finra_short_sale.py:216-279`
- Manual backfill script: `scripts/backfill_finra_short_sale.py:17-64`
- Feature computation: `src/features/shorting.py:32-55`, `src/features/shorting.py:77-95`, `src/features/shorting.py:186-210`
- Daily DAG alternative path: `dags/dag_daily_data.py:808-881`

### Findings

`short_sale_volume_daily` currently contains only 91 rows total:

```text
min trade_date: 2025-12-11
max trade_date: 2026-04-23
rows: 91
distinct dates: 91
distinct tickers: 1
```

Recent rows are all AAPL / ADF only:

```text
2026-04-14..2026-04-23: 1 row/day, 1 ticker/day, 1 market/day
ticker=AAPL, market=ADF, total_volume=1000
file_etag='etag-YYYY-MM-DD' or 'etag-current'
```

Those values match the unit-test fixture shape in `tests/test_features/test_shorting.py:250-348` (`ticker=AAPL`, `market=ADF`, `file_etag=etag-current`, alternating short-volume rows). This strongly indicates the runtime DB has test/synthetic FINRA rows, not real FINRA market-wide rows.

The daily DAG does not call `FINRAShortSaleSource` or `scripts/backfill_finra_short_sale.py`; `dags/dag_daily_data.py:822-823` appends only Polygon `short-interest` to alternative sources. So this was not a silent FINRA fetch failure in `fetch_alternative_data`; FINRA daily short-sale volume is simply not wired into the daily refresh path.

Wrapper self-heal then ran:

```text
scripts/backfill_shorting_features.py --use-dynamic-universe --recent-fridays 2
output dates: [2026-04-24, 2026-05-01]
tickers: 498
saved 7968 feature rows, batch_id=063ca337-bf25-4ab3-bb44-a0b74d96179d
```

But because the source table had no real multi-ticker FINRA rows, it wrote null features:

```text
short_sale_ratio_5d 2026-04-24 batch 063ca337...: 498 rows, 1 non-null
short_sale_ratio_5d 2026-05-01 batch 063ca337...: 498 rows, 0 non-null
```

The bundle currently requires `short_sale_ratio_5d`; `short_sale_ratio_1d` and `short_sale_accel` were also all null in the self-heal batch, but they are not the validator's sparse required feature in this bundle.

### Broken-since date

- Source table is invalid for the current DB as far back as its minimum date (`2025-12-11`): one AAPL/ADF row per date, not market-wide FINRA data.
- Launch-blocking source gap begins `2026-04-24`: no `short_sale_volume_daily` rows exist from `2026-04-24` through `2026-05-01`.
- Launch-blocking feature failure is latest `short_sale_ratio_5d` on `2026-05-01`: 498 rows, 0 non-null coverage.

## Root Cause 2: `high_vix_x_beta`

### Responsible modules

- Technical beta: `src/features/technical.py:289-300`, `src/features/technical.py:392-411`
- Composite feature: `src/features/pipeline.py:1357-1358`
- VIX z-score helper: `src/features/pipeline.py:1424-1433`
- Daily feature refresh slicing: `dags/dag_daily_data.py:1189-1194`
- Feature exporter: `scripts/build_feature_matrix.py:148-202`, `scripts/run_ic_screening.py:799-805`

### Findings

This is not a beta data-source failure:

```text
2026-04-30 stock_beta_252: 617 rows, 608 non-null (98.54%)
2026-04-30 vix:            617 rows, 617 non-null (100.00%)
2026-04-30 high_vix_x_beta: 617 rows, 0 non-null
```

`high_vix_x_beta` was dense through `2026-04-16`, then became 0% non-null starting `2026-04-17`:

```text
2026-04-16 high_vix_x_beta: 617 rows, 608 non-null
2026-04-17 high_vix_x_beta: 618 rows, 0 non-null
2026-04-30 high_vix_x_beta: 617 rows, 0 non-null
```

The composite is:

```python
vix_z = _rolling_date_zscore(wide.get("vix"), window=252)
high_vix_x_beta = vix_z.clip(lower=0) * stock_beta_252
```

`_rolling_date_zscore` requires `min_periods = max(20, min(window, 60))`, so it needs 60 date-level observations inside the feature frame. But the daily/incremental feature path computes `macro = _compute_broadcast_macro_features(output_prices, as_of_ts)` only for the requested output window; it does not include the 60+ historical macro dates needed for the rolling VIX z-score.

When the feature cache is advanced incrementally (`refresh_start = latest_feature_date + 1`), output windows become one/few dates. `stock_beta_252` still works because technical features pull 520 calendar days of price history, but `high_vix_x_beta` collapses to NaN because the VIX z-score has no warmup history in the output slice.

### Broken-since date

`high_vix_x_beta` first became fully null on `2026-04-17` and remained fully null through `2026-04-30`.

## Immediate Backfill / Recovery Plan

Do not run these until approved; these are the recommended commands.

### 1. Repair FINRA daily source rows

Minimum to unblock the bundle's `short_sale_ratio_5d` for `2026-05-01`:

```bash
.venv/bin/python scripts/backfill_finra_short_sale.py \
  --start-date 2026-04-24 \
  --end-date 2026-05-01 \
  --markets CNMS,ADF,BNY \
  --force-refetch
```

Safer repair for the whole shorting family, including `short_sale_accel` and `abnormal_off_exchange_shorting`:

```bash
.venv/bin/python scripts/backfill_finra_short_sale.py \
  --start-date 2026-01-01 \
  --end-date 2026-05-01 \
  --markets CNMS,ADF,BNY \
  --force-refetch
```

Then recompute shorting features for the latest launch date:

```bash
.venv/bin/python scripts/backfill_shorting_features.py \
  --use-dynamic-universe \
  --as-of 2026-05-01
```

Expected validator target after this: `short_sale_ratio_5d` latest `2026-05-01` with >100 non-null tickers.

### 2. Repair `high_vix_x_beta`

Recompute a wide enough feature window so `_rolling_date_zscore` has at least 60 date observations before the launch date. This is heavier but uses the canonical exporter:

```bash
.venv/bin/python scripts/build_feature_matrix.py \
  --start-date 2026-01-02 \
  --end-date 2026-04-30 \
  --as-of 2026-05-02T05:00:00+00:00 \
  --output-path data/features/all_features.parquet \
  --batch-size 25 \
  --max-workers 8 \
  --sync-feature-store \
  --clear-feature-store-range \
  --metadata-output data/features/feature_export_20260502_repair.json
```

If runtime/memory is a concern, the code fix should be to give the VIX z-score composite its own historical macro window instead of requiring a wide output recompute. For immediate launch recovery, the wide recompute is the safer operational path.

### 3. Validate before rerun

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
from src.data.db.session import get_engine
from src.models.bundle_validator import BundleValidator

bv = BundleValidator(Path("data/models/bundles/w12_60d_ridge_swbuf_v3/bundle.json"))
with get_engine().connect() as conn:
    rec = bv.validate_recency(conn, max_stale_days=7, min_coverage_count=100)
print(rec.passed)
print(rec.sparse_features)
for name in ["high_vix_x_beta", "short_sale_ratio_5d"]:
    print(name, rec.metadata["per_feature_stats"].get(name))
PY
```

### 4. Rerun wrapper

Preferred, because it preserves email/report/monitor behavior:

```bash
bash scripts/run_w12_greyscale_once.sh
```

Direct diagnostic rerun if needed:

```bash
.venv/bin/python scripts/run_greyscale_live.py \
  --bundle-path data/models/bundles/w12_60d_ridge_swbuf_v3/bundle.json \
  --report-dir data/reports/greyscale \
  --reference-feature-matrix-path data/features/feature_matrix_w11.parquet \
  --dry-run \
  --as-of 2026-05-02T05:00:00+00:00
```

## Prevention

P2 guardrails to add after launch recovery:

1. Add FINRA daily short-sale volume to `daily_data_pipeline` explicitly; do not confuse Polygon `/stocks/v1/short-interest` with FINRA daily RegSHO short volume.
2. Add source-table coverage checks in daily DAG:
   - `short_sale_volume_daily` latest PIT-visible session must have `count(distinct ticker) >= 100`.
   - `stock_prices`, `fundamentals_pit`, and other per-ticker source tables should use the same threshold.
3. Add a test/fixture isolation guard: test-shaped `file_etag like 'etag-%'` / one-ticker AAPL rows should never remain in the runtime DB.
4. Add feature-family coverage assertions after `update_features_cache`: required bundle features should fail the DAG immediately if latest coverage is below threshold, not only during the Saturday wrapper.
5. Fix `high_vix_x_beta` generation so macro rolling z-scores are computed over a historical macro window independent of the output slice. The current incremental feature-cache path can keep producing nulls for any rolling date-level macro composite.

## Bottom Line

The 5/2 launch did not fail because price PIT data was late. It failed because:

1. FINRA daily short-sale volume is not in the daily DAG and the runtime table contains only test-shaped AAPL/ADF rows, so the wrapper's shorting self-heal wrote all-null `short_sale_ratio_5d` for `2026-05-01`.
2. `high_vix_x_beta` uses dense beta and VIX inputs, but its rolling VIX z-score is computed on too narrow an incremental feature slice, making the composite all null since `2026-04-17`.

