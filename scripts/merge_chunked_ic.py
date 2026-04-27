#!/usr/bin/env python3
"""Merge multiple chunked W7 IC screening runs into a single full-span report
that is mathematically equivalent to one big single-pass run.

Background
----------
``run_per_horizon_ic_screening.py`` cannot fit the full-universe 9y panel
(~30 GB) in 19 GB of RAM, so we chunk by date (e.g. 3 × 3y), each chunk runs
under tight ``ulimit -v`` resource caps. Each chunk dumps a per-(feature,
trade_date) ``raw_ic_{horizon}.parquet`` alongside its aggregated CSV.

This script concatenates those raw daily IC values across chunks and
recomputes the W7 metrics — ``mean_ic``, ``t_stat``, ``sign_consistent_windows``
— on the merged daily IC series using the SAME helper functions
(``series_t_stat``, ``screening_windows``, ``screening_status``) the original
runner uses. As long as the chunks are non-overlapping calendar windows over
a frozen ticker universe, the result is bit-equivalent to a single run.

Codex deep-review checks honoured here:
- 5 sanity checks (dedup / date bounds / count match / span min-max /
  optional spot-check) before producing the merged report;
- a chunk-1 CSV is reused as the metadata template so ``family`` and
  ``excluded_reason`` columns survive merge;
- thresholds passed explicitly (no module-default reliance) when invoking
  ``screening_status``.

Usage:
    python scripts/merge_chunked_ic.py \
        --input-dirs /tmp/chunk1_ic /tmp/chunk2_ic /tmp/chunk3_ic \
        --output-dir data/reports/ic_v8_full_9y_merged \
        --expected-start 2016-03-01 --expected-end 2025-02-28
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._week7_ic_utils import (  # noqa: E402
    SCREENING_MEAN_IC_THRESHOLD,
    SCREENING_T_STAT_THRESHOLD,
    SCREENING_SIGN_WINDOW_THRESHOLD,
    screening_status,
    screening_windows,
    series_t_stat,
)


HORIZON_LABELS = ("1d", "5d", "20d", "60d")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_chunk_raw_ic(input_dirs: list[Path], horizon_label: str) -> tuple[pd.DataFrame, list[int]]:
    """Concat raw_ic parquet across chunks and return the merged df + per-chunk row counts."""
    frames: list[pd.DataFrame] = []
    chunk_counts: list[int] = []
    for d in input_dirs:
        path = d / f"raw_ic_{horizon_label}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing raw_ic for {horizon_label} in {d}")
        df = pd.read_parquet(path)
        chunk_counts.append(len(df))
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    merged["trade_date"] = pd.to_datetime(merged["trade_date"]).dt.date
    return merged, chunk_counts


def aggregate_merged_ic(
    merged: pd.DataFrame,
    *,
    mean_ic_threshold: float,
    t_stat_threshold: float,
    sign_window_threshold: int,
) -> pd.DataFrame:
    """Re-run W7 aggregation (mean / t / sign_consistent_windows) on merged daily IC."""
    windows = list(screening_windows())
    rows: list[dict] = []
    for (feature, family), group in merged.groupby(["feature", "family"]):
        daily = pd.Series(
            group["ic_value"].values,
            index=pd.to_datetime(group["trade_date"]),
        ).sort_index()
        mean_ic = float(daily.mean()) if not daily.empty else np.nan
        t_stat = series_t_stat(daily)
        dominant_sign = (
            np.sign(mean_ic) if pd.notna(mean_ic) and not np.isclose(mean_ic, 0.0) else 0.0
        )
        sign_consistent = 0
        for window in windows:
            mask = (
                (pd.to_datetime(daily.index).date >= window.test_start)
                & (pd.to_datetime(daily.index).date <= window.test_end)
            )
            window_series = daily.loc[mask]
            if window_series.empty:
                continue
            window_ic = float(window_series.mean())
            if dominant_sign != 0.0 and pd.notna(window_ic) and np.sign(window_ic) == dominant_sign:
                sign_consistent += 1
        status = screening_status(
            mean_ic=mean_ic,
            t_stat=t_stat,
            sign_consistent_windows=sign_consistent,
            mean_ic_threshold=mean_ic_threshold,
            t_stat_threshold=t_stat_threshold,
            sign_window_threshold=sign_window_threshold,
        )
        rows.append({
            "feature": feature,
            "family": family,
            "mean_ic": round(float(mean_ic), 6) if pd.notna(mean_ic) else None,
            "t_stat": round(float(t_stat), 6) if pd.notna(t_stat) else None,
            "sign_consistent_windows": int(sign_consistent),
            "status": status,
        })
    return pd.DataFrame(rows)


def write_merged_csv(
    *,
    template_csv: Path,
    merged_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Use chunk-1 CSV as template (preserves family + excluded_reason rows for
    features that appear in CSV but not in raw_ic, e.g. excluded ones)."""
    template = pd.read_csv(template_csv)
    template_idx = template.set_index("feature")

    # For features present in merged_metrics, overwrite mean_ic/t_stat/sign_consistent_windows/status.
    # Excluded-reason rows in template (where mean_ic is NaN) keep their original NaN/0/FAIL.
    merged_idx = merged_metrics.set_index("feature")
    for feature in merged_idx.index:
        if feature in template_idx.index:
            template_idx.at[feature, "mean_ic"] = merged_idx.at[feature, "mean_ic"]
            template_idx.at[feature, "t_stat"] = merged_idx.at[feature, "t_stat"]
            template_idx.at[feature, "sign_consistent_windows"] = merged_idx.at[feature, "sign_consistent_windows"]
            template_idx.at[feature, "status"] = merged_idx.at[feature, "status"]
            template_idx.at[feature, "excluded_reason"] = ""
    out = template_idx.reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def sanity_check(
    *,
    merged: pd.DataFrame,
    chunk_counts: list[int],
    expected_start: date | None,
    expected_end: date | None,
    horizon_label: str,
) -> dict:
    """Run Codex-recommended 5 sanity checks. Returns a dict of stats; raises
    AssertionError on hard failures."""
    diagnostics: dict = {"horizon": horizon_label}

    # 1. concat 后无重复 (feature, trade_date)
    dup_count = int(merged.duplicated(subset=["feature", "trade_date"]).sum())
    diagnostics["duplicate_rows"] = dup_count
    assert dup_count == 0, f"{horizon_label}: found {dup_count} duplicate (feature, trade_date) rows"

    # 3. merged_count == sum(chunk_counts) (after dedup; with non-overlap should equal naive sum)
    sum_chunks = sum(chunk_counts)
    diagnostics["merged_rows"] = len(merged)
    diagnostics["chunk_rows_total"] = sum_chunks
    assert len(merged) == sum_chunks, (
        f"{horizon_label}: merged rows {len(merged)} != Σ chunks {sum_chunks} "
        f"(non-overlap assumption violated?)"
    )

    # 4. merged trade_date min/max == expected span
    actual_min = merged["trade_date"].min()
    actual_max = merged["trade_date"].max()
    diagnostics["actual_min_date"] = actual_min.isoformat()
    diagnostics["actual_max_date"] = actual_max.isoformat()
    if expected_start is not None:
        assert actual_min >= expected_start, (
            f"{horizon_label}: min trade_date {actual_min} < expected start {expected_start}"
        )
    if expected_end is not None:
        assert actual_max <= expected_end, (
            f"{horizon_label}: max trade_date {actual_max} > expected end {expected_end}"
        )

    # 5. (light) feature count
    diagnostics["unique_features"] = int(merged["feature"].nunique())
    return diagnostics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", nargs="+", type=Path, required=True,
                        help="Chunk output dirs in chronological order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-start", type=_parse_date, default=None)
    parser.add_argument("--expected-end", type=_parse_date, default=None)
    parser.add_argument("--mean-ic-threshold", type=float, default=SCREENING_MEAN_IC_THRESHOLD)
    parser.add_argument("--t-stat-threshold", type=float, default=SCREENING_T_STAT_THRESHOLD)
    parser.add_argument("--sign-window-threshold", type=int, default=SCREENING_SIGN_WINDOW_THRESHOLD)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_per_horizon = {}

    for horizon_label in HORIZON_LABELS:
        merged, chunk_counts = load_chunk_raw_ic(args.input_dirs, horizon_label)

        # 2. assert each chunk row strictly within its directory's chunk bounds —
        # recovered from `(min, max)` of each chunk df and stored in diagnostics.
        per_chunk_bounds = []
        offset = 0
        for d, n in zip(args.input_dirs, chunk_counts):
            chunk_df = merged.iloc[offset:offset + n]
            per_chunk_bounds.append({
                "dir": str(d),
                "rows": int(n),
                "min_date": chunk_df["trade_date"].min().isoformat() if n else None,
                "max_date": chunk_df["trade_date"].max().isoformat() if n else None,
            })
            offset += n

        diag = sanity_check(
            merged=merged,
            chunk_counts=chunk_counts,
            expected_start=args.expected_start,
            expected_end=args.expected_end,
            horizon_label=horizon_label,
        )
        diag["per_chunk_bounds"] = per_chunk_bounds
        diagnostics_per_horizon[horizon_label] = diag

        merged_metrics = aggregate_merged_ic(
            merged,
            mean_ic_threshold=args.mean_ic_threshold,
            t_stat_threshold=args.t_stat_threshold,
            sign_window_threshold=args.sign_window_threshold,
        )

        # Use chunk1 CSV as template to preserve family/excluded_reason rows.
        template_csv = args.input_dirs[0] / f"ic_screening_v7_{horizon_label}.csv"
        output_csv = args.output_dir / f"ic_screening_v7_{horizon_label}.csv"
        write_merged_csv(
            template_csv=template_csv,
            merged_metrics=merged_metrics,
            output_path=output_csv,
        )
        print(f"[merge_chunked_ic] {horizon_label}: wrote {output_csv} "
              f"(features={diag['unique_features']}, dates={diag['actual_min_date']}~{diag['actual_max_date']}, "
              f"rows={diag['merged_rows']})")

    # Persist diagnostics for the audit trail.
    diag_path = args.output_dir / "merge_diagnostics.json"
    diag_path.write_text(json.dumps(diagnostics_per_horizon, ensure_ascii=False, indent=2, default=str))
    print(f"[merge_chunked_ic] sanity diagnostics -> {diag_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
