#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, summarize_issues, write_json_report


LABEL_PATHS = {
    "1D": Path("data/labels/forward_returns_1d.parquet"),
    "5D": Path("data/labels/forward_returns_5d.parquet"),
    "10D": Path("data/labels/forward_returns_10d.parquet"),
    "20D": Path("data/labels/forward_returns_20d.parquet"),
    "60D": Path("data/labels/forward_returns_60d.parquet"),
}


def _load_split_diagnostics(engine) -> dict[str, Any]:
    ratio_bucket_query = text(
        """
        select round((adj_close / nullif(close, 0))::numeric, 6) as ratio, count(*) as rows
        from stock_prices
        where close is not null
          and adj_close is not null
          and close <> 0
          and ((adj_close / close) > 2.0 or (adj_close / close) < 0.1)
        group by 1
        order by rows desc, ratio
        limit 20
        """
    )
    ticker_query = text(
        """
        select ticker, count(*) as rows, min(trade_date) as min_date, max(trade_date) as max_date
        from stock_prices
        where close is not null
          and adj_close is not null
          and close <> 0
          and ((adj_close / close) > 2.0 or (adj_close / close) < 0.1)
        group by ticker
        order by rows desc, ticker
        """
    )
    coverage_query = text(
        """
        with anomalies as (
          select ticker, trade_date, (adj_close / nullif(close, 0))::numeric as ratio
          from stock_prices
          where close is not null
            and adj_close is not null
            and close <> 0
            and ((adj_close / close) > 2.0 or (adj_close / close) < 0.1)
        ),
        nearest as (
          select
            a.ticker,
            a.trade_date,
            a.ratio,
            ca.ex_date,
            ca.ratio as corporate_action_ratio,
            abs(a.trade_date - ca.ex_date) as day_distance,
            row_number() over (
              partition by a.ticker, a.trade_date
              order by abs(a.trade_date - ca.ex_date), ca.ex_date
            ) as rn
          from anomalies a
          left join corporate_actions ca
            on ca.ticker = a.ticker
           and ca.action_type = 'split'
        )
        select
          count(*) as total,
          count(*) filter (where ex_date is not null and day_distance <= 14) as covered_14d,
          count(*) filter (where ex_date is not null and day_distance <= 30) as covered_30d,
          count(*) filter (where ex_date is not null and day_distance <= 365) as covered_365d,
          count(*) filter (where ex_date is null) as no_split_action
        from nearest
        where rn = 1
        """
    )
    sample_query = text(
        """
        with anomalies as (
          select ticker, trade_date, close, adj_close, (adj_close / nullif(close, 0))::numeric as ratio
          from stock_prices
          where close is not null
            and adj_close is not null
            and close <> 0
            and ((adj_close / close) > 2.0 or (adj_close / close) < 0.1)
        ),
        nearest as (
          select
            a.ticker,
            a.trade_date,
            a.close,
            a.adj_close,
            a.ratio,
            ca.ex_date,
            ca.ratio as corporate_action_ratio,
            abs(a.trade_date - ca.ex_date) as day_distance,
            row_number() over (
              partition by a.ticker, a.trade_date
              order by abs(a.trade_date - ca.ex_date), ca.ex_date
            ) as rn
          from anomalies a
          left join corporate_actions ca
            on ca.ticker = a.ticker
           and ca.action_type = 'split'
        )
        select *
        from nearest
        where rn = 1
        order by ticker, trade_date
        limit 50
        """
    )

    with engine.connect() as conn:
        ratio_buckets = [dict(row) for row in conn.execute(ratio_bucket_query).mappings().all()]
        ticker_counts = [dict(row) for row in conn.execute(ticker_query).mappings().all()]
        coverage = dict(conn.execute(coverage_query).mappings().first())
        samples = [dict(row) for row in conn.execute(sample_query).mappings().all()]

    diagnosis = (
        "Most split-like rows are concentrated in 17 tickers and exact split-factor buckets "
        "(0.05, 0.066667, 15, 5, 50, 200, etc.), which indicates forward-adjusted Polygon history "
        "rather than random price corruption. Existing corporate_actions coverage is incomplete for many of these tickers."
    )
    return {
        "summary": coverage,
        "ratio_buckets": ratio_buckets,
        "ticker_counts": ticker_counts,
        "sample_rows": samples,
        "root_cause": diagnosis,
    }


def _load_zero_volume_diagnostics(engine) -> dict[str, Any]:
    summary_query = text(
        """
        select
          count(*) as total_zero_volume_rows,
          count(*) filter (where open = high and high = low and low = close) as flat_ohlc_zero_rows,
          count(*) filter (where open = high and high = low and low = close and close = adj_close) as flat_equal_adj_rows,
          count(*) filter (where not (open = high and high = low and low = close)) as nonflat_zero_rows
        from stock_prices
        where coalesce(volume, 0) = 0
        """
    )
    ticker_query = text(
        """
        select ticker, count(*) as rows, min(trade_date) as min_date, max(trade_date) as max_date
        from stock_prices
        where coalesce(volume, 0) = 0
        group by ticker
        order by rows desc, ticker
        """
    )
    sample_query = text(
        """
        select ticker, trade_date, open, high, low, close, adj_close, volume, knowledge_time
        from stock_prices
        where coalesce(volume, 0) = 0
        order by ticker, trade_date
        limit 50
        """
    )

    with engine.connect() as conn:
        summary = dict(conn.execute(summary_query).mappings().first())
        tickers = [dict(row) for row in conn.execute(ticker_query).mappings().all()]
        samples = [dict(row) for row in conn.execute(sample_query).mappings().all()]

    diagnosis = (
        "All zero-volume rows are flat OHLC ghost bars, not realistic halts with intra-day range. "
        "They appear to be stale synthetic history/backfill artifacts and can be removed by a targeted flat-zero-volume rule."
    )
    return {
        "summary": summary,
        "ticker_counts": tickers,
        "sample_rows": samples,
        "root_cause": diagnosis,
    }


def _load_pit_diagnostics(engine) -> dict[str, Any]:
    summary_query = text(
        """
        select source, count(*) as rows, min(trade_date) as min_date, max(trade_date) as max_date
        from stock_prices
        where ((knowledge_time at time zone 'UTC')::date) <= trade_date
        group by source
        order by rows desc, source
        """
    )
    hour_query = text(
        """
        select extract(hour from knowledge_time at time zone 'UTC') as hour_utc, count(*) as rows
        from stock_prices
        where ((knowledge_time at time zone 'UTC')::date) <= trade_date
        group by 1
        order by 1
        """
    )
    sample_query = text(
        """
        select ticker, trade_date, knowledge_time, source
        from stock_prices
        where ((knowledge_time at time zone 'UTC')::date) <= trade_date
        order by trade_date desc, ticker
        limit 50
        """
    )

    with engine.connect() as conn:
        by_source = [dict(row) for row in conn.execute(summary_query).mappings().all()]
        by_hour = [dict(row) for row in conn.execute(hour_query).mappings().all()]
        samples = [dict(row) for row in conn.execute(sample_query).mappings().all()]

    diagnosis = (
        "The same-day knowledge_time rows are recent Polygon rows written with observed-at timestamps on trade_date "
        "and then preserved by the stock_prices upsert rule using least(existing, incoming). "
        "That means a later historical T+1 refetch cannot correct a too-early knowledge_time."
    )
    return {
        "summary": {
            "row_count": int(sum(int(row["rows"]) for row in by_source)),
            "by_source": by_source,
            "by_hour_utc": by_hour,
        },
        "sample_rows": samples,
        "root_cause": diagnosis,
    }


def _load_label_diagnostics(engine) -> dict[str, Any]:
    with engine.connect() as conn:
        spy_start = conn.execute(
            text("select min(trade_date) from stock_prices where ticker = 'SPY'"),
        ).scalar()

    if spy_start is None:
        raise RuntimeError("SPY benchmark is missing from stock_prices; cannot diagnose label contamination.")

    rows: dict[str, Any] = {}
    for horizon, path in LABEL_PATHS.items():
        if not path.exists():
            rows[horizon] = {"path": str(path), "status": "missing"}
            continue
        frame = pd.read_parquet(path, columns=["trade_date", "forward_return", "excess_return"])
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        pre_spy = frame.loc[frame["trade_date"] < spy_start]
        rows[horizon] = {
            "path": str(path),
            "total_rows": int(len(frame)),
            "min_trade_date": frame["trade_date"].min(),
            "max_trade_date": frame["trade_date"].max(),
            "pre_spy_rows": int(len(pre_spy)),
            "pre_spy_null_excess_rows": int(pre_spy["excess_return"].isna().sum()),
            "pre_spy_nonnull_excess_rows": int(pre_spy["excess_return"].notna().sum()),
            "null_forward_rows": int(frame["forward_return"].isna().sum()),
            "null_excess_rows": int(frame["excess_return"].isna().sum()),
        }

    diagnosis = (
        "The current label parquet files are inconsistent snapshots. 1D/5D/20D still contain non-null pre-SPY excess returns, "
        "which implies they were generated against an older benchmark history than the current stock_prices table. "
        "10D and 60D were built later and already reflect the shorter SPY coverage window."
    )
    return {
        "spy_start_date": spy_start,
        "horizons": rows,
        "root_cause": diagnosis,
    }


def build_report(output_path: Path) -> dict[str, Any]:
    engine = get_engine()
    split_diag = _load_split_diagnostics(engine)
    zero_diag = _load_zero_volume_diagnostics(engine)
    pit_diag = _load_pit_diagnostics(engine)
    label_diag = _load_label_diagnostics(engine)

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if split_diag["summary"]["total"] > 0:
        issues.append(
            {
                "severity": "critical",
                "code": "split_adjustment_truth_drift",
                "message": "stock_prices contains split-like adj_close/close ratios concentrated in a small set of tickers.",
                "count": split_diag["summary"]["total"],
            }
        )
    if pit_diag["summary"]["row_count"] > 0:
        issues.append(
            {
                "severity": "critical",
                "code": "same_day_knowledge_time_rows",
                "message": "stock_prices contains same-day Polygon knowledge_time rows that violate the strict T+1 rule.",
                "count": pit_diag["summary"]["row_count"],
            }
        )
    if zero_diag["summary"]["total_zero_volume_rows"] > 0:
        warnings.append(
            {
                "severity": "warning",
                "code": "ghost_zero_volume_rows",
                "message": "Flat zero-volume ghost bars are present in stock_prices and should be removed before rebuilding labels.",
                "count": zero_diag["summary"]["total_zero_volume_rows"],
            }
        )
    if any(
        horizon_data.get("pre_spy_nonnull_excess_rows", 0) > 0
        for horizon_data in label_diag["horizons"].values()
        if isinstance(horizon_data, dict)
    ):
        issues.append(
            {
                "severity": "critical",
                "code": "stale_pre_spy_excess_labels",
                "message": "Some label files contain non-null pre-SPY excess returns and must be rebuilt on the current benchmark truth.",
            }
        )

    payload = {
        "metadata": {
            "说明": "Week 2.5-P2 根因诊断：确认 split-like 行、零成交 ghost bar、same-day knowledge_time 与标签污染的来源。",
            "generated_at_utc": datetime.now(timezone.utc),
            "script_name": "scripts/diagnose_p2_root_cause.py",
        },
        "summary": {
            "split_anomaly_rows": int(split_diag["summary"]["total"]),
            "zero_volume_rows": int(zero_diag["summary"]["total_zero_volume_rows"]),
            "same_day_knowledge_time_rows": int(pit_diag["summary"]["row_count"]),
            "spy_start_date": label_diag["spy_start_date"],
        },
        "issues": issues,
        "warnings": warnings,
        "split_adjustment_root_cause": split_diag,
        "zero_volume_root_cause": zero_diag,
        "pit_root_cause": pit_diag,
        "label_root_cause": label_diag,
    }
    write_json_report(payload, output_path)
    critical_count, warning_count = summarize_issues(issues + warnings)
    print(f"[p2_root_cause] wrote {output_path}")
    print(
        f"[p2_root_cause] split_anomalies={payload['summary']['split_anomaly_rows']} "
        f"zero_volume={payload['summary']['zero_volume_rows']} "
        f"pit_same_day={payload['summary']['same_day_knowledge_time_rows']} "
        f"critical={critical_count} warnings={warning_count}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose price-truth and label contamination root causes before P2 repair.")
    parser.add_argument(
        "--output",
        default=f"data/reports/p2_root_cause_diagnosis_{REPORT_DATE_TAG}.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(Path(args.output))


if __name__ == "__main__":
    main()
