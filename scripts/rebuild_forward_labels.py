#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, write_json_report
from src.data.db.pit import get_prices_pit
from src.labels.forward_returns import compute_forward_returns
from scripts.run_ic_screening import write_parquet_atomic


DEFAULT_HORIZONS = (1, 5, 10, 20, 60)


def load_rebuild_price_panel(*, horizons: Iterable[int], as_of: date, benchmark_ticker: str) -> tuple[pd.DataFrame, date]:
    engine = get_engine()
    with engine.connect() as conn:
        tickers = conn.execute(text("select distinct ticker from stock_prices order by ticker")).scalars().all()
        latest_pit_trade_date = conn.execute(
            text(
                """
                select max(trade_date) filter (where knowledge_time <= :as_of) as latest_pit_trade_date
                from stock_prices
                """
            ),
            {"as_of": datetime.combine(as_of, time.max, tzinfo=timezone.utc)},
        ).scalar()

    if latest_pit_trade_date is None:
        raise RuntimeError("No PIT-visible trade_date found in stock_prices.")

    start_date = date(2015, 1, 2)
    prices = get_prices_pit(
        tickers=tickers,
        start_date=start_date,
        end_date=latest_pit_trade_date,
        as_of=datetime.combine(as_of, time.max, tzinfo=timezone.utc),
    )
    if prices.empty:
        raise RuntimeError("No PIT prices available to rebuild forward labels.")

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    return prices, latest_pit_trade_date


def annotate_label_validity(labels: pd.DataFrame, *, benchmark_start: date) -> pd.DataFrame:
    annotated = labels.copy()
    annotated["trade_date"] = pd.to_datetime(annotated["trade_date"]).dt.date
    annotated["forward_return"] = pd.to_numeric(annotated["forward_return"], errors="coerce")
    annotated["excess_return"] = pd.to_numeric(annotated["excess_return"], errors="coerce")

    invalid_reason = pd.Series(pd.NA, index=annotated.index, dtype="object")
    benchmark_missing_mask = annotated["trade_date"] < benchmark_start
    forward_missing_mask = annotated["forward_return"].isna()
    excess_missing_mask = annotated["excess_return"].isna()

    invalid_reason.loc[benchmark_missing_mask & excess_missing_mask] = "benchmark_unavailable_pre_spy"
    invalid_reason.loc[~benchmark_missing_mask & forward_missing_mask] = "insufficient_forward_window"
    invalid_reason.loc[
        ~benchmark_missing_mask & ~forward_missing_mask & excess_missing_mask
    ] = "benchmark_unavailable"

    annotated["is_valid_excess"] = annotated["excess_return"].notna()
    annotated["invalid_reason"] = invalid_reason
    return annotated


def rebuild_labels_for_horizon(
    *,
    prices: pd.DataFrame,
    horizon: int,
    benchmark_ticker: str,
    benchmark_start: date,
    output_path: Path,
) -> dict[str, object]:
    labels = compute_forward_returns(prices_df=prices, horizons=[horizon], benchmark_ticker=benchmark_ticker)
    labels = labels.loc[labels["horizon"].astype(int) == int(horizon)].copy()
    labels = labels.loc[labels["ticker"].astype(str).str.upper() != benchmark_ticker.upper()].copy()
    annotated = annotate_label_validity(labels, benchmark_start=benchmark_start)
    annotated.sort_values(["trade_date", "ticker"], inplace=True)
    annotated.reset_index(drop=True, inplace=True)
    write_parquet_atomic(annotated, output_path)
    reason_counts = {
        str(key): int(value)
        for key, value in annotated["invalid_reason"].value_counts(dropna=True).to_dict().items()
    }
    print(
        f"[rebuild_forward_labels] horizon={horizon} rows={len(annotated)} "
        f"valid_excess={int(annotated['is_valid_excess'].sum())} output={output_path}",
    )
    return {
        "path": str(output_path),
        "rows": int(len(annotated)),
        "min_trade_date": annotated["trade_date"].min(),
        "max_trade_date": annotated["trade_date"].max(),
        "null_forward_return_rows": int(annotated["forward_return"].isna().sum()),
        "null_excess_return_rows": int(annotated["excess_return"].isna().sum()),
        "invalid_reason_counts": reason_counts,
    }


def build_manifest(output_path: Path, *, benchmark_start: date, latest_pit_trade_date: date, horizon_summaries: dict[str, object]) -> dict[str, object]:
    payload = {
        "metadata": {
            "说明": "Week 2.5-P2 labels rebuild summary on repaired stock_prices truth.",
            "generated_at_utc": datetime.now(timezone.utc),
            "script_name": "scripts/rebuild_forward_labels.py",
        },
        "summary": {
            "benchmark_start_date": benchmark_start,
            "latest_pit_trade_date": latest_pit_trade_date,
            "horizons_rebuilt": list(horizon_summaries.keys()),
        },
        "horizons": horizon_summaries,
    }
    write_json_report(payload, output_path)
    print(f"[rebuild_forward_labels] wrote {output_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild forward label parquet files on repaired stock_prices truth.")
    parser.add_argument(
        "--as-of",
        default=date.today().isoformat(),
        help="As-of date for PIT price visibility.",
    )
    parser.add_argument(
        "--benchmark-ticker",
        default="SPY",
        help="Benchmark ticker used for excess returns.",
    )
    parser.add_argument(
        "--horizons",
        default="1,5,10,20,60",
        help="Comma-separated horizons to rebuild.",
    )
    parser.add_argument(
        "--manifest-output",
        default=f"data/reports/rebuild_forward_labels_{REPORT_DATE_TAG}.json",
        help="Summary manifest JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    as_of = date.fromisoformat(str(args.as_of))
    horizons = tuple(int(token) for token in str(args.horizons).split(",") if token.strip())
    benchmark_ticker = str(args.benchmark_ticker).upper()
    prices, latest_pit_trade_date = load_rebuild_price_panel(
        horizons=horizons,
        as_of=as_of,
        benchmark_ticker=benchmark_ticker,
    )
    benchmark_prices = prices.loc[prices["ticker"] == benchmark_ticker, "trade_date"]
    if benchmark_prices.empty:
        raise RuntimeError(f"Benchmark ticker {benchmark_ticker} missing from PIT price panel.")
    benchmark_start = benchmark_prices.min()

    horizon_summaries: dict[str, object] = {}
    for horizon in horizons:
        output_path = Path(f"data/labels/forward_returns_{horizon}d.parquet")
        horizon_summaries[f"{horizon}D"] = rebuild_labels_for_horizon(
            prices=prices,
            horizon=horizon,
            benchmark_ticker=benchmark_ticker,
            benchmark_start=benchmark_start,
            output_path=output_path,
        )

    build_manifest(
        Path(args.manifest_output),
        benchmark_start=benchmark_start,
        latest_pit_trade_date=latest_pit_trade_date,
        horizon_summaries=horizon_summaries,
    )


if __name__ == "__main__":
    main()
