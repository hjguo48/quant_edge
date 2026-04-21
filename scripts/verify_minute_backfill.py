#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any

import pandas as pd
import sqlalchemy as sa

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from src.data.db.session import get_engine
from src.data.polygon_minute import PolygonMinuteClient
from src.features.intraday import aggregate_minute_to_daily


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample-verify minute flat-file backfill against REST minute aggregates.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-output", default="data/reports/minute_backfill_verification_20260417.json")
    return parser.parse_args(argv)


def parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


def bp_delta(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return None
    midpoint = (abs(float(left)) + abs(float(right))) / 2.0
    if midpoint == 0:
        return 0.0
    return abs(float(left) - float(right)) / midpoint * 10_000.0


def load_sample_candidates(start_date: date, end_date: date) -> list[tuple[str, date]]:
    query = sa.text(
        """
        select distinct ticker, trade_date
        from stock_minute_aggs
        where trade_date >= :start_date
          and trade_date <= :end_date
        order by trade_date, ticker
        """,
    )
    with get_engine().connect() as conn:
        frame = pd.read_sql_query(query, conn, params={"start_date": start_date, "end_date": end_date})
    if frame.empty:
        return []
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    return [(str(row["ticker"]).upper(), row["trade_date"]) for row in frame.to_dict(orient="records")]


def load_db_minute_daily(ticker: str, trade_day: date) -> dict[str, float] | None:
    query = sa.text(
        """
        select ticker, trade_date, minute_ts, open, high, low, close, volume, vwap, transactions
        from stock_minute_aggs
        where ticker = :ticker
          and trade_date = :trade_date
        order by minute_ts
        """,
    )
    with get_engine().connect() as conn:
        frame = pd.read_sql_query(query, conn, params={"ticker": ticker, "trade_date": trade_day}, parse_dates=["minute_ts"])
    if frame.empty:
        return None
    daily = aggregate_minute_to_daily(frame).iloc[0]
    return {field: float(daily[field]) for field in ("open", "high", "low", "close", "volume")}


def load_rest_minute_daily(ticker: str, trade_day: date) -> dict[str, float] | None:
    client = PolygonMinuteClient(min_request_interval=0.0)
    frame = client.get_minute_aggs(ticker, trade_day, trade_day)
    if frame.empty:
        return None
    daily = aggregate_minute_to_daily(frame).iloc[0]
    return {field: float(daily[field]) for field in ("open", "high", "low", "close", "volume")}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    candidates = load_sample_candidates(start_date, end_date)
    if not candidates:
        raise RuntimeError("No stock_minute_aggs samples are available in the requested window.")

    rng = random.Random(args.seed)
    sample = rng.sample(candidates, k=min(args.sample_size, len(candidates)))
    rows: list[dict[str, Any]] = []
    for ticker, trade_day in sample:
        db_daily = load_db_minute_daily(ticker, trade_day)
        rest_daily = load_rest_minute_daily(ticker, trade_day)
        if db_daily is None or rest_daily is None:
            continue
        deltas = {field: bp_delta(db_daily[field], rest_daily[field]) for field in ("open", "high", "low", "close", "volume")}
        rows.append(
            {
                "ticker": ticker,
                "trade_date": trade_day.isoformat(),
                "db_flat_minute_agg": db_daily,
                "rest_minute_agg": rest_daily,
                "delta_bp": deltas,
            },
        )

    summary = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_name": "verify_minute_backfill.py",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "requested_sample_size": args.sample_size,
            "actual_sample_size": len(rows),
        },
        "samples": rows,
        "thresholds": {
            "ohl_bp_max": 1.0,
            "close_bp_max": 10.0,
            "volume": "warning_only",
        },
    }
    output_path = REPO_ROOT / args.report_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
