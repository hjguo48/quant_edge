#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from sqlalchemy import bindparam, text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, write_json_report
from src.data.corporate_actions import fetch_corporate_actions
from src.data.db.models import StockPrice
from src.data.db.session import get_session_factory
from src.data.sources.polygon import PolygonDataSource


NEW_YORK_TZ = ZoneInfo("America/New_York")


def _next_business_close_utc(trade_date: date) -> datetime:
    next_day = trade_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    local_close = datetime.combine(next_day, time(16, 0), tzinfo=NEW_YORK_TZ)
    return local_close.astimezone(timezone.utc)


def load_anomaly_tickers(engine) -> list[dict[str, date]]:
    query = text(
        """
        select ticker, min(trade_date) as min_date, max(trade_date) as max_date, count(*) as rows
        from stock_prices
        where close is not null
          and adj_close is not null
          and close <> 0
          and ((adj_close / close) > 2.0 or (adj_close / close) < 0.1)
        group by ticker
        order by rows desc, ticker
        """
    )
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(query).mappings().all()]


def backfill_corporate_actions_for_anomalies(anomaly_windows: list[dict[str, date]]) -> int:
    if not anomaly_windows:
        return 0
    tickers = [row["ticker"] for row in anomaly_windows]
    start_date = min(row["min_date"] for row in anomaly_windows)
    end_date = max(row["max_date"] for row in anomaly_windows)
    frame = fetch_corporate_actions(tickers, start_date, end_date, min_request_interval=0.0)
    return int(len(frame))


def refetch_adjusted_history_for_anomalies(anomaly_windows: list[dict[str, date]]) -> dict[str, int]:
    source = PolygonDataSource(min_request_interval=0.0)
    repaired_counts: dict[str, int] = {}
    for row in anomaly_windows:
        ticker = str(row["ticker"]).upper()
        frame = source.fetch_adjusted_historical(
            [ticker],
            row["min_date"],
            row["max_date"],
            knowledge_time_mode="historical",
        )
        repaired_counts[ticker] = int(len(frame))
        print(
            f"[fix_price_truth] refetched adjusted history for {ticker} "
            f"{row['min_date']} -> {row['max_date']} rows={len(frame)}",
        )
    return repaired_counts


def delete_flat_zero_volume_rows() -> int:
    session_factory = get_session_factory()
    with session_factory() as session:
        try:
            result = session.execute(
                sa.delete(StockPrice).where(
                    sa.func.coalesce(StockPrice.volume, 0) == 0,
                    StockPrice.open == StockPrice.high,
                    StockPrice.high == StockPrice.low,
                    StockPrice.low == StockPrice.close,
                ),
            )
            session.commit()
        except Exception:
            session.rollback()
            raise
    return int(result.rowcount or 0)


def normalize_same_day_knowledge_time() -> int:
    engine = get_engine()
    select_statement = text(
        """
        select ticker, trade_date
        from stock_prices
        where ((knowledge_time at time zone 'UTC')::date) <= trade_date
        """
    )
    update_statement = text(
        """
        update stock_prices
        set knowledge_time = :b_knowledge_time
        where ticker = :b_ticker
          and trade_date = :b_trade_date
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(select_statement).mappings().all()
    if not rows:
        return 0

    updates = [
        {
            "b_ticker": row["ticker"],
            "b_trade_date": row["trade_date"],
            "b_knowledge_time": _next_business_close_utc(row["trade_date"]),
        }
        for row in rows
    ]
    with engine.begin() as conn:
        conn.execute(update_statement, updates)
    return len(updates)


def build_repair_summary(output_path: Path, *, corporate_action_rows: int, adjusted_refetch_rows: dict[str, int], deleted_zero_rows: int, pit_rows_fixed: int) -> dict:
    payload = {
        "metadata": {
            "说明": "Week 2.5-P2 一次性 price truth 修复摘要。",
            "generated_at_utc": datetime.now(timezone.utc),
            "script_name": "scripts/fix_price_truth.py",
        },
        "summary": {
            "corporate_actions_backfilled": int(corporate_action_rows),
            "adjusted_refetch_tickers": len(adjusted_refetch_rows),
            "adjusted_refetch_total_rows": int(sum(adjusted_refetch_rows.values())),
            "flat_zero_volume_rows_deleted": int(deleted_zero_rows),
            "same_day_knowledge_time_rows_fixed": int(pit_rows_fixed),
        },
        "adjusted_refetch_rows_by_ticker": adjusted_refetch_rows,
    }
    write_json_report(payload, output_path)
    print(f"[fix_price_truth] wrote {output_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair stock_prices truth before rebuilding labels.")
    parser.add_argument(
        "--output",
        default=f"data/reports/fix_price_truth_summary_{REPORT_DATE_TAG}.json",
        help="Repair summary JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    engine = get_engine()
    anomaly_windows = load_anomaly_tickers(engine)
    corporate_action_rows = backfill_corporate_actions_for_anomalies(anomaly_windows)
    adjusted_refetch_rows = refetch_adjusted_history_for_anomalies(anomaly_windows)
    deleted_zero_rows = delete_flat_zero_volume_rows()
    pit_rows_fixed = normalize_same_day_knowledge_time()
    build_repair_summary(
        output_path,
        corporate_action_rows=corporate_action_rows,
        adjusted_refetch_rows=adjusted_refetch_rows,
        deleted_zero_rows=deleted_zero_rows,
        pit_rows_fixed=pit_rows_fixed,
    )


if __name__ == "__main__":
    main()
