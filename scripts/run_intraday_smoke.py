#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import uuid

import exchange_calendars as xcals
from loguru import logger
import pandas as pd
import sqlalchemy as sa

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic, write_parquet_atomic
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_engine
from src.data.polygon_minute import EASTERN, MINUTE_COLUMNS, PolygonMinuteClient
from src.features.intraday import aggregate_minute_to_daily, compute_intraday_features
from src.features.pipeline import FeaturePipeline, prepare_feature_export_frame

SMOKE_TICKERS = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "JPM",
    "XOM",
    "UNH",
)
XNYS = xcals.get_calendar("XNYS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 3.0 smoke run for Polygon minute aggregates -> intraday features -> labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", default="2026-01-02")
    parser.add_argument("--end-date", default="2026-01-08")
    parser.add_argument("--tickers", nargs="*", default=list(SMOKE_TICKERS))
    parser.add_argument(
        "--feature-output",
        default="data/features/intraday_smoke_features_20260417.parquet",
    )
    parser.add_argument(
        "--label-output",
        default="data/labels/intraday_smoke_labels_20260417.parquet",
    )
    parser.add_argument(
        "--report-output",
        default="data/reports/week3_smoke_report_20260417.json",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if pd.isna(value):
        return None
    return value


def load_minute_slice(*, tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    query = sa.text(
        """
        select
            ticker,
            trade_date,
            minute_ts,
            open,
            high,
            low,
            close,
            volume,
            vwap,
            transactions,
            event_time,
            knowledge_time,
            batch_id
        from stock_minute_aggs
        where ticker = any(:tickers)
          and trade_date >= :start_date
          and trade_date <= :end_date
        order by ticker, minute_ts
        """,
    )
    with get_engine().connect() as conn:
        frame = pd.read_sql_query(
            query,
            conn,
            params={
                "tickers": list(tickers),
                "start_date": start_date,
                "end_date": end_date,
            },
            parse_dates=["minute_ts", "event_time", "knowledge_time"],
        )
    return frame


def load_daily_prices(*, tickers: tuple[str, ...], start_date: date, end_date: date) -> pd.DataFrame:
    lookback_start = start_date - pd.offsets.BDay(5)
    return get_prices_pit(
        tickers=tickers,
        start_date=pd.Timestamp(lookback_start).date(),
        end_date=end_date,
        as_of=datetime.now(timezone.utc),
    )


def build_smoke_labels(minute_frame: pd.DataFrame) -> pd.DataFrame:
    minute_daily = aggregate_minute_to_daily(minute_frame)
    if minute_daily.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "label_name", "label_value"])
    minute_daily.sort_values(["ticker", "trade_date"], inplace=True)
    minute_daily["next_open"] = minute_daily.groupby("ticker")["open"].shift(-1)
    minute_daily["next_close"] = minute_daily.groupby("ticker")["close"].shift(-1)
    minute_daily["next_open_ret_1d"] = (minute_daily["next_open"] - minute_daily["close"]) / minute_daily["close"]
    minute_daily["next_close_ret_1d"] = (minute_daily["next_close"] - minute_daily["close"]) / minute_daily["close"]

    rows: list[dict[str, object]] = []
    for row in minute_daily.itertuples(index=False):
        for label_name in ("next_open_ret_1d", "next_close_ret_1d"):
            label_value = getattr(row, label_name)
            if pd.isna(label_value):
                continue
            rows.append(
                {
                    "ticker": row.ticker,
                    "trade_date": row.trade_date,
                    "label_name": label_name,
                    "label_value": float(label_value),
                },
            )
    return pd.DataFrame(rows, columns=["ticker", "trade_date", "label_name", "label_value"])


def validate_schema() -> dict[str, Any]:
    engine = get_engine()
    inspector = sa.inspect(engine)
    columns = {column["name"]: column for column in inspector.get_columns("stock_minute_aggs")}
    pk = inspector.get_pk_constraint("stock_minute_aggs")
    indexes = inspector.get_indexes("stock_minute_aggs")
    with engine.connect() as conn:
        hypertable_count = int(
            conn.execute(
                sa.text(
                    """
                    select count(*)
                    from timescaledb_information.hypertables
                    where hypertable_schema = 'public'
                      and hypertable_name = 'stock_minute_aggs'
                    """,
                ),
            ).scalar_one(),
        )

    expected_columns = {
        "id",
        "ticker",
        "trade_date",
        "minute_ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
        "event_time",
        "knowledge_time",
        "batch_id",
    }
    index_names = {index["name"] for index in indexes}
    schema_pass = (
        expected_columns <= set(columns)
        and set(pk.get("constrained_columns") or []) == {"ticker", "minute_ts"}
        and {"idx_stock_minute_aggs_trade_date", "idx_stock_minute_aggs_knowledge_time"} <= index_names
        and hypertable_count == 1
    )
    return {
        "pass": bool(schema_pass),
        "expected_columns": sorted(expected_columns),
        "actual_columns": sorted(columns),
        "primary_key": pk.get("constrained_columns") or [],
        "indexes": sorted(index_names),
        "is_hypertable": hypertable_count == 1,
    }


def validate_timezones(minute_frame: pd.DataFrame) -> dict[str, Any]:
    if minute_frame.empty:
        return {"pass": False, "checked_rows": 0, "sample": [], "reason": "no minute rows"}
    sample = minute_frame.sample(n=min(10, len(minute_frame)), random_state=42).copy()
    sample["minute_ts"] = pd.to_datetime(sample["minute_ts"], utc=True)
    sample_rows: list[dict[str, object]] = []
    pass_count = 0
    for row in sample.itertuples(index=False):
        minute_ts_utc = pd.Timestamp(row.minute_ts)
        if minute_ts_utc.tzinfo is None:
            minute_ts_utc = minute_ts_utc.tz_localize("UTC")
        else:
            minute_ts_utc = minute_ts_utc.tz_convert("UTC")
        minute_ts_et = minute_ts_utc.tz_convert(EASTERN)
        local_time = minute_ts_et.time()
        time_ok = local_time >= datetime.strptime("09:30", "%H:%M").time() and local_time <= datetime.strptime("16:00", "%H:%M").time()
        trade_date_ok = minute_ts_et.date() == row.trade_date
        trading_minute_ok = XNYS.is_trading_minute(minute_ts_utc)
        passed = bool(time_ok and trade_date_ok and trading_minute_ok)
        pass_count += int(passed)
        sample_rows.append(
            {
                "ticker": row.ticker,
                "trade_date": row.trade_date,
                "minute_ts_utc": minute_ts_utc.isoformat(),
                "minute_ts_et": minute_ts_et.isoformat(),
                "time_ok": bool(time_ok),
                "trade_date_ok": bool(trade_date_ok),
                "trading_minute_ok": bool(trading_minute_ok),
                "pass": passed,
            },
        )
    return {
        "pass": pass_count == len(sample_rows),
        "checked_rows": len(sample_rows),
        "pass_rate": float(pass_count / len(sample_rows)) if sample_rows else 0.0,
        "sample": sample_rows,
    }


def validate_trading_days(*, minute_frame: pd.DataFrame, start_date: date, end_date: date) -> dict[str, Any]:
    actual_dates = sorted(pd.to_datetime(minute_frame["trade_date"]).dt.date.unique().tolist())
    expected_sessions = [ts.date() for ts in XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    missing = sorted(set(expected_sessions) - set(actual_dates))
    extra = sorted(set(actual_dates) - set(expected_sessions))
    return {
        "pass": not missing and not extra,
        "actual_trade_dates": actual_dates,
        "expected_trade_dates": expected_sessions,
        "missing_trade_dates": missing,
        "unexpected_trade_dates": extra,
    }


def validate_minute_to_day_consistency(minute_frame: pd.DataFrame, daily_prices: pd.DataFrame) -> dict[str, Any]:
    minute_daily = aggregate_minute_to_daily(minute_frame)
    daily_reference = daily_prices.loc[
        :,
        ["ticker", "trade_date", "open", "high", "low", "close", "volume"],
    ].copy()
    daily_reference["ticker"] = daily_reference["ticker"].astype(str).str.upper()
    daily_reference["trade_date"] = pd.to_datetime(daily_reference["trade_date"]).dt.date

    merged = minute_daily.merge(
        daily_reference,
        on=["ticker", "trade_date"],
        how="inner",
        suffixes=("_minute", "_daily"),
    )
    field_thresholds = {
        "open": 1e-4,
        "high": 1e-4,
        "low": 1e-4,
        "close": 1e-4,
        "volume": 5e-3,
    }
    summary: dict[str, Any] = {}
    pass_flag = True
    for field, threshold in field_thresholds.items():
        minute_values = pd.to_numeric(merged[f"{field}_minute"], errors="coerce")
        daily_values = pd.to_numeric(merged[f"{field}_daily"], errors="coerce")
        denominator = daily_values.abs().replace(0, pd.NA)
        rel_diff = ((minute_values - daily_values).abs() / denominator).fillna(0.0)
        max_diff = float(rel_diff.max()) if not rel_diff.empty else 0.0
        pass_flag &= max_diff < threshold
        summary[field] = {
            "max_abs_rel_diff": max_diff,
            "threshold": threshold,
            "pass": bool(max_diff < threshold),
        }
    return {
        "pass": bool(pass_flag),
        "row_count": int(len(merged)),
        "fields": summary,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in args.tickers if ticker))
    minute_client = PolygonMinuteClient(min_request_interval=0.15)

    logger.info(
        "running intraday smoke for {} tickers between {} and {}",
        len(tickers),
        start_date,
        end_date,
    )

    frames = [minute_client.get_minute_aggs(ticker, start_date, end_date) for ticker in tickers]
    minute_frame = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(
        not frame.empty for frame in frames
    ) else pd.DataFrame(columns=MINUTE_COLUMNS)
    if minute_frame.empty:
        raise RuntimeError("Polygon minute smoke returned no rows.")

    ingest_batch_id = str(uuid.uuid4())
    minute_frame["batch_id"] = ingest_batch_id
    rows_saved = minute_client.persist_minute_aggs(minute_frame)
    persisted = load_minute_slice(tickers=tickers, start_date=start_date, end_date=end_date)

    daily_prices = load_daily_prices(tickers=tickers, start_date=start_date, end_date=end_date)
    features = compute_intraday_features(minute_df=persisted, daily_prices_df=daily_prices)
    features = prepare_feature_export_frame(features)
    feature_batch_id = str(uuid.uuid4())
    FeaturePipeline().save_to_store(features, batch_id=feature_batch_id)
    feature_output_path = REPO_ROOT / args.feature_output
    write_parquet_atomic(features, feature_output_path)

    labels = build_smoke_labels(persisted)
    label_output_path = REPO_ROOT / args.label_output
    write_parquet_atomic(labels, label_output_path)

    report = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_name": "run_intraday_smoke.py",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "tickers": list(tickers),
        },
        "ingest": {
            "rows_fetched": int(len(minute_frame)),
            "rows_saved": int(rows_saved),
            "rows_loaded": int(len(persisted)),
            "rows_by_ticker": {ticker: int(count) for ticker, count in persisted.groupby("ticker").size().items()},
            "rows_by_trade_date": {
                trade_date.isoformat(): int(count)
                for trade_date, count in persisted.groupby("trade_date").size().items()
            },
            "batch_id": ingest_batch_id,
        },
        "features": {
            "rows": int(len(features)),
            "rows_by_feature": {
                feature_name: int(count)
                for feature_name, count in features.groupby("feature_name").size().items()
            },
            "feature_store_batch_id": feature_batch_id,
            "parquet_path": str(feature_output_path),
        },
        "labels": {
            "rows": int(len(labels)),
            "rows_by_label": {
                label_name: int(count)
                for label_name, count in labels.groupby("label_name").size().items()
            },
            "parquet_path": str(label_output_path),
        },
        "verification": {
            "schema": validate_schema(),
            "timezone": validate_timezones(persisted),
            "trading_days": validate_trading_days(
                minute_frame=persisted,
                start_date=start_date,
                end_date=end_date,
            ),
            "minute_to_day_consistency": validate_minute_to_day_consistency(
                persisted,
                daily_prices.loc[
                    (pd.to_datetime(daily_prices["trade_date"]).dt.date >= start_date)
                    & (pd.to_datetime(daily_prices["trade_date"]).dt.date <= end_date)
                ].copy(),
            ),
        },
    }
    verification = report["verification"]
    report["pass"] = bool(
        verification["schema"]["pass"]
        and verification["timezone"]["pass"]
        and verification["trading_days"]["pass"]
        and verification["minute_to_day_consistency"]["pass"]
    )

    report_path = REPO_ROOT / args.report_output
    write_json_atomic(report_path, json_safe(report))
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
