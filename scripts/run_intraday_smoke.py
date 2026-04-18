#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
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
from src.data.db.models import PriceReconciliationEvent
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_engine, get_session_factory
from src.data.polygon_minute import (
    EASTERN,
    MINUTE_COLUMNS,
    PolygonMinuteClient,
    REGULAR_SESSION_END,
    REGULAR_SESSION_START,
)
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
BLOCKER_THRESHOLD_BP = 10.0
WARNING_CLOSE_THRESHOLD_BP = 10.0
WARNING_VOLUME_THRESHOLD_BP = 500.0
EXPECTED_BARS_PER_SESSION = 391
ALLOWED_MISSING_BARS = 3


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


def bp_delta(left: float | int | None, right: float | int | None) -> float | None:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return None
    midpoint = (abs(float(left)) + abs(float(right))) / 2.0
    if midpoint == 0:
        return 0.0
    return abs(float(left) - float(right)) / midpoint * 10_000.0


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
    event_columns = {column["name"]: column for column in inspector.get_columns("price_reconciliation_events")}
    pk = inspector.get_pk_constraint("stock_minute_aggs")
    indexes = inspector.get_indexes("stock_minute_aggs")
    event_indexes = inspector.get_indexes("price_reconciliation_events")
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
    expected_event_columns = {
        "id",
        "ticker",
        "trade_date",
        "field",
        "stock_prices_value",
        "minute_agg_value",
        "delta_bp",
        "severity",
        "detected_at",
        "batch_id",
    }
    index_names = {index["name"] for index in indexes}
    event_index_names = {index["name"] for index in event_indexes}
    schema_pass = (
        expected_columns <= set(columns)
        and expected_event_columns <= set(event_columns)
        and set(pk.get("constrained_columns") or []) == {"ticker", "minute_ts"}
        and {"idx_stock_minute_aggs_trade_date", "idx_stock_minute_aggs_knowledge_time"} <= index_names
        and {
            "idx_price_reconciliation_events_trade_date",
            "idx_price_reconciliation_events_severity",
            "idx_price_reconciliation_events_batch_id",
        } <= event_index_names
        and hypertable_count == 1
    )
    return {
        "pass": bool(schema_pass),
        "expected_columns": sorted(expected_columns),
        "actual_columns": sorted(columns),
        "expected_event_columns": sorted(expected_event_columns),
        "actual_event_columns": sorted(event_columns),
        "primary_key": pk.get("constrained_columns") or [],
        "indexes": sorted(index_names),
        "event_indexes": sorted(event_index_names),
        "is_hypertable": hypertable_count == 1,
    }


def validate_timezones(minute_frame: pd.DataFrame) -> dict[str, Any]:
    if minute_frame.empty:
        return {"pass": False, "checked_rows": 0, "sample": [], "reason": "no minute rows"}
    frame = minute_frame.copy()
    frame["minute_ts"] = pd.to_datetime(frame["minute_ts"], utc=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["minute_ts_et"] = frame["minute_ts"].dt.tz_convert(EASTERN)
    frame["trade_date_ok"] = frame["minute_ts_et"].dt.date == frame["trade_date"]
    frame["session_date_ok"] = frame["minute_ts_et"].dt.date.map(lambda d: bool(XNYS.is_session(pd.Timestamp(d))))
    session_open_map: dict[date, pd.Timestamp] = {}
    session_close_map: dict[date, pd.Timestamp] = {}
    for trade_day in frame["trade_date"].dropna().unique():
        session_label = pd.Timestamp(trade_day)
        if XNYS.is_session(session_label):
            session_open_map[trade_day] = pd.Timestamp(XNYS.session_open(session_label))
            session_close_map[trade_day] = pd.Timestamp(XNYS.session_close(session_label))
        else:
            session_open_map[trade_day] = pd.NaT
            session_close_map[trade_day] = pd.NaT
    session_open = frame["trade_date"].map(session_open_map)
    session_close = frame["trade_date"].map(session_close_map)
    frame["session_time_ok"] = (
        session_open.notna()
        & (frame["minute_ts"] >= session_open)
        & (frame["minute_ts"] < session_close)
    )
    frame["pass"] = frame["session_time_ok"] & frame["trade_date_ok"] & frame["session_date_ok"]
    failures = frame.loc[~frame["pass"]].copy()
    sample_source = failures if not failures.empty else frame.sample(n=min(10, len(frame)), random_state=42)
    sample_rows: list[dict[str, object]] = []
    for record in sample_source.head(10).to_dict(orient="records"):
        sample_rows.append(
            {
                "ticker": record["ticker"],
                "trade_date": record["trade_date"],
                "minute_ts_utc": pd.Timestamp(record["minute_ts"]).isoformat(),
                "minute_ts_et": pd.Timestamp(record["minute_ts_et"]).isoformat(),
                "time_ok": bool(record["session_time_ok"]),
                "trade_date_ok": bool(record["trade_date_ok"]),
                "session_date_ok": bool(record["session_date_ok"]),
                "pass": bool(record["pass"]),
            },
        )
    return {
        "pass": bool(frame["pass"].all()),
        "checked_rows": int(len(frame)),
        "failure_count": int((~frame["pass"]).sum()),
        "pass_rate": float(frame["pass"].mean()) if len(frame) else 0.0,
        "sample": sample_rows,
    }


def validate_trading_days(
    *,
    minute_frame: pd.DataFrame,
    start_date: date,
    end_date: date,
    expected_tickers: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    actual_dates = sorted(pd.to_datetime(minute_frame["trade_date"]).dt.date.unique().tolist())
    expected_sessions = [ts.date() for ts in XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))]
    missing = sorted(set(expected_sessions) - set(actual_dates))
    extra = sorted(set(actual_dates) - set(expected_sessions))
    frame = minute_frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    tickers = (
        sorted(frame["ticker"].unique().tolist())
        if expected_tickers is None
        else sorted({str(ticker).upper() for ticker in expected_tickers})
    )
    expected_dates_set = set(expected_sessions)
    per_ticker_missing: dict[str, list[date]] = {}
    for ticker in tickers:
        ticker_dates = set(frame.loc[frame["ticker"] == ticker, "trade_date"].tolist())
        ticker_missing = sorted(expected_dates_set - ticker_dates)
        if ticker_missing:
            per_ticker_missing[ticker] = ticker_missing
    return {
        "pass": not missing and not extra and not per_ticker_missing,
        "actual_trade_dates": actual_dates,
        "expected_trade_dates": expected_sessions,
        "missing_trade_dates": missing,
        "unexpected_trade_dates": extra,
        "per_ticker_missing": per_ticker_missing,
    }


def validate_minute_internal_consistency(minute_frame: pd.DataFrame) -> dict[str, Any]:
    if minute_frame.empty:
        return {
            "pass": False,
            "checks": {},
            "reason": "no minute rows",
        }

    frame = minute_frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["minute_ts"] = pd.to_datetime(frame["minute_ts"], utc=True)
    frame["minute_ts_et"] = frame["minute_ts"].dt.tz_convert(EASTERN)

    duplicate_rows = frame.loc[frame.duplicated(["ticker", "minute_ts"], keep=False)].copy()
    overlap_check = {
        "pass": duplicate_rows.empty,
        "duplicate_row_count": int(len(duplicate_rows)),
        "sample": duplicate_rows.loc[:, ["ticker", "trade_date", "minute_ts"]].head(10).to_dict(orient="records"),
    }

    monotonic_failures: list[dict[str, object]] = []
    for (ticker, trade_day), group in frame.groupby(["ticker", "trade_date"], sort=False):
        deltas = group["minute_ts"].diff().dropna()
        if (deltas <= pd.Timedelta(0)).any():
            monotonic_failures.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_day,
                    "row_count": int(len(group)),
                },
            )
    monotonic_check = {
        "pass": not monotonic_failures,
        "failure_count": int(len(monotonic_failures)),
        "sample": monotonic_failures[:10],
    }

    gap_failures: list[dict[str, object]] = []
    for (ticker, trade_day), group in frame.groupby(["ticker", "trade_date"]):
        session_label = pd.Timestamp(trade_day)
        session_open_utc = pd.Timestamp(XNYS.session_open(session_label))
        session_close_utc = pd.Timestamp(XNYS.session_close(session_label))
        expected_index = pd.date_range(
            session_open_utc,
            session_close_utc - pd.Timedelta(minutes=1),
            freq="min",
        ).tz_convert(EASTERN)
        expected_minute_set = set(expected_index.time)
        observed_minutes = set(group["minute_ts_et"].dt.time.tolist())
        missing_minutes = sorted(expected_minute_set - observed_minutes)
        missing_count = len(missing_minutes)
        if missing_count > ALLOWED_MISSING_BARS:
            gap_failures.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_day,
                    "observed_bars": int(len(group)),
                    "expected_bars": int(len(expected_index)),
                    "missing_bars": int(missing_count),
                    "missing_sample": [minute.isoformat() for minute in missing_minutes[:10]],
                },
            )
    gap_check = {
        "pass": not gap_failures,
        "tolerance_missing_bars": ALLOWED_MISSING_BARS,
        "expected_bars_per_session": "dynamic_by_session",
        "failure_count": int(len(gap_failures)),
        "sample": gap_failures[:10],
    }

    ohlc_invalid = frame.loc[
        (pd.to_numeric(frame["high"], errors="coerce") < pd.concat([pd.to_numeric(frame["open"], errors="coerce"), pd.to_numeric(frame["close"], errors="coerce")], axis=1).max(axis=1))
        | (pd.to_numeric(frame["low"], errors="coerce") > pd.concat([pd.to_numeric(frame["open"], errors="coerce"), pd.to_numeric(frame["close"], errors="coerce")], axis=1).min(axis=1))
    ].copy()
    ohlc_check = {
        "pass": ohlc_invalid.empty,
        "failure_count": int(len(ohlc_invalid)),
        "sample": ohlc_invalid.loc[:, ["ticker", "trade_date", "minute_ts", "open", "high", "low", "close"]]
        .head(10)
        .to_dict(orient="records"),
    }

    activity_invalid = frame.loc[
        (pd.to_numeric(frame["volume"], errors="coerce") < 0)
        | (pd.to_numeric(frame["transactions"], errors="coerce") < 0)
    ].copy()
    activity_check = {
        "pass": activity_invalid.empty,
        "failure_count": int(len(activity_invalid)),
        "sample": activity_invalid.loc[:, ["ticker", "trade_date", "minute_ts", "volume", "transactions"]]
        .head(10)
        .to_dict(orient="records"),
    }

    checks = {
        "gap_free": gap_check,
        "no_overlap": overlap_check,
        "strictly_increasing": monotonic_check,
        "ohlc_internal_consistency": ohlc_check,
        "non_negative_activity": activity_check,
    }
    return {
        "pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
    }


def validate_corporate_action_alignment(
    minute_frame: pd.DataFrame,
    daily_prices: pd.DataFrame,
) -> dict[str, Any]:
    minute_daily = aggregate_minute_to_daily(minute_frame)
    if minute_daily.empty:
        return {"pass": False, "reason": "no minute rows"}
    minute_daily["trade_date"] = pd.to_datetime(minute_daily["trade_date"]).dt.date
    daily = daily_prices.loc[:, ["ticker", "trade_date", "close"]].copy()
    daily["ticker"] = daily["ticker"].astype(str).str.upper()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce").astype(float)
    daily.sort_values(["ticker", "trade_date"], inplace=True)
    daily["prev_close"] = daily.groupby("ticker")["close"].shift(1)

    merged = minute_daily.merge(daily.loc[:, ["ticker", "trade_date", "prev_close"]], on=["ticker", "trade_date"], how="left")
    if merged.empty:
        return {"pass": False, "reason": "minute/daily merge empty"}

    split_actions: pd.DataFrame
    with get_engine().connect() as conn:
        split_actions = pd.read_sql_query(
            sa.text(
                """
                select ticker, ex_date, ratio, action_type
                from corporate_actions
                where action_type ilike '%split%'
                  and ticker = any(:tickers)
                  and ex_date between :start_date and :end_date
                order by ticker, ex_date
                """,
            ),
            conn,
            params={
                "tickers": merged["ticker"].dropna().astype(str).str.upper().unique().tolist(),
                "start_date": min(merged["trade_date"]) - timedelta(days=2),
                "end_date": max(merged["trade_date"]),
            },
        )

    if split_actions.empty:
        return {"pass": True, "checked_rows": 0, "anomalies": []}

    split_actions["ticker"] = split_actions["ticker"].astype(str).str.upper()
    split_actions["ex_date"] = pd.to_datetime(split_actions["ex_date"]).dt.date
    anomalies: list[dict[str, Any]] = []
    for action in split_actions.itertuples(index=False):
        row = merged.loc[(merged["ticker"] == action.ticker) & (merged["trade_date"] == action.ex_date)]
        if row.empty:
            continue
        row = row.iloc[0]
        if pd.isna(row["prev_close"]) or pd.isna(row["open"]) or pd.isna(action.ratio):
            continue
        observed_ratio = float(row["open"]) / float(row["prev_close"]) if float(row["prev_close"]) != 0 else None
        if observed_ratio is None:
            continue
        ratio = float(action.ratio)
        candidate_ratios = [ratio]
        if ratio != 0:
            candidate_ratios.append(1.0 / ratio)
        best_expected = min(candidate_ratios, key=lambda candidate: abs(observed_ratio - candidate))
        relative_error = abs(observed_ratio - best_expected) / abs(best_expected) if best_expected != 0 else 0.0
        if relative_error > 0.10:
            anomalies.append(
                {
                    "ticker": action.ticker,
                    "trade_date": action.ex_date,
                    "observed_ratio": observed_ratio,
                    "expected_ratio": best_expected,
                    "declared_ratio": ratio,
                    "relative_error": relative_error,
                },
            )
    return {
        "pass": not anomalies,
        "checked_rows": int(len(split_actions)),
        "anomalies": anomalies[:20],
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
    if merged.empty:
        return {
            "pass": False,
            "row_count": 0,
            "fields": {},
            "warning_events": [],
            "warning_event_count": 0,
            "concentrated_warning_days": [],
            "reason": "minute/daily merge empty",
        }
    field_specs = {
        "open": {"threshold_bp": BLOCKER_THRESHOLD_BP, "severity": "blocker"},
        "high": {"threshold_bp": BLOCKER_THRESHOLD_BP, "severity": "blocker"},
        "low": {"threshold_bp": BLOCKER_THRESHOLD_BP, "severity": "blocker"},
        "close": {"threshold_bp": WARNING_CLOSE_THRESHOLD_BP, "severity": "warning"},
        "volume": {"threshold_bp": WARNING_VOLUME_THRESHOLD_BP, "severity": "warning"},
    }
    summary: dict[str, Any] = {}
    blocker_pass = True
    warning_events: list[dict[str, Any]] = []
    for field, spec in field_specs.items():
        minute_values = pd.to_numeric(merged[f"{field}_minute"], errors="coerce").astype(float)
        daily_values = pd.to_numeric(merged[f"{field}_daily"], errors="coerce").astype(float)
        rel_diffs = []
        bp_diffs = []
        for minute_value, daily_value in zip(minute_values.tolist(), daily_values.tolist()):
            if pd.isna(minute_value) or pd.isna(daily_value):
                rel_diffs.append(0.0)
                bp_diffs.append(None)
                continue
            denominator = abs(float(daily_value)) if float(daily_value) != 0 else None
            rel_diffs.append(abs(float(minute_value) - float(daily_value)) / denominator if denominator else 0.0)
            bp_diffs.append(bp_delta(float(minute_value), float(daily_value)))
        rel_diff = pd.Series(rel_diffs)
        bp_diff = pd.Series(bp_diffs, dtype="float64")
        max_rel_diff = float(rel_diff.max()) if not rel_diff.empty else 0.0
        max_bp = float(bp_diff.max()) if not bp_diff.dropna().empty else 0.0
        field_pass = max_bp <= spec["threshold_bp"]
        if spec["severity"] == "blocker":
            blocker_pass &= field_pass
        else:
            flagged = merged.loc[bp_diff > spec["threshold_bp"], ["ticker", "trade_date", f"{field}_daily", f"{field}_minute"]].copy()
            flagged["delta_bp"] = bp_diff.loc[flagged.index].astype(float)
            for row in flagged.itertuples(index=False):
                warning_events.append(
                    {
                        "ticker": str(row.ticker).upper(),
                        "trade_date": row.trade_date,
                        "field": field,
                        "stock_prices_value": float(getattr(row, f"{field}_daily")),
                        "minute_agg_value": float(getattr(row, f"{field}_minute")),
                        "delta_bp": float(row.delta_bp),
                        "severity": "warning",
                    },
                )
        summary[field] = {
            "max_abs_rel_diff": max_rel_diff,
            "max_abs_bp": max_bp,
            "threshold_bp": float(spec["threshold_bp"]),
            "severity": spec["severity"],
            "pass": bool(field_pass),
        }
    concentrated_warning_days: list[dict[str, Any]] = []
    if warning_events:
        warning_frame = pd.DataFrame(warning_events)
        grouped = (
            warning_frame.groupby("trade_date")["ticker"]
            .nunique()
            .reset_index(name="affected_tickers")
            .sort_values(["affected_tickers", "trade_date"], ascending=[False, True])
        )
        for row in grouped.itertuples(index=False):
            if row.affected_tickers > 3:
                concentrated_warning_days.append(
                    {
                        "trade_date": row.trade_date,
                        "affected_tickers": int(row.affected_tickers),
                    },
                )
    return {
        "pass": bool(blocker_pass),
        "row_count": int(len(merged)),
        "fields": summary,
        "warning_events": warning_events,
        "warning_event_count": int(len(warning_events)),
        "concentrated_warning_days": concentrated_warning_days,
    }


def persist_reconciliation_events(
    events: list[dict[str, Any]],
    *,
    batch_id: str,
    detected_at: datetime,
    session_factory: Any | None = None,
) -> int:
    if not events:
        return 0
    session_factory = session_factory or get_session_factory()
    rows = [
        PriceReconciliationEvent(
            ticker=str(event["ticker"]).upper(),
            trade_date=event["trade_date"],
            field=event["field"],
            stock_prices_value=event["stock_prices_value"],
            minute_agg_value=event["minute_agg_value"],
            delta_bp=event["delta_bp"],
            severity=event.get("severity", "warning"),
            detected_at=detected_at,
            batch_id=batch_id,
        )
        for event in events
    ]
    session = session_factory()
    try:
        session.add_all(rows)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        if hasattr(session, "close"):
            session.close()
    return len(rows)


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

    filtered_daily_prices = daily_prices.loc[
        (pd.to_datetime(daily_prices["trade_date"]).dt.date >= start_date)
        & (pd.to_datetime(daily_prices["trade_date"]).dt.date <= end_date)
    ].copy()
    timezone_result = validate_timezones(persisted)
    trading_days_result = validate_trading_days(
        minute_frame=persisted,
        start_date=start_date,
        end_date=end_date,
        expected_tickers=list(tickers),
    )
    internal_consistency = validate_minute_internal_consistency(persisted)
    corporate_action_alignment = validate_corporate_action_alignment(persisted, filtered_daily_prices)
    minute_to_day = validate_minute_to_day_consistency(persisted, filtered_daily_prices)
    reconciliation_batch_id = str(uuid.uuid4())
    reconciliation_event_count = persist_reconciliation_events(
        minute_to_day["warning_events"],
        batch_id=reconciliation_batch_id,
        detected_at=datetime.now(timezone.utc),
    )

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
            "timezone": timezone_result,
            "trading_days": trading_days_result,
            "minute_internal_consistency": internal_consistency,
            "corporate_action_alignment": corporate_action_alignment,
            "minute_to_day_consistency": minute_to_day,
            "reconciliation_events": {
                "table_name": "price_reconciliation_events",
                "batch_id": reconciliation_batch_id,
                "inserted_rows": int(reconciliation_event_count),
            },
        },
    }
    verification = report["verification"]
    report["pass"] = bool(
        verification["schema"]["pass"]
        and verification["timezone"]["pass"]
        and verification["trading_days"]["pass"]
        and verification["minute_internal_consistency"]["pass"]
        and verification["corporate_action_alignment"]["pass"]
        and verification["minute_to_day_consistency"]["pass"]
    )

    report_path = REPO_ROOT / args.report_output
    write_json_atomic(report_path, json_safe(report))
    print(json.dumps(json_safe(report), indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
