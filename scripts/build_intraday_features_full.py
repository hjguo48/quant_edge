#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import uuid

import exchange_calendars as xcals
from loguru import logger
import numpy as np
import pandas as pd
import sqlalchemy as sa

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from src.data.db.models import MinuteBackfillState, StockMinuteAggs, UniverseMembership
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_engine, get_session_factory
from src.features.intraday import INTRADAY_FEATURE_NAMES, compute_intraday_features
from src.features.pipeline import FeaturePipeline, prepare_feature_export_frame

XNYS = xcals.get_calendar("XNYS")
DEFAULT_START_DATE = date(2016, 4, 20)
DEFAULT_END_DATE = date(2026, 4, 17)
DEFAULT_CHUNK_SIZE = 200
DEFAULT_REPORT_OUTPUT = f"data/reports/intraday_feature_build_{date.today().strftime('%Y%m%d')}.json"
INTRADAY_FEATURE_SET = set(INTRADAY_FEATURE_NAMES)
FEATURE_COLUMNS = ["ticker", "trade_date", "feature_name", "feature_value", "is_filled"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the full historical intraday feature panel into feature_store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date", default=DEFAULT_END_DATE.isoformat())
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-workers", type=int, default=min(4, max(1, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument(
        "--features",
        default=",".join(INTRADAY_FEATURE_NAMES),
        help="Comma-separated subset of intraday features to build.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def month_windows(start_date: date, end_date: date) -> list[tuple[date, date]]:
    windows: list[tuple[date, date]] = []
    current = start_date.replace(day=1)
    while current <= end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        window_start = max(start_date, current)
        window_end = min(end_date, next_month - timedelta(days=1))
        if window_start <= window_end:
            windows.append((window_start, window_end))
        current = next_month
    return windows


def chunked(values: Sequence[str], chunk_size: int) -> Iterator[list[str]]:
    for start_index in range(0, len(values), chunk_size):
        yield list(values[start_index : start_index + chunk_size])


def _next_session_close(trade_date: date) -> datetime:
    session_label = XNYS.date_to_session(pd.Timestamp(trade_date), direction="previous")
    next_session = XNYS.next_session(session_label)
    close = pd.Timestamp(XNYS.session_close(next_session))
    return close.to_pydatetime()


def _history_start_for_window(window_start: date, *, calendar_days: int = 70) -> date:
    return window_start - timedelta(days=calendar_days)


def load_membership_rows(*, start_date: date, end_date: date, index_name: str = "SP500") -> list[dict[str, object]]:
    statement = (
        sa.select(
            UniverseMembership.ticker,
            UniverseMembership.effective_date,
            UniverseMembership.end_date,
        )
        .where(
            UniverseMembership.index_name == index_name,
            UniverseMembership.effective_date <= end_date,
            sa.or_(
                UniverseMembership.end_date.is_(None),
                UniverseMembership.end_date > start_date,
            ),
        )
        .order_by(UniverseMembership.ticker, UniverseMembership.effective_date)
    )
    with get_engine().connect() as conn:
        rows = conn.execute(statement).mappings().all()
    return [dict(row) for row in rows]


def build_expected_pairs(
    *,
    session_dates: Sequence[date],
    membership_rows: Sequence[dict[str, object]],
) -> tuple[pd.DataFrame, list[date]]:
    pairs: list[dict[str, object]] = []
    skipped_dates: list[date] = []
    for trade_day in session_dates:
        active = sorted(
            {
                str(row["ticker"]).upper()
                for row in membership_rows
                if row["effective_date"] <= trade_day
                and (row["end_date"] is None or row["end_date"] > trade_day)
            },
        )
        if not active:
            skipped_dates.append(trade_day)
            continue
        for ticker in active:
            pairs.append({"ticker": ticker, "trade_date": trade_day})
    return pd.DataFrame(pairs, columns=["ticker", "trade_date"]), skipped_dates


def build_expected_feature_rows(expected_pairs: pd.DataFrame) -> pd.DataFrame:
    return build_expected_feature_rows_for_names(expected_pairs, INTRADAY_FEATURE_NAMES)


def build_expected_feature_rows_for_names(
    expected_pairs: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if expected_pairs.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in expected_pairs.itertuples(index=False):
        for feature_name in feature_names:
            rows.append(
                {
                    "ticker": str(row.ticker).upper(),
                    "trade_date": row.trade_date,
                    "feature_name": feature_name,
                    "feature_value": np.nan,
                    "is_filled": True,
                },
            )
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS)


def load_existing_feature_rows(
    *,
    start_date: date,
    end_date: date,
    tickers: Sequence[str],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name"])
    query = sa.text(
        """
        select ticker, calc_date as trade_date, feature_name
        from feature_store
        where calc_date >= :start_date
          and calc_date <= :end_date
          and ticker = any(:tickers)
          and feature_name = any(:feature_names)
        """,
    )
    with get_engine().connect() as conn:
        frame = pd.read_sql_query(
            query,
            conn,
            params={
                "start_date": start_date,
                "end_date": end_date,
                "tickers": list(tickers),
                "feature_names": list(feature_names),
            },
            parse_dates=["trade_date"],
        )
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["feature_name"] = frame["feature_name"].astype(str)
    return frame.loc[:, ["ticker", "trade_date", "feature_name"]].drop_duplicates()


def merge_intraday_rows(expected_rows: pd.DataFrame, computed_rows: pd.DataFrame) -> pd.DataFrame:
    if expected_rows.empty:
        return expected_rows.copy()
    computed = prepare_feature_export_frame(computed_rows) if not computed_rows.empty else pd.DataFrame(columns=FEATURE_COLUMNS)
    if not computed.empty:
        computed = computed.loc[computed["feature_name"].isin(INTRADAY_FEATURE_NAMES)].copy()
        computed["is_filled"] = computed["is_filled"].fillna(False).astype(bool) | computed["feature_value"].isna()
    merged = expected_rows.merge(
        computed,
        on=["ticker", "trade_date", "feature_name"],
        how="left",
        suffixes=("_default", ""),
    )
    merged["feature_value"] = merged["feature_value"].combine_first(merged["feature_value_default"])
    merged["is_filled"] = merged["is_filled"].fillna(merged["is_filled_default"]).astype(bool)
    merged.loc[merged["feature_value"].isna(), "is_filled"] = True
    return prepare_feature_export_frame(merged.loc[:, FEATURE_COLUMNS])


def filter_existing_feature_rows(frame: pd.DataFrame, existing_rows: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty or existing_rows.empty:
        return frame.copy(), 0
    existing_keys = {
        (str(row.ticker).upper(), row.trade_date, str(row.feature_name))
        for row in existing_rows.itertuples(index=False)
    }
    mask = [
        (str(row.ticker).upper(), row.trade_date, str(row.feature_name)) not in existing_keys
        for row in frame.itertuples(index=False)
    ]
    skipped_count = int(len(frame) - sum(mask))
    return frame.loc[mask].reset_index(drop=True), skipped_count


def build_chunk_frame(
    *,
    tickers: Sequence[str],
    target_start: date,
    target_end: date,
    expected_pairs: pd.DataFrame,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if expected_pairs.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    history_start = _history_start_for_window(target_start)
    as_of = _next_session_close(target_end)
    minute_history = load_intraday_minute_history_build(
        tickers=tickers,
        start_trade_date=history_start,
        end_trade_date=target_end,
        as_of=as_of,
    )
    daily_prices = get_prices_pit(
        tickers=tuple(tickers),
        start_date=history_start,
        end_date=target_end,
        as_of=as_of,
    )
    computed = compute_intraday_features(
        minute_df=minute_history,
        daily_prices_df=daily_prices,
    )
    if not computed.empty:
        computed = computed.loc[
            (pd.to_datetime(computed["trade_date"]).dt.date >= target_start)
            & (pd.to_datetime(computed["trade_date"]).dt.date <= target_end)
            & (computed["ticker"].astype(str).str.upper().isin([str(ticker).upper() for ticker in tickers]))
            & (computed["feature_name"].isin(feature_names))
        ].copy()
    expected_rows = build_expected_feature_rows_for_names(expected_pairs, feature_names)
    return merge_intraday_rows(expected_rows, computed)


def load_intraday_minute_history_build(
    *,
    tickers: Sequence[str],
    start_trade_date: date,
    end_trade_date: date,
    as_of: date | datetime,
) -> pd.DataFrame:
    normalized_tickers = tuple(dict.fromkeys(str(ticker).upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        return pd.DataFrame(
            columns=["ticker", "trade_date", "minute_ts", "open", "high", "low", "close", "volume", "vwap", "transactions"],
        )
    cutoff = as_of if isinstance(as_of, datetime) else datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)
    statement = (
        sa.select(
            StockMinuteAggs.ticker,
            StockMinuteAggs.trade_date,
            StockMinuteAggs.minute_ts,
            StockMinuteAggs.open,
            StockMinuteAggs.high,
            StockMinuteAggs.low,
            StockMinuteAggs.close,
            StockMinuteAggs.volume,
            StockMinuteAggs.vwap,
            StockMinuteAggs.transactions,
        )
        .where(
            StockMinuteAggs.ticker.in_(normalized_tickers),
            StockMinuteAggs.trade_date >= start_trade_date,
            StockMinuteAggs.trade_date <= end_trade_date,
            StockMinuteAggs.knowledge_time <= cutoff,
        )
        .order_by(StockMinuteAggs.ticker, StockMinuteAggs.minute_ts)
    )
    session_factory = get_session_factory()
    with session_factory() as session:
        session.execute(sa.text("set local max_parallel_workers_per_gather = 0"))
        expected_sessions = [
            pd.Timestamp(session_label).date()
            for session_label in XNYS.sessions_in_range(pd.Timestamp(start_trade_date), pd.Timestamp(end_trade_date))
        ]
        if expected_sessions:
            state_query = sa.select(MinuteBackfillState.trading_date, MinuteBackfillState.status).where(
                MinuteBackfillState.trading_date >= expected_sessions[0],
                MinuteBackfillState.trading_date <= expected_sessions[-1],
            )
            state_rows = session.execute(state_query).mappings().all()
            state_map = {
                pd.Timestamp(row["trading_date"]).date(): str(row["status"]).lower()
                for row in state_rows
            }
            incomplete = [
                trade_day
                for trade_day in expected_sessions
                if state_map.get(trade_day) not in {"completed", "skipped_holiday"}
            ]
            if incomplete:
                logger.error(
                    "minute_history backfill incomplete (build path): {} days missing ({}...) range {}~{}",
                    len(incomplete),
                    [trade_day.isoformat() for trade_day in incomplete[:5]],
                    start_trade_date,
                    end_trade_date,
                )
        rows = session.execute(statement).mappings().all()
    if not rows:
        logger.error(
            "minute_history empty (build path): tickers={} range={}~{}",
            normalized_tickers,
            start_trade_date,
            end_trade_date,
        )
        return pd.DataFrame(
            columns=["ticker", "trade_date", "minute_ts", "open", "high", "low", "close", "volume", "vwap", "transactions"],
        )
    loaded_tickers = {str(row["ticker"]).upper() for row in rows}
    missing_tickers = sorted(set(normalized_tickers) - loaded_tickers)
    if missing_tickers:
        logger.error(
            "minute_history partial coverage (build path): missing tickers {} range {}~{}",
            missing_tickers,
            start_trade_date,
            end_trade_date,
        )
    return pd.DataFrame(rows)


def _coerce_session_dates(start_date: date, end_date: date) -> list[date]:
    return [timestamp.date() for timestamp in XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))]


def process_month_window(
    *,
    window_start: date,
    window_end: date,
    chunk_size: int,
    max_workers: int,
    feature_names: Sequence[str],
) -> dict[str, object]:
    session_dates = _coerce_session_dates(window_start, window_end)
    membership_rows = load_membership_rows(start_date=window_start, end_date=window_end)
    expected_pairs, skipped_universe_days = build_expected_pairs(
        session_dates=session_dates,
        membership_rows=membership_rows,
    )
    result: dict[str, object] = {
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "session_day_count": len(session_dates),
        "skipped_universe_days": [trade_day.isoformat() for trade_day in skipped_universe_days],
        "expected_ticker_dates": int(len(expected_pairs)),
        "expected_feature_rows": int(len(expected_pairs) * len(feature_names)),
        "rows_written": 0,
        "existing_feature_rows_skipped": 0,
        "missing_rows_padded": 0,
        "batches_processed": 0,
    }
    if expected_pairs.empty:
        return result

    unique_tickers = sorted(expected_pairs["ticker"].astype(str).str.upper().unique().tolist())
    batches = list(enumerate(chunked(unique_tickers, chunk_size), start=1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_ticker_batch,
                batch_index=batch_index,
                total_batches=max(1, len(batches)),
                ticker_batch=ticker_batch,
                expected_pairs=expected_pairs.loc[expected_pairs["ticker"].isin(ticker_batch)].copy(),
                window_start=window_start,
                window_end=window_end,
                feature_names=feature_names,
            ): batch_index
            for batch_index, ticker_batch in batches
        }
        for future in as_completed(futures):
            batch_summary = future.result()
            result["rows_written"] = int(result["rows_written"]) + int(batch_summary["rows_written"])
            result["existing_feature_rows_skipped"] = int(result["existing_feature_rows_skipped"]) + int(batch_summary["existing_feature_rows_skipped"])
            result["missing_rows_padded"] = int(result["missing_rows_padded"]) + int(batch_summary["missing_rows_padded"])
            result["batches_processed"] = int(result["batches_processed"]) + int(batch_summary["batches_processed"])
    return result


def process_ticker_batch(
    *,
    batch_index: int,
    total_batches: int,
    ticker_batch: Sequence[str],
    expected_pairs: pd.DataFrame,
    window_start: date,
    window_end: date,
    feature_names: Sequence[str],
) -> dict[str, int]:
    if expected_pairs.empty:
        return {
            "rows_written": 0,
            "existing_feature_rows_skipped": 0,
            "missing_rows_padded": 0,
            "batches_processed": 0,
        }
    existing_rows = load_existing_feature_rows(
        start_date=window_start,
        end_date=window_end,
        tickers=ticker_batch,
        feature_names=feature_names,
    )
    existing_pair_counts = Counter(
        (str(row.ticker).upper(), row.trade_date)
        for row in existing_rows.itertuples(index=False)
        if str(row.feature_name) in set(feature_names)
    )
    complete_pairs = {
        pair
        for pair, count in existing_pair_counts.items()
        if count >= len(feature_names)
    }
    if complete_pairs:
        expected_pairs = expected_pairs.loc[
            [
                (str(row.ticker).upper(), row.trade_date) not in complete_pairs
                for row in expected_pairs.itertuples(index=False)
            ]
        ].copy()
    if expected_pairs.empty:
        return {
            "rows_written": 0,
            "existing_feature_rows_skipped": int(len(existing_rows)),
            "missing_rows_padded": 0,
            "batches_processed": 0,
        }
    logger.info(
        "building intraday month {} batch {}/{} for {} ticker-dates across {} tickers",
        window_start.strftime("%Y-%m"),
        batch_index,
        total_batches,
        len(expected_pairs),
        len(ticker_batch),
    )
    batch_frame = build_chunk_frame(
        tickers=ticker_batch,
        target_start=window_start,
        target_end=window_end,
        expected_pairs=expected_pairs,
        feature_names=feature_names,
    )
    padded_count = int(batch_frame["feature_value"].isna().sum())
    filtered_frame, skipped_existing = filter_existing_feature_rows(batch_frame, existing_rows)
    if filtered_frame.empty:
        return {
            "rows_written": 0,
            "existing_feature_rows_skipped": int(skipped_existing),
            "missing_rows_padded": int(padded_count),
            "batches_processed": 0,
        }
    batch_id = f"intraday-full-{window_start.strftime('%Y%m')}-{batch_index:03d}-{uuid.uuid4().hex[:8]}"
    rows_written = FeaturePipeline().save_to_store(filtered_frame, batch_id=batch_id, batch_size=25_000)
    return {
        "rows_written": int(rows_written),
        "existing_feature_rows_skipped": int(skipped_existing),
        "missing_rows_padded": int(padded_count),
        "batches_processed": 1,
    }


def build_intraday_feature_history(
    *,
    start_date: date,
    end_date: date,
    chunk_size: int,
    max_workers: int,
    feature_names: Sequence[str],
) -> dict[str, object]:
    month_results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    started_at = datetime.now(timezone.utc)
    for window_start, window_end in month_windows(start_date, end_date):
        try:
            month_results.append(
                process_month_window(
                    window_start=window_start,
                    window_end=window_end,
                    chunk_size=chunk_size,
                    max_workers=max_workers,
                    feature_names=feature_names,
                ),
            )
        except Exception as exc:
            logger.opt(exception=exc).error(
                "intraday feature build failed for {} -> {}",
                window_start,
                window_end,
            )
            errors.append(
                {
                    "window_start": window_start.isoformat(),
                    "window_end": window_end.isoformat(),
                    "error": str(exc),
                },
            )
    finished_at = datetime.now(timezone.utc)
    total_expected_ticker_dates = sum(int(month["expected_ticker_dates"]) for month in month_results)
    total_expected_feature_rows = sum(int(month["expected_feature_rows"]) for month in month_results)
    total_rows_written = sum(int(month["rows_written"]) for month in month_results)
    total_skipped_existing = sum(int(month["existing_feature_rows_skipped"]) for month in month_results)
    total_missing_padded = sum(int(month["missing_rows_padded"]) for month in month_results)
    return {
        "metadata": {
            "script_name": "build_intraday_features_full.py",
            "generated_at_utc": finished_at.isoformat(),
            "started_at_utc": started_at.isoformat(),
            "duration_seconds": (finished_at - started_at).total_seconds(),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "chunk_size": int(chunk_size),
            "max_workers": int(max_workers),
            "feature_names": list(feature_names),
        },
        "summary": {
            "month_window_count": len(month_results),
            "rows_written": int(total_rows_written),
            "expected_ticker_dates": int(total_expected_ticker_dates),
            "expected_feature_rows": int(total_expected_feature_rows),
            "existing_feature_rows_skipped": int(total_skipped_existing),
            "missing_rows_padded": int(total_missing_padded),
            "error_window_count": int(len(errors)),
        },
        "month_results": month_results,
        "error_days": errors,
        "status": "ok" if not errors else "error",
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    start_date = date.fromisoformat(args.start_date)
    end_date = date.fromisoformat(args.end_date)
    feature_names = tuple(
        name.strip()
        for name in str(args.features).split(",")
        if name.strip()
    )
    unknown_features = sorted(set(feature_names) - INTRADAY_FEATURE_SET)
    if unknown_features:
        raise SystemExit(f"unknown intraday feature(s): {', '.join(unknown_features)}")
    summary = build_intraday_feature_history(
        start_date=start_date,
        end_date=end_date,
        chunk_size=int(args.chunk_size),
        max_workers=int(args.max_workers),
        feature_names=feature_names,
    )
    report_path = REPO_ROOT / args.report_output
    write_json_atomic(report_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
