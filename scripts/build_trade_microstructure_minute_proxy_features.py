"""Build Week 4 minute-level proxy features from stock_minute_aggs.

This is a fast fallback path for gate validation when Polygon trade-level REST
sampling is too slow. It does not write feature_store and does not call Polygon.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    import scripts.preflight_trades_estimator as preflight_trades_estimator
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import preflight_trades_estimator as preflight_trades_estimator

from src.config.week4_trades import Week4TradesConfig
from src.data.db.models import StockMinuteAggs
from src.data.db.session import get_session_factory
from src.features.trade_microstructure import compute_knowledge_time
from src.features.trade_microstructure_minute_proxy import (
    compute_late_day_aggressiveness_minute,
    compute_offhours_trade_ratio_minute,
    compute_trade_imbalance_proxy_minute,
)
from src.universe.top_liquidity import get_top_liquidity_tickers

DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_OUTPUT_PATH = Path("data/features/trade_microstructure_minute_proxy.parquet")
DEFAULT_START_DATE = date(2019, 1, 1)
DEFAULT_END_DATE = date(2026, 4, 20)
DEFAULT_TOP_N = 50
EASTERN = ZoneInfo("America/New_York")
XNYS = xcals.get_calendar("XNYS")

OUTPUT_COLUMNS = [
    "event_date",
    "ticker",
    "knowledge_time_regular",
    "knowledge_time_offhours",
    "trade_imbalance_proxy",
    "late_day_aggressiveness",
    "offhours_trade_ratio",
    "run_config_hash",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build minute-level proxy trade microstructure features.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args(argv)


def session_dates(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        return []
    return [pd.Timestamp(label).date() for label in XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))]


def load_existing_rows(output_path: Path, config_hash: str) -> tuple[pd.DataFrame, set[tuple[str, date]]]:
    if not output_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS), set()
    existing = pd.read_parquet(output_path)
    if existing.empty:
        return existing, set()
    if set(existing["run_config_hash"].astype(str).unique()) != {config_hash}:
        logger.warning("existing proxy parquet config_hash mismatch; full recompute")
        return pd.DataFrame(columns=OUTPUT_COLUMNS), set()
    existing["event_date"] = pd.to_datetime(existing["event_date"]).dt.date
    existing["ticker"] = existing["ticker"].astype(str).str.upper()
    completed = {(row.ticker, row.event_date) for row in existing.itertuples(index=False)}
    return existing, completed


def load_minute_bars_for_ticker_date(session: Session, ticker: str, trading_date: date) -> pd.DataFrame:
    start_utc, end_utc = _et_window_utc(trading_date, "04:00", "20:00")
    stmt = (
        select(
            StockMinuteAggs.ticker,
            StockMinuteAggs.trade_date,
            StockMinuteAggs.minute_ts,
            StockMinuteAggs.open,
            StockMinuteAggs.close,
            StockMinuteAggs.high,
            StockMinuteAggs.low,
            StockMinuteAggs.volume,
            StockMinuteAggs.transactions,
            StockMinuteAggs.vwap,
        )
        .where(
            StockMinuteAggs.ticker == ticker.upper(),
            StockMinuteAggs.trade_date == trading_date,
            StockMinuteAggs.minute_ts >= start_utc,
            StockMinuteAggs.minute_ts < end_utc,
        )
        .order_by(StockMinuteAggs.minute_ts)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(
            columns=[
                "ticker",
                "trading_date",
                "minute_ts",
                "open",
                "close",
                "high",
                "low",
                "volume",
                "transactions",
                "vwap",
            ],
        )
    frame = pd.DataFrame.from_records(
        rows,
        columns=["ticker", "trading_date", "minute_ts", "open", "close", "high", "low", "volume", "transactions", "vwap"],
    )
    return frame


def compute_proxy_feature_row(
    *,
    ticker: str,
    event_date: date,
    minute_bars: pd.DataFrame,
    config: Week4TradesConfig,
    config_hash: str,
) -> dict[str, Any]:
    late_window = tuple(config.features.late_day_window_et)
    pre_window = tuple(config.features.offhours_window_et_pre)
    post_window = tuple(config.features.offhours_window_et_post)

    return {
        "event_date": event_date,
        "ticker": ticker.upper(),
        "knowledge_time_regular": compute_knowledge_time(event_date, "trade_imbalance_proxy"),
        "knowledge_time_offhours": compute_knowledge_time(event_date, "offhours_trade_ratio"),
        "trade_imbalance_proxy": compute_trade_imbalance_proxy_minute(minute_bars),
        "late_day_aggressiveness": compute_late_day_aggressiveness_minute(minute_bars, late_window_et=late_window),
        "offhours_trade_ratio": compute_offhours_trade_ratio_minute(
            minute_bars,
            pre_et=pre_window,
            post_et=post_window,
        ),
        "run_config_hash": config_hash,
    }


def write_features(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame[OUTPUT_COLUMNS].copy()
    ordered["event_date"] = pd.to_datetime(ordered["event_date"]).dt.date
    ordered.sort_values(["event_date", "ticker"], inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    ordered.to_parquet(output_path, index=False)


def build_minute_proxy_features(
    *,
    config: Week4TradesConfig,
    config_hash: str,
    start_date: date,
    end_date: date,
    top_n: int,
    output_path: Path,
    resume: bool = True,
    session_factory: Callable[[], Session] | None = None,
    top_liquidity_fn: Callable[..., list[str]] = get_top_liquidity_tickers,
    load_minutes_fn: Callable[[Session, str, date], pd.DataFrame] = load_minute_bars_for_ticker_date,
    flush_every: int = 10_000,
) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    factory = session_factory or get_session_factory()
    if resume:
        existing_frame, completed = load_existing_rows(output_path, config_hash)
    else:
        existing_frame, completed = pd.DataFrame(columns=OUTPUT_COLUMNS), set()

    new_rows: list[dict[str, Any]] = []
    skipped = 0
    dates_processed = 0
    ticker_days_seen = 0
    with factory() as session:
        for trading_date in session_dates(start_date, end_date):
            dates_processed += 1
            tickers = [
                str(ticker).upper()
                for ticker in top_liquidity_fn(trading_date, top_n=top_n, session_factory=factory)[:top_n]
            ]
            for ticker in tickers:
                ticker_days_seen += 1
                if resume and (ticker, trading_date) in completed:
                    skipped += 1
                    continue
                minute_bars = load_minutes_fn(session, ticker, trading_date)
                new_rows.append(
                    compute_proxy_feature_row(
                        ticker=ticker,
                        event_date=trading_date,
                        minute_bars=minute_bars,
                        config=config,
                        config_hash=config_hash,
                    ),
                )
                if flush_every > 0 and len(new_rows) % flush_every == 0:
                    write_features(_combine_feature_frames(existing_frame, new_rows), output_path)

    combined = _combine_feature_frames(existing_frame, new_rows)
    if not combined.empty:
        write_features(combined, output_path)

    summary = {
        "dates_processed": dates_processed,
        "ticker_days_seen": ticker_days_seen,
        "rows_skipped_resume": skipped,
        "rows_computed": len(new_rows),
        "rows_total": len(combined),
        "top_n": top_n,
        "config_hash": config_hash,
        "output_path": str(output_path),
    }
    logger.info("minute proxy trade microstructure summary: {}", summary)
    return summary


def _combine_feature_frames(existing_frame: pd.DataFrame, new_rows: list[dict[str, Any]]) -> pd.DataFrame:
    new_frame = pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS) if new_rows else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if not existing_frame.empty and not new_frame.empty:
        combined = pd.concat([existing_frame, new_frame], ignore_index=True)
    elif not existing_frame.empty:
        combined = existing_frame.copy()
    else:
        combined = new_frame
    if not combined.empty:
        combined.drop_duplicates(["event_date", "ticker"], keep="last", inplace=True)
    return combined


def _et_window_utc(trading_date: date, start_hhmm: str, end_hhmm: str) -> tuple[datetime, datetime]:
    start = _combine_et(trading_date, start_hhmm)
    end = _combine_et(trading_date, end_hhmm)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def _combine_et(trading_date: date, hhmm: str) -> datetime:
    hour_raw, minute_raw = hhmm.split(":", 1)
    return datetime.combine(trading_date, time(int(hour_raw), int(minute_raw)), tzinfo=EASTERN)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = preflight_trades_estimator.load_config(args.config)
    config_hash = preflight_trades_estimator.compute_config_hash(config)
    summary = build_minute_proxy_features(
        config=config,
        config_hash=config_hash,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        output_path=args.output,
        resume=args.resume,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script execution
    sys.exit(main())
