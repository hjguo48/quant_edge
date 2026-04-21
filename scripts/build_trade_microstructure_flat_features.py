"""Build Week 4 trade microstructure features directly from Polygon trades flat files."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from loguru import logger

try:
    import scripts.preflight_trades_estimator as preflight_trades_estimator
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import preflight_trades_estimator as preflight_trades_estimator

from src.config.week4_trades import Week4TradesConfig
from src.data.polygon_trades_flat import PolygonTradesFlatFilesClient
from src.features.trade_microstructure import (
    compute_knowledge_time,
    compute_large_trade_ratio,
    compute_late_day_aggressiveness,
    compute_off_exchange_volume_ratio,
    compute_offhours_trade_ratio,
    compute_trade_imbalance_proxy,
)
from src.universe.top_liquidity import get_top_liquidity_tickers

DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_START_DATE = date(2021, 9, 3)
DEFAULT_END_DATE = date(2022, 2, 25)
DEFAULT_OUTPUT_PATH = Path("data/features/trade_microstructure_flat_W5.parquet")
DEFAULT_TOP_N = 50
XNYS = xcals.get_calendar("XNYS")

OUTPUT_COLUMNS = [
    "event_date",
    "ticker",
    "knowledge_time_regular",
    "knowledge_time_offhours",
    "trade_imbalance_proxy",
    "large_trade_ratio",
    "late_day_aggressiveness",
    "offhours_trade_ratio",
    "off_exchange_volume_ratio",
    "run_config_hash",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trade microstructure features from Polygon trades flat files.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--flush-days", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1, help="Reserved for future parallel day downloads; current path is sequential.")
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
        logger.warning("existing flat trade feature parquet config_hash mismatch; full recompute")
        return pd.DataFrame(columns=OUTPUT_COLUMNS), set()
    existing["event_date"] = pd.to_datetime(existing["event_date"]).dt.date
    existing["ticker"] = existing["ticker"].astype(str).str.upper()
    completed = {(row.ticker, row.event_date) for row in existing.itertuples(index=False)}
    return existing, completed


def compute_feature_row(
    *,
    ticker: str,
    event_date: date,
    trades: pd.DataFrame,
    config: Week4TradesConfig,
    config_hash: str,
) -> dict[str, Any]:
    condition_allow = set(config.features.condition_allow_list) or None
    return {
        "event_date": event_date,
        "ticker": ticker.upper(),
        "knowledge_time_regular": compute_knowledge_time(event_date, "trade_imbalance_proxy"),
        "knowledge_time_offhours": compute_knowledge_time(event_date, "offhours_trade_ratio"),
        "trade_imbalance_proxy": compute_trade_imbalance_proxy(trades, condition_allow=condition_allow),
        "large_trade_ratio": compute_large_trade_ratio(
            trades,
            size_threshold_dollars=float(config.features.size_threshold_dollars),
        ),
        "late_day_aggressiveness": compute_late_day_aggressiveness(
            trades,
            late_day_window_et=tuple(config.features.late_day_window_et),
        ),
        "offhours_trade_ratio": compute_offhours_trade_ratio(
            trades,
            pre_window_et=tuple(config.features.offhours_window_et_pre),
            post_window_et=tuple(config.features.offhours_window_et_post),
        ),
        "off_exchange_volume_ratio": compute_off_exchange_volume_ratio(
            trades,
            trf_exchange_codes=set(config.features.trf_exchange_codes),
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


def build_flat_features(
    *,
    config: Week4TradesConfig,
    config_hash: str,
    start_date: date,
    end_date: date,
    top_n: int,
    output_path: Path,
    resume: bool = True,
    flush_days: int = 10,
    concurrency: int = 1,
    client: PolygonTradesFlatFilesClient | None = None,
    top_liquidity_fn: Callable[..., list[str]] = get_top_liquidity_tickers,
    session_factory: Callable | None = None,
) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if concurrency != 1:
        logger.warning("concurrency={} requested; flat trade builder currently processes one day at a time", concurrency)

    if resume:
        existing_frame, completed = load_existing_rows(output_path, config_hash)
    else:
        existing_frame, completed = pd.DataFrame(columns=OUTPUT_COLUMNS), set()

    flat_client = client or PolygonTradesFlatFilesClient()
    new_rows: list[dict[str, Any]] = []
    dates_processed = 0
    days_downloaded = 0
    ticker_days_seen = 0
    rows_skipped_resume = 0
    for trading_date in session_dates(start_date, end_date):
        dates_processed += 1
        top_tickers = [
            str(ticker).upper()
            for ticker in top_liquidity_fn(trading_date, top_n=top_n, session_factory=session_factory)[:top_n]
        ]
        pending = [ticker for ticker in top_tickers if not (resume and (ticker, trading_date) in completed)]
        rows_skipped_resume += len(top_tickers) - len(pending)
        ticker_days_seen += len(top_tickers)
        if not pending:
            continue

        logger.info("loading trades flat file for {} ({} pending tickers)", trading_date, len(pending))
        day_trades = flat_client.load_day_for_tickers(trading_date, pending)
        days_downloaded += 1
        if not day_trades.empty:
            day_trades["ticker"] = day_trades["ticker"].astype(str).str.upper()

        for ticker in pending:
            ticker_trades = (
                day_trades.loc[day_trades["ticker"] == ticker].copy()
                if not day_trades.empty
                else pd.DataFrame(columns=day_trades.columns)
            )
            new_rows.append(
                compute_feature_row(
                    ticker=ticker,
                    event_date=trading_date,
                    trades=ticker_trades,
                    config=config,
                    config_hash=config_hash,
                ),
            )

        if flush_days > 0 and days_downloaded % flush_days == 0:
            write_features(_combine_feature_frames(existing_frame, new_rows), output_path)
            logger.info("flushed {} rows after {} downloaded days", len(new_rows), days_downloaded)

    combined = _combine_feature_frames(existing_frame, new_rows)
    if not combined.empty:
        write_features(combined, output_path)

    summary = {
        "dates_processed": dates_processed,
        "days_downloaded": days_downloaded,
        "ticker_days_seen": ticker_days_seen,
        "rows_skipped_resume": rows_skipped_resume,
        "rows_computed": len(new_rows),
        "rows_total": len(combined),
        "top_n": top_n,
        "config_hash": config_hash,
        "output_path": str(output_path),
    }
    logger.info("flat trade microstructure feature summary: {}", summary)
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = preflight_trades_estimator.load_config(args.config)
    config_hash = preflight_trades_estimator.compute_config_hash(config)
    summary = build_flat_features(
        config=config,
        config_hash=config_hash,
        start_date=args.start_date,
        end_date=args.end_date,
        top_n=args.top_n,
        output_path=args.output,
        resume=args.resume,
        flush_days=args.flush_days,
        concurrency=args.concurrency,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script execution
    sys.exit(main())
