"""Week 4 Task 8: 从 stock_trades_sampled 批量计算 5 个 trade microstructure 特征.

CLI:
    uv run python scripts/build_trade_microstructure_features.py \
        --config configs/research/week4_trades_sampling.yaml \
        --start-date 2016-04-17 \
        --end-date 2026-04-17 \
        --output data/features/trade_microstructure_features.parquet

输出 parquet schema:
    event_date, ticker, knowledge_time_regular, knowledge_time_offhours,
    trade_imbalance_proxy, large_trade_ratio, late_day_aggressiveness,
    offhours_trade_ratio, off_exchange_volume_ratio, run_config_hash

Resume: 已有 parquet 中 (ticker, event_date) 跳过 (若 config_hash 一致); config_hash 变则全部重算.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

try:
    import scripts.preflight_trades_estimator as preflight_trades_estimator
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import preflight_trades_estimator as preflight_trades_estimator

from src.config.week4_trades import Week4TradesConfig
from src.data.db.models import StockTradesSampled
from src.data.db.session import get_session_factory
from src.features.trade_microstructure import (
    compute_knowledge_time,
    compute_large_trade_ratio,
    compute_late_day_aggressiveness,
    compute_off_exchange_volume_ratio,
    compute_offhours_trade_ratio,
    compute_trade_imbalance_proxy,
)

DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_OUTPUT_PATH = Path("data/features/trade_microstructure_features.parquet")
DEFAULT_START_DATE = date(2016, 4, 17)
DEFAULT_END_DATE = date(2026, 4, 17)

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
    parser = argparse.ArgumentParser(description="Build Week 4 trade microstructure features from sampled trades.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute all (ticker, event_date) rows even if output parquet already has them.",
    )
    return parser.parse_args(argv)


def iter_ticker_dates(session: Session, start_date: date, end_date: date) -> Iterable[tuple[str, date]]:
    """流式返回 (ticker, trading_date) 对, 按 (trading_date, ticker) 排序."""
    stmt = (
        select(StockTradesSampled.ticker, StockTradesSampled.trading_date)
        .where(StockTradesSampled.trading_date >= start_date)
        .where(StockTradesSampled.trading_date <= end_date)
        .group_by(StockTradesSampled.ticker, StockTradesSampled.trading_date)
        .order_by(StockTradesSampled.trading_date, StockTradesSampled.ticker)
    )
    for row in session.execute(stmt):
        yield str(row[0]).upper(), row[1]


def load_trades_for_group(session: Session, ticker: str, trading_date: date) -> pd.DataFrame:
    """加载 (ticker, trading_date) 的所有 trades 构造 DataFrame, 供 Task 7 特征函数消费."""
    stmt = (
        select(
            StockTradesSampled.sip_timestamp,
            StockTradesSampled.price,
            StockTradesSampled.size,
            StockTradesSampled.exchange,
            StockTradesSampled.trf_id,
            StockTradesSampled.trf_timestamp,
            StockTradesSampled.conditions,
        )
        .where(StockTradesSampled.ticker == ticker)
        .where(StockTradesSampled.trading_date == trading_date)
        .order_by(StockTradesSampled.sip_timestamp)
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(
            columns=["sip_timestamp", "price", "size", "exchange", "trf_id", "trf_timestamp", "conditions"],
        )
    frame = pd.DataFrame.from_records(
        rows,
        columns=["sip_timestamp", "price", "size", "exchange", "trf_id", "trf_timestamp", "conditions"],
    )
    return frame


def compute_feature_row(
    ticker: str,
    event_date: date,
    trades: pd.DataFrame,
    config: Week4TradesConfig,
    config_hash: str,
) -> dict[str, Any]:
    """Compute 5 trade microstructure features + PIT knowledge times for one (ticker, event_date)."""
    condition_allow = set(config.features.condition_allow_list) or None
    trf_exchange_codes = set(config.features.trf_exchange_codes)
    late_day_window = tuple(config.features.late_day_window_et)
    pre_window = tuple(config.features.offhours_window_et_pre)
    post_window = tuple(config.features.offhours_window_et_post)
    size_threshold = float(config.features.size_threshold_dollars)

    trade_imbalance = compute_trade_imbalance_proxy(trades, condition_allow=condition_allow)
    large_trade = compute_large_trade_ratio(trades, size_threshold_dollars=size_threshold)
    late_day = compute_late_day_aggressiveness(trades, late_day_window_et=late_day_window)
    offhours = compute_offhours_trade_ratio(trades, pre_window_et=pre_window, post_window_et=post_window)
    off_exchange = compute_off_exchange_volume_ratio(trades, trf_exchange_codes=trf_exchange_codes)

    return {
        "event_date": event_date,
        "ticker": ticker.upper(),
        "knowledge_time_regular": compute_knowledge_time(event_date, "trade_imbalance_proxy"),
        "knowledge_time_offhours": compute_knowledge_time(event_date, "offhours_trade_ratio"),
        "trade_imbalance_proxy": trade_imbalance,
        "large_trade_ratio": large_trade,
        "late_day_aggressiveness": late_day,
        "offhours_trade_ratio": offhours,
        "off_exchange_volume_ratio": off_exchange,
        "run_config_hash": config_hash,
    }


def load_existing_rows(output_path: Path, config_hash: str) -> tuple[pd.DataFrame, set[tuple[str, date]]]:
    """读已有 parquet; 若 config_hash 不匹配, 返空 frame + 空 set 触发全量重算."""
    if not output_path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS), set()
    existing = pd.read_parquet(output_path)
    if existing.empty:
        return existing, set()
    existing_hashes = set(existing["run_config_hash"].astype(str).unique())
    if existing_hashes != {config_hash}:
        logger.warning(
            "existing parquet config_hash {} != current {} — full recompute",
            sorted(existing_hashes),
            config_hash,
        )
        return pd.DataFrame(columns=OUTPUT_COLUMNS), set()
    existing["event_date"] = pd.to_datetime(existing["event_date"]).dt.date
    existing["ticker"] = existing["ticker"].astype(str).str.upper()
    completed = {(row.ticker, row.event_date) for row in existing.itertuples(index=False)}
    return existing, completed


def write_features(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame[OUTPUT_COLUMNS].copy()
    ordered["event_date"] = pd.to_datetime(ordered["event_date"]).dt.date
    ordered.sort_values(["event_date", "ticker"], inplace=True)
    ordered.reset_index(drop=True, inplace=True)
    ordered.to_parquet(output_path, index=False)


def build_features(
    *,
    config: Week4TradesConfig,
    config_hash: str,
    start_date: date,
    end_date: date,
    output_path: Path,
    session_factory: Callable[[], Session] | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """Main loop — iterate (ticker, event_date) pairs, compute features, write parquet.

    Returns summary dict with counts (skipped, computed, written).
    """
    factory = session_factory or get_session_factory()
    if resume:
        existing_frame, completed = load_existing_rows(output_path, config_hash)
    else:
        existing_frame, completed = pd.DataFrame(columns=OUTPUT_COLUMNS), set()

    new_rows: list[dict[str, Any]] = []
    skipped = 0
    groups_processed = 0
    with factory() as session:
        for ticker, event_date in iter_ticker_dates(session, start_date, end_date):
            groups_processed += 1
            if resume and (ticker, event_date) in completed:
                skipped += 1
                continue
            trades = load_trades_for_group(session, ticker, event_date)
            if trades.empty:
                continue
            row = compute_feature_row(ticker, event_date, trades, config, config_hash)
            new_rows.append(row)

    new_frame = pd.DataFrame(new_rows, columns=OUTPUT_COLUMNS) if new_rows else pd.DataFrame(columns=OUTPUT_COLUMNS)
    combined = pd.concat([existing_frame, new_frame], ignore_index=True) if not existing_frame.empty else new_frame
    if not combined.empty:
        combined = combined.drop_duplicates(["event_date", "ticker"], keep="last")
        write_features(combined, output_path)

    summary = {
        "groups_processed": groups_processed,
        "groups_skipped_resume": skipped,
        "rows_computed": len(new_rows),
        "rows_total": len(combined),
        "config_hash": config_hash,
        "output_path": str(output_path),
    }
    logger.info("trade microstructure features summary: {}", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = preflight_trades_estimator.load_config(args.config)
    config_hash = preflight_trades_estimator.compute_config_hash(config)
    summary = build_features(
        config=config,
        config_hash=config_hash,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover - direct script execution
    sys.exit(main())
