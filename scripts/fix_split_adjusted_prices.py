from __future__ import annotations

import argparse
from datetime import date
import sys

from loguru import logger
import sqlalchemy as sa

from _data_ops import configure_logging, current_market_data_end_date, ensure_tables_exist, parse_date
from src.data.db.models import StockPrice
from src.data.db.session import get_session_factory
from src.data.sources.polygon import PolygonDataSource, normalize_polygon_ticker

DEFAULT_TICKER = "BKNG"
DEFAULT_START_DATE = date(2015, 1, 2)
DEFAULT_PRE_SPLIT_DATE = date(2026, 4, 2)
DEFAULT_POST_SPLIT_DATE = date(2026, 4, 6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-fetch fully adjusted Polygon price history for a split ticker and upsert it into stock_prices.",
    )
    parser.add_argument("--ticker", default=DEFAULT_TICKER)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE.isoformat())
    parser.add_argument("--end-date")
    parser.add_argument("--pre-check-date", default=DEFAULT_PRE_SPLIT_DATE.isoformat())
    parser.add_argument("--post-check-date", default=DEFAULT_POST_SPLIT_DATE.isoformat())
    parser.add_argument(
        "--max-gap-pct",
        type=float,
        default=0.10,
        help="Maximum allowed absolute close-to-close jump between the verification dates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging("fix_split_adjusted_prices")
    ensure_tables_exist(required_tables=("stocks", "stock_prices"))

    ticker = normalize_polygon_ticker(args.ticker)
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date) if args.end_date else current_market_data_end_date()
    pre_check_date = parse_date(args.pre_check_date)
    post_check_date = parse_date(args.post_check_date)

    logger.info(
        "re-fetching fully adjusted Polygon history for {} between {} and {}",
        ticker,
        start_date,
        end_date,
    )
    source = PolygonDataSource(min_request_interval=0.0)
    frame = source.fetch_adjusted_historical([ticker], start_date, end_date)
    if frame.empty:
        logger.error("no adjusted rows returned for {}", ticker)
        return 2

    logger.info("persisted {} fully adjusted rows for {}", len(frame), ticker)
    verification = _load_verification_closes(
        ticker=ticker,
        dates=(pre_check_date, post_check_date),
    )
    if pre_check_date not in verification or post_check_date not in verification:
        logger.error(
            "missing verification closes for {} on {} or {}",
            ticker,
            pre_check_date,
            post_check_date,
        )
        return 2

    pre_close = verification[pre_check_date]
    post_close = verification[post_check_date]
    if pre_close == 0:
        logger.error("pre-split verification close is zero for {} on {}", ticker, pre_check_date)
        return 2

    pct_change = (post_close / pre_close) - 1.0
    logger.info(
        "verification {} close {:.4f} -> {} close {:.4f} ({:+.2%})",
        pre_check_date,
        pre_close,
        post_check_date,
        post_close,
        pct_change,
    )
    if abs(pct_change) > args.max_gap_pct:
        logger.error(
            "adjusted close continuity check failed for {}: absolute jump {:+.2%} exceeds {:.2%}",
            ticker,
            pct_change,
            args.max_gap_pct,
        )
        return 2

    logger.info("adjusted split repair verified successfully for {}", ticker)
    return 0


def _load_verification_closes(*, ticker: str, dates: tuple[date, date]) -> dict[date, float]:
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(
            sa.select(StockPrice.trade_date, StockPrice.close)
            .where(
                StockPrice.ticker == ticker,
                StockPrice.trade_date.in_(dates),
            )
            .order_by(StockPrice.trade_date),
        ).all()

    return {
        trade_date: float(close_value)
        for trade_date, close_value in rows
        if close_value is not None
    }


if __name__ == "__main__":
    sys.exit(main())
