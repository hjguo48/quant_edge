#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
import sys

from loguru import logger
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TICKERS_FILE = REPO_ROOT / "data/universe/sp500_tickers.txt"
DEFAULT_ENDPOINTS = ("grades", "ratings", "price_target", "earnings_calendar")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Week 5 FMP analyst-related sources into migration 008 tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tickers", help="Comma-separated ticker override for quick smoke runs.")
    parser.add_argument(
        "--tickers-file",
        default=str(DEFAULT_TICKERS_FILE),
        help="Ticker file, one symbol per line. Falls back to latest active universe when missing.",
    )
    parser.add_argument(
        "--endpoints",
        default=",".join(DEFAULT_ENDPOINTS),
        help="Comma-separated endpoint names: grades, ratings, price_target, earnings_calendar",
    )
    parser.add_argument("--start-date", help="Inclusive start date (YYYY-MM-DD). Mainly used for earnings_calendar.")
    parser.add_argument("--end-date", help="Inclusive end date (YYYY-MM-DD). Mainly used for earnings_calendar.")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and parse only; do not write to the database.")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def parse_date(raw_value: str | None, *, default: date) -> date:
    if raw_value is None:
        return default
    return date.fromisoformat(raw_value)


def parse_csv_list(raw_value: str) -> list[str]:
    return [segment.strip().upper() for segment in raw_value.split(",") if segment.strip()]


def resolve_tickers(args: argparse.Namespace) -> list[str]:
    if args.tickers:
        tickers = parse_csv_list(args.tickers)
        if tickers:
            return tickers

    ticker_file = Path(args.tickers_file)
    if ticker_file.exists():
        tickers = [
            line.strip().upper()
            for line in ticker_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if tickers:
            return tickers

    from src.data.db.session import get_engine
    from src.universe.active import resolve_active_universe

    engine = get_engine()
    with engine.connect() as connection:
        latest_trade_date = connection.execute(text("select max(trade_date) from stock_prices")).scalar()
    resolved_date = latest_trade_date or date.today()
    tickers, source = resolve_active_universe(resolved_date)
    logger.info("Resolved {} tickers from {} (trade_date={})", len(tickers), source, resolved_date)
    return tickers


def parse_endpoints(raw_value: str) -> list[str]:
    endpoints = [segment.strip().lower() for segment in raw_value.split(",") if segment.strip()]
    invalid = [endpoint for endpoint in endpoints if endpoint not in DEFAULT_ENDPOINTS]
    if invalid:
        raise ValueError(f"Unsupported endpoints: {invalid}. Expected subset of {DEFAULT_ENDPOINTS}.")
    return endpoints


def run_ticker_source(
    *,
    endpoint_name: str,
    source: object,
    tickers: list[str],
    start_date: date,
    end_date: date,
    dry_run: bool,
) -> int:
    total_rows = 0
    for index, ticker in enumerate(tickers, start=1):
        logger.info("{} {}/{}: {}", endpoint_name, index, len(tickers), ticker)
        if dry_run:
            frame = source.fetch_ticker(ticker)
            if not frame.empty and "event_date" in frame.columns:
                frame = frame.loc[(frame["event_date"] >= start_date) & (frame["event_date"] <= end_date)].copy()
            total_rows += len(frame)
            logger.info("{} {} dry-run rows={}", endpoint_name, ticker, len(frame))
            continue
        inserted = source.fetch_historical([ticker], start_date, end_date)
        total_rows += int(inserted)
        logger.info("{} {} inserted={}", endpoint_name, ticker, inserted)
    return total_rows


def run_earnings_calendar(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    dry_run: bool,
) -> int:
    from src.data.sources.fmp_earnings_calendar import FMPEarningsCalendarSource

    source = FMPEarningsCalendarSource()
    if dry_run:
        frame = source.fetch_range(start_date, end_date)
        filtered = frame.loc[frame["ticker"].isin(set(tickers))].copy()
        logger.info("earnings_calendar dry-run rows={}", len(filtered))
        return len(filtered)
    inserted = source.fetch_historical(tickers, start_date, end_date)
    logger.info("earnings_calendar inserted={}", inserted)
    return int(inserted)


def main(argv: list[str] | None = None) -> int:
    from src.data.sources.fmp_grades import FMPGradesSource
    from src.data.sources.fmp_price_target import FMPPriceTargetSource
    from src.data.sources.fmp_ratings import FMPRatingsSource

    configure_logging()
    args = parse_args(argv)
    tickers = resolve_tickers(args)
    endpoints = parse_endpoints(args.endpoints)
    end_date = parse_date(args.end_date, default=date.today())
    start_date = parse_date(args.start_date, default=end_date - timedelta(days=365))

    summary: dict[str, int] = {}
    for endpoint in endpoints:
        if endpoint == "grades":
            summary[endpoint] = run_ticker_source(
                endpoint_name=endpoint,
                source=FMPGradesSource(),
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                dry_run=args.dry_run,
            )
        elif endpoint == "ratings":
            summary[endpoint] = run_ticker_source(
                endpoint_name=endpoint,
                source=FMPRatingsSource(),
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                dry_run=args.dry_run,
            )
        elif endpoint == "price_target":
            summary[endpoint] = run_ticker_source(
                endpoint_name=endpoint,
                source=FMPPriceTargetSource(),
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                dry_run=args.dry_run,
            )
        elif endpoint == "earnings_calendar":
            summary[endpoint] = run_earnings_calendar(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                dry_run=args.dry_run,
            )

    logger.info("FMP analyst backfill summary: {}", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
