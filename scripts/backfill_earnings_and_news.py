"""Backfill all alternative data sources for S&P 500 universe.

Usage:
    python scripts/backfill_earnings_and_news.py --source earnings --start 2018-01-01
    python scripts/backfill_earnings_and_news.py --source news --start 2020-01-01
    python scripts/backfill_earnings_and_news.py --source analyst --start 2018-01-01
    python scripts/backfill_earnings_and_news.py --source short-interest --start 2017-01-01
    python scripts/backfill_earnings_and_news.py --source insider --start 2018-01-01
    python scripts/backfill_earnings_and_news.py --source sec-filings --start 2018-01-01
    python scripts/backfill_earnings_and_news.py --source all --start 2018-01-01
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def get_sp500_tickers() -> list[str]:
    """Get all tickers that have ever been in our universe (stocks table)."""
    from src.data.db.session import get_engine
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        # Use stocks table for full coverage (includes historical + current members)
        result = conn.execute(
            text("SELECT ticker FROM stocks ORDER BY ticker")
        )
        tickers = [row[0] for row in result]

    logger.info("Found {} tickers for backfill", len(tickers))
    return tickers


def get_current_universe_tickers() -> list[str]:
    """Get the current active S&P 500-style universe from universe_membership."""
    from src.data.db.session import get_engine
    from src.universe.active import resolve_active_universe
    from sqlalchemy import text

    engine = get_engine()
    with engine.connect() as conn:
        latest_trade_date = conn.execute(text("SELECT max(trade_date) FROM stock_prices")).scalar() or date.today()
    tickers, source = resolve_active_universe(latest_trade_date)
    if not tickers:
        logger.warning("No active universe tickers found; falling back to full stocks table.")
        return get_sp500_tickers()
    logger.info(
        "Found {} current universe tickers for incremental refresh from {} (trade_date={})",
        len(tickers),
        source,
        latest_trade_date,
    )
    return tickers


def create_tables() -> None:
    """Create all alternative data tables if they don't exist."""
    from src.data.db.session import get_engine
    from src.data.sources.fmp_earnings import EarningsEstimate
    from src.data.sources.polygon_news import NewsSentiment
    from src.data.sources.fmp_analyst import AnalystEstimate
    from src.data.sources.polygon_short_interest import ShortInterest
    from src.data.sources.fmp_insider import InsiderTrade
    from src.data.sources.fmp_sec_filings import SecFiling

    engine = get_engine()
    for model in [EarningsEstimate, NewsSentiment, AnalystEstimate, ShortInterest, InsiderTrade, SecFiling]:
        model.__table__.create(engine, checkfirst=True)
    logger.info("Ensured all alternative data tables exist")


def get_incremental_since_date(source: str, end_date: date) -> date:
    """Resolve a conservative incremental start date from table state."""
    from src.data.db.session import get_engine
    from sqlalchemy import text

    default_lookbacks = {
        "earnings": 180,
        "short-interest": 45,
        "insider": 120,
    }
    if source == "analyst":
        # Analyst estimates are forward-looking snapshots; refresh a recent anchor window
        # and let the source repopulate the near/future quarters on each daily run.
        return end_date - timedelta(days=30)

    table_map = {
        "earnings": "earnings_estimates",
        "short-interest": "short_interest",
        "insider": "insider_trades",
    }
    table_name = table_map[source]
    engine = get_engine()
    with engine.connect() as conn:
        latest_knowledge_time = conn.execute(
            text(f"SELECT max(knowledge_time) FROM {table_name}"),
        ).scalar()

    if latest_knowledge_time is None:
        return end_date - timedelta(days=default_lookbacks[source])

    if isinstance(latest_knowledge_time, datetime):
        latest_date = latest_knowledge_time.date()
    else:
        latest_date = latest_knowledge_time
    latest_date = min(latest_date, end_date)

    # Small overlap prevents off-by-one / late-arriving updates from being missed.
    return latest_date - timedelta(days=2)


def backfill_earnings(
    tickers: list[str],
    start_date: date,
    end_date: date,
    batch_size: int = 25,
) -> int:
    """Backfill earnings estimates for all tickers."""
    from src.data.sources.fmp_earnings import FMPEarningsSource

    source = FMPEarningsSource()
    total_rows = 0

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size

        logger.info(
            "earnings batch {}/{}: {} tickers ({} -> {})",
            batch_num,
            total_batches,
            len(batch),
            batch[0],
            batch[-1],
        )

        try:
            frame = source.fetch_historical(batch, start_date, end_date)
            total_rows += len(frame)
            logger.info(
                "earnings batch {}/{} done: {} rows (total: {})",
                batch_num,
                total_batches,
                len(frame),
                total_rows,
            )
        except Exception as exc:
            logger.error(
                "earnings batch {}/{} failed: {}",
                batch_num,
                total_batches,
                exc,
            )

    return total_rows


def backfill_news(
    tickers: list[str],
    start_date: date,
    end_date: date,
    batch_size: int = 10,
) -> int:
    """Backfill news sentiment for all tickers."""
    from src.data.sources.polygon_news import PolygonNewsSource

    source = PolygonNewsSource()
    total_rows = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info(
            "news {}/{}: {}",
            i,
            len(tickers),
            ticker,
        )

        try:
            frame = source.fetch_historical([ticker], start_date, end_date)
            total_rows += len(frame)
            if i % 50 == 0:
                logger.info("news progress: {}/{} tickers, {} total rows", i, len(tickers), total_rows)
        except Exception as exc:
            logger.error("news {} failed: {}", ticker, exc)

    return total_rows


def backfill_analyst(
    tickers: list[str], start_date: date, end_date: date, batch_size: int = 25,
) -> int:
    from src.data.sources.fmp_analyst import FMPAnalystSource
    source = FMPAnalystSource()
    total = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        bn = i // batch_size + 1
        tb = (len(tickers) + batch_size - 1) // batch_size
        try:
            frame = source.fetch_historical(batch, start_date, end_date)
            total += len(frame)
            logger.info("analyst batch {}/{}: {} rows (total: {})", bn, tb, len(frame), total)
        except Exception as exc:
            logger.error("analyst batch {}/{} failed: {}", bn, tb, exc)
    return total


def backfill_short_interest(
    tickers: list[str], start_date: date, end_date: date,
) -> int:
    from src.data.sources.polygon_short_interest import PolygonShortInterestSource
    source = PolygonShortInterestSource()
    total = 0
    for i, ticker in enumerate(tickers, 1):
        try:
            frame = source.fetch_historical([ticker], start_date, end_date)
            total += len(frame)
            if i % 50 == 0:
                logger.info("short-interest progress: {}/{}, {} total rows", i, len(tickers), total)
        except Exception as exc:
            logger.error("short-interest {} failed: {}", ticker, exc)
    return total


def backfill_insider(
    tickers: list[str], start_date: date, end_date: date,
) -> int:
    from src.data.sources.fmp_insider import FMPInsiderSource
    source = FMPInsiderSource()
    total = 0
    for i, ticker in enumerate(tickers, 1):
        try:
            frame = source.fetch_historical([ticker], start_date, end_date)
            total += len(frame)
            if i % 50 == 0:
                logger.info("insider progress: {}/{}, {} total rows", i, len(tickers), total)
        except Exception as exc:
            logger.error("insider {} failed: {}", ticker, exc)
    return total


def backfill_sec_filings(
    tickers: list[str], start_date: date, end_date: date,
) -> int:
    from src.data.sources.fmp_sec_filings import FMPSecFilingsSource
    source = FMPSecFilingsSource()
    total = 0
    for i, ticker in enumerate(tickers, 1):
        try:
            frame = source.fetch_historical([ticker], start_date, end_date)
            total += len(frame)
            if i % 50 == 0:
                logger.info("sec-filings progress: {}/{}, {} total rows", i, len(tickers), total)
        except Exception as exc:
            logger.error("sec-filings {} failed: {}", ticker, exc)
    return total


def backfill_etf_prices(start_date: date, end_date: date) -> int:
    from src.data.sources.polygon import PolygonDataSource
    from src.features.sector_rotation import SECTOR_ROTATION_ETF_TICKERS

    source = PolygonDataSource()
    tickers = list(SECTOR_ROTATION_ETF_TICKERS)
    frame = source.fetch_historical(tickers, start_date, end_date)
    logger.info(
        "ETF price backfill complete for {} tickers: {} rows",
        len(tickers),
        len(frame),
    )
    return int(len(frame))


def incremental_earnings(
    tickers: list[str],
    end_date: date,
    batch_size: int = 25,
) -> tuple[int, date]:
    from src.data.sources.fmp_earnings import FMPEarningsSource

    source = FMPEarningsSource()
    since_date = get_incremental_since_date("earnings", end_date)
    total_rows = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        try:
            frame = source.fetch_incremental(batch, since_date)
            total_rows += len(frame)
            logger.info(
                "earnings incremental batch {}/{}: {} rows (total: {}) since {}",
                batch_num,
                total_batches,
                len(frame),
                total_rows,
                since_date,
            )
        except Exception as exc:
            logger.error("earnings incremental batch {}/{} failed: {}", batch_num, total_batches, exc)
    return total_rows, since_date


def incremental_short_interest(
    tickers: list[str],
    end_date: date,
) -> tuple[int, date]:
    from src.data.sources.polygon_short_interest import PolygonShortInterestSource

    source = PolygonShortInterestSource()
    since_date = get_incremental_since_date("short-interest", end_date)
    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        try:
            frame = source.fetch_incremental([ticker], since_date)
            total_rows += len(frame)
            if i % 50 == 0:
                logger.info(
                    "short-interest incremental progress: {}/{} tickers, {} total rows since {}",
                    i,
                    len(tickers),
                    total_rows,
                    since_date,
                )
        except Exception as exc:
            logger.error("short-interest incremental {} failed: {}", ticker, exc)
    return total_rows, since_date


def incremental_insider(
    tickers: list[str],
    end_date: date,
) -> tuple[int, date]:
    from src.data.sources.fmp_insider import FMPInsiderSource

    source = FMPInsiderSource()
    since_date = get_incremental_since_date("insider", end_date)
    total_rows = 0
    for i, ticker in enumerate(tickers, 1):
        try:
            frame = source.fetch_incremental([ticker], since_date)
            total_rows += len(frame)
            if i % 50 == 0:
                logger.info(
                    "insider incremental progress: {}/{} tickers, {} total rows since {}",
                    i,
                    len(tickers),
                    total_rows,
                    since_date,
                )
        except Exception as exc:
            logger.error("insider incremental {} failed: {}", ticker, exc)
    return total_rows, since_date


def incremental_analyst(
    tickers: list[str],
    end_date: date,
    batch_size: int = 25,
) -> tuple[int, date, date]:
    from src.data.sources.fmp_analyst import FMPAnalystSource

    source = FMPAnalystSource()
    refresh_start = get_incremental_since_date("analyst", end_date)
    forward_end = end_date + timedelta(days=365 * 3)
    total_rows = 0
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(tickers) + batch_size - 1) // batch_size
        try:
            frame = source.fetch_historical(batch, refresh_start, forward_end)
            total_rows += len(frame)
            logger.info(
                "analyst incremental batch {}/{}: {} rows (total: {}) fiscal_date {} -> {}",
                batch_num,
                total_batches,
                len(frame),
                total_rows,
                refresh_start,
                forward_end,
            )
        except Exception as exc:
            logger.error("analyst incremental batch {}/{} failed: {}", batch_num, total_batches, exc)
    return total_rows, refresh_start, forward_end


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill earnings and news sentiment data")
    parser.add_argument(
        "--source",
        choices=["earnings", "news", "analyst", "short-interest", "insider", "sec-filings", "etf-prices", "all"],
        default="all",
        help="Which data source to backfill",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2018-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        nargs="*",
        default=None,
        help="Specific tickers (default: all SP500)",
    )
    parser.add_argument(
        "--earnings-batch-size",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run source-specific incremental refresh instead of historical backfill.",
    )
    parser.add_argument(
        "--current-universe",
        action="store_true",
        help="Use the current active universe_membership tickers instead of the full stocks table.",
    )
    args = parser.parse_args(argv)

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end) if args.end else date.today()
    default_tickers = get_current_universe_tickers if args.current_universe else get_sp500_tickers
    tickers = args.tickers or default_tickers()

    logger.info(
        "Backfilling {} for {} tickers from {} to {}",
        args.source,
        len(tickers),
        start_date,
        end_date,
    )

    create_tables()

    if args.source in ("earnings", "all") and args.incremental:
        t0 = time.monotonic()
        n, since_date = incremental_earnings(tickers, end_date, batch_size=args.earnings_batch_size)
        logger.info(
            "Earnings incremental complete: {} rows in {:.0f}s since {}",
            n,
            time.monotonic() - t0,
            since_date,
        )
    elif args.source in ("earnings", "all"):
        t0 = time.monotonic()
        n = backfill_earnings(tickers, start_date, end_date, batch_size=args.earnings_batch_size)
        logger.info("Earnings backfill complete: {} rows in {:.0f}s", n, time.monotonic() - t0)

    if args.source in ("news", "all"):
        news_start = max(start_date, date(2020, 1, 1))
        t0 = time.monotonic()
        n = backfill_news(tickers, news_start, end_date)
        logger.info("News backfill complete: {} rows in {:.0f}s", n, time.monotonic() - t0)

    if args.source in ("analyst", "all") and args.incremental:
        t0 = time.monotonic()
        n, refresh_start, forward_end = incremental_analyst(
            tickers,
            end_date,
            batch_size=args.earnings_batch_size,
        )
        logger.info(
            "Analyst estimates incremental complete: {} rows in {:.0f}s fiscal_date {} -> {}",
            n,
            time.monotonic() - t0,
            refresh_start,
            forward_end,
        )
    elif args.source in ("analyst", "all"):
        t0 = time.monotonic()
        n = backfill_analyst(tickers, start_date, end_date, batch_size=args.earnings_batch_size)
        logger.info("Analyst estimates backfill complete: {} rows in {:.0f}s", n, time.monotonic() - t0)

    if args.source in ("short-interest", "all") and args.incremental:
        t0 = time.monotonic()
        n, since_date = incremental_short_interest(tickers, end_date)
        logger.info(
            "Short interest incremental complete: {} rows in {:.0f}s since {}",
            n,
            time.monotonic() - t0,
            since_date,
        )
    elif args.source in ("short-interest", "all"):
        t0 = time.monotonic()
        n = backfill_short_interest(tickers, start_date, end_date)
        logger.info("Short interest backfill complete: {} rows in {:.0f}s", n, time.monotonic() - t0)

    if args.source in ("insider", "all") and args.incremental:
        t0 = time.monotonic()
        n, since_date = incremental_insider(tickers, end_date)
        logger.info(
            "Insider trading incremental complete: {} rows in {:.0f}s since {}",
            n,
            time.monotonic() - t0,
            since_date,
        )
    elif args.source in ("insider", "all"):
        t0 = time.monotonic()
        n = backfill_insider(tickers, start_date, end_date)
        logger.info("Insider trading backfill complete: {} rows in {:.0f}s", n, time.monotonic() - t0)

    if args.source in ("sec-filings", "all"):
        t0 = time.monotonic()
        n = backfill_sec_filings(tickers, start_date, end_date)
        logger.info("SEC filings backfill complete: {} rows in {:.0f}s", n, time.monotonic() - t0)

    if args.source in ("etf-prices", "all"):
        t0 = time.monotonic()
        n = backfill_etf_prices(start_date, end_date)
        logger.info("ETF price backfill complete: {} rows in {:.0f}s", n, time.monotonic() - t0)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
