from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sys

import sqlalchemy as sa
from loguru import logger
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.models import Stock, UniverseMembership
from src.data.db.session import get_engine, get_session_factory
from src.universe.builder import backfill_universe_membership


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    verify_as_of = args.verify_as_of or min(args.end_date, date.today())
    logger.info(
        "populating {} membership from {} to {} verify_as_of={}",
        args.index_name,
        args.start_date,
        args.end_date,
        verify_as_of,
    )

    method = "builder"
    rows_written = 0
    try:
        if args.force_stocks_fallback:
            raise RuntimeError("forced stocks fallback")
        rows_written = backfill_universe_membership(
            args.start_date,
            args.end_date,
            index_name=args.index_name,
        )
    except Exception as exc:
        if args.no_stocks_fallback:
            raise
        logger.warning("builder backfill failed; falling back to current stocks snapshot: {}", exc)
        rows_written = populate_from_current_stocks(
            start_date=args.start_date,
            index_name=args.index_name,
        )
        method = "stocks_fallback"

    active_count = count_active_members(
        as_of=verify_as_of,
        index_name=args.index_name,
    )
    logger.info(
        "universe population completed method={} rows_written={} active_count_on_{}={}",
        method,
        rows_written,
        verify_as_of,
        active_count,
    )
    if active_count <= 0:
        raise RuntimeError(
            f"{args.index_name} membership is still empty as of {verify_as_of.isoformat()} after {method}.",
        )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill universe_membership using the existing builder with a stocks-table fallback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", type=parse_date, default=date(2024, 1, 1))
    parser.add_argument("--end-date", type=parse_date, default=date.today())
    parser.add_argument("--verify-as-of", type=parse_date, default=date(2026, 3, 31))
    parser.add_argument("--index-name", default="SP500")
    parser.add_argument("--no-stocks-fallback", action="store_true")
    parser.add_argument("--force-stocks-fallback", action="store_true")
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def count_active_members(*, as_of: date, index_name: str) -> int:
    engine = get_engine()
    with engine.connect() as conn:
        return int(
            conn.execute(
                text(
                    """
                    select count(*)
                    from universe_membership
                    where index_name = :index_name
                      and effective_date <= :as_of
                      and (end_date is null or end_date > :as_of)
                    """,
                ),
                {"index_name": index_name, "as_of": as_of},
            ).scalar()
            or 0,
        )


def populate_from_current_stocks(*, start_date: date, index_name: str) -> int:
    session_factory = get_session_factory()
    rows = load_current_stock_rows(index_name=index_name, start_date=start_date)
    if not rows:
        raise RuntimeError("No current stocks were available for fallback universe population.")

    with session_factory() as session:
        try:
            session.execute(
                sa.delete(UniverseMembership).where(
                    UniverseMembership.index_name == index_name,
                    sa.or_(
                        UniverseMembership.effective_date >= start_date,
                        sa.and_(
                            UniverseMembership.effective_date < start_date,
                            sa.or_(
                                UniverseMembership.end_date.is_(None),
                                UniverseMembership.end_date > start_date,
                            ),
                        ),
                    ),
                ),
            )
            statement = insert(UniverseMembership).values(rows)
            upsert = statement.on_conflict_do_update(
                constraint="uq_universe_membership_entry",
                set_={
                    "end_date": statement.excluded.end_date,
                    "reason": statement.excluded.reason,
                },
            )
            session.execute(upsert)
            session.commit()
        except Exception:
            session.rollback()
            raise

    return len(rows)


def load_current_stock_rows(*, index_name: str, start_date: date) -> list[dict[str, object]]:
    engine = get_engine()
    with engine.connect() as conn:
        tickers = [
            str(row[0]).upper()
            for row in conn.execute(
                sa.select(Stock.ticker)
                .where(
                    Stock.ticker.is_not(None),
                    Stock.ticker != "SPY",
                    sa.or_(Stock.delist_date.is_(None), Stock.delist_date > start_date),
                )
                .order_by(Stock.ticker),
            ).all()
        ]
    return [
        {
            "ticker": ticker,
            "index_name": index_name,
            "effective_date": start_date,
            "end_date": None,
            "reason": "stocks_fallback_backfill",
        }
        for ticker in tickers
    ]


if __name__ == "__main__":
    raise SystemExit(main())
