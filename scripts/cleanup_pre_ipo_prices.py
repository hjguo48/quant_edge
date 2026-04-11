from __future__ import annotations

import argparse

from loguru import logger
import sqlalchemy as sa

from src.data.db.session import get_session_factory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean up pre-IPO ghost prices from stock_prices.")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete matching rows. Default behavior is dry-run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Preview row limit for dry-run output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_factory = get_session_factory()

    count_sql = sa.text(
        """
        SELECT COUNT(*)
        FROM stock_prices sp
        JOIN stocks s ON s.ticker = sp.ticker
        WHERE s.ipo_date IS NOT NULL
          AND sp.trade_date < s.ipo_date
        """,
    )
    preview_sql = sa.text(
        """
        SELECT sp.ticker, sp.trade_date, s.ipo_date, sp.close, sp.adj_close
        FROM stock_prices sp
        JOIN stocks s ON s.ticker = sp.ticker
        WHERE s.ipo_date IS NOT NULL
          AND sp.trade_date < s.ipo_date
        ORDER BY sp.ticker, sp.trade_date
        LIMIT :limit
        """,
    )
    delete_sql = sa.text(
        """
        DELETE FROM stock_prices sp
        USING stocks s
        WHERE s.ticker = sp.ticker
          AND s.ipo_date IS NOT NULL
          AND sp.trade_date < s.ipo_date
        """,
    )

    with session_factory() as session:
        row_count = int(session.execute(count_sql).scalar() or 0)
        preview_rows = session.execute(preview_sql, {"limit": args.limit}).all()

        logger.info("found {} pre-IPO stock_prices rows", row_count)
        for ticker, trade_date, ipo_date, close, adj_close in preview_rows:
            logger.info(
                "preview ticker={} trade_date={} ipo_date={} close={} adj_close={}",
                ticker,
                trade_date,
                ipo_date,
                close,
                adj_close,
            )

        if not args.apply:
            logger.info("dry-run mode, no rows deleted")
            return

        session.execute(delete_sql)
        session.commit()
        logger.info("deleted {} pre-IPO stock_prices rows", row_count)


if __name__ == "__main__":
    main()
