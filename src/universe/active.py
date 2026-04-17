from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import text

from src.data.db.session import get_engine
from src.universe.builder import backfill_universe_membership

DEFAULT_INDEX_NAME = "SP500"
DEFAULT_BENCHMARK_TICKER = "SPY"


def _coerce_trade_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def resolve_active_universe(
    trade_date: date | datetime,
    *,
    as_of: date | datetime | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    index_name: str = DEFAULT_INDEX_NAME,
) -> tuple[list[str], str]:
    """Resolve the live tradable universe from a single shared code path.

    The preferred source is ``universe_membership``. If it is stale or empty,
    the fallback is the ``stocks`` table filtered to names that still have PIT-
    visible price data on the requested trade date.
    """

    effective_trade_date = _coerce_trade_date(trade_date)
    benchmark = benchmark_ticker.upper()
    as_of_value = as_of or datetime.combine(effective_trade_date, datetime.max.time())

    membership_sql = text(
        """
        select distinct ticker
        from universe_membership
        where index_name = :index_name
          and effective_date <= :trade_date
          and (end_date is null or end_date > :trade_date)
          and upper(ticker) <> :benchmark
        order by ticker
        """,
    )
    membership_with_prices_sql = text(
        """
        with active_members as (
            select distinct ticker
            from universe_membership
            where index_name = :index_name
              and effective_date <= :trade_date
              and (end_date is null or end_date > :trade_date)
              and upper(ticker) <> :benchmark
        )
        select distinct sp.ticker
        from stock_prices sp
        join active_members am on am.ticker = sp.ticker
        where sp.trade_date = :trade_date
          and sp.knowledge_time <= :as_of
        order by sp.ticker
        """,
    )
    stocks_with_prices_sql = text(
        """
        select distinct s.ticker
        from stocks s
        join stock_prices sp on sp.ticker = s.ticker
        where s.ticker is not null
          and upper(s.ticker) <> :benchmark
          and sp.trade_date = :trade_date
          and sp.knowledge_time <= :as_of
        order by s.ticker
        """,
    )
    stocks_sql = text(
        """
        select ticker
        from stocks
        where ticker is not null
          and upper(ticker) <> :benchmark
        order by ticker
        """,
    )

    engine = get_engine()
    with engine.connect() as conn:
        membership_rows = conn.execute(
            membership_sql,
            {
                "index_name": index_name,
                "trade_date": effective_trade_date,
                "benchmark": benchmark,
            },
        ).scalars().all()
        if membership_rows:
            price_visible_rows = conn.execute(
                membership_with_prices_sql,
                {
                    "index_name": index_name,
                    "trade_date": effective_trade_date,
                    "benchmark": benchmark,
                    "as_of": as_of_value,
                },
            ).scalars().all()
            if price_visible_rows:
                return [str(ticker).upper() for ticker in price_visible_rows], "universe_membership"
            return [str(ticker).upper() for ticker in membership_rows], "universe_membership_no_price_filter"

        stocks_with_prices = conn.execute(
            stocks_with_prices_sql,
            {
                "trade_date": effective_trade_date,
                "benchmark": benchmark,
                "as_of": as_of_value,
            },
        ).scalars().all()
        if stocks_with_prices:
            return [str(ticker).upper() for ticker in stocks_with_prices], "stocks_price_fallback"

        stock_rows = conn.execute(
            stocks_sql,
            {"benchmark": benchmark},
        ).scalars().all()
    return [str(ticker).upper() for ticker in stock_rows], "stocks_fallback"


def get_active_universe(
    trade_date: date | datetime,
    *,
    as_of: date | datetime | None = None,
    benchmark_ticker: str = DEFAULT_BENCHMARK_TICKER,
    index_name: str = DEFAULT_INDEX_NAME,
) -> list[str]:
    tickers, _ = resolve_active_universe(
        trade_date,
        as_of=as_of,
        benchmark_ticker=benchmark_ticker,
        index_name=index_name,
    )
    return tickers


def ensure_monthly_universe_membership(
    as_of: date | datetime,
    *,
    index_name: str = DEFAULT_INDEX_NAME,
) -> dict[str, Any]:
    """Ensure the current month has live universe membership coverage."""

    effective_date = _coerce_trade_date(as_of)
    month_start = effective_date.replace(day=1)
    engine = get_engine()
    with engine.connect() as conn:
        active_count = int(
            conn.execute(
                text(
                    """
                    select count(distinct ticker)
                    from universe_membership
                    where index_name = :index_name
                      and effective_date <= :trade_date
                      and (end_date is null or end_date > :trade_date)
                    """,
                ),
                {"index_name": index_name, "trade_date": effective_date},
            ).scalar()
            or 0,
        )
        current_month_rows = int(
            conn.execute(
                text(
                    """
                    select count(*)
                    from universe_membership
                    where index_name = :index_name
                      and effective_date >= :month_start
                      and effective_date <= :trade_date
                    """,
                ),
                {
                    "index_name": index_name,
                    "month_start": month_start,
                    "trade_date": effective_date,
                },
            ).scalar()
            or 0,
        )

    if active_count > 0 and current_month_rows > 0:
        return {
            "status": "up_to_date",
            "rows_written": 0,
            "active_count": active_count,
            "month_start": month_start.isoformat(),
            "trade_date": effective_date.isoformat(),
        }

    rows_written = backfill_universe_membership(
        start_date=month_start,
        end_date=effective_date,
        index_name=index_name,
    )
    return {
        "status": "refreshed",
        "rows_written": int(rows_written),
        "active_count": active_count,
        "month_start": month_start.isoformat(),
        "trade_date": effective_date.isoformat(),
    }
