from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import sqlalchemy as sa
from loguru import logger
from sqlalchemy.orm import Session

from src.data.db.models import StockPrice
from src.data.db.session import get_session_factory
from src.universe.history import get_historical_members

EASTERN = ZoneInfo("America/New_York")


def _as_of_eod_utc(as_of_date: date) -> datetime:
    return datetime.combine(as_of_date, time(23, 59, 59, 999999), tzinfo=EASTERN).astimezone(
        timezone.utc,
    )


def get_top_liquidity_tickers(
    as_of_date: date,
    *,
    top_n: int = 200,
    lookback_days: int = 20,
    session_factory: Callable[[], Session] | None = None,
) -> list[str]:
    """Return PIT S&P 500 members ranked by recent average dollar volume.

    The membership list comes from ``get_historical_members``. Price rows are
    additionally constrained by ``knowledge_time <= as_of_date 23:59:59 ET`` so
    rows not visible at the as-of close cannot influence the ranking.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    members = tuple(get_historical_members(as_of_date, "SP500"))
    if not members:
        logger.warning("No historical universe members found for {}", as_of_date)
        return []

    # Fetch extra calendar days so sparse holidays/weekends still allow the
    # ranked CTE to select the last `lookback_days` available trading rows.
    history_start = as_of_date - timedelta(days=max(lookback_days * 3, 30))
    cutoff = _as_of_eod_utc(as_of_date)

    ranked = (
        sa.select(
            StockPrice.ticker.label("ticker"),
            StockPrice.trade_date.label("trade_date"),
            (StockPrice.close * StockPrice.volume).label("dollar_volume"),
            sa.func.row_number()
            .over(partition_by=StockPrice.ticker, order_by=StockPrice.trade_date.desc())
            .label("row_num"),
        )
        .where(
            StockPrice.ticker.in_(members),
            StockPrice.trade_date >= history_start,
            StockPrice.trade_date <= as_of_date,
            StockPrice.knowledge_time <= cutoff,
            StockPrice.close.is_not(None),
            StockPrice.volume.is_not(None),
        )
        .subquery()
    )

    statement = (
        sa.select(
            ranked.c.ticker,
            sa.func.avg(ranked.c.dollar_volume).label("dollar_adv"),
            sa.func.count().label("observations"),
        )
        .where(ranked.c.row_num <= lookback_days)
        .group_by(ranked.c.ticker)
        .order_by(sa.desc("dollar_adv"), ranked.c.ticker)
        .limit(top_n)
    )

    factory = session_factory or get_session_factory()
    with factory() as session:
        rows = session.execute(statement).mappings().all()

    if not rows:
        logger.warning(
            "No PIT-visible stock_prices rows found for top liquidity ranking as_of={} cutoff={}",
            as_of_date,
            cutoff.isoformat(),
        )
        return []

    max_observations = max(int(row["observations"] or 0) for row in rows)
    if max_observations < 5:
        logger.warning(
            "Top-liquidity ranking for {} has only {} visible observations per ticker at best; "
            "falling back to available history.",
            as_of_date,
            max_observations,
        )

    return [str(row["ticker"]).upper() for row in rows]
