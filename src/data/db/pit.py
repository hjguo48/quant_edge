from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timezone

import pandas as pd
import sqlalchemy as sa

from src.data.db.models import FundamentalsPIT, StockPrice, UniverseMembership
from src.data.db.session import get_session_factory


def _coerce_as_of(as_of: date | datetime) -> datetime:
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=timezone.utc)
        return as_of.astimezone(timezone.utc)
    return datetime.combine(as_of, time.max, tzinfo=timezone.utc)


def _normalize_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker))


def get_fundamentals_pit(
    ticker: str,
    as_of: date | datetime,
    metric_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    as_of_ts = _coerce_as_of(as_of)
    normalized_metrics = tuple(metric_names) if metric_names else None

    ranked = (
        sa.select(
            FundamentalsPIT.id.label("id"),
            FundamentalsPIT.ticker.label("ticker"),
            FundamentalsPIT.fiscal_period.label("fiscal_period"),
            FundamentalsPIT.metric_name.label("metric_name"),
            FundamentalsPIT.metric_value.label("metric_value"),
            FundamentalsPIT.event_time.label("event_time"),
            FundamentalsPIT.knowledge_time.label("knowledge_time"),
            FundamentalsPIT.is_restated.label("is_restated"),
            FundamentalsPIT.source.label("source"),
            sa.func.row_number()
            .over(
                partition_by=(
                    FundamentalsPIT.ticker,
                    FundamentalsPIT.fiscal_period,
                    FundamentalsPIT.metric_name,
                ),
                order_by=(
                    FundamentalsPIT.knowledge_time.desc(),
                    FundamentalsPIT.event_time.desc(),
                    FundamentalsPIT.id.desc(),
                ),
            )
            .label("row_num"),
        )
        .where(
            FundamentalsPIT.ticker == ticker.upper(),
            FundamentalsPIT.knowledge_time <= as_of_ts,
        )
    )
    if normalized_metrics:
        ranked = ranked.where(FundamentalsPIT.metric_name.in_(normalized_metrics))

    subquery = ranked.subquery()
    statement = (
        sa.select(
            subquery.c.id,
            subquery.c.ticker,
            subquery.c.fiscal_period,
            subquery.c.metric_name,
            subquery.c.metric_value,
            subquery.c.event_time,
            subquery.c.knowledge_time,
            subquery.c.is_restated,
            subquery.c.source,
        )
        .where(subquery.c.row_num == 1)
        .order_by(subquery.c.event_time, subquery.c.metric_name)
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "ticker",
                "fiscal_period",
                "metric_name",
                "metric_value",
                "event_time",
                "knowledge_time",
                "is_restated",
                "source",
            ],
        )

    return pd.DataFrame(rows)


def get_prices_pit(
    tickers: Sequence[str],
    start_date: date | datetime,
    end_date: date | datetime,
    as_of: date | datetime,
) -> pd.DataFrame:
    normalized_tickers = _normalize_tickers(tickers)
    if not normalized_tickers:
        return pd.DataFrame(
            columns=[
                "ticker",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
                "knowledge_time",
                "source",
            ],
        )

    as_of_ts = _coerce_as_of(as_of)
    start = start_date.date() if isinstance(start_date, datetime) else start_date
    end = end_date.date() if isinstance(end_date, datetime) else end_date

    statement = (
        sa.select(
            StockPrice.ticker,
            StockPrice.trade_date,
            StockPrice.open,
            StockPrice.high,
            StockPrice.low,
            StockPrice.close,
            StockPrice.adj_close,
            StockPrice.volume,
            StockPrice.knowledge_time,
            StockPrice.source,
        )
        .where(
            StockPrice.ticker.in_(normalized_tickers),
            StockPrice.trade_date >= start,
            StockPrice.trade_date <= end,
            StockPrice.knowledge_time <= as_of_ts,
        )
        .order_by(StockPrice.trade_date, StockPrice.ticker)
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    return pd.DataFrame(rows)


def get_universe_pit(as_of: date | datetime, index_name: str = "SP500") -> list[str]:
    as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
    statement = (
        sa.select(sa.distinct(UniverseMembership.ticker))
        .where(
            UniverseMembership.index_name == index_name,
            UniverseMembership.effective_date <= as_of_date,
            sa.or_(
                UniverseMembership.end_date.is_(None),
                UniverseMembership.end_date > as_of_date,
            ),
        )
        .order_by(UniverseMembership.ticker)
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        return list(session.execute(statement).scalars().all())
