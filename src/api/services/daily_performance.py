from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Literal

import exchange_calendars as xcals
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.portfolio import (
    DailyPerformancePoint,
    DailyPerformanceTranche,
    DailyPortfolioPerformanceResponse,
)
from src.data.db.models import PaperPortfolioAudit, StockPrice

PerformanceHorizon = Literal["1d", "5d", "20d", "60d"]
HORIZON_DAYS: dict[PerformanceHorizon, int] = {
    "1d": 1,
    "5d": 5,
    "20d": 20,
    "60d": 60,
}
BENCHMARK_TICKER = "SPY"
XNYS = xcals.get_calendar("XNYS")


async def compute_daily_portfolio_performance(
    db: AsyncSession,
    *,
    horizon: PerformanceHorizon = "60d",
    bundle_version: str | None = None,
    as_of: datetime | None = None,
) -> DailyPortfolioPerformanceResponse:
    as_of_utc = as_of or datetime.now(timezone.utc)
    resolved_bundle = bundle_version or await _latest_bundle_version(db)
    if resolved_bundle is None:
        return DailyPortfolioPerformanceResponse(horizon=horizon)

    audit_rows = await _load_paper_portfolio_rows(db, bundle_version=resolved_bundle)
    if not audit_rows:
        return DailyPortfolioPerformanceResponse(horizon=horizon, bundle_version=resolved_bundle)

    grouped: dict[date, dict[str, float]] = defaultdict(dict)
    for signal_date, ticker, target_weight in audit_rows:
        grouped[signal_date][ticker.upper()] = float(target_weight)

    horizon_days = HORIZON_DAYS[horizon]
    tranches: list[DailyPerformanceTranche] = []
    latest_horizon_end: date | None = None

    for tranche_index, signal_date in enumerate(sorted(grouped), start=1):
        weights = grouped[signal_date]
        session_dates = _tranche_session_dates(
            signal_date=signal_date,
            horizon_days=horizon_days,
            as_of_date=as_of_utc.date(),
        )
        if not session_dates:
            tranches.append(
                DailyPerformanceTranche(
                    signal_date=signal_date.isoformat(),
                    tranche_index=tranche_index,
                    tickers_dropped=sorted(weights),
                )
            )
            continue

        tickers = sorted(set(weights) | {BENCHMARK_TICKER})
        price_rows = await _load_open_prices(
            db,
            tickers=tickers,
            start_date=session_dates[0],
            end_date=session_dates[-1],
            as_of=as_of_utc,
        )
        price_by_ticker = _price_rows_to_lookup(price_rows)
        tranche = _build_tranche(
            signal_date=signal_date,
            tranche_index=tranche_index,
            weights=weights,
            session_dates=session_dates,
            price_by_ticker=price_by_ticker,
        )
        tranches.append(tranche)
        if tranche.horizon_end_date is not None:
            tranche_end = date.fromisoformat(tranche.horizon_end_date)
            latest_horizon_end = tranche_end if latest_horizon_end is None else max(latest_horizon_end, tranche_end)

    return DailyPortfolioPerformanceResponse(
        horizon=horizon,
        bundle_version=resolved_bundle,
        weeks_count=len(tranches),
        latest_horizon_end_date=latest_horizon_end.isoformat() if latest_horizon_end else None,
        tranches=tranches,
    )


async def _latest_bundle_version(db: AsyncSession) -> str | None:
    result = await db.execute(
        sa.select(PaperPortfolioAudit.bundle_version)
        .order_by(PaperPortfolioAudit.generated_at_utc.desc())
        .limit(1)
    )
    value = result.scalar_one_or_none()
    return None if value is None else str(value)


async def _load_paper_portfolio_rows(
    db: AsyncSession,
    *,
    bundle_version: str,
) -> list[tuple[date, str, Decimal]]:
    result = await db.execute(
        sa.select(
            PaperPortfolioAudit.signal_date,
            PaperPortfolioAudit.ticker,
            PaperPortfolioAudit.target_weight,
        )
        .where(PaperPortfolioAudit.bundle_version == bundle_version)
        .order_by(PaperPortfolioAudit.signal_date.asc(), PaperPortfolioAudit.ticker.asc())
    )
    return [(row[0], str(row[1]), row[2]) for row in result.all()]


async def _load_open_prices(
    db: AsyncSession,
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    as_of: datetime,
) -> list[tuple[str, date, Decimal]]:
    result = await db.execute(
        sa.select(StockPrice.ticker, StockPrice.trade_date, StockPrice.open)
        .where(
            StockPrice.ticker.in_(tickers),
            StockPrice.trade_date >= start_date,
            StockPrice.trade_date <= end_date,
            StockPrice.knowledge_time <= as_of,
            StockPrice.open.is_not(None),
        )
        .order_by(StockPrice.ticker.asc(), StockPrice.trade_date.asc())
    )
    return [(str(row[0]).upper(), row[1], row[2]) for row in result.all()]


def _tranche_session_dates(
    *,
    signal_date: date,
    horizon_days: int,
    as_of_date: date,
) -> list[date]:
    if horizon_days < 1:
        return []
    signal_ts = pd.Timestamp(signal_date)
    if XNYS.is_session(signal_ts):
        entry_session = XNYS.next_session(signal_ts)
    else:
        entry_session = XNYS.date_to_session(signal_ts, direction="next")

    as_of_ts = pd.Timestamp(as_of_date)
    if XNYS.is_session(as_of_ts):
        as_of_session = as_of_ts
    else:
        as_of_session = XNYS.date_to_session(as_of_ts, direction="previous")
    if as_of_session < entry_session:
        return []

    lookahead_end = entry_session + pd.Timedelta(days=max(horizon_days * 3, 10))
    horizon_sessions = XNYS.sessions_in_range(entry_session, lookahead_end)[:horizon_days]
    if len(horizon_sessions) == 0:
        return []
    horizon_end = horizon_sessions[-1]
    end_session = min(horizon_end, as_of_session)
    sessions = XNYS.sessions_in_range(entry_session, end_session)
    return [pd.Timestamp(session).date() for session in sessions]


def _price_rows_to_lookup(rows: list[tuple[str, date, Decimal]]) -> dict[str, dict[date, float]]:
    lookup: dict[str, dict[date, float]] = defaultdict(dict)
    for ticker, trade_date, open_price in rows:
        lookup[ticker.upper()][trade_date] = float(open_price)
    return lookup


def _build_tranche(
    *,
    signal_date: date,
    tranche_index: int,
    weights: dict[str, float],
    session_dates: list[date],
    price_by_ticker: dict[str, dict[date, float]],
) -> DailyPerformanceTranche:
    entry_date = session_dates[0]
    original_weight_sum = sum(float(weight) for weight in weights.values())
    usable_weights: dict[str, float] = {}
    dropped: list[str] = []
    for ticker, weight in weights.items():
        entry_open = price_by_ticker.get(ticker, {}).get(entry_date)
        if entry_open is None or entry_open <= 0.0:
            dropped.append(ticker)
            continue
        usable_weights[ticker] = float(weight)

    remaining_sum = sum(usable_weights.values())
    if original_weight_sum > 0.0 and remaining_sum > 0.0:
        scale = original_weight_sum / remaining_sum
        usable_weights = {ticker: weight * scale for ticker, weight in usable_weights.items()}

    spy_entry_open = price_by_ticker.get(BENCHMARK_TICKER, {}).get(entry_date)
    points: list[DailyPerformancePoint] = []
    if usable_weights and spy_entry_open is not None and spy_entry_open > 0.0:
        for tranche_day, trade_date in enumerate(session_dates, start=1):
            spy_open = price_by_ticker.get(BENCHMARK_TICKER, {}).get(trade_date)
            if spy_open is None:
                continue
            portfolio_return = 0.0
            for ticker, weight in usable_weights.items():
                entry_open = price_by_ticker[ticker][entry_date]
                current_open = price_by_ticker.get(ticker, {}).get(trade_date)
                if current_open is None:
                    continue
                portfolio_return += weight * ((current_open - entry_open) / entry_open)
            spy_return = (spy_open - spy_entry_open) / spy_entry_open
            points.append(
                DailyPerformancePoint(
                    date=trade_date.isoformat(),
                    cumulative_portfolio=float(portfolio_return),
                    cumulative_spy=float(spy_return),
                    cumulative_excess=float(portfolio_return - spy_return),
                    tranche_day=tranche_day,
                )
            )

    horizon_end_date = points[-1].date if points else None
    return DailyPerformanceTranche(
        signal_date=signal_date.isoformat(),
        tranche_index=tranche_index,
        entry_date=entry_date.isoformat(),
        horizon_end_date=horizon_end_date,
        tickers_used=sorted(usable_weights),
        tickers_dropped=sorted(dropped),
        series=points,
    )
