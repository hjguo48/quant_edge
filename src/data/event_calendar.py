from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from src.config.week4_trades import Week4TradesConfig
from src.data.db.models import StockPrice
from src.data.db.session import get_session_factory
from src.data.sources.fmp_earnings import EarningsEstimate
from src.universe.top_liquidity import get_top_liquidity_tickers

EASTERN = ZoneInfo("America/New_York")
XNYS = xcals.get_calendar("XNYS")
VALID_REASONS = {"earnings", "gap", "weak_window", "top_liquidity"}


@dataclass(frozen=True)
class SamplingEvent:
    ticker: str
    trading_date: date
    reason: str

    def __post_init__(self) -> None:
        normalized_reason = self.reason.strip().lower()
        if normalized_reason not in VALID_REASONS:
            raise ValueError(f"unsupported sampling reason: {self.reason!r}")
        object.__setattr__(self, "ticker", self.ticker.upper())
        object.__setattr__(self, "reason", normalized_reason)


def _coerce_config(config: dict[str, Any] | Week4TradesConfig) -> Week4TradesConfig:
    if isinstance(config, Week4TradesConfig):
        return config
    return Week4TradesConfig.model_validate(config)


def _sessions_in_range(start: date, end: date) -> list[date]:
    if end < start:
        return []
    sessions = XNYS.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
    return [pd.Timestamp(session).date() for session in sessions]


def _eod_utc(trading_date: date) -> datetime:
    return datetime.combine(trading_date, time(23, 59, 59, 999999), tzinfo=EASTERN).astimezone(
        timezone.utc,
    )


def _bounded_sessions(start: date, end: date, lower: date, upper: date) -> list[date]:
    return _sessions_in_range(max(start, lower), min(end, upper))


def _add_event(events: list[SamplingEvent], ticker: str, trading_date: date, reason: str) -> None:
    events.append(SamplingEvent(ticker=ticker, trading_date=trading_date, reason=reason))


def _earnings_events(
    *,
    session: Session,
    start_date: date,
    end_date: date,
    window_days: int,
) -> list[SamplingEvent]:
    query_start = start_date - timedelta(days=window_days + 7)
    query_end = end_date + timedelta(days=window_days + 7)
    statement = (
        sa.select(
            EarningsEstimate.ticker,
            EarningsEstimate.fiscal_date,
            EarningsEstimate.knowledge_time,
        )
        .where(
            EarningsEstimate.fiscal_date >= query_start,
            EarningsEstimate.fiscal_date <= query_end,
        )
        .order_by(EarningsEstimate.ticker, EarningsEstimate.fiscal_date, EarningsEstimate.knowledge_time)
    )
    rows = session.execute(statement).mappings().all()
    events: list[SamplingEvent] = []
    seen: set[tuple[str, date, str]] = set()
    for row in rows:
        fiscal_date = pd.Timestamp(row["fiscal_date"]).date()
        ticker = str(row["ticker"]).upper()
        knowledge_time = row["knowledge_time"]
        if knowledge_time.tzinfo is None:
            knowledge_time = knowledge_time.replace(tzinfo=timezone.utc)
        else:
            knowledge_time = knowledge_time.astimezone(timezone.utc)
        for trading_date in _bounded_sessions(
            fiscal_date - timedelta(days=window_days),
            fiscal_date + timedelta(days=window_days),
            start_date,
            end_date,
        ):
            if knowledge_time > _eod_utc(trading_date):
                continue
            key = (ticker, trading_date, "earnings")
            if key not in seen:
                _add_event(events, ticker, trading_date, "earnings")
                seen.add(key)
    return events


def _gap_events(
    *,
    session: Session,
    start_date: date,
    end_date: date,
    gap_threshold_pct: float,
) -> list[SamplingEvent]:
    statement = (
        sa.select(
            StockPrice.ticker,
            StockPrice.trade_date,
            StockPrice.open,
            StockPrice.close,
            StockPrice.knowledge_time,
        )
        .where(
            StockPrice.trade_date >= start_date - timedelta(days=10),
            StockPrice.trade_date <= end_date,
        )
        .order_by(StockPrice.ticker, StockPrice.trade_date)
    )
    rows = session.execute(statement).mappings().all()
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_ticker.setdefault(str(row["ticker"]).upper(), []).append(dict(row))

    events: list[SamplingEvent] = []
    for ticker, ticker_rows in by_ticker.items():
        for idx, row in enumerate(ticker_rows):
            trading_date = pd.Timestamp(row["trade_date"]).date()
            if trading_date < start_date or trading_date > end_date:
                continue
            current_cutoff = _eod_utc(trading_date)
            knowledge_time = row["knowledge_time"]
            if knowledge_time.tzinfo is None:
                knowledge_time = knowledge_time.replace(tzinfo=timezone.utc)
            else:
                knowledge_time = knowledge_time.astimezone(timezone.utc)
            if knowledge_time > current_cutoff or row["open"] is None:
                continue

            prev_close = None
            for prev in reversed(ticker_rows[:idx]):
                prev_date = pd.Timestamp(prev["trade_date"]).date()
                if prev_date >= trading_date:
                    continue
                prev_knowledge_time = prev["knowledge_time"]
                if prev_knowledge_time.tzinfo is None:
                    prev_knowledge_time = prev_knowledge_time.replace(tzinfo=timezone.utc)
                else:
                    prev_knowledge_time = prev_knowledge_time.astimezone(timezone.utc)
                if prev_knowledge_time <= current_cutoff and prev["close"] not in (None, 0):
                    prev_close = prev["close"]
                    break
            if prev_close in (None, 0):
                continue
            gap = abs((float(row["open"]) - float(prev_close)) / float(prev_close))
            if gap >= gap_threshold_pct:
                _add_event(events, ticker, trading_date, "gap")
    return events


def _weak_window_events(
    config: Week4TradesConfig,
    start_date: date,
    end_date: date,
    session_factory: Callable[[], Session] | None,
) -> list[SamplingEvent]:
    events: list[SamplingEvent] = []
    pilot = config.sampling.pilot
    for window in pilot.weak_windows:
        for trading_date in _bounded_sessions(window.start, window.end, start_date, end_date):
            tickers = get_top_liquidity_tickers(
                trading_date,
                top_n=pilot.weak_window_top_n,
                session_factory=session_factory,
            )
            for ticker in tickers:
                _add_event(events, ticker, trading_date, "weak_window")
    return events


def _stage2_top_liquidity_events(
    config: Week4TradesConfig,
    start_date: date,
    end_date: date,
    session_factory: Callable[[], Session] | None,
) -> list[SamplingEvent]:
    events: list[SamplingEvent] = []
    stage2 = config.sampling.stage2
    for trading_date in _sessions_in_range(start_date, end_date):
        tickers = get_top_liquidity_tickers(
            trading_date,
            top_n=stage2.top_n_liquidity,
            lookback_days=stage2.top_liquidity_lookback_days,
            session_factory=session_factory,
        )
        for ticker in tickers:
            _add_event(events, ticker, trading_date, "top_liquidity")
    return events


def build_sampling_plan(
    *,
    start_date: date,
    end_date: date,
    config: dict[str, Any] | Week4TradesConfig,
    session_factory: Callable[[], Session] | None = None,
) -> list[SamplingEvent]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    parsed = _coerce_config(config)
    factory = session_factory or get_session_factory()
    events: list[SamplingEvent] = []
    pilot = parsed.sampling.pilot

    with factory() as session:
        if "earnings" in pilot.reasons:
            events.extend(
                _earnings_events(
                    session=session,
                    start_date=start_date,
                    end_date=end_date,
                    window_days=pilot.earnings_window_days,
                ),
            )
        if "gap" in pilot.reasons:
            events.extend(
                _gap_events(
                    session=session,
                    start_date=start_date,
                    end_date=end_date,
                    gap_threshold_pct=pilot.gap_threshold_pct,
                ),
            )

    if "weak_window" in pilot.reasons:
        events.extend(_weak_window_events(parsed, start_date, end_date, session_factory))

    if parsed.stage == "stage2":
        events.extend(_stage2_top_liquidity_events(parsed, start_date, end_date, session_factory))

    return sorted(events, key=lambda event: (event.trading_date, event.ticker, event.reason))
