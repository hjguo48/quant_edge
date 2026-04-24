from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import sqlalchemy as sa

from src.data.db.session import get_session_factory
from src.data.finra_short_sale import ShortSaleVolume

EASTERN = ZoneInfo("America/New_York")
XNYS = xcals.get_calendar("XNYS")


def compute_short_sale_ratio_1d(
    ticker: str,
    as_of: date,
    session_factory: Callable | None = None,
) -> float | None:
    session_label = _current_or_previous_session(as_of)
    daily = _load_daily_short_ratios(
        ticker=ticker,
        start_date=session_label,
        end_date=session_label,
        as_of_end=_as_of_end_utc(as_of),
        session_factory=session_factory,
    )
    if daily.empty:
        return None
    value = daily["ratio"].iloc[0]
    return _float_or_none(value)


def compute_short_sale_ratio_5d(
    ticker: str,
    as_of: date,
    session_factory: Callable | None = None,
) -> float | None:
    sessions = _recent_sessions(as_of, 5)
    if not sessions:
        return None
    daily = _load_daily_short_ratios(
        ticker=ticker,
        start_date=sessions[0],
        end_date=sessions[-1],
        as_of_end=_as_of_end_utc(as_of),
        session_factory=session_factory,
    )
    ratios = daily.set_index("trade_date")["ratio"].reindex(sessions).dropna()
    if len(ratios) < 3:
        return None
    return float(ratios.mean())


def compute_short_sale_accel(
    ticker: str,
    as_of: date,
    session_factory: Callable | None = None,
) -> float | None:
    sessions = _recent_sessions(as_of, 20)
    if not sessions:
        return None
    daily = _load_daily_short_ratios(
        ticker=ticker,
        start_date=sessions[0],
        end_date=sessions[-1],
        as_of_end=_as_of_end_utc(as_of),
        session_factory=session_factory,
    )
    ratios = daily.set_index("trade_date")["ratio"].reindex(sessions).dropna()
    ma_5 = ratios.tail(5)
    ma_20 = ratios.tail(20)
    if len(ma_5) < 3 or len(ma_20) < 15:
        return None
    return float(ma_5.mean() - ma_20.mean())


def compute_abnormal_off_exchange_shorting(
    ticker: str,
    as_of: date,
    session_factory: Callable | None = None,
) -> float | None:
    sessions = _recent_sessions(as_of, 91)
    if len(sessions) < 2:
        return None
    current_session = sessions[-1]
    history_sessions = sessions[:-1]
    market_daily = _load_market_short_ratios(
        ticker=ticker,
        start_date=history_sessions[0],
        end_date=current_session,
        as_of_end=_as_of_end_utc(as_of),
        session_factory=session_factory,
    )
    adf_daily = (
        market_daily.loc[market_daily["market"] == "ADF", ["trade_date", "ratio"]]
        .drop_duplicates(subset=["trade_date"])
        .set_index("trade_date")["ratio"]
    )
    today_ratio = _float_or_none(adf_daily.reindex([current_session]).iloc[0] if current_session in adf_daily.index else None)
    if today_ratio is None:
        return None
    history = adf_daily.reindex(history_sessions).dropna()
    if len(history) < 60:
        return None
    std = float(history.std(ddof=0))
    if std <= 0:
        return None
    mean = float(history.mean())
    return float((today_ratio - mean) / std)


def _load_daily_short_ratios(
    *,
    ticker: str,
    start_date: date,
    end_date: date,
    as_of_end: datetime,
    session_factory: Callable | None,
) -> pd.DataFrame:
    market_daily = _load_market_short_ratios(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        as_of_end=as_of_end,
        session_factory=session_factory,
    )
    if market_daily.empty:
        return pd.DataFrame(columns=["trade_date", "short_volume", "total_volume", "ratio"])

    combined = (
        market_daily.groupby("trade_date", as_index=False)[["short_volume", "total_volume"]]
        .sum()
        .sort_values("trade_date")
    )
    combined["ratio"] = combined.apply(
        lambda row: _safe_ratio(row["short_volume"], row["total_volume"]),
        axis=1,
    )
    return combined


def _load_market_short_ratios(
    *,
    ticker: str,
    start_date: date,
    end_date: date,
    as_of_end: datetime,
    session_factory: Callable | None,
) -> pd.DataFrame:
    factory = session_factory or get_session_factory()
    stmt = (
        sa.select(
            ShortSaleVolume.trade_date.label("trade_date"),
            ShortSaleVolume.market.label("market"),
            sa.func.sum(ShortSaleVolume.short_volume).label("short_volume"),
            sa.func.sum(ShortSaleVolume.total_volume).label("total_volume"),
        )
        .where(
            ShortSaleVolume.ticker == ticker.upper(),
            ShortSaleVolume.trade_date >= start_date,
            ShortSaleVolume.trade_date <= end_date,
            ShortSaleVolume.knowledge_time <= as_of_end,
        )
        .group_by(ShortSaleVolume.trade_date, ShortSaleVolume.market)
        .order_by(ShortSaleVolume.trade_date, ShortSaleVolume.market)
    )

    with factory() as session:
        rows = session.execute(stmt).mappings().all()

    frame = pd.DataFrame(rows, columns=["trade_date", "market", "short_volume", "total_volume"])
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["short_volume"] = pd.to_numeric(frame["short_volume"], errors="coerce")
    frame["total_volume"] = pd.to_numeric(frame["total_volume"], errors="coerce")
    frame["ratio"] = frame.apply(lambda row: _safe_ratio(row["short_volume"], row["total_volume"]), axis=1)
    return frame


def _current_or_previous_session(as_of: date) -> date:
    as_of_ts = pd.Timestamp(as_of)
    if XNYS.is_session(as_of_ts):
        return as_of_ts.date()
    return XNYS.previous_session(as_of_ts).date()


def _recent_sessions(as_of: date, count: int) -> list[date]:
    if count <= 0:
        return []
    current_session = pd.Timestamp(_current_or_previous_session(as_of))
    lookback_start = current_session - pd.Timedelta(days=max(count * 4, 30))
    sessions = XNYS.sessions_in_range(lookback_start, current_session)
    if len(sessions) == 0:
        return []
    return [session.date() for session in sessions[-count:]]


def _as_of_end_utc(as_of: date) -> datetime:
    local_dt = datetime.combine(as_of, time(hour=23, minute=59, second=59), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    numerator_float = float(numerator)
    denominator_float = float(denominator)
    if denominator_float == 0:
        return None
    return numerator_float / denominator_float


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)
