from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.data.db.models import StockPrice
from src.data.db.session import get_session_factory
from src.data.sources.fmp_grades import GradesEvent
from src.data.sources.fmp_price_target import PriceTargetEvent
from src.data.sources.fmp_ratings import RatingEvent

EASTERN = ZoneInfo("America/New_York")


def compute_net_grade_change(
    ticker: str,
    as_of: date,
    horizon_days: int,
    session_factory: Callable | None = None,
) -> int | None:
    normalized_ticker = ticker.upper()
    start_date = as_of - timedelta(days=horizon_days)
    as_of_end = _as_of_end_utc(as_of)
    factory = session_factory or get_session_factory()

    stmt = sa.select(sa.func.coalesce(sa.func.sum(GradesEvent.grade_score_change), 0)).where(
        GradesEvent.ticker == normalized_ticker,
        GradesEvent.knowledge_time <= as_of_end,
        GradesEvent.event_date >= start_date,
        GradesEvent.event_date <= as_of,
    )

    with factory() as session:
        value = session.execute(stmt).scalar_one()
    return int(value)


def compute_upgrade_count(
    ticker: str,
    as_of: date,
    horizon_days: int = 20,
    session_factory: Callable | None = None,
) -> int | None:
    normalized_ticker = ticker.upper()
    start_date = as_of - timedelta(days=horizon_days)
    as_of_end = _as_of_end_utc(as_of)
    factory = session_factory or get_session_factory()

    stmt = sa.select(sa.func.count()).where(
        GradesEvent.ticker == normalized_ticker,
        GradesEvent.knowledge_time <= as_of_end,
        GradesEvent.event_date >= start_date,
        GradesEvent.event_date <= as_of,
        GradesEvent.grade_score_change > 0,
    )

    with factory() as session:
        value = session.execute(stmt).scalar_one()
    return int(value)


def compute_downgrade_count(
    ticker: str,
    as_of: date,
    horizon_days: int = 20,
    session_factory: Callable | None = None,
) -> int | None:
    normalized_ticker = ticker.upper()
    start_date = as_of - timedelta(days=horizon_days)
    as_of_end = _as_of_end_utc(as_of)
    factory = session_factory or get_session_factory()

    stmt = sa.select(sa.func.count()).where(
        GradesEvent.ticker == normalized_ticker,
        GradesEvent.knowledge_time <= as_of_end,
        GradesEvent.event_date >= start_date,
        GradesEvent.event_date <= as_of,
        GradesEvent.grade_score_change < 0,
    )

    with factory() as session:
        value = session.execute(stmt).scalar_one()
    return int(value)


def compute_consensus_upside(
    ticker: str,
    as_of: date,
    session_factory: Callable | None = None,
) -> float | None:
    """(consensus_target - close) / close.

    The FMP consensus snapshot endpoint returns only the *current* aggregate, so
    historical PIT lookups for past dates always exclude it (knowledge_time =
    fetch time, not event time). We therefore proxy the consensus by averaging
    per-analyst price targets in the most recent 60 days that meet PIT.
    """
    normalized_ticker = ticker.upper()
    start_date = as_of - timedelta(days=60)
    as_of_end = _as_of_end_utc(as_of)
    factory = session_factory or get_session_factory()

    # Per-analyst targets within the lookback (PIT-safe). Take last revision per
    # firm so the proxy mirrors a real consensus calc.
    ranked_targets = (
        sa.select(
            PriceTargetEvent.analyst_firm.label("analyst_firm"),
            PriceTargetEvent.target_price.label("target_price"),
            sa.func.row_number()
            .over(
                partition_by=PriceTargetEvent.analyst_firm,
                order_by=(PriceTargetEvent.event_date.desc(), PriceTargetEvent.knowledge_time.desc()),
            )
            .label("rn"),
        )
        .where(
            PriceTargetEvent.ticker == normalized_ticker,
            PriceTargetEvent.is_consensus.is_(False),
            PriceTargetEvent.knowledge_time <= as_of_end,
            PriceTargetEvent.event_date >= start_date,
            PriceTargetEvent.event_date <= as_of,
            PriceTargetEvent.analyst_firm.is_not(None),
        )
        .subquery()
    )
    stmt = sa.select(sa.func.avg(ranked_targets.c.target_price)).where(ranked_targets.c.rn == 1)

    with factory() as session:
        consensus_target = session.execute(stmt).scalar_one_or_none()
        latest_close = _latest_close(normalized_ticker, as_of, session)

    if consensus_target is None or latest_close is None or latest_close == 0:
        return None
    return (float(consensus_target) - latest_close) / latest_close


def compute_target_price_drift(
    ticker: str,
    as_of: date,
    horizon_days: int = 60,
    session_factory: Callable | None = None,
) -> float | None:
    normalized_ticker = ticker.upper()
    start_date = as_of - timedelta(days=horizon_days)
    as_of_end = _as_of_end_utc(as_of)
    factory = session_factory or get_session_factory()

    stmt = (
        sa.select(PriceTargetEvent.event_date, PriceTargetEvent.target_price)
        .where(
            PriceTargetEvent.ticker == normalized_ticker,
            PriceTargetEvent.is_consensus.is_(False),
            PriceTargetEvent.knowledge_time <= as_of_end,
            PriceTargetEvent.event_date >= start_date,
            PriceTargetEvent.event_date <= as_of,
        )
        .order_by(PriceTargetEvent.event_date, PriceTargetEvent.knowledge_time)
    )

    with factory() as session:
        rows = session.execute(stmt).mappings().all()
        latest_close = _latest_close(normalized_ticker, as_of, session)

    if latest_close is None or latest_close == 0 or not rows:
        return None

    frame = pd.DataFrame(rows, columns=["event_date", "target_price"])
    frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.date
    frame["target_price"] = pd.to_numeric(frame["target_price"], errors="coerce")
    frame = frame.dropna(subset=["target_price"])
    if frame.empty or frame["event_date"].nunique() < 5:
        return None

    min_event_date = frame["event_date"].min()
    x = np.array([(event_day - min_event_date).days for event_day in frame["event_date"]], dtype=float)
    y = frame["target_price"].astype(float).to_numpy()
    if len(np.unique(x)) < 2:
        return None

    slope_raw = float(np.polyfit(x, y, 1)[0])
    return slope_raw * float(horizon_days) / latest_close


def compute_target_dispersion_proxy(
    ticker: str,
    as_of: date,
    session_factory: Callable | None = None,
) -> float | None:
    normalized_ticker = ticker.upper()
    start_date = as_of - timedelta(days=60)
    as_of_end = _as_of_end_utc(as_of)
    factory = session_factory or get_session_factory()

    ranked_targets = (
        sa.select(
            PriceTargetEvent.analyst_firm.label("analyst_firm"),
            PriceTargetEvent.target_price.label("target_price"),
            sa.func.row_number()
            .over(
                partition_by=PriceTargetEvent.analyst_firm,
                order_by=(PriceTargetEvent.event_date.desc(), PriceTargetEvent.knowledge_time.desc()),
            )
            .label("rn"),
        )
        .where(
            PriceTargetEvent.ticker == normalized_ticker,
            PriceTargetEvent.is_consensus.is_(False),
            PriceTargetEvent.knowledge_time <= as_of_end,
            PriceTargetEvent.event_date >= start_date,
            PriceTargetEvent.event_date <= as_of,
            PriceTargetEvent.analyst_firm.is_not(None),
        )
        .subquery()
    )

    stmt = sa.select(ranked_targets.c.analyst_firm, ranked_targets.c.target_price).where(ranked_targets.c.rn == 1)

    with factory() as session:
        rows = session.execute(stmt).mappings().all()

    frame = pd.DataFrame(rows, columns=["analyst_firm", "target_price"])
    if frame.empty or frame["analyst_firm"].nunique() < 3:
        return None

    prices = pd.to_numeric(frame["target_price"], errors="coerce").dropna()
    if len(prices) < 3:
        return None
    mean_value = float(prices.mean())
    if mean_value == 0:
        return None
    return float(prices.std(ddof=0) / mean_value)


def compute_coverage_change_proxy(
    ticker: str,
    as_of: date,
    horizon_days: int = 60,
    session_factory: Callable | None = None,
) -> int | None:
    normalized_ticker = ticker.upper()
    recent_start = as_of - timedelta(days=horizon_days)
    prior_start = as_of - timedelta(days=2 * horizon_days)
    prior_end = recent_start
    as_of_end = _as_of_end_utc(as_of)
    factory = session_factory or get_session_factory()

    stmt = sa.select(PriceTargetEvent.analyst_firm, PriceTargetEvent.event_date).where(
        PriceTargetEvent.ticker == normalized_ticker,
        PriceTargetEvent.is_consensus.is_(False),
        PriceTargetEvent.knowledge_time <= as_of_end,
        PriceTargetEvent.event_date >= prior_start,
        PriceTargetEvent.event_date <= as_of,
        PriceTargetEvent.analyst_firm.is_not(None),
    )

    with factory() as session:
        rows = session.execute(stmt).mappings().all()

    frame = pd.DataFrame(rows, columns=["analyst_firm", "event_date"])
    if frame.empty:
        return None

    frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.date
    if not (frame["event_date"] <= prior_end).any():
        return None

    recent_firms = {
        str(row.analyst_firm)
        for row in frame.itertuples(index=False)
        if recent_start <= row.event_date <= as_of
    }
    prior_firms = {
        str(row.analyst_firm)
        for row in frame.itertuples(index=False)
        if prior_start <= row.event_date <= prior_end
    }
    return len(recent_firms) - len(prior_firms)


def compute_financial_health_trend(
    ticker: str,
    as_of: date,
    horizon_days: int = 60,
    session_factory: Callable | None = None,
) -> float | None:
    normalized_ticker = ticker.upper()
    as_of_end = _as_of_end_utc(as_of)
    prior_cutoff = as_of - timedelta(days=horizon_days)
    factory = session_factory or get_session_factory()

    current_stmt = (
        sa.select(RatingEvent.rating_score)
        .where(
            RatingEvent.ticker == normalized_ticker,
            RatingEvent.knowledge_time <= as_of_end,
            RatingEvent.event_date <= as_of,
        )
        .order_by(RatingEvent.event_date.desc(), RatingEvent.knowledge_time.desc())
        .limit(1)
    )
    prior_stmt = (
        sa.select(RatingEvent.rating_score)
        .where(
            RatingEvent.ticker == normalized_ticker,
            RatingEvent.knowledge_time <= as_of_end,
            RatingEvent.event_date <= prior_cutoff,
        )
        .order_by(RatingEvent.event_date.desc(), RatingEvent.knowledge_time.desc())
        .limit(1)
    )

    with factory() as session:
        current_rating = session.execute(current_stmt).scalar_one_or_none()
        prior_rating = session.execute(prior_stmt).scalar_one_or_none()

    if current_rating is None or prior_rating is None:
        return None
    return float(current_rating) - float(prior_rating)


def _as_of_end_utc(as_of: date) -> datetime:
    local_dt = datetime.combine(as_of, time(hour=23, minute=59, second=59), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)


def _latest_close(ticker: str, as_of: date, session: sa.orm.Session) -> float | None:
    stmt = (
        sa.select(StockPrice.close)
        .where(
            StockPrice.ticker == ticker.upper(),
            StockPrice.trade_date <= as_of,
            StockPrice.knowledge_time <= _as_of_end_utc(as_of),
        )
        .order_by(StockPrice.trade_date.desc(), StockPrice.knowledge_time.desc())
        .limit(1)
    )
    close_value = session.execute(stmt).scalar_one_or_none()
    if close_value is None:
        return None
    return float(close_value)
