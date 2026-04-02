from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd
import sqlalchemy as sa
from loguru import logger

from src.data.db.pit import get_prices_pit, get_universe_pit
from src.data.db.session import get_session_factory
from src.data.sources.fred import MACRO_SERIES_TABLE

# MACRO_REGIME: these features are broadcast to every ticker on a given date and
# therefore act as market-state context rather than standalone stock selectors.
MACRO_FEATURE_NAMES = (
    "vix",
    "vix_change_5d",
    "vix_rank",
    "yield_10y",
    "yield_spread_10y2y",
    "credit_spread",
    "credit_spread_change",
    "ffr",
    "sp500_breadth",
    "market_ret_20d",
)

_PRIMARY_MACRO_SERIES = ("VIXCLS", "DGS10", "DGS2", "BAA10Y", "AAA10Y", "FEDFUNDS")


def compute_macro_features(as_of: date | datetime, lookback_days: int = 252) -> pd.DataFrame:
    as_of_ts = _coerce_as_of_datetime(as_of)
    as_of_date = as_of_ts.date()
    series_history = _load_macro_histories(as_of_ts, lookback_days=max(lookback_days, 252) + 30)

    vix_series = series_history.get("VIXCLS", pd.Series(dtype=float))
    yield_10y_series = series_history.get("DGS10", pd.Series(dtype=float))
    yield_2y_series = series_history.get("DGS2", pd.Series(dtype=float))
    baa_10y_series = series_history.get("BAA10Y", pd.Series(dtype=float))
    aaa_10y_series = series_history.get("AAA10Y", pd.Series(dtype=float))
    ffr_series = series_history.get("FEDFUNDS", pd.Series(dtype=float))
    credit_spread_series = _credit_spread_series(baa_10y_series, aaa_10y_series)

    if yield_2y_series.empty:
        yield_2y_series = ffr_series

    breadth = _sp500_breadth(as_of_ts)
    market_ret_20d = _market_return(as_of_ts, benchmark_ticker="SPY", horizon=20)

    features = {
        "vix": _latest_value(vix_series),
        "vix_change_5d": _pct_change(vix_series, 5),
        "vix_rank": _percentile_rank(vix_series.tail(lookback_days)),
        "yield_10y": _latest_value(yield_10y_series),
        "yield_spread_10y2y": _spread(yield_10y_series, yield_2y_series),
        "credit_spread": _latest_value(credit_spread_series),
        "credit_spread_change": _difference(credit_spread_series, 20),
        "ffr": _latest_value(ffr_series),
        "sp500_breadth": breadth,
        "market_ret_20d": market_ret_20d,
    }

    frame = pd.DataFrame(
        [
            {
                "trade_date": as_of_date,
                "feature_name": feature_name,
                "feature_value": feature_value,
            }
            for feature_name, feature_value in features.items()
        ],
    )
    logger.info("computed {} macro features for {}", len(frame), as_of_date)
    return frame


def _load_macro_histories(as_of: datetime, lookback_days: int) -> dict[str, pd.Series]:
    start_date = as_of.date() - timedelta(days=lookback_days + 30)
    ranked = (
        sa.select(
            MACRO_SERIES_TABLE.c.series_id,
            MACRO_SERIES_TABLE.c.observation_date,
            MACRO_SERIES_TABLE.c.value,
            MACRO_SERIES_TABLE.c.knowledge_time,
            sa.func.row_number()
            .over(
                partition_by=(
                    MACRO_SERIES_TABLE.c.series_id,
                    MACRO_SERIES_TABLE.c.observation_date,
                ),
                order_by=(
                    MACRO_SERIES_TABLE.c.knowledge_time.desc(),
                    MACRO_SERIES_TABLE.c.id.desc(),
                ),
            )
            .label("row_num"),
        )
        .where(
            MACRO_SERIES_TABLE.c.series_id.in_(_PRIMARY_MACRO_SERIES),
            MACRO_SERIES_TABLE.c.observation_date >= start_date,
            MACRO_SERIES_TABLE.c.observation_date <= as_of.date(),
            MACRO_SERIES_TABLE.c.knowledge_time <= as_of,
        )
    ).subquery()

    statement = (
        sa.select(
            ranked.c.series_id,
            ranked.c.observation_date,
            ranked.c.value,
        )
        .where(ranked.c.row_num == 1)
        .order_by(ranked.c.series_id, ranked.c.observation_date)
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}

    frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.date
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    histories: dict[str, pd.Series] = {}
    for series_id, group in frame.groupby("series_id", sort=False):
        histories[str(series_id)] = pd.Series(
            group["value"].to_numpy(dtype=float),
            index=pd.Index(group["observation_date"], name="observation_date"),
            name=str(series_id),
        )
    return histories


def _sp500_breadth(as_of: datetime) -> float:
    tickers = get_universe_pit(as_of, index_name="SP500")
    if not tickers:
        return np.nan

    prices = get_prices_pit(
        tickers=tickers,
        start_date=as_of.date() - timedelta(days=40),
        end_date=as_of.date(),
        as_of=as_of,
    )
    if prices.empty:
        return np.nan

    prepared = prices.copy()
    prepared["adj_close"] = pd.to_numeric(prepared["adj_close"], errors="coerce")
    prepared["close"] = pd.to_numeric(prepared["close"], errors="coerce")
    prepared.sort_values(["ticker", "trade_date"], inplace=True)
    prepared["base_price"] = prepared["adj_close"].fillna(prepared["close"])
    prepared["ret_20d"] = prepared.groupby("ticker")["base_price"].pct_change(20)
    latest = prepared.groupby("ticker", group_keys=False).tail(1)
    valid = latest["ret_20d"].dropna()
    if valid.empty:
        return np.nan
    return float((valid > 0).mean())


def _market_return(as_of: datetime, benchmark_ticker: str, horizon: int) -> float:
    prices = get_prices_pit(
        tickers=[benchmark_ticker],
        start_date=as_of.date() - timedelta(days=max(horizon * 3, 40)),
        end_date=as_of.date(),
        as_of=as_of,
    )
    if prices.empty:
        return np.nan

    prepared = prices.copy()
    prepared.sort_values("trade_date", inplace=True)
    prepared["base_price"] = pd.to_numeric(prepared["adj_close"], errors="coerce").fillna(
        pd.to_numeric(prepared["close"], errors="coerce"),
    )
    prepared["ret"] = prepared["base_price"].pct_change(horizon)
    if prepared["ret"].dropna().empty:
        return np.nan
    return float(prepared["ret"].dropna().iloc[-1])


def _latest_value(series: pd.Series) -> float:
    if series.empty:
        return np.nan
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return float(non_null.iloc[-1])


def _pct_change(series: pd.Series, periods: int) -> float:
    if series.empty or len(series.dropna()) <= periods:
        return np.nan
    changed = series.astype(float).pct_change(periods)
    non_null = changed.dropna()
    return float(non_null.iloc[-1]) if not non_null.empty else np.nan


def _difference(series: pd.Series, periods: int) -> float:
    if series.empty or len(series.dropna()) <= periods:
        return np.nan
    diff = series.astype(float).diff(periods)
    non_null = diff.dropna()
    return float(non_null.iloc[-1]) if not non_null.empty else np.nan


def _percentile_rank(series: pd.Series) -> float:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    latest = non_null.iloc[-1]
    return float((non_null <= latest).mean())


def _spread(left: pd.Series, right: pd.Series) -> float:
    left_value = _latest_value(left)
    right_value = _latest_value(right)
    if pd.isna(left_value) or pd.isna(right_value):
        return np.nan
    return float(left_value - right_value)


def _credit_spread_series(baa_10y_series: pd.Series, aaa_10y_series: pd.Series) -> pd.Series:
    if baa_10y_series.empty:
        return pd.Series(dtype=float)
    if aaa_10y_series.empty:
        return baa_10y_series

    aligned = pd.concat(
        [
            baa_10y_series.rename("baa_10y"),
            aaa_10y_series.rename("aaa_10y"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)

    spread = aligned["baa_10y"] - aligned["aaa_10y"]
    spread.name = "credit_spread"
    return spread


def _coerce_as_of_datetime(as_of: date | datetime) -> datetime:
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=timezone.utc)
        return as_of.astimezone(timezone.utc)
    return datetime.combine(as_of, time.max, tzinfo=timezone.utc)
