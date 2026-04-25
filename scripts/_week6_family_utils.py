from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any

import exchange_calendars as xcals
from loguru import logger
import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.config import settings
from src.data.db.models import StockPrice
from src.data.db.session import get_session_factory
from src.features.macro import _load_macro_histories
from src.features.pipeline import FeaturePipeline
from src.features.registry import FeatureDefinition, FeatureRegistry
from src.universe.history import get_historical_members

XNYS = xcals.get_calendar("XNYS")
MISSING_FLAG_PREFIX = "is_missing_"
WEEK5_FAMILIES = frozenset({"shorting", "analyst_proxy"})
PIPELINE_VALUE_NULL_FAMILIES = frozenset({"composite", "sector_rotation"})
DEFAULT_SAMPLE_TICKERS = 20
DEFAULT_SAMPLE_DATES_PER_MONTH = 1
DEFAULT_SAMPLE_MONTHS = 12


@dataclass(frozen=True)
class SampleContext:
    start_date: date
    end_date: date
    universe: tuple[str, ...]
    sampled_tickers: tuple[str, ...]
    sampled_trade_dates: tuple[date, ...]

    @property
    def universe_size(self) -> int:
        return len(self.universe)


def parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def empty_observations_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "family", "is_missing"])


def latest_available_trade_date(session_factory: Callable | None = None) -> date:
    factory = session_factory or get_session_factory()
    with factory() as session:
        latest = session.execute(sa.select(sa.func.max(StockPrice.trade_date))).scalar_one_or_none()
    if isinstance(latest, date):
        return latest
    today = pd.Timestamp(date.today())
    last_session = today if XNYS.is_session(today) else XNYS.previous_session(today)
    return last_session.date()


def trailing_month_window(end_date: date, months: int = DEFAULT_SAMPLE_MONTHS) -> tuple[date, date]:
    if months <= 0:
        raise ValueError("months must be positive.")
    start_year = end_date.year
    start_month = end_date.month - (months - 1)
    while start_month <= 0:
        start_month += 12
        start_year -= 1
    return date(start_year, start_month, 1), end_date


def iter_month_labels(start_date: date, end_date: date) -> Iterator[str]:
    cursor = date(start_date.year, start_date.month, 1)
    while cursor <= end_date:
        yield f"{cursor.year:04d}-{cursor.month:02d}"
        next_month = cursor.month + 1
        next_year = cursor.year
        if next_month == 13:
            next_month = 1
            next_year += 1
        cursor = date(next_year, next_month, 1)


def build_sample_context(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sample_tickers: int = DEFAULT_SAMPLE_TICKERS,
    sample_dates_per_month: int = DEFAULT_SAMPLE_DATES_PER_MONTH,
    universe_fetcher: Callable[[date, str], list[str]] = get_historical_members,
    session_factory: Callable | None = None,
    index_name: str = "SP500",
) -> SampleContext:
    effective_end = end_date or latest_available_trade_date(session_factory=session_factory)
    effective_start = start_date or trailing_month_window(effective_end)[0]
    if effective_start > effective_end:
        raise ValueError("start_date must be on or before end_date.")

    universe = tuple(sorted(set(universe_fetcher(effective_end, index_name))))
    sampled_tickers = _sample_items(universe, sample_tickers, seed=42)
    sampled_trade_dates = sample_monthly_trade_dates(
        start_date=effective_start,
        end_date=effective_end,
        sample_dates_per_month=sample_dates_per_month,
        seed=1337,
    )
    return SampleContext(
        start_date=effective_start,
        end_date=effective_end,
        universe=universe,
        sampled_tickers=sampled_tickers,
        sampled_trade_dates=sampled_trade_dates,
    )


def sample_monthly_trade_dates(
    *,
    start_date: date,
    end_date: date,
    sample_dates_per_month: int = DEFAULT_SAMPLE_DATES_PER_MONTH,
    seed: int = 1337,
) -> tuple[date, ...]:
    sessions = XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    if len(sessions) == 0:
        return tuple()

    grouped: dict[str, list[date]] = {}
    for session_label in sessions:
        trade_day = session_label.date()
        month_key = f"{trade_day.year:04d}-{trade_day.month:02d}"
        grouped.setdefault(month_key, []).append(trade_day)

    rng = np.random.default_rng(seed)
    sampled: list[date] = []
    for month_key in sorted(grouped):
        month_days = grouped[month_key]
        if sample_dates_per_month <= 0 or sample_dates_per_month >= len(month_days):
            sampled.extend(month_days)
            continue
        indices = sorted(rng.choice(len(month_days), size=sample_dates_per_month, replace=False).tolist())
        sampled.extend(month_days[index] for index in indices)
    return tuple(sorted(sampled))


def build_feature_family_map(registry: FeatureRegistry) -> dict[str, str]:
    return {definition.name: definition.category for definition in registry.list_features()}


def collect_feature_observations(
    context: SampleContext,
    *,
    registry: FeatureRegistry,
    session_factory: Callable | None = None,
) -> pd.DataFrame:
    if not context.sampled_tickers or not context.sampled_trade_dates:
        return empty_observations_frame()

    feature_family_map = build_feature_family_map(registry)
    grid = _initialize_observation_grid(
        sampled_tickers=context.sampled_tickers,
        sampled_trade_dates=context.sampled_trade_dates,
        feature_family_map=feature_family_map,
    )
    if grid.empty:
        return grid

    pipeline_features = tuple(
        definition.name
        for definition in registry.list_features()
        if definition.category not in WEEK5_FAMILIES
    )
    week5_definitions = tuple(
        definition
        for definition in registry.list_features()
        if definition.category in WEEK5_FAMILIES
    )

    observed_frames: list[pd.DataFrame] = []
    pipeline_observations = collect_pipeline_missing_observations(
        sampled_tickers=context.sampled_tickers,
        sampled_trade_dates=context.sampled_trade_dates,
        feature_family_map={name: feature_family_map[name] for name in pipeline_features},
    )
    if not pipeline_observations.empty:
        observed_frames.append(pipeline_observations)

    week5_observations = collect_scalar_missing_observations(
        sampled_tickers=context.sampled_tickers,
        sampled_trade_dates=context.sampled_trade_dates,
        feature_definitions=week5_definitions,
        session_factory=session_factory,
    )
    if not week5_observations.empty:
        observed_frames.append(week5_observations)

    if not observed_frames:
        return grid.sort_values(["trade_date", "ticker", "family", "feature_name"]).reset_index(drop=True)

    observed = pd.concat(observed_frames, ignore_index=True)
    merged = grid.merge(
        observed,
        on=["ticker", "trade_date", "feature_name"],
        how="left",
        suffixes=("", "_observed"),
    )
    observed_mask = merged["is_missing_observed"].notna()
    merged.loc[observed_mask, "is_missing"] = merged.loc[observed_mask, "is_missing_observed"].astype(bool)
    merged["is_missing"] = merged["is_missing"].astype(bool)
    return (
        merged[["ticker", "trade_date", "feature_name", "family", "is_missing"]]
        .sort_values(["trade_date", "ticker", "family", "feature_name"])
        .reset_index(drop=True)
    )


def collect_pipeline_missing_observations(
    *,
    sampled_tickers: Sequence[str],
    sampled_trade_dates: Sequence[date],
    feature_family_map: dict[str, str],
) -> pd.DataFrame:
    if not sampled_tickers or not sampled_trade_dates or not feature_family_map:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "is_missing"])

    pipeline = FeaturePipeline()
    frames: list[pd.DataFrame] = []
    feature_name_set = set(feature_family_map)
    for trade_day in sampled_trade_dates:
        as_of = next_session_date(trade_day)
        logger.info(
            "week6 family report running pipeline for {} tickers on {} as_of {}",
            len(sampled_tickers),
            trade_day,
            as_of,
        )
        frame = pipeline.run(
            tickers=sampled_tickers,
            start_date=trade_day,
            end_date=trade_day,
            as_of=as_of,
            allow_missing_intraday=True,
        )
        if frame.empty:
            continue
        trade_day_rows = frame.loc[
            pd.to_datetime(frame["trade_date"]).dt.date == trade_day,
            ["ticker", "trade_date", "feature_name", "feature_value"],
        ].copy()
        if trade_day_rows.empty:
            continue
        observations = _extract_pipeline_missing_rows(
            trade_day_rows=trade_day_rows,
            feature_name_set=feature_name_set,
            feature_family_map=feature_family_map,
        )
        if observations.empty:
            continue
        frames.append(observations)

    if not frames:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "is_missing"])
    return pd.concat(frames, ignore_index=True)


def collect_scalar_missing_observations(
    *,
    sampled_tickers: Sequence[str],
    sampled_trade_dates: Sequence[date],
    feature_definitions: Sequence[FeatureDefinition],
    session_factory: Callable | None = None,
) -> pd.DataFrame:
    if not sampled_tickers or not sampled_trade_dates or not feature_definitions:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "is_missing"])

    factory = session_factory or get_session_factory()
    rows: list[dict[str, Any]] = []
    for trade_day in sampled_trade_dates:
        for ticker in sampled_tickers:
            for definition in feature_definitions:
                try:
                    value = definition.compute_fn(ticker=ticker, as_of=trade_day, session_factory=factory)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed computing {definition.name} for {ticker} on {trade_day}: {exc}"
                    ) from exc
                rows.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_day,
                        "feature_name": definition.name,
                        "is_missing": scalar_is_missing(value),
                    },
                )
    return pd.DataFrame(rows, columns=["ticker", "trade_date", "feature_name", "is_missing"])


def next_session_date(trade_day: date) -> date:
    session_label = pd.Timestamp(trade_day)
    return XNYS.next_session(session_label).date()


def scalar_is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (pd.DataFrame, pd.Series, list, tuple, dict, set)):
        return False
    missing = pd.isna(value)
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


@contextmanager
def temporary_enable_week5_flags(enable: bool) -> Iterator[None]:
    original_shorting = settings.ENABLE_SHORTING_FEATURES
    original_analyst = settings.ENABLE_ANALYST_PROXY_FEATURES
    if enable:
        settings.ENABLE_SHORTING_FEATURES = True
        settings.ENABLE_ANALYST_PROXY_FEATURES = True
    try:
        yield
    finally:
        settings.ENABLE_SHORTING_FEATURES = original_shorting
        settings.ENABLE_ANALYST_PROXY_FEATURES = original_analyst


def _initialize_observation_grid(
    *,
    sampled_tickers: Sequence[str],
    sampled_trade_dates: Sequence[date],
    feature_family_map: dict[str, str],
) -> pd.DataFrame:
    if not sampled_tickers or not sampled_trade_dates or not feature_family_map:
        return empty_observations_frame()
    grid = pd.MultiIndex.from_product(
        [tuple(sampled_tickers), tuple(sampled_trade_dates), tuple(sorted(feature_family_map))],
        names=["ticker", "trade_date", "feature_name"],
    ).to_frame(index=False)
    grid["family"] = grid["feature_name"].map(feature_family_map)
    grid["is_missing"] = True
    return grid


def _sample_items(items: Sequence[str], sample_size: int, *, seed: int) -> tuple[str, ...]:
    normalized = tuple(items)
    if sample_size <= 0 or sample_size >= len(normalized):
        return normalized
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(normalized), size=sample_size, replace=False).tolist())
    return tuple(normalized[index] for index in indices)


def _extract_pipeline_missing_rows(
    *,
    trade_day_rows: pd.DataFrame,
    feature_name_set: set[str],
    feature_family_map: dict[str, str],
) -> pd.DataFrame:
    if trade_day_rows.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "is_missing"])

    prepared = trade_day_rows.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"]).dt.date
    prepared["feature_name"] = prepared["feature_name"].astype(str)

    flag_rows = prepared.loc[
        prepared["feature_name"].str.startswith(MISSING_FLAG_PREFIX),
        ["ticker", "trade_date", "feature_name", "feature_value"],
    ].copy()
    if not flag_rows.empty:
        flag_rows["feature_name"] = flag_rows["feature_name"].str.removeprefix(MISSING_FLAG_PREFIX)
        flag_rows = flag_rows.loc[flag_rows["feature_name"].isin(feature_name_set)].copy()
        flag_rows["is_missing"] = pd.to_numeric(flag_rows["feature_value"], errors="coerce").fillna(1.0) >= 0.5
        flag_rows = flag_rows[["ticker", "trade_date", "feature_name", "is_missing"]]

    raw_rows = prepared.loc[
        ~prepared["feature_name"].str.startswith(MISSING_FLAG_PREFIX),
        ["ticker", "trade_date", "feature_name", "feature_value"],
    ].copy()
    raw_rows = raw_rows.loc[raw_rows["feature_name"].isin(feature_name_set)].copy()
    if not raw_rows.empty:
        raw_rows["family"] = raw_rows["feature_name"].map(feature_family_map)
        raw_rows["is_missing"] = pd.to_numeric(raw_rows["feature_value"], errors="coerce").isna()
        raw_rows = raw_rows.loc[raw_rows["family"].isin(PIPELINE_VALUE_NULL_FAMILIES)].copy()
        raw_rows = raw_rows[["ticker", "trade_date", "feature_name", "is_missing"]]
    high_vix_rows = _extract_high_vix_missing_rows(prepared=prepared, feature_name_set=feature_name_set)

    if flag_rows.empty and raw_rows.empty and high_vix_rows.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "is_missing"])
    combined = pd.concat([flag_rows, raw_rows, high_vix_rows], ignore_index=True)
    return combined.drop_duplicates(
        subset=["ticker", "trade_date", "feature_name"],
        keep="last",
    ).reset_index(drop=True)


def _extract_high_vix_missing_rows(*, prepared: pd.DataFrame, feature_name_set: set[str]) -> pd.DataFrame:
    if "high_vix_x_beta" not in feature_name_set or prepared.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "is_missing"])

    trade_day = pd.to_datetime(prepared["trade_date"]).dt.date.iloc[0]
    context_available = _high_vix_context_available(trade_day)
    stock_beta_missing = _feature_missing_map(prepared=prepared, feature_name="stock_beta_252")

    rows = []
    for ticker in sorted(prepared["ticker"].astype(str).unique()):
        rows.append(
            {
                "ticker": ticker,
                "trade_date": trade_day,
                "feature_name": "high_vix_x_beta",
                "is_missing": (not context_available) or stock_beta_missing.get(ticker, True),
            },
        )
    return pd.DataFrame(rows, columns=["ticker", "trade_date", "feature_name", "is_missing"])


def _feature_missing_map(*, prepared: pd.DataFrame, feature_name: str) -> dict[str, bool]:
    flag_name = f"{MISSING_FLAG_PREFIX}{feature_name}"
    flag_rows = prepared.loc[prepared["feature_name"] == flag_name, ["ticker", "feature_value"]].copy()
    if not flag_rows.empty:
        return {
            str(row["ticker"]): bool(pd.to_numeric(row["feature_value"], errors="coerce") >= 0.5)
            for _, row in flag_rows.iterrows()
        }

    raw_rows = prepared.loc[prepared["feature_name"] == feature_name, ["ticker", "feature_value"]].copy()
    return {
        str(row["ticker"]): bool(pd.isna(pd.to_numeric(row["feature_value"], errors="coerce")))
        for _, row in raw_rows.iterrows()
    }


def _high_vix_context_available(trade_day: date) -> bool:
    as_of = datetime.combine(trade_day, time.max, tzinfo=timezone.utc)
    series_history = _load_macro_histories(as_of, lookback_days=282)
    vix_series = series_history.get("VIXCLS", pd.Series(dtype=float)).dropna().astype(float)
    if len(vix_series) < 60:
        return False
    trailing = vix_series.tail(252)
    if len(trailing) < 60:
        return False
    latest = trailing.iloc[-1]
    rolling_std = trailing.std(ddof=0)
    return bool(np.isfinite(latest) and np.isfinite(rolling_std) and not np.isclose(rolling_std, 0.0))
