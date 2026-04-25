# ruff: noqa: E402
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
from loguru import logger
import numpy as np
import pandas as pd
import sqlalchemy as sa
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_single_window_validation import fill_feature_matrix
from scripts.run_walkforward_comparison import WindowSpec, WINDOWS
from src.data.db.models import FeatureStore, StockPrice
from src.data.db.session import get_session_factory
from src.data.finra_short_sale import ShortSaleVolume
from src.data.sources.fmp_grades import GradesEvent
from src.data.sources.fmp_price_target import PriceTargetEvent
from src.data.sources.fmp_ratings import RatingEvent
from src.features import registry as registry_module
from src.features.registry import FeatureRegistry
from src.models.evaluation import information_coefficient_series

try:
    from scripts._week6_family_utils import (
        build_feature_family_map,
        build_sample_context,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _week6_family_utils import (
        build_feature_family_map,
        build_sample_context,
    )


XNYS = xcals.get_calendar("XNYS")
EASTERN = ZoneInfo("America/New_York")
DEFAULT_SAMPLE_TICKERS = 50
DEFAULT_REBALANCE_WEEKDAY = 4
DEFAULT_PANEL_CACHE_PATH = Path("data/features/week7_sample_panel.parquet")
DEFAULT_HORIZON_FAMILIES_PATH = Path("configs/research/horizon_families.yaml")
DEFAULT_MISSINGNESS_AUDIT_PATH = Path("data/reports/missingness_audit.json")
DEFAULT_RETAINED_FEATURES_PATH = Path("data/reports/horizon_retained_features.yaml")
DEFAULT_HISTORICAL_FEATURE_PARQUET_CANDIDATES = (
    Path("data/features/all_features_v6.parquet"),
    Path("data/features/all_features_v5.parquet"),
    Path("data/features/all_features_v4.parquet"),
    Path("data/features/all_features.parquet"),
)
HORIZON_DAY_MAP = {"1d": 1, "5d": 5, "20d": 20, "60d": 60}
SCREENING_MEAN_IC_THRESHOLD = 0.015
SCREENING_T_STAT_THRESHOLD = 2.0
SCREENING_SIGN_WINDOW_THRESHOLD = 7
WEEK5_FAMILIES = frozenset({"shorting", "analyst_proxy"})
SHORTING_FEATURES = frozenset(registry_module._SHORTING_FEATURE_METADATA)
ANALYST_PROXY_FEATURES = frozenset(registry_module._ANALYST_PROXY_FEATURE_METADATA)


@dataclass(frozen=True)
class PanelContext:
    start_date: date
    end_date: date
    sampled_tickers: tuple[str, ...]
    sampled_trade_dates: tuple[date, ...]
    universe_size: int


def screening_windows() -> tuple[WindowSpec, ...]:
    selected: list[WindowSpec] = []
    for window in WINDOWS:
        if window.window_id.startswith("W-") or window.window_id == "W0":
            continue
        selected.append(window)
    return tuple(selected)


def parse_date_arg(value: str) -> date:
    return date.fromisoformat(value)


def default_screening_bounds() -> tuple[date, date]:
    windows = screening_windows()
    return min(window.train_start for window in windows), max(window.test_end for window in windows)


def build_panel_context(
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sample_tickers: int = DEFAULT_SAMPLE_TICKERS,
    rebalance_weekday: int = DEFAULT_REBALANCE_WEEKDAY,
) -> PanelContext:
    default_start, default_end = default_screening_bounds()
    effective_start = start_date or default_start
    effective_end = end_date or default_end
    sample_context = build_sample_context(
        start_date=effective_start,
        end_date=effective_end,
        sample_tickers=sample_tickers,
        sample_dates_per_month=1,
    )
    trade_dates = rebalance_trade_dates(
        start_date=effective_start,
        end_date=effective_end,
        rebalance_weekday=rebalance_weekday,
    )
    return PanelContext(
        start_date=effective_start,
        end_date=effective_end,
        sampled_tickers=tuple(sample_context.sampled_tickers),
        sampled_trade_dates=trade_dates,
        universe_size=sample_context.universe_size,
    )


def rebalance_trade_dates(
    *,
    start_date: date,
    end_date: date,
    rebalance_weekday: int = DEFAULT_REBALANCE_WEEKDAY,
) -> tuple[date, ...]:
    sessions = XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return tuple(
        session.date()
        for session in sessions
        if session.day_of_week == rebalance_weekday
    )


def load_horizon_families(path: Path = DEFAULT_HORIZON_FAMILIES_PATH) -> dict[str, dict[str, Any]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    horizons = payload.get("horizons")
    if not isinstance(horizons, dict):
        raise RuntimeError(f"Invalid horizon families payload at {path}: missing horizons mapping.")
    normalized: dict[str, dict[str, Any]] = {}
    for horizon, config in horizons.items():
        families = list(config.get("families") or [])
        if not families:
            raise RuntimeError(f"{path} horizon {horizon!r} has no included families.")
        normalized[str(horizon)] = {
            "families": [str(family) for family in families],
            "excluded_families": [str(family) for family in list(config.get("excluded_families") or [])],
            "rationale": str(config.get("rationale") or ""),
        }
    return normalized


def load_missingness_exclusion_map(path: Path = DEFAULT_MISSINGNESS_AUDIT_PATH) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    exclusions: dict[str, str] = {}
    for feature_name in summary.get("data_source_block_features") or []:
        exclusions[str(feature_name)] = "data_source_block"
    for feature_name in summary.get("sample_disabled_features") or []:
        exclusions[str(feature_name)] = "sample_disabled_pipeline"
    for feature_name in summary.get("dropped_features") or []:
        exclusions.setdefault(str(feature_name), "missingness_drop")
    return exclusions


def build_registry_feature_maps(registry: FeatureRegistry) -> tuple[dict[str, str], list[str]]:
    family_by_feature = build_feature_family_map(registry)
    return family_by_feature, sorted(family_by_feature)


def horizon_feature_names(
    *,
    registry: FeatureRegistry,
    horizon_label: str,
    included_families: Sequence[str],
    exclusion_map: Mapping[str, str],
) -> list[str]:
    definitions = registry.list_features()
    selected: list[str] = []
    for definition in definitions:
        if definition.category not in included_families:
            continue
        if definition.name in exclusion_map:
            continue
        if not feature_supported_for_horizon(
            feature_name=definition.name,
            family=definition.category,
            horizon_label=horizon_label,
        ):
            continue
        selected.append(definition.name)
    return sorted(selected)


def feature_supported_for_horizon(*, feature_name: str, family: str, horizon_label: str) -> bool:
    if family == "shorting":
        metadata = registry_module._SHORTING_FEATURE_METADATA.get(feature_name)
        return metadata is not None and horizon_label in metadata["horizon_applicability"]
    if family == "analyst_proxy":
        metadata = registry_module._ANALYST_PROXY_FEATURE_METADATA.get(feature_name)
        return metadata is not None and horizon_label in metadata["horizon_applicability"]
    return True


def build_feature_exclusion_map(
    *,
    all_features: Sequence[str],
    family_by_feature: Mapping[str, str],
    horizon_label: str,
    included_families: Sequence[str],
    missingness_exclusions: Mapping[str, str],
) -> dict[str, str]:
    included_family_set = set(included_families)
    result: dict[str, str] = {}
    for feature_name in all_features:
        if feature_name in missingness_exclusions:
            result[feature_name] = str(missingness_exclusions[feature_name])
            continue
        family = family_by_feature[feature_name]
        if family not in included_family_set:
            result[feature_name] = "family_not_in_horizon"
            continue
        if not feature_supported_for_horizon(
            feature_name=feature_name,
            family=family,
            horizon_label=horizon_label,
        ):
            result[feature_name] = "feature_not_applicable_to_horizon"
    return result


def panel_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".meta.json")


def build_or_load_week7_panel(
    *,
    context: PanelContext,
    feature_names: Sequence[str],
    cache_path: Path = DEFAULT_PANEL_CACHE_PATH,
    historical_feature_parquet: Path | None = None,
    session_factory: Callable | None = None,
) -> pd.DataFrame:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = panel_meta_path(cache_path)
    parquet_path = historical_feature_parquet or select_historical_feature_parquet()
    expected_meta = {
        "start_date": context.start_date.isoformat(),
        "end_date": context.end_date.isoformat(),
        "sampled_tickers": list(context.sampled_tickers),
        "sampled_trade_dates": [trade_date.isoformat() for trade_date in context.sampled_trade_dates],
        "feature_names": list(feature_names),
        "historical_feature_parquet": str(parquet_path) if parquet_path is not None else None,
    }
    if cache_path.exists() and meta_path.exists():
        existing_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if existing_meta == expected_meta:
            logger.info("loading cached Week 7 sample panel from {}", cache_path)
            return pd.read_parquet(cache_path)

    feature_names = list(dict.fromkeys(str(name) for name in feature_names))
    family_by_feature, _ = build_registry_feature_maps(
        registry_module.FeatureRegistry(),
    )
    feature_store_names = [name for name in feature_names if family_by_feature.get(name) not in WEEK5_FAMILIES]
    week5_names = [name for name in feature_names if family_by_feature.get(name) in WEEK5_FAMILIES]

    frames: list[pd.DataFrame] = []
    if feature_store_names and parquet_path is not None:
        frames.append(
            load_historical_parquet_panel(
                parquet_path=parquet_path,
                tickers=context.sampled_tickers,
                trade_dates=context.sampled_trade_dates,
                feature_names=feature_store_names,
            ),
        )
    if feature_store_names:
        frames.append(
            load_feature_store_panel(
                tickers=context.sampled_tickers,
                trade_dates=context.sampled_trade_dates,
                feature_names=feature_store_names,
                session_factory=session_factory,
            ),
        )
    if week5_names:
        frames.append(
            build_week5_feature_panel(
                tickers=context.sampled_tickers,
                trade_dates=context.sampled_trade_dates,
                feature_names=week5_names,
                session_factory=session_factory,
            ),
        )

    if frames:
        panel = pd.concat(frames, ignore_index=True)
    else:
        panel = pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel["trade_date"] = pd.to_datetime(panel["trade_date"]).dt.date
    panel["feature_name"] = panel["feature_name"].astype(str)
    panel["feature_value"] = pd.to_numeric(panel["feature_value"], errors="coerce")
    panel.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    panel.drop_duplicates(["trade_date", "ticker", "feature_name"], keep="last", inplace=True)
    panel.reset_index(drop=True, inplace=True)
    panel.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps(expected_meta, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        "saved Week 7 sample panel to {} rows={} dates={} tickers={} features={}",
        cache_path,
        len(panel),
        panel["trade_date"].nunique() if not panel.empty else 0,
        panel["ticker"].nunique() if not panel.empty else 0,
        panel["feature_name"].nunique() if not panel.empty else 0,
    )
    return panel


def load_feature_store_panel(
    *,
    tickers: Sequence[str],
    trade_dates: Sequence[date],
    feature_names: Sequence[str],
    session_factory: Callable | None = None,
) -> pd.DataFrame:
    if not tickers or not trade_dates or not feature_names:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    factory = session_factory or get_session_factory()
    ranked = (
        sa.select(
            FeatureStore.ticker.label("ticker"),
            FeatureStore.calc_date.label("trade_date"),
            FeatureStore.feature_name.label("feature_name"),
            FeatureStore.feature_value.label("feature_value"),
            sa.func.row_number()
            .over(
                partition_by=(
                    FeatureStore.ticker,
                    FeatureStore.calc_date,
                    FeatureStore.feature_name,
                ),
                order_by=FeatureStore.id.desc(),
            )
            .label("rn"),
        )
        .where(
            FeatureStore.ticker.in_([str(ticker).upper() for ticker in tickers]),
            FeatureStore.calc_date.in_(list(trade_dates)),
            FeatureStore.feature_name.in_(list(feature_names)),
        )
        .subquery()
    )
    stmt = sa.select(
        ranked.c.ticker,
        ranked.c.trade_date,
        ranked.c.feature_name,
        ranked.c.feature_value,
    ).where(ranked.c.rn == 1)
    with factory() as session:
        rows = session.execute(stmt).mappings().all()
    frame = pd.DataFrame(rows, columns=["ticker", "trade_date", "feature_name", "feature_value"])
    if frame.empty:
        return frame
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
    return frame


def select_historical_feature_parquet() -> Path | None:
    for candidate in DEFAULT_HISTORICAL_FEATURE_PARQUET_CANDIDATES:
        if candidate.exists():
            return candidate
    logger.warning("no historical feature parquet candidate found; relying on feature_store only")
    return None


def load_historical_parquet_panel(
    *,
    parquet_path: Path,
    tickers: Sequence[str],
    trade_dates: Sequence[date],
    feature_names: Sequence[str],
) -> pd.DataFrame:
    if not tickers or not trade_dates or not feature_names:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    start_date = min(trade_dates)
    end_date = max(trade_dates)
    frame = pd.read_parquet(
        parquet_path,
        columns=["ticker", "trade_date", "feature_name", "feature_value"],
        filters=[
            ("trade_date", ">=", start_date),
            ("trade_date", "<=", end_date),
            ("feature_name", "in", list(feature_names)),
            ("ticker", "in", [str(ticker).upper() for ticker in tickers]),
        ],
    )
    if frame.empty:
        return frame
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["feature_name"] = frame["feature_name"].astype(str)
    frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
    frame = frame.loc[frame["trade_date"].isin(list(trade_dates))].copy()
    return frame


def build_week5_feature_panel(
    *,
    tickers: Sequence[str],
    trade_dates: Sequence[date],
    feature_names: Sequence[str],
    session_factory: Callable | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    requested = set(str(name) for name in feature_names)
    if requested & SHORTING_FEATURES:
        frames.append(
            build_shorting_panel(
                tickers=tickers,
                trade_dates=trade_dates,
                requested_features=sorted(requested & SHORTING_FEATURES),
                session_factory=session_factory,
            ),
        )
    if requested & ANALYST_PROXY_FEATURES:
        frames.append(
            build_analyst_proxy_panel(
                tickers=tickers,
                trade_dates=trade_dates,
                requested_features=sorted(requested & ANALYST_PROXY_FEATURES),
                session_factory=session_factory,
            ),
        )
    if not frames:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    return pd.concat(frames, ignore_index=True)


def build_shorting_panel(
    *,
    tickers: Sequence[str],
    trade_dates: Sequence[date],
    requested_features: Sequence[str],
    session_factory: Callable | None = None,
) -> pd.DataFrame:
    if not tickers or not trade_dates or not requested_features:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    factory = session_factory or get_session_factory()
    start_date = min(trade_dates) - timedelta(days=400)
    end_date = max(trade_dates)
    stmt = (
        sa.select(
            ShortSaleVolume.ticker.label("ticker"),
            ShortSaleVolume.trade_date.label("trade_date"),
            ShortSaleVolume.market.label("market"),
            sa.func.sum(ShortSaleVolume.short_volume).label("short_volume"),
            sa.func.sum(ShortSaleVolume.total_volume).label("total_volume"),
        )
        .where(
            ShortSaleVolume.ticker.in_([str(ticker).upper() for ticker in tickers]),
            ShortSaleVolume.trade_date >= start_date,
            ShortSaleVolume.trade_date <= end_date,
            ShortSaleVolume.knowledge_time <= as_of_end_utc(end_date),
        )
        .group_by(ShortSaleVolume.ticker, ShortSaleVolume.trade_date, ShortSaleVolume.market)
        .order_by(ShortSaleVolume.ticker, ShortSaleVolume.trade_date, ShortSaleVolume.market)
    )
    with factory() as session:
        rows = session.execute(stmt).mappings().all()
    frame = pd.DataFrame(rows, columns=["ticker", "trade_date", "market", "short_volume", "total_volume"])
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["short_volume"] = pd.to_numeric(frame["short_volume"], errors="coerce")
    frame["total_volume"] = pd.to_numeric(frame["total_volume"], errors="coerce")
    frame["ratio"] = np.where(frame["total_volume"].fillna(0.0) > 0.0, frame["short_volume"] / frame["total_volume"], np.nan)

    sessions = tuple(
        session.date() for session in XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    )
    position_by_session = {session: idx for idx, session in enumerate(sessions)}
    rows_out: list[dict[str, object]] = []
    requested = set(requested_features)

    for ticker, ticker_frame in frame.groupby("ticker", sort=True):
        combined = (
            ticker_frame.groupby("trade_date", as_index=False)[["short_volume", "total_volume"]]
            .sum()
            .sort_values("trade_date")
        )
        combined["ratio"] = np.where(
            combined["total_volume"].fillna(0.0) > 0.0,
            combined["short_volume"] / combined["total_volume"],
            np.nan,
        )
        combined_ratio = combined.set_index("trade_date")["ratio"]
        adf_ratio = (
            ticker_frame.loc[ticker_frame["market"] == "ADF", ["trade_date", "ratio"]]
            .drop_duplicates(subset=["trade_date"], keep="last")
            .set_index("trade_date")["ratio"]
        )

        for trade_date in trade_dates:
            session = current_or_previous_session(trade_date)
            session_position = position_by_session.get(session)
            if session_position is None:
                continue
            if "short_sale_ratio_1d" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "short_sale_ratio_1d",
                        "feature_value": combined_ratio.get(session),
                    },
                )
            if "short_sale_ratio_5d" in requested:
                recent = list(sessions[max(0, session_position - 4) : session_position + 1])
                ratios = combined_ratio.reindex(recent).dropna()
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "short_sale_ratio_5d",
                        "feature_value": float(ratios.mean()) if len(ratios) >= 3 else np.nan,
                    },
                )
            if "short_sale_accel" in requested:
                recent = list(sessions[max(0, session_position - 19) : session_position + 1])
                ratios = combined_ratio.reindex(recent).dropna()
                ma_5 = ratios.tail(5)
                ma_20 = ratios.tail(20)
                value = np.nan
                if len(ma_5) >= 3 and len(ma_20) >= 15:
                    value = float(ma_5.mean() - ma_20.mean())
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "short_sale_accel",
                        "feature_value": value,
                    },
                )
            if "abnormal_off_exchange_shorting" in requested:
                recent = list(sessions[max(0, session_position - 90) : session_position + 1])
                if len(recent) < 2:
                    value = np.nan
                else:
                    today_ratio = adf_ratio.get(recent[-1])
                    history = adf_ratio.reindex(recent[:-1]).dropna()
                    std = float(history.std(ddof=0)) if len(history) >= 60 else 0.0
                    if pd.isna(today_ratio) or len(history) < 60 or std <= 0.0:
                        value = np.nan
                    else:
                        value = float((float(today_ratio) - float(history.mean())) / std)
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "abnormal_off_exchange_shorting",
                        "feature_value": value,
                    },
                )

    return pd.DataFrame(rows_out, columns=["ticker", "trade_date", "feature_name", "feature_value"])


def build_analyst_proxy_panel(
    *,
    tickers: Sequence[str],
    trade_dates: Sequence[date],
    requested_features: Sequence[str],
    session_factory: Callable | None = None,
) -> pd.DataFrame:
    if not tickers or not trade_dates or not requested_features:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    factory = session_factory or get_session_factory()
    requested = set(requested_features)
    min_trade_date = min(trade_dates)
    max_trade_date = max(trade_dates)
    lookback_start = min_trade_date - timedelta(days=130)

    with factory() as session:
        grades_rows = session.execute(
            sa.select(
                GradesEvent.ticker,
                GradesEvent.event_date,
                GradesEvent.grade_score_change,
            ).where(
                GradesEvent.ticker.in_([str(ticker).upper() for ticker in tickers]),
                GradesEvent.event_date >= lookback_start,
                GradesEvent.event_date <= max_trade_date,
                GradesEvent.knowledge_time <= as_of_end_utc(max_trade_date),
            ).order_by(GradesEvent.ticker, GradesEvent.event_date, GradesEvent.id)
        ).mappings().all()
        targets_rows = session.execute(
            sa.select(
                PriceTargetEvent.ticker,
                PriceTargetEvent.event_date,
                PriceTargetEvent.knowledge_time,
                PriceTargetEvent.analyst_firm,
                PriceTargetEvent.target_price,
                PriceTargetEvent.is_consensus,
            ).where(
                PriceTargetEvent.ticker.in_([str(ticker).upper() for ticker in tickers]),
                PriceTargetEvent.event_date >= lookback_start,
                PriceTargetEvent.event_date <= max_trade_date,
                PriceTargetEvent.knowledge_time <= as_of_end_utc(max_trade_date),
            ).order_by(
                PriceTargetEvent.ticker,
                PriceTargetEvent.event_date,
                PriceTargetEvent.knowledge_time,
                PriceTargetEvent.id,
            )
        ).mappings().all()
        ratings_rows = session.execute(
            sa.select(
                RatingEvent.ticker,
                RatingEvent.event_date,
                RatingEvent.rating_score,
            ).where(
                RatingEvent.ticker.in_([str(ticker).upper() for ticker in tickers]),
                RatingEvent.event_date >= lookback_start,
                RatingEvent.event_date <= max_trade_date,
                RatingEvent.knowledge_time <= as_of_end_utc(max_trade_date),
            ).order_by(RatingEvent.ticker, RatingEvent.event_date)
        ).mappings().all()
        price_rows = session.execute(
            sa.select(
                StockPrice.ticker,
                StockPrice.trade_date,
                StockPrice.knowledge_time,
                StockPrice.close,
            ).where(
                StockPrice.ticker.in_([str(ticker).upper() for ticker in tickers]),
                StockPrice.trade_date >= lookback_start,
                StockPrice.trade_date <= max_trade_date,
                StockPrice.knowledge_time <= as_of_end_utc(max_trade_date),
            ).order_by(StockPrice.ticker, StockPrice.knowledge_time, StockPrice.trade_date)
        ).mappings().all()

    grades = pd.DataFrame(grades_rows)
    targets = pd.DataFrame(targets_rows)
    ratings = pd.DataFrame(ratings_rows)
    prices = pd.DataFrame(price_rows)

    if not grades.empty:
        grades["event_date"] = pd.to_datetime(grades["event_date"]).dt.date
        grades["grade_score_change"] = pd.to_numeric(grades["grade_score_change"], errors="coerce")
    if not targets.empty:
        targets["event_date"] = pd.to_datetime(targets["event_date"]).dt.date
        targets["knowledge_time"] = pd.to_datetime(targets["knowledge_time"], utc=True)
        targets["target_price"] = pd.to_numeric(targets["target_price"], errors="coerce")
        targets["analyst_firm"] = targets["analyst_firm"].where(targets["analyst_firm"].notna())
    if not ratings.empty:
        ratings["event_date"] = pd.to_datetime(ratings["event_date"]).dt.date
        ratings["rating_score"] = pd.to_numeric(ratings["rating_score"], errors="coerce")
    close_lookup = build_visible_close_lookup(
        price_rows=prices,
        tickers=tickers,
        trade_dates=trade_dates,
    )

    rows_out: list[dict[str, object]] = []
    normalized_trade_dates = list(trade_dates)

    for ticker in [str(item).upper() for item in tickers]:
        ticker_grades = grades.loc[grades["ticker"] == ticker].copy() if not grades.empty else pd.DataFrame()
        ticker_targets = targets.loc[targets["ticker"] == ticker].copy() if not targets.empty else pd.DataFrame()
        ticker_ratings = ratings.loc[ratings["ticker"] == ticker].copy() if not ratings.empty else pd.DataFrame()
        if not ticker_targets.empty:
            ticker_targets = ticker_targets.loc[ticker_targets["is_consensus"] == False].copy()  # noqa: E712

        for trade_date in normalized_trade_dates:
            if "net_grade_change_5d" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "net_grade_change_5d",
                        "feature_value": sum_grade_changes(ticker_grades, trade_date=trade_date, horizon_days=5),
                    },
                )
            if "net_grade_change_20d" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "net_grade_change_20d",
                        "feature_value": sum_grade_changes(ticker_grades, trade_date=trade_date, horizon_days=20),
                    },
                )
            if "net_grade_change_60d" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "net_grade_change_60d",
                        "feature_value": sum_grade_changes(ticker_grades, trade_date=trade_date, horizon_days=60),
                    },
                )
            if "upgrade_count" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "upgrade_count",
                        "feature_value": count_grade_changes(
                            ticker_grades,
                            trade_date=trade_date,
                            horizon_days=20,
                            positive=True,
                        ),
                    },
                )
            if "downgrade_count" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "downgrade_count",
                        "feature_value": count_grade_changes(
                            ticker_grades,
                            trade_date=trade_date,
                            horizon_days=20,
                            positive=False,
                        ),
                    },
                )
            if "consensus_upside" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "consensus_upside",
                        "feature_value": compute_consensus_upside_from_frame(
                            ticker_targets,
                            trade_date=trade_date,
                            latest_close=close_lookup.get((ticker, trade_date)),
                        ),
                    },
                )
            if "target_price_drift" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "target_price_drift",
                        "feature_value": compute_target_price_drift_from_frame(
                            ticker_targets,
                            trade_date=trade_date,
                            latest_close=close_lookup.get((ticker, trade_date)),
                        ),
                    },
                )
            if "target_dispersion_proxy" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "target_dispersion_proxy",
                        "feature_value": compute_target_dispersion_from_frame(
                            ticker_targets,
                            trade_date=trade_date,
                        ),
                    },
                )
            if "coverage_change_proxy" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "coverage_change_proxy",
                        "feature_value": compute_coverage_change_from_frame(
                            ticker_targets,
                            trade_date=trade_date,
                        ),
                    },
                )
            if "financial_health_trend" in requested:
                rows_out.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "feature_name": "financial_health_trend",
                        "feature_value": compute_financial_health_trend_from_frame(
                            ticker_ratings,
                            trade_date=trade_date,
                        ),
                    },
                )

    return pd.DataFrame(rows_out, columns=["ticker", "trade_date", "feature_name", "feature_value"])


def build_visible_close_lookup(
    *,
    price_rows: pd.DataFrame,
    tickers: Sequence[str],
    trade_dates: Sequence[date],
) -> dict[tuple[str, date], float]:
    if price_rows.empty:
        return {}
    frame = price_rows.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["knowledge_time"] = pd.to_datetime(frame["knowledge_time"], utc=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame.sort_values(["ticker", "knowledge_time", "trade_date"], inplace=True)
    lookup: dict[tuple[str, date], float] = {}
    trade_dates_sorted = sorted(trade_dates)
    for ticker in [str(item).upper() for item in tickers]:
        ticker_rows = frame.loc[frame["ticker"] == ticker].copy()
        if ticker_rows.empty:
            continue
        current_close = np.nan
        current_trade_date: date | None = None
        pointer = 0
        rows = list(ticker_rows.itertuples(index=False))
        for trade_date in trade_dates_sorted:
            cutoff = as_of_end_utc(trade_date)
            while pointer < len(rows) and rows[pointer].knowledge_time <= cutoff:
                row = rows[pointer]
                if current_trade_date is None or row.trade_date >= current_trade_date:
                    current_trade_date = row.trade_date
                    current_close = float(row.close) if pd.notna(row.close) else np.nan
                pointer += 1
            if current_trade_date is not None and current_trade_date <= trade_date and pd.notna(current_close):
                lookup[(ticker, trade_date)] = float(current_close)
    return lookup


def sum_grade_changes(frame: pd.DataFrame, *, trade_date: date, horizon_days: int) -> float:
    if frame.empty:
        return np.nan
    start_date = trade_date - timedelta(days=horizon_days)
    window = frame.loc[(frame["event_date"] >= start_date) & (frame["event_date"] <= trade_date), "grade_score_change"].dropna()
    if window.empty:
        return 0.0
    return float(window.sum())


def count_grade_changes(frame: pd.DataFrame, *, trade_date: date, horizon_days: int, positive: bool) -> float:
    if frame.empty:
        return 0.0
    start_date = trade_date - timedelta(days=horizon_days)
    window = frame.loc[(frame["event_date"] >= start_date) & (frame["event_date"] <= trade_date), "grade_score_change"].dropna()
    if positive:
        return float((window > 0).sum())
    return float((window < 0).sum())


def compute_consensus_upside_from_frame(frame: pd.DataFrame, *, trade_date: date, latest_close: float | None) -> float:
    if frame.empty or latest_close is None or latest_close == 0:
        return np.nan
    start_date = trade_date - timedelta(days=60)
    window = frame.loc[(frame["event_date"] >= start_date) & (frame["event_date"] <= trade_date)].copy()
    if window.empty:
        return np.nan
    window.sort_values(["event_date", "knowledge_time"], inplace=True)
    window.dropna(subset=["analyst_firm", "target_price"], inplace=True)
    if window.empty:
        return np.nan
    latest = window.drop_duplicates(subset=["analyst_firm"], keep="last")
    if latest.empty:
        return np.nan
    consensus_target = latest["target_price"].mean()
    if pd.isna(consensus_target):
        return np.nan
    return float((float(consensus_target) - float(latest_close)) / float(latest_close))


def compute_target_price_drift_from_frame(frame: pd.DataFrame, *, trade_date: date, latest_close: float | None) -> float:
    if frame.empty or latest_close is None or latest_close == 0:
        return np.nan
    start_date = trade_date - timedelta(days=60)
    window = frame.loc[(frame["event_date"] >= start_date) & (frame["event_date"] <= trade_date), ["event_date", "target_price"]].copy()
    window["target_price"] = pd.to_numeric(window["target_price"], errors="coerce")
    window.dropna(subset=["target_price"], inplace=True)
    if window.empty or window["event_date"].nunique() < 5:
        return np.nan
    min_event_date = window["event_date"].min()
    x = np.array([(event_date - min_event_date).days for event_date in window["event_date"]], dtype=float)
    if len(np.unique(x)) < 2:
        return np.nan
    y = window["target_price"].astype(float).to_numpy()
    slope = float(np.polyfit(x, y, 1)[0])
    return float((slope * 60.0) / float(latest_close))


def compute_target_dispersion_from_frame(frame: pd.DataFrame, *, trade_date: date) -> float:
    if frame.empty:
        return np.nan
    start_date = trade_date - timedelta(days=60)
    window = frame.loc[(frame["event_date"] >= start_date) & (frame["event_date"] <= trade_date)].copy()
    window.dropna(subset=["analyst_firm", "target_price"], inplace=True)
    if window.empty:
        return np.nan
    window.sort_values(["event_date", "knowledge_time"], inplace=True)
    latest = window.drop_duplicates(subset=["analyst_firm"], keep="last")
    if latest["analyst_firm"].nunique() < 3:
        return np.nan
    prices = pd.to_numeric(latest["target_price"], errors="coerce").dropna()
    if len(prices) < 3:
        return np.nan
    mean_value = float(prices.mean())
    if mean_value == 0.0:
        return np.nan
    return float(prices.std(ddof=0) / mean_value)


def compute_coverage_change_from_frame(frame: pd.DataFrame, *, trade_date: date) -> float:
    if frame.empty:
        return np.nan
    recent_start = trade_date - timedelta(days=60)
    prior_start = trade_date - timedelta(days=120)
    prior_end = recent_start
    window = frame.loc[(frame["event_date"] >= prior_start) & (frame["event_date"] <= trade_date)].copy()
    window.dropna(subset=["analyst_firm"], inplace=True)
    if window.empty or not (window["event_date"] <= prior_end).any():
        return np.nan
    recent_firms = {
        str(row.analyst_firm)
        for row in window.itertuples(index=False)
        if recent_start <= row.event_date <= trade_date
    }
    prior_firms = {
        str(row.analyst_firm)
        for row in window.itertuples(index=False)
        if prior_start <= row.event_date <= prior_end
    }
    return float(len(recent_firms) - len(prior_firms))


def compute_financial_health_trend_from_frame(frame: pd.DataFrame, *, trade_date: date) -> float:
    if frame.empty:
        return np.nan
    current = frame.loc[frame["event_date"] <= trade_date, "rating_score"].dropna()
    prior = frame.loc[frame["event_date"] <= (trade_date - timedelta(days=60)), "rating_score"].dropna()
    if current.empty or prior.empty:
        return np.nan
    return float(current.iloc[-1] - prior.iloc[-1])


def current_or_previous_session(value: date) -> date:
    session_label = pd.Timestamp(value)
    if XNYS.is_session(session_label):
        return session_label.date()
    return XNYS.previous_session(session_label).date()


def as_of_end_utc(trade_date: date) -> datetime:
    local_dt = datetime.combine(trade_date, time(23, 59, 59), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)


def load_label_series(
    *,
    tickers: Sequence[str],
    trade_dates: Sequence[date],
    horizon_days: int,
    label_path: Path | None = None,
) -> pd.Series:
    path = label_path or Path(f"data/labels/forward_returns_{horizon_days}d.parquet")
    labels = pd.read_parquet(path)
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    filtered = labels.loc[
        labels["ticker"].isin([str(ticker).upper() for ticker in tickers])
        & (labels["horizon"] == horizon_days)
        & (labels["trade_date"].dt.date.isin(list(trade_dates)))
    ].copy()
    series = (
        filtered.set_index(["trade_date", "ticker"])["excess_return"]
        .sort_index()
        .dropna()
    )
    return series


def build_wide_feature_matrix(panel: pd.DataFrame, *, feature_names: Sequence[str]) -> pd.DataFrame:
    if panel.empty:
        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["trade_date", "ticker"])
        return pd.DataFrame(index=index, columns=list(feature_names), dtype=float)
    subset = panel.loc[panel["feature_name"].isin(list(feature_names)), ["ticker", "trade_date", "feature_name", "feature_value"]].copy()
    if subset.empty:
        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["trade_date", "ticker"])
        return pd.DataFrame(index=index, columns=list(feature_names), dtype=float)
    subset["trade_date"] = pd.to_datetime(subset["trade_date"])
    subset["ticker"] = subset["ticker"].astype(str).str.upper()
    subset["feature_value"] = pd.to_numeric(subset["feature_value"], errors="coerce")
    subset.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    subset.drop_duplicates(["trade_date", "ticker", "feature_name"], keep="last", inplace=True)
    matrix = subset.pivot(index=["trade_date", "ticker"], columns="feature_name", values="feature_value").sort_index()
    matrix = matrix.reindex(columns=list(feature_names))
    matrix.columns.name = None
    return matrix


def compute_feature_screening_metrics(
    *,
    feature_series: pd.Series,
    label_series: pd.Series,
    windows: Sequence[WindowSpec],
) -> dict[str, Any]:
    aligned = pd.concat(
        [label_series.rename("y_true"), feature_series.rename("y_pred")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return {
            "mean_ic": np.nan,
            "t_stat": np.nan,
            "sign_consistent_windows": 0,
            "window_ics": {},
            "daily_ic_count": 0,
            "n_obs": 0,
        }

    daily_ic = information_coefficient_series(aligned["y_true"], aligned["y_pred"])
    mean_ic = float(daily_ic.mean()) if not daily_ic.empty else np.nan
    t_stat = series_t_stat(daily_ic)
    dominant_sign = np.sign(mean_ic) if pd.notna(mean_ic) and not np.isclose(mean_ic, 0.0) else 0.0
    sign_consistent_windows = 0
    window_ics: dict[str, float | None] = {}
    for window in windows:
        mask = (
            (pd.to_datetime(daily_ic.index).date >= window.test_start)
            & (pd.to_datetime(daily_ic.index).date <= window.test_end)
        )
        window_series = daily_ic.loc[mask]
        window_ic = float(window_series.mean()) if not window_series.empty else np.nan
        window_ics[window.window_id] = None if pd.isna(window_ic) else window_ic
        if dominant_sign != 0.0 and pd.notna(window_ic) and np.sign(window_ic) == dominant_sign:
            sign_consistent_windows += 1

    return {
        "mean_ic": mean_ic,
        "t_stat": t_stat,
        "sign_consistent_windows": int(sign_consistent_windows),
        "window_ics": window_ics,
        "daily_ic_count": int(len(daily_ic)),
        "n_obs": int(len(aligned)),
    }


def series_t_stat(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if len(clean) < 2:
        return np.nan
    std = float(clean.std(ddof=1))
    if std <= 0.0 or np.isnan(std):
        return np.nan
    return float(clean.mean() / (std / np.sqrt(len(clean))))


def screening_status(
    *,
    mean_ic: float,
    t_stat: float,
    sign_consistent_windows: int,
    mean_ic_threshold: float = SCREENING_MEAN_IC_THRESHOLD,
    t_stat_threshold: float = SCREENING_T_STAT_THRESHOLD,
    sign_window_threshold: int = SCREENING_SIGN_WINDOW_THRESHOLD,
) -> str:
    if pd.isna(mean_ic) or pd.isna(t_stat):
        return "FAIL"
    if abs(float(mean_ic)) < float(mean_ic_threshold):
        return "FAIL"
    if abs(float(t_stat)) < float(t_stat_threshold):
        return "FAIL"
    if int(sign_consistent_windows) < int(sign_window_threshold):
        return "FAIL"
    return "PASS"


def selected_top_families(rows: pd.DataFrame, *, top_n: int = 3) -> list[str]:
    passed = rows.loc[rows["status"] == "PASS"].copy()
    if passed.empty:
        return []
    summary = (
        passed.assign(abs_mean_ic=lambda frame: frame["mean_ic"].abs())
        .groupby("family", as_index=False)
        .agg(
            pass_count=("feature", "count"),
            avg_abs_ic=("abs_mean_ic", "mean"),
        )
        .sort_values(["pass_count", "avg_abs_ic", "family"], ascending=[False, False, True])
    )
    return summary["family"].head(top_n).astype(str).tolist()


def build_retained_features_payload(
    *,
    report_rows_by_horizon: Mapping[str, pd.DataFrame],
) -> dict[str, Any]:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizons": {},
    }
    for horizon, rows in report_rows_by_horizon.items():
        passed = rows.loc[rows["status"] == "PASS", "feature"].astype(str).tolist()
        payload["horizons"][horizon] = {
            "retained": passed,
            "top_3_families": selected_top_families(rows),
        }
    return payload


def write_yaml_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def summarize_panel_coverage(panel: pd.DataFrame, feature_names: Sequence[str]) -> dict[str, Any]:
    if panel.empty:
        return {"rows": 0, "dates": 0, "tickers": 0, "features": 0}
    return {
        "rows": int(len(panel)),
        "dates": int(pd.to_datetime(panel["trade_date"]).nunique()),
        "tickers": int(panel["ticker"].astype(str).str.upper().nunique()),
        "features": int(len(set(feature_names))),
    }


def prepare_filled_matrix(matrix: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    subset = matrix.reindex(columns=list(feature_names))
    if subset.empty:
        return subset
    return fill_feature_matrix(subset)
