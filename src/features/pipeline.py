from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from loguru import logger
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.data.db.models import FeatureStore
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_session_factory
from src.features.fundamental import FUNDAMENTAL_FEATURE_NAMES, compute_fundamental_features
from src.features.macro import MACRO_FEATURE_NAMES, compute_macro_features
from src.features.preprocessing import preprocess_features
from src.features.sector import SECTOR_RELATIVE_RETAINED, compute_sector_relative_from_raw_features
from src.features.technical import TECHNICAL_FEATURE_NAMES, compute_technical_features

COMPOSITE_FEATURE_NAMES = (
    "ret_vol_interaction_20d",
    "ret_vol_interaction_60d",
    "mom_vol_adj_20d",
    "mom_vol_adj_60d",
    "value_mom_pe",
    "value_mom_pb",
    "quality_value_roe_pb",
    "quality_value_roa_pe",
    "fcf_mom_20d",
    "leverage_vol_20d",
    "valuation_spread_pb_pe",
    "margin_quality_combo",
    "profitability_combo",
    "liquidity_momentum",
    "mean_reversion_combo",
    "trend_confirmation",
    "risk_sentiment",
    "spread_stress",
    "breadth_momentum",
    "macro_risk_on",
)


class FeaturePipeline:
    def __init__(self) -> None:
        self.last_batch_id: str | None = None

    def run(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
        as_of: date | datetime,
    ) -> pd.DataFrame:
        """Generate PIT-safe features for market dates in the requested window.

        `as_of` controls which source records are visible through PIT queries. The
        returned `trade_date` values remain market dates, not guaranteed feature
        availability timestamps. For example, daily price inputs may only become
        knowable after their market date because Week 2 price ingestion stores
        daily bars with T+1 `knowledge_time`.
        """
        normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker))
        if not normalized_tickers:
            raise ValueError("At least one ticker is required.")

        start = _coerce_date(start_date)
        end = _coerce_date(end_date)
        as_of_ts = _coerce_as_of_datetime(as_of)
        if as_of_ts.date() < end:
            raise ValueError("as_of must be on or after end_date for PIT feature generation.")
        if as_of_ts.date() == end:
            logger.warning(
                "feature pipeline received as_of={} equal to end_date={}; "
                "the last market date may be unavailable when source prices use T+1 knowledge_time",
                as_of_ts.date(),
                end,
            )
        # Residual-momentum features need roughly 252 regression days plus a
        # 60-day aggregation window, so the raw PIT pull needs a wider buffer
        # than the original 400-calendar-day lookback.
        history_start = start - timedelta(days=520)

        logger.info(
            "running feature pipeline for {} tickers from {} to {} as_of {}",
            len(normalized_tickers),
            start,
            end,
            as_of_ts,
        )
        price_tickers = tuple(dict.fromkeys([*normalized_tickers, "SPY"]))
        prices = get_prices_pit(
            tickers=price_tickers,
            start_date=history_start,
            end_date=end,
            as_of=as_of_ts,
        )
        if prices.empty:
            logger.warning("feature pipeline found no PIT prices for requested tickers")
            return _empty_feature_frame()

        prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
        prices.sort_values(["ticker", "trade_date"], inplace=True)
        stock_prices = prices.loc[prices["ticker"].isin(normalized_tickers)].copy()
        market_prices = prices.loc[prices["ticker"] == "SPY"].copy()
        output_prices = stock_prices.loc[
            (stock_prices["trade_date"] >= start) & (stock_prices["trade_date"] <= end)
        ].copy()
        if output_prices.empty:
            logger.warning("feature pipeline has no prices inside the requested output window")
            return _empty_feature_frame()

        technical = compute_technical_features(stock_prices, market_prices_df=market_prices)
        technical = technical.loc[
            (technical["trade_date"] >= start)
            & (technical["trade_date"] <= end)
            & (technical["ticker"].isin(normalized_tickers))
        ].copy()

        fundamental_frames = []
        for ticker in normalized_tickers:
            ticker_prices = output_prices.loc[output_prices["ticker"] == ticker]
            if ticker_prices.empty:
                continue
            fundamental_frames.append(
                compute_fundamental_features(
                    ticker=ticker,
                    as_of=as_of_ts,
                    prices_df=ticker_prices,
                ),
            )
        fundamentals = (
            pd.concat(fundamental_frames, ignore_index=True)
            if fundamental_frames
            else _empty_feature_frame()
        )

        macro = self._compute_broadcast_macro_features(output_prices, as_of_ts)
        base_features = pd.concat([technical, fundamentals, macro], ignore_index=True)
        composite = compute_composite_features(base_features)
        all_features = pd.concat([base_features, composite], ignore_index=True)

        # S1.4: Sector-relative features (computed from raw fundamentals before rank normalization)
        sector_rel_frames: list[pd.DataFrame] = []
        for td in sorted(set(fundamentals["trade_date"].unique()) if not fundamentals.empty else []):
            td_date = td if isinstance(td, date) else td.date() if hasattr(td, "date") else td
            cross_section = all_features.loc[all_features["trade_date"] == td]
            sector_rel = compute_sector_relative_from_raw_features(cross_section, td_date)
            if not sector_rel.empty:
                sector_rel_frames.append(sector_rel)
        if sector_rel_frames:
            all_features = pd.concat([all_features, *sector_rel_frames], ignore_index=True)

        # TODO(phase1-w3-w4): IMPLEMENTATION_PLAN 3.10 IC-based feature screening
        # remains notebook/report work and is intentionally outside this pipeline.
        processed = preprocess_features(all_features)

        batch_id = str(uuid.uuid4())
        self.last_batch_id = batch_id
        processed.attrs["batch_id"] = batch_id
        logger.info("feature pipeline completed batch {} with {} rows", batch_id, len(processed))
        return processed

    def save_to_store(
        self,
        features_df: pd.DataFrame,
        batch_id: str,
        *,
        batch_size: int = 10_000,
    ) -> int:
        if features_df.empty:
            return 0

        session_factory = get_session_factory()
        rows_saved = 0
        with session_factory() as session:
            try:
                for start in range(0, len(features_df), batch_size):
                    batch = features_df.iloc[start : start + batch_size]
                    records = [
                        {
                            "ticker": str(row.ticker).upper(),
                            # calc_date stores the market date for the feature row. The
                            # actual availability time remains governed by PIT input
                            # knowledge_time and the batch's as_of cutoff used in run().
                            "calc_date": _coerce_date(row.trade_date),
                            "feature_name": str(row.feature_name),
                            "feature_value": _to_decimal(row.feature_value),
                            "is_filled": bool(getattr(row, "is_filled", False)),
                            "batch_id": batch_id,
                        }
                        for row in batch.itertuples(index=False)
                    ]
                    statement = insert(FeatureStore).values(records)
                    upsert = statement.on_conflict_do_update(
                        constraint="uq_feature_store_batch",
                        set_={
                            "feature_value": statement.excluded.feature_value,
                            "is_filled": statement.excluded.is_filled,
                        },
                    )
                    session.execute(upsert)
                    rows_saved += len(records)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.opt(exception=exc).error("failed to save feature batch {} to store", batch_id)
                raise

        logger.info("saved {} feature rows to feature_store for batch {}", rows_saved, batch_id)
        return rows_saved

    def save_to_parquet(
        self,
        features_df: pd.DataFrame,
        batch_id: str,
        output_dir: str = "data/features/",
    ) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"{batch_id}.parquet"
        parquet_frame = features_df.copy()
        parquet_frame["batch_id"] = batch_id
        try:
            parquet_frame.to_parquet(file_path, index=False)
        except Exception as exc:
            logger.opt(exception=exc).error("failed to write parquet feature batch {}", batch_id)
            raise

        logger.info("saved feature batch {} to {}", batch_id, file_path)
        return file_path

    def _compute_broadcast_macro_features(
        self,
        prices_df: pd.DataFrame,
        as_of: datetime,
    ) -> pd.DataFrame:
        date_ticker_pairs = prices_df[["ticker", "trade_date"]].drop_duplicates().sort_values(
            ["trade_date", "ticker"],
        )
        macro_frames = []
        for trade_date in date_ticker_pairs["trade_date"].drop_duplicates().tolist():
            effective_as_of = min(trade_date, as_of.date())
            macro_frames.append(compute_macro_features(as_of=effective_as_of))

        if not macro_frames:
            return _empty_feature_frame()

        macro_by_date = pd.concat(macro_frames, ignore_index=True)
        broadcast = date_ticker_pairs.merge(macro_by_date, on="trade_date", how="left")
        return broadcast[["ticker", "trade_date", "feature_name", "feature_value"]]


def compute_composite_features(base_features_df: pd.DataFrame) -> pd.DataFrame:
    if base_features_df.empty:
        return _empty_feature_frame()

    wide = (
        base_features_df.pivot_table(
            index=["ticker", "trade_date"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        )
        .sort_index()
    )
    if wide.empty:
        return _empty_feature_frame()

    required_base_features = [
        "ret_20d",
        "ret_60d",
        "volume_ratio_20d",
        "vol_20d",
        "vol_60d",
        "momentum_rank_60d",
        "pe_ratio",
        "pb_ratio",
        "roe",
        "roa",
        "fcf_yield",
        "debt_to_equity",
        "gross_margin",
        "operating_margin",
        "bb_position",
        "rsi_14",
        "macd_histogram",
        "stoch_d",
        "market_ret_20d",
        "vix_change_5d",
        "credit_spread",
        "yield_spread_10y2y",
        "sp500_breadth",
    ]
    wide = wide.reindex(columns=sorted(set(list(wide.columns) + required_base_features)))

    composite = pd.DataFrame(index=wide.index)
    composite["ret_vol_interaction_20d"] = wide.get("ret_20d") * wide.get("volume_ratio_20d")
    composite["ret_vol_interaction_60d"] = wide.get("ret_60d") * wide.get("volume_ratio_20d")
    composite["mom_vol_adj_20d"] = _safe_series_divide(wide.get("ret_20d"), wide.get("vol_20d"))
    composite["mom_vol_adj_60d"] = _safe_series_divide(wide.get("ret_60d"), wide.get("vol_60d"))
    composite["value_mom_pe"] = wide.get("momentum_rank_60d") - wide.get("pe_ratio")
    composite["value_mom_pb"] = wide.get("momentum_rank_60d") - wide.get("pb_ratio")
    composite["quality_value_roe_pb"] = wide.get("roe") - wide.get("pb_ratio")
    composite["quality_value_roa_pe"] = wide.get("roa") - wide.get("pe_ratio")
    composite["fcf_mom_20d"] = wide.get("fcf_yield") * wide.get("ret_20d")
    composite["leverage_vol_20d"] = wide.get("debt_to_equity") * wide.get("vol_20d")
    composite["valuation_spread_pb_pe"] = wide.get("pb_ratio") - wide.get("pe_ratio")
    composite["margin_quality_combo"] = wide.get("gross_margin") + wide.get("operating_margin")
    composite["profitability_combo"] = wide.get("roe") + wide.get("roa")
    composite["liquidity_momentum"] = wide.get("volume_ratio_20d") * wide.get("ret_20d")
    composite["mean_reversion_combo"] = wide.get("bb_position") - _safe_series_divide(wide.get("rsi_14"), 100)
    composite["trend_confirmation"] = wide.get("macd_histogram") + _safe_series_divide(wide.get("stoch_d"), 100)
    # MACRO_REGIME: these composites depend only on market-wide inputs and
    # should be treated as regime/context overlays rather than stock selectors.
    composite["risk_sentiment"] = wide.get("market_ret_20d") - wide.get("vix_change_5d")
    composite["spread_stress"] = wide.get("credit_spread") - wide.get("yield_spread_10y2y")
    composite["breadth_momentum"] = wide.get("sp500_breadth") + wide.get("market_ret_20d")
    composite["macro_risk_on"] = wide.get("yield_spread_10y2y") - wide.get("credit_spread")

    long_frame = (
        composite.reset_index()
        .melt(
            id_vars=["ticker", "trade_date"],
            value_vars=list(COMPOSITE_FEATURE_NAMES),
            var_name="feature_name",
            value_name="feature_value",
        )
        .sort_values(["trade_date", "ticker", "feature_name"])
        .reset_index(drop=True)
    )
    return long_frame


def _safe_series_divide(left: pd.Series | None, right: pd.Series | float | int | None) -> pd.Series:
    if left is None:
        return pd.Series(dtype=float)
    if isinstance(right, pd.Series):
        denominator = right.replace(0, np.nan)
    elif right in (None, 0):
        denominator = np.nan
    else:
        denominator = right
    return left / denominator


def _coerce_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value


def _coerce_as_of_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime.combine(value, datetime.max.time(), tzinfo=timezone.utc)


def _to_decimal(value: object) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    return Decimal(str(round(float(value), 8)))


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
