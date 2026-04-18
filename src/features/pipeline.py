from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from loguru import logger
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.data.db.models import FeatureStore, StockMinuteAggs
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_session_factory
from src.data.sources.fmp_analyst import AnalystEstimate
from src.data.sources.fmp_earnings import EarningsEstimate
from src.data.sources.fmp_insider import InsiderTrade
from src.data.sources.fmp_sec_filings import SecFiling
from src.data.sources.polygon_short_interest import ShortInterest
from src.features.alternative import ALTERNATIVE_FEATURE_NAMES
from src.features.fundamental import FUNDAMENTAL_FEATURE_NAMES, compute_fundamental_features
from src.features.intraday import INTRADAY_FEATURE_NAMES, compute_intraday_features
from src.features.macro import MACRO_FEATURE_NAMES, compute_macro_features
from src.features.preprocessing import preprocess_features
from src.features.sector import SECTOR_RELATIVE_RETAINED, compute_sector_relative_from_raw_features, load_sector_map_pit
from src.features.sector_rotation import SECTOR_ROTATION_ETF_TICKERS, compute_sector_rotation_features
from src.features.technical import TECHNICAL_FEATURE_NAMES, _attach_pit_shares_outstanding, compute_technical_features

COMPOSITE_FEATURE_NAMES = (
    "ret_vol_interaction_20d",
    "ret_vol_interaction_60d",
    "mom_vol_adj_20d",
    "mom_vol_adj_60d",
    "breadth_pct_above_20dma",
    "return_dispersion_20d",
    "narrow_leadership_score",
    "high_vix_x_beta",
    "credit_widening_x_leverage",
    "curve_inverted_x_growth",
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
FEATURE_EXPORT_COLUMNS = ["ticker", "trade_date", "feature_name", "feature_value", "is_filled"]


class IntradayHistoryError(RuntimeError):
    """Raised when intraday minute history is required but unavailable."""


def prepare_feature_export_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize pipeline output into the canonical export contract.

    This frame is the single source for both parquet research caches and
    feature_store writes. Any path that persists features should use this exact
    transformation first.
    """
    required_columns = {"ticker", "trade_date", "feature_name", "feature_value"}
    missing = sorted(required_columns - set(features_df.columns))
    if missing:
        raise ValueError(f"features_df is missing required columns for export: {missing}")

    frame = features_df.loc[:, [column for column in FEATURE_EXPORT_COLUMNS if column in features_df.columns]].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["feature_name"] = frame["feature_name"].astype(str)
    frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
    if "is_filled" not in frame.columns:
        frame["is_filled"] = False
    else:
        frame["is_filled"] = frame["is_filled"].fillna(False).astype(bool)
    frame = frame[FEATURE_EXPORT_COLUMNS]
    frame.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    frame.drop_duplicates(["trade_date", "ticker", "feature_name"], keep="last", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def feature_store_records_from_frame(frame: pd.DataFrame, *, batch_id: str) -> list[dict[str, object]]:
    prepared = prepare_feature_export_frame(frame)
    return [
        {
            "ticker": str(row.ticker).upper(),
            "calc_date": _coerce_date(row.trade_date),
            "feature_name": str(row.feature_name),
            "feature_value": _to_decimal(row.feature_value),
            "is_filled": bool(getattr(row, "is_filled", False)),
            "batch_id": batch_id,
        }
        for row in prepared.itertuples(index=False)
    ]


class FeaturePipeline:
    def __init__(self) -> None:
        self.last_batch_id: str | None = None

    def run(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
        as_of: date | datetime,
        *,
        allow_missing_intraday: bool = False,
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
        price_tickers = tuple(dict.fromkeys([*normalized_tickers, *SECTOR_ROTATION_ETF_TICKERS]))
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

        alternative = compute_alternative_features_batch(
            prices_df=stock_prices,
            output_start=start,
            output_end=end,
            as_of=as_of_ts,
        )
        minute_history = load_intraday_minute_history(
            tickers=normalized_tickers,
            start_trade_date=start - timedelta(days=90),
            end_trade_date=end,
            as_of=as_of_ts,
            allow_missing=allow_missing_intraday,
        )
        intraday = compute_intraday_features(
            minute_df=minute_history,
            daily_prices_df=stock_prices,
        )
        if not intraday.empty:
            intraday = intraday.loc[
                (pd.to_datetime(intraday["trade_date"]).dt.date >= start)
                & (pd.to_datetime(intraday["trade_date"]).dt.date <= end)
            ].copy()
        macro = self._compute_broadcast_macro_features(output_prices, as_of_ts)
        base_features = pd.concat([technical, fundamentals, alternative, intraday, macro], ignore_index=True)
        sector_rotation = compute_sector_rotation_features(
            base_features_df=base_features,
            prices_df=prices,
        )
        base_features = pd.concat([base_features, sector_rotation], ignore_index=True)
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
        prepared = prepare_feature_export_frame(features_df)

        session_factory = get_session_factory()
        rows_saved = 0
        with session_factory() as session:
            try:
                for start in range(0, len(prepared), batch_size):
                    batch = prepared.iloc[start : start + batch_size]
                    # calc_date stores the market date for the feature row. The
                    # actual availability time remains governed by PIT input
                    # knowledge_time and the batch's as_of cutoff used in run().
                    records = feature_store_records_from_frame(batch, batch_id=batch_id)
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
        parquet_frame = prepare_feature_export_frame(features_df)
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


def compute_alternative_features_batch(
    *,
    prices_df: pd.DataFrame,
    output_start: date,
    output_end: date,
    as_of: date | datetime,
) -> pd.DataFrame:
    """Compute PIT-safe alternative features for a batch of tickers.

    This preloads sparse alternative-data tables once per batch, then walks each
    ticker's daily calendar in memory. It avoids the one-query-per-row pattern
    that would otherwise make screening prohibitively slow.
    """
    if prices_df.empty:
        return _empty_feature_frame()

    prepared_prices = prices_df.copy()
    prepared_prices["ticker"] = prepared_prices["ticker"].astype(str).str.upper()
    prepared_prices["trade_date"] = pd.to_datetime(prepared_prices["trade_date"]).dt.date
    prepared_prices.sort_values(["ticker", "trade_date"], inplace=True)
    prepared_prices = _attach_pit_shares_outstanding(prepared_prices)

    normalized_tickers = tuple(prepared_prices["ticker"].dropna().astype(str).str.upper().unique().tolist())
    if not normalized_tickers:
        return _empty_feature_frame()

    max_trade_date = min(output_end, _coerce_as_of_datetime(as_of).date())
    sector_map = load_sector_map_pit(max_trade_date)
    histories = load_alternative_histories_batch(
        tickers=normalized_tickers,
        start_trade_date=output_start,
        max_trade_date=max_trade_date,
    )

    frames: list[pd.DataFrame] = []
    for ticker in normalized_tickers:
        ticker_prices = prepared_prices.loc[prepared_prices["ticker"] == ticker].copy()
        if ticker_prices.empty:
            continue
        frames.append(
            _compute_alternative_features_for_ticker_history(
                ticker=ticker,
                ticker_prices=ticker_prices,
                output_start=output_start,
                output_end=output_end,
                earnings_history=histories["earnings"].get(ticker, _empty_earnings_history()),
                analyst_history=histories["analyst"].get(ticker, _empty_analyst_history()),
                short_interest_history=histories["short_interest"].get(ticker, _empty_short_interest_history()),
                insider_history=histories["insider"].get(ticker, _empty_insider_history()),
                sec_filing_history=histories["sec_filings"].get(ticker, _empty_sec_filing_history()),
            ),
        )

    if not frames:
        return _empty_feature_frame()
    wide = pd.concat(frames, ignore_index=True)
    if wide.empty:
        return _empty_feature_frame()
    wide = _apply_short_interest_sector_relative(wide, sector_map=sector_map)
    extra_cols = set(wide.columns) - {"ticker", "trade_date"} - set(ALTERNATIVE_FEATURE_NAMES)
    if extra_cols:
        wide = wide.drop(columns=list(extra_cols))
    melted = wide.melt(
        id_vars=["ticker", "trade_date"],
        value_vars=[c for c in ALTERNATIVE_FEATURE_NAMES if c in wide.columns],
        var_name="feature_name",
        value_name="feature_value",
    )
    return melted.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def load_alternative_histories_batch(
    *,
    tickers: Sequence[str],
    start_trade_date: date,
    max_trade_date: date,
) -> dict[str, dict[str, pd.DataFrame]]:
    normalized_tickers = tuple(dict.fromkeys(str(ticker).upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        return {
            "earnings": {},
            "analyst": {},
            "short_interest": {},
            "insider": {},
            "sec_filings": {},
        }

    session_factory = get_session_factory()
    cutoff = datetime.combine(max_trade_date, time.max, tzinfo=timezone.utc)
    insider_start = start_trade_date - timedelta(days=730)
    filing_start = start_trade_date - timedelta(days=365)

    def _load_frame(statement: sa.sql.Select, empty_frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
        try:
            with session_factory() as session:
                rows = session.execute(statement).mappings().all()
        except Exception as exc:
            logger.warning("alternative feature preload skipped {} because query failed: {}", source_name, exc)
            return empty_frame
        frame = pd.DataFrame(rows)
        if frame.empty:
            return empty_frame
        return frame

    earnings_stmt = (
        sa.select(
            EarningsEstimate.ticker,
            EarningsEstimate.fiscal_date,
            EarningsEstimate.eps_estimated,
            EarningsEstimate.eps_actual,
            EarningsEstimate.knowledge_time,
        )
        .where(
            EarningsEstimate.ticker.in_(normalized_tickers),
            EarningsEstimate.knowledge_time <= cutoff,
            EarningsEstimate.fiscal_date <= max_trade_date,
        )
        .order_by(EarningsEstimate.ticker, EarningsEstimate.knowledge_time, EarningsEstimate.fiscal_date)
    )
    analyst_stmt = (
        sa.select(
            AnalystEstimate.ticker,
            AnalystEstimate.fiscal_date,
            AnalystEstimate.period,
            AnalystEstimate.eps_avg,
            AnalystEstimate.revenue_avg,
            AnalystEstimate.num_analysts_eps,
            AnalystEstimate.knowledge_time,
        )
        .where(
            AnalystEstimate.ticker.in_(normalized_tickers),
            AnalystEstimate.period == "quarter",
            AnalystEstimate.knowledge_time <= cutoff,
            AnalystEstimate.fiscal_date <= max_trade_date,
        )
        .order_by(AnalystEstimate.ticker, AnalystEstimate.knowledge_time, AnalystEstimate.fiscal_date)
    )
    short_interest_stmt = (
        sa.select(
            ShortInterest.ticker,
            ShortInterest.settlement_date,
            ShortInterest.short_interest,
            ShortInterest.avg_daily_volume,
            ShortInterest.days_to_cover,
            ShortInterest.knowledge_time,
        )
        .where(
            ShortInterest.ticker.in_(normalized_tickers),
            ShortInterest.knowledge_time <= cutoff,
            ShortInterest.settlement_date <= max_trade_date,
        )
        .order_by(ShortInterest.ticker, ShortInterest.knowledge_time, ShortInterest.settlement_date)
    )
    insider_stmt = (
        sa.select(
            InsiderTrade.ticker,
            InsiderTrade.filing_date,
            InsiderTrade.reporting_cik,
            InsiderTrade.transaction_type,
            InsiderTrade.securities_transacted,
            InsiderTrade.price,
            InsiderTrade.acquisition_or_disposition,
            InsiderTrade.type_of_owner,
            InsiderTrade.knowledge_time,
        )
        .where(
            InsiderTrade.ticker.in_(normalized_tickers),
            InsiderTrade.knowledge_time <= cutoff,
            InsiderTrade.filing_date >= insider_start,
            InsiderTrade.filing_date <= max_trade_date,
        )
        .order_by(InsiderTrade.ticker, InsiderTrade.knowledge_time, InsiderTrade.filing_date, InsiderTrade.id)
    )
    sec_filings_stmt = (
        sa.select(
            SecFiling.ticker,
            SecFiling.filing_date,
            SecFiling.accepted_date,
            SecFiling.form_type,
            SecFiling.knowledge_time,
        )
        .where(
            SecFiling.ticker.in_(normalized_tickers),
            SecFiling.knowledge_time <= cutoff,
            SecFiling.accepted_date <= cutoff,
            sa.or_(SecFiling.filing_date.is_(None), SecFiling.filing_date >= filing_start),
        )
        .order_by(SecFiling.ticker, SecFiling.knowledge_time, SecFiling.accepted_date, SecFiling.id)
    )

    earnings = _load_frame(earnings_stmt, _empty_earnings_history(), "earnings_estimates")
    analyst = _load_frame(analyst_stmt, _empty_analyst_history(), "analyst_estimates")
    short_interest = _load_frame(short_interest_stmt, _empty_short_interest_history(), "short_interest")
    insider = _load_frame(insider_stmt, _empty_insider_history(), "insider_trades")
    sec_filings = _load_frame(sec_filings_stmt, _empty_sec_filing_history(), "sec_filings")

    histories = {
        "earnings": _split_history_by_ticker(_prepare_earnings_history(earnings)),
        "analyst": _split_history_by_ticker(_prepare_analyst_history(analyst)),
        "short_interest": _split_history_by_ticker(_prepare_short_interest_history(short_interest)),
        "insider": _split_history_by_ticker(_prepare_insider_history(insider)),
        "sec_filings": _split_history_by_ticker(_prepare_sec_filing_history(sec_filings)),
    }
    logger.info(
        "preloaded alternative histories for {} tickers: earnings={}, analyst={}, short_interest={}, insider={}, sec_filings={}",
        len(normalized_tickers),
        sum(len(frame) for frame in histories["earnings"].values()),
        sum(len(frame) for frame in histories["analyst"].values()),
        sum(len(frame) for frame in histories["short_interest"].values()),
        sum(len(frame) for frame in histories["insider"].values()),
        sum(len(frame) for frame in histories["sec_filings"].values()),
    )
    return histories


def load_intraday_minute_history(
    *,
    tickers: Sequence[str],
    start_trade_date: date,
    end_trade_date: date,
    as_of: date | datetime,
    allow_missing: bool = False,
) -> pd.DataFrame:
    normalized_tickers = tuple(dict.fromkeys(str(ticker).upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        return pd.DataFrame(
            columns=["ticker", "trade_date", "minute_ts", "open", "high", "low", "close", "volume", "vwap", "transactions"],
        )

    cutoff = _coerce_as_of_datetime(as_of)
    statement = (
        sa.select(
            StockMinuteAggs.ticker,
            StockMinuteAggs.trade_date,
            StockMinuteAggs.minute_ts,
            StockMinuteAggs.open,
            StockMinuteAggs.high,
            StockMinuteAggs.low,
            StockMinuteAggs.close,
            StockMinuteAggs.volume,
            StockMinuteAggs.vwap,
            StockMinuteAggs.transactions,
        )
        .where(
            StockMinuteAggs.ticker.in_(normalized_tickers),
            StockMinuteAggs.trade_date >= start_trade_date,
            StockMinuteAggs.trade_date <= end_trade_date,
            StockMinuteAggs.knowledge_time <= cutoff,
        )
        .order_by(StockMinuteAggs.ticker, StockMinuteAggs.minute_ts)
    )
    session_factory = get_session_factory()
    try:
        with session_factory() as session:
            rows = session.execute(statement).mappings().all()
    except Exception as exc:
        if allow_missing:
            logger.error("minute_history_missing (allow_missing=True): {}", exc)
            return pd.DataFrame(
                columns=["ticker", "trade_date", "minute_ts", "open", "high", "low", "close", "volume", "vwap", "transactions"],
            )
        raise IntradayHistoryError(
            f"minute history unavailable for {','.join(normalized_tickers)} {start_trade_date}~{end_trade_date}: {exc}",
        ) from exc

    if not rows:
        if allow_missing:
            logger.error(
                "minute_history empty (allow_missing=True): tickers={} range={}~{}",
                normalized_tickers,
                start_trade_date,
                end_trade_date,
            )
            return pd.DataFrame(
                columns=["ticker", "trade_date", "minute_ts", "open", "high", "low", "close", "volume", "vwap", "transactions"],
            )
        raise IntradayHistoryError(
            "minute history empty for "
            f"{','.join(normalized_tickers)} {start_trade_date}~{end_trade_date} "
            "(query succeeded but returned 0 rows; check minute_backfill_state coverage)",
        )
    return pd.DataFrame(rows)


def _compute_alternative_features_for_ticker_history(
    *,
    ticker: str,
    ticker_prices: pd.DataFrame,
    output_start: date,
    output_end: date,
    earnings_history: pd.DataFrame,
    analyst_history: pd.DataFrame,
    short_interest_history: pd.DataFrame,
    insider_history: pd.DataFrame,
    sec_filing_history: pd.DataFrame,
) -> pd.DataFrame:
    prices = ticker_prices.copy()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    prices["pit_shares_outstanding"] = pd.to_numeric(prices.get("pit_shares_outstanding"), errors="coerce")
    prices.sort_values("trade_date", inplace=True)

    output_prices = prices.loc[
        (prices["trade_date"] >= output_start) & (prices["trade_date"] <= output_end)
    ].copy()
    if output_prices.empty:
        return _empty_feature_frame()

    prices["overnight_gap"] = (prices["open"] - prices["close"].shift(1)) / prices["close"].shift(1)
    prices["volume_surge"] = prices["volume"] / prices["volume"].shift(1).rolling(20, min_periods=20).mean()
    prices["ret_5d"] = prices["close"].pct_change(5)
    prices["ret_20d"] = prices["close"].pct_change(20)
    prices["market_cap"] = prices["close"] * prices["pit_shares_outstanding"]
    daily_map = (
        prices.loc[
            (prices["trade_date"] >= output_start) & (prices["trade_date"] <= output_end),
            ["trade_date", "overnight_gap", "volume_surge", "ret_5d", "ret_20d", "market_cap"],
        ]
        .set_index("trade_date")
        .to_dict("index")
    )

    earnings_records = earnings_history.to_dict("records")
    analyst_records = analyst_history.to_dict("records")
    short_records = short_interest_history.to_dict("records")
    insider_records = insider_history.to_dict("records")
    sec_records = sec_filing_history.to_dict("records")

    earnings_pointer = 0
    analyst_pointer = 0
    short_pointer = 0
    insider_pointer = 0
    sec_pointer = 0

    active_earnings: dict[date, dict[str, object]] = {}
    active_analyst: dict[date, dict[str, object]] = {}
    latest_short: dict[str, object] | None = None
    previous_short: dict[str, object] | None = None
    active_insider_window: list[dict[str, object]] = []
    active_sec_history: list[dict[str, object]] = []

    snapshots: list[dict[str, object]] = []
    for trade_date in output_prices["trade_date"].tolist():
        cutoff = datetime.combine(trade_date, time.max, tzinfo=timezone.utc)

        while earnings_pointer < len(earnings_records) and earnings_records[earnings_pointer]["knowledge_time"] <= cutoff:
            record = earnings_records[earnings_pointer]
            active_earnings[record["fiscal_date"]] = record
            earnings_pointer += 1

        while analyst_pointer < len(analyst_records) and analyst_records[analyst_pointer]["knowledge_time"] <= cutoff:
            record = analyst_records[analyst_pointer]
            active_analyst[record["fiscal_date"]] = record
            analyst_pointer += 1

        while short_pointer < len(short_records) and short_records[short_pointer]["knowledge_time"] <= cutoff:
            previous_short = latest_short
            latest_short = short_records[short_pointer]
            short_pointer += 1

        while insider_pointer < len(insider_records) and insider_records[insider_pointer]["knowledge_time"] <= cutoff:
            active_insider_window.append(insider_records[insider_pointer])
            insider_pointer += 1

        while sec_pointer < len(sec_records) and sec_records[sec_pointer]["knowledge_time"] <= cutoff:
            active_sec_history.append(sec_records[sec_pointer])
            sec_pointer += 1

        lookback_start = trade_date - timedelta(days=730)
        while active_insider_window and active_insider_window[0]["filing_date"] < lookback_start:
            active_insider_window.pop(0)

        daily_values = daily_map.get(trade_date, {})
        snapshot = {
            "ticker": ticker,
            "trade_date": trade_date,
            **_summarize_earnings_features(
                active_earnings,
                trade_date=trade_date,
                ret_5d=daily_values.get("ret_5d", np.nan),
            ),
            **_summarize_analyst_features(active_analyst),
            **_summarize_short_interest_features(
                latest_short,
                previous_short,
                short_history=short_records[:short_pointer],
            ),
            **_summarize_insider_features(
                active_insider_window=active_insider_window,
                market_cap=_coerce_float(daily_values.get("market_cap")),
                trade_date=trade_date,
            ),
            **_summarize_sec_filing_features(active_sec_history, trade_date=trade_date),
            "overnight_gap": daily_values.get("overnight_gap", np.nan),
            "volume_surge": daily_values.get("volume_surge", np.nan),
            "_ret_5d": daily_values.get("ret_5d", np.nan),
            "_ret_20d": daily_values.get("ret_20d", np.nan),
        }
        snapshots.append(snapshot)

    return pd.DataFrame(snapshots)


def _summarize_earnings_features(
    active_earnings: dict[date, dict[str, object]],
    *,
    trade_date: date,
    ret_5d: float | object = np.nan,
) -> dict[str, float]:
    result = {
        "earnings_surprise_latest": np.nan,
        "earnings_surprise_avg_4q": np.nan,
        "earnings_beat_streak": np.nan,
        "earnings_surprise_recency": np.nan,
        "earnings_beat_recency": np.nan,
        "earnings_surprise_recency_20d": np.nan,
        "earnings_beat_recency_30d": np.nan,
        "surprise_flip_qoq": np.nan,
        "surprise_vs_history": np.nan,
        "pead_setup": np.nan,
    }
    valid: list[tuple[date, float]] = []
    for record in active_earnings.values():
        eps_actual = record.get("eps_actual")
        eps_estimated = record.get("eps_estimated")
        if eps_actual is None or eps_estimated is None:
            continue
        eps_actual_float = float(eps_actual)
        eps_estimated_float = float(eps_estimated)
        if eps_estimated_float == 0:
            continue
        valid.append(
            (
                record["fiscal_date"],
                (eps_actual_float - eps_estimated_float) / abs(eps_estimated_float),
            ),
        )
    valid.sort(key=lambda item: item[0], reverse=True)
    surprises = [surprise for _, surprise in valid[:4]]
    if not surprises:
        return result

    result["earnings_surprise_latest"] = surprises[0]
    result["earnings_surprise_avg_4q"] = float(np.mean(surprises))
    streak = 0
    for surprise in surprises:
        if surprise > 0:
            streak += 1
        else:
            break
    result["earnings_beat_streak"] = float(streak)
    latest_fiscal_date = valid[0][0]
    if latest_fiscal_date is not None:
        days_since = max((trade_date - latest_fiscal_date).days, 0)
        recency_decay_30d = float(np.exp(-days_since / 30.0))
        recency_decay_20d = float(np.exp(-days_since / 20.0))
        result["earnings_surprise_recency"] = result["earnings_surprise_latest"] * recency_decay_30d
        result["earnings_beat_recency"] = result["earnings_beat_streak"] * recency_decay_30d
        result["earnings_surprise_recency_20d"] = result["earnings_surprise_latest"] * recency_decay_20d
        result["earnings_beat_recency_30d"] = result["earnings_beat_streak"] * recency_decay_30d
        momentum = 0.0 if pd.isna(ret_5d) else max(0.0, float(ret_5d))
        result["pead_setup"] = result["earnings_surprise_recency_20d"] * momentum
    if len(surprises) >= 2:
        result["surprise_flip_qoq"] = surprises[0] - surprises[1]
    result["surprise_vs_history"] = result["earnings_surprise_latest"] - result["earnings_surprise_avg_4q"]
    return result


def _summarize_analyst_features(active_analyst: dict[date, dict[str, object]]) -> dict[str, float]:
    result = {
        "eps_revision_direction": 0.0,
        "revenue_revision_pct": np.nan,
        "analyst_coverage": np.nan,
    }
    if not active_analyst:
        return result

    latest_records = sorted(active_analyst.values(), key=lambda row: row["fiscal_date"], reverse=True)[:2]
    latest = latest_records[0]
    if latest.get("num_analysts_eps") is not None:
        result["analyst_coverage"] = float(latest["num_analysts_eps"])
    if len(latest_records) < 2:
        return result

    prior = latest_records[1]
    if latest.get("eps_avg") is not None and prior.get("eps_avg") is not None and float(prior["eps_avg"]) != 0:
        diff = float(latest["eps_avg"]) - float(prior["eps_avg"])
        result["eps_revision_direction"] = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
    if latest.get("revenue_avg") is not None and prior.get("revenue_avg") is not None and float(prior["revenue_avg"]) != 0:
        result["revenue_revision_pct"] = (float(latest["revenue_avg"]) - float(prior["revenue_avg"])) / abs(
            float(prior["revenue_avg"]),
        )
    return result


def _summarize_short_interest_features(
    latest_short: dict[str, object] | None,
    previous_short: dict[str, object] | None,
    *,
    short_history: list[dict[str, object]],
) -> dict[str, float]:
    result = {
        "short_interest_ratio": np.nan,
        "short_interest_change": np.nan,
        "short_interest_sector_rel": np.nan,
        "short_interest_change_20d": np.nan,
        "short_interest_abnormal_1y": np.nan,
        "short_squeeze_setup": np.nan,
        "crowding_unwind_risk": np.nan,
    }
    if latest_short is None:
        return result

    days_to_cover = latest_short.get("days_to_cover")
    short_interest = latest_short.get("short_interest")
    avg_daily_volume = latest_short.get("avg_daily_volume")
    if days_to_cover is not None:
        result["short_interest_ratio"] = float(days_to_cover)
    elif short_interest is not None and avg_daily_volume is not None and float(avg_daily_volume) > 0:
        result["short_interest_ratio"] = float(short_interest) / float(avg_daily_volume)

    if (
        previous_short is not None
        and latest_short.get("short_interest") is not None
        and previous_short.get("short_interest") is not None
        and float(previous_short["short_interest"]) > 0
    ):
        result["short_interest_change"] = (
            float(latest_short["short_interest"]) - float(previous_short["short_interest"])
        ) / float(previous_short["short_interest"])
        result["short_interest_change_20d"] = result["short_interest_change"]
    history_ratios: list[float] = []
    for record in short_history:
        if record.get("days_to_cover") is not None:
            history_ratios.append(float(record["days_to_cover"]))
        elif (
            record.get("short_interest") is not None
            and record.get("avg_daily_volume") is not None
            and float(record["avg_daily_volume"]) > 0
        ):
            history_ratios.append(float(record["short_interest"]) / float(record["avg_daily_volume"]))
    if history_ratios:
        median = float(np.nanmedian(history_ratios))
        std = float(np.nanstd(history_ratios))
        if std > 0 and pd.notna(result["short_interest_ratio"]):
            result["short_interest_abnormal_1y"] = (float(result["short_interest_ratio"]) - median) / std
    return result


def _summarize_insider_features(
    *,
    active_insider_window: list[dict[str, object]],
    market_cap: float | None,
    trade_date: date,
) -> dict[str, float]:
    buy_count = 0
    sell_count = 0
    buy_value = 0.0
    buy_insider_counts: Counter[str] = Counter()
    buy_intensity_20d = 0.0
    buy_intensity_60d = 0.0
    sell_intensity_60d = 0.0
    cluster_weight_30d = 0.0
    ceo_cfo_buy_30d = 0.0
    director_officer_sell_30d = 0.0
    buy_intensity_90d = 0.0
    annual_buy_bins: dict[int, float] = {}

    for record in active_insider_window:
        filing_date = record.get("filing_date")
        if filing_date is None:
            continue
        filing_day = pd.to_datetime(filing_date).date()
        age = max((trade_date - filing_day).days, 0)
        value = _transaction_value(record)
        owner_type = record.get("type_of_owner")
        role_weight = _role_weight(owner_type)
        if _is_purchase_record(record):
            if age <= 90:
                buy_count += 1
                buy_value += value
                cik = str(record.get("reporting_cik") or "").strip()
                if cik:
                    buy_insider_counts[cik] += 1
            if age <= 20:
                buy_intensity_20d += value * role_weight * float(np.exp(-age / 20.0))
            if age <= 60:
                buy_intensity_60d += value * role_weight * float(np.exp(-age / 60.0))
            if age <= 30:
                cluster_weight_30d += float(np.exp(-age / 30.0))
                if _owner_is_ceo_cfo(owner_type):
                    ceo_cfo_buy_30d += value * role_weight * float(np.exp(-age / 30.0))
            if age <= 90:
                buy_intensity_90d += value * role_weight
            annual_buy_bins[filing_day.year] = annual_buy_bins.get(filing_day.year, 0.0) + value * role_weight
        elif _is_sale_record(record):
            if age <= 90:
                sell_count += 1
            if age <= 60:
                sell_intensity_60d += value * role_weight * float(np.exp(-age / 60.0))
            if age <= 30 and _owner_is_director_or_officer(owner_type):
                director_officer_sell_30d += value * role_weight * float(np.exp(-age / 30.0))

    total = buy_count + sell_count
    result = {
        "insider_net_buy_ratio": np.nan if total == 0 else (buy_count - sell_count) / total,
        "insider_buy_value": buy_value if buy_count > 0 else 0.0,
        "insider_cluster_buy": 1.0 if len(buy_insider_counts) >= 3 else 0.0,
        "insider_buy_intensity_20d": np.nan,
        "insider_net_intensity_60d": np.nan,
        "insider_cluster_buy_30d_w": cluster_weight_30d,
        "insider_abnormal_buy_90d": np.nan,
        "insider_role_skew_30d": ceo_cfo_buy_30d - director_officer_sell_30d,
    }
    if market_cap is not None and market_cap > 0:
        result["insider_buy_intensity_20d"] = buy_intensity_20d / market_cap
        result["insider_net_intensity_60d"] = (buy_intensity_60d - sell_intensity_60d) / market_cap
        if annual_buy_bins:
            annual_baseline = float(np.nanmedian([value / market_cap for value in annual_buy_bins.values()]))
            if annual_baseline > 0:
                result["insider_abnormal_buy_90d"] = (buy_intensity_90d / market_cap) / annual_baseline
    return result


def _summarize_sec_filing_features(active_sec_history: list[dict[str, object]], *, trade_date: date) -> dict[str, float]:
    result = {
        "days_since_last_8k": np.nan,
        "days_since_last_10q": np.nan,
        "days_since_last_10k": np.nan,
        "recent_8k_count_5d": 0.0,
        "recent_8k_count_20d": 0.0,
        "recent_8k_count_60d": 0.0,
        "has_recent_8k_5d": 0.0,
        "filing_burst_20d": 0.0,
    }
    if not active_sec_history:
        return result

    latest_8k: date | None = None
    latest_10q: date | None = None
    latest_10k: date | None = None
    for record in reversed(active_sec_history):
        filing_day = pd.to_datetime(record.get("filing_date") or record.get("accepted_date")).date()
        age = max((trade_date - filing_day).days, 0)
        form_type = str(record.get("form_type") or "").upper()
        if form_type.startswith("8-K"):
            result["recent_8k_count_60d"] += float(age <= 60)
            result["recent_8k_count_20d"] += float(age <= 20)
            recent_5d = float(age <= 5)
            result["recent_8k_count_5d"] += recent_5d
            result["has_recent_8k_5d"] = max(result["has_recent_8k_5d"], recent_5d)
            if latest_8k is None:
                latest_8k = filing_day
        if form_type.startswith("10-Q") and latest_10q is None:
            latest_10q = filing_day
        if form_type.startswith("10-K") and latest_10k is None:
            latest_10k = filing_day
        if age <= 20:
            result["filing_burst_20d"] += 1.0

    if latest_8k is not None:
        result["days_since_last_8k"] = float(max((trade_date - latest_8k).days, 0))
    if latest_10q is not None:
        result["days_since_last_10q"] = float(max((trade_date - latest_10q).days, 0))
    if latest_10k is not None:
        result["days_since_last_10k"] = float(max((trade_date - latest_10k).days, 0))
    return result


def _is_purchase_record(record: dict[str, object]) -> bool:
    transaction_type = str(record.get("transaction_type") or "")
    acquisition_or_disposition = str(record.get("acquisition_or_disposition") or "")
    return "P-Purchase" in transaction_type or (
        acquisition_or_disposition == "A" and "Award" not in transaction_type
    )


def _is_sale_record(record: dict[str, object]) -> bool:
    transaction_type = str(record.get("transaction_type") or "")
    acquisition_or_disposition = str(record.get("acquisition_or_disposition") or "")
    return "S-Sale" in transaction_type or acquisition_or_disposition == "D"


def _transaction_value(record: dict[str, object]) -> float:
    shares = record.get("securities_transacted")
    price = record.get("price")
    shares_float = float(shares) if shares is not None else 0.0
    price_float = float(price) if price is not None else 0.0
    return shares_float * price_float


def _role_weight(type_of_owner: object) -> float:
    normalized = str(type_of_owner or "").strip().lower()
    if any(token in normalized for token in ("chief executive", "ceo", "chief financial", "cfo")):
        return 1.5
    if "director" in normalized and "officer" not in normalized:
        return 0.75
    return 1.0


def _owner_is_ceo_cfo(type_of_owner: object) -> bool:
    normalized = str(type_of_owner or "").strip().lower()
    return any(token in normalized for token in ("chief executive", "ceo", "chief financial", "cfo"))


def _owner_is_director_or_officer(type_of_owner: object) -> bool:
    normalized = str(type_of_owner or "").strip().lower()
    return ("director" in normalized) or ("officer" in normalized)


def _coerce_float(value: object) -> float | None:
    try:
        converted = float(value) if value is not None else np.nan
    except (TypeError, ValueError):
        return None
    if np.isnan(converted) or np.isinf(converted):
        return None
    return converted


def _apply_short_interest_sector_relative(wide: pd.DataFrame, *, sector_map: dict[str, str]) -> pd.DataFrame:
    if wide.empty or "short_interest_ratio" not in wide.columns:
        return wide

    enriched = wide.copy()
    enriched["sector"] = enriched["ticker"].astype(str).str.upper().map(sector_map)
    stats = (
        enriched.groupby(["trade_date", "sector"], dropna=False)["short_interest_ratio"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "_sector_si_mean", "std": "_sector_si_std"})
    )
    enriched = enriched.merge(stats, on=["trade_date", "sector"], how="left")
    denominator = enriched["_sector_si_std"].replace(0, np.nan)
    enriched["short_interest_sector_rel"] = (
        enriched["short_interest_ratio"] - enriched["_sector_si_mean"]
    ) / denominator
    ret_5d = pd.to_numeric(enriched.get("_ret_5d"), errors="coerce")
    ret_20d = pd.to_numeric(enriched.get("_ret_20d"), errors="coerce")
    volume_surge = pd.to_numeric(enriched.get("volume_surge"), errors="coerce")
    sector_rel = pd.to_numeric(enriched.get("short_interest_sector_rel"), errors="coerce")
    enriched["short_squeeze_setup"] = sector_rel * np.maximum(ret_5d.fillna(0.0), 0.0) * volume_surge.fillna(1.0)
    enriched["crowding_unwind_risk"] = sector_rel * np.minimum(ret_20d.fillna(0.0), 0.0)
    return enriched.drop(columns=["sector", "_sector_si_mean", "_sector_si_std", "_ret_5d", "_ret_20d"], errors="ignore")


def _prepare_earnings_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_earnings_history()
    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["fiscal_date"] = pd.to_datetime(prepared["fiscal_date"]).dt.date
    prepared["eps_estimated"] = pd.to_numeric(prepared["eps_estimated"], errors="coerce")
    prepared["eps_actual"] = pd.to_numeric(prepared["eps_actual"], errors="coerce")
    prepared["knowledge_time"] = pd.to_datetime(prepared["knowledge_time"], utc=True)
    prepared.sort_values(["ticker", "knowledge_time", "fiscal_date"], inplace=True)
    return prepared.reset_index(drop=True)


def _prepare_analyst_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_analyst_history()
    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["fiscal_date"] = pd.to_datetime(prepared["fiscal_date"]).dt.date
    prepared["eps_avg"] = pd.to_numeric(prepared["eps_avg"], errors="coerce")
    prepared["revenue_avg"] = pd.to_numeric(prepared["revenue_avg"], errors="coerce")
    prepared["num_analysts_eps"] = pd.to_numeric(prepared["num_analysts_eps"], errors="coerce")
    prepared["knowledge_time"] = pd.to_datetime(prepared["knowledge_time"], utc=True)
    prepared.sort_values(["ticker", "knowledge_time", "fiscal_date"], inplace=True)
    return prepared.reset_index(drop=True)


def _prepare_short_interest_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_short_interest_history()
    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["settlement_date"] = pd.to_datetime(prepared["settlement_date"]).dt.date
    prepared["short_interest"] = pd.to_numeric(prepared["short_interest"], errors="coerce")
    prepared["avg_daily_volume"] = pd.to_numeric(prepared["avg_daily_volume"], errors="coerce")
    prepared["days_to_cover"] = pd.to_numeric(prepared["days_to_cover"], errors="coerce")
    prepared["knowledge_time"] = pd.to_datetime(prepared["knowledge_time"], utc=True)
    prepared.sort_values(["ticker", "knowledge_time", "settlement_date"], inplace=True)
    return prepared.reset_index(drop=True)


def _prepare_insider_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_insider_history()
    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["filing_date"] = pd.to_datetime(prepared["filing_date"]).dt.date
    prepared["securities_transacted"] = pd.to_numeric(prepared["securities_transacted"], errors="coerce")
    prepared["price"] = pd.to_numeric(prepared["price"], errors="coerce")
    prepared["knowledge_time"] = pd.to_datetime(prepared["knowledge_time"], utc=True)
    prepared.sort_values(["ticker", "knowledge_time", "filing_date"], inplace=True)
    return prepared.reset_index(drop=True)


def _prepare_sec_filing_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_sec_filing_history()
    prepared = frame.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["filing_date"] = pd.to_datetime(prepared["filing_date"], errors="coerce").dt.date
    prepared["accepted_date"] = pd.to_datetime(prepared["accepted_date"], utc=True, errors="coerce")
    prepared["knowledge_time"] = pd.to_datetime(prepared["knowledge_time"], utc=True, errors="coerce")
    prepared.sort_values(["ticker", "knowledge_time", "accepted_date"], inplace=True)
    return prepared.reset_index(drop=True)


def _split_history_by_ticker(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty or "ticker" not in frame.columns:
        return {}
    return {
        str(ticker).upper(): group.reset_index(drop=True)
        for ticker, group in frame.groupby("ticker", sort=False)
    }


def _empty_earnings_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "fiscal_date", "eps_estimated", "eps_actual", "knowledge_time"],
    )


def _empty_analyst_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "fiscal_date", "period", "eps_avg", "revenue_avg", "num_analysts_eps", "knowledge_time"],
    )


def _empty_short_interest_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "settlement_date", "short_interest", "avg_daily_volume", "days_to_cover", "knowledge_time"],
    )


def _empty_insider_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "filing_date",
            "reporting_cik",
            "transaction_type",
            "securities_transacted",
            "price",
            "acquisition_or_disposition",
            "type_of_owner",
            "knowledge_time",
        ],
    )


def _empty_sec_filing_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ticker", "filing_date", "accepted_date", "form_type", "knowledge_time"],
    )


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
        "stock_beta_252",
        "above_20dma",
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
        "vix",
        "credit_spread",
        "credit_spread_change",
        "yield_spread_10y2y",
        "sp500_breadth",
        "revenue_growth_yoy",
    ]
    wide = wide.reindex(columns=sorted(set(list(wide.columns) + required_base_features)))

    composite = pd.DataFrame(index=wide.index)
    composite["ret_vol_interaction_20d"] = wide.get("ret_20d") * wide.get("volume_ratio_20d")
    composite["ret_vol_interaction_60d"] = wide.get("ret_60d") * wide.get("volume_ratio_20d")
    composite["mom_vol_adj_20d"] = _safe_series_divide(wide.get("ret_20d"), wide.get("vol_20d"))
    composite["mom_vol_adj_60d"] = _safe_series_divide(wide.get("ret_60d"), wide.get("vol_60d"))
    breadth_pct_above_20dma = _cross_sectional_mean(wide.get("above_20dma"))
    return_dispersion_20d = _cross_sectional_std(wide.get("ret_20d"))
    median_ret_20d = _cross_sectional_median(wide.get("ret_20d"))
    relative_ret_20d = wide.get("ret_20d") - median_ret_20d
    # Convert market-context series into stock-level selectors. Pure date-level
    # broadcasts have no cross-sectional variance and therefore undefined IC.
    composite["breadth_pct_above_20dma"] = wide.get("above_20dma") - breadth_pct_above_20dma
    composite["return_dispersion_20d"] = _safe_series_divide(relative_ret_20d, return_dispersion_20d)
    composite["narrow_leadership_score"] = (
        (wide.get("market_ret_20d") - median_ret_20d)
        * _safe_series_divide(relative_ret_20d, return_dispersion_20d)
    )
    vix_z = _rolling_date_zscore(wide.get("vix"), window=252)
    composite["high_vix_x_beta"] = vix_z.clip(lower=0) * wide.get("stock_beta_252")
    composite["credit_widening_x_leverage"] = wide.get("credit_spread_change").clip(lower=0) * wide.get("debt_to_equity")
    composite["curve_inverted_x_growth"] = wide.get("yield_spread_10y2y").clip(upper=0) * (-wide.get("revenue_growth_yoy"))
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


def _cross_sectional_mean(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return series.groupby(level="trade_date").transform("mean")


def _cross_sectional_median(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return series.groupby(level="trade_date").transform("median")


def _cross_sectional_std(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return series.groupby(level="trade_date").transform(lambda values: values.std(ddof=0))


def _rolling_date_zscore(series: pd.Series | None, *, window: int) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    per_date = series.groupby(level="trade_date").first().sort_index().astype(float)
    min_periods = max(20, min(window, 60))
    rolling_median = per_date.rolling(window, min_periods=min_periods).median()
    rolling_std = per_date.rolling(window, min_periods=min_periods).std(ddof=0)
    zscore = (per_date - rolling_median) / rolling_std.replace(0, np.nan)
    date_index = series.index.get_level_values("trade_date")
    return pd.Series(date_index.map(zscore), index=series.index, dtype=float)


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
