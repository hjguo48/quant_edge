from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import gc
import json
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import sys
from typing import Any
import uuid

from loguru import logger
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.pit import get_prices_pit
from src.data.db.models import FundamentalsPIT
from src.data.db.session import get_engine
from src.data.db.session import get_session_factory
import src.features.fundamental as fundamental_module
import src.features.pipeline as pipeline_module
import src.features.preprocessing as preprocessing_module
from src.features.fundamental import FUNDAMENTAL_FEATURE_NAMES
from src.features.macro import MACRO_FEATURE_NAMES
from src.features.pipeline import COMPOSITE_FEATURE_NAMES, FeaturePipeline
from src.features.technical import TECHNICAL_FEATURE_NAMES
from src.labels.forward_returns import compute_forward_returns
from src.models.evaluation import icir, information_coefficient, rank_information_coefficient

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only in misconfigured environments.
    pq = None


FEATURE_START_DATE = date(2016, 3, 1)
FEATURE_END_DATE = date(2025, 6, 30)
AS_OF_DATE = date(2026, 3, 31)
LABEL_BUFFER_DAYS = 120
FEATURE_BATCH_SIZE = 25
FEATURE_PROGRESS_INTERVAL = 50
DEFAULT_MAX_WORKERS = min(12, max(1, (os.cpu_count() or 1) - 1))
IC_THRESHOLD = 0.01
HORIZON_DAYS = 5
SIGN_CONSISTENCY_THRESHOLD = 0.6
CORRELATION_THRESHOLD = 0.85
RIDGE_FEATURE_LIMIT = 35
TREE_FEATURE_LIMIT = 18
DEDUP_REBALANCE_WEEKDAY = 4
REPORT_SCHEMA_VERSION = 2
MISSING_INDICATOR_PREFIX = "is_missing_"

FEATURE_COLUMNS = ["ticker", "trade_date", "feature_name", "feature_value", "is_filled"]
LABEL_COLUMNS = ["ticker", "trade_date", "horizon", "forward_return", "excess_return"]
DATE_LEVEL_NAME = "trade_date"
TICKER_LEVEL_NAME = "ticker"

BASE_CANDIDATE_FEATURE_NAMES = (
    *TECHNICAL_FEATURE_NAMES,
    *FUNDAMENTAL_FEATURE_NAMES,
    *MACRO_FEATURE_NAMES,
    *COMPOSITE_FEATURE_NAMES,
)
BASE_FEATURE_DOMAIN_BY_NAME = {
    **{name: "technical" for name in TECHNICAL_FEATURE_NAMES},
    **{name: "fundamental" for name in FUNDAMENTAL_FEATURE_NAMES},
    **{name: "macro" for name in MACRO_FEATURE_NAMES},
    **{name: "composite" for name in COMPOSITE_FEATURE_NAMES},
}
CANDIDATE_FEATURE_NAMES = (
    *BASE_CANDIDATE_FEATURE_NAMES,
    *(f"{MISSING_INDICATOR_PREFIX}{name}" for name in BASE_CANDIDATE_FEATURE_NAMES),
)
FEATURE_DOMAIN_BY_NAME = {
    **BASE_FEATURE_DOMAIN_BY_NAME,
    **{
        f"{MISSING_INDICATOR_PREFIX}{name}": BASE_FEATURE_DOMAIN_BY_NAME[name]
        for name in BASE_CANDIDATE_FEATURE_NAMES
    },
}
MODEL_FEATURE_LIMITS = {
    "ridge": RIDGE_FEATURE_LIMIT,
    "xgboost": TREE_FEATURE_LIMIT,
    "lightgbm": TREE_FEATURE_LIMIT,
}
FEATURE_SERIES_CACHE: dict[tuple[str, tuple[str, ...]], pd.Series] = {}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    install_runtime_optimizations()

    feature_start = parse_date(args.feature_start_date)
    feature_end = parse_date(args.feature_end_date)
    as_of = parse_date(args.as_of)
    label_end = min(feature_end + timedelta(days=args.label_buffer_days), as_of)

    if as_of < feature_end:
        raise ValueError("as_of must be on or after feature_end_date.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.max_workers <= 0:
        raise ValueError("max_workers must be positive.")

    feature_output_path = REPO_ROOT / args.feature_output
    label_output_path = REPO_ROOT / args.label_output
    report_output_path = REPO_ROOT / args.report_output
    batch_dir = feature_output_path.parent / f"{feature_output_path.stem}_batches"
    manifest_path = batch_dir / "manifest.json"

    tickers = load_universe_tickers()
    logger.info(
        "IC screening configured for {} tickers, feature window {} -> {}, label window {} -> {}, as_of {}",
        len(tickers),
        feature_start,
        feature_end,
        feature_start,
        label_end,
        as_of,
    )

    feature_summary = build_or_load_feature_cache(
        tickers=tickers,
        feature_start=feature_start,
        feature_end=feature_end,
        as_of=as_of,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        progress_interval=args.progress_interval,
        feature_output_path=feature_output_path,
        batch_dir=batch_dir,
        manifest_path=manifest_path,
    )
    labels = build_or_load_labels(
        tickers=tickers,
        feature_start=feature_start,
        feature_end=feature_end,
        label_end=label_end,
        as_of=as_of,
        horizon=args.horizon,
        label_output_path=label_output_path,
    )
    report = build_or_load_ic_report(
        labels=labels,
        batch_dir=batch_dir,
        feature_output_path=feature_output_path,
        report_output_path=report_output_path,
        ic_threshold=args.ic_threshold,
        sign_consistency_threshold=args.sign_consistency_threshold,
        correlation_threshold=args.correlation_threshold,
        rebalance_weekday=args.rebalance_weekday,
        ridge_feature_limit=args.ridge_feature_limit,
        tree_feature_limit=args.tree_feature_limit,
    )

    log_report_summary(feature_summary=feature_summary, report=report, ic_threshold=args.ic_threshold)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PIT-safe feature ICs versus 5D forward excess returns with resumable caching.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature-start-date", default=FEATURE_START_DATE.isoformat())
    parser.add_argument("--feature-end-date", default=FEATURE_END_DATE.isoformat())
    parser.add_argument("--as-of", default=AS_OF_DATE.isoformat())
    parser.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    parser.add_argument("--batch-size", type=int, default=FEATURE_BATCH_SIZE)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--progress-interval", type=int, default=FEATURE_PROGRESS_INTERVAL)
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS)
    parser.add_argument("--ic-threshold", type=float, default=IC_THRESHOLD)
    parser.add_argument("--sign-consistency-threshold", type=float, default=SIGN_CONSISTENCY_THRESHOLD)
    parser.add_argument("--correlation-threshold", type=float, default=CORRELATION_THRESHOLD)
    parser.add_argument("--rebalance-weekday", type=int, default=DEDUP_REBALANCE_WEEKDAY)
    parser.add_argument("--ridge-feature-limit", type=int, default=RIDGE_FEATURE_LIMIT)
    parser.add_argument("--tree-feature-limit", type=int, default=TREE_FEATURE_LIMIT)
    parser.add_argument("--feature-output", default="data/features/all_features.parquet")
    parser.add_argument("--label-output", default="data/labels/forward_returns_5d.parquet")
    parser.add_argument("--report-output", default="data/features/ic_screening_report.csv")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_universe_tickers() -> list[str]:
    engine = get_engine()
    statement = text(
        """
        SELECT DISTINCT ticker
        FROM stock_prices
        WHERE ticker <> 'SPY'
        ORDER BY ticker
        """
    )
    with engine.connect() as connection:
        tickers = [str(row[0]).upper() for row in connection.execute(statement).fetchall()]

    if not tickers:
        raise RuntimeError("No non-SPY tickers found in stock_prices.")
    return tickers


def install_runtime_optimizations() -> None:
    pipeline_module.compute_fundamental_features = compute_fundamental_features_fast
    pipeline_module.compute_macro_features = compute_macro_features_cached
    pipeline_module.preprocess_features = preprocess_candidate_features
    pipeline_module.FeaturePipeline.run = run_feature_pipeline_fast
    logger.info(
        "installed runtime optimizations for batch-level fundamentals preload, "
        "single-query SPY reuse, macro caching, candidate-only preprocessing, "
        "and fast screening pipeline execution",
    )


def preprocess_candidate_features(features_df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    finalized = preprocessing_module.preprocess_features(features_df, method=method)
    logger.info(
        "preprocessed {} candidate feature rows into {} rows with missing indicators preserved",
        len(features_df),
        len(finalized),
    )
    return finalized


@lru_cache(maxsize=None)
def compute_macro_features_cached(as_of: date | datetime) -> pd.DataFrame:
    from src.features.macro import compute_macro_features

    return compute_macro_features(as_of=as_of).copy()


def compute_fundamental_features_fast(
    ticker: str,
    as_of: date | datetime,
    prices_df: pd.DataFrame,
    *,
    raw_pit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if prices_df.empty:
        return fundamental_module._empty_feature_frame()

    max_as_of = fundamental_module._coerce_as_of_date(as_of)
    prepared_prices = prices_df.copy()
    prepared_prices["ticker"] = prepared_prices["ticker"].astype(str).str.upper()
    prepared_prices["trade_date"] = pd.to_datetime(prepared_prices["trade_date"]).dt.date
    prepared_prices["close"] = pd.to_numeric(prepared_prices["close"], errors="coerce")
    prepared_prices = prepared_prices.loc[
        (prepared_prices["ticker"] == ticker.upper()) & (prepared_prices["trade_date"] <= max_as_of)
    ].sort_values("trade_date")
    if prepared_prices.empty:
        return fundamental_module._empty_feature_frame()

    if raw_pit is None:
        raw_pit = load_raw_fundamentals_history(
            ticker=ticker.upper(),
            max_trade_date=prepared_prices["trade_date"].max(),
        )
    active_rows: dict[tuple[str, str], dict[str, Any]] = {}
    active_frame = fundamental_module._empty_feature_frame()
    current_history = pd.DataFrame()
    raw_records = raw_pit.to_dict("records") if not raw_pit.empty else []
    pointer = 0
    rows: list[dict[str, object]] = []

    for trade_date, close in prepared_prices[["trade_date", "close"]].itertuples(index=False):
        trade_cutoff = datetime.combine(trade_date, time.max, tzinfo=timezone.utc)
        state_changed = False
        while pointer < len(raw_records) and raw_records[pointer]["knowledge_time"] <= trade_cutoff:
            record = raw_records[pointer]
            active_rows[(str(record["fiscal_period"]), str(record["metric_name"]))] = record
            pointer += 1
            state_changed = True
        if state_changed:
            active_frame = (
                pd.DataFrame(active_rows.values())
                if active_rows
                else fundamental_module._empty_feature_frame()
            )
            current_history = (
                fundamental_module._build_pit_history(active_frame)
                if not active_frame.empty
                else pd.DataFrame()
            )

        features = calculate_feature_snapshot_from_history(
            history=current_history,
            price=float(close) if not pd.isna(close) else np.nan,
        )
        for feature_name, feature_value in features.items():
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "trade_date": trade_date,
                    "feature_name": feature_name,
                    "feature_value": feature_value,
                },
            )

    feature_frame = pd.DataFrame(rows)
    logger.info(
        "computed {} PIT fundamental feature rows for {} across {} dates (optimized)",
        len(feature_frame),
        ticker.upper(),
        prepared_prices["trade_date"].nunique(),
    )
    return feature_frame


def run_feature_pipeline_fast(
    self: FeaturePipeline,
    tickers: list[str] | tuple[str, ...],
    start_date: date | datetime,
    end_date: date | datetime,
    as_of: date | datetime,
) -> pd.DataFrame:
    normalized_tickers = tuple(dict.fromkeys(ticker.strip().upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        raise ValueError("At least one ticker is required.")

    start = pipeline_module._coerce_date(start_date)
    end = pipeline_module._coerce_date(end_date)
    as_of_ts = pipeline_module._coerce_as_of_datetime(as_of)
    if as_of_ts.date() < end:
        raise ValueError("as_of must be on or after end_date for PIT feature generation.")

    history_start = start - timedelta(days=520)
    price_tickers = tuple(dict.fromkeys([*normalized_tickers, "SPY"]))
    logger.info(
        "running fast feature pipeline for {} tickers from {} to {} as_of {}",
        len(normalized_tickers),
        start,
        end,
        as_of_ts,
    )
    prices = get_prices_pit(
        tickers=price_tickers,
        start_date=history_start,
        end_date=end,
        as_of=as_of_ts,
    )
    if prices.empty:
        logger.warning("fast feature pipeline found no PIT prices for requested tickers")
        return pipeline_module._empty_feature_frame()

    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    prices.sort_values(["ticker", "trade_date"], inplace=True)
    stock_prices = prices.loc[prices["ticker"].isin(normalized_tickers)].copy()
    market_prices = prices.loc[prices["ticker"] == "SPY"].copy()
    output_prices = stock_prices.loc[
        (stock_prices["trade_date"] >= start) & (stock_prices["trade_date"] <= end)
    ].copy()
    if output_prices.empty:
        logger.warning("fast feature pipeline has no prices inside the requested output window")
        return pipeline_module._empty_feature_frame()

    technical = pipeline_module.compute_technical_features(
        stock_prices,
        market_prices_df=market_prices,
    )
    technical = technical.loc[
        (technical["trade_date"] >= start)
        & (technical["trade_date"] <= end)
        & (technical["ticker"].isin(normalized_tickers))
    ].copy()

    raw_fundamentals = load_raw_fundamentals_history_batch(
        tickers=normalized_tickers,
        max_trade_date=output_prices["trade_date"].max(),
    )
    raw_fundamentals_by_ticker = {
        str(ticker).upper(): frame.reset_index(drop=True)
        for ticker, frame in raw_fundamentals.groupby("ticker", sort=False)
    }
    empty_raw_history = empty_raw_fundamentals_history()

    fundamental_frames: list[pd.DataFrame] = []
    for ticker in normalized_tickers:
        ticker_prices = output_prices.loc[output_prices["ticker"] == ticker]
        if ticker_prices.empty:
            continue
        fundamental_frames.append(
            compute_fundamental_features_fast(
                ticker=ticker,
                as_of=as_of_ts,
                prices_df=ticker_prices,
                raw_pit=raw_fundamentals_by_ticker.get(ticker, empty_raw_history),
            ),
        )
    fundamentals = (
        pd.concat(fundamental_frames, ignore_index=True)
        if fundamental_frames
        else pipeline_module._empty_feature_frame()
    )

    macro = self._compute_broadcast_macro_features(output_prices, as_of_ts)
    base_features = pd.concat([technical, fundamentals, macro], ignore_index=True)
    composite = pipeline_module.compute_composite_features(base_features)
    all_features = pd.concat([base_features, composite], ignore_index=True)

    sector_rel_frames: list[pd.DataFrame] = []
    for td in sorted(set(fundamentals["trade_date"].unique()) if not fundamentals.empty else []):
        td_date = td if isinstance(td, date) else td.date() if hasattr(td, "date") else td
        cross_section = all_features.loc[all_features["trade_date"] == td]
        sector_rel = pipeline_module.compute_sector_relative_from_raw_features(cross_section, td_date)
        if not sector_rel.empty:
            sector_rel_frames.append(sector_rel)
    if sector_rel_frames:
        all_features = pd.concat([all_features, *sector_rel_frames], ignore_index=True)

    processed = pipeline_module.preprocess_features(all_features)
    batch_id = str(uuid.uuid4())
    self.last_batch_id = batch_id
    processed.attrs["batch_id"] = batch_id
    logger.info("fast feature pipeline completed batch {} with {} rows", batch_id, len(processed))
    return processed


def calculate_feature_snapshot_from_history(*, history: pd.DataFrame, price: float) -> dict[str, float]:
    return fundamental_module._calculate_feature_snapshot_from_history(history=history, price=price)


def load_raw_fundamentals_history(*, ticker: str, max_trade_date: date) -> pd.DataFrame:
    batch = load_raw_fundamentals_history_batch(
        tickers=[ticker.upper()],
        max_trade_date=max_trade_date,
    )
    if batch.empty:
        return empty_raw_fundamentals_history()
    return batch.loc[batch["ticker"] == ticker.upper()].reset_index(drop=True)


def load_raw_fundamentals_history_batch(*, tickers: list[str] | tuple[str, ...], max_trade_date: date) -> pd.DataFrame:
    normalized_tickers = tuple(dict.fromkeys(str(ticker).upper() for ticker in tickers if ticker))
    if not normalized_tickers:
        return empty_raw_fundamentals_history()

    cutoff = datetime.combine(max_trade_date, time.max, tzinfo=timezone.utc)
    statement = (
        sa.select(
            FundamentalsPIT.id,
            FundamentalsPIT.ticker,
            FundamentalsPIT.fiscal_period,
            FundamentalsPIT.metric_name,
            FundamentalsPIT.metric_value,
            FundamentalsPIT.event_time,
            FundamentalsPIT.knowledge_time,
            FundamentalsPIT.is_restated,
            FundamentalsPIT.source,
        )
        .where(
            FundamentalsPIT.ticker.in_(normalized_tickers),
            FundamentalsPIT.metric_name.in_(fundamental_module._PIT_METRIC_NAMES),
            FundamentalsPIT.knowledge_time <= cutoff,
            FundamentalsPIT.event_time <= max_trade_date,
        )
        .order_by(
            FundamentalsPIT.knowledge_time,
            FundamentalsPIT.event_time,
            FundamentalsPIT.id,
        )
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    frame = pd.DataFrame(rows)
    if frame.empty:
        return empty_raw_fundamentals_history()

    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["knowledge_time"] = pd.to_datetime(frame["knowledge_time"], utc=True)
    frame["event_time"] = pd.to_datetime(frame["event_time"]).dt.date
    return frame


def empty_raw_fundamentals_history() -> pd.DataFrame:
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


def build_or_load_feature_cache(
    *,
    tickers: list[str],
    feature_start: date,
    feature_end: date,
    as_of: date,
    batch_size: int,
    max_workers: int,
    progress_interval: int,
    feature_output_path: Path,
    batch_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_tickers = tickers
    manifest_batch_size = batch_size
    expected_manifest = build_manifest(
        tickers=tickers,
        feature_start=feature_start,
        feature_end=feature_end,
        as_of=as_of,
        batch_size=batch_size,
    )
    force_rebuild = not feature_output_path.exists()
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != expected_manifest:
            if feature_output_path.exists():
                logger.info(
                    "using existing legacy feature cache at {}; missing indicators will be synthesized downstream",
                    feature_output_path,
                )
                manifest_tickers = [str(ticker).upper() for ticker in existing_manifest.get("tickers", tickers)]
                manifest_batch_size = int(existing_manifest.get("batch_size", batch_size))
            else:
                logger.info("feature cache manifest changed at {}; rebuilding screening batches in place", manifest_path)
                force_rebuild = True
                write_json_atomic(manifest_path, expected_manifest)
        else:
            manifest_tickers = [str(ticker).upper() for ticker in existing_manifest.get("tickers", tickers)]
            manifest_batch_size = int(existing_manifest.get("batch_size", batch_size))
    else:
        force_rebuild = True
        write_json_atomic(manifest_path, expected_manifest)

    batch_specs = build_batch_specs(tickers=manifest_tickers, batch_size=manifest_batch_size, batch_dir=batch_dir)
    completed_tickers = 0 if force_rebuild else count_completed_tickers(batch_specs=batch_specs)

    if force_rebuild or not feature_output_path.exists():
        pending_specs: list[tuple[int, dict[str, Any]]] = []
        for batch_index, batch_spec in enumerate(batch_specs, start=1):
            if batch_spec["path"].exists() and not force_rebuild:
                logger.info(
                    "skipping existing feature batch {}/{} at {}",
                    batch_index,
                    len(batch_specs),
                    batch_spec["path"],
                )
                continue
            pending_specs.append((batch_index, batch_spec))

        worker_count = min(max_workers, max(1, len(pending_specs)))
        if pending_specs:
            logger.info(
                "building {} pending feature batches with up to {} worker processes",
                len(pending_specs),
                worker_count,
            )
        if worker_count <= 1:
            for batch_index, batch_spec in pending_specs:
                result = build_feature_batch_worker(
                    batch_index=batch_index,
                    total_batches=len(batch_specs),
                    batch_tickers=batch_spec["tickers"],
                    feature_start=feature_start,
                    feature_end=feature_end,
                    as_of=as_of,
                    batch_path=str(batch_spec["path"]),
                )
                completed_tickers += int(result["ticker_count"])
                if completed_tickers % progress_interval == 0 or completed_tickers == len(tickers):
                    logger.info("feature progress: processed {}/{} tickers", completed_tickers, len(tickers))
        elif pending_specs:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_to_spec = {
                    executor.submit(
                        build_feature_batch_worker,
                        batch_index=batch_index,
                        total_batches=len(batch_specs),
                        batch_tickers=batch_spec["tickers"],
                        feature_start=feature_start,
                        feature_end=feature_end,
                        as_of=as_of,
                        batch_path=str(batch_spec["path"]),
                    ): (batch_index, batch_spec)
                    for batch_index, batch_spec in pending_specs
                }
                for future in as_completed(future_to_spec):
                    batch_index, batch_spec = future_to_spec[future]
                    result = future.result()
                    completed_tickers += int(result["ticker_count"])
                    logger.info(
                        "completed feature batch {}/{} at {} rows={} tickers={}",
                        batch_index,
                        len(batch_specs),
                        batch_spec["path"],
                        result["row_count"],
                        result["ticker_count"],
                    )
                    if completed_tickers % progress_interval == 0 or completed_tickers == len(tickers):
                        logger.info("feature progress: processed {}/{} tickers", completed_tickers, len(tickers))

        combine_feature_batches(batch_specs=batch_specs, output_path=feature_output_path)
    else:
        logger.info("using existing feature cache at {}", feature_output_path)

    feature_summary = summarize_feature_batches(batch_specs=batch_specs, feature_output_path=feature_output_path)
    logger.info(
        "feature cache ready: {} rows across {} tickers and {} feature names",
        feature_summary["row_count"],
        feature_summary["ticker_count"],
        feature_summary["feature_count"],
    )
    return feature_summary


def build_feature_batch_worker(
    *,
    batch_index: int,
    total_batches: int,
    batch_tickers: list[str],
    feature_start: date,
    feature_end: date,
    as_of: date,
    batch_path: str,
) -> dict[str, Any]:
    install_runtime_optimizations()
    pipeline = FeaturePipeline()
    logger.info(
        "running feature batch {}/{} for {} tickers ({} -> {})",
        batch_index,
        total_batches,
        len(batch_tickers),
        batch_tickers[0],
        batch_tickers[-1],
    )
    features = pipeline.run(
        tickers=batch_tickers,
        start_date=feature_start,
        end_date=feature_end,
        as_of=as_of,
    )
    filtered = prepare_feature_batch(features)
    write_parquet_atomic(filtered, Path(batch_path))
    row_count = int(len(filtered))
    ticker_count = int(len(batch_tickers))
    del features
    del filtered
    gc.collect()
    return {
        "batch_path": batch_path,
        "row_count": row_count,
        "ticker_count": ticker_count,
    }


def build_manifest(
    *,
    tickers: list[str],
    feature_start: date,
    feature_end: date,
    as_of: date,
    batch_size: int,
) -> dict[str, Any]:
    return {
        "tickers": tickers,
        "feature_start_date": feature_start.isoformat(),
        "feature_end_date": feature_end.isoformat(),
        "as_of": as_of.isoformat(),
        "batch_size": batch_size,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
    }


def build_batch_specs(*, tickers: list[str], batch_size: int, batch_dir: Path) -> list[dict[str, Any]]:
    batch_specs: list[dict[str, Any]] = []
    batch_dir.mkdir(parents=True, exist_ok=True)

    for start_index in range(0, len(tickers), batch_size):
        batch_number = (start_index // batch_size) + 1
        batch_tickers = tickers[start_index : start_index + batch_size]
        first = batch_tickers[0]
        last = batch_tickers[-1]
        batch_path = batch_dir / f"batch_{batch_number:03d}_{first}_{last}.parquet"
        batch_specs.append({"tickers": batch_tickers, "path": batch_path})
    return batch_specs


def count_completed_tickers(*, batch_specs: list[dict[str, Any]]) -> int:
    return sum(len(batch_spec["tickers"]) for batch_spec in batch_specs if batch_spec["path"].exists())


def prepare_feature_batch(features: pd.DataFrame) -> pd.DataFrame:
    filtered = features.loc[features["feature_name"].isin(CANDIDATE_FEATURE_NAMES), FEATURE_COLUMNS].copy()
    filtered["ticker"] = filtered["ticker"].astype(str).str.upper()
    filtered["trade_date"] = pd.to_datetime(filtered["trade_date"]).dt.date
    filtered["feature_name"] = filtered["feature_name"].astype(str)
    filtered["feature_value"] = pd.to_numeric(filtered["feature_value"], errors="coerce")
    filtered["is_filled"] = filtered["is_filled"].fillna(False).astype(bool)
    filtered.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    filtered.reset_index(drop=True, inplace=True)
    return filtered


def combine_feature_batches(*, batch_specs: list[dict[str, Any]], output_path: Path) -> None:
    if pq is None:
        raise RuntimeError("pyarrow is required to combine batch parquet files.")

    missing_batches = [str(batch_spec["path"]) for batch_spec in batch_specs if not batch_spec["path"].exists()]
    if missing_batches:
        raise RuntimeError(f"Cannot build {output_path}; missing batch files: {missing_batches}")

    temp_path = temp_path_for(output_path)
    if temp_path.exists():
        temp_path.unlink()

    writer: pq.ParquetWriter | None = None
    try:
        total_rows = 0
        for batch_spec in batch_specs:
            table = pq.read_table(batch_spec["path"])
            if writer is None:
                writer = pq.ParquetWriter(temp_path, table.schema, compression="snappy")
            writer.write_table(table)
            total_rows += table.num_rows
        if writer is not None:
            writer.close()
            writer = None
        temp_path.replace(output_path)
        logger.info("combined {} feature batches into {} rows at {}", len(batch_specs), total_rows, output_path)
    finally:
        if writer is not None:
            writer.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def summarize_feature_batches(*, batch_specs: list[dict[str, Any]], feature_output_path: Path) -> dict[str, Any]:
    missing_batch_paths = [batch_spec["path"] for batch_spec in batch_specs if not batch_spec["path"].exists()]
    if missing_batch_paths and feature_output_path.exists():
        frame = pd.read_parquet(feature_output_path, columns=["ticker", "feature_name"])
        return {
            "row_count": int(len(frame)),
            "ticker_count": int(frame["ticker"].astype(str).str.upper().nunique()),
            "feature_count": int(frame["feature_name"].astype(str).nunique()),
        }

    ticker_set: set[str] = set()
    feature_set: set[str] = set()
    row_count = 0
    for batch_spec in batch_specs:
        batch_path = batch_spec["path"]
        frame = pd.read_parquet(batch_path, columns=["ticker", "feature_name"])
        ticker_set.update(frame["ticker"].astype(str).str.upper().unique().tolist())
        feature_set.update(frame["feature_name"].astype(str).unique().tolist())
        row_count += len(frame)
    return {
        "row_count": row_count,
        "ticker_count": len(ticker_set),
        "feature_count": len(feature_set),
    }


def build_or_load_labels(
    *,
    tickers: list[str],
    feature_start: date,
    feature_end: date,
    label_end: date,
    as_of: date,
    horizon: int,
    label_output_path: Path,
) -> pd.DataFrame:
    label_output_path.parent.mkdir(parents=True, exist_ok=True)
    if label_output_path.exists():
        logger.info("using existing label cache at {}", label_output_path)
        labels = pd.read_parquet(label_output_path)
        labels["trade_date"] = pd.to_datetime(labels["trade_date"]).dt.date
        return labels

    label_tickers = tickers + ["SPY"]
    prices = get_prices_pit(
        tickers=label_tickers,
        start_date=feature_start,
        end_date=label_end,
        as_of=as_of,
    )
    if prices.empty:
        raise RuntimeError("No PIT prices found for label generation.")

    logger.info(
        "loaded {} PIT price rows across {} tickers for forward-return labels",
        len(prices),
        prices["ticker"].nunique(),
    )
    labels = compute_forward_returns(prices_df=prices, horizons=[horizon], benchmark_ticker="SPY")
    labels = labels.loc[
        (labels["horizon"] == horizon)
        & (labels["ticker"].astype(str).str.upper() != "SPY")
        & (pd.to_datetime(labels["trade_date"]).dt.date >= feature_start)
        & (pd.to_datetime(labels["trade_date"]).dt.date <= feature_end),
        LABEL_COLUMNS,
    ].copy()
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"]).dt.date
    labels["forward_return"] = pd.to_numeric(labels["forward_return"], errors="coerce")
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    labels.sort_values(["trade_date", "ticker"], inplace=True)
    labels.reset_index(drop=True, inplace=True)
    write_parquet_atomic(labels, label_output_path)
    logger.info(
        "saved {} {}D forward-return labels across {} tickers to {}",
        len(labels),
        horizon,
        labels["ticker"].nunique(),
        label_output_path,
    )
    return labels


def build_or_load_ic_report(
    *,
    labels: pd.DataFrame,
    batch_dir: Path,
    feature_output_path: Path,
    report_output_path: Path,
    ic_threshold: float,
    sign_consistency_threshold: float,
    correlation_threshold: float,
    rebalance_weekday: int,
    ridge_feature_limit: int,
    tree_feature_limit: int,
) -> pd.DataFrame:
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_output_path.exists():
        cached_report = pd.read_csv(report_output_path)
        if report_matches_current_schema(cached_report):
            logger.info("using existing IC report at {}", report_output_path)
            return cached_report
        logger.info("rebuilding IC report at {} because the schema or feature set changed", report_output_path)

    label_series = build_label_series(labels)
    feature_source_paths = available_feature_source_paths(batch_dir=batch_dir, feature_output_path=feature_output_path)
    if not feature_source_paths:
        raise RuntimeError("No feature parquet files found for IC screening.")

    records: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(CANDIDATE_FEATURE_NAMES, start=1):
        logger.info(
            "computing IC metrics for feature {}/{}: {}",
            feature_index,
            len(CANDIDATE_FEATURE_NAMES),
            feature_name,
        )
        feature_series = load_feature_series(feature_name=feature_name, batch_paths=feature_source_paths)
        aligned = align_label_and_feature(label_series=label_series, feature_series=feature_series)
        records.append(
            compute_feature_stability_record(
                feature_name=feature_name,
                aligned=aligned,
                ic_threshold=ic_threshold,
                sign_consistency_threshold=sign_consistency_threshold,
            ),
        )
        gc.collect()

    report = pd.DataFrame(records)
    report = annotate_model_feature_sets(
        report=report,
        batch_paths=feature_source_paths,
        rebalance_weekday=rebalance_weekday,
        correlation_threshold=correlation_threshold,
        ridge_feature_limit=ridge_feature_limit,
        tree_feature_limit=tree_feature_limit,
    )
    report = report.sort_values(["retained", "stability_score", "feature_name"], ascending=[False, False, True]).reset_index(
        drop=True,
    )
    write_csv_atomic(report, report_output_path)
    logger.info("wrote IC screening report for {} features to {}", len(report), report_output_path)
    return report


def report_matches_current_schema(report: pd.DataFrame) -> bool:
    required_columns = {
        "feature_name",
        "domain",
        "feature_kind",
        "base_feature_name",
        "report_schema_version",
        "signed_ic",
        "rank_ic",
        "icir",
        "sign_consistency",
        "stability_score",
        "window_count",
        "window_signed_ic_mean",
        "window_rank_ic_mean",
        "window_icir_mean",
        "passed",
        "retained",
        "retained_ridge",
        "retained_xgboost",
        "retained_lightgbm",
    }
    if not required_columns.issubset(report.columns):
        return False
    return set(report["feature_name"].astype(str)) == set(CANDIDATE_FEATURE_NAMES)


def available_feature_source_paths(*, batch_dir: Path, feature_output_path: Path) -> list[Path]:
    batch_paths = sorted(batch_dir.glob("*.parquet"))
    if batch_paths:
        return batch_paths
    if feature_output_path.exists():
        return [feature_output_path]
    return []


def compute_feature_stability_record(
    *,
    feature_name: str,
    aligned: pd.DataFrame,
    ic_threshold: float,
    sign_consistency_threshold: float,
) -> dict[str, Any]:
    if aligned.empty:
        window_metrics = pd.DataFrame(
            columns=["window_id", "signed_ic", "rank_ic", "icir", "n_obs", "n_dates", "n_tickers"],
        )
        signed_ic = float("nan")
        rank_ic_value = float("nan")
        icir_value = float("nan")
        row_count = 0
        date_count = 0
        ticker_count = 0
    else:
        y_true = aligned["y_true"]
        y_pred = aligned["y_pred"]
        signed_ic = information_coefficient(y_true=y_true, y_pred=y_pred)
        rank_ic_value = rank_information_coefficient(y_true=y_true, y_pred=y_pred)
        icir_value = icir(y_true=y_true, y_pred=y_pred)
        row_count = int(len(aligned))
        date_count = int(aligned.index.get_level_values(DATE_LEVEL_NAME).nunique())
        ticker_count = int(aligned.index.get_level_values(TICKER_LEVEL_NAME).nunique())
        window_metrics = compute_window_metrics(aligned)

    valid_window_ics = window_metrics["signed_ic"].dropna() if "signed_ic" in window_metrics else pd.Series(dtype=float)
    positive_window_share = float((valid_window_ics > 0).mean()) if not valid_window_ics.empty else float("nan")
    negative_window_share = float((valid_window_ics < 0).mean()) if not valid_window_ics.empty else float("nan")
    sign_consistency = (
        max(positive_window_share, negative_window_share)
        if not valid_window_ics.empty
        else float("nan")
    )
    dominant_sign = infer_dominant_sign(
        signed_ic=signed_ic,
        positive_window_share=positive_window_share,
        negative_window_share=negative_window_share,
    )
    window_signed_ic_mean = nanmean(valid_window_ics.tolist())
    window_rank_ic_mean = nanmean(window_metrics["rank_ic"].dropna().tolist()) if "rank_ic" in window_metrics else float("nan")
    window_icir_mean = nanmean(window_metrics["icir"].dropna().tolist()) if "icir" in window_metrics else float("nan")
    stability_reference_ic = window_signed_ic_mean if pd.notna(window_signed_ic_mean) else signed_ic
    stability_score = (
        abs(stability_reference_ic) * sign_consistency
        if pd.notna(stability_reference_ic) and pd.notna(sign_consistency)
        else float("nan")
    )
    passed = bool(
        pd.notna(stability_reference_ic)
        and abs(stability_reference_ic) >= ic_threshold
        and pd.notna(sign_consistency)
        and sign_consistency >= sign_consistency_threshold
    )
    feature_kind = "missing_indicator" if is_missing_indicator_feature(feature_name) else "base"

    return {
        "feature_name": feature_name,
        "base_feature_name": base_feature_name(feature_name),
        "feature_kind": feature_kind,
        "domain": FEATURE_DOMAIN_BY_NAME[feature_name],
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "ic": signed_ic,
        "signed_ic": signed_ic,
        "rank_ic": rank_ic_value,
        "icir": icir_value,
        "abs_ic": abs(signed_ic) if pd.notna(signed_ic) else float("nan"),
        "window_count": int(len(window_metrics)),
        "window_signed_ic_mean": window_signed_ic_mean,
        "window_rank_ic_mean": window_rank_ic_mean,
        "window_icir_mean": window_icir_mean,
        "window_signed_ic_std": nanstd(valid_window_ics.tolist()),
        "positive_window_share": positive_window_share,
        "negative_window_share": negative_window_share,
        "sign_consistency": sign_consistency,
        "dominant_sign": dominant_sign,
        "stability_score": stability_score,
        "n_obs": int(row_count),
        "n_dates": int(date_count),
        "n_tickers": int(ticker_count),
        "passed": passed,
        "window_metrics_json": json.dumps(
            [
                {key: normalize_json_value(value) for key, value in row.items()}
                for row in window_metrics.to_dict(orient="records")
            ],
            sort_keys=True,
        ),
    }


def compute_window_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    if aligned.empty:
        return pd.DataFrame(columns=["window_id", "signed_ic", "rank_ic", "icir", "n_obs", "n_dates", "n_tickers"])

    frame = aligned.reset_index().copy()
    frame[DATE_LEVEL_NAME] = pd.to_datetime(frame[DATE_LEVEL_NAME])
    frame["window_id"] = frame[DATE_LEVEL_NAME].map(half_year_window_id)

    records: list[dict[str, Any]] = []
    for window_id, group in frame.groupby("window_id", sort=True):
        indexed = group.set_index([DATE_LEVEL_NAME, TICKER_LEVEL_NAME]).sort_index()
        y_true = indexed["y_true"]
        y_pred = indexed["y_pred"]
        records.append(
            {
                "window_id": str(window_id),
                "signed_ic": information_coefficient(y_true=y_true, y_pred=y_pred),
                "rank_ic": rank_information_coefficient(y_true=y_true, y_pred=y_pred),
                "icir": icir(y_true=y_true, y_pred=y_pred),
                "n_obs": int(len(indexed)),
                "n_dates": int(indexed.index.get_level_values(DATE_LEVEL_NAME).nunique()),
                "n_tickers": int(indexed.index.get_level_values(TICKER_LEVEL_NAME).nunique()),
            },
        )

    return pd.DataFrame(records)


def half_year_window_id(trade_date: Any) -> str:
    timestamp = pd.Timestamp(trade_date)
    half = 1 if timestamp.month <= 6 else 2
    return f"{timestamp.year}H{half}"


def infer_dominant_sign(
    *,
    signed_ic: float,
    positive_window_share: float,
    negative_window_share: float,
) -> str:
    if pd.notna(positive_window_share) and positive_window_share > negative_window_share:
        return "positive"
    if pd.notna(negative_window_share) and negative_window_share > positive_window_share:
        return "negative"
    if pd.notna(signed_ic) and signed_ic > 0:
        return "positive"
    if pd.notna(signed_ic) and signed_ic < 0:
        return "negative"
    return "mixed"


def annotate_model_feature_sets(
    *,
    report: pd.DataFrame,
    batch_paths: list[Path],
    rebalance_weekday: int,
    correlation_threshold: float,
    ridge_feature_limit: int,
    tree_feature_limit: int,
) -> pd.DataFrame:
    annotated = report.copy()
    if annotated.empty:
        return annotated

    available_features = set(annotated["feature_name"].astype(str))
    stability_passed = annotated.loc[annotated["passed"].astype(bool), "feature_name"].astype(str).tolist()
    correlation_candidates = sorted(
        set(stability_passed)
        | {
            missing_indicator_name(feature_name)
            for feature_name in stability_passed
            if not is_missing_indicator_feature(feature_name)
            and missing_indicator_name(feature_name) in available_features
        },
    )
    feature_matrix = load_feature_matrix_for_correlation(
        feature_names=correlation_candidates,
        batch_paths=batch_paths,
        rebalance_weekday=rebalance_weekday,
    )
    correlation_matrix = compute_feature_correlation_matrix(feature_matrix)
    selection = build_model_feature_sets(
        report=annotated,
        correlation_matrix=correlation_matrix,
        correlation_threshold=correlation_threshold,
        model_feature_limits={
            "ridge": int(ridge_feature_limit),
            "xgboost": int(tree_feature_limit),
            "lightgbm": int(tree_feature_limit),
        },
    )

    cluster_id_map = selection["cluster_id_by_feature"]
    cluster_size_map = selection["cluster_size_by_feature"]
    cluster_rep_map = selection["cluster_representative_by_feature"]
    annotated["corr_cluster_id"] = annotated["feature_name"].map(cluster_id_map)
    annotated["corr_cluster_size"] = annotated["feature_name"].map(cluster_size_map)
    annotated["corr_cluster_representative"] = annotated["feature_name"].map(cluster_rep_map)

    retained_union: set[str] = set()
    for model_name, selected_features in selection["feature_sets"].items():
        retained_union.update(selected_features)
        selected_rank = {feature_name: rank for rank, feature_name in enumerate(selected_features, start=1)}
        annotated[f"retained_{model_name}"] = annotated["feature_name"].isin(selected_features)
        annotated[f"selection_rank_{model_name}"] = annotated["feature_name"].map(selected_rank)

    annotated["retained"] = annotated["feature_name"].isin(retained_union)
    return annotated


def build_model_feature_sets(
    *,
    report: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    correlation_threshold: float,
    model_feature_limits: dict[str, int],
) -> dict[str, Any]:
    candidate_rows = report.loc[report["passed"].astype(bool)].copy()
    candidate_rows["feature_name"] = candidate_rows["feature_name"].astype(str)
    candidate_features = candidate_rows["feature_name"].tolist()
    if not candidate_features:
        return {
            "feature_sets": {model_name: [] for model_name in model_feature_limits},
            "cluster_id_by_feature": {},
            "cluster_size_by_feature": {},
            "cluster_representative_by_feature": {},
        }

    components = build_correlation_components(
        feature_names=candidate_features,
        correlation_matrix=correlation_matrix,
        correlation_threshold=correlation_threshold,
    )
    row_by_feature = report.set_index("feature_name", drop=False)
    representatives: list[tuple[str, list[str]]] = []
    cluster_id_by_feature: dict[str, int] = {}
    cluster_size_by_feature: dict[str, int] = {}
    cluster_representative_by_feature: dict[str, str] = {}

    for cluster_index, members in enumerate(components, start=1):
        ordered_members = sorted(members, key=lambda name: selection_priority(row_by_feature.loc[name]), reverse=True)
        representative = ordered_members[0]
        representatives.append((representative, ordered_members))
        for feature_name in ordered_members:
            cluster_id_by_feature[feature_name] = cluster_index
            cluster_size_by_feature[feature_name] = len(ordered_members)
            cluster_representative_by_feature[feature_name] = representative

    representatives.sort(key=lambda item: selection_priority(row_by_feature.loc[item[0]]), reverse=True)

    feature_sets: dict[str, list[str]] = {}
    for model_name, limit in model_feature_limits.items():
        selected = [representative for representative, _ in representatives[:limit]]
        selected = augment_with_missing_indicators(
            selected_features=selected,
            available_features=set(report["feature_name"].astype(str)),
            correlation_matrix=correlation_matrix,
            correlation_threshold=correlation_threshold,
            limit=limit,
        )
        feature_sets[model_name] = selected

    return {
        "feature_sets": feature_sets,
        "cluster_id_by_feature": cluster_id_by_feature,
        "cluster_size_by_feature": cluster_size_by_feature,
        "cluster_representative_by_feature": cluster_representative_by_feature,
    }


def build_correlation_components(
    *,
    feature_names: list[str],
    correlation_matrix: pd.DataFrame,
    correlation_threshold: float,
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {feature_name: set() for feature_name in feature_names}
    for left_index, left_name in enumerate(feature_names):
        for right_name in feature_names[left_index + 1 :]:
            if left_name not in correlation_matrix.index or right_name not in correlation_matrix.columns:
                continue
            corr_value = correlation_matrix.at[left_name, right_name]
            if pd.notna(corr_value) and abs(float(corr_value)) > correlation_threshold:
                adjacency[left_name].add(right_name)
                adjacency[right_name].add(left_name)

    visited: set[str] = set()
    components: list[list[str]] = []
    for feature_name in feature_names:
        if feature_name in visited:
            continue
        stack = [feature_name]
        component: list[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - visited))
        components.append(component)
    return components


def augment_with_missing_indicators(
    *,
    selected_features: list[str],
    available_features: set[str],
    correlation_matrix: pd.DataFrame,
    correlation_threshold: float,
    limit: int,
) -> list[str]:
    augmented = list(selected_features)
    for feature_name in list(selected_features):
        if len(augmented) >= limit or is_missing_indicator_feature(feature_name):
            continue
        companion = missing_indicator_name(feature_name)
        if companion not in available_features or companion in augmented:
            continue
        if can_add_feature(
            candidate_feature=companion,
            selected_features=augmented,
            correlation_matrix=correlation_matrix,
            correlation_threshold=correlation_threshold,
        ):
            augmented.append(companion)
    return augmented[:limit]


def can_add_feature(
    *,
    candidate_feature: str,
    selected_features: list[str],
    correlation_matrix: pd.DataFrame,
    correlation_threshold: float,
) -> bool:
    if candidate_feature not in correlation_matrix.index:
        return True
    for selected_feature in selected_features:
        if selected_feature not in correlation_matrix.columns:
            continue
        corr_value = correlation_matrix.at[candidate_feature, selected_feature]
        if pd.notna(corr_value) and abs(float(corr_value)) > correlation_threshold:
            return False
    return True


def selection_priority(row: pd.Series) -> tuple[float, float, float, int, str]:
    stability_score = float(row.get("stability_score", float("nan")))
    window_ic = float(row.get("window_signed_ic_mean", float("nan")))
    rank_ic_value = float(row.get("rank_ic", float("nan")))
    sign_consistency = float(row.get("sign_consistency", float("nan")))
    return (
        -np.inf if pd.isna(stability_score) else stability_score,
        -np.inf if pd.isna(abs(window_ic)) else abs(window_ic),
        -np.inf if pd.isna(abs(rank_ic_value)) else abs(rank_ic_value),
        0 if row.get("feature_kind") == "missing_indicator" else 1,
        str(row.get("feature_name", "")),
    )


def load_feature_matrix_for_correlation(
    *,
    feature_names: list[str],
    batch_paths: list[Path],
    rebalance_weekday: int,
) -> pd.DataFrame:
    if not feature_names:
        return pd.DataFrame(columns=feature_names)

    slices: list[pd.DataFrame] = []
    for batch_path in batch_paths:
        frame = pd.read_parquet(
            batch_path,
            filters=[("feature_name", "in", feature_names)],
            columns=["ticker", "trade_date", "feature_name", "feature_value"],
        )
        if frame.empty:
            continue
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame.loc[frame["trade_date"].dt.weekday == rebalance_weekday].copy()
        if frame.empty:
            continue
        frame["feature_name"] = frame["feature_name"].astype(str)
        frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
        slices.append(frame)

    if not slices:
        return pd.DataFrame(columns=feature_names)

    combined = pd.concat(slices, ignore_index=True)
    combined.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    combined.drop_duplicates(["trade_date", "ticker", "feature_name"], keep="last", inplace=True)
    matrix = (
        combined.set_index(["trade_date", "ticker", "feature_name"])["feature_value"]
        .unstack("feature_name")
        .sort_index()
        .reindex(columns=feature_names)
    )
    return matrix.astype(np.float32)


def compute_feature_correlation_matrix(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    if feature_matrix.empty:
        return pd.DataFrame(index=feature_matrix.columns, columns=feature_matrix.columns, dtype=float)
    corr = feature_matrix.corr(method="pearson", min_periods=50)
    corr = corr.reindex(index=feature_matrix.columns, columns=feature_matrix.columns)
    if not corr.empty:
        np.fill_diagonal(corr.values, 1.0)
    return corr


def is_missing_indicator_feature(feature_name: str) -> bool:
    return str(feature_name).startswith(MISSING_INDICATOR_PREFIX)


def base_feature_name(feature_name: str) -> str:
    name = str(feature_name)
    if is_missing_indicator_feature(name):
        return name[len(MISSING_INDICATOR_PREFIX) :]
    return name


def missing_indicator_name(feature_name: str) -> str:
    base_name = base_feature_name(feature_name)
    return f"{MISSING_INDICATOR_PREFIX}{base_name}"


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def build_label_series(labels: pd.DataFrame) -> pd.Series:
    prepared = labels.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"]).dt.date
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["excess_return"] = pd.to_numeric(prepared["excess_return"], errors="coerce")
    series = prepared.set_index([DATE_LEVEL_NAME, TICKER_LEVEL_NAME])["excess_return"].sort_index()
    series.index = series.index.set_names([DATE_LEVEL_NAME, TICKER_LEVEL_NAME])
    return series


def load_feature_series(*, feature_name: str, batch_paths: list[Path]) -> pd.Series:
    cache_key = (str(feature_name), tuple(str(path) for path in batch_paths))
    cached = FEATURE_SERIES_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    slices: list[pd.Series] = []
    for batch_path in batch_paths:
        frame = pd.read_parquet(
            batch_path,
            filters=[("feature_name", "==", feature_name)],
            columns=["ticker", "trade_date", "feature_name", "feature_value"],
        )
        if frame.empty:
            continue
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
        series = frame.set_index([DATE_LEVEL_NAME, TICKER_LEVEL_NAME])["feature_value"]
        series.index = series.index.set_names([DATE_LEVEL_NAME, TICKER_LEVEL_NAME])
        slices.append(series.sort_index())

    if not slices:
        if is_missing_indicator_feature(feature_name):
            derived = derive_missing_indicator_series(feature_name=feature_name, batch_paths=batch_paths)
            FEATURE_SERIES_CACHE[cache_key] = derived.copy()
            return derived
        empty = pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=[DATE_LEVEL_NAME, TICKER_LEVEL_NAME]),
            name=feature_name,
        )
        FEATURE_SERIES_CACHE[cache_key] = empty.copy()
        return empty
    series = pd.concat(slices).sort_index()
    FEATURE_SERIES_CACHE[cache_key] = series.copy()
    return series


def derive_missing_indicator_series(*, feature_name: str, batch_paths: list[Path]) -> pd.Series:
    base_name = base_feature_name(feature_name)
    base_series = load_feature_series(feature_name=base_name, batch_paths=batch_paths)
    if base_series.empty:
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=[DATE_LEVEL_NAME, TICKER_LEVEL_NAME]),
            name=feature_name,
        )
    indicator = base_series.isna().astype(float)
    indicator.name = feature_name
    return indicator.sort_index()


def align_label_and_feature(*, label_series: pd.Series, feature_series: pd.Series) -> pd.DataFrame:
    aligned = pd.concat(
        [label_series.rename("y_true"), feature_series.rename("y_pred")],
        axis=1,
        join="inner",
    ).dropna()
    aligned.sort_index(inplace=True)
    return aligned


def log_report_summary(*, feature_summary: dict[str, Any], report: pd.DataFrame, ic_threshold: float) -> None:
    pass_count = int(report["passed"].sum())
    fail_count = int((~report["passed"]).sum())
    retained_union = int(report["retained"].sum()) if "retained" in report.columns else 0
    retained_ridge = int(report["retained_ridge"].sum()) if "retained_ridge" in report.columns else 0
    retained_xgboost = int(report["retained_xgboost"].sum()) if "retained_xgboost" in report.columns else 0
    retained_lightgbm = int(report["retained_lightgbm"].sum()) if "retained_lightgbm" in report.columns else 0
    logger.info(
        "IC screening summary: total_features={} stability_passed={} rejected={} retained_union={} ridge={} xgboost={} lightgbm={} feature_tickers={}",
        len(report),
        pass_count,
        fail_count,
        retained_union,
        retained_ridge,
        retained_xgboost,
        retained_lightgbm,
        feature_summary["ticker_count"],
    )

    top20 = report.sort_values(["retained", "stability_score", "feature_name"], ascending=[False, False, True]).head(20)
    logger.info("Top 20 features by stability score:")
    for row in top20.itertuples(index=False):
        logger.info(
            "  {} | domain={} | kind={} | signed_IC={:.6f} | RankIC={:.6f} | ICIR={} | sign_consistency={} | retained={} | n_obs={} | n_dates={} | n_tickers={}",
            row.feature_name,
            row.domain,
            row.feature_kind,
            row.signed_ic if pd.notna(row.signed_ic) else float("nan"),
            row.rank_ic if pd.notna(row.rank_ic) else float("nan"),
            format_metric(row.icir),
            format_metric(row.sign_consistency),
            bool(getattr(row, "retained", False)),
            row.n_obs,
            row.n_dates,
            row.n_tickers,
        )

    rejected = report.loc[~report["passed"], "feature_name"].tolist()
    logger.info("Rejected features after stability screening (|window IC| / sign-consistency) with threshold {}: {}", ic_threshold, rejected)


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"


def nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=float)))


def nanstd(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanstd(np.asarray(values, dtype=float)))


def write_parquet_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_path_for(output_path)
    if temp_path.exists():
        temp_path.unlink()
    frame.to_parquet(temp_path, index=False)
    temp_path.replace(output_path)


def write_csv_atomic(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_path_for(output_path)
    if temp_path.exists():
        temp_path.unlink()
    frame.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def write_json_atomic(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_path_for(output_path)
    if temp_path.exists():
        temp_path.unlink()
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temp_path.replace(output_path)


def temp_path_for(path: Path) -> Path:
    if path.suffix:
        return path.with_name(f"{path.stem}.tmp{path.suffix}")
    return path.with_name(f"{path.name}.tmp")


if __name__ == "__main__":
    raise SystemExit(main())
