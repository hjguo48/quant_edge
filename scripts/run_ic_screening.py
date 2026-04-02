from __future__ import annotations

import argparse
from functools import lru_cache
import gc
import json
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import sys
from typing import Any

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


FEATURE_START_DATE = date(2019, 7, 1)
FEATURE_END_DATE = date(2025, 6, 30)
AS_OF_DATE = date(2026, 3, 31)
LABEL_BUFFER_DAYS = 120
FEATURE_BATCH_SIZE = 25
FEATURE_PROGRESS_INTERVAL = 50
IC_THRESHOLD = 0.01
HORIZON_DAYS = 5

FEATURE_COLUMNS = ["ticker", "trade_date", "feature_name", "feature_value", "is_filled"]
LABEL_COLUMNS = ["ticker", "trade_date", "horizon", "forward_return", "excess_return"]
DATE_LEVEL_NAME = "trade_date"
TICKER_LEVEL_NAME = "ticker"

CANDIDATE_FEATURE_NAMES = (
    *TECHNICAL_FEATURE_NAMES,
    *FUNDAMENTAL_FEATURE_NAMES,
    *MACRO_FEATURE_NAMES,
    *COMPOSITE_FEATURE_NAMES,
)
FEATURE_DOMAIN_BY_NAME = {
    **{name: "technical" for name in TECHNICAL_FEATURE_NAMES},
    **{name: "fundamental" for name in FUNDAMENTAL_FEATURE_NAMES},
    **{name: "macro" for name in MACRO_FEATURE_NAMES},
    **{name: "composite" for name in COMPOSITE_FEATURE_NAMES},
}


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
        report_output_path=report_output_path,
        ic_threshold=args.ic_threshold,
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
    parser.add_argument("--progress-interval", type=int, default=FEATURE_PROGRESS_INTERVAL)
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS)
    parser.add_argument("--ic-threshold", type=float, default=IC_THRESHOLD)
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
    logger.info("installed runtime optimizations for fundamental PIT reuse, macro caching, and candidate-only preprocessing")


def preprocess_candidate_features(features_df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    prepared = preprocessing_module._prepare_feature_frame(features_df)
    forward_filled = preprocessing_module.forward_fill_features(prepared, max_days=90)
    winsorized = preprocessing_module.winsorize_features(forward_filled, z_threshold=5.0)
    normalized = preprocessing_module.rank_normalize_features(winsorized, method=method)
    finalized = normalized.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)
    logger.info(
        "preprocessed {} candidate feature rows into {} rows without missing flags",
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


def calculate_feature_snapshot_from_history(*, history: pd.DataFrame, price: float) -> dict[str, float]:
    features = {feature_name: np.nan for feature_name in fundamental_module.FUNDAMENTAL_FEATURE_NAMES}
    if history.empty:
        return features

    latest = history.iloc[-1]
    shares_outstanding = fundamental_module._latest_metric(history, "weighted_average_shares_outstanding")
    market_cap = fundamental_module._market_cap(price, shares_outstanding)
    equity = fundamental_module._safe_subtract(latest.get("total_assets"), latest.get("total_liabilities"))
    revenue_ttm = fundamental_module._ttm(history, "revenue")
    eps_ttm = fundamental_module._ttm(history, "eps")
    operating_cash_flow_ttm = fundamental_module._ttm(history, "operating_cash_flow")
    free_cash_flow_ttm = fundamental_module._free_cash_flow_ttm(history, operating_cash_flow_ttm)
    ebitda_ttm = fundamental_module._ttm(history, "ebitda")
    dividend_per_share = fundamental_module._first_non_nan(
        latest.get("annual_dividend"),
        latest.get("dividend_per_share"),
    )
    if pd.notna(dividend_per_share) and pd.isna(latest.get("annual_dividend")):
        dividend_per_share = dividend_per_share * 4
    cash = fundamental_module._first_non_nan(latest.get("cash"), latest.get("cash_and_cash_equivalents"))
    consensus_eps = fundamental_module._first_non_nan(latest.get("consensus_eps"), latest.get("eps_consensus"))
    total_debt = fundamental_module._first_non_nan(latest.get("total_debt"), latest.get("total_liabilities"))

    revenue_per_share = (
        revenue_ttm / shares_outstanding
        if pd.notna(revenue_ttm) and shares_outstanding is not None and shares_outstanding > 0
        else np.nan
    )

    features["pe_ratio"] = fundamental_module._safe_divide(price, eps_ttm)
    features["pb_ratio"] = fundamental_module._safe_divide(price, latest.get("book_value_per_share"))
    features["ps_ratio"] = fundamental_module._safe_divide(price, revenue_per_share)
    enterprise_value = fundamental_module._safe_add(market_cap, total_debt)
    enterprise_value = fundamental_module._safe_subtract(enterprise_value, cash)
    features["ev_ebitda"] = fundamental_module._safe_divide(enterprise_value, ebitda_ttm)
    features["fcf_yield"] = fundamental_module._safe_divide(free_cash_flow_ttm, market_cap)
    features["dividend_yield"] = fundamental_module._safe_divide(dividend_per_share, price)
    features["roe"] = fundamental_module._safe_divide(latest.get("net_income"), equity)
    features["roa"] = fundamental_module._safe_divide(latest.get("net_income"), latest.get("total_assets"))
    features["gross_margin"] = fundamental_module._safe_divide(latest.get("gross_profit"), latest.get("revenue"))
    features["operating_margin"] = fundamental_module._safe_divide(
        latest.get("operating_income"),
        latest.get("revenue"),
    )
    features["revenue_growth_yoy"] = fundamental_module._yoy_growth(history, "revenue")
    features["earnings_growth_yoy"] = fundamental_module._yoy_growth(history, "net_income")
    features["debt_to_equity"] = fundamental_module._safe_divide(total_debt, equity)
    features["current_ratio"] = fundamental_module._safe_divide(
        latest.get("current_assets"),
        latest.get("current_liabilities"),
    )
    eps_surprise_denom = abs(consensus_eps) if pd.notna(consensus_eps) else np.nan
    features["eps_surprise"] = (
        fundamental_module._safe_divide(latest.get("eps") - consensus_eps, eps_surprise_denom)
        if pd.notna(latest.get("eps")) and pd.notna(consensus_eps)
        else np.nan
    )
    return features


def load_raw_fundamentals_history(*, ticker: str, max_trade_date: date) -> pd.DataFrame:
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
            FundamentalsPIT.ticker == ticker.upper(),
            FundamentalsPIT.metric_name.in_(fundamental_module._PIT_METRIC_NAMES),
            FundamentalsPIT.knowledge_time <= cutoff,
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

    frame["metric_value"] = pd.to_numeric(frame["metric_value"], errors="coerce")
    frame["knowledge_time"] = pd.to_datetime(frame["knowledge_time"], utc=True)
    frame["event_time"] = pd.to_datetime(frame["event_time"]).dt.date
    return frame


def build_or_load_feature_cache(
    *,
    tickers: list[str],
    feature_start: date,
    feature_end: date,
    as_of: date,
    batch_size: int,
    progress_interval: int,
    feature_output_path: Path,
    batch_dir: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    batch_dir.mkdir(parents=True, exist_ok=True)
    feature_output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_manifest = build_manifest(
        tickers=tickers,
        feature_start=feature_start,
        feature_end=feature_end,
        as_of=as_of,
        batch_size=batch_size,
    )
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text())
        if existing_manifest != expected_manifest:
            raise RuntimeError(
                f"Existing manifest at {manifest_path} does not match current parameters. "
                "Delete the batch directory or rerun with matching settings.",
            )
    else:
        write_json_atomic(manifest_path, expected_manifest)

    batch_specs = build_batch_specs(tickers=tickers, batch_size=batch_size, batch_dir=batch_dir)
    pipeline = FeaturePipeline()
    completed_tickers = count_completed_tickers(batch_specs=batch_specs)

    if not feature_output_path.exists():
        for batch_index, batch_spec in enumerate(batch_specs, start=1):
            if batch_spec["path"].exists():
                logger.info(
                    "skipping existing feature batch {}/{} at {}",
                    batch_index,
                    len(batch_specs),
                    batch_spec["path"],
                )
                continue

            batch_tickers = batch_spec["tickers"]
            logger.info(
                "running feature batch {}/{} for {} tickers ({} -> {})",
                batch_index,
                len(batch_specs),
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
            write_parquet_atomic(filtered, batch_spec["path"])
            completed_tickers += len(batch_tickers)
            if completed_tickers % progress_interval == 0 or completed_tickers == len(tickers):
                logger.info("feature progress: processed {}/{} tickers", completed_tickers, len(tickers))
            del features
            del filtered
            gc.collect()

        combine_feature_batches(batch_specs=batch_specs, output_path=feature_output_path)
    else:
        logger.info("using existing feature cache at {}", feature_output_path)

    feature_summary = summarize_feature_batches(batch_specs=batch_specs)
    logger.info(
        "feature cache ready: {} rows across {} tickers and {} feature names",
        feature_summary["row_count"],
        feature_summary["ticker_count"],
        feature_summary["feature_count"],
    )
    return feature_summary


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


def summarize_feature_batches(*, batch_specs: list[dict[str, Any]]) -> dict[str, Any]:
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
    report_output_path: Path,
    ic_threshold: float,
) -> pd.DataFrame:
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_output_path.exists():
        logger.info("using existing IC report at {}", report_output_path)
        return pd.read_csv(report_output_path)

    label_series = build_label_series(labels)
    batch_paths = sorted(batch_dir.glob("*.parquet"))
    if not batch_paths:
        raise RuntimeError("No feature batch parquet files found for IC screening.")

    records: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(CANDIDATE_FEATURE_NAMES, start=1):
        logger.info(
            "computing IC metrics for feature {}/{}: {}",
            feature_index,
            len(CANDIDATE_FEATURE_NAMES),
            feature_name,
        )
        feature_series = load_feature_series(feature_name=feature_name, batch_paths=batch_paths)
        aligned = align_label_and_feature(label_series=label_series, feature_series=feature_series)
        if aligned.empty:
            ic_value = float("nan")
            rank_ic_value = float("nan")
            icir_value = float("nan")
            row_count = 0
            date_count = 0
            ticker_count = 0
        else:
            y_true = aligned["y_true"]
            y_pred = aligned["y_pred"]
            ic_value = information_coefficient(y_true=y_true, y_pred=y_pred)
            rank_ic_value = rank_information_coefficient(y_true=y_true, y_pred=y_pred)
            icir_value = icir(y_true=y_true, y_pred=y_pred)
            row_count = len(aligned)
            date_count = aligned.index.get_level_values(DATE_LEVEL_NAME).nunique()
            ticker_count = aligned.index.get_level_values(TICKER_LEVEL_NAME).nunique()

        abs_ic = abs(ic_value) if pd.notna(ic_value) else float("nan")
        records.append(
            {
                "feature_name": feature_name,
                "domain": FEATURE_DOMAIN_BY_NAME[feature_name],
                "ic": ic_value,
                "rank_ic": rank_ic_value,
                "icir": icir_value,
                "abs_ic": abs_ic,
                "n_obs": row_count,
                "n_dates": date_count,
                "n_tickers": ticker_count,
                "passed": bool(pd.notna(abs_ic) and abs_ic >= ic_threshold),
            },
        )
        gc.collect()

    report = pd.DataFrame(records).sort_values(["abs_ic", "feature_name"], ascending=[False, True]).reset_index(
        drop=True,
    )
    write_csv_atomic(report, report_output_path)
    logger.info("wrote IC screening report for {} features to {}", len(report), report_output_path)
    return report


def build_label_series(labels: pd.DataFrame) -> pd.Series:
    prepared = labels.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"]).dt.date
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["excess_return"] = pd.to_numeric(prepared["excess_return"], errors="coerce")
    series = prepared.set_index([DATE_LEVEL_NAME, TICKER_LEVEL_NAME])["excess_return"].sort_index()
    series.index = series.index.set_names([DATE_LEVEL_NAME, TICKER_LEVEL_NAME])
    return series


def load_feature_series(*, feature_name: str, batch_paths: list[Path]) -> pd.Series:
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
        return pd.Series(
            dtype=float,
            index=pd.MultiIndex.from_arrays([[], []], names=[DATE_LEVEL_NAME, TICKER_LEVEL_NAME]),
            name=feature_name,
        )
    return pd.concat(slices).sort_index()


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
    logger.info(
        "IC screening summary: total_features={} passed={} rejected={} feature_tickers={}",
        len(report),
        pass_count,
        fail_count,
        feature_summary["ticker_count"],
    )

    top20 = report.sort_values(["abs_ic", "feature_name"], ascending=[False, True]).head(20)
    logger.info("Top 20 features by |IC|:")
    for row in top20.itertuples(index=False):
        logger.info(
            "  {} | domain={} | IC={:.6f} | RankIC={:.6f} | ICIR={} | n_obs={} | n_dates={} | n_tickers={}",
            row.feature_name,
            row.domain,
            row.ic if pd.notna(row.ic) else float("nan"),
            row.rank_ic if pd.notna(row.rank_ic) else float("nan"),
            format_metric(row.icir),
            row.n_obs,
            row.n_dates,
            row.n_tickers,
        )

    rejected = report.loc[~report["passed"], "feature_name"].tolist()
    logger.info("Rejected features with |IC| < {}: {}", ic_threshold, rejected)


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"


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
