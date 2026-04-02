from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
import subprocess
import sys
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import install_runtime_optimizations, write_json_atomic, write_parquet_atomic
from src.data.db.pit import get_prices_pit
from src.labels.forward_returns import compute_forward_returns
from src.mlflow_config import get_mlflow_tracking_uri
from src.models.baseline import DEFAULT_ALPHA_GRID, RidgeBaselineModel
from src.models.evaluation import (
    EvaluationSummary,
    information_coefficient_series,
    rank_information_coefficient_series,
)
from src.models.experiment import ExperimentTracker, ValidationWindowConfig, run_single_window_validation

TRAIN_START = date(2018, 1, 1)
TRAIN_END = date(2020, 12, 31)
VALIDATION_START = date(2021, 1, 1)
VALIDATION_END = date(2021, 6, 30)
TEST_START = date(2021, 7, 1)
TEST_END = date(2021, 12, 31)
AS_OF_DATE = date(2026, 3, 31)
HORIZON_DAYS = 5
REBALANCE_WEEKDAY = 4
PASS_IC_THRESHOLD = 0.01
BATCH_SIZE = 50
DEFAULT_ALL_FEATURES_PATH = "data/features/all_features.parquet"
DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_v2.csv"
DEFAULT_FEATURE_MATRIX_PATH = "data/features/window1_feature_matrix.parquet"
DEFAULT_LABEL_CACHE_PATH = "data/labels/window1_forward_returns_5d.parquet"
DEFAULT_REPORT_PATH = "data/reports/single_window_validation.json"
BENCHMARK_TICKER = "SPY"
CACHED_FEATURE_START = date(2019, 7, 1)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    config = ValidationWindowConfig(
        train_start=parse_date(args.train_start),
        train_end=parse_date(args.train_end),
        validation_start=parse_date(args.validation_start),
        validation_end=parse_date(args.validation_end),
        test_start=parse_date(args.test_start),
        test_end=parse_date(args.test_end),
        rebalance_weekday=args.rebalance_weekday,
        target_horizon=f"{args.horizon}D",
        pass_ic_threshold=args.pass_ic_threshold,
        refit_on_train_plus_validation=True,
    )
    as_of = parse_date(args.as_of)

    if as_of < config.test_end:
        raise ValueError("as_of must be on or after test_end.")

    retained_features = load_retained_features(REPO_ROOT / args.ic_report_path)
    tickers = load_universe_tickers(REPO_ROOT / args.label_source_path)
    logger.info("loaded {} retained features and {} tickers", len(retained_features), len(tickers))

    feature_matrix, feature_diagnostics = build_or_load_feature_matrix(
        tickers=tickers,
        retained_features=retained_features,
        config=config,
        as_of=as_of,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_path=REPO_ROOT / args.feature_matrix_path,
        batch_size=args.batch_size,
    )
    label_series, label_diagnostics = build_or_load_label_series(
        tickers=tickers,
        config=config,
        as_of=as_of,
        label_cache_path=REPO_ROOT / args.label_cache_path,
        horizon=args.horizon,
        benchmark_ticker=args.benchmark_ticker,
    )

    aligned_X, aligned_y = align_panel(feature_matrix, label_series)
    logger.info(
        "aligned panel rows={} dates={} tickers={}",
        len(aligned_X),
        aligned_X.index.get_level_values("trade_date").nunique(),
        aligned_X.index.get_level_values("ticker").nunique(),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    tracked_result = run_tracked_validation(
        X=aligned_X,
        y=aligned_y,
        config=config,
        timestamp=timestamp,
    )

    analysis = compute_split_metrics(
        X=aligned_X,
        y=aligned_y,
        config=config,
        alpha_grid=DEFAULT_ALPHA_GRID,
    )
    validate_result_consistency(tracked_result=tracked_result, analysis=analysis)

    report = build_report_payload(
        args=args,
        config=config,
        as_of=as_of,
        retained_features=retained_features,
        feature_diagnostics=feature_diagnostics,
        label_diagnostics=label_diagnostics,
        aligned_X=aligned_X,
        aligned_y=aligned_y,
        tracked_result=tracked_result,
        analysis=analysis,
    )
    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved single-window validation report to {}", report_path)

    log_summary(analysis=analysis, tracked_result=tracked_result, report_path=report_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Window 1 Ridge baseline validation on retained IC-screened features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-start", default=TRAIN_START.isoformat())
    parser.add_argument("--train-end", default=TRAIN_END.isoformat())
    parser.add_argument("--validation-start", default=VALIDATION_START.isoformat())
    parser.add_argument("--validation-end", default=VALIDATION_END.isoformat())
    parser.add_argument("--test-start", default=TEST_START.isoformat())
    parser.add_argument("--test-end", default=TEST_END.isoformat())
    parser.add_argument("--as-of", default=AS_OF_DATE.isoformat())
    parser.add_argument("--horizon", type=int, default=HORIZON_DAYS)
    parser.add_argument("--rebalance-weekday", type=int, default=REBALANCE_WEEKDAY)
    parser.add_argument("--pass-ic-threshold", type=float, default=PASS_IC_THRESHOLD)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--label-source-path", default="data/labels/forward_returns_5d.parquet")
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
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


def load_retained_features(report_path: Path) -> list[str]:
    report = pd.read_csv(report_path)
    retained = report.loc[report["retained"].astype(bool), "feature_name"].astype(str).tolist()
    if not retained:
        raise RuntimeError(f"No retained features found in {report_path}.")
    return retained


def load_universe_tickers(label_source_path: Path) -> list[str]:
    if label_source_path.exists():
        tickers = (
            pd.read_parquet(label_source_path, columns=["ticker"])["ticker"]
            .astype(str)
            .str.upper()
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if tickers:
            return tickers

    raise RuntimeError(f"Ticker source parquet not found or empty: {label_source_path}")


def build_or_load_feature_matrix(
    *,
    tickers: list[str],
    retained_features: list[str],
    config: ValidationWindowConfig,
    as_of: date,
    all_features_path: Path,
    feature_matrix_path: Path,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if feature_matrix_path.exists():
        matrix = pd.read_parquet(feature_matrix_path)
        matrix = restore_feature_matrix_index(matrix)
        diagnostics = summarize_feature_matrix(matrix, retained_features, source="cache")
        logger.info("loaded cached feature matrix from {}", feature_matrix_path)
        return matrix, diagnostics

    cached_matrix = pd.DataFrame(columns=retained_features)
    if all_features_path.exists():
        cached_long = pd.read_parquet(
            all_features_path,
            columns=["ticker", "trade_date", "feature_name", "feature_value"],
            filters=[
                ("trade_date", ">=", max(config.train_start, CACHED_FEATURE_START)),
                ("trade_date", "<=", config.test_end),
                ("feature_name", "in", retained_features),
            ],
        )
        cached_matrix = long_to_feature_matrix(cached_long, retained_features)
        logger.info(
            "loaded cached feature slice rows={} dates={} tickers={}",
            len(cached_long),
            cached_matrix.index.get_level_values("trade_date").nunique() if not cached_matrix.empty else 0,
            cached_matrix.index.get_level_values("ticker").nunique() if not cached_matrix.empty else 0,
        )

    gap_matrix = pd.DataFrame(columns=retained_features)
    if config.train_start < CACHED_FEATURE_START:
        gap_end = min(config.test_end, CACHED_FEATURE_START - timedelta(days=1))
        install_runtime_optimizations()
        gap_matrix = compute_feature_gap_matrix(
            tickers=tickers,
            retained_features=retained_features,
            start_date=config.train_start,
            end_date=gap_end,
            as_of=as_of,
            batch_size=batch_size,
        )

    matrix = (
        pd.concat([gap_matrix, cached_matrix])
        .sort_index()
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
    )
    matrix = matrix.reindex(columns=retained_features)
    diagnostics = summarize_feature_matrix(matrix, retained_features, source="fresh")
    filled = fill_feature_matrix(matrix)
    diagnostics["missing_cells_after_fill"] = int(filled.isna().sum().sum())
    diagnostics["rows_with_any_missing_after_fill"] = int(filled.isna().any(axis=1).sum())

    persisted = feature_matrix_to_frame(filled)
    write_parquet_atomic(persisted, feature_matrix_path)
    logger.info("saved feature matrix cache to {}", feature_matrix_path)
    return filled, diagnostics


def compute_feature_gap_matrix(
    *,
    tickers: list[str],
    retained_features: list[str],
    start_date: date,
    end_date: date,
    as_of: date,
    batch_size: int,
) -> pd.DataFrame:
    from src.features.pipeline import FeaturePipeline

    pipeline = FeaturePipeline()
    matrices: list[pd.DataFrame] = []
    total = len(tickers)

    for offset in range(0, total, batch_size):
        batch = tickers[offset : offset + batch_size]
        logger.info(
            "computing feature gap batch {}-{} of {} tickers for {} -> {}",
            offset + 1,
            min(offset + len(batch), total),
            total,
            start_date,
            end_date,
        )
        batch_long = pipeline.run(
            tickers=batch,
            start_date=start_date,
            end_date=end_date,
            as_of=as_of,
        )
        batch_matrix = long_to_feature_matrix(
            batch_long.loc[batch_long["feature_name"].isin(retained_features)].copy(),
            retained_features,
        )
        matrices.append(batch_matrix)

    if not matrices:
        return pd.DataFrame(columns=retained_features)

    return pd.concat(matrices).sort_index()


def long_to_feature_matrix(features_long: pd.DataFrame, retained_features: list[str]) -> pd.DataFrame:
    if features_long.empty:
        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=["trade_date", "ticker"])
        return pd.DataFrame(index=index, columns=retained_features, dtype=float)

    prepared = features_long.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
    prepared["feature_name"] = prepared["feature_name"].astype(str)
    prepared["feature_value"] = pd.to_numeric(prepared["feature_value"], errors="coerce")
    prepared.sort_values(["trade_date", "ticker", "feature_name"], inplace=True)
    prepared.drop_duplicates(["trade_date", "ticker", "feature_name"], keep="last", inplace=True)

    matrix = (
        prepared.set_index(["trade_date", "ticker", "feature_name"])["feature_value"]
        .unstack("feature_name")
        .sort_index()
    )
    matrix = matrix.reindex(columns=retained_features)
    matrix.index = matrix.index.set_names(["trade_date", "ticker"])
    return matrix


def summarize_feature_matrix(
    matrix: pd.DataFrame,
    retained_features: list[str],
    *,
    source: str,
) -> dict[str, Any]:
    if matrix.empty:
        raise RuntimeError("Feature matrix is empty.")

    missing_by_feature = (
        matrix.isna().sum().sort_values(ascending=False).head(15).astype(int).to_dict()
    )
    return {
        "source": source,
        "rows": int(len(matrix)),
        "dates": int(matrix.index.get_level_values("trade_date").nunique()),
        "tickers": int(matrix.index.get_level_values("ticker").nunique()),
        "min_date": matrix.index.get_level_values("trade_date").min().date().isoformat(),
        "max_date": matrix.index.get_level_values("trade_date").max().date().isoformat(),
        "feature_count": int(len(retained_features)),
        "missing_cells_before_fill": int(matrix.isna().sum().sum()),
        "rows_with_any_missing_before_fill": int(matrix.isna().any(axis=1).sum()),
        "top_missing_features_before_fill": missing_by_feature,
    }


def fill_feature_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    filled = matrix.replace([np.inf, -np.inf], np.nan).sort_index().copy()
    cross_section_medians = filled.groupby(level="trade_date").transform("median")
    filled = filled.fillna(cross_section_medians).fillna(0.5)
    if filled.isna().any(axis=None):
        raise RuntimeError("Feature matrix still contains NaN values after filling.")
    return filled.astype(float)


def feature_matrix_to_frame(matrix: pd.DataFrame) -> pd.DataFrame:
    frame = matrix.reset_index()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    return frame


def restore_feature_matrix_index(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    matrix = prepared.set_index(["trade_date", "ticker"]).sort_index()
    return matrix


def build_or_load_label_series(
    *,
    tickers: list[str],
    config: ValidationWindowConfig,
    as_of: date,
    label_cache_path: Path,
    horizon: int,
    benchmark_ticker: str,
) -> tuple[pd.Series, dict[str, Any]]:
    if label_cache_path.exists():
        labels = pd.read_parquet(label_cache_path)
        logger.info("loaded cached label parquet from {}", label_cache_path)
    else:
        label_price_end = min(config.test_end + timedelta(days=max(horizon * 5, 35)), as_of)
        price_tickers = list(dict.fromkeys([*tickers, benchmark_ticker.upper()]))
        prices = get_prices_pit(
            tickers=price_tickers,
            start_date=config.train_start,
            end_date=label_price_end,
            as_of=_as_of_datetime(as_of),
        )
        if prices.empty:
            raise RuntimeError("No PIT prices available for label computation.")
        labels = compute_forward_returns(
            prices_df=prices,
            horizons=(horizon,),
            benchmark_ticker=benchmark_ticker,
        )
        labels = labels.loc[
            (labels["horizon"] == horizon)
            & (labels["ticker"].astype(str).str.upper() != benchmark_ticker.upper())
            & (pd.to_datetime(labels["trade_date"]) >= pd.Timestamp(config.train_start))
            & (pd.to_datetime(labels["trade_date"]) <= pd.Timestamp(config.test_end))
        ].copy()
        write_parquet_atomic(labels, label_cache_path)
        logger.info("saved label cache to {}", label_cache_path)

    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    series = (
        labels.set_index(["trade_date", "ticker"])["excess_return"]
        .sort_index()
        .dropna()
    )
    diagnostics = {
        "rows": int(len(series)),
        "dates": int(series.index.get_level_values("trade_date").nunique()),
        "tickers": int(series.index.get_level_values("ticker").nunique()),
        "min_date": series.index.get_level_values("trade_date").min().date().isoformat(),
        "max_date": series.index.get_level_values("trade_date").max().date().isoformat(),
        "horizon_days": int(horizon),
        "benchmark_ticker": benchmark_ticker.upper(),
    }
    return series, diagnostics


def align_panel(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    aligned_index = X.index.intersection(y.index)
    if aligned_index.empty:
        raise RuntimeError("No aligned observations between features and labels.")
    return X.loc[aligned_index].sort_index(), y.loc[aligned_index].sort_index()


def compute_split_metrics(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    config: ValidationWindowConfig,
    alpha_grid: tuple[float, ...],
) -> dict[str, Any]:
    train_X, train_y = slice_window(X=X, y=y, start_date=config.train_start, end_date=config.train_end, rebalance_weekday=config.rebalance_weekday)
    validation_X, validation_y = slice_window(
        X=X,
        y=y,
        start_date=config.validation_start,
        end_date=config.validation_end,
        rebalance_weekday=config.rebalance_weekday,
    )
    test_X, test_y = slice_window(
        X=X,
        y=y,
        start_date=config.test_start,
        end_date=config.test_end,
        rebalance_weekday=config.rebalance_weekday,
    )

    model = RidgeBaselineModel(alpha_grid=alpha_grid)
    selection = model.select_alpha(train_X, train_y, validation_X, validation_y)
    model.train(train_X, train_y)

    train_predictions = model.predict(train_X)
    validation_predictions = model.predict(validation_X)
    train_metrics = model.evaluate(train_y, train_predictions)
    validation_metrics = model.evaluate(validation_y, validation_predictions)

    final_train_X = train_X
    final_train_y = train_y
    if config.refit_on_train_plus_validation:
        final_train_X = pd.concat([train_X, validation_X]).sort_index()
        final_train_y = pd.concat([train_y, validation_y]).sort_index()
    model.train(final_train_X, final_train_y)
    test_predictions = model.predict(test_X)
    test_metrics = model.evaluate(test_y, test_predictions)

    split_predictions = {
        "train": train_predictions,
        "validation": validation_predictions,
        "test": test_predictions,
    }
    split_targets = {
        "train": train_y,
        "validation": validation_y,
        "test": test_y,
    }
    split_features = {
        "train": train_X,
        "validation": validation_X,
        "test": test_X,
    }
    split_metrics = {
        "train": train_metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    return {
        "selection": {
            "best_hyperparams": float(selection.best_hyperparams),
            "best_ic": float(selection.best_ic),
            "scores_by_alpha": {str(alpha): float(score) for alpha, score in selection.scores_by_alpha.items()},
        },
        "metrics": {name: metrics_to_dict(summary) for name, summary in split_metrics.items()},
        "window_rows": {
            name: {
                "rows": int(len(split_features[name])),
                "dates": int(split_features[name].index.get_level_values("trade_date").nunique()),
                "tickers": int(split_features[name].index.get_level_values("ticker").nunique()),
            }
            for name in split_features
        },
        "series": {
            name: build_split_series(y_true=split_targets[name], y_pred=split_predictions[name])
            for name in split_predictions
        },
    }


def run_tracked_validation(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    config: ValidationWindowConfig,
    timestamp: str,
) -> Any:
    fallback_tracking_uri = get_mlflow_tracking_uri()
    tracker_attempts = (
        ExperimentTracker(),
        ExperimentTracker(tracking_uri=fallback_tracking_uri),
    )
    last_exception: Exception | None = None

    for tracker in tracker_attempts:
        model = RidgeBaselineModel(alpha_grid=DEFAULT_ALPHA_GRID)
        try:
            result = run_single_window_validation(
                model=model,
                X=X,
                y=y,
                config=config,
                tracker=tracker,
                timestamp=timestamp,
            )
            if tracker.tracking_uri:
                logger.warning(
                    "MLflow logging used fallback tracking URI {} after configured tracking failed",
                    tracker.tracking_uri,
                )
            return result
        except PermissionError as exc:
            last_exception = exc
            if tracker.tracking_uri:
                raise
            logger.warning(
                "configured MLflow artifact store is not writable ({}); retrying with local tracking URI {}",
                exc,
                fallback_tracking_uri,
            )
        except Exception as exc:
            last_exception = exc
            if tracker.tracking_uri or not is_mlflow_artifact_permission_error(exc):
                raise
            logger.warning(
                "configured MLflow logging failed with artifact permission error; retrying with local tracking URI {}",
                fallback_tracking_uri,
            )

    if last_exception is None:
        raise RuntimeError("MLflow logging failed without an explicit exception.")
    raise last_exception


def slice_window(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    start_date: date,
    end_date: date,
    rebalance_weekday: int,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_dates = pd.to_datetime(pd.Index(X.index.get_level_values("trade_date")))
    target_dates = pd.to_datetime(pd.Index(y.index.get_level_values("trade_date")))

    feature_mask = (
        (feature_dates >= pd.Timestamp(start_date))
        & (feature_dates <= pd.Timestamp(end_date))
        & (feature_dates.weekday == rebalance_weekday)
    )
    target_mask = (
        (target_dates >= pd.Timestamp(start_date))
        & (target_dates <= pd.Timestamp(end_date))
        & (target_dates.weekday == rebalance_weekday)
    )

    sliced_X = X.loc[feature_mask].sort_index()
    sliced_y = y.loc[target_mask].sort_index()
    aligned_index = sliced_X.index.intersection(sliced_y.index)
    if aligned_index.empty:
        raise RuntimeError(f"No aligned rows in window {start_date} -> {end_date}.")
    return sliced_X.loc[aligned_index], sliced_y.loc[aligned_index]


def build_split_series(*, y_true: pd.Series, y_pred: pd.Series) -> dict[str, list[dict[str, Any]]]:
    ic_series = information_coefficient_series(y_true=y_true, y_pred=y_pred)
    rank_ic_series = rank_information_coefficient_series(y_true=y_true, y_pred=y_pred)
    returns_frame = per_date_return_frame(y_true=y_true, y_pred=y_pred)

    return {
        "ic_series": series_to_records(ic_series),
        "rank_ic_series": series_to_records(rank_ic_series),
        "return_series": dataframe_to_records(returns_frame),
    }


def is_mlflow_artifact_permission_error(exc: Exception) -> bool:
    message = str(exc)
    return "Permission denied: '/mlflow'" in message or "log_artifact" in message


def per_date_return_frame(*, y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    aligned = (
        pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1, join="inner")
        .dropna()
        .sort_index()
    )
    records: list[dict[str, Any]] = []

    for trade_date, frame in aligned.groupby(level="trade_date", sort=True):
        cross_section = frame.droplevel("trade_date")
        decile_size = max(1, int(np.ceil(len(cross_section) * 0.10)))
        top = cross_section.nlargest(decile_size, "y_pred")["y_true"].mean()
        bottom = cross_section.nsmallest(decile_size, "y_pred")["y_true"].mean()
        records.append(
            {
                "trade_date": pd.Timestamp(trade_date),
                "cross_section_size": int(len(cross_section)),
                "top_decile_return": float(top),
                "long_short_return": float(top - bottom),
            },
        )

    frame = pd.DataFrame(records).sort_values("trade_date").reset_index(drop=True)
    if frame.empty:
        return frame

    frame["top_decile_cum_return"] = (1.0 + frame["top_decile_return"]).cumprod() - 1.0
    frame["long_short_cum_return"] = (1.0 + frame["long_short_return"]).cumprod() - 1.0
    return frame


def metrics_to_dict(summary: EvaluationSummary) -> dict[str, float]:
    payload = asdict(summary)
    return {key: float(value) if pd.notna(value) else float("nan") for key, value in payload.items()}


def series_to_records(series: pd.Series) -> list[dict[str, Any]]:
    if series.empty:
        return []
    records = []
    for index_value, value in series.items():
        timestamp = pd.Timestamp(index_value)
        records.append({"trade_date": timestamp.date().isoformat(), "value": float(value)})
    return records


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "trade_date": pd.Timestamp(row.trade_date).date().isoformat(),
                "cross_section_size": int(row.cross_section_size),
                "top_decile_return": float(row.top_decile_return),
                "long_short_return": float(row.long_short_return),
                "top_decile_cum_return": float(row.top_decile_cum_return),
                "long_short_cum_return": float(row.long_short_cum_return),
            },
        )
    return records


def build_report_payload(
    *,
    args: argparse.Namespace,
    config: ValidationWindowConfig,
    as_of: date,
    retained_features: list[str],
    feature_diagnostics: dict[str, Any],
    label_diagnostics: dict[str, Any],
    aligned_X: pd.DataFrame,
    aligned_y: pd.Series,
    tracked_result: Any,
    analysis: dict[str, Any],
) -> dict[str, Any]:
    validation_ic = float(analysis["metrics"]["validation"]["ic"])
    test_ic = float(analysis["metrics"]["test"]["ic"])
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_single_window_validation.py",
        "config": {
            "train_start": config.train_start.isoformat(),
            "train_end": config.train_end.isoformat(),
            "validation_start": config.validation_start.isoformat(),
            "validation_end": config.validation_end.isoformat(),
            "test_start": config.test_start.isoformat(),
            "test_end": config.test_end.isoformat(),
            "rebalance_weekday": config.rebalance_weekday,
            "target_horizon": config.target_horizon,
            "pass_ic_threshold": config.pass_ic_threshold,
            "as_of": as_of.isoformat(),
            "benchmark_ticker": args.benchmark_ticker.upper(),
        },
        "inputs": {
            "ic_report_path": str(REPO_ROOT / args.ic_report_path),
            "feature_source_path": str(REPO_ROOT / args.all_features_path),
            "feature_matrix_cache_path": str(REPO_ROOT / args.feature_matrix_path),
            "label_cache_path": str(REPO_ROOT / args.label_cache_path),
            "report_path": str(REPO_ROOT / args.report_path),
            "retained_feature_count": len(retained_features),
            "retained_features": retained_features,
            "feature_fill_strategy": "cross-sectional median by trade_date, then 0.5 fallback",
        },
        "data_summary": {
            "feature_matrix": feature_diagnostics,
            "labels": label_diagnostics,
            "aligned_rows": int(len(aligned_X)),
            "aligned_dates": int(aligned_X.index.get_level_values("trade_date").nunique()),
            "aligned_tickers": int(aligned_X.index.get_level_values("ticker").nunique()),
            "aligned_min_date": aligned_X.index.get_level_values("trade_date").min().date().isoformat(),
            "aligned_max_date": aligned_X.index.get_level_values("trade_date").max().date().isoformat(),
            "label_rows": int(len(aligned_y)),
        },
        "alpha_selection": analysis["selection"],
        "metrics": analysis["metrics"],
        "window_rows": analysis["window_rows"],
        "gate": {
            "validation_ic": validation_ic,
            "validation_passed": bool(pd.notna(validation_ic) and validation_ic > config.pass_ic_threshold),
            "test_ic": test_ic,
            "test_passed": bool(pd.notna(test_ic) and test_ic > config.pass_ic_threshold),
            "framework_passed": bool(tracked_result.passed),
            "note": "framework_passed follows src.models.experiment.run_single_window_validation and is based on test IC; validation_passed is reported separately for the task gate.",
        },
        "mlflow": {
            "logged": tracked_result.logged_run is not None,
            "tracking_uri": tracked_result.logged_run.tracking_uri if tracked_result.logged_run else None,
            "experiment_name": tracked_result.logged_run.experiment_name if tracked_result.logged_run else None,
            "experiment_id": tracked_result.logged_run.experiment_id if tracked_result.logged_run else None,
            "run_id": tracked_result.logged_run.run_id if tracked_result.logged_run else None,
        },
        "series": analysis["series"],
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.floating):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if pd.isna(value) or np.isinf(value):
            return None
        return value
    if isinstance(value, np.integer):
        return int(value)
    return value


def validate_result_consistency(*, tracked_result: Any, analysis: dict[str, Any]) -> None:
    validation_ic = float(analysis["metrics"]["validation"]["ic"])
    test_ic = float(analysis["metrics"]["test"]["ic"])

    if not np.isclose(validation_ic, tracked_result.validation_metrics.ic, equal_nan=True):
        raise RuntimeError("Validation IC mismatch between tracked run and analysis model.")
    if not np.isclose(test_ic, tracked_result.test_metrics.ic, equal_nan=True):
        raise RuntimeError("Test IC mismatch between tracked run and analysis model.")


def current_git_branch() -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _as_of_datetime(as_of: date) -> datetime:
    return datetime.combine(as_of, time.max, tzinfo=timezone.utc)


def log_summary(*, analysis: dict[str, Any], tracked_result: Any, report_path: Path) -> None:
    for split_name in ("train", "validation", "test"):
        metrics = analysis["metrics"][split_name]
        logger.info(
            "{} | IC={:.6f} RankIC={:.6f} ICIR={} HitRate={:.6f} TopDecile={:.6f} LongShort={:.6f}",
            split_name,
            metrics["ic"],
            metrics["rank_ic"],
            format_metric(metrics["icir"]),
            metrics["hit_rate"],
            metrics["top_decile_return"],
            metrics["long_short_return"],
        )

    logger.info(
        "gate validation_passed={} framework_passed={} mlflow_run_id={}",
        analysis["metrics"]["validation"]["ic"] > PASS_IC_THRESHOLD,
        tracked_result.passed,
        tracked_result.logged_run.run_id if tracked_result.logged_run else "none",
    )
    logger.info("report available at {}", report_path)


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
