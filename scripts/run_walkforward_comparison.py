from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
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
from scripts.run_single_window_validation import (
    feature_matrix_to_frame,
    fill_feature_matrix,
    long_to_feature_matrix,
    restore_feature_matrix_index,
)
from src.data.db.pit import get_prices_pit
from src.labels.forward_returns import compute_forward_returns
from src.models.baseline import DEFAULT_ALPHA_GRID, RidgeBaselineModel
from src.models.evaluation import EvaluationSummary, evaluate_predictions
from src.models.experiment import ExperimentTracker, LoggedModelRun, ValidationWindowConfig
from src.models.tree import LightGBMModel, XGBoostModel

TARGET_HORIZON_DAYS = 60
REBALANCE_WEEKDAY = 4
AS_OF_DATE = date(2026, 3, 31)
BENCHMARK_TICKER = "SPY"
DEFAULT_ALL_FEATURES_PATH = "data/features/all_features.parquet"
DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_60d.csv"
DEFAULT_LABEL_CACHE_PATH = "data/labels/forward_returns_60d.parquet"
DEFAULT_FEATURE_MATRIX_CACHE_PATH = "data/features/walkforward_feature_matrix_60d.parquet"
DEFAULT_REPORT_PATH = "data/reports/walkforward_comparison_60d.json"
PASS_IC_THRESHOLD = 0.01
TREE_SEARCH_N_ITER = 100
TREE_N_JOBS = 4
RANDOM_STATE = 42
LABEL_BUFFER_DAYS = 120


@dataclass(frozen=True)
class WindowSpec:
    window_id: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date
    test_start: date
    test_end: date

    def to_validation_config(self, *, rebalance_weekday: int, horizon_days: int) -> ValidationWindowConfig:
        return ValidationWindowConfig(
            train_start=self.train_start,
            train_end=self.train_end,
            validation_start=self.validation_start,
            validation_end=self.validation_end,
            test_start=self.test_start,
            test_end=self.test_end,
            rebalance_weekday=rebalance_weekday,
            target_horizon=f"{horizon_days}D",
            pass_ic_threshold=PASS_IC_THRESHOLD,
            refit_on_train_plus_validation=True,
        )


WINDOWS: tuple[WindowSpec, ...] = (
    WindowSpec("W1", date(2016, 3, 1), date(2019, 2, 28), date(2019, 3, 1), date(2019, 8, 31), date(2019, 9, 1), date(2020, 2, 29)),
    WindowSpec("W2", date(2016, 9, 1), date(2019, 8, 31), date(2019, 9, 1), date(2020, 2, 29), date(2020, 3, 1), date(2020, 8, 31)),
    WindowSpec("W3", date(2017, 3, 1), date(2020, 2, 29), date(2020, 3, 1), date(2020, 8, 31), date(2020, 9, 1), date(2021, 2, 28)),
    WindowSpec("W4", date(2017, 9, 1), date(2020, 8, 31), date(2020, 9, 1), date(2021, 2, 28), date(2021, 3, 1), date(2021, 8, 31)),
    WindowSpec("W5", date(2018, 3, 1), date(2021, 2, 28), date(2021, 3, 1), date(2021, 8, 31), date(2021, 9, 1), date(2022, 2, 28)),
    WindowSpec("W6", date(2018, 9, 1), date(2021, 8, 31), date(2021, 9, 1), date(2022, 2, 28), date(2022, 3, 1), date(2022, 8, 31)),
    WindowSpec("W7", date(2019, 3, 1), date(2022, 2, 28), date(2022, 3, 1), date(2022, 8, 31), date(2022, 9, 1), date(2023, 2, 28)),
    WindowSpec("W8", date(2019, 9, 1), date(2022, 8, 31), date(2022, 9, 1), date(2023, 2, 28), date(2023, 3, 1), date(2023, 8, 31)),
    WindowSpec("W9", date(2020, 3, 1), date(2023, 2, 28), date(2023, 3, 1), date(2023, 8, 31), date(2023, 9, 1), date(2024, 2, 29)),
    WindowSpec("W10", date(2020, 9, 1), date(2023, 8, 31), date(2023, 9, 1), date(2024, 2, 29), date(2024, 3, 1), date(2024, 8, 31)),
    WindowSpec("W11", date(2021, 3, 1), date(2024, 2, 29), date(2024, 3, 1), date(2024, 8, 31), date(2024, 9, 1), date(2025, 2, 28)),
)

SUPPORTED_MODELS = ("ridge", "xgboost", "lightgbm")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    install_runtime_optimizations()

    windows = select_windows(limit=args.window_limit)
    if not windows:
        raise RuntimeError("No windows selected.")

    as_of_date = parse_date(args.as_of)
    retained_features = load_retained_features(REPO_ROOT / args.ic_report_path)
    model_names = parse_model_names(args.models)
    logger.info(
        "walk-forward comparison configured for {} windows, {} models, {} retained 60D features",
        len(windows),
        len(model_names),
        len(retained_features),
    )

    global_start = min(window.train_start for window in windows)
    global_end = max(window.test_end for window in windows)

    feature_matrix = build_or_load_feature_matrix(
        all_features_path=REPO_ROOT / args.all_features_path,
        cache_path=REPO_ROOT / args.feature_matrix_cache_path,
        retained_features=retained_features,
        start_date=global_start,
        end_date=global_end,
        rebalance_weekday=args.rebalance_weekday,
    )
    labels = build_or_load_label_series(
        label_cache_path=REPO_ROOT / args.label_cache_path,
        tickers=sorted(feature_matrix.index.get_level_values("ticker").unique()),
        start_date=global_start,
        end_date=global_end,
        as_of=as_of_date,
        horizon=args.horizon,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=args.benchmark_ticker,
        rebalance_weekday=args.rebalance_weekday,
    )
    aligned_X, aligned_y = align_panel(feature_matrix, labels)
    logger.info(
        "aligned Friday panel rows={} dates={} tickers={} features={}",
        len(aligned_X),
        aligned_X.index.get_level_values("trade_date").nunique(),
        aligned_X.index.get_level_values("ticker").nunique(),
        aligned_X.shape[1],
    )

    tracker = None if args.disable_mlflow else ExperimentTracker()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    report_payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of_date.isoformat(),
        "target_horizon": f"{args.horizon}D",
        "retained_feature_count": len(retained_features),
        "retained_features": retained_features,
        "windows": [],
        "summary": {},
    }

    for position, window in enumerate(windows, start=1):
        logger.info("running window {}/{} {} (train {} -> {}, val {} -> {}, test {} -> {})",
                    position,
                    len(windows),
                    window.window_id,
                    window.train_start,
                    window.train_end,
                    window.validation_start,
                    window.validation_end,
                    window.test_start,
                    window.test_end)
        config = window.to_validation_config(
            rebalance_weekday=args.rebalance_weekday,
            horizon_days=args.horizon,
        )
        window_result = run_window(
            window=window,
            config=config,
            X=aligned_X,
            y=aligned_y,
            model_names=model_names,
            tracker=tracker,
            timestamp=timestamp,
            alpha_grid=DEFAULT_ALPHA_GRID,
            search_n_iter=args.search_n_iter,
            tree_n_jobs=args.tree_n_jobs,
            random_state=args.random_state,
        )
        report_payload["windows"].append(window_result)
        report_payload["summary"] = summarize_results(report_payload["windows"], model_names)
        write_json_atomic(REPO_ROOT / args.report_path, json_safe(report_payload))
        logger.info("saved partial walk-forward report to {}", REPO_ROOT / args.report_path)

    logger.info("completed {} windows across {} models", len(windows), len(model_names))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 60D multi-window walk-forward comparison across ridge, xgboost, and lightgbm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--feature-matrix-cache-path", default=DEFAULT_FEATURE_MATRIX_CACHE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--as-of", default=AS_OF_DATE.isoformat())
    parser.add_argument("--horizon", type=int, default=TARGET_HORIZON_DAYS)
    parser.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--rebalance-weekday", type=int, default=REBALANCE_WEEKDAY)
    parser.add_argument("--search-n-iter", type=int, default=TREE_SEARCH_N_ITER)
    parser.add_argument("--tree-n-jobs", type=int, default=TREE_N_JOBS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--models", default="ridge,xgboost,lightgbm")
    parser.add_argument("--window-limit", type=int)
    parser.add_argument("--disable-mlflow", action="store_true")
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


def parse_model_names(value: str) -> list[str]:
    names = [name.strip().lower() for name in value.split(",") if name.strip()]
    invalid = sorted(set(names) - set(SUPPORTED_MODELS))
    if invalid:
        raise ValueError(f"Unsupported model names: {invalid}")
    if not names:
        raise ValueError("At least one model must be selected.")
    return names


def select_windows(*, limit: int | None) -> list[WindowSpec]:
    windows = list(WINDOWS)
    if limit is not None:
        if limit <= 0:
            raise ValueError("window_limit must be positive.")
        windows = windows[:limit]
    return windows


def load_retained_features(report_path: Path) -> list[str]:
    report = pd.read_csv(report_path)
    retention_column = "passed" if "passed" in report.columns else "retained"
    retained = report.loc[report[retention_column].astype(bool), "feature_name"].astype(str).tolist()
    if not retained:
        raise RuntimeError(f"No retained features found in {report_path}.")
    return retained


def build_or_load_feature_matrix(
    *,
    all_features_path: Path,
    cache_path: Path,
    retained_features: list[str],
    start_date: date,
    end_date: date,
    rebalance_weekday: int,
) -> pd.DataFrame:
    metadata_path = cache_path.with_suffix(".meta.json")
    expected_metadata = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "rebalance_weekday": int(rebalance_weekday),
        "retained_features": retained_features,
    }

    if cache_path.exists() and metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text())
        if cached_metadata == expected_metadata:
            cached = pd.read_parquet(cache_path)
            matrix = restore_feature_matrix_index(cached)
            logger.info("loaded cached walk-forward feature matrix from {}", cache_path)
            return matrix

    logger.info(
        "building Friday-only feature matrix from {} for {} -> {} with {} retained features",
        all_features_path,
        start_date,
        end_date,
        len(retained_features),
    )
    features_long = pd.read_parquet(
        all_features_path,
        columns=["ticker", "trade_date", "feature_name", "feature_value"],
        filters=[
            ("trade_date", ">=", start_date),
            ("trade_date", "<=", end_date),
            ("feature_name", "in", retained_features),
        ],
    )
    features_long["trade_date"] = pd.to_datetime(features_long["trade_date"])
    features_long = features_long.loc[features_long["trade_date"].dt.weekday == rebalance_weekday].copy()
    matrix = long_to_feature_matrix(features_long, retained_features)
    matrix = fill_feature_matrix(matrix)
    write_parquet_atomic(feature_matrix_to_frame(matrix), cache_path)
    write_json_atomic(metadata_path, expected_metadata)
    logger.info(
        "saved walk-forward feature matrix cache to {} rows={} dates={} tickers={} features={}",
        cache_path,
        len(matrix),
        matrix.index.get_level_values("trade_date").nunique(),
        matrix.index.get_level_values("ticker").nunique(),
        matrix.shape[1],
    )
    return matrix


def build_or_load_label_series(
    *,
    label_cache_path: Path,
    tickers: list[str],
    start_date: date,
    end_date: date,
    as_of: date,
    horizon: int,
    label_buffer_days: int,
    benchmark_ticker: str,
    rebalance_weekday: int,
) -> pd.Series:
    if label_cache_path.exists():
        labels = pd.read_parquet(label_cache_path)
        logger.info("loaded cached {}D labels from {}", horizon, label_cache_path)
    else:
        price_end = min(end_date + timedelta(days=label_buffer_days), as_of)
        prices = get_prices_pit(
            tickers=list(dict.fromkeys([*tickers, benchmark_ticker.upper()])),
            start_date=start_date,
            end_date=price_end,
            as_of=as_of_datetime(as_of),
        )
        if prices.empty:
            raise RuntimeError("No PIT prices available for label computation.")
        labels = compute_forward_returns(
            prices_df=prices,
            horizons=(horizon,),
            benchmark_ticker=benchmark_ticker,
        )
        write_parquet_atomic(labels, label_cache_path)
        logger.info("saved {}D label cache to {}", horizon, label_cache_path)

    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels = labels.loc[
        (labels["horizon"] == horizon)
        & (labels["ticker"] != benchmark_ticker.upper())
        & (labels["trade_date"] >= pd.Timestamp(start_date))
        & (labels["trade_date"] <= pd.Timestamp(end_date))
        & (labels["trade_date"].dt.weekday == rebalance_weekday)
    ].copy()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    series = labels.set_index(["trade_date", "ticker"])["excess_return"].sort_index().dropna()
    if series.empty:
        raise RuntimeError("No usable labels available after filtering.")
    return series


def align_panel(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    aligned_index = X.index.intersection(y.index)
    if aligned_index.empty:
        raise RuntimeError("No aligned observations between features and labels.")
    return X.loc[aligned_index].sort_index(), y.loc[aligned_index].sort_index()


def run_window(
    *,
    window: WindowSpec,
    config: ValidationWindowConfig,
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str],
    tracker: ExperimentTracker | None,
    timestamp: str,
    alpha_grid: tuple[float, ...],
    search_n_iter: int,
    tree_n_jobs: int,
    random_state: int,
) -> dict[str, Any]:
    train_X, train_y = slice_window(
        X=X,
        y=y,
        start_date=config.train_start,
        end_date=config.train_end,
        rebalance_weekday=config.rebalance_weekday,
    )
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

    window_payload: dict[str, Any] = {
        "window_id": window.window_id,
        "dates": {
            "train_start": window.train_start.isoformat(),
            "train_end": window.train_end.isoformat(),
            "validation_start": window.validation_start.isoformat(),
            "validation_end": window.validation_end.isoformat(),
            "test_start": window.test_start.isoformat(),
            "test_end": window.test_end.isoformat(),
            "train": f"{window.train_start:%Y-%m} -> {window.train_end:%Y-%m}",
            "validation": f"{window.validation_start:%Y-%m} -> {window.validation_end:%Y-%m}",
            "test": f"{window.test_start:%Y-%m} -> {window.test_end:%Y-%m}",
        },
        "row_counts": {
            "train": build_split_shape(train_X),
            "validation": build_split_shape(validation_X),
            "test": build_split_shape(test_X),
        },
        "results": {},
    }

    for model_name in model_names:
        if model_name == "ridge":
            result = run_ridge_model(
                train_X=train_X,
                train_y=train_y,
                validation_X=validation_X,
                validation_y=validation_y,
                test_X=test_X,
                test_y=test_y,
                window_id=window.window_id,
                target_horizon=config.target_horizon,
                tracker=tracker,
                timestamp=timestamp,
                alpha_grid=alpha_grid,
            )
        elif model_name == "xgboost":
            result = run_tree_model(
                model=XGBoostModel(
                    random_state=random_state,
                    n_jobs=tree_n_jobs,
                    search_n_iter=search_n_iter,
                ),
                train_X=train_X,
                train_y=train_y,
                validation_X=validation_X,
                validation_y=validation_y,
                test_X=test_X,
                test_y=test_y,
                window_id=window.window_id,
                target_horizon=config.target_horizon,
                tracker=tracker,
                timestamp=timestamp,
                search_n_iter=search_n_iter,
            )
        elif model_name == "lightgbm":
            result = run_tree_model(
                model=LightGBMModel(
                    random_state=random_state,
                    n_jobs=tree_n_jobs,
                    search_n_iter=search_n_iter,
                ),
                train_X=train_X,
                train_y=train_y,
                validation_X=validation_X,
                validation_y=validation_y,
                test_X=test_X,
                test_y=test_y,
                window_id=window.window_id,
                target_horizon=config.target_horizon,
                tracker=tracker,
                timestamp=timestamp,
                search_n_iter=search_n_iter,
            )
        else:  # pragma: no cover - guarded by parse_model_names.
            raise ValueError(f"Unsupported model: {model_name}")

        window_payload["results"][model_name] = result
        logger.info(
            "{} {} test_ic={:.6f} rank_ic={:.6f} hit_rate={:.6f}",
            window.window_id,
            model_name,
            result["test_metrics"]["ic"],
            result["test_metrics"]["rank_ic"],
            result["test_metrics"]["hit_rate"],
        )

    return window_payload


def run_ridge_model(
    *,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    validation_X: pd.DataFrame,
    validation_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    window_id: str,
    target_horizon: str,
    tracker: ExperimentTracker | None,
    timestamp: str,
    alpha_grid: tuple[float, ...],
) -> dict[str, Any]:
    model = RidgeBaselineModel(alpha_grid=alpha_grid)
    selection = model.select_alpha(train_X, train_y, validation_X, validation_y)
    model.train(train_X, train_y)
    validation_pred = model.predict(validation_X)
    validation_metrics = evaluate_predictions(validation_y, validation_pred)

    final_train_X = pd.concat([train_X, validation_X]).sort_index()
    final_train_y = pd.concat([train_y, validation_y]).sort_index()
    model.train(final_train_X, final_train_y)
    test_pred = model.predict(test_X)
    test_metrics = evaluate_predictions(test_y, test_pred)

    logged_run = log_model_run(
        tracker=tracker,
        model=model,
        window_id=window_id,
        target_horizon=target_horizon,
        timestamp=timestamp,
        params={
            **model.get_params(),
            "alpha_grid_size": len(alpha_grid),
            "best_validation_ic": selection.best_ic,
        },
        metrics=metric_payload(validation_metrics=validation_metrics, test_metrics=test_metrics),
    )
    return {
        "best_hyperparams": {"alpha": float(selection.best_hyperparams)},
        "selection": {
            "best_ic": float(selection.best_ic),
            "scores_by_alpha": {str(alpha): float(score) for alpha, score in selection.scores_by_alpha.items()},
        },
        "validation_metrics": evaluation_to_dict(validation_metrics),
        "test_metrics": evaluation_to_dict(test_metrics),
        "logged_run": logged_run_to_dict(logged_run),
    }


def run_tree_model(
    *,
    model: XGBoostModel | LightGBMModel,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    validation_X: pd.DataFrame,
    validation_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    window_id: str,
    target_horizon: str,
    tracker: ExperimentTracker | None,
    timestamp: str,
    search_n_iter: int,
) -> dict[str, Any]:
    selection = model.select_hyperparameters(
        train_X,
        train_y,
        validation_X,
        validation_y,
        n_iter=search_n_iter,
        tracker=None,
        target_horizon=target_horizon,
        window_id=window_id,
        timestamp=timestamp,
    )
    model.train(train_X, train_y)
    validation_pred = model.predict(validation_X)
    validation_metrics = evaluate_predictions(validation_y, validation_pred)

    final_train_X = pd.concat([train_X, validation_X]).sort_index()
    final_train_y = pd.concat([train_y, validation_y]).sort_index()
    model.train(final_train_X, final_train_y)
    test_pred = model.predict(test_X)
    test_metrics = evaluate_predictions(test_y, test_pred)

    logged_run = log_model_run(
        tracker=tracker,
        model=model,
        window_id=window_id,
        target_horizon=target_horizon,
        timestamp=timestamp,
        params={
            **model.get_params(),
            "search_n_iter_requested": int(search_n_iter),
            "search_n_iter_actual": int(selection.n_iter),
            "best_validation_ic": float(selection.best_ic),
        },
        metrics=metric_payload(validation_metrics=validation_metrics, test_metrics=test_metrics),
    )
    return {
        "best_hyperparams": {str(key): normalize_value(value) for key, value in selection.best_params.items()},
        "selection": {
            "best_ic": float(selection.best_ic),
            "search_n_iter": int(selection.n_iter),
            "trial_count": int(len(selection.trials)),
        },
        "validation_metrics": evaluation_to_dict(validation_metrics),
        "test_metrics": evaluation_to_dict(test_metrics),
        "logged_run": logged_run_to_dict(logged_run),
    }


def log_model_run(
    *,
    tracker: ExperimentTracker | None,
    model: Any,
    window_id: str,
    target_horizon: str,
    timestamp: str,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> LoggedModelRun | None:
    if tracker is None:
        return None
    try:
        return tracker.log_training_run(
            model=model,
            target_horizon=target_horizon,
            window_id=window_id,
            params=params,
            metrics=metrics,
            timestamp=timestamp,
        )
    except Exception as exc:  # pragma: no cover - depends on external MLflow environment.
        logger.warning("MLflow logging failed for {} {}: {}", window_id, model.model_type, exc)
        return None


def metric_payload(*, validation_metrics: EvaluationSummary, test_metrics: EvaluationSummary) -> dict[str, float]:
    payload = {
        **prefix_metrics("validation", validation_metrics),
        **prefix_metrics("test", test_metrics),
        "top_decile_return": float(test_metrics.top_decile_return),
        "long_short_return": float(test_metrics.long_short_return),
        "turnover": float(test_metrics.turnover),
    }
    return payload


def prefix_metrics(prefix: str, summary: EvaluationSummary) -> dict[str, float]:
    return {
        f"{prefix}_{metric_name}": float(metric_value)
        for metric_name, metric_value in summary.to_dict().items()
    }


def evaluation_to_dict(summary: EvaluationSummary) -> dict[str, float]:
    return {key: float(value) for key, value in summary.to_dict().items()}


def logged_run_to_dict(run: LoggedModelRun | None) -> dict[str, str] | None:
    if run is None:
        return None
    return {
        "tracking_uri": run.tracking_uri,
        "experiment_name": run.experiment_name,
        "experiment_id": run.experiment_id,
        "run_id": run.run_id,
    }


def build_split_shape(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "dates": int(frame.index.get_level_values("trade_date").nunique()),
        "tickers": int(frame.index.get_level_values("ticker").nunique()),
    }


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


def summarize_results(windows: list[dict[str, Any]], model_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for model_name in model_names:
        test_ics = [window["results"][model_name]["test_metrics"]["ic"] for window in windows if model_name in window["results"]]
        rank_ics = [window["results"][model_name]["test_metrics"]["rank_ic"] for window in windows if model_name in window["results"]]
        icirs = [window["results"][model_name]["test_metrics"]["icir"] for window in windows if model_name in window["results"]]
        hit_rates = [window["results"][model_name]["test_metrics"]["hit_rate"] for window in windows if model_name in window["results"]]
        top_deciles = [window["results"][model_name]["test_metrics"]["top_decile_return"] for window in windows if model_name in window["results"]]

        win_count = 0
        for window in windows:
            test_ic_by_model = {
                name: window["results"][name]["test_metrics"]["ic"]
                for name in model_names
                if name in window["results"]
            }
            if not test_ic_by_model:
                continue
            best_ic = max(test_ic_by_model.values())
            if np.isclose(test_ic_by_model.get(model_name, float("-inf")), best_ic):
                win_count += 1

        summary[model_name] = {
            "windows_completed": int(len(test_ics)),
            "mean_test_ic": nanmean(test_ics),
            "std_test_ic": nanstd(test_ics),
            "mean_test_rank_ic": nanmean(rank_ics),
            "mean_test_icir": nanmean(icirs),
            "mean_hit_rate": nanmean(hit_rates),
            "mean_top_decile_return": nanmean(top_deciles),
            "win_count": int(win_count),
        }
    return summary


def nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=float)))


def nanstd(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanstd(np.asarray(values, dtype=float)))


def as_of_datetime(as_of: date) -> datetime:
    return datetime.combine(as_of, time.max, tzinfo=timezone.utc)


def normalize_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
            return None
        return item
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
