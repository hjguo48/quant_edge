from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
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

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import (
    align_panel,
    build_or_load_label_series,
    build_split_series,
    configure_logging,
    current_git_branch,
    json_safe,
    load_retained_features,
    metrics_to_dict,
    restore_feature_matrix_index,
    slice_window,
)
from src.mlflow_config import get_mlflow_tracking_uri
from src.models.experiment import ExperimentTracker, ValidationWindowConfig
from src.models.tree import (
    LightGBMModel,
    XGBoostModel,
    compare_model_metrics,
    feature_importance_frame,
    zero_contribution_features,
)

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
DEFAULT_N_JOBS = min(8, max(1, os.cpu_count() or 1))

DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_v2.csv"
DEFAULT_FEATURE_MATRIX_PATH = "data/features/window1_feature_matrix.parquet"
DEFAULT_LABEL_CACHE_PATH = "data/labels/window1_forward_returns_5d.parquet"
DEFAULT_BASELINE_REPORT_PATH = "data/reports/single_window_validation.json"
DEFAULT_REPORT_PATH = "data/reports/tree_vs_baseline_comparison.json"
DEFAULT_BENCHMARK_TICKER = "SPY"

COMPACT_XGBOOST_SEARCH_SPACE = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [5],
    "reg_alpha": [0.0],
    "reg_lambda": [1.0],
}

COMPACT_LIGHTGBM_SEARCH_SPACE = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 300],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "min_child_samples": [20],
    "reg_alpha": [0.0],
    "reg_lambda": [1.0],
}


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

    retained_features = load_retained_features(REPO_ROOT / args.ic_report_path)
    if len(retained_features) != 58:
        logger.warning("retained feature count is {} instead of 58", len(retained_features))

    feature_matrix = restore_feature_matrix_index(pd.read_parquet(REPO_ROOT / args.feature_matrix_path))
    label_series, label_diagnostics = build_or_load_label_series(
        tickers=feature_matrix.index.get_level_values("ticker").unique().tolist(),
        config=config,
        as_of=as_of,
        label_cache_path=REPO_ROOT / args.label_cache_path,
        horizon=args.horizon,
        benchmark_ticker=args.benchmark_ticker,
    )
    aligned_X, aligned_y = align_panel(feature_matrix, label_series)

    baseline_report = load_baseline_report(REPO_ROOT / args.baseline_report_path)
    baseline_tracking_uri = resolve_tracking_uri(baseline_report)
    tracker = ExperimentTracker(tracking_uri=baseline_tracking_uri)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    logger.info(
        "running tree comparison on {} aligned rows across {} dates and {} tickers using {} retained features",
        len(aligned_X),
        aligned_X.index.get_level_values("trade_date").nunique(),
        aligned_X.index.get_level_values("ticker").nunique(),
        len(retained_features),
    )

    xgboost_result = run_tree_model(
        model_cls=XGBoostModel,
        model_name="xgboost",
        X=aligned_X,
        y=aligned_y,
        config=config,
        tracker=tracker,
        tracking_uri=baseline_tracking_uri,
        timestamp=timestamp,
        n_jobs=args.n_jobs,
        search_space=COMPACT_XGBOOST_SEARCH_SPACE,
    )
    lightgbm_result = run_tree_model(
        model_cls=LightGBMModel,
        model_name="lightgbm",
        X=aligned_X,
        y=aligned_y,
        config=config,
        tracker=tracker,
        tracking_uri=baseline_tracking_uri,
        timestamp=timestamp,
        n_jobs=args.n_jobs,
        search_space=COMPACT_LIGHTGBM_SEARCH_SPACE,
    )

    report = build_report_payload(
        args=args,
        config=config,
        as_of=as_of,
        retained_features=retained_features,
        feature_matrix=aligned_X,
        label_diagnostics=label_diagnostics,
        baseline_report=baseline_report,
        baseline_tracking_uri=baseline_tracking_uri,
        xgboost_result=xgboost_result,
        lightgbm_result=lightgbm_result,
    )
    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved tree comparison report to {}", report_path)
    log_summary(report)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare XGBoost and LightGBM against the Ridge baseline on Window 1 using cached real data.",
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
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--baseline-report-path", default=DEFAULT_BASELINE_REPORT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_baseline_report(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        raise FileNotFoundError(f"Baseline report not found: {report_path}")
    return json.loads(report_path.read_text())


def resolve_tracking_uri(baseline_report: dict[str, Any]) -> str:
    tracking_uri = str(baseline_report.get("mlflow", {}).get("tracking_uri") or "")
    if tracking_uri:
        return tracking_uri
    return get_mlflow_tracking_uri()


def run_tree_model(
    *,
    model_cls: type,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    config: ValidationWindowConfig,
    tracker: ExperimentTracker,
    tracking_uri: str,
    timestamp: str,
    n_jobs: int,
    search_space: dict[str, list[Any]],
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

    search_n_iter = total_combinations(search_space)
    model = model_cls(n_jobs=n_jobs, search_n_iter=search_n_iter)
    logger.info("{} search configured with {} exhaustive trials", model_name, search_n_iter)

    selection = model.select_hyperparameters(
        X_train=train_X,
        y_train=train_y,
        X_val=validation_X,
        y_val=validation_y,
        n_iter=search_n_iter,
        search_space=search_space,
        tracker=tracker,
        target_horizon=config.target_horizon,
        window_id=f"{model_name}_window1_search",
        timestamp=timestamp,
    )

    model.train(train_X, train_y)
    validation_predictions = model.predict(validation_X)
    validation_metrics = model.evaluate(validation_y, validation_predictions)

    final_train_X = train_X
    final_train_y = train_y
    if config.refit_on_train_plus_validation:
        final_train_X = pd.concat([train_X, validation_X]).sort_index()
        final_train_y = pd.concat([train_y, validation_y]).sort_index()
    model.train(final_train_X, final_train_y)
    test_predictions = model.predict(test_X)
    test_metrics = model.evaluate(test_y, test_predictions)

    logged_run = model.log_training_run(
        target_horizon=config.target_horizon,
        window_id=f"{model_name}_window1",
        metrics=prefixed_metrics(validation_metrics, "validation") | prefixed_metrics(test_metrics, "test"),
        tracking_uri=tracking_uri,
        timestamp=timestamp,
        extra_params={
            "train_start": config.train_start.isoformat(),
            "train_end": config.train_end.isoformat(),
            "validation_start": config.validation_start.isoformat(),
            "validation_end": config.validation_end.isoformat(),
            "test_start": config.test_start.isoformat(),
            "test_end": config.test_end.isoformat(),
            "rebalance_weekday": config.rebalance_weekday,
            "search_trials": search_n_iter,
        },
    )

    importance = feature_importance_frame(model)
    zero_features = zero_contribution_features(model)

    return {
        "model_type": model.model_type,
        "selection": {
            "best_params": selection.best_params,
            "best_ic": float(selection.best_ic),
            "n_iter": int(selection.n_iter),
            "top_trials": trial_records(selection.trials, top_n=10),
        },
        "metrics": {
            "validation": metrics_to_dict(validation_metrics),
            "test": metrics_to_dict(test_metrics),
        },
        "window_rows": {
            "train": summarize_window(train_X),
            "validation": summarize_window(validation_X),
            "test": summarize_window(test_X),
        },
        "series": {
            "validation": build_split_series(y_true=validation_y, y_pred=validation_predictions),
            "test": build_split_series(y_true=test_y, y_pred=test_predictions),
        },
        "feature_importance_top20": importance.head(20).to_dict(orient="records"),
        "zero_contribution_features": zero_features,
        "mlflow": {
            "logged": logged_run is not None,
            "tracking_uri": logged_run.tracking_uri if logged_run else None,
            "experiment_name": logged_run.experiment_name if logged_run else None,
            "experiment_id": logged_run.experiment_id if logged_run else None,
            "run_id": logged_run.run_id if logged_run else None,
        },
    }


def total_combinations(search_space: dict[str, list[Any]]) -> int:
    total = 1
    for values in search_space.values():
        total *= len(values)
    return int(total)


def trial_records(trials: pd.DataFrame, *, top_n: int) -> list[dict[str, Any]]:
    if trials.empty:
        return []
    return trials.head(top_n).to_dict(orient="records")


def summarize_window(X: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(X)),
        "dates": int(X.index.get_level_values("trade_date").nunique()),
        "tickers": int(X.index.get_level_values("ticker").nunique()),
    }


def prefixed_metrics(metrics: Any, prefix: str) -> dict[str, float]:
    payload = metrics_to_dict(metrics)
    return {f"{prefix}_{key}": float(value) for key, value in payload.items()}


def comparison_table(
    *,
    baseline_metrics: dict[str, float],
    xgboost_metrics: dict[str, float],
    lightgbm_metrics: dict[str, float],
) -> list[dict[str, Any]]:
    metric_names = [
        "ic",
        "rank_ic",
        "icir",
        "hit_rate",
        "top_decile_return",
        "long_short_return",
        "turnover",
    ]
    rows: list[dict[str, Any]] = []
    for metric_name in metric_names:
        rows.append(
            {
                "metric": metric_name,
                "ridge": float(baseline_metrics[metric_name]),
                "xgboost": float(xgboost_metrics[metric_name]),
                "lightgbm": float(lightgbm_metrics[metric_name]),
            }
        )
    return rows


def build_report_payload(
    *,
    args: argparse.Namespace,
    config: ValidationWindowConfig,
    as_of: date,
    retained_features: list[str],
    feature_matrix: pd.DataFrame,
    label_diagnostics: dict[str, Any],
    baseline_report: dict[str, Any],
    baseline_tracking_uri: str,
    xgboost_result: dict[str, Any],
    lightgbm_result: dict[str, Any],
) -> dict[str, Any]:
    baseline_validation = baseline_report["metrics"]["validation"]
    baseline_test = baseline_report["metrics"]["test"]
    xgb_validation = xgboost_result["metrics"]["validation"]
    xgb_test = xgboost_result["metrics"]["test"]
    lgb_validation = lightgbm_result["metrics"]["validation"]
    lgb_test = lightgbm_result["metrics"]["test"]

    xgb_vs_ridge_test = compare_model_metrics(baseline_test, xgb_test, baseline_name="ridge", candidate_name="xgboost")
    lgb_vs_ridge_test = compare_model_metrics(baseline_test, lgb_test, baseline_name="ridge", candidate_name="lightgbm")
    xgb_vs_ridge_validation = compare_model_metrics(
        baseline_validation,
        xgb_validation,
        baseline_name="ridge",
        candidate_name="xgboost",
    )
    lgb_vs_ridge_validation = compare_model_metrics(
        baseline_validation,
        lgb_validation,
        baseline_name="ridge",
        candidate_name="lightgbm",
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_tree_comparison.py",
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
            "feature_matrix_path": str(REPO_ROOT / args.feature_matrix_path),
            "label_cache_path": str(REPO_ROOT / args.label_cache_path),
            "baseline_report_path": str(REPO_ROOT / args.baseline_report_path),
            "report_path": str(REPO_ROOT / args.report_path),
            "retained_feature_count": len(retained_features),
            "retained_features": retained_features,
        },
        "data_summary": {
            "aligned_rows": int(len(feature_matrix)),
            "aligned_dates": int(feature_matrix.index.get_level_values("trade_date").nunique()),
            "aligned_tickers": int(feature_matrix.index.get_level_values("ticker").nunique()),
            "aligned_min_date": feature_matrix.index.get_level_values("trade_date").min().date().isoformat(),
            "aligned_max_date": feature_matrix.index.get_level_values("trade_date").max().date().isoformat(),
            "labels": label_diagnostics,
        },
        "search_spaces": {
            "xgboost": COMPACT_XGBOOST_SEARCH_SPACE,
            "lightgbm": COMPACT_LIGHTGBM_SEARCH_SPACE,
        },
        "baseline": {
            "metrics": baseline_report["metrics"],
            "window_rows": baseline_report["window_rows"],
            "mlflow": baseline_report.get("mlflow", {}),
        },
        "xgboost": xgboost_result,
        "lightgbm": lightgbm_result,
        "comparison": {
            "validation_metrics": comparison_table(
                baseline_metrics=baseline_validation,
                xgboost_metrics=xgb_validation,
                lightgbm_metrics=lgb_validation,
            ),
            "test_metrics": comparison_table(
                baseline_metrics=baseline_test,
                xgboost_metrics=xgb_test,
                lightgbm_metrics=lgb_test,
            ),
            "delta_vs_ridge": {
                "validation": {
                    "xgboost": xgb_vs_ridge_validation.reset_index().to_dict(orient="records"),
                    "lightgbm": lgb_vs_ridge_validation.reset_index().to_dict(orient="records"),
                },
                "test": {
                    "xgboost": xgb_vs_ridge_test.reset_index().to_dict(orient="records"),
                    "lightgbm": lgb_vs_ridge_test.reset_index().to_dict(orient="records"),
                },
            },
        },
        "mlflow": {
            "tracking_uri": baseline_tracking_uri,
            "baseline": baseline_report.get("mlflow", {}),
            "xgboost": xgboost_result["mlflow"],
            "lightgbm": lightgbm_result["mlflow"],
        },
    }


def log_summary(report: dict[str, Any]) -> None:
    for model_name in ("baseline", "xgboost", "lightgbm"):
        if model_name == "baseline":
            metrics = report[model_name]["metrics"]["test"]
        else:
            metrics = report[model_name]["metrics"]["test"]
        logger.info(
            "{} test | IC={:.6f} RankIC={:.6f} ICIR={} HitRate={:.6f} TopDecile={:.6f} LongShort={:.6f}",
            model_name,
            metrics["ic"],
            metrics["rank_ic"],
            format_metric(metrics["icir"]),
            metrics["hit_rate"],
            metrics["top_decile_return"],
            metrics["long_short_return"],
        )

    ridge_ic = report["baseline"]["metrics"]["test"]["ic"]
    xgb_ic = report["xgboost"]["metrics"]["test"]["ic"]
    lgb_ic = report["lightgbm"]["metrics"]["test"]["ic"]
    logger.info(
        "IC lift vs ridge | xgboost={:.6f} | lightgbm={:.6f}",
        xgb_ic - ridge_ic,
        lgb_ic - ridge_ic,
    )
    logger.info(
        "zero contribution features | xgboost={} | lightgbm={}",
        report["xgboost"]["zero_contribution_features"],
        report["lightgbm"]["zero_contribution_features"],
    )


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
