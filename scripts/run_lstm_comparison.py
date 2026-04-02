from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

from loguru import logger
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
from src.models.deep import LSTMModel, LSTM_SEARCH_SPACE
from src.models.experiment import ExperimentTracker, ValidationWindowConfig

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
DEEP_MODEL_ADOPTION_DELTA = 0.005
DEFAULT_N_JOBS = min(8, max(1, os.cpu_count() or 1))

DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_v2.csv"
DEFAULT_FEATURE_MATRIX_PATH = "data/features/window1_feature_matrix.parquet"
DEFAULT_LABEL_CACHE_PATH = "data/labels/window1_forward_returns_5d.parquet"
DEFAULT_TREE_REPORT_PATH = "data/reports/tree_vs_baseline_comparison.json"
DEFAULT_REPORT_PATH = "data/reports/lstm_vs_tree_comparison.json"
DEFAULT_BENCHMARK_TICKER = "SPY"


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
    feature_matrix = restore_feature_matrix_index(pd.read_parquet(REPO_ROOT / args.feature_matrix_path))
    feature_matrix = feature_matrix.loc[:, retained_features]
    label_series, label_diagnostics = build_or_load_label_series(
        tickers=feature_matrix.index.get_level_values("ticker").unique().tolist(),
        config=config,
        as_of=as_of,
        label_cache_path=REPO_ROOT / args.label_cache_path,
        horizon=args.horizon,
        benchmark_ticker=args.benchmark_ticker,
    )
    aligned_X, aligned_y = align_panel(feature_matrix, label_series)

    tree_report = load_json(REPO_ROOT / args.tree_report_path)
    tracking_uri = resolve_tracking_uri(tree_report)
    tracker = ExperimentTracker(tracking_uri=tracking_uri)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    train_history_X = history_until(aligned_X, config.train_end)
    train_validation_history_X = history_until(aligned_X, config.validation_end)
    full_test_history_X = history_until(aligned_X, config.test_end)

    _, y_train = slice_window(
        X=aligned_X,
        y=aligned_y,
        start_date=config.train_start,
        end_date=config.train_end,
        rebalance_weekday=config.rebalance_weekday,
    )
    _, y_validation = slice_window(
        X=aligned_X,
        y=aligned_y,
        start_date=config.validation_start,
        end_date=config.validation_end,
        rebalance_weekday=config.rebalance_weekday,
    )
    _, y_test = slice_window(
        X=aligned_X,
        y=aligned_y,
        start_date=config.test_start,
        end_date=config.test_end,
        rebalance_weekday=config.rebalance_weekday,
    )

    y_train_validation = pd.concat([y_train, y_validation]).sort_index()
    logger.info(
        "running LSTM comparison on {} retained features, daily panel rows={}, train anchors={}, validation anchors={}, test anchors={}",
        len(retained_features),
        len(aligned_X),
        len(y_train),
        len(y_validation),
        len(y_test),
    )

    search_model = LSTMModel(
        n_jobs=args.n_jobs,
        search_n_iter=args.search_n_iter,
        max_epochs=args.max_epochs,
        device=args.device,
    )
    selection = search_model.select_hyperparameters(
        X_train=train_history_X,
        y_train=y_train,
        X_val=train_validation_history_X,
        y_val=y_validation,
        tracker=tracker,
        target_horizon=config.target_horizon,
        window_id="lstm_window1_search",
        timestamp=timestamp,
    )

    evaluation_model = build_lstm_model(
        best_params=selection.best_params,
        args=args,
        search_n_iter=args.search_n_iter,
    )
    evaluation_model.search_result_ = selection
    evaluation_model.fit_with_validation(
        X_train=train_history_X,
        y_train=y_train,
        X_val=train_validation_history_X,
        y_val=y_validation,
    )

    train_predictions = evaluation_model.predict_for_index(train_history_X, y_train.index)
    validation_predictions = evaluation_model.predict_for_index(train_validation_history_X, y_validation.index)
    train_target = y_train.loc[train_predictions.index]
    validation_target = y_validation.loc[validation_predictions.index]
    train_metrics = evaluation_model.evaluate(train_target, train_predictions)
    validation_metrics = evaluation_model.evaluate(validation_target, validation_predictions)
    validation_training_info = evaluation_model.training_info_.to_dict() if evaluation_model.training_info_ else {}

    final_model = build_lstm_model(
        best_params=selection.best_params,
        args=args,
        search_n_iter=args.search_n_iter,
    )
    final_model.search_result_ = selection
    final_model.train(train_validation_history_X, y_train_validation)
    test_predictions = final_model.predict_for_index(full_test_history_X, y_test.index)
    test_target = y_test.loc[test_predictions.index]
    test_metrics = final_model.evaluate(test_target, test_predictions)
    final_training_info = final_model.training_info_.to_dict() if final_model.training_info_ else {}

    logged_run = final_model.log_training_run(
        target_horizon=config.target_horizon,
        window_id="lstm_window1",
        metrics=(
            prefixed_metrics(train_metrics, "train")
            | prefixed_metrics(validation_metrics, "validation")
            | prefixed_metrics(test_metrics, "test")
        ),
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
            "search_n_iter": args.search_n_iter,
            "retained_feature_count": len(retained_features),
        },
    )

    report = build_report_payload(
        args=args,
        config=config,
        as_of=as_of,
        retained_features=retained_features,
        aligned_X=aligned_X,
        label_diagnostics=label_diagnostics,
        tree_report=tree_report,
        tracking_uri=tracking_uri,
        selection=selection,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_training_info=validation_training_info,
        final_training_info=final_training_info,
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        train_target=train_target,
        validation_target=validation_target,
        test_target=test_target,
        logged_run=logged_run,
    )

    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved LSTM comparison report to {}", report_path)
    log_summary(report)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Window 1 LSTM comparison against baseline and tree models on cached real data.",
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
    parser.add_argument("--search-n-iter", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--tree-report-path", default=DEFAULT_TREE_REPORT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    return json.loads(path.read_text())


def resolve_tracking_uri(tree_report: dict[str, Any]) -> str:
    tracking_uri = str(tree_report.get("mlflow", {}).get("tracking_uri") or "")
    if tracking_uri:
        return tracking_uri
    return get_mlflow_tracking_uri()


def history_until(X: pd.DataFrame, end_date: date) -> pd.DataFrame:
    dates = pd.to_datetime(X.index.get_level_values("trade_date"))
    return X.loc[dates <= pd.Timestamp(end_date)].sort_index()


def build_lstm_model(
    *,
    best_params: dict[str, Any],
    args: argparse.Namespace,
    search_n_iter: int,
) -> LSTMModel:
    return LSTMModel(
        hidden_size=int(best_params["hidden_size"]),
        num_layers=int(best_params["num_layers"]),
        dropout=float(best_params["dropout"]),
        learning_rate=float(best_params["learning_rate"]),
        weight_decay=float(best_params["weight_decay"]),
        batch_size=int(best_params["batch_size"]),
        search_n_iter=int(search_n_iter),
        max_epochs=args.max_epochs,
        n_jobs=args.n_jobs,
        device=args.device,
    )


def prefixed_metrics(metrics: Any, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{name}": float(value)
        for name, value in metrics_to_dict(metrics).items()
    }


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in candidate
        if key in baseline
    }


def build_report_payload(
    *,
    args: argparse.Namespace,
    config: ValidationWindowConfig,
    as_of: date,
    retained_features: list[str],
    aligned_X: pd.DataFrame,
    label_diagnostics: dict[str, Any],
    tree_report: dict[str, Any],
    tracking_uri: str,
    selection: Any,
    train_metrics: Any,
    validation_metrics: Any,
    test_metrics: Any,
    validation_training_info: dict[str, Any],
    final_training_info: dict[str, Any],
    train_predictions: pd.Series,
    validation_predictions: pd.Series,
    test_predictions: pd.Series,
    train_target: pd.Series,
    validation_target: pd.Series,
    test_target: pd.Series,
    logged_run: Any,
) -> dict[str, Any]:
    ridge_test = tree_report["baseline"]["metrics"]["test"]
    xgboost_test = tree_report["xgboost"]["metrics"]["test"]
    lightgbm_test = tree_report["lightgbm"]["metrics"]["test"]
    lstm_train = metrics_to_dict(train_metrics)
    lstm_validation = metrics_to_dict(validation_metrics)
    lstm_test = metrics_to_dict(test_metrics)

    lstm_vs_xgboost_ic = float(lstm_test["ic"]) - float(xgboost_test["ic"])
    significant = bool(lstm_vs_xgboost_ic >= DEEP_MODEL_ADOPTION_DELTA)
    if significant:
        recommendation = "adopt_lstm"
        reason = (
            f"LSTM test IC improved over XGBoost by {lstm_vs_xgboost_ic:.6f}, "
            f"meeting the {DEEP_MODEL_ADOPTION_DELTA:.3f} adoption threshold."
        )
    elif float(xgboost_test["ic"]) > float(ridge_test["ic"]):
        recommendation = "keep_tree"
        reason = (
            f"LSTM test IC improvement over XGBoost was {lstm_vs_xgboost_ic:.6f}, "
            f"below the {DEEP_MODEL_ADOPTION_DELTA:.3f} threshold, and XGBoost still beats Ridge."
        )
    else:
        recommendation = "keep_baseline"
        reason = (
            f"LSTM test IC improvement over XGBoost was {lstm_vs_xgboost_ic:.6f}, "
            f"below the {DEEP_MODEL_ADOPTION_DELTA:.3f} threshold, and Ridge remains the strongest non-deep baseline."
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_lstm_comparison.py",
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
            "deep_model_adoption_delta": DEEP_MODEL_ADOPTION_DELTA,
        },
        "inputs": {
            "ic_report_path": str(REPO_ROOT / args.ic_report_path),
            "feature_matrix_path": str(REPO_ROOT / args.feature_matrix_path),
            "label_cache_path": str(REPO_ROOT / args.label_cache_path),
            "tree_report_path": str(REPO_ROOT / args.tree_report_path),
            "report_path": str(REPO_ROOT / args.report_path),
            "retained_feature_count": len(retained_features),
            "retained_features": retained_features,
        },
        "data_summary": {
            "aligned_rows": int(len(aligned_X)),
            "aligned_dates": int(aligned_X.index.get_level_values("trade_date").nunique()),
            "aligned_tickers": int(aligned_X.index.get_level_values("ticker").nunique()),
            "aligned_min_date": aligned_X.index.get_level_values("trade_date").min().date().isoformat(),
            "aligned_max_date": aligned_X.index.get_level_values("trade_date").max().date().isoformat(),
            "labels": label_diagnostics,
        },
        "lstm": {
            "best_params": selection.best_params,
            "search_space": LSTM_SEARCH_SPACE,
            "search_n_iter": selection.n_iter,
            "selection": {
                "best_ic": float(selection.best_ic),
                "top_trials": selection.trials.head(10).to_dict(orient="records"),
            },
            "metrics": {
                "train": lstm_train,
                "validation": lstm_validation,
                "test": lstm_test,
            },
            "training_info": final_training_info,
            "validation_training_info": validation_training_info,
            "mlflow": {
                "logged": logged_run is not None,
                "tracking_uri": logged_run.tracking_uri if logged_run else None,
                "experiment_name": logged_run.experiment_name if logged_run else None,
                "experiment_id": logged_run.experiment_id if logged_run else None,
                "run_id": logged_run.run_id if logged_run else None,
            },
            "series": {
                "train": build_split_series(y_true=train_target, y_pred=train_predictions),
                "validation": build_split_series(y_true=validation_target, y_pred=validation_predictions),
                "test": build_split_series(y_true=test_target, y_pred=test_predictions),
            },
        },
        "comparison": {
            "models": ["ridge", "xgboost", "lightgbm", "lstm"],
            "test_metrics": {
                "ridge": ridge_test,
                "xgboost": xgboost_test,
                "lightgbm": lightgbm_test,
                "lstm": lstm_test,
            },
            "delta_vs_ridge": metric_delta(lstm_test, ridge_test),
            "delta_vs_xgboost": {
                **metric_delta(lstm_test, xgboost_test),
                "significant": significant,
            },
            "delta_vs_lightgbm": metric_delta(lstm_test, lightgbm_test),
            "recommendation": recommendation,
        },
        "decision": {
            "lstm_ic_improvement_over_xgboost": lstm_vs_xgboost_ic,
            "threshold": DEEP_MODEL_ADOPTION_DELTA,
            "adopt_deep_model": significant,
            "reason": reason,
        },
        "references": {
            "ridge": tree_report["baseline"],
            "xgboost": tree_report["xgboost"],
            "lightgbm": tree_report["lightgbm"],
            "tracking_uri": tracking_uri,
        },
    }


def log_summary(report: dict[str, Any]) -> None:
    for model_name in ("ridge", "xgboost", "lightgbm", "lstm"):
        metrics = report["comparison"]["test_metrics"][model_name]
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

    logger.info(
        "LSTM vs XGBoost IC delta={:.6f} significant={} recommendation={}",
        report["decision"]["lstm_ic_improvement_over_xgboost"],
        report["comparison"]["delta_vs_xgboost"]["significant"],
        report["comparison"]["recommendation"],
    )


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "nan"
    return f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
