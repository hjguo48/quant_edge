from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import pickle
import sys
from typing import Any

from loguru import logger
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import install_runtime_optimizations, write_json_atomic
from scripts.run_ic_weighted_fusion import extract_best_params, instantiate_model
from scripts.run_regime_analysis import extract_window_dates
from scripts.run_single_window_validation import restore_feature_matrix_index
from scripts.run_walkforward_comparison import (
    DEFAULT_FEATURE_MATRIX_CACHE_PATH,
    DEFAULT_IC_REPORT_PATH,
    DEFAULT_LABEL_CACHE_PATH,
    DEFAULT_REPORT_PATH,
    REBALANCE_WEEKDAY,
    TARGET_HORIZON_DAYS,
    align_panel,
    json_safe,
    load_retained_features,
    slice_window,
)
from src.models.evaluation import evaluate_predictions
from src.models.experiment import ExperimentTracker

DEFAULT_FUSION_REPORT_PATH = "data/reports/fusion_analysis_60d.json"
DEFAULT_REGIME_REPORT_PATH = "data/reports/regime_analysis_60d.json"
DEFAULT_MODELS_DIR = "data/models"
DEFAULT_MANIFEST_PATH = "data/models/fusion_model_bundle_60d.json"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_FALLBACK_TRACKING_URI = f"sqlite:///{(REPO_ROOT / 'mlruns_r86.db').resolve().as_posix()}"
DEFAULT_WINDOW_ID = "W11"
DEFAULT_TREE_N_JOBS = 4
DEFAULT_RANDOM_STATE = 42
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_IC_TOLERANCE = 0.001

MODEL_FILENAMES = {
    "ridge": "fusion_ridge_60d.pkl",
    "xgboost": "fusion_xgboost_60d.pkl",
    "lightgbm": "fusion_lightgbm_60d.pkl",
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    install_runtime_optimizations()

    comparison = json.loads((REPO_ROOT / args.comparison_report).read_text())
    fusion = json.loads((REPO_ROOT / args.fusion_report).read_text())
    regime = json.loads((REPO_ROOT / args.regime_report).read_text())
    window = load_window(comparison, window_id=args.window_id)
    window_dates = extract_window_dates(window)
    seed_weights = load_seed_weights(fusion, window_id=args.window_id)
    regime_weights = load_regime_weights(regime)
    retained_features = load_retained_features(REPO_ROOT / args.ic_report_path)

    feature_matrix = load_feature_matrix(
        path=REPO_ROOT / args.feature_matrix_path,
        retained_features=retained_features,
    )
    label_series = load_label_series(
        path=REPO_ROOT / args.label_cache_path,
        start_date=window_dates["train_start"],
        end_date=window_dates["test_end"],
        horizon=args.horizon,
        rebalance_weekday=args.rebalance_weekday,
        benchmark_ticker=args.benchmark_ticker,
    )
    aligned_X, aligned_y = align_panel(feature_matrix, label_series)

    train_X, train_y = slice_window(
        X=aligned_X,
        y=aligned_y,
        start_date=window_dates["train_start"],
        end_date=window_dates["train_end"],
        rebalance_weekday=args.rebalance_weekday,
    )
    validation_X, validation_y = slice_window(
        X=aligned_X,
        y=aligned_y,
        start_date=window_dates["validation_start"],
        end_date=window_dates["validation_end"],
        rebalance_weekday=args.rebalance_weekday,
    )
    test_X, test_y = slice_window(
        X=aligned_X,
        y=aligned_y,
        start_date=window_dates["test_start"],
        end_date=window_dates["test_end"],
        rebalance_weekday=args.rebalance_weekday,
    )

    final_train_X = pd.concat([train_X, validation_X]).sort_index()
    final_train_y = pd.concat([train_y, validation_y]).sort_index()
    output_dir = REPO_ROOT / args.models_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    manifest: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "window_id": args.window_id,
        "horizon_days": int(args.horizon),
        "rebalance_weekday": int(args.rebalance_weekday),
        "tracking_uri": None if args.disable_mlflow else args.tracking_uri,
        "fallback_tracking_uri": None if args.disable_mlflow else args.fallback_tracking_uri,
        "source_artifacts": {
            "comparison_report": str((REPO_ROOT / args.comparison_report).resolve()),
            "fusion_report": str((REPO_ROOT / args.fusion_report).resolve()),
            "regime_report": str((REPO_ROOT / args.regime_report).resolve()),
            "ic_report": str((REPO_ROOT / args.ic_report_path).resolve()),
            "feature_matrix": str((REPO_ROOT / args.feature_matrix_path).resolve()),
            "label_cache": str((REPO_ROOT / args.label_cache_path).resolve()),
        },
        "window_dates": {name: value.isoformat() for name, value in window_dates.items()},
        "row_counts": {
            "train": build_split_counts(train_X),
            "validation": build_split_counts(validation_X),
            "test": build_split_counts(test_X),
            "train_plus_validation": build_split_counts(final_train_X),
        },
        "retained_features": retained_features,
        "seed_weights": seed_weights,
        "regime_weights": regime_weights,
        "models": {},
    }

    for model_name in ("ridge", "xgboost", "lightgbm"):
        params = extract_best_params(window, model_name)
        model = instantiate_model(
            model_name=model_name,
            best_params=params,
            tree_n_jobs=args.tree_n_jobs,
            random_state=args.random_state,
        )
        model.train(final_train_X, final_train_y)
        test_predictions = model.predict(test_X)
        test_metrics = evaluate_predictions(test_y, test_predictions)
        reference_metrics = window["results"][model_name]["test_metrics"]
        reference_ic = float(reference_metrics["ic"])
        ic_abs_diff = abs(float(test_metrics.ic) - reference_ic)
        if ic_abs_diff > args.ic_tolerance:
            raise RuntimeError(
                f"{model_name} test_ic mismatch {test_metrics.ic:.6f} vs {reference_ic:.6f} "
                f"(diff {ic_abs_diff:.6f} > tolerance {args.ic_tolerance:.6f})",
            )

        model_path = output_dir / MODEL_FILENAMES[model_name]
        with model_path.open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logged_run = None
        if not args.disable_mlflow:
            logged_run = log_training_run_with_fallback(
                model=model,
                params={
                    **params,
                    "source_window": args.window_id,
                    "retained_feature_count": len(retained_features),
                },
                metrics={
                    **{f"test_{name}": float(value) for name, value in test_metrics.to_dict().items()},
                    "reference_test_ic": reference_ic,
                    "reference_test_rank_ic": float(reference_metrics["rank_ic"]),
                    "reference_test_hit_rate": float(reference_metrics["hit_rate"]),
                    "reference_test_top_decile_return": float(reference_metrics["top_decile_return"]),
                    "abs_test_ic_diff": float(ic_abs_diff),
                    "top_decile_return": float(test_metrics.top_decile_return),
                    "long_short_return": float(test_metrics.long_short_return),
                    "turnover": float(test_metrics.turnover),
                },
                target_horizon=f"{args.horizon}D",
                window_id=f"{args.window_id}_fusion_live",
                timestamp=timestamp,
                tracking_uri=args.tracking_uri,
                fallback_tracking_uri=args.fallback_tracking_uri,
            )

        manifest["models"][model_name] = {
            "artifact_path": str(model_path.resolve()),
            "best_hyperparams": params,
            "feature_count": int(len(model.feature_names_)),
            "feature_names": list(model.feature_names_),
            "test_metrics": {key: float(value) for key, value in test_metrics.to_dict().items()},
            "reference_test_metrics": {key: float(value) for key, value in reference_metrics.items()},
            "abs_test_ic_diff": float(ic_abs_diff),
            "mlflow_run": None if logged_run is None else {
                "tracking_uri": logged_run.tracking_uri,
                "experiment_name": logged_run.experiment_name,
                "experiment_id": logged_run.experiment_id,
                "run_id": logged_run.run_id,
            },
        }
        logger.info(
            "trained {} rows={} features={} test_ic={:.6f} ref_ic={:.6f} diff={:.6f} saved={}",
            model_name,
            len(final_train_X),
            len(model.feature_names_),
            test_metrics.ic,
            reference_ic,
            ic_abs_diff,
            model_path,
        )

    write_json_atomic(REPO_ROOT / args.manifest_path, json_safe(manifest))
    logger.info("saved fusion model manifest to {}", REPO_ROOT / args.manifest_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and persist the W11 60D fusion component models for the live greyscale pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--comparison-report", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--fusion-report", default=DEFAULT_FUSION_REPORT_PATH)
    parser.add_argument("--regime-report", default=DEFAULT_REGIME_REPORT_PATH)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_CACHE_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--fallback-tracking-uri", default=DEFAULT_FALLBACK_TRACKING_URI)
    parser.add_argument("--window-id", default=DEFAULT_WINDOW_ID)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--horizon", type=int, default=TARGET_HORIZON_DAYS)
    parser.add_argument("--rebalance-weekday", type=int, default=REBALANCE_WEEKDAY)
    parser.add_argument("--tree-n-jobs", type=int, default=DEFAULT_TREE_N_JOBS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--ic-tolerance", type=float, default=DEFAULT_IC_TOLERANCE)
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def load_window(comparison_payload: dict[str, Any], *, window_id: str) -> dict[str, Any]:
    for window in comparison_payload.get("windows", []):
        if str(window.get("window_id")) == window_id:
            return window
    raise KeyError(f"Window {window_id!r} not found in comparison report.")


def load_seed_weights(fusion_payload: dict[str, Any], *, window_id: str) -> dict[str, float]:
    for window in fusion_payload.get("per_window", []):
        if str(window.get("window_id")) == window_id:
            weights = window.get("avg_weights", {})
            if not weights:
                raise RuntimeError(f"Window {window_id!r} is missing avg_weights in fusion report.")
            return normalize_weights(weights)
    raise KeyError(f"Window {window_id!r} not found in fusion report.")


def load_regime_weights(regime_payload: dict[str, Any]) -> dict[str, float]:
    weights = regime_payload.get("regime_weights", {})
    return {
        "low": float(weights.get("low", 1.0)),
        "mid": float(weights.get("mid", 1.0)),
        "high": float(weights.get("high", 1.0)),
        "unknown": float(weights.get("unknown", 1.0)),
    }


def load_feature_matrix(*, path: Path, retained_features: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    matrix = restore_feature_matrix_index(frame)
    matrix = matrix.reindex(columns=retained_features)
    matrix = matrix.sort_index()
    if matrix.empty:
        raise RuntimeError(f"Feature matrix cache {path} is empty.")
    return matrix


def load_label_series(
    *,
    path: Path,
    start_date: date,
    end_date: date,
    horizon: int,
    rebalance_weekday: int,
    benchmark_ticker: str,
) -> pd.Series:
    labels = pd.read_parquet(path)
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    filtered = labels.loc[
        (labels["horizon"] == horizon)
        & (labels["ticker"] != benchmark_ticker.upper())
        & (labels["trade_date"] >= pd.Timestamp(start_date))
        & (labels["trade_date"] <= pd.Timestamp(end_date))
        & (labels["trade_date"].dt.weekday == rebalance_weekday)
    ].copy()
    series = filtered.set_index(["trade_date", "ticker"])["excess_return"].sort_index().dropna()
    if series.empty:
        raise RuntimeError(f"Label cache {path} produced no usable {horizon}D labels.")
    return series


def build_split_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "dates": int(frame.index.get_level_values("trade_date").nunique()),
        "tickers": int(frame.index.get_level_values("ticker").nunique()),
    }


def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    normalized = {str(key): float(value) for key, value in weights.items()}
    total = sum(normalized.values())
    if total <= 0.0:
        raise RuntimeError("Weight block sums to zero.")
    return {key: value / total for key, value in normalized.items()}


def log_training_run_with_fallback(
    *,
    model: Any,
    params: dict[str, Any],
    metrics: dict[str, float],
    target_horizon: str,
    window_id: str,
    timestamp: str,
    tracking_uri: str,
    fallback_tracking_uri: str | None,
):
    candidates = [tracking_uri]
    if fallback_tracking_uri and fallback_tracking_uri not in candidates:
        candidates.append(fallback_tracking_uri)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            tracker = ExperimentTracker(tracking_uri=candidate)
            logged_run = tracker.log_training_run(
                model=model,
                target_horizon=target_horizon,
                window_id=window_id,
                params=params,
                metrics=metrics,
                timestamp=timestamp,
            )
            if candidate != tracking_uri:
                logger.warning(
                    "MLflow primary tracking URI {} failed; logged to fallback {} instead",
                    tracking_uri,
                    candidate,
                )
            return logged_run
        except Exception as exc:  # pragma: no cover - external MLflow environments vary.
            last_error = exc
            logger.warning("MLflow logging failed for {}: {}", candidate, exc)

    if last_error is not None:
        raise last_error
    return None


if __name__ == "__main__":
    raise SystemExit(main())
