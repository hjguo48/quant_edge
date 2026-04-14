from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
import math
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
from scripts.run_regime_analysis import (
    MODEL_NAMES,
    classify_vix,
    extract_date_field,
    extract_window_dates,
    load_macro_series,
    parse_date,
    select_windows,
)
from scripts.run_walkforward_comparison import (
    DEFAULT_ALL_FEATURES_PATH,
    DEFAULT_IC_REPORT_PATH,
    DEFAULT_LABEL_CACHE_PATH,
    align_panel,
    build_or_load_feature_matrix,
    build_or_load_label_series,
    load_retained_features,
    slice_window,
)
from src.models.baseline import RidgeBaselineModel
from src.models.evaluation import evaluate_predictions, information_coefficient_series
from src.models.tree import LightGBMModel, XGBoostModel

DEFAULT_COMPARISON_REPORT = "data/reports/walkforward_comparison_60d.json"
DEFAULT_REGIME_REPORT = "data/reports/regime_analysis_60d.json"
DEFAULT_OUTPUT_PATH = "data/reports/fusion_analysis_60d.json"
DEFAULT_FUSION_FEATURE_MATRIX_CACHE_PATH = "data/features/fusion_feature_matrix_60d.parquet"
DEFAULT_AS_OF = date(2026, 3, 31)
DEFAULT_HORIZON = 60
DEFAULT_LABEL_BUFFER_DAYS = 120
DEFAULT_REBALANCE_WEEKDAY = 4
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_ROLLING_WINDOW = 60
DEFAULT_TEMPERATURE = 5.0
DEFAULT_TREE_N_JOBS = 4
DEFAULT_RANDOM_STATE = 42


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    comparison_path = REPO_ROOT / args.comparison_report
    regime_path = REPO_ROOT / args.regime_report
    output_path = REPO_ROOT / args.output
    comparison = json.loads(comparison_path.read_text())
    windows = select_windows(comparison.get("windows", []), limit=args.window_limit)
    if not windows:
        raise RuntimeError(f"No windows available in {comparison_path}.")

    required_models = [name.strip().lower() for name in args.models.split(",") if name.strip()]
    missing_models = [name for name in required_models if name not in MODEL_NAMES]
    if missing_models:
        raise ValueError(f"Unsupported model names: {missing_models}")

    regime_weights = load_regime_weights(regime_path)
    retained_features = load_retained_features(REPO_ROOT / args.ic_report_path)
    as_of = parse_date(args.as_of)
    global_start = min(extract_window_dates(window)["train_start"] for window in windows)
    global_end = max(extract_window_dates(window)["test_end"] for window in windows)

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
        as_of=as_of,
        horizon=args.horizon,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=args.benchmark_ticker,
        rebalance_weekday=args.rebalance_weekday,
    )
    aligned_X, aligned_y = align_panel(feature_matrix, labels)
    vix_history = load_macro_series(
        series_id="VIXCLS",
        start_date=global_start,
        end_date=global_end,
        as_of=as_of,
    )
    logger.info(
        "fusion analysis configured for {} windows, {} models, rows={} dates={} tickers={} features={}",
        len(windows),
        len(required_models),
        len(aligned_X),
        aligned_X.index.get_level_values("trade_date").nunique(),
        aligned_X.index.get_level_values("ticker").nunique(),
        aligned_X.shape[1],
    )

    per_window: list[dict[str, Any]] = []
    effective_rolling_points = max(1, int(math.ceil(args.rolling_window / 5)))
    for position, window in enumerate(windows, start=1):
        logger.info("running fusion window {}/{} {}", position, len(windows), window.get("window_id", position))
        result = run_window_fusion(
            window=window,
            X=aligned_X,
            y=aligned_y,
            vix_history=vix_history,
            required_models=required_models,
            regime_weights=regime_weights,
            rolling_window=effective_rolling_points,
            temperature=args.temperature,
            tree_n_jobs=args.tree_n_jobs,
            random_state=args.random_state,
            low_threshold=args.low_threshold,
            high_threshold=args.high_threshold,
            rebalance_weekday=args.rebalance_weekday,
        )
        if result is not None:
            per_window.append(result)

    if not per_window:
        raise RuntimeError("No windows were processed successfully.")

    report = {
        "fusion_config": {
            "comparison_report": str(comparison_path),
            "regime_report": str(regime_path),
            "rolling_window_trading_days": int(args.rolling_window),
            "rolling_window_rebalance_points": int(effective_rolling_points),
            "temperature": float(args.temperature),
            "regime_weights": regime_weights,
            "models": required_models,
            "regime_note": (
                "Positive scalar regime multipliers do not change cross-sectional IC; "
                "regime_weighted_mean_daily_ic is included to reflect exposure scaling."
            ),
        },
        "per_window": per_window,
        "summary": summarize_fusion(per_window, required_models),
    }
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved fusion analysis report to {}", output_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild per-window model predictions and evaluate equal-weight vs rolling-IC fusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--comparison-report", default=DEFAULT_COMPARISON_REPORT)
    parser.add_argument("--regime-report", default=DEFAULT_REGIME_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--feature-matrix-cache-path", default=DEFAULT_FUSION_FEATURE_MATRIX_CACHE_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF.isoformat())
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-buffer-days", type=int, default=DEFAULT_LABEL_BUFFER_DAYS)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--rebalance-weekday", type=int, default=DEFAULT_REBALANCE_WEEKDAY)
    parser.add_argument("--rolling-window", type=int, default=DEFAULT_ROLLING_WINDOW)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--tree-n-jobs", type=int, default=DEFAULT_TREE_N_JOBS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--models", default="ridge,xgboost,lightgbm")
    parser.add_argument("--low-threshold", type=float, default=20.0)
    parser.add_argument("--high-threshold", type=float, default=30.0)
    parser.add_argument("--window-limit", type=int)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def load_regime_weights(path: Path) -> dict[str, float]:
    if not path.exists():
        logger.warning("regime report {} missing; using neutral weights", path)
        return {"low": 1.0, "mid": 1.0, "high": 1.0, "unknown": 1.0}
    payload = json.loads(path.read_text())
    weights = payload.get("regime_weights", {})
    return {
        "low": float(weights.get("low", 1.0)),
        "mid": float(weights.get("mid", 1.0)),
        "high": float(weights.get("high", 1.0)),
        "unknown": float(weights.get("unknown", 1.0)),
    }


def run_window_fusion(
    *,
    window: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    vix_history: pd.Series,
    required_models: list[str],
    regime_weights: dict[str, float],
    rolling_window: int,
    temperature: float,
    tree_n_jobs: int,
    random_state: int,
    low_threshold: float,
    high_threshold: float,
    rebalance_weekday: int,
) -> dict[str, Any] | None:
    dates = extract_window_dates(window)
    available_models = [name for name in required_models if name in window.get("results", {})]
    missing = sorted(set(required_models) - set(available_models))
    if missing:
        logger.warning("skipping {} because models are missing from comparison report: {}", window.get("window_id"), missing)
        return None

    train_X, train_y = slice_window(
        X=X,
        y=y,
        start_date=dates["train_start"],
        end_date=dates["train_end"],
        rebalance_weekday=rebalance_weekday,
    )
    validation_X, validation_y = slice_window(
        X=X,
        y=y,
        start_date=dates["validation_start"],
        end_date=dates["validation_end"],
        rebalance_weekday=rebalance_weekday,
    )
    test_X, test_y = slice_window(
        X=X,
        y=y,
        start_date=dates["test_start"],
        end_date=dates["test_end"],
        rebalance_weekday=rebalance_weekday,
    )

    combined_train_X = pd.concat([train_X, validation_X]).sort_index()
    combined_train_y = pd.concat([train_y, validation_y]).sort_index()

    raw_predictions: dict[str, pd.Series] = {}
    model_metrics: dict[str, dict[str, float]] = {}
    daily_ic_frame = pd.DataFrame(index=sorted(test_X.index.get_level_values("trade_date").unique()))

    for model_name in available_models:
        params = extract_best_params(window, model_name)
        model = instantiate_model(
            model_name=model_name,
            best_params=params,
            tree_n_jobs=tree_n_jobs,
            random_state=random_state,
        )
        model.train(combined_train_X, combined_train_y)
        predictions = model.predict(test_X).rename(model_name)
        raw_predictions[model_name] = predictions
        metrics = evaluate_predictions(test_y, predictions)
        model_metrics[model_name] = metrics.to_dict()
        daily_ic_frame[model_name] = information_coefficient_series(test_y, predictions).reindex(
            daily_ic_frame.index,
        )

    normalized_predictions = {
        model_name: cross_sectional_zscore(predictions)
        for model_name, predictions in raw_predictions.items()
    }
    prediction_frame = pd.concat(
        [test_y.rename("actual"), *[series.rename(name) for name, series in normalized_predictions.items()]],
        axis=1,
        join="inner",
    ).dropna(subset=["actual"])
    prediction_frame.sort_index(inplace=True)

    equal_weight_frame = build_equal_weights(
        dates=prediction_frame.index.get_level_values("trade_date").unique(),
        model_names=available_models,
    )
    rolling_weight_frame = build_rolling_weights(
        daily_ic_frame=daily_ic_frame[available_models],
        model_names=available_models,
        rolling_window=rolling_window,
        temperature=temperature,
    )
    regime_scalar = build_regime_scalar(
        vix_history=vix_history,
        dates=prediction_frame.index.get_level_values("trade_date").unique(),
        regime_weights=regime_weights,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    equal_pred = combine_predictions(prediction_frame, equal_weight_frame, available_models).rename("equal_weight")
    weighted_pred = combine_predictions(prediction_frame, rolling_weight_frame, available_models).rename("ic_weighted")
    regime_pred = apply_regime_scalar(weighted_pred, regime_scalar).rename("ic_weighted_regime")

    equal_metrics = evaluate_predictions(test_y, equal_pred)
    weighted_metrics = evaluate_predictions(test_y, weighted_pred)
    regime_metrics = evaluate_predictions(test_y, regime_pred)

    equal_daily_ic = information_coefficient_series(test_y, equal_pred)
    weighted_daily_ic = information_coefficient_series(test_y, weighted_pred)
    regime_daily_ic = information_coefficient_series(test_y, regime_pred)

    avg_weights = {
        model_name: float(rolling_weight_frame[model_name].mean())
        for model_name in available_models
    }
    window_id = str(window.get("window_id", f"W{dates['test_start'].isoformat()}"))

    return {
        "window_id": window_id,
        "train_start": dates["train_start"].isoformat(),
        "train_end": dates["train_end"].isoformat(),
        "val_start": dates["validation_start"].isoformat(),
        "val_end": dates["validation_end"].isoformat(),
        "test_start": dates["test_start"].isoformat(),
        "test_end": dates["test_end"].isoformat(),
        **{f"{model_name}_ic": float(model_metrics[model_name]["ic"]) for model_name in available_models},
        "equal_weight_fusion_ic": float(equal_metrics.ic),
        "ic_weighted_fusion_ic": float(weighted_metrics.ic),
        "ic_weighted_regime_fusion_ic": float(regime_metrics.ic),
        "equal_weight_fusion_metrics": evaluation_to_dict(equal_metrics),
        "ic_weighted_fusion_metrics": evaluation_to_dict(weighted_metrics),
        "ic_weighted_regime_fusion_metrics": evaluation_to_dict(regime_metrics),
        "equal_weight_mean_daily_ic": safe_mean(equal_daily_ic),
        "ic_weighted_mean_daily_ic": safe_mean(weighted_daily_ic),
        "regime_weighted_mean_daily_ic": weighted_mean_daily_ic(regime_daily_ic, regime_scalar),
        "avg_weights": avg_weights,
        "n_test_dates": int(test_X.index.get_level_values("trade_date").nunique()),
        "n_test_rows": int(len(test_X)),
    }


def extract_best_params(window: dict[str, Any], model_name: str) -> dict[str, Any]:
    result = window.get("results", {}).get(model_name, {})
    params = result.get("best_hyperparams") or result.get("best_params") or {}
    if model_name == "ridge" and isinstance(params, (int, float)):
        return {"alpha": float(params)}
    return {str(key): normalize_json_value(value) for key, value in dict(params).items()}


def normalize_json_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def instantiate_model(
    *,
    model_name: str,
    best_params: dict[str, Any],
    tree_n_jobs: int,
    random_state: int,
) -> RidgeBaselineModel | XGBoostModel | LightGBMModel:
    if model_name == "ridge":
        alpha = float(best_params.get("alpha", 1.0))
        return RidgeBaselineModel(alpha=alpha)
    if model_name == "xgboost":
        return XGBoostModel(random_state=random_state, n_jobs=tree_n_jobs, **best_params)
    if model_name == "lightgbm":
        return LightGBMModel(random_state=random_state, n_jobs=tree_n_jobs, **best_params)
    raise ValueError(f"Unsupported model {model_name}")


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    frames: list[pd.Series] = []
    for _, group in series.groupby(level="trade_date", sort=True):
        values = group.astype(float)
        centered = values - values.mean()
        std = values.std(ddof=0)
        if not np.isfinite(std) or np.isclose(std, 0.0):
            frames.append(pd.Series(0.0, index=values.index, dtype=float))
        else:
            frames.append((centered / std).astype(float))
    return pd.concat(frames).sort_index()


def build_equal_weights(*, dates: pd.Index, model_names: list[str]) -> pd.DataFrame:
    weight = 1.0 / len(model_names)
    frame = pd.DataFrame(index=pd.Index(pd.to_datetime(dates), name="trade_date"))
    for model_name in model_names:
        frame[model_name] = weight
    return frame


def build_rolling_weights(
    *,
    daily_ic_frame: pd.DataFrame,
    model_names: list[str],
    rolling_window: int,
    temperature: float,
    negative_ic_action: str = "zero",
) -> pd.DataFrame:
    history = daily_ic_frame.copy().sort_index()
    score_frame = history.shift(1).rolling(rolling_window, min_periods=rolling_window).mean()
    weights = pd.DataFrame(index=history.index, columns=model_names, dtype=float)
    equal = np.full(len(model_names), 1.0 / len(model_names), dtype=float)

    for trade_date in weights.index:
        scores = score_frame.loc[trade_date, model_names].to_numpy(dtype=float)
        if np.isfinite(scores).all():
            gated_scores = scores.copy()
            if negative_ic_action == "zero":
                gated_scores[gated_scores < 0] = 0.0
            elif negative_ic_action == "halve":
                gated_scores[gated_scores < 0] *= 0.5
            if gated_scores.sum() == 0:
                weights.loc[trade_date, model_names] = equal
            else:
                weights.loc[trade_date, model_names] = softmax(gated_scores * temperature)
        else:
            weights.loc[trade_date, model_names] = equal
    return weights


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    total = exps.sum()
    if not np.isfinite(total) or np.isclose(total, 0.0):
        return np.full_like(values, 1.0 / len(values), dtype=float)
    return exps / total


def build_regime_scalar(
    *,
    vix_history: pd.Series,
    dates: pd.Index,
    regime_weights: dict[str, float],
    low_threshold: float,
    high_threshold: float,
) -> pd.Series:
    trade_dates = pd.Index(pd.to_datetime(dates).date, name="trade_date")
    source = pd.Series(vix_history.to_numpy(dtype=float), index=pd.Index(vix_history.index, name="trade_date"))
    aligned = source.reindex(trade_dates, method="ffill")
    values = []
    for trade_date, vix_value in aligned.items():
        regime = classify_vix(float(vix_value), low_threshold=low_threshold, high_threshold=high_threshold)
        values.append(float(regime_weights.get(regime, 1.0)))
    return pd.Series(values, index=pd.Index(pd.to_datetime(trade_dates), name="trade_date"), name="regime_weight")


def combine_predictions(
    prediction_frame: pd.DataFrame,
    weight_frame: pd.DataFrame,
    model_names: list[str],
) -> pd.Series:
    parts: list[pd.Series] = []
    for trade_date, group in prediction_frame.groupby(level="trade_date", sort=True):
        weights = weight_frame.loc[pd.Timestamp(trade_date), model_names].to_numpy(dtype=float)
        scores = group.loc[:, model_names].to_numpy(dtype=float)
        combined = scores @ weights
        parts.append(pd.Series(combined, index=group.index, dtype=float))
    return pd.concat(parts).sort_index()


def apply_regime_scalar(predictions: pd.Series, regime_scalar: pd.Series) -> pd.Series:
    scaled_parts: list[pd.Series] = []
    for trade_date, group in predictions.groupby(level="trade_date", sort=True):
        scalar = float(regime_scalar.loc[pd.Timestamp(trade_date)])
        scaled_parts.append((group.astype(float) * scalar).astype(float))
    return pd.concat(scaled_parts).sort_index()


def weighted_mean_daily_ic(daily_ic: pd.Series, regime_scalar: pd.Series) -> float:
    aligned_weights = regime_scalar.reindex(pd.to_datetime(daily_ic.index))
    values = daily_ic.to_numpy(dtype=float)
    weights = aligned_weights.to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    values = series.to_numpy(dtype=float)
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanmean(values))


def evaluation_to_dict(summary: Any) -> dict[str, float]:
    return {key: float(value) for key, value in summary.to_dict().items()}


def summarize_fusion(per_window: list[dict[str, Any]], required_models: list[str]) -> dict[str, Any]:
    summary = {
        "equal_weight_fusion": summarize_metric_block(per_window, "equal_weight_fusion_ic"),
        "ic_weighted_fusion": summarize_metric_block(per_window, "ic_weighted_fusion_ic"),
        "ic_weighted_regime_fusion": summarize_metric_block(per_window, "ic_weighted_regime_fusion_ic"),
        "best_single_model": {},
        "improvement_vs_equal_weight": float("nan"),
    }

    best_name = None
    best_score = float("-inf")
    for model_name in required_models:
        mean_ic = mean_ignore_nan([window.get(f"{model_name}_ic", float("nan")) for window in per_window])
        if np.isfinite(mean_ic) and mean_ic > best_score:
            best_name = model_name
            best_score = mean_ic

    summary["best_single_model"] = {
        "name": best_name,
        "mean_ic": float(best_score) if np.isfinite(best_score) else None,
    }
    equal_mean = summary["equal_weight_fusion"]["mean_ic"]
    weighted_mean = summary["ic_weighted_fusion"]["mean_ic"]
    if equal_mean is not None and weighted_mean is not None:
        summary["improvement_vs_equal_weight"] = float(weighted_mean - equal_mean)
    return summary


def summarize_metric_block(per_window: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [float(window.get(key, float("nan"))) for window in per_window]
    arr = np.asarray(values, dtype=float)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return {"mean_ic": None, "std_ic": None, "n_windows": 0}
    return {
        "mean_ic": float(valid.mean()),
        "std_ic": float(valid.std(ddof=0)),
        "n_windows": int(valid.size),
    }


def mean_ignore_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


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
