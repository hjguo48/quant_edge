from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
import warnings

from loguru import logger
import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_extended_walkforward import extended_walkforward_windows
from scripts.run_holding_period_experiment import (
    load_calibrated_cost_parameters,
    load_predictions_by_window,
    make_report_safe_payload,
    run_holding_period_experiment,
)
from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import (
    align_panel,
    configure_logging,
    current_git_branch,
    json_safe,
    load_retained_features,
    metrics_to_dict,
    restore_feature_matrix_index,
)
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.engine import WalkForwardEngine, build_universe_by_date
from src.config import DEFAULT_MLFLOW_TRACKING_URI
from src.labels.forward_returns import compute_forward_returns
from src.models.baseline import DEFAULT_ALPHA_GRID
from src.models.evaluation import (
    evaluate_predictions,
    rank_information_coefficient,
)
from src.stats.bootstrap import bootstrap_return_statistics
from src.stats.spa import run_spa_fallback, series_from_records

EXPECTED_BRANCH = "feature/alpha-enhancement"
BENCHMARK_TICKER = "SPY"
SCREENING_HORIZONS = (5, 10, 20, 60)
MODEL_HORIZONS = (10, 20)
FUSION_HORIZONS = (60, 20, 10)
IC_THRESHOLD = 0.02
FALLBACK_FEATURE_COUNT = 10
SELECTION_MODE_THRESHOLD = "threshold_gt_0.02"
SELECTION_MODE_FALLBACK = "fallback_top_positive"

DEFAULT_FEATURE_MATRIX_PATH = "data/features/extended_walkforward_feature_matrix.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_BASELINE_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_v2.csv"
DEFAULT_EXTENDED_REPORT_PATH = "data/reports/extended_walkforward.json"
DEFAULT_HOLDING_PERIOD_REPORT_PATH = "data/reports/holding_period_experiment.json"
DEFAULT_SCREENING_REPORT_PATH = "data/reports/short_horizon_ic_screening.json"
DEFAULT_MODELS_REPORT_PATH = "data/reports/short_horizon_models.json"
DEFAULT_FUSION_REPORT_PATH = "data/reports/signal_fusion_experiment.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"This script must run on branch {EXPECTED_BRANCH}. Found {branch!r}.")

    feature_matrix = restore_feature_matrix_index(pd.read_parquet(REPO_ROOT / args.feature_matrix_path))
    candidate_features = load_retained_features(REPO_ROOT / args.ic_report_path)
    feature_matrix = feature_matrix.loc[:, candidate_features]
    prices = pd.read_parquet(REPO_ROOT / args.prices_path)
    baseline_predictions_frame = pd.read_parquet(REPO_ROOT / args.baseline_predictions_path)
    extended_report = json.loads((REPO_ROOT / args.extended_report_path).read_text())
    holding_period_report = json.loads((REPO_ROOT / args.holding_period_report_path).read_text())

    logger.info(
        "loaded feature rows={} candidate_features={} price rows={} baseline prediction rows={}",
        len(feature_matrix),
        len(candidate_features),
        len(prices),
        len(baseline_predictions_frame),
    )

    labels = compute_forward_returns(prices, horizons=SCREENING_HORIZONS, benchmark_ticker=args.benchmark_ticker)
    label_series_by_horizon = build_label_series_by_horizon(labels)

    screening_report_path = REPO_ROOT / args.screening_report_path
    if args.reuse_screening and screening_report_path.exists():
        screening_payload = json.loads(screening_report_path.read_text())
        selected_features_by_horizon = {
            horizon: list(screening_payload["comparison"][f"{horizon}D"]["selected_features"])
            for horizon in MODEL_HORIZONS
        }
        logger.info("reused short-horizon IC screening report from {}", screening_report_path)
    else:
        screening_payload, selected_features_by_horizon = run_short_horizon_screening(
            feature_matrix=feature_matrix,
            label_series_by_horizon=label_series_by_horizon,
            candidate_features=candidate_features,
            threshold=args.ic_threshold,
        )
        write_json_atomic(screening_report_path, json_safe(screening_payload))
        logger.info("saved short-horizon IC screening report to {}", screening_report_path)

    cost_params = load_calibrated_cost_parameters(REPO_ROOT / args.holding_period_report_path)
    cost_model = AlmgrenChrissCostModel(
        eta=float(cost_params["eta"]),
        gamma=float(cost_params["gamma"]),
        commission_per_share=float(cost_params["commission_per_share"]),
        min_spread_bps=float(cost_params["min_spread_bps"]),
    )
    scheme_config = extract_base_scheme(holding_period_report)

    short_model_payload, model_prediction_bundles = run_short_horizon_models(
        feature_matrix=feature_matrix,
        label_series_by_horizon=label_series_by_horizon,
        selected_features_by_horizon=selected_features_by_horizon,
        prices=prices,
        cost_model=cost_model,
        scheme_config=scheme_config,
        benchmark_ticker=args.benchmark_ticker,
    )
    write_json_atomic(REPO_ROOT / args.models_report_path, json_safe(make_report_safe_payload(short_model_payload)))
    logger.info("saved short-horizon model report to {}", REPO_ROOT / args.models_report_path)

    baseline_60d_bundle = load_predictions_by_window(baseline_predictions_frame)
    baseline_60d_series = series_from_records(
        holding_period_report["holding_periods"]["4W"]["aggregate"]["period_net_excess_records"],
        date_key="trade_date",
        value_key="value",
    )
    baseline_60d_summary = holding_period_report["holding_periods"]["4W"]["aggregate"]
    baseline_60d_ic = float(extended_report["walkforward"]["aggregate"]["mean_test_ic"])

    fusion_payload = run_signal_fusion(
        baseline_60d_bundle=baseline_60d_bundle,
        short_model_payload=short_model_payload,
        model_prediction_bundles=model_prediction_bundles,
        label_series_60d=label_series_by_horizon[60],
        prices=prices,
        cost_model=cost_model,
        scheme_config=scheme_config,
        benchmark_ticker=args.benchmark_ticker,
        baseline_60d_series=baseline_60d_series,
        baseline_60d_summary=baseline_60d_summary,
        baseline_60d_ic=baseline_60d_ic,
    )
    write_json_atomic(REPO_ROOT / args.fusion_report_path, json_safe(make_report_safe_payload(fusion_payload)))
    logger.info("saved signal fusion experiment report to {}", REPO_ROOT / args.fusion_report_path)

    best = fusion_payload["best_configuration"]
    logger.info(
        "best fusion config={} annualized_net_excess={:.6f} bootstrap_lower={:.6f}",
        best["label"],
        best["annualized_net_excess"],
        best["bootstrap"]["sharpe_ci_lower"],
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run short-horizon screening, 10D/20D Ridge models, and multi-horizon signal fusion on the 8-window cached panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--baseline-predictions-path", default=DEFAULT_BASELINE_PREDICTIONS_PATH)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--extended-report-path", default=DEFAULT_EXTENDED_REPORT_PATH)
    parser.add_argument("--holding-period-report-path", default=DEFAULT_HOLDING_PERIOD_REPORT_PATH)
    parser.add_argument("--screening-report-path", default=DEFAULT_SCREENING_REPORT_PATH)
    parser.add_argument("--models-report-path", default=DEFAULT_MODELS_REPORT_PATH)
    parser.add_argument("--fusion-report-path", default=DEFAULT_FUSION_REPORT_PATH)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--ic-threshold", type=float, default=IC_THRESHOLD)
    parser.add_argument("--reuse-screening", action="store_true")
    return parser.parse_args(argv)


def build_label_series_by_horizon(labels: pd.DataFrame) -> dict[int, pd.Series]:
    prepared = labels.copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["excess_return"] = pd.to_numeric(prepared["excess_return"], errors="coerce")
    series_by_horizon: dict[int, pd.Series] = {}
    for horizon, frame in prepared.groupby("horizon", sort=True):
        series_by_horizon[int(horizon)] = (
            frame.set_index(["trade_date", "ticker"])["excess_return"]
            .dropna()
            .sort_index()
        )
    return series_by_horizon


def run_short_horizon_screening(
    *,
    feature_matrix: pd.DataFrame,
    label_series_by_horizon: dict[int, pd.Series],
    candidate_features: list[str],
    threshold: float,
) -> tuple[dict[str, Any], dict[int, list[str]]]:
    comparison: dict[str, dict[str, Any]] = {}
    selected_features_by_horizon: dict[int, list[str]] = {}

    for horizon in SCREENING_HORIZONS:
        report_rows: list[dict[str, Any]] = []
        y = label_series_by_horizon[horizon]
        aligned_index = feature_matrix.index.intersection(y.index)
        X = feature_matrix.loc[aligned_index, candidate_features]
        y_aligned = y.loc[aligned_index]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConstantInputWarning)
            for feature_name in candidate_features:
                rank_ic = rank_information_coefficient(y_aligned, X[feature_name])
                report_rows.append(
                    {
                        "feature_name": feature_name,
                        "rank_ic": float(rank_ic) if pd.notna(rank_ic) else float("nan"),
                    },
                )

        report_rows.sort(
            key=lambda row: (
                -np.inf
                if not np.isfinite(row["rank_ic"])
                else row["rank_ic"]
            ),
            reverse=True,
        )
        passed = [row["feature_name"] for row in report_rows if np.isfinite(row["rank_ic"]) and row["rank_ic"] > threshold]
        selection_mode = SELECTION_MODE_THRESHOLD
        if horizon in MODEL_HORIZONS and not passed:
            passed = [
                row["feature_name"]
                for row in report_rows
                if np.isfinite(row["rank_ic"]) and row["rank_ic"] > 0.0
            ][:FALLBACK_FEATURE_COUNT]
            if not passed:
                passed = [
                    row["feature_name"]
                    for row in report_rows
                    if np.isfinite(row["rank_ic"])
                ][:FALLBACK_FEATURE_COUNT]
            selection_mode = SELECTION_MODE_FALLBACK
        if horizon in MODEL_HORIZONS:
            selected_features_by_horizon[horizon] = passed

        comparison[f"{horizon}D"] = {
            "candidate_feature_count": len(candidate_features),
            "selected_feature_count": len(passed),
            "selection_mode": selection_mode if horizon in MODEL_HORIZONS else "comparison_only",
            "threshold": threshold,
            "top_features": report_rows[:20],
            "selected_features": passed,
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_signal_fusion_experiment.py",
        "candidate_feature_source": DEFAULT_IC_REPORT_PATH,
        "candidate_feature_count": len(candidate_features),
        "comparison": comparison,
    }
    return payload, selected_features_by_horizon


def extract_base_scheme(holding_period_report: dict[str, Any]) -> dict[str, Any]:
    base = holding_period_report["base_scheme"]
    return {
        "weighting_scheme": str(base["weighting_scheme"]),
        "selection_pct": float(base["selection_pct"]),
        "sell_buffer_pct": float(base["sell_buffer_pct"]),
        "min_trade_weight": float(base["min_trade_weight"]),
        "max_weight": float(base["max_weight"]),
        "min_holdings": int(base["min_holdings"]),
    }


def run_short_horizon_models(
    *,
    feature_matrix: pd.DataFrame,
    label_series_by_horizon: dict[int, pd.Series],
    selected_features_by_horizon: dict[int, list[str]],
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    scheme_config: dict[str, Any],
    benchmark_ticker: str,
) -> tuple[dict[str, Any], dict[int, dict[str, pd.Series]]]:
    windows = extended_walkforward_windows()
    engine = WalkForwardEngine(
        alpha_grid=DEFAULT_ALPHA_GRID,
        benchmark_ticker=benchmark_ticker,
        tracking_uri=DEFAULT_MLFLOW_TRACKING_URI,
    )
    prediction_bundles: dict[int, dict[str, pd.Series]] = {}
    reports: dict[str, Any] = {}

    all_trade_dates: list[pd.Timestamp] = []
    for horizon in MODEL_HORIZONS:
        selected_features = selected_features_by_horizon[horizon]
        if not selected_features:
            raise RuntimeError(f"No selected features available for {horizon}D.")

        X = feature_matrix.loc[:, selected_features]
        y = label_series_by_horizon[horizon]
        aligned_X, aligned_y = align_panel(X, y)

        window_reports: list[dict[str, Any]] = []
        prediction_bundle: dict[str, pd.Series] = {}
        metadata_bundle: dict[str, dict[str, Any]] = {}
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        for window in windows:
            result = engine.run_window(
                X=aligned_X,
                y=aligned_y,
                prices=None,
                window=window,
                target_horizon=f"{horizon}D",
                window_id=f"{window.window_id}_{horizon}D_alpha_enhancement",
                timestamp=timestamp,
                simulate_portfolio=False,
            )
            prediction_bundle[window.window_id] = result.test_predictions
            metadata_bundle[window.window_id] = {
                "window_id": window.window_id,
                "train_period": f"{window.train_start.isoformat()} -> {window.train_end.isoformat()}",
                "validation_period": f"{window.validation_start.isoformat()} -> {window.validation_end.isoformat()}",
                "test_period": f"{window.test_start.isoformat()} -> {window.test_end.isoformat()}",
                "best_hyperparams": float(result.best_hyperparams),
                "test_metrics": metrics_to_dict(result.test_metrics),
            }
            window_reports.append(
                {
                    "window_id": window.window_id,
                    "train_period": metadata_bundle[window.window_id]["train_period"],
                    "validation_period": metadata_bundle[window.window_id]["validation_period"],
                    "test_period": metadata_bundle[window.window_id]["test_period"],
                    "best_hyperparams": float(result.best_hyperparams),
                    "train_metrics": metrics_to_dict(result.train_metrics),
                    "validation_metrics": metrics_to_dict(result.validation_metrics),
                    "test_metrics": metrics_to_dict(result.test_metrics),
                    "train_rows": int(result.train_rows),
                    "validation_rows": int(result.validation_rows),
                    "test_rows": int(result.test_rows),
                    "mlflow": {
                        "tracking_uri": result.mlflow_run.tracking_uri if result.mlflow_run else None,
                        "experiment_name": result.mlflow_run.experiment_name if result.mlflow_run else None,
                        "experiment_id": result.mlflow_run.experiment_id if result.mlflow_run else None,
                        "run_id": result.mlflow_run.run_id if result.mlflow_run else None,
                    },
                },
            )
            all_trade_dates.extend(pd.to_datetime(result.test_predictions.index.get_level_values("trade_date")).tolist())

        prediction_bundles[horizon] = prediction_bundle
        universe_by_date = build_universe_by_date(trade_dates=sorted(set(all_trade_dates)))
        portfolio_payload = run_holding_period_experiment(
            holding_period="4W",
            prediction_bundle=prediction_bundle,
            window_metadata=metadata_bundle,
            prices=prices,
            universe_by_date=universe_by_date,
            benchmark_ticker=benchmark_ticker,
            cost_model=cost_model,
            scheme_config=scheme_config,
        )
        reports[f"{horizon}D"] = {
            "selected_feature_count": len(selected_features),
            "selected_features": selected_features,
            "walkforward": window_reports,
            "aggregate": portfolio_payload["aggregate"],
        }

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_signal_fusion_experiment.py",
        "holding_period": "4W",
        "scheme_config": scheme_config,
        "models": reports,
    }
    return payload, prediction_bundles


def run_signal_fusion(
    *,
    baseline_60d_bundle: dict[str, pd.Series],
    short_model_payload: dict[str, Any],
    model_prediction_bundles: dict[int, dict[str, pd.Series]],
    label_series_60d: pd.Series,
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    scheme_config: dict[str, Any],
    benchmark_ticker: str,
    baseline_60d_series: pd.Series,
    baseline_60d_summary: dict[str, Any],
    baseline_60d_ic: float,
) -> dict[str, Any]:
    ic_weights = build_ic_weights(
        baseline_60d_ic=baseline_60d_ic,
        short_model_payload=short_model_payload,
    )
    combined_bundles = {
        "equal_weight": combine_prediction_bundles(
            baseline_60d_bundle=baseline_60d_bundle,
            model_prediction_bundles=model_prediction_bundles,
            method="equal_weight",
            ic_weights=ic_weights,
        ),
        "ic_weighted": combine_prediction_bundles(
            baseline_60d_bundle=baseline_60d_bundle,
            model_prediction_bundles=model_prediction_bundles,
            method="ic_weighted",
            ic_weights=ic_weights,
        ),
        "recursive_overlay": combine_prediction_bundles(
            baseline_60d_bundle=baseline_60d_bundle,
            model_prediction_bundles=model_prediction_bundles,
            method="recursive_overlay",
            ic_weights=ic_weights,
        ),
    }

    all_trade_dates = {
        pd.Timestamp(value)
        for bundle in combined_bundles.values()
        for series in bundle.values()
        for value in pd.to_datetime(series.index.get_level_values("trade_date"))
    }
    universe_by_date = build_universe_by_date(trade_dates=sorted(all_trade_dates))
    fusion_reports: dict[str, Any] = {}

    for method, bundle in combined_bundles.items():
        window_metadata = build_fusion_window_metadata(bundle=bundle, label_series_60d=label_series_60d)
        portfolio_payload = run_holding_period_experiment(
            holding_period="4W",
            prediction_bundle=bundle,
            window_metadata=window_metadata,
            prices=prices,
            universe_by_date=universe_by_date,
            benchmark_ticker=benchmark_ticker,
            cost_model=cost_model,
            scheme_config=scheme_config,
        )
        competitor_series = portfolio_payload["aggregate"]["period_net_excess_series"]
        spa = run_spa_fallback(
            benchmark_series=baseline_60d_series,
            competitors={method: competitor_series},
            benchmark_name="60D_4W",
        )
        fusion_reports[method] = {
            "method": method,
            "weights": describe_method_weights(method=method, ic_weights=ic_weights),
            "aggregate": portfolio_payload["aggregate"],
            "windows": portfolio_payload["windows"],
            "spa_vs_60d": spa.to_dict(),
        }

    best_method, best_payload = select_best_fusion_method(fusion_reports)
    best_aggregate = best_payload["aggregate"]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_signal_fusion_experiment.py",
        "baseline_60d_4w": {
            "mean_test_ic": baseline_60d_ic,
            "aggregate": baseline_60d_summary,
        },
        "fusion_methods": {
            name: make_report_safe_payload(payload)
            for name, payload in fusion_reports.items()
        },
        "best_configuration": {
            "label": best_method,
            "annualized_net_excess": float(best_aggregate["annualized_net_excess"]),
            "annualized_gross_excess": float(best_aggregate["annualized_gross_excess"]),
            "sharpe": float(best_aggregate["sharpe"]),
            "bootstrap": best_aggregate["bootstrap"],
            "average_turnover": float(best_aggregate["average_turnover"]),
            "max_drawdown": float(best_aggregate["max_drawdown"]),
            "spa_vs_60d": fusion_reports[best_method]["spa_vs_60d"],
        },
        "final_recommendation": {
            "net_excess_threshold": 0.05,
            "bootstrap_ci_threshold": 0.0,
            "net_excess_pass": bool(best_aggregate["annualized_net_excess"] > 0.05),
            "bootstrap_pass": bool(best_aggregate["bootstrap"]["sharpe_ci_lower"] > 0.0),
            "recommended_method": best_method,
            "rationale": build_recommendation_text(
                method=best_method,
                aggregate=best_aggregate,
            ),
        },
    }


def build_ic_weights(
    *,
    baseline_60d_ic: float,
    short_model_payload: dict[str, Any],
) -> dict[str, float]:
    raw = {
        "60D": max(float(baseline_60d_ic), 0.0),
        "20D": max(float(short_model_payload["models"]["20D"]["aggregate"]["mean_ic"]), 0.0),
        "10D": max(float(short_model_payload["models"]["10D"]["aggregate"]["mean_ic"]), 0.0),
    }
    total = sum(raw.values())
    if total <= 0.0:
        return {key: 1.0 / len(raw) for key in raw}
    return {key: value / total for key, value in raw.items()}


def combine_prediction_bundles(
    *,
    baseline_60d_bundle: dict[str, pd.Series],
    model_prediction_bundles: dict[int, dict[str, pd.Series]],
    method: str,
    ic_weights: dict[str, float],
) -> dict[str, pd.Series]:
    combined: dict[str, pd.Series] = {}
    for window_id, scores_60d in baseline_60d_bundle.items():
        ranked_60d = cross_sectional_rank(scores_60d)
        ranked_20d = cross_sectional_rank(model_prediction_bundles[20][window_id])
        ranked_10d = cross_sectional_rank(model_prediction_bundles[10][window_id])
        frame = pd.concat(
            [
                ranked_60d.rename("60D"),
                ranked_20d.rename("20D"),
                ranked_10d.rename("10D"),
            ],
            axis=1,
        ).fillna(0.5)

        if method == "equal_weight":
            score = frame.mean(axis=1)
        elif method == "ic_weighted":
            score = (
                ic_weights["60D"] * frame["60D"]
                + ic_weights["20D"] * frame["20D"]
                + ic_weights["10D"] * frame["10D"]
            )
        elif method == "recursive_overlay":
            overlay = 0.5 * (frame["20D"] + frame["10D"])
            score = frame["60D"] + 0.35 * (overlay - 0.5)
        else:
            raise ValueError(f"Unsupported fusion method: {method}")

        combined[window_id] = cross_sectional_rank(score.rename("score"))
    return combined


def cross_sectional_rank(scores: pd.Series) -> pd.Series:
    series = pd.Series(scores, dtype=float).dropna().sort_index()
    if not isinstance(series.index, pd.MultiIndex):
        return series.rank(pct=True, method="average")
    return series.groupby(level="trade_date", sort=True).rank(pct=True, method="average")


def build_fusion_window_metadata(
    *,
    bundle: dict[str, pd.Series],
    label_series_60d: pd.Series,
) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    windows = {window.window_id: window for window in extended_walkforward_windows()}
    for window_id, predictions in bundle.items():
        aligned_index = predictions.index.intersection(label_series_60d.index)
        summary = evaluate_predictions(label_series_60d.loc[aligned_index], predictions.loc[aligned_index])
        window = windows[window_id]
        metadata[window_id] = {
            "window_id": window_id,
            "train_period": f"{window.train_start.isoformat()} -> {window.train_end.isoformat()}",
            "validation_period": f"{window.validation_start.isoformat()} -> {window.validation_end.isoformat()}",
            "test_period": f"{window.test_start.isoformat()} -> {window.test_end.isoformat()}",
            "best_hyperparams": 0.0,
            "test_metrics": metrics_to_dict(summary),
        }
    return metadata


def describe_method_weights(*, method: str, ic_weights: dict[str, float]) -> dict[str, float | str]:
    if method == "equal_weight":
        return {"60D": 1 / 3, "20D": 1 / 3, "10D": 1 / 3}
    if method == "ic_weighted":
        return dict(ic_weights)
    if method == "recursive_overlay":
        return {
            "base_60D": 1.0,
            "overlay_20D": 0.175,
            "overlay_10D": 0.175,
            "note": "60D base rank plus short-horizon timing overlay before final reranking.",
        }
    raise ValueError(f"Unsupported fusion method: {method}")


def select_best_fusion_method(fusion_reports: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    ranked = sorted(
        fusion_reports.items(),
        key=lambda item: (
            float(item[1]["aggregate"]["bootstrap"]["sharpe_ci_lower"]),
            float(item[1]["aggregate"]["annualized_net_excess"]),
            float(item[1]["aggregate"]["sharpe"]),
        ),
        reverse=True,
    )
    return ranked[0]


def build_recommendation_text(*, method: str, aggregate: dict[str, Any]) -> str:
    if aggregate["bootstrap"]["sharpe_ci_lower"] > 0.0 and aggregate["annualized_net_excess"] > 0.05:
        return f"{method} clears both the 5% net-excess gate and the bootstrap lower-bound gate."
    if aggregate["annualized_net_excess"] > 0.05:
        return f"{method} clears the net-excess gate but still leaves the bootstrap lower bound at or below zero."
    return f"{method} does not improve economics enough to justify replacing the pure 60D baseline."


if __name__ == "__main__":
    raise SystemExit(main())
