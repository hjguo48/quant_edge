from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import mlflow
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic, write_parquet_atomic
from scripts.run_single_window_validation import (
    align_panel,
    configure_logging,
    current_git_branch,
    json_safe,
)
from scripts.run_walkforward_backtest import (
    DEFAULT_FULL_FEATURE_MATRIX_PATH,
    DEFAULT_MULTI_LABEL_CACHE_PATH,
    DEFAULT_PRICE_CACHE_PATH,
    LOCAL_MLFLOW_URI,
    build_label_series_by_horizon,
    build_or_load_multi_horizon_labels,
    build_or_load_prices,
    build_or_load_full_feature_matrix,
    normalize_prices,
    walkforward_windows,
)
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.engine import WalkForwardEngine, build_universe_by_date
from src.backtest.execution import simulate_portfolio
from src.mlflow_config import ensure_experiment as ensure_mlflow_experiment, setup_mlflow
from src.models.baseline import DEFAULT_ALPHA_GRID

AS_OF_DATE = pd.Timestamp("2026-03-31").date()
DEFAULT_WEEK8_REPORT_PATH = "data/reports/walkforward_backtest.json"
DEFAULT_PREDICTIONS_CACHE_PATH = "data/backtest/portfolio_comparison_predictions.parquet"
DEFAULT_MODEL_METADATA_PATH = "data/backtest/portfolio_comparison_model_metadata.json"
DEFAULT_REPORT_PATH = "data/reports/portfolio_comparison.json"
DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_v2.csv"
DEFAULT_ALL_FEATURES_PATH = "data/features/all_features.parquet"
DEFAULT_WINDOW1_FEATURE_MATRIX_PATH = "data/features/window1_feature_matrix.parquet"

SCHEME_CONFIGS: dict[str, dict[str, Any]] = {
    "equal_weight": {
        "weighting_scheme": "equal_weight",
        "selection_pct": 0.10,
        "sell_buffer_pct": None,
        "min_trade_weight": 0.0,
        "max_weight": 0.05,
        "min_holdings": 20,
        "description": "Week 8 baseline: equal-weight top decile with full weekly rebalance.",
    },
    "equal_weight_buffered": {
        "weighting_scheme": "equal_weight",
        "selection_pct": 0.20,
        "sell_buffer_pct": 0.25,
        "min_trade_weight": 0.01,
        "max_weight": 0.05,
        "min_holdings": 20,
        "description": "Equal-weight broad book with 20% entry, 25% exit buffer, and 1% trade buffer.",
    },
    "vol_inverse": {
        "weighting_scheme": "vol_inverse",
        "selection_pct": 0.10,
        "sell_buffer_pct": None,
        "min_trade_weight": 0.0,
        "max_weight": 0.05,
        "min_holdings": 20,
        "description": "Top decile weighted by inverse 20D volatility.",
    },
    "vol_inverse_buffered": {
        "weighting_scheme": "vol_inverse",
        "selection_pct": 0.20,
        "sell_buffer_pct": 0.25,
        "min_trade_weight": 0.01,
        "max_weight": 0.05,
        "min_holdings": 20,
        "description": "Inverse-vol broad book with sell buffer and 1% trade buffer.",
    },
    "black_litterman": {
        "weighting_scheme": "black_litterman",
        "selection_pct": 0.20,
        "sell_buffer_pct": 0.25,
        "min_trade_weight": 0.01,
        "max_weight": 0.05,
        "min_holdings": 20,
        "bl_lookback_days": 60,
        "description": "Long-only capped Black-Litterman with model-score views on the top 20% candidate set.",
    },
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    if current_git_branch() != "feature/week9-portfolio-optimization":
        raise RuntimeError("This script must run on branch feature/week9-portfolio-optimization.")

    week8_report = json.loads((REPO_ROOT / args.week8_report_path).read_text())
    optimal_horizon = str(week8_report["horizon_experiment"]["optimal_horizon"])
    optimal_horizon_days = int(optimal_horizon.removesuffix("D"))
    logger.info("loaded Week 8 report with optimal horizon {}", optimal_horizon)

    retained_features = list(week8_report["inputs"]["retained_features"])
    full_feature_matrix = build_or_load_full_feature_matrix(
        retained_features=retained_features,
        window1_feature_matrix_path=REPO_ROOT / args.window1_feature_matrix_path,
        all_features_path=REPO_ROOT / args.all_features_path,
        output_path=REPO_ROOT / args.full_feature_matrix_path,
        start_date=pd.Timestamp("2018-01-01").date(),
        end_date=pd.Timestamp("2023-12-31").date(),
    )
    tickers = full_feature_matrix.index.get_level_values("ticker").unique().tolist()
    prices = build_or_load_prices(
        tickers=tickers,
        benchmark_ticker=args.benchmark_ticker,
        start_date=pd.Timestamp("2018-01-01").date(),
        end_date=pd.Timestamp("2024-12-31").date(),
        as_of=AS_OF_DATE,
        cache_path=REPO_ROOT / args.price_cache_path,
    )
    labels = build_or_load_multi_horizon_labels(
        prices=prices,
        horizons=(optimal_horizon_days,),
        benchmark_ticker=args.benchmark_ticker,
        cache_path=REPO_ROOT / args.label_cache_path,
        start_date=pd.Timestamp("2018-01-01").date(),
        end_date=pd.Timestamp("2023-12-31").date(),
    )
    label_series = build_label_series_by_horizon(labels)[optimal_horizon_days]
    aligned_X, aligned_y = align_panel(full_feature_matrix, label_series)
    universe_by_date = build_universe_by_date(
        trade_dates=aligned_X.index.get_level_values("trade_date").unique().sort_values(),
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    engine = WalkForwardEngine(
        alpha_grid=DEFAULT_ALPHA_GRID,
        benchmark_ticker=args.benchmark_ticker,
        tracking_uri=LOCAL_MLFLOW_URI,
    )

    prediction_bundle = build_or_load_predictions(
        engine=engine,
        X=aligned_X,
        y=aligned_y,
        target_horizon=optimal_horizon,
        timestamp=timestamp,
        prediction_cache_path=REPO_ROOT / args.predictions_cache_path,
        metadata_cache_path=REPO_ROOT / args.model_metadata_path,
    )
    prices = normalize_prices(prices)
    cost_model = AlmgrenChrissCostModel()

    comparison_experiment = ensure_experiment(f"portfolio_comparison_{optimal_horizon}_{timestamp}")
    scheme_reports: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for scheme_name, config in SCHEME_CONFIGS.items():
        logger.info("simulating scheme {}", scheme_name)
        scheme_payload = run_scheme(
            scheme_name=scheme_name,
            scheme_config=config,
            prediction_bundle=prediction_bundle,
            prices=prices,
            cost_model=cost_model,
            universe_by_date=universe_by_date,
            benchmark_ticker=args.benchmark_ticker,
            mlflow_experiment_id=comparison_experiment["experiment_id"],
        )
        scheme_reports[scheme_name] = scheme_payload
        comparison_rows.append(
            {
                "scheme": scheme_name,
                "mean_test_ic": float(scheme_payload["aggregate"]["mean_ic"]),
                "ann_gross_excess": float(scheme_payload["aggregate"]["annualized_gross_excess"]),
                "ann_net_excess": float(scheme_payload["aggregate"]["annualized_net_excess"]),
                "cost_drag": float(scheme_payload["aggregate"]["total_cost_drag"]),
                "avg_turnover": float(scheme_payload["aggregate"]["average_turnover"]),
            },
        )

    comparison_rows.sort(key=lambda row: row["ann_net_excess"], reverse=True)
    best_scheme = comparison_rows[0]["scheme"]
    best_net_excess = float(comparison_rows[0]["ann_net_excess"])

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_portfolio_comparison.py",
        "inputs": {
            "week8_report_path": str(REPO_ROOT / args.week8_report_path),
            "prediction_cache_path": str(REPO_ROOT / args.predictions_cache_path),
            "model_metadata_path": str(REPO_ROOT / args.model_metadata_path),
            "price_cache_path": str(REPO_ROOT / args.price_cache_path),
            "label_cache_path": str(REPO_ROOT / args.label_cache_path),
            "report_path": str(REPO_ROOT / args.report_path),
            "optimal_horizon": optimal_horizon,
            "benchmark_ticker": args.benchmark_ticker.upper(),
        },
        "schemes": scheme_reports,
        "comparison_matrix": comparison_rows,
        "best_scheme": best_scheme,
        "go_nogo_update": {
            "best_net_excess": best_net_excess,
            "threshold": 0.05,
            "pass": bool(best_net_excess > 0.05),
        },
        "mlflow": {
            "tracking_uri": LOCAL_MLFLOW_URI,
            "experiment_name": comparison_experiment["experiment_name"],
            "experiment_id": comparison_experiment["experiment_id"],
        },
    }

    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved portfolio comparison report to {}", report_path)
    logger.info(
        "best scheme={} annualized_net_excess={:.6f} pass={}",
        best_scheme,
        best_net_excess,
        report["go_nogo_update"]["pass"],
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare portfolio weighting and turnover-control schemes on the Week 8 walk-forward windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--week8-report-path", default=DEFAULT_WEEK8_REPORT_PATH)
    parser.add_argument("--window1-feature-matrix-path", default=DEFAULT_WINDOW1_FEATURE_MATRIX_PATH)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--full-feature-matrix-path", default=DEFAULT_FULL_FEATURE_MATRIX_PATH)
    parser.add_argument("--price-cache-path", default=DEFAULT_PRICE_CACHE_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_MULTI_LABEL_CACHE_PATH)
    parser.add_argument("--predictions-cache-path", default=DEFAULT_PREDICTIONS_CACHE_PATH)
    parser.add_argument("--model-metadata-path", default=DEFAULT_MODEL_METADATA_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def build_or_load_predictions(
    *,
    engine: WalkForwardEngine,
    X: pd.DataFrame,
    y: pd.Series,
    target_horizon: str,
    timestamp: str,
    prediction_cache_path: Path,
    metadata_cache_path: Path,
) -> dict[str, Any]:
    if prediction_cache_path.exists() and metadata_cache_path.exists():
        logger.info("loading cached walk-forward predictions from {}", prediction_cache_path)
        prediction_frame = pd.read_parquet(prediction_cache_path)
        metadata = json.loads(metadata_cache_path.read_text())
    else:
        records: list[dict[str, Any]] = []
        windows_metadata: list[dict[str, Any]] = []
        for window in walkforward_windows():
            result = engine.run_window(
                X=X,
                y=y,
                prices=None,
                window=window,
                target_horizon=target_horizon,
                window_id=f"{window.window_id}_predictions",
                timestamp=timestamp,
                simulate_portfolio=False,
            )
            for index_key, score in result.test_predictions.items():
                trade_date, ticker = index_key
                records.append(
                    {
                        "window_id": window.window_id,
                        "trade_date": pd.Timestamp(trade_date),
                        "ticker": str(ticker).upper(),
                        "score": float(score),
                    },
                )
            windows_metadata.append(
                {
                    "window_id": window.window_id,
                    "train_period": f"{window.train_start.isoformat()} -> {window.train_end.isoformat()}",
                    "validation_period": f"{window.validation_start.isoformat()} -> {window.validation_end.isoformat()}",
                    "test_period": f"{window.test_start.isoformat()} -> {window.test_end.isoformat()}",
                    "best_hyperparams": float(result.best_hyperparams),
                    "train_metrics": result.train_metrics.to_dict(),
                    "validation_metrics": result.validation_metrics.to_dict(),
                    "test_metrics": result.test_metrics.to_dict(),
                    "mlflow": {
                        "tracking_uri": result.mlflow_run.tracking_uri if result.mlflow_run else None,
                        "experiment_name": result.mlflow_run.experiment_name if result.mlflow_run else None,
                        "experiment_id": result.mlflow_run.experiment_id if result.mlflow_run else None,
                        "run_id": result.mlflow_run.run_id if result.mlflow_run else None,
                    },
                },
            )

        prediction_frame = pd.DataFrame(records).sort_values(["window_id", "trade_date", "ticker"]).reset_index(drop=True)
        metadata = {"target_horizon": target_horizon, "windows": windows_metadata}
        write_parquet_atomic(prediction_frame, prediction_cache_path)
        write_json_atomic(metadata_cache_path, json_safe(metadata))
        logger.info("saved prediction cache to {}", prediction_cache_path)

    series_by_window: dict[str, pd.Series] = {}
    for window_id, frame in prediction_frame.groupby("window_id", sort=True):
        series = frame.set_index(["trade_date", "ticker"])["score"].sort_index()
        series_by_window[str(window_id)] = series

    metadata_by_window = {entry["window_id"]: entry for entry in metadata["windows"]}
    return {
        "target_horizon": metadata["target_horizon"],
        "predictions": series_by_window,
        "windows": metadata_by_window,
    }


def run_scheme(
    *,
    scheme_name: str,
    scheme_config: dict[str, Any],
    prediction_bundle: dict[str, Any],
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    universe_by_date: dict[pd.Timestamp, set[str]],
    benchmark_ticker: str,
    mlflow_experiment_id: str,
) -> dict[str, Any]:
    window_results: list[dict[str, Any]] = []
    for window in walkforward_windows():
        metadata = prediction_bundle["windows"][window.window_id]
        predictions = prediction_bundle["predictions"][window.window_id]
        portfolio = simulate_portfolio(
            predictions=predictions,
            prices=prices,
            cost_model=cost_model,
            weighting_scheme=scheme_config["weighting_scheme"],
            benchmark_ticker=benchmark_ticker,
            universe_by_date=universe_by_date,
            selection_pct=float(scheme_config["selection_pct"]),
            sell_buffer_pct=scheme_config.get("sell_buffer_pct"),
            min_trade_weight=float(scheme_config["min_trade_weight"]),
            max_weight=float(scheme_config["max_weight"]),
            min_holdings=int(scheme_config["min_holdings"]),
            bl_lookback_days=int(scheme_config.get("bl_lookback_days", 60)),
        )
        window_results.append(
            {
                "window_id": window.window_id,
                "train_period": metadata["train_period"],
                "validation_period": metadata["validation_period"],
                "test_period": metadata["test_period"],
                "best_hyperparams": metadata.get("best_hyperparams", metadata.get("best_alpha")),
                "test_ic": float(metadata["test_metrics"]["ic"]),
                "test_rank_ic": float(metadata["test_metrics"]["rank_ic"]),
                "test_icir": float(metadata["test_metrics"]["icir"]),
                "test_hit_rate": float(metadata["test_metrics"]["hit_rate"]),
                "gross_return": float(portfolio.gross_return),
                "net_return": float(portfolio.net_return),
                "annualized_gross_excess": float(portfolio.annualized_excess_gross),
                "annualized_net_excess": float(portfolio.annualized_excess_net),
                "turnover": float(portfolio.average_turnover),
                "portfolio": portfolio.to_dict(),
                "model_mlflow": metadata["mlflow"],
            },
        )

    aggregate = aggregate_scheme_results(window_results)
    mlflow_run = log_scheme_run(
        scheme_name=scheme_name,
        scheme_config=scheme_config,
        aggregate=aggregate,
        window_results=window_results,
        experiment_id=mlflow_experiment_id,
    )
    return {
        "config": scheme_config,
        "aggregate": aggregate,
        "windows": window_results,
        "mlflow": mlflow_run,
    }


def aggregate_scheme_results(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    periods: list[dict[str, Any]] = []
    cost_breakdown = {
        "commission": 0.0,
        "spread": 0.0,
        "temporary_impact": 0.0,
        "permanent_impact": 0.0,
        "gap_penalty": 0.0,
        "total": 0.0,
    }
    for row in window_results:
        portfolio = row["portfolio"]
        for period in portfolio["periods"]:
            periods.append({"window_id": row["window_id"], **period})
        for key in cost_breakdown:
            cost_breakdown[key] += float(portfolio["cost_breakdown"].get(key, 0.0))

    periods_frame = pd.DataFrame(periods)
    if periods_frame.empty:
        annualized_gross = 0.0
        annualized_net = 0.0
        annualized_benchmark = 0.0
    else:
        periods_frame["execution_date"] = pd.to_datetime(periods_frame["execution_date"])
        periods_frame["exit_date"] = pd.to_datetime(periods_frame["exit_date"])
        periods_frame.sort_values(["execution_date", "window_id"], inplace=True)
        gross_curve = (1.0 + periods_frame["gross_return"]).cumprod()
        net_curve = (1.0 + periods_frame["net_return"]).cumprod()
        benchmark_curve = (1.0 + periods_frame["benchmark_return"]).cumprod()
        total_days = max((periods_frame["exit_date"].iloc[-1] - periods_frame["execution_date"].iloc[0]).days, 1)
        annualized_gross = annualize_return(float(gross_curve.iloc[-1] - 1.0), total_days)
        annualized_net = annualize_return(float(net_curve.iloc[-1] - 1.0), total_days)
        annualized_benchmark = annualize_return(float(benchmark_curve.iloc[-1] - 1.0), total_days)

    return {
        "mean_ic": float(np.mean([row["test_ic"] for row in window_results])),
        "mean_rank_ic": float(np.mean([row["test_rank_ic"] for row in window_results])),
        "mean_icir": float(np.mean([row["test_icir"] for row in window_results])),
        "mean_hit_rate": float(np.mean([row["test_hit_rate"] for row in window_results])),
        "annualized_gross_excess": float(annualized_gross - annualized_benchmark),
        "annualized_net_excess": float(annualized_net - annualized_benchmark),
        "annualized_gross_return": float(annualized_gross),
        "annualized_net_return": float(annualized_net),
        "annualized_benchmark_return": float(annualized_benchmark),
        "total_cost_drag": float(annualized_gross - annualized_net),
        "average_turnover": float(np.mean([row["turnover"] for row in window_results])),
        "cost_breakdown": {key: float(value) for key, value in cost_breakdown.items()},
        "window_count": int(len(window_results)),
    }


def log_scheme_run(
    *,
    scheme_name: str,
    scheme_config: dict[str, Any],
    aggregate: dict[str, Any],
    window_results: list[dict[str, Any]],
    experiment_id: str,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(LOCAL_MLFLOW_URI)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(
            {
                "run_kind": "portfolio_comparison",
                "scheme": scheme_name,
                "target_horizon": "from_week8_report",
            },
        )
        mlflow.log_params({key: str(value) for key, value in scheme_config.items() if key != "description"})
        mlflow.log_param("description", scheme_config["description"])
        mlflow.log_metrics(
            {
                "mean_ic": float(aggregate["mean_ic"]),
                "mean_rank_ic": float(aggregate["mean_rank_ic"]),
                "mean_icir": float(aggregate["mean_icir"]),
                "mean_hit_rate": float(aggregate["mean_hit_rate"]),
                "annualized_gross_excess": float(aggregate["annualized_gross_excess"]),
                "annualized_net_excess": float(aggregate["annualized_net_excess"]),
                "total_cost_drag": float(aggregate["total_cost_drag"]),
                "average_turnover": float(aggregate["average_turnover"]),
            },
        )
        for window in window_results:
            mlflow.log_metric(f"{window['window_id']}_net_excess", float(window["annualized_net_excess"]))
            mlflow.log_metric(f"{window['window_id']}_gross_return", float(window["gross_return"]))
            mlflow.log_metric(f"{window['window_id']}_net_return", float(window["net_return"]))

        mlflow.log_dict(
            {
                "scheme": scheme_name,
                "config": scheme_config,
                "aggregate": aggregate,
                "windows": window_results,
            },
            f"portfolio/{scheme_name}.json",
        )

    return {
        "tracking_uri": LOCAL_MLFLOW_URI,
        "experiment_id": experiment_id,
        "run_id": run.info.run_id,
    }


def ensure_experiment(experiment_name: str) -> dict[str, str]:
    tracking_uri = setup_mlflow(tracking_uri=LOCAL_MLFLOW_URI)
    experiment_id = ensure_mlflow_experiment(experiment_name, tracking_uri=tracking_uri)
    return {"experiment_name": experiment_name, "experiment_id": experiment_id}


def annualize_return(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


if __name__ == "__main__":
    raise SystemExit(main())
