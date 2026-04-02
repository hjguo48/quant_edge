from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
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

from scripts import run_alpha_report as alpha_report_module
from scripts.run_ic_screening import write_json_atomic, write_parquet_atomic
from scripts.run_portfolio_comparison import ensure_experiment
from scripts.run_single_window_validation import (
    align_panel,
    configure_logging,
    current_git_branch,
    json_safe,
    load_retained_features,
)
from scripts.run_walkforward_backtest import (
    DEFAULT_ALL_FEATURES_PATH,
    DEFAULT_IC_REPORT_PATH,
    DEFAULT_WINDOW1_FEATURE_MATRIX_PATH,
    LOCAL_MLFLOW_URI,
    aggregate_walkforward_results,
    build_label_series_by_horizon,
    build_or_load_full_feature_matrix,
    build_or_load_multi_horizon_labels,
    build_or_load_prices,
    mlflow_payload,
    normalize_prices,
)
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.engine import WalkForwardEngine, WalkForwardWindowConfig, build_universe_by_date
from src.backtest.execution import simulate_portfolio
from src.models.baseline import DEFAULT_ALPHA_GRID

AS_OF_DATE = date(2026, 3, 31)
FEATURE_START = date(2018, 1, 1)
FEATURE_END = date(2025, 6, 30)
PRICE_END_BUFFER_DAYS = 300
TARGET_HORIZON_DAYS = 60
TARGET_HORIZON = f"{TARGET_HORIZON_DAYS}D"
REBALANCE_WEEKDAY = 4
EXPECTED_BRANCH = "feature/week10-bootstrap-fix"

DEFAULT_BASELINE_WALKFORWARD_REPORT_PATH = "data/reports/walkforward_backtest.json"
DEFAULT_EXTENDED_FEATURE_MATRIX_PATH = "data/features/extended_walkforward_feature_matrix.parquet"
DEFAULT_EXTENDED_PRICE_CACHE_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_EXTENDED_LABEL_CACHE_PATH = "data/labels/extended_walkforward_forward_returns_multi.parquet"
DEFAULT_EXTENDED_PREDICTIONS_CACHE_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_EXTENDED_METADATA_PATH = "data/backtest/extended_walkforward_model_metadata.json"
DEFAULT_EXTENDED_REPORT_PATH = "data/reports/extended_walkforward.json"
DEFAULT_EXTENDED_PORTFOLIO_REPORT_PATH = "data/reports/extended_portfolio_buffered.json"
DEFAULT_ALPHA_REPORT_V2_PATH = "data/reports/phase1_alpha_report_v2.json"

BUFFERED_SCHEME_CONFIG: dict[str, Any] = {
    "weighting_scheme": "equal_weight",
    "selection_pct": 0.20,
    "sell_buffer_pct": 0.25,
    "min_trade_weight": 0.01,
    "max_weight": 0.05,
    "min_holdings": 20,
    "description": "Equal-weight broad book with 20% entry, 25% exit buffer, and 1% trade buffer.",
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"This script must run on branch {EXPECTED_BRANCH}. Found {branch!r}.")

    baseline_walkforward_report = json.loads((REPO_ROOT / args.baseline_walkforward_report_path).read_text())
    retained_features = load_retained_features(REPO_ROOT / args.ic_report_path)
    logger.info("loaded {} retained features for extended walk-forward", len(retained_features))

    feature_matrix = build_or_load_full_feature_matrix(
        retained_features=retained_features,
        window1_feature_matrix_path=REPO_ROOT / args.window1_feature_matrix_path,
        all_features_path=REPO_ROOT / args.all_features_path,
        output_path=REPO_ROOT / args.extended_feature_matrix_path,
        start_date=FEATURE_START,
        end_date=FEATURE_END,
    )
    tickers = feature_matrix.index.get_level_values("ticker").unique().tolist()
    logger.info(
        "extended feature matrix rows={} dates={} tickers={}",
        len(feature_matrix),
        feature_matrix.index.get_level_values("trade_date").nunique(),
        len(tickers),
    )

    prices = build_or_load_prices(
        tickers=tickers,
        benchmark_ticker=args.benchmark_ticker,
        start_date=FEATURE_START,
        end_date=min(AS_OF_DATE, FEATURE_END + timedelta(days=PRICE_END_BUFFER_DAYS)),
        as_of=AS_OF_DATE,
        cache_path=REPO_ROOT / args.extended_price_cache_path,
    )
    labels = build_or_load_multi_horizon_labels(
        prices=prices,
        horizons=(TARGET_HORIZON_DAYS,),
        benchmark_ticker=args.benchmark_ticker,
        cache_path=REPO_ROOT / args.extended_label_cache_path,
        start_date=FEATURE_START,
        end_date=FEATURE_END,
    )
    label_series = build_label_series_by_horizon(labels)[TARGET_HORIZON_DAYS]
    aligned_X, aligned_y = align_panel(feature_matrix, label_series)
    logger.info(
        "aligned panel rows={} dates={} tickers={}",
        len(aligned_X),
        aligned_X.index.get_level_values("trade_date").nunique(),
        aligned_X.index.get_level_values("ticker").nunique(),
    )

    universe_by_date = build_universe_by_date(
        trade_dates=aligned_X.index.get_level_values("trade_date").unique().sort_values(),
    )
    windows = extended_walkforward_windows()
    engine = WalkForwardEngine(
        alpha_grid=DEFAULT_ALPHA_GRID,
        benchmark_ticker=args.benchmark_ticker,
        tracking_uri=LOCAL_MLFLOW_URI,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    prediction_bundle = build_or_load_predictions(
        engine=engine,
        X=aligned_X,
        y=aligned_y,
        windows=windows,
        target_horizon=TARGET_HORIZON,
        timestamp=timestamp,
        prediction_cache_path=REPO_ROOT / args.extended_predictions_cache_path,
        metadata_cache_path=REPO_ROOT / args.extended_metadata_path,
    )

    prices = normalize_prices(prices)
    cost_model = AlmgrenChrissCostModel()
    portfolio_experiment = ensure_experiment(f"portfolio_extended_{TARGET_HORIZON}_{timestamp}")
    scheme_payload = run_buffered_scheme(
        scheme_name="equal_weight_buffered",
        scheme_config=BUFFERED_SCHEME_CONFIG,
        prediction_bundle=prediction_bundle,
        prices=prices,
        cost_model=cost_model,
        universe_by_date=universe_by_date,
        benchmark_ticker=args.benchmark_ticker,
        mlflow_experiment_id=portfolio_experiment["experiment_id"],
        windows=windows,
    )

    walkforward_windows_payload = build_walkforward_window_results(
        prediction_bundle=prediction_bundle,
        scheme_payload=scheme_payload,
        windows=windows,
    )
    walkforward_aggregate = aggregate_walkforward_results(walkforward_windows_payload)
    walkforward_aggregate["average_turnover"] = float(scheme_payload["aggregate"]["average_turnover"])

    extended_report = build_extended_walkforward_report(
        args=args,
        baseline_walkforward_report=baseline_walkforward_report,
        retained_features=retained_features,
        feature_matrix=feature_matrix,
        prices=prices,
        labels=labels,
        walkforward_windows_payload=walkforward_windows_payload,
        walkforward_aggregate=walkforward_aggregate,
        scheme_payload=scheme_payload,
        cost_model=cost_model,
        portfolio_experiment=portfolio_experiment,
    )
    extended_report_path = REPO_ROOT / args.extended_report_path
    write_json_atomic(extended_report_path, json_safe(extended_report))
    logger.info("saved extended walk-forward report to {}", extended_report_path)

    portfolio_report = build_portfolio_report(
        args=args,
        scheme_payload=scheme_payload,
        target_horizon=TARGET_HORIZON,
        portfolio_experiment=portfolio_experiment,
    )
    portfolio_report_path = REPO_ROOT / args.extended_portfolio_report_path
    write_json_atomic(portfolio_report_path, json_safe(portfolio_report))
    logger.info("saved extended buffered portfolio report to {}", portfolio_report_path)

    alpha_exit = alpha_report_module.main(
        [
            "--expected-branch",
            EXPECTED_BRANCH,
            "--walkforward-report",
            str(Path(args.extended_report_path)),
            "--portfolio-report",
            str(Path(args.extended_portfolio_report_path)),
            "--prediction-cache",
            str(Path(args.extended_predictions_cache_path)),
            "--label-cache",
            str(Path(args.extended_label_cache_path)),
            "--report-path",
            str(Path(args.alpha_report_v2_path)),
        ],
    )
    if alpha_exit != 0:
        raise RuntimeError(f"run_alpha_report exited with status {alpha_exit}")

    alpha_report = json.loads((REPO_ROOT / args.alpha_report_v2_path).read_text())
    logger.info(
        "extended bootstrap CI lower={:.6f} decision={}",
        alpha_report["statistical_tests"]["bootstrap_ci"]["sharpe_ci_lower"],
        alpha_report["final_decision"]["decision"],
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extend the Ridge 60D walk-forward backtest to 8 windows and regenerate the Phase 1 alpha report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--baseline-walkforward-report-path", default=DEFAULT_BASELINE_WALKFORWARD_REPORT_PATH)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--window1-feature-matrix-path", default=DEFAULT_WINDOW1_FEATURE_MATRIX_PATH)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--extended-feature-matrix-path", default=DEFAULT_EXTENDED_FEATURE_MATRIX_PATH)
    parser.add_argument("--extended-price-cache-path", default=DEFAULT_EXTENDED_PRICE_CACHE_PATH)
    parser.add_argument("--extended-label-cache-path", default=DEFAULT_EXTENDED_LABEL_CACHE_PATH)
    parser.add_argument("--extended-predictions-cache-path", default=DEFAULT_EXTENDED_PREDICTIONS_CACHE_PATH)
    parser.add_argument("--extended-metadata-path", default=DEFAULT_EXTENDED_METADATA_PATH)
    parser.add_argument("--extended-report-path", default=DEFAULT_EXTENDED_REPORT_PATH)
    parser.add_argument("--extended-portfolio-report-path", default=DEFAULT_EXTENDED_PORTFOLIO_REPORT_PATH)
    parser.add_argument("--alpha-report-v2-path", default=DEFAULT_ALPHA_REPORT_V2_PATH)
    return parser.parse_args(argv)


def extended_walkforward_windows() -> list[WalkForwardWindowConfig]:
    return [
        WalkForwardWindowConfig(
            window_id="W1",
            train_start=date(2018, 1, 1),
            train_end=date(2020, 12, 31),
            validation_start=date(2021, 1, 1),
            validation_end=date(2021, 6, 30),
            test_start=date(2021, 7, 1),
            test_end=date(2021, 12, 31),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
        WalkForwardWindowConfig(
            window_id="W2",
            train_start=date(2018, 7, 1),
            train_end=date(2021, 6, 30),
            validation_start=date(2021, 7, 1),
            validation_end=date(2021, 12, 31),
            test_start=date(2022, 1, 1),
            test_end=date(2022, 6, 30),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
        WalkForwardWindowConfig(
            window_id="W3",
            train_start=date(2019, 1, 1),
            train_end=date(2021, 12, 31),
            validation_start=date(2022, 1, 1),
            validation_end=date(2022, 6, 30),
            test_start=date(2022, 7, 1),
            test_end=date(2022, 12, 31),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
        WalkForwardWindowConfig(
            window_id="W4",
            train_start=date(2019, 7, 1),
            train_end=date(2022, 6, 30),
            validation_start=date(2022, 7, 1),
            validation_end=date(2022, 12, 31),
            test_start=date(2023, 1, 1),
            test_end=date(2023, 6, 30),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
        WalkForwardWindowConfig(
            window_id="W5",
            train_start=date(2020, 1, 1),
            train_end=date(2022, 12, 31),
            validation_start=date(2023, 1, 1),
            validation_end=date(2023, 6, 30),
            test_start=date(2023, 7, 1),
            test_end=date(2023, 12, 31),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
        WalkForwardWindowConfig(
            window_id="W6",
            train_start=date(2020, 7, 1),
            train_end=date(2023, 6, 30),
            validation_start=date(2023, 7, 1),
            validation_end=date(2023, 12, 31),
            test_start=date(2024, 1, 1),
            test_end=date(2024, 6, 30),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
        WalkForwardWindowConfig(
            window_id="W7",
            train_start=date(2021, 1, 1),
            train_end=date(2023, 12, 31),
            validation_start=date(2024, 1, 1),
            validation_end=date(2024, 6, 30),
            test_start=date(2024, 7, 1),
            test_end=date(2024, 12, 31),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
        WalkForwardWindowConfig(
            window_id="W8",
            train_start=date(2021, 7, 1),
            train_end=date(2024, 6, 30),
            validation_start=date(2024, 7, 1),
            validation_end=date(2024, 12, 31),
            test_start=date(2025, 1, 1),
            test_end=date(2025, 6, 30),
            rebalance_weekday=REBALANCE_WEEKDAY,
        ),
    ]


def build_or_load_predictions(
    *,
    engine: WalkForwardEngine,
    X: pd.DataFrame,
    y: pd.Series,
    windows: list[WalkForwardWindowConfig],
    target_horizon: str,
    timestamp: str,
    prediction_cache_path: Path,
    metadata_cache_path: Path,
) -> dict[str, Any]:
    if prediction_cache_path.exists() and metadata_cache_path.exists():
        logger.info("loading cached extended walk-forward predictions from {}", prediction_cache_path)
        prediction_frame = pd.read_parquet(prediction_cache_path)
        metadata = json.loads(metadata_cache_path.read_text())
    else:
        records: list[dict[str, Any]] = []
        windows_metadata: list[dict[str, Any]] = []
        for window in windows:
            result = engine.run_window(
                X=X,
                y=y,
                prices=None,
                window=window,
                target_horizon=target_horizon,
                window_id=f"{window.window_id}_extended_predictions",
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
                    "train_rows": int(result.train_rows),
                    "validation_rows": int(result.validation_rows),
                    "test_rows": int(result.test_rows),
                    "train_metrics": result.train_metrics.to_dict(),
                    "validation_metrics": result.validation_metrics.to_dict(),
                    "test_metrics": result.test_metrics.to_dict(),
                    "mlflow": mlflow_payload(result),
                },
            )

        prediction_frame = pd.DataFrame(records).sort_values(["window_id", "trade_date", "ticker"]).reset_index(drop=True)
        metadata = {
            "target_horizon": target_horizon,
            "windows": windows_metadata,
        }
        write_parquet_atomic(prediction_frame, prediction_cache_path)
        write_json_atomic(metadata_cache_path, json_safe(metadata))
        logger.info("saved extended prediction cache to {}", prediction_cache_path)

    predictions_by_window: dict[str, pd.Series] = {}
    for window_id, frame in prediction_frame.groupby("window_id", sort=True):
        predictions_by_window[str(window_id)] = frame.set_index(["trade_date", "ticker"])["score"].sort_index()

    metadata_by_window = {entry["window_id"]: entry for entry in metadata["windows"]}
    return {
        "target_horizon": metadata["target_horizon"],
        "predictions": predictions_by_window,
        "windows": metadata_by_window,
    }


def run_buffered_scheme(
    *,
    scheme_name: str,
    scheme_config: dict[str, Any],
    prediction_bundle: dict[str, Any],
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    universe_by_date: dict[pd.Timestamp, set[str]],
    benchmark_ticker: str,
    mlflow_experiment_id: str,
    windows: list[WalkForwardWindowConfig],
) -> dict[str, Any]:
    window_results: list[dict[str, Any]] = []
    for window in windows:
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
            sell_buffer_pct=float(scheme_config["sell_buffer_pct"]) if scheme_config.get("sell_buffer_pct") is not None else None,
            min_trade_weight=float(scheme_config["min_trade_weight"]),
            max_weight=float(scheme_config["max_weight"]),
            min_holdings=int(scheme_config["min_holdings"]),
        )
        window_results.append(
            {
                "window_id": window.window_id,
                "train_period": metadata["train_period"],
                "validation_period": metadata["validation_period"],
                "test_period": metadata["test_period"],
                "best_hyperparams": metadata["best_hyperparams"],
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
        target_horizon=prediction_bundle["target_horizon"],
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
    target_horizon: str,
) -> dict[str, Any]:
    mlflow.set_tracking_uri(LOCAL_MLFLOW_URI)
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.set_tags(
            {
                "run_kind": "portfolio_comparison",
                "scheme": scheme_name,
                "target_horizon": target_horizon,
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
            f"portfolio/{scheme_name}_extended.json",
        )

    return {
        "tracking_uri": LOCAL_MLFLOW_URI,
        "experiment_id": experiment_id,
        "run_id": run.info.run_id,
    }


def build_walkforward_window_results(
    *,
    prediction_bundle: dict[str, Any],
    scheme_payload: dict[str, Any],
    windows: list[WalkForwardWindowConfig],
) -> list[dict[str, Any]]:
    portfolio_by_window = {row["window_id"]: row for row in scheme_payload["windows"]}
    payload: list[dict[str, Any]] = []
    for window in windows:
        metadata = prediction_bundle["windows"][window.window_id]
        portfolio_row = portfolio_by_window[window.window_id]
        payload.append(
            {
                "window_id": window.window_id,
                "train_period": metadata["train_period"],
                "validation_period": metadata["validation_period"],
                "test_period": metadata["test_period"],
                "best_hyperparams": metadata["best_hyperparams"],
                "train_rows": int(metadata.get("train_rows", 0)),
                "validation_rows": int(metadata.get("validation_rows", 0)),
                "test_rows": int(metadata.get("test_rows", len(prediction_bundle["predictions"][window.window_id]))),
                "train_metrics": metadata["train_metrics"],
                "validation_metrics": metadata["validation_metrics"],
                "test_metrics": metadata["test_metrics"],
                "portfolio": portfolio_row["portfolio"],
                "mlflow": metadata["mlflow"],
            },
        )
    return payload


def build_extended_walkforward_report(
    *,
    args: argparse.Namespace,
    baseline_walkforward_report: dict[str, Any],
    retained_features: list[str],
    feature_matrix: pd.DataFrame,
    prices: pd.DataFrame,
    labels: pd.DataFrame,
    walkforward_windows_payload: list[dict[str, Any]],
    walkforward_aggregate: dict[str, Any],
    scheme_payload: dict[str, Any],
    cost_model: AlmgrenChrissCostModel,
    portfolio_experiment: dict[str, str],
) -> dict[str, Any]:
    mean_oos_ic = float(walkforward_aggregate["mean_test_ic"])
    windows_above_threshold = int(sum(row["test_metrics"]["ic"] > 0.03 for row in walkforward_windows_payload))
    annualized_excess_net = float(scheme_payload["aggregate"]["annualized_net_excess"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_extended_walkforward.py",
        "inputs": {
            "baseline_walkforward_report_path": str(REPO_ROOT / args.baseline_walkforward_report_path),
            "ic_report_path": str(REPO_ROOT / args.ic_report_path),
            "window1_feature_matrix_path": str(REPO_ROOT / args.window1_feature_matrix_path),
            "all_features_path": str(REPO_ROOT / args.all_features_path),
            "extended_feature_matrix_path": str(REPO_ROOT / args.extended_feature_matrix_path),
            "extended_price_cache_path": str(REPO_ROOT / args.extended_price_cache_path),
            "extended_label_cache_path": str(REPO_ROOT / args.extended_label_cache_path),
            "extended_predictions_cache_path": str(REPO_ROOT / args.extended_predictions_cache_path),
            "extended_metadata_path": str(REPO_ROOT / args.extended_metadata_path),
            "extended_portfolio_report_path": str(REPO_ROOT / args.extended_portfolio_report_path),
            "alpha_report_v2_path": str(REPO_ROOT / args.alpha_report_v2_path),
            "benchmark_ticker": args.benchmark_ticker.upper(),
            "retained_feature_count": len(retained_features),
            "retained_features": retained_features,
            "target_horizon": TARGET_HORIZON,
            "buffered_scheme_config": BUFFERED_SCHEME_CONFIG,
        },
        "data_summary": {
            "feature_rows": int(len(feature_matrix)),
            "feature_dates": int(feature_matrix.index.get_level_values("trade_date").nunique()),
            "feature_tickers": int(feature_matrix.index.get_level_values("ticker").nunique()),
            "feature_min_date": feature_matrix.index.get_level_values("trade_date").min().date().isoformat(),
            "feature_max_date": feature_matrix.index.get_level_values("trade_date").max().date().isoformat(),
            "price_rows": int(len(prices)),
            "price_dates": int(pd.to_datetime(prices["trade_date"]).nunique()),
            "price_tickers": int(prices["ticker"].nunique()),
            "label_rows": int(len(labels)),
            "label_horizons": sorted(labels["horizon"].drop_duplicates().astype(int).tolist()),
        },
        "horizon_experiment": baseline_walkforward_report["horizon_experiment"],
        "walkforward": {
            "windows": walkforward_windows_payload,
            "aggregate": walkforward_aggregate,
        },
        "portfolio": {
            "scheme": "equal_weight_buffered",
            "config": BUFFERED_SCHEME_CONFIG,
            "aggregate": scheme_payload["aggregate"],
            "windows": scheme_payload["windows"],
        },
        "cost_model": cost_model.get_params(),
        "execution_assumptions": {
            "signal_timing": "Friday close signal, next trading day execution",
            "execution_price_proxy": "50% adjusted open + 50% adjusted OHLC typical price",
            "portfolio_construction": "equal-weight buffered long-only portfolio",
            "buffer_rules": {
                "selection_pct": BUFFERED_SCHEME_CONFIG["selection_pct"],
                "sell_buffer_pct": BUFFERED_SCHEME_CONFIG["sell_buffer_pct"],
                "min_trade_weight": BUFFERED_SCHEME_CONFIG["min_trade_weight"],
            },
            "universe_filtering": (
                "The engine attempts to intersect signals with universe_membership, "
                "but falls back to the model signal cross-section when the overlap is below 100 names. "
                "The current universe_membership history remains too sparse to act as a hard filter."
            ),
        },
        "go_nogo_preview": {
            "mean_oos_ic": mean_oos_ic,
            "ic_threshold": 0.03,
            "windows_above_threshold": windows_above_threshold,
            "required_windows_above_threshold": 3,
            "ic_pass": bool(mean_oos_ic > 0.03 and windows_above_threshold >= 3),
            "cost_adjusted_excess": annualized_excess_net,
            "excess_threshold": 0.05,
            "excess_pass": bool(annualized_excess_net > 0.05),
        },
        "mlflow": {
            "tracking_uri": LOCAL_MLFLOW_URI,
            "window_run_ids": [
                {
                    "window_id": row["window_id"],
                    "run_id": row["mlflow"]["run_id"],
                    "experiment_name": row["mlflow"]["experiment_name"],
                }
                for row in walkforward_windows_payload
            ],
            "portfolio_experiment": portfolio_experiment,
            "portfolio_run": scheme_payload["mlflow"],
        },
    }


def build_portfolio_report(
    *,
    args: argparse.Namespace,
    scheme_payload: dict[str, Any],
    target_horizon: str,
    portfolio_experiment: dict[str, str],
) -> dict[str, Any]:
    aggregate = scheme_payload["aggregate"]
    comparison_row = {
        "scheme": "equal_weight_buffered",
        "mean_test_ic": float(aggregate["mean_ic"]),
        "ann_gross_excess": float(aggregate["annualized_gross_excess"]),
        "ann_net_excess": float(aggregate["annualized_net_excess"]),
        "cost_drag": float(aggregate["total_cost_drag"]),
        "avg_turnover": float(aggregate["average_turnover"]),
    }
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_extended_walkforward.py",
        "inputs": {
            "extended_report_path": str(REPO_ROOT / args.extended_report_path),
            "target_horizon": target_horizon,
            "benchmark_ticker": args.benchmark_ticker.upper(),
        },
        "schemes": {
            "equal_weight_buffered": scheme_payload,
        },
        "comparison_matrix": [comparison_row],
        "best_scheme": "equal_weight_buffered",
        "go_nogo_update": {
            "best_net_excess": float(aggregate["annualized_net_excess"]),
            "threshold": 0.05,
            "pass": bool(aggregate["annualized_net_excess"] > 0.05),
        },
        "mlflow": {
            "tracking_uri": LOCAL_MLFLOW_URI,
            "experiment_name": portfolio_experiment["experiment_name"],
            "experiment_id": portfolio_experiment["experiment_id"],
            "run_id": scheme_payload["mlflow"]["run_id"],
        },
    }


def annualize_return(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


if __name__ == "__main__":
    raise SystemExit(main())
