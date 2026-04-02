from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
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

from scripts.run_ic_screening import write_json_atomic, write_parquet_atomic
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
from src.backtest.engine import (
    WalkForwardEngine,
    WalkForwardWindowConfig,
    build_universe_by_date,
)
from src.data.db.pit import get_prices_pit
from src.labels.forward_returns import compute_forward_returns
from src.mlflow_config import get_mlflow_tracking_uri
from src.models.baseline import DEFAULT_ALPHA_GRID

AS_OF_DATE = date(2026, 3, 31)
GLOBAL_START = date(2018, 1, 1)
GLOBAL_END = date(2023, 12, 31)
REBALANCE_WEEKDAY = 4
HORIZONS = (1, 2, 5, 10, 20, 60)
BENCHMARK_TICKER = "SPY"

DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_v2.csv"
DEFAULT_WINDOW1_FEATURE_MATRIX_PATH = "data/features/window1_feature_matrix.parquet"
DEFAULT_ALL_FEATURES_PATH = "data/features/all_features.parquet"
DEFAULT_FULL_FEATURE_MATRIX_PATH = "data/features/walkforward_feature_matrix.parquet"
DEFAULT_PRICE_CACHE_PATH = "data/backtest/walkforward_prices.parquet"
DEFAULT_MULTI_LABEL_CACHE_PATH = "data/labels/walkforward_forward_returns_multi.parquet"
DEFAULT_REPORT_PATH = "data/reports/walkforward_backtest.json"
LOCAL_MLFLOW_URI = get_mlflow_tracking_uri()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    if current_git_branch() != "feature/week8-walkforward-backtest":
        raise RuntimeError("This script must run on branch feature/week8-walkforward-backtest.")

    retained_features = load_retained_features(REPO_ROOT / args.ic_report_path)
    logger.info("loaded {} retained features", len(retained_features))

    full_feature_matrix = build_or_load_full_feature_matrix(
        retained_features=retained_features,
        window1_feature_matrix_path=REPO_ROOT / args.window1_feature_matrix_path,
        all_features_path=REPO_ROOT / args.all_features_path,
        output_path=REPO_ROOT / args.full_feature_matrix_path,
        start_date=GLOBAL_START,
        end_date=GLOBAL_END,
    )
    tickers = full_feature_matrix.index.get_level_values("ticker").unique().tolist()
    logger.info(
        "full feature matrix rows={} dates={} tickers={}",
        len(full_feature_matrix),
        full_feature_matrix.index.get_level_values("trade_date").nunique(),
        len(tickers),
    )

    price_end = min(AS_OF_DATE, GLOBAL_END + timedelta(days=max(HORIZONS) * 5 + 40))
    prices = build_or_load_prices(
        tickers=tickers,
        benchmark_ticker=args.benchmark_ticker,
        start_date=GLOBAL_START,
        end_date=price_end,
        as_of=AS_OF_DATE,
        cache_path=REPO_ROOT / args.price_cache_path,
    )
    labels = build_or_load_multi_horizon_labels(
        prices=prices,
        horizons=HORIZONS,
        benchmark_ticker=args.benchmark_ticker,
        cache_path=REPO_ROOT / args.multi_label_cache_path,
        start_date=GLOBAL_START,
        end_date=GLOBAL_END,
    )
    label_series_by_horizon = build_label_series_by_horizon(labels)

    engine = WalkForwardEngine(
        alpha_grid=DEFAULT_ALPHA_GRID,
        benchmark_ticker=args.benchmark_ticker,
        tracking_uri=LOCAL_MLFLOW_URI,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    window1 = walkforward_windows()[0]

    horizon_matrix, optimal_horizon = run_horizon_experiment(
        engine=engine,
        feature_matrix=full_feature_matrix,
        label_series_by_horizon=label_series_by_horizon,
        window=window1,
        timestamp=timestamp,
    )
    logger.info("selected optimal horizon {} from Window 1 validation matrix", optimal_horizon)

    optimal_horizon_days = int(optimal_horizon.removesuffix("D"))
    aligned_X, aligned_y = align_panel(full_feature_matrix, label_series_by_horizon[optimal_horizon_days])
    signal_dates = aligned_X.index.get_level_values("trade_date").unique().sort_values()
    universe_by_date = build_universe_by_date(trade_dates=signal_dates)
    logger.info("built universe lookup for {} dates", len(universe_by_date))

    cost_model = AlmgrenChrissCostModel()
    walkforward_results = run_walkforward_windows(
        engine=engine,
        feature_matrix=aligned_X,
        label_series=aligned_y,
        prices=prices,
        windows=walkforward_windows(),
        target_horizon=optimal_horizon,
        timestamp=timestamp,
        cost_model=cost_model,
        universe_by_date=universe_by_date,
    )
    aggregate = aggregate_walkforward_results(walkforward_results)

    report = build_report_payload(
        args=args,
        retained_features=retained_features,
        full_feature_matrix=full_feature_matrix,
        prices=prices,
        labels=labels,
        horizon_matrix=horizon_matrix,
        optimal_horizon=optimal_horizon,
        walkforward_results=walkforward_results,
        aggregate=aggregate,
        cost_model=cost_model,
    )
    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved walk-forward report to {}", report_path)
    log_summary(report)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 8 multi-horizon Ridge experiments and the 5-window walk-forward backtest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--window1-feature-matrix-path", default=DEFAULT_WINDOW1_FEATURE_MATRIX_PATH)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--full-feature-matrix-path", default=DEFAULT_FULL_FEATURE_MATRIX_PATH)
    parser.add_argument("--price-cache-path", default=DEFAULT_PRICE_CACHE_PATH)
    parser.add_argument("--multi-label-cache-path", default=DEFAULT_MULTI_LABEL_CACHE_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def walkforward_windows() -> list[WalkForwardWindowConfig]:
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
    ]


def build_or_load_full_feature_matrix(
    *,
    retained_features: list[str],
    window1_feature_matrix_path: Path,
    all_features_path: Path,
    output_path: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if output_path.exists():
        logger.info("loading cached walk-forward feature matrix from {}", output_path)
        return restore_feature_matrix_index(pd.read_parquet(output_path))

    prefix = restore_feature_matrix_index(pd.read_parquet(window1_feature_matrix_path))
    prefix = prefix.loc[
        (prefix.index.get_level_values("trade_date") >= pd.Timestamp(start_date))
        & (prefix.index.get_level_values("trade_date") <= pd.Timestamp(min(end_date, date(2021, 12, 31)))),
        retained_features,
    ].sort_index()

    suffix_matrices: list[pd.DataFrame] = []
    suffix_start = date(2022, 1, 1)
    if end_date >= suffix_start:
        for year in range(suffix_start.year, end_date.year + 1):
            chunk_start = max(suffix_start, date(year, 1, 1))
            chunk_end = min(end_date, date(year, 12, 31))
            logger.info("loading retained all_features slice {} -> {}", chunk_start, chunk_end)
            long_slice = pd.read_parquet(
                all_features_path,
                columns=["ticker", "trade_date", "feature_name", "feature_value"],
                filters=[
                    ("trade_date", ">=", chunk_start),
                    ("trade_date", "<=", chunk_end),
                    ("feature_name", "in", retained_features),
                ],
            )
            if long_slice.empty:
                continue
            chunk_matrix = long_to_feature_matrix(long_slice, retained_features)
            chunk_matrix = fill_feature_matrix(chunk_matrix)
            suffix_matrices.append(chunk_matrix)

    matrices = [prefix, *suffix_matrices]
    combined = (
        pd.concat(matrices)
        .sort_index()
        .loc[lambda frame: ~frame.index.duplicated(keep="last")]
        .reindex(columns=retained_features)
    )
    write_parquet_atomic(feature_matrix_to_frame(combined), output_path)
    logger.info("saved combined walk-forward feature matrix to {}", output_path)
    return combined


def build_or_load_prices(
    *,
    tickers: list[str],
    benchmark_ticker: str,
    start_date: date,
    end_date: date,
    as_of: date,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        logger.info("loading cached walk-forward prices from {}", cache_path)
        prices = pd.read_parquet(cache_path)
    else:
        query_tickers = list(dict.fromkeys([*tickers, benchmark_ticker.upper()]))
        prices = get_prices_pit(
            tickers=query_tickers,
            start_date=start_date,
            end_date=end_date,
            as_of=as_of,
        )
        if prices.empty:
            raise RuntimeError("No PIT prices available for walk-forward backtest.")
        prices = normalize_prices(prices)
        write_parquet_atomic(prices, cache_path)
        logger.info("saved walk-forward price cache to {}", cache_path)

    return normalize_prices(prices)


def build_or_load_multi_horizon_labels(
    *,
    prices: pd.DataFrame,
    horizons: tuple[int, ...],
    benchmark_ticker: str,
    cache_path: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    if cache_path.exists():
        logger.info("loading cached multi-horizon labels from {}", cache_path)
        labels = pd.read_parquet(cache_path)
    else:
        labels = compute_forward_returns(
            prices_df=prices,
            horizons=horizons,
            benchmark_ticker=benchmark_ticker,
        )
        labels = labels.loc[
            (labels["ticker"].astype(str).str.upper() != benchmark_ticker.upper())
            & (pd.to_datetime(labels["trade_date"]) >= pd.Timestamp(start_date))
            & (pd.to_datetime(labels["trade_date"]) <= pd.Timestamp(end_date))
        ].copy()
        write_parquet_atomic(labels, cache_path)
        logger.info("saved multi-horizon label cache to {}", cache_path)

    prepared = labels.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"])
    prepared["horizon"] = prepared["horizon"].astype(int)
    prepared["forward_return"] = pd.to_numeric(prepared["forward_return"], errors="coerce")
    prepared["excess_return"] = pd.to_numeric(prepared["excess_return"], errors="coerce")
    prepared.sort_values(["horizon", "trade_date", "ticker"], inplace=True)
    return prepared


def build_label_series_by_horizon(labels: pd.DataFrame) -> dict[int, pd.Series]:
    series_by_horizon: dict[int, pd.Series] = {}
    for horizon, frame in labels.groupby("horizon", sort=True):
        series = (
            frame.set_index(["trade_date", "ticker"])["excess_return"]
            .sort_index()
            .dropna()
        )
        series_by_horizon[int(horizon)] = series
    return series_by_horizon


def run_horizon_experiment(
    *,
    engine: WalkForwardEngine,
    feature_matrix: pd.DataFrame,
    label_series_by_horizon: dict[int, pd.Series],
    window: WalkForwardWindowConfig,
    timestamp: str,
) -> tuple[list[dict[str, Any]], str]:
    matrix: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        X, y = align_panel(feature_matrix, label_series_by_horizon[horizon])
        result = engine.run_window(
            X=X,
            y=y,
            prices=None,
            window=window,
            target_horizon=f"{horizon}D",
            window_id=f"{window.window_id}_{horizon}D",
            timestamp=timestamp,
            simulate_portfolio=False,
        )
        matrix.append(
            {
                "horizon": f"{horizon}D",
                "model": "ridge",
                "best_hyperparams": result.best_hyperparams,
                "train_ic": float(result.train_metrics.ic),
                "train_rank_ic": float(result.train_metrics.rank_ic),
                "train_icir": float(result.train_metrics.icir),
                "train_hit_rate": float(result.train_metrics.hit_rate),
                "train_top_decile": float(result.train_metrics.top_decile_return),
                "train_long_short": float(result.train_metrics.long_short_return),
                "train_turnover": float(result.train_metrics.turnover),
                "validation_ic": float(result.validation_metrics.ic),
                "validation_rank_ic": float(result.validation_metrics.rank_ic),
                "validation_icir": float(result.validation_metrics.icir),
                "validation_hit_rate": float(result.validation_metrics.hit_rate),
                "validation_top_decile": float(result.validation_metrics.top_decile_return),
                "validation_long_short": float(result.validation_metrics.long_short_return),
                "validation_turnover": float(result.validation_metrics.turnover),
                "test_ic": float(result.test_metrics.ic),
                "test_rank_ic": float(result.test_metrics.rank_ic),
                "test_icir": float(result.test_metrics.icir),
                "test_hit_rate": float(result.test_metrics.hit_rate),
                "test_top_decile": float(result.test_metrics.top_decile_return),
                "test_long_short": float(result.test_metrics.long_short_return),
                "test_turnover": float(result.test_metrics.turnover),
                "train_rows": result.train_rows,
                "validation_rows": result.validation_rows,
                "test_rows": result.test_rows,
                "mlflow": mlflow_payload(result),
            },
        )

    sorted_matrix = sorted(
        matrix,
        key=lambda row: (
            _nan_safe(row["validation_ic"]),
            _nan_safe(row["test_ic"]),
            -_nan_safe(row["validation_turnover"]),
        ),
        reverse=True,
    )
    optimal_horizon = str(sorted_matrix[0]["horizon"])
    return sorted_matrix, optimal_horizon


def run_walkforward_windows(
    *,
    engine: WalkForwardEngine,
    feature_matrix: pd.DataFrame,
    label_series: pd.Series,
    prices: pd.DataFrame,
    windows: list[WalkForwardWindowConfig],
    target_horizon: str,
    timestamp: str,
    cost_model: AlmgrenChrissCostModel,
    universe_by_date: dict[pd.Timestamp, set[str]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for window in windows:
        result = engine.run_window(
            X=feature_matrix,
            y=label_series,
            prices=prices,
            window=window,
            target_horizon=target_horizon,
            window_id=window.window_id,
            timestamp=timestamp,
            cost_model=cost_model,
            universe_by_date=universe_by_date,
            simulate_portfolio=True,
        )
        results.append(
            {
                "window_id": window.window_id,
                "train_period": f"{window.train_start.isoformat()} -> {window.train_end.isoformat()}",
                "validation_period": f"{window.validation_start.isoformat()} -> {window.validation_end.isoformat()}",
                "test_period": f"{window.test_start.isoformat()} -> {window.test_end.isoformat()}",
                "best_hyperparams": result.best_hyperparams,
                "train_rows": result.train_rows,
                "validation_rows": result.validation_rows,
                "test_rows": result.test_rows,
                "train_metrics": metrics_to_dict(result.train_metrics),
                "validation_metrics": metrics_to_dict(result.validation_metrics),
                "test_metrics": metrics_to_dict(result.test_metrics),
                "portfolio": result.portfolio.to_dict() if result.portfolio is not None else None,
                "mlflow": mlflow_payload(result),
            },
        )
    return results


def aggregate_walkforward_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    window_metrics = {
        "mean_test_ic": float(np.mean([row["test_metrics"]["ic"] for row in results])),
        "mean_test_rank_ic": float(np.mean([row["test_metrics"]["rank_ic"] for row in results])),
        "mean_test_icir": float(np.mean([row["test_metrics"]["icir"] for row in results])),
        "mean_test_hit_rate": float(np.mean([row["test_metrics"]["hit_rate"] for row in results])),
    }

    all_periods: list[dict[str, Any]] = []
    cost_breakdown = {
        "commission": 0.0,
        "spread": 0.0,
        "temporary_impact": 0.0,
        "permanent_impact": 0.0,
        "gap_penalty": 0.0,
        "total": 0.0,
    }
    for window in results:
        portfolio = window.get("portfolio") or {}
        for period in portfolio.get("periods", []):
            all_periods.append({"window_id": window["window_id"], **period})
        for key in cost_breakdown:
            cost_breakdown[key] += float(portfolio.get("cost_breakdown", {}).get(key, 0.0))

    periods_frame = pd.DataFrame(all_periods)
    if periods_frame.empty:
        return {
            **window_metrics,
            "annualized_excess_gross": 0.0,
            "annualized_excess_net": 0.0,
            "annualized_gross_return": 0.0,
            "annualized_net_return": 0.0,
            "annualized_benchmark_return": 0.0,
            "total_cost_drag": 0.0,
            "windows_above_ic_threshold": 0,
            "period_count": 0,
            "periods": [],
            "cost_breakdown": cost_breakdown,
        }

    periods_frame["execution_date"] = pd.to_datetime(periods_frame["execution_date"])
    periods_frame["exit_date"] = pd.to_datetime(periods_frame["exit_date"])
    periods_frame.sort_values(["execution_date", "window_id"], inplace=True)

    gross_curve = (1.0 + periods_frame["gross_return"]).cumprod()
    net_curve = (1.0 + periods_frame["net_return"]).cumprod()
    benchmark_curve = (1.0 + periods_frame["benchmark_return"]).cumprod()
    periods_frame["gross_cum_return"] = gross_curve - 1.0
    periods_frame["net_cum_return"] = net_curve - 1.0
    periods_frame["benchmark_cum_return"] = benchmark_curve - 1.0
    periods_frame["gross_cum_excess"] = periods_frame["gross_cum_return"] - periods_frame["benchmark_cum_return"]
    periods_frame["net_cum_excess"] = periods_frame["net_cum_return"] - periods_frame["benchmark_cum_return"]

    total_days = max((periods_frame["exit_date"].iloc[-1] - periods_frame["execution_date"].iloc[0]).days, 1)
    annualized_gross = annualize_return(float(gross_curve.iloc[-1] - 1.0), total_days)
    annualized_net = annualize_return(float(net_curve.iloc[-1] - 1.0), total_days)
    annualized_benchmark = annualize_return(float(benchmark_curve.iloc[-1] - 1.0), total_days)

    return {
        **window_metrics,
        "annualized_excess_gross": float(annualized_gross - annualized_benchmark),
        "annualized_excess_net": float(annualized_net - annualized_benchmark),
        "annualized_gross_return": float(annualized_gross),
        "annualized_net_return": float(annualized_net),
        "annualized_benchmark_return": float(annualized_benchmark),
        "total_cost_drag": float(annualized_gross - annualized_net),
        "windows_above_ic_threshold": int(sum(row["test_metrics"]["ic"] > 0.03 for row in results)),
        "period_count": int(len(periods_frame)),
        "periods": dataframe_to_records(periods_frame),
        "cost_breakdown": {key: float(value) for key, value in cost_breakdown.items()},
    }


def build_report_payload(
    *,
    args: argparse.Namespace,
    retained_features: list[str],
    full_feature_matrix: pd.DataFrame,
    prices: pd.DataFrame,
    labels: pd.DataFrame,
    horizon_matrix: list[dict[str, Any]],
    optimal_horizon: str,
    walkforward_results: list[dict[str, Any]],
    aggregate: dict[str, Any],
    cost_model: AlmgrenChrissCostModel,
) -> dict[str, Any]:
    optimal_row = next(row for row in horizon_matrix if row["horizon"] == optimal_horizon)
    mean_oos_ic = float(aggregate["mean_test_ic"])
    windows_above_threshold = int(aggregate["windows_above_ic_threshold"])
    annualized_excess_net = float(aggregate["annualized_excess_net"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_walkforward_backtest.py",
        "inputs": {
            "ic_report_path": str(REPO_ROOT / args.ic_report_path),
            "window1_feature_matrix_path": str(REPO_ROOT / args.window1_feature_matrix_path),
            "all_features_path": str(REPO_ROOT / args.all_features_path),
            "full_feature_matrix_path": str(REPO_ROOT / args.full_feature_matrix_path),
            "price_cache_path": str(REPO_ROOT / args.price_cache_path),
            "multi_label_cache_path": str(REPO_ROOT / args.multi_label_cache_path),
            "report_path": str(REPO_ROOT / args.report_path),
            "benchmark_ticker": args.benchmark_ticker.upper(),
            "retained_feature_count": len(retained_features),
            "retained_features": retained_features,
        },
        "data_summary": {
            "feature_rows": int(len(full_feature_matrix)),
            "feature_dates": int(full_feature_matrix.index.get_level_values("trade_date").nunique()),
            "feature_tickers": int(full_feature_matrix.index.get_level_values("ticker").nunique()),
            "feature_min_date": full_feature_matrix.index.get_level_values("trade_date").min().date().isoformat(),
            "feature_max_date": full_feature_matrix.index.get_level_values("trade_date").max().date().isoformat(),
            "price_rows": int(len(prices)),
            "price_dates": int(pd.to_datetime(prices["trade_date"]).nunique()),
            "price_tickers": int(prices["ticker"].nunique()),
            "label_rows": int(len(labels)),
            "label_horizons": sorted(labels["horizon"].drop_duplicates().astype(int).tolist()),
        },
        "horizon_experiment": {
            "matrix": horizon_matrix,
            "optimal_horizon": optimal_horizon,
            "rationale": (
                f"Selected {optimal_horizon} because it had the strongest Window 1 validation IC "
                f"({optimal_row['validation_ic']:.6f}) without using test data for selection. "
                f"Its Window 1 test IC was {optimal_row['test_ic']:.6f}."
            ),
        },
        "walkforward": {
            "windows": walkforward_results,
            "aggregate": aggregate,
        },
        "cost_model": cost_model.get_params(),
        "execution_assumptions": {
            "signal_timing": "Friday close signal, next trading day execution",
            "execution_price_proxy": "50% adjusted open + 50% adjusted OHLC typical price",
            "portfolio_construction": "equal-weight top decile long-only portfolio",
            "universe_filtering": (
                "The engine attempts to intersect signals with universe_membership, "
                "but falls back to the model signal cross-section when the overlap is below 100 names. "
                "The current universe_membership history is too sparse/disjoint to serve as a hard trading filter "
                "for 2021-2023."
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
                for row in walkforward_results
            ],
        },
    }


def normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    for column in ("open", "high", "low", "close", "adj_close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values(["trade_date", "ticker"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


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
    matrix.index = matrix.index.set_names(["trade_date", "ticker"])
    return matrix.reindex(columns=retained_features)


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


def annualize_return(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


def dataframe_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "window_id": str(row.window_id),
                "signal_date": pd.Timestamp(row.signal_date).date().isoformat(),
                "execution_date": pd.Timestamp(row.execution_date).date().isoformat(),
                "exit_date": pd.Timestamp(row.exit_date).date().isoformat(),
                "gross_return": float(row.gross_return),
                "net_return": float(row.net_return),
                "benchmark_return": float(row.benchmark_return),
                "gross_excess_return": float(row.gross_excess_return),
                "net_excess_return": float(row.net_excess_return),
                "gross_cum_return": float(row.gross_cum_return),
                "net_cum_return": float(row.net_cum_return),
                "benchmark_cum_return": float(row.benchmark_cum_return),
                "gross_cum_excess": float(row.gross_cum_excess),
                "net_cum_excess": float(row.net_cum_excess),
                "turnover": float(row.turnover),
                "cost_rate": float(row.cost_rate),
                "selected_count": int(row.selected_count),
            },
        )
    return records


def mlflow_payload(result: Any) -> dict[str, Any]:
    logged = getattr(result, "mlflow_run", None)
    return {
        "logged": logged is not None,
        "tracking_uri": logged.tracking_uri if logged is not None else None,
        "experiment_name": logged.experiment_name if logged is not None else None,
        "experiment_id": logged.experiment_id if logged is not None else None,
        "run_id": logged.run_id if logged is not None else None,
    }


def log_summary(report: dict[str, Any]) -> None:
    logger.info("optimal horizon: {}", report["horizon_experiment"]["optimal_horizon"])
    logger.info(
        "walk-forward mean_test_ic={:.6f} annualized_excess_net={:.6f} ic_pass={} excess_pass={}",
        report["walkforward"]["aggregate"]["mean_test_ic"],
        report["walkforward"]["aggregate"]["annualized_excess_net"],
        report["go_nogo_preview"]["ic_pass"],
        report["go_nogo_preview"]["excess_pass"],
    )


def _nan_safe(value: Any) -> float:
    numeric = float(value)
    if np.isnan(numeric):
        return float("-inf")
    return numeric


if __name__ == "__main__":
    raise SystemExit(main())
