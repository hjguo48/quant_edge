from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
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

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from scripts.run_horizon_fusion import (
    extract_ridge_alpha,
    parse_horizon_days,
    prepare_horizon_artifacts,
    rebuild_ridge_predictions,
    select_report_windows,
    slice_all_splits,
)
from scripts.run_turnover_optimization import simulate_score_weighted_controlled
from scripts.run_walkforward_comparison import LABEL_BUFFER_DAYS, REBALANCE_WEEKDAY, parse_date
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.engine import build_universe_by_date
from src.backtest.execution import PortfolioBacktestResult, prepare_execution_price_frame, simulate_portfolio
from src.data.db.pit import get_prices_pit
from src.stats.bootstrap import bootstrap_return_statistics, sharpe_ratio
from src.stats.spa import run_spa_fallback

ALLOWED_BRANCHES = {"feature/alpha-enhancement", "feature/s2-stage0-monetization"}
DEFAULT_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_WALKFORWARD_REPORT_PATH = "data/reports/extended_walkforward.json"
DEFAULT_COST_CALIBRATION_REPORT_PATH = "data/reports/portfolio_optimization_comparison.json"
DEFAULT_REPORT_PATH = "data/reports/holding_period_experiment.json"
DEFAULT_COMPARISON_REPORT_PATH = "data/reports/walkforward_comparison_60d_ridge_v2.json"
DEFAULT_BUNDLE_PATH = "data/models/fusion_model_bundle_60d.json"
DEFAULT_ALL_FEATURES_PATH = "data/features/all_features_v2.parquet"
DEFAULT_FEATURE_MATRIX_CACHE_PATH = "data/features/walkforward_feature_matrix_60d_v2.parquet"
DEFAULT_LABEL_CACHE_PATH = "data/labels/forward_returns_60d.parquet"
DEFAULT_PHASE_E_PRICE_CACHE_PATH = "data/backtest/phase_e_ridge_v2_prices.parquet"
DEFAULT_PHASE_E_REPORT_PATH = "data/reports/s2_holding_period_comparison.json"
BENCHMARK_TICKER = "SPY"
DEFAULT_SELECTION_PCT = 0.20
DEFAULT_SELL_BUFFER_PCT = 0.25
DEFAULT_MIN_TRADE_WEIGHT = 0.01
DEFAULT_MAX_WEIGHT = 0.05
DEFAULT_MIN_HOLDINGS = 20
DEFAULT_ETA = 0.426
DEFAULT_GAMMA = 0.942
HOLDING_PERIOD_ORDER = ("1W", "2W", "4W", "8W")
PHASE_E_HOLDING_PERIOD_ORDER = ("20D", "40D", "60D")
SELL_BUFFER_OPTIONS: tuple[tuple[str, float | None], ...] = (
    ("0%", None),
    ("25%", 0.25),
    ("50%", 0.50),
    ("75%", 0.75),
)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch not in ALLOWED_BRANCHES:
        raise RuntimeError(
            f"This script must run on one of {sorted(ALLOWED_BRANCHES)}. Found {branch!r}.",
        )

    prediction_source = resolve_prediction_source(args)
    if prediction_source == "comparison":
        return run_phase_e_holding_period_experiment(args, branch)

    predictions_frame = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices = pd.read_parquet(REPO_ROOT / args.prices_path)
    walkforward_report = json.loads((REPO_ROOT / args.walkforward_report_path).read_text())
    cost_params = load_calibrated_cost_parameters(REPO_ROOT / args.cost_calibration_report_path)

    logger.info(
        "loaded predictions rows={} windows={} prices rows={} tickers={}",
        len(predictions_frame),
        predictions_frame["window_id"].nunique(),
        len(prices),
        prices["ticker"].nunique(),
    )

    prediction_bundle = load_predictions_by_window(predictions_frame)
    window_metadata = {
        str(window["window_id"]): window
        for window in walkforward_report["walkforward"]["windows"]
    }
    trade_dates = (
        pd.DatetimeIndex(pd.to_datetime(predictions_frame["trade_date"]))
        .sort_values()
        .unique()
    )
    universe_by_date = build_universe_by_date(trade_dates=trade_dates)
    cost_model = AlmgrenChrissCostModel(
        eta=float(cost_params["eta"]),
        gamma=float(cost_params["gamma"]),
        commission_per_share=float(cost_params["commission_per_share"]),
        min_spread_bps=float(cost_params["min_spread_bps"]),
    )

    base_scheme_config = {
        "weighting_scheme": "equal_weight",
        "selection_pct": float(args.selection_pct),
        "sell_buffer_pct": float(args.sell_buffer_pct),
        "min_trade_weight": float(args.min_trade_weight),
        "max_weight": float(args.max_weight),
        "min_holdings": int(args.min_holdings),
    }

    holding_period_reports: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []

    for holding_period in HOLDING_PERIOD_ORDER:
        logger.info("running holding-period experiment {}", holding_period)
        payload = run_holding_period_experiment(
            holding_period=holding_period,
            prediction_bundle=prediction_bundle,
            window_metadata=window_metadata,
            prices=prices,
            universe_by_date=universe_by_date,
            benchmark_ticker=args.benchmark_ticker,
            cost_model=cost_model,
            scheme_config=base_scheme_config,
        )
        holding_period_reports[holding_period] = payload
        comparison_rows.append(
            {
                "holding_period": holding_period,
                "annualized_gross_excess": float(payload["aggregate"]["annualized_gross_excess"]),
                "annualized_net_excess": float(payload["aggregate"]["annualized_net_excess"]),
                "sharpe": float(payload["aggregate"]["sharpe"]),
                "average_turnover": float(payload["aggregate"]["average_turnover"]),
                "total_cost_drag": float(payload["aggregate"]["total_cost_drag"]),
                "max_drawdown": float(payload["aggregate"]["max_drawdown"]),
                "rebalance_count": int(payload["aggregate"]["rebalance_count"]),
                "bootstrap_sharpe_ci_lower": float(payload["aggregate"]["bootstrap"]["sharpe_ci_lower"]),
                "bootstrap_sharpe_ci_upper": float(payload["aggregate"]["bootstrap"]["sharpe_ci_upper"]),
            },
        )

    comparison_rows.sort(key=lambda row: row["annualized_net_excess"], reverse=True)
    best_holding_period = str(comparison_rows[0]["holding_period"])
    logger.info(
        "best holding period under default buffer={} is {} with annualized net excess {:.6f}",
        args.sell_buffer_pct,
        best_holding_period,
        comparison_rows[0]["annualized_net_excess"],
    )

    sell_buffer_sensitivity = run_sell_buffer_sensitivity(
        holding_period=best_holding_period,
        prediction_bundle=prediction_bundle,
        window_metadata=window_metadata,
        prices=prices,
        universe_by_date=universe_by_date,
        benchmark_ticker=args.benchmark_ticker,
        cost_model=cost_model,
        base_scheme_config=base_scheme_config,
    )

    best_period_series = holding_period_reports[best_holding_period]["aggregate"]["period_net_excess_series"]
    baseline_series = holding_period_reports["1W"]["aggregate"]["period_net_excess_series"]
    spa_result = run_spa_fallback(
        benchmark_series=baseline_series,
        competitors={best_holding_period: best_period_series},
        benchmark_name="1W_equal_weight_buffered",
    )

    recommended_config = select_recommended_configuration(
        holding_period_reports=holding_period_reports,
        sell_buffer_sensitivity=sell_buffer_sensitivity,
    )
    go_nogo_impact = build_go_nogo_impact(recommended_config)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": "scripts/run_holding_period_experiment.py",
        "inputs": {
            "predictions_path": str(REPO_ROOT / args.predictions_path),
            "prices_path": str(REPO_ROOT / args.prices_path),
            "walkforward_report_path": str(REPO_ROOT / args.walkforward_report_path),
            "cost_calibration_report_path": str(REPO_ROOT / args.cost_calibration_report_path),
            "report_path": str(REPO_ROOT / args.report_path),
            "benchmark_ticker": args.benchmark_ticker.upper(),
        },
        "calibrated_cost_model": {
            **cost_params,
            "source": str(REPO_ROOT / args.cost_calibration_report_path),
        },
        "schedule_assumptions": {
            "1W": "Every available weekly signal date.",
            "2W": "Every other weekly signal date, starting from the first date in each window.",
            "4W": "First available signal date of each calendar month.",
            "8W": "First available signal date of every second calendar month, starting from the first month in each window.",
        },
        "base_scheme": {
            **base_scheme_config,
            "description": "Equal-weight buffered book with top-20% entry universe, 5% max weight, 20-stock minimum, and calibrated Almgren-Chriss costs.",
        },
        "comparison_matrix": comparison_rows,
        "holding_periods": {
            holding_period: make_report_safe_payload(payload)
            for holding_period, payload in holding_period_reports.items()
        },
        "sell_buffer_sensitivity": make_report_safe_payload(sell_buffer_sensitivity),
        "spa_test": {
            "metric": "period_net_excess_return",
            "best_holding_period": best_holding_period,
            "result": spa_result.to_dict(),
        },
        "recommended_configuration": recommended_config,
        "go_nogo_impact": go_nogo_impact,
    }

    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved holding-period experiment report to {}", report_path)
    logger.info(
        "recommended config={} annualized_net_excess={:.6f} bootstrap_lower={:.6f} pass_net_excess={}",
        recommended_config["label"],
        recommended_config["annualized_net_excess"],
        recommended_config["bootstrap"]["sharpe_ci_lower"],
        go_nogo_impact["net_excess_pass"],
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 1W/2W/4W/8W holding periods for the 8-window 60D Ridge predictions using calibrated execution costs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prediction-source",
        choices=("auto", "legacy", "comparison"),
        default="auto",
    )
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--walkforward-report-path", default=DEFAULT_WALKFORWARD_REPORT_PATH)
    parser.add_argument("--cost-calibration-report-path", default=DEFAULT_COST_CALIBRATION_REPORT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--comparison-report")
    parser.add_argument("--bundle-path", default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--feature-matrix-cache-path", default=DEFAULT_FEATURE_MATRIX_CACHE_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--phase-e-price-cache-path", default=DEFAULT_PHASE_E_PRICE_CACHE_PATH)
    parser.add_argument("--phase-e-report-path", default=DEFAULT_PHASE_E_REPORT_PATH)
    parser.add_argument("--as-of")
    parser.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    parser.add_argument("--rebalance-weekday", type=int, default=REBALANCE_WEEKDAY)
    parser.add_argument("--holding-periods", default="20D,40D,60D")
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--selection-pct", type=float, default=DEFAULT_SELECTION_PCT)
    parser.add_argument("--sell-buffer-pct", type=float, default=DEFAULT_SELL_BUFFER_PCT)
    parser.add_argument("--min-trade-weight", type=float, default=DEFAULT_MIN_TRADE_WEIGHT)
    parser.add_argument("--max-weight", type=float, default=DEFAULT_MAX_WEIGHT)
    parser.add_argument("--min-holdings", type=int, default=DEFAULT_MIN_HOLDINGS)
    return parser.parse_args(argv)


def load_calibrated_cost_parameters(report_path: Path) -> dict[str, float]:
    params = {
        "eta": DEFAULT_ETA,
        "gamma": DEFAULT_GAMMA,
        "commission_per_share": 0.005,
        "min_spread_bps": 2.0,
    }
    if not report_path.exists():
        return params

    payload = json.loads(report_path.read_text())
    calibrated = payload.get("cost_calibration", {}).get("calibrated")
    if isinstance(calibrated, dict):
        params["eta"] = float(calibrated.get("eta", params["eta"]))
        params["gamma"] = float(calibrated.get("gamma", params["gamma"]))
    return params


def load_predictions_by_window(predictions_frame: pd.DataFrame) -> dict[str, pd.Series]:
    frame = predictions_frame.copy()
    frame["window_id"] = frame["window_id"].astype(str)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.dropna(subset=["score"]).sort_values(["window_id", "trade_date", "ticker"])

    bundle: dict[str, pd.Series] = {}
    for window_id, group in frame.groupby("window_id", sort=True):
        bundle[window_id] = group.set_index(["trade_date", "ticker"])["score"].sort_index()
    return bundle


def run_holding_period_experiment(
    *,
    holding_period: str,
    prediction_bundle: dict[str, pd.Series],
    window_metadata: dict[str, dict[str, Any]],
    prices: pd.DataFrame | None,
    execution: pd.DataFrame | None,
    universe_by_date: dict[pd.Timestamp, set[str]] | None,
    benchmark_ticker: str,
    cost_model: AlmgrenChrissCostModel,
    scheme_config: dict[str, Any],
) -> dict[str, Any]:
    window_reports: list[dict[str, Any]] = []
    period_rows: list[dict[str, Any]] = []

    for window_id in sorted(prediction_bundle):
        predictions = prediction_bundle[window_id]
        filtered_predictions, selected_signal_dates = thin_predictions(
            predictions=predictions,
            holding_period=holding_period,
        )
        if str(scheme_config["weighting_scheme"]) == "score_weighted":
            if execution is None:
                raise RuntimeError("execution price frame is required for score_weighted holding-period tests.")
            portfolio = simulate_score_weighted_controlled(
                predictions=filtered_predictions,
                execution=execution,
                cost_model=cost_model,
                benchmark_ticker=benchmark_ticker,
                selection_pct=float(scheme_config["selection_pct"]),
                sell_buffer_pct=scheme_config.get("sell_buffer_pct"),
                min_trade_weight=float(scheme_config["min_trade_weight"]),
                max_weight=float(scheme_config["max_weight"]),
                min_holdings=int(scheme_config["min_holdings"]),
                weight_shrinkage=float(scheme_config.get("weight_shrinkage", 0.0)),
                no_trade_zone=float(scheme_config.get("no_trade_zone", 0.0)),
                turnover_penalty_lambda=float(scheme_config.get("turnover_penalty_lambda", 0.0)),
            )
        else:
            if prices is None:
                raise RuntimeError("prices are required for legacy holding-period experiment mode.")
            portfolio = simulate_portfolio(
                predictions=filtered_predictions,
                prices=prices,
                cost_model=cost_model,
                weighting_scheme=str(scheme_config["weighting_scheme"]),
                benchmark_ticker=benchmark_ticker,
                universe_by_date=universe_by_date,
                selection_pct=float(scheme_config["selection_pct"]),
                sell_buffer_pct=scheme_config["sell_buffer_pct"],
                min_trade_weight=float(scheme_config["min_trade_weight"]),
                max_weight=float(scheme_config["max_weight"]),
                min_holdings=int(scheme_config["min_holdings"]),
            )
        metadata = window_metadata[window_id]
        window_payload = build_window_payload(
            window_id=window_id,
            metadata=metadata,
            holding_period=holding_period,
            selected_signal_dates=selected_signal_dates,
            portfolio=portfolio,
        )
        window_reports.append(window_payload)
        for period in portfolio.periods:
            period_rows.append({"window_id": window_id, **period.to_dict()})

    aggregate = aggregate_window_reports(window_reports=window_reports, period_rows=period_rows)
    return {
        "holding_period": holding_period,
        "windows": window_reports,
        "aggregate": aggregate,
    }


def thin_predictions(
    *,
    predictions: pd.Series,
    holding_period: str,
) -> tuple[pd.Series, list[str]]:
    signal_dates = pd.DatetimeIndex(
        pd.to_datetime(predictions.index.get_level_values("trade_date")),
    ).sort_values().unique()
    selected_dates = select_signal_dates(signal_dates=signal_dates, holding_period=holding_period)
    mask = pd.DatetimeIndex(
        pd.to_datetime(predictions.index.get_level_values("trade_date")),
    ).isin(selected_dates)
    filtered = predictions[mask].sort_index()
    return filtered, [pd.Timestamp(value).date().isoformat() for value in selected_dates]


def select_signal_dates(
    *,
    signal_dates: pd.DatetimeIndex,
    holding_period: str,
) -> pd.DatetimeIndex:
    if signal_dates.empty:
        return signal_dates

    normalized = pd.DatetimeIndex(signal_dates).sort_values().unique()
    if holding_period.endswith("D"):
        holding_days = int(holding_period.removesuffix("D"))
        step = max(1, int(round(holding_days / 5.0)))
        return normalized[::step]
    if holding_period == "1W":
        return normalized
    if holding_period == "2W":
        return normalized[::2]

    month_starts = (
        pd.Series(normalized, index=normalized)
        .groupby(normalized.to_period("M"))
        .first()
    )
    monthly_dates = pd.DatetimeIndex(month_starts.tolist()).sort_values()
    if holding_period == "4W":
        return monthly_dates
    if holding_period == "8W":
        return monthly_dates[::2]
    raise ValueError(f"Unsupported holding period: {holding_period}")


def build_window_payload(
    *,
    window_id: str,
    metadata: dict[str, Any],
    holding_period: str,
    selected_signal_dates: list[str],
    portfolio: PortfolioBacktestResult,
) -> dict[str, Any]:
    annualization = infer_annualization_factor(portfolio.periods)
    net_excess_series = period_return_series(portfolio.periods, value_key="net_excess_return")
    gross_excess_series = period_return_series(portfolio.periods, value_key="gross_excess_return")
    return {
        "window_id": window_id,
        "holding_period": holding_period,
        "train_period": metadata["train_period"],
        "validation_period": metadata["validation_period"],
        "test_period": metadata["test_period"],
        "test_metrics": metadata["test_metrics"],
        "best_hyperparams": float(metadata["best_hyperparams"]),
        "signal_date_count": int(len(selected_signal_dates)),
        "signal_dates_selected": selected_signal_dates,
        "rebalance_count": int(len(portfolio.periods)),
        "annualization_factor": int(annualization),
        "annualized_gross_excess": float(portfolio.annualized_excess_gross),
        "annualized_net_excess": float(portfolio.annualized_excess_net),
        "sharpe": float(sharpe_ratio(net_excess_series, annualization=annualization)) if len(net_excess_series) >= 2 else float("nan"),
        "max_drawdown": float(max_drawdown(net_excess_series)) if not net_excess_series.empty else float("nan"),
        "mean_period_gross_excess": float(gross_excess_series.mean()) if not gross_excess_series.empty else float("nan"),
        "mean_period_net_excess": float(net_excess_series.mean()) if not net_excess_series.empty else float("nan"),
        "portfolio": portfolio.to_dict(),
    }


def aggregate_window_reports(
    *,
    window_reports: list[dict[str, Any]],
    period_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    periods_frame = pd.DataFrame(period_rows)
    if periods_frame.empty:
        empty_bootstrap = {
            "block_size": 4,
            "n_bootstrap": 0,
            "ci_level": 0.95,
            "n_observations": 0,
            "sharpe_estimate": float("nan"),
            "sharpe_ci_lower": float("nan"),
            "sharpe_ci_upper": float("nan"),
            "sharpe_p_value": float("nan"),
            "mean_excess_estimate": float("nan"),
            "mean_excess_ci_lower": float("nan"),
            "mean_excess_ci_upper": float("nan"),
            "mean_excess_p_value": float("nan"),
            "annualized_excess_estimate": float("nan"),
            "annualized_excess_ci_lower": float("nan"),
            "annualized_excess_ci_upper": float("nan"),
            "annualized_excess_p_value": float("nan"),
        }
        return {
            "mean_ic": float(np.mean([row["test_metrics"]["ic"] for row in window_reports])) if window_reports else float("nan"),
            "mean_rank_ic": float(np.mean([row["test_metrics"]["rank_ic"] for row in window_reports])) if window_reports else float("nan"),
            "mean_icir": float(np.mean([row["test_metrics"]["icir"] for row in window_reports])) if window_reports else float("nan"),
            "mean_hit_rate": float(np.mean([row["test_metrics"]["hit_rate"] for row in window_reports])) if window_reports else float("nan"),
            "annualized_gross_return": 0.0,
            "annualized_net_return": 0.0,
            "annualized_benchmark_return": 0.0,
            "annualized_gross_excess": 0.0,
            "annualized_net_excess": 0.0,
            "total_cost_drag": 0.0,
            "average_turnover": 0.0,
            "cost_breakdown": empty_cost_breakdown(),
            "rebalance_count": 0,
            "signal_date_count": int(sum(row["signal_date_count"] for row in window_reports)),
            "annualization_factor": 0,
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "bootstrap": empty_bootstrap,
            "period_net_excess_records": [],
            "period_net_excess_series": pd.Series(dtype=float),
        }

    periods_frame["execution_date"] = pd.to_datetime(periods_frame["execution_date"])
    periods_frame["exit_date"] = pd.to_datetime(periods_frame["exit_date"])
    periods_frame.sort_values(["execution_date", "window_id"], inplace=True)

    gross_curve = (1.0 + periods_frame["gross_return"].astype(float)).cumprod()
    net_curve = (1.0 + periods_frame["net_return"].astype(float)).cumprod()
    benchmark_curve = (1.0 + periods_frame["benchmark_return"].astype(float)).cumprod()
    total_days = max((periods_frame["exit_date"].iloc[-1] - periods_frame["execution_date"].iloc[0]).days, 1)

    annualized_gross = annualize_return(float(gross_curve.iloc[-1] - 1.0), total_days)
    annualized_net = annualize_return(float(net_curve.iloc[-1] - 1.0), total_days)
    annualized_benchmark = annualize_return(float(benchmark_curve.iloc[-1] - 1.0), total_days)

    annualization_factor = infer_annualization_factor_from_frame(periods_frame)
    net_excess_series = pd.Series(
        periods_frame["net_excess_return"].astype(float).to_numpy(),
        index=periods_frame["execution_date"],
        dtype=float,
        name="net_excess_return",
    )
    bootstrap = bootstrap_return_statistics(
        net_excess_series,
        block_size=4,
        n_bootstrap=10_000,
        ci_level=0.95,
        annualization=annualization_factor,
        seed=42,
    )

    cost_breakdown = empty_cost_breakdown()
    for row in window_reports:
        for key, value in row["portfolio"]["cost_breakdown"].items():
            cost_breakdown[key] += float(value)

    return {
        "mean_ic": float(np.mean([row["test_metrics"]["ic"] for row in window_reports])),
        "mean_rank_ic": float(np.mean([row["test_metrics"]["rank_ic"] for row in window_reports])),
        "mean_icir": float(np.mean([row["test_metrics"]["icir"] for row in window_reports])),
        "mean_hit_rate": float(np.mean([row["test_metrics"]["hit_rate"] for row in window_reports])),
        "annualized_gross_return": float(annualized_gross),
        "annualized_net_return": float(annualized_net),
        "annualized_benchmark_return": float(annualized_benchmark),
        "annualized_gross_excess": float(annualized_gross - annualized_benchmark),
        "annualized_net_excess": float(annualized_net - annualized_benchmark),
        "total_cost_drag": float(annualized_gross - annualized_net),
        "average_turnover": float(periods_frame["turnover"].astype(float).mean()),
        "cost_breakdown": {key: float(value) for key, value in cost_breakdown.items()},
        "rebalance_count": int(len(periods_frame)),
        "signal_date_count": int(sum(row["signal_date_count"] for row in window_reports)),
        "annualization_factor": int(annualization_factor),
        "sharpe": float(sharpe_ratio(net_excess_series, annualization=annualization_factor)),
        "max_drawdown": float(max_drawdown(net_excess_series)),
        "bootstrap": bootstrap.to_dict(),
        "period_net_excess_records": [
            {"trade_date": idx.date().isoformat(), "value": float(value)}
            for idx, value in net_excess_series.items()
        ],
        "period_net_excess_series": net_excess_series,
    }


def period_return_series(periods: list[dict[str, Any]] | list[Any], *, value_key: str) -> pd.Series:
    if not periods:
        return pd.Series(dtype=float, name=value_key)
    frame = pd.DataFrame(
        [
            period if isinstance(period, dict) else period.to_dict()
            for period in periods
        ],
    )
    if frame.empty:
        return pd.Series(dtype=float, name=value_key)
    return pd.Series(
        data=pd.to_numeric(frame[value_key], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(frame["execution_date"]),
        dtype=float,
        name=value_key,
    ).dropna().sort_index()


def infer_annualization_factor(periods: list[Any]) -> int:
    if not periods:
        return 52
    frame = pd.DataFrame([period.to_dict() if hasattr(period, "to_dict") else period for period in periods])
    return infer_annualization_factor_from_frame(frame)


def infer_annualization_factor_from_frame(periods_frame: pd.DataFrame) -> int:
    if periods_frame.empty:
        return 52
    execution_dates = pd.to_datetime(periods_frame["execution_date"])
    exit_dates = pd.to_datetime(periods_frame["exit_date"])
    avg_days = float((exit_dates - execution_dates).dt.days.mean())
    if not np.isfinite(avg_days) or avg_days <= 0.0:
        return 52
    return max(1, int(round(365.25 / avg_days)))


def annualize_return(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


def max_drawdown(excess_returns: pd.Series) -> float:
    if excess_returns.empty:
        return float("nan")
    wealth = (1.0 + excess_returns.astype(float)).cumprod()
    running_peak = wealth.cummax()
    safe_peak = running_peak.where(running_peak > 0.0, np.nan)
    drawdown = ((safe_peak - wealth) / safe_peak).fillna(0.0)
    return float(drawdown.max())


def run_sell_buffer_sensitivity(
    *,
    holding_period: str,
    prediction_bundle: dict[str, pd.Series],
    window_metadata: dict[str, dict[str, Any]],
    prices: pd.DataFrame,
    universe_by_date: dict[pd.Timestamp, set[str]],
    benchmark_ticker: str,
    cost_model: AlmgrenChrissCostModel,
    base_scheme_config: dict[str, Any],
) -> dict[str, Any]:
    sensitivity_rows: list[dict[str, Any]] = []
    detailed_runs: dict[str, dict[str, Any]] = {}

    for label, buffer_value in SELL_BUFFER_OPTIONS:
        logger.info("sell-buffer sensitivity {} on {}", label, holding_period)
        config = dict(base_scheme_config)
        config["sell_buffer_pct"] = buffer_value
        payload = run_holding_period_experiment(
            holding_period=holding_period,
            prediction_bundle=prediction_bundle,
            window_metadata=window_metadata,
            prices=prices,
            universe_by_date=universe_by_date,
            benchmark_ticker=benchmark_ticker,
            cost_model=cost_model,
            scheme_config=config,
        )
        detailed_runs[label] = payload
        sensitivity_rows.append(
            {
                "sell_buffer_label": label,
                "sell_buffer_pct": buffer_value,
                "annualized_gross_excess": float(payload["aggregate"]["annualized_gross_excess"]),
                "annualized_net_excess": float(payload["aggregate"]["annualized_net_excess"]),
                "sharpe": float(payload["aggregate"]["sharpe"]),
                "average_turnover": float(payload["aggregate"]["average_turnover"]),
                "total_cost_drag": float(payload["aggregate"]["total_cost_drag"]),
                "max_drawdown": float(payload["aggregate"]["max_drawdown"]),
                "rebalance_count": int(payload["aggregate"]["rebalance_count"]),
                "bootstrap_sharpe_ci_lower": float(payload["aggregate"]["bootstrap"]["sharpe_ci_lower"]),
                "bootstrap_sharpe_ci_upper": float(payload["aggregate"]["bootstrap"]["sharpe_ci_upper"]),
            },
        )

    sensitivity_rows.sort(key=lambda row: row["annualized_net_excess"], reverse=True)
    best_label = str(sensitivity_rows[0]["sell_buffer_label"])
    return {
        "holding_period": holding_period,
        "results": sensitivity_rows,
        "best_sell_buffer": best_label,
        "best_run": detailed_runs[best_label],
    }


def select_recommended_configuration(
    *,
    holding_period_reports: dict[str, dict[str, Any]],
    sell_buffer_sensitivity: dict[str, Any],
) -> dict[str, Any]:
    best_period = str(sell_buffer_sensitivity["holding_period"])
    best_buffer = str(sell_buffer_sensitivity["best_sell_buffer"])
    best_buffer_payload = sell_buffer_sensitivity["best_run"]
    baseline_payload = holding_period_reports[best_period]

    return {
        "label": f"{best_period} / sell_buffer {best_buffer}",
        "holding_period": best_period,
        "sell_buffer_label": best_buffer,
        "sell_buffer_pct": next(
            row["sell_buffer_pct"]
            for row in sell_buffer_sensitivity["results"]
            if row["sell_buffer_label"] == best_buffer
        ),
        "annualized_net_excess": float(best_buffer_payload["aggregate"]["annualized_net_excess"]),
        "annualized_gross_excess": float(best_buffer_payload["aggregate"]["annualized_gross_excess"]),
        "sharpe": float(best_buffer_payload["aggregate"]["sharpe"]),
        "average_turnover": float(best_buffer_payload["aggregate"]["average_turnover"]),
        "total_cost_drag": float(best_buffer_payload["aggregate"]["total_cost_drag"]),
        "max_drawdown": float(best_buffer_payload["aggregate"]["max_drawdown"]),
        "rebalance_count": int(best_buffer_payload["aggregate"]["rebalance_count"]),
        "bootstrap": best_buffer_payload["aggregate"]["bootstrap"],
        "improvement_vs_default_buffer": {
            "annualized_net_excess": float(best_buffer_payload["aggregate"]["annualized_net_excess"] - baseline_payload["aggregate"]["annualized_net_excess"]),
            "sharpe": float(best_buffer_payload["aggregate"]["sharpe"] - baseline_payload["aggregate"]["sharpe"]),
            "average_turnover": float(best_buffer_payload["aggregate"]["average_turnover"] - baseline_payload["aggregate"]["average_turnover"]),
        },
    }


def build_go_nogo_impact(recommended_config: dict[str, Any]) -> dict[str, Any]:
    bootstrap = recommended_config["bootstrap"]
    best_net_excess = float(recommended_config["annualized_net_excess"])
    sharpe_ci_lower = float(bootstrap["sharpe_ci_lower"])
    return {
        "net_excess_threshold": 0.05,
        "bootstrap_sharpe_ci_threshold": 0.0,
        "best_net_excess": best_net_excess,
        "best_bootstrap_sharpe_ci_lower": sharpe_ci_lower,
        "net_excess_pass": bool(best_net_excess > 0.05),
        "bootstrap_pass": bool(np.isfinite(sharpe_ci_lower) and sharpe_ci_lower > 0.0),
        "impact_assessment": (
            "GO threshold recovered on economics."
            if best_net_excess > 0.05
            else "Economics still below the 5% net-excess gate."
        ),
    }


def make_report_safe_payload(value: Any) -> Any:
    if isinstance(value, dict):
        safe: dict[str, Any] = {}
        for key, item in value.items():
            if key == "period_net_excess_series":
                continue
            safe[str(key)] = make_report_safe_payload(item)
        return safe
    if isinstance(value, list):
        return [make_report_safe_payload(item) for item in value]
    return value


def empty_cost_breakdown() -> dict[str, float]:
    return {
        "commission": 0.0,
        "spread": 0.0,
        "temporary_impact": 0.0,
        "permanent_impact": 0.0,
        "gap_penalty": 0.0,
        "total": 0.0,
    }


def resolve_prediction_source(args: argparse.Namespace) -> str:
    if args.prediction_source != "auto":
        return str(args.prediction_source)
    return "comparison" if args.comparison_report else "legacy"


def run_phase_e_holding_period_experiment(args: argparse.Namespace, branch: str) -> int:
    comparison_path = REPO_ROOT / (
        args.comparison_report or DEFAULT_COMPARISON_REPORT_PATH
    )
    report_payload = json.loads(comparison_path.read_text())
    as_of = parse_date(args.as_of) if args.as_of else parse_date(str(report_payload["as_of"]))
    benchmark_ticker = str(
        report_payload.get("split_config", {}).get("calendar_ticker", args.benchmark_ticker),
    ).upper()
    rebalance_weekday = int(args.rebalance_weekday)
    holding_periods = parse_phase_e_holding_periods(args.holding_periods)

    prediction_bundle, window_metadata, comparison_windows, horizon_days = reconstruct_phase_e_prediction_bundle(
        comparison_path=comparison_path,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=REPO_ROOT / args.feature_matrix_cache_path,
        label_cache_path=REPO_ROOT / args.label_cache_path,
        as_of=as_of,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )
    prices = build_or_load_phase_e_prices(
        prediction_bundle=prediction_bundle,
        benchmark_ticker=benchmark_ticker,
        comparison_windows=comparison_windows,
        as_of=as_of,
        cache_path=REPO_ROOT / args.phase_e_price_cache_path,
    )
    execution = prepare_execution_price_frame(prices)

    bundle = json.loads((REPO_ROOT / args.bundle_path).read_text())
    scheme_config = build_phase_e_score_weighted_config(bundle.get("turnover_controls", {}))
    cost_params = load_calibrated_cost_parameters(REPO_ROOT / args.cost_calibration_report_path)
    cost_model = AlmgrenChrissCostModel(
        eta=float(cost_params["eta"]),
        gamma=float(cost_params["gamma"]),
        commission_per_share=float(cost_params["commission_per_share"]),
        min_spread_bps=float(cost_params["min_spread_bps"]),
    )

    holding_period_reports: dict[str, dict[str, Any]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for holding_period in holding_periods:
        logger.info("running Phase E holding-period experiment {}", holding_period)
        payload = run_holding_period_experiment(
            holding_period=holding_period,
            prediction_bundle=prediction_bundle,
            window_metadata=window_metadata,
            prices=None,
            execution=execution,
            universe_by_date=None,
            benchmark_ticker=benchmark_ticker,
            cost_model=cost_model,
            scheme_config=scheme_config,
        )
        holding_period_reports[holding_period] = payload
        comparison_rows.append(
            {
                "holding_period": holding_period,
                "annualized_net_excess": float(payload["aggregate"]["annualized_net_excess"]),
                "annualized_gross_excess": float(payload["aggregate"]["annualized_gross_excess"]),
                "sharpe": float(payload["aggregate"]["sharpe"]),
                "average_turnover": float(payload["aggregate"]["average_turnover"]),
                "total_cost_drag": float(payload["aggregate"]["total_cost_drag"]),
                "rebalance_count": int(payload["aggregate"]["rebalance_count"]),
            },
        )

    comparison_rows.sort(key=lambda row: row["annualized_net_excess"], reverse=True)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": "scripts/run_holding_period_experiment.py",
        "mode": "phase_e_comparison_report",
        "inputs": {
            "comparison_report": str(comparison_path),
            "bundle_path": str(REPO_ROOT / args.bundle_path),
            "all_features_path": str(REPO_ROOT / args.all_features_path),
            "feature_matrix_cache_path": str(REPO_ROOT / args.feature_matrix_cache_path),
            "label_cache_path": str(REPO_ROOT / args.label_cache_path),
            "phase_e_price_cache_path": str(REPO_ROOT / args.phase_e_price_cache_path),
            "cost_calibration_report_path": str(REPO_ROOT / args.cost_calibration_report_path),
            "as_of": as_of.isoformat(),
            "benchmark_ticker": benchmark_ticker,
            "rebalance_weekday": rebalance_weekday,
            "horizon_days": horizon_days,
        },
        "phase_e_summary": {
            "mean_test_ic": float(report_payload["summary"]["ridge"]["mean_test_ic"]),
            "mean_test_icir": float(report_payload["summary"]["ridge"]["mean_test_icir"]),
            "positive_windows": int(report_payload["summary"]["ridge"]["win_count"]),
            "windows_completed": int(report_payload["summary"]["ridge"]["windows_completed"]),
            "retained_feature_count": int(report_payload["retained_feature_count"]),
        },
        "score_weighted_config": scheme_config,
        "schedule_assumptions": {
            "20D": "Every 4th weekly signal date (~20 trading days).",
            "40D": "Every 8th weekly signal date (~40 trading days).",
            "60D": "Every 12th weekly signal date (~60 trading days).",
        },
        "comparison_matrix": comparison_rows,
        "holding_periods": {
            holding_period: make_report_safe_payload(payload)
            for holding_period, payload in holding_period_reports.items()
        },
        "best_holding_period": comparison_rows[0]["holding_period"] if comparison_rows else None,
    }
    output_path = REPO_ROOT / args.phase_e_report_path
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved Phase E holding-period report to {}", output_path)
    return 0


def parse_phase_e_holding_periods(raw: str) -> list[str]:
    parts = [item.strip().upper() for item in str(raw).split(",") if item.strip()]
    if not parts:
        raise ValueError("holding_periods must not be empty")
    unsupported = sorted(set(parts) - set(PHASE_E_HOLDING_PERIOD_ORDER))
    if unsupported:
        raise ValueError(f"Unsupported holding periods: {unsupported}")
    return parts


def build_phase_e_score_weighted_config(turnover_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "weighting_scheme": "score_weighted",
        "selection_pct": float(turnover_cfg.get("selection_pct", 0.25)),
        "sell_buffer_pct": float(turnover_cfg.get("sell_buffer_pct", 0.40)),
        "min_trade_weight": float(turnover_cfg.get("min_trade_weight", 0.005)),
        "max_weight": float(turnover_cfg.get("max_weight", 0.05)),
        "min_holdings": int(turnover_cfg.get("min_holdings", 20)),
        "weight_shrinkage": float(turnover_cfg.get("weight_shrinkage", 0.0)),
        "no_trade_zone": float(turnover_cfg.get("no_trade_zone", 0.0)),
        "turnover_penalty_lambda": float(turnover_cfg.get("turnover_penalty_lambda", 0.0)),
    }


def reconstruct_phase_e_prediction_bundle(
    *,
    comparison_path: Path,
    all_features_path: Path,
    feature_matrix_cache_path: Path,
    label_cache_path: Path,
    as_of: date,
    label_buffer_days: int,
    benchmark_ticker: str,
    rebalance_weekday: int,
) -> tuple[dict[str, pd.Series], dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    report_payload = json.loads(comparison_path.read_text())
    comparison_windows = select_report_windows(report_payload, limit=None)
    horizon_days = parse_horizon_days(report_payload)
    artifacts = prepare_horizon_artifacts(
        label=f"{horizon_days}D",
        horizon_days=horizon_days,
        report_path=comparison_path,
        report_payload=report_payload,
        windows=comparison_windows,
        all_features_path=all_features_path,
        feature_matrix_cache_path=feature_matrix_cache_path,
        label_cache_path=label_cache_path,
        as_of=as_of,
        label_buffer_days=label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )

    prediction_bundle: dict[str, pd.Series] = {}
    window_metadata: dict[str, dict[str, Any]] = {}
    for window in comparison_windows:
        dates = {
            key: parse_date(str(window["dates"][key]))
            for key in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            )
        }
        split = slice_all_splits(
            X=artifacts.feature_matrix,
            y=artifacts.labels,
            dates=dates,
            rebalance_weekday=rebalance_weekday,
        )
        alpha = extract_ridge_alpha(window)
        _, test_pred = rebuild_ridge_predictions(
            train_X=split["train_X"],
            train_y=split["train_y"],
            validation_X=split["validation_X"],
            validation_y=split["validation_y"],
            test_X=split["test_X"],
            alpha=alpha,
        )
        window_id = str(window["window_id"])
        prediction_bundle[window_id] = test_pred.sort_index()
        window_metadata[window_id] = {
            "train_period": str(window["dates"]["train"]),
            "validation_period": str(window["dates"]["validation"]),
            "test_period": str(window["dates"]["test"]),
            "test_metrics": dict(window["results"]["ridge"]["test_metrics"]),
            "best_hyperparams": alpha,
        }

    return prediction_bundle, window_metadata, comparison_windows, horizon_days


def build_or_load_phase_e_prices(
    *,
    prediction_bundle: dict[str, pd.Series],
    benchmark_ticker: str,
    comparison_windows: list[dict[str, Any]],
    as_of: date,
    cache_path: Path,
) -> pd.DataFrame:
    if cache_path.exists():
        logger.info("loading cached Phase E price history from {}", cache_path)
        return pd.read_parquet(cache_path)

    tickers = sorted(
        {
            str(ticker).upper()
            for predictions in prediction_bundle.values()
            for ticker in predictions.index.get_level_values("ticker")
        },
    )
    start_date = min(parse_date(str(window["dates"]["test_start"])) for window in comparison_windows)
    end_date = max(parse_date(str(window["dates"]["test_end"])) for window in comparison_windows) + timedelta(days=14)
    prices = get_prices_pit(
        tickers=[*tickers, benchmark_ticker.upper()],
        start_date=start_date,
        end_date=end_date,
        as_of=as_of,
    )
    if prices.empty:
        raise RuntimeError("No PIT prices available for Phase E holding-period experiment.")
    prices.to_parquet(cache_path, index=False)
    logger.info("cached Phase E price history to {}", cache_path)
    return prices


if __name__ == "__main__":
    raise SystemExit(main())
