"""S1.12: SPA retest — score_weighted_controlled vs equal_weight.

Re-runs the best optimized score-weighted config and equal-weight baseline,
collects period-level net excess returns, and runs a paired t-test (SPA fallback)
to determine whether the optimized scheme is statistically superior.

Usage:
    python scripts/run_spa_retest.py
    python scripts/run_spa_retest.py --turnover-report data/reports/turnover_optimization.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

from loguru import logger
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from scripts.run_turnover_optimization import (
    build_prediction_series_by_window,
    simulate_score_weighted_controlled,
    aggregate_window_portfolios,
)
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.execution import (
    PortfolioBacktestResult,
    prepare_execution_price_frame,
    simulate_portfolio,
)
from src.stats.spa import run_spa_fallback

DEFAULT_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_TURNOVER_REPORT_PATH = "data/reports/turnover_optimization.json"
DEFAULT_REPORT_PATH = "data/reports/spa_retest.json"
BENCHMARK_TICKER = "SPY"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    logger.info("S1.12: SPA retest — score_weighted_controlled vs equal_weight")

    # Load turnover optimization report for best config
    turnover_report_path = REPO_ROOT / args.turnover_report
    with open(turnover_report_path) as f:
        turnover_report = json.load(f)

    best_config_data = turnover_report["phase_b_gate"]["best_passing_config"]
    if best_config_data is None:
        logger.error("No passing config found in turnover report — cannot run SPA retest")
        return 1

    best_config = best_config_data["config"]
    best_scheme_name = best_config_data["scheme_name"]
    logger.info("Best config from Phase B: {} (net_excess={:.4f})",
                best_scheme_name, best_config_data["annualized_net_excess"])

    # Also grab top-3 Pareto configs for comprehensive comparison
    pareto_configs = []
    for entry in turnover_report.get("pareto_front", []):
        if entry.get("period_count", 0) > 0:
            pareto_configs.append(entry)
    pareto_configs.sort(key=lambda r: r["annualized_net_excess"], reverse=True)
    pareto_top3 = pareto_configs[:3]

    # Load data
    predictions_df = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices_df = pd.read_parquet(REPO_ROOT / args.prices_path)
    logger.info("loaded predictions={} prices={}", len(predictions_df), len(prices_df))

    predictions_by_window = build_prediction_series_by_window(predictions_df)
    cost_model = AlmgrenChrissCostModel()

    logger.info("pre-computing execution price frame...")
    execution = prepare_execution_price_frame(prices_df)
    logger.info("execution frame ready: {} rows", len(execution))

    # --- Run equal-weight baseline (benchmark for SPA)
    logger.info("=== Running equal-weight baseline ===")
    ew_periods = collect_period_returns(
        run_equal_weight(predictions_by_window, prices_df, cost_model, BENCHMARK_TICKER)
    )
    ew_agg = aggregate_from_periods(ew_periods)
    logger.info("EW baseline: {} periods, net_excess={:.4f}, turnover={:.4f}",
                len(ew_periods), ew_agg["annualized_net_excess"], ew_agg["average_turnover"])

    # --- Run best score-weighted config
    logger.info("=== Running best score-weighted config ===")
    sw_best_portfolios = run_sw_config(
        best_config, predictions_by_window, execution, cost_model, BENCHMARK_TICKER
    )
    sw_best_periods = collect_period_returns(sw_best_portfolios)
    sw_best_agg = aggregate_from_periods(sw_best_periods)
    logger.info("SW best: {} periods, net_excess={:.4f}, turnover={:.4f}",
                len(sw_best_periods), sw_best_agg["annualized_net_excess"], sw_best_agg["average_turnover"])

    # --- Run Pareto top-3 configs
    pareto_results: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(pareto_top3):
        cfg = entry["config"]
        scheme_name = entry["scheme_name"]
        if scheme_name == best_scheme_name:
            # Already ran this one
            pareto_results[scheme_name] = {
                "periods": sw_best_periods,
                "agg": sw_best_agg,
                "config": cfg,
            }
            continue
        logger.info("=== Running Pareto #{}: {} ===", i + 1, scheme_name)
        portfolios = run_sw_config(cfg, predictions_by_window, execution, cost_model, BENCHMARK_TICKER)
        periods = collect_period_returns(portfolios)
        agg = aggregate_from_periods(periods)
        logger.info("  {} periods, net_excess={:.4f}, turnover={:.4f}",
                    len(periods), agg["annualized_net_excess"], agg["average_turnover"])
        pareto_results[scheme_name] = {"periods": periods, "agg": agg, "config": cfg}

    # --- Build SPA test: score_weighted_controlled vs equal_weight
    logger.info("=== Running SPA tests ===")

    # Convert period returns to pd.Series indexed by date
    ew_series = _periods_to_series(ew_periods)

    # Primary test: best SW vs EW
    sw_best_series = _periods_to_series(sw_best_periods)
    primary_spa = run_spa_fallback(
        benchmark_series=ew_series,
        competitors={"score_weighted_controlled": sw_best_series},
        benchmark_name="equal_weight",
    )
    logger.info("Primary SPA: p={:.6f} significant={}", primary_spa.p_value, primary_spa.significant)

    # Comprehensive test: all Pareto configs vs EW
    all_competitors: dict[str, pd.Series] = {}
    for scheme_name, data in pareto_results.items():
        all_competitors[scheme_name] = _periods_to_series(data["periods"])
    # Add best if not already in pareto
    if best_scheme_name not in all_competitors:
        all_competitors[best_scheme_name] = sw_best_series

    comprehensive_spa = run_spa_fallback(
        benchmark_series=ew_series,
        competitors=all_competitors,
        benchmark_name="equal_weight",
    )
    logger.info("Comprehensive SPA: p={:.6f} significant={}", comprehensive_spa.p_value, comprehensive_spa.significant)

    # --- Build report
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_spa_retest.py",
        "task": "S1.12",
        "description": "SPA retest: score_weighted_controlled vs equal_weight after Phase B turnover optimization",
        "source_config": {
            "turnover_report": str(args.turnover_report),
            "best_scheme": best_scheme_name,
            "best_config": best_config,
        },
        "schemes": {
            "equal_weight": {
                "config": {"weighting_scheme": "equal_weight", "selection_pct": 0.20, "sell_buffer_pct": 0.25, "min_trade_weight": 0.01},
                **ew_agg,
                "period_count": len(ew_periods),
            },
            best_scheme_name: {
                "config": best_config,
                **sw_best_agg,
                "period_count": len(sw_best_periods),
            },
        },
        "spa_primary": {
            "description": f"{best_scheme_name} vs equal_weight",
            "result": primary_spa.to_dict(),
        },
        "spa_comprehensive": {
            "description": "All Pareto-optimal configs vs equal_weight",
            "result": comprehensive_spa.to_dict(),
        },
        "conclusion": _build_conclusion(primary_spa, sw_best_agg, ew_agg, best_scheme_name),
    }

    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved SPA retest report to {}", report_path)

    # Summary
    logger.info("=" * 60)
    logger.info("S1.12 SPA Retest Summary")
    logger.info("  EW baseline:  net_excess={:.4f}  turnover={:.4f}",
                ew_agg["annualized_net_excess"], ew_agg["average_turnover"])
    logger.info("  SW optimized: net_excess={:.4f}  turnover={:.4f}",
                sw_best_agg["annualized_net_excess"], sw_best_agg["average_turnover"])
    logger.info("  SPA p-value: {:.6f}  significant: {}",
                primary_spa.p_value, primary_spa.significant)
    if primary_spa.comparisons:
        c = primary_spa.comparisons[0]
        logger.info("  t-statistic: {:.4f}  n_obs: {}  mean_delta: {:.6f}",
                    c.t_statistic, c.n_observations, c.mean_ic_delta)
    logger.info("  Decision: {}", report["conclusion"]["decision"])

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1.12 SPA retest")
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--turnover-report", default=DEFAULT_TURNOVER_REPORT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


# ===================================================================
# Simulation runners
# ===================================================================

def run_sw_config(
    config: dict[str, Any],
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
) -> list[PortfolioBacktestResult]:
    """Run score-weighted controlled config across all windows, return per-window results."""
    window_portfolios: list[PortfolioBacktestResult] = []
    for window_id in sorted(predictions_by_window):
        portfolio = simulate_score_weighted_controlled(
            predictions=predictions_by_window[window_id],
            execution=execution,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
            selection_pct=float(config["selection_pct"]),
            sell_buffer_pct=config.get("sell_buffer_pct"),
            min_trade_weight=float(config["min_trade_weight"]),
            max_weight=float(config["max_weight"]),
            min_holdings=int(config["min_holdings"]),
            weight_shrinkage=float(config.get("weight_shrinkage", 0.0)),
            no_trade_zone=float(config.get("no_trade_zone", 0.0)),
            turnover_penalty_lambda=float(config.get("turnover_penalty_lambda", 0.0)),
        )
        window_portfolios.append(portfolio)
    return window_portfolios


def run_equal_weight(
    predictions_by_window: dict[str, pd.Series],
    prices_df: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
) -> list[PortfolioBacktestResult]:
    """Run equal-weight baseline across all windows."""
    window_portfolios: list[PortfolioBacktestResult] = []
    for window_id in sorted(predictions_by_window):
        portfolio = simulate_portfolio(
            predictions=predictions_by_window[window_id],
            prices=prices_df,
            cost_model=cost_model,
            weighting_scheme="equal_weight",
            benchmark_ticker=benchmark_ticker,
            universe_by_date=None,
            selection_pct=0.20,
            sell_buffer_pct=0.25,
            min_trade_weight=0.01,
            max_weight=0.05,
            min_holdings=20,
        )
        window_portfolios.append(portfolio)
    return window_portfolios


# ===================================================================
# Period-level extraction
# ===================================================================

def collect_period_returns(portfolios: list[PortfolioBacktestResult]) -> list[dict[str, Any]]:
    """Extract period-level net excess returns from portfolio results."""
    periods: list[dict[str, Any]] = []
    for portfolio in portfolios:
        for period in portfolio.periods:
            periods.append({
                "execution_date": period.execution_date,
                "exit_date": period.exit_date,
                "net_excess_return": period.net_excess_return,
                "gross_excess_return": period.gross_excess_return,
                "gross_return": period.gross_return,
                "net_return": period.net_return,
                "benchmark_return": period.benchmark_return,
                "turnover": period.turnover,
            })
    return periods


def aggregate_from_periods(periods: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics from period-level data."""
    if not periods:
        return {
            "annualized_net_excess": 0.0,
            "annualized_gross_excess": 0.0,
            "average_turnover": 0.0,
            "sharpe_proxy": 0.0,
            "total_cost_drag": 0.0,
        }
    df = pd.DataFrame(periods)
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df.sort_values("execution_date", inplace=True)

    gross_curve = (1.0 + df["gross_return"]).cumprod()
    net_curve = (1.0 + df["net_return"]).cumprod()
    bench_curve = (1.0 + df["benchmark_return"]).cumprod()
    total_days = max((df["exit_date"].iloc[-1] - df["execution_date"].iloc[0]).days, 1)

    ann_gross = _annualize(float(gross_curve.iloc[-1] - 1.0), total_days)
    ann_net = _annualize(float(net_curve.iloc[-1] - 1.0), total_days)
    ann_bench = _annualize(float(bench_curve.iloc[-1] - 1.0), total_days)

    net_excess_series = df["net_return"] - df["benchmark_return"]
    sharpe_proxy = (
        float(net_excess_series.mean() / net_excess_series.std() * np.sqrt(52))
        if len(net_excess_series) > 1 and net_excess_series.std() > 0
        else 0.0
    )

    return {
        "annualized_gross_excess": float(ann_gross - ann_bench),
        "annualized_net_excess": float(ann_net - ann_bench),
        "annualized_gross_return": float(ann_gross),
        "annualized_net_return": float(ann_net),
        "annualized_benchmark_return": float(ann_bench),
        "total_cost_drag": float(ann_gross - ann_net),
        "average_turnover": float(df["turnover"].mean()),
        "sharpe_proxy": sharpe_proxy,
    }


def _annualize(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0:
        return -1.0
    return float(base ** (365.25 / total_days) - 1.0)


def _periods_to_series(periods: list[dict[str, Any]]) -> pd.Series:
    """Convert period records to a pd.Series of net_excess_return indexed by date."""
    if not periods:
        return pd.Series(dtype=float)
    df = pd.DataFrame(periods)
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    return pd.Series(
        df["net_excess_return"].astype(float).values,
        index=df["execution_date"],
        dtype=float,
    ).sort_index()


def _build_conclusion(
    spa_result: Any,
    sw_agg: dict[str, Any],
    ew_agg: dict[str, Any],
    sw_name: str,
) -> dict[str, Any]:
    """Build conclusion based on SPA test result."""
    if spa_result.significant:
        decision = f"adopt_{sw_name}"
        rationale = (
            f"{sw_name} delivers significantly higher net excess return ({sw_agg['annualized_net_excess']:.4f} "
            f"vs {ew_agg['annualized_net_excess']:.4f}) with lower turnover "
            f"({sw_agg['average_turnover']:.4f} vs {ew_agg['average_turnover']:.4f}). "
            f"SPA p-value={spa_result.p_value:.6f} confirms statistical significance."
        )
    else:
        decision = "keep_equal_weight"
        rationale = (
            f"{sw_name} shows higher net excess ({sw_agg['annualized_net_excess']:.4f} "
            f"vs {ew_agg['annualized_net_excess']:.4f}) but the SPA test "
            f"(p={spa_result.p_value:.6f}) does not establish significance at alpha=0.05. "
            f"Note: with weekly rebalancing periods (n~196), statistical power is limited."
        )

    return {
        "decision": decision,
        "rationale": rationale,
        "spa_significant": spa_result.significant,
        "spa_p_value": spa_result.p_value,
        "net_excess_delta": sw_agg["annualized_net_excess"] - ew_agg["annualized_net_excess"],
        "turnover_delta": sw_agg["average_turnover"] - ew_agg["average_turnover"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
