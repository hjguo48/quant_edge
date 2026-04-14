from __future__ import annotations

"""Compare baseline score-weighted sizing against confidence-tier sizing."""

import argparse
from datetime import date, datetime, timezone
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

from scripts.run_holding_period_experiment import (
    DEFAULT_ALL_FEATURES_PATH,
    DEFAULT_BUNDLE_PATH,
    DEFAULT_COMPARISON_REPORT_PATH,
    DEFAULT_COST_CALIBRATION_REPORT_PATH,
    DEFAULT_FEATURE_MATRIX_CACHE_PATH,
    DEFAULT_LABEL_CACHE_PATH,
    DEFAULT_PHASE_E_PRICE_CACHE_PATH,
    BENCHMARK_TICKER,
    build_or_load_phase_e_prices,
    build_phase_e_score_weighted_config,
    load_calibrated_cost_parameters,
    reconstruct_phase_e_prediction_bundle,
)
from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from scripts.run_turnover_optimization import aggregate_window_portfolios, simulate_score_weighted_controlled
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.execution import (
    PortfolioBacktestResult,
    PortfolioPeriodResult,
    build_execution_schedule,
    prepare_execution_price_frame,
)
from src.portfolio.confidence_sizing import DEFAULT_CONFIDENCE_TIERS, confidence_weighted_portfolio

ALLOWED_BRANCHES = {"feature/s2-stage0-monetization"}
DEFAULT_REPORT_PATH = "data/reports/s2_confidence_sizing_comparison.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch not in ALLOWED_BRANCHES:
        raise RuntimeError(
            f"This script must run on one of {sorted(ALLOWED_BRANCHES)}. Found {branch!r}.",
        )

    comparison_path = REPO_ROOT / args.comparison_report
    report_payload = json.loads(comparison_path.read_text())
    as_of = date.fromisoformat(args.as_of) if args.as_of else date.fromisoformat(str(report_payload["as_of"]))
    benchmark_ticker = str(
        report_payload.get("split_config", {}).get("calendar_ticker", args.benchmark_ticker),
    ).upper()

    prediction_bundle, window_metadata, comparison_windows, horizon_days = reconstruct_phase_e_prediction_bundle(
        comparison_path=comparison_path,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=REPO_ROOT / args.feature_matrix_cache_path,
        label_cache_path=REPO_ROOT / args.label_cache_path,
        as_of=as_of,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=args.rebalance_weekday,
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
    baseline_config = build_phase_e_score_weighted_config(bundle.get("turnover_controls", {}))
    confidence_config = build_confidence_config(args, baseline_config)

    cost_params = load_calibrated_cost_parameters(REPO_ROOT / args.cost_calibration_report_path)
    cost_model = AlmgrenChrissCostModel(
        eta=float(cost_params["eta"]),
        gamma=float(cost_params["gamma"]),
        commission_per_share=float(cost_params["commission_per_share"]),
        min_spread_bps=float(cost_params["min_spread_bps"]),
    )

    baseline_portfolios: list[PortfolioBacktestResult] = []
    confidence_portfolios: list[PortfolioBacktestResult] = []
    per_window: dict[str, Any] = {}

    for window_id in sorted(prediction_bundle):
        predictions = prediction_bundle[window_id]
        logger.info("confidence sizing comparison window={}", window_id)

        baseline_portfolio = simulate_score_weighted_controlled(
            predictions=predictions,
            execution=execution,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
            selection_pct=float(baseline_config["selection_pct"]),
            sell_buffer_pct=baseline_config.get("sell_buffer_pct"),
            min_trade_weight=float(baseline_config["min_trade_weight"]),
            max_weight=float(baseline_config["max_weight"]),
            min_holdings=int(baseline_config["min_holdings"]),
            weight_shrinkage=float(baseline_config.get("weight_shrinkage", 0.0)),
            no_trade_zone=float(baseline_config.get("no_trade_zone", 0.0)),
            turnover_penalty_lambda=float(baseline_config.get("turnover_penalty_lambda", 0.0)),
        )
        confidence_portfolio = simulate_confidence_weighted_controlled(
            predictions=predictions,
            execution=execution,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
            tiers=confidence_config["tiers"],
            selection_pct=float(confidence_config["selection_pct"]),
            sell_buffer_pct=confidence_config.get("sell_buffer_pct"),
            min_trade_weight=float(confidence_config["min_trade_weight"]),
            max_weight=float(confidence_config["max_weight"]),
            min_holdings=int(confidence_config["min_holdings"]),
            weight_shrinkage=float(confidence_config.get("weight_shrinkage", 0.0)),
            no_trade_zone=float(confidence_config.get("no_trade_zone", 0.0)),
            turnover_penalty_lambda=float(confidence_config.get("turnover_penalty_lambda", 0.0)),
        )

        baseline_portfolios.append(baseline_portfolio)
        confidence_portfolios.append(confidence_portfolio)

        per_window[window_id] = {
            "test_period": window_metadata[window_id]["test_period"],
            "ridge_test_metrics": window_metadata[window_id]["test_metrics"],
            "score_weighted": with_sharpe_alias(aggregate_window_portfolios([baseline_portfolio])),
            "confidence_weighted": with_sharpe_alias(aggregate_window_portfolios([confidence_portfolio])),
        }

    baseline_summary = with_sharpe_alias(aggregate_window_portfolios(baseline_portfolios))
    confidence_summary = with_sharpe_alias(aggregate_window_portfolios(confidence_portfolios))
    comparison = {
        "annualized_net_excess_delta": float(
            confidence_summary["annualized_net_excess"] - baseline_summary["annualized_net_excess"],
        ),
        "sharpe_delta": float(confidence_summary["sharpe"] - baseline_summary["sharpe"]),
        "average_turnover_delta": float(
            confidence_summary["average_turnover"] - baseline_summary["average_turnover"],
        ),
        "total_cost_drag_delta": float(
            confidence_summary["total_cost_drag"] - baseline_summary["total_cost_drag"],
        ),
        "winner_by_net_excess": (
            "confidence_weighted"
            if confidence_summary["annualized_net_excess"] >= baseline_summary["annualized_net_excess"]
            else "score_weighted"
        ),
    }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": "scripts/run_confidence_sizing_experiment.py",
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
            "horizon_days": horizon_days,
        },
        "baseline_score_weighted_config": baseline_config,
        "confidence_weighted_config": {
            **{key: value for key, value in confidence_config.items() if key != "tiers"},
            "tiers": [
                {"upper_rank_pct": float(bound), "multiplier": float(multiplier)}
                for bound, multiplier in confidence_config["tiers"]
            ],
        },
        "summary": {
            "score_weighted": baseline_summary,
            "confidence_weighted": confidence_summary,
            "comparison": comparison,
        },
        "per_window": per_window,
    }
    output_path = REPO_ROOT / args.output
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved confidence sizing comparison report to {}", output_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare score-weighted vs confidence-tier weighted monetization on Phase E Ridge v2 predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--comparison-report", default=DEFAULT_COMPARISON_REPORT_PATH)
    parser.add_argument("--bundle-path", default=DEFAULT_BUNDLE_PATH)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--feature-matrix-cache-path", default=DEFAULT_FEATURE_MATRIX_CACHE_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--phase-e-price-cache-path", default=DEFAULT_PHASE_E_PRICE_CACHE_PATH)
    parser.add_argument("--cost-calibration-report-path", default=DEFAULT_COST_CALIBRATION_REPORT_PATH)
    parser.add_argument("--output", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--as-of")
    parser.add_argument("--label-buffer-days", type=int, default=60)
    parser.add_argument("--rebalance-weekday", type=int, default=4)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    return parser.parse_args(argv)


def build_confidence_config(args: argparse.Namespace, baseline_config: dict[str, Any]) -> dict[str, Any]:
    effective_selection_pct = max(
        float(baseline_config.get("selection_pct", 0.25)),
        float(DEFAULT_CONFIDENCE_TIERS[-1][0]),
    )
    return {
        **baseline_config,
        "selection_pct": effective_selection_pct,
        "tiers": DEFAULT_CONFIDENCE_TIERS,
    }


def simulate_confidence_weighted_controlled(
    *,
    predictions: pd.Series,
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str = "SPY",
    initial_capital: float = 1_000_000.0,
    tiers: tuple[tuple[float, float], ...] = DEFAULT_CONFIDENCE_TIERS,
    selection_pct: float = 0.30,
    sell_buffer_pct: float | None = 0.40,
    min_trade_weight: float = 0.005,
    max_weight: float = 0.05,
    min_holdings: int = 20,
    weight_shrinkage: float = 0.0,
    no_trade_zone: float = 0.0,
    turnover_penalty_lambda: float = 0.0,
) -> PortfolioBacktestResult:
    if predictions.empty:
        return PortfolioBacktestResult(
            periods=[],
            gross_return=0.0,
            net_return=0.0,
            benchmark_return=0.0,
            gross_excess_return=0.0,
            net_excess_return=0.0,
            annualized_gross_return=0.0,
            annualized_net_return=0.0,
            annualized_benchmark_return=0.0,
            annualized_excess_gross=0.0,
            annualized_excess_net=0.0,
            total_cost_drag=0.0,
            average_turnover=0.0,
            cost_breakdown=empty_cost_breakdown(),
        )

    benchmark = benchmark_ticker.upper()
    signal_dates = pd.DatetimeIndex(
        pd.to_datetime(predictions.index.get_level_values("trade_date")),
    ).sort_values().unique()
    trade_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    schedule = build_execution_schedule(signal_dates, trade_dates)

    current_weights: dict[str, float] = {}
    portfolio_value = float(initial_capital)
    periods: list[PortfolioPeriodResult] = []
    cost_totals = empty_cost_breakdown()

    for signal_date, next_signal_date in zip(signal_dates[:-1], signal_dates[1:]):
        execution_date = schedule.get(pd.Timestamp(signal_date))
        exit_date = schedule.get(pd.Timestamp(next_signal_date))
        if execution_date is None or exit_date is None or exit_date <= execution_date:
            continue

        score_frame = (
            predictions.xs(signal_date, level="trade_date")
            .dropna()
            .astype(float)
            .sort_values(ascending=False)
        )
        if (execution_date, benchmark) not in execution.index or (exit_date, benchmark) not in execution.index:
            continue

        entry_slice = execution.xs(execution_date, level="trade_date")
        exit_slice = execution.xs(exit_date, level="trade_date")
        eligible = (
            set(score_frame.index.astype(str))
            & set(entry_slice.index.astype(str))
            & set(exit_slice.index.astype(str))
        )
        eligible.discard(benchmark)
        if len(eligible) < max(min_holdings, 2):
            continue

        filtered_scores = score_frame.loc[score_frame.index.astype(str).isin(eligible)].sort_values(ascending=False)
        if filtered_scores.empty:
            continue

        if turnover_penalty_lambda > 0.0 and current_weights:
            adjusted_scores = filtered_scores.copy()
            for ticker in adjusted_scores.index.astype(str):
                if ticker in current_weights:
                    adjusted_scores.loc[ticker] += turnover_penalty_lambda * current_weights[ticker]
            filtered_scores = adjusted_scores.sort_values(ascending=False)

        target_weights = confidence_weighted_portfolio(
            filtered_scores,
            current_weights,
            tiers=tiers,
            selection_pct=selection_pct,
            sell_buffer_pct=0.0 if sell_buffer_pct is None else float(sell_buffer_pct),
            weight_shrinkage=weight_shrinkage,
            no_trade_zone=no_trade_zone,
            min_trade_weight=min_trade_weight,
            max_weight=max_weight,
            min_holdings=min_holdings,
        )
        if not target_weights:
            continue

        previous_weights = current_weights.copy()
        period_costs = empty_cost_breakdown()
        all_trade_tickers = set(previous_weights) | set(target_weights)
        for ticker in all_trade_tickers:
            delta_weight = target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)
            if abs(delta_weight) < 1e-12 or ticker not in entry_slice.index:
                continue
            bar = entry_slice.loc[ticker]
            order_notional = abs(delta_weight) * portfolio_value
            order_shares = order_notional / max(float(bar["execution_price"]), 1e-12)
            estimate = cost_model.estimate_trade(
                order_shares=order_shares,
                execution_price=float(bar["execution_price"]),
                sigma_20d=float(bar["sigma_20d"]),
                adv_20d_shares=float(bar["adv_20d_shares"]),
                open_gap=float(bar["open_gap"]),
                execution_volume_ratio=float(bar["volume_ratio"]),
            )
            period_costs["commission"] += estimate.commission_cost
            period_costs["spread"] += estimate.spread_cost
            period_costs["temporary_impact"] += estimate.temporary_cost
            period_costs["permanent_impact"] += estimate.permanent_cost
            period_costs["gap_penalty"] += estimate.gap_cost
            period_costs["total"] += estimate.total_cost

        cost_rate = period_costs["total"] / portfolio_value if portfolio_value else 0.0

        asset_returns: dict[str, float] = {}
        gross_return = 0.0
        gaps: list[float] = []
        for ticker, weight in target_weights.items():
            if ticker not in entry_slice.index or ticker not in exit_slice.index:
                continue
            entry_bar = entry_slice.loc[ticker]
            exit_bar = exit_slice.loc[ticker]
            entry_price = max(float(entry_bar["execution_price"]), 1e-12)
            exit_price = float(exit_bar["execution_price"])
            realized = (exit_price / entry_price) - 1.0
            asset_returns[ticker] = realized
            gross_return += weight * realized
            gaps.append(abs(float(entry_bar["open_gap"])))

        bench_entry = entry_slice.loc[benchmark]
        bench_exit = exit_slice.loc[benchmark]
        benchmark_return = (
            float(bench_exit["execution_price"]) / max(float(bench_entry["execution_price"]), 1e-12)
        ) - 1.0
        net_return = ((1.0 - cost_rate) * (1.0 + gross_return)) - 1.0

        portfolio_value *= 1.0 + net_return
        drift_denominator = 1.0 + gross_return
        if abs(drift_denominator) < 1e-12:
            current_weights = {}
        else:
            current_weights = {
                ticker: float((weight * (1.0 + asset_returns.get(ticker, 0.0))) / drift_denominator)
                for ticker, weight in target_weights.items()
                if (weight * (1.0 + asset_returns.get(ticker, 0.0))) > 1e-8
            }

        turnover = 0.5 * sum(
            abs(target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0))
            for ticker in set(target_weights) | set(previous_weights)
        )
        periods.append(
            PortfolioPeriodResult(
                signal_date=pd.Timestamp(signal_date).date().isoformat(),
                execution_date=pd.Timestamp(execution_date).date().isoformat(),
                exit_date=pd.Timestamp(exit_date).date().isoformat(),
                universe_size=int(len(filtered_scores)),
                selected_count=int(len(target_weights)),
                turnover=float(turnover),
                cost_rate=float(cost_rate),
                gross_return=float(gross_return),
                net_return=float(net_return),
                benchmark_return=float(benchmark_return),
                gross_excess_return=float(gross_return - benchmark_return),
                net_excess_return=float(net_return - benchmark_return),
                avg_gap=float(np.mean(gaps)) if gaps else 0.0,
                cost_breakdown={key: float(value) for key, value in period_costs.items()},
                selected_tickers=list(target_weights.keys()),
            ),
        )
        for key, value in period_costs.items():
            cost_totals[key] += float(value)

    return build_backtest_result(periods=periods, cost_totals=cost_totals)


def build_backtest_result(
    *,
    periods: list[PortfolioPeriodResult],
    cost_totals: dict[str, float],
) -> PortfolioBacktestResult:
    if not periods:
        return PortfolioBacktestResult(
            periods=[],
            gross_return=0.0,
            net_return=0.0,
            benchmark_return=0.0,
            gross_excess_return=0.0,
            net_excess_return=0.0,
            annualized_gross_return=0.0,
            annualized_net_return=0.0,
            annualized_benchmark_return=0.0,
            annualized_excess_gross=0.0,
            annualized_excess_net=0.0,
            total_cost_drag=0.0,
            average_turnover=0.0,
            cost_breakdown=cost_totals,
        )

    cumulative_gross = np.prod([1.0 + period.gross_return for period in periods]) - 1.0
    cumulative_net = np.prod([1.0 + period.net_return for period in periods]) - 1.0
    cumulative_benchmark = np.prod([1.0 + period.benchmark_return for period in periods]) - 1.0
    start_date = pd.Timestamp(periods[0].execution_date)
    end_date = pd.Timestamp(periods[-1].exit_date)
    total_days = max((end_date - start_date).days, 1)
    annualized_gross = annualize_return(float(cumulative_gross), total_days)
    annualized_net = annualize_return(float(cumulative_net), total_days)
    annualized_benchmark = annualize_return(float(cumulative_benchmark), total_days)

    return PortfolioBacktestResult(
        periods=periods,
        gross_return=float(cumulative_gross),
        net_return=float(cumulative_net),
        benchmark_return=float(cumulative_benchmark),
        gross_excess_return=float(cumulative_gross - cumulative_benchmark),
        net_excess_return=float(cumulative_net - cumulative_benchmark),
        annualized_gross_return=float(annualized_gross),
        annualized_net_return=float(annualized_net),
        annualized_benchmark_return=float(annualized_benchmark),
        annualized_excess_gross=float(annualized_gross - annualized_benchmark),
        annualized_excess_net=float(annualized_net - annualized_benchmark),
        total_cost_drag=float(annualized_gross - annualized_net),
        average_turnover=float(np.mean([period.turnover for period in periods])),
        cost_breakdown={key: float(value) for key, value in cost_totals.items()},
    )


def annualize_return(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


def empty_cost_breakdown() -> dict[str, float]:
    return {
        "commission": 0.0,
        "spread": 0.0,
        "temporary_impact": 0.0,
        "permanent_impact": 0.0,
        "gap_penalty": 0.0,
        "total": 0.0,
    }


def with_sharpe_alias(metrics: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(metrics)
    enriched["sharpe"] = float(metrics.get("sharpe_proxy", 0.0))
    return enriched


if __name__ == "__main__":
    raise SystemExit(main())
