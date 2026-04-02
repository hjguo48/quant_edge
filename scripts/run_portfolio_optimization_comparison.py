from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.execution import (
    PortfolioBacktestResult,
    PortfolioPeriodResult,
    build_execution_schedule,
    prepare_execution_price_frame,
    select_candidate_tickers,
    simulate_portfolio,
)
from src.data.db.session import get_engine
from src.portfolio.black_litterman import (
    BlackLittermanPosterior,
    black_litterman_portfolio,
    build_black_litterman_posterior,
)
from src.portfolio.constraints import CVXPYOptimizer, PortfolioConstraints, apply_turnover_buffer
from src.portfolio.equal_weight import equal_weight_portfolio
from src.portfolio.shrinkage import CovarianceResult
from src.portfolio.vol_weighted import vol_inverse_portfolio
from src.risk.portfolio_risk import PortfolioRiskEngine, compute_sector_weights
from src.stats.spa import run_spa_fallback

EXPECTED_BRANCH = "feature/week15-portfolio-optimization"
BENCHMARK_TICKER = "SPY"
DEFAULT_EXTENDED_REPORT_PATH = "data/reports/extended_walkforward.json"
DEFAULT_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_REPORT_PATH = "data/reports/portfolio_optimization_comparison.json"

BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "equal_weight_buffered": {
        "weighting_scheme": "equal_weight",
        "selection_pct": 0.20,
        "sell_buffer_pct": 0.25,
        "min_trade_weight": 0.01,
        "max_weight": 0.05,
        "min_holdings": 20,
    },
    "vol_inverse_buffered": {
        "weighting_scheme": "vol_inverse",
        "selection_pct": 0.20,
        "sell_buffer_pct": 0.25,
        "min_trade_weight": 0.01,
        "max_weight": 0.05,
        "min_holdings": 20,
    },
}

OPTIMIZED_CONFIGS: dict[str, dict[str, Any]] = {
    "black_litterman_upgraded": {
        "selection_pct": 0.20,
        "sell_buffer_pct": 0.25,
        "max_weight": 0.10,
        "min_holdings": 20,
        "lookback_days": 60,
        "tau": 0.05,
        "risk_aversion": 2.5,
        "score_scale": 0.05,
        "lambda_risk": 1.0,
        "lambda_turnover": 0.005,
        "max_sector_deviation": 0.10,
        "beta_bounds": (0.8, 1.2),
        "portfolio_size": 1e7,
        "use_cvxpy": False,
    },
    "cvxpy_optimized": {
        "selection_pct": 0.20,
        "sell_buffer_pct": 0.25,
        "max_weight": 0.10,
        "min_holdings": 20,
        "lookback_days": 60,
        "tau": 0.05,
        "risk_aversion": 2.5,
        "score_scale": 0.05,
        "lambda_risk": 1.0,
        "lambda_turnover": 0.005,
        "max_sector_deviation": 0.10,
        "beta_bounds": (0.8, 1.2),
        "portfolio_size": 1e7,
        "use_cvxpy": True,
    },
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}.")

    extended_report = json.loads((REPO_ROOT / args.extended_report_path).read_text())
    predictions = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices = pd.read_parquet(REPO_ROOT / args.prices_path)
    sector_map = load_sector_map()

    window_metadata = {
        str(window["window_id"]): window
        for window in extended_report["walkforward"]["windows"]
    }
    predictions_by_window = build_prediction_series_by_window(predictions)

    calibration = calibrate_cost_model(
        predictions_by_window=predictions_by_window,
        prices=prices,
        scheme_config=BASELINE_CONFIGS["equal_weight_buffered"],
        benchmark_ticker=args.benchmark_ticker,
    )
    calibrated_cost_model = AlmgrenChrissCostModel(
        eta=float(calibration["calibrated"]["eta"]),
        gamma=float(calibration["calibrated"]["gamma"]),
    )
    logger.info(
        "cost calibration ratio={:.4f} eta {:.4f}->{:.4f} gamma {:.4f}->{:.4f}",
        calibration["calibration_ratio"],
        calibration["original"]["eta"],
        calibration["calibrated"]["eta"],
        calibration["original"]["gamma"],
        calibration["calibrated"]["gamma"],
    )

    scheme_reports: dict[str, dict[str, Any]] = {}
    shrinkage_records: list[CovarianceResult] = []

    for scheme_name, config in BASELINE_CONFIGS.items():
        scheme_reports[scheme_name] = run_baseline_scheme(
            scheme_name=scheme_name,
            scheme_config=config,
            predictions_by_window=predictions_by_window,
            prices=prices,
            cost_model=calibrated_cost_model,
            benchmark_ticker=args.benchmark_ticker,
            window_metadata=window_metadata,
        )

    for scheme_name, config in OPTIMIZED_CONFIGS.items():
        scheme_payload = run_optimized_scheme(
            scheme_name=scheme_name,
            scheme_config=config,
            predictions_by_window=predictions_by_window,
            prices=prices,
            cost_model=calibrated_cost_model,
            benchmark_ticker=args.benchmark_ticker,
            window_metadata=window_metadata,
            sector_map=sector_map,
        )
        shrinkage_records.extend(scheme_payload["raw_diagnostics"]["shrinkage_records"])
        scheme_payload.pop("raw_diagnostics", None)
        scheme_reports[scheme_name] = scheme_payload

    spa_payload = build_spa_payload(scheme_reports)
    shrinkage_stats = summarize_shrinkage(shrinkage_records)
    recommendation = build_recommendation(
        scheme_reports=scheme_reports,
        spa_payload=spa_payload,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": Path(__file__).name,
        "inputs": {
            "extended_report_path": str(REPO_ROOT / args.extended_report_path),
            "predictions_path": str(REPO_ROOT / args.predictions_path),
            "prices_path": str(REPO_ROOT / args.prices_path),
            "benchmark_ticker": args.benchmark_ticker.upper(),
            "optimal_horizon": str(extended_report["horizon_experiment"]["optimal_horizon"]),
        },
        "schemes": scheme_reports,
        "spa_test": spa_payload,
        "cost_calibration": calibration,
        "shrinkage_stats": shrinkage_stats,
        "recommendation": recommendation,
    }

    output_path = REPO_ROOT / args.report_path
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved portfolio optimization comparison report to {}", output_path)
    logger.info(
        "best_scheme={} best_net_excess={:.6f} recommendation={}",
        recommendation["best_scheme"],
        recommendation["best_net_excess"],
        recommendation["decision"],
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare buffered, Black-Litterman, and CVXPY portfolio schemes on the 8-window walk-forward cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--extended-report-path", default=DEFAULT_EXTENDED_REPORT_PATH)
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


def load_sector_map() -> dict[str, str]:
    engine = get_engine()
    with engine.connect() as conn:
        stocks = pd.read_sql(text("select ticker, sector from stocks"), conn)
    stocks["ticker"] = stocks["ticker"].astype(str).str.upper()
    stocks["sector"] = stocks["sector"].fillna("Unknown").astype(str)
    return stocks.set_index("ticker")["sector"].to_dict()


def build_prediction_series_by_window(predictions: pd.DataFrame) -> dict[str, pd.Series]:
    frame = predictions.copy()
    frame["window_id"] = frame["window_id"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    series_by_window: dict[str, pd.Series] = {}
    for window_id, window_frame in frame.groupby("window_id", sort=True):
        series = (
            window_frame
            .sort_values(["trade_date", "ticker"])
            .set_index(["trade_date", "ticker"])["score"]
            .astype(float)
        )
        series_by_window[str(window_id)] = series
    return series_by_window


def run_baseline_scheme(
    *,
    scheme_name: str,
    scheme_config: dict[str, Any],
    predictions_by_window: dict[str, pd.Series],
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
    window_metadata: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    window_results: list[dict[str, Any]] = []
    for window_id in sorted(predictions_by_window):
        portfolio = simulate_portfolio(
            predictions=predictions_by_window[window_id],
            prices=prices,
            cost_model=cost_model,
            weighting_scheme=scheme_config["weighting_scheme"],
            benchmark_ticker=benchmark_ticker,
            universe_by_date=None,
            selection_pct=float(scheme_config["selection_pct"]),
            sell_buffer_pct=scheme_config["sell_buffer_pct"],
            min_trade_weight=float(scheme_config["min_trade_weight"]),
            max_weight=float(scheme_config["max_weight"]),
            min_holdings=int(scheme_config["min_holdings"]),
        )
        window_results.append(build_window_payload(
            window_id=window_id,
            metadata=window_metadata[window_id],
            portfolio=portfolio,
        ))

    aggregate = aggregate_scheme_results(window_results)
    return {
        "config": scheme_config,
        "aggregate": aggregate,
        "windows": window_results,
        "series": build_period_series(window_results),
    }


def run_optimized_scheme(
    *,
    scheme_name: str,
    scheme_config: dict[str, Any],
    predictions_by_window: dict[str, pd.Series],
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
    window_metadata: dict[str, dict[str, Any]],
    sector_map: dict[str, str],
) -> dict[str, Any]:
    window_results: list[dict[str, Any]] = []
    diagnostics = {
        "shrinkage_records": [],
        "solver_status_counts": Counter(),
        "risk_rule_triggers": Counter(),
    }
    for window_id in sorted(predictions_by_window):
        portfolio, window_diag = simulate_optimized_portfolio(
            predictions=predictions_by_window[window_id],
            prices=prices,
            cost_model=cost_model,
            scheme_name=scheme_name,
            scheme_config=scheme_config,
            sector_map=sector_map,
            benchmark_ticker=benchmark_ticker,
        )
        window_results.append(build_window_payload(
            window_id=window_id,
            metadata=window_metadata[window_id],
            portfolio=portfolio,
            extra_payload=window_diag["window_summary"],
        ))
        diagnostics["shrinkage_records"].extend(window_diag["shrinkage_records"])
        diagnostics["solver_status_counts"].update(window_diag["solver_status_counts"])
        diagnostics["risk_rule_triggers"].update(window_diag["risk_rule_triggers"])

    aggregate = aggregate_scheme_results(window_results)
    return {
        "config": scheme_config,
        "aggregate": aggregate,
        "windows": window_results,
        "series": build_period_series(window_results),
        "diagnostics": {
            "shrinkage_records": [record.to_dict() for record in diagnostics["shrinkage_records"]],
            "solver_status_counts": dict(diagnostics["solver_status_counts"]),
            "risk_rule_triggers": dict(diagnostics["risk_rule_triggers"]),
        },
        "raw_diagnostics": {
            "shrinkage_records": diagnostics["shrinkage_records"],
        },
    }


def simulate_optimized_portfolio(
    *,
    predictions: pd.Series,
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    scheme_name: str,
    scheme_config: dict[str, Any],
    sector_map: dict[str, str],
    benchmark_ticker: str,
    initial_capital: float = 1_000_000.0,
) -> tuple[PortfolioBacktestResult, dict[str, Any]]:
    execution = prepare_execution_price_frame(prices)
    returns_history = (
        execution["daily_return"]
        .unstack("ticker")
        .sort_index()
        .replace([np.inf, -np.inf], np.nan)
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
    cumulative_gross = 1.0
    cumulative_net = 1.0
    cumulative_benchmark = 1.0
    shrinkage_records: list[CovarianceResult] = []
    solver_status_counts: Counter[str] = Counter()
    risk_rule_triggers: Counter[str] = Counter()

    risk_engine = PortfolioRiskEngine()
    optimizer = CVXPYOptimizer()
    constraints = PortfolioConstraints(
        max_weight=float(scheme_config["max_weight"]),
        min_holdings=int(scheme_config["min_holdings"]),
    )

    for signal_date, next_signal_date in zip(signal_dates[:-1], signal_dates[1:]):
        execution_date = schedule.get(pd.Timestamp(signal_date))
        exit_date = schedule.get(pd.Timestamp(next_signal_date))
        if execution_date is None or exit_date is None or exit_date <= execution_date:
            continue

        score_frame = predictions.xs(signal_date, level="trade_date").dropna().astype(float).sort_values(ascending=False)
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
        if len(eligible) < max(int(scheme_config["min_holdings"]), 2):
            continue

        filtered_scores = score_frame.loc[score_frame.index.astype(str).isin(eligible)].sort_values(ascending=False)
        if filtered_scores.empty:
            continue
        ranking = filtered_scores.index.astype(str).tolist()
        candidate_tickers = select_candidate_tickers(
            ranking=ranking,
            current_weights=current_weights,
            selection_pct=float(scheme_config["selection_pct"]),
            sell_buffer_pct=float(scheme_config["sell_buffer_pct"]),
            min_holdings=int(scheme_config["min_holdings"]),
            max_weight=float(scheme_config["max_weight"]),
        )
        candidate_scores = filtered_scores.reindex(candidate_tickers).dropna()
        if candidate_scores.empty:
            continue

        trailing = returns_history.loc[:execution_date].iloc[:-1]
        if benchmark not in trailing.columns:
            continue
        trailing_candidate_returns = trailing.reindex(columns=candidate_scores.index.astype(str))
        spy_returns = trailing[benchmark].dropna()
        benchmark_weights = equal_weight_mapping(filtered_scores.index.astype(str).tolist())
        benchmark_sector_weights = compute_sector_weights(benchmark_weights, sector_map)
        dollar_liquidity = (
            entry_slice.loc[candidate_scores.index, "adv_20d_shares"]
            * entry_slice.loc[candidate_scores.index, "execution_price"]
        )

        posterior: BlackLittermanPosterior | None = None
        raw_weights: dict[str, float]
        try:
            posterior = build_black_litterman_posterior(
                scores=candidate_scores,
                trailing_returns=trailing_candidate_returns,
                dollar_liquidity=dollar_liquidity,
                lookback_days=int(scheme_config["lookback_days"]),
                tau=float(scheme_config["tau"]),
                risk_aversion=float(scheme_config["risk_aversion"]),
                score_scale=float(scheme_config["score_scale"]),
            )
            shrinkage_records.append(posterior.covariance_result)
        except ValueError:
            posterior = None

        if scheme_name == "black_litterman_upgraded":
            raw_weights = black_litterman_portfolio(
                candidate_scores,
                trailing_returns=trailing_candidate_returns,
                dollar_liquidity=dollar_liquidity,
                n_stocks=len(candidate_scores),
                selection_pct=1.0,
                constraints=constraints,
                lookback_days=int(scheme_config["lookback_days"]),
                tau=float(scheme_config["tau"]),
                risk_aversion=float(scheme_config["risk_aversion"]),
                score_scale=float(scheme_config["score_scale"]),
                covariance_method="ledoit_wolf",
                use_cvxpy=False,
            )
            solver_status_counts["scipy_post_mean_variance"] += 1
        else:
            if posterior is None:
                raw_weights = equal_weight_portfolio(
                    candidate_scores,
                    n_stocks=len(candidate_scores),
                    selection_pct=1.0,
                    constraints=constraints,
                )
                solver_status_counts["posterior_fallback_equal_weight"] += 1
            else:
                stock_betas = compute_stock_betas(
                    return_history=trailing_candidate_returns.reindex(columns=posterior.tickers),
                    spy_returns=spy_returns,
                )
                previous_vector = vectorize_weights(current_weights, posterior.tickers)
                optimization = optimizer.optimize(
                    expected_returns=posterior.posterior_returns,
                    covariance=posterior.covariance_result.matrix,
                    tickers=posterior.tickers,
                    prev_weights=previous_vector,
                    sector_map={ticker: sector_map.get(ticker, "Unknown") for ticker in posterior.tickers},
                    benchmark_sector_weights=benchmark_sector_weights,
                    stock_betas=stock_betas,
                    adv=dollar_liquidity.reindex(posterior.tickers).fillna(0.0).to_numpy(dtype=float),
                    portfolio_size=float(scheme_config["portfolio_size"]),
                    max_weight=float(scheme_config["max_weight"]),
                    max_sector_deviation=float(scheme_config["max_sector_deviation"]),
                    beta_bounds=tuple(float(value) for value in scheme_config["beta_bounds"]),
                    min_holdings=int(scheme_config["min_holdings"]),
                    lambda_risk=float(scheme_config["lambda_risk"]),
                    lambda_turnover=float(scheme_config["lambda_turnover"]),
                )
                raw_weights = optimization.weights
                solver_status_counts[str(optimization.solver_status)] += 1

        constrained = risk_engine.apply_all_constraints(
            weights=raw_weights,
            benchmark_weights=benchmark_weights,
            sector_map=sector_map,
            return_history=trailing_candidate_returns,
            spy_returns=spy_returns,
            current_weights=current_weights,
            candidate_ranking=ranking,
            max_single_stock_weight=float(scheme_config["max_weight"]),
            max_sector_deviation=float(scheme_config["max_sector_deviation"]),
            beta_target_bounds=tuple(float(value) for value in scheme_config["beta_bounds"]),
            min_holdings=int(scheme_config["min_holdings"]),
        )
        for entry in constrained.audit_trail:
            if entry.triggered:
                risk_rule_triggers[entry.rule_name] += 1

        target_weights = {str(ticker): float(weight) for ticker, weight in constrained.weights.items() if weight > 0.0}
        if not target_weights:
            continue

        previous_weights = current_weights.copy()
        period_costs = empty_cost_breakdown()
        all_trade_tickers = set(previous_weights) | set(target_weights)
        for ticker in all_trade_tickers:
            if ticker not in entry_slice.index:
                continue
            delta_weight = target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)
            if np.isclose(delta_weight, 0.0, atol=1e-12):
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

        cumulative_gross *= 1.0 + gross_return
        cumulative_net *= 1.0 + net_return
        cumulative_benchmark *= 1.0 + benchmark_return
        portfolio_value *= 1.0 + net_return

        drift_denominator = 1.0 + gross_return
        if np.isclose(drift_denominator, 0.0, atol=1e-12):
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
                selected_tickers=list(target_weights),
            ),
        )
        for key, value in period_costs.items():
            cost_totals[key] += float(value)

    portfolio = build_backtest_result(periods=periods, cost_totals=cost_totals)
    diagnostics = {
        "shrinkage_records": shrinkage_records,
        "solver_status_counts": solver_status_counts,
        "risk_rule_triggers": risk_rule_triggers,
        "window_summary": {
            "solver_status_counts": dict(solver_status_counts),
            "risk_rule_triggers": dict(risk_rule_triggers),
        },
    }
    return portfolio, diagnostics


def build_window_payload(
    *,
    window_id: str,
    metadata: dict[str, Any],
    portfolio: PortfolioBacktestResult,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "window_id": window_id,
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
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload


def aggregate_scheme_results(window_results: list[dict[str, Any]]) -> dict[str, Any]:
    periods: list[dict[str, Any]] = []
    cost_breakdown = empty_cost_breakdown()
    for row in window_results:
        portfolio = row["portfolio"]
        for period in portfolio["periods"]:
            periods.append({"window_id": row["window_id"], **period})
        for key in cost_breakdown:
            cost_breakdown[key] += float(portfolio["cost_breakdown"].get(key, 0.0))

    periods_frame = pd.DataFrame(periods)
    if periods_frame.empty:
        annualized_gross = annualized_net = annualized_benchmark = 0.0
        sharpe = 0.0
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
        sharpe = compute_annualized_sharpe(periods_frame["net_excess_return"])

    return {
        "mean_ic": float(np.mean([row["test_ic"] for row in window_results])) if window_results else 0.0,
        "mean_rank_ic": float(np.mean([row["test_rank_ic"] for row in window_results])) if window_results else 0.0,
        "mean_icir": float(np.mean([row["test_icir"] for row in window_results])) if window_results else 0.0,
        "mean_hit_rate": float(np.mean([row["test_hit_rate"] for row in window_results])) if window_results else 0.0,
        "annualized_gross_excess": float(annualized_gross - annualized_benchmark),
        "annualized_net_excess": float(annualized_net - annualized_benchmark),
        "annualized_gross_return": float(annualized_gross),
        "annualized_net_return": float(annualized_net),
        "annualized_benchmark_return": float(annualized_benchmark),
        "total_cost_drag": float(annualized_gross - annualized_net),
        "average_turnover": float(np.mean([row["turnover"] for row in window_results])) if window_results else 0.0,
        "cost_breakdown": {key: float(value) for key, value in cost_breakdown.items()},
        "window_count": int(len(window_results)),
        "sharpe": float(sharpe),
    }


def build_period_series(window_results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    net_excess_records: list[dict[str, Any]] = []
    for row in window_results:
        for period in row["portfolio"]["periods"]:
            net_excess_records.append(
                {
                    "trade_date": period["execution_date"],
                    "value": float(period["net_excess_return"]),
                    "window_id": row["window_id"],
                },
            )
    return {"net_excess_return": net_excess_records}


def build_spa_payload(scheme_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    benchmark_series = period_records_to_series(scheme_reports["equal_weight_buffered"]["series"]["net_excess_return"])
    bl_series = period_records_to_series(scheme_reports["black_litterman_upgraded"]["series"]["net_excess_return"])
    cvx_series = period_records_to_series(scheme_reports["cvxpy_optimized"]["series"]["net_excess_return"])
    return {
        "bl_vs_ew": run_spa_fallback(
            benchmark_series=benchmark_series,
            competitors={"black_litterman_upgraded": bl_series},
            benchmark_name="equal_weight_buffered",
        ).to_dict(),
        "cvxpy_vs_ew": run_spa_fallback(
            benchmark_series=benchmark_series,
            competitors={"cvxpy_optimized": cvx_series},
            benchmark_name="equal_weight_buffered",
        ).to_dict(),
    }


def period_records_to_series(records: list[dict[str, Any]]) -> pd.Series:
    if not records:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(records)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    series = pd.Series(frame["value"].astype(float).to_numpy(dtype=float), index=frame["trade_date"])
    return series.sort_index()


def summarize_shrinkage(records: list[CovarianceResult]) -> dict[str, Any]:
    if not records:
        return {
            "condition_number_before": None,
            "condition_number_after": None,
            "shrinkage_coefficient": None,
            "n_samples": 0,
        }

    return {
        "condition_number_before": float(np.mean([record.condition_number_before for record in records])),
        "condition_number_after": float(np.mean([record.condition_number_after for record in records])),
        "shrinkage_coefficient": float(np.mean([record.shrinkage_coefficient for record in records])),
        "n_samples": int(len(records)),
    }


def build_recommendation(
    *,
    scheme_reports: dict[str, dict[str, Any]],
    spa_payload: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(
        (
            (name, float(payload["aggregate"]["annualized_net_excess"]))
            for name, payload in scheme_reports.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    best_scheme, best_net_excess = ordered[0]
    bl_significant = bool(spa_payload["bl_vs_ew"]["significant"])
    cvx_significant = bool(spa_payload["cvxpy_vs_ew"]["significant"])

    if best_scheme == "cvxpy_optimized" and cvx_significant:
        decision = "adopt_cvxpy_optimized"
        rationale = "CVXPY delivered the highest net excess and cleared the SPA significance check versus equal weight."
    elif best_scheme == "black_litterman_upgraded" and bl_significant:
        decision = "adopt_black_litterman_upgraded"
        rationale = "Upgraded Black-Litterman delivered the highest net excess and cleared the SPA significance check."
    elif best_scheme in {"cvxpy_optimized", "black_litterman_upgraded"}:
        decision = "keep_equal_weight_buffered"
        rationale = "The optimized scheme improved the headline metric, but the SPA fallback did not establish significance versus the buffered equal-weight baseline."
    else:
        decision = "keep_equal_weight_buffered"
        rationale = "Equal-weight buffered remains the strongest post-cost scheme on this comparison."

    return {
        "best_scheme": best_scheme,
        "best_net_excess": float(best_net_excess),
        "decision": decision,
        "rationale": rationale,
    }


def calibrate_cost_model(
    *,
    predictions_by_window: dict[str, pd.Series],
    prices: pd.DataFrame,
    scheme_config: dict[str, Any],
    benchmark_ticker: str,
) -> dict[str, Any]:
    model = AlmgrenChrissCostModel()
    samples = collect_impact_samples(
        predictions_by_window=predictions_by_window,
        prices=prices,
        cost_model=model,
        scheme_config=scheme_config,
        benchmark_ticker=benchmark_ticker,
    )
    if not samples:
        return {
            "original": model.get_params() | {"eta": model.eta, "gamma": model.gamma},
            "calibrated": {"eta": model.eta, "gamma": model.gamma},
            "calibration_ratio": 1.0,
            "sample_count": 0,
        }

    ratios = np.array([sample["ratio"] for sample in samples if np.isfinite(sample["ratio"])], dtype=float)
    if ratios.size == 0:
        ratio = 1.0
    else:
        ratio = float(np.clip(np.median(ratios), 0.25, 3.0))
    if abs(ratio - 1.0) > 0.5:
        calibrated_eta = float(model.eta * ratio)
        calibrated_gamma = float(model.gamma * ratio)
    else:
        calibrated_eta = float(model.eta)
        calibrated_gamma = float(model.gamma)

    return {
        "original": {"eta": float(model.eta), "gamma": float(model.gamma)},
        "calibrated": {"eta": calibrated_eta, "gamma": calibrated_gamma},
        "calibration_ratio": ratio,
        "sample_count": int(len(samples)),
    }


def collect_impact_samples(
    *,
    predictions_by_window: dict[str, pd.Series],
    prices: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    scheme_config: dict[str, Any],
    benchmark_ticker: str,
) -> list[dict[str, float]]:
    execution = prepare_execution_price_frame(prices)
    signal_dates = pd.DatetimeIndex(
        pd.to_datetime(
            pd.Index(
                [date for series in predictions_by_window.values() for date, _ in series.index],
            ),
        ),
    ).sort_values().unique()
    trade_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    schedule = build_execution_schedule(signal_dates, trade_dates)
    current_weights: dict[str, float] = {}
    portfolio_value = 1_000_000.0
    benchmark = benchmark_ticker.upper()
    samples: list[dict[str, float]] = []

    for window_id in sorted(predictions_by_window):
        predictions = predictions_by_window[window_id]
        local_dates = pd.DatetimeIndex(pd.to_datetime(predictions.index.get_level_values("trade_date"))).sort_values().unique()
        constraints = PortfolioConstraints(
            max_weight=float(scheme_config["max_weight"]),
            min_holdings=int(scheme_config["min_holdings"]),
        )

        for signal_date, next_signal_date in zip(local_dates[:-1], local_dates[1:]):
            execution_date = schedule.get(pd.Timestamp(signal_date))
            exit_date = schedule.get(pd.Timestamp(next_signal_date))
            if execution_date is None or exit_date is None or exit_date <= execution_date:
                continue

            score_frame = predictions.xs(signal_date, level="trade_date").dropna().astype(float).sort_values(ascending=False)
            entry_slice = execution.xs(execution_date, level="trade_date")
            exit_slice = execution.xs(exit_date, level="trade_date")
            eligible = (
                set(score_frame.index.astype(str))
                & set(entry_slice.index.astype(str))
                & set(exit_slice.index.astype(str))
            )
            eligible.discard(benchmark)
            filtered_scores = score_frame.loc[score_frame.index.astype(str).isin(eligible)].sort_values(ascending=False)
            if filtered_scores.empty:
                continue

            ranking = filtered_scores.index.astype(str).tolist()
            candidate_tickers = select_candidate_tickers(
                ranking=ranking,
                current_weights=current_weights,
                selection_pct=float(scheme_config["selection_pct"]),
                sell_buffer_pct=float(scheme_config["sell_buffer_pct"]),
                min_holdings=int(scheme_config["min_holdings"]),
                max_weight=float(scheme_config["max_weight"]),
            )
            candidate_scores = filtered_scores.reindex(candidate_tickers).dropna()
            if candidate_scores.empty:
                continue

            target_weights = equal_weight_portfolio(
                candidate_scores,
                n_stocks=len(candidate_scores),
                selection_pct=1.0,
                constraints=constraints,
            )
            target_weights = apply_turnover_buffer(
                target_weights,
                current_weights={ticker: weight for ticker, weight in current_weights.items() if ticker in ranking},
                min_trade_weight=float(scheme_config["min_trade_weight"]),
                ranking=ranking,
                constraints=constraints,
            )
            if not target_weights:
                continue

            previous_weights = current_weights.copy()
            for ticker in set(previous_weights) | set(target_weights):
                if ticker not in entry_slice.index:
                    continue
                delta_weight = target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)
                if np.isclose(delta_weight, 0.0, atol=1e-12):
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
                predicted_rate = float(estimate.temporary_impact_rate + estimate.permanent_impact_rate)
                realized_rate = abs(float(bar["execution_price"] - bar["open_px"])) / max(float(bar["open_px"]), 1e-12)
                if predicted_rate <= 1e-8:
                    continue
                samples.append(
                    {
                        "predicted_rate": predicted_rate,
                        "realized_rate": realized_rate,
                        "ratio": realized_rate / predicted_rate,
                    },
                )

            gross_return = 0.0
            asset_returns: dict[str, float] = {}
            for ticker, weight in target_weights.items():
                if ticker not in entry_slice.index or ticker not in exit_slice.index:
                    continue
                realized = (float(exit_slice.loc[ticker, "execution_price"]) / max(float(entry_slice.loc[ticker, "execution_price"]), 1e-12)) - 1.0
                asset_returns[ticker] = realized
                gross_return += weight * realized
            drift_denominator = 1.0 + gross_return
            if np.isclose(drift_denominator, 0.0, atol=1e-12):
                current_weights = {}
            else:
                current_weights = {
                    ticker: float((weight * (1.0 + asset_returns.get(ticker, 0.0))) / drift_denominator)
                    for ticker, weight in target_weights.items()
                    if (weight * (1.0 + asset_returns.get(ticker, 0.0))) > 1e-8
                }

    return samples


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


def compute_annualized_sharpe(returns: pd.Series | Sequence[float]) -> float:
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2 or np.isclose(float(series.std(ddof=1)), 0.0, atol=1e-12):
        return 0.0
    return float(np.sqrt(52.0) * series.mean() / series.std(ddof=1))


def empty_cost_breakdown() -> dict[str, float]:
    return {
        "commission": 0.0,
        "spread": 0.0,
        "temporary_impact": 0.0,
        "permanent_impact": 0.0,
        "gap_penalty": 0.0,
        "total": 0.0,
    }


def equal_weight_mapping(tickers: list[str]) -> dict[str, float]:
    if not tickers:
        return {}
    weight = 1.0 / len(tickers)
    return {str(ticker).upper(): float(weight) for ticker in tickers}


def vectorize_weights(weights: dict[str, float], tickers: list[str]) -> np.ndarray:
    return (
        pd.Series(weights, dtype=float)
        .rename(index=str)
        .rename_axis("ticker")
        .pipe(lambda series: series.set_axis(series.index.astype(str).str.upper()))
        .reindex(tickers)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )


def compute_stock_betas(
    *,
    return_history: pd.DataFrame,
    spy_returns: pd.Series,
) -> np.ndarray | None:
    if return_history.empty or spy_returns.empty:
        return None
    benchmark = pd.Series(spy_returns, dtype=float).rename("SPY").dropna()
    joined = return_history.join(benchmark, how="inner")
    benchmark_var = float(joined["SPY"].var())
    if benchmark_var <= 1e-12:
        return None
    betas: list[float] = []
    for ticker in return_history.columns.astype(str):
        aligned = joined[[ticker, "SPY"]].dropna()
        if len(aligned) < 20:
            betas.append(1.0)
            continue
        covariance = float(aligned[ticker].cov(aligned["SPY"]))
        betas.append(covariance / benchmark_var)
    return np.asarray(betas, dtype=float)


if __name__ == "__main__":
    raise SystemExit(main())
