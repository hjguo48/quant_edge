"""Phase B (S1.8–S1.10): Turnover control parameter sweep.

Sweeps hysteresis, weight-shrinkage, no-trade zone, and turnover-penalty
parameters on the score_weighted scheme using cached walk-forward predictions.
Outputs a JSON report with Pareto frontier analysis.

Usage:
    python scripts/run_turnover_optimization.py
    python scripts/run_turnover_optimization.py --report-path data/reports/turnover_optimization.json
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

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
from src.backtest.cost_model import AlmgrenChrissCostModel
from src.backtest.execution import (
    PortfolioBacktestResult,
    PortfolioPeriodResult,
    build_execution_schedule,
    prepare_execution_price_frame,
    select_candidate_tickers,
)
from src.portfolio.constraints import (
    PortfolioConstraints,
    apply_turnover_buffer,
    apply_weight_constraints,
    normalize_weights,
    cap_weights,
)

DEFAULT_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_REPORT_PATH = "data/reports/turnover_optimization.json"
BENCHMARK_TICKER = "SPY"

# ---------------------------------------------------------------------------
# S1.8: Hysteresis sweep grid
# ---------------------------------------------------------------------------
HYSTERESIS_GRID = {
    "sell_buffer_pct": [0.20, 0.25, 0.30, 0.35, 0.40],
    "min_trade_weight": [0.005, 0.01, 0.015, 0.02, 0.03],
    "selection_pct": [0.15, 0.20, 0.25],
}

# ---------------------------------------------------------------------------
# S1.9: Shrinkage + No-trade zone grid
# ---------------------------------------------------------------------------
SHRINKAGE_GRID = {
    "weight_shrinkage": [0.0, 0.1, 0.3, 0.5],
    "no_trade_zone": [0.0, 0.001, 0.002, 0.005],
}

# ---------------------------------------------------------------------------
# S1.10: Turnover penalty grid (applied as score dampening)
# ---------------------------------------------------------------------------
TURNOVER_PENALTY_GRID = {
    "turnover_penalty_lambda": [0.0, 0.5, 1.0, 2.0],
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    logger.info("Phase B: Turnover optimization sweep")

    predictions_df = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices_df = pd.read_parquet(REPO_ROOT / args.prices_path)
    logger.info("loaded predictions={} prices={}", len(predictions_df), len(prices_df))

    predictions_by_window = build_prediction_series_by_window(predictions_df)
    cost_model = AlmgrenChrissCostModel()

    # Pre-compute execution price frame ONCE (big speedup)
    logger.info("pre-computing execution price frame...")
    execution = prepare_execution_price_frame(prices_df)
    logger.info("execution frame ready: {} rows", len(execution))

    # --- Phase 1: Coarse sweep (S1.8 hysteresis only, shrinkage=0, penalty=0)
    logger.info("=== Phase 1: Coarse hysteresis sweep ===")
    coarse_results = run_hysteresis_sweep(
        predictions_by_window=predictions_by_window,
        execution=execution,
        cost_model=cost_model,
        benchmark_ticker=BENCHMARK_TICKER,
    )
    logger.info("coarse sweep: {} configurations tested", len(coarse_results))

    # Filter to configs with turnover <= 15% (pre-filter for fine sweep)
    feasible_coarse = [r for r in coarse_results if r["average_turnover"] <= 0.15]
    if not feasible_coarse:
        feasible_coarse = sorted(coarse_results, key=lambda r: r["average_turnover"])[:10]
    logger.info("feasible coarse configs: {}", len(feasible_coarse))

    # Pick top-5 by net excess among feasible
    top_hysteresis = sorted(feasible_coarse, key=lambda r: r["annualized_net_excess"], reverse=True)[:5]

    # --- Phase 2: Fine sweep (S1.9 shrinkage + no-trade zone on top hysteresis configs)
    logger.info("=== Phase 2: Shrinkage + no-trade zone sweep ===")
    fine_results = run_shrinkage_sweep(
        base_configs=top_hysteresis,
        predictions_by_window=predictions_by_window,
        execution=execution,
        cost_model=cost_model,
        benchmark_ticker=BENCHMARK_TICKER,
    )
    logger.info("fine sweep: {} configurations tested", len(fine_results))

    # --- Phase 3: Turnover penalty sweep (S1.10) on top fine configs
    feasible_fine = [r for r in fine_results if r["average_turnover"] <= 0.12]
    if not feasible_fine:
        feasible_fine = sorted(fine_results, key=lambda r: r["average_turnover"])[:5]
    top_fine = sorted(feasible_fine, key=lambda r: r["annualized_net_excess"], reverse=True)[:3]

    logger.info("=== Phase 3: Turnover penalty sweep ===")
    penalty_results = run_penalty_sweep(
        base_configs=top_fine,
        predictions_by_window=predictions_by_window,
        execution=execution,
        cost_model=cost_model,
        benchmark_ticker=BENCHMARK_TICKER,
    )
    logger.info("penalty sweep: {} configurations tested", len(penalty_results))

    # --- Also run equal_weight baseline variants for comparison
    logger.info("=== Equal-weight baselines ===")
    ew_results = run_equal_weight_baselines(
        predictions_by_window=predictions_by_window,
        execution=execution,
        prices_df=prices_df,
        cost_model=cost_model,
        benchmark_ticker=BENCHMARK_TICKER,
    )

    # --- Combine all results and build Pareto analysis
    all_results = coarse_results + fine_results + penalty_results + ew_results
    pareto_front = compute_pareto_front(all_results)

    # Find best config meeting Phase B gate
    gate_passing = [
        r for r in all_results
        if r["average_turnover"] <= 0.10
        and r["annualized_net_excess"] >= 0.035
        and r["total_cost_drag"] <= 0.025
    ]
    gate_passing.sort(key=lambda r: r["annualized_net_excess"], reverse=True)

    best_config = gate_passing[0] if gate_passing else None
    gate_pass = best_config is not None

    # Relax: best config with turnover <= 10%
    low_turnover = [r for r in all_results if r["average_turnover"] <= 0.10]
    low_turnover.sort(key=lambda r: r["annualized_net_excess"], reverse=True)
    best_low_turnover = low_turnover[0] if low_turnover else None

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_turnover_optimization.py",
        "phase_b_gate": {
            "pass": gate_pass,
            "criteria": {
                "net_excess_threshold": 0.035,
                "turnover_threshold": 0.10,
                "cost_drag_threshold": 0.025,
            },
            "best_passing_config": json_safe(best_config) if best_config else None,
        },
        "best_low_turnover_config": json_safe(best_low_turnover) if best_low_turnover else None,
        "sweep_summary": {
            "coarse_count": len(coarse_results),
            "fine_count": len(fine_results),
            "penalty_count": len(penalty_results),
            "ew_count": len(ew_results),
            "total_configs": len(all_results),
            "pareto_count": len(pareto_front),
        },
        "pareto_front": [json_safe(r) for r in pareto_front],
        "top_10_by_net_excess": [
            json_safe(r) for r in sorted(all_results, key=lambda r: r["annualized_net_excess"], reverse=True)[:10]
        ],
        "top_10_by_sharpe_proxy": [
            json_safe(r) for r in sorted(all_results, key=lambda r: r.get("sharpe_proxy", -999), reverse=True)[:10]
        ],
        "equal_weight_baselines": [json_safe(r) for r in ew_results],
    }

    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved turnover optimization report to {}", report_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Phase B Gate: {}", "PASS" if gate_pass else "FAIL")
    if best_config:
        logger.info(
            "Best passing: scheme={} net_excess={:.4f} turnover={:.4f} cost_drag={:.4f}",
            best_config["scheme_name"],
            best_config["annualized_net_excess"],
            best_config["average_turnover"],
            best_config["total_cost_drag"],
        )
    if best_low_turnover:
        logger.info(
            "Best low-turnover: scheme={} net_excess={:.4f} turnover={:.4f} cost_drag={:.4f}",
            best_low_turnover["scheme_name"],
            best_low_turnover["annualized_net_excess"],
            best_low_turnover["average_turnover"],
            best_low_turnover["total_cost_drag"],
        )
    for r in sorted(all_results, key=lambda r: r["annualized_net_excess"], reverse=True)[:5]:
        logger.info(
            "  {} | net_excess={:.4f} turnover={:.4f} cost={:.4f} gross={:.4f}",
            r["scheme_name"],
            r["annualized_net_excess"],
            r["average_turnover"],
            r["total_cost_drag"],
            r["annualized_gross_excess"],
        )

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B turnover optimization sweep")
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    return parser.parse_args(argv)


# ===================================================================
# Data loading
# ===================================================================

def build_prediction_series_by_window(df: pd.DataFrame) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for window_id, frame in df.groupby("window_id", sort=True):
        series = frame.set_index(["trade_date", "ticker"])["score"].sort_index()
        result[str(window_id)] = series
    return result


# ===================================================================
# S1.8: Hysteresis sweep
# ===================================================================

def run_hysteresis_sweep(
    *,
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    grid = list(itertools.product(
        HYSTERESIS_GRID["sell_buffer_pct"],
        HYSTERESIS_GRID["min_trade_weight"],
        HYSTERESIS_GRID["selection_pct"],
    ))
    logger.info("hysteresis grid: {} combinations", len(grid))

    for idx, (sell_buf, min_trade, sel_pct) in enumerate(grid):
        scheme_name = f"sw_hyst_sb{sell_buf}_mt{min_trade}_sp{sel_pct}"
        config = {
            "selection_pct": sel_pct,
            "sell_buffer_pct": sell_buf,
            "min_trade_weight": min_trade,
            "max_weight": 0.05,
            "min_holdings": 20,
            "weight_shrinkage": 0.0,
            "no_trade_zone": 0.0,
            "turnover_penalty_lambda": 0.0,
        }
        result = run_score_weighted_config(
            scheme_name=scheme_name,
            config=config,
            predictions_by_window=predictions_by_window,
            execution=execution,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
        )
        results.append(result)
        if (idx + 1) % 15 == 0:
            logger.info("  hysteresis {}/{} done", idx + 1, len(grid))

    return results


# ===================================================================
# S1.9: Shrinkage + no-trade zone sweep
# ===================================================================

def run_shrinkage_sweep(
    *,
    base_configs: list[dict[str, Any]],
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    shrinkage_grid = list(itertools.product(
        SHRINKAGE_GRID["weight_shrinkage"],
        SHRINKAGE_GRID["no_trade_zone"],
    ))
    # Skip (0.0, 0.0) as it duplicates the base config
    shrinkage_grid = [(s, n) for s, n in shrinkage_grid if not (s == 0.0 and n == 0.0)]

    total = len(base_configs) * len(shrinkage_grid)
    logger.info("shrinkage grid: {} base × {} shrinkage = {} combos", len(base_configs), len(shrinkage_grid), total)

    count = 0
    for base in base_configs:
        base_cfg = base["config"]
        for shrinkage, no_trade in shrinkage_grid:
            scheme_name = (
                f"sw_shrink_sb{base_cfg['sell_buffer_pct']}_mt{base_cfg['min_trade_weight']}"
                f"_sp{base_cfg['selection_pct']}_sh{shrinkage}_nt{no_trade}"
            )
            config = {**base_cfg, "weight_shrinkage": shrinkage, "no_trade_zone": no_trade}
            result = run_score_weighted_config(
                scheme_name=scheme_name,
                config=config,
                predictions_by_window=predictions_by_window,
                execution=execution,
                cost_model=cost_model,
                benchmark_ticker=benchmark_ticker,
            )
            results.append(result)
            count += 1
            if count % 20 == 0:
                logger.info("  shrinkage {}/{} done", count, total)

    return results


# ===================================================================
# S1.10: Turnover penalty sweep
# ===================================================================

def run_penalty_sweep(
    *,
    base_configs: list[dict[str, Any]],
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    penalties = [p for p in TURNOVER_PENALTY_GRID["turnover_penalty_lambda"] if p > 0.0]

    total = len(base_configs) * len(penalties)
    logger.info("penalty grid: {} base × {} penalties = {} combos", len(base_configs), len(penalties), total)

    for base in base_configs:
        base_cfg = base["config"]
        for penalty in penalties:
            scheme_name = (
                f"sw_penalty_sb{base_cfg['sell_buffer_pct']}_mt{base_cfg['min_trade_weight']}"
                f"_sp{base_cfg['selection_pct']}_sh{base_cfg.get('weight_shrinkage', 0)}"
                f"_nt{base_cfg.get('no_trade_zone', 0)}_lam{penalty}"
            )
            config = {**base_cfg, "turnover_penalty_lambda": penalty}
            result = run_score_weighted_config(
                scheme_name=scheme_name,
                config=config,
                predictions_by_window=predictions_by_window,
                execution=execution,
                cost_model=cost_model,
                benchmark_ticker=benchmark_ticker,
            )
            results.append(result)

    return results


# ===================================================================
# Equal-weight baselines
# ===================================================================

def run_equal_weight_baselines(
    *,
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    prices_df: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
) -> list[dict[str, Any]]:
    """Run a few equal-weight configs for comparison."""
    from src.backtest.execution import simulate_portfolio

    configs = [
        ("ew_baseline", {"weighting_scheme": "equal_weight", "selection_pct": 0.20, "sell_buffer_pct": 0.25, "min_trade_weight": 0.01, "max_weight": 0.05, "min_holdings": 20}),
        ("ew_tight", {"weighting_scheme": "equal_weight", "selection_pct": 0.15, "sell_buffer_pct": 0.30, "min_trade_weight": 0.02, "max_weight": 0.05, "min_holdings": 20}),
        ("ew_loose", {"weighting_scheme": "equal_weight", "selection_pct": 0.25, "sell_buffer_pct": 0.35, "min_trade_weight": 0.015, "max_weight": 0.05, "min_holdings": 20}),
    ]
    results = []
    for scheme_name, cfg in configs:
        window_results = []
        for window_id in sorted(predictions_by_window):
            portfolio = simulate_portfolio(
                predictions=predictions_by_window[window_id],
                prices=prices_df,
                cost_model=cost_model,
                weighting_scheme=cfg["weighting_scheme"],
                benchmark_ticker=benchmark_ticker,
                universe_by_date=None,
                selection_pct=float(cfg["selection_pct"]),
                sell_buffer_pct=cfg.get("sell_buffer_pct"),
                min_trade_weight=float(cfg["min_trade_weight"]),
                max_weight=float(cfg["max_weight"]),
                min_holdings=int(cfg["min_holdings"]),
            )
            window_results.append(portfolio)

        agg = aggregate_window_portfolios(window_results)
        results.append({
            "scheme_name": scheme_name,
            "config": cfg,
            "weighting": "equal_weight",
            **agg,
        })
    return results


# ===================================================================
# Core simulation: score-weighted with all turnover controls
# ===================================================================

def run_score_weighted_config(
    *,
    scheme_name: str,
    config: dict[str, Any],
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
) -> dict[str, Any]:
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

    agg = aggregate_window_portfolios(window_portfolios)
    return {
        "scheme_name": scheme_name,
        "config": config,
        "weighting": "score_weighted",
        **agg,
    }


def simulate_score_weighted_controlled(
    *,
    predictions: pd.Series,
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str = "SPY",
    initial_capital: float = 1_000_000.0,
    selection_pct: float = 0.20,
    sell_buffer_pct: float | None = 0.25,
    min_trade_weight: float = 0.01,
    max_weight: float = 0.05,
    min_holdings: int = 20,
    weight_shrinkage: float = 0.0,
    no_trade_zone: float = 0.0,
    turnover_penalty_lambda: float = 0.0,
    directional_throttle_signal: pd.Series | None = None,
    adverse_buy_threshold: float = 1.0,
    adverse_sell_threshold: float = 1.0,
    adverse_trade_scale: float = 0.5,
) -> PortfolioBacktestResult:
    """Score-weighted portfolio with full turnover controls.

    Args:
        execution: Pre-computed execution price frame from prepare_execution_price_frame().
        directional_throttle_signal: Optional Series indexed by (trade_date, ticker)
            carrying a directional signal z (e.g., 1D z-score). When |z| exceeds the
            corresponding threshold and direction is adverse to the intended trade,
            scale that ticker's Δw by adverse_trade_scale.
        adverse_buy_threshold: |z| threshold for adverse-buy detection (z < -threshold
            on a buy = scale down the buy).
        adverse_sell_threshold: |z| threshold for adverse-sell detection (z > +threshold
            on a sell = scale down the sell).
        adverse_trade_scale: Scale factor (0..1) applied to adverse Δw.

    Turnover controls (applied in order):
    1. Hysteresis: sell_buffer_pct widens exit band, min_trade_weight skips small trades
    2. Weight shrinkage: w_final = (1 - shrinkage) * w_target + shrinkage * w_prev
    3. No-trade zone: positions where |Δw| < no_trade_zone keep previous weight
    4. Turnover penalty: dampens score changes by penalizing distance from current weights
    5. Directional throttle: scale Δw against adverse short-horizon signal (W11 1D throttle)
    """
    if predictions.empty:
        return _empty_backtest_result()

    benchmark = benchmark_ticker.upper()
    signal_dates = pd.DatetimeIndex(
        pd.to_datetime(predictions.index.get_level_values("trade_date")),
    ).sort_values().unique()
    trade_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    schedule = build_execution_schedule(signal_dates, trade_dates)

    constraints = PortfolioConstraints(
        max_weight=max_weight,
        min_holdings=min_holdings,
        turnover_buffer=min_trade_weight,
    )

    current_weights: dict[str, float] = {}
    portfolio_value = float(initial_capital)
    periods: list[PortfolioPeriodResult] = []
    cost_totals = _empty_cost_breakdown()

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

        ranking = filtered_scores.index.astype(str).tolist()

        # --- S1.10: Turnover penalty — dampen scores toward current holdings
        if turnover_penalty_lambda > 0.0 and current_weights:
            adjusted_scores = filtered_scores.copy()
            for ticker in adjusted_scores.index:
                ticker_str = str(ticker)
                if ticker_str in current_weights:
                    # Boost score of currently held positions
                    adjusted_scores.loc[ticker] += turnover_penalty_lambda * current_weights[ticker_str]
            adjusted_scores = adjusted_scores.sort_values(ascending=False)
            ranking = adjusted_scores.index.astype(str).tolist()
            filtered_scores = adjusted_scores

        # Candidate selection with hysteresis
        candidate_tickers = select_candidate_tickers(
            ranking=ranking,
            current_weights=current_weights,
            selection_pct=selection_pct,
            sell_buffer_pct=sell_buffer_pct,
            min_holdings=min_holdings,
            max_weight=max_weight,
        )
        candidate_scores = filtered_scores.reindex(candidate_tickers).dropna()
        if candidate_scores.empty:
            continue

        # Build raw score-weighted targets
        target_weights = _build_score_weights(candidate_scores, constraints)
        if not target_weights:
            continue

        # --- S1.9a: Weight shrinkage — blend toward previous weights
        if weight_shrinkage > 0.0 and current_weights:
            target_weights = _apply_weight_shrinkage(
                target_weights=target_weights,
                previous_weights=current_weights,
                shrinkage=weight_shrinkage,
                constraints=constraints,
                ranking=ranking,
            )

        # --- S1.9b: No-trade zone — keep previous weight if change is tiny
        if no_trade_zone > 0.0 and current_weights:
            target_weights = _apply_no_trade_zone(
                target_weights=target_weights,
                previous_weights=current_weights,
                threshold=no_trade_zone,
                constraints=constraints,
                ranking=ranking,
            )

        # --- S1.8: Turnover buffer (min_trade_weight)
        if min_trade_weight > 0.0:
            buffer_reference_weights = {
                ticker: weight
                for ticker, weight in current_weights.items()
                if ticker in set(ranking)
            }
            target_weights = apply_turnover_buffer(
                target_weights,
                current_weights=buffer_reference_weights,
                min_trade_weight=min_trade_weight,
                ranking=ranking,
                constraints=constraints,
            )
        else:
            target_weights = apply_weight_constraints(
                target_weights,
                ranking=ranking,
                constraints=constraints,
            )

        if not target_weights:
            continue

        # --- W11 1D throttle: scale Δw against adverse short-horizon signal
        # NOTE: this scales target_weights without renormalizing — adverse trades
        # leave cash on the table (implicit cash position). It is NOT pure
        # execution timing; it reduces total invested exposure on adverse 1D days.
        # Confirmed semantically intentional per W11 design (Codex review).
        if directional_throttle_signal is not None:
            try:
                throttle_today = directional_throttle_signal.xs(signal_date, level="trade_date")
            except KeyError:
                throttle_today = None
            if throttle_today is not None and not throttle_today.empty:
                previous_for_throttle = current_weights
                throttled_targets = dict(target_weights)
                for ticker in set(target_weights) | set(previous_for_throttle):
                    delta = target_weights.get(ticker, 0.0) - previous_for_throttle.get(ticker, 0.0)
                    if abs(delta) < 1e-12:
                        continue
                    z1 = float(throttle_today.get(ticker, 0.0))
                    if not math.isfinite(z1):
                        continue
                    is_buy = delta > 0.0
                    is_sell = delta < 0.0
                    adverse = (is_buy and z1 <= -adverse_buy_threshold) or \
                              (is_sell and z1 >= adverse_sell_threshold)
                    if adverse:
                        new_target = previous_for_throttle.get(ticker, 0.0) + adverse_trade_scale * delta
                        if abs(new_target) < 1e-12:
                            throttled_targets.pop(ticker, None)
                        else:
                            throttled_targets[ticker] = float(new_target)
                target_weights = throttled_targets

        # --- Execute and track
        previous_weights = current_weights.copy()
        period_costs = _empty_cost_breakdown()
        all_trade_tickers = set(previous_weights) | set(target_weights)
        for ticker in all_trade_tickers:
            delta_weight = target_weights.get(ticker, 0.0) - previous_weights.get(ticker, 0.0)
            if abs(delta_weight) < 1e-12:
                continue
            if ticker not in entry_slice.index:
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

    return _build_backtest_result(periods=periods, cost_totals=cost_totals)


# ===================================================================
# Weight construction helpers
# ===================================================================

def _build_score_weights(
    candidate_scores: pd.Series,
    constraints: PortfolioConstraints,
) -> dict[str, float]:
    pos_scores = candidate_scores[candidate_scores > 0.0]
    if pos_scores.empty:
        return {}
    raw = pos_scores / pos_scores.sum()
    raw = raw.clip(upper=float(constraints.max_weight))
    total = float(raw.sum())
    if total <= 0.0:
        return {}
    raw = raw / total
    return {str(ticker): float(weight) for ticker, weight in raw.items() if weight > 0.0}


def _apply_weight_shrinkage(
    *,
    target_weights: dict[str, float],
    previous_weights: dict[str, float],
    shrinkage: float,
    constraints: PortfolioConstraints,
    ranking: list[str],
) -> dict[str, float]:
    """Blend target weights toward previous: w = (1-λ)*target + λ*prev."""
    all_tickers = set(target_weights) | set(previous_weights)
    blended: dict[str, float] = {}
    for ticker in all_tickers:
        t = target_weights.get(ticker, 0.0)
        p = previous_weights.get(ticker, 0.0)
        w = (1.0 - shrinkage) * t + shrinkage * p
        if w > 1e-8:
            blended[ticker] = w

    # Re-normalize
    total = sum(blended.values())
    if total <= 0:
        return target_weights
    return {t: w / total for t, w in blended.items()}


def _apply_no_trade_zone(
    *,
    target_weights: dict[str, float],
    previous_weights: dict[str, float],
    threshold: float,
    constraints: PortfolioConstraints,
    ranking: list[str],
) -> dict[str, float]:
    """Keep previous weight for positions where |Δw| < threshold."""
    result: dict[str, float] = {}
    adjustable_tickers = []

    all_tickers = set(target_weights) | set(previous_weights)
    for ticker in all_tickers:
        t = target_weights.get(ticker, 0.0)
        p = previous_weights.get(ticker, 0.0)
        if abs(t - p) < threshold and p > 0:
            result[ticker] = p  # Keep previous
        else:
            if t > 0:
                adjustable_tickers.append(ticker)
                result[ticker] = t

    # Re-normalize
    total = sum(result.values())
    if total <= 0:
        return target_weights
    return {t: w / total for t, w in result.items() if w > 0}


# ===================================================================
# Aggregation
# ===================================================================

def aggregate_window_portfolios(portfolios: list[PortfolioBacktestResult]) -> dict[str, Any]:
    all_periods: list[dict[str, Any]] = []
    cost_totals = _empty_cost_breakdown()

    for portfolio in portfolios:
        for period in portfolio.periods:
            all_periods.append({
                "execution_date": period.execution_date,
                "exit_date": period.exit_date,
                "gross_return": period.gross_return,
                "net_return": period.net_return,
                "benchmark_return": period.benchmark_return,
                "turnover": period.turnover,
                "cost_rate": period.cost_rate,
            })
        for key in cost_totals:
            cost_totals[key] += portfolio.cost_breakdown.get(key, 0.0)

    if not all_periods:
        return {
            "annualized_gross_excess": 0.0,
            "annualized_net_excess": 0.0,
            "annualized_gross_return": 0.0,
            "annualized_net_return": 0.0,
            "annualized_benchmark_return": 0.0,
            "total_cost_drag": 0.0,
            "average_turnover": 0.0,
            "sharpe_proxy": 0.0,
            "period_count": 0,
        }

    df = pd.DataFrame(all_periods)
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
        "period_count": len(df),
    }


def compute_pareto_front(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find Pareto-optimal configs: maximize net_excess, minimize turnover."""
    # Sort by turnover ascending
    sorted_results = sorted(results, key=lambda r: r["average_turnover"])
    pareto: list[dict[str, Any]] = []
    best_excess = -float("inf")

    for r in sorted_results:
        if r["annualized_net_excess"] > best_excess:
            pareto.append(r)
            best_excess = r["annualized_net_excess"]

    return pareto


# ===================================================================
# Utilities
# ===================================================================

def _annualize(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


def _empty_cost_breakdown() -> dict[str, float]:
    return {
        "commission": 0.0,
        "spread": 0.0,
        "temporary_impact": 0.0,
        "permanent_impact": 0.0,
        "gap_penalty": 0.0,
        "total": 0.0,
    }


def _build_backtest_result(
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

    cumulative_gross = 1.0
    cumulative_net = 1.0
    cumulative_benchmark = 1.0
    for p in periods:
        cumulative_gross *= 1.0 + p.gross_return
        cumulative_net *= 1.0 + p.net_return
        cumulative_benchmark *= 1.0 + p.benchmark_return

    start_date = pd.Timestamp(periods[0].execution_date)
    end_date = pd.Timestamp(periods[-1].exit_date)
    total_days = max((end_date - start_date).days, 1)

    gross_total = cumulative_gross - 1.0
    net_total = cumulative_net - 1.0
    bench_total = cumulative_benchmark - 1.0
    ann_gross = _annualize(gross_total, total_days)
    ann_net = _annualize(net_total, total_days)
    ann_bench = _annualize(bench_total, total_days)

    return PortfolioBacktestResult(
        periods=periods,
        gross_return=float(gross_total),
        net_return=float(net_total),
        benchmark_return=float(bench_total),
        gross_excess_return=float(gross_total - bench_total),
        net_excess_return=float(net_total - bench_total),
        annualized_gross_return=float(ann_gross),
        annualized_net_return=float(ann_net),
        annualized_benchmark_return=float(ann_bench),
        annualized_excess_gross=float(ann_gross - ann_bench),
        annualized_excess_net=float(ann_net - ann_bench),
        total_cost_drag=float(ann_gross - ann_net),
        average_turnover=float(np.mean([p.turnover for p in periods])),
        cost_breakdown={k: float(v) for k, v in cost_totals.items()},
    )


def _empty_backtest_result() -> PortfolioBacktestResult:
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
        cost_breakdown=_empty_cost_breakdown(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
