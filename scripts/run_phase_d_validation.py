"""Phase D (S1.17–S1.21): Full walk-forward validation + statistical tests.

S1.17 — 8-window walk-forward with Phase B optimal turnover controls
S1.18 — Hyperparameter sensitivity check (verify current params are near-optimal)
S1.19 — Lightweight stacker experiment (out-of-fold stacking vs adaptive fusion)
S1.20 — DSR/SPA final test (new vs old fusion)
S1.21 — G3 Gate retest (6 checks)

Usage:
    python scripts/run_phase_d_validation.py
    python scripts/run_phase_d_validation.py --skip-stacker  # skip S1.19
"""
from __future__ import annotations

import argparse
import json
import math
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
from scipy import stats

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
from src.stats.bootstrap import bootstrap_return_statistics
from src.stats.spa import run_spa_fallback

DEFAULT_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_FUSION_REPORT_PATH = "data/reports/fusion_analysis_60d.json"
DEFAULT_COMPARISON_REPORT_PATH = "data/reports/walkforward_comparison_60d.json"
DEFAULT_TURNOVER_REPORT_PATH = "data/reports/turnover_optimization.json"
DEFAULT_REPORT_PATH = "data/reports/phase_d_validation.json"
BENCHMARK_TICKER = "SPY"

# Phase B optimal config
OPTIMAL_SW_CONFIG = {
    "sell_buffer_pct": 0.40,
    "min_trade_weight": 0.005,
    "selection_pct": 0.25,
    "weight_shrinkage": 0.50,
    "no_trade_zone": 0.005,
    "turnover_penalty_lambda": 0.0,
    "max_weight": 0.05,
    "min_holdings": 20,
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    logger.info("Phase D: Full walk-forward validation")

    predictions_df = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices_df = pd.read_parquet(REPO_ROOT / args.prices_path)
    logger.info("loaded predictions={} prices={}", len(predictions_df), len(prices_df))

    predictions_by_window = build_prediction_series_by_window(predictions_df)
    cost_model = AlmgrenChrissCostModel()

    logger.info("pre-computing execution price frame...")
    execution = prepare_execution_price_frame(prices_df)
    logger.info("execution frame ready: {} rows", len(execution))

    # Load fusion/comparison reports for IC data
    fusion_data = json.loads((REPO_ROOT / args.fusion_report).read_text())
    comparison_data = json.loads((REPO_ROOT / args.comparison_report).read_text())

    # =====================================================================
    # S1.17: Full walk-forward with Phase B optimal config
    # =====================================================================
    logger.info("=" * 60)
    logger.info("S1.17: Full walk-forward validation with Phase B optimal config")

    s17_report = run_s17_walkforward(
        predictions_by_window, execution, prices_df, cost_model,
        fusion_data, comparison_data,
    )

    # =====================================================================
    # S1.18: Hyperparameter sensitivity check
    # =====================================================================
    logger.info("=" * 60)
    logger.info("S1.18: Hyperparameter sensitivity check")

    s18_report = run_s18_hyperparam_sensitivity(
        predictions_by_window, execution, cost_model,
    )

    # =====================================================================
    # S1.19: Stacker experiment
    # =====================================================================
    if args.skip_stacker:
        logger.info("S1.19: Skipped (--skip-stacker)")
        s19_report = {"status": "skipped"}
    else:
        logger.info("=" * 60)
        logger.info("S1.19: Stacker experiment")
        s19_report = run_s19_stacker(
            comparison_data, fusion_data,
        )

    # =====================================================================
    # S1.20: DSR/SPA final test
    # =====================================================================
    logger.info("=" * 60)
    logger.info("S1.20: DSR/SPA final test")

    sw_periods = s17_report["sw_controlled"]["period_details"]
    ew_periods = s17_report["ew_baseline"]["period_details"]
    s20_report = run_s20_spa_dsr(sw_periods, ew_periods, fusion_data)

    # =====================================================================
    # S1.21: G3 Gate retest
    # =====================================================================
    logger.info("=" * 60)
    logger.info("S1.21: G3 Gate retest")

    s21_report = run_s21_g3_gate(
        s17_report, s20_report, fusion_data,
    )

    # =====================================================================
    # Final report
    # =====================================================================
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_phase_d_validation.py",
        "optimal_config": OPTIMAL_SW_CONFIG,
        "s1_17_walkforward": _strip_period_details(s17_report),
        "s1_18_hyperparam": s18_report,
        "s1_19_stacker": s19_report,
        "s1_20_spa_dsr": s20_report,
        "s1_21_g3_gate": s21_report,
    }

    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved Phase D report to {}", report_path)

    # Summary
    logger.info("=" * 60)
    logger.info("Phase D Summary:")
    logger.info("  S1.17 net_excess={:.4f} turnover={:.4f} sharpe={:.4f}",
                s17_report["sw_controlled"]["agg"]["annualized_net_excess"],
                s17_report["sw_controlled"]["agg"]["average_turnover"],
                s17_report["sw_controlled"]["agg"]["sharpe_proxy"])
    logger.info("  S1.20 SPA p={:.4f} significant={}",
                s20_report["spa_primary"]["p_value"],
                s20_report["spa_primary"]["significant"])
    logger.info("  S1.21 G3 Gate: {}/{} PASS → {}",
                s21_report["total_passed"], s21_report["total_checks"],
                "PASS" if s21_report["gate_pass"] else "FAIL")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase D validation")
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--fusion-report", default=DEFAULT_FUSION_REPORT_PATH)
    parser.add_argument("--comparison-report", default=DEFAULT_COMPARISON_REPORT_PATH)
    parser.add_argument("--turnover-report", default=DEFAULT_TURNOVER_REPORT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    parser.add_argument("--skip-stacker", action="store_true")
    return parser.parse_args(argv)


# ===================================================================
# S1.17: Full walk-forward validation
# ===================================================================

def run_s17_walkforward(
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    prices_df: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    fusion_data: dict,
    comparison_data: dict,
) -> dict[str, Any]:
    """Run full walk-forward with optimal config + EW baseline, per-window metrics."""

    # Score-weighted with Phase B optimal config
    logger.info("Running SW controlled (Phase B optimal) across {} windows...",
                len(predictions_by_window))
    sw_portfolios: list[PortfolioBacktestResult] = []
    sw_per_window: dict[str, dict] = {}

    for window_id in sorted(predictions_by_window):
        portfolio = simulate_score_weighted_controlled(
            predictions=predictions_by_window[window_id],
            execution=execution,
            cost_model=cost_model,
            benchmark_ticker=BENCHMARK_TICKER,
            **OPTIMAL_SW_CONFIG,
        )
        sw_portfolios.append(portfolio)
        window_agg = aggregate_window_portfolios([portfolio])
        sw_per_window[window_id] = window_agg

    sw_all_periods = _collect_periods(sw_portfolios)
    sw_agg = _aggregate_from_periods(sw_all_periods)

    # Equal-weight baseline
    logger.info("Running EW baseline across {} windows...", len(predictions_by_window))
    ew_portfolios: list[PortfolioBacktestResult] = []
    ew_per_window: dict[str, dict] = {}

    for window_id in sorted(predictions_by_window):
        portfolio = simulate_portfolio(
            predictions=predictions_by_window[window_id],
            prices=prices_df,
            cost_model=cost_model,
            weighting_scheme="equal_weight",
            benchmark_ticker=BENCHMARK_TICKER,
            universe_by_date=None,
            selection_pct=0.20,
            sell_buffer_pct=0.25,
            min_trade_weight=0.01,
            max_weight=0.05,
            min_holdings=20,
        )
        ew_portfolios.append(portfolio)
        window_agg = aggregate_window_portfolios([portfolio])
        ew_per_window[window_id] = window_agg

    ew_all_periods = _collect_periods(ew_portfolios)
    ew_agg = _aggregate_from_periods(ew_all_periods)

    # Per-window IC from fusion report (map to our window IDs)
    per_window_ic = {}
    fusion_windows = {w["window_id"]: w for w in fusion_data.get("per_window", [])}
    # Map extended walkforward W1-W8 to comparison W4-W11 (last 8 of 11)
    # Actually just use whichever windows match
    for window_id in sorted(predictions_by_window):
        fw = fusion_windows.get(window_id, {})
        per_window_ic[window_id] = {
            "fusion_ic": fw.get("ic_weighted_fusion_ic"),
            "ridge_ic": fw.get("ridge_ic"),
            "xgboost_ic": fw.get("xgboost_ic"),
            "lightgbm_ic": fw.get("lightgbm_ic"),
        }

    logger.info("S1.17 results:")
    logger.info("  SW controlled: net_excess={:.4f} turnover={:.4f} sharpe={:.4f}",
                sw_agg["annualized_net_excess"], sw_agg["average_turnover"], sw_agg["sharpe_proxy"])
    logger.info("  EW baseline:   net_excess={:.4f} turnover={:.4f} sharpe={:.4f}",
                ew_agg["annualized_net_excess"], ew_agg["average_turnover"], ew_agg["sharpe_proxy"])

    for wid in sorted(sw_per_window):
        sw = sw_per_window[wid]
        ew = ew_per_window[wid]
        logger.info("  {}: SW net={:.4f} TO={:.4f} | EW net={:.4f} TO={:.4f}",
                    wid, sw["annualized_net_excess"], sw["average_turnover"],
                    ew["annualized_net_excess"], ew["average_turnover"])

    return {
        "sw_controlled": {
            "config": OPTIMAL_SW_CONFIG,
            "agg": sw_agg,
            "per_window": sw_per_window,
            "period_details": sw_all_periods,
            "n_periods": len(sw_all_periods),
        },
        "ew_baseline": {
            "agg": ew_agg,
            "per_window": ew_per_window,
            "period_details": ew_all_periods,
            "n_periods": len(ew_all_periods),
        },
        "per_window_ic": per_window_ic,
        "n_windows": len(predictions_by_window),
    }


# ===================================================================
# S1.18: Hyperparameter sensitivity check
# ===================================================================

def run_s18_hyperparam_sensitivity(
    predictions_by_window: dict[str, pd.Series],
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
) -> dict[str, Any]:
    """Check if current Phase B params are near-optimal by perturbing each parameter.

    For each param, test ±1 step from optimal. If net excess changes < 0.5%,
    the optimum is robust (not on a cliff edge).
    """
    perturbations = {
        "sell_buffer_pct": [0.35, 0.40, 0.45],
        "weight_shrinkage": [0.4, 0.5, 0.6],
        "no_trade_zone": [0.003, 0.005, 0.007],
        "min_trade_weight": [0.003, 0.005, 0.007],
        "selection_pct": [0.20, 0.25, 0.30],
    }

    results: dict[str, list[dict]] = {}
    total_configs = sum(len(vals) for vals in perturbations.values())
    done = 0

    for param_name, values in perturbations.items():
        param_results = []
        for val in values:
            config = {**OPTIMAL_SW_CONFIG, param_name: val}
            scheme_name = f"sens_{param_name}={val}"

            window_portfolios = []
            for window_id in sorted(predictions_by_window):
                portfolio = simulate_score_weighted_controlled(
                    predictions=predictions_by_window[window_id],
                    execution=execution,
                    cost_model=cost_model,
                    benchmark_ticker=BENCHMARK_TICKER,
                    **config,
                )
                window_portfolios.append(portfolio)

            agg = aggregate_window_portfolios(window_portfolios)
            param_results.append({
                "value": val,
                "net_excess": agg["annualized_net_excess"],
                "turnover": agg["average_turnover"],
                "sharpe": agg["sharpe_proxy"],
                "cost_drag": agg["total_cost_drag"],
            })
            done += 1

        results[param_name] = param_results

        # Log sensitivity
        excesses = [r["net_excess"] for r in param_results]
        logger.info("  {}: net_excess range [{:.4f}, {:.4f}], spread={:.4f}",
                    param_name, min(excesses), max(excesses), max(excesses) - min(excesses))

    # Check robustness: is the optimal near the peak for each param?
    robust = True
    for param_name, param_results in results.items():
        optimal_val = OPTIMAL_SW_CONFIG[param_name]
        optimal_excess = next(
            (r["net_excess"] for r in param_results if r["value"] == optimal_val), None
        )
        best_excess = max(r["net_excess"] for r in param_results)
        if optimal_excess is not None and best_excess - optimal_excess > 0.005:
            robust = False
            logger.warning("  {} is not at peak: optimal={:.4f} vs best={:.4f}",
                          param_name, optimal_excess, best_excess)

    logger.info("S1.18 robustness: {}", "ROBUST" if robust else "SENSITIVITY DETECTED")

    return {
        "perturbations": results,
        "robust": robust,
        "optimal_config": OPTIMAL_SW_CONFIG,
    }


# ===================================================================
# S1.19: Stacker experiment
# ===================================================================

def run_s19_stacker(
    comparison_data: dict,
    fusion_data: dict,
) -> dict[str, Any]:
    """Lightweight stacking experiment using out-of-fold model predictions.

    Uses per-window IC data to simulate what a meta-learner (linear stacker)
    would achieve vs the IC-weighted adaptive fusion.
    """
    # Collect per-window, per-model IC
    windows = fusion_data.get("per_window", [])
    if len(windows) < 5:
        return {"status": "insufficient_windows", "n_windows": len(windows)}

    model_names = ["ridge", "xgboost", "lightgbm"]
    ic_matrix = []
    fusion_ics = []

    for w in windows:
        row = [float(w.get(f"{m}_ic", 0.0)) for m in model_names]
        ic_matrix.append(row)
        fusion_ics.append(float(w.get("ic_weighted_fusion_ic", 0.0)))

    ic_matrix = np.array(ic_matrix)  # (n_windows, 3)
    fusion_ics = np.array(fusion_ics)

    # Leave-one-out stacking: for each window, train weights on all OTHER windows
    stacker_ics = []
    stacker_weights_per_window = []

    for i in range(len(windows)):
        # Train: all windows except i
        train_ic = np.delete(ic_matrix, i, axis=0)
        # Optimal weights = proportional to mean IC (non-negative)
        mean_train_ic = train_ic.mean(axis=0)
        weights = np.maximum(mean_train_ic, 0.0)
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(len(model_names)) / len(model_names)

        # Predict: weighted combination of model ICs for window i
        stacker_ic = float(np.dot(ic_matrix[i], weights))
        stacker_ics.append(stacker_ic)
        stacker_weights_per_window.append({
            m: float(w) for m, w in zip(model_names, weights)
        })

    stacker_ics = np.array(stacker_ics)

    # Compare stacker vs adaptive fusion
    delta = stacker_ics - fusion_ics
    if len(delta) > 1 and delta.std() > 0:
        t_stat, p_val_two = stats.ttest_1samp(delta, 0.0)
        p_val = float(p_val_two / 2.0) if t_stat > 0 else float(1.0 - p_val_two / 2.0)
    else:
        t_stat, p_val = 0.0, 1.0

    stacker_mean_ic = float(stacker_ics.mean())
    fusion_mean_ic = float(fusion_ics.mean())
    significant = p_val < 0.05

    logger.info("S1.19 Stacker: mean_IC={:.4f} vs Fusion={:.4f}, delta={:.4f}, p={:.4f} sig={}",
                stacker_mean_ic, fusion_mean_ic,
                stacker_mean_ic - fusion_mean_ic, p_val, significant)

    decision = "adopt_stacker" if significant and stacker_mean_ic > fusion_mean_ic else "keep_adaptive_fusion"

    return {
        "status": "completed",
        "stacker_mean_ic": stacker_mean_ic,
        "fusion_mean_ic": fusion_mean_ic,
        "ic_delta": float(stacker_mean_ic - fusion_mean_ic),
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "significant": significant,
        "decision": decision,
        "per_window": [
            {
                "window_id": w["window_id"],
                "stacker_ic": float(s),
                "fusion_ic": float(f),
                "stacker_weights": wt,
            }
            for w, s, f, wt in zip(windows, stacker_ics, fusion_ics, stacker_weights_per_window)
        ],
    }


# ===================================================================
# S1.20: DSR/SPA final test
# ===================================================================

def run_s20_spa_dsr(
    sw_periods: list[dict],
    ew_periods: list[dict],
    fusion_data: dict,
) -> dict[str, Any]:
    """SPA test (SW controlled vs EW) + DSR on fusion IC."""

    # SPA: period-level net excess comparison
    sw_series = _periods_to_series(sw_periods, "net_excess_return")
    ew_series = _periods_to_series(ew_periods, "net_excess_return")

    spa_result = run_spa_fallback(
        benchmark_series=ew_series,
        competitors={"score_weighted_controlled": sw_series},
        benchmark_name="equal_weight",
    )

    primary = spa_result.comparisons[0] if spa_result.comparisons else None
    spa_report = {
        "p_value": float(spa_result.p_value),
        "significant": spa_result.significant,
        "method": spa_result.method,
        "n_observations": primary.n_observations if primary else 0,
        "t_statistic": float(primary.t_statistic) if primary else 0.0,
        "mean_delta": float(primary.mean_ic_delta) if primary else 0.0,
    }

    logger.info("S1.20 SPA: p={:.4f} significant={} t={:.4f}",
                spa_report["p_value"], spa_report["significant"], spa_report["t_statistic"])

    # DSR: Deflated Sharpe Ratio
    # Compute from fusion IC series across windows
    # n_trials = actual MLflow experiment count (114 runs)
    fusion_ics = np.array([
        float(w.get("ic_weighted_fusion_ic", 0.0))
        for w in fusion_data.get("per_window", [])
    ])

    dsr_report = _compute_simple_dsr(fusion_ics, n_trials=114)
    logger.info("S1.20 DSR: sharpe={:.4f} dsr_p={:.4f} significant={}",
                dsr_report["observed_sharpe"], dsr_report["p_value"], dsr_report["significant"])

    return {
        "spa_primary": spa_report,
        "dsr": dsr_report,
    }


def _compute_simple_dsr(
    ic_series: np.ndarray,
    n_trials: int = 50,  # approximate total experiments run
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Simplified Deflated Sharpe Ratio.

    Adjusts for multiple testing by estimating the expected maximum Sharpe
    ratio under the null (no skill), accounting for skewness and kurtosis.
    """
    n = len(ic_series)
    if n < 3:
        return {"observed_sharpe": 0.0, "p_value": 1.0, "significant": False, "reason": "insufficient_data"}

    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std(ddof=1))
    if std_ic < 1e-10:
        return {"observed_sharpe": 0.0, "p_value": 1.0, "significant": False, "reason": "zero_variance"}

    observed_sharpe = mean_ic / std_ic

    # Expected max Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    expected_max_sharpe = (
        (1.0 - euler_mascheroni) * stats.norm.ppf(1.0 - 1.0 / n_trials)
        + euler_mascheroni * stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    )

    # Skewness and kurtosis adjustment
    skew = float(stats.skew(ic_series))
    kurt = float(stats.kurtosis(ic_series, fisher=True))

    # Adjusted Sharpe (Lo 2002 + Bailey & de Prado 2014)
    adjustment = 1.0 - skew * observed_sharpe / 3.0 + (kurt - 3.0) * observed_sharpe ** 2 / 24.0
    adjusted_sharpe = observed_sharpe * math.sqrt(max(adjustment, 0.01))

    # Test: is adjusted Sharpe > expected max under null?
    # Use normal approximation for the distribution of SR
    se_sharpe = math.sqrt((1.0 + 0.5 * observed_sharpe ** 2) / n)
    z_stat = (adjusted_sharpe - expected_max_sharpe) / se_sharpe if se_sharpe > 0 else 0.0
    p_value = float(1.0 - stats.norm.cdf(z_stat))

    return {
        "observed_sharpe": float(observed_sharpe),
        "adjusted_sharpe": float(adjusted_sharpe),
        "expected_max_sharpe_null": float(expected_max_sharpe),
        "skewness": float(skew),
        "kurtosis": float(kurt),
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "n_trials": n_trials,
        "n_observations": int(n),
    }


# ===================================================================
# S1.21: G3 Gate retest
# ===================================================================

def run_s21_g3_gate(
    s17_report: dict,
    s20_report: dict,
    fusion_data: dict,
) -> dict[str, Any]:
    """Retest 6 G3 Gate checks using Phase D results."""

    sw_agg = s17_report["sw_controlled"]["agg"]
    fusion_ics = np.array([
        float(w.get("ic_weighted_fusion_ic", 0.0))
        for w in fusion_data.get("per_window", [])
    ])

    # Check 1: OOS IC above threshold (0.03)
    mean_ic = float(fusion_ics.mean())
    positive_windows = int((fusion_ics > 0).sum())
    check_1 = {
        "name": "oos_ic_above_threshold",
        "value": mean_ic,
        "threshold": 0.03,
        "positive_windows": positive_windows,
        "n_windows": len(fusion_ics),
        "passed": bool(mean_ic > 0.03),
    }

    # Check 2: IC t-test significant
    if len(fusion_ics) > 1 and fusion_ics.std() > 0:
        t_stat, p_two = stats.ttest_1samp(fusion_ics, 0.0)
        p_val = float(p_two / 2.0) if t_stat > 0 else float(1.0 - p_two / 2.0)
    else:
        t_stat, p_val = 0.0, 1.0
    check_2 = {
        "name": "ic_ttest_significant",
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "threshold": 0.05,
        "passed": bool(p_val < 0.05),
    }

    # Check 3: Bootstrap CI lower bound > 0 (annualized)
    sw_periods = s17_report["sw_controlled"]["period_details"]
    net_excess = np.array([p["net_excess_return"] for p in sw_periods])
    boot_result = bootstrap_return_statistics(
        net_excess, block_size=4, n_bootstrap=10000, annualization=52,
    )
    check_3 = {
        "name": "bootstrap_ci_positive",
        "annualized_excess": boot_result.annualized_excess_estimate,
        "annualized_ci_lower": boot_result.annualized_excess_ci_lower,
        "annualized_ci_upper": boot_result.annualized_excess_ci_upper,
        "sharpe": boot_result.sharpe_estimate,
        "sharpe_ci_lower": boot_result.sharpe_ci_lower,
        "sharpe_ci_upper": boot_result.sharpe_ci_upper,
        "passed": bool(boot_result.annualized_excess_ci_lower > 0),
    }

    # Check 4: DSR significant
    dsr = s20_report["dsr"]
    check_4 = {
        "name": "dsr_significant",
        "dsr_p_value": dsr["p_value"],
        "observed_sharpe": dsr["observed_sharpe"],
        "passed": dsr["significant"],
    }

    # Check 5: SPA vs baseline significant
    spa = s20_report["spa_primary"]
    check_5 = {
        "name": "spa_vs_baseline",
        "spa_p_value": spa["p_value"],
        "passed": spa["significant"],
    }

    # Check 6: Cost-adjusted annualized excess > threshold
    ann_net_excess = sw_agg["annualized_net_excess"]
    check_6 = {
        "name": "cost_adjusted_excess",
        "value": ann_net_excess,
        "threshold_base": 0.035,
        "threshold_stretch": 0.05,
        "passed_base": bool(ann_net_excess >= 0.035),
        "passed_stretch": bool(ann_net_excess >= 0.05),
        "passed": bool(ann_net_excess >= 0.035),
    }

    checks = [check_1, check_2, check_3, check_4, check_5, check_6]
    total_passed = sum(1 for c in checks if c["passed"])
    gate_pass = total_passed >= 6  # strict: all 6 must pass

    for c in checks:
        status = "PASS" if c["passed"] else "FAIL"
        logger.info("  {}: {} ({})", c["name"], status,
                    {k: v for k, v in c.items() if k not in ("name", "passed")})

    logger.info("G3 Gate: {}/{} → {}", total_passed, len(checks),
                "PASS" if gate_pass else "FAIL")

    return {
        "checks": {c["name"]: c for c in checks},
        "total_passed": total_passed,
        "total_checks": len(checks),
        "gate_pass": gate_pass,
    }


# ===================================================================
# Utilities
# ===================================================================

def _collect_periods(portfolios: list[PortfolioBacktestResult]) -> list[dict[str, Any]]:
    periods = []
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


def _aggregate_from_periods(periods: list[dict]) -> dict[str, Any]:
    if not periods:
        return {"annualized_net_excess": 0, "annualized_gross_excess": 0,
                "average_turnover": 0, "sharpe_proxy": 0, "total_cost_drag": 0}
    df = pd.DataFrame(periods)
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df.sort_values("execution_date", inplace=True)

    gross_curve = (1 + df["gross_return"]).cumprod()
    net_curve = (1 + df["net_return"]).cumprod()
    bench_curve = (1 + df["benchmark_return"]).cumprod()
    total_days = max((df["exit_date"].iloc[-1] - df["execution_date"].iloc[0]).days, 1)

    def ann(total_ret: float) -> float:
        base = 1.0 + total_ret
        return float(base ** (365.25 / total_days) - 1.0) if base > 0 else -1.0

    ag = ann(float(gross_curve.iloc[-1] - 1))
    an = ann(float(net_curve.iloc[-1] - 1))
    ab = ann(float(bench_curve.iloc[-1] - 1))

    net_ex = df["net_return"] - df["benchmark_return"]
    sharpe = float(net_ex.mean() / net_ex.std() * np.sqrt(52)) if len(net_ex) > 1 and net_ex.std() > 0 else 0.0

    return {
        "annualized_gross_excess": float(ag - ab),
        "annualized_net_excess": float(an - ab),
        "annualized_gross_return": float(ag),
        "annualized_net_return": float(an),
        "annualized_benchmark_return": float(ab),
        "total_cost_drag": float(ag - an),
        "average_turnover": float(df["turnover"].mean()),
        "sharpe_proxy": sharpe,
        "period_count": len(df),
    }


def _periods_to_series(periods: list[dict], key: str = "net_excess_return") -> pd.Series:
    if not periods:
        return pd.Series(dtype=float)
    df = pd.DataFrame(periods)
    df["execution_date"] = pd.to_datetime(df["execution_date"])
    return pd.Series(df[key].astype(float).values, index=df["execution_date"]).sort_index()


def _bootstrap_ci(
    returns: np.ndarray,
    n_boot: int = 5000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Block bootstrap 95% CI for mean return."""
    rng = np.random.default_rng(42)
    n = len(returns)
    if n < 2:
        return (0.0, 0.0)
    block_size = max(1, int(np.sqrt(n)))
    means = []
    for _ in range(n_boot):
        indices = []
        while len(indices) < n:
            start = rng.integers(0, n)
            indices.extend(range(start, min(start + block_size, n)))
        indices = indices[:n]
        means.append(float(returns[indices].mean()))
    alpha = (1.0 - ci) / 2.0
    return (float(np.percentile(means, alpha * 100)), float(np.percentile(means, (1 - alpha) * 100)))


def _strip_period_details(report: dict) -> dict:
    """Remove bulky period_details from report for JSON output."""
    out = {}
    for key, val in report.items():
        if isinstance(val, dict) and "period_details" in val:
            out[key] = {k: v for k, v in val.items() if k != "period_details"}
        else:
            out[key] = val
    return out


if __name__ == "__main__":
    raise SystemExit(main())
