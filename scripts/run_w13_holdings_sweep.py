#!/usr/bin/env python3
"""W13.x — Holdings cap × Layer-3 risk sensitivity sweep.

Reuses the exact W10 truth-table data path so the baseline reproduces
7.34% net excess. Forks the score_weighted_buffered simulator to add an
optional ``max_holdings`` cap (applied AFTER candidate selection /
hysteresis, BEFORE score-weight normalization) and an optional Layer-3
PortfolioRiskEngine pass with PIT-correct trailing returns.

8 configs:
    baseline                — no cap, no Layer 3 (sanity-check vs W10 truth table)
    cap50_only / cap30_only / cap20_only       — cap, no Layer 3
    layer3_only             — no cap, Layer 3
    cap50_plus_layer3 / cap30_plus_layer3 / cap20_plus_layer3 — cap + Layer 3

Output:
    data/reports/w13_holdings_sweep_<as_of>.json
    data/reports/W13_holdings_sweep_verdict_<as_of>.md
    ASCII table to stdout

Usage:
    python scripts/run_w13_holdings_sweep.py
        [--report data/reports/walkforward_v9full9y_60d_ridge_13w.json]
        [--feature-matrix-path data/features/walkforward_v9full9y_fm_60d.parquet]
        [--label-path data/labels/forward_returns_60d_v9full9y.parquet]
        [--window-limit N]
        [--out data/reports/w13_holdings_sweep_2026-04-27.json]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_horizon_fusion import (  # noqa: E402
    parse_horizon_days,
    prepare_horizon_artifacts,
    select_report_windows,
)
from scripts.run_w10_truth_table import (  # noqa: E402
    BENCHMARK_TICKER,
    REBALANCE_WEEKDAY,
    aggregate_truth_row,
    build_predictions,
    load_report,
)
from scripts.run_walkforward_comparison import (  # noqa: E402
    DEFAULT_ALL_FEATURES_PATH,
    LABEL_BUFFER_DAYS,
    parse_date,
)
from scripts.run_turnover_optimization import _build_backtest_result  # noqa: E402
from src.backtest.cost_model import AlmgrenChrissCostModel  # noqa: E402
from src.backtest.execution import (  # noqa: E402
    PortfolioBacktestResult,
    PortfolioPeriodResult,
    build_execution_schedule,
    prepare_execution_price_frame,
    select_candidate_tickers,
)
from src.data.db.pit import get_prices_pit  # noqa: E402
from src.portfolio.constraints import (  # noqa: E402
    PortfolioConstraints,
    apply_turnover_buffer,
    apply_weight_constraints,
)
from src.risk.portfolio_risk import PortfolioRiskEngine  # noqa: E402
from scripts.run_live_pipeline import build_return_history, load_sector_map  # noqa: E402

# Live W12 history window — match scripts/run_greyscale_live.py DEFAULT_HISTORY_LOOKBACK_DAYS
LIVE_HISTORY_LOOKBACK_DAYS = 400

# Champion settings (matching W10 truth-table score_weighted_buffered + bundle v3)
CHAMPION_PARAMS = {
    "selection_pct": 0.20,
    "sell_buffer_pct": 0.25,
    "min_trade_weight": 0.01,
    "max_weight": 0.05,
    "min_holdings": 20,
}

# Layer 3 thresholds — matched to bundle v3 / live W12 settings.
LAYER3_DEFAULTS = {
    "max_single_stock_weight": 0.05,
    "max_sector_deviation": 0.15,   # research default; live uses ~0.09 implicitly
    "cvar_floor": -0.05,
    "cvar_confidence": 0.99,
    "correlation_threshold": 0.85,
    "turnover_cap": 0.40,
    "min_holdings": 20,
}

# 8 sweep configurations
SWEEP_CONFIGS = [
    {"name": "baseline",              "max_holdings": None, "layer3": False},
    {"name": "cap50_only",            "max_holdings": 50,   "layer3": False},
    {"name": "cap30_only",            "max_holdings": 30,   "layer3": False},
    {"name": "cap20_only",            "max_holdings": 20,   "layer3": False},
    {"name": "layer3_only",           "max_holdings": None, "layer3": True},
    {"name": "cap50_plus_layer3",     "max_holdings": 50,   "layer3": True},
    {"name": "cap30_plus_layer3",     "max_holdings": 30,   "layer3": True},
    {"name": "cap20_plus_layer3",     "max_holdings": 20,   "layer3": True},
]

# Loose Layer 3 thresholds — for sensitivity test
LAYER3_LOOSE = {
    "max_single_stock_weight": 0.05,
    "max_sector_deviation": 0.25,
    "cvar_floor": -0.10,
    "cvar_confidence": 0.99,
    "correlation_threshold": 0.92,
    "turnover_cap": 0.40,
    "min_holdings": 20,
}
# Medium loose Layer 3 thresholds
LAYER3_MEDIUM = {
    "max_single_stock_weight": 0.05,
    "max_sector_deviation": 0.20,
    "cvar_floor": -0.08,
    "cvar_confidence": 0.99,
    "correlation_threshold": 0.90,
    "turnover_cap": 0.40,
    "min_holdings": 20,
}

LAYER3_TUNING_CONFIGS = [
    {"name": "layer3_medium",          "max_holdings": None, "layer3": True, "layer3_kwargs": LAYER3_MEDIUM},
    {"name": "layer3_loose",           "max_holdings": None, "layer3": True, "layer3_kwargs": LAYER3_LOOSE},
    {"name": "cap30_plus_layer3_med",  "max_holdings": 30,   "layer3": True, "layer3_kwargs": LAYER3_MEDIUM},
    {"name": "cap30_plus_layer3_loose","max_holdings": 30,   "layer3": True, "layer3_kwargs": LAYER3_LOOSE},
]


@dataclass
class ConfigSummary:
    name: str
    truth_row: dict[str, Any]
    holdings_pre_risk_mean: float
    holdings_final_mean: float
    holdings_final_min: int
    holdings_final_max: int
    gross_exposure_mean: float
    cash_weight_mean: float
    cvar_trigger_rate: float
    avg_haircut_rounds: float


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--report", default="data/reports/walkforward_v9full9y_60d_ridge_13w.json")
    p.add_argument("--feature-matrix-path", default="data/features/walkforward_v9full9y_fm_60d.parquet")
    p.add_argument("--label-path", default="data/labels/forward_returns_60d_v9full9y.parquet")
    p.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    p.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    p.add_argument("--price-buffer-days", type=int, default=120)
    p.add_argument("--window-limit", type=int, default=None,
                   help="Limit to first N walk-forward windows (debug speedup)")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: data/reports/w13_holdings_sweep_<utc>.json)")
    p.add_argument("--cost-mult", type=float, default=1.0,
                   help="Cost multiplier vs baseline (1.0 = production)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Score weight construction (forked from run_turnover_optimization to add max_holdings)
# ---------------------------------------------------------------------------

def build_score_weights(candidate_scores: pd.Series, max_weight: float) -> dict[str, float]:
    """Replicate _build_score_weights from run_turnover_optimization.py."""
    pos_scores = candidate_scores[candidate_scores > 0.0]
    if pos_scores.empty:
        return {}
    raw = pos_scores / pos_scores.sum()
    raw = raw.clip(upper=float(max_weight))
    total = float(raw.sum())
    if total <= 0.0:
        return {}
    raw = raw / total
    return {str(ticker): float(weight) for ticker, weight in raw.items() if weight > 0.0}


# ---------------------------------------------------------------------------
# PIT trailing return panel
# ---------------------------------------------------------------------------

def build_return_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Build (trade_date × ticker) daily return panel via live's cleaned helper.

    Codex finding (2026-04-27): the previous custom helper used raw pct_change()
    on wide close prices, which kept extreme outliers (e.g. META 2022-06-09 at
    +1300%) that live's `build_return_history` already strips out. Switching to
    the live helper makes the sweep return panel feed-identical to live W12.
    """
    return_history, _spy_returns = build_return_history(prices)
    return_history.index = pd.DatetimeIndex(return_history.index)
    return return_history


# ---------------------------------------------------------------------------
# Forked simulator with max_holdings + Layer 3
# ---------------------------------------------------------------------------

@dataclass
class PeriodAudit:
    holdings_pre_risk: int = 0
    holdings_final: int = 0
    gross_exposure_final: float = 0.0
    cash_weight_final: float = 0.0
    cvar_triggered: bool = False
    cvar_iterations: int = 0


def simulate_with_cap_and_layer3(
    *,
    predictions: pd.Series,
    execution: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    return_panel: pd.DataFrame,
    sector_map: dict[str, str],
    benchmark_ticker: str,
    max_holdings: int | None,
    layer3: bool,
    layer3_kwargs: dict[str, Any] | None = None,
    initial_capital: float = 1_000_000.0,
) -> tuple[PortfolioBacktestResult, list[PeriodAudit]]:
    benchmark = benchmark_ticker.upper()
    risk_engine = PortfolioRiskEngine() if layer3 else None

    signal_dates = pd.DatetimeIndex(
        pd.to_datetime(predictions.index.get_level_values("trade_date"))
    ).sort_values().unique()
    trade_dates = execution.index.get_level_values("trade_date").unique().sort_values()
    schedule = build_execution_schedule(signal_dates, trade_dates)

    constraints = PortfolioConstraints(
        max_weight=CHAMPION_PARAMS["max_weight"],
        min_holdings=CHAMPION_PARAMS["min_holdings"],
        turnover_buffer=CHAMPION_PARAMS["min_trade_weight"],
    )

    current_weights: dict[str, float] = {}
    portfolio_value = float(initial_capital)
    periods: list[PortfolioPeriodResult] = []
    audits: list[PeriodAudit] = []
    cost_totals: dict[str, float] = defaultdict(float)

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
        if len(eligible) < max(CHAMPION_PARAMS["min_holdings"], 2):
            continue

        filtered_scores = score_frame.loc[score_frame.index.astype(str).isin(eligible)].sort_values(ascending=False)
        if filtered_scores.empty:
            continue
        ranking = filtered_scores.index.astype(str).tolist()

        # Candidate selection (hysteresis)
        candidate_tickers = select_candidate_tickers(
            ranking=ranking,
            current_weights=current_weights,
            selection_pct=CHAMPION_PARAMS["selection_pct"],
            sell_buffer_pct=CHAMPION_PARAMS["sell_buffer_pct"],
            min_holdings=CHAMPION_PARAMS["min_holdings"],
            max_weight=CHAMPION_PARAMS["max_weight"],
        )

        # *** max_holdings cap — applied AFTER hysteresis, BEFORE score-weight normalization ***
        if max_holdings is not None and len(candidate_tickers) > max_holdings:
            ranked_candidates = [t for t in ranking if t in set(candidate_tickers)]
            candidate_tickers = ranked_candidates[:max_holdings]

        candidate_scores = filtered_scores.reindex(candidate_tickers).dropna()
        if candidate_scores.empty:
            continue

        target_weights = build_score_weights(candidate_scores, max_weight=CHAMPION_PARAMS["max_weight"])
        if not target_weights:
            continue

        # Turnover buffer (matches simulate_score_weighted_controlled)
        if CHAMPION_PARAMS["min_trade_weight"] > 0.0:
            buffer_reference_weights = {
                ticker: w for ticker, w in current_weights.items() if ticker in set(ranking)
            }
            target_weights = apply_turnover_buffer(
                target_weights,
                current_weights=buffer_reference_weights,
                min_trade_weight=CHAMPION_PARAMS["min_trade_weight"],
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

        audit = PeriodAudit(holdings_pre_risk=len(target_weights))

        # *** Layer 3 — PIT-correct trailing returns ***
        cvar_triggered = False
        cvar_iterations = 0
        if layer3 and risk_engine is not None:
            # PIT trailing returns up to (but excluding) execution_date.
            # Codex finding: cap trailing history at LIVE_HISTORY_LOOKBACK_DAYS
            # (400 days), matching scripts/run_greyscale_live.py. Without this
            # cap the sweep used expanding-since-2018 history and CVaR was
            # permanently anchored to COVID tails, inflating trigger rate.
            exec_ts = pd.Timestamp(execution_date)
            history_start = exec_ts - pd.Timedelta(days=LIVE_HISTORY_LOOKBACK_DAYS)
            trailing = return_panel.loc[history_start:exec_ts].iloc[:-1]
            if benchmark not in trailing.columns:
                # Fallback — Layer 3 cannot run without SPY
                pass
            else:
                target_tickers = list(target_weights.keys())
                trailing_target_returns = trailing.reindex(columns=target_tickers).dropna(how="all")
                spy_returns = trailing[benchmark].dropna()
                # Equal-weight benchmark on the candidate-ranking universe
                eq_universe = filtered_scores.index.astype(str).tolist()
                benchmark_weights = {t: 1.0 / len(eq_universe) for t in eq_universe} if eq_universe else {}

                kw = layer3_kwargs if layer3_kwargs is not None else LAYER3_DEFAULTS
                constrained = risk_engine.apply_all_constraints(
                    weights=target_weights,
                    benchmark_weights=benchmark_weights,
                    sector_map=sector_map,
                    return_history=trailing_target_returns,
                    spy_returns=spy_returns,
                    current_weights=current_weights,
                    candidate_ranking=ranking,
                    max_single_stock_weight=kw["max_single_stock_weight"],
                    max_sector_deviation=kw["max_sector_deviation"],
                    cvar_floor=kw["cvar_floor"],
                    cvar_confidence=kw["cvar_confidence"],
                    correlation_threshold=kw["correlation_threshold"],
                    turnover_cap=kw["turnover_cap"],
                    min_holdings=kw["min_holdings"],
                )
                target_weights = dict(constrained.weights)
                audit.gross_exposure_final = float(constrained.gross_exposure)
                audit.cash_weight_final = float(constrained.cash_weight)
                # Find CVaR audit entry
                for entry in constrained.audit_trail:
                    if entry.rule_name == "cvar_haircut":
                        cvar_triggered = bool(entry.triggered)
                        cvar_iterations = int(entry.after.get("iterations", 0))
                        break

        if not target_weights:
            continue

        audit.holdings_final = len([w for w in target_weights.values() if w > 1e-9])
        if not layer3:
            audit.gross_exposure_final = sum(target_weights.values())
            audit.cash_weight_final = max(0.0, 1.0 - audit.gross_exposure_final)
        audit.cvar_triggered = cvar_triggered
        audit.cvar_iterations = cvar_iterations

        # Cost computation (replica of simulate_score_weighted_controlled)
        previous_weights = dict(current_weights)
        all_weight_keys = set(target_weights) | set(previous_weights)
        period_costs: dict[str, float] = defaultdict(float)
        for ticker in all_weight_keys:
            target_w = target_weights.get(ticker, 0.0)
            previous_w = previous_weights.get(ticker, 0.0)
            delta_weight = target_w - previous_w
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
            )
        )
        audits.append(audit)
        for key, value in period_costs.items():
            cost_totals[key] += float(value)

    backtest_result = _build_backtest_result(periods=periods, cost_totals=dict(cost_totals))
    return backtest_result, audits


# ---------------------------------------------------------------------------
# Aggregation per config
# ---------------------------------------------------------------------------

def aggregate_config(name: str, result: PortfolioBacktestResult, audits: list[PeriodAudit]) -> ConfigSummary:
    if not result.periods:
        empty = {
            "strategy": name, "cost_mult": 1.0, "gate_on": False, "n_periods": 0,
            "gross_ann_excess": float("nan"), "net_ann_excess": float("nan"),
            "cost_drag_ann": 0.0, "avg_turnover": 0.0,
            "sharpe": float("nan"), "ir": float("nan"), "max_drawdown": float("nan"),
        }
        return ConfigSummary(
            name=name, truth_row=empty,
            holdings_pre_risk_mean=0.0, holdings_final_mean=0.0,
            holdings_final_min=0, holdings_final_max=0,
            gross_exposure_mean=0.0, cash_weight_mean=0.0,
            cvar_trigger_rate=0.0, avg_haircut_rounds=0.0,
        )

    periods_df = pd.DataFrame([p.to_dict() for p in result.periods])
    periods_df["signal_date"] = pd.to_datetime(periods_df["signal_date"])
    periods_df["execution_date"] = pd.to_datetime(periods_df["execution_date"])
    periods_df["exit_date"] = pd.to_datetime(periods_df["exit_date"])
    periods_df = periods_df.sort_values("execution_date").reset_index(drop=True)

    truth_row = aggregate_truth_row(
        strategy=name, cost_mult=1.0, gate_on=False, periods=periods_df,
    )

    holdings_pre = [a.holdings_pre_risk for a in audits]
    holdings_final = [a.holdings_final for a in audits]
    gross = [a.gross_exposure_final for a in audits]
    cash = [a.cash_weight_final for a in audits]
    cvar_trig = [1 if a.cvar_triggered else 0 for a in audits]
    haircut_rounds = [a.cvar_iterations for a in audits if a.cvar_triggered]

    return ConfigSummary(
        name=name,
        truth_row=truth_row,
        holdings_pre_risk_mean=float(np.mean(holdings_pre)) if holdings_pre else 0.0,
        holdings_final_mean=float(np.mean(holdings_final)) if holdings_final else 0.0,
        holdings_final_min=int(np.min(holdings_final)) if holdings_final else 0,
        holdings_final_max=int(np.max(holdings_final)) if holdings_final else 0,
        gross_exposure_mean=float(np.mean(gross)) if gross else 0.0,
        cash_weight_mean=float(np.mean(cash)) if cash else 0.0,
        cvar_trigger_rate=float(np.mean(cvar_trig)) if cvar_trig else 0.0,
        avg_haircut_rounds=float(np.mean(haircut_rounds)) if haircut_rounds else 0.0,
    )


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_table(summaries: list[ConfigSummary]) -> str:
    header = (
        f"{'config':<22s} {'h_pre':>6s} {'h_post':>6s} {'h_min':>5s} {'h_max':>5s} "
        f"{'gross':>7s} {'cash':>6s} {'net_ex':>8s} {'IR':>6s} {'max_DD':>7s} "
        f"{'turn':>6s} {'cvar%':>6s}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for s in summaries:
        r = s.truth_row
        lines.append(
            f"{s.name:<22s} {s.holdings_pre_risk_mean:>6.1f} {s.holdings_final_mean:>6.1f} "
            f"{s.holdings_final_min:>5d} {s.holdings_final_max:>5d} "
            f"{s.gross_exposure_mean:>7.1%} {s.cash_weight_mean:>6.1%} "
            f"{r['net_ann_excess']:>8.2%} {r['ir']:>6.2f} {r['max_drawdown']:>7.2%} "
            f"{r['avg_turnover']:>6.2%} {s.cvar_trigger_rate:>6.1%}"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_verdict(summaries: list[ConfigSummary]) -> str:
    baseline = next((s for s in summaries if s.name == "baseline"), None)
    base_excess = baseline.truth_row["net_ann_excess"] if baseline else None
    base_ir = baseline.truth_row["ir"] if baseline else None
    base_dd = baseline.truth_row["max_drawdown"] if baseline else None

    md = ["# W13 Holdings Sweep Verdict", ""]
    md.append(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    md.append(f"**Baseline:** {baseline.name if baseline else 'N/A'}")
    if baseline:
        md.append(
            f"**Baseline metrics:** net excess {base_excess:.2%}, IR {base_ir:.2f}, "
            f"max DD {base_dd:.2%}, mean holdings {baseline.holdings_final_mean:.0f}"
        )
    md.append("")
    md.append("## Comparison Table")
    md.append("")
    md.append("| Config | Mean Holdings | Net Excess (Δ vs base) | IR (Δ) | Max DD (Δ) | Turnover | CVaR Hit | Final Cash |")
    md.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        r = s.truth_row
        if baseline and s.name != baseline.name:
            d_excess = r["net_ann_excess"] - base_excess
            d_ir = r["ir"] - base_ir
            d_dd = r["max_drawdown"] - base_dd
            d_excess_str = f" ({d_excess:+.2%})"
            d_ir_str = f" ({d_ir:+.2f})"
            d_dd_str = f" ({d_dd:+.2%})"
        else:
            d_excess_str = d_ir_str = d_dd_str = ""
        md.append(
            f"| `{s.name}` | {s.holdings_final_mean:.0f} (post) "
            f"/ {s.holdings_pre_risk_mean:.0f} (pre) "
            f"| {r['net_ann_excess']:.2%}{d_excess_str} "
            f"| {r['ir']:.2f}{d_ir_str} "
            f"| {r['max_drawdown']:.2%}{d_dd_str} "
            f"| {r['avg_turnover']:.2%} "
            f"| {s.cvar_trigger_rate:.1%} "
            f"| {s.cash_weight_mean:.1%} |"
        )
    md.append("")
    md.append("## Key Decompositions")
    md.append("")
    if baseline:
        layer3_only = next((s for s in summaries if s.name == "layer3_only"), None)
        cap30 = next((s for s in summaries if s.name == "cap30_only"), None)
        cap30_l3 = next((s for s in summaries if s.name == "cap30_plus_layer3"), None)
        if layer3_only:
            md.append(
                f"- **Layer 3 alone** moves net excess by "
                f"{layer3_only.truth_row['net_ann_excess'] - base_excess:+.2%} and IR by "
                f"{layer3_only.truth_row['ir'] - base_ir:+.2f}."
            )
        if cap30:
            md.append(
                f"- **30-stock cap alone** moves net excess by "
                f"{cap30.truth_row['net_ann_excess'] - base_excess:+.2%} and IR by "
                f"{cap30.truth_row['ir'] - base_ir:+.2f}."
            )
        if cap30_l3:
            md.append(
                f"- **30-stock cap + Layer 3** (live-style) moves net excess by "
                f"{cap30_l3.truth_row['net_ann_excess'] - base_excess:+.2%} and IR by "
                f"{cap30_l3.truth_row['ir'] - base_ir:+.2f}."
            )
    md.append("")
    md.append("## Notes")
    md.append("- Baseline reproduces W10 truth-table `score_weighted_buffered` at cost_mult=1.0, gate_off.")
    md.append("- Layer 3 uses PIT-correct trailing returns (each period sees only history up to execution_date).")
    md.append("- CVaR config: 99% confidence, floor -5%, 20% gross haircut, max 3 rounds.")
    md.append("- Layer 3 sector deviation cap = 15% (research default; live W12 ran tighter ~9%).")
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    report_path = REPO_ROOT / args.report
    payload = load_report(report_path)
    horizon_days = parse_horizon_days(payload)
    windows = select_report_windows(payload, limit=args.window_limit)
    as_of = parse_date(str(payload["as_of"]))
    benchmark_ticker = str(
        payload.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER)
    ).upper()
    rebalance_weekday = int(
        payload.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY)
    )

    logger.info("Stage 1: rebuilding Ridge predictions across {} windows", len(windows))
    artifacts = prepare_horizon_artifacts(
        label=f"{horizon_days}D",
        horizon_days=horizon_days,
        report_path=report_path,
        report_payload=payload,
        windows=windows,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=REPO_ROOT / args.feature_matrix_path,
        label_cache_path=REPO_ROOT / args.label_path,
        as_of=as_of,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )
    predictions = build_predictions(
        windows=windows, artifacts=artifacts, rebalance_weekday=rebalance_weekday,
    )
    logger.info("predictions: {} rows × {} dates × {} tickers",
                len(predictions),
                predictions.index.get_level_values("trade_date").nunique(),
                predictions.index.get_level_values("ticker").nunique())

    logger.info("Stage 2: loading PIT prices")
    signal_dates = pd.DatetimeIndex(predictions.index.get_level_values("trade_date")).sort_values().unique()
    tickers = sorted(set(predictions.index.get_level_values("ticker").astype(str).tolist()) | {benchmark_ticker})
    price_start = pd.Timestamp(signal_dates.min()).date()
    price_end = (pd.Timestamp(signal_dates.max()) + pd.Timedelta(days=args.price_buffer_days)).date()
    prices = get_prices_pit(tickers=tickers, start_date=price_start, end_date=price_end, as_of=as_of)
    if prices.empty:
        raise RuntimeError("No PIT prices returned for sweep.")

    execution_frame = prepare_execution_price_frame(prices)
    return_panel = build_return_panel(prices)
    logger.info("execution frame: {} rows; return panel: {} dates × {} tickers",
                len(execution_frame), len(return_panel), return_panel.shape[1])

    logger.info("Stage 3: loading sector map")
    sector_map = load_sector_map()
    logger.info("sector_map: {} tickers", len(sector_map))

    cost_model = AlmgrenChrissCostModel()
    if abs(args.cost_mult - 1.0) > 1e-9:
        cost_model = AlmgrenChrissCostModel(eta=cost_model.eta * args.cost_mult,
                                            gamma=cost_model.gamma * args.cost_mult)

    all_configs = SWEEP_CONFIGS + LAYER3_TUNING_CONFIGS
    logger.info("Stage 4: running {} configs", len(all_configs))
    summaries: list[ConfigSummary] = []
    for cfg in all_configs:
        logger.info("=== config: {} (max_holdings={}, layer3={}) ===",
                    cfg["name"], cfg["max_holdings"], cfg["layer3"])
        result, audits = simulate_with_cap_and_layer3(
            predictions=predictions,
            execution=execution_frame,
            cost_model=cost_model,
            return_panel=return_panel,
            sector_map=sector_map,
            benchmark_ticker=benchmark_ticker,
            max_holdings=cfg["max_holdings"],
            layer3=cfg["layer3"],
            layer3_kwargs=cfg.get("layer3_kwargs"),
        )
        summary = aggregate_config(cfg["name"], result, audits)
        logger.info(
            "  → n_periods={}, mean_holdings_post={:.1f}, net_excess={:.2%}, IR={:.2f}, max_DD={:.2%}",
            summary.truth_row["n_periods"],
            summary.holdings_final_mean,
            summary.truth_row["net_ann_excess"],
            summary.truth_row["ir"],
            summary.truth_row["max_drawdown"],
        )
        summaries.append(summary)

    table = format_table(summaries)
    print()
    print(table)
    print()

    out_path = (REPO_ROOT / args.out) if args.out else (
        REPO_ROOT / f"data/reports/w13_holdings_sweep_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%SZ')}.json"
    )
    out_payload = {
        "as_of": as_of.isoformat(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "report_path": str(report_path.relative_to(REPO_ROOT)),
        "horizon_days": int(horizon_days),
        "n_windows": len(windows),
        "n_signal_dates": int(len(signal_dates)),
        "champion_params": CHAMPION_PARAMS,
        "layer3_defaults": LAYER3_DEFAULTS,
        "configs": [
            {
                "name": s.name,
                **s.truth_row,
                "holdings_pre_risk_mean": s.holdings_pre_risk_mean,
                "holdings_final_mean": s.holdings_final_mean,
                "holdings_final_min": s.holdings_final_min,
                "holdings_final_max": s.holdings_final_max,
                "gross_exposure_mean": s.gross_exposure_mean,
                "cash_weight_mean": s.cash_weight_mean,
                "cvar_trigger_rate": s.cvar_trigger_rate,
                "avg_haircut_rounds": s.avg_haircut_rounds,
            }
            for s in summaries
        ],
        "ascii_table": table,
    }
    out_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True, default=str))
    logger.info("wrote {}", out_path)

    verdict_path = out_path.with_name(out_path.stem.replace("w13_holdings_sweep_", "W13_holdings_sweep_verdict_") + ".md")
    verdict_path.write_text(format_verdict(summaries))
    logger.info("wrote {}", verdict_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
