from __future__ import annotations

"""W10.5 P0-3: Capacity sweep on champion at $1M / $5M / $10M / $25M.

Tests Almgren-Chriss non-linear cost scaling on the W10 champion
(score_weighted_buffered + 60D Ridge alone, no gate, cost_mult=1.0).

Pass gate (per Codex W11_W12_plan):
- At $10M: net_ann_excess > 5%, IR > 0.5, max_drawdown < 25%
- Cost drag increase vs $1M ≤ 150 bps annualized

Output: data/reports/w10_capacity_sweep.json + console table.
"""

import argparse
import json
import math
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

from scripts.run_horizon_fusion import (  # noqa: E402
    extract_ridge_alpha,
    parse_horizon_days,
    prepare_horizon_artifacts,
    rebuild_ridge_predictions,
    select_report_windows,
    slice_all_splits,
)
from scripts.run_ic_screening import write_json_atomic  # noqa: E402
from scripts.run_turnover_optimization import simulate_score_weighted_controlled  # noqa: E402
from scripts.run_walkforward_comparison import (  # noqa: E402
    BENCHMARK_TICKER,
    DEFAULT_ALL_FEATURES_PATH,
    LABEL_BUFFER_DAYS,
    REBALANCE_WEEKDAY,
    json_safe,
    parse_date,
)
from src.backtest.cost_model import AlmgrenChrissCostModel  # noqa: E402
from src.backtest.execution import prepare_execution_price_frame  # noqa: E402
from src.data.db.pit import get_prices_pit  # noqa: E402

DEFAULT_REPORT_60D = "data/reports/walkforward_v9full9y_60d_ridge_13w.json"
DEFAULT_FEATURE_MATRIX_60D = "data/features/walkforward_v9full9y_fm_60d.parquet"
DEFAULT_LABEL_60D = "data/labels/forward_returns_60d_v9full9y.parquet"
DEFAULT_OUTPUT = "data/reports/w10_capacity_sweep.json"

DEFAULT_ETA = 0.426
DEFAULT_GAMMA = 0.942

CAPITAL_SIZES_USD = [1_000_000, 5_000_000, 10_000_000, 25_000_000]

# Champion config (matches W10 truth table champion)
CHAMPION_PARAMS = {
    "selection_pct": 0.20,
    "sell_buffer_pct": 0.25,
    "min_trade_weight": 0.01,
    "max_weight": 0.05,
    "min_holdings": 20,
    "weight_shrinkage": 0.0,
    "no_trade_zone": 0.0,
    "turnover_penalty_lambda": 0.0,
}

# Pass gate (per W11_W12_plan)
GATE_NET_EXCESS_MIN = 0.05
GATE_IR_MIN = 0.5
GATE_DD_MAX = 0.25
GATE_COST_DRAG_INCREASE_MAX = 0.015  # 150 bps


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    report_path = REPO_ROOT / args.report
    payload = json.loads(report_path.read_text())
    horizon_days = parse_horizon_days(payload)
    windows = select_report_windows(payload, limit=args.window_limit)

    as_of = parse_date(str(payload["as_of"]))
    benchmark_ticker = str(payload.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER)).upper()
    rebalance_weekday = int(payload.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY))

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
        windows=windows,
        artifacts=artifacts,
        rebalance_weekday=rebalance_weekday,
    )
    logger.info("predictions: {} rows, {} dates", len(predictions),
                predictions.index.get_level_values("trade_date").nunique())

    logger.info("Stage 2: loading PIT prices")
    signal_dates = pd.DatetimeIndex(predictions.index.get_level_values("trade_date")).sort_values().unique()
    tickers = sorted(set(predictions.index.get_level_values("ticker").astype(str).tolist()) | {benchmark_ticker})
    price_start = pd.Timestamp(signal_dates.min()).date()
    price_end = (pd.Timestamp(signal_dates.max()) + pd.Timedelta(days=args.price_buffer_days)).date()
    prices = get_prices_pit(tickers=tickers, start_date=price_start, end_date=price_end, as_of=as_of)
    if prices.empty:
        raise RuntimeError("No PIT prices returned for backtest.")
    execution_frame = prepare_execution_price_frame(prices)

    cost_model = AlmgrenChrissCostModel(eta=DEFAULT_ETA, gamma=DEFAULT_GAMMA)

    logger.info("Stage 3: capacity sweep across {} sizes", len(CAPITAL_SIZES_USD))
    rows: list[dict[str, Any]] = []
    baseline_cost_drag = None
    for capital in CAPITAL_SIZES_USD:
        logger.info("  initial_capital=${:,.0f}", capital)
        portfolio = simulate_score_weighted_controlled(
            predictions=predictions,
            execution=execution_frame,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
            initial_capital=float(capital),
            **CHAMPION_PARAMS,
        )
        if not portfolio.periods:
            logger.warning("    empty result")
            continue
        row = aggregate_metrics(portfolio, capital)
        if baseline_cost_drag is None:
            baseline_cost_drag = row["cost_drag_ann"]
        row["cost_drag_increase_vs_1m"] = float(row["cost_drag_ann"] - baseline_cost_drag)
        row["pass_net"] = bool(row["net_ann_excess"] > GATE_NET_EXCESS_MIN)
        row["pass_ir"] = bool(row["ir"] > GATE_IR_MIN)
        row["pass_dd"] = bool(row["max_drawdown"] < GATE_DD_MAX)
        row["pass_cost_increase"] = bool(row["cost_drag_increase_vs_1m"] <= GATE_COST_DRAG_INCREASE_MAX)
        row["pass_all"] = bool(row["pass_net"] and row["pass_ir"] and row["pass_dd"] and row["pass_cost_increase"])
        rows.append(row)

    sweep_df = pd.DataFrame(rows)

    # Verdict: identify max passing capital
    max_passing = max((r["initial_capital_usd"] for r in rows if r["pass_all"]), default=0)
    target_10m_passes = next((r["pass_all"] for r in rows if r["initial_capital_usd"] == 10_000_000), False)
    verdict = {
        "passed_at_10m": bool(target_10m_passes),
        "max_passing_capital_usd": int(max_passing),
        "n_passing_capacities": int(sum(r["pass_all"] for r in rows)),
        "thresholds": {
            "net_ann_excess_min": GATE_NET_EXCESS_MIN,
            "ir_min": GATE_IR_MIN,
            "max_drawdown_max": GATE_DD_MAX,
            "cost_drag_increase_max_vs_1m": GATE_COST_DRAG_INCREASE_MAX,
        },
    }

    output = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "as_of": as_of.isoformat(),
        "horizon_days": int(horizon_days),
        "report_path": str(report_path),
        "champion_strategy": "score_weighted_buffered",
        "champion_params": CHAMPION_PARAMS,
        "cost_model": {"eta": DEFAULT_ETA, "gamma": DEFAULT_GAMMA},
        "capital_sizes_usd": CAPITAL_SIZES_USD,
        "rows": rows,
        "verdict": verdict,
    }

    output_path = REPO_ROOT / args.output
    write_json_atomic(output_path, json_safe(output))
    logger.info("saved capacity sweep to {}", output_path)
    print_summary(sweep_df, verdict)
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--report", default=DEFAULT_REPORT_60D)
    p.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_60D)
    p.add_argument("--label-path", default=DEFAULT_LABEL_60D)
    p.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--window-limit", type=int)
    p.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    p.add_argument("--price-buffer-days", type=int, default=7)
    return p.parse_args(argv)


def configure_logging():
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def build_predictions(*, windows, artifacts, rebalance_weekday):
    parts: list[pd.Series] = []
    for window in windows:
        date_keys = ("train_start", "train_end", "validation_start", "validation_end", "test_start", "test_end")
        dates = {key: parse_date(str(window["dates"][key])) for key in date_keys}
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
        parts.append(test_pred.rename("score"))
    return pd.concat(parts).sort_index()


def aggregate_metrics(portfolio, capital: int) -> dict[str, Any]:
    periods = pd.DataFrame([p.to_dict() for p in portfolio.periods])
    periods["execution_date"] = pd.to_datetime(periods["execution_date"])
    periods["exit_date"] = pd.to_datetime(periods["exit_date"])
    periods = periods.sort_values("execution_date").reset_index(drop=True)

    n_periods = len(periods)
    total_days = max((periods["exit_date"].iloc[-1] - periods["execution_date"].iloc[0]).days, 1)
    periods_per_year = float(n_periods * 365.25 / total_days)

    cum_gross = float((1.0 + periods["gross_return"]).prod() - 1.0)
    cum_net = float((1.0 + periods["net_return"]).prod() - 1.0)
    cum_bench = float((1.0 + periods["benchmark_return"]).prod() - 1.0)

    def _annualize(r):
        b = 1.0 + r
        return -1.0 if b <= 0 else b ** (365.25 / max(total_days, 1)) - 1.0

    ann_gross = _annualize(cum_gross)
    ann_net = _annualize(cum_net)
    ann_bench = _annualize(cum_bench)

    net_excess = periods["net_return"] - periods["benchmark_return"]
    if n_periods > 1 and net_excess.std() > 0:
        ir = float(net_excess.mean() / net_excess.std() * math.sqrt(periods_per_year))
        sharpe = float(periods["net_return"].mean() / periods["net_return"].std() * math.sqrt(periods_per_year))
    else:
        ir = float("nan")
        sharpe = float("nan")

    excess_wealth = (1.0 + net_excess).cumprod()
    running_peak = excess_wealth.cummax()
    safe_peak = running_peak.where(running_peak > 0, np.nan)
    drawdown = ((safe_peak - excess_wealth) / safe_peak).fillna(0.0)
    max_dd = float(drawdown.max())

    return {
        "initial_capital_usd": int(capital),
        "n_periods": int(n_periods),
        "gross_ann_excess": float(ann_gross - ann_bench),
        "net_ann_excess": float(ann_net - ann_bench),
        "cost_drag_ann": float(ann_gross - ann_net),
        "avg_turnover": float(periods["turnover"].mean()),
        "ir": ir,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def print_summary(df: pd.DataFrame, verdict: dict) -> None:
    print()
    print("=" * 78)
    print("W10.5 P0-3 — Capacity Sweep on Champion (score_weighted_buffered, 60D Ridge)")
    print("=" * 78)
    cols = ["initial_capital_usd", "gross_ann_excess", "net_ann_excess",
            "cost_drag_ann", "cost_drag_increase_vs_1m",
            "ir", "sharpe", "max_drawdown",
            "pass_net", "pass_ir", "pass_dd", "pass_cost_increase", "pass_all"]
    display = df[cols].copy()
    display["initial_capital_usd"] = display["initial_capital_usd"].map(lambda x: f"${x/1e6:.0f}M")
    for c in ("gross_ann_excess", "net_ann_excess", "cost_drag_ann",
              "cost_drag_increase_vs_1m", "ir", "sharpe", "max_drawdown"):
        display[c] = display[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    print(display.to_string(index=False))
    print()
    print("Pass gates:")
    print(f"  net_ann_excess > {GATE_NET_EXCESS_MIN:.2f}")
    print(f"  IR > {GATE_IR_MIN:.2f}")
    print(f"  max_drawdown < {GATE_DD_MAX:.2f}")
    print(f"  cost_drag_increase_vs_1m ≤ {GATE_COST_DRAG_INCREASE_MAX*1e4:.0f} bps")
    print()
    status = "PASS" if verdict["passed_at_10m"] else "FAIL"
    print(f"=== VERDICT @ $10M: {status} ===")
    print(f"  Max passing capital: ${verdict['max_passing_capital_usd']/1e6:.0f}M")
    print(f"  N passing capacity sizes: {verdict['n_passing_capacities']}/{len(CAPITAL_SIZES_USD)}")


if __name__ == "__main__":
    raise SystemExit(main())
