from __future__ import annotations

"""W10 production-style post-cost truth table.

Tests 4 portfolio strategies × 3 Almgren-Chriss cost sensitivity bands ×
{with, without} Rule B regime gate (dispersion<0.10 AND credit_spread<-1.5)
on 60D Ridge predictions from the W9.1 13-window walk-forward report.

Strategies:
  - equal_weight_top_decile
  - vol_inverse_buffered
  - score_weighted_buffered
  - black_litterman_buffered

Output: data/reports/w10_truth_table_60d.json + console truth table.
Verdict (triple gate): net_ann_excess > 5% AND IR > 0.5 AND max_drawdown < 25%.
"""

import argparse
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
from src.backtest.engine import build_universe_by_date  # noqa: E402
from src.backtest.execution import (  # noqa: E402
    PortfolioBacktestResult,
    prepare_execution_price_frame,
    simulate_portfolio,
    simulate_top_decile_portfolio,
)
from src.data.db.pit import get_prices_pit  # noqa: E402

DEFAULT_REPORT_60D = "data/reports/walkforward_v9full9y_60d_ridge_13w.json"
DEFAULT_FEATURE_MATRIX_60D = "data/features/walkforward_v9full9y_fm_60d.parquet"
DEFAULT_LABEL_60D = "data/labels/forward_returns_60d_v9full9y.parquet"
DEFAULT_REGIME_TABLE = "data/features/regime_table_2016-2025.parquet"
DEFAULT_OUTPUT = "data/reports/w10_truth_table_60d.json"

DEFAULT_ETA = 0.426
DEFAULT_GAMMA = 0.942
COST_MULTIPLIERS = (0.75, 1.0, 1.25)

REGIME_DISPERSION_THRESHOLD = 0.10
REGIME_CREDIT_SPREAD_THRESHOLD = -1.5
REGIME_GATE_EXPOSURE = 0.5

VERDICT_NET_EXCESS_MIN = 0.05
VERDICT_IR_MIN = 0.5
VERDICT_DRAWDOWN_MAX = 0.25


STRATEGY_CONFIGS: dict[str, dict[str, Any]] = {
    "equal_weight_top_decile": {
        "engine": "simulate_top_decile_portfolio",
        "params": {},
    },
    "vol_inverse_buffered": {
        "engine": "simulate_portfolio",
        "params": {
            "weighting_scheme": "vol_inverse",
            "selection_pct": 0.20,
            "sell_buffer_pct": 0.25,
            "min_trade_weight": 0.01,
            "max_weight": 0.05,
            "min_holdings": 20,
        },
    },
    "score_weighted_buffered": {
        "engine": "simulate_score_weighted_controlled",
        "params": {
            "selection_pct": 0.20,
            "sell_buffer_pct": 0.25,
            "min_trade_weight": 0.01,
            "max_weight": 0.05,
            "min_holdings": 20,
            "weight_shrinkage": 0.0,
            "no_trade_zone": 0.0,
            "turnover_penalty_lambda": 0.0,
        },
    },
    "black_litterman_buffered": {
        "engine": "simulate_portfolio",
        "params": {
            "weighting_scheme": "black_litterman",
            "selection_pct": 0.20,
            "sell_buffer_pct": 0.25,
            "min_trade_weight": 0.01,
            "max_weight": 0.10,
            "min_holdings": 20,
            "bl_lookback_days": 60,
        },
    },
}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    report_path = REPO_ROOT / args.report
    payload = load_report(report_path)
    horizon_days = parse_horizon_days(payload)
    windows = select_report_windows(payload, limit=args.window_limit)

    as_of = parse_date(args.as_of) if args.as_of else parse_date(str(payload["as_of"]))
    benchmark_ticker = (
        args.benchmark_ticker
        if args.benchmark_ticker
        else str(payload.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER))
    ).upper()
    rebalance_weekday = (
        int(args.rebalance_weekday) if args.rebalance_weekday is not None
        else int(payload.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY))
    )

    logger.info("Stage 1: rebuilding Ridge predictions across {} windows", len(windows))
    artifacts = prepare_horizon_artifacts(
        label="60D",
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
    logger.info(
        "predictions: {} rows, {} dates, {} tickers",
        len(predictions),
        predictions.index.get_level_values("trade_date").nunique(),
        predictions.index.get_level_values("ticker").nunique(),
    )

    logger.info("Stage 2: loading PIT prices")
    signal_dates = pd.DatetimeIndex(predictions.index.get_level_values("trade_date")).sort_values().unique()
    tickers = sorted(set(predictions.index.get_level_values("ticker").astype(str).tolist()) | {benchmark_ticker})
    price_start = pd.Timestamp(signal_dates.min()).date()
    price_end = (pd.Timestamp(signal_dates.max()) + pd.Timedelta(days=args.price_buffer_days)).date()
    prices = get_prices_pit(tickers=tickers, start_date=price_start, end_date=price_end, as_of=as_of)
    if prices.empty:
        raise RuntimeError("No PIT prices returned for backtest.")
    universe_by_date = build_universe_by_date(trade_dates=signal_dates, index_name="SP500")
    execution_frame = prepare_execution_price_frame(prices)

    logger.info("Stage 3: loading regime table")
    regime_gate = load_regime_gate(REPO_ROOT / args.regime_table)

    logger.info(
        "Stage 4: running {} strategies × {} cost bands × 2 gate states",
        len(STRATEGY_CONFIGS),
        len(COST_MULTIPLIERS),
    )
    rows: list[dict[str, Any]] = []
    for strategy_name in STRATEGY_CONFIGS:
        for cost_mult in COST_MULTIPLIERS:
            cost_model = AlmgrenChrissCostModel(
                eta=DEFAULT_ETA * cost_mult,
                gamma=DEFAULT_GAMMA * cost_mult,
            )
            logger.info("strategy={} cost_mult={:.2f}", strategy_name, cost_mult)
            portfolio = run_strategy(
                name=strategy_name,
                config=STRATEGY_CONFIGS[strategy_name],
                predictions=predictions,
                prices=prices,
                execution_frame=execution_frame,
                cost_model=cost_model,
                benchmark_ticker=benchmark_ticker,
                universe_by_date=universe_by_date,
            )
            for gate_on in (False, True):
                row = build_truth_row(
                    strategy=strategy_name,
                    cost_mult=cost_mult,
                    gate_on=gate_on,
                    portfolio=portfolio,
                    regime_gate=regime_gate,
                )
                rows.append(row)

    truth_table = pd.DataFrame(rows)
    verdict = build_w10_verdict(truth_table)

    output = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "as_of": as_of.isoformat(),
        "horizon_days": int(horizon_days),
        "report_path": str(report_path),
        "benchmark_ticker": benchmark_ticker,
        "cost_model_base": {"eta": DEFAULT_ETA, "gamma": DEFAULT_GAMMA},
        "cost_multipliers": list(COST_MULTIPLIERS),
        "regime_gate_rule": (
            f"universe_return_dispersion_20d<{REGIME_DISPERSION_THRESHOLD} "
            f"AND credit_spread_baa10y<{REGIME_CREDIT_SPREAD_THRESHOLD} "
            f"=> exposure × {REGIME_GATE_EXPOSURE}"
        ),
        "verdict_thresholds": {
            "net_ann_excess_min": VERDICT_NET_EXCESS_MIN,
            "ir_min": VERDICT_IR_MIN,
            "max_drawdown_max": VERDICT_DRAWDOWN_MAX,
        },
        "rows": rows,
        "verdict": verdict,
    }
    output_path = REPO_ROOT / args.output
    write_json_atomic(output_path, json_safe(output))
    logger.info("saved truth table report to {}", output_path)
    print_truth_table(truth_table, verdict)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--report", default=DEFAULT_REPORT_60D)
    parser.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_60D)
    parser.add_argument("--label-path", default=DEFAULT_LABEL_60D)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--regime-table", default=DEFAULT_REGIME_TABLE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--as-of")
    parser.add_argument("--benchmark-ticker")
    parser.add_argument("--rebalance-weekday", type=int)
    parser.add_argument("--window-limit", type=int)
    parser.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    parser.add_argument("--price-buffer-days", type=int, default=7)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level:<5} | {message}")


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward report not found: {path}")
    payload = json.loads(path.read_text())
    if "windows" not in payload:
        raise RuntimeError(f"Walk-forward report missing windows: {path}")
    return payload


def build_predictions(
    *,
    windows: list[dict[str, Any]],
    artifacts: Any,
    rebalance_weekday: int,
) -> pd.Series:
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


def load_regime_gate(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.sort_values("trade_date").set_index("trade_date")
    gate = (
        (df["universe_return_dispersion_20d"] < REGIME_DISPERSION_THRESHOLD)
        & (df["credit_spread_baa10y"] < REGIME_CREDIT_SPREAD_THRESHOLD)
    )
    return gate.astype(bool)


def run_strategy(
    *,
    name: str,
    config: dict[str, Any],
    predictions: pd.Series,
    prices: pd.DataFrame,
    execution_frame: pd.DataFrame,
    cost_model: AlmgrenChrissCostModel,
    benchmark_ticker: str,
    universe_by_date: dict[pd.Timestamp, set[str]] | None,
) -> PortfolioBacktestResult:
    engine = config["engine"]
    params = dict(config["params"])
    if engine == "simulate_top_decile_portfolio":
        return simulate_top_decile_portfolio(
            predictions=predictions,
            prices=prices,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
            universe_by_date=universe_by_date,
        )
    if engine == "simulate_portfolio":
        return simulate_portfolio(
            predictions=predictions,
            prices=prices,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
            universe_by_date=universe_by_date,
            **params,
        )
    if engine == "simulate_score_weighted_controlled":
        return simulate_score_weighted_controlled(
            predictions=predictions,
            execution=execution_frame,
            cost_model=cost_model,
            benchmark_ticker=benchmark_ticker,
            **params,
        )
    raise ValueError(f"Unknown engine: {engine}")


def build_truth_row(
    *,
    strategy: str,
    cost_mult: float,
    gate_on: bool,
    portfolio: PortfolioBacktestResult,
    regime_gate: pd.Series,
) -> dict[str, Any]:
    if not portfolio.periods:
        return {
            "strategy": strategy,
            "cost_mult": float(cost_mult),
            "gate_on": bool(gate_on),
            "n_periods": 0,
            "gross_ann_excess": float("nan"),
            "net_ann_excess": float("nan"),
            "cost_drag_ann": 0.0,
            "avg_turnover": 0.0,
            "sharpe": float("nan"),
            "ir": float("nan"),
            "max_drawdown": float("nan"),
        }

    periods = pd.DataFrame([p.to_dict() for p in portfolio.periods])
    periods["signal_date"] = pd.to_datetime(periods["signal_date"])
    periods["execution_date"] = pd.to_datetime(periods["execution_date"])
    periods["exit_date"] = pd.to_datetime(periods["exit_date"])
    periods = periods.sort_values("execution_date").reset_index(drop=True)

    if gate_on:
        gate_lookup = regime_gate.reindex(periods["signal_date"], method="ffill").fillna(False)
        gate_mask = gate_lookup.to_numpy(dtype=bool)
        if gate_mask.any():
            scaled_gross = np.where(gate_mask, periods["gross_return"] * REGIME_GATE_EXPOSURE, periods["gross_return"])
            scaled_cost = np.where(gate_mask, periods["cost_rate"] * REGIME_GATE_EXPOSURE, periods["cost_rate"])
            scaled_turnover = np.where(gate_mask, periods["turnover"] * REGIME_GATE_EXPOSURE, periods["turnover"])
            scaled_net = (1.0 - scaled_cost) * (1.0 + scaled_gross) - 1.0
            periods["gross_return"] = scaled_gross
            periods["cost_rate"] = scaled_cost
            periods["turnover"] = scaled_turnover
            periods["net_return"] = scaled_net
            periods["gross_excess_return"] = periods["gross_return"] - periods["benchmark_return"]
            periods["net_excess_return"] = periods["net_return"] - periods["benchmark_return"]

    return aggregate_truth_row(
        strategy=strategy,
        cost_mult=cost_mult,
        gate_on=gate_on,
        periods=periods,
    )


def aggregate_truth_row(
    *,
    strategy: str,
    cost_mult: float,
    gate_on: bool,
    periods: pd.DataFrame,
) -> dict[str, Any]:
    n_periods = int(len(periods))
    total_days = max((periods["exit_date"].iloc[-1] - periods["execution_date"].iloc[0]).days, 1)
    periods_per_year = float(n_periods * 365.25 / total_days) if total_days > 0 else 0.0

    cum_gross = float((1.0 + periods["gross_return"]).prod() - 1.0)
    cum_net = float((1.0 + periods["net_return"]).prod() - 1.0)
    cum_bench = float((1.0 + periods["benchmark_return"]).prod() - 1.0)

    ann_gross = _annualize(cum_gross, total_days)
    ann_net = _annualize(cum_net, total_days)
    ann_bench = _annualize(cum_bench, total_days)

    net_excess_per = periods["net_return"] - periods["benchmark_return"]
    if n_periods > 1 and net_excess_per.std() > 0 and periods_per_year > 0:
        ir = float(net_excess_per.mean() / net_excess_per.std() * np.sqrt(periods_per_year))
        sharpe = float(periods["net_return"].mean() / periods["net_return"].std() * np.sqrt(periods_per_year))
    else:
        ir = float("nan")
        sharpe = float("nan")

    excess_wealth = (1.0 + net_excess_per).cumprod()
    running_peak = excess_wealth.cummax()
    safe_peak = running_peak.where(running_peak > 0, np.nan)
    drawdown = ((safe_peak - excess_wealth) / safe_peak).fillna(0.0)
    max_dd = float(drawdown.max())

    return {
        "strategy": strategy,
        "cost_mult": float(cost_mult),
        "gate_on": bool(gate_on),
        "n_periods": n_periods,
        "gross_ann_excess": float(ann_gross - ann_bench),
        "net_ann_excess": float(ann_net - ann_bench),
        "cost_drag_ann": float(ann_gross - ann_net),
        "avg_turnover": float(periods["turnover"].mean()),
        "sharpe": sharpe,
        "ir": ir,
        "max_drawdown": max_dd,
    }


def _annualize(total_return: float, total_days: int) -> float:
    base = 1.0 + float(total_return)
    if base <= 0.0:
        return -1.0
    return float(base ** (365.25 / max(total_days, 1)) - 1.0)


def build_w10_verdict(truth_table: pd.DataFrame) -> dict[str, Any]:
    pass_mask = (
        (truth_table["net_ann_excess"] > VERDICT_NET_EXCESS_MIN)
        & (truth_table["ir"] > VERDICT_IR_MIN)
        & (truth_table["max_drawdown"] < VERDICT_DRAWDOWN_MAX)
    )
    passing = truth_table[pass_mask].copy()
    if not passing.empty:
        best = passing.sort_values("net_ann_excess", ascending=False).iloc[0].to_dict()
        return {
            "passed": True,
            "n_passing_combos": int(len(passing)),
            "champion": _serialize_row(best),
        }
    closest = truth_table.sort_values("net_ann_excess", ascending=False).iloc[0].to_dict()
    return {
        "passed": False,
        "n_passing_combos": 0,
        "best_attempt": _serialize_row(closest),
    }


def _serialize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in row.items():
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            out[key] = value.item()
        else:
            out[key] = value
    return out


def print_truth_table(truth_table: pd.DataFrame, verdict: dict[str, Any]) -> None:
    print("\n=== W10 Truth Table (60D Ridge) ===")
    cols = ["strategy", "cost_mult", "gate_on", "n_periods",
            "gross_ann_excess", "net_ann_excess", "cost_drag_ann",
            "avg_turnover", "sharpe", "ir", "max_drawdown"]
    display = truth_table[cols].copy()
    for c in ("gross_ann_excess", "net_ann_excess", "cost_drag_ann",
              "avg_turnover", "sharpe", "ir", "max_drawdown"):
        display[c] = display[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "nan")
    display["cost_mult"] = display["cost_mult"].map(lambda x: f"{x:.2f}")
    print(display.to_string(index=False))
    print(f"\n=== Verdict: {'PASS' if verdict['passed'] else 'FAIL'} ===")
    print(f"  thresholds: net_ann_excess > {VERDICT_NET_EXCESS_MIN:.2f}, "
          f"IR > {VERDICT_IR_MIN:.2f}, max_drawdown < {VERDICT_DRAWDOWN_MAX:.2f}")
    if verdict.get("passed"):
        c = verdict["champion"]
        print(f"  champion: {c['strategy']} | cost_mult={c['cost_mult']:.2f} | gate_on={c['gate_on']}")
        print(f"            net_ann_excess={c['net_ann_excess']:.4f} "
              f"IR={c['ir']:.4f} max_dd={c['max_drawdown']:.4f}")
        print(f"  n_passing_combos: {verdict['n_passing_combos']}")
    else:
        c = verdict["best_attempt"]
        print(f"  best_attempt: {c['strategy']} | cost_mult={c['cost_mult']:.2f} | gate_on={c['gate_on']}")
        print(f"                net_ann_excess={c['net_ann_excess']:.4f} "
              f"IR={c['ir']:.4f} max_dd={c['max_drawdown']:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
