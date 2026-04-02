from __future__ import annotations

import argparse
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
from src.data.db.session import get_engine
from src.risk.portfolio_risk import (
    PortfolioRiskEngine,
    compute_portfolio_beta,
    compute_portfolio_cvar,
    compute_sector_weights,
    sector_weight_deviation,
)

EXPECTED_BRANCH = "feature/week13-risk-engine"
TARGET_WINDOW = "W6"
TARGET_SECTOR = "Financial Services"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}.")

    buffered_report = load_json(REPO_ROOT / args.portfolio_report_path)
    predictions = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices = pd.read_parquet(REPO_ROOT / args.prices_path)
    sector_map = load_sector_map()

    validation_case = build_validation_case(
        buffered_report=buffered_report,
        predictions=predictions,
        prices=prices,
        sector_map=sector_map,
        target_window=args.window_id,
        target_sector=args.target_sector,
        sector_overweight_buffer=args.synthetic_sector_overweight_buffer,
    )

    engine = PortfolioRiskEngine()
    constrained = engine.apply_all_constraints(
        weights=validation_case["synthetic_weights"],
        benchmark_weights=validation_case["benchmark_weights"],
        sector_map=sector_map,
        return_history=validation_case["return_history"],
        spy_returns=validation_case["spy_returns"],
        current_weights=validation_case["synthetic_weights"],
        candidate_ranking=validation_case["candidate_ranking"],
        max_sector_deviation=args.max_sector_deviation,
    )

    benchmark_sector_weights = validation_case["benchmark_sector_weights"]
    before_sector_weights = validation_case["synthetic_sector_weights"]
    after_sector_weights = constrained.sector_weights
    before_deviation = sector_weight_deviation(before_sector_weights, benchmark_sector_weights)
    after_deviation = sector_weight_deviation(after_sector_weights, benchmark_sector_weights)

    before_target_sector_deviation = float(before_deviation.get(args.target_sector, 0.0))
    after_target_sector_deviation = float(after_deviation.get(args.target_sector, 0.0))
    if before_target_sector_deviation <= args.max_sector_deviation:
        raise RuntimeError(
            f"Synthetic validation case did not breach the sector limit: "
            f"{before_target_sector_deviation:.4f} <= {args.max_sector_deviation:.4f}."
        )
    if after_target_sector_deviation > args.max_sector_deviation + 1e-9:
        raise RuntimeError(
            f"Sector constraint failed to correct {args.target_sector}: "
            f"{after_target_sector_deviation:.4f} > {args.max_sector_deviation:.4f}."
        )

    expected_rule_order = [
        ("P0", "single_stock_cap"),
        ("P0", "sector_deviation_cap"),
        ("P0", "cvar_haircut"),
        ("P1", "beta_alignment"),
        ("P1", "correlation_dedup"),
        ("P1", "turnover_cap"),
        ("P2", "minimum_holdings"),
        ("P2", "stress_test"),
    ]
    actual_rule_order = [(entry.priority, entry.rule_name) for entry in constrained.audit_trail]
    if actual_rule_order != expected_rule_order:
        raise RuntimeError(f"Unexpected rule order: {actual_rule_order}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": Path(__file__).name,
        "window_id": args.window_id,
        "signal_date": validation_case["signal_date"].date().isoformat(),
        "target_sector": args.target_sector,
        "benchmark_source": "equal_weight_prediction_universe_on_signal_date",
        "example_method": validation_case["example_method"],
        "actual_w6_selected_sector_weights": validation_case["actual_sector_weights"],
        "synthetic_before": {
            "weights": validation_case["synthetic_weights"],
            "sector_weights": before_sector_weights,
            "sector_deviation": before_deviation,
            "portfolio_beta": compute_portfolio_beta(validation_case["synthetic_weights"], validation_case["beta_lookup"]),
            "cvar_99": compute_portfolio_cvar(
                validation_case["synthetic_weights"],
                return_history=validation_case["return_history"],
                confidence=0.99,
            ),
        },
        "benchmark": {
            "sector_weights": benchmark_sector_weights,
            "universe_size": validation_case["universe_size"],
        },
        "after_constraints": constrained.to_dict(),
        "sector_validation": {
            "limit": float(args.max_sector_deviation),
            "before_target_sector_deviation": before_target_sector_deviation,
            "after_target_sector_deviation": after_target_sector_deviation,
            "corrected": True,
        },
        "audit_rule_order": actual_rule_order,
        "inputs": {
            "portfolio_report_path": str(REPO_ROOT / args.portfolio_report_path),
            "predictions_path": str(REPO_ROOT / args.predictions_path),
            "prices_path": str(REPO_ROOT / args.prices_path),
        },
    }

    output_path = REPO_ROOT / args.output_path
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved risk engine validation report to {}", output_path)
    logger.info(
        "validated {} deviation {:.4f} -> {:.4f}",
        args.target_sector,
        before_target_sector_deviation,
        after_target_sector_deviation,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the ordered portfolio risk engine on a W6-style sector concentration case.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--portfolio-report-path", default="data/reports/extended_portfolio_buffered.json")
    parser.add_argument("--predictions-path", default="data/backtest/extended_walkforward_predictions.parquet")
    parser.add_argument("--prices-path", default="data/backtest/extended_walkforward_prices.parquet")
    parser.add_argument("--output-path", default="data/reports/risk_engine_validation.json")
    parser.add_argument("--window-id", default=TARGET_WINDOW)
    parser.add_argument("--target-sector", default=TARGET_SECTOR)
    parser.add_argument("--max-sector-deviation", type=float, default=0.15)
    parser.add_argument("--synthetic-sector-overweight-buffer", type=float, default=0.20)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_sector_map() -> dict[str, str]:
    engine = get_engine()
    with engine.connect() as conn:
        stocks = pd.read_sql(text("select ticker, sector from stocks"), conn)
    stocks["ticker"] = stocks["ticker"].astype(str).str.upper()
    stocks["sector"] = stocks["sector"].fillna("Unknown").astype(str)
    return stocks.set_index("ticker")["sector"].to_dict()


def build_validation_case(
    *,
    buffered_report: dict[str, Any],
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    sector_map: dict[str, str],
    target_window: str,
    target_sector: str,
    sector_overweight_buffer: float,
) -> dict[str, Any]:
    predictions = predictions.copy()
    predictions["window_id"] = predictions["window_id"].astype(str)
    predictions["ticker"] = predictions["ticker"].astype(str).str.upper()
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])

    prices = prices.copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")

    window = next(
        entry
        for entry in buffered_report["schemes"]["equal_weight_buffered"]["windows"]
        if entry["window_id"] == target_window
    )
    periods = pd.DataFrame(window["portfolio"]["periods"])
    periods["signal_date"] = pd.to_datetime(periods["signal_date"])

    ranked_periods = []
    for period in periods.itertuples(index=False):
        signal_date = pd.Timestamp(period.signal_date)
        universe_scores = predictions.loc[
            (predictions["window_id"] == target_window) & (predictions["trade_date"] == signal_date)
        ].copy()
        if universe_scores.empty:
            continue

        selected_tickers = [str(ticker).upper() for ticker in period.selected_tickers]
        actual_weights = {ticker: 1.0 / len(selected_tickers) for ticker in selected_tickers}
        benchmark_weights = {
            str(ticker).upper(): 1.0 / len(universe_scores)
            for ticker in universe_scores["ticker"].tolist()
        }
        benchmark_sector_weights = compute_sector_weights(benchmark_weights, sector_map)
        actual_sector_weights = compute_sector_weights(actual_weights, sector_map)
        deviation = sector_weight_deviation(actual_sector_weights, benchmark_sector_weights)
        ranked_periods.append(
            {
                "signal_date": signal_date,
                "selected_tickers": selected_tickers,
                "universe_scores": universe_scores,
                "benchmark_weights": benchmark_weights,
                "benchmark_sector_weights": benchmark_sector_weights,
                "actual_sector_weights": actual_sector_weights,
                "target_sector_deviation": float(deviation.get(target_sector, 0.0)),
            },
        )

    if not ranked_periods:
        raise RuntimeError(f"No usable periods found for window {target_window}.")

    chosen = max(ranked_periods, key=lambda row: row["target_sector_deviation"])
    score_frame = (
        chosen["universe_scores"]
        .set_index("ticker")["score"]
        .astype(float)
        .sort_values(ascending=False)
    )
    synthetic_weights = build_synthetic_sector_overweight_portfolio(
        scores=score_frame,
        selected_tickers=chosen["selected_tickers"],
        sector_map=sector_map,
        benchmark_sector_weight=float(chosen["benchmark_sector_weights"].get(target_sector, 0.0)),
        target_sector=target_sector,
        overweight_buffer=sector_overweight_buffer,
    )

    return_history, spy_returns = build_return_history(
        prices=prices,
        signal_date=chosen["signal_date"],
        tickers=sorted(set(score_frame.index) | set(chosen["selected_tickers"])),
    )
    beta_lookup = compute_beta_lookup(return_history=return_history, spy_returns=spy_returns)

    return {
        "signal_date": chosen["signal_date"],
        "candidate_ranking": score_frame.index.tolist(),
        "synthetic_weights": synthetic_weights,
        "synthetic_sector_weights": compute_sector_weights(synthetic_weights, sector_map),
        "actual_sector_weights": chosen["actual_sector_weights"],
        "benchmark_weights": chosen["benchmark_weights"],
        "benchmark_sector_weights": chosen["benchmark_sector_weights"],
        "return_history": return_history,
        "spy_returns": spy_returns,
        "beta_lookup": beta_lookup,
        "universe_size": int(len(chosen["benchmark_weights"])),
        "example_method": (
            f"Start from the W6 rebalance with the largest {target_sector} tilt and "
            f"reallocate the selected basket so {target_sector} carries benchmark+{sector_overweight_buffer:.0%} weight."
        ),
    }


def build_synthetic_sector_overweight_portfolio(
    *,
    scores: pd.Series,
    selected_tickers: list[str],
    sector_map: dict[str, str],
    benchmark_sector_weight: float,
    target_sector: str,
    overweight_buffer: float,
) -> dict[str, float]:
    selected_scores = scores.reindex(selected_tickers).dropna()
    if selected_scores.empty:
        raise RuntimeError("Selected basket does not overlap the signal scores.")

    sector_tickers = [ticker for ticker in selected_scores.index if sector_map.get(ticker, "Unknown") == target_sector]
    other_tickers = [ticker for ticker in selected_scores.index if ticker not in sector_tickers]
    if not sector_tickers or not other_tickers:
        raise RuntimeError(f"Unable to build a synthetic {target_sector} overweight case.")

    target_sector_weight = min(0.45, max(benchmark_sector_weight + overweight_buffer, 0.30))
    sector_weights = rank_weight_group(selected_scores.reindex(sector_tickers))
    other_weights = rank_weight_group(selected_scores.reindex(other_tickers))

    synthetic = {}
    for ticker, weight in sector_weights.items():
        synthetic[str(ticker)] = float(weight * target_sector_weight)
    for ticker, weight in other_weights.items():
        synthetic[str(ticker)] = float(weight * (1.0 - target_sector_weight))
    return synthetic


def rank_weight_group(scores: pd.Series) -> dict[str, float]:
    ordered = scores.astype(float).sort_values(ascending=False)
    if ordered.empty:
        return {}
    raw = pd.Series(
        np.linspace(len(ordered), 1.0, len(ordered), dtype=float),
        index=ordered.index,
        dtype=float,
    )
    raw /= float(raw.sum())
    return {str(ticker): float(weight) for ticker, weight in raw.items()}


def build_return_history(
    *,
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
    tickers: list[str],
    lookback_days: int = 60,
) -> tuple[pd.DataFrame, pd.Series]:
    required = sorted(set(tickers) | {"SPY"})
    subset = prices.loc[prices["ticker"].isin(required), ["trade_date", "ticker", "adj_close"]].copy()
    if subset.empty:
        raise RuntimeError("Price history subset is empty.")

    close_matrix = (
        subset.pivot_table(index="trade_date", columns="ticker", values="adj_close")
        .sort_index()
    )
    returns = close_matrix.pct_change(fill_method=None).loc[lambda frame: frame.index < signal_date].tail(int(lookback_days))
    if returns.empty:
        raise RuntimeError("No return history available before the validation signal date.")
    spy_returns = returns.get("SPY")
    return returns.drop(columns=["SPY"], errors="ignore"), pd.Series(spy_returns, dtype=float).dropna()


def compute_beta_lookup(
    *,
    return_history: pd.DataFrame,
    spy_returns: pd.Series,
) -> dict[str, float]:
    if return_history.empty or spy_returns.empty:
        return {}

    joined = return_history.join(spy_returns.rename("SPY"), how="inner")
    benchmark_var = float(joined["SPY"].var())
    if benchmark_var <= 1e-12:
        return {}

    beta_lookup: dict[str, float] = {}
    for ticker in return_history.columns:
        aligned = joined[[ticker, "SPY"]].dropna()
        if len(aligned) < 20:
            continue
        covariance = float(aligned[ticker].cov(aligned["SPY"]))
        beta_lookup[str(ticker)] = covariance / benchmark_var
    return beta_lookup


if __name__ == "__main__":
    raise SystemExit(main())
