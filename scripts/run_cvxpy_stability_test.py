from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_portfolio_optimization_comparison import (
    DEFAULT_EXTENDED_REPORT_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_PRICES_PATH,
    OPTIMIZED_CONFIGS,
    build_prediction_series_by_window,
    load_sector_map,
    simulate_optimized_portfolio,
)
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from src.backtest.cost_model import AlmgrenChrissCostModel, CALIBRATION_NOTE

EXPECTED_BRANCH = "feature/batch2-cvxpy-cost-fix"
DEFAULT_BASELINE_REPORT_PATH = "data/reports/portfolio_optimization_comparison.json"
DEFAULT_OUTPUT_PATH = "data/reports/cvxpy_stability_test.json"
TARGET_FALLBACK_RATE = 0.15
STATUS_KEYS = ("optimal", "optimal_relaxed", "fallback")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"This script must run on branch {EXPECTED_BRANCH}. Found {branch!r}.")

    baseline_report = json.loads((REPO_ROOT / args.baseline_report_path).read_text())
    before_counts = Counter(
        {
            str(key): int(value)
            for key, value in baseline_report["schemes"]["cvxpy_optimized"]["diagnostics"]["solver_status_counts"].items()
        },
    )

    extended_report = json.loads((REPO_ROOT / args.extended_report_path).read_text())
    predictions = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices = pd.read_parquet(REPO_ROOT / args.prices_path)
    predictions_by_window = build_prediction_series_by_window(predictions)
    sector_map = load_sector_map()

    window_metadata = {
        str(window["window_id"]): window
        for window in extended_report["walkforward"]["windows"]
    }

    config = dict(OPTIMIZED_CONFIGS["cvxpy_optimized"])
    cost_model = AlmgrenChrissCostModel()
    logger.info(
        "running stability test with current defaults eta={} gamma={}",
        cost_model.eta,
        cost_model.gamma,
    )

    after_counts: Counter[str] = Counter()
    window_results: list[dict[str, Any]] = []

    for window_id in sorted(predictions_by_window):
        logger.info("simulating current CVXPY path for {}", window_id)
        _, diagnostics = simulate_optimized_portfolio(
            predictions=predictions_by_window[window_id],
            prices=prices,
            cost_model=cost_model,
            scheme_name="cvxpy_optimized",
            scheme_config=config,
            sector_map=sector_map,
            benchmark_ticker=args.benchmark_ticker,
        )
        counts = Counter({str(key): int(value) for key, value in diagnostics["solver_status_counts"].items()})
        after_counts.update(counts)
        window_results.append(
            {
                "window_id": window_id,
                "test_period": window_metadata[window_id]["test_period"],
                "solver_status_counts": dict(counts),
                "fallback_rate": fallback_rate(counts),
                "relaxed_share": relaxed_share(counts),
            },
        )

    before_rate = fallback_rate(before_counts)
    after_rate = fallback_rate(after_counts)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": "scripts/run_cvxpy_stability_test.py",
        "inputs": {
            "baseline_report_path": str(REPO_ROOT / args.baseline_report_path),
            "extended_report_path": str(REPO_ROOT / args.extended_report_path),
            "predictions_path": str(REPO_ROOT / args.predictions_path),
            "prices_path": str(REPO_ROOT / args.prices_path),
            "benchmark_ticker": args.benchmark_ticker.upper(),
        },
        "cost_model_defaults": {
            **cost_model.get_params(),
            "calibration_note": CALIBRATION_NOTE,
        },
        "constraint_relaxation_strategy": {
            "attempts": [
                "standard constraints",
                "relaxed beta to at least [0.7, 1.3] with adaptive sector lower bounds and max sector deviation >= 15%",
                "relaxed beta to at least [0.6, 1.4] with adaptive sector lower bounds and max sector deviation >= 20%",
            ],
            "solver_sequence": ["OSQP", "CLARABEL", "SCS"],
            "status_values": ["optimal", "optimal_relaxed", "fallback"],
            "fallback_logic_changed": False,
        },
        "before": {
            "source": str(REPO_ROOT / args.baseline_report_path),
            "solver_status_counts": dict(before_counts),
            "fallback_rate": before_rate,
        },
        "after": {
            "solver_status_counts": dict(after_counts),
            "fallback_rate": after_rate,
            "relaxed_share": relaxed_share(after_counts),
            "target_fallback_rate": TARGET_FALLBACK_RATE,
            "pass": bool(after_rate < TARGET_FALLBACK_RATE),
            "windows": window_results,
        },
        "comparison": {
            "fallback_rate_improvement": before_rate - after_rate,
            "fallback_count_improvement": int(before_counts.get("fallback", 0) - after_counts.get("fallback", 0)),
        },
    }

    output_path = REPO_ROOT / args.output_path
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved CVXPY stability report to {}", output_path)
    logger.info(
        "fallback rate before={:.4f} after={:.4f} target_pass={}",
        before_rate,
        after_rate,
        report["after"]["pass"],
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run the cached Week 15-16 CVXPY path and compare current solver stability against the original report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--baseline-report-path", default=DEFAULT_BASELINE_REPORT_PATH)
    parser.add_argument("--extended-report-path", default=DEFAULT_EXTENDED_REPORT_PATH)
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args(argv)


def fallback_rate(counts: Counter[str]) -> float:
    total = sum(int(counts.get(key, 0)) for key in STATUS_KEYS)
    if total <= 0:
        return 0.0
    return float(counts.get("fallback", 0)) / float(total)


def relaxed_share(counts: Counter[str]) -> float:
    total = sum(int(counts.get(key, 0)) for key in STATUS_KEYS)
    if total <= 0:
        return 0.0
    return float(counts.get("optimal_relaxed", 0)) / float(total)


if __name__ == "__main__":
    raise SystemExit(main())
