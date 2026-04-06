from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
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

from scripts.run_greyscale_live import (
    BENCHMARK_TICKER,
    FUSION_NAME,
    MODEL_NAMES,
    build_realized_label_series_from_reports,
    compute_report_realized_ics,
    compute_pairwise_rank_correlations,
    configure_logging,
    load_greyscale_reports,
)
from scripts.run_ic_screening import write_json_atomic
from scripts.run_live_pipeline import load_db_state
from src.risk.portfolio_risk import compute_turnover

DEFAULT_REPORT_DIR = "data/reports/greyscale"
DEFAULT_OUTPUT_PATH = "data/reports/greyscale/g4_gate_summary.json"
DEFAULT_HORIZON = 60
DEFAULT_REQUIRED_WEEKS = 4
DEFAULT_LIVE_IC_THRESHOLD = 0.06
DEFAULT_MODEL_CONSISTENCY_THRESHOLD = 0.5
DEFAULT_TURNOVER_THRESHOLD = 0.45


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    report_dir = REPO_ROOT / args.report_dir
    reports = load_greyscale_reports(report_dir)
    as_of = parse_as_of(args.as_of)
    db_state = load_db_state(as_of=as_of)
    latest_pit_trade_date = db_state["latest_pit_trade_date"]
    realized_labels = build_realized_label_series_from_reports(
        reports=reports,
        latest_pit_trade_date=latest_pit_trade_date or as_of.date(),
        as_of=as_of,
        horizon=args.horizon,
        benchmark_ticker=args.benchmark_ticker,
    )

    per_week: list[dict[str, Any]] = []
    previous_weights: dict[str, float] = {}
    for report in reports:
        realized_ics = compute_report_realized_ics(
            report=report,
            realized_labels=realized_labels,
            model_names=list(MODEL_NAMES),
        )
        pairwise_corr = compute_pairwise_rank_correlations(
            {
                model_name: pd.Series(report.get("score_vectors", {}).get(model_name, {}), dtype=float)
                for model_name in MODEL_NAMES
            },
        )
        current_weights = {
            str(ticker): float(weight)
            for ticker, weight in report.get("live_outputs", {}).get("target_weights_after_risk", {}).items()
        }
        turnover = float(compute_turnover(current_weights, previous_weights)) if previous_weights else 0.0
        previous_weights = current_weights
        per_week.append(
            {
                "week_number": int(report.get("week_number", 0)),
                "signal_date": str(report.get("live_outputs", {}).get("signal_date")),
                "generated_at_utc": str(report.get("generated_at_utc", "")),
                "realized_ics": realized_ics,
                "pairwise_rank_correlation": pairwise_corr,
                "risk_status": {
                    "layer1_pass": bool(report.get("risk_checks", {}).get("layer1_data", {}).get("pass", False)),
                    "layer2_pass": bool(report.get("risk_checks", {}).get("layer2_signal", {}).get("pass", False)),
                    "layer3_pass": bool(report.get("risk_checks", {}).get("layer3_portfolio", {}).get("pass", False)),
                    "layer4_pass": bool(report.get("risk_checks", {}).get("layer4_operational", {}).get("pass", False)),
                },
                "turnover_vs_previous": turnover,
                "weight_source": str(report.get("fusion", {}).get("weight_source", "")),
                "holding_count_after_risk": int(report.get("portfolio_metrics", {}).get("holding_count_after_risk", 0)),
            },
        )

    summary = summarize_monitoring(
        per_week=per_week,
        required_weeks=args.required_weeks,
        live_ic_threshold=args.live_ic_threshold,
        turnover_threshold=args.turnover_threshold,
        model_consistency_threshold=args.model_consistency_threshold,
    )
    output = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_utc": as_of.isoformat(),
        "latest_pit_trade_date": None if latest_pit_trade_date is None else latest_pit_trade_date.isoformat(),
        "horizon_days": int(args.horizon),
        "required_weeks": int(args.required_weeks),
        "per_week": per_week,
        "summary": summary,
    }
    write_json_atomic(REPO_ROOT / args.output_path, output)
    logger.info("saved greyscale monitor summary to {}", REPO_ROOT / args.output_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize greyscale live reports and evaluate the G4 gate state.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--as-of")
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--required-weeks", type=int, default=DEFAULT_REQUIRED_WEEKS)
    parser.add_argument("--live-ic-threshold", type=float, default=DEFAULT_LIVE_IC_THRESHOLD)
    parser.add_argument("--model-consistency-threshold", type=float, default=DEFAULT_MODEL_CONSISTENCY_THRESHOLD)
    parser.add_argument("--turnover-threshold", type=float, default=DEFAULT_TURNOVER_THRESHOLD)
    return parser.parse_args(argv)


def parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if "T" in value:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return datetime.combine(date.fromisoformat(value), datetime.max.time(), tzinfo=timezone.utc)


def summarize_monitoring(
    *,
    per_week: list[dict[str, Any]],
    required_weeks: int,
    live_ic_threshold: float,
    turnover_threshold: float,
    model_consistency_threshold: float,
) -> dict[str, Any]:
    matured = [week for week in per_week if np.isfinite(week["realized_ics"].get(FUSION_NAME, np.nan))]
    matured_ic_values = [float(week["realized_ics"][FUSION_NAME]) for week in matured]
    pairwise_values = [
        float(value)
        for week in per_week
        for value in week["pairwise_rank_correlation"].values()
        if np.isfinite(value)
    ]
    layer12_halts = [
        week for week in per_week
        if (not week["risk_status"]["layer1_pass"]) or (not week["risk_status"]["layer2_pass"])
    ]

    recent_weeks = per_week[-required_weeks:] if len(per_week) >= required_weeks else per_week
    recent_matured = [week for week in recent_weeks if np.isfinite(week["realized_ics"].get(FUSION_NAME, np.nan))]
    recent_ic_values = [float(week["realized_ics"][FUSION_NAME]) for week in recent_matured]
    recent_turnover = [float(week["turnover_vs_previous"]) for week in recent_weeks]
    recent_pairwise = [
        float(value)
        for week in recent_weeks
        for value in week["pairwise_rank_correlation"].values()
        if np.isfinite(value)
    ]

    gate_ready = len(recent_weeks) >= required_weeks and len(recent_matured) >= required_weeks
    checks = {
        "live_ic_above_threshold": {
            "value": float(np.mean(recent_ic_values)) if recent_ic_values else None,
            "threshold": float(live_ic_threshold),
            "passed": bool(recent_ic_values and np.mean(recent_ic_values) > live_ic_threshold) if gate_ready else None,
        },
        "positive_ic_weeks": {
            "value": int(sum(value > 0.0 for value in recent_ic_values)),
            "threshold": f">= {max(1, required_weeks - 1)}/{required_weeks}",
            "passed": bool(sum(value > 0.0 for value in recent_ic_values) >= max(1, required_weeks - 1)) if gate_ready else None,
        },
        "no_layer12_halts": {
            "value": int(sum(
                (not week["risk_status"]["layer1_pass"]) or (not week["risk_status"]["layer2_pass"])
                for week in recent_weeks
            )),
            "threshold": 0,
            "passed": bool(all(week["risk_status"]["layer1_pass"] and week["risk_status"]["layer2_pass"] for week in recent_weeks)) if len(recent_weeks) >= required_weeks else None,
        },
        "turnover_below_cap": {
            "value": float(np.mean(recent_turnover)) if recent_turnover else None,
            "threshold": float(turnover_threshold),
            "passed": bool(recent_turnover and np.mean(recent_turnover) < turnover_threshold) if len(recent_weeks) >= required_weeks else None,
        },
        "model_consistency_above_threshold": {
            "value": float(np.mean(recent_pairwise)) if recent_pairwise else None,
            "threshold": float(model_consistency_threshold),
            "passed": bool(recent_pairwise and np.mean(recent_pairwise) > model_consistency_threshold) if len(recent_weeks) >= required_weeks else None,
        },
    }

    if not per_week:
        gate_status = "PENDING"
    elif not gate_ready:
        gate_status = "PENDING"
    elif all(value["passed"] for value in checks.values()):
        gate_status = "PASS"
    else:
        gate_status = "FAIL"

    return {
        "reports_seen": int(len(per_week)),
        "matured_weeks": int(len(matured)),
        "mean_live_ic": float(np.mean(matured_ic_values)) if matured_ic_values else None,
        "rolling_live_ic_std": float(np.std(matured_ic_values[-required_weeks:], ddof=0)) if len(matured_ic_values) >= 2 else None,
        "positive_live_ic_weeks": int(sum(value > 0.0 for value in matured_ic_values)),
        "mean_pairwise_rank_correlation": float(np.mean(pairwise_values)) if pairwise_values else None,
        "mean_turnover": float(np.mean([week["turnover_vs_previous"] for week in per_week])) if per_week else None,
        "layer12_halt_count": int(len(layer12_halts)),
        "gate_status": gate_status,
        "gate_rule": "PASS requires 5/5 checks after 4 matured weekly reports; otherwise PENDING or FAIL.",
        "checks": checks,
    }


if __name__ == "__main__":
    raise SystemExit(main())
