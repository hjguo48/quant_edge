from __future__ import annotations

import argparse
from dataclasses import asdict
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

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from src.portfolio.constraints import CVXPYOptimizer
from src.risk.data_risk import DataRiskMonitor, RiskSeverity as DataSeverity
from src.risk.operational_risk import OperationalRiskMonitor, RiskSeverity as OperationalSeverity
from src.risk.portfolio_risk import (
    PortfolioRiskEngine,
    compute_portfolio_beta,
    compute_portfolio_cvar,
    compute_sector_weights,
    sector_weight_deviation,
)
from src.risk.signal_risk import SignalRiskMonitor, RiskSeverity as SignalSeverity

EXPECTED_BRANCH = "feature/week17-shadow-mode"
DEFAULT_OUTPUT_PATH = "data/reports/stress_test_report.json"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_OPTIMIZATION_REPORT_PATH = "data/reports/portfolio_optimization_comparison.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}.")

    rng = np.random.default_rng(args.seed)
    data_monitor = DataRiskMonitor()
    signal_monitor = SignalRiskMonitor()
    portfolio_engine = PortfolioRiskEngine()
    operational_monitor = OperationalRiskMonitor()
    portfolio_context = build_portfolio_test_context(rng)

    layer1 = run_layer1_tests(data_monitor=data_monitor, rng=rng)
    layer2 = run_layer2_tests(signal_monitor=signal_monitor, rng=rng)
    layer3 = run_layer3_tests(
        engine=portfolio_engine,
        context=portfolio_context,
        prices_path=REPO_ROOT / args.prices_path,
    )
    layer4 = run_layer4_tests(operational_monitor=operational_monitor)

    all_tests = [*layer1, *layer2, *layer3, *layer4]
    passed = sum(1 for test in all_tests if test["passed"])
    total = len(all_tests)
    pass_rate = float(passed / total) if total else 0.0

    fallback_analysis = analyze_cvxpy_fallback(
        optimization_report_path=REPO_ROOT / args.optimization_report_path,
        context=portfolio_context,
    )
    cost_calibration_note = build_cost_calibration_note(REPO_ROOT / args.optimization_report_path)

    if passed != total:
        failed_names = [test["test_id"] for test in all_tests if not test["passed"]]
        raise RuntimeError(f"Stress suite did not pass completely: {failed_names}")

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": Path(__file__).name,
        "summary": {
            "total_tests": int(total),
            "passed": int(passed),
            "failed": int(total - passed),
            "pass_rate": pass_rate,
            "all_passed": True,
        },
        "layers": {
            "layer1_data_risk": layer1,
            "layer2_signal_risk": layer2,
            "layer3_portfolio_risk": layer3,
            "layer4_operational_risk": layer4,
        },
        "cvxpy_fallback_analysis": fallback_analysis,
        "cost_calibration_note": cost_calibration_note,
    }

    output_path = REPO_ROOT / args.output_path
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved stress test report to {}", output_path)
    logger.info("stress test pass rate {:.2%} ({} / {})", pass_rate, passed, total)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a four-layer risk-engine stress test suite and document optimizer fallback behavior.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--optimization-report-path", default=DEFAULT_OPTIMIZATION_REPORT_PATH)
    parser.add_argument("--seed", type=int, default=20260402)
    return parser.parse_args(argv)


def run_layer1_tests(*, data_monitor: DataRiskMonitor, rng: np.random.Generator) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    universe_size = 100

    missing_green = data_monitor.check_missing_rate(
        data=pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(97)]}),
        universe_size=universe_size,
    )
    tests.append(
        build_test_result(
            layer="layer1_data_risk",
            test_id="1.1.green",
            name="Missing rate 3% remains GREEN",
            passed=missing_green.severity == DataSeverity.GREEN,
            expected={"severity": DataSeverity.GREEN.value},
            observed=missing_green.to_dict(),
        ),
    )

    missing_yellow = data_monitor.check_missing_rate(
        data=pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(92)]}),
        universe_size=universe_size,
    )
    tests.append(
        build_test_result(
            layer="layer1_data_risk",
            test_id="1.1.yellow",
            name="Missing rate 8% escalates to YELLOW",
            passed=missing_yellow.severity == DataSeverity.YELLOW and not missing_yellow.halt_pipeline,
            expected={"severity": DataSeverity.YELLOW.value, "halt_pipeline": False},
            observed=missing_yellow.to_dict(),
        ),
    )

    missing_red = data_monitor.check_missing_rate(
        data=pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(80)]}),
        universe_size=universe_size,
    )
    tests.append(
        build_test_result(
            layer="layer1_data_risk",
            test_id="1.1.red",
            name="Missing rate 20% escalates to RED and halts pipeline",
            passed=missing_red.severity == DataSeverity.RED and missing_red.halt_pipeline,
            expected={"severity": DataSeverity.RED.value, "halt_pipeline": True},
            observed=missing_red.to_dict(),
        ),
    )

    historical = pd.DataFrame(
        {
            "feature_a": rng.normal(0.0, 1.0, size=500),
            "feature_b": rng.normal(1.0, 0.5, size=500),
        },
    )
    current_normal = pd.DataFrame(
        {
            "feature_a": rng.normal(0.0, 1.0, size=120),
            "feature_b": rng.normal(1.0, 0.5, size=120),
        },
    )
    no_shift_alerts = data_monitor.check_feature_distribution(
        current_features=current_normal,
        historical_features=historical,
    )
    tests.append(
        build_test_result(
            layer="layer1_data_risk",
            test_id="1.2.normal",
            name="Normal feature distribution shows no KS drift alerts",
            passed=len(no_shift_alerts) == 0,
            expected={"alerts": 0},
            observed={"alerts": [alert.to_dict() for alert in no_shift_alerts]},
        ),
    )

    current_shifted = pd.DataFrame(
        {
            "feature_a": rng.normal(3.0, 1.0, size=120),
            "feature_b": rng.normal(-2.0, 0.5, size=120),
        },
    )
    shift_alerts = data_monitor.check_feature_distribution(
        current_features=current_shifted,
        historical_features=historical,
    )
    tests.append(
        build_test_result(
            layer="layer1_data_risk",
            test_id="1.2.shifted",
            name="Shifted feature distribution triggers KS alerts",
            passed=len(shift_alerts) >= 1,
            expected={"alerts_at_least": 1},
            observed={"alerts": [alert.to_dict() for alert in shift_alerts]},
        ),
    )

    api_green = data_monitor.check_api_health(
        response_times=[0.25, 0.30, 0.28],
        error_count=1,
        consecutive_failures=1,
    )
    tests.append(
        build_test_result(
            layer="layer1_data_risk",
            test_id="1.3.greenish",
            name="Single API failure stays non-failover",
            passed=api_green.severity == DataSeverity.YELLOW and not api_green.switch_to_backup,
            expected={"severity": DataSeverity.YELLOW.value, "switch_to_backup": False},
            observed=api_green.to_dict(),
        ),
    )

    api_failover = data_monitor.check_api_health(
        response_times=[2.5, 2.8, 3.1],
        error_count=3,
        consecutive_failures=3,
    )
    tests.append(
        build_test_result(
            layer="layer1_data_risk",
            test_id="1.3.failover",
            name="Three consecutive API failures trigger backup-source switch",
            passed=api_failover.severity == DataSeverity.RED and api_failover.switch_to_backup,
            expected={"severity": DataSeverity.RED.value, "switch_to_backup": True},
            observed=api_failover.to_dict(),
        ),
    )

    return tests


def run_layer2_tests(*, signal_monitor: SignalRiskMonitor, rng: np.random.Generator) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    healthy_ic = signal_monitor.check_rolling_ic(
        ic_history=[0.08] * 16 + [0.07, 0.09, 0.08, 0.07],
    )
    tests.append(
        build_test_result(
            layer="layer2_signal_risk",
            test_id="2.1.green",
            name="Healthy IC history remains GREEN",
            passed=healthy_ic.severity == SignalSeverity.GREEN and not healthy_ic.switch_model,
            expected={"severity": SignalSeverity.GREEN.value, "switch_model": False},
            observed=healthy_ic.to_dict(),
        ),
    )

    degraded_ic = signal_monitor.check_rolling_ic(
        ic_history=[0.08] * 16 + [0.01, 0.01, 0.01, 0.01],
    )
    tests.append(
        build_test_result(
            layer="layer2_signal_risk",
            test_id="2.1.degraded",
            name="Four consecutive low-IC periods trigger switch alert",
            passed=degraded_ic.severity == SignalSeverity.RED and degraded_ic.switch_model,
            expected={"severity": SignalSeverity.RED.value, "switch_model": True},
            observed=degraded_ic.to_dict(),
        ),
    )

    scores = pd.Series(np.linspace(-1.0, 1.0, 200), name="score")
    realized_good = pd.Series(scores.to_numpy() + rng.normal(0.0, 0.05, size=len(scores)), name="realized")
    good_calibration = signal_monitor.check_calibration(scores, realized_good)
    tests.append(
        build_test_result(
            layer="layer2_signal_risk",
            test_id="2.2.green",
            name="Monotonic score-to-return mapping passes calibration",
            passed=good_calibration.severity == SignalSeverity.GREEN,
            expected={"severity": SignalSeverity.GREEN.value},
            observed=good_calibration.to_dict(),
        ),
    )

    realized_bad = pd.Series((-scores).to_numpy() + rng.normal(0.0, 0.05, size=len(scores)), name="realized")
    bad_calibration = signal_monitor.check_calibration(scores, realized_bad)
    tests.append(
        build_test_result(
            layer="layer2_signal_risk",
            test_id="2.2.reversal",
            name="Rank-reversed returns trigger calibration alert",
            passed=bad_calibration.severity == SignalSeverity.RED,
            expected={"severity": SignalSeverity.RED.value},
            observed=bad_calibration.to_dict(),
        ),
    )

    challenger_three = signal_monitor.check_champion_vs_challenger(
        champion_ic=0.04,
        challenger_ic=0.05,
        consecutive_challenger_wins=3,
    )
    tests.append(
        build_test_result(
            layer="layer2_signal_risk",
            test_id="2.3.no_switch",
            name="Three challenger wins are not enough to recommend a switch",
            passed=challenger_three.severity == SignalSeverity.GREEN and not challenger_three.recommend_switch,
            expected={"severity": SignalSeverity.GREEN.value, "recommend_switch": False},
            observed=challenger_three.to_dict(),
        ),
    )

    challenger_four = signal_monitor.check_champion_vs_challenger(
        champion_ic=0.04,
        challenger_ic=0.05,
        consecutive_challenger_wins=4,
    )
    tests.append(
        build_test_result(
            layer="layer2_signal_risk",
            test_id="2.3.switch",
            name="Four challenger wins recommend a model switch",
            passed=challenger_four.severity == SignalSeverity.YELLOW and challenger_four.recommend_switch,
            expected={"severity": SignalSeverity.YELLOW.value, "recommend_switch": True},
            observed=challenger_four.to_dict(),
        ),
    )

    return tests


def run_layer3_tests(
    *,
    engine: PortfolioRiskEngine,
    context: dict[str, Any],
    prices_path: Path,
) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    benchmark_sector_weights = compute_sector_weights(context["benchmark_weights"], context["sector_map"])

    concentrated_weights = make_concentrated_weights(context["tickers"][:10], top_weight=0.30)
    single_stock = engine.apply_all_constraints(
        weights=concentrated_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=context["return_history"],
        spy_returns=context["spy_returns"],
        current_weights=concentrated_weights,
        candidate_ranking=context["ranking"],
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.1",
            name="Single-stock 30% concentration is capped at 10%",
            passed=max(single_stock.weights.values()) <= 0.10 + 1e-9,
            expected={"max_weight": 0.10},
            observed=single_stock.to_dict(),
        ),
    )

    sector_overweight = make_sector_overweight_weights(
        tickers=context["tickers"],
        sector_map=context["sector_map"],
        benchmark_sector_weights=benchmark_sector_weights,
        target_sector="Financial Services",
        target_sector_weight=benchmark_sector_weights["Financial Services"] + 0.20,
    )
    sector_result = engine.apply_all_constraints(
        weights=sector_overweight,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=context["return_history"],
        spy_returns=context["spy_returns"],
        current_weights=sector_overweight,
        candidate_ranking=context["ranking"],
    )
    sector_deviation_after = sector_weight_deviation(sector_result.sector_weights, benchmark_sector_weights)
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.2",
            name="Financial Services overweight is clipped back inside ±15%",
            passed=float(sector_deviation_after.get("Financial Services", 0.0)) <= 0.15 + 1e-9,
            expected={"max_sector_deviation": 0.15},
            observed={
                "before_deviation": sector_weight_deviation(
                    compute_sector_weights(sector_overweight, context["sector_map"]),
                    benchmark_sector_weights,
                ),
                "after": sector_result.to_dict(),
            },
        ),
    )

    low_beta_weights = make_beta_extreme_weights(
        beta_map=context["beta_map"],
        target="low",
    )
    low_beta_result = engine.apply_all_constraints(
        weights=low_beta_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=context["return_history"],
        spy_returns=context["spy_returns"],
        current_weights=low_beta_weights,
        candidate_ranking=context["ranking"],
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.3.low_beta",
            name="Low-beta portfolio is nudged into the target beta band",
            passed=low_beta_result.portfolio_beta is not None and 0.8 <= low_beta_result.portfolio_beta <= 1.2,
            expected={"beta_bounds": [0.8, 1.2]},
            observed=low_beta_result.to_dict(),
        ),
    )

    high_beta_weights = make_beta_extreme_weights(
        beta_map=context["beta_map"],
        target="high",
    )
    high_beta_result = engine.apply_all_constraints(
        weights=high_beta_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=context["return_history"],
        spy_returns=context["spy_returns"],
        current_weights=high_beta_weights,
        candidate_ranking=context["ranking"],
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.3.high_beta",
            name="High-beta portfolio is nudged back into the target beta band",
            passed=high_beta_result.portfolio_beta is not None and 0.8 <= high_beta_result.portfolio_beta <= 1.2,
            expected={"beta_bounds": [0.8, 1.2]},
            observed=high_beta_result.to_dict(),
        ),
    )

    corr_weights = {
        context["tickers"][0]: 0.18,
        context["tickers"][1]: 0.17,
        **{ticker: 0.65 / 13.0 for ticker in context["tickers"][2:15]},
    }
    corr_result = engine.apply_all_constraints(
        weights=corr_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=context["return_history"],
        spy_returns=context["spy_returns"],
        current_weights=corr_weights,
        candidate_ranking=context["ranking"],
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.4",
            name="Highly correlated pair drops the weaker-ranked name",
            passed=any(
                entry.rule_name == "correlation_dedup"
                and entry.triggered
                and int(entry.after.get("high_corr_pairs", 1)) == 0
                for entry in corr_result.audit_trail
            ),
            expected={"correlation_dedup_triggered": True, "high_corr_pairs_after": 0},
            observed={
                "portfolio": corr_result.to_dict(),
                "audit_trail": [entry.to_dict() for entry in corr_result.audit_trail],
            },
        ),
    )

    current_weights = {ticker: 1.0 / 20.0 for ticker in context["tickers"][:20]}
    turnover_target = {ticker: 1.0 / 20.0 for ticker in context["tickers"][10:30]}
    turnover_result = engine.apply_all_constraints(
        weights=turnover_target,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=context["return_history"],
        spy_returns=context["spy_returns"],
        current_weights=current_weights,
        candidate_ranking=context["ranking"],
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.5",
            name="60%+ one-period turnover is clipped to the 40% cap",
            passed=turnover_result.turnover <= 0.40 + 1e-9,
            expected={"turnover_cap": 0.40},
            observed=turnover_result.to_dict(),
        ),
    )

    cvar_history = add_tail_crash(
        return_history=context["return_history"],
        affected_tickers=context["tickers"][:20],
        shock=-0.12,
        crash_days=3,
    )
    cvar_weights = {ticker: 1.0 / 20.0 for ticker in context["tickers"][:20]}
    cvar_result = engine.apply_all_constraints(
        weights=cvar_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=cvar_history,
        spy_returns=context["spy_returns"],
        current_weights=cvar_weights,
        candidate_ranking=context["ranking"],
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.6",
            name="CVaR breach triggers a 20% gross haircut",
            passed=cvar_result.gross_exposure <= 0.80 + 1e-6 and cvar_result.cvar_99 is not None,
            expected={"gross_exposure_after": 0.80},
            observed={
                "before_cvar": compute_portfolio_cvar(cvar_weights, return_history=cvar_history, confidence=0.99),
                "after": cvar_result.to_dict(),
            },
        ),
    )

    min_holdings_weights = {ticker: 1.0 / 10.0 for ticker in context["tickers"][:10]}
    min_holdings_result = engine.apply_all_constraints(
        weights=min_holdings_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=context["return_history"],
        spy_returns=context["spy_returns"],
        current_weights=min_holdings_weights,
        candidate_ranking=context["ranking"],
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.7",
            name="Portfolio with 10 names is expanded to at least 20 holdings",
            passed=min_holdings_result.holding_count >= 20,
            expected={"minimum_holdings": 20},
            observed=min_holdings_result.to_dict(),
        ),
    )

    crash_history = load_actual_crash_return_history(prices_path=prices_path, tickers=context["tickers"])
    crash_weights = {ticker: 1.0 / 20.0 for ticker in context["tickers"][:20]}
    crash_result = engine.apply_all_constraints(
        weights=crash_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=crash_history["return_history"],
        spy_returns=crash_history["spy_returns"],
        current_weights=crash_weights,
        candidate_ranking=context["ranking"],
        stress_test_shock=-0.20,
        stress_warning_threshold=-0.12,
    )
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.8",
            name="2020 crash scenario produces a stress warning without invalid weights",
            passed=(
                any("Stress scenario return" in warning for warning in crash_result.warnings)
                and all(weight >= 0.0 for weight in crash_result.weights.values())
                and not any(pd.isna(list(crash_result.weights.values())))
            ),
            expected={"stress_warning": True, "negative_weights": False, "nan_weights": False},
            observed=crash_result.to_dict(),
        ),
    )

    combined_weights = make_sector_overweight_weights(
        tickers=context["tickers"],
        sector_map=context["sector_map"],
        benchmark_sector_weights=benchmark_sector_weights,
        target_sector="Financial Services",
        target_sector_weight=0.58,
    )
    combined_weights[context["tickers"][0]] = 0.30
    combined_weights = normalize_weight_map(combined_weights)
    combined_current = {ticker: 1.0 / 20.0 for ticker in context["tickers"][10:30]}
    combined_history = add_tail_crash(
        return_history=context["return_history"],
        affected_tickers=context["tickers"][:20],
        shock=-0.10,
        crash_days=4,
    )
    combined_result = engine.apply_all_constraints(
        weights=combined_weights,
        benchmark_weights=context["benchmark_weights"],
        sector_map=context["sector_map"],
        beta_map=context["beta_map"],
        return_history=combined_history,
        spy_returns=context["spy_returns"],
        current_weights=combined_current,
        candidate_ranking=context["ranking"],
        stress_test_shock=-0.20,
        stress_warning_threshold=-0.12,
    )
    triggered = [(entry.priority, entry.rule_name) for entry in combined_result.audit_trail if entry.triggered]
    priority_values = [entry.priority for entry in combined_result.audit_trail]
    ordering_ok = priority_values == sorted(priority_values)
    tests.append(
        build_test_result(
            layer="layer3_portfolio_risk",
            test_id="3.9",
            name="Combined stress case preserves P0→P1→P2 rule ordering",
            passed=ordering_ok and len(triggered) >= 4,
            expected={"ordered_priorities": ["P0", "P1", "P2"], "triggered_rules_at_least": 4},
            observed={
                "triggered_rules": triggered,
                "audit_trail": [entry.to_dict() for entry in combined_result.audit_trail],
            },
        ),
    )

    return tests


def run_layer4_tests(*, operational_monitor: OperationalRiskMonitor) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []

    timeout_alert = operational_monitor.check_timeout(runtime_seconds=7_200.0)
    tests.append(
        build_test_result(
            layer="layer4_operational_risk",
            test_id="4.1",
            name="Pipeline runtime over one hour triggers timeout abort",
            passed=timeout_alert.severity == OperationalSeverity.RED and timeout_alert.halt_pipeline,
            expected={"severity": OperationalSeverity.RED.value, "halt_pipeline": True},
            observed=timeout_alert.to_dict(),
        ),
    )

    fail_safe_alert = operational_monitor.check_fail_safe_mode(critical_alerts=["data_pipeline_failure"])
    tests.append(
        build_test_result(
            layer="layer4_operational_risk",
            test_id="4.2",
            name="Fail-safe mode holds positions and stops trading on critical failure",
            passed=fail_safe_alert.severity == OperationalSeverity.RED and fail_safe_alert.fail_safe_mode,
            expected={"severity": OperationalSeverity.RED.value, "fail_safe_mode": True},
            observed=fail_safe_alert.to_dict(),
        ),
    )

    audit_events = [
        operational_monitor.audit_decision(action="load_signals", actor="shadow_mode", details={"window": "W6"}),
        operational_monitor.audit_decision(action="portfolio_risk_check", actor="shadow_mode", details={"triggered": 2}),
    ]
    report = operational_monitor.run_all_checks(
        runtime_seconds=42.0,
        critical_alerts=[],
        audit_events=audit_events,
    )
    timestamps_preserved = all(record.timestamp_utc for record in report.audit_log)
    tests.append(
        build_test_result(
            layer="layer4_operational_risk",
            test_id="4.3",
            name="Audit log preserves all entries with timestamps",
            passed=len(report.audit_log) == 3 and timestamps_preserved,
            expected={"audit_entries": 3, "timestamps_present": True},
            observed=report.to_dict(),
        ),
    )

    return tests


def analyze_cvxpy_fallback(
    *,
    optimization_report_path: Path,
    context: dict[str, Any],
) -> dict[str, Any]:
    report_payload: dict[str, Any] | None = None
    if optimization_report_path.exists():
        report_payload = json.loads(optimization_report_path.read_text())

    report_counts: dict[str, int] = {}
    fallback_rate = None
    if report_payload is not None:
        report_counts = {
            str(key): int(value)
            for key, value in report_payload["schemes"]["cvxpy_optimized"]["diagnostics"]["solver_status_counts"].items()
        }
        total = max(sum(report_counts.values()), 1)
        fallback_rate = float(report_counts.get("fallback", 0) / total)

    optimizer = CVXPYOptimizer()
    synthetic_case = build_cvxpy_infeasible_case(context)
    full_result = optimizer.optimize(**synthetic_case["full_constraints"])
    relaxed_result = optimizer.optimize(**synthetic_case["relaxed_constraints"])
    reproduced = full_result.solver_status == "fallback" and relaxed_result.solver_status in {"optimal", "optimal_inaccurate"}

    root_cause = (
        "Observed fallback appears driven by feasibility pressure from combined hard sector and beta bounds, "
        "not by a missing solver dependency. The synthetic reproduction falls back under full constraints and "
        "solves once those bounds are relaxed."
        if reproduced
        else "Observed fallback could not be cleanly reproduced with the synthetic case; retain the current implementation and monitor."
    )

    return {
        "report_source": str(optimization_report_path),
        "historical_solver_status_counts": report_counts,
        "historical_fallback_rate": fallback_rate,
        "synthetic_reproduction": {
            "reproduced": reproduced,
            "full_constraints_status": full_result.solver_status,
            "relaxed_constraints_status": relaxed_result.solver_status,
            "full_constraints_portfolio_beta": full_result.portfolio_beta,
            "relaxed_constraints_portfolio_beta": relaxed_result.portfolio_beta,
            "full_constraints_sector_weights": full_result.sector_weights,
            "relaxed_constraints_sector_weights": relaxed_result.sector_weights,
        },
        "action_taken": "documented_only",
        "root_cause_assessment": root_cause,
        "recommended_follow_up": (
            "Keep the current fallback path. Any material reduction in fallback would require softening production "
            "constraint semantics rather than a narrow numerical fix."
        ),
    }


def build_cost_calibration_note(optimization_report_path: Path) -> dict[str, Any]:
    if not optimization_report_path.exists():
        return {
            "available": False,
            "message": "Week 15 optimization report was not available.",
        }
    report = json.loads(optimization_report_path.read_text())
    calibration = report.get("cost_calibration", {})
    return {
        "available": True,
        "original": calibration.get("original"),
        "calibrated": calibration.get("calibrated"),
        "calibration_ratio": calibration.get("calibration_ratio"),
        "message": (
            "Week 15 showed that Almgren-Chriss impact needed a 3x uplift to match realized slippage. "
            "This remains a documentation and future-research item rather than a Week 17 code change."
        ),
    }


def build_portfolio_test_context(rng: np.random.Generator) -> dict[str, Any]:
    tickers = [f"T{index:02d}" for index in range(30)]
    sectors = {
        **{ticker: "Financial Services" for ticker in tickers[:10]},
        **{ticker: "Technology" for ticker in tickers[10:20]},
        **{ticker: "Healthcare" for ticker in tickers[20:30]},
    }
    betas = {ticker: float(beta) for ticker, beta in zip(tickers, np.linspace(0.55, 1.45, len(tickers)))}
    dates = pd.bdate_range("2024-01-02", periods=140)
    market = rng.normal(0.0005, 0.012, size=len(dates))
    shared_pair = market * 0.9 + rng.normal(0.0, 0.002, size=len(dates))
    frame = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for ticker in tickers:
        frame[ticker] = betas[ticker] * market + rng.normal(0.0, 0.008, size=len(dates))
    frame[tickers[0]] = shared_pair + rng.normal(0.0, 0.001, size=len(dates))
    frame[tickers[1]] = shared_pair + rng.normal(0.0, 0.001, size=len(dates))
    spy_returns = pd.Series(market, index=dates, name="SPY", dtype=float)
    benchmark_weights = {ticker: 1.0 / len(tickers) for ticker in tickers}
    ranking = tickers.copy()
    return {
        "tickers": tickers,
        "sector_map": sectors,
        "beta_map": betas,
        "return_history": frame,
        "spy_returns": spy_returns,
        "benchmark_weights": benchmark_weights,
        "ranking": ranking,
    }


def load_actual_crash_return_history(*, prices_path: Path, tickers: list[str]) -> dict[str, Any]:
    if not prices_path.exists():
        raise FileNotFoundError(prices_path)

    prices = pd.read_parquet(prices_path)
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["adj_close"] = pd.to_numeric(prices["adj_close"], errors="coerce")
    subset = prices.loc[
        prices["trade_date"].between(pd.Timestamp("2020-02-20"), pd.Timestamp("2020-03-31"))
        & prices["ticker"].isin(["SPY", *tickers])
    , ["trade_date", "ticker", "adj_close"]].copy()
    subset.sort_values(["ticker", "trade_date"], inplace=True)
    returns = subset.assign(daily_return=subset.groupby("ticker")["adj_close"].pct_change())
    wide = returns.pivot(index="trade_date", columns="ticker", values="daily_return").sort_index()
    spy_returns = wide.pop("SPY").dropna()
    wide = wide.reindex(columns=tickers).dropna(how="all")
    return {"return_history": wide, "spy_returns": spy_returns}


def build_cvxpy_infeasible_case(context: dict[str, Any]) -> dict[str, Any]:
    tickers = context["tickers"][:12]
    expected_returns = np.linspace(0.03, 0.01, len(tickers))
    covariance = np.eye(len(tickers)) * 0.02
    prev_weights = np.array([1.0 / len(tickers)] * len(tickers))
    sector_map = {ticker: context["sector_map"][ticker] for ticker in tickers}
    benchmark_sector_weights = {
        "Financial Services": 0.34,
        "Technology": 0.33,
        "Healthcare": 0.33,
    }
    stock_betas = np.array([0.45] * 8 + [1.55] * 4, dtype=float)
    adv = np.array([3_000_000.0] * len(tickers), dtype=float)

    full = {
        "expected_returns": expected_returns,
        "covariance": covariance,
        "tickers": tickers,
        "prev_weights": prev_weights,
        "sector_map": sector_map,
        "benchmark_sector_weights": benchmark_sector_weights,
        "stock_betas": stock_betas,
        "adv": adv,
        "portfolio_size": 1e7,
        "max_weight": 0.10,
        "max_sector_deviation": 0.02,
        "beta_bounds": (0.95, 1.05),
        "min_holdings": 8,
        "lambda_risk": 1.0,
        "lambda_turnover": 0.005,
    }
    relaxed = dict(full)
    relaxed["max_sector_deviation"] = 0.20
    relaxed["beta_bounds"] = (0.70, 1.30)
    return {"full_constraints": full, "relaxed_constraints": relaxed}


def make_concentrated_weights(tickers: list[str], *, top_weight: float) -> dict[str, float]:
    remainder = 1.0 - float(top_weight)
    base = remainder / max(len(tickers) - 1, 1)
    weights = {ticker: base for ticker in tickers}
    weights[tickers[0]] = float(top_weight)
    return normalize_weight_map(weights)


def make_sector_overweight_weights(
    *,
    tickers: list[str],
    sector_map: dict[str, str],
    benchmark_sector_weights: dict[str, float],
    target_sector: str,
    target_sector_weight: float,
) -> dict[str, float]:
    target_members = [ticker for ticker in tickers if sector_map[ticker] == target_sector]
    other_members = [ticker for ticker in tickers if sector_map[ticker] != target_sector]
    weights: dict[str, float] = {}
    target_weight = min(max(target_sector_weight, 0.0), 0.90)
    other_total = max(1.0 - target_weight, 0.0)
    for ticker in target_members:
        weights[ticker] = target_weight / max(len(target_members), 1)
    for ticker in other_members:
        weights[ticker] = other_total / max(len(other_members), 1)
    return normalize_weight_map(weights)


def make_beta_extreme_weights(*, beta_map: dict[str, float], target: str) -> dict[str, float]:
    ordered = sorted(beta_map.items(), key=lambda item: item[1])
    low = [ticker for ticker, _ in ordered[:15]]
    high = [ticker for ticker, _ in ordered[-15:]]
    if target == "low":
        weights = {ticker: 0.95 / len(low) for ticker in low}
        for ticker in high[:5]:
            weights[ticker] = 0.05 / 5.0
        return normalize_weight_map(weights)
    weights = {ticker: 0.95 / len(high) for ticker in high}
    for ticker in low[:5]:
        weights[ticker] = 0.05 / 5.0
    return normalize_weight_map(weights)


def add_tail_crash(
    *,
    return_history: pd.DataFrame,
    affected_tickers: list[str],
    shock: float,
    crash_days: int,
) -> pd.DataFrame:
    history = return_history.copy()
    impacted_dates = history.index[-int(crash_days) :]
    history.loc[impacted_dates, affected_tickers] = float(shock)
    return history


def normalize_weight_map(weights: dict[str, float]) -> dict[str, float]:
    series = pd.Series(weights, dtype=float).clip(lower=0.0)
    total = float(series.sum())
    if total <= 0.0:
        return {}
    normalized = series / total
    return {str(ticker): float(weight) for ticker, weight in normalized.items()}


def build_test_result(
    *,
    layer: str,
    test_id: str,
    name: str,
    passed: bool,
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer": layer,
        "test_id": test_id,
        "name": name,
        "passed": bool(passed),
        "expected": expected,
        "observed": observed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
