from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from src.data.db.session import get_engine
from src.models.evaluation import information_coefficient_series
from src.stats.bootstrap import bootstrap_return_statistics
from src.stats.dsr import compute_deflated_sharpe
from src.stats.ic_test import run_ic_ttest, run_windowed_ic_tests
from src.stats.spa import run_spa_fallback, series_from_records

DEFAULT_MLFLOW_TRACKING_URI = "file:///home/jiahao/quant_edge/mlruns"
BEST_SCHEME_NAME = "equal_weight_buffered"
OPTIMAL_HORIZON = 60
MAX_RESULTS = 5_000


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if args.expected_branch and branch != args.expected_branch:
        raise RuntimeError(f"Expected git branch {args.expected_branch!r}, found {branch!r}.")

    single_window_report = load_json(REPO_ROOT / args.single_window_report)
    tree_report = load_json(REPO_ROOT / args.tree_report)
    lstm_report = load_json(REPO_ROOT / args.lstm_report)
    walkforward_report = load_json(REPO_ROOT / args.walkforward_report)
    portfolio_report = load_json(REPO_ROOT / args.portfolio_report)

    data_overview = load_data_overview(
        ic_report_path=REPO_ROOT / args.ic_report,
        walkforward_report=walkforward_report,
    )
    walkforward_ic_series = load_walkforward_ic_series(
        prediction_path=REPO_ROOT / args.prediction_cache,
        label_path=REPO_ROOT / args.label_cache,
        horizon=args.optimal_horizon,
    )
    combined_ic_series = pd.concat(walkforward_ic_series.values()).sort_index()
    ic_ttest_result = run_ic_ttest(combined_ic_series, alternative="greater")
    ic_window_results = run_windowed_ic_tests(walkforward_ic_series, alternative="greater")

    best_scheme = str(portfolio_report["best_scheme"])
    if best_scheme != BEST_SCHEME_NAME:
        logger.warning("report best_scheme={} differs from expected {}", best_scheme, BEST_SCHEME_NAME)
    best_scheme_periods = load_portfolio_periods(portfolio_report, scheme_name=best_scheme)
    net_excess_returns = pd.Series(
        best_scheme_periods["net_excess_return"].to_numpy(dtype=float),
        index=best_scheme_periods["signal_date"],
        name="net_excess_return",
        dtype=float,
    ).sort_index()
    bootstrap_result = bootstrap_return_statistics(
        net_excess_returns,
        block_size=args.bootstrap_block_size,
        n_bootstrap=args.bootstrap_samples,
        ci_level=args.bootstrap_ci_level,
        annualization=args.annualization,
        seed=args.bootstrap_seed,
    )
    dsr_result = compute_deflated_sharpe(
        net_excess_returns,
        tracking_uri=args.mlflow_tracking_uri,
        max_results=args.mlflow_max_results,
        annualization=args.annualization,
    )
    spa_result = run_spa_test(
        single_window_report=single_window_report,
        tree_report=tree_report,
        lstm_report=lstm_report,
    )
    max_drawdown = compute_max_drawdown(net_excess_returns)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": Path(__file__).name,
        "data_overview": data_overview,
        "model_comparison": build_model_comparison(
            single_window_report=single_window_report,
            tree_report=tree_report,
            lstm_report=lstm_report,
            walkforward_report=walkforward_report,
        ),
        "walkforward": build_walkforward_section(
            walkforward_report=walkforward_report,
            ic_ttest_result=ic_ttest_result,
            ic_window_results=ic_window_results,
        ),
        "portfolio": build_portfolio_section(
            portfolio_report=portfolio_report,
            scheme_name=best_scheme,
            period_frame=best_scheme_periods,
        ),
        "statistical_tests": {
            "ic_ttest": {
                "overall": ic_ttest_result.to_dict(),
                "per_window": {window_id: result.to_dict() for window_id, result in ic_window_results.items()},
            },
            "bootstrap_ci": bootstrap_result.to_dict(),
            "dsr": dsr_result.to_dict(),
            "spa": spa_result.to_dict(),
        },
        "max_drawdown": {
            "value": max_drawdown["value"],
            "threshold": 0.20,
            "pass": max_drawdown["pass"],
            "peak_date": max_drawdown["peak_date"],
            "trough_date": max_drawdown["trough_date"],
        },
    }

    report["go_nogo_checklist"] = build_checklist(
        walkforward_report=walkforward_report,
        portfolio_report=portfolio_report,
        scheme_name=best_scheme,
        bootstrap_result=bootstrap_result,
        dsr_result=dsr_result,
        max_drawdown=max_drawdown,
    )
    report["final_decision"] = determine_final_decision(report["go_nogo_checklist"], report)

    output_path = REPO_ROOT / args.report_path
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved phase 1 alpha report to {}", output_path)
    log_summary(report)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 10 statistical tests and generate the final Phase 1 alpha report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--single-window-report", default="data/reports/single_window_validation.json")
    parser.add_argument("--tree-report", default="data/reports/tree_vs_baseline_comparison.json")
    parser.add_argument("--lstm-report", default="data/reports/lstm_vs_tree_comparison.json")
    parser.add_argument("--walkforward-report", default="data/reports/walkforward_backtest.json")
    parser.add_argument("--portfolio-report", default="data/reports/portfolio_comparison.json")
    parser.add_argument("--ic-report", default="data/features/ic_screening_report_v2.csv")
    parser.add_argument("--prediction-cache", default="data/backtest/portfolio_comparison_predictions.parquet")
    parser.add_argument("--label-cache", default="data/labels/walkforward_forward_returns_multi.parquet")
    parser.add_argument("--report-path", default="data/reports/phase1_alpha_report.json")
    parser.add_argument("--expected-branch", default=None)
    parser.add_argument("--mlflow-tracking-uri", default=DEFAULT_MLFLOW_TRACKING_URI)
    parser.add_argument("--mlflow-max-results", type=int, default=MAX_RESULTS)
    parser.add_argument("--optimal-horizon", type=int, default=OPTIMAL_HORIZON)
    parser.add_argument("--annualization", type=int, default=52)
    parser.add_argument("--bootstrap-block-size", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-ci-level", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def load_data_overview(*, ic_report_path: Path, walkforward_report: dict[str, Any]) -> dict[str, Any]:
    engine = get_engine()
    with engine.connect() as connection:
        universe_tickers = int(connection.execute(text("SELECT count(*) FROM stocks WHERE ticker <> 'SPY'")).scalar_one())
        price_rows = int(connection.execute(text("SELECT count(*) FROM stock_prices")).scalar_one())
        fundamentals_rows = int(connection.execute(text("SELECT count(*) FROM fundamentals_pit")).scalar_one())
        min_trade_date, max_trade_date = connection.execute(
            text("SELECT min(trade_date), max(trade_date) FROM stock_prices")
        ).one()

    feature_report = pd.read_csv(ic_report_path)
    retained_count = int(feature_report["retained"].astype(bool).sum())
    candidate_count = int(len(feature_report))

    return {
        "tickers": universe_tickers,
        "benchmark_ticker": "SPY",
        "prices": price_rows,
        "fundamentals": fundamentals_rows,
        "market_data_date_range": f"{min_trade_date} to {max_trade_date}",
        "research_date_range": (
            f"{walkforward_report['data_summary']['feature_min_date']} "
            f"to {walkforward_report['data_summary']['feature_max_date']}"
        ),
        "features": {
            "candidates": candidate_count,
            "retained": retained_count,
        },
    }


def load_walkforward_ic_series(
    *,
    prediction_path: Path,
    label_path: Path,
    horizon: int,
) -> dict[str, pd.Series]:
    predictions = pd.read_parquet(prediction_path)
    labels = pd.read_parquet(label_path, filters=[("horizon", "==", horizon)])
    predictions["trade_date"] = pd.to_datetime(predictions["trade_date"])
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])

    merged = predictions.merge(
        labels[["ticker", "trade_date", "excess_return"]],
        on=["ticker", "trade_date"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise RuntimeError("Prediction and label join for walk-forward IC series is empty.")

    series_by_window: dict[str, pd.Series] = {}
    for window_id, frame in merged.groupby("window_id", sort=True):
        indexed = frame.set_index(["trade_date", "ticker"]).sort_index()
        ic_series = information_coefficient_series(
            y_true=indexed["excess_return"],
            y_pred=indexed["score"],
        )
        if ic_series.empty:
            raise RuntimeError(f"No IC series generated for window {window_id}.")
        series_by_window[str(window_id)] = ic_series.sort_index()

    return series_by_window


def load_portfolio_periods(portfolio_report: dict[str, Any], *, scheme_name: str) -> pd.DataFrame:
    scheme = portfolio_report["schemes"][scheme_name]
    rows: list[dict[str, Any]] = []
    for window in scheme["windows"]:
        for period in window["portfolio"]["periods"]:
            rows.append(
                {
                    "window_id": window["window_id"],
                    "signal_date": pd.Timestamp(period["signal_date"]),
                    "execution_date": pd.Timestamp(period["execution_date"]),
                    "exit_date": pd.Timestamp(period["exit_date"]),
                    "gross_return": float(period["gross_return"]),
                    "net_return": float(period["net_return"]),
                    "gross_excess_return": float(period["gross_excess_return"]),
                    "net_excess_return": float(period["net_excess_return"]),
                    "cost_rate": float(period["cost_rate"]),
                    "turnover": float(period["turnover"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No period-level portfolio returns found for scheme {scheme_name}.")
    return frame.sort_values(["signal_date", "window_id"]).reset_index(drop=True)


def build_model_comparison(
    *,
    single_window_report: dict[str, Any],
    tree_report: dict[str, Any],
    lstm_report: dict[str, Any],
    walkforward_report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "window1_5d": {
            "ridge": single_window_report["metrics"]["test"],
            "xgboost": tree_report["xgboost"]["metrics"]["test"],
            "lightgbm": tree_report["lightgbm"]["metrics"]["test"],
            "lstm": lstm_report["comparison"]["test_metrics"]["lstm"],
        },
        "horizon_matrix": walkforward_report["horizon_experiment"]["matrix"],
        "optimal": {
            "model": "ridge",
            "horizon": walkforward_report["horizon_experiment"]["optimal_horizon"],
            "rationale": walkforward_report["horizon_experiment"]["rationale"],
        },
    }


def build_walkforward_section(
    *,
    walkforward_report: dict[str, Any],
    ic_ttest_result: Any,
    ic_window_results: dict[str, Any],
) -> dict[str, Any]:
    windows = []
    for window in walkforward_report["walkforward"]["windows"]:
        windows.append(
            {
                "window_id": window["window_id"],
                "train_period": window["train_period"],
                "validation_period": window["validation_period"],
                "test_period": window["test_period"],
                "test_metrics": window["test_metrics"],
                "ic_ttest": ic_window_results[window["window_id"]].to_dict(),
            }
        )

    aggregate = dict(walkforward_report["walkforward"]["aggregate"])
    aggregate["ic_ttest"] = ic_ttest_result.to_dict()
    return {"windows": windows, "aggregate": aggregate}


def build_portfolio_section(
    *,
    portfolio_report: dict[str, Any],
    scheme_name: str,
    period_frame: pd.DataFrame,
) -> dict[str, Any]:
    aggregate = dict(portfolio_report["schemes"][scheme_name]["aggregate"])
    aggregate["weekly_periods"] = int(len(period_frame))
    return {
        "best_scheme": scheme_name,
        **aggregate,
    }


def run_spa_test(
    *,
    single_window_report: dict[str, Any],
    tree_report: dict[str, Any],
    lstm_report: dict[str, Any],
):
    ridge_series = series_from_records(single_window_report["series"]["test"]["ic_series"])
    competitors = {
        "xgboost": series_from_records(tree_report["xgboost"]["series"]["test"]["ic_series"]),
        "lightgbm": series_from_records(tree_report["lightgbm"]["series"]["test"]["ic_series"]),
        "lstm": series_from_records(lstm_report["lstm"]["series"]["test"]["ic_series"]),
    }
    return run_spa_fallback(ridge_series, competitors, benchmark_name="ridge")


def compute_max_drawdown(excess_returns: pd.Series) -> dict[str, Any]:
    values = pd.Series(excess_returns, dtype=float).dropna().sort_index()
    wealth = (1.0 + values).cumprod()
    running_peak = wealth.cummax()
    drawdown = 1.0 - (wealth / running_peak)
    trough_date = drawdown.idxmax()
    peak_date = wealth.loc[:trough_date].idxmax()
    max_drawdown = float(drawdown.max())
    return {
        "value": max_drawdown,
        "pass": bool(max_drawdown < 0.20),
        "peak_date": peak_date.isoformat(),
        "trough_date": trough_date.isoformat(),
    }


def build_checklist(
    *,
    walkforward_report: dict[str, Any],
    portfolio_report: dict[str, Any],
    scheme_name: str,
    bootstrap_result: Any,
    dsr_result: Any,
    max_drawdown: dict[str, Any],
) -> list[dict[str, Any]]:
    walkforward_aggregate = walkforward_report["walkforward"]["aggregate"]
    portfolio_aggregate = portfolio_report["schemes"][scheme_name]["aggregate"]

    checklist = [
        {
            "criterion": "OOS IC > 0.03",
            "value": float(walkforward_aggregate["mean_test_ic"]),
            "threshold": 0.03,
            "pass": bool(walkforward_aggregate["mean_test_ic"] > 0.03),
        },
        {
            "criterion": "DSR p < 0.05",
            "value": float(dsr_result.p_value),
            "threshold": 0.05,
            "pass": bool(dsr_result.p_value < 0.05),
        },
        {
            "criterion": "Net excess > 5%",
            "value": float(portfolio_aggregate["annualized_net_excess"]),
            "threshold": 0.05,
            "pass": bool(portfolio_aggregate["annualized_net_excess"] > 0.05),
        },
        {
            "criterion": "Bootstrap CI > 0",
            "value": float(bootstrap_result.sharpe_ci_lower),
            "threshold": 0.0,
            "pass": bool(bootstrap_result.sharpe_ci_lower > 0.0),
        },
        {
            "criterion": "Max DD < 20%",
            "value": float(max_drawdown["value"]),
            "threshold": 0.20,
            "pass": bool(max_drawdown["pass"]),
        },
        {
            "criterion": "Turnover < 30%",
            "value": float(portfolio_aggregate["average_turnover"]),
            "threshold": 0.30,
            "pass": bool(portfolio_aggregate["average_turnover"] < 0.30),
        },
    ]
    return checklist


def determine_final_decision(checklist: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    passed_count = sum(1 for item in checklist if item["pass"])
    total = len(checklist)
    mean_ic = float(report["walkforward"]["aggregate"]["mean_test_ic"])
    net_excess = float(report["portfolio"]["annualized_net_excess"])

    if mean_ic <= 0.0 and net_excess <= 0.0:
        decision = "PIVOT"
        rationale = "Signal quality and economics both failed, so the alpha thesis does not hold."
    elif passed_count == total:
        decision = "GO"
        rationale = "All Phase 1 gates passed, including significance, economics, and drawdown control."
    elif passed_count >= 4:
        decision = "CONDITIONAL_GO"
        failed = ", ".join(item["criterion"] for item in checklist if not item["pass"])
        rationale = f"{passed_count}/{total} gates passed. Remaining issues: {failed}."
    else:
        decision = "NO_GO"
        failed = ", ".join(item["criterion"] for item in checklist if not item["pass"])
        rationale = f"Only {passed_count}/{total} gates passed. Blocking issues: {failed}."

    return {
        "passed_count": passed_count,
        "total": total,
        "decision": decision,
        "rationale": rationale,
    }


def log_summary(report: dict[str, Any]) -> None:
    decision = report["final_decision"]
    logger.info(
        "decision={} passed={}/{} mean_oos_ic={:.6f} net_excess={:.6f}",
        decision["decision"],
        decision["passed_count"],
        decision["total"],
        report["walkforward"]["aggregate"]["mean_test_ic"],
        report["portfolio"]["annualized_net_excess"],
    )
    logger.info(
        "dsr_p={:.6f} bootstrap_sharpe_ci=({:.6f}, {:.6f}) max_dd={:.6f}",
        report["statistical_tests"]["dsr"]["p_value"],
        report["statistical_tests"]["bootstrap_ci"]["sharpe_ci_lower"],
        report["statistical_tests"]["bootstrap_ci"]["sharpe_ci_upper"],
        report["max_drawdown"]["value"],
    )


if __name__ == "__main__":
    raise SystemExit(main())
