from __future__ import annotations

"""Compare 1D / 5D / 60D Ridge walk-forward signals on a post-cost basis.

This script rebuilds the Ridge test predictions from existing walk-forward
reports, runs the standard top-decile execution simulator with the calibrated
Almgren-Chriss cost model, and aggregates gross/net excess performance across
all test windows.

The comparison is therefore based on the same execution stack already used by
the repo's portfolio backtests rather than on a naive turnover * bps shortcut.
"""

import argparse
from datetime import date, timedelta
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

from scripts.run_horizon_fusion import (  # noqa: E402
    extract_ridge_alpha,
    parse_horizon_days,
    prepare_horizon_artifacts,
    rebuild_ridge_predictions,
    select_report_windows,
    slice_all_splits,
)
from scripts.run_ic_screening import write_json_atomic  # noqa: E402
from scripts.run_turnover_optimization import aggregate_window_portfolios  # noqa: E402
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
from src.backtest.execution import simulate_top_decile_portfolio  # noqa: E402
from src.data.db.pit import get_prices_pit  # noqa: E402

DEFAULT_REPORT_1D = "data/reports/walkforward_comparison_1d_v5_13w.json"
DEFAULT_REPORT_5D = "data/reports/walkforward_comparison_5d_v5_13w.json"
DEFAULT_REPORT_60D = "data/reports/walkforward_comparison_60d_v5_no_analyst.json"
DEFAULT_FEATURE_MATRIX_CACHE_1D = "data/features/walkforward_feature_matrix_1d_v5_13w.parquet"
DEFAULT_FEATURE_MATRIX_CACHE_5D = "data/features/walkforward_feature_matrix_5d_v5_13w.parquet"
DEFAULT_FEATURE_MATRIX_CACHE_60D = "data/features/walkforward_feature_matrix_60d_v5_no_analyst.parquet"
DEFAULT_LABEL_CACHE_1D = "data/labels/forward_returns_1d.parquet"
DEFAULT_LABEL_CACHE_5D = "data/labels/forward_returns_5d.parquet"
DEFAULT_LABEL_CACHE_60D = "data/labels/forward_returns_60d.parquet"
DEFAULT_OUTPUT = "data/reports/post_cost_comparison.json"
HORIZON_LABELS = ("1D", "5D", "60D")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    reports = {
        "1D": REPO_ROOT / args.report_1d,
        "5D": REPO_ROOT / args.report_5d,
        "60D": REPO_ROOT / args.report_60d,
    }
    feature_matrix_caches = {
        "1D": REPO_ROOT / args.feature_matrix_cache_1d,
        "5D": REPO_ROOT / args.feature_matrix_cache_5d,
        "60D": REPO_ROOT / args.feature_matrix_cache_60d,
    }
    label_caches = {
        "1D": REPO_ROOT / args.label_cache_1d,
        "5D": REPO_ROOT / args.label_cache_5d,
        "60D": REPO_ROOT / args.label_cache_60d,
    }

    payloads = {label: load_report(path) for label, path in reports.items()}
    as_of = resolve_shared_as_of(payloads=payloads, override=args.as_of)
    benchmark_ticker = resolve_shared_benchmark(payloads=payloads, override=args.benchmark_ticker)
    rebalance_weekday = resolve_shared_rebalance_weekday(payloads=payloads, override=args.rebalance_weekday)

    cost_model = AlmgrenChrissCostModel(
        eta=args.eta,
        gamma=args.gamma,
        commission_per_share=args.commission_per_share,
        min_spread_bps=args.min_spread_bps,
        gap_penalty_threshold=args.gap_penalty_threshold,
        gap_penalty_multiplier=args.gap_penalty_multiplier,
        low_volume_threshold=args.low_volume_threshold,
        low_volume_temp_impact_multiplier=args.low_volume_temp_impact_multiplier,
    )

    horizon_results: dict[str, dict[str, Any]] = {}
    for label in HORIZON_LABELS:
        report_payload = payloads[label]
        horizon_days = parse_horizon_days(report_payload)
        windows = select_report_windows(report_payload, limit=args.window_limit)
        artifacts = prepare_horizon_artifacts(
            label=label,
            horizon_days=horizon_days,
            report_path=reports[label],
            report_payload=report_payload,
            windows=windows,
            all_features_path=REPO_ROOT / args.all_features_path,
            feature_matrix_cache_path=feature_matrix_caches[label],
            label_cache_path=label_caches[label],
            as_of=as_of,
            label_buffer_days=args.label_buffer_days,
            benchmark_ticker=benchmark_ticker,
            rebalance_weekday=rebalance_weekday,
        )
        result = run_horizon_post_cost(
            label=label,
            report_payload=report_payload,
            windows=windows,
            artifacts=artifacts,
            as_of=as_of,
            benchmark_ticker=benchmark_ticker,
            rebalance_weekday=rebalance_weekday,
            cost_model=cost_model,
            price_buffer_days=args.price_buffer_days,
        )
        horizon_results[label] = result

    comparison = summarize_comparison(horizon_results)
    output = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "as_of": as_of.isoformat(),
        "benchmark_ticker": benchmark_ticker,
        "all_features_path": str(REPO_ROOT / args.all_features_path),
        "cost_model": {
            "name": "AlmgrenChrissCostModel",
            **cost_model.get_params(),
        },
        "horizons": horizon_results,
        "comparison": comparison,
    }

    output_path = REPO_ROOT / args.output
    write_json_atomic(output_path, json_safe(output))
    logger.info("saved post-cost comparison report to {}", output_path)
    print_stdout_table(horizon_results=horizon_results, comparison=comparison)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Ridge walk-forward predictions for 1D/5D/60D reports and compare "
            "post-cost top-decile portfolio performance with the calibrated execution model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--report-1d", default=DEFAULT_REPORT_1D)
    parser.add_argument("--report-5d", default=DEFAULT_REPORT_5D)
    parser.add_argument("--report-60d", default=DEFAULT_REPORT_60D)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--feature-matrix-cache-1d", default=DEFAULT_FEATURE_MATRIX_CACHE_1D)
    parser.add_argument("--feature-matrix-cache-5d", default=DEFAULT_FEATURE_MATRIX_CACHE_5D)
    parser.add_argument("--feature-matrix-cache-60d", default=DEFAULT_FEATURE_MATRIX_CACHE_60D)
    parser.add_argument("--label-cache-1d", default=DEFAULT_LABEL_CACHE_1D)
    parser.add_argument("--label-cache-5d", default=DEFAULT_LABEL_CACHE_5D)
    parser.add_argument("--label-cache-60d", default=DEFAULT_LABEL_CACHE_60D)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--as-of")
    parser.add_argument("--benchmark-ticker")
    parser.add_argument("--rebalance-weekday", type=int)
    parser.add_argument("--window-limit", type=int)
    parser.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    parser.add_argument("--price-buffer-days", type=int, default=7)
    parser.add_argument("--eta", type=float, default=0.426)
    parser.add_argument("--gamma", type=float, default=0.942)
    parser.add_argument("--commission-per-share", type=float, default=0.005)
    parser.add_argument("--min-spread-bps", type=float, default=2.0)
    parser.add_argument("--gap-penalty-threshold", type=float, default=0.02)
    parser.add_argument("--gap-penalty-multiplier", type=float, default=0.5)
    parser.add_argument("--low-volume-threshold", type=float, default=0.30)
    parser.add_argument("--low-volume-temp-impact-multiplier", type=float, default=2.0)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def load_report(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Walk-forward report does not exist: {path}")
    payload = json.loads(path.read_text())
    if "windows" not in payload:
        raise RuntimeError(f"Walk-forward report is missing windows: {path}")
    return payload


def resolve_shared_as_of(*, payloads: dict[str, dict[str, Any]], override: str | None) -> date:
    if override:
        return parse_date(override)
    as_of_values = {label: parse_date(str(payload["as_of"])) for label, payload in payloads.items()}
    values = set(as_of_values.values())
    if len(values) != 1:
        raise RuntimeError(f"Report as_of mismatch: {as_of_values}")
    return next(iter(values))


def resolve_shared_benchmark(*, payloads: dict[str, dict[str, Any]], override: str | None) -> str:
    if override:
        return override.upper()
    tickers = {
        label: str(payload.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER)).upper()
        for label, payload in payloads.items()
    }
    values = set(tickers.values())
    if len(values) != 1:
        raise RuntimeError(f"Benchmark ticker mismatch: {tickers}")
    return next(iter(values))


def resolve_shared_rebalance_weekday(
    *,
    payloads: dict[str, dict[str, Any]],
    override: int | None,
) -> int:
    if override is not None:
        return int(override)
    weekdays = {
        label: int(payload.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY))
        for label, payload in payloads.items()
    }
    values = set(weekdays.values())
    if len(values) != 1:
        raise RuntimeError(f"Rebalance weekday mismatch: {weekdays}")
    return next(iter(values))


def extract_window_dates(window: dict[str, Any]) -> dict[str, date]:
    payload = window["dates"]
    return {
        "train_start": parse_date(str(payload["train_start"])),
        "train_end": parse_date(str(payload["train_end"])),
        "validation_start": parse_date(str(payload["validation_start"])),
        "validation_end": parse_date(str(payload["validation_end"])),
        "test_start": parse_date(str(payload["test_start"])),
        "test_end": parse_date(str(payload["test_end"])),
    }


def run_horizon_post_cost(
    *,
    label: str,
    report_payload: dict[str, Any],
    windows: list[dict[str, Any]],
    artifacts: Any,
    as_of: date,
    benchmark_ticker: str,
    rebalance_weekday: int,
    cost_model: AlmgrenChrissCostModel,
    price_buffer_days: int,
) -> dict[str, Any]:
    horizon_days = parse_horizon_days(report_payload)
    prediction_parts: list[pd.Series] = []
    prediction_counts: list[dict[str, Any]] = []

    for position, window in enumerate(windows, start=1):
        window_id = str(window["window_id"])
        dates = extract_window_dates(window)
        logger.info(
            "rebuilding {} predictions window {}/{} {} test {} -> {}",
            label,
            position,
            len(windows),
            window_id,
            dates["test_start"],
            dates["test_end"],
        )
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
        prediction_parts.append(test_pred.rename("score"))
        prediction_counts.append(
            {
                "window_id": window_id,
                "alpha": float(alpha),
                "test_rows": int(len(test_pred)),
                "test_dates": int(test_pred.index.get_level_values("trade_date").nunique()),
            },
        )

    predictions = pd.concat(prediction_parts).sort_index()
    signal_dates = pd.DatetimeIndex(predictions.index.get_level_values("trade_date")).sort_values().unique()
    tickers = sorted(set(predictions.index.get_level_values("ticker").astype(str).tolist()) | {benchmark_ticker})
    price_start = pd.Timestamp(signal_dates.min()).date()
    price_end = (pd.Timestamp(signal_dates.max()) + pd.Timedelta(days=price_buffer_days)).date()
    prices = get_prices_pit(
        tickers=tickers,
        start_date=price_start,
        end_date=price_end,
        as_of=as_of,
    )
    if prices.empty:
        raise RuntimeError(f"No PIT prices returned for {label} execution backtest.")
    universe_by_date = build_universe_by_date(trade_dates=signal_dates, index_name="SP500")

    portfolio = simulate_top_decile_portfolio(
        predictions=predictions,
        prices=prices,
        cost_model=cost_model,
        benchmark_ticker=benchmark_ticker,
        universe_by_date=universe_by_date,
    )
    aggregate = aggregate_window_portfolios([portfolio])
    observed_signal_dates = int(signal_dates.size)
    total_days = max((signal_dates.max() - signal_dates.min()).days, 1)
    observed_rebalances_per_year = float(observed_signal_dates * 365.25 / total_days)

    report_summary = report_payload.get("summary", {}).get("ridge", {})
    return {
        "report_path": str(artifacts.report_path),
        "target_horizon_days": int(horizon_days),
        "retained_feature_count": int(len(artifacts.retained_features)),
        "report_mean_ic": float(report_summary.get("mean_test_ic", float("nan"))),
        "report_mean_rank_ic": float(report_summary.get("mean_test_rank_ic", float("nan"))),
        "report_mean_icir": float(report_summary.get("mean_test_icir", float("nan"))),
        "report_mean_top_decile_return": float(report_summary.get("mean_top_decile_return", float("nan"))),
        "report_windows_completed": int(report_summary.get("windows_completed", len(windows))),
        "gross_ann_excess": float(aggregate["annualized_gross_excess"]),
        "net_ann_excess": float(aggregate["annualized_net_excess"]),
        "gross_ann_return": float(aggregate["annualized_gross_return"]),
        "net_ann_return": float(aggregate["annualized_net_return"]),
        "benchmark_ann_return": float(aggregate["annualized_benchmark_return"]),
        "cost_drag": float(aggregate["total_cost_drag"]),
        "turnover": float(aggregate["average_turnover"]),
        "sharpe_est": float(aggregate["sharpe_proxy"]),
        "period_count": int(aggregate["period_count"]),
        "observed_rebalances_per_year": observed_rebalances_per_year,
        "schedule_cadence": describe_schedule(observed_rebalances_per_year),
        "price_range": {
            "start": price_start.isoformat(),
            "end": price_end.isoformat(),
        },
        "prediction_summary": {
            "rows": int(len(predictions)),
            "dates": int(signal_dates.size),
            "tickers": int(predictions.index.get_level_values("ticker").nunique()),
            "per_window": prediction_counts,
        },
    }


def describe_schedule(observed_rebalances_per_year: float) -> str:
    if observed_rebalances_per_year > 200:
        return "daily_like"
    if observed_rebalances_per_year > 40:
        return "weekly_like"
    if observed_rebalances_per_year > 10:
        return "monthly_like"
    return "sparse"


def summarize_comparison(horizon_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    best_horizon = max(
        horizon_results.items(),
        key=lambda item: item[1]["net_ann_excess"],
    )[0]
    daily_exceeds_weekly = bool(
        horizon_results["1D"]["net_ann_excess"] > horizon_results["60D"]["net_ann_excess"],
    )
    short_exceeds_weekly = bool(
        max(horizon_results["1D"]["net_ann_excess"], horizon_results["5D"]["net_ann_excess"])
        > horizon_results["60D"]["net_ann_excess"]
    )
    return {
        "best_horizon": best_horizon,
        "daily_exceeds_weekly": daily_exceeds_weekly,
        "short_horizon_exceeds_weekly": short_exceeds_weekly,
        "gate_pass": daily_exceeds_weekly,
        "net_excess_ranking": [
            {"horizon": label, "net_ann_excess": float(payload["net_ann_excess"])}
            for label, payload in sorted(
                horizon_results.items(),
                key=lambda item: item[1]["net_ann_excess"],
                reverse=True,
            )
        ],
        "gate_note": (
            "Comparison uses reconstructed top-decile portfolios under the current "
            "walk-forward report schedule. If the 1D report is still weekly-rebalanced, "
            "this is not yet a true daily live-switch gate."
        ),
    }


def print_stdout_table(*, horizon_results: dict[str, dict[str, Any]], comparison: dict[str, Any]) -> None:
    print("\nPost-cost comparison")
    print(
        f"{'Horizon':<8} {'Cadence':<12} {'GrossEx':>12} {'CostDrag':>12} "
        f"{'NetEx':>12} {'Turnover':>12} {'Sharpe':>10}",
    )
    for label in HORIZON_LABELS:
        payload = horizon_results[label]
        print(
            f"{label:<8} "
            f"{payload['schedule_cadence']:<12} "
            f"{payload['gross_ann_excess']:>12.4f} "
            f"{payload['cost_drag']:>12.4f} "
            f"{payload['net_ann_excess']:>12.4f} "
            f"{payload['turnover']:>12.4f} "
            f"{payload['sharpe_est']:>10.4f}",
        )
    print("\nComparison")
    print(f"Best horizon: {comparison['best_horizon']}")
    print(f"Daily exceeds weekly: {comparison['daily_exceeds_weekly']}")
    print(f"Short horizon exceeds weekly: {comparison['short_horizon_exceeds_weekly']}")
    print(f"Gate pass: {comparison['gate_pass']}")


if __name__ == "__main__":
    raise SystemExit(main())
