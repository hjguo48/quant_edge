from __future__ import annotations

"""Generate per-quintile expected-return statistics from Ridge walk-forward scores."""

import argparse
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

from scripts.run_horizon_fusion import (  # noqa: E402
    extract_ridge_alpha,
    parse_horizon_days,
    prepare_horizon_artifacts,
    select_report_windows,
    slice_all_splits,
    rebuild_ridge_predictions,
)
from scripts.run_ic_screening import write_json_atomic  # noqa: E402
from scripts.run_walkforward_comparison import (  # noqa: E402
    BENCHMARK_TICKER,
    DEFAULT_ALL_FEATURES_PATH,
    LABEL_BUFFER_DAYS,
    REBALANCE_WEEKDAY,
    json_safe,
    parse_date,
)
from src.stats.bootstrap import bootstrap_return_statistics  # noqa: E402

DEFAULT_COMPARISON_REPORT = "data/reports/walkforward_comparison_60d_ridge_v2.json"
DEFAULT_FEATURE_MATRIX_CACHE_PATH = "data/features/walkforward_feature_matrix_60d_v2.parquet"
DEFAULT_LABEL_CACHE_PATH = "data/labels/forward_returns_60d.parquet"
DEFAULT_OUTPUT_PATH = "data/reports/quintile_expected_returns.json"
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 12
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_CI_LEVEL = 0.95
QUINTILE_COUNT = 5


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    comparison_path = REPO_ROOT / args.comparison_report
    report_payload = json.loads(comparison_path.read_text())
    windows = select_report_windows(report_payload, limit=args.window_limit)
    horizon_days = parse_horizon_days(report_payload)
    as_of = parse_date(args.as_of) if args.as_of else parse_date(str(report_payload["as_of"]))
    benchmark_ticker = str(
        report_payload.get("split_config", {}).get("calendar_ticker", args.benchmark_ticker),
    ).upper()
    rebalance_weekday = int(args.rebalance_weekday)

    artifacts = prepare_horizon_artifacts(
        label=f"{horizon_days}D",
        horizon_days=horizon_days,
        report_path=comparison_path,
        report_payload=report_payload,
        windows=windows,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=REPO_ROOT / args.feature_matrix_cache_path,
        label_cache_path=REPO_ROOT / args.label_cache_path,
        as_of=as_of,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )

    quintile_rows: list[dict[str, Any]] = []
    for position, window in enumerate(windows, start=1):
        window_id = str(window["window_id"])
        dates = {
            key: parse_date(str(window["dates"][key]))
            for key in (
                "train_start",
                "train_end",
                "validation_start",
                "validation_end",
                "test_start",
                "test_end",
            )
        }
        logger.info(
            "processing window {}/{} {} test {} -> {}",
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
        quintile_rows.extend(
            compute_window_quintile_rows(
                window_id=window_id,
                scores=test_pred,
                forward_returns=split["test_y"],
            ),
        )

    rows_frame = pd.DataFrame(quintile_rows)
    if rows_frame.empty:
        raise RuntimeError("No quintile observations were generated from the comparison report.")

    annualization = max(1, int(round(252 / horizon_days)))
    quintiles_payload = build_quintile_payload(
        rows_frame=rows_frame,
        annualization=annualization,
        block_size=args.bootstrap_block_size,
        n_bootstrap=args.n_bootstrap,
        ci_level=args.ci_level,
        seed=args.seed,
    )

    output_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_report": str(comparison_path),
        "data_source": "walk_forward_quintile_bootstrap",
        "score_source": args.score_source or comparison_path.stem,
        "quintile_definition": "1=highest predicted scores, 5=lowest predicted scores",
        "horizon_days": int(horizon_days),
        "ci_level": float(args.ci_level),
        "bootstrap_block_size": int(args.bootstrap_block_size),
        "n_bootstrap": int(args.n_bootstrap),
        "annualization_factor": int(annualization),
        "window_count": int(len(windows)),
        "retained_feature_count": int(report_payload.get("retained_feature_count", 0)),
        "quintiles": quintiles_payload,
    }

    output_path = REPO_ROOT / args.output
    write_json_atomic(output_path, json_safe(output_payload))
    logger.info("saved quintile expected-return report to {}", output_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild Phase E Ridge walk-forward predictions and compute per-quintile "
            "expected return / Sharpe bootstrap statistics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--comparison-report", default=DEFAULT_COMPARISON_REPORT)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--feature-matrix-cache-path", default=DEFAULT_FEATURE_MATRIX_CACHE_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--score-source", default=None)
    parser.add_argument("--as-of")
    parser.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    parser.add_argument("--benchmark-ticker", default=BENCHMARK_TICKER)
    parser.add_argument("--rebalance-weekday", type=int, default=REBALANCE_WEEKDAY)
    parser.add_argument("--window-limit", type=int)
    parser.add_argument("--bootstrap-block-size", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_SIZE)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--ci-level", type=float, default=DEFAULT_CI_LEVEL)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def compute_window_quintile_rows(
    *,
    window_id: str,
    scores: pd.Series,
    forward_returns: pd.Series,
) -> list[dict[str, Any]]:
    aligned = pd.concat(
        [scores.rename("score"), forward_returns.rename("forward_return")],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return []

    if not isinstance(aligned.index, pd.MultiIndex):
        raise RuntimeError("Expected MultiIndex test predictions with trade_date and ticker.")

    rows: list[dict[str, Any]] = []
    for trade_date, frame in aligned.groupby(level="trade_date", sort=True):
        cross_section = frame.droplevel("trade_date").copy()
        if cross_section.empty:
            continue
        cross_section["quintile"] = assign_quintiles(cross_section["score"], n_quintiles=QUINTILE_COUNT)
        for quintile, bucket in cross_section.groupby("quintile", sort=True):
            rows.append(
                {
                    "window_id": window_id,
                    "trade_date": pd.Timestamp(trade_date),
                    "quintile": int(quintile),
                    "mean_forward_return": float(bucket["forward_return"].mean()),
                    "stock_count": int(len(bucket)),
                },
            )
    return rows


def assign_quintiles(scores: pd.Series, *, n_quintiles: int) -> pd.Series:
    if scores.empty:
        return pd.Series(dtype=int)

    ordered_rank = scores.rank(method="first", ascending=False)
    quintiles = np.floor((ordered_rank - 1.0) * n_quintiles / len(scores)).astype(int) + 1
    return pd.Series(quintiles, index=scores.index, dtype=int)


def build_quintile_payload(
    *,
    rows_frame: pd.DataFrame,
    annualization: int,
    block_size: int,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for quintile in range(1, QUINTILE_COUNT + 1):
        quintile_rows = rows_frame.loc[rows_frame["quintile"] == quintile].copy()
        if quintile_rows.empty:
            continue

        quintile_rows.sort_values(["trade_date", "window_id"], inplace=True)
        series = pd.Series(
            quintile_rows["mean_forward_return"].to_numpy(dtype=float),
            index=pd.MultiIndex.from_frame(quintile_rows[["trade_date", "window_id"]]),
            name="forward_return",
        )
        bootstrap = bootstrap_return_statistics(
            series,
            block_size=block_size,
            n_bootstrap=n_bootstrap,
            ci_level=ci_level,
            annualization=annualization,
            seed=seed,
        )
        payload[str(quintile)] = {
            "annualized_excess": {
                "estimate": float(bootstrap.annualized_excess_estimate),
                "ci_lower": float(bootstrap.annualized_excess_ci_lower),
                "ci_upper": float(bootstrap.annualized_excess_ci_upper),
            },
            "sharpe": {
                "estimate": float(bootstrap.sharpe_estimate),
                "ci_lower": float(bootstrap.sharpe_ci_lower),
                "ci_upper": float(bootstrap.sharpe_ci_upper),
            },
            "mean_forward_return": {
                "estimate": float(bootstrap.mean_excess_estimate),
                "ci_lower": float(bootstrap.mean_excess_ci_lower),
                "ci_upper": float(bootstrap.mean_excess_ci_upper),
            },
            "n_observations": int(bootstrap.n_observations),
            "n_stock_observations": int(quintile_rows["stock_count"].sum()),
            "mean_stock_count_per_date": float(quintile_rows["stock_count"].mean()),
        }
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
