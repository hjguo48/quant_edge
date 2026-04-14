from __future__ import annotations

"""Rebuild and evaluate rank-level fusion of two Ridge walk-forward horizons.

This script reads two existing walk-forward comparison reports, rebuilds the
underlying Ridge validation/test predictions for each window, and combines the
two horizon scores on cross-sectional percentile ranks. The fused signal is
evaluated against the longer-horizon label target by default.

IC-weighted fusion uses per-window validation ICs measured against the
evaluation-horizon validation labels, with negative ICs clipped to zero.
"""

import argparse
from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
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
from scripts.run_walkforward_comparison import (
    BENCHMARK_TICKER,
    DEFAULT_ALL_FEATURES_PATH,
    LABEL_BUFFER_DAYS,
    REBALANCE_WEEKDAY,
    align_panel,
    build_or_load_feature_matrix,
    build_or_load_label_series,
    evaluation_to_dict,
    json_safe,
    parse_date,
    slice_window,
)
from src.models.baseline import RidgeBaselineModel
from src.models.evaluation import evaluate_predictions

DEFAULT_SHORT_REPORT = "data/reports/walkforward_comparison_20d.json"
DEFAULT_LONG_REPORT = "data/reports/walkforward_comparison_60d.json"
DEFAULT_OUTPUT = "data/reports/horizon_fusion_ridge_20d_60d.json"


@dataclass(frozen=True)
class HorizonArtifacts:
    label: str
    horizon_days: int
    report_path: Path
    report_payload: dict[str, Any]
    windows: list[dict[str, Any]]
    retained_features: list[str]
    feature_matrix: pd.DataFrame
    labels: pd.Series
    feature_matrix_cache_path: Path
    label_cache_path: Path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    short_report_path = REPO_ROOT / args.short_report
    long_report_path = REPO_ROOT / args.long_report
    output_path = REPO_ROOT / args.output

    short_report = load_report(short_report_path)
    long_report = load_report(long_report_path)

    short_horizon = parse_horizon_days(short_report)
    long_horizon = parse_horizon_days(long_report)
    if short_horizon >= long_horizon:
        raise RuntimeError(
            f"Expected short horizon < long horizon, got {short_horizon}D and {long_horizon}D.",
        )

    as_of = resolve_shared_as_of(
        short_report=short_report,
        long_report=long_report,
        override=args.as_of,
    )
    benchmark_ticker = resolve_shared_benchmark_ticker(
        short_report=short_report,
        long_report=long_report,
        override=args.benchmark_ticker,
    )
    rebalance_weekday = resolve_shared_rebalance_weekday(
        short_report=short_report,
        long_report=long_report,
        override=args.rebalance_weekday,
    )

    short_windows = select_report_windows(short_report, limit=args.window_limit)
    long_windows = select_report_windows(long_report, limit=args.window_limit)
    aligned_windows = align_window_pairs(short_windows=short_windows, long_windows=long_windows)

    logger.info(
        "horizon fusion configured for {} aligned windows using {}D + {}D Ridge evaluated on {}D labels",
        len(aligned_windows),
        short_horizon,
        long_horizon,
        long_horizon,
    )

    short_artifacts = prepare_horizon_artifacts(
        label=f"{short_horizon}D",
        horizon_days=short_horizon,
        report_path=short_report_path,
        report_payload=short_report,
        windows=short_windows,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=resolve_feature_matrix_cache_path(
            override=args.short_feature_matrix_cache_path,
            horizon_days=short_horizon,
        ),
        label_cache_path=resolve_label_cache_path(
            override=args.short_label_cache_path,
            horizon_days=short_horizon,
        ),
        as_of=as_of,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )
    long_artifacts = prepare_horizon_artifacts(
        label=f"{long_horizon}D",
        horizon_days=long_horizon,
        report_path=long_report_path,
        report_payload=long_report,
        windows=long_windows,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=resolve_feature_matrix_cache_path(
            override=args.long_feature_matrix_cache_path,
            horizon_days=long_horizon,
        ),
        label_cache_path=resolve_label_cache_path(
            override=args.long_label_cache_path,
            horizon_days=long_horizon,
        ),
        as_of=as_of,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )

    per_window: list[dict[str, Any]] = []
    for position, pair in enumerate(aligned_windows, start=1):
        logger.info(
            "running horizon fusion window {}/{} {} test {} -> {}",
            position,
            len(aligned_windows),
            pair["window_id"],
            pair["test_start"],
            pair["test_end"],
        )
        per_window.append(
            run_window_fusion(
                window_id=pair["window_id"],
                short_window=pair["short_window"],
                long_window=pair["long_window"],
                short_artifacts=short_artifacts,
                long_artifacts=long_artifacts,
                rebalance_weekday=rebalance_weekday,
            ),
        )

    report = {
        "fusion_config": {
            "short_report": str(short_report_path),
            "long_report": str(long_report_path),
            "all_features_path": str(REPO_ROOT / args.all_features_path),
            "short_feature_matrix_cache_path": str(short_artifacts.feature_matrix_cache_path),
            "long_feature_matrix_cache_path": str(long_artifacts.feature_matrix_cache_path),
            "short_label_cache_path": str(short_artifacts.label_cache_path),
            "long_label_cache_path": str(long_artifacts.label_cache_path),
            "as_of": as_of.isoformat(),
            "benchmark_ticker": benchmark_ticker,
            "rebalance_weekday": int(rebalance_weekday),
            "short_horizon_days": int(short_horizon),
            "long_horizon_days": int(long_horizon),
            "evaluation_horizon_days": int(long_horizon),
            "rank_fusion": "cross_sectional_percentile_rank -> weighted sum -> final rerank",
            "ic_weighting_source": (
                f"per-window validation IC against {long_horizon}D validation labels, "
                "with negative weights clipped to zero"
            ),
        },
        "per_window": per_window,
        "summary": summarize_fusion(
            per_window=per_window,
            short_label=short_artifacts.label,
            long_label=long_artifacts.label,
        ),
    }
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved horizon fusion report to {}", output_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild 20D/60D-style Ridge walk-forward predictions and evaluate "
            "rank-level multi-horizon fusion against the longer-horizon labels."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--short-report", default=DEFAULT_SHORT_REPORT)
    parser.add_argument("--long-report", default=DEFAULT_LONG_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--short-feature-matrix-cache-path")
    parser.add_argument("--long-feature-matrix-cache-path")
    parser.add_argument("--short-label-cache-path")
    parser.add_argument("--long-label-cache-path")
    parser.add_argument("--as-of")
    parser.add_argument("--label-buffer-days", type=int, default=LABEL_BUFFER_DAYS)
    parser.add_argument("--benchmark-ticker")
    parser.add_argument("--rebalance-weekday", type=int)
    parser.add_argument("--window-limit", type=int)
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


def parse_horizon_days(report: dict[str, Any]) -> int:
    raw = str(report.get("target_horizon", "")).strip()
    match = re.search(r"(\d+)", raw)
    if match is None:
        raise RuntimeError(f"Unable to parse target_horizon from report: {raw!r}")
    return int(match.group(1))


def resolve_shared_as_of(
    *,
    short_report: dict[str, Any],
    long_report: dict[str, Any],
    override: str | None,
) -> date:
    if override:
        return parse_date(override)
    short_as_of = parse_date(str(short_report["as_of"]))
    long_as_of = parse_date(str(long_report["as_of"]))
    if short_as_of != long_as_of:
        raise RuntimeError(f"Report as_of mismatch: {short_as_of} vs {long_as_of}")
    return short_as_of


def resolve_shared_benchmark_ticker(
    *,
    short_report: dict[str, Any],
    long_report: dict[str, Any],
    override: str | None,
) -> str:
    if override:
        return override.upper()
    short_ticker = str(short_report.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER)).upper()
    long_ticker = str(long_report.get("split_config", {}).get("calendar_ticker", BENCHMARK_TICKER)).upper()
    if short_ticker != long_ticker:
        raise RuntimeError(f"Benchmark ticker mismatch between reports: {short_ticker} vs {long_ticker}")
    return short_ticker


def resolve_shared_rebalance_weekday(
    *,
    short_report: dict[str, Any],
    long_report: dict[str, Any],
    override: int | None,
) -> int:
    if override is not None:
        return int(override)
    short_weekday = int(short_report.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY))
    long_weekday = int(long_report.get("split_config", {}).get("rebalance_weekday", REBALANCE_WEEKDAY))
    if short_weekday != long_weekday:
        raise RuntimeError(f"Rebalance weekday mismatch between reports: {short_weekday} vs {long_weekday}")
    return short_weekday


def select_report_windows(report: dict[str, Any], *, limit: int | None) -> list[dict[str, Any]]:
    windows = list(report.get("windows", []))
    if limit is not None:
        if limit <= 0:
            raise ValueError("window_limit must be positive.")
        windows = windows[:limit]
    if not windows:
        raise RuntimeError("No windows available after applying window_limit.")
    return windows


def align_window_pairs(
    *,
    short_windows: list[dict[str, Any]],
    long_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(short_windows) != len(long_windows):
        raise RuntimeError(
            f"Window count mismatch between reports: {len(short_windows)} vs {len(long_windows)}.",
        )

    aligned: list[dict[str, Any]] = []
    for short_window, long_window in zip(short_windows, long_windows, strict=True):
        short_id = str(short_window.get("window_id"))
        long_id = str(long_window.get("window_id"))
        if short_id != long_id:
            raise RuntimeError(f"Window ID mismatch: {short_id} vs {long_id}")

        short_dates = extract_window_dates(short_window)
        long_dates = extract_window_dates(long_window)
        if short_dates["test_start"] != long_dates["test_start"] or short_dates["test_end"] != long_dates["test_end"]:
            raise RuntimeError(
                f"{short_id} test window mismatch: "
                f"{short_dates['test_start']}->{short_dates['test_end']} vs "
                f"{long_dates['test_start']}->{long_dates['test_end']}"
            )

        aligned.append(
            {
                "window_id": short_id,
                "test_start": short_dates["test_start"],
                "test_end": short_dates["test_end"],
                "short_window": short_window,
                "long_window": long_window,
            },
        )
    return aligned


def extract_window_dates(window: dict[str, Any]) -> dict[str, date]:
    payload = window.get("dates", {})
    return {
        "train_start": parse_date(str(payload["train_start"])),
        "train_end": parse_date(str(payload["train_end"])),
        "validation_start": parse_date(str(payload["validation_start"])),
        "validation_end": parse_date(str(payload["validation_end"])),
        "test_start": parse_date(str(payload["test_start"])),
        "test_end": parse_date(str(payload["test_end"])),
    }


def resolve_feature_matrix_cache_path(*, override: str | None, horizon_days: int) -> Path:
    if override:
        return REPO_ROOT / override
    return REPO_ROOT / f"data/features/walkforward_feature_matrix_{horizon_days}d.parquet"


def resolve_label_cache_path(*, override: str | None, horizon_days: int) -> Path:
    if override:
        return REPO_ROOT / override
    return REPO_ROOT / f"data/labels/forward_returns_{horizon_days}d.parquet"


def prepare_horizon_artifacts(
    *,
    label: str,
    horizon_days: int,
    report_path: Path,
    report_payload: dict[str, Any],
    windows: list[dict[str, Any]],
    all_features_path: Path,
    feature_matrix_cache_path: Path,
    label_cache_path: Path,
    as_of: date,
    label_buffer_days: int,
    benchmark_ticker: str,
    rebalance_weekday: int,
) -> HorizonArtifacts:
    retained_features = list(report_payload.get("retained_features") or [])
    if not retained_features:
        raise RuntimeError(f"{report_path} is missing retained_features.")

    global_start = min(extract_window_dates(window)["train_start"] for window in windows)
    global_end = max(extract_window_dates(window)["test_end"] for window in windows)

    feature_matrix = build_or_load_feature_matrix(
        all_features_path=all_features_path,
        cache_path=feature_matrix_cache_path,
        retained_features=retained_features,
        start_date=global_start,
        end_date=global_end,
        rebalance_weekday=rebalance_weekday,
    )
    labels = build_or_load_label_series(
        label_cache_path=label_cache_path,
        tickers=sorted(feature_matrix.index.get_level_values("ticker").unique()),
        start_date=global_start,
        end_date=global_end,
        as_of=as_of,
        horizon=horizon_days,
        label_buffer_days=label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )
    aligned_X, aligned_y = align_panel(feature_matrix, labels)

    logger.info(
        "{} artifacts rows={} dates={} tickers={} features={}",
        label,
        len(aligned_X),
        aligned_X.index.get_level_values("trade_date").nunique(),
        aligned_X.index.get_level_values("ticker").nunique(),
        aligned_X.shape[1],
    )
    return HorizonArtifacts(
        label=label,
        horizon_days=horizon_days,
        report_path=report_path,
        report_payload=report_payload,
        windows=windows,
        retained_features=retained_features,
        feature_matrix=aligned_X,
        labels=aligned_y,
        feature_matrix_cache_path=feature_matrix_cache_path,
        label_cache_path=label_cache_path,
    )


def run_window_fusion(
    *,
    window_id: str,
    short_window: dict[str, Any],
    long_window: dict[str, Any],
    short_artifacts: HorizonArtifacts,
    long_artifacts: HorizonArtifacts,
    rebalance_weekday: int,
) -> dict[str, Any]:
    short_dates = extract_window_dates(short_window)
    long_dates = extract_window_dates(long_window)

    short_split = slice_all_splits(
        X=short_artifacts.feature_matrix,
        y=short_artifacts.labels,
        dates=short_dates,
        rebalance_weekday=rebalance_weekday,
    )
    long_split = slice_all_splits(
        X=long_artifacts.feature_matrix,
        y=long_artifacts.labels,
        dates=long_dates,
        rebalance_weekday=rebalance_weekday,
    )

    short_alpha = extract_ridge_alpha(short_window)
    long_alpha = extract_ridge_alpha(long_window)

    short_validation_pred, short_test_pred = rebuild_ridge_predictions(
        train_X=short_split["train_X"],
        train_y=short_split["train_y"],
        validation_X=short_split["validation_X"],
        validation_y=short_split["validation_y"],
        test_X=short_split["test_X"],
        alpha=short_alpha,
    )
    long_validation_pred, long_test_pred = rebuild_ridge_predictions(
        train_X=long_split["train_X"],
        train_y=long_split["train_y"],
        validation_X=long_split["validation_X"],
        validation_y=long_split["validation_y"],
        test_X=long_split["test_X"],
        alpha=long_alpha,
    )

    eval_validation_y = long_split["validation_y"]
    eval_test_y = long_split["test_y"]

    short_validation_rank = cross_sectional_rank(short_validation_pred)
    long_validation_rank = cross_sectional_rank(long_validation_pred)
    short_test_rank = cross_sectional_rank(short_test_pred)
    long_test_rank = cross_sectional_rank(long_test_pred)

    short_validation_rank, long_validation_rank, eval_validation_y = align_three_series(
        short_validation_rank,
        long_validation_rank,
        eval_validation_y,
    )
    short_test_rank, long_test_rank, eval_test_y = align_three_series(
        short_test_rank,
        long_test_rank,
        eval_test_y,
    )

    short_validation_metrics = evaluate_predictions(eval_validation_y, short_validation_rank)
    long_validation_metrics = evaluate_predictions(eval_validation_y, long_validation_rank)
    short_test_metrics = evaluate_predictions(eval_test_y, short_test_rank)
    long_test_metrics = evaluate_predictions(eval_test_y, long_test_rank)

    ic_weights = build_ic_weights(
        short_validation_ic=short_validation_metrics.ic,
        long_validation_ic=long_validation_metrics.ic,
        short_label=short_artifacts.label,
        long_label=long_artifacts.label,
    )

    equal_weight_pred = fuse_ranked_predictions(
        short_scores=short_test_rank,
        long_scores=long_test_rank,
        short_weight=0.5,
        long_weight=0.5,
    ).rename("equal_weight")
    ic_weighted_pred = fuse_ranked_predictions(
        short_scores=short_test_rank,
        long_scores=long_test_rank,
        short_weight=ic_weights[short_artifacts.label],
        long_weight=ic_weights[long_artifacts.label],
    ).rename("ic_weighted")

    equal_weight_metrics = evaluate_predictions(eval_test_y, equal_weight_pred)
    ic_weighted_metrics = evaluate_predictions(eval_test_y, ic_weighted_pred)

    return {
        "window_id": window_id,
        "test_start": long_dates["test_start"].isoformat(),
        "test_end": long_dates["test_end"].isoformat(),
        "short_horizon_days": int(short_artifacts.horizon_days),
        "long_horizon_days": int(long_artifacts.horizon_days),
        "short_ridge_alpha": float(short_alpha),
        "long_ridge_alpha": float(long_alpha),
        "component_validation_metrics": {
            short_artifacts.label: evaluation_to_dict(short_validation_metrics),
            long_artifacts.label: evaluation_to_dict(long_validation_metrics),
        },
        "component_test_metrics": {
            short_artifacts.label: evaluation_to_dict(short_test_metrics),
            long_artifacts.label: evaluation_to_dict(long_test_metrics),
        },
        "ic_weights": {key: float(value) for key, value in ic_weights.items()},
        "equal_weight_fusion_metrics": evaluation_to_dict(equal_weight_metrics),
        "ic_weighted_fusion_metrics": evaluation_to_dict(ic_weighted_metrics),
        "n_validation_rows": int(len(eval_validation_y)),
        "n_test_rows": int(len(eval_test_y)),
        "n_test_dates": int(eval_test_y.index.get_level_values("trade_date").nunique()),
    }


def slice_all_splits(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    dates: dict[str, date],
    rebalance_weekday: int,
) -> dict[str, pd.DataFrame | pd.Series]:
    train_X, train_y = slice_window(
        X=X,
        y=y,
        start_date=dates["train_start"],
        end_date=dates["train_end"],
        rebalance_weekday=rebalance_weekday,
    )
    validation_X, validation_y = slice_window(
        X=X,
        y=y,
        start_date=dates["validation_start"],
        end_date=dates["validation_end"],
        rebalance_weekday=rebalance_weekday,
    )
    test_X, test_y = slice_window(
        X=X,
        y=y,
        start_date=dates["test_start"],
        end_date=dates["test_end"],
        rebalance_weekday=rebalance_weekday,
    )
    return {
        "train_X": train_X,
        "train_y": train_y,
        "validation_X": validation_X,
        "validation_y": validation_y,
        "test_X": test_X,
        "test_y": test_y,
    }


def extract_ridge_alpha(window: dict[str, Any]) -> float:
    ridge_payload = window.get("results", {}).get("ridge")
    if ridge_payload is None:
        raise RuntimeError(f"Window {window.get('window_id')} does not contain Ridge results.")
    params = ridge_payload.get("best_hyperparams") or {}
    alpha = params.get("alpha")
    if alpha is None:
        raise RuntimeError(f"Window {window.get('window_id')} is missing ridge alpha.")
    return float(alpha)


def rebuild_ridge_predictions(
    *,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    validation_X: pd.DataFrame,
    validation_y: pd.Series,
    test_X: pd.DataFrame,
    alpha: float,
) -> tuple[pd.Series, pd.Series]:
    model = RidgeBaselineModel(alpha=alpha)
    model.train(train_X, train_y)
    validation_pred = model.predict(validation_X).rename("validation_score")

    final_train_X = pd.concat([train_X, validation_X]).sort_index()
    final_train_y = pd.concat([train_y, validation_y]).sort_index()
    model.train(final_train_X, final_train_y)
    test_pred = model.predict(test_X).rename("test_score")
    return validation_pred, test_pred


def cross_sectional_rank(scores: pd.Series) -> pd.Series:
    if scores.empty:
        return scores.astype(float)
    series = pd.Series(scores, dtype=float).dropna().sort_index()
    if not isinstance(series.index, pd.MultiIndex):
        return series.rank(pct=True, method="average")
    ranked_parts: list[pd.Series] = []
    for _, group in series.groupby(level="trade_date", sort=True):
        ranked_parts.append(group.rank(pct=True, method="average"))
    return pd.concat(ranked_parts).sort_index()


def align_three_series(
    left: pd.Series,
    right: pd.Series,
    target: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    aligned_index = left.index.intersection(right.index).intersection(target.index)
    if aligned_index.empty:
        raise RuntimeError("No common aligned rows available for horizon fusion.")
    return (
        left.loc[aligned_index].sort_index(),
        right.loc[aligned_index].sort_index(),
        target.loc[aligned_index].sort_index(),
    )


def build_ic_weights(
    *,
    short_validation_ic: float,
    long_validation_ic: float,
    short_label: str,
    long_label: str,
) -> dict[str, float]:
    raw = {
        short_label: max(float(short_validation_ic), 0.0),
        long_label: max(float(long_validation_ic), 0.0),
    }
    total = sum(raw.values())
    if np.isclose(total, 0.0):
        return {short_label: 0.5, long_label: 0.5}
    return {key: float(value / total) for key, value in raw.items()}


def fuse_ranked_predictions(
    *,
    short_scores: pd.Series,
    long_scores: pd.Series,
    short_weight: float,
    long_weight: float,
) -> pd.Series:
    frame = pd.concat(
        [
            short_scores.rename("short"),
            long_scores.rename("long"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if frame.empty:
        raise RuntimeError("No aligned rank scores available to fuse.")
    blended = short_weight * frame["short"] + long_weight * frame["long"]
    return cross_sectional_rank(blended.rename("score"))


def summarize_fusion(
    *,
    per_window: list[dict[str, Any]],
    short_label: str,
    long_label: str,
) -> dict[str, Any]:
    summary = {
        "components": {
            short_label: summarize_component_block(per_window, short_label),
            long_label: summarize_component_block(per_window, long_label),
        },
        "equal_weight_fusion": summarize_method_block(per_window, "equal_weight_fusion_metrics"),
        "ic_weighted_fusion": summarize_method_block(per_window, "ic_weighted_fusion_metrics"),
        "avg_ic_weights": {
            short_label: nanmean([window["ic_weights"][short_label] for window in per_window]),
            long_label: nanmean([window["ic_weights"][long_label] for window in per_window]),
        },
        "best_method": {},
        "improvement_vs_long_component_ic": float("nan"),
    }

    best_name = None
    best_score = float("-inf")
    for method_name in ("equal_weight_fusion", "ic_weighted_fusion"):
        mean_ic = summary[method_name]["mean_ic"]
        if mean_ic is not None and mean_ic > best_score:
            best_name = method_name
            best_score = float(mean_ic)
    summary["best_method"] = {
        "name": best_name,
        "mean_ic": float(best_score) if np.isfinite(best_score) else None,
    }

    long_mean_ic = summary["components"][long_label]["mean_test_ic"]
    best_mean_ic = summary["best_method"]["mean_ic"]
    if long_mean_ic is not None and best_mean_ic is not None:
        summary["improvement_vs_long_component_ic"] = float(best_mean_ic - long_mean_ic)
    return summary


def summarize_component_block(per_window: list[dict[str, Any]], label: str) -> dict[str, Any]:
    validation_ics = [
        float(window["component_validation_metrics"][label]["ic"])
        for window in per_window
    ]
    test_ics = [
        float(window["component_test_metrics"][label]["ic"])
        for window in per_window
    ]
    test_rank_ics = [
        float(window["component_test_metrics"][label]["rank_ic"])
        for window in per_window
    ]
    test_icirs = [
        float(window["component_test_metrics"][label]["icir"])
        for window in per_window
    ]
    top_deciles = [
        float(window["component_test_metrics"][label]["top_decile_return"])
        for window in per_window
    ]
    return {
        "mean_validation_ic_for_weighting": nanmean(validation_ics),
        "mean_test_ic": nanmean(test_ics),
        "std_test_ic": nanstd(test_ics),
        "mean_test_rank_ic": nanmean(test_rank_ics),
        "mean_test_icir": nanmean(test_icirs),
        "mean_top_decile_return": nanmean(top_deciles),
        "n_windows": int(len(per_window)),
    }


def summarize_method_block(per_window: list[dict[str, Any]], key: str) -> dict[str, Any]:
    ics = [float(window[key]["ic"]) for window in per_window]
    rank_ics = [float(window[key]["rank_ic"]) for window in per_window]
    icirs = [float(window[key]["icir"]) for window in per_window]
    hit_rates = [float(window[key]["hit_rate"]) for window in per_window]
    top_deciles = [float(window[key]["top_decile_return"]) for window in per_window]
    long_shorts = [float(window[key]["long_short_return"]) for window in per_window]
    turnovers = [float(window[key]["turnover"]) for window in per_window]
    return {
        "mean_ic": nanmean(ics),
        "std_ic": nanstd(ics),
        "mean_rank_ic": nanmean(rank_ics),
        "mean_icir": nanmean(icirs),
        "mean_hit_rate": nanmean(hit_rates),
        "mean_top_decile_return": nanmean(top_deciles),
        "mean_long_short_return": nanmean(long_shorts),
        "mean_turnover": nanmean(turnovers),
        "n_windows": int(len(per_window)),
    }


def nanmean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanmean(np.asarray(values, dtype=float)))


def nanstd(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.nanstd(np.asarray(values, dtype=float)))


if __name__ == "__main__":
    raise SystemExit(main())
