from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from math import ceil
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
from scripts.run_ic_weighted_fusion import (
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_FUSION_FEATURE_MATRIX_CACHE_PATH,
    DEFAULT_HORIZON,
    DEFAULT_LABEL_BUFFER_DAYS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_REBALANCE_WEEKDAY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TREE_N_JOBS,
    build_rolling_weights,
    combine_predictions,
    cross_sectional_zscore,
    extract_best_params,
    instantiate_model,
)
from scripts.run_regime_analysis import extract_window_dates, parse_date, select_windows
from scripts.run_walkforward_comparison import (
    DEFAULT_ALL_FEATURES_PATH,
    DEFAULT_IC_REPORT_PATH,
    DEFAULT_LABEL_CACHE_PATH,
    align_panel,
    build_or_load_feature_matrix,
    build_or_load_label_series,
    load_retained_features,
    slice_window,
)
from src.mlflow_config import get_mlflow_tracking_uri, normalize_tracking_uri
from src.models.evaluation import information_coefficient_series
from src.models.experiment import ExperimentTracker
from src.stats.bootstrap import bootstrap_return_statistics
from src.stats.dsr import compute_deflated_sharpe
from src.stats.ic_test import run_ic_ttest
from src.stats.spa import run_spa_fallback

DEFAULT_COMPARISON_REPORT = "data/reports/walkforward_comparison_60d.json"
DEFAULT_FUSION_REPORT = "data/reports/fusion_analysis_60d.json"
DEFAULT_OUTPUT_PATH = "data/reports/g3_gate_results.json"
DEFAULT_CHAMPION_KEY = "ic_weighted_fusion"
DEFAULT_FUSION_REPORT_FORMAT = "auto"
DEFAULT_TRACKING_URI = None
DEFAULT_ALPHA = 0.05
DEFAULT_IC_THRESHOLD = 0.03
DEFAULT_BOOTSTRAP_BLOCK_SIZE = 12
DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_COST_PER_TURNOVER_BPS = 50.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    comparison_path = REPO_ROOT / args.comparison_report
    fusion_path = REPO_ROOT / args.fusion_report
    output_path = REPO_ROOT / args.output

    comparison = json.loads(comparison_path.read_text())
    fusion = json.loads(fusion_path.read_text())
    comparison_windows = select_windows(comparison.get("windows", []), limit=args.window_limit)
    champion_report = normalize_champion_report(
        fusion,
        champion_key=args.champion_key,
        report_format=args.fusion_report_format,
        window_limit=args.window_limit,
    )
    fusion_windows = [record["raw_window"] for record in champion_report["window_records"]]
    if len(comparison_windows) != len(champion_report["window_records"]):
        raise RuntimeError(
            "Window count mismatch "
            f"comparison={len(comparison_windows)} "
            f"fusion={len(champion_report['window_records'])}.",
        )
    validate_window_alignment(comparison_windows, fusion_windows)

    mean_test_ic = champion_report["mean_ic"]
    window_ic_series = pd.Series(
        [float(window["ic"]) for window in champion_report["window_records"]],
        index=[str(window["window_id"]) for window in champion_report["window_records"]],
        name="window_ic",
        dtype=float,
    )

    reconstruction = reconstruct_daily_artifacts(
        comparison_windows=comparison_windows,
        fusion_windows=fusion_windows,
        all_features_path=REPO_ROOT / args.all_features_path,
        feature_matrix_cache_path=REPO_ROOT / args.feature_matrix_cache_path,
        ic_report_path=REPO_ROOT / args.ic_report_path,
        label_cache_path=REPO_ROOT / args.label_cache_path,
        as_of=parse_date(args.as_of),
        horizon=args.horizon,
        label_buffer_days=args.label_buffer_days,
        benchmark_ticker=args.benchmark_ticker,
        rebalance_weekday=args.rebalance_weekday,
        rolling_window_days=args.rolling_window,
        temperature=args.temperature,
        tree_n_jobs=args.tree_n_jobs,
        random_state=args.random_state,
    )

    check_1 = {
        "value": mean_test_ic,
        "threshold": float(args.ic_threshold),
        "passed": bool(mean_test_ic > args.ic_threshold),
        "positive_windows": int((window_ic_series > 0.0).sum()),
        "n_windows": int(len(window_ic_series)),
        "source": f"{fusion_path}:{args.champion_key}:{champion_report['report_format']}",
    }

    ttest_result = run_ic_ttest(window_ic_series, alternative="greater", alpha=args.alpha)
    check_2 = {
        **ttest_result.to_dict(),
        "threshold": float(args.alpha),
        "passed": bool(ttest_result.p_value < args.alpha),
        "method": "one_sided_ttest_on_window_ics",
    }

    bootstrap_result = bootstrap_return_statistics(
        reconstruction["top_decile_excess"],
        block_size=args.bootstrap_block_size,
        n_bootstrap=args.n_bootstrap,
        annualization=max(1, int(round(252 / args.horizon))),
    )
    check_3 = {
        **bootstrap_result.to_dict(),
        "passed": bool(bootstrap_result.mean_excess_ci_lower > 0.0),
        "method": "top_decile_excess_per_test_date_proxy",
    }

    tracking_uri = discover_tracking_uri(args.tracking_uri)
    dsr_result = compute_deflated_sharpe(
        reconstruction["fusion_daily_ic"],
        tracking_uri=tracking_uri,
        annualization=52,
        alpha=args.alpha,
    )
    check_4 = {
        **dsr_result.to_dict(),
        "tracking_uri": tracking_uri,
        "passed": bool(dsr_result.p_value < args.alpha),
        "method": "daily_cross_sectional_ic_proxy",
    }

    spa_result = run_spa_fallback(
        reconstruction["ridge_daily_ic"],
        {"new_fusion": reconstruction["fusion_daily_ic"]},
        benchmark_name="ridge_proxy",
        alpha=args.alpha,
    )
    primary_comparison = spa_result.comparisons[0] if spa_result.comparisons else None
    check_5 = {
        **spa_result.to_dict(),
        "passed": bool(spa_result.significant),
        "benchmark_proxy": "ridge_from_r8.2",
    }
    if primary_comparison is not None:
        check_5["comparison"] = primary_comparison.to_dict()

    cost_result = estimate_cost_adjusted_excess(
        top_decile_excess=reconstruction["top_decile_excess"],
        turnover_series=reconstruction["turnover_series"],
        horizon_days=args.horizon,
        cost_per_turnover_bps=args.cost_per_turnover_bps,
    )
    check_6 = {
        **cost_result,
        "threshold": 0.05,
        "passed": bool(cost_result["value"] > 0.05),
    }

    checks = {
        "oos_ic_above_threshold": check_1,
        "ic_ttest_significant": check_2,
        "bootstrap_ci_positive": check_3,
        "dsr_significant": check_4,
        "spa_vs_old_champion": check_5,
        "cost_adjusted_excess": check_6,
    }
    total_passed = sum(int(bool(check["passed"])) for check in checks.values())
    total_checks = len(checks)
    gate_result = classify_gate_result(total_passed, total_checks)

    report = {
        "gate": "G3",
        "champion": f"{args.champion_key}_{args.horizon}d",
        "mean_test_ic": mean_test_ic,
        "window_count": int(len(window_ic_series)),
        "positive_window_count": int((window_ic_series > 0.0).sum()),
        "checks": checks,
        "total_passed": int(total_passed),
        "total_checks": int(total_checks),
        "gate_result": gate_result,
        "gate_rule": "PASS requires 6/6, CONDITIONAL_GO requires 5/6",
        "artifacts": {
            "comparison_report": str(comparison_path),
            "fusion_report": str(fusion_path),
            "fusion_report_format": champion_report["report_format"],
            "tracking_uri": tracking_uri,
        },
        "series_reconstruction": {
            "daily_fusion_ic_observations": int(len(reconstruction["fusion_daily_ic"])),
            "daily_ridge_ic_observations": int(len(reconstruction["ridge_daily_ic"])),
            "top_decile_excess_observations": int(len(reconstruction["top_decile_excess"])),
            "turnover_observations": int(len(reconstruction["turnover_series"])),
        },
    }
    write_json_atomic(output_path, json_safe(report))
    logger.info(
        "saved G3 gate report to {} total_passed={}/{} gate_result={}",
        output_path,
        total_passed,
        total_checks,
        gate_result,
    )
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the G3 gate statistics for the 60D IC-weighted fusion champion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--comparison-report", default=DEFAULT_COMPARISON_REPORT)
    parser.add_argument("--fusion-report", default=DEFAULT_FUSION_REPORT)
    parser.add_argument(
        "--fusion-report-format",
        choices=("auto", "fusion", "comparison"),
        default=DEFAULT_FUSION_REPORT_FORMAT,
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--champion-key", default=DEFAULT_CHAMPION_KEY)
    parser.add_argument("--all-features-path", default=DEFAULT_ALL_FEATURES_PATH)
    parser.add_argument("--feature-matrix-cache-path", default=DEFAULT_FUSION_FEATURE_MATRIX_CACHE_PATH)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--label-cache-path", default=DEFAULT_LABEL_CACHE_PATH)
    parser.add_argument("--as-of", default="2026-03-31")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--label-buffer-days", type=int, default=DEFAULT_LABEL_BUFFER_DAYS)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--rebalance-weekday", type=int, default=DEFAULT_REBALANCE_WEEKDAY)
    parser.add_argument("--rolling-window", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--tree-n-jobs", type=int, default=DEFAULT_TREE_N_JOBS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--ic-threshold", type=float, default=DEFAULT_IC_THRESHOLD)
    parser.add_argument("--bootstrap-block-size", type=int, default=DEFAULT_BOOTSTRAP_BLOCK_SIZE)
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    parser.add_argument("--cost-per-turnover-bps", type=float, default=DEFAULT_COST_PER_TURNOVER_BPS)
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI)
    parser.add_argument("--window-limit", type=int)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def normalize_champion_report(
    payload: dict[str, Any],
    *,
    champion_key: str,
    report_format: str,
    window_limit: int | None,
) -> dict[str, Any]:
    resolved_format = detect_champion_report_format(payload, requested=report_format)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise KeyError("Champion report is missing a summary payload.")
    if champion_key not in summary:
        raise KeyError(
            f"Champion key {champion_key!r} is not present in report summary. "
            f"Available keys: {sorted(summary.keys())}",
        )

    champion_summary = summary[champion_key]
    if resolved_format == "fusion":
        raw_windows = select_windows(payload.get("per_window", []), limit=window_limit)
        mean_ic = float(champion_summary["mean_ic"])
        window_records = [
            {
                "window_id": str(window["window_id"]),
                "ic": float(window[f"{champion_key}_ic"]),
                "raw_window": window,
            }
            for window in raw_windows
        ]
    else:
        raw_windows = select_windows(payload.get("windows", []), limit=window_limit)
        mean_ic = float(champion_summary["mean_test_ic"])
        window_records = []
        for window in raw_windows:
            results = window.get("results", {})
            champion_result = results.get(champion_key)
            if not isinstance(champion_result, dict):
                raise KeyError(
                    f"Window {window.get('window_id')} is missing results for champion {champion_key!r}.",
                )
            test_metrics = champion_result.get("test_metrics", {})
            if "ic" not in test_metrics:
                raise KeyError(
                    f"Window {window.get('window_id')} champion {champion_key!r} is missing test_metrics.ic.",
                )
            window_records.append(
                {
                    "window_id": str(window["window_id"]),
                    "ic": float(test_metrics["ic"]),
                    "raw_window": window,
                },
            )

    if not window_records:
        raise RuntimeError("Champion report does not contain any selected windows.")

    return {
        "report_format": resolved_format,
        "mean_ic": mean_ic,
        "summary": champion_summary,
        "window_records": window_records,
    }


def detect_champion_report_format(payload: dict[str, Any], *, requested: str) -> str:
    if requested == "fusion":
        return "fusion"
    if requested == "comparison":
        return "comparison"
    if isinstance(payload.get("per_window"), list):
        return "fusion"
    if isinstance(payload.get("windows"), list):
        return "comparison"
    raise RuntimeError(
        "Unable to auto-detect fusion report format. "
        "Expected either 'per_window' (fusion_analysis) or 'windows' (walkforward_comparison).",
    )


def validate_window_alignment(
    comparison_windows: list[dict[str, Any]],
    fusion_windows: list[dict[str, Any]],
) -> None:
    for position, (comparison_window, fusion_window) in enumerate(
        zip(comparison_windows, fusion_windows, strict=True),
        start=1,
    ):
        comparison_id = str(comparison_window.get("window_id", f"W{position}"))
        fusion_id = str(fusion_window.get("window_id", f"W{position}"))
        if comparison_id != fusion_id:
            raise RuntimeError(
                f"Window ID mismatch at position {position}: comparison={comparison_id} fusion={fusion_id}.",
            )

        comparison_dates = extract_window_dates(comparison_window)
        fusion_dates = extract_window_dates(fusion_window)
        if (
            comparison_dates["test_start"] != fusion_dates["test_start"]
            or comparison_dates["test_end"] != fusion_dates["test_end"]
        ):
            raise RuntimeError(
                "Test window mismatch at position "
                f"{position}: comparison=({comparison_dates['test_start']},{comparison_dates['test_end']}) "
                f"fusion=({fusion_dates['test_start']},{fusion_dates['test_end']}).",
            )


def reconstruct_daily_artifacts(
    *,
    comparison_windows: list[dict[str, Any]],
    fusion_windows: list[dict[str, Any]],
    all_features_path: Path,
    feature_matrix_cache_path: Path,
    ic_report_path: Path,
    label_cache_path: Path,
    as_of: date,
    horizon: int,
    label_buffer_days: int,
    benchmark_ticker: str,
    rebalance_weekday: int,
    rolling_window_days: int,
    temperature: float,
    tree_n_jobs: int,
    random_state: int,
) -> dict[str, pd.Series]:
    retained_features = load_retained_features(ic_report_path)
    global_start = min(extract_window_dates(window)["train_start"] for window in comparison_windows)
    global_end = max(extract_window_dates(window)["test_end"] for window in comparison_windows)
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
        horizon=horizon,
        label_buffer_days=label_buffer_days,
        benchmark_ticker=benchmark_ticker,
        rebalance_weekday=rebalance_weekday,
    )
    aligned_X, aligned_y = align_panel(feature_matrix, labels)
    logger.info(
        "reconstructing daily gate series rows={} dates={} tickers={} features={}",
        len(aligned_X),
        aligned_X.index.get_level_values("trade_date").nunique(),
        aligned_X.index.get_level_values("ticker").nunique(),
        aligned_X.shape[1],
    )

    effective_rolling_points = max(1, int(ceil(rolling_window_days / 5)))
    fusion_ic_parts: list[pd.Series] = []
    ridge_ic_parts: list[pd.Series] = []
    top_decile_parts: list[pd.Series] = []
    turnover_parts: list[pd.Series] = []

    for position, (comparison_window, fusion_window) in enumerate(
        zip(comparison_windows, fusion_windows, strict=True),
        start=1,
    ):
        window_id = str(comparison_window.get("window_id", f"W{position}"))
        logger.info("reconstructing gate series for {} ({}/{})", window_id, position, len(comparison_windows))
        dates = extract_window_dates(comparison_window)
        train_X, train_y = slice_window(
            X=aligned_X,
            y=aligned_y,
            start_date=dates["train_start"],
            end_date=dates["train_end"],
            rebalance_weekday=rebalance_weekday,
        )
        validation_X, validation_y = slice_window(
            X=aligned_X,
            y=aligned_y,
            start_date=dates["validation_start"],
            end_date=dates["validation_end"],
            rebalance_weekday=rebalance_weekday,
        )
        test_X, test_y = slice_window(
            X=aligned_X,
            y=aligned_y,
            start_date=dates["test_start"],
            end_date=dates["test_end"],
            rebalance_weekday=rebalance_weekday,
        )

        combined_train_X = pd.concat([train_X, validation_X]).sort_index()
        combined_train_y = pd.concat([train_y, validation_y]).sort_index()
        model_names = ["ridge", "xgboost", "lightgbm"]
        raw_predictions: dict[str, pd.Series] = {}
        daily_ic_frame = pd.DataFrame(
            index=sorted(test_X.index.get_level_values("trade_date").unique()),
        )

        for model_name in model_names:
            params = extract_best_params(comparison_window, model_name)
            model = instantiate_model(
                model_name=model_name,
                best_params=params,
                tree_n_jobs=tree_n_jobs,
                random_state=random_state,
            )
            model.train(combined_train_X, combined_train_y)
            predictions = model.predict(test_X).rename(model_name)
            raw_predictions[model_name] = predictions
            daily_ic_frame[model_name] = information_coefficient_series(test_y, predictions).reindex(
                daily_ic_frame.index,
            )

        normalized_predictions = {
            model_name: cross_sectional_zscore(predictions)
            for model_name, predictions in raw_predictions.items()
        }
        prediction_frame = pd.concat(
            [test_y.rename("actual"), *[series.rename(name) for name, series in normalized_predictions.items()]],
            axis=1,
            join="inner",
        ).dropna(subset=["actual"])
        prediction_frame.sort_index(inplace=True)

        rolling_weights = build_rolling_weights(
            daily_ic_frame=daily_ic_frame[model_names],
            model_names=model_names,
            rolling_window=effective_rolling_points,
            temperature=temperature,
        )
        fusion_pred = combine_predictions(prediction_frame, rolling_weights, model_names).rename("fusion")
        fusion_ic_parts.append(information_coefficient_series(test_y, fusion_pred))
        ridge_ic_parts.append(information_coefficient_series(test_y, raw_predictions["ridge"]))
        top_decile_parts.append(per_date_top_decile_excess(test_y, fusion_pred))
        turnover_parts.append(per_date_turnover(fusion_pred))

    return {
        "fusion_daily_ic": pd.concat(fusion_ic_parts).sort_index(),
        "ridge_daily_ic": pd.concat(ridge_ic_parts).sort_index(),
        "top_decile_excess": pd.concat(top_decile_parts).sort_index(),
        "turnover_series": pd.concat(turnover_parts).sort_index(),
    }


def per_date_top_decile_excess(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    aligned = pd.concat(
        [pd.Series(y_true, dtype=float).rename("actual"), pd.Series(y_pred, dtype=float).rename("score")],
        axis=1,
        join="inner",
    ).dropna()
    values: list[tuple[pd.Timestamp, float]] = []
    for trade_date, frame in aligned.groupby(level="trade_date", sort=True):
        decile_size = max(1, int(np.ceil(len(frame) * 0.10)))
        selected = frame.nlargest(decile_size, "score")
        values.append((pd.Timestamp(trade_date), float(selected["actual"].mean())))
    return pd.Series({trade_date: value for trade_date, value in values}, dtype=float, name="top_decile_excess")


def per_date_turnover(y_pred: pd.Series) -> pd.Series:
    aligned = pd.Series(y_pred, dtype=float).dropna().sort_index()
    values: list[tuple[pd.Timestamp, float]] = []
    previous_selection: set[object] | None = None
    for trade_date, frame in aligned.groupby(level="trade_date", sort=True):
        decile_size = max(1, int(np.ceil(len(frame) * 0.10)))
        selected = set(frame.nlargest(decile_size).index.get_level_values("ticker"))
        if previous_selection is not None and previous_selection:
            overlap = len(previous_selection & selected)
            turnover = 1.0 - (overlap / len(previous_selection))
            values.append((pd.Timestamp(trade_date), float(turnover)))
        previous_selection = selected
    return pd.Series({trade_date: value for trade_date, value in values}, dtype=float, name="turnover")


def discover_tracking_uri(explicit_uri: str | None) -> str:
    candidates: list[str] = []
    if explicit_uri:
        candidates.append(normalize_tracking_uri(explicit_uri))
    try:
        candidates.append(normalize_tracking_uri(get_mlflow_tracking_uri()))
    except Exception:
        pass
    candidates.append("http://127.0.0.1:5001")
    local_sqlite = REPO_ROOT / "mlruns_r82.db"
    if local_sqlite.exists():
        candidates.append(normalize_tracking_uri(f"sqlite:///{local_sqlite.resolve().as_posix()}"))

    seen: set[str] = set()
    unique_candidates = [candidate for candidate in candidates if candidate and not (candidate in seen or seen.add(candidate))]

    for candidate in unique_candidates:
        try:
            runs = ExperimentTracker(tracking_uri=candidate).search_runs(max_results=25)
            if runs:
                logger.info("selected MLflow tracking URI {} with {} visible runs", candidate, len(runs))
                return candidate
            logger.warning("tracking URI {} is reachable but returned no runs", candidate)
        except Exception as exc:
            logger.warning("tracking URI {} is unavailable: {}", candidate, exc)

    raise RuntimeError("Unable to find a usable MLflow tracking URI with visible runs.")


def estimate_cost_adjusted_excess(
    *,
    top_decile_excess: pd.Series,
    turnover_series: pd.Series,
    horizon_days: int,
    cost_per_turnover_bps: float,
) -> dict[str, Any]:
    returns = pd.Series(top_decile_excess, dtype=float).dropna()
    if returns.empty:
        raise RuntimeError("No top-decile excess observations available for cost estimate.")

    if (returns <= -1.0).any():
        gross_annualized = -1.0
    else:
        log_mean = float(np.log1p(returns).mean())
        gross_annualized = float(np.expm1(log_mean * (252.0 / horizon_days)))

    mean_turnover = float(pd.Series(turnover_series, dtype=float).dropna().mean()) if len(turnover_series) else 0.0
    annual_turnover_cost = float(mean_turnover * 52.0 * (cost_per_turnover_bps / 10_000.0))
    net_value = float(gross_annualized - annual_turnover_cost)
    return {
        "value": net_value,
        "gross_annualized_excess": gross_annualized,
        "annual_turnover_cost": annual_turnover_cost,
        "mean_turnover": mean_turnover,
        "cost_per_turnover_bps": float(cost_per_turnover_bps),
        "n_obs": int(len(returns)),
        "method": "estimated_top_decile_60d_geometric_annualization_minus_turnover_cost",
    }


def classify_gate_result(total_passed: int, total_checks: int) -> str:
    if total_passed == total_checks:
        return "PASS"
    if total_passed >= total_checks - 1:
        return "CONDITIONAL_GO"
    return "FAIL"


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
            return None
        return item
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return {str(index): json_safe(item) for index, item in value.items()}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
