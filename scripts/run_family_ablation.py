# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.backtest.engine import slice_panel
from src.features.registry import build_feature_registry
from src.models.baseline import DEFAULT_ALPHA_GRID, RidgeBaselineModel
from src.models.evaluation import information_coefficient_series

try:
    from scripts._week6_family_utils import temporary_enable_week5_flags
    from scripts._week7_ic_utils import (
        DEFAULT_HORIZON_FAMILIES_PATH,
        DEFAULT_MISSINGNESS_AUDIT_PATH,
        DEFAULT_PANEL_CACHE_PATH,
        DEFAULT_REBALANCE_WEEKDAY,
        DEFAULT_SAMPLE_TICKERS,
        HORIZON_DAY_MAP,
        build_or_load_week7_panel,
        build_panel_context,
        build_registry_feature_maps,
        build_wide_feature_matrix,
        fill_feature_matrix,
        horizon_feature_names,
        load_horizon_families,
        load_label_series,
        load_missingness_exclusion_map,
        parse_date_arg,
        screening_windows,
        series_t_stat,
        summarize_panel_coverage,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _week6_family_utils import temporary_enable_week5_flags
    from _week7_ic_utils import (
        DEFAULT_HORIZON_FAMILIES_PATH,
        DEFAULT_MISSINGNESS_AUDIT_PATH,
        DEFAULT_PANEL_CACHE_PATH,
        DEFAULT_REBALANCE_WEEKDAY,
        DEFAULT_SAMPLE_TICKERS,
        HORIZON_DAY_MAP,
        build_or_load_week7_panel,
        build_panel_context,
        build_registry_feature_maps,
        build_wide_feature_matrix,
        fill_feature_matrix,
        horizon_feature_names,
        load_horizon_families,
        load_label_series,
        load_missingness_exclusion_map,
        parse_date_arg,
        screening_windows,
        series_t_stat,
        summarize_panel_coverage,
    )


DEFAULT_OUTPUT_DIR = Path("data/reports")
DEFAULT_SCREENING_DIR = Path("data/reports/ic_v7")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 7 only-one-family and leave-one-family-out ridge ablations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", type=parse_date_arg, default=None)
    parser.add_argument("--end-date", type=parse_date_arg, default=None)
    parser.add_argument("--sample-tickers", type=int, default=DEFAULT_SAMPLE_TICKERS)
    parser.add_argument("--rebalance-weekday", type=int, default=DEFAULT_REBALANCE_WEEKDAY)
    parser.add_argument("--horizons", default="1d,5d,20d,60d")
    parser.add_argument("--enable-flags", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--screening-dir", type=Path, default=DEFAULT_SCREENING_DIR)
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE_PATH)
    parser.add_argument("--horizon-families", type=Path, default=DEFAULT_HORIZON_FAMILIES_PATH)
    parser.add_argument("--missingness-audit", type=Path, default=DEFAULT_MISSINGNESS_AUDIT_PATH)
    return parser.parse_args(argv)


def parse_horizon_labels(value: str) -> list[str]:
    labels = []
    for raw in str(value).split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item.isdigit():
            item = f"{item}d"
        if item not in HORIZON_DAY_MAP:
            raise ValueError(f"Unsupported horizon {raw!r}. Expected one of {sorted(HORIZON_DAY_MAP)}.")
        labels.append(item)
    if not labels:
        raise ValueError("At least one horizon must be selected.")
    return list(dict.fromkeys(labels))


def build_family_feature_sets(
    *,
    retained_rows: pd.DataFrame,
) -> dict[str, list[str]]:
    passed = retained_rows.loc[retained_rows["status"] == "PASS", ["feature", "family"]].copy()
    return {
        str(family): sorted(group["feature"].astype(str).tolist())
        for family, group in passed.groupby("family", sort=True)
    }


def run_feature_subset_walkforward(
    *,
    feature_matrix: pd.DataFrame,
    label_series: pd.Series,
    horizon_label: str,
    rebalance_weekday: int = DEFAULT_REBALANCE_WEEKDAY,
) -> dict[str, Any]:
    if feature_matrix.empty:
        return {
            "ic": None,
            "t_stat": None,
            "window_count": 0,
            "feature_count": 0,
            "per_window_ic": {},
        }
    predictions: list[pd.Series] = []
    per_window_ic: dict[str, float] = {}
    for window in screening_windows():
        try:
            train_X, train_y = slice_panel(
                X=feature_matrix,
                y=label_series,
                start_date=window.train_start,
                end_date=window.train_end,
                rebalance_weekday=rebalance_weekday,
            )
            validation_X, validation_y = slice_panel(
                X=feature_matrix,
                y=label_series,
                start_date=window.validation_start,
                end_date=window.validation_end,
                rebalance_weekday=rebalance_weekday,
            )
            test_X, test_y = slice_panel(
                X=feature_matrix,
                y=label_series,
                start_date=window.test_start,
                end_date=window.test_end,
                rebalance_weekday=rebalance_weekday,
            )
        except RuntimeError:
            continue

        selector = RidgeBaselineModel(alpha_grid=DEFAULT_ALPHA_GRID)
        selection = selector.select_alpha(train_X, train_y, validation_X, validation_y)

        final_train_X = pd.concat([train_X, validation_X]).sort_index()
        final_train_y = pd.concat([train_y, validation_y]).sort_index()
        model = RidgeBaselineModel(alpha=selection.best_hyperparams, alpha_grid=DEFAULT_ALPHA_GRID)
        model.train(final_train_X, final_train_y)
        test_predictions = model.predict(test_X)

        predictions.append(test_predictions)
        window_ic = information_coefficient_series(test_y, test_predictions).mean()
        if pd.notna(window_ic):
            per_window_ic[window.window_id] = float(window_ic)

    if not predictions:
        return {
            "ic": None,
            "t_stat": None,
            "window_count": 0,
            "feature_count": int(feature_matrix.shape[1]),
            "per_window_ic": {},
        }
    combined_predictions = pd.concat(predictions).sort_index()
    aligned_truth = label_series.loc[combined_predictions.index]
    ic_series = information_coefficient_series(aligned_truth, combined_predictions)
    mean_ic = float(ic_series.mean()) if not ic_series.empty else None
    t_stat = series_t_stat(ic_series)
    return {
        "ic": None if mean_ic is None or pd.isna(mean_ic) else round(mean_ic, 6),
        "t_stat": None if pd.isna(t_stat) else round(float(t_stat), 6),
        "window_count": len(per_window_ic),
        "feature_count": int(feature_matrix.shape[1]),
        "per_window_ic": {window_id: round(value, 6) for window_id, value in per_window_ic.items()},
    }


def rank_only_one_family(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda item: (-float(item["ic"]), item["family"]) if item["ic"] is not None else (float("inf"), item["family"]),
    )
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx if item["ic"] is not None else None
    return ranked


def rank_leave_one_family_out(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda item: (float(item["ic_delta"]), item["family"]) if item["ic_delta"] is not None else (float("inf"), item["family"]),
    )
    for idx, item in enumerate(ranked, start=1):
        item["rank"] = idx if item["ic_delta"] is not None else None
    return ranked


def load_retained_rows(screening_dir: Path, horizon_label: str) -> pd.DataFrame:
    path = screening_dir / f"ic_screening_v7_{horizon_label}.csv"
    if not path.exists():
        raise RuntimeError(f"Missing screening output for {horizon_label}: {path}")
    return pd.read_csv(path)


def generate_family_ablation_report(
    *,
    horizon_label: str,
    feature_matrix: pd.DataFrame,
    label_series: pd.Series,
    retained_rows: pd.DataFrame,
    rebalance_weekday: int = DEFAULT_REBALANCE_WEEKDAY,
) -> dict[str, Any]:
    family_sets = build_family_feature_sets(retained_rows=retained_rows)
    retained_features = retained_rows.loc[retained_rows["status"] == "PASS", "feature"].astype(str).tolist()
    baseline_metrics = run_feature_subset_walkforward(
        feature_matrix=feature_matrix.loc[:, retained_features],
        label_series=label_series,
        horizon_label=horizon_label,
        rebalance_weekday=rebalance_weekday,
    )
    baseline_ic = baseline_metrics["ic"]

    only_one_rows: list[dict[str, Any]] = []
    leave_one_rows: list[dict[str, Any]] = []
    full_feature_set = set(retained_features)

    for family, family_features in family_sets.items():
        subset = [feature for feature in retained_features if feature in family_features]
        if subset:
            metrics = run_feature_subset_walkforward(
                feature_matrix=feature_matrix.loc[:, subset],
                label_series=label_series,
                horizon_label=horizon_label,
                rebalance_weekday=rebalance_weekday,
            )
            only_one_rows.append(
                {
                    "family": family,
                    "feature_count": len(subset),
                    "ic": metrics["ic"],
                    "t_stat": metrics["t_stat"],
                    "window_count": metrics["window_count"],
                },
            )
        else:
            only_one_rows.append(
                {
                    "family": family,
                    "feature_count": 0,
                    "ic": None,
                    "t_stat": None,
                    "window_count": 0,
                },
            )

        lofo_features = [feature for feature in retained_features if feature not in set(family_features)]
        if not lofo_features:
            leave_one_rows.append(
                {
                    "family": family,
                    "feature_count_removed": len(family_features),
                    "feature_count_remaining": 0,
                    "ic": None,
                    "t_stat": None,
                    "ic_delta": None,
                    "window_count": 0,
                },
            )
            continue
        metrics = run_feature_subset_walkforward(
            feature_matrix=feature_matrix.loc[:, lofo_features],
            label_series=label_series,
            horizon_label=horizon_label,
            rebalance_weekday=rebalance_weekday,
        )
        ic_value = metrics["ic"]
        ic_delta = None
        if baseline_ic is not None and ic_value is not None:
            ic_delta = round(float(ic_value) - float(baseline_ic), 6)
        leave_one_rows.append(
            {
                "family": family,
                "feature_count_removed": len(full_feature_set & set(family_features)),
                "feature_count_remaining": len(lofo_features),
                "ic": ic_value,
                "t_stat": metrics["t_stat"],
                "ic_delta": ic_delta,
                "window_count": metrics["window_count"],
            },
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "horizon": horizon_label,
        "model_type": "ridge_baseline",
        "baseline_full_ic": baseline_ic,
        "baseline_full_t_stat": baseline_metrics["t_stat"],
        "baseline_window_count": baseline_metrics["window_count"],
        "retained_feature_count": len(retained_features),
        "only_one_family": rank_only_one_family(only_one_rows),
        "leave_one_family_out": rank_leave_one_family_out(leave_one_rows),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_horizon_labels(args.horizons)

    with temporary_enable_week5_flags(args.enable_flags):
        registry = build_feature_registry()
        family_by_feature, _ = build_registry_feature_maps(registry)
        horizon_families = load_horizon_families(args.horizon_families)
        missingness_exclusions = load_missingness_exclusion_map(args.missingness_audit)
        context = build_panel_context(
            start_date=args.start_date,
            end_date=args.end_date,
            sample_tickers=args.sample_tickers,
            rebalance_weekday=args.rebalance_weekday,
        )

        union_features: set[str] = set()
        for horizon_label in horizons:
            union_features.update(
                horizon_feature_names(
                    registry=registry,
                    horizon_label=horizon_label,
                    included_families=horizon_families[horizon_label]["families"],
                    exclusion_map=missingness_exclusions,
                ),
            )
        panel = build_or_load_week7_panel(
            context=context,
            feature_names=sorted(union_features),
            cache_path=args.panel_cache,
        )
        matrix = build_wide_feature_matrix(panel, feature_names=sorted(union_features))
        payload_summary = summarize_panel_coverage(panel, sorted(union_features))

        for horizon_label in horizons:
            retained_rows = load_retained_rows(args.screening_dir, horizon_label)
            retained_features = retained_rows.loc[retained_rows["status"] == "PASS", "feature"].astype(str).tolist()
            filled_matrix = fill_feature_matrix(matrix.reindex(columns=retained_features))
            label_series = load_label_series(
                tickers=context.sampled_tickers,
                trade_dates=context.sampled_trade_dates,
                horizon_days=HORIZON_DAY_MAP[horizon_label],
            )
            report = generate_family_ablation_report(
                horizon_label=horizon_label,
                feature_matrix=filled_matrix,
                label_series=label_series,
                retained_rows=retained_rows,
                rebalance_weekday=args.rebalance_weekday,
            )
            report["sample"] = {
                "start_date": context.start_date.isoformat(),
                "end_date": context.end_date.isoformat(),
                "sample_tickers": len(context.sampled_tickers),
                "sample_trade_dates": len(context.sampled_trade_dates),
                "universe_size": context.universe_size,
            }
            report["panel"] = payload_summary
            write_json(output_dir / f"family_ablation_{horizon_label}.json", report)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
