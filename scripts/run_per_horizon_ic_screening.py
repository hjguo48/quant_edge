# ruff: noqa: E402
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.registry import build_feature_registry

try:
    from scripts._week6_family_utils import temporary_enable_week5_flags
    from scripts._week7_ic_utils import (
        DEFAULT_HORIZON_FAMILIES_PATH,
        DEFAULT_MISSINGNESS_AUDIT_PATH,
        DEFAULT_PANEL_CACHE_PATH,
        DEFAULT_RETAINED_FEATURES_PATH,
        DEFAULT_REBALANCE_WEEKDAY,
        DEFAULT_SAMPLE_TICKERS,
        HORIZON_DAY_MAP,
        SCREENING_MEAN_IC_THRESHOLD,
        SCREENING_SIGN_WINDOW_THRESHOLD,
        SCREENING_T_STAT_THRESHOLD,
        build_feature_exclusion_map,
        build_or_load_week7_panel,
        build_panel_context,
        build_registry_feature_maps,
        build_retained_features_payload,
        build_wide_feature_matrix,
        compute_feature_screening_metrics,
        horizon_feature_names,
        load_horizon_families,
        load_label_series,
        load_missingness_exclusion_map,
        parse_date_arg,
        screening_status,
        screening_windows,
        summarize_panel_coverage,
        write_yaml_atomic,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _week6_family_utils import temporary_enable_week5_flags
    from _week7_ic_utils import (
        DEFAULT_HORIZON_FAMILIES_PATH,
        DEFAULT_MISSINGNESS_AUDIT_PATH,
        DEFAULT_PANEL_CACHE_PATH,
        DEFAULT_RETAINED_FEATURES_PATH,
        DEFAULT_REBALANCE_WEEKDAY,
        DEFAULT_SAMPLE_TICKERS,
        HORIZON_DAY_MAP,
        SCREENING_MEAN_IC_THRESHOLD,
        SCREENING_SIGN_WINDOW_THRESHOLD,
        SCREENING_T_STAT_THRESHOLD,
        build_feature_exclusion_map,
        build_or_load_week7_panel,
        build_panel_context,
        build_registry_feature_maps,
        build_retained_features_payload,
        build_wide_feature_matrix,
        compute_feature_screening_metrics,
        horizon_feature_names,
        load_horizon_families,
        load_label_series,
        load_missingness_exclusion_map,
        parse_date_arg,
        screening_status,
        screening_windows,
        summarize_panel_coverage,
        write_yaml_atomic,
    )


DEFAULT_OUTPUT_DIR = Path("data/reports/ic_v7")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Week 7 per-horizon IC screening on the sampled Friday panel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", type=parse_date_arg, default=None)
    parser.add_argument("--end-date", type=parse_date_arg, default=None)
    parser.add_argument("--sample-tickers", type=int, default=DEFAULT_SAMPLE_TICKERS)
    parser.add_argument("--rebalance-weekday", type=int, default=DEFAULT_REBALANCE_WEEKDAY)
    parser.add_argument("--enable-flags", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel-cache", type=Path, default=DEFAULT_PANEL_CACHE_PATH)
    parser.add_argument("--retained-output", type=Path, default=DEFAULT_RETAINED_FEATURES_PATH)
    parser.add_argument("--horizon-families", type=Path, default=DEFAULT_HORIZON_FAMILIES_PATH)
    parser.add_argument("--missingness-audit", type=Path, default=DEFAULT_MISSINGNESS_AUDIT_PATH)
    parser.add_argument("--mean-ic-threshold", type=float, default=SCREENING_MEAN_IC_THRESHOLD)
    parser.add_argument("--t-stat-threshold", type=float, default=SCREENING_T_STAT_THRESHOLD)
    parser.add_argument("--sign-window-threshold", type=int, default=SCREENING_SIGN_WINDOW_THRESHOLD)
    parser.add_argument(
        "--frozen-universe-path",
        type=Path,
        default=None,
        help="Path to a frozen ticker JSON (from scripts/freeze_universe.py). "
             "When set, build_sample_context skips its date-dependent universe "
             "fetcher and uses this list — required for chunked-merge "
             "equivalence so all chunks share one ticker set.",
    )
    return parser.parse_args(argv)


def screen_horizon_features(
    *,
    feature_matrix: pd.DataFrame,
    label_series: pd.Series,
    all_features: list[str],
    family_by_feature: dict[str, str],
    included_features: set[str],
    exclusion_map: dict[str, str],
    mean_ic_threshold: float,
    t_stat_threshold: float,
    sign_window_threshold: int,
    windows: list[Any],
    raw_ic_collector: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in all_features:
        family = family_by_feature[feature_name]
        excluded_reason = exclusion_map.get(feature_name)
        if excluded_reason is not None or feature_name not in included_features:
            rows.append(
                {
                    "feature": feature_name,
                    "family": family,
                    "mean_ic": None,
                    "t_stat": None,
                    "sign_consistent_windows": 0,
                    "status": "FAIL",
                    "excluded_reason": excluded_reason or "not_selected",
                },
            )
            continue

        feature_series = feature_matrix.get(feature_name)
        if feature_series is None:
            rows.append(
                {
                    "feature": feature_name,
                    "family": family,
                    "mean_ic": None,
                    "t_stat": None,
                    "sign_consistent_windows": 0,
                    "status": "FAIL",
                    "excluded_reason": "panel_source_missing",
                },
            )
            continue

        metrics = compute_feature_screening_metrics(
            feature_series=feature_series,
            label_series=label_series,
            windows=windows,
        )
        if raw_ic_collector is not None:
            daily_ic = metrics.get("daily_ic")
            if daily_ic is not None and not daily_ic.empty:
                for trade_dt, ic_value in daily_ic.items():
                    if pd.isna(ic_value):
                        continue
                    raw_ic_collector.append({
                        "feature": feature_name,
                        "family": family,
                        "trade_date": pd.Timestamp(trade_dt).date(),
                        "ic_value": float(ic_value),
                    })
        status = screening_status(
            mean_ic=metrics["mean_ic"],
            t_stat=metrics["t_stat"],
            sign_consistent_windows=metrics["sign_consistent_windows"],
            mean_ic_threshold=mean_ic_threshold,
            t_stat_threshold=t_stat_threshold,
            sign_window_threshold=sign_window_threshold,
        )
        rows.append(
            {
                "feature": feature_name,
                "family": family,
                "mean_ic": round(float(metrics["mean_ic"]), 6) if pd.notna(metrics["mean_ic"]) else None,
                "t_stat": round(float(metrics["t_stat"]), 6) if pd.notna(metrics["t_stat"]) else None,
                "sign_consistent_windows": int(metrics["sign_consistent_windows"]),
                "status": status,
                "excluded_reason": "",
            },
        )
    return pd.DataFrame(rows)


def csv_output_path(output_dir: Path, horizon_label: str) -> Path:
    return output_dir / f"ic_screening_v7_{horizon_label}.csv"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with temporary_enable_week5_flags(args.enable_flags):
        registry = build_feature_registry()
        family_by_feature, all_features = build_registry_feature_maps(registry)
        horizon_families = load_horizon_families(args.horizon_families)
        missingness_exclusions = load_missingness_exclusion_map(args.missingness_audit)
        context = build_panel_context(
            start_date=args.start_date,
            end_date=args.end_date,
            sample_tickers=args.sample_tickers,
            rebalance_weekday=args.rebalance_weekday,
            frozen_universe_path=args.frozen_universe_path,
        )

        union_features: set[str] = set()
        for horizon_label, config in horizon_families.items():
            union_features.update(
                horizon_feature_names(
                    registry=registry,
                    horizon_label=horizon_label,
                    included_families=config["families"],
                    exclusion_map=missingness_exclusions,
                ),
            )

        panel = build_or_load_week7_panel(
            context=context,
            feature_names=sorted(union_features),
            cache_path=args.panel_cache,
        )
        feature_matrix = build_wide_feature_matrix(panel, feature_names=sorted(union_features))

        report_rows_by_horizon: dict[str, pd.DataFrame] = {}
        for horizon_label, config in horizon_families.items():
            horizon_days = HORIZON_DAY_MAP[horizon_label]
            label_series = load_label_series(
                tickers=context.sampled_tickers,
                trade_dates=context.sampled_trade_dates,
                horizon_days=horizon_days,
            )
            included_features = set(
                horizon_feature_names(
                    registry=registry,
                    horizon_label=horizon_label,
                    included_families=config["families"],
                    exclusion_map=missingness_exclusions,
                ),
            )
            exclusion_map = build_feature_exclusion_map(
                all_features=all_features,
                family_by_feature=family_by_feature,
                horizon_label=horizon_label,
                included_families=config["families"],
                missingness_exclusions=missingness_exclusions,
            )
            raw_ic_rows: list[dict[str, Any]] = []
            rows = screen_horizon_features(
                feature_matrix=feature_matrix,
                label_series=label_series,
                all_features=all_features,
                family_by_feature=family_by_feature,
                included_features=included_features,
                exclusion_map=exclusion_map,
                mean_ic_threshold=args.mean_ic_threshold,
                t_stat_threshold=args.t_stat_threshold,
                sign_window_threshold=args.sign_window_threshold,
                windows=list(screening_windows()),
                raw_ic_collector=raw_ic_rows,
            )
            rows.to_csv(csv_output_path(output_dir, horizon_label), index=False)
            report_rows_by_horizon[horizon_label] = rows
            # Dump per-(feature, trade_date) raw IC for chunked-merge workflows.
            # Caller can concat these across chunks and recompute exact W7
            # mean_ic / t_stat / sign_consistent_windows via merge_chunked_ic.py.
            if raw_ic_rows:
                raw_path = output_dir / f"raw_ic_{horizon_label}.parquet"
                pd.DataFrame(raw_ic_rows).to_parquet(raw_path, index=False)

        retained_payload = build_retained_features_payload(report_rows_by_horizon=report_rows_by_horizon)
        retained_payload["generated_at"] = datetime.now(timezone.utc).isoformat()
        retained_payload["sample"] = {
            "start_date": context.start_date.isoformat(),
            "end_date": context.end_date.isoformat(),
            "sample_tickers": len(context.sampled_tickers),
            "sample_trade_dates": len(context.sampled_trade_dates),
            "universe_size": context.universe_size,
        }
        retained_payload["panel"] = summarize_panel_coverage(panel, sorted(union_features))
        write_yaml_atomic(args.retained_output, retained_payload)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
