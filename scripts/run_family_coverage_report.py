from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.features.registry import build_feature_registry

try:
    from scripts._week6_family_utils import (
        DEFAULT_SAMPLE_DATES_PER_MONTH,
        DEFAULT_SAMPLE_TICKERS,
        SampleContext,
        build_feature_family_map,
        build_sample_context,
        collect_feature_observations,
        iter_month_labels,
        parse_date_arg,
        temporary_enable_week5_flags,
    )
except ModuleNotFoundError:  # pragma: no cover
    from _week6_family_utils import (
        DEFAULT_SAMPLE_DATES_PER_MONTH,
        DEFAULT_SAMPLE_TICKERS,
        SampleContext,
        build_feature_family_map,
        build_sample_context,
        collect_feature_observations,
        iter_month_labels,
        parse_date_arg,
        temporary_enable_week5_flags,
    )

DEFAULT_OUTPUT = Path("data/reports/family_coverage_report.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a monthly family coverage report from sampled feature panels.")
    parser.add_argument("--start-date", type=parse_date_arg, default=None)
    parser.add_argument("--end-date", type=parse_date_arg, default=None)
    parser.add_argument("--sample-tickers", type=int, default=DEFAULT_SAMPLE_TICKERS)
    parser.add_argument(
        "--sample-dates",
        dest="sample_dates_per_month",
        type=int,
        default=DEFAULT_SAMPLE_DATES_PER_MONTH,
        help="Trading dates sampled per month. Default: 1.",
    )
    parser.add_argument("--enable-flags", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def generate_family_coverage_report(
    *,
    context: SampleContext,
    observations: pd.DataFrame,
    family_by_feature: dict[str, str],
) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    monthly_labels = tuple(iter_month_labels(context.start_date, context.end_date))

    for family in sorted(set(family_by_feature.values())):
        family_features = sorted(name for name, feature_family in family_by_feature.items() if feature_family == family)
        family_obs = observations.loc[observations["family"] == family].copy()
        available_dates = family_obs.loc[~family_obs["is_missing"], "trade_date"]

        monthly_coverage: list[dict[str, Any]] = []
        for month in monthly_labels:
            month_obs = family_obs.loc[
                pd.to_datetime(family_obs["trade_date"]).dt.strftime("%Y-%m") == month
            ].copy()
            if month_obs.empty:
                monthly_coverage.append(
                    {
                        "month": month,
                        "missing_rate": 1.0,
                        "tickers_with_data": 0,
                    },
                )
                continue

            per_day_tickers = (
                month_obs.loc[~month_obs["is_missing"]]
                .groupby("trade_date")["ticker"]
                .nunique()
            )
            monthly_coverage.append(
                {
                    "month": month,
                    "missing_rate": round(float(month_obs["is_missing"].mean()), 6),
                    "tickers_with_data": int(round(float(per_day_tickers.mean()))) if not per_day_tickers.empty else 0,
                },
            )

        families[family] = {
            "feature_count": len(family_features),
            "monthly_coverage": monthly_coverage,
            "first_available_date": _serialize_date(available_dates.min() if not available_dates.empty else None),
            "last_available_date": _serialize_date(available_dates.max() if not available_dates.empty else None),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe_size": context.universe_size,
        "date_range": {
            "start": context.start_date.isoformat(),
            "end": context.end_date.isoformat(),
        },
        "families": families,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with temporary_enable_week5_flags(args.enable_flags):
        registry = build_feature_registry()
        family_by_feature = build_feature_family_map(registry)
        context = build_sample_context(
            start_date=args.start_date,
            end_date=args.end_date,
            sample_tickers=args.sample_tickers,
            sample_dates_per_month=args.sample_dates_per_month,
        )
        observations = collect_feature_observations(context, registry=registry)
        report = generate_family_coverage_report(
            context=context,
            observations=observations,
            family_by_feature=family_by_feature,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


def _serialize_date(value: Any) -> str | None:
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
