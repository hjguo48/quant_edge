from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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
        parse_date_arg,
        temporary_enable_week5_flags,
    )

DEFAULT_OUTPUT = Path("data/reports/missingness_audit.json")
VENDOR_RELATED_CATEGORIES = frozenset({"vendor_bias", "data_source_block"})

FEATURE_RULE_OVERRIDES: dict[str, dict[str, str]] = {
    "abnormal_off_exchange_shorting": {
        "category": "data_source_block",
        "rationale": "ADF off-exchange coverage depends on FINRA public CDN access; persistent gaps reflect a blocked data path, not issuer economics.",
        "recommendation": "drop_pending_paid_subscription",
    },
    "target_price_drift": {
        "category": "real_sparsity",
        "rationale": "This signal needs repeated analyst target revisions inside a 60-day lookback, so missingness mostly reflects genuine analyst coverage sparsity.",
        "recommendation": "keep",
    },
}

DEFAULT_FAMILY_RATIONALES = {
    "analyst": "Coverage depends on live analyst participation and is structurally sparse outside large-cap names.",
    "analyst_proxy": "These features require enough recent analyst actions or target updates, so gaps are economically meaningful coverage sparsity.",
    "earnings": "Coverage is materially thinner in older history; missingness is mostly an era effect rather than a broken feature definition.",
    "fundamental": "Fundamental data arrives on filing cadence and is thinner for IPOs and younger issuers.",
    "insider": "Insider activity is episodic, so missingness often reflects the absence of recent transactions.",
    "sec_filing": "SEC filing features are event-driven, and missingness often reflects no recent filing activity or younger listing history.",
    "short_interest": "Short-interest coverage is structurally sparser than price-based families and varies with exchange and vendor coverage.",
    "shorting": "FINRA short-sale features should usually be dense when venue files are available; large gaps are suspicious unless the feature is ADF-specific.",
}

DENSE_FAMILIES = frozenset(
    {
        "composite",
        "daily",
        "intraday",
        "macro",
        "sector_rotation",
        "technical",
        "trade_microstructure",
    },
)
SPARSE_BUT_REAL_FAMILIES = frozenset({"analyst", "analyst_proxy", "insider", "short_interest"})
TEMPORAL_FAMILIES = frozenset({"fundamental", "sec_filing"})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a per-feature missingness audit on a sampled Week 6 panel.")
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


def generate_missingness_audit(
    *,
    context: SampleContext,
    observations: pd.DataFrame,
    family_by_feature: dict[str, str],
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    feature_names = sorted(family_by_feature)
    for feature_name in feature_names:
        family = family_by_feature[feature_name]
        feature_obs = observations.loc[observations["feature_name"] == feature_name]
        missing_rate = float(feature_obs["is_missing"].mean()) if not feature_obs.empty else 1.0
        classification = classify_missingness(feature_name=feature_name, family=family, missing_rate=missing_rate)
        features.append(
            {
                "feature": feature_name,
                "family": family,
                "missing_rate": round(missing_rate, 6),
                "category": classification["category"],
                "rationale": classification["rationale"],
                "recommendation": classification["recommendation"],
            },
        )

    keep_count = sum(1 for item in features if item["recommendation"] == "keep")
    drop_count = sum(1 for item in features if str(item["recommendation"]).startswith("drop"))
    convert_count = sum(1 for item in features if item["recommendation"] == "convert_to_imputed")
    vendor_bias_count = sum(1 for item in features if item["category"] in VENDOR_RELATED_CATEGORIES)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date_range": {
            "start": context.start_date.isoformat(),
            "end": context.end_date.isoformat(),
        },
        "universe_size": context.universe_size,
        "features": features,
        "summary": {
            "keep": keep_count,
            "drop": drop_count,
            "convert_to_imputed": convert_count,
            "vendor_bias": vendor_bias_count,
        },
    }


def classify_missingness(*, feature_name: str, family: str, missing_rate: float) -> dict[str, str]:
    override = FEATURE_RULE_OVERRIDES.get(feature_name)
    if override is not None:
        return override

    if family == "earnings":
        if missing_rate < 0.10:
            return {
                "category": "real",
                "rationale": "Recent earnings observations are mostly present; residual gaps do not dominate the feature.",
                "recommendation": "keep",
            }
        return {
            "category": "era",
            "rationale": DEFAULT_FAMILY_RATIONALES[family],
            "recommendation": "convert_to_imputed",
        }

    if family in DENSE_FAMILIES:
        if missing_rate < 0.10:
            return {
                "category": "real",
                "rationale": "This price-linked family is broadly available in the sample and the remaining gaps are not economically concerning.",
                "recommendation": "keep",
            }
        if missing_rate < 0.40:
            return {
                "category": "temporal",
                "rationale": "This family is mostly dense, so moderate missingness is more consistent with startup periods and local coverage gaps than a structural signal.",
                "recommendation": "convert_to_imputed",
            }
        return {
            "category": "vendor_bias",
            "rationale": "A dense family should not be this sparse. The gap profile is more consistent with a vendor or ingestion coverage issue than economics.",
            "recommendation": "drop",
        }

    if family in SPARSE_BUT_REAL_FAMILIES:
        if missing_rate < 0.10:
            return {
                "category": "real",
                "rationale": "Observed coverage is already high, so the remaining gaps are not material.",
                "recommendation": "keep",
            }
        if missing_rate < 0.40:
            return {
                "category": "temporal",
                "rationale": DEFAULT_FAMILY_RATIONALES[family],
                "recommendation": "convert_to_imputed",
            }
        if missing_rate < 0.80:
            return {
                "category": "real_sparsity",
                "rationale": DEFAULT_FAMILY_RATIONALES[family],
                "recommendation": "keep",
            }
        return {
            "category": "real_sparsity",
            "rationale": DEFAULT_FAMILY_RATIONALES[family],
            "recommendation": "convert_to_imputed",
        }

    if family in TEMPORAL_FAMILIES:
        if missing_rate < 0.10:
            return {
                "category": "real",
                "rationale": "The filing-backed signal is mostly available in the sample.",
                "recommendation": "keep",
            }
        if missing_rate < 0.80:
            return {
                "category": "temporal",
                "rationale": DEFAULT_FAMILY_RATIONALES[family],
                "recommendation": "convert_to_imputed",
            }
        return {
            "category": "era",
            "rationale": DEFAULT_FAMILY_RATIONALES[family],
            "recommendation": "convert_to_imputed",
        }

    if family == "shorting":
        if missing_rate < 0.10:
            return {
                "category": "real",
                "rationale": "The main FINRA venue files are broadly available, so the feature is usable as-is.",
                "recommendation": "keep",
            }
        if missing_rate < 0.40:
            return {
                "category": "temporal",
                "rationale": DEFAULT_FAMILY_RATIONALES[family],
                "recommendation": "convert_to_imputed",
            }
        return {
            "category": "vendor_bias",
            "rationale": DEFAULT_FAMILY_RATIONALES[family],
            "recommendation": "drop",
        }

    if missing_rate < 0.10:
        return {
            "category": "real",
            "rationale": "Missingness is low enough that the feature behaves like a normal retained signal.",
            "recommendation": "keep",
        }
    if missing_rate < 0.40:
        return {
            "category": "temporal",
            "rationale": "Moderate missingness is consistent with event timing or listing-history gaps.",
            "recommendation": "convert_to_imputed",
        }
    if missing_rate < 0.80:
        return {
            "category": "vendor_bias",
            "rationale": "The feature is sparse enough that coverage quality, not economics, is the main concern.",
            "recommendation": "drop",
        }
    return {
        "category": "vendor_bias",
        "rationale": "The feature is effectively absent on the sampled panel and is not reliable enough to retain.",
        "recommendation": "drop",
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
        report = generate_missingness_audit(
            context=context,
            observations=observations,
            family_by_feature=family_by_feature,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
