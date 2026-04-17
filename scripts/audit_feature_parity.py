#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import (
    REPO_ROOT,
    REPORT_DATE_TAG,
    extract_feature_sets,
    get_engine,
    infer_feature_contract,
    load_current_state,
    load_family_registry,
    summarize_issues,
    write_json_report,
)


def load_feature_store_stats(engine) -> tuple[dict[str, dict[str, Any]], str | None, str | None]:
    query = text(
        """
        select
            feature_name,
            count(*) as row_count,
            avg(case when feature_value is null then 1.0 else 0.0 end) as null_rate,
            min(calc_date) as min_date,
            max(calc_date) as max_date,
            bool_or(coalesce(is_filled, false)) as any_filled
        from feature_store
        group by feature_name
        order by feature_name
        """
    )
    bounds_query = text(
        """
        select min(calc_date) as min_date, max(calc_date) as max_date
        from feature_store
        """
    )
    with engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
        bounds = conn.execute(bounds_query).mappings().first()
    stats = {
        row["feature_name"]: {
            "row_count": int(row["row_count"]),
            "null_rate": float(row["null_rate"] or 0.0),
            "min_date": row["min_date"],
            "max_date": row["max_date"],
            "dtype": "numeric(20,8)",
            "any_filled": bool(row["any_filled"]),
        }
        for row in rows
    }
    return stats, bounds["min_date"], bounds["max_date"]


def get_parquet_date_range(parquet_path: Path) -> tuple[Any, Any]:
    parquet = pq.ParquetFile(parquet_path)
    mins = []
    maxs = []
    for idx in range(parquet.metadata.num_row_groups):
        stats = parquet.metadata.row_group(idx).column(1).statistics
        if stats is not None:
            mins.append(stats.min)
            maxs.append(stats.max)
    return min(mins), max(maxs)


def scan_parquet_stats(
    parquet_path: Path,
    min_date,
    max_date,
    *,
    overlap_exists: bool,
    sample_trade_date=None,
) -> dict[str, dict[str, Any]]:
    dataset = ds.dataset(parquet_path, format="parquet")
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"row_count": 0, "null_count": 0, "sample_dtype": "double"}
    )
    if overlap_exists:
        scanner = dataset.scanner(
            columns=["trade_date", "feature_name", "feature_value"],
            filter=(ds.field("trade_date") >= min_date) & (ds.field("trade_date") <= max_date),
            batch_size=250_000,
        )
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if frame.empty:
                continue
            grouped = frame.groupby("feature_name", dropna=False)["feature_value"]
            counts = grouped.size()
            null_counts = grouped.apply(lambda s: int(s.isna().sum()))
            for feature_name, row_count in counts.items():
                stats[str(feature_name)]["row_count"] += int(row_count)
                stats[str(feature_name)]["null_count"] += int(null_counts.loc[feature_name])
    else:
        scanner = dataset.scanner(
            columns=["feature_name"],
            filter=ds.field("trade_date") == sample_trade_date,
            batch_size=250_000,
        )
        for batch in scanner.to_batches():
            names = set(batch.column("feature_name").to_pylist())
            for feature_name in names:
                stats[str(feature_name)]["row_count"] = None
                stats[str(feature_name)]["null_count"] = None

    normalized = {}
    for feature_name, row in stats.items():
        row_count = row["row_count"]
        normalized[feature_name] = {
            "row_count": int(row_count) if row_count is not None else None,
            "null_rate": float(row["null_count"] / row_count) if row_count else None,
            "dtype": row["sample_dtype"],
        }
    return normalized


def build_report(parquet_path: Path, output_path: Path) -> dict[str, Any]:
    current_state = load_current_state()
    family_registry = load_family_registry()
    family_map = family_registry["feature_to_family"]
    feature_sets = extract_feature_sets()
    expected_features = set(family_map.keys())

    engine = get_engine()
    feature_store_stats, min_date, max_date = load_feature_store_stats(engine)
    parquet_min_date, parquet_max_date = get_parquet_date_range(parquet_path)
    overlap_exists = not (parquet_max_date < min_date or parquet_min_date > max_date)
    parquet_stats = scan_parquet_stats(
        parquet_path,
        min_date,
        max_date,
        overlap_exists=overlap_exists,
        sample_trade_date=parquet_max_date,
    )

    feature_store_features = set(feature_store_stats.keys())
    parquet_features = set(parquet_stats.keys())
    all_features = sorted(expected_features | feature_store_features | parquet_features)

    bundle_path = REPO_ROOT / current_state["live_champion"]["bundle_path"]
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    live_bundle_features = bundle["retained_features"]
    bundle_missing = sorted([feature for feature in live_bundle_features if feature not in feature_store_features])

    issues: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if bundle_missing:
        issues.append(
            {
                "severity": "critical",
                "code": "live_bundle_feature_missing_from_feature_store",
                "message": "当前 live champion bundle 的部分 retained features 不存在于 feature_store。",
                "details": bundle_missing,
            }
        )
    if not overlap_exists:
        warnings.append(
            {
                "severity": "warning",
                "code": "parquet_snapshot_stale_vs_live_store",
                "message": "研究 parquet 与 live feature_store 没有日期重叠，说明当前研究快照比 live 特征落后。",
                "parquet_date_range": [parquet_min_date, parquet_max_date],
                "feature_store_date_range": [min_date, max_date],
            }
        )

    only_in_feature_store = sorted(feature_store_features - parquet_features)
    only_in_parquet = sorted(parquet_features - feature_store_features)
    if only_in_feature_store:
        warnings.append(
            {
                "severity": "warning",
                "code": "legacy_live_only_features",
                "message": "feature_store 中存在研究 parquet 不含的 live-only 或历史遗留特征。",
                "count": len(only_in_feature_store),
            }
        )
    if only_in_parquet:
        warnings.append(
            {
                "severity": "warning",
                "code": "research_only_features",
                "message": "研究 parquet 中存在 live feature_store 当前未落地的特征。",
                "count": len(only_in_parquet),
            }
        )

    feature_rows = []
    mismatched = []
    gap_report: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "shared": 0,
            "only_in_feature_store": 0,
            "only_in_parquet": 0,
            "mismatched": 0,
        }
    )

    for feature_name in all_features:
        contract = infer_feature_contract(feature_name, feature_sets, family_map)
        store_row = feature_store_stats.get(feature_name)
        parquet_row = parquet_stats.get(feature_name)
        exists_in_feature_store = store_row is not None
        exists_in_parquet = parquet_row is not None
        null_rate_store = store_row["null_rate"] if store_row else None
        null_rate_parquet = parquet_row["null_rate"] if parquet_row else None
        null_rate_delta = (
            abs(null_rate_store - null_rate_parquet)
            if null_rate_store is not None and null_rate_parquet is not None
            else None
        )
        matches = bool(
            exists_in_feature_store
            and exists_in_parquet
            and (null_rate_delta is None or null_rate_delta <= 0.15)
        )
        family = contract["family"]
        gap_report[family]["total"] += 1
        if exists_in_feature_store and exists_in_parquet:
            gap_report[family]["shared"] += 1
        elif exists_in_feature_store:
            gap_report[family]["only_in_feature_store"] += 1
        elif exists_in_parquet:
            gap_report[family]["only_in_parquet"] += 1
        if exists_in_feature_store and exists_in_parquet and not matches:
            gap_report[family]["mismatched"] += 1
            mismatched.append(
                {
                    "feature_name": feature_name,
                    "family": family,
                    "db_null_rate": null_rate_store,
                    "parquet_null_rate": null_rate_parquet,
                    "null_rate_delta": null_rate_delta,
                }
            )

        feature_rows.append(
            {
                "name": feature_name,
                "family": family,
                "domain": contract["domain"],
                "dtype": {
                    "feature_store": store_row["dtype"] if store_row else None,
                    "parquet": parquet_row["dtype"] if parquet_row else None,
                },
                "null_rate": {
                    "feature_store": null_rate_store,
                    "parquet": null_rate_parquet,
                },
                "lag_rule": contract["lag_rule"],
                "pit_rule": contract["pit_rule"],
                "source_table": contract["source_tables"],
                "exists_in_feature_store": exists_in_feature_store,
                "exists_in_parquet": exists_in_parquet,
                "matches": matches,
                "live_bundle_required": feature_name in live_bundle_features,
            }
        )

    payload = {
        "metadata": {
            "说明": "对比当前 live feature_store 与研究 parquet 在相同近期日期窗口内的特征覆盖与一致性。",
            "parquet_path": str(parquet_path.relative_to(REPO_ROOT)),
            "parity_window": {"start": min_date, "end": max_date, "overlap_exists": overlap_exists},
            "parquet_date_range": {"start": parquet_min_date, "end": parquet_max_date},
            "current_state": "configs/research/current_state.yaml",
            "family_registry": "configs/research/family_registry.yaml",
        },
        "summary": {
            "expected_feature_count": len(expected_features),
            "feature_store_feature_count": len(feature_store_features),
            "parquet_feature_count": len(parquet_features),
            "bundle_feature_count": len(live_bundle_features),
            "bundle_missing_feature_count": len(bundle_missing),
        },
        "issues": issues,
        "warnings": warnings,
        "bundle_missing_features": bundle_missing,
        "only_in_feature_store": only_in_feature_store,
        "only_in_parquet": only_in_parquet,
        "mismatched": mismatched,
        "gap_report": dict(sorted(gap_report.items())),
        "features": feature_rows,
    }
    write_json_report(payload, output_path)
    critical_count, warning_count = summarize_issues(issues + warnings)
    print(f"[feature_parity] wrote {output_path}")
    print(
        f"[feature_parity] store={len(feature_store_features)} parquet={len(parquet_features)} "
        f"bundle_missing={len(bundle_missing)} critical={critical_count} warnings={warning_count}"
    )
    if bundle_missing:
        print("[feature_parity] ALERT bundle features missing from feature_store:", ", ".join(bundle_missing))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit feature parity between live feature_store and research parquet.")
    parser.add_argument(
        "--parquet-path",
        default="data/features/all_features_v5.parquet",
        help="Research feature parquet to compare against the live feature_store.",
    )
    parser.add_argument(
        "--output",
        default=f"data/reports/feature_parity_audit_{REPORT_DATE_TAG}.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(REPO_ROOT / args.parquet_path, REPO_ROOT / args.output)


if __name__ == "__main__":
    main()
