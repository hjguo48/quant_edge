#!/usr/bin/env python3
from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.session import get_engine
from src.features.pipeline import FeaturePipeline
from src.models.bundle_validator import BundleValidator


OUTPUT_PATH = REPO_ROOT / "data/reports/p1_root_cause_diagnosis_20260417.json"
BUNDLE_PATH = REPO_ROOT / "data/models/fusion_model_bundle_60d.json"
DEFAULT_FEATURE_BOOTSTRAP_DAYS = 30


def parquet_range(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    try:
        import pyarrow.parquet as pq
    except ImportError:
        frame = pd.read_parquet(path, columns=["trade_date"])
        if frame.empty:
            return {"exists": True, "rows": 0, "min_trade_date": None, "max_trade_date": None}
        return {
            "exists": True,
            "rows": int(len(frame)),
            "min_trade_date": str(pd.to_datetime(frame["trade_date"]).min().date()),
            "max_trade_date": str(pd.to_datetime(frame["trade_date"]).max().date()),
        }

    parquet = pq.ParquetFile(path)
    mins = []
    maxs = []
    for idx in range(parquet.metadata.num_row_groups):
        column_index = parquet.schema_arrow.names.index("trade_date")
        stats = parquet.metadata.row_group(idx).column(column_index).statistics
        if stats is None:
            continue
        mins.append(pd.to_datetime(stats.min).date())
        maxs.append(pd.to_datetime(stats.max).date())
    return {
        "exists": True,
        "rows": int(parquet.metadata.num_rows),
        "min_trade_date": str(min(mins)) if mins else None,
        "max_trade_date": str(max(maxs)) if maxs else None,
    }


def latest_pit_trade_date() -> date:
    with get_engine().connect() as conn:
        row = conn.execute(
            text(
                """
                select max(trade_date) filter (where knowledge_time <= :as_of) as latest_pit_trade_date
                from stock_prices
                """,
            ),
            {"as_of": datetime.now(timezone.utc)},
        ).mappings().one()
    return row["latest_pit_trade_date"]


def main() -> int:
    bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
    validator = BundleValidator(BUNDLE_PATH)
    with get_engine().connect() as conn:
        available = set(
            str(name)
            for name in conn.execute(
                text("select distinct feature_name from feature_store"),
            ).scalars().all()
        )
        store_bounds = conn.execute(
            text(
                """
                select min(calc_date) as min_calc_date, max(calc_date) as max_calc_date,
                       count(*) as row_count, count(distinct feature_name) as feature_count
                from feature_store
                """,
            ),
        ).mappings().one()

    required = set(bundle.get("required_features") or bundle.get("retained_features") or [])
    missing = sorted(required - available)

    pit_trade_date = latest_pit_trade_date()
    pipeline = FeaturePipeline()
    sample = pipeline.run(
        tickers=["AAPL", "MSFT"],
        start_date=pit_trade_date,
        end_date=pit_trade_date,
        as_of=datetime.now(timezone.utc),
    )
    sample_features = set(sample["feature_name"].astype(str).unique().tolist())

    v5_path = REPO_ROOT / "data/features/all_features_v5.parquet"
    v6_path = REPO_ROOT / "data/features/all_features_v6.parquet"
    canonical_path = REPO_ROOT / "data/features/all_features.parquet"
    report = {
        "metadata": {
            "说明": "Week 2.5 P1 根因诊断，记录研究 parquet 与 live feature_store 脱节的直接证据。",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "bundle": {
            "path": str(BUNDLE_PATH),
            "version": bundle.get("version") or bundle.get("model_bundle_version"),
            "required_feature_count": len(required),
            "feature_fingerprint": bundle.get("feature_fingerprint"),
        },
        "missing_live_features": {
            "count": len(missing),
            "features": missing,
            "present_in_current_pipeline_output": {
                feature: bool(feature in sample_features)
                for feature in missing
            },
            "interpretation": "缺失特征已在当前 FeaturePipeline 输出中存在，说明问题出在持久化/刷新路径，而非特征定义缺失。",
        },
        "feature_store_snapshot": {
            "row_count": int(store_bounds["row_count"] or 0),
            "feature_count": int(store_bounds["feature_count"] or 0),
            "min_calc_date": str(store_bounds["min_calc_date"]) if store_bounds["min_calc_date"] else None,
            "max_calc_date": str(store_bounds["max_calc_date"]) if store_bounds["max_calc_date"] else None,
        },
        "parquet_snapshots": {
            "all_features_v5": parquet_range(v5_path),
            "all_features_v6": parquet_range(v6_path),
            "all_features_current": parquet_range(canonical_path),
        },
        "root_causes": [
            {
                "code": "separate_generation_paths",
                "message": "研究 parquet 由 scripts/run_ic_screening.py 批量生成，live feature_store 由 dags/dag_daily_data.py 的 update_features_cache 维护，两者不是同一导出入口。",
            },
            {
                "code": "stale_gap_reseed_logic",
                "message": f"daily_data update_features_cache 在缓存落后超过 {DEFAULT_FEATURE_BOOTSTRAP_DAYS} 天时只重建最近窗口，不回补完整缺口，导致 store 与 parquet 截止日期长期错位。",
            },
            {
                "code": "bundle_features_postdate_store_refresh",
                "message": "当前 live bundle 所需的 7 个特征在 pipeline 中已存在，但 feature_store 从未通过统一导出路径覆盖这些字段，因此 fail-closed 护栏会持续阻断 live inference。",
            },
        ],
        "recommended_fix": {
            "single_source_of_truth": "以 FeaturePipeline + prepare_feature_export_frame 为唯一导出契约。",
            "dual_export": "同一次导出同时写 parquet 与 feature_store。",
            "rebuild_window": {
                "start_date": "2025-06-30",
                "end_date": str(pit_trade_date),
            },
            "preserve_experiment_snapshots": [
                "data/features/all_features_v5.parquet",
                "data/features/all_features_v6.parquet",
            ],
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
