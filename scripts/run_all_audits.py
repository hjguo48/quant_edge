#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPO_ROOT, REPORT_DATE_TAG
from audit_benchmark import build_report as build_benchmark_report
from audit_feature_parity import build_report as build_feature_parity_report
from audit_price_truth import build_report as build_price_truth_report
from audit_universe import build_report as build_universe_report


def main() -> None:
    outputs = {
        "feature_parity": REPO_ROOT / f"data/reports/feature_parity_audit_{REPORT_DATE_TAG}.json",
        "price_truth": REPO_ROOT / f"data/reports/price_truth_audit_{REPORT_DATE_TAG}.json",
        "universe": REPO_ROOT / f"data/reports/universe_audit_{REPORT_DATE_TAG}.json",
        "benchmark": REPO_ROOT / f"data/reports/benchmark_audit_{REPORT_DATE_TAG}.json",
    }
    build_feature_parity_report(REPO_ROOT / "data/features/all_features_v5.parquet", outputs["feature_parity"])
    build_price_truth_report(outputs["price_truth"])
    build_universe_report(outputs["universe"])
    build_benchmark_report(outputs["benchmark"])

    print("[audit_runner] complete")
    for name, path in outputs.items():
        print(f"[audit_runner] {name}: {path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
