#!/usr/bin/env python3
"""Backfill paper_portfolio_audit from existing greyscale week_*.json exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.session import get_engine
from src.data.paper_portfolio_audit import save_paper_portfolio_report

DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "reports" / "greyscale"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def list_week_reports(report_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in report_dir.glob("week_*.json")
        if "bak" not in path.name and "contaminated" not in path.name
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    paths = list_week_reports(report_dir)
    if not paths:
        print(f"no week_*.json reports found in {report_dir}")
        return 0

    total_rows = 0
    with get_engine().begin() as conn:
        for path in paths:
            report = json.loads(path.read_text())
            weights = (report.get("live_outputs") or {}).get("target_weights_after_risk") or {}
            signal_date = (report.get("live_outputs") or {}).get("signal_date")
            if args.dry_run:
                rows_saved = len(weights)
            else:
                rows_saved = save_paper_portfolio_report(report, run_id=path.stem, conn=conn)
            total_rows += rows_saved
            action = "would backfill" if args.dry_run else "backfilled"
            print(f"{action} {rows_saved:4d} rows from {path.name} signal_date={signal_date}")

    print(f"total_rows={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
