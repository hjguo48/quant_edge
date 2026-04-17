#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import REPORT_DATE_TAG, get_engine, write_json_report
from src.universe.builder import backfill_universe_membership as backfill_membership_rows


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def build_monthly_coverage(engine, *, start_date: date, end_date: date) -> dict[str, int]:
    month = start_date.replace(day=1)
    final_month = end_date.replace(day=1)
    coverage: dict[str, int] = {}
    with engine.connect() as conn:
        while month <= final_month:
            count = int(
                conn.execute(
                    text(
                        """
                        select count(distinct ticker)
                        from universe_membership
                        where index_name = 'SP500'
                          and effective_date <= :as_of
                          and (end_date is null or end_date > :as_of)
                        """
                    ),
                    {"as_of": month},
                ).scalar()
                or 0
            )
            coverage[month.isoformat()] = count
            if month.month == 12:
                month = month.replace(year=month.year + 1, month=1)
            else:
                month = month.replace(month=month.month + 1)
    return coverage


def run_backfill(*, start_date: date, end_date: date, strict_fmp: bool, output_path: Path) -> dict:
    rows_written = backfill_membership_rows(
        start_date=start_date,
        end_date=end_date,
        index_name="SP500",
        strict_fmp=strict_fmp,
    )
    engine = get_engine()
    with engine.connect() as conn:
        summary = conn.execute(
            text(
                """
                select min(effective_date) as min_date,
                       max(effective_date) as max_date,
                       count(*) as rows,
                       count(distinct ticker) as tickers
                from universe_membership
                """
            )
        ).mappings().first()
    coverage = build_monthly_coverage(engine, start_date=start_date, end_date=end_date)
    payload = {
        "metadata": {
            "说明": "Week 2.5-P3 PIT universe membership 历史回填结果。",
            "generated_at_utc": datetime.now(timezone.utc),
            "script_name": "scripts/backfill_universe_membership.py",
        },
        "summary": {
            "rows_written": int(rows_written),
            "membership_rows": int(summary["rows"] or 0),
            "membership_tickers": int(summary["tickers"] or 0),
            "membership_date_range": [summary["min_date"], summary["max_date"]],
            "strict_fmp": bool(strict_fmp),
        },
        "membership_coverage": coverage,
    }
    write_json_report(payload, output_path)
    print(f"[backfill_universe_membership] wrote {output_path}")
    print(
        f"[backfill_universe_membership] rows_written={rows_written} "
        f"coverage_start={summary['min_date']} coverage_end={summary['max_date']}"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical SP500 universe membership intervals.")
    parser.add_argument("--start-date", type=_parse_date, default=date(2016, 1, 1))
    parser.add_argument("--end-date", type=_parse_date, default=date(2025, 12, 31))
    parser.add_argument("--strict-fmp", action="store_true")
    parser.add_argument(
        "--output",
        default=f"data/reports/universe_backfill_{REPORT_DATE_TAG}.json",
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        strict_fmp=args.strict_fmp,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
