#!/usr/bin/env python3
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import sys

from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from _audit_common import get_engine, write_json_report

CHECK_DATES = [
    date(2016, 1, 1),
    date(2017, 6, 1),
    date(2018, 12, 31),
    date(2020, 6, 15),
    date(2023, 1, 1),
    date(2023, 12, 15),
    date(2024, 6, 1),
    date(2025, 12, 31),
]
MIN_EXPECTED = 490
MAX_EXPECTED = 515
DEFAULT_OUTPUT = Path("data/reports/p3_pit_verification_20260418.json")


def run_verification(*, output_path: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    engine = get_engine()
    counts: dict[str, int] = {}
    with engine.connect() as conn:
        for as_of in CHECK_DATES:
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
                    {"as_of": as_of},
                ).scalar()
                or 0
            )
            counts[as_of.isoformat()] = count
            if count < MIN_EXPECTED or count > MAX_EXPECTED:
                raise RuntimeError(
                    f"PIT universe coverage failed for {as_of.isoformat()}: "
                    f"{count} tickers not in [{MIN_EXPECTED}, {MAX_EXPECTED}]",
                )

    payload: dict[str, object] = {
        "metadata": {
            "说明": "Week 2.5 P3 universe_membership PIT coverage verification.",
            "generated_at_utc": datetime.now(timezone.utc),
            "script_name": "scripts/verify_universe_pit_coverage.py",
            "accepted_range": [MIN_EXPECTED, MAX_EXPECTED],
        },
        "summary": {
            "check_dates": [as_of.isoformat() for as_of in CHECK_DATES],
            "count_range": [min(counts.values()), max(counts.values())],
            "all_dates_passed": True,
        },
        "counts_by_date": counts,
    }
    write_json_report(payload, output_path)
    print(f"[verify_universe_pit_coverage] wrote {output_path}")
    for as_of, count in counts.items():
        print(f"{as_of}: {count}")
    return payload


def main() -> None:
    run_verification()


if __name__ == "__main__":
    main()
