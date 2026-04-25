#!/usr/bin/env python3
"""Repair short_interest rows whose ``knowledge_time`` was set with the legacy
``settlement_date + 3 calendar days`` heuristic instead of the FINRA-correct
``settlement_date + 7 business days`` publication cycle.

Background: ``src/data/sources/polygon_short_interest.py`` previously stored
all rows with ``kt = settlement_date + 3 calendar days``, which is 5+ days
ahead of FINRA's actual mid-/end-of-month publication. An interim audit fix
used 8 BD; Codex deep review confirmed the official schedule is reports due
on T+2 BD and publication on T+7 BD, so the canonical kt is settlement +
7 business days.

Usage:
    python scripts/fix_short_interest_pit_lag.py [--dry-run]
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import sys

from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.session import get_engine  # noqa: E402


def add_business_days(start: date, n: int) -> date:
    cur = start
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            added += 1
    return cur


def expected_kt(settlement_date: date) -> datetime:
    return datetime.combine(
        add_business_days(settlement_date, 7),
        datetime.max.time(),
        tzinfo=timezone.utc,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                """
                SELECT settlement_date,
                       MIN(knowledge_time) AS old_kt,
                       COUNT(*) AS n
                  FROM short_interest
                 GROUP BY settlement_date
                 ORDER BY settlement_date
                """
            )
        ).all()

        repaired_dates = 0
        repaired_rows = 0
        for settlement_date, old_kt, n in rows:
            new_kt = expected_kt(settlement_date)
            old_kt_aware = old_kt if old_kt.tzinfo else old_kt.replace(tzinfo=timezone.utc)
            if abs((new_kt - old_kt_aware).total_seconds()) < 1:
                continue
            repaired_dates += 1
            repaired_rows += n
            if args.dry_run:
                continue
            conn.execute(
                text(
                    """
                    UPDATE short_interest
                       SET knowledge_time = :new_kt
                     WHERE settlement_date = :sd
                    """
                ),
                {"new_kt": new_kt, "sd": settlement_date},
            )

        action = "would repair" if args.dry_run else "repaired"
        print(
            f"[fix_short_interest_pit_lag] {action} {repaired_dates} settlement_dates "
            f"({repaired_rows} rows)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
