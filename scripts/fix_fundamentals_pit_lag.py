#!/usr/bin/env python3
"""Repair fundamentals_pit rows where vendor (FMP) returned a non-causal
``acceptedDate`` / ``fillingDate`` so the stored ``knowledge_time`` ended up at
or before the fiscal-period ``event_time``.

10-Q filings arrive 35-45 days after fiscal period end, and 10-K arrives 60-90
days later — so any row where ``knowledge_time::date <= event_time::date``
implies the financial number is "knowable" before quarter end, which would let
backtests peek into the future. Audit on 2026-04-25 found 15,429 such rows
across 154 S&P 500 tickers (most concentrated in 2015-2018).

This script shifts the affected rows' ``knowledge_time`` to the conservative
``event_time + 45 days at 16:00 America/New_York`` fallback, matching
``FMPDataSource._fallback_knowledge_time``. Source-side validation is added in
``src/data/sources/fmp.py::_parse_knowledge_time`` to prevent regression on
future fetches.

Usage:
    python scripts/fix_fundamentals_pit_lag.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.session import get_engine  # noqa: E402


COUNT_SQL = text(
    """
    SELECT COUNT(*) AS n
      FROM fundamentals_pit
     WHERE ((knowledge_time AT TIME ZONE 'UTC')::date) <= event_time::date
    """
)

UPDATE_SQL = text(
    """
    UPDATE fundamentals_pit
       SET knowledge_time = (
               (event_time + INTERVAL '45 days')::timestamp + TIME '16:00'
           ) AT TIME ZONE 'America/New_York'
     WHERE ((knowledge_time AT TIME ZONE 'UTC')::date) <= event_time::date
    """
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report count only, no UPDATE.")
    args = parser.parse_args()

    engine = get_engine()
    with engine.begin() as conn:
        before = conn.execute(COUNT_SQL).scalar() or 0
        print(f"[fix_fundamentals_pit_lag] rows to repair: {before}")
        if before == 0:
            print("[fix_fundamentals_pit_lag] nothing to do.")
            return 0
        if args.dry_run:
            print("[fix_fundamentals_pit_lag] --dry-run set, no UPDATE issued.")
            return 0

        result = conn.execute(UPDATE_SQL)
        repaired = result.rowcount or 0
        print(f"[fix_fundamentals_pit_lag] UPDATE rowcount: {repaired}")

        after = conn.execute(COUNT_SQL).scalar() or 0
        print(f"[fix_fundamentals_pit_lag] remaining lag<=0 rows: {after}")
        if after != 0:
            print("[fix_fundamentals_pit_lag] WARNING: residual lag<=0 rows remain after UPDATE")
            return 2

    print("[fix_fundamentals_pit_lag] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
