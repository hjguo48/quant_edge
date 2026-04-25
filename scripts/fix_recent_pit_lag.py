#!/usr/bin/env python3
"""Backfill PIT correction for stock_prices rows where knowledge_time was set to
observed_at (same-day market close + ~10 min) instead of the historical convention
(trade_date + 1 day at 16:00 ET = 20:00 UTC).

Root cause: ``dags/dag_daily_data.py`` previously used ``knowledge_time_mode='observed_at'``
on ``polygon_source.fetch_grouped_daily_range`` / ``fetch_historical`` calls, so the
``market_close_fast_pipeline`` DAG wrote 3,710 rows over 2026-04-17 ~ 2026-04-24 with
``knowledge_time = trade_date 20:10 UTC`` (lag = 0 calendar day). Combined with the
``LEAST`` UPSERT in ``persist_prices``, these rows could not be repaired by a later
historical re-fetch.

This script identifies rows where ``knowledge_time::date <= trade_date`` and shifts
their ``knowledge_time`` to ``trade_date + 1 day at 16:00 America/New_York`` (i.e. the
historical convention) using a single SQL UPDATE inside a transaction.

Usage:
    python scripts/fix_recent_pit_lag.py [--dry-run]
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


UPDATE_SQL = text(
    """
    UPDATE stock_prices
       SET knowledge_time = (
               (trade_date + INTERVAL '1 day')::timestamp
               + TIME '16:00'
           ) AT TIME ZONE 'America/New_York'
     WHERE ((knowledge_time AT TIME ZONE 'UTC')::date) <= trade_date
    """
)

COUNT_SQL = text(
    """
    SELECT COUNT(*) AS n
      FROM stock_prices
     WHERE ((knowledge_time AT TIME ZONE 'UTC')::date) <= trade_date
    """
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report count only, no UPDATE.")
    args = parser.parse_args()

    engine = get_engine()
    with engine.begin() as conn:
        before = conn.execute(COUNT_SQL).scalar() or 0
        print(f"[fix_recent_pit_lag] rows to repair: {before}")
        if before == 0:
            print("[fix_recent_pit_lag] nothing to do.")
            return 0
        if args.dry_run:
            print("[fix_recent_pit_lag] --dry-run set, no UPDATE issued.")
            return 0

        result = conn.execute(UPDATE_SQL)
        repaired = result.rowcount or 0
        print(f"[fix_recent_pit_lag] UPDATE rowcount: {repaired}")

        after = conn.execute(COUNT_SQL).scalar() or 0
        print(f"[fix_recent_pit_lag] remaining lag<=0 rows: {after}")
        if after != 0:
            print("[fix_recent_pit_lag] WARNING: residual lag<=0 rows remain after UPDATE")
            return 2

    print("[fix_recent_pit_lag] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
