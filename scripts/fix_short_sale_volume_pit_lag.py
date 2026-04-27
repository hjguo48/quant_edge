#!/usr/bin/env python3
"""Repair ``short_sale_volume_daily`` rows whose ``knowledge_time`` was set to
``trade_date 18:00 ET`` (the FINRA publication time on the same day) instead of
the lag-1 convention used by ``stock_prices`` and the rest of the daily PIT
layer.

Why the same-day timestamp was leaky
------------------------------------
W7-style backtests use ``as_of = trade_date 23:59 NYT`` (≈ ``trade_date+1
03:59 UTC``). With kt = trade_date 22:00 UTC, that row is visible during
trade_date's signal computation, so today's FINRA off-exchange short volume
could leak into today's predictions. Aligning kt to ``trade_date + 1 day
16:00 NYT`` matches ``PolygonDataSource._historical_knowledge_time`` and
keeps short-volume features one full day behind the trade date.

Source-side fix is in ``src/data/finra_short_sale.py::_knowledge_time``.

Usage:
    python scripts/fix_short_sale_volume_pit_lag.py [--dry-run]
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
      FROM short_sale_volume_daily
     WHERE ((knowledge_time AT TIME ZONE 'UTC')::date) <= trade_date
    """
)

UPDATE_SQL = text(
    """
    UPDATE short_sale_volume_daily
       SET knowledge_time = (
               (trade_date + INTERVAL '1 day')::timestamp + TIME '16:00'
           ) AT TIME ZONE 'America/New_York'
     WHERE ((knowledge_time AT TIME ZONE 'UTC')::date) <= trade_date
    """
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    with engine.begin() as conn:
        before = conn.execute(COUNT_SQL).scalar() or 0
        print(f"[fix_short_sale_volume_pit_lag] rows to repair: {before}")
        if before == 0:
            print("[fix_short_sale_volume_pit_lag] nothing to do.")
            return 0
        if args.dry_run:
            print("[fix_short_sale_volume_pit_lag] --dry-run set, no UPDATE issued.")
            return 0

        result = conn.execute(UPDATE_SQL)
        print(f"[fix_short_sale_volume_pit_lag] UPDATE rowcount: {result.rowcount}")

        after = conn.execute(COUNT_SQL).scalar() or 0
        print(f"[fix_short_sale_volume_pit_lag] residual lag<=0 rows: {after}")
        if after != 0:
            return 2

    print("[fix_short_sale_volume_pit_lag] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
