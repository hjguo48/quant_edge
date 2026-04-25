#!/usr/bin/env python3
"""Repair fundamentals_pit rows whose stored ``knowledge_time`` is implausibly
close to the fiscal-period ``event_time`` for the SEC filing class.

Two repair populations:
  1. Q1/Q2/Q3 (10-Q): kt must be strictly later than event_time, otherwise
     shift to ``event_time + 45 days @ 16:00 NYT`` (large/accelerated/non-acc
     deadlines all converge at <=45 days).
  2. Q4 / FY (10-K): kt must be at least 60 days after event_time (large
     accelerated filer deadline). If less, shift to ``event_time + 90 days @
     16:00 NYT`` (non-accelerated filer ceiling, conservative for any S&P 500
     issuer).

Source-side guards in ``FMPDataSource._parse_knowledge_time`` /
``_fallback_knowledge_time`` enforce the same period-aware policy on new
fetches. The audit on 2026-04-25 (and Codex deep review) found that the
previous flat 45-day fallback still allowed 16k+ Q4 rows to sit inside the
60-day 10-K deadline window, leaking year-end fundamentals into backtests.

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


# 10-K filing deadline applies only to statement-derived metrics (income /
# balance / cash flow + book_value_per_share which is derived from the same
# balance sheet). Dividend and consensus metrics derive their kt from
# declaration_date or earnings_date and are PIT-safe even inside the 60-day
# window; they must be excluded from the Q4 repair.
STATEMENT_METRICS = (
    "revenue",
    "gross_profit",
    "operating_income",
    "net_income",
    "ebitda",
    "eps",
    "weighted_average_shares_outstanding",
    "total_assets",
    "total_liabilities",
    "total_debt",
    "cash_and_cash_equivalents",
    "current_assets",
    "current_liabilities",
    "total_stockholders_equity",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "book_value_per_share",
)

QUARTERLY_COUNT_SQL = text(
    """
    SELECT COUNT(*) AS n
      FROM fundamentals_pit
     WHERE fiscal_period NOT LIKE '%Q4'
       AND fiscal_period NOT LIKE '%FY'
       AND metric_name = ANY(:statement_metrics)
       AND ((knowledge_time AT TIME ZONE 'UTC')::date) <= event_time::date
    """
)

QUARTERLY_UPDATE_SQL = text(
    """
    UPDATE fundamentals_pit
       SET knowledge_time = (
               (event_time + INTERVAL '45 days')::timestamp + TIME '16:00'
           ) AT TIME ZONE 'America/New_York'
     WHERE fiscal_period NOT LIKE '%Q4'
       AND fiscal_period NOT LIKE '%FY'
       AND metric_name = ANY(:statement_metrics)
       AND ((knowledge_time AT TIME ZONE 'UTC')::date) <= event_time::date
    """
)

ANNUAL_COUNT_SQL = text(
    """
    SELECT COUNT(*) AS n
      FROM fundamentals_pit
     WHERE (fiscal_period LIKE '%Q4' OR fiscal_period LIKE '%FY')
       AND metric_name = ANY(:statement_metrics)
       AND (((knowledge_time AT TIME ZONE 'UTC')::date) - event_time::date) < 60
    """
)

# Delete duplicate bad rows within the same (ticker, fiscal_period,
# metric_name) group before shifting kt. After the shift all surviving rows in
# such a group would collapse onto the same new_kt and violate the unique
# constraint, so we keep the row with the highest id (most recent insert /
# version) and discard older duplicates.
ANNUAL_DEDUP_SQL = text(
    """
    DELETE FROM fundamentals_pit f1
     USING fundamentals_pit f2
     WHERE f1.ticker = f2.ticker
       AND f1.fiscal_period = f2.fiscal_period
       AND f1.metric_name = f2.metric_name
       AND f1.id < f2.id
       AND (f1.fiscal_period LIKE '%Q4' OR f1.fiscal_period LIKE '%FY')
       AND f1.metric_name = ANY(:statement_metrics)
       AND f2.metric_name = ANY(:statement_metrics)
       AND (((f1.knowledge_time AT TIME ZONE 'UTC')::date) - f1.event_time::date) < 60
       AND (((f2.knowledge_time AT TIME ZONE 'UTC')::date) - f2.event_time::date) < 60
    """
)

ANNUAL_UPDATE_SQL = text(
    """
    UPDATE fundamentals_pit
       SET knowledge_time = (
               (event_time + INTERVAL '90 days')::timestamp + TIME '16:00'
           ) AT TIME ZONE 'America/New_York'
     WHERE (fiscal_period LIKE '%Q4' OR fiscal_period LIKE '%FY')
       AND metric_name = ANY(:statement_metrics)
       AND (((knowledge_time AT TIME ZONE 'UTC')::date) - event_time::date) < 60
    """
)

# Final post-condition: no statement-metric row may have kt at or before
# event_time, and no Q4/FY statement-metric row may sit inside the 60-day
# 10-K deadline. Dividend / consensus rows are intentionally excluded.
RESIDUAL_SQL = text(
    """
    SELECT
        SUM(CASE WHEN metric_name = ANY(:statement_metrics)
                  AND ((knowledge_time AT TIME ZONE 'UTC')::date) <= event_time::date
                 THEN 1 ELSE 0 END) AS kt_le_evt,
        SUM(CASE WHEN metric_name = ANY(:statement_metrics)
                  AND (fiscal_period LIKE '%Q4' OR fiscal_period LIKE '%FY')
                  AND (((knowledge_time AT TIME ZONE 'UTC')::date) - event_time::date) < 60
                 THEN 1 ELSE 0 END) AS annual_lag_under_60d
      FROM fundamentals_pit
    """
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report counts only, no UPDATE.")
    args = parser.parse_args()

    params = {"statement_metrics": list(STATEMENT_METRICS)}
    engine = get_engine()
    with engine.begin() as conn:
        quarterly = conn.execute(QUARTERLY_COUNT_SQL, params).scalar() or 0
        annual = conn.execute(ANNUAL_COUNT_SQL, params).scalar() or 0
        print(
            f"[fix_fundamentals_pit_lag] quarterly statement rows to repair (kt<=evt): {quarterly}; "
            f"annual statement rows to repair (lag<60d): {annual}"
        )
        if quarterly == 0 and annual == 0:
            print("[fix_fundamentals_pit_lag] nothing to do.")
            return 0
        if args.dry_run:
            print("[fix_fundamentals_pit_lag] --dry-run set, no UPDATE issued.")
            return 0

        if quarterly > 0:
            r = conn.execute(QUARTERLY_UPDATE_SQL, params)
            print(f"[fix_fundamentals_pit_lag] quarterly UPDATE rowcount: {r.rowcount}")
        if annual > 0:
            r = conn.execute(ANNUAL_DEDUP_SQL, params)
            print(f"[fix_fundamentals_pit_lag] annual DEDUP rowcount: {r.rowcount}")
            r = conn.execute(ANNUAL_UPDATE_SQL, params)
            print(f"[fix_fundamentals_pit_lag] annual UPDATE rowcount: {r.rowcount}")

        residual = conn.execute(RESIDUAL_SQL, params).mappings().one()
        print(
            f"[fix_fundamentals_pit_lag] residual statement rows: kt<=evt={residual['kt_le_evt']}, "
            f"annual_lag<60d={residual['annual_lag_under_60d']}"
        )
        if residual["kt_le_evt"] or residual["annual_lag_under_60d"]:
            print("[fix_fundamentals_pit_lag] WARNING: residual rows remain after UPDATE")
            return 2

    print("[fix_fundamentals_pit_lag] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
