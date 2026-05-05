#!/usr/bin/env python3
"""Normalize duplicate active universe_membership intervals.

Dry-run is the default. Pass --apply to close older active rows at the next
active effective_date and add a partial unique index preventing recurrence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import date
from pathlib import Path
import re
import sys
from typing import Any

from sqlalchemy import Connection, text

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.session import get_engine

ACTIVE_INDEX_NAME = "uq_universe_membership_one_active"


def _quote_identifier(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return f'"{name}"'


@contextmanager
def _managed_connection(conn: Connection | None, *, apply: bool):
    if conn is not None:
        yield conn
        return

    engine = get_engine()
    if apply:
        with engine.begin() as owned:
            yield owned
    else:
        with engine.connect() as owned:
            yield owned


def fetch_duplicate_active_rows(
    conn: Connection,
    *,
    index_name: str | None = None,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        text(
            """
            with duplicate_groups as (
                select ticker, index_name
                from universe_membership
                where end_date is null
                  and (:index_name is null or index_name = :index_name)
                group by ticker, index_name
                having count(*) > 1
            )
            select
                um.id,
                um.ticker,
                um.index_name,
                um.effective_date,
                um.end_date,
                um.reason
            from universe_membership um
            join duplicate_groups dg
              on dg.ticker = um.ticker
             and dg.index_name = um.index_name
            where um.end_date is null
            order by um.ticker, um.index_name, um.effective_date, um.id
            """
        ),
        {"index_name": index_name},
    ).mappings()
    return [dict(row) for row in rows]


def build_interval_repair_plan(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["ticker"]), str(row["index_name"]))].append(row)

    plan: list[dict[str, Any]] = []
    for (ticker, index_name), group_rows in sorted(grouped.items()):
        ordered = sorted(group_rows, key=lambda row: (row["effective_date"], row["id"]))
        for current, next_row in zip(ordered, ordered[1:]):
            next_effective = next_row["effective_date"]
            current_effective = current["effective_date"]
            if not isinstance(next_effective, date) or not isinstance(current_effective, date):
                raise TypeError(f"Unexpected effective_date types for {ticker}/{index_name}")
            if next_effective <= current_effective:
                raise ValueError(
                    f"Non-increasing active intervals for {ticker}/{index_name}: "
                    f"{current_effective} -> {next_effective}"
                )
            plan.append(
                {
                    "id": current["id"],
                    "ticker": ticker,
                    "index_name": index_name,
                    "effective_date": current_effective,
                    "new_end_date": next_effective,
                    "next_active_id": next_row["id"],
                }
            )
    return plan


def count_duplicate_active_groups(
    conn: Connection,
    *,
    index_name: str | None = None,
) -> int:
    return int(
        conn.execute(
            text(
                """
                select count(*)
                from (
                    select ticker, index_name
                    from universe_membership
                    where end_date is null
                      and (:index_name is null or index_name = :index_name)
                    group by ticker, index_name
                    having count(*) > 1
                ) duplicate_groups
                """
            ),
            {"index_name": index_name},
        ).scalar()
        or 0
    )


def create_active_membership_index(
    conn: Connection,
    *,
    index_name: str = ACTIVE_INDEX_NAME,
) -> None:
    if conn.dialect.name != "postgresql":
        return
    quoted = _quote_identifier(index_name)
    conn.execute(
        text(
            f"""
            create unique index if not exists {quoted}
            on universe_membership (ticker, index_name)
            where end_date is null
            """
        )
    )


def repair_universe_membership_intervals(
    *,
    conn: Connection | None = None,
    apply: bool = False,
    index_name: str | None = None,
    active_index_name: str = ACTIVE_INDEX_NAME,
) -> dict[str, Any]:
    with _managed_connection(conn, apply=apply) as active_conn:
        duplicate_rows = fetch_duplicate_active_rows(active_conn, index_name=index_name)
        plan = build_interval_repair_plan(duplicate_rows)
        duplicate_groups_before = count_duplicate_active_groups(active_conn, index_name=index_name)

        if apply:
            for change in plan:
                active_conn.execute(
                    text(
                        """
                        update universe_membership
                        set end_date = :new_end_date
                        where id = :id
                          and end_date is null
                        """
                    ),
                    {"id": change["id"], "new_end_date": change["new_end_date"]},
                )
            create_active_membership_index(active_conn, index_name=active_index_name)
            duplicate_groups_after = count_duplicate_active_groups(active_conn, index_name=index_name)
            if duplicate_groups_after:
                raise RuntimeError(f"Universe membership still has {duplicate_groups_after} duplicate active groups")
        else:
            duplicate_groups_after = duplicate_groups_before

    return {
        "apply": apply,
        "duplicate_groups_before": duplicate_groups_before,
        "active_rows_in_duplicate_groups": len(duplicate_rows),
        "rows_to_close": len(plan),
        "duplicate_groups_after": duplicate_groups_after,
        "sample_plan": plan[:10],
        "active_index_name": active_index_name if apply else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply interval repairs and create the active-row index.")
    parser.add_argument("--index-name", default=None, help="Optional universe index filter, e.g. SP500.")
    parser.add_argument("--sample", type=int, default=10, help="Number of planned updates to print.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = repair_universe_membership_intervals(apply=args.apply, index_name=args.index_name)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[fix_universe_membership_intervals] mode={mode}")
    print(
        "duplicate_groups_before={duplicate_groups_before} "
        "active_rows_in_duplicate_groups={active_rows_in_duplicate_groups} "
        "rows_to_close={rows_to_close} "
        "duplicate_groups_after={duplicate_groups_after}".format(**result)
    )
    sample = result["sample_plan"][: args.sample]
    if sample:
        print("sample planned row closures:")
        for change in sample:
            print(
                "  id={id} {ticker}/{index_name} {effective_date} -> end_date={new_end_date} "
                "(next_active_id={next_active_id})".format(**change)
            )
    if not args.apply:
        print("No changes made. Re-run with --apply to update intervals and add the partial unique index.")


if __name__ == "__main__":
    main()
