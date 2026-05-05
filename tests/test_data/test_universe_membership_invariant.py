from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from scripts.fix_universe_membership_intervals import (
    count_duplicate_active_groups,
    repair_universe_membership_intervals,
)


@pytest.fixture
def temp_universe_membership_conn(db_engine) -> Connection:
    with db_engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(
                text(
                    """
                    create temp table universe_membership (
                        like public.universe_membership
                        including defaults
                        including constraints
                    )
                    """
                )
            )
            yield conn
        finally:
            transaction.rollback()


def _seed_duplicate_active_rows(conn: Connection) -> None:
    conn.execute(
        text(
            """
            insert into universe_membership
                (id, ticker, index_name, effective_date, end_date, reason)
            values
                (1, 'AAPL', 'SP500', '2016-01-01', null, 'historical_backfill_anchor'),
                (2, 'AAPL', 'SP500', '2024-01-01', null, 'historical_backfill_anchor'),
                (3, 'AAPL', 'SP500', '2026-05-01', null, 'historical_backfill_anchor'),
                (4, 'MSFT', 'SP500', '2016-01-01', null, 'historical_backfill_anchor')
            """
        )
    )


def test_universe_membership_repair_closes_older_active_intervals(temp_universe_membership_conn):
    _seed_duplicate_active_rows(temp_universe_membership_conn)

    result = repair_universe_membership_intervals(
        conn=temp_universe_membership_conn,
        apply=True,
        active_index_name="uq_universe_membership_one_active_test",
    )

    assert result["duplicate_groups_before"] == 1
    assert result["rows_to_close"] == 2
    assert result["duplicate_groups_after"] == 0
    assert count_duplicate_active_groups(temp_universe_membership_conn) == 0

    rows = temp_universe_membership_conn.execute(
        text(
            """
            select id, effective_date, end_date
            from universe_membership
            where ticker = 'AAPL'
            order by effective_date
            """
        )
    ).mappings().all()

    assert [(row["id"], row["end_date"]) for row in rows] == [
        (1, date(2024, 1, 1)),
        (2, date(2026, 5, 1)),
        (3, None),
    ]


def test_universe_membership_dry_run_does_not_mutate(temp_universe_membership_conn):
    _seed_duplicate_active_rows(temp_universe_membership_conn)

    result = repair_universe_membership_intervals(conn=temp_universe_membership_conn, apply=False)

    assert result["duplicate_groups_before"] == 1
    assert result["rows_to_close"] == 2
    assert count_duplicate_active_groups(temp_universe_membership_conn) == 1


def test_universe_membership_active_unique_index_rejects_second_active_row(temp_universe_membership_conn):
    _seed_duplicate_active_rows(temp_universe_membership_conn)
    repair_universe_membership_intervals(
        conn=temp_universe_membership_conn,
        apply=True,
        active_index_name="uq_universe_membership_one_active_test",
    )

    with pytest.raises(IntegrityError):
        temp_universe_membership_conn.execute(
            text(
                """
                insert into universe_membership
                    (id, ticker, index_name, effective_date, end_date, reason)
                values
                    (5, 'AAPL', 'SP500', '2026-06-01', null, 'regression_check')
                """
            )
        )
