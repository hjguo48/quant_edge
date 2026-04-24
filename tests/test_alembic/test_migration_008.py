from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError


ROOT_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI = ROOT_DIR / "alembic.ini"


def _alembic_config() -> Config:
    return Config(str(ALEMBIC_INI))


def _insert_short_sale_row(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                insert into short_sale_volume_daily (
                    ticker, trade_date, knowledge_time, market, short_volume,
                    short_exempt_volume, total_volume, file_etag
                ) values (
                    :ticker, :trade_date, :knowledge_time, :market, :short_volume,
                    :short_exempt_volume, :total_volume, :file_etag
                )
                """,
            ),
            {
                "ticker": "AAPL",
                "trade_date": date(2026, 4, 23),
                "knowledge_time": datetime(2026, 4, 23, 18, 0, tzinfo=timezone.utc),
                "market": "CNMS",
                "short_volume": 10_000,
                "short_exempt_volume": 250,
                "total_volume": 100_000,
                "file_etag": "etag-1",
            },
        )


def _insert_grade_row(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                insert into grades_events (
                    ticker, event_date, knowledge_time, analyst_firm, prior_grade,
                    new_grade, action, grade_score_change
                ) values (
                    :ticker, :event_date, :knowledge_time, :analyst_firm, :prior_grade,
                    :new_grade, :action, :grade_score_change
                )
                """,
            ),
            {
                "ticker": "AAPL",
                "event_date": date(2026, 4, 23),
                "knowledge_time": datetime(2026, 4, 23, 23, 59, tzinfo=timezone.utc),
                "analyst_firm": "Goldman Sachs",
                "prior_grade": "Hold",
                "new_grade": "Buy",
                "action": "upgrade",
                "grade_score_change": 1,
            },
        )


def _insert_rating_row(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                insert into ratings_events (
                    ticker, event_date, knowledge_time, rating_score, rating_recommendation,
                    dcf_rating, pe_rating, roe_rating
                ) values (
                    :ticker, :event_date, :knowledge_time, :rating_score, :rating_recommendation,
                    :dcf_rating, :pe_rating, :roe_rating
                )
                """,
            ),
            {
                "ticker": "AAPL",
                "event_date": date(2026, 4, 23),
                "knowledge_time": datetime(2026, 4, 23, 23, 59, tzinfo=timezone.utc),
                "rating_score": 4,
                "rating_recommendation": "Buy",
                "dcf_rating": Decimal("4.25"),
                "pe_rating": Decimal("3.50"),
                "roe_rating": Decimal("4.75"),
            },
        )


def _insert_price_target_row(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                insert into price_target_events (
                    ticker, event_date, knowledge_time, analyst_firm, target_price,
                    prior_target, price_when_published, is_consensus
                ) values (
                    :ticker, :event_date, :knowledge_time, :analyst_firm, :target_price,
                    :prior_target, :price_when_published, :is_consensus
                )
                """,
            ),
            {
                "ticker": "AAPL",
                "event_date": date(2026, 4, 23),
                "knowledge_time": datetime(2026, 4, 23, 23, 59, tzinfo=timezone.utc),
                "analyst_firm": "Morgan Stanley",
                "target_price": Decimal("225.0000"),
                "prior_target": Decimal("210.0000"),
                "price_when_published": Decimal("198.5000"),
                "is_consensus": False,
            },
        )


def _insert_earnings_calendar_row(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            sa.text(
                """
                insert into earnings_calendar (
                    ticker, announce_date, knowledge_time, timing, fiscal_period_end,
                    eps_estimate, eps_actual, revenue_estimate, revenue_actual
                ) values (
                    :ticker, :announce_date, :knowledge_time, :timing, :fiscal_period_end,
                    :eps_estimate, :eps_actual, :revenue_estimate, :revenue_actual
                )
                """,
            ),
            {
                "ticker": "AAPL",
                "announce_date": date(2026, 4, 30),
                "knowledge_time": datetime(2026, 4, 29, 23, 59, tzinfo=timezone.utc),
                "timing": "BMO",
                "fiscal_period_end": date(2026, 3, 31),
                "eps_estimate": Decimal("1.7200"),
                "eps_actual": Decimal("1.8100"),
                "revenue_estimate": 95_000_000_000,
                "revenue_actual": 96_500_000_000,
            },
        )


def test_migration_008_upgrade_downgrade_reupgrade_and_constraints(db_engine: Engine) -> None:
    cfg = _alembic_config()
    command.downgrade(cfg, "007")
    command.upgrade(cfg, "008")

    inspector = inspect(db_engine)
    expected_tables = {
        "short_sale_volume_daily",
        "grades_events",
        "ratings_events",
        "price_target_events",
        "earnings_calendar",
    }
    assert expected_tables <= set(inspector.get_table_names())

    short_sale_columns = {column["name"]: column for column in inspector.get_columns("short_sale_volume_daily")}
    assert {"ticker", "trade_date", "knowledge_time", "market", "file_etag"} <= set(short_sale_columns)
    assert short_sale_columns["knowledge_time"]["nullable"] is False
    assert inspector.get_pk_constraint("short_sale_volume_daily")["constrained_columns"] == [
        "ticker",
        "trade_date",
        "market",
    ]

    grade_columns = {column["name"]: column for column in inspector.get_columns("grades_events")}
    assert {"id", "ticker", "event_date", "analyst_firm", "grade_score_change"} <= set(grade_columns)
    assert grade_columns["knowledge_time"]["nullable"] is False

    rating_columns = {column["name"]: column for column in inspector.get_columns("ratings_events")}
    assert {"ticker", "event_date", "rating_score", "dcf_rating", "pe_rating", "roe_rating"} <= set(rating_columns)
    assert inspector.get_pk_constraint("ratings_events")["constrained_columns"] == ["ticker", "event_date"]

    target_columns = {column["name"]: column for column in inspector.get_columns("price_target_events")}
    assert {"id", "ticker", "event_date", "target_price", "is_consensus"} <= set(target_columns)
    assert target_columns["knowledge_time"]["nullable"] is False

    earnings_columns = {column["name"]: column for column in inspector.get_columns("earnings_calendar")}
    assert {
        "ticker",
        "announce_date",
        "knowledge_time",
        "timing",
        "fiscal_period_end",
        "eps_estimate",
        "eps_actual",
        "revenue_estimate",
        "revenue_actual",
    } <= set(earnings_columns)
    assert inspector.get_pk_constraint("earnings_calendar")["constrained_columns"] == ["ticker", "announce_date"]

    indexes = {index["name"] for table in expected_tables for index in inspector.get_indexes(table)}
    assert {
        "ix_short_sale_kt",
        "ix_grades_kt",
        "ix_ratings_kt",
        "ix_target_kt",
        "ix_earnings_cal_kt",
    } <= indexes

    _insert_short_sale_row(db_engine)
    with pytest.raises(IntegrityError):
        _insert_short_sale_row(db_engine)

    _insert_grade_row(db_engine)
    with pytest.raises(IntegrityError):
        _insert_grade_row(db_engine)

    _insert_rating_row(db_engine)
    with pytest.raises(IntegrityError):
        _insert_rating_row(db_engine)

    _insert_price_target_row(db_engine)
    with pytest.raises(IntegrityError):
        _insert_price_target_row(db_engine)

    _insert_earnings_calendar_row(db_engine)
    with pytest.raises(IntegrityError):
        _insert_earnings_calendar_row(db_engine)

    with db_engine.connect() as conn:
        counts = {
            "short_sale_volume_daily": conn.execute(sa.text("select count(*) from short_sale_volume_daily")).scalar_one(),
            "grades_events": conn.execute(sa.text("select count(*) from grades_events")).scalar_one(),
            "ratings_events": conn.execute(sa.text("select count(*) from ratings_events")).scalar_one(),
            "price_target_events": conn.execute(sa.text("select count(*) from price_target_events")).scalar_one(),
            "earnings_calendar": conn.execute(sa.text("select count(*) from earnings_calendar")).scalar_one(),
        }
    assert counts == {name: 1 for name in counts}

    command.downgrade(cfg, "007")
    inspector = inspect(db_engine)
    assert expected_tables.isdisjoint(set(inspector.get_table_names()))

    command.upgrade(cfg, "008")
    inspector = inspect(db_engine)
    assert expected_tables <= set(inspector.get_table_names())
