from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.data.db.pit as pit_module
from src.data.db.models import Base, FundamentalsPIT, StockPrice, UniverseMembership
from src.data.db.pit import get_fundamentals_pit, get_prices_pit, get_universe_pit


def _session_factory(db_engine: Engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_tables(db_engine: Engine) -> None:
    Base.metadata.create_all(
        bind=db_engine,
        tables=[
            FundamentalsPIT.__table__,
            StockPrice.__table__,
            UniverseMembership.__table__,
        ],
        checkfirst=True,
    )


def _sqlite_session_factory(*tables) -> sessionmaker:
    engine = sa.create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=list(tables), checkfirst=True)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_fundamentals_pit_respects_knowledge_time(db_engine: Engine) -> None:
    _ensure_tables(db_engine)
    session_factory = _session_factory(db_engine)

    with session_factory() as session:
        session.execute(sa.delete(FundamentalsPIT).where(FundamentalsPIT.ticker == "PITF"))
        session.add_all(
            [
                FundamentalsPIT(
                    ticker="PITF",
                    fiscal_period="2024Q1",
                    metric_name="revenue",
                    metric_value=Decimal("100.0"),
                    event_time=date(2024, 3, 31),
                    knowledge_time=datetime(2024, 5, 10, 21, tzinfo=timezone.utc),
                    source="test",
                ),
                FundamentalsPIT(
                    ticker="PITF",
                    fiscal_period="2024Q1",
                    metric_name="revenue",
                    metric_value=Decimal("125.0"),
                    event_time=date(2024, 3, 31),
                    knowledge_time=datetime(2024, 5, 20, 21, tzinfo=timezone.utc),
                    source="test",
                ),
            ],
        )
        session.commit()

    try:
        early = get_fundamentals_pit(
            ticker="PITF",
            as_of=datetime(2024, 5, 15, 12, tzinfo=timezone.utc),
        )
        assert len(early) == 1
        assert float(early.iloc[0]["metric_value"]) == 100.0

        late = get_fundamentals_pit(
            ticker="PITF",
            as_of=datetime(2024, 5, 25, 12, tzinfo=timezone.utc),
        )
        assert len(late) == 1
        assert float(late.iloc[0]["metric_value"]) == 125.0
    finally:
        with session_factory() as session:
            session.execute(sa.delete(FundamentalsPIT).where(FundamentalsPIT.ticker == "PITF"))
            session.commit()


def test_prices_pit_respects_knowledge_time(db_engine: Engine) -> None:
    _ensure_tables(db_engine)
    session_factory = _session_factory(db_engine)

    with session_factory() as session:
        session.execute(sa.delete(StockPrice).where(StockPrice.ticker == "PITP"))
        session.add_all(
            [
                StockPrice(
                    ticker="PITP",
                    trade_date=date(2024, 1, 2),
                    open=Decimal("10"),
                    high=Decimal("11"),
                    low=Decimal("9"),
                    close=Decimal("10.5"),
                    adj_close=Decimal("10.5"),
                    volume=1_000_000,
                    knowledge_time=datetime(2024, 1, 3, 21, tzinfo=timezone.utc),
                    source="test",
                ),
                StockPrice(
                    ticker="PITP",
                    trade_date=date(2024, 1, 3),
                    open=Decimal("11"),
                    high=Decimal("12"),
                    low=Decimal("10"),
                    close=Decimal("11.5"),
                    adj_close=Decimal("11.5"),
                    volume=1_500_000,
                    knowledge_time=datetime(2024, 1, 5, 21, tzinfo=timezone.utc),
                    source="test",
                ),
            ],
        )
        session.commit()

    try:
        result = get_prices_pit(
            tickers=["PITP"],
            start_date=date(2024, 1, 2),
            end_date=date(2024, 1, 3),
            as_of=datetime(2024, 1, 4, 12, tzinfo=timezone.utc),
        )
        assert len(result) == 1
        assert result.iloc[0]["trade_date"] == date(2024, 1, 2)
    finally:
        with session_factory() as session:
            session.execute(sa.delete(StockPrice).where(StockPrice.ticker == "PITP"))
            session.commit()


def test_universe_pit_returns_correct_members(db_engine: Engine) -> None:
    _ensure_tables(db_engine)
    session_factory = _session_factory(db_engine)

    with session_factory() as session:
        session.execute(
            sa.delete(UniverseMembership).where(UniverseMembership.ticker.in_(["PITU1", "PITU2"])),
        )
        session.add_all(
            [
                UniverseMembership(
                    ticker="PITU1",
                    index_name="SP500",
                    effective_date=date(2024, 1, 1),
                    end_date=date(2024, 3, 1),
                    reason="initial",
                ),
                UniverseMembership(
                    ticker="PITU2",
                    index_name="SP500",
                    effective_date=date(2024, 3, 1),
                    end_date=None,
                    reason="replacement",
                ),
            ],
        )
        session.commit()

    try:
        january_members = get_universe_pit(
            as_of=datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
            index_name="SP500",
        )
        march_members = get_universe_pit(
            as_of=datetime(2024, 3, 15, 12, tzinfo=timezone.utc),
            index_name="SP500",
        )

        assert january_members == ["PITU1"]
        assert march_members == ["PITU2"]
    finally:
        with session_factory() as session:
            session.execute(
                sa.delete(UniverseMembership).where(UniverseMembership.ticker.in_(["PITU1", "PITU2"])),
            )
            session.commit()


def test_fundamentals_pit_returns_shares_outstanding_snapshot_without_external_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = _sqlite_session_factory(FundamentalsPIT.__table__)
    monkeypatch.setattr(pit_module, "get_session_factory", lambda: session_factory)

    with session_factory() as session:
        session.add_all(
            [
                FundamentalsPIT(
                    ticker="UNIT",
                    fiscal_period="2024Q1",
                    metric_name="weighted_average_shares_outstanding",
                    metric_value=Decimal("1000000"),
                    event_time=date(2024, 3, 31),
                    knowledge_time=datetime(2024, 5, 10, 21, tzinfo=timezone.utc),
                    source="test",
                ),
                FundamentalsPIT(
                    ticker="UNIT",
                    fiscal_period="2024Q2",
                    metric_name="weighted_average_shares_outstanding",
                    metric_value=Decimal("1100000"),
                    event_time=date(2024, 6, 30),
                    knowledge_time=datetime(2024, 8, 9, 21, tzinfo=timezone.utc),
                    source="test",
                ),
            ],
        )
        session.commit()

    result = pit_module.get_fundamentals_pit(
        ticker="UNIT",
        as_of=datetime(2024, 8, 15, 12, tzinfo=timezone.utc),
        metric_names=["weighted_average_shares_outstanding"],
    )

    assert len(result) == 2
    assert float(result.iloc[-1]["metric_value"]) == 1_100_000.0


def test_fundamentals_pit_excludes_rows_until_knowledge_time_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_factory = _sqlite_session_factory(FundamentalsPIT.__table__)
    monkeypatch.setattr(pit_module, "get_session_factory", lambda: session_factory)

    with session_factory() as session:
        session.add(
            FundamentalsPIT(
                ticker="TIME",
                fiscal_period="2024Q1",
                metric_name="shares_outstanding",
                metric_value=Decimal("2500000"),
                event_time=date(2024, 3, 31),
                knowledge_time=datetime(2024, 5, 15, 21, tzinfo=timezone.utc),
                source="test",
            ),
        )
        session.commit()

    before_release = pit_module.get_fundamentals_pit(
        ticker="TIME",
        as_of=datetime(2024, 4, 15, 12, tzinfo=timezone.utc),
    )
    after_release = pit_module.get_fundamentals_pit(
        ticker="TIME",
        as_of=datetime(2024, 5, 16, 12, tzinfo=timezone.utc),
    )

    assert before_release.empty
    assert len(after_release) == 1
    assert pd.Timestamp(after_release.iloc[0]["knowledge_time"], tz="UTC").date() > after_release.iloc[0]["event_time"]
