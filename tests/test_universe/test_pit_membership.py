from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.pool import StaticPool

import src.data.db.pit as pit_module
import src.universe.active as active_module
from src.data.db.models import Base, StockPrice, UniverseMembership
from src.data.db.pit import get_universe_pit


def _engine():
    engine = sa.create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[UniverseMembership.__table__, StockPrice.__table__], checkfirst=True)
    return engine


def test_get_universe_pit_uses_interval_membership(monkeypatch) -> None:
    engine = _engine()
    session_factory = sa.orm.sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(pit_module, "get_session_factory", lambda: session_factory)

    with session_factory() as session:
        session.add_all(
            [
                UniverseMembership(
                    ticker="AAA",
                    index_name="SP500",
                    effective_date=date(2016, 1, 1),
                    end_date=date(2017, 7, 1),
                    reason="anchor",
                ),
                UniverseMembership(
                    ticker="BBB",
                    index_name="SP500",
                    effective_date=date(2017, 7, 1),
                    end_date=None,
                    reason="replacement",
                ),
            ]
        )
        session.commit()

    early = get_universe_pit(datetime(2017, 3, 15, 12, tzinfo=timezone.utc), index_name="SP500")
    late = get_universe_pit(datetime(2017, 8, 15, 12, tzinfo=timezone.utc), index_name="SP500")

    assert early == ["AAA"]
    assert late == ["BBB"]


def test_resolve_active_universe_prefers_membership_and_filters_to_price_visible(monkeypatch) -> None:
    engine = _engine()
    session_factory = sa.orm.sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(active_module, "get_engine", lambda: engine)

    with session_factory() as session:
        session.add_all(
            [
                UniverseMembership(
                    ticker="AAA",
                    index_name="SP500",
                    effective_date=date(2026, 1, 1),
                    end_date=None,
                    reason="anchor",
                ),
                UniverseMembership(
                    ticker="BBB",
                    index_name="SP500",
                    effective_date=date(2026, 1, 1),
                    end_date=None,
                    reason="anchor",
                ),
                StockPrice(
                    ticker="AAA",
                    trade_date=date(2026, 4, 17),
                    open=Decimal("10"),
                    high=Decimal("11"),
                    low=Decimal("9"),
                    close=Decimal("10.5"),
                    adj_close=Decimal("10.5"),
                    volume=1000,
                    knowledge_time=datetime(2026, 4, 18, 20, tzinfo=timezone.utc),
                    source="test",
                ),
            ]
        )
        session.commit()

    tickers, source = active_module.resolve_active_universe(
        trade_date=date(2026, 4, 17),
        as_of=datetime(2026, 4, 18, 21, tzinfo=timezone.utc),
    )

    assert tickers == ["AAA"]
    assert source == "universe_membership"
