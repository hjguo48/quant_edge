from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.pool import StaticPool

import src.universe.top_liquidity as top_liquidity_module
from src.data.db.models import Base, StockPrice


def _engine():
    engine = sa.create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[StockPrice.__table__], checkfirst=True)
    return engine


def _session_factory(engine):
    return sa.orm.sessionmaker(bind=engine, expire_on_commit=False)


def _price(
    ticker: str,
    trade_date: date,
    *,
    close: str,
    volume: int,
    knowledge_time: datetime | None = None,
) -> StockPrice:
    return StockPrice(
        ticker=ticker,
        trade_date=trade_date,
        open=Decimal(close),
        high=Decimal(close),
        low=Decimal(close),
        close=Decimal(close),
        adj_close=Decimal(close),
        volume=volume,
        knowledge_time=knowledge_time or datetime(2026, 1, 10, 20, tzinfo=timezone.utc),
        source="test",
    )


def test_top_n_ranks_by_average_dollar_volume(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    monkeypatch.setattr(
        top_liquidity_module,
        "get_historical_members",
        lambda as_of, index_name="SP500": ["AAA", "BBB", "CCC", "DDD", "EEE"],
    )

    with session_factory() as session:
        session.add_all(
            [
                _price("AAA", date(2026, 1, 8), close="10", volume=100),
                _price("AAA", date(2026, 1, 9), close="10", volume=100),
                _price("BBB", date(2026, 1, 8), close="20", volume=100),
                _price("BBB", date(2026, 1, 9), close="20", volume=100),
                _price("CCC", date(2026, 1, 8), close="5", volume=100),
                _price("CCC", date(2026, 1, 9), close="5", volume=100),
                _price("DDD", date(2026, 1, 8), close="30", volume=100),
                _price("DDD", date(2026, 1, 9), close="30", volume=100),
                _price("EEE", date(2026, 1, 8), close="1", volume=100),
                _price("EEE", date(2026, 1, 9), close="1", volume=100),
            ],
        )
        session.commit()

    tickers = top_liquidity_module.get_top_liquidity_tickers(
        date(2026, 1, 10),
        top_n=3,
        lookback_days=2,
        session_factory=session_factory,
    )

    assert tickers == ["DDD", "BBB", "AAA"]


def test_top_liquidity_respects_pit_knowledge_time(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    monkeypatch.setattr(
        top_liquidity_module,
        "get_historical_members",
        lambda as_of, index_name="SP500": ["AAA", "BBB"],
    )

    with session_factory() as session:
        session.add_all(
            [
                _price(
                    "AAA",
                    date(2026, 1, 9),
                    close="1000",
                    volume=1000,
                    knowledge_time=datetime(2026, 1, 11, 5, 1, tzinfo=timezone.utc),
                ),
                _price("BBB", date(2026, 1, 9), close="10", volume=100),
            ],
        )
        session.commit()

    tickers = top_liquidity_module.get_top_liquidity_tickers(
        date(2026, 1, 10),
        top_n=2,
        lookback_days=2,
        session_factory=session_factory,
    )

    assert tickers == ["BBB"]


def test_top_liquidity_filters_to_historical_members(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    monkeypatch.setattr(
        top_liquidity_module,
        "get_historical_members",
        lambda as_of, index_name="SP500": ["AAA"],
    )

    with session_factory() as session:
        session.add_all(
            [
                _price("AAA", date(2026, 1, 9), close="10", volume=100),
                _price("BBB", date(2026, 1, 9), close="1000", volume=1000),
            ],
        )
        session.commit()

    tickers = top_liquidity_module.get_top_liquidity_tickers(
        date(2026, 1, 10),
        top_n=2,
        lookback_days=2,
        session_factory=session_factory,
    )

    assert tickers == ["AAA"]


def test_top_liquidity_falls_back_to_available_history_and_logs_warning(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    warnings: list[tuple[str, tuple]] = []

    class FakeLogger:
        def warning(self, message: str, *args) -> None:
            warnings.append((message, args))

    monkeypatch.setattr(top_liquidity_module, "logger", FakeLogger())
    monkeypatch.setattr(
        top_liquidity_module,
        "get_historical_members",
        lambda as_of, index_name="SP500": ["AAA", "BBB"],
    )

    with session_factory() as session:
        session.add_all(
            [
                _price("AAA", date(2026, 1, 9), close="10", volume=100),
                _price("BBB", date(2026, 1, 9), close="20", volume=100),
            ],
        )
        session.commit()

    tickers = top_liquidity_module.get_top_liquidity_tickers(
        date(2026, 1, 10),
        top_n=2,
        lookback_days=20,
        session_factory=session_factory,
    )

    assert tickers == ["BBB", "AAA"]
    assert any("falling back to available history" in message for message, _ in warnings)


def test_top_liquidity_calls_get_historical_members(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    calls: list[tuple[date, str]] = []

    def fake_get_historical_members(as_of, index_name="SP500"):
        calls.append((as_of, index_name))
        return ["AAA"]

    monkeypatch.setattr(top_liquidity_module, "get_historical_members", fake_get_historical_members)

    with session_factory() as session:
        session.add(_price("AAA", date(2026, 1, 9), close="10", volume=100))
        session.commit()

    tickers = top_liquidity_module.get_top_liquidity_tickers(
        date(2026, 1, 10),
        top_n=1,
        lookback_days=2,
        session_factory=session_factory,
    )

    assert tickers == ["AAA"]
    assert calls == [(date(2026, 1, 10), "SP500")]
