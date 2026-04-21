from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import sqlalchemy as sa
from sqlalchemy.pool import StaticPool

import src.data.event_calendar as event_calendar
from src.data.db.models import Base, StockPrice
from src.data.sources.fmp_earnings import EarningsEstimate


def _engine():
    engine = sa.create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine, tables=[StockPrice.__table__, EarningsEstimate.__table__], checkfirst=True)
    return engine


def _session_factory(engine):
    return sa.orm.sessionmaker(bind=engine, expire_on_commit=False)


def _config(
    *,
    stage: str = "pilot",
    reasons: list[str] | None = None,
    weak_window_top_n: int = 2,
    weak_windows: list[dict] | None = None,
    earnings_window_days: int = 3,
) -> dict:
    return {
        "version": 1,
        "stage": stage,
        "sampling": {
            "pilot": {
                "reasons": ["earnings", "gap", "weak_window"] if reasons is None else reasons,
                "earnings_window_days": earnings_window_days,
                "gap_threshold_pct": 0.03,
                "weak_window_top_n": weak_window_top_n,
                "weak_windows": weak_windows
                or [{"name": "W5", "start": "2025-01-02", "end": "2025-01-03"}],
            },
            "stage2": {"top_n_liquidity": 3, "top_liquidity_lookback_days": 20},
        },
        "polygon": {
            "entitlement_delay_minutes": 15,
            "rest_max_pages_per_request": 50,
            "rest_page_size": 50000,
            "rest_min_interval_seconds": 0.05,
            "retry_max": 3,
        },
        "budgets": {
            "max_daily_api_calls": 50000,
            "max_storage_gb": 200,
            "max_rows_per_ticker_day": 2000000,
            "expected_pilot_ticker_days": 30000,
        },
        "features": {
            "size_threshold_dollars": 1000000,
            "size_threshold_min_cap_dollars": 250000,
            "condition_allow_list": [],
            "trf_exchange_codes": [4, 202],
            "late_day_window_et": ["15:00", "16:00"],
            "offhours_window_et_pre": ["04:00", "09:30"],
            "offhours_window_et_post": ["16:00", "20:00"],
        },
        "gate": {
            "coverage_min_pct": 95.0,
            "feature_missing_max_pct": 30.0,
            "feature_outlier_max_pct": 5.0,
            "min_passing_features": 2,
            "ic_threshold": 0.015,
            "abs_tstat_threshold": 2.0,
            "sign_consistent_windows_min": 7,
        },
    }


def _earnings(ticker: str, fiscal_date: date, knowledge_time: datetime) -> EarningsEstimate:
    return EarningsEstimate(
        ticker=ticker,
        fiscal_date=fiscal_date,
        eps_estimated=Decimal("1.0"),
        eps_actual=Decimal("1.1"),
        revenue_estimated=100,
        revenue_actual=110,
        knowledge_time=knowledge_time,
        source="test",
    )


def _price(
    ticker: str,
    trade_date: date,
    *,
    open_: str,
    close: str,
    knowledge_time: datetime | None = None,
) -> StockPrice:
    return StockPrice(
        ticker=ticker,
        trade_date=trade_date,
        open=Decimal(open_),
        high=Decimal(open_),
        low=Decimal(close),
        close=Decimal(close),
        adj_close=Decimal(close),
        volume=1000,
        knowledge_time=knowledge_time or datetime(2025, 1, 3, 20, tzinfo=timezone.utc),
        source="test",
    )


def test_earnings_reason_expands_to_session_window() -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    with session_factory() as session:
        session.add(_earnings("AAPL", date(2025, 2, 1), datetime(2025, 1, 28, 20, tzinfo=timezone.utc)))
        session.commit()

    events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 29),
        end_date=date(2025, 2, 4),
        config=_config(reasons=["earnings"], earnings_window_days=3),
        session_factory=session_factory,
    )

    assert [(event.ticker, event.trading_date, event.reason) for event in events] == [
        ("AAPL", date(2025, 1, 29), "earnings"),
        ("AAPL", date(2025, 1, 30), "earnings"),
        ("AAPL", date(2025, 1, 31), "earnings"),
        ("AAPL", date(2025, 2, 3), "earnings"),
        ("AAPL", date(2025, 2, 4), "earnings"),
    ]


def test_gap_reason_detects_open_to_previous_close_gap() -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    with session_factory() as session:
        session.add_all(
            [
                _price("AAPL", date(2025, 1, 2), open_="100", close="100"),
                _price("AAPL", date(2025, 1, 3), open_="104", close="105"),
            ],
        )
        session.commit()

    events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 3),
        end_date=date(2025, 1, 3),
        config=_config(reasons=["gap"]),
        session_factory=session_factory,
    )

    assert events == [event_calendar.SamplingEvent("AAPL", date(2025, 1, 3), "gap")]


def test_gap_reason_respects_price_knowledge_time() -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    with session_factory() as session:
        session.add_all(
            [
                _price(
                    "AAPL",
                    date(2025, 1, 2),
                    open_="100",
                    close="100",
                    knowledge_time=datetime(2025, 1, 4, 20, tzinfo=timezone.utc),
                ),
                _price("AAPL", date(2025, 1, 3), open_="104", close="105"),
            ],
        )
        session.commit()

    events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 3),
        end_date=date(2025, 1, 3),
        config=_config(reasons=["gap"]),
        session_factory=session_factory,
    )

    assert events == []


def test_weak_window_uses_top_liquidity_cap(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    calls: list[tuple[date, int]] = []

    def fake_top_liquidity(as_of_date, *, top_n=200, lookback_days=20, session_factory=None):
        calls.append((as_of_date, top_n))
        return ["AAA", "BBB", "CCC", "DDD", "EEE"][:top_n]

    monkeypatch.setattr(event_calendar, "get_top_liquidity_tickers", fake_top_liquidity)

    events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        config=_config(reasons=["weak_window"], weak_window_top_n=2),
        session_factory=session_factory,
    )

    assert calls == [(date(2025, 1, 2), 2), (date(2025, 1, 3), 2)]
    assert len(events) == 4
    assert all(event.reason == "weak_window" for event in events)
    assert max(sum(event.trading_date == day for event in events) for day, _ in calls) == 2


def test_top_liquidity_reason_is_stage2_only(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)

    monkeypatch.setattr(
        event_calendar,
        "get_top_liquidity_tickers",
        lambda as_of_date, *, top_n=200, lookback_days=20, session_factory=None: ["AAA"],
    )

    pilot_events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 2),
        config=_config(stage="pilot", reasons=[]),
        session_factory=session_factory,
    )
    stage2_events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 2),
        config=_config(stage="stage2", reasons=[]),
        session_factory=session_factory,
    )

    assert pilot_events == []
    assert stage2_events == [event_calendar.SamplingEvent("AAA", date(2025, 1, 2), "top_liquidity")]


def test_same_ticker_date_keeps_multiple_reasons() -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    with session_factory() as session:
        session.add(_earnings("AAPL", date(2025, 1, 3), datetime(2025, 1, 1, 20, tzinfo=timezone.utc)))
        session.add_all(
            [
                _price("AAPL", date(2025, 1, 2), open_="100", close="100"),
                _price("AAPL", date(2025, 1, 3), open_="104", close="105"),
            ],
        )
        session.commit()

    events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 3),
        end_date=date(2025, 1, 3),
        config=_config(reasons=["earnings", "gap"], earnings_window_days=0),
        session_factory=session_factory,
    )

    assert events == [
        event_calendar.SamplingEvent("AAPL", date(2025, 1, 3), "earnings"),
        event_calendar.SamplingEvent("AAPL", date(2025, 1, 3), "gap"),
    ]


def test_stage_pilot_and_stage2_switching(monkeypatch) -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    monkeypatch.setattr(
        event_calendar,
        "get_top_liquidity_tickers",
        lambda as_of_date, *, top_n=200, lookback_days=20, session_factory=None: ["AAA"][:top_n],
    )

    pilot = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 2),
        config=_config(stage="pilot", reasons=["weak_window"]),
        session_factory=session_factory,
    )
    stage2 = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 2),
        config=_config(stage="stage2", reasons=["weak_window"]),
        session_factory=session_factory,
    )

    assert [event.reason for event in pilot] == ["weak_window"]
    assert [event.reason for event in stage2] == ["top_liquidity", "weak_window"]


def test_earnings_pit_blocks_unpublished_event() -> None:
    engine = _engine()
    session_factory = _session_factory(engine)
    with session_factory() as session:
        session.add(_earnings("AAPL", date(2025, 1, 3), datetime(2025, 1, 4, 20, tzinfo=timezone.utc)))
        session.commit()

    events = event_calendar.build_sampling_plan(
        start_date=date(2025, 1, 3),
        end_date=date(2025, 1, 3),
        config=_config(reasons=["earnings"], earnings_window_days=0),
        session_factory=session_factory,
    )

    assert events == []
