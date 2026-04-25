from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from src.data.sources.base import DataSourceTransientError, RetryConfig
from src.data.sources.fmp_earnings_calendar import EarningsCalendar, FMPEarningsCalendarSource


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: Any = None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], int]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> _FakeResponse:
        self.calls.append((url, params or {}, timeout))
        if not self.responses:
            raise AssertionError("Unexpected GET call")
        return self.responses.pop(0)


def _retry_config() -> RetryConfig:
    return RetryConfig(max_attempts=5, initial_delay=0, backoff_factor=1, max_delay=0, jitter=0)


def _client(
    fake_session: _FakeSession,
    *,
    now_fn: Any | None = None,
) -> FMPEarningsCalendarSource:
    return FMPEarningsCalendarSource(
        min_request_interval=0,
        retry_config=_retry_config(),
        http_session=fake_session,
        now_fn=now_fn,
    )


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_table(db_engine) -> None:
    EarningsCalendar.__table__.create(bind=db_engine, checkfirst=True)


_TEST_TICKER = "TSTERN"  # unique prefix avoids clobbering real SP500 data


def _truncate_table(db_engine) -> None:
    _ensure_table(db_engine)
    with db_engine.begin() as conn:
        conn.execute(
            sa.text("DELETE FROM earnings_calendar WHERE ticker = :t"),
            {"t": _TEST_TICKER},
        )


def test_fetch_range_parses_rows_and_pit_knowledge_time() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(
                payload=[
                    {
                        "symbol": _TEST_TICKER,
                        "date": "2026-04-23",
                        "epsEstimated": 1.25,
                        "eps": None,
                        "time": "amc",
                        "fiscalDateEnding": "2026-03-31",
                        "revenueEstimated": 100000000,
                        "revenue": None,
                        "updatedFromDate": None,
                    },
                    {
                        "symbol": "MSFT",
                        "date": "2026-04-23",
                        "epsEstimated": 2.25,
                        "eps": 2.5,
                        "time": "bmo",
                        "fiscalDateEnding": "2026-03-31",
                        "revenueEstimated": 200000000,
                        "revenue": 210000000,
                        "updatedFromDate": "2026-04-24",
                    },
                ],
            ),
        ],
    )

    frame = _client(fake_session).fetch_range(date(2026, 4, 23), date(2026, 4, 23))

    assert len(frame) == 2
    aapl_row = frame.loc[frame["ticker"] == _TEST_TICKER].iloc[0]
    msft_row = frame.loc[frame["ticker"] == "MSFT"].iloc[0]
    assert aapl_row["knowledge_time"] == datetime(2026, 4, 24, 3, 59, tzinfo=timezone.utc)
    assert msft_row["knowledge_time"] == datetime(2026, 4, 25, 3, 59, tzinfo=timezone.utc)
    assert msft_row["eps_actual"] == Decimal("2.5")


def test_fetch_range_splits_requests_into_30_day_chunks() -> None:
    # 2025-01-01 → 2026-02-01 spans 397 days; 30-day chunks → 14 calls.
    fake_session = _FakeSession([_FakeResponse(payload=[]) for _ in range(20)])

    _client(fake_session).fetch_range(date(2025, 1, 1), date(2026, 2, 1))

    # 397 days / 30 = 14 chunks (last is partial).
    assert len(fake_session.calls) == 14


def test_fetch_range_returns_empty_on_404() -> None:
    frame = _client(_FakeSession([_FakeResponse(status_code=404)])).fetch_range(date(2026, 4, 23), date(2026, 4, 23))

    assert frame.empty


def test_fetch_range_retries_on_429_then_succeeds() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(status_code=429, text="rate limited"),
            _FakeResponse(payload=[]),
        ],
    )

    frame = _client(fake_session).fetch_range(date(2026, 4, 23), date(2026, 4, 23))

    assert frame.empty
    assert len(fake_session.calls) == 2


def test_fetch_range_raises_on_persistent_5xx() -> None:
    fake_session = _FakeSession([_FakeResponse(status_code=503, text="server error") for _ in range(_retry_config().max_attempts)])

    with pytest.raises(DataSourceTransientError):
        _client(fake_session).fetch_range(date(2026, 4, 23), date(2026, 4, 23))


def test_revised_eps_advances_knowledge_time(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    now_dt = datetime(2026, 4, 25, 12, 0, tzinfo=timezone.utc)
    fake_session = _FakeSession(
        [
            _FakeResponse(
                payload=[
                    {
                        "symbol": _TEST_TICKER,
                        "date": "2026-04-23",
                        "epsEstimated": 1.25,
                        "eps": None,
                        "time": "amc",
                        "fiscalDateEnding": "2026-03-31",
                        "revenueEstimated": 100000000,
                        "revenue": None,
                        "updatedFromDate": None,
                    },
                ],
            ),
            _FakeResponse(
                payload=[
                    {
                        "symbol": _TEST_TICKER,
                        "date": "2026-04-23",
                        "epsEstimated": 1.25,
                        "eps": 1.5,
                        "time": "amc",
                        "fiscalDateEnding": "2026-03-31",
                        "revenueEstimated": 100000000,
                        "revenue": 101000000,
                        "updatedFromDate": "2026-04-24",
                    },
                ],
            ),
        ],
    )
    client = _client(fake_session, now_fn=lambda: now_dt)

    assert client.fetch_historical([_TEST_TICKER], date(2026, 4, 23), date(2026, 4, 23)) == 1
    assert client.fetch_historical([_TEST_TICKER], date(2026, 4, 23), date(2026, 4, 23)) == 1

    with session_factory() as session:
        row = session.execute(
            sa.select(EarningsCalendar).where(EarningsCalendar.ticker == _TEST_TICKER)
        ).scalar_one()
    assert row.eps_actual == Decimal("1.5")
    assert row.knowledge_time >= now_dt
