from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from src.data.sources.base import DataSourceTransientError, RetryConfig
from src.data.sources.fmp_ratings import FMPRatingsSource, RatingEvent


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


def _client(fake_session: _FakeSession) -> FMPRatingsSource:
    return FMPRatingsSource(min_request_interval=0, retry_config=_retry_config(), http_session=fake_session)


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_table(db_engine) -> None:
    RatingEvent.__table__.create(bind=db_engine, checkfirst=True)


_TEST_TICKER = "TSTRTG"  # unique prefix avoids clobbering real SP500 data


def _truncate_table(db_engine) -> None:
    _ensure_table(db_engine)
    with db_engine.begin() as conn:
        conn.execute(
            sa.text("DELETE FROM ratings_events WHERE ticker = :t"),
            {"t": _TEST_TICKER},
        )


def test_fetch_ticker_parses_ratings_and_pit_knowledge_time() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(
                payload=[
                    {
                        "date": "2026-04-23",
                        "rating": "A",
                        "overallScore": 4,
                        "discountedCashFlowScore": 3.25,
                        "priceToEarningsScore": 2.75,
                        "returnOnEquityScore": 4.5,
                    },
                ],
            ),
        ],
    )

    frame = _client(fake_session).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert frame["rating_score"].iloc[0] == 4
    assert frame["dcf_rating"].iloc[0] == Decimal("3.25")
    assert frame["knowledge_time"].iloc[0] == datetime(2026, 4, 24, 3, 59, tzinfo=timezone.utc)


def test_fetch_ticker_returns_empty_on_404() -> None:
    frame = _client(_FakeSession([_FakeResponse(status_code=404)])).fetch_ticker("AAPL")

    assert frame.empty


def test_fetch_ticker_retries_on_5xx_then_succeeds() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(status_code=500, text="server error"),
            _FakeResponse(
                payload=[
                    {
                        "date": "2026-04-23",
                        "rating": "B",
                        "overallScore": 5,
                        "discountedCashFlowScore": 4,
                        "priceToEarningsScore": 4,
                        "returnOnEquityScore": 5,
                    },
                ],
            ),
        ],
    )

    frame = _client(fake_session).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert len(fake_session.calls) == 2


def test_fetch_ticker_raises_on_persistent_429() -> None:
    fake_session = _FakeSession([_FakeResponse(status_code=429, text="rate limited") for _ in range(_retry_config().max_attempts)])

    with pytest.raises(DataSourceTransientError):
        _client(fake_session).fetch_ticker("AAPL")


def test_fetch_historical_persists_and_updates_row(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    fake_session = _FakeSession(
        [
            _FakeResponse(
                payload=[
                    {
                        "date": "2026-04-23",
                        "rating": "B",
                        "overallScore": 4,
                        "discountedCashFlowScore": 2.0,
                        "priceToEarningsScore": 2.0,
                        "returnOnEquityScore": 3.0,
                    },
                ],
            ),
            _FakeResponse(
                payload=[
                    {
                        "date": "2026-04-23",
                        "rating": "A",
                        "overallScore": 5,
                        "discountedCashFlowScore": 3.0,
                        "priceToEarningsScore": 3.0,
                        "returnOnEquityScore": 4.0,
                    },
                ],
            ),
        ],
    )
    client = _client(fake_session)

    assert client.fetch_historical([_TEST_TICKER], date(2026, 4, 1), date(2026, 4, 30)) == 1
    assert client.fetch_historical([_TEST_TICKER], date(2026, 4, 1), date(2026, 4, 30)) == 1

    with session_factory() as session:
        row = session.execute(
            sa.select(RatingEvent).where(RatingEvent.ticker == _TEST_TICKER)
        ).scalar_one()
    assert row.rating_score == 5
    assert row.rating_recommendation == "A"
