from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from src.data.sources.base import DataSourceTransientError, RetryConfig
from src.data.sources.fmp_grades import FMPGradesSource, GradesEvent


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


def _client(fake_session: _FakeSession) -> FMPGradesSource:
    return FMPGradesSource(min_request_interval=0, retry_config=_retry_config(), http_session=fake_session)


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_table(db_engine) -> None:
    GradesEvent.__table__.create(bind=db_engine, checkfirst=True)


_TEST_TICKER = "TSTGRD"  # unique prefix avoids clobbering real SP500 data


def _truncate_table(db_engine) -> None:
    _ensure_table(db_engine)
    with db_engine.begin() as conn:
        # NEVER truncate the full table — production data would be wiped.
        conn.execute(
            sa.text("DELETE FROM grades_events WHERE ticker = :t"),
            {"t": _TEST_TICKER},
        )


def test_fetch_ticker_parses_grades_and_pit_knowledge_time() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(
                payload=[
                    {
                        "date": "2026-04-23",
                        "newGrade": "Buy",
                        "previousGrade": "Hold",
                        "gradingCompany": "Acme Research",
                        "action": "upgraded",
                    },
                ],
            ),
        ],
    )

    frame = _client(fake_session).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert frame["ticker"].iloc[0] == "AAPL"
    assert frame["grade_score_change"].iloc[0] == 2
    assert frame["knowledge_time"].iloc[0] == datetime(2026, 4, 24, 3, 59, tzinfo=timezone.utc)


def test_fetch_ticker_returns_empty_frame_on_404() -> None:
    frame = _client(_FakeSession([_FakeResponse(status_code=404)])).fetch_ticker("AAPL")

    assert frame.empty
    assert list(frame.columns) == [
        "ticker",
        "event_date",
        "knowledge_time",
        "analyst_firm",
        "prior_grade",
        "new_grade",
        "action",
        "grade_score_change",
    ]


def test_fetch_ticker_retries_on_429_then_succeeds() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(status_code=429, text="rate limited"),
            _FakeResponse(
                payload=[
                    {
                        "date": "2026-04-23",
                        "newGrade": "Hold",
                        "previousGrade": "Sell",
                        "gradingCompany": "Acme Research",
                        "action": "upgraded",
                    },
                ],
            ),
        ],
    )

    frame = _client(fake_session).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert len(fake_session.calls) == 2


def test_fetch_ticker_raises_on_persistent_5xx() -> None:
    fake_session = _FakeSession([_FakeResponse(status_code=503, text="oops") for _ in range(_retry_config().max_attempts)])

    with pytest.raises(DataSourceTransientError):
        _client(fake_session).fetch_ticker("AAPL")

    assert len(fake_session.calls) == _retry_config().max_attempts


def test_fetch_historical_persists_rows(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    fake_session = _FakeSession(
        [
            _FakeResponse(
                payload=[
                    {
                        "date": "2026-04-23",
                        "newGrade": "Buy",
                        "previousGrade": "Hold",
                        "gradingCompany": "Acme Research",
                        "action": "upgraded",
                    },
                ],
            ),
        ],
    )

    inserted = _client(fake_session).fetch_historical([_TEST_TICKER], date(2026, 4, 1), date(2026, 4, 30))

    assert inserted == 1
    with session_factory() as session:
        row = session.execute(
            sa.select(GradesEvent).where(GradesEvent.ticker == _TEST_TICKER)
        ).scalar_one()
    assert row.ticker == _TEST_TICKER
    assert row.new_grade == "Buy"
