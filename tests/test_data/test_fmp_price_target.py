from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import src.data.sources.fmp_price_target as price_target_module
from src.data.sources.base import DataSourceTransientError, RetryConfig
from src.data.sources.fmp_price_target import FMPPriceTargetSource, PriceTargetEvent


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
) -> FMPPriceTargetSource:
    return FMPPriceTargetSource(
        min_request_interval=0,
        retry_config=_retry_config(),
        http_session=fake_session,
        now_fn=now_fn,
    )


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_table(db_engine) -> None:
    PriceTargetEvent.__table__.create(bind=db_engine, checkfirst=True)


def _truncate_table(db_engine) -> None:
    _ensure_table(db_engine)
    with db_engine.begin() as conn:
        conn.execute(sa.text("truncate table price_target_events restart identity"))


def test_fetch_ticker_merges_consensus_and_legacy_rows() -> None:
    now_dt = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
    fake_session = _FakeSession(
        [
            _FakeResponse(
                payload={
                    "symbol": "AAPL",
                    "targetConsensus": 210,
                    "targetMedian": 205,
                    "numAnalysts": 20,
                },
            ),
            _FakeResponse(
                payload=[
                    {
                        "publishedDate": "2026-04-23T08:00:00",
                        "analystName": "Jane Doe",
                        "analystCompany": "Acme Research",
                        "adjPriceTarget": 205,
                        "priceTarget": 210,
                        "targetChange": 10,
                    },
                ],
            ),
        ],
    )

    frame = _client(fake_session, now_fn=lambda: now_dt).fetch_ticker("AAPL")

    assert len(frame) == 2
    consensus_row = frame.loc[frame["is_consensus"]].iloc[0]
    analyst_row = frame.loc[~frame["is_consensus"]].iloc[0]
    assert consensus_row["knowledge_time"] == now_dt
    assert consensus_row["target_price"] == Decimal("210")
    assert analyst_row["knowledge_time"] == datetime(2026, 4, 24, 3, 59, tzinfo=timezone.utc)
    assert analyst_row["prior_target"] == Decimal("200")


def test_fetch_ticker_legacy_404_falls_back_to_consensus_only(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[tuple[str, tuple[Any, ...]]] = []

    class _FakeLogger:
        def warning(self, message: str, *args: Any) -> None:
            warnings.append((message, args))

    fake_session = _FakeSession(
        [
            _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 210}),
            _FakeResponse(status_code=404, text="not found"),
        ],
    )
    monkeypatch.setattr(price_target_module, "logger", _FakeLogger())

    frame = _client(fake_session, now_fn=lambda: datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert bool(frame["is_consensus"].iloc[0]) is True
    assert warnings


def test_fetch_ticker_legacy_403_deprecation_falls_back_to_consensus_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warnings: list[tuple[str, tuple[Any, ...]]] = []

    class _FakeLogger:
        def warning(self, message: str, *args: Any) -> None:
            warnings.append((message, args))

    fake_session = _FakeSession(
        [
            _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 210}),
            _FakeResponse(
                status_code=403,
                text="Legacy Endpoint : Due to Legacy endpoints being no longer supported",
            ),
        ],
    )
    monkeypatch.setattr(price_target_module, "logger", _FakeLogger())

    frame = _client(fake_session, now_fn=lambda: datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert bool(frame["is_consensus"].iloc[0]) is True
    assert warnings


def test_fetch_ticker_retries_on_429_then_succeeds() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(status_code=429, text="rate limited"),
            _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 210}),
            _FakeResponse(payload=[]),
        ],
    )

    frame = _client(fake_session, now_fn=lambda: datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert len(fake_session.calls) == 3


def test_fetch_ticker_raises_on_persistent_5xx() -> None:
    fake_session = _FakeSession([_FakeResponse(status_code=500, text="server error") for _ in range(_retry_config().max_attempts)])

    with pytest.raises(DataSourceTransientError):
        _client(fake_session).fetch_ticker("AAPL")


def test_fetch_historical_persists_consensus_without_duplicates(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    now_dt = datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)
    fake_session = _FakeSession(
        [
            _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 210}),
            _FakeResponse(status_code=404, text="not found"),
            _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 212}),
            _FakeResponse(status_code=404, text="not found"),
        ],
    )
    client = _client(fake_session, now_fn=lambda: now_dt)

    assert client.fetch_historical(["AAPL"], date(2026, 4, 1), date(2026, 4, 30)) == 1
    assert client.fetch_historical(["AAPL"], date(2026, 4, 1), date(2026, 4, 30)) == 1

    with session_factory() as session:
        rows = session.execute(sa.select(PriceTargetEvent)).scalars().all()
    assert len(rows) == 1
    assert rows[0].target_price == Decimal("212")
