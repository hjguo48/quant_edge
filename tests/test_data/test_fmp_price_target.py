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


_TEST_TICKER = "TSTPT"  # unique prefix avoids clobbering real SP500 data


def _truncate_table(db_engine) -> None:
    _ensure_table(db_engine)
    with db_engine.begin() as conn:
        conn.execute(
            sa.text("DELETE FROM price_target_events WHERE ticker = :t"),
            {"t": _TEST_TICKER},
        )


def test_fetch_ticker_merges_consensus_and_per_analyst_news_rows() -> None:
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
                        "symbol": "AAPL",
                        "publishedDate": "2026-04-23T08:00:00.000Z",
                        "analystName": "Jane Doe",
                        "analystCompany": "Acme Research",
                        "adjPriceTarget": 205,
                        "priceTarget": 210,
                        "priceWhenPosted": 200,
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
    # knowledge_time uses EOD(event_date) in America/New_York → 03:59 UTC next day.
    # This keeps PIT alignment with the Week 5 lag-rule gate.
    assert analyst_row["knowledge_time"] == datetime(2026, 4, 24, 3, 59, tzinfo=timezone.utc)
    assert analyst_row["target_price"] == Decimal("205")  # adjPriceTarget preferred
    assert analyst_row["price_when_published"] == Decimal("200")
    # news endpoint does not provide targetChange; prior_target stays None.
    assert analyst_row["prior_target"] is None


def test_fetch_ticker_news_404_keeps_consensus_only() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 210}),
            _FakeResponse(status_code=404, text="not found"),
        ],
    )

    frame = _client(fake_session, now_fn=lambda: datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)).fetch_ticker("AAPL")

    assert len(frame) == 1
    assert bool(frame["is_consensus"].iloc[0]) is True


def test_fetch_ticker_news_pagination_terminates_on_short_page() -> None:
    # Page 0 returns 100 rows (full page) → request page 1.
    # Page 1 returns 2 rows (<100) → terminate pagination.
    page0_records = [
        {
            "symbol": "AAPL",
            "publishedDate": f"2026-04-{(idx % 25) + 1:02d}T08:00:00.000Z",
            "analystCompany": f"Firm {idx}",
            "adjPriceTarget": 200 + idx,
            "priceTarget": 200 + idx,
        }
        for idx in range(100)
    ]
    page1_records = [
        {
            "symbol": "AAPL",
            "publishedDate": "2023-01-15T08:00:00.000Z",
            "analystCompany": "Older Firm",
            "adjPriceTarget": 150,
            "priceTarget": 150,
        },
        {
            "symbol": "AAPL",
            "publishedDate": "2022-11-01T08:00:00.000Z",
            "analystCompany": "Even Older Firm",
            "adjPriceTarget": 140,
            "priceTarget": 140,
        },
    ]
    fake_session = _FakeSession(
        [
            _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 210}),
            _FakeResponse(payload=page0_records),
            _FakeResponse(payload=page1_records),
        ],
    )

    frame = _client(fake_session, now_fn=lambda: datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)).fetch_ticker("AAPL")

    # 1 consensus + 100 page0 + 2 page1 = 103 rows total.
    assert len(frame) == 103
    # Only 3 HTTP calls (consensus + 2 news pages, not page 2).
    assert len(fake_session.calls) == 3


def test_fetch_ticker_news_dedupes_same_day_same_firm_by_publication_time() -> None:
    """Same (event_date, analyst_firm) revisions: latest publishedDate wins, not payload order.

    Regresses a critical bug where EOD-based knowledge_time made same-day
    revisions tie on sort, so HTTP response order decided the winner.
    """
    older = {
        "symbol": "AAPL",
        "publishedDate": "2026-04-23T08:00:00.000Z",
        "analystCompany": "Acme Research",
        "adjPriceTarget": 200,
        "priceTarget": 200,
    }
    newer = {
        "symbol": "AAPL",
        "publishedDate": "2026-04-23T15:00:00.000Z",
        "analystCompany": "Acme Research",
        "adjPriceTarget": 250,
        "priceTarget": 250,
    }

    for order_label, payload in [("older-first", [older, newer]), ("newer-first", [newer, older])]:
        fake_session = _FakeSession(
            [
                _FakeResponse(payload={"symbol": "AAPL", "targetConsensus": 210}),
                _FakeResponse(payload=payload),
            ],
        )
        frame = _client(fake_session, now_fn=lambda: datetime(2026, 4, 24, 12, 0, tzinfo=timezone.utc)).fetch_ticker("AAPL")
        analyst_rows = frame.loc[~frame["is_consensus"]]
        assert len(analyst_rows) == 1, f"{order_label}: expected 1 dedup'd row"
        assert analyst_rows.iloc[0]["target_price"] == Decimal("250"), (
            f"{order_label}: newer publishedDate should win regardless of HTTP order"
        )


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
            _FakeResponse(payload={"symbol": _TEST_TICKER, "targetConsensus": 210}),
            _FakeResponse(status_code=404, text="not found"),
            _FakeResponse(payload={"symbol": _TEST_TICKER, "targetConsensus": 212}),
            _FakeResponse(status_code=404, text="not found"),
        ],
    )
    client = _client(fake_session, now_fn=lambda: now_dt)

    assert client.fetch_historical([_TEST_TICKER], date(2026, 4, 1), date(2026, 4, 30)) == 1
    assert client.fetch_historical([_TEST_TICKER], date(2026, 4, 1), date(2026, 4, 30)) == 1

    with session_factory() as session:
        rows = session.execute(
            sa.select(PriceTargetEvent).where(PriceTargetEvent.ticker == _TEST_TICKER)
        ).scalars().all()
    assert len(rows) == 1
    assert rows[0].target_price == Decimal("212")
