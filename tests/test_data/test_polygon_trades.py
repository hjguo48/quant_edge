from __future__ import annotations

from datetime import date, timezone
from decimal import Decimal
import json
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

import src.data.polygon_trades as polygon_trades
from src.data.polygon_trades import PolygonTradesClient
from src.data.sources.base import DataSourceTransientError, RetryConfig


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200, text: str | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any] | None, int]] = []

    def get(self, url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> _FakeResponse:
        self.calls.append((url, params, timeout))
        if not self.responses:
            raise AssertionError("Unexpected HTTP call")
        return self.responses.pop(0)


def _retry_config() -> RetryConfig:
    return RetryConfig(
        max_attempts=3,
        initial_delay=0,
        backoff_factor=1,
        max_delay=0,
        jitter=0,
    )


def _client(fake_session: _FakeSession) -> PolygonTradesClient:
    return PolygonTradesClient(
        api_key="test-key",
        min_request_interval=0,
        retry_config=_retry_config(),
        http_session=fake_session,
    )


def _ns(raw: str) -> int:
    return int(pd.Timestamp(raw, tz="UTC").value)


def _trade(
    *,
    trade_id: str = "tr-1",
    seq: int | None = 1,
    exchange: int | None = 4,
    price: str = "101.25",
    size: str = "100",
    sip_ts: str = "2026-01-05 14:30:00",
    participant_ts: str = "2026-01-05 14:29:59.999999",
    conditions: list[int] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": trade_id,
        "sip_timestamp": _ns(sip_ts),
        "participant_timestamp": _ns(participant_ts),
        "price": price,
        "size": size,
        "decimal_size": "100.00000000",
        "tape": 1,
        "conditions": conditions if conditions is not None else [12, 37],
        "correction": 0,
        "trf_timestamp": _ns("2026-01-05 14:30:00.000001"),
        "trf_id": "TRF1",
    }
    if seq is not None:
        payload["sequence_number"] = seq
    if exchange is not None:
        payload["exchange"] = exchange
    return payload


def test_fetch_trades_for_day_paginates_three_pages_and_normalizes_records() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(
                {
                    "results": [_trade(trade_id="tr-1", seq=1)],
                    "next_url": "https://api.polygon.io/v3/trades/AAPL?cursor=page2",
                },
            ),
            _FakeResponse(
                {
                    "results": [_trade(trade_id="tr-2", seq=2, price="102.50")],
                    "next_url": "https://api.polygon.io/v3/trades/AAPL?cursor=page3",
                },
            ),
            _FakeResponse({"results": [_trade(trade_id="tr-3", seq=None, exchange=None, price="103.50")]}),
        ],
    )

    records, page_count = _client(fake_session).fetch_trades_for_day("AAPL", date(2026, 1, 5))

    assert len(records) == 3
    assert page_count == 3
    assert records[0].ticker == "AAPL"
    assert records[0].trading_date == date(2026, 1, 5)
    assert records[0].sip_timestamp.tzinfo == timezone.utc
    assert records[0].price == Decimal("101.25")
    assert records[0].size == Decimal("100")
    assert records[0].decimal_size == Decimal("100.00000000")
    assert records[0].conditions == [12, 37]
    assert records[2].exchange == -1
    assert records[2].sequence_number < 0
    assert len(fake_session.calls) == 3

    first_url, first_params, _ = fake_session.calls[0]
    assert first_url == "https://api.polygon.io/v3/trades/AAPL"
    assert first_params == {
        "timestamp": "2026-01-05",
        "limit": 50_000,
        "apiKey": "test-key",
    }

    second_url, second_params, _ = fake_session.calls[1]
    assert second_params is None
    assert parse_qs(urlparse(second_url).query)["apiKey"] == ["test-key"]


def test_fetch_trades_for_day_retries_429_then_succeeds() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse({"error": "rate limit"}, status_code=429, text="rate limit"),
            _FakeResponse({"results": [_trade(seq=10)]}),
        ],
    )

    records, page_count = _client(fake_session).fetch_trades_for_day("MSFT", date(2026, 1, 5))

    assert len(records) == 1
    assert page_count == 2
    assert records[0].sequence_number == 10
    assert len(fake_session.calls) == 2


def test_fetch_trades_for_day_raises_after_429_retries_exhausted() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse({"error": "rate limit"}, status_code=429, text="rate limit"),
            _FakeResponse({"error": "rate limit"}, status_code=429, text="rate limit"),
            _FakeResponse({"error": "rate limit"}, status_code=429, text="rate limit"),
        ],
    )

    with pytest.raises(DataSourceTransientError):
        _client(fake_session).fetch_trades_for_day("NVDA", date(2026, 1, 5))

    assert len(fake_session.calls) == 3


def test_fetch_trades_for_day_retries_5xx_then_succeeds() -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse({"error": "upstream"}, status_code=503, text="upstream"),
            _FakeResponse({"results": [_trade(seq=20)]}),
        ],
    )

    records, page_count = _client(fake_session).fetch_trades_for_day("GOOGL", date(2026, 1, 5))

    assert len(records) == 1
    assert page_count == 2
    assert records[0].sequence_number == 20
    assert len(fake_session.calls) == 2


def test_fetch_trades_for_day_empty_response_returns_empty_iterator() -> None:
    fake_session = _FakeSession([_FakeResponse({"results": []})])

    records, page_count = _client(fake_session).fetch_trades_for_day("META", date(2026, 1, 5))

    assert records == []
    assert page_count == 1
    assert len(fake_session.calls) == 1


def test_fetch_trades_for_day_max_pages_truncates_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_session = _FakeSession(
        [
            _FakeResponse(
                {
                    "results": [_trade(trade_id="tr-1", seq=1)],
                    "next_url": "https://api.polygon.io/v3/trades/TSLA?cursor=page2",
                },
            ),
            _FakeResponse(
                {
                    "results": [_trade(trade_id="tr-2", seq=2)],
                    "next_url": "https://api.polygon.io/v3/trades/TSLA?cursor=page3",
                },
            ),
            _FakeResponse({"results": [_trade(trade_id="tr-3", seq=3)]}),
        ],
    )
    warnings: list[tuple[str, tuple[Any, ...]]] = []

    class _FakeLogger:
        def warning(self, message: str, *args: Any) -> None:
            warnings.append((message, args))

    monkeypatch.setattr(polygon_trades, "logger", _FakeLogger())

    records, page_count = _client(fake_session).fetch_trades_for_day(
        "TSLA",
        date(2026, 1, 5),
        max_pages=2,
    )

    assert [record.sequence_number for record in records] == [1, 2]
    assert page_count == 2
    assert len(fake_session.calls) == 2
    assert warnings
    assert "max_pages" in warnings[0][0]
