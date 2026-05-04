from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.data.sources.fmp_analyst import FMPAnalystSource


class _FakeResponse:
    ok = True
    status_code = 200
    text = "ok"

    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self._payload = payload

    def json(self) -> list[dict[str, Any]]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(self, url: str, *, params: dict[str, Any], timeout: int) -> _FakeResponse:
        self.calls.append((url, params))
        return _FakeResponse(self.payload)


def test_fmp_analyst_source_uses_fetch_time_for_forward_estimate_pit() -> None:
    observed_at = datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc)
    fake_session = _FakeSession(
        [
            {
                "date": "2026-06-30",
                "epsAvg": 1.2,
                "epsHigh": 1.4,
                "epsLow": 1.0,
                "revenueAvg": 120_000_000,
                "revenueHigh": 130_000_000,
                "revenueLow": 110_000_000,
                "numAnalystsEps": None,
                "numAnalystsRevenue": 7,
            },
        ],
    )
    source = FMPAnalystSource(api_key="test-key", min_request_interval=0)
    source._http_session = fake_session

    rows = source._fetch_ticker("AAPL", observed_at=observed_at)

    assert rows[0]["fiscal_date"].isoformat() == "2026-06-30"
    assert rows[0]["knowledge_time"] == observed_at
    assert rows[0]["num_analysts_revenue"] == 7
