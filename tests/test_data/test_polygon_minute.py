from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.polygon_minute import EASTERN, PolygonMinuteClient


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = "ok"

    def json(self) -> dict[str, object]:
        return self._payload


class _FakeSession:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, dict[str, object] | None]] = []

    def get(self, url: str, params: dict[str, object] | None = None, timeout: int = 30) -> _FakeResponse:
        self.calls.append((url, params))
        return _FakeResponse(self.payload)


def _utc_ms(raw: str) -> int:
    return int(pd.Timestamp(raw, tz="UTC").timestamp() * 1000)


def test_get_minute_aggs_filters_regular_session_and_assigns_next_session_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "results": [
            {"t": _utc_ms("2026-01-05 14:29"), "o": 99.0, "h": 99.5, "l": 98.9, "c": 99.2, "v": 100, "vw": 99.1, "n": 10},
            {"t": _utc_ms("2026-01-05 14:30"), "o": 100.0, "h": 101.0, "l": 99.8, "c": 100.5, "v": 1000, "vw": 100.4, "n": 100},
            {"t": _utc_ms("2026-01-05 20:59"), "o": 101.0, "h": 101.5, "l": 100.8, "c": 101.2, "v": 2000, "vw": 101.1, "n": 200},
            {"t": _utc_ms("2026-01-05 21:00"), "o": 101.3, "h": 101.4, "l": 101.2, "c": 101.3, "v": 50, "vw": 101.3, "n": 5},
        ],
    }
    client = PolygonMinuteClient(api_key="test-key")
    fake_session = _FakeSession(payload)

    monkeypatch.setattr(client, "_get_http_session", lambda: fake_session)

    frame = client.get_minute_aggs("AAPL", date(2026, 1, 5), date(2026, 1, 5))

    # Polygon bar `t` is the START of a 1-min window.
    # Valid regular bars: t in [session_open, session_close) → 09:30 and 15:59 ET.
    # Excluded: 09:29 ET (pre-open) and 16:00 ET (bar covers 16:00-16:01, post-close).
    assert len(frame) == 2
    assert frame["ticker"].tolist() == ["AAPL", "AAPL"]
    assert frame["trade_date"].tolist() == [date(2026, 1, 5), date(2026, 1, 5)]

    minute_ts = pd.to_datetime(frame["minute_ts"])
    minute_ts_et = minute_ts.dt.tz_convert(EASTERN)
    assert minute_ts_et.dt.strftime("%H:%M").tolist() == ["09:30", "15:59"]

    knowledge_times = pd.to_datetime(frame["knowledge_time"], utc=True)
    assert set(knowledge_times.dt.strftime("%Y-%m-%d %H:%M:%S%z")) == {"2026-01-06 21:00:00+0000"}


def test_get_minute_aggs_uses_canonical_ticker_and_polygon_provider_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"results": [{"t": _utc_ms("2026-01-05 14:30"), "o": 100.0, "h": 100.0, "l": 100.0, "c": 100.0, "v": 1, "vw": 100.0, "n": 1}]}
    client = PolygonMinuteClient(api_key="test-key")
    fake_session = _FakeSession(payload)
    monkeypatch.setattr(client, "_get_http_session", lambda: fake_session)

    frame = client.get_minute_aggs("BRK-B", date(2026, 1, 5), date(2026, 1, 5))

    assert frame["ticker"].tolist() == ["BRK-B"]
    assert fake_session.calls
    request_url, request_params = fake_session.calls[0]
    assert "/BRK.B/range/1/minute/" in request_url
    assert request_params is not None
    assert request_params["adjusted"] == "true"


def test_is_regular_session_bar_excludes_16_00_and_early_close() -> None:
    # Polygon bar `t` is START of minute window; 16:00 ET is first post-close bar.
    bar_15_59 = pd.Timestamp("2026-01-05 20:59", tz="UTC")  # 15:59 ET
    bar_16_00 = pd.Timestamp("2026-01-05 21:00", tz="UTC")  # 16:00 ET → post-close
    assert PolygonMinuteClient._is_regular_session_bar(bar_15_59) is True
    assert PolygonMinuteClient._is_regular_session_bar(bar_16_00) is False

    # Early-close day: day after Thanksgiving 2025-11-28, market closes 13:00 ET.
    bar_12_59_early = pd.Timestamp("2025-11-28 17:59", tz="UTC")  # 12:59 ET
    bar_13_00_early = pd.Timestamp("2025-11-28 18:00", tz="UTC")  # 13:00 ET → post-close
    bar_15_00_early = pd.Timestamp("2025-11-28 20:00", tz="UTC")  # 15:00 ET → post-close
    assert PolygonMinuteClient._is_regular_session_bar(bar_12_59_early) is True
    assert PolygonMinuteClient._is_regular_session_bar(bar_13_00_early) is False
    assert PolygonMinuteClient._is_regular_session_bar(bar_15_00_early) is False

    # Non-session day (weekend).
    bar_weekend = pd.Timestamp("2026-01-03 15:00", tz="UTC")  # Saturday
    assert PolygonMinuteClient._is_regular_session_bar(bar_weekend) is False
