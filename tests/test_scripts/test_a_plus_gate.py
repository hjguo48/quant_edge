from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from scripts.run_intraday_smoke import (
    persist_reconciliation_events,
    validate_minute_internal_consistency,
    validate_minute_to_day_consistency,
)


def _minute_rows_for_day(
    *,
    ticker: str = "AAA",
    trade_day: date = date(2026, 1, 5),
    open_price: float = 100.0,
    close_price: float = 101.0,
) -> pd.DataFrame:
    minute_index = pd.date_range("2026-01-05 14:30", "2026-01-05 21:00", freq="min", tz="UTC")
    prices = pd.Series(
        [open_price + (close_price - open_price) * idx / (len(minute_index) - 1) for idx in range(len(minute_index))]
    )
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": trade_day,
            "minute_ts": minute_index,
            "open": prices,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices,
            "volume": 100,
            "vwap": prices,
            "transactions": 10,
        },
    )
    return frame


def _daily_row(
    *,
    ticker: str = "AAA",
    trade_day: date = date(2026, 1, 5),
    open_price: float = 100.0,
    high_price: float = 101.05,
    low_price: float = 99.95,
    close_price: float = 101.0,
    volume: float = 39_100,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "trade_date": trade_day,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume,
            },
        ],
    )


def test_blocker_fails_on_ohl_diff() -> None:
    minute_df = _minute_rows_for_day()
    daily_df = _daily_row(open_price=100.30)

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["pass"] is False
    assert result["fields"]["open"]["pass"] is False
    assert result["fields"]["open"]["severity"] == "blocker"


def test_warning_only_on_close() -> None:
    minute_df = _minute_rows_for_day(close_price=101.0)
    daily_df = _daily_row(close_price=101.25)

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["pass"] is True
    assert result["fields"]["close"]["pass"] is False
    assert result["fields"]["close"]["severity"] == "warning"
    assert len(result["warning_events"]) == 1
    assert result["warning_events"][0]["field"] == "close"


def test_minute_internal_consistency_gap() -> None:
    minute_df = _minute_rows_for_day().drop(index=[0, 1, 2, 3]).reset_index(drop=True)

    result = validate_minute_internal_consistency(minute_df)

    assert result["pass"] is False
    assert result["checks"]["gap_free"]["pass"] is False


def test_minute_internal_consistency_overlap() -> None:
    minute_df = _minute_rows_for_day()
    minute_df = pd.concat([minute_df, minute_df.iloc[[0]]], ignore_index=True)

    result = validate_minute_internal_consistency(minute_df)

    assert result["pass"] is False
    assert result["checks"]["no_overlap"]["pass"] is False


class _FakeSession:
    def __init__(self) -> None:
        self.added: list[object] = []
        self.committed = False
        self.closed = False

    def add_all(self, rows: list[object]) -> None:
        self.added.extend(rows)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        raise AssertionError("rollback should not be called in success path")

    def close(self) -> None:
        self.closed = True


def test_reconciliation_event_written() -> None:
    fake_session = _FakeSession()
    events = [
        {
            "ticker": "AAA",
            "trade_date": date(2026, 1, 5),
            "field": "close",
            "stock_prices_value": 101.25,
            "minute_agg_value": 101.0,
            "delta_bp": 24.72,
            "severity": "warning",
        },
    ]

    inserted = persist_reconciliation_events(
        events,
        batch_id="batch-1",
        detected_at=datetime.now(timezone.utc),
        session_factory=lambda: fake_session,
    )

    assert inserted == 1
    assert fake_session.committed is True
    assert fake_session.closed is True
    assert len(fake_session.added) == 1
    row = fake_session.added[0]
    assert row.ticker == "AAA"
    assert row.field == "close"
    assert row.batch_id == "batch-1"
