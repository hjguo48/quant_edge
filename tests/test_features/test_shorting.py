from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from src.data.finra_short_sale import ShortSaleVolume
from src.features.shorting import (
    compute_abnormal_off_exchange_shorting,
    compute_short_sale_accel,
    compute_short_sale_ratio_1d,
    compute_short_sale_ratio_5d,
)

EASTERN = ZoneInfo("America/New_York")
XNYS = xcals.get_calendar("XNYS")


def _session_dates(start: str, end: str) -> list[date]:
    return [session.date() for session in XNYS.sessions_in_range(start, end)]


def _kt(day: date, hour: int = 18, minute: int = 0, second: int = 0) -> datetime:
    return datetime.combine(day, time(hour=hour, minute=minute, second=second), tzinfo=EASTERN).astimezone(timezone.utc)


def _seed_rows(session_factory: sessionmaker, rows: Iterable[dict]) -> None:
    with session_factory() as session:
        session.execute(sa.delete(ShortSaleVolume))
        session.add_all([ShortSaleVolume(**row) for row in rows])
        session.commit()


def test_compute_short_sale_ratio_1d_aggregates_across_markets(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    trade_day = date(2026, 4, 23)
    _seed_rows(
        sf,
        [
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "CNMS",
                "short_volume": 10,
                "short_exempt_volume": 1,
                "total_volume": 100,
                "file_etag": "a",
            },
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "ADF",
                "short_volume": 20,
                "short_exempt_volume": 2,
                "total_volume": 200,
                "file_etag": "b",
            },
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "BNY",
                "short_volume": 30,
                "short_exempt_volume": 3,
                "total_volume": 300,
                "file_etag": "c",
            },
        ],
    )

    result = compute_short_sale_ratio_1d("AAPL", trade_day, session_factory=sf)

    assert result == pytest.approx(0.1)


def test_compute_short_sale_ratio_1d_returns_none_when_missing(short_sale_session_factory) -> None:
    sf = short_sale_session_factory

    result = compute_short_sale_ratio_1d("AAPL", date(2026, 4, 23), session_factory=sf)

    assert result is None


def test_compute_short_sale_ratio_1d_filters_future_knowledge_time(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    trade_day = date(2026, 4, 23)
    _seed_rows(
        sf,
        [
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day, 23, 59, 59) + timedelta(seconds=2),
                "market": "CNMS",
                "short_volume": 10,
                "short_exempt_volume": 1,
                "total_volume": 100,
                "file_etag": "future",
            },
        ],
    )

    result = compute_short_sale_ratio_1d("AAPL", trade_day, session_factory=sf)

    assert result is None


def test_compute_short_sale_ratio_5d_averages_available_days(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    sessions = _session_dates("2026-04-16", "2026-04-23")
    ratios = [0.1, 0.2, 0.3]
    rows = []
    for trade_day, ratio in zip(sessions[-3:], ratios, strict=False):
        rows.append(
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "CNMS",
                "short_volume": int(ratio * 1000),
                "short_exempt_volume": 0,
                "total_volume": 1000,
                "file_etag": f"etag-{trade_day}",
            },
        )
    _seed_rows(sf, rows)

    result = compute_short_sale_ratio_5d("AAPL", sessions[-1], session_factory=sf)

    assert result == pytest.approx(sum(ratios) / len(ratios))


def test_compute_short_sale_ratio_5d_returns_none_with_less_than_three_days(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    sessions = _session_dates("2026-04-16", "2026-04-23")
    rows = []
    for trade_day, ratio in zip(sessions[-2:], [0.1, 0.2], strict=False):
        rows.append(
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "CNMS",
                "short_volume": int(ratio * 1000),
                "short_exempt_volume": 0,
                "total_volume": 1000,
                "file_etag": f"etag-{trade_day}",
            },
        )
    _seed_rows(sf, rows)

    result = compute_short_sale_ratio_5d("AAPL", sessions[-1], session_factory=sf)

    assert result is None


def test_compute_short_sale_accel_returns_ma5_minus_ma20(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    sessions = _session_dates("2026-03-25", "2026-04-23")[-20:]
    rows = []
    for trade_day in sessions[:-5]:
        rows.append(
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "CNMS",
                "short_volume": 100,
                "short_exempt_volume": 0,
                "total_volume": 1000,
                "file_etag": f"etag-{trade_day}",
            },
        )
    for trade_day in sessions[-5:]:
        rows.append(
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "CNMS",
                "short_volume": 300,
                "short_exempt_volume": 0,
                "total_volume": 1000,
                "file_etag": f"etag-{trade_day}",
            },
        )
    _seed_rows(sf, rows)

    result = compute_short_sale_accel("AAPL", sessions[-1], session_factory=sf)

    assert result == pytest.approx(0.15)


def test_compute_short_sale_accel_returns_none_when_20d_history_insufficient(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    sessions = _session_dates("2026-04-01", "2026-04-23")[-14:]
    rows = [
        {
            "ticker": "AAPL",
            "trade_date": trade_day,
            "knowledge_time": _kt(trade_day),
            "market": "CNMS",
            "short_volume": 100,
            "short_exempt_volume": 0,
            "total_volume": 1000,
            "file_etag": f"etag-{trade_day}",
        }
        for trade_day in sessions
    ]
    _seed_rows(sf, rows)

    result = compute_short_sale_accel("AAPL", sessions[-1], session_factory=sf)

    assert result is None


def test_compute_abnormal_off_exchange_shorting_returns_adf_zscore(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    sessions = _session_dates("2025-12-01", "2026-04-23")[-91:]
    rows = []
    for index, trade_day in enumerate(sessions[:-1]):
        ratio = 0.1 if index % 2 == 0 else 0.2
        rows.append(
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "ADF",
                "short_volume": int(ratio * 1000),
                "short_exempt_volume": 0,
                "total_volume": 1000,
                "file_etag": f"etag-{trade_day}",
            },
        )
    rows.append(
        {
            "ticker": "AAPL",
            "trade_date": sessions[-1],
            "knowledge_time": _kt(sessions[-1]),
            "market": "ADF",
            "short_volume": 300,
            "short_exempt_volume": 0,
            "total_volume": 1000,
            "file_etag": "etag-current",
        },
    )
    _seed_rows(sf, rows)

    result = compute_abnormal_off_exchange_shorting("AAPL", sessions[-1], session_factory=sf)

    assert result == pytest.approx(3.0, rel=1e-6)


def test_compute_abnormal_off_exchange_shorting_returns_none_when_history_insufficient(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    sessions = _session_dates("2026-01-01", "2026-04-23")[-60:]
    rows = []
    for trade_day in sessions[:-1]:
        rows.append(
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "ADF",
                "short_volume": 100,
                "short_exempt_volume": 0,
                "total_volume": 1000,
                "file_etag": f"etag-{trade_day}",
            },
        )
    rows.append(
        {
            "ticker": "AAPL",
            "trade_date": sessions[-1],
            "knowledge_time": _kt(sessions[-1]),
            "market": "ADF",
            "short_volume": 300,
            "short_exempt_volume": 0,
            "total_volume": 1000,
            "file_etag": "etag-current",
        },
    )
    _seed_rows(sf, rows)

    result = compute_abnormal_off_exchange_shorting("AAPL", sessions[-1], session_factory=sf)

    assert result is None


def test_compute_abnormal_off_exchange_shorting_filters_future_knowledge_time(short_sale_session_factory) -> None:
    sf = short_sale_session_factory
    sessions = _session_dates("2025-12-01", "2026-04-23")[-91:]
    rows = []
    for index, trade_day in enumerate(sessions[:-1]):
        ratio = 0.1 if index % 2 == 0 else 0.2
        rows.append(
            {
                "ticker": "AAPL",
                "trade_date": trade_day,
                "knowledge_time": _kt(trade_day),
                "market": "ADF",
                "short_volume": int(ratio * 1000),
                "short_exempt_volume": 0,
                "total_volume": 1000,
                "file_etag": f"etag-{trade_day}",
            },
        )
    rows.append(
        {
            "ticker": "AAPL",
            "trade_date": sessions[-1],
            "knowledge_time": _kt(sessions[-1], 23, 59, 59).replace(microsecond=0) + timedelta(seconds=2),
            "market": "ADF",
            "short_volume": 300,
            "short_exempt_volume": 0,
            "total_volume": 1000,
            "file_etag": "etag-current",
        },
    )
    _seed_rows(sf, rows)

    result = compute_abnormal_off_exchange_shorting("AAPL", sessions[-1], session_factory=sf)

    assert result is None
