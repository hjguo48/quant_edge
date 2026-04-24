from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

import src.data.finra_short_sale as finra_module
from src.data.finra_short_sale import FINRAShortSaleSource, ShortSaleVolume
from src.data.sources.base import RetryConfig


class _FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class _FakeSession:
    def __init__(
        self,
        *,
        head_responses: list[_FakeResponse] | None = None,
        get_responses: list[_FakeResponse] | None = None,
    ) -> None:
        self.head_responses = list(head_responses or [])
        self.get_responses = list(get_responses or [])
        self.head_calls: list[tuple[str, int]] = []
        self.get_calls: list[tuple[str, int]] = []

    def head(self, url: str, timeout: int = 30) -> _FakeResponse:
        self.head_calls.append((url, timeout))
        if not self.head_responses:
            raise AssertionError("Unexpected HEAD call")
        return self.head_responses.pop(0)

    def get(self, url: str, timeout: int = 30) -> _FakeResponse:
        self.get_calls.append((url, timeout))
        if not self.get_responses:
            raise AssertionError("Unexpected GET call")
        return self.get_responses.pop(0)


def _retry_config() -> RetryConfig:
    return RetryConfig(
        max_attempts=5,
        initial_delay=0,
        backoff_factor=1,
        max_delay=0,
        jitter=0,
    )


def _client(fake_session: _FakeSession) -> FINRAShortSaleSource:
    return FINRAShortSaleSource(
        min_request_interval=0,
        retry_config=_retry_config(),
        http_session=fake_session,
    )


def _sample_file(rows: int = 50) -> str:
    header = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
    lines = [header]
    for index in range(rows):
        lines.append(
            f"20260423|T{index:02d}|{100 + index}|{10 + index}|{1000 + index}|CNMS",
        )
    return "\n".join(lines)


def _legacy_file() -> str:
    return "\n".join(
        [
            "Date|Symbol|ShortVolume|TotalVolume|Market",
            "20260423|AAPL|120|1000|CNMS",
        ],
    )


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_table(db_engine) -> None:
    ShortSaleVolume.__table__.create(bind=db_engine, checkfirst=True)


def _truncate_table(db_engine) -> None:
    _ensure_table(db_engine)
    with db_engine.begin() as conn:
        conn.execute(sa.text("truncate table short_sale_volume_daily"))


def test_fetch_day_parses_pipe_delimited_file_and_legacy_short_exempt_fallback() -> None:
    fake_session = _FakeSession(
        get_responses=[
            _FakeResponse(
                status_code=200,
                text=_sample_file(),
                headers={"ETag": '"etag-1"'},
            ),
            _FakeResponse(
                status_code=200,
                text=_legacy_file(),
                headers={"ETag": '"etag-legacy"'},
            ),
        ],
    )
    client = _client(fake_session)

    frame, etag = client.fetch_day(date(2026, 4, 23), "CNMS")
    legacy_frame, legacy_etag = client.fetch_day(date(2026, 4, 23), "CNMS")

    assert len(frame) == 50
    assert etag == "etag-1"
    assert frame["ticker"].iloc[0] == "T00"
    assert frame["knowledge_time"].iloc[0] == datetime(2026, 4, 23, 22, 0, tzinfo=timezone.utc)
    assert frame["short_exempt_volume"].iloc[0] == 10
    assert legacy_etag == "etag-legacy"
    assert len(legacy_frame) == 1
    assert pd.isna(legacy_frame["short_exempt_volume"].iloc[0])


def test_fetch_day_returns_empty_frame_on_404() -> None:
    fake_session = _FakeSession(get_responses=[_FakeResponse(status_code=404, text="not found")])

    frame, etag = _client(fake_session).fetch_day(date(2026, 4, 26), "CNMS")

    assert frame.empty
    assert list(frame.columns) == finra_module.FINRA_COLUMNS
    assert etag is None


def test_fetch_day_skips_malformed_lines_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    warnings: list[tuple[str, tuple[Any, ...]]] = []

    class _FakeLogger:
        def warning(self, message: str, *args: Any) -> None:
            warnings.append((message, args))

    fake_session = _FakeSession(
        get_responses=[
            _FakeResponse(
                status_code=200,
                text="\n".join(
                    [
                        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
                        "20260423|AAPL|100|10|1000|CNMS",
                        "20260423|MSFT|101|11|1001",
                    ],
                ),
            ),
        ],
    )
    monkeypatch.setattr(finra_module, "logger", _FakeLogger())

    frame, _ = _client(fake_session).fetch_day(date(2026, 4, 23), "CNMS")

    assert len(frame) == 1
    assert frame["ticker"].tolist() == ["AAPL"]
    assert warnings
    assert "malformed line" in warnings[0][0]


def test_fetch_day_retries_429_then_succeeds() -> None:
    fake_session = _FakeSession(
        get_responses=[
            _FakeResponse(status_code=429, text="rate limited"),
            _FakeResponse(status_code=200, text=_sample_file(rows=1), headers={"ETag": '"etag-retry"'}),
        ],
    )

    frame, etag = _client(fake_session).fetch_day(date(2026, 4, 23), "CNMS")

    assert len(frame) == 1
    assert etag == "etag-retry"
    assert len(fake_session.get_calls) == 2


def test_fetch_historical_skips_when_etag_unchanged_unless_force_refetch(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    with session_factory() as session:
        session.execute(
            insert(ShortSaleVolume).values(
                ticker="AAPL",
                trade_date=date(2026, 4, 23),
                knowledge_time=datetime(2026, 4, 23, 22, 0, tzinfo=timezone.utc),
                market="CNMS",
                short_volume=100,
                short_exempt_volume=10,
                total_volume=1000,
                file_etag="etag-same",
            ),
        )
        session.commit()

    fake_session = _FakeSession(
        head_responses=[_FakeResponse(status_code=200, headers={"ETag": '"etag-same"'})],
        get_responses=[
            _FakeResponse(status_code=200, text=_sample_file(rows=1), headers={"ETag": '"etag-same"'})
        ],
    )
    client = _client(fake_session)

    skipped = client.fetch_historical(
        start_date=date(2026, 4, 23),
        end_date=date(2026, 4, 23),
        markets=["CNMS"],
        session_factory=session_factory,
    )
    forced = client.fetch_historical(
        start_date=date(2026, 4, 23),
        end_date=date(2026, 4, 23),
        markets=["CNMS"],
        session_factory=session_factory,
        force_refetch=True,
    )

    assert skipped == 0
    assert forced == 1
    assert len(fake_session.head_calls) == 1
    assert len(fake_session.get_calls) == 1


def test_fetch_historical_etag_changed_reparses_and_upserts(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    with session_factory() as session:
        session.execute(
            insert(ShortSaleVolume).values(
                ticker="AAPL",
                trade_date=date(2026, 4, 23),
                knowledge_time=datetime(2026, 4, 23, 22, 0, tzinfo=timezone.utc),
                market="CNMS",
                short_volume=100,
                short_exempt_volume=10,
                total_volume=1000,
                file_etag="etag-old",
            ),
        )
        session.commit()

    fake_session = _FakeSession(
        head_responses=[_FakeResponse(status_code=200, headers={"ETag": '"etag-new"'})],
        get_responses=[
            _FakeResponse(
                status_code=200,
                text="\n".join(
                    [
                        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market",
                        "20260423|AAPL|250|15|1200|CNMS",
                    ],
                ),
                headers={"ETag": '"etag-new"'},
            ),
        ],
    )

    inserted = _client(fake_session).fetch_historical(
        start_date=date(2026, 4, 23),
        end_date=date(2026, 4, 23),
        markets=["CNMS"],
        session_factory=session_factory,
    )

    assert inserted == 1
    with session_factory() as session:
        row = session.execute(
            sa.select(ShortSaleVolume).where(
                ShortSaleVolume.ticker == "AAPL",
                ShortSaleVolume.trade_date == date(2026, 4, 23),
                ShortSaleVolume.market == "CNMS",
            ),
        ).scalar_one()
    assert row.short_volume == 250
    assert row.total_volume == 1200
    assert row.file_etag == "etag-new"
