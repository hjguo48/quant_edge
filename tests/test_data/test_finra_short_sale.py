from __future__ import annotations

from datetime import date, datetime, timezone
import sys
from typing import Any
import types

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import sessionmaker

import src.data.finra_short_sale as finra_module
from src.data.finra_short_sale import FINRAShortSaleSource, ShortSaleVolume, _knowledge_time
from src.data.sources.base import DataSourceTransientError, RetryConfig


def test_knowledge_time_uses_lag_one_next_day_close() -> None:
    """FINRA daily short-sale rows must publish on the next business day at the
    16:00 NYT close, matching the ``stock_prices`` historical convention.
    Same-day kt would let today's short volume leak into today's signal because
    ``_as_of_end_utc(trade_date)`` already advances past 18:00 NYT
    (data audit 2026-04-25 P1 follow-up A1).
    """
    kt = _knowledge_time(date(2026, 4, 17))
    # 2026-04-17 + 1d = 2026-04-18 16:00 EDT = 2026-04-18 20:00 UTC
    assert kt == datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    # Friday → Saturday is intentional (wall-clock convention, not business day).
    kt_friday = _knowledge_time(date(2026, 4, 24))
    assert kt_friday == datetime(2026, 4, 25, 20, 0, tzinfo=timezone.utc)


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


def _client(
    fake_session: _FakeSession,
    *,
    now_fn: Any | None = None,
) -> FINRAShortSaleSource:
    return FINRAShortSaleSource(
        min_request_interval=0,
        retry_config=_retry_config(),
        http_session=fake_session,
        now_fn=now_fn,
    )


def _sample_file(rows: int = 50) -> str:
    header = "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market"
    lines = [header]
    for index in range(rows):
        lines.append(
            f"20150615|T{index:02d}|{100 + index}|{10 + index}|{1000 + index}|CNMS",
        )
    return "\n".join(lines)


def _legacy_file() -> str:
    return "\n".join(
        [
            "Date|Symbol|ShortVolume|TotalVolume|Market",
            "20150615|TSTFR|120|1000|CNMS",
        ],
    )


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_table(db_engine) -> None:
    ShortSaleVolume.__table__.create(bind=db_engine, checkfirst=True)


_TEST_TICKER = "TSTFR"  # unique prefix avoids clobbering real FINRA data


def _truncate_table(db_engine) -> None:
    _ensure_table(db_engine)
    with db_engine.begin() as conn:
        # NEVER truncate the full table — production has 13M+ rows.
        # Remove TEST_TICKER + numeric-suffixed test stubs (T00..T49 from _sample_file).
        conn.execute(
            sa.text(
                "DELETE FROM short_sale_volume_daily "
                "WHERE ticker = :t "
                "OR (ticker ~ '^T[0-9]+$' AND trade_date < '2021-01-01')"
            ),
            {"t": _TEST_TICKER},
        )


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

    frame, etag = client.fetch_day(date(2015, 6, 15), "CNMS")
    legacy_frame, legacy_etag = client.fetch_day(date(2015, 6, 15), "CNMS")

    assert len(frame) == 50
    assert etag == "etag-1"
    assert frame["ticker"].iloc[0] == "T00"
    # _knowledge_time pins kt to (trade_date+1) 16:00 NYT to keep the lag-1
    # convention consistent with stock_prices (data audit 2026-04-25 A1).
    assert frame["knowledge_time"].iloc[0] == datetime(2015, 6, 16, 20, 0, tzinfo=timezone.utc)
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
                        "20150615|TSTFR|100|10|1000|CNMS",
                        "20150615|MSFT|101|11|1001",
                    ],
                ),
            ),
        ],
    )
    monkeypatch.setattr(finra_module, "logger", _FakeLogger())

    frame, _ = _client(fake_session).fetch_day(date(2015, 6, 15), "CNMS")

    assert len(frame) == 1
    assert frame["ticker"].tolist() == [_TEST_TICKER]
    assert warnings
    assert "malformed line" in warnings[0][0]


def test_fetch_day_retries_429_then_succeeds() -> None:
    fake_session = _FakeSession(
        get_responses=[
            _FakeResponse(status_code=429, text="rate limited"),
            _FakeResponse(status_code=200, text=_sample_file(rows=1), headers={"ETag": '"etag-retry"'}),
        ],
    )

    frame, etag = _client(fake_session).fetch_day(date(2015, 6, 15), "CNMS")

    assert len(frame) == 1
    assert etag == "etag-retry"
    assert len(fake_session.get_calls) == 2


def test_fetch_day_raises_on_persistent_429() -> None:
    fake_session = _FakeSession(
        get_responses=[_FakeResponse(status_code=429, text="rate limited") for _ in range(_retry_config().max_attempts)],
    )

    with pytest.raises(DataSourceTransientError):
        _client(fake_session).fetch_day(date(2015, 6, 15), "CNMS")

    assert len(fake_session.get_calls) == _retry_config().max_attempts


def test_fetch_historical_skips_when_etag_unchanged_unless_force_refetch(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    with session_factory() as session:
        session.execute(
            insert(ShortSaleVolume).values(
                ticker=_TEST_TICKER,
                trade_date=date(2015, 6, 15),
                knowledge_time=datetime(2015, 6, 15, 22, 0, tzinfo=timezone.utc),
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
        start_date=date(2015, 6, 15),
        end_date=date(2015, 6, 15),
        markets=["CNMS"],
        session_factory=session_factory,
    )
    forced = client.fetch_historical(
        start_date=date(2015, 6, 15),
        end_date=date(2015, 6, 15),
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
                ticker=_TEST_TICKER,
                trade_date=date(2015, 6, 15),
                knowledge_time=datetime(2015, 6, 15, 22, 0, tzinfo=timezone.utc),
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
                        "20150615|TSTFR|250|15|1200|CNMS",
                    ],
                ),
                headers={"ETag": '"etag-new"'},
            ),
        ],
    )

    now_dt = datetime(2015, 6, 16, 12, 0, tzinfo=timezone.utc)
    inserted = _client(fake_session, now_fn=lambda: now_dt).fetch_historical(
        start_date=date(2015, 6, 15),
        end_date=date(2015, 6, 15),
        markets=["CNMS"],
        session_factory=session_factory,
    )

    assert inserted == 1
    with session_factory() as session:
        row = session.execute(
            sa.select(ShortSaleVolume).where(
                ShortSaleVolume.ticker == _TEST_TICKER,
                ShortSaleVolume.trade_date == date(2015, 6, 15),
                ShortSaleVolume.market == "CNMS",
            ),
        ).scalar_one()
    assert row.short_volume == 250
    assert row.total_volume == 1200
    assert row.file_etag == "etag-new"
    assert row.knowledge_time == now_dt


def test_etag_change_advances_knowledge_time(db_engine) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)
    original_kt = datetime(2015, 6, 15, 22, 0, tzinfo=timezone.utc)
    with session_factory() as session:
        session.execute(
            insert(ShortSaleVolume).values(
                ticker=_TEST_TICKER,
                trade_date=date(2015, 6, 15),
                knowledge_time=original_kt,
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
                        "20150615|TSTFR|250|15|1200|CNMS",
                    ],
                ),
                headers={"ETag": '"etag-new"'},
            ),
        ],
    )
    now_dt = datetime(2015, 6, 16, 12, 0, tzinfo=timezone.utc)

    _client(fake_session, now_fn=lambda: now_dt).fetch_historical(
        start_date=date(2015, 6, 15),
        end_date=date(2015, 6, 15),
        markets=["CNMS"],
        session_factory=session_factory,
    )

    with session_factory() as session:
        updated = session.execute(
            sa.select(ShortSaleVolume.knowledge_time).where(
                ShortSaleVolume.ticker == _TEST_TICKER,
                ShortSaleVolume.trade_date == date(2015, 6, 15),
                ShortSaleVolume.market == "CNMS",
            ),
        ).scalar_one()

    assert updated >= now_dt


def test_session_has_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RequestsSession:
        def __init__(self) -> None:
            self.headers: dict[str, str] = {}
            self.trust_env = True

    fake_requests = types.SimpleNamespace(Session=_RequestsSession)
    monkeypatch.setitem(sys.modules, "requests", fake_requests)
    source = FINRAShortSaleSource(min_request_interval=0, retry_config=_retry_config())

    session = source._get_http_session()

    assert "QuantEdge" in session.headers["User-Agent"]


def test_fetch_incremental_returns_persisted_frame(
    db_engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _truncate_table(db_engine)
    session_factory = _session_factory(db_engine)

    class _FixedDate(date):
        @classmethod
        def today(cls) -> date:
            return cls(2015, 6, 15)

    fake_session = _FakeSession(
        head_responses=[
            _FakeResponse(status_code=200, headers={"ETag": '"etag-cnms"'}),
            _FakeResponse(status_code=404, text="not found"),
            _FakeResponse(status_code=404, text="not found"),
        ],
        get_responses=[
            _FakeResponse(status_code=200, text=_sample_file(rows=1), headers={"ETag": '"etag-cnms"'})
        ],
    )
    monkeypatch.setattr(finra_module, "date", _FixedDate)
    monkeypatch.setattr(finra_module, "get_session_factory", lambda: session_factory)

    frame = _client(fake_session).fetch_incremental([_TEST_TICKER], date(2015, 6, 15))

    assert not frame.empty
    assert list(frame.columns) == [column.name for column in ShortSaleVolume.__table__.columns]
    assert frame["file_etag"].iloc[0] == "etag-cnms"
