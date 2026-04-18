from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import scripts.run_minute_backfill as minute_backfill_module
from scripts.run_minute_backfill import (
    ProcessResult,
    load_universe_whitelist_for_date,
    process_trade_day,
    run_backfill,
    should_skip_resume,
)
from src.data.polygon_flat_files import FlatFileLoadResult


class _FakeFlatClient:
    def __init__(self, result: FlatFileLoadResult) -> None:
        self.result = result
        self.calls: list[tuple[str, date]] = []

    def sample_day(self, trading_date: date, *, universe_tickers, sample_rows: int = 1_000) -> FlatFileLoadResult:
        self.calls.append(("sample", trading_date))
        return self.result

    def load_day(self, trading_date: date, *, universe_tickers) -> FlatFileLoadResult:
        self.calls.append(("load", trading_date))
        return self.result


class _IdempotentWriter:
    def __init__(self) -> None:
        self.keys: set[tuple[str, object]] = set()

    def __call__(self, frame: pd.DataFrame) -> int:
        for row in frame.itertuples(index=False):
            self.keys.add((row.ticker, row.minute_ts))
        return len(self.keys)


def _result() -> FlatFileLoadResult:
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "trade_date": date(2026, 1, 5),
                "minute_ts": pd.Timestamp("2026-01-05 09:30", tz="America/New_York"),
                "open": 100.0,
                "high": 101.0,
                "low": 99.5,
                "close": 100.5,
                "volume": 1000,
                "vwap": 100.2,
                "transactions": 10,
                "event_time": pd.Timestamp("2026-01-05 09:30", tz="America/New_York"),
                "knowledge_time": pd.Timestamp("2026-01-06 16:00", tz="America/New_York"),
                "batch_id": "",
            },
        ],
    )
    return FlatFileLoadResult(
        source_file="s3://flatfiles/us_stocks_sip/minute_aggs_v1/2026/01/2026-01-05.csv.gz",
        checksum_md5="abc",
        rows_raw=2,
        rows_kept=1,
        tickers_loaded=1,
        frame=frame,
    )


def test_process_trade_day_uses_universe_filtered_flat_file_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.run_minute_backfill.is_trading_day", lambda _: True)
    result = _result()
    client = _FakeFlatClient(result)
    writer = _IdempotentWriter()

    processed = process_trade_day(
        trading_date=date(2026, 1, 5),
        client=client,
        universe_tickers=["AAPL"],
        dry_run=False,
        writer=writer,
    )

    assert processed == ProcessResult(
        trading_date=date(2026, 1, 5),
        status="completed",
        source_file=result.source_file,
        rows_raw=2,
        rows_kept=1,
        tickers_loaded=1,
        checksum="abc",
        error_message=None,
    )
    assert client.calls == [("load", date(2026, 1, 5))]
    assert len(writer.keys) == 1


def test_process_trade_day_dry_run_does_not_call_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.run_minute_backfill.is_trading_day", lambda _: True)
    result = _result()
    client = _FakeFlatClient(result)
    writer = _IdempotentWriter()

    processed = process_trade_day(
        trading_date=date(2026, 1, 5),
        client=client,
        universe_tickers=["AAPL"],
        dry_run=True,
        writer=writer,
    )

    assert processed.status == "completed"
    assert client.calls == [("sample", date(2026, 1, 5))]
    assert len(writer.keys) == 0


def test_should_skip_resume_for_completed_or_holiday() -> None:
    state = {
        date(2026, 1, 5): {"status": "completed"},
        date(2026, 1, 6): {"status": "skipped_holiday"},
        date(2026, 1, 7): {"status": "failed"},
    }

    assert should_skip_resume(state, date(2026, 1, 5)) is True
    assert should_skip_resume(state, date(2026, 1, 6)) is True
    assert should_skip_resume(state, date(2026, 1, 7)) is False


def test_writer_idempotence_same_day_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.run_minute_backfill.is_trading_day", lambda _: True)
    result = _result()
    client = _FakeFlatClient(result)
    writer = _IdempotentWriter()

    first = process_trade_day(
        trading_date=date(2026, 1, 5),
        client=client,
        universe_tickers=["AAPL"],
        dry_run=False,
        writer=writer,
    )
    second = process_trade_day(
        trading_date=date(2026, 1, 5),
        client=client,
        universe_tickers=["AAPL"],
        dry_run=False,
        writer=writer,
    )

    assert first.rows_kept == second.rows_kept == 1
    assert len(writer.keys) == 1


def test_load_universe_whitelist_for_date_filters_pit_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResult:
        def __init__(self, values):
            self._values = values

        def scalars(self):
            return self

        def all(self):
            return self._values

    class FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, statement, params=None):
            trade_day = params["trading_date"]
            if trade_day == date(2025, 3, 15):
                return FakeResult(["AAPL", "MSFT"])
            if trade_day == date(2025, 7, 15):
                return FakeResult(["MSFT"])
            return FakeResult([])

    class FakeEngine:
        def connect(self):
            return FakeConn()

    monkeypatch.setattr(minute_backfill_module, "get_engine", lambda: FakeEngine())

    assert load_universe_whitelist_for_date(date(2025, 3, 15)) == ["AAPL", "MSFT"]
    assert load_universe_whitelist_for_date(date(2025, 7, 15)) == ["MSFT"]
    assert load_universe_whitelist_for_date(date(2015, 12, 1)) == []


def test_run_backfill_calls_pit_universe_per_day(monkeypatch: pytest.MonkeyPatch) -> None:
    seen_universes: list[tuple[date, list[str] | None]] = []

    monkeypatch.setattr(
        minute_backfill_module,
        "iter_calendar_days",
        lambda start_date, end_date: [date(2026, 1, 5), date(2026, 1, 6)],
    )
    monkeypatch.setattr(minute_backfill_module, "is_trading_day", lambda _: True)
    monkeypatch.setattr(
        minute_backfill_module,
        "load_universe_whitelist_for_date",
        lambda trading_date: ["AAPL"] if trading_date == date(2026, 1, 5) else ["MSFT", "NVDA"],
    )
    monkeypatch.setattr(minute_backfill_module, "PolygonFlatFilesClient", lambda min_request_interval=0.0: object())
    monkeypatch.setattr(
        minute_backfill_module,
        "process_trade_day",
        lambda **kwargs: (
            seen_universes.append((kwargs["trading_date"], kwargs["universe_tickers"])),
            ProcessResult(trading_date=kwargs["trading_date"], status="completed"),
        )[1],
    )

    summary = run_backfill(
        minute_backfill_module.argparse.Namespace(
            start_date="2026-01-05",
            end_date="2026-01-06",
            resume=False,
            dry_run=True,
            universe_from_membership=True,
            report_output="ignored.json",
        ),
    )

    assert summary["summary"]["completed_days"] == 2
    assert seen_universes == [
        (date(2026, 1, 5), ["AAPL"]),
        (date(2026, 1, 6), ["MSFT", "NVDA"]),
    ]


def test_run_backfill_fails_fast_when_session_day_has_empty_pit_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        minute_backfill_module,
        "iter_calendar_days",
        lambda start_date, end_date: [date(2026, 1, 5)],
    )
    monkeypatch.setattr(minute_backfill_module, "is_trading_day", lambda _: True)
    monkeypatch.setattr(minute_backfill_module, "load_universe_whitelist_for_date", lambda _: [])

    with pytest.raises(RuntimeError, match="empty whitelist"):
        run_backfill(
            minute_backfill_module.argparse.Namespace(
                start_date="2026-01-05",
                end_date="2026-01-05",
                resume=False,
                dry_run=True,
                universe_from_membership=True,
                report_output="ignored.json",
            ),
        )


def test_run_backfill_holiday_skips_empty_pit_whitelist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        minute_backfill_module,
        "iter_calendar_days",
        lambda start_date, end_date: [date(2026, 1, 4)],
    )
    monkeypatch.setattr(minute_backfill_module, "is_trading_day", lambda _: False)
    monkeypatch.setattr(
        minute_backfill_module,
        "load_universe_whitelist_for_date",
        lambda _: (_ for _ in ()).throw(AssertionError("holiday should not load PIT whitelist")),
    )
    monkeypatch.setattr(minute_backfill_module, "PolygonFlatFilesClient", lambda min_request_interval=0.0: object())
    monkeypatch.setattr(
        minute_backfill_module,
        "process_trade_day",
        lambda **kwargs: ProcessResult(trading_date=kwargs["trading_date"], status="skipped_holiday"),
    )

    summary = run_backfill(
        minute_backfill_module.argparse.Namespace(
            start_date="2026-01-04",
            end_date="2026-01-04",
            resume=False,
            dry_run=True,
            universe_from_membership=True,
            report_output="ignored.json",
        ),
    )

    assert summary["summary"]["skipped_holidays"] == 1


def test_run_backfill_not_fails_on_unset_membership_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        minute_backfill_module,
        "load_universe_whitelist_for_date",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("membership whitelist should not be loaded")),
    )
    monkeypatch.setattr(minute_backfill_module, "iter_calendar_days", lambda start_date, end_date: [])

    summary = run_backfill(
        minute_backfill_module.argparse.Namespace(
            start_date="2026-01-05",
            end_date="2026-01-05",
            resume=False,
            dry_run=True,
            universe_from_membership=False,
            report_output="ignored.json",
        ),
    )

    assert summary["summary"]["failed_days"] == 0
    assert summary["metadata"]["universe_size"] is None
