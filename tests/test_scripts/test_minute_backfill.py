from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from scripts.run_minute_backfill import ProcessResult, process_trade_day, should_skip_resume
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
