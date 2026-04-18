from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest

import scripts.run_intraday_smoke as smoke_module
from scripts.run_intraday_smoke import (
    main,
    persist_reconciliation_events,
    validate_trading_days,
    validate_timezones,
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


def test_validate_minute_to_day_consistency_flags_all_nan_as_blocker() -> None:
    minute_df = _minute_rows_for_day()
    minute_df["close"] = float("nan")
    daily_df = _daily_row()

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["pass"] is False
    assert result["fields"]["close"]["pass"] is False
    assert result["fields"]["close"]["nan_pairs"] == 1
    assert result["nan_pair_count"] >= 1


def test_validate_minute_to_day_consistency_counts_partial_nan() -> None:
    trade_days = [date(2026, 1, 5 + offset) for offset in range(10)]
    minute_df = pd.concat(
        [
            _minute_rows_for_day(ticker="AAA", trade_day=trade_day)
            for trade_day in trade_days
        ],
        ignore_index=True,
    )
    minute_df.loc[minute_df["trade_date"] == trade_days[0], "close"] = float("nan")
    daily_df = pd.concat(
        [
            _daily_row(ticker="AAA", trade_day=trade_day)
            for trade_day in trade_days
        ],
        ignore_index=True,
    )

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["pass"] is False
    assert result["fields"]["close"]["nan_pairs"] == 1
    assert result["nan_pair_count"] == 1


def test_validate_minute_to_day_consistency_flags_missing_daily_row() -> None:
    minute_df = _minute_rows_for_day(ticker="AAA", trade_day=date(2026, 1, 5))
    daily_df = pd.DataFrame(columns=["ticker", "trade_date", "open", "high", "low", "close", "volume"])

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["pass"] is False
    assert result["minute_only_count"] == 1
    assert result["daily_only_count"] == 0
    assert result["ticker_day_mismatch"] is True


def test_validate_minute_to_day_consistency_flags_missing_minute_row() -> None:
    minute_df = _minute_rows_for_day(ticker="AAA", trade_day=date(2026, 1, 5))
    daily_df = pd.concat(
        [
            _daily_row(ticker="AAA", trade_day=date(2026, 1, 5)),
            _daily_row(ticker="BBB", trade_day=date(2026, 1, 5)),
        ],
        ignore_index=True,
    )

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["pass"] is False
    assert result["minute_only_count"] == 0
    assert result["daily_only_count"] == 1
    assert result["ticker_day_mismatch"] is True


def test_validate_minute_to_day_consistency_pass_with_full_overlap() -> None:
    minute_df = _minute_rows_for_day()
    daily_df = _daily_row()

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["overlap_count"] == 1
    assert result["minute_only_count"] == 0
    assert result["daily_only_count"] == 0
    assert result["ticker_day_mismatch"] is False
    assert result["pass"] is True


def test_validate_minute_to_day_consistency_no_crash_on_partial_overlap() -> None:
    minute_df = pd.concat(
        [
            _minute_rows_for_day(ticker="AAA", trade_day=date(2026, 1, 5), close_price=101.0),
            _minute_rows_for_day(ticker="BBB", trade_day=date(2026, 1, 5), close_price=102.0),
        ],
        ignore_index=True,
    )
    daily_df = _daily_row(ticker="AAA", trade_day=date(2026, 1, 5), close_price=101.25)

    result = validate_minute_to_day_consistency(minute_df, daily_df)

    assert result["pass"] is False
    assert result["overlap_count"] == 1
    assert result["minute_only_count"] == 1
    assert result["fields"]["close"]["pass"] is False
    assert result["warning_event_count"] == 1


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


def test_validate_minute_internal_consistency_flags_nan_ohlc() -> None:
    minute_df = _minute_rows_for_day()
    minute_df.loc[0, "high"] = float("nan")

    result = validate_minute_internal_consistency(minute_df)

    assert result["pass"] is False
    assert result["checks"]["ohlc_internal_consistency"]["pass"] is False
    assert result["checks"]["ohlc_internal_consistency"]["nan_ohlc_count"] == 1


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


def test_smoke_gap_check_handles_early_close_day() -> None:
    trade_day = date(2025, 11, 28)
    minute_index = pd.date_range(
        "2025-11-28 14:30",
        "2025-11-28 17:59",
        freq="min",
        tz="UTC",
    )
    prices = pd.Series([100.0 + idx * 0.01 for idx in range(len(minute_index))])
    minute_df = pd.DataFrame(
        {
            "ticker": "AAA",
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

    result = validate_minute_internal_consistency(minute_df)

    assert result["pass"] is True
    assert result["checks"]["gap_free"]["pass"] is True


def test_validate_timezones_uses_half_open_interval() -> None:
    trade_day = date(2026, 1, 5)
    minute_df = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": trade_day,
                "minute_ts": pd.Timestamp("2026-01-05 20:59", tz="UTC"),
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 100,
                "vwap": 100.0,
                "transactions": 10,
            },
            {
                "ticker": "AAA",
                "trade_date": trade_day,
                "minute_ts": pd.Timestamp("2026-01-05 21:00", tz="UTC"),
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 100,
                "vwap": 100.0,
                "transactions": 10,
            },
        ],
    )

    result = validate_timezones(minute_df)

    assert result["pass"] is False
    assert result["failure_count"] == 1
    assert result["sample"][0]["minute_ts_et"].endswith("16:00:00-05:00")
    assert result["sample"][0]["time_ok"] is False


def test_smoke_validate_trading_days_catches_per_ticker_dropout() -> None:
    expected_dates = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]
    frames = []
    expected_tickers = [f"T{idx:02d}" for idx in range(10)]
    for ticker in expected_tickers:
        for trade_day in expected_dates:
            if ticker == "T01" and trade_day == date(2026, 1, 7):
                continue
            frames.append(_minute_rows_for_day(ticker=ticker, trade_day=trade_day))
    minute_df = pd.concat(frames, ignore_index=True)

    result = validate_trading_days(
        minute_frame=minute_df,
        start_date=expected_dates[0],
        end_date=expected_dates[-1],
        expected_tickers=expected_tickers,
    )

    assert result["pass"] is False
    assert result["missing_trade_dates"] == []
    assert result["per_ticker_missing"]["T01"] == [date(2026, 1, 7)]


def test_smoke_main_exit_code_reflects_report_pass(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    minute_frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2026, 1, 5),
                "minute_ts": pd.Timestamp("2026-01-05 14:30", tz="UTC"),
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 100,
                "vwap": 100.0,
                "transactions": 10,
                "event_time": pd.Timestamp("2026-01-05 14:30", tz="UTC"),
                "knowledge_time": pd.Timestamp("2026-01-06 21:00", tz="UTC"),
                "batch_id": "batch",
            },
        ],
    )
    feature_frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2026, 1, 5),
                "feature_name": "gap_pct",
                "feature_value": 0.01,
                "is_filled": False,
            },
        ],
    )
    label_frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2026, 1, 5),
                "label_name": "next_open_ret_1d",
                "label_value": 0.01,
            },
        ],
    )

    class FakeMinuteClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_minute_aggs(self, ticker, start_date, end_date):
            return minute_frame.copy()

        def persist_minute_aggs(self, frame):
            return len(frame)

    saved_batches: list[str] = []

    monkeypatch.setattr(smoke_module, "PolygonMinuteClient", FakeMinuteClient)
    monkeypatch.setattr(smoke_module, "load_minute_slice", lambda **kwargs: minute_frame.copy())
    monkeypatch.setattr(
        smoke_module,
        "load_daily_prices",
        lambda **kwargs: pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "trade_date": date(2026, 1, 5),
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "close": 100.0,
                    "volume": 100,
                },
            ],
        ),
    )
    monkeypatch.setattr(smoke_module, "compute_intraday_features", lambda **kwargs: feature_frame.copy())
    monkeypatch.setattr(smoke_module, "prepare_feature_export_frame", lambda frame: frame.copy())
    monkeypatch.setattr(
        smoke_module.FeaturePipeline,
        "save_to_store",
        lambda self, features_df, batch_id: saved_batches.append(batch_id) or len(features_df),
    )
    monkeypatch.setattr(smoke_module, "write_parquet_atomic", lambda frame, path: None)
    monkeypatch.setattr(smoke_module, "build_smoke_labels", lambda frame: label_frame.copy())
    monkeypatch.setattr(smoke_module, "validate_schema", lambda: {"pass": True})
    monkeypatch.setattr(smoke_module, "validate_timezones", lambda frame: {"pass": True})
    monkeypatch.setattr(smoke_module, "validate_trading_days", lambda **kwargs: {"pass": True})
    monkeypatch.setattr(smoke_module, "validate_minute_internal_consistency", lambda frame: {"pass": True})
    monkeypatch.setattr(smoke_module, "validate_corporate_action_alignment", lambda *args, **kwargs: {"pass": True})
    monkeypatch.setattr(
        smoke_module,
        "validate_minute_to_day_consistency",
        lambda *args, **kwargs: {"pass": False, "warning_events": [], "fields": {}},
    )
    monkeypatch.setattr(smoke_module, "persist_reconciliation_events", lambda *args, **kwargs: 0)
    monkeypatch.setattr(smoke_module, "write_json_atomic", lambda path, payload: None)

    exit_code = main(
        [
            "--start-date",
            "2026-01-05",
            "--end-date",
            "2026-01-05",
            "--tickers",
            "AAA",
            "--feature-output",
            str(tmp_path / "features.parquet"),
            "--label-output",
            str(tmp_path / "labels.parquet"),
            "--report-output",
            str(tmp_path / "report.json"),
        ],
    )

    assert exit_code == 1
    assert len(saved_batches) == 1
