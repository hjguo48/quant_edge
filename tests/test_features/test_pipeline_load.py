from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import src.features.pipeline as pipeline_module
from src.features.pipeline import FeaturePipeline, IntradayHistoryError, load_intraday_minute_history


class _BrokenSession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement):
        raise RuntimeError("db down")


class _EmptyResult:
    def mappings(self):
        return self

    def all(self):
        return []


class _EmptySession:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement):
        return _EmptyResult()


class _RowsResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return list(self._rows)


class _RowsSession:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement):
        return _RowsResult(self._rows)


class _SequentialSession:
    def __init__(self, responses):
        self._responses = list(responses)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, statement):
        if not self._responses:
            raise AssertionError("unexpected execute call with no queued response")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _RowsResult(response)


def _patch_sessions(monkeypatch: pytest.MonkeyPatch, responses) -> None:
    monkeypatch.setattr(pipeline_module, "get_session_factory", lambda: (lambda: _SequentialSession(responses)))


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"])


def test_load_intraday_minute_history_raises_on_db_error_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "get_session_factory", lambda: (lambda: _BrokenSession()))

    with pytest.raises(IntradayHistoryError, match="minute history unavailable"):
        load_intraday_minute_history(
            tickers=["AAPL"],
            start_trade_date=date(2026, 1, 1),
            end_trade_date=date(2026, 1, 5),
            as_of=date(2026, 1, 6),
        )


def test_load_intraday_minute_history_returns_empty_with_allow_missing_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline_module, "get_session_factory", lambda: (lambda: _BrokenSession()))

    frame = load_intraday_minute_history(
        tickers=["AAPL"],
        start_trade_date=date(2026, 1, 1),
        end_trade_date=date(2026, 1, 5),
        as_of=date(2026, 1, 6),
        allow_missing=True,
    )

    assert frame.empty
    assert list(frame.columns) == [
        "ticker",
        "trade_date",
        "minute_ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]


def test_load_intraday_minute_history_raises_on_empty_rows_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_sessions(
        monkeypatch,
        [
            [{"trading_date": date(2026, 1, 1), "status": "completed"}, {"trading_date": date(2026, 1, 2), "status": "completed"}, {"trading_date": date(2026, 1, 5), "status": "completed"}],
            [],
        ],
    )

    with pytest.raises(IntradayHistoryError, match="minute history empty"):
        load_intraday_minute_history(
            tickers=["AAPL"],
            start_trade_date=date(2026, 1, 1),
            end_trade_date=date(2026, 1, 5),
            as_of=date(2026, 1, 6),
        )


def test_load_intraday_minute_history_returns_empty_with_allow_missing_on_empty_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sessions(
        monkeypatch,
        [
            [{"trading_date": date(2026, 1, 1), "status": "completed"}, {"trading_date": date(2026, 1, 2), "status": "completed"}, {"trading_date": date(2026, 1, 5), "status": "completed"}],
            [],
        ],
    )

    frame = load_intraday_minute_history(
        tickers=["AAPL"],
        start_trade_date=date(2026, 1, 1),
        end_trade_date=date(2026, 1, 5),
        as_of=date(2026, 1, 6),
        allow_missing=True,
    )

    assert frame.empty
    assert list(frame.columns) == [
        "ticker",
        "trade_date",
        "minute_ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "vwap",
        "transactions",
    ]


def test_feature_pipeline_run_fails_closed_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    prices = pd.DataFrame(
        [
            {"ticker": "AAPL", "trade_date": date(2026, 1, 5), "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000},
            {"ticker": "AAPL", "trade_date": date(2026, 1, 6), "open": 101.0, "high": 102.0, "low": 100.0, "close": 101.5, "volume": 1100},
            {"ticker": "SPY", "trade_date": date(2026, 1, 5), "open": 400.0, "high": 401.0, "low": 399.0, "close": 400.5, "volume": 10_000},
            {"ticker": "SPY", "trade_date": date(2026, 1, 6), "open": 401.0, "high": 402.0, "low": 400.0, "close": 401.5, "volume": 10_500},
        ],
    )

    monkeypatch.setattr(pipeline_module, "get_prices_pit", lambda *args, **kwargs: prices.copy())
    monkeypatch.setattr(pipeline_module, "compute_technical_features", lambda *args, **kwargs: _empty_feature_frame())
    monkeypatch.setattr(pipeline_module, "compute_fundamental_features", lambda *args, **kwargs: _empty_feature_frame())
    monkeypatch.setattr(pipeline_module, "compute_alternative_features_batch", lambda *args, **kwargs: _empty_feature_frame())
    monkeypatch.setattr(pipeline_module, "compute_sector_rotation_features", lambda *args, **kwargs: _empty_feature_frame())
    monkeypatch.setattr(pipeline_module, "compute_composite_features", lambda *args, **kwargs: _empty_feature_frame())
    monkeypatch.setattr(pipeline_module, "preprocess_features", lambda frame: frame)
    monkeypatch.setattr(
        FeaturePipeline,
        "_compute_broadcast_macro_features",
        lambda self, *args, **kwargs: _empty_feature_frame(),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_intraday_minute_history",
        lambda *args, **kwargs: (_ for _ in ()).throw(IntradayHistoryError("db down")),
    )

    with pytest.raises(IntradayHistoryError, match="db down"):
        FeaturePipeline().run(
            tickers=["AAPL"],
            start_date=date(2026, 1, 5),
            end_date=date(2026, 1, 6),
            as_of=date(2026, 1, 7),
        )


def test_load_intraday_minute_history_raises_on_partial_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "ticker": "AAPL",
            "trade_date": date(2026, 1, 5),
            "minute_ts": pd.Timestamp("2026-01-05 14:30:00+00:00"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "vwap": 100.2,
            "transactions": 10,
        },
    ]
    _patch_sessions(
        monkeypatch,
        [
            [{"trading_date": date(2026, 1, 1), "status": "completed"}, {"trading_date": date(2026, 1, 2), "status": "completed"}, {"trading_date": date(2026, 1, 5), "status": "completed"}],
            rows,
        ],
    )

    with pytest.raises(IntradayHistoryError, match="minute history partial"):
        load_intraday_minute_history(
            tickers=["AAPL", "MSFT"],
            start_trade_date=date(2026, 1, 1),
            end_trade_date=date(2026, 1, 5),
            as_of=date(2026, 1, 6),
        )


def test_load_intraday_minute_history_logs_partial_coverage_with_allow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "ticker": "AAPL",
            "trade_date": date(2026, 1, 5),
            "minute_ts": pd.Timestamp("2026-01-05 14:30:00+00:00"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "vwap": 100.2,
            "transactions": 10,
        },
    ]
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []

    _patch_sessions(
        monkeypatch,
        [
            [{"trading_date": date(2026, 1, 1), "status": "completed"}, {"trading_date": date(2026, 1, 2), "status": "completed"}, {"trading_date": date(2026, 1, 5), "status": "completed"}],
            rows,
        ],
    )
    monkeypatch.setattr(pipeline_module.logger, "error", lambda *args, **kwargs: logged.append((args, kwargs)))

    frame = load_intraday_minute_history(
        tickers=["AAPL", "MSFT"],
        start_trade_date=date(2026, 1, 1),
        end_trade_date=date(2026, 1, 5),
        as_of=date(2026, 1, 6),
        allow_missing=True,
    )

    assert not frame.empty
    assert any("partial coverage" in str(args[0]) for args, _ in logged)


def test_load_intraday_minute_history_raises_on_incomplete_backfill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_module.XNYS,
        "sessions_in_range",
        lambda start, end: pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"]),
    )
    _patch_sessions(
        monkeypatch,
        [[
            {"trading_date": date(2026, 1, 5), "status": "completed"},
            {"trading_date": date(2026, 1, 6), "status": "failed"},
            {"trading_date": date(2026, 1, 7), "status": "completed"},
        ]],
    )

    with pytest.raises(IntradayHistoryError, match="incomplete"):
        load_intraday_minute_history(
            tickers=["AAPL"],
            start_trade_date=date(2026, 1, 5),
            end_trade_date=date(2026, 1, 7),
            as_of=date(2026, 1, 8),
        )


def test_load_intraday_minute_history_accepts_skipped_holiday(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "ticker": "AAPL",
            "trade_date": date(2026, 1, 5),
            "minute_ts": pd.Timestamp("2026-01-05 14:30:00+00:00"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "vwap": 100.2,
            "transactions": 10,
        },
    ]
    monkeypatch.setattr(
        pipeline_module.XNYS,
        "sessions_in_range",
        lambda start, end: pd.DatetimeIndex(["2026-01-05", "2026-01-06"]),
    )
    _patch_sessions(
        monkeypatch,
        [[
            {"trading_date": date(2026, 1, 5), "status": "completed"},
            {"trading_date": date(2026, 1, 6), "status": "skipped_holiday"},
        ], rows],
    )

    frame = load_intraday_minute_history(
        tickers=["AAPL"],
        start_trade_date=date(2026, 1, 5),
        end_trade_date=date(2026, 1, 6),
        as_of=date(2026, 1, 7),
    )

    assert not frame.empty


def test_load_intraday_minute_history_raises_on_missing_state_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_module.XNYS,
        "sessions_in_range",
        lambda start, end: pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"]),
    )
    _patch_sessions(
        monkeypatch,
        [[{"trading_date": date(2026, 1, 5), "status": "completed"}]],
    )

    with pytest.raises(IntradayHistoryError, match="incomplete"):
        load_intraday_minute_history(
            tickers=["AAPL"],
            start_trade_date=date(2026, 1, 5),
            end_trade_date=date(2026, 1, 7),
            as_of=date(2026, 1, 8),
        )


def test_load_intraday_minute_history_logs_incomplete_with_allow_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        {
            "ticker": "AAPL",
            "trade_date": date(2026, 1, 5),
            "minute_ts": pd.Timestamp("2026-01-05 14:30:00+00:00"),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000,
            "vwap": 100.2,
            "transactions": 10,
        },
    ]
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        pipeline_module.XNYS,
        "sessions_in_range",
        lambda start, end: pd.DatetimeIndex(["2026-01-05", "2026-01-06", "2026-01-07"]),
    )
    _patch_sessions(
        monkeypatch,
        [[{"trading_date": date(2026, 1, 5), "status": "completed"}], rows],
    )
    monkeypatch.setattr(pipeline_module.logger, "error", lambda *args, **kwargs: logged.append((args, kwargs)))

    frame = load_intraday_minute_history(
        tickers=["AAPL"],
        start_trade_date=date(2026, 1, 5),
        end_trade_date=date(2026, 1, 7),
        as_of=date(2026, 1, 8),
        allow_missing=True,
    )

    assert not frame.empty
    assert any("backfill incomplete" in str(args[0]) for args, _ in logged)
