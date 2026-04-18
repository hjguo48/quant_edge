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
    monkeypatch.setattr(pipeline_module, "get_session_factory", lambda: (lambda: _EmptySession()))

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
    monkeypatch.setattr(pipeline_module, "get_session_factory", lambda: (lambda: _EmptySession()))

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
