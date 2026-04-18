from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

import src.features.pipeline as pipeline_module
from src.features.intraday import (
    compute_close_to_vwap,
    compute_last_30m_ret,
    compute_open_30m_ret,
    compute_realized_vol_1d,
    compute_transactions_count_zscore,
    compute_volume_curve_surprise,
)
from src.features.pipeline import FeaturePipeline, feature_store_records_from_frame, prepare_feature_export_frame


def test_open_30m_ret_happy_path() -> None:
    minute_bars = pd.DataFrame(
        {
            "session_minute": list(range(570, 600)),
            "open": [100.0 + idx * 0.1 for idx in range(30)],
            "close": [100.2 + idx * 0.1 for idx in range(30)],
            "minute_ts": pd.date_range("2026-01-05 14:30", periods=30, freq="min", tz="UTC"),
        },
    )

    value, is_filled = compute_open_30m_ret(minute_bars)

    assert value == pytest.approx(minute_bars["close"].iloc[-1] / minute_bars["open"].iloc[0] - 1.0)
    assert is_filled is False


def test_open_30m_ret_missing_bars() -> None:
    minute_bars = pd.DataFrame(
        {
            "session_minute": list(range(570, 580)),
            "open": [100.0] * 10,
            "close": [101.0] * 10,
            "minute_ts": pd.date_range("2026-01-05 14:30", periods=10, freq="min", tz="UTC"),
        },
    )

    value, is_filled = compute_open_30m_ret(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_open_30m_ret_requires_exact_boundary_minutes() -> None:
    minute_bars = pd.DataFrame(
        {
            "session_minute": list(range(571, 600)),
            "open": [100.0 + idx * 0.1 for idx in range(29)],
            "close": [100.2 + idx * 0.1 for idx in range(29)],
            "minute_ts": pd.date_range("2026-01-05 14:31", periods=29, freq="min", tz="UTC"),
        },
    )

    value, is_filled = compute_open_30m_ret(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_last_30m_ret_happy_path() -> None:
    minute_bars = pd.DataFrame(
        {
            "session_minute": list(range(930, 960)),
            "open": [200.0 + idx * 0.2 for idx in range(30)],
            "close": [200.1 + idx * 0.2 for idx in range(30)],
            "minute_ts": pd.date_range("2026-01-05 20:30", periods=30, freq="min", tz="UTC"),
        },
    )

    value, is_filled = compute_last_30m_ret(minute_bars)

    assert value == pytest.approx(minute_bars["close"].iloc[-1] / minute_bars["open"].iloc[0] - 1.0)
    assert is_filled is False


def test_last_30m_ret_missing_bars() -> None:
    minute_bars = pd.DataFrame(
        {
            "session_minute": list(range(930, 942)),
            "open": [200.0] * 12,
            "close": [201.0] * 12,
            "minute_ts": pd.date_range("2026-01-05 20:30", periods=12, freq="min", tz="UTC"),
        },
    )

    value, is_filled = compute_last_30m_ret(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_last_30m_ret_requires_exact_boundary_minutes() -> None:
    minute_bars = pd.DataFrame(
        {
            "session_minute": list(range(930, 959)),
            "open": [200.0 + idx * 0.2 for idx in range(29)],
            "close": [200.1 + idx * 0.2 for idx in range(29)],
            "minute_ts": pd.date_range("2026-01-05 20:30", periods=29, freq="min", tz="UTC"),
        },
    )

    value, is_filled = compute_last_30m_ret(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_realized_vol_1d_happy_path() -> None:
    closes = 100.0 * np.exp(np.linspace(0.0, 0.06, 300) + 0.002 * np.sin(np.arange(300)))
    minute_bars = pd.DataFrame({"close": closes})

    value, is_filled = compute_realized_vol_1d(minute_bars)
    expected = np.log(pd.Series(closes) / pd.Series(closes).shift(1)).dropna().std(ddof=0) * np.sqrt(390.0)

    assert value == pytest.approx(expected)
    assert is_filled is False


def test_realized_vol_1d_missing_bars() -> None:
    minute_bars = pd.DataFrame({"close": np.linspace(100.0, 101.0, 299)})

    value, is_filled = compute_realized_vol_1d(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_volume_curve_surprise_happy_path() -> None:
    history = pd.DataFrame(
        [
            {bucket: float(100 + bucket + day) for bucket in range(13)}
            for day in range(20)
        ],
    )
    means = history.mean(axis=0)
    stds = history.std(axis=0, ddof=0)
    today = means + 2.0 * stds

    value, is_filled = compute_volume_curve_surprise(
        history_bucket_matrix=history,
        today_bucket_vector=today,
    )

    assert value == pytest.approx(2.0)
    assert is_filled is False


def test_volume_curve_surprise_missing_history() -> None:
    history = pd.DataFrame([{bucket: 100.0 + bucket for bucket in range(13)} for _ in range(10)])
    today = pd.Series({bucket: 110.0 + bucket for bucket in range(13)})

    value, is_filled = compute_volume_curve_surprise(
        history_bucket_matrix=history,
        today_bucket_vector=today,
    )

    assert np.isnan(value)
    assert is_filled is True


def test_close_to_vwap_happy_path() -> None:
    minute_bars = pd.DataFrame(
        {
            "close": [100.0, 102.0, 104.0],
            "volume": [1.0, 2.0, 1.0],
            "vwap": [99.0, 101.0, 103.0],
            "minute_ts": pd.to_datetime(
                ["2026-01-05 20:57:00+00:00", "2026-01-05 20:58:00+00:00", "2026-01-05 20:59:00+00:00"],
                utc=True,
            ),
        },
    )

    value, is_filled = compute_close_to_vwap(minute_bars)

    expected_vwap = (99.0 * 1.0 + 101.0 * 2.0 + 103.0 * 1.0) / 4.0
    assert value == pytest.approx((104.0 - expected_vwap) / expected_vwap)
    assert is_filled is False


def test_close_to_vwap_zero_volume() -> None:
    minute_bars = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "volume": [0.0, 0.0, 0.0],
            "vwap": [100.0, 101.0, 102.0],
            "minute_ts": pd.to_datetime(
                ["2026-01-05 20:57:00+00:00", "2026-01-05 20:58:00+00:00", "2026-01-05 20:59:00+00:00"],
                utc=True,
            ),
        },
    )

    value, is_filled = compute_close_to_vwap(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_close_to_vwap_flags_is_filled_on_nan_close() -> None:
    minute_bars = pd.DataFrame(
        {
            "close": [100.0, 101.0, np.nan],
            "volume": [1.0, 2.0, 3.0],
            "vwap": [100.0, 101.0, 102.0],
            "minute_ts": pd.to_datetime(
                ["2026-01-05 20:57:00+00:00", "2026-01-05 20:58:00+00:00", "2026-01-05 20:59:00+00:00"],
                utc=True,
            ),
        },
    )

    value, is_filled = compute_close_to_vwap(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_close_to_vwap_uses_volume_weighted_minute_vwap() -> None:
    minute_bars = pd.DataFrame(
        {
            "close": [111.0, 111.0],
            "volume": [1000.0, 9000.0],
            "vwap": [100.0, 110.0],
            "minute_ts": pd.to_datetime(
                ["2026-01-05 20:58:00+00:00", "2026-01-05 20:59:00+00:00"],
                utc=True,
            ),
        },
    )

    value, is_filled = compute_close_to_vwap(minute_bars)

    expected_vwap = (100.0 * 1000.0 + 110.0 * 9000.0) / 10000.0
    assert expected_vwap == pytest.approx(109.0)
    assert value == pytest.approx((111.0 - expected_vwap) / expected_vwap)
    assert is_filled is False


def test_close_to_vwap_flags_is_filled_on_missing_session_tail() -> None:
    minute_bars = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "volume": [1.0, 2.0, 3.0],
            "vwap": [100.0, 101.0, 102.0],
            "minute_ts": pd.to_datetime(
                ["2026-01-05 20:56:00+00:00", "2026-01-05 20:57:00+00:00", "2026-01-05 20:58:00+00:00"],
                utc=True,
            ),
        },
    )

    value, is_filled = compute_close_to_vwap(minute_bars)

    assert np.isnan(value)
    assert is_filled is True


def test_transactions_count_zscore_happy_path() -> None:
    txn_history = pd.Series(np.linspace(100.0, 119.0, 20))

    value, is_filled = compute_transactions_count_zscore(
        txn_today=130.0,
        txn_history=txn_history,
    )

    expected = (130.0 - txn_history.mean()) / txn_history.std(ddof=0)
    assert value == pytest.approx(expected)
    assert is_filled is False


def test_transactions_count_zscore_missing_history() -> None:
    txn_history = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])

    value, is_filled = compute_transactions_count_zscore(
        txn_today=110.0,
        txn_history=txn_history,
    )

    assert np.isnan(value)
    assert is_filled is True


def test_transactions_count_zscore_flags_is_filled_on_nan_txn_today() -> None:
    txn_history = pd.Series(np.linspace(100.0, 119.0, 20))

    value, is_filled = compute_transactions_count_zscore(
        txn_today=float("nan"),
        txn_history=txn_history,
    )

    assert np.isnan(value)
    assert is_filled is True


def test_all_9_intraday_features_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    tickers = [f"T{idx:02d}" for idx in range(10)]
    output_dates = pd.bdate_range("2026-02-02", periods=20)
    prior_dates = pd.bdate_range(end=output_dates[0] - pd.Timedelta(days=1), periods=25)
    all_dates = prior_dates.append(output_dates)

    minute_history = _build_minute_history(tickers=tickers, trade_dates=all_dates)
    daily_prices = _build_daily_prices_from_minute(minute_history)

    empty_frame = pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    monkeypatch.setattr(pipeline_module, "get_prices_pit", lambda *args, **kwargs: daily_prices.copy())
    monkeypatch.setattr(pipeline_module, "compute_technical_features", lambda *args, **kwargs: empty_frame.copy())
    monkeypatch.setattr(pipeline_module, "compute_fundamental_features", lambda *args, **kwargs: empty_frame.copy())
    monkeypatch.setattr(pipeline_module, "compute_alternative_features_batch", lambda *args, **kwargs: empty_frame.copy())
    monkeypatch.setattr(pipeline_module, "compute_sector_rotation_features", lambda *args, **kwargs: empty_frame.copy())
    monkeypatch.setattr(pipeline_module, "compute_composite_features", lambda *args, **kwargs: empty_frame.copy())
    monkeypatch.setattr(
        FeaturePipeline,
        "_compute_broadcast_macro_features",
        lambda self, *args, **kwargs: empty_frame.copy(),
    )
    monkeypatch.setattr(pipeline_module, "load_intraday_minute_history", lambda *args, **kwargs: minute_history.copy())
    monkeypatch.setattr(pipeline_module, "preprocess_features", lambda frame: frame)

    pipeline = FeaturePipeline()
    result = pipeline.run(
        tickers=tickers,
        start_date=output_dates[0].date(),
        end_date=output_dates[-1].date(),
        as_of=(output_dates[-1] + pd.Timedelta(days=1)).date(),
    )

    assert set(result["feature_name"].unique()) == {
        "gap_pct",
        "overnight_ret",
        "intraday_ret",
        "open_30m_ret",
        "last_30m_ret",
        "realized_vol_1d",
        "volume_curve_surprise",
        "close_to_vwap",
        "transactions_count_zscore",
    }
    assert len(result) == len(tickers) * len(output_dates) * 9

    missing_rates = result.groupby("feature_name")["feature_value"].apply(lambda series: float(series.isna().mean()))
    assert all(rate < 0.1 for rate in missing_rates)

    curve_values = result.loc[result["feature_name"] == "volume_curve_surprise", "feature_value"]
    assert curve_values.abs().max() < 1e-12

    prepared = prepare_feature_export_frame(result)
    records = feature_store_records_from_frame(result, batch_id="intraday-test")
    assert len(prepared) == len(result)
    assert len(records) == len(result)
    assert all(record["batch_id"] == "intraday-test" for record in records)


def _build_minute_history(*, tickers: list[str], trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker_idx, ticker in enumerate(tickers):
        base_offset = ticker_idx * 5.0
        for day_idx, trade_day in enumerate(trade_dates):
            timestamps = _session_timestamps(trade_day.date())
            day_base = 100.0 + base_offset + day_idx
            for minute_idx, minute_ts in enumerate(timestamps):
                open_px = day_base + 0.01 * minute_idx
                close_px = open_px + 0.002
                rows.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_day.date(),
                        "minute_ts": minute_ts,
                        "open": open_px,
                        "high": close_px + 0.001,
                        "low": open_px - 0.001,
                        "close": close_px,
                        "volume": 1_000 + (minute_idx % 25) * 10 + ticker_idx,
                        "vwap": (open_px + close_px) / 2.0,
                        "transactions": 100 + (minute_idx % 11) + ticker_idx,
                    },
                )
    return pd.DataFrame(rows)


def _build_daily_prices_from_minute(minute_history: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        minute_history.groupby(["ticker", "trade_date"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .sort_values(["ticker", "trade_date"])
    )
    grouped["adj_close"] = grouped["close"]
    spy_rows = grouped.loc[grouped["ticker"] == grouped["ticker"].iloc[0]].copy()
    spy_rows["ticker"] = "SPY"
    return pd.concat([grouped, spy_rows], ignore_index=True)


def _session_timestamps(trade_day: date) -> pd.DatetimeIndex:
    morning = pd.date_range(
        f"{trade_day.isoformat()} 09:30",
        f"{trade_day.isoformat()} 13:59",
        freq="min",
        tz="America/New_York",
    )
    close_window = pd.date_range(
        f"{trade_day.isoformat()} 15:30",
        f"{trade_day.isoformat()} 15:59",
        freq="min",
        tz="America/New_York",
    )
    return morning.append(close_window).tz_convert("UTC")
