from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.trade_microstructure_minute_proxy import (
    compute_late_day_aggressiveness_minute,
    compute_offhours_trade_ratio_minute,
    compute_trade_imbalance_proxy_minute,
)


def _et(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw, tz="America/New_York").tz_convert("UTC")


def _bars(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "ticker": "AAPL",
        "trading_date": pd.Timestamp("2026-01-05").date(),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
        "transactions": 100,
        "vwap": 100.0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_minute_proxy_happy_path_tick_rule_late_and_offhours() -> None:
    minute_df = _bars(
        [
            {"minute_ts": _et("2026-01-05 08:00"), "close": 99, "volume": 100},
            {"minute_ts": _et("2026-01-05 10:00"), "close": 100, "volume": 100},
            {"minute_ts": _et("2026-01-05 10:01"), "close": 101, "volume": 100},
            {"minute_ts": _et("2026-01-05 15:00"), "close": 100, "volume": 100},
            {"minute_ts": _et("2026-01-05 15:01"), "close": 99, "volume": 100},
            {"minute_ts": _et("2026-01-05 19:00"), "close": 100, "volume": 100},
        ],
    )

    assert compute_trade_imbalance_proxy_minute(minute_df) == pytest.approx(-0.25)
    assert compute_late_day_aggressiveness_minute(minute_df) == pytest.approx(2.0)
    assert compute_offhours_trade_ratio_minute(minute_df) == pytest.approx(2 / 6)


def test_minute_proxy_empty_frame_returns_nan() -> None:
    minute_df = pd.DataFrame(columns=["minute_ts", "close", "volume"])

    assert np.isnan(compute_trade_imbalance_proxy_minute(minute_df))
    assert np.isnan(compute_late_day_aggressiveness_minute(minute_df))
    assert np.isnan(compute_offhours_trade_ratio_minute(minute_df))


def test_offhours_ratio_returns_zero_with_only_regular_session_bars() -> None:
    minute_df = _bars(
        [
            {"minute_ts": _et("2026-01-05 10:00"), "close": 100, "volume": 100},
            {"minute_ts": _et("2026-01-05 10:01"), "close": 101, "volume": 100},
        ],
    )

    assert compute_offhours_trade_ratio_minute(minute_df) == pytest.approx(0.0)


def test_regular_features_return_nan_with_only_offhours_bars() -> None:
    minute_df = _bars(
        [
            {"minute_ts": _et("2026-01-05 08:00"), "close": 100, "volume": 100},
            {"minute_ts": _et("2026-01-05 19:00"), "close": 101, "volume": 100},
        ],
    )

    assert np.isnan(compute_trade_imbalance_proxy_minute(minute_df))
    assert np.isnan(compute_late_day_aggressiveness_minute(minute_df))
    assert compute_offhours_trade_ratio_minute(minute_df) == pytest.approx(1.0)


def test_naive_timestamp_raises_value_error() -> None:
    minute_df = _bars([{"minute_ts": pd.Timestamp("2026-01-05 10:00"), "close": 100, "volume": 100}])

    with pytest.raises(ValueError, match="timezone-aware"):
        compute_trade_imbalance_proxy_minute(minute_df)


def test_five_minute_bar_spec_works() -> None:
    minute_df = _bars(
        [
            {"minute_ts": _et("2026-01-05 10:00"), "close": 100, "volume": 100},
            {"minute_ts": _et("2026-01-05 10:05"), "close": 101, "volume": 200},
            {"minute_ts": _et("2026-01-05 10:10"), "close": 100, "volume": 100},
        ],
    )

    assert compute_trade_imbalance_proxy_minute(minute_df) == pytest.approx(0.25)
