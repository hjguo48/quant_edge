from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.features.intraday import aggregate_minute_to_daily, compute_intraday_features


def test_compute_intraday_features_calculates_gap_and_intraday_return() -> None:
    minute_df = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "minute_ts": pd.Timestamp("2026-01-05 14:30", tz="UTC"), "open": 102.0, "high": 102.0, "low": 102.0, "close": 102.0, "volume": 100, "vwap": 102.0, "transactions": 10},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "minute_ts": pd.Timestamp("2026-01-05 20:59", tz="UTC"), "open": 104.0, "high": 104.0, "low": 104.0, "close": 104.0, "volume": 150, "vwap": 104.0, "transactions": 15},
            {"ticker": "AAA", "trade_date": date(2026, 1, 6), "minute_ts": pd.Timestamp("2026-01-06 14:30", tz="UTC"), "open": 103.0, "high": 103.0, "low": 103.0, "close": 103.0, "volume": 100, "vwap": 103.0, "transactions": 10},
            {"ticker": "AAA", "trade_date": date(2026, 1, 6), "minute_ts": pd.Timestamp("2026-01-06 20:59", tz="UTC"), "open": 101.0, "high": 101.0, "low": 101.0, "close": 101.0, "volume": 100, "vwap": 101.0, "transactions": 10},
        ],
    )
    daily_prices = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 1, 2), "close": 100.0},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "close": 104.0},
            {"ticker": "AAA", "trade_date": date(2026, 1, 6), "close": 101.0},
        ],
    )

    features = compute_intraday_features(minute_df=minute_df, daily_prices_df=daily_prices)
    pivot = features.pivot_table(index=["ticker", "trade_date"], columns="feature_name", values="feature_value")

    assert float(pivot.loc[("AAA", date(2026, 1, 5)), "gap_pct"]) == pytest.approx(0.02)
    assert float(pivot.loc[("AAA", date(2026, 1, 5)), "overnight_ret"]) == pytest.approx(0.02)
    assert float(pivot.loc[("AAA", date(2026, 1, 5)), "intraday_ret"]) == pytest.approx(2.0 / 102.0)

    assert float(pivot.loc[("AAA", date(2026, 1, 6)), "gap_pct"]) == pytest.approx((103.0 - 104.0) / 104.0)
    assert float(pivot.loc[("AAA", date(2026, 1, 6)), "intraday_ret"]) == pytest.approx((101.0 - 103.0) / 103.0)


def test_aggregate_minute_to_daily_uses_first_last_max_min_and_sums() -> None:
    minute_df = pd.DataFrame(
        [
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "minute_ts": pd.Timestamp("2026-01-05 14:30", tz="UTC"), "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.1, "volume": 100, "vwap": 10.05, "transactions": 10},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "minute_ts": pd.Timestamp("2026-01-05 14:31", tz="UTC"), "open": 10.1, "high": 10.5, "low": 10.0, "close": 10.4, "volume": 200, "vwap": 10.3, "transactions": 20},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "minute_ts": pd.Timestamp("2026-01-05 20:59", tz="UTC"), "open": 10.4, "high": 10.6, "low": 10.3, "close": 10.5, "volume": 300, "vwap": 10.45, "transactions": 30},
        ],
    )

    aggregated = aggregate_minute_to_daily(minute_df)

    assert len(aggregated) == 1
    row = aggregated.iloc[0]
    assert row["open"] == 10.0
    assert row["high"] == 10.6
    assert row["low"] == 9.9
    assert row["close"] == 10.5
    assert row["volume"] == 600
    assert row["transactions"] == 60
