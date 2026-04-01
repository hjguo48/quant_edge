from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.corporate_actions import adjust_for_dividends, adjust_for_splits


def test_adjust_for_splits_multi_ticker() -> None:
    prices_df = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "open": 100.0, "high": 110.0, "low": 95.0, "close": 100.0, "adj_close": 100.0, "volume": 1_000},
            {"ticker": "AAA", "trade_date": date(2024, 1, 4), "open": 52.0, "high": 55.0, "low": 50.0, "close": 54.0, "adj_close": 54.0, "volume": 2_000},
            {"ticker": "BBB", "trade_date": date(2024, 1, 2), "open": 200.0, "high": 210.0, "low": 190.0, "close": 205.0, "adj_close": 205.0, "volume": 500},
        ],
    )
    splits_df = pd.DataFrame(
        [
            {"ticker": "AAA", "ex_date": date(2024, 1, 3), "ratio": 2.0},
        ],
    )

    adjusted = adjust_for_splits(prices_df, splits_df)

    aaa_pre_split = adjusted.loc[
        (adjusted["ticker"] == "AAA") & (adjusted["trade_date"] == date(2024, 1, 2))
    ].iloc[0]
    aaa_post_split = adjusted.loc[
        (adjusted["ticker"] == "AAA") & (adjusted["trade_date"] == date(2024, 1, 4))
    ].iloc[0]
    bbb_row = adjusted.loc[
        (adjusted["ticker"] == "BBB") & (adjusted["trade_date"] == date(2024, 1, 2))
    ].iloc[0]

    assert aaa_pre_split["close"] == pytest.approx(50.0)
    assert aaa_pre_split["volume"] == pytest.approx(2_000.0)
    assert aaa_post_split["close"] == pytest.approx(54.0)
    assert bbb_row["close"] == pytest.approx(205.0)
    assert bbb_row["volume"] == pytest.approx(500.0)


def test_adjust_for_dividends_multi_ticker() -> None:
    prices_df = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 1), "open": 99.0, "high": 101.0, "low": 98.0, "close": 100.0, "adj_close": 100.0, "volume": 1_000},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "open": 101.0, "high": 103.0, "low": 100.0, "close": 102.0, "adj_close": 102.0, "volume": 1_100},
            {"ticker": "AAA", "trade_date": date(2024, 1, 3), "open": 97.0, "high": 99.0, "low": 96.0, "close": 98.0, "adj_close": 98.0, "volume": 1_200},
            {"ticker": "BBB", "trade_date": date(2024, 1, 2), "open": 50.0, "high": 52.0, "low": 49.0, "close": 51.0, "adj_close": 51.0, "volume": 700},
        ],
    )
    dividends_df = pd.DataFrame(
        [
            {"ticker": "AAA", "ex_date": date(2024, 1, 3), "cash_amount": 5.0},
        ],
    )

    adjusted = adjust_for_dividends(prices_df, dividends_df)
    adjustment_factor = (102.0 - 5.0) / 102.0

    aaa_pre_dividend = adjusted.loc[
        (adjusted["ticker"] == "AAA") & (adjusted["trade_date"] == date(2024, 1, 1))
    ].iloc[0]
    aaa_ex_date = adjusted.loc[
        (adjusted["ticker"] == "AAA") & (adjusted["trade_date"] == date(2024, 1, 3))
    ].iloc[0]
    bbb_row = adjusted.loc[
        (adjusted["ticker"] == "BBB") & (adjusted["trade_date"] == date(2024, 1, 2))
    ].iloc[0]

    assert aaa_pre_dividend["close"] == pytest.approx(100.0 * adjustment_factor)
    assert aaa_ex_date["close"] == pytest.approx(98.0)
    assert bbb_row["close"] == pytest.approx(51.0)
