from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.data.corporate_actions import adjust_for_dividends, adjust_for_splits, fetch_corporate_actions


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


def test_fetch_corporate_actions_uses_polygon_ticker_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_tickers: list[str] = []
    persisted: dict[str, object] = {}

    def fake_iterate_polygon_results(
        session: object,
        url: str,
        params: dict[str, object],
        *,
        throttle: object,
    ) -> list[dict[str, object]]:
        requested_tickers.append(str(params["ticker"]))
        if "splits" in url:
            return [{"ticker": "BRK.B", "execution_date": "2024-01-03", "split_from": 1, "split_to": 2}]
        return [{"ticker": "BRK.B", "ex_dividend_date": "2024-01-04", "cash_amount": 1.25}]

    monkeypatch.setattr("src.data.corporate_actions._get_http_session", lambda: object())
    monkeypatch.setattr("src.data.corporate_actions._iterate_polygon_results", fake_iterate_polygon_results)
    monkeypatch.setattr(
        "src.data.corporate_actions._persist_corporate_actions",
        lambda frame, tickers, start, end: persisted.update(
            {
                "frame": frame.copy(),
                "tickers": tickers,
                "start": start,
                "end": end,
            },
        ),
    )

    frame = fetch_corporate_actions(["BRK-B"], date(2024, 1, 1), date(2024, 1, 31))

    assert requested_tickers == ["BRK.B", "BRK.B"]
    assert frame["ticker"].tolist() == ["BRK-B", "BRK-B"]
    assert frame["details_json"].tolist()[0]["ticker"] == "BRK-B"
    assert frame["details_json"].tolist()[1]["ticker"] == "BRK-B"
    assert persisted["tickers"] == ("BRK-B",)
