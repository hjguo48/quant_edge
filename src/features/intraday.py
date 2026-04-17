from __future__ import annotations

from datetime import date

import pandas as pd

INTRADAY_FEATURE_NAMES = (
    "gap_pct",
    "overnight_ret",
    "intraday_ret",
)


def aggregate_minute_to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "transactions",
            ],
        )

    frame = minute_df.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame.sort_values(["ticker", "trade_date", "minute_ts"], inplace=True)
    aggregated = (
        frame.groupby(["ticker", "trade_date"], as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            vwap=("vwap", "mean"),
            transactions=("transactions", "sum"),
        )
    )
    return aggregated


def compute_intraday_features(
    *,
    minute_df: pd.DataFrame,
    daily_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    if minute_df.empty:
        return _empty_feature_frame()

    intraday_daily = aggregate_minute_to_daily(minute_df)
    prices = daily_prices_df.loc[:, ["ticker", "trade_date", "close"]].copy()
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices.sort_values(["ticker", "trade_date"], inplace=True)
    prices["prev_close"] = prices.groupby("ticker")["close"].shift(1)

    feature_base = intraday_daily.merge(
        prices.loc[:, ["ticker", "trade_date", "prev_close"]],
        on=["ticker", "trade_date"],
        how="left",
    )
    feature_base["gap_pct"] = (feature_base["open"] - feature_base["prev_close"]) / feature_base["prev_close"]
    feature_base["overnight_ret"] = feature_base["gap_pct"]
    feature_base["intraday_ret"] = (feature_base["close"] - feature_base["open"]) / feature_base["open"]

    rows: list[dict[str, object]] = []
    for row in feature_base.itertuples(index=False):
        trade_date = row.trade_date if isinstance(row.trade_date, date) else pd.Timestamp(row.trade_date).date()
        rows.extend(
            [
                {
                    "ticker": row.ticker,
                    "trade_date": trade_date,
                    "feature_name": "gap_pct",
                    "feature_value": row.gap_pct,
                    "is_filled": False,
                },
                {
                    "ticker": row.ticker,
                    "trade_date": trade_date,
                    "feature_name": "overnight_ret",
                    "feature_value": row.overnight_ret,
                    "is_filled": False,
                },
                {
                    "ticker": row.ticker,
                    "trade_date": trade_date,
                    "feature_name": "intraday_ret",
                    "feature_value": row.intraday_ret,
                    "is_filled": False,
                },
            ],
        )
    return pd.DataFrame(rows, columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"])


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"])
