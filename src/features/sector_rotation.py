from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.features.sector import (
    GICS_RECLASS_DATE,
    GICS_TO_FMP,
    _FROM_CONSUMER_TO_COMM,
    _FROM_TECH_TO_COMM,
    load_sector_map_pit,
)

SECTOR_ROTATION_FEATURE_NAMES = (
    "sector_rel_ret_5d",
    "sector_volume_surge",
    "sector_pressure",
    "stock_vs_sector_20d",
    "sector_pressure_x_divergence",
)

SECTOR_ROTATION_ETF_TICKERS = (
    "SPY",
    "QQQ",
    "IWM",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLP",
    "XLY",
    "XLU",
    "XLB",
    "XLRE",
    "XLC",
)

SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Consumer Cyclical": "XLY",
    "Utilities": "XLU",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def compute_sector_rotation_features(
    *,
    base_features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
) -> pd.DataFrame:
    if base_features_df.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    wide = (
        base_features_df.pivot_table(
            index=["ticker", "trade_date"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    if wide.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    prepared_prices = prices_df.copy()
    prepared_prices["ticker"] = prepared_prices["ticker"].astype(str).str.upper()
    prepared_prices["trade_date"] = pd.to_datetime(prepared_prices["trade_date"]).dt.date
    for column in ("close", "adj_close", "volume"):
        prepared_prices[column] = pd.to_numeric(prepared_prices[column], errors="coerce")
    prepared_prices["price"] = prepared_prices["adj_close"].fillna(prepared_prices["close"])
    prepared_prices.sort_values(["ticker", "trade_date"], inplace=True)

    etf_prices = prepared_prices.loc[
        prepared_prices["ticker"].isin(SECTOR_ROTATION_ETF_TICKERS),
        ["ticker", "trade_date", "price", "volume"],
    ].copy()
    if etf_prices.empty:
        return _empty_sector_rotation_frame(wide)

    etf_prices["ret_5d"] = etf_prices.groupby("ticker")["price"].pct_change(5)
    etf_prices["ret_20d"] = etf_prices.groupby("ticker")["price"].pct_change(20)
    etf_prices["adv20"] = etf_prices.groupby("ticker")["volume"].transform(
        lambda values: values.rolling(20, min_periods=20).mean(),
    )
    etf_prices["sector_volume_surge"] = etf_prices["volume"] / etf_prices["adv20"]

    spy_ret = etf_prices.loc[etf_prices["ticker"] == "SPY", ["trade_date", "ret_5d"]].rename(
        columns={"ret_5d": "spy_ret_5d"},
    )
    etf_metrics = etf_prices.merge(spy_ret, on="trade_date", how="left")
    etf_metrics["sector_rel_ret_5d"] = etf_metrics["ret_5d"] - etf_metrics["spy_ret_5d"]

    grouped = etf_metrics.groupby("ticker", sort=False)
    etf_metrics["sector_rel_ret_5d_z"] = grouped["sector_rel_ret_5d"].transform(_rolling_zscore)
    etf_metrics["sector_volume_surge_z"] = grouped["sector_volume_surge"].transform(_rolling_zscore)
    etf_metrics["sector_pressure"] = etf_metrics["sector_rel_ret_5d_z"] * etf_metrics["sector_volume_surge_z"]
    etf_metrics = etf_metrics[["ticker", "trade_date", "ret_20d", "sector_rel_ret_5d", "sector_volume_surge", "sector_pressure"]]
    etf_metrics = etf_metrics.rename(columns={"ticker": "sector_etf", "ret_20d": "sector_ret_20d"})

    current_sector_map = load_sector_map_pit()
    wide["sector"] = wide.apply(
        lambda row: _sector_for_trade_date(
            ticker=str(row["ticker"]).upper(),
            trade_date=pd.to_datetime(row["trade_date"]).date(),
            current_sector=current_sector_map.get(str(row["ticker"]).upper()),
        ),
        axis=1,
    )
    wide["sector_etf"] = wide["sector"].map(SECTOR_TO_ETF)

    merged = wide.merge(etf_metrics, on=["trade_date", "sector_etf"], how="left")
    merged["stock_vs_sector_20d"] = merged.get("ret_20d") - merged.get("sector_ret_20d")
    merged["sector_pressure_x_divergence"] = merged.get("sector_pressure") * merged.get("stock_vs_sector_20d")

    long = merged.melt(
        id_vars=["ticker", "trade_date"],
        value_vars=list(SECTOR_ROTATION_FEATURE_NAMES),
        var_name="feature_name",
        value_name="feature_value",
    )
    return long.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def _empty_sector_rotation_frame(wide: pd.DataFrame) -> pd.DataFrame:
    frame = wide.loc[:, ["ticker", "trade_date"]].copy()
    if frame.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    rows = []
    for feature_name in SECTOR_ROTATION_FEATURE_NAMES:
        subset = frame.copy()
        subset["feature_name"] = feature_name
        subset["feature_value"] = np.nan
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def _rolling_zscore(series: pd.Series) -> pd.Series:
    min_periods = 20
    rolling_mean = series.rolling(252, min_periods=min_periods).mean()
    rolling_std = series.rolling(252, min_periods=min_periods).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


def _sector_for_trade_date(*, ticker: str, trade_date: date, current_sector: str | None) -> str | None:
    sector = GICS_TO_FMP.get(str(current_sector or ""), str(current_sector or ""))
    if trade_date < GICS_RECLASS_DATE and sector == "Communication Services":
        if ticker in _FROM_TECH_TO_COMM:
            return "Technology"
        if ticker in _FROM_CONSUMER_TO_COMM:
            return "Consumer Cyclical"
    return sector or None
