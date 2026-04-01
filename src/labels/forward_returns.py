from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from loguru import logger


def compute_forward_returns(
    prices_df: pd.DataFrame,
    horizons: Sequence[int] = (1, 2, 5, 10, 20, 60),
    benchmark_ticker: str = "SPY",
) -> pd.DataFrame:
    required_columns = {"ticker", "trade_date", "adj_close", "close"}
    missing_columns = sorted(required_columns - set(prices_df.columns))
    if missing_columns:
        raise ValueError(f"prices_df is missing required columns: {missing_columns}")

    prepared = prices_df.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"]).dt.date
    prepared["base_price"] = pd.to_numeric(prepared["adj_close"], errors="coerce").fillna(
        pd.to_numeric(prepared["close"], errors="coerce"),
    )
    prepared.sort_values(["ticker", "trade_date"], inplace=True)

    benchmark_symbol = benchmark_ticker.upper()
    benchmark_frame = prepared.loc[prepared["ticker"] == benchmark_symbol, ["trade_date", "base_price"]].copy()
    if benchmark_frame.empty:
        logger.warning("benchmark ticker {} missing from price input; excess returns will be NaN", benchmark_symbol)

    label_frames: list[pd.DataFrame] = []
    for horizon in horizons:
        horizon_frame = prepared[["ticker", "trade_date", "base_price"]].copy()
        horizon_frame["future_price"] = horizon_frame.groupby("ticker")["base_price"].shift(-horizon)
        horizon_frame["forward_return"] = (
            horizon_frame["future_price"] - horizon_frame["base_price"]
        ) / horizon_frame["base_price"]

        if not benchmark_frame.empty:
            benchmark_horizon = benchmark_frame.copy()
            benchmark_horizon["future_price"] = benchmark_horizon["base_price"].shift(-horizon)
            benchmark_horizon["benchmark_return"] = (
                benchmark_horizon["future_price"] - benchmark_horizon["base_price"]
            ) / benchmark_horizon["base_price"]
            horizon_frame = horizon_frame.merge(
                benchmark_horizon[["trade_date", "benchmark_return"]],
                on="trade_date",
                how="left",
            )
            horizon_frame["excess_return"] = horizon_frame["forward_return"] - horizon_frame["benchmark_return"]
        else:
            horizon_frame["excess_return"] = pd.NA

        horizon_frame["horizon"] = int(horizon)
        label_frames.append(
            horizon_frame[["ticker", "trade_date", "horizon", "forward_return", "excess_return"]],
        )

    labels = pd.concat(label_frames, ignore_index=True)
    logger.info("computed {} forward-return labels across {} horizons", len(labels), len(horizons))
    return labels
