from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EASTERN = ZoneInfo("America/New_York")
_FIRST_30_START = 9 * 60 + 30
_FIRST_30_END = 10 * 60
_LAST_30_START = 15 * 60 + 30
_LAST_30_END = 16 * 60
_INTRADAY_BUCKET_COUNT = 13

INTRADAY_FEATURE_NAMES = (
    "gap_pct",
    "overnight_ret",
    "intraday_ret",
    "open_30m_ret",
    "last_30m_ret",
    "realized_vol_1d",
    "volume_curve_surprise",
    "close_to_vwap",
    "transactions_count_zscore",
)


def aggregate_minute_to_daily(minute_df: pd.DataFrame) -> pd.DataFrame:
    frame = _prepare_minute_frame(minute_df)
    if frame.empty:
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


def compute_open_30m_ret(minute_bars: pd.DataFrame) -> tuple[float, bool]:
    return _compute_window_return(minute_bars, start_minute=_FIRST_30_START, end_minute=_FIRST_30_END)


def compute_last_30m_ret(minute_bars: pd.DataFrame) -> tuple[float, bool]:
    return _compute_window_return(minute_bars, start_minute=_LAST_30_START, end_minute=_LAST_30_END)


def compute_realized_vol_1d(minute_bars: pd.DataFrame) -> tuple[float, bool]:
    if len(minute_bars) < 300:
        return np.nan, True
    closes = pd.to_numeric(minute_bars["close"], errors="coerce")
    log_returns = np.log(closes / closes.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(log_returns) < 299:
        return np.nan, True
    return float(log_returns.std(ddof=0) * np.sqrt(390.0)), False


def compute_volume_curve_surprise(
    *,
    history_bucket_matrix: pd.DataFrame,
    today_bucket_vector: pd.Series,
) -> tuple[float, bool]:
    if history_bucket_matrix.empty or len(history_bucket_matrix.index) < 20:
        return np.nan, True

    aligned_history = history_bucket_matrix.reindex(columns=range(_INTRADAY_BUCKET_COUNT)).astype(float)
    aligned_today = today_bucket_vector.reindex(range(_INTRADAY_BUCKET_COUNT)).astype(float)
    means = aligned_history.mean(axis=0)
    stds = aligned_history.std(axis=0, ddof=0)
    zscores: list[float] = []
    for bucket in range(_INTRADAY_BUCKET_COUNT):
        today_value = aligned_today.iloc[bucket]
        mean_value = means.iloc[bucket]
        std_value = stds.iloc[bucket]
        if pd.isna(today_value) or pd.isna(mean_value):
            continue
        if pd.isna(std_value) or std_value == 0:
            zscores.append(0.0 if np.isclose(today_value, mean_value) else np.nan)
            continue
        zscores.append(float((today_value - mean_value) / std_value))

    finite = [value for value in zscores if np.isfinite(value)]
    if not finite:
        return np.nan, True
    return float(np.mean(finite)), False


def compute_close_to_vwap(minute_bars: pd.DataFrame) -> tuple[float, bool]:
    volumes = pd.to_numeric(minute_bars["volume"], errors="coerce").fillna(0.0)
    if float(volumes.sum()) <= 0.0:
        return np.nan, True
    closes = pd.to_numeric(minute_bars["close"], errors="coerce")
    day_close = float(closes.iloc[-1])
    if pd.isna(day_close):
        return np.nan, True
    day_vwap = float((closes * volumes).sum() / volumes.sum())
    if not np.isfinite(day_vwap) or day_vwap == 0.0:
        return np.nan, True
    return float((day_close - day_vwap) / day_vwap), False


def compute_transactions_count_zscore(
    *,
    txn_today: float,
    txn_history: pd.Series,
) -> tuple[float, bool]:
    if pd.isna(txn_today):
        return np.nan, True
    clean_history = pd.to_numeric(txn_history, errors="coerce").dropna().astype(float)
    if len(clean_history.index) < 10:
        return np.nan, True
    mean_value = float(clean_history.mean())
    std_value = float(clean_history.std(ddof=0))
    if std_value == 0.0:
        return (0.0, False) if np.isclose(float(txn_today), mean_value) else (np.nan, True)
    return float((float(txn_today) - mean_value) / std_value), False


def compute_intraday_features(
    *,
    minute_df: pd.DataFrame,
    daily_prices_df: pd.DataFrame,
) -> pd.DataFrame:
    frame = _prepare_minute_frame(minute_df)
    if frame.empty:
        return _empty_feature_frame()

    intraday_daily = aggregate_minute_to_daily(frame)
    if intraday_daily.empty:
        return _empty_feature_frame()

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

    bucket_daily = _build_daily_bucket_matrix(frame)
    txn_daily = (
        frame.groupby(["ticker", "trade_date"], as_index=False)["transactions"]
        .sum(min_count=1)
        .rename(columns={"transactions": "txn_today"})
    )

    per_day_metrics: list[dict[str, object]] = []
    for (ticker, trade_date), group in frame.groupby(["ticker", "trade_date"], sort=True):
        group = group.sort_values("minute_ts")
        open_30m_ret, open_30m_filled = compute_open_30m_ret(group)
        last_30m_ret, last_30m_filled = compute_last_30m_ret(group)
        realized_vol_1d, realized_vol_filled = compute_realized_vol_1d(group)
        close_to_vwap, close_to_vwap_filled = compute_close_to_vwap(group)
        txn_today = float(pd.to_numeric(group["transactions"], errors="coerce").fillna(0.0).sum())
        per_day_metrics.append(
            {
                "ticker": ticker,
                "trade_date": trade_date,
                "open_30m_ret": open_30m_ret,
                "open_30m_ret_is_filled": open_30m_filled,
                "last_30m_ret": last_30m_ret,
                "last_30m_ret_is_filled": last_30m_filled,
                "realized_vol_1d": realized_vol_1d,
                "realized_vol_1d_is_filled": realized_vol_filled,
                "close_to_vwap": close_to_vwap,
                "close_to_vwap_is_filled": close_to_vwap_filled,
                "txn_today": txn_today,
            },
        )

    per_day = pd.DataFrame(per_day_metrics)
    if per_day.empty:
        return _empty_feature_frame()

    curve_rows: list[dict[str, object]] = []
    for ticker, ticker_buckets in bucket_daily.groupby("ticker", sort=True):
        ticker_buckets = ticker_buckets.sort_values("trade_date").reset_index(drop=True)
        ticker_txn = txn_daily.loc[txn_daily["ticker"] == ticker].sort_values("trade_date").reset_index(drop=True)
        txn_map = ticker_txn.set_index("trade_date")["txn_today"]
        for idx, row in ticker_buckets.iterrows():
            trade_date = row["trade_date"]
            prior_buckets = ticker_buckets.iloc[max(0, idx - 30) : idx]
            bucket_history = prior_buckets.reindex(columns=range(_INTRADAY_BUCKET_COUNT), fill_value=0.0)
            today_vector = row.reindex(range(_INTRADAY_BUCKET_COUNT), fill_value=0.0)
            volume_curve_surprise, volume_curve_filled = compute_volume_curve_surprise(
                history_bucket_matrix=bucket_history,
                today_bucket_vector=today_vector,
            )
            prior_txn = txn_map.loc[txn_map.index < trade_date].tail(20)
            transactions_count_zscore, txn_filled = compute_transactions_count_zscore(
                txn_today=float(txn_map.get(trade_date, np.nan)),
                txn_history=prior_txn,
            )
            curve_rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "volume_curve_surprise": volume_curve_surprise,
                    "volume_curve_surprise_is_filled": volume_curve_filled,
                    "transactions_count_zscore": transactions_count_zscore,
                    "transactions_count_zscore_is_filled": txn_filled,
                },
            )

    curve_frame = pd.DataFrame(curve_rows)
    combined = feature_base.merge(
        per_day.drop(columns=["txn_today"]),
        on=["ticker", "trade_date"],
        how="left",
    ).merge(
        curve_frame,
        on=["ticker", "trade_date"],
        how="left",
    )

    feature_specs = [
        ("gap_pct", "gap_pct", "gap_pct", False),
        ("overnight_ret", "overnight_ret", "overnight_ret", False),
        ("intraday_ret", "intraday_ret", "intraday_ret", False),
        ("open_30m_ret", "open_30m_ret", "open_30m_ret_is_filled", True),
        ("last_30m_ret", "last_30m_ret", "last_30m_ret_is_filled", True),
        ("realized_vol_1d", "realized_vol_1d", "realized_vol_1d_is_filled", True),
        ("volume_curve_surprise", "volume_curve_surprise", "volume_curve_surprise_is_filled", True),
        ("close_to_vwap", "close_to_vwap", "close_to_vwap_is_filled", True),
        ("transactions_count_zscore", "transactions_count_zscore", "transactions_count_zscore_is_filled", True),
    ]

    rows: list[dict[str, object]] = []
    for row in combined.itertuples(index=False):
        trade_day = row.trade_date if isinstance(row.trade_date, date) else pd.Timestamp(row.trade_date).date()
        for feature_name, value_attr, filled_attr, has_filled_flag in feature_specs:
            rows.append(
                {
                    "ticker": row.ticker,
                    "trade_date": trade_day,
                    "feature_name": feature_name,
                    "feature_value": getattr(row, value_attr),
                    "is_filled": bool(getattr(row, filled_attr)) if has_filled_flag else False,
                },
            )
    return pd.DataFrame(rows, columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"])


def _prepare_minute_frame(minute_df: pd.DataFrame) -> pd.DataFrame:
    if minute_df.empty:
        return pd.DataFrame()
    frame = minute_df.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["minute_ts"] = pd.to_datetime(frame["minute_ts"], utc=True, errors="coerce")
    frame = frame.loc[frame["minute_ts"].notna()].copy()
    frame["minute_ts_et"] = frame["minute_ts"].dt.tz_convert(EASTERN)
    frame["trade_date"] = pd.to_datetime(frame.get("trade_date", frame["minute_ts_et"].dt.date)).dt.date
    frame["session_minute"] = frame["minute_ts_et"].dt.hour * 60 + frame["minute_ts_et"].dt.minute
    for column in ("open", "high", "low", "close", "volume", "vwap", "transactions"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.sort_values(["ticker", "trade_date", "minute_ts"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _compute_window_return(
    minute_bars: pd.DataFrame,
    *,
    start_minute: int,
    end_minute: int,
) -> tuple[float, bool]:
    window = minute_bars.loc[
        (minute_bars["session_minute"] >= start_minute)
        & (minute_bars["session_minute"] < end_minute)
    ].sort_values("minute_ts")
    if len(window.index) < 25:
        return np.nan, True
    first_open = pd.to_numeric(window["open"], errors="coerce").iloc[0]
    last_close = pd.to_numeric(window["close"], errors="coerce").iloc[-1]
    if pd.isna(first_open) or pd.isna(last_close) or float(first_open) == 0.0:
        return np.nan, True
    return float(last_close / first_open - 1.0), False


def _build_daily_bucket_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    bucket_frame = frame.copy()
    bucket_frame["bucket"] = ((bucket_frame["session_minute"] - _FIRST_30_START) // 30).astype(int)
    bucket_frame = bucket_frame.loc[bucket_frame["bucket"].between(0, _INTRADAY_BUCKET_COUNT - 1)].copy()
    grouped = (
        bucket_frame.groupby(["ticker", "trade_date", "bucket"], as_index=False)["volume"]
        .sum(min_count=1)
        .pivot_table(index=["ticker", "trade_date"], columns="bucket", values="volume", fill_value=0.0)
        .reindex(columns=range(_INTRADAY_BUCKET_COUNT), fill_value=0.0)
        .reset_index()
    )
    grouped.columns.name = None
    return grouped


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value", "is_filled"])
