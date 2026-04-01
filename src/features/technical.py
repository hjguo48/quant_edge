from __future__ import annotations

from datetime import date, datetime, time, timezone
from math import sqrt

import numpy as np
import pandas as pd
import sqlalchemy as sa
from loguru import logger

from src.data.db.models import FundamentalsPIT
from src.data.db.session import get_session_factory

TECHNICAL_FEATURE_NAMES = (
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "ret_60d",
    "high_52w_ratio",
    "low_52w_ratio",
    "momentum_rank_20d",
    "momentum_rank_60d",
    "vol_20d",
    "vol_60d",
    "atr_14",
    "gk_vol",
    "vol_rank",
    "vol_change",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "obv_slope",
    "vwap_deviation",
    "amihud",
    "turnover_rate",
    "rsi_14",
    "rsi_28",
    "macd_signal",
    "macd_histogram",
    "bb_width",
    "bb_position",
    "adx_14",
    "stoch_k",
    "stoch_d",
    "cci_20",
)

_ANNUALIZATION_FACTOR = sqrt(252.0)
_PIT_SHARES_METRIC_NAME = "weighted_average_shares_outstanding"


def compute_technical_features(prices_df: pd.DataFrame) -> pd.DataFrame:
    prepared = _prepare_prices(prices_df)
    if prepared.empty:
        return _empty_feature_frame()

    prepared = _attach_pit_shares_outstanding(prepared)
    feature_frames: list[pd.DataFrame] = []

    for ticker, group in prepared.groupby("ticker", sort=False):
        feature_frames.append(_compute_ticker_features(group.copy()))

    combined = pd.concat(feature_frames, ignore_index=True)
    combined["momentum_rank_20d"] = _cross_sectional_rank(combined, "ret_20d")
    combined["momentum_rank_60d"] = _cross_sectional_rank(combined, "ret_60d")
    combined["vol_rank"] = _cross_sectional_rank(combined, "vol_20d")

    long_frame = _to_long_feature_frame(combined, TECHNICAL_FEATURE_NAMES)
    logger.info(
        "computed {} technical feature rows across {} tickers",
        len(long_frame),
        combined["ticker"].nunique(),
    )
    return long_frame


def _prepare_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {
        "ticker",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    }
    missing_columns = sorted(required_columns - set(prices_df.columns))
    if missing_columns:
        raise ValueError(f"prices_df is missing required columns: {missing_columns}")

    prepared = prices_df.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"]).dt.date
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared.sort_values(["ticker", "trade_date"], inplace=True)
    prepared.drop_duplicates(subset=["ticker", "trade_date"], keep="last", inplace=True)
    return prepared.reset_index(drop=True)


def _attach_pit_shares_outstanding(prices_df: pd.DataFrame) -> pd.DataFrame:
    shares_by_date = _load_pit_shares_outstanding(
        prices_df[["ticker", "trade_date"]].drop_duplicates().reset_index(drop=True),
    )
    if shares_by_date.empty:
        frame = prices_df.copy()
        frame["pit_shares_outstanding"] = np.nan
        return frame
    return prices_df.merge(shares_by_date, on=["ticker", "trade_date"], how="left")


def _load_pit_shares_outstanding(date_pairs: pd.DataFrame) -> pd.DataFrame:
    if date_pairs.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "pit_shares_outstanding"])

    normalized_pairs = date_pairs.copy()
    normalized_pairs["ticker"] = normalized_pairs["ticker"].astype(str).str.upper()
    normalized_pairs["trade_date"] = pd.to_datetime(normalized_pairs["trade_date"]).dt.date
    normalized_pairs.sort_values(["ticker", "trade_date"], inplace=True)

    normalized_tickers = tuple(normalized_pairs["ticker"].drop_duplicates())
    max_trade_date = normalized_pairs["trade_date"].max()
    as_of_cutoff = datetime.combine(max_trade_date, time.max, tzinfo=timezone.utc)

    statement = (
        sa.select(
            FundamentalsPIT.id,
            FundamentalsPIT.ticker,
            FundamentalsPIT.fiscal_period,
            FundamentalsPIT.event_time,
            FundamentalsPIT.knowledge_time,
            FundamentalsPIT.metric_value,
        )
        .where(
            FundamentalsPIT.ticker.in_(normalized_tickers),
            FundamentalsPIT.metric_name == _PIT_SHARES_METRIC_NAME,
            FundamentalsPIT.knowledge_time <= as_of_cutoff,
        )
        .order_by(
            FundamentalsPIT.ticker,
            FundamentalsPIT.knowledge_time,
            FundamentalsPIT.event_time,
            FundamentalsPIT.id,
        )
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    pit_rows = pd.DataFrame(rows)
    if pit_rows.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "pit_shares_outstanding"])

    pit_rows["event_time"] = pd.to_datetime(pit_rows["event_time"]).dt.date
    pit_rows["knowledge_time"] = pd.to_datetime(pit_rows["knowledge_time"], utc=True)
    pit_rows["metric_value"] = pd.to_numeric(pit_rows["metric_value"], errors="coerce")
    pit_rows.dropna(subset=["metric_value", "knowledge_time"], inplace=True)
    if pit_rows.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "pit_shares_outstanding"])

    shares_frames: list[pd.DataFrame] = []
    for ticker, trade_dates in normalized_pairs.groupby("ticker", sort=False):
        ticker_rows = pit_rows.loc[pit_rows["ticker"] == ticker].copy()
        if ticker_rows.empty:
            continue

        share_steps = _build_pit_share_steps(ticker_rows)
        if share_steps.empty:
            continue
        share_steps.sort_values("knowledge_time", inplace=True)

        trade_frame = trade_dates[["trade_date"]].drop_duplicates().sort_values("trade_date").copy()
        trade_frame["trade_as_of"] = (
            pd.to_datetime(trade_frame["trade_date"]).dt.tz_localize(timezone.utc)
            + pd.Timedelta(days=1)
            - pd.Timedelta(microseconds=1)
        )
        mapped = pd.merge_asof(
            trade_frame,
            share_steps,
            left_on="trade_as_of",
            right_on="knowledge_time",
            direction="backward",
        )
        mapped["ticker"] = ticker
        shares_frames.append(mapped[["ticker", "trade_date", "pit_shares_outstanding"]])

    if not shares_frames:
        return pd.DataFrame(columns=["ticker", "trade_date", "pit_shares_outstanding"])

    return pd.concat(shares_frames, ignore_index=True)


def _build_pit_share_steps(pit_rows: pd.DataFrame) -> pd.DataFrame:
    if pit_rows.empty:
        return pd.DataFrame(columns=["knowledge_time", "pit_shares_outstanding"])

    latest_by_period: dict[str, pd.Series] = {}
    checkpoints: list[dict[str, object]] = []
    ordered = pit_rows.sort_values(["knowledge_time", "event_time", "id"]).reset_index(drop=True)

    for knowledge_time, knowledge_group in ordered.groupby("knowledge_time", sort=True):
        for row in knowledge_group.to_dict("records"):
            latest_by_period[str(row["fiscal_period"])] = row

        latest_row = max(
            latest_by_period.values(),
            key=lambda row: (
                row["event_time"],
                row["fiscal_period"],
                row["knowledge_time"],
                row["id"],
            ),
        )
        checkpoints.append(
            {
                "knowledge_time": knowledge_time,
                "pit_shares_outstanding": float(latest_row["metric_value"]),
            },
        )

    return pd.DataFrame(checkpoints)


def _compute_ticker_features(group: pd.DataFrame) -> pd.DataFrame:
    adj_close = group["adj_close"].fillna(group["close"])
    returns_1d = adj_close.pct_change()
    group["ret_5d"] = adj_close.pct_change(5)
    group["ret_10d"] = adj_close.pct_change(10)
    group["ret_20d"] = adj_close.pct_change(20)
    group["ret_60d"] = adj_close.pct_change(60)
    group["high_52w_ratio"] = group["close"] / group["close"].rolling(252, min_periods=20).max()
    group["low_52w_ratio"] = group["close"] / group["close"].rolling(252, min_periods=20).min()

    group["vol_20d"] = returns_1d.rolling(20, min_periods=20).std(ddof=0) * _ANNUALIZATION_FACTOR
    group["vol_60d"] = returns_1d.rolling(60, min_periods=60).std(ddof=0) * _ANNUALIZATION_FACTOR
    group["atr_14"] = _average_true_range(group, 14) / group["close"].replace(0, np.nan)
    group["gk_vol"] = _garman_klass_volatility(group, 20)
    group["vol_change"] = group["vol_20d"] / group["vol_60d"].replace(0, np.nan)

    volume = group["volume"]
    group["volume_ratio_5d"] = volume.rolling(5, min_periods=5).mean() / volume.rolling(20, min_periods=20).mean()
    group["volume_ratio_20d"] = volume.rolling(20, min_periods=20).mean() / volume.rolling(60, min_periods=60).mean()
    group["obv_slope"] = _obv_slope(group)
    typical_price = (group["high"] + group["low"] + group["close"]) / 3.0
    group["vwap_deviation"] = (group["close"] - typical_price) / typical_price.replace(0, np.nan)
    dollar_volume = group["close"].abs() * volume
    amihud_daily = returns_1d.abs() / dollar_volume.replace(0, np.nan)
    group["amihud"] = amihud_daily.rolling(20, min_periods=20).mean()
    shares_outstanding = pd.to_numeric(group.get("pit_shares_outstanding"), errors="coerce")
    group["turnover_rate"] = volume / shares_outstanding.replace(0, np.nan)

    group["rsi_14"] = _rsi(adj_close, 14)
    group["rsi_28"] = _rsi(adj_close, 28)
    macd_line, macd_signal = _macd(adj_close)
    group["macd_signal"] = macd_signal
    group["macd_histogram"] = macd_line - macd_signal
    group["bb_width"], group["bb_position"] = _bollinger_bands(group["close"], 20, 2.0)
    group["adx_14"] = _adx(group, 14)
    group["stoch_k"], group["stoch_d"] = _stochastic(group, 14, 3)
    group["cci_20"] = _cci(group, 20)

    return group


def _average_true_range(group: pd.DataFrame, window: int) -> pd.Series:
    prev_close = group["close"].shift(1)
    true_range = pd.concat(
        [
            group["high"] - group["low"],
            (group["high"] - prev_close).abs(),
            (group["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _garman_klass_volatility(group: pd.DataFrame, window: int) -> pd.Series:
    safe_open = group["open"].replace(0, np.nan)
    safe_high = group["high"].replace(0, np.nan)
    safe_low = group["low"].replace(0, np.nan)
    safe_close = group["close"].replace(0, np.nan)
    log_hl = np.log(safe_high / safe_low)
    log_co = np.log(safe_close / safe_open)
    gk_variance = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
    rolling_variance = gk_variance.rolling(window, min_periods=window).mean().clip(lower=0)
    return np.sqrt(rolling_variance * 252)


def _obv_slope(group: pd.DataFrame, window: int = 20) -> pd.Series:
    direction = np.sign(group["close"].diff()).fillna(0.0)
    obv = (direction * group["volume"].fillna(0.0)).cumsum()
    return obv.rolling(window, min_periods=window).apply(_normalized_slope, raw=True)


def _normalized_slope(values: np.ndarray) -> float:
    if len(values) == 0 or np.isnan(values).all():
        return np.nan

    x = np.arange(len(values), dtype=float)
    y = values.astype(float)
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 2:
        return np.nan

    x = x[valid_mask]
    y = y[valid_mask]
    x_mean = x.mean()
    y_mean = y.mean()
    denominator = np.sum((x - x_mean) ** 2)
    if denominator == 0:
        return np.nan

    slope = np.sum((x - x_mean) * (y - y_mean)) / denominator
    scale = np.nanmean(np.abs(y))
    if scale == 0 or np.isnan(scale):
        return 0.0
    return float(slope / scale)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd_line, signal_line


def _bollinger_bands(series: pd.Series, window: int, num_std: float) -> tuple[pd.Series, pd.Series]:
    middle = series.rolling(window, min_periods=window).mean()
    rolling_std = series.rolling(window, min_periods=window).std(ddof=0)
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    width = (upper - lower) / middle.replace(0, np.nan)
    position = (series - lower) / (upper - lower).replace(0, np.nan)
    return width, position


def _adx(group: pd.DataFrame, window: int) -> pd.Series:
    up_move = group["high"].diff()
    down_move = -group["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = _average_true_range(group, window)
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, adjust=False, min_periods=window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _stochastic(group: pd.DataFrame, window: int, smooth_window: int) -> tuple[pd.Series, pd.Series]:
    rolling_low = group["low"].rolling(window, min_periods=window).min()
    rolling_high = group["high"].rolling(window, min_periods=window).max()
    stoch_k = 100 * (group["close"] - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    stoch_d = stoch_k.rolling(smooth_window, min_periods=smooth_window).mean()
    return stoch_k, stoch_d


def _cci(group: pd.DataFrame, window: int) -> pd.Series:
    typical_price = (group["high"] + group["low"] + group["close"]) / 3.0
    sma = typical_price.rolling(window, min_periods=window).mean()
    mean_abs_deviation = typical_price.rolling(window, min_periods=window).apply(
        lambda values: float(np.mean(np.abs(values - np.mean(values)))),
        raw=True,
    )
    return (typical_price - sma) / (0.015 * mean_abs_deviation.replace(0, np.nan))


def _cross_sectional_rank(frame: pd.DataFrame, feature_name: str) -> pd.Series:
    ranked = frame.groupby("trade_date")[feature_name].transform(_rank_to_unit_interval)
    return ranked


def _rank_to_unit_interval(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if len(non_null) == 1:
        return pd.Series(np.where(series.notna(), 0.5, np.nan), index=series.index, dtype=float)

    ranked = non_null.rank(method="average")
    normalized = (ranked - 1) / (len(non_null) - 1)
    return normalized.reindex(series.index)


def _to_long_feature_frame(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    long_frame = frame.melt(
        id_vars=["ticker", "trade_date"],
        value_vars=list(feature_columns),
        var_name="feature_name",
        value_name="feature_value",
    )
    return long_frame.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
