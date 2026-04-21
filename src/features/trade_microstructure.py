from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EASTERN = ZoneInfo("America/New_York")
REGULAR_SESSION_FEATURES = {
    "trade_imbalance_proxy",
    "large_trade_ratio",
    "late_day_aggressiveness",
    "off_exchange_volume_ratio",
}
OFFHOURS_FEATURES = {"offhours_trade_ratio"}
TRADE_MICROSTRUCTURE_FEATURE_NAMES = (
    "trade_imbalance_proxy",
    "large_trade_ratio",
    "late_day_aggressiveness",
    "offhours_trade_ratio",
    "off_exchange_volume_ratio",
)


def compute_trade_imbalance_proxy(
    trades: pd.DataFrame,
    *,
    condition_allow: set[int] | None = None,
) -> float:
    """Lee-Ready-style tick-rule imbalance over regular-session trades."""
    frame = _regular_session_trades(_prepare_trades(trades))
    frame = _filter_conditions(frame, condition_allow)
    return _tick_rule_imbalance(frame)


def compute_large_trade_ratio(
    trades: pd.DataFrame,
    *,
    size_threshold_dollars: float = 1_000_000,
) -> float:
    """Dollar volume share from regular-session trades above the configured threshold."""
    frame = _regular_session_trades(_prepare_trades(trades))
    if frame.empty:
        return np.nan
    dollars = _trade_dollars(frame)
    total = float(dollars.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.nan
    large = float(dollars.loc[dollars >= float(size_threshold_dollars)].sum())
    return float(np.clip(large / total, 0.0, 1.0))


def compute_late_day_aggressiveness(
    trades: pd.DataFrame,
    *,
    late_day_window_et: tuple[str, str] = ("15:00", "16:00"),
) -> float:
    """Late-day absolute imbalance divided by full regular-session absolute imbalance."""
    regular = _regular_session_trades(_prepare_trades(trades))
    full_imbalance = _tick_rule_imbalance(regular)
    if pd.isna(full_imbalance) or np.isclose(float(full_imbalance), 0.0):
        return np.nan

    late = _time_window(regular, late_day_window_et[0], late_day_window_et[1])
    late_imbalance = _tick_rule_imbalance(late)
    if pd.isna(late_imbalance):
        return np.nan
    return float(np.clip(abs(float(late_imbalance)) / abs(float(full_imbalance)), 0.0, 5.0))


def compute_offhours_trade_ratio(
    trades: pd.DataFrame,
    *,
    pre_window_et: tuple[str, str] = ("04:00", "09:30"),
    post_window_et: tuple[str, str] = ("16:00", "20:00"),
) -> float:
    """Pre/post-market share of full analyzed-day trade volume."""
    frame = _prepare_trades(trades)
    pre = _time_window(frame, pre_window_et[0], pre_window_et[1])
    regular = _time_window(frame, "09:30", "16:00")
    post = _time_window(frame, post_window_et[0], post_window_et[1])
    analyzed = pd.concat([pre, regular, post], ignore_index=True)
    total_volume = float(pd.to_numeric(analyzed.get("size"), errors="coerce").fillna(0.0).sum())
    if not np.isfinite(total_volume) or total_volume <= 0.0:
        return np.nan
    offhours_volume = float(
        pd.to_numeric(pre.get("size"), errors="coerce").fillna(0.0).sum()
        + pd.to_numeric(post.get("size"), errors="coerce").fillna(0.0).sum(),
    )
    return float(np.clip(offhours_volume / total_volume, 0.0, 1.0))


def compute_off_exchange_volume_ratio(
    trades: pd.DataFrame,
    *,
    trf_exchange_codes: set[int],
) -> float:
    """Dollar volume share from regular-session TRF/off-exchange trades."""
    frame = _regular_session_trades(_prepare_trades(trades))
    if frame.empty:
        return np.nan
    dollars = _trade_dollars(frame)
    total = float(dollars.sum())
    if not np.isfinite(total) or total <= 0.0:
        return np.nan

    exchanges = pd.to_numeric(frame["exchange"], errors="coerce")
    trf_id_present = frame["trf_id"].notna() if "trf_id" in frame.columns else pd.Series(False, index=frame.index)
    trf_ts_present = (
        frame["trf_timestamp"].notna()
        if "trf_timestamp" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    off_exchange = exchanges.isin(set(trf_exchange_codes)) | trf_id_present | trf_ts_present
    off_exchange_dollars = float(dollars.loc[off_exchange].sum())
    return float(np.clip(off_exchange_dollars / total, 0.0, 1.0))


def compute_knowledge_time(trading_date: date, feature_name: str) -> datetime:
    if feature_name in OFFHOURS_FEATURES:
        local_time = time(20, 15)
    elif feature_name in REGULAR_SESSION_FEATURES:
        local_time = time(16, 15)
    else:
        raise ValueError(f"unknown trade microstructure feature: {feature_name}")
    return datetime.combine(trading_date, local_time, tzinfo=EASTERN).astimezone(timezone.utc)


def compute_trade_microstructure_features(*args, **kwargs) -> pd.DataFrame:
    raise NotImplementedError("Trade microstructure batch feature construction is implemented in Week 4 Task 8.")


def _prepare_trades(trades: pd.DataFrame) -> pd.DataFrame:
    columns = ["sip_timestamp", "price", "size", "exchange", "trf_id", "trf_timestamp", "conditions"]
    if trades.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = trades.copy()
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan

    frame["sip_timestamp"] = pd.to_datetime(frame["sip_timestamp"], utc=True, errors="coerce")
    frame["sip_timestamp_et"] = frame["sip_timestamp"].dt.tz_convert(EASTERN)
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["size"] = pd.to_numeric(frame["size"], errors="coerce")
    frame["exchange"] = pd.to_numeric(frame["exchange"], errors="coerce")
    frame["trf_timestamp"] = pd.to_datetime(frame["trf_timestamp"], utc=True, errors="coerce")
    frame.sort_values("sip_timestamp", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _regular_session_trades(frame: pd.DataFrame) -> pd.DataFrame:
    return _time_window(frame, "09:30", "16:00")


def _time_window(frame: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    start = _parse_hhmm(start_hhmm)
    end = _parse_hhmm(end_hhmm)
    minutes = frame["sip_timestamp_et"].dt.hour * 60 + frame["sip_timestamp_et"].dt.minute
    mask = (minutes >= start) & (minutes < end)
    return frame.loc[mask].copy()


def _parse_hhmm(value: str) -> int:
    hour_raw, minute_raw = value.split(":", 1)
    return int(hour_raw) * 60 + int(minute_raw)


def _filter_conditions(frame: pd.DataFrame, condition_allow: set[int] | None) -> pd.DataFrame:
    if not condition_allow:
        return frame

    allow = {int(condition) for condition in condition_allow}

    def allowed(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        if isinstance(value, (list, tuple, set)):
            return {int(condition) for condition in value}.issubset(allow)
        return int(value) in allow

    return frame.loc[frame["conditions"].map(allowed)].copy()


def _tick_rule_imbalance(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan
    prices = pd.to_numeric(frame["price"], errors="coerce")
    sizes = pd.to_numeric(frame["size"], errors="coerce").fillna(0.0)
    valid = prices.notna() & sizes.gt(0.0)
    prices = prices.loc[valid].reset_index(drop=True)
    sizes = sizes.loc[valid].reset_index(drop=True)
    if prices.empty:
        return np.nan

    signs: list[int] = []
    last_sign = 0
    previous_price: float | None = None
    for price in prices:
        price_value = float(price)
        if previous_price is None:
            sign = 0
        elif price_value > previous_price:
            sign = 1
        elif price_value < previous_price:
            sign = -1
        else:
            sign = last_sign
        signs.append(sign)
        if sign != 0:
            last_sign = sign
        previous_price = price_value

    total_size = float(sizes.sum())
    if total_size <= 0.0:
        return np.nan
    signed_size = float((pd.Series(signs, dtype=float) * sizes).sum())
    return float(np.clip(signed_size / total_size, -1.0, 1.0))


def _trade_dollars(frame: pd.DataFrame) -> pd.Series:
    prices = pd.to_numeric(frame["price"], errors="coerce").fillna(0.0)
    sizes = pd.to_numeric(frame["size"], errors="coerce").fillna(0.0)
    return prices * sizes
