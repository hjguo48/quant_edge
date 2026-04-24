from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

EASTERN = ZoneInfo("America/New_York")


def compute_trade_imbalance_proxy_minute(minute_df: pd.DataFrame) -> float:
    """Minute-bar tick-rule imbalance over regular-session bars."""
    frame = _regular_session(_prepare_minute_bars(minute_df))
    return _minute_imbalance(frame)


def compute_late_day_aggressiveness_minute(
    minute_df: pd.DataFrame,
    late_window_et: tuple[str, str] = ("15:00", "16:00"),
) -> float:
    """Late-window absolute minute imbalance divided by full-session absolute imbalance."""
    regular = _regular_session(_prepare_minute_bars(minute_df))
    full_imbalance = _minute_imbalance(regular)
    if pd.isna(full_imbalance) or np.isclose(float(full_imbalance), 0.0):
        return np.nan

    late = _time_window(regular, late_window_et[0], late_window_et[1])
    late_imbalance = _minute_imbalance(late)
    if pd.isna(late_imbalance):
        return np.nan
    return float(np.clip(abs(float(late_imbalance)) / abs(float(full_imbalance)), 0.0, 5.0))


def compute_offhours_trade_ratio_minute(
    minute_df: pd.DataFrame,
    pre_et: tuple[str, str] = ("04:00", "09:30"),
    post_et: tuple[str, str] = ("16:00", "20:00"),
) -> float:
    """Pre/post-market minute volume divided by analyzed full-day volume."""
    frame = _prepare_minute_bars(minute_df)
    pre = _time_window(frame, pre_et[0], pre_et[1])
    regular = _time_window(frame, "09:30", "16:00")
    post = _time_window(frame, post_et[0], post_et[1])
    analyzed = pd.concat([pre, regular, post], ignore_index=True)
    total_volume = float(pd.to_numeric(analyzed.get("volume"), errors="coerce").fillna(0.0).sum())
    if not np.isfinite(total_volume) or total_volume <= 0.0:
        return np.nan

    offhours_volume = float(
        pd.to_numeric(pre.get("volume"), errors="coerce").fillna(0.0).sum()
        + pd.to_numeric(post.get("volume"), errors="coerce").fillna(0.0).sum(),
    )
    return float(np.clip(offhours_volume / total_volume, 0.0, 1.0))


def _prepare_minute_bars(minute_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["minute_ts", "close", "volume"]
    if minute_df.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        frame = minute_df.copy()
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan

    timestamps = pd.to_datetime(frame["minute_ts"], errors="coerce")
    if timestamps.empty:
        timestamps = pd.Series(pd.to_datetime([], utc=True), index=frame.index)
    if not timestamps.empty and timestamps.dt.tz is None:
        raise ValueError("minute_ts must be timezone-aware")

    frame["minute_ts"] = timestamps.dt.tz_convert("UTC")
    frame["minute_ts_et"] = frame["minute_ts"].dt.tz_convert(EASTERN)
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
    frame.sort_values("minute_ts", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def _regular_session(frame: pd.DataFrame) -> pd.DataFrame:
    return _time_window(frame, "09:30", "16:00")


def _time_window(frame: pd.DataFrame, start_hhmm: str, end_hhmm: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    start = _parse_hhmm(start_hhmm)
    end = _parse_hhmm(end_hhmm)
    minutes = frame["minute_ts_et"].dt.hour * 60 + frame["minute_ts_et"].dt.minute
    return frame.loc[(minutes >= start) & (minutes < end)].copy()


def _parse_hhmm(value: str) -> int:
    hour_raw, minute_raw = value.split(":", 1)
    return int(hour_raw) * 60 + int(minute_raw)


def _minute_imbalance(frame: pd.DataFrame) -> float:
    if frame.empty:
        return np.nan

    closes = pd.to_numeric(frame["close"], errors="coerce")
    volumes = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    valid = closes.notna() & volumes.gt(0.0)
    closes = closes.loc[valid].reset_index(drop=True)
    volumes = volumes.loc[valid].reset_index(drop=True)
    if closes.empty:
        return np.nan

    signs = np.sign(closes.diff()).fillna(0.0)
    total_volume = float(volumes.sum())
    if total_volume <= 0.0:
        return np.nan
    signed_volume = float((signs * volumes).sum())
    return float(np.clip(signed_volume / total_volume, -1.0, 1.0))
