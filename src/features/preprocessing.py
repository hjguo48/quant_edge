from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from src.features.fundamental import FUNDAMENTAL_FEATURE_NAMES


def preprocess_features(features_df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    prepared = _prepare_feature_frame(features_df)
    prepared["_raw_missing"] = prepared["feature_value"].isna()

    forward_filled = forward_fill_features(prepared, max_days=90)
    winsorized = winsorize_features(forward_filled, z_threshold=5.0)
    normalized = rank_normalize_features(winsorized, method=method)
    raw_missing = normalized.pop("_raw_missing").astype(bool)
    finalized = add_missing_flags(normalized, raw_missing)
    logger.info("preprocessed {} feature rows into {} rows", len(features_df), len(finalized))
    return finalized


def forward_fill_features(features_df: pd.DataFrame, max_days: int = 90) -> pd.DataFrame:
    frame = _prepare_feature_frame(features_df)
    frame["is_filled"] = frame.get("is_filled", False).astype(bool)

    fundamental_mask = frame["feature_name"].isin(FUNDAMENTAL_FEATURE_NAMES)
    fundamental = frame.loc[fundamental_mask].copy()
    if not fundamental.empty:
        filled_groups = [
            _forward_fill_group(group, max_days=max_days)
            for _, group in fundamental.groupby(["ticker", "feature_name"], sort=False)
        ]
        frame.loc[fundamental.index, ["feature_value", "is_filled"]] = pd.concat(
            filled_groups,
            ignore_index=False,
        )[["feature_value", "is_filled"]]

    return frame.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def winsorize_features(features_df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
    frame = _prepare_feature_frame(features_df)

    def clip_group(series: pd.Series) -> pd.Series:
        non_null = series.dropna()
        if len(non_null) < 2:
            return series
        mean = non_null.mean()
        std = non_null.std(ddof=0)
        if pd.isna(std) or std == 0:
            return series
        lower = mean - z_threshold * std
        upper = mean + z_threshold * std
        return series.clip(lower=lower, upper=upper)

    frame["feature_value"] = frame.groupby(
        ["trade_date", "feature_name"],
        sort=False,
    )["feature_value"].transform(clip_group)
    return frame


def rank_normalize_features(features_df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
    if method != "rank":
        raise ValueError(f"Unsupported normalization method: {method}")

    frame = _prepare_feature_frame(features_df)
    frame["feature_value"] = frame.groupby(
        ["trade_date", "feature_name"],
        sort=False,
    )["feature_value"].transform(_rank_to_unit_interval)
    return frame


def add_missing_flags(features_df: pd.DataFrame, raw_missing_mask: pd.Series) -> pd.DataFrame:
    frame = _prepare_feature_frame(features_df)
    if len(raw_missing_mask) != len(frame):
        raise ValueError("raw_missing_mask must align with features_df.")

    flag_frame = frame[["ticker", "trade_date", "feature_name"]].copy()
    flag_frame["feature_name"] = "is_missing_" + flag_frame["feature_name"].astype(str)
    flag_frame["feature_value"] = raw_missing_mask.astype(float).to_numpy()
    flag_frame["is_filled"] = False

    combined = pd.concat([frame, flag_frame], ignore_index=True)
    return combined.sort_values(["trade_date", "ticker", "feature_name"]).reset_index(drop=True)


def _forward_fill_group(group: pd.DataFrame, max_days: int) -> pd.DataFrame:
    sorted_group = group.sort_values("trade_date").copy()
    original_values = sorted_group["feature_value"].copy()
    last_valid_dates = sorted_group["trade_date"].where(original_values.notna()).ffill()
    filled_values = original_values.ffill()

    age_days = (
        pd.to_datetime(sorted_group["trade_date"]) - pd.to_datetime(last_valid_dates)
    ).dt.days
    valid_fill = last_valid_dates.notna() & (age_days <= max_days)
    sorted_group["feature_value"] = filled_values.where(valid_fill | original_values.notna())
    sorted_group["is_filled"] = original_values.isna() & sorted_group["feature_value"].notna()
    return sorted_group


def _rank_to_unit_interval(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if len(non_null) == 1:
        return pd.Series(np.where(series.notna(), 0.5, np.nan), index=series.index, dtype=float)

    ranked = non_null.rank(method="average")
    normalized = (ranked - 1) / (len(non_null) - 1)
    return normalized.reindex(series.index)


def _prepare_feature_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"ticker", "trade_date", "feature_name", "feature_value"}
    missing_columns = sorted(required_columns - set(features_df.columns))
    if missing_columns:
        raise ValueError(f"features_df is missing required columns: {missing_columns}")

    frame = features_df.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["feature_name"] = frame["feature_name"].astype(str)
    frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
    if "is_filled" not in frame.columns:
        frame["is_filled"] = False
    else:
        frame["is_filled"] = frame["is_filled"].fillna(False).astype(bool)
    return frame
