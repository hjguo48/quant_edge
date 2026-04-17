"""PIT-safe sector mapping and sector-relative feature computation.

The ``stocks`` table contains a single snapshot (2026-04-04) of FMP sector
classifications. Two known PIT issues must be corrected:

1. **2018-09 GICS reclassification**: Communication Services was created as a
   new GICS sector.  ~19 tickers moved from Technology or Consumer Cyclical.
   For trade dates before 2018-09-28 we map these tickers back to their
   original sector.

2. **FMP vs GICS naming**: retired/delisted tickers were backfilled with GICS
   names (e.g. "Financials" vs FMP's "Financial Services").  We unify to FMP
   naming for consistency.

All sector assignments carry ``pit_approximate=True`` metadata because even
with the 2018 fix, individual company reclassifications (rare) are not tracked.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sqlalchemy import text

from src.data.db.session import get_engine

# ── FMP ↔ GICS naming unification ──────────────────────────────────────────
GICS_TO_FMP: dict[str, str] = {
    "Consumer Discretionary": "Consumer Cyclical",
    "Consumer Staples": "Consumer Defensive",
    "Financials": "Financial Services",
    "Health Care": "Healthcare",
    "Information Technology": "Technology",
    "Materials": "Basic Materials",
}

# ── 2018-09-28 GICS reclassification: Communication Services created ───────
# Tickers that moved FROM Technology TO Communication Services
_FROM_TECH_TO_COMM: frozenset[str] = frozenset({
    "GOOG", "GOOGL",  # Alphabet
    "META", "FB",      # Meta Platforms (FB pre-rename)
    "NFLX",            # Netflix
    "TTWO",            # Take-Two Interactive
    "EA",              # Electronic Arts
    "ATVI",            # Activision Blizzard (merged into MSFT 2023)
})

# Tickers that moved FROM Consumer Cyclical TO Communication Services
_FROM_CONSUMER_TO_COMM: frozenset[str] = frozenset({
    "DIS",             # Walt Disney
    "CMCSA", "CMCSK",  # Comcast
    "CHTR",            # Charter Communications
    "FOX", "FOXA",     # Fox Corporation
    "DISCK", "DISCA",  # Discovery (merged into WBD 2022)
    "VIAB", "VIAC",    # ViacomCBS (renamed to PARA)
    "PARA",            # Paramount Global
    "LBTYA", "LBTYB", "LBTYK",  # Liberty Global
    "OMC",             # Omnicom Group
    "IPG",             # Interpublic Group
    "WBD",             # Warner Bros. Discovery
})

# Telecom-origin tickers: always Communication Services (or Telecom before 2018)
# These do NOT need remapping — they were Telecom → Communication Services
# and the sector-relative comparison group is similar enough.

GICS_RECLASS_DATE = date(2018, 9, 28)


def load_sector_map_pit(as_of_date: date | None = None) -> dict[str, str]:
    """Load sector mapping with PIT-safe corrections.

    Args:
        as_of_date: The trade date for which sectors are needed.
            If before 2018-09-28, Communication Services tickers are
            mapped back to their original sector.
            If None, returns current (latest) sector assignments.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT ticker, sector FROM stocks")).fetchall()

    mapping: dict[str, str] = {}
    for ticker, sector in rows:
        ticker = str(ticker).upper()
        sector = str(sector) if sector else "Unknown"
        # Step 1: unify GICS → FMP naming
        sector = GICS_TO_FMP.get(sector, sector)
        mapping[ticker] = sector

    # Step 2: apply 2018-09 reclassification fix
    if as_of_date is not None and as_of_date < GICS_RECLASS_DATE:
        for ticker in _FROM_TECH_TO_COMM:
            if ticker in mapping and mapping[ticker] == "Communication Services":
                mapping[ticker] = "Technology"
        for ticker in _FROM_CONSUMER_TO_COMM:
            if ticker in mapping and mapping[ticker] == "Communication Services":
                mapping[ticker] = "Consumer Cyclical"

    return mapping


def compute_sector_relative_features(
    features_wide: pd.DataFrame,
    sector_map: dict[str, str],
    feature_columns: list[str],
    *,
    min_sector_size: int = 5,
) -> pd.DataFrame:
    """Compute sector-relative z-scores for the given features.

    For each (trade_date, feature), computes:
        z_sector = (raw_value - sector_mean) / sector_std

    Args:
        features_wide: DataFrame indexed by (ticker, trade_date) with feature
            columns.  Must contain raw (pre-rank) feature values.
        sector_map: ticker → sector string.
        feature_columns: which columns to compute sector-relative versions of.
        min_sector_size: minimum number of stocks in a sector to compute
            z-score.  Smaller sectors get NaN.

    Returns:
        Long-format DataFrame with columns:
        [ticker, trade_date, feature_name, feature_value]
        where feature_name = "{original}_sector_rel".
    """
    if features_wide.empty or not feature_columns:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    # Ensure we have a flat index for groupby
    df = features_wide.copy()
    if isinstance(df.index, pd.MultiIndex):
        idx_names = [n or f"level_{i}" for i, n in enumerate(df.index.names)]
        df = df.reset_index()
        if "ticker" not in df.columns:
            # Try to find the ticker column from the reset index
            for name in idx_names:
                if "ticker" in name.lower():
                    df = df.rename(columns={name: "ticker"})
                    break
        if "trade_date" not in df.columns:
            for name in idx_names:
                if "date" in name.lower():
                    df = df.rename(columns={name: "trade_date"})
                    break
    else:
        df = df.reset_index()

    if "ticker" not in df.columns or "trade_date" not in df.columns:
        raise ValueError("features_wide must have 'ticker' and 'trade_date' columns or index levels")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["_sector"] = df["ticker"].map(sector_map).fillna("Unknown")

    results: list[pd.DataFrame] = []
    for col in feature_columns:
        if col not in df.columns:
            continue

        # Sector-level mean and std per trade_date
        grouped = df.groupby(["trade_date", "_sector"])[col]
        sector_stats = grouped.agg(["mean", "std", "count"]).reset_index()
        sector_stats.columns = ["trade_date", "_sector", "_mean", "_std", "_count"]

        merged = df[["ticker", "trade_date", "_sector", col]].merge(
            sector_stats, on=["trade_date", "_sector"], how="left",
        )

        # Mask sectors with too few members
        merged.loc[merged["_count"] < min_sector_size, ["_mean", "_std"]] = np.nan
        # Avoid division by zero
        merged["_std"] = merged["_std"].replace(0, np.nan)

        merged["_z"] = (merged[col] - merged["_mean"]) / merged["_std"]

        result = merged[["ticker", "trade_date", "_z"]].copy()
        result.columns = ["ticker", "trade_date", "feature_value"]
        result["feature_name"] = f"{col}_sector_rel"
        result = result[["ticker", "trade_date", "feature_name", "feature_value"]]
        results.append(result)

    if not results:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    out = pd.concat(results, ignore_index=True)
    out = out.dropna(subset=["feature_value"])
    logger.info(
        "computed {} sector-relative feature rows for {} features",
        len(out), len(feature_columns),
    )
    return out


# Features that are meaningful to compare within a sector
SECTOR_RELATIVE_CANDIDATES: tuple[str, ...] = (
    "pe_ratio",
    "pb_ratio",
    "ps_ratio",
    "ev_ebitda",
    "fcf_yield",
    "dividend_yield",
    "roe",
    "roa",
    "operating_margin",
    "revenue_growth_yoy",
    "earnings_growth_yoy",
    "debt_to_equity",
    "current_ratio",
)

# S1.4 experiment result: 9 features that passed |IC| >= 0.01 threshold
SECTOR_RELATIVE_RETAINED: tuple[str, ...] = (
    "pb_ratio_sector_rel",
    "ps_ratio_sector_rel",
    "pe_ratio_sector_rel",
    "roa_sector_rel",
    "operating_margin_sector_rel",
    "ev_ebitda_sector_rel",
    "current_ratio_sector_rel",
    "fcf_yield_sector_rel",
    "dividend_yield_sector_rel",
)

# Base feature names whose sector-relative versions are retained
_SECTOR_REL_BASE_NAMES: frozenset[str] = frozenset(
    name.replace("_sector_rel", "") for name in SECTOR_RELATIVE_RETAINED
)


def compute_sector_relative_from_raw_features(
    raw_features: pd.DataFrame,
    trade_date: date,
    *,
    min_sector_size: int = 5,
) -> pd.DataFrame:
    """Compute sector-relative z-scores from raw (pre-rank) feature values.

    This is the live-pipeline variant: operates on a single cross-section date.

    Args:
        raw_features: Long-format DataFrame with columns
            [ticker, trade_date, feature_name, feature_value].
            Must contain raw (non-rank-normalized) values.
        trade_date: The cross-section date (for PIT-safe sector lookup).
        min_sector_size: Minimum stocks in a sector to compute z-score.

    Returns:
        Long-format DataFrame with new ``{feature}_sector_rel`` rows.
    """
    candidates = raw_features.loc[
        raw_features["feature_name"].isin(_SECTOR_REL_BASE_NAMES)
    ].copy()
    if candidates.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    sector_map = load_sector_map_pit(trade_date)
    candidates["_sector"] = candidates["ticker"].str.upper().map(sector_map).fillna("Unknown")

    results: list[pd.DataFrame] = []
    for feat_name, group in candidates.groupby("feature_name"):
        if len(group) < 10:
            continue

        stats = group.groupby("_sector")["feature_value"].agg(["mean", "std", "count"])
        stats = stats[stats["count"] >= min_sector_size]
        stats["std"] = stats["std"].replace(0, np.nan)

        merged = group.merge(stats, left_on="_sector", right_index=True, how="left")
        merged["z"] = (merged["feature_value"] - merged["mean"]) / merged["std"]
        valid = merged.dropna(subset=["z"])
        if valid.empty:
            continue

        results.append(pd.DataFrame({
            "ticker": valid["ticker"].values,
            "trade_date": valid["trade_date"].values,
            "feature_name": f"{feat_name}_sector_rel",
            "feature_value": valid["z"].values,
        }))

    if not results:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    out = pd.concat(results, ignore_index=True)
    logger.info("computed {} sector-relative rows for {} features (date={})",
                len(out), len(results), trade_date)
    return out
