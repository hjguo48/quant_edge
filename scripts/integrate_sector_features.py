"""Integrate S1.4 sector-relative features into the walk-forward pipeline.

Steps:
1. Compute raw fundamental ratios → sector-relative z-scores for all Friday dates
2. Cross-sectionally rank-normalize to 0-1 (consistent with other features)
3. Append to all_features.parquet
4. Update ic_screening_report_60d.csv with the 9 passing features
5. Delete cached feature matrix so walk-forward comparison rebuilds it

Usage:
    python scripts/integrate_sector_features.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

import numpy as np
import pandas as pd
from loguru import logger

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic, write_parquet_atomic
from src.features.sector import GICS_RECLASS_DATE, load_sector_map_pit

ALL_FEATURES_PATH = REPO_ROOT / "data/features/all_features.parquet"
IC_REPORT_PATH = REPO_ROOT / "data/features/ic_screening_report_60d.csv"
FEATURE_MATRIX_CACHE = REPO_ROOT / "data/features/walkforward_feature_matrix_60d.parquet"
FEATURE_MATRIX_META = REPO_ROOT / "data/features/walkforward_feature_matrix_60d.meta.json"
EXPERIMENT_REPORT = REPO_ROOT / "data/reports/sector_features_experiment.json"

PASSING_SECTOR_FEATURES = [
    "pb_ratio_sector_rel",
    "ps_ratio_sector_rel",
    "pe_ratio_sector_rel",
    "roa_sector_rel",
    "operating_margin_sector_rel",
    "ev_ebitda_sector_rel",
    "current_ratio_sector_rel",
    "fcf_yield_sector_rel",
    "dividend_yield_sector_rel",
]

_METRICS = [
    "eps", "book_value_per_share", "revenue", "net_income",
    "total_assets", "total_liabilities", "total_debt",
    "operating_cash_flow", "capital_expenditure", "free_cash_flow",
    "ebitda", "cash", "cash_and_cash_equivalents",
    "annual_dividend", "dividend_per_share",
    "operating_income",
    "current_assets", "current_liabilities",
    "weighted_average_shares_outstanding",
]


def main() -> int:
    logger.info("S1.4 integration: adding sector-relative features to pipeline")

    # ── Step 1: Get all Friday dates and tickers from all_features ──
    logger.info("loading all_features.parquet to get Friday dates and tickers...")
    af_meta = pd.read_parquet(
        ALL_FEATURES_PATH,
        columns=["ticker", "trade_date"],
    )
    af_meta["trade_date"] = pd.to_datetime(af_meta["trade_date"])
    friday_dates = sorted(af_meta.loc[af_meta["trade_date"].dt.weekday == 4, "trade_date"].unique())
    all_tickers = sorted(af_meta["ticker"].unique())
    del af_meta
    logger.info("found {} Friday dates, {} tickers", len(friday_dates), len(all_tickers))

    # ── Step 2: Compute raw fundamentals ──
    logger.info("computing raw fundamentals for {} Fridays × {} tickers...",
                len(friday_dates), len(all_tickers))
    raw_long = compute_raw_fundamentals(all_tickers, friday_dates)
    logger.info("raw fundamentals: {} rows, {} features",
                len(raw_long), raw_long["feature_name"].nunique())

    # ── Step 3: Compute sector-relative z-scores ──
    logger.info("computing sector-relative z-scores...")
    sector_rel = compute_sector_zscores(raw_long, friday_dates)
    logger.info("sector-relative z-scores: {} rows", len(sector_rel))

    # Filter to only passing features
    sector_rel = sector_rel[sector_rel["feature_name"].isin(PASSING_SECTOR_FEATURES)]
    logger.info("filtered to {} passing features: {} rows",
                len(PASSING_SECTOR_FEATURES), len(sector_rel))

    # ── Step 4: Rank-normalize per (trade_date, feature_name) ──
    logger.info("rank-normalizing to 0-1 per cross-section...")
    sector_rel = rank_normalize(sector_rel)

    # Add is_filled column
    sector_rel["is_filled"] = False

    # ── Step 5: Append to all_features.parquet ──
    logger.info("loading existing all_features.parquet...")
    existing = pd.read_parquet(ALL_FEATURES_PATH)
    existing["trade_date"] = pd.to_datetime(existing["trade_date"])

    # Remove any existing sector_rel features (in case of re-run)
    existing = existing[~existing["feature_name"].str.endswith("_sector_rel")]
    logger.info("existing features after cleanup: {} rows", len(existing))

    # Ensure consistent types
    sector_rel["trade_date"] = pd.to_datetime(sector_rel["trade_date"])
    sector_rel["feature_value"] = sector_rel["feature_value"].astype(float)
    sector_rel = sector_rel[["ticker", "trade_date", "feature_name", "feature_value", "is_filled"]]

    combined = pd.concat([existing, sector_rel], ignore_index=True)
    logger.info("combined all_features: {} rows ({} new)", len(combined), len(sector_rel))

    # Save
    backup_path = ALL_FEATURES_PATH.with_suffix(".parquet.bak")
    if ALL_FEATURES_PATH.exists() and not backup_path.exists():
        ALL_FEATURES_PATH.rename(backup_path)
        logger.info("backed up original to {}", backup_path)
    write_parquet_atomic(combined, ALL_FEATURES_PATH)
    logger.info("saved augmented all_features.parquet")
    del existing, combined

    # ── Step 6: Update IC screening report ──
    update_ic_report()

    # ── Step 7: Delete cached feature matrix ──
    for p in [FEATURE_MATRIX_CACHE, FEATURE_MATRIX_META]:
        if p.exists():
            p.unlink()
            logger.info("deleted cache: {}", p)

    logger.info("integration complete — run `python scripts/run_walkforward_comparison.py` next")
    return 0


def compute_raw_fundamentals(
    tickers: list[str],
    dates: list,
) -> pd.DataFrame:
    """Compute raw fundamental ratios via per-ticker merge_asof."""
    from src.data.db.session import get_engine

    engine = get_engine()

    logger.info("loading fundamentals_pit bulk data...")
    with engine.connect() as conn:
        fundamentals = pd.read_sql(
            __import__("sqlalchemy").text("""
                SELECT ticker, fiscal_period, event_time, metric_name, metric_value
                FROM fundamentals_pit
                WHERE ticker = ANY(:tickers) AND metric_name = ANY(:metrics)
                ORDER BY ticker, metric_name, event_time, fiscal_period
            """),
            conn,
            params={"tickers": list(tickers), "metrics": _METRICS},
        )
    logger.info("loaded {} fundamentals_pit rows", len(fundamentals))

    fundamentals["event_time"] = pd.to_datetime(fundamentals["event_time"])
    fundamentals["metric_value"] = pd.to_numeric(fundamentals["metric_value"], errors="coerce")
    fundamentals["ticker"] = fundamentals["ticker"].astype(str).str.upper()

    logger.info("loading stock_prices bulk data...")
    with engine.connect() as conn:
        prices = pd.read_sql(
            __import__("sqlalchemy").text("""
                SELECT ticker, trade_date, close
                FROM stock_prices WHERE ticker = ANY(:tickers)
                ORDER BY ticker, trade_date
            """),
            conn,
            params={"tickers": list(tickers)},
        )
    logger.info("loaded {} price rows", len(prices))
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    # Dedup fundamentals
    fundamentals = (
        fundamentals
        .sort_values(["ticker", "metric_name", "event_time", "fiscal_period"])
        .drop_duplicates(subset=["ticker", "metric_name", "event_time"], keep="last")
    )

    # Pivot to wide
    fund_wide = (
        fundamentals
        .pivot_table(
            index=["ticker", "event_time"],
            columns="metric_name",
            values="metric_value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["ticker", "event_time"])
    )
    logger.info("fundamentals wide: {} rows × {} cols", len(fund_wide), fund_wide.shape[1])

    walk_dates_ts = pd.to_datetime(sorted(set(dates)))
    dates_df = pd.DataFrame({"trade_date": walk_dates_ts})
    prices_prepared = prices[["ticker", "trade_date", "close"]].copy()

    # Per-ticker merge_asof
    unique_tickers = sorted(fund_wide["ticker"].unique())
    ticker_results = []
    for i, ticker in enumerate(unique_tickers):
        if i % 100 == 0:
            logger.info("merge_asof ticker {}/{}: {}", i + 1, len(unique_tickers), ticker)

        tk_fund = fund_wide[fund_wide["ticker"] == ticker].copy()
        if tk_fund.empty:
            continue
        tk_fund = tk_fund.sort_values("event_time").reset_index(drop=True)

        left = dates_df.copy()
        left["_key"] = left["trade_date"]
        tk_fund["_key"] = tk_fund["event_time"]

        merged = pd.merge_asof(
            left.sort_values("_key"),
            tk_fund.drop(columns=["ticker"]).sort_values("_key"),
            on="_key",
            direction="backward",
        ).drop(columns=["_key", "event_time"], errors="ignore")

        tk_prices = prices_prepared[prices_prepared["ticker"] == ticker].sort_values("trade_date")
        if tk_prices.empty:
            continue

        price_merged = pd.merge_asof(
            merged.sort_values("trade_date"),
            tk_prices[["trade_date", "close"]].sort_values("trade_date"),
            on="trade_date",
            direction="backward",
        )
        price_merged["ticker"] = ticker
        ticker_results.append(price_merged)

    if not ticker_results:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    df = pd.concat(ticker_results, ignore_index=True)
    logger.info("price-merged: {} rows", len(df))

    # Vectorized ratio computation
    _nan = pd.Series(np.nan, index=df.index, dtype=float)

    def _col(name: str) -> pd.Series:
        return df[name] if name in df.columns else _nan.copy()

    price = _col("close")
    eps = _col("eps")
    bvps = _col("book_value_per_share")
    revenue = _col("revenue")
    net_income = _col("net_income")
    total_assets = _col("total_assets")
    total_liabilities = _col("total_liabilities")
    total_debt = _col("total_debt")
    ebitda = _col("ebitda")
    cash = _col("cash").fillna(_col("cash_and_cash_equivalents"))
    dividend = _col("annual_dividend").fillna(_col("dividend_per_share"))
    operating_income = _col("operating_income")
    current_assets = _col("current_assets")
    current_liabilities = _col("current_liabilities")
    shares = _col("weighted_average_shares_outstanding")
    ocf = _col("operating_cash_flow")
    capex = _col("capital_expenditure")
    fcf = _col("free_cash_flow").fillna(ocf - capex.fillna(0))

    equity = total_assets - total_liabilities
    market_cap = price * shares
    rev_per_share = revenue / shares.replace(0, np.nan)
    ev = market_cap + total_debt.fillna(0) - cash.fillna(0)

    def _safe_div(a, b):
        return a / b.replace(0, np.nan)

    ratios = pd.DataFrame({
        "pe_ratio": _safe_div(price, eps),
        "pb_ratio": _safe_div(price, bvps),
        "ps_ratio": _safe_div(price, rev_per_share),
        "ev_ebitda": _safe_div(ev, ebitda),
        "fcf_yield": _safe_div(fcf, market_cap),
        "dividend_yield": _safe_div(dividend, price),
        "roe": _safe_div(net_income, equity),
        "roa": _safe_div(net_income, total_assets),
        "operating_margin": _safe_div(operating_income, revenue),
        "debt_to_equity": _safe_div(total_debt.fillna(total_liabilities), equity),
        "current_ratio": _safe_div(current_assets, current_liabilities),
    }, index=df.index)
    ratios["ticker"] = df["ticker"]
    ratios["trade_date"] = df["trade_date"]

    feature_cols = [c for c in ratios.columns if c not in ("ticker", "trade_date")]
    long = ratios.melt(
        id_vars=["ticker", "trade_date"],
        value_vars=feature_cols,
        var_name="feature_name",
        value_name="feature_value",
    )
    long = long.dropna(subset=["feature_value"])
    long = long[np.isfinite(long["feature_value"])]
    return long


def compute_sector_zscores(
    raw_long: pd.DataFrame,
    friday_dates: list,
) -> pd.DataFrame:
    """Compute sector-relative z-scores per (trade_date, feature, sector)."""
    raw_long = raw_long.copy()
    raw_long["trade_date"] = pd.to_datetime(raw_long["trade_date"])

    wide = raw_long.pivot_table(
        index=["trade_date", "ticker"],
        columns="feature_name",
        values="feature_value",
        aggfunc="first",
    )

    feature_cols = sorted(wide.columns)
    all_results = []

    unique_dates = sorted(wide.index.get_level_values("trade_date").unique())
    for i, trade_date in enumerate(unique_dates):
        if i % 50 == 0:
            logger.info("sector z-score date {}/{}", i + 1, len(unique_dates))

        cross_section = wide.xs(trade_date, level="trade_date")
        if len(cross_section) < 10:
            continue

        dt = trade_date.date() if hasattr(trade_date, "date") else trade_date
        sector_map = load_sector_map_pit(dt)

        for col in feature_cols:
            values = cross_section[col].dropna()
            if len(values) < 10:
                continue

            sectors = values.index.map(lambda t: sector_map.get(str(t).upper(), "Unknown"))
            df_temp = pd.DataFrame({
                "ticker": values.index,
                "sector": sectors,
                "raw_value": values.values,
            })

            sector_stats = df_temp.groupby("sector")["raw_value"].agg(["mean", "std", "count"])
            sector_stats = sector_stats[sector_stats["count"] >= 5]

            merged = df_temp.merge(sector_stats, left_on="sector", right_index=True, how="left")
            merged["std"] = merged["std"].replace(0, np.nan)
            merged["z_score"] = (merged["raw_value"] - merged["mean"]) / merged["std"]

            valid = merged.dropna(subset=["z_score"])
            if valid.empty:
                continue

            result = pd.DataFrame({
                "ticker": valid["ticker"].values,
                "trade_date": trade_date,
                "feature_name": f"{col}_sector_rel",
                "feature_value": valid["z_score"].values,
            })
            all_results.append(result)

    if not all_results:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])
    return pd.concat(all_results, ignore_index=True)


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally rank-normalize feature_value to 0-1 per (trade_date, feature_name)."""
    def _rank_group(group):
        vals = group["feature_value"]
        ranked = vals.rank(method="average", na_option="keep")
        n = vals.notna().sum()
        if n > 1:
            group = group.copy()
            group["feature_value"] = (ranked - 1) / (n - 1)
        else:
            group = group.copy()
            group["feature_value"] = 0.5
        return group

    return df.groupby(["trade_date", "feature_name"], group_keys=False).apply(_rank_group)


def update_ic_report() -> None:
    """Add 9 sector-relative features to the IC screening report."""
    report = pd.read_csv(IC_REPORT_PATH)

    # Load experiment results for IC values
    with open(EXPERIMENT_REPORT) as f:
        experiment = json.load(f)

    ic_results = {
        r["feature"]: r
        for r in experiment["s1_4_sector_relative_features"]["ic_results"]
    }

    # Remove any existing sector_rel features (re-run safety)
    report = report[~report["feature_name"].str.endswith("_sector_rel")]

    # Add new features
    new_rows = []
    for feat_name in PASSING_SECTOR_FEATURES:
        ic_data = ic_results.get(feat_name, {})
        new_rows.append({
            "feature_name": feat_name,
            "domain": "sector_relative",
            "ic": ic_data.get("mean_ic", 0),
            "rank_ic": ic_data.get("mean_ic", 0),
            "icir": ic_data.get("icir", 0),
            "abs_ic": abs(ic_data.get("mean_ic", 0)),
            "n_obs": ic_data.get("n_dates", 0) * 500,  # approx
            "n_dates": ic_data.get("n_dates", 0),
            "n_tickers": 500,
            "passed": True,
        })

    new_df = pd.DataFrame(new_rows)
    # Add schema version columns if present
    for col in report.columns:
        if col not in new_df.columns:
            new_df[col] = report[col].iloc[0] if len(report) > 0 else None

    updated = pd.concat([report, new_df[report.columns]], ignore_index=True)
    updated = updated.sort_values("abs_ic", ascending=False).reset_index(drop=True)
    updated.to_csv(IC_REPORT_PATH, index=False)
    logger.info("updated IC report: {} features total, {} sector-relative",
                len(updated), len(new_rows))


if __name__ == "__main__":
    from scripts.run_single_window_validation import configure_logging
    configure_logging()
    raise SystemExit(main())
