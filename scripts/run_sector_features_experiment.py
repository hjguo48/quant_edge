"""S1.4 / S1.5: Sector-relative features + sector-neutral labels experiment.

Computes raw fundamental ratios from fundamentals_pit (bypassing rank
normalization), then creates sector-relative z-scores as new features.
Also tests sector-demeaned forward returns as an alternative label.

Usage:
    python scripts/run_sector_features_experiment.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from sqlalchemy import text

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import (
    configure_logging,
    fill_feature_matrix,
    long_to_feature_matrix,
)
from scripts.run_walkforward_comparison import json_safe
from src.data.db.session import get_engine
from src.features.sector import (
    SECTOR_RELATIVE_CANDIDATES,
    compute_sector_relative_features,
    load_sector_map_pit,
)

DEFAULT_FEATURE_MATRIX_PATH = "data/features/walkforward_feature_matrix_60d.parquet"
DEFAULT_LABELS_PATH = "data/labels/forward_returns_60d.parquet"
DEFAULT_REPORT_PATH = "data/reports/sector_features_experiment.json"

# Walk-forward window definitions (same as run_walkforward_comparison.py)
REBALANCE_WEEKDAY = 4  # Friday


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    logger.info("S1.4/S1.5: Sector-relative features + sector-neutral labels experiment")

    # ── Step 1: Load existing feature matrix (rank-normalized, for baseline) ──
    existing_matrix = load_existing_feature_matrix(REPO_ROOT / args.feature_matrix_path)
    walk_dates = sorted(existing_matrix.index.get_level_values("trade_date").unique())
    all_tickers = sorted(existing_matrix.index.get_level_values("ticker").unique())
    logger.info(
        "loaded existing feature matrix: {} dates, {} tickers, {} features",
        len(walk_dates), len(all_tickers), existing_matrix.shape[1],
    )

    # ── Step 2: Compute raw fundamental ratios from DB ──────────────────────
    logger.info("computing raw fundamental ratios from fundamentals_pit + stock_prices...")
    raw_fundamentals = compute_raw_fundamentals_batch(
        tickers=all_tickers,
        dates=[d.date() if hasattr(d, 'date') else d for d in walk_dates],
    )
    logger.info("raw fundamentals: {} rows, {} features", len(raw_fundamentals),
                raw_fundamentals["feature_name"].nunique())

    # ── Step 3: Compute sector-relative features ────────────────────────────
    logger.info("computing sector-relative z-scores with PIT-safe sector mapping...")
    sector_rel_features = compute_sector_relative_batch(
        raw_fundamentals=raw_fundamentals,
        dates=[d.date() if hasattr(d, 'date') else d for d in walk_dates],
    )
    logger.info("sector-relative features: {} rows", len(sector_rel_features))

    # ── Step 4: Build augmented feature matrix ──────────────────────────────
    sector_rel_names = sorted(sector_rel_features["feature_name"].unique())
    augmented_features = sector_rel_names
    logger.info("new sector-relative features: {}", augmented_features)

    sector_matrix = long_to_feature_matrix(sector_rel_features, augmented_features)
    sector_matrix = fill_feature_matrix(sector_matrix)

    # Align with existing matrix
    common_idx = existing_matrix.index.intersection(sector_matrix.index)
    combined_matrix = pd.concat(
        [existing_matrix.loc[common_idx], sector_matrix.loc[common_idx]],
        axis=1,
    )
    logger.info(
        "combined matrix: {} rows, {} features (existing={}, new={})",
        len(combined_matrix), combined_matrix.shape[1],
        existing_matrix.shape[1], len(augmented_features),
    )

    # ── Step 5: Load labels and compute IC ──────────────────────────────────
    labels = load_labels(REPO_ROOT / args.labels_path)
    aligned_X, aligned_y = align_panel(combined_matrix, labels)
    logger.info("aligned panel: {} rows", len(aligned_X))

    # IC analysis for new sector-relative features
    ic_results = compute_per_feature_ic(aligned_X, aligned_y, augmented_features)

    # IC for existing fundamental features (baseline comparison)
    existing_fundamental = [f for f in SECTOR_RELATIVE_CANDIDATES if f in existing_matrix.columns]
    ic_baseline = compute_per_feature_ic(aligned_X, aligned_y, existing_fundamental)

    # ── Step 6: S1.5 — Sector-neutral labels ────────────────────────────────
    logger.info("S1.5: computing sector-neutral (sector-demeaned) labels...")
    sector_neutral_ic = compute_sector_neutral_label_ic(
        features=aligned_X,
        labels=aligned_y,
        feature_columns=list(existing_matrix.columns),
        dates=[d.date() if hasattr(d, 'date') else d for d in walk_dates],
    )

    # ── Step 7: Report ──────────────────────────────────────────────────────
    report = build_report(
        ic_results=ic_results,
        ic_baseline=ic_baseline,
        sector_neutral_ic=sector_neutral_ic,
        augmented_features=augmented_features,
        n_dates=len(walk_dates),
        n_tickers=len(all_tickers),
        combined_shape=combined_matrix.shape,
    )
    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved experiment report to {}", report_path)

    # Summary
    print_summary(ic_results, ic_baseline, sector_neutral_ic)
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# Raw fundamental ratio computation (batch)
# ═══════════════════════════════════════════════════════════════════════════

def compute_raw_fundamentals_batch(
    tickers: list[str],
    dates: list[date],
) -> pd.DataFrame:
    """Compute raw fundamental ratios for all tickers × dates (vectorized).

    Strategy: build a (ticker, metric) → latest PIT-visible value snapshot
    for each as_of date using pandas merge_asof, then compute ratios in bulk.
    """
    engine = get_engine()
    _METRICS = [
        "eps", "book_value_per_share", "revenue", "net_income",
        "total_assets", "total_liabilities", "total_debt",
        "operating_cash_flow", "capital_expenditure", "free_cash_flow",
        "ebitda", "cash", "cash_and_cash_equivalents",
        "annual_dividend", "dividend_per_share",
        "gross_profit", "operating_income",
        "current_assets", "current_liabilities",
        "weighted_average_shares_outstanding",
    ]

    # ── Bulk load ───────────────────────────────────────────────────────
    logger.info("loading fundamentals_pit bulk data...")
    with engine.connect() as conn:
        fundamentals = pd.read_sql(
            text("""
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
            text("""
                SELECT ticker, trade_date, close
                FROM stock_prices
                WHERE ticker = ANY(:tickers)
                ORDER BY ticker, trade_date
            """),
            conn,
            params={"tickers": list(tickers)},
        )
    logger.info("loaded {} price rows", len(prices))
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()

    # ── Pre-compute latest PIT snapshot per (ticker, metric) ────────────
    # For each (ticker, metric_name), keep only the latest row by
    # (event_time, fiscal_period) — this is the "last known" value.
    # Then we can merge_asof against walk-forward dates.
    logger.info("building latest-PIT snapshots per (ticker, metric)...")
    fundamentals = (
        fundamentals
        .sort_values(["ticker", "metric_name", "event_time", "fiscal_period"])
        .drop_duplicates(subset=["ticker", "metric_name", "event_time"], keep="last")
    )

    # For each (ticker, metric_name), build a time-series of the latest value
    # known at each event_time. We want: for as_of date D, what was the last
    # known value of each metric?
    # Use groupby + merge_asof approach per metric for efficiency.

    walk_dates_ts = pd.to_datetime(sorted(set(dates)))

    # Build a scaffold: all (ticker, date) combinations we need
    scaffold = pd.MultiIndex.from_product(
        [sorted(set(tickers)), walk_dates_ts],
        names=["ticker", "trade_date"],
    ).to_frame(index=False)

    # Per-ticker merge_asof (avoids global sort requirement)
    logger.info("per-ticker merge_asof for {} metrics across {} dates × {} tickers...",
                len(_METRICS), len(walk_dates_ts), len(tickers))

    # Pivot fundamentals to wide: (ticker, event_time) → metric columns
    fund_wide = (
        fundamentals
        .sort_values(["ticker", "metric_name", "event_time", "fiscal_period"])
        .drop_duplicates(subset=["ticker", "metric_name", "event_time"], keep="last")
        .pivot_table(
            index=["ticker", "event_time"],
            columns="metric_name",
            values="metric_value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["ticker", "event_time"])
    )
    logger.info("fundamentals wide table: {} rows × {} cols", len(fund_wide), fund_wide.shape[1])

    dates_df = pd.DataFrame({"trade_date": walk_dates_ts})
    prices_prepared = prices[["ticker", "trade_date", "close"]].copy()

    ticker_results = []
    unique_tickers = sorted(fund_wide["ticker"].unique())
    for i, ticker in enumerate(unique_tickers):
        if i % 100 == 0:
            logger.info("processing ticker {}/{}: {}", i + 1, len(unique_tickers), ticker)

        # Fundamental merge_asof for this ticker
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

        # Price merge_asof for this ticker
        tk_prices = prices_prepared[prices_prepared["ticker"] == ticker].sort_values("trade_date").reset_index(drop=True)
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

    price_merged = pd.concat(ticker_results, ignore_index=True)
    logger.info("price-merged table: {} rows", len(price_merged))

    # ── Vectorized ratio computation ────────────────────────────────────
    logger.info("computing ratios vectorized...")
    df = price_merged
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

    # Melt to long format
    feature_cols = [c for c in ratios.columns if c not in ("ticker", "trade_date")]
    long = ratios.melt(
        id_vars=["ticker", "trade_date"],
        value_vars=feature_cols,
        var_name="feature_name",
        value_name="feature_value",
    )
    long = long.dropna(subset=["feature_value"])
    long = long[np.isfinite(long["feature_value"])]
    long["trade_date"] = long["trade_date"].dt.date

    logger.info("raw fundamentals computed: {} rows, {} features",
                len(long), long["feature_name"].nunique())
    return long


# ═══════════════════════════════════════════════════════════════════════════
# Sector-relative computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_sector_relative_batch(
    raw_fundamentals: pd.DataFrame,
    dates: list[date],
) -> pd.DataFrame:
    """Compute sector-relative z-scores for each (date, sector, feature)."""
    if raw_fundamentals.empty:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    # Pivot to wide
    wide = (
        raw_fundamentals
        .pivot_table(
            index=["trade_date", "ticker"],
            columns="feature_name",
            values="feature_value",
            aggfunc="first",
        )
    )

    feature_cols = [c for c in SECTOR_RELATIVE_CANDIDATES if c in wide.columns]
    if not feature_cols:
        logger.warning("no sector-relative candidate features found in raw fundamentals")
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    all_results: list[pd.DataFrame] = []

    for trade_date in sorted(wide.index.get_level_values("trade_date").unique()):
        cross_section = wide.xs(trade_date, level="trade_date")
        if len(cross_section) < 10:
            continue

        dt = trade_date if isinstance(trade_date, date) else trade_date.date()
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

            # Sector z-score
            sector_stats = df_temp.groupby("sector")["raw_value"].agg(["mean", "std", "count"])
            sector_stats = sector_stats[sector_stats["count"] >= 5]  # min sector size

            merged = df_temp.merge(sector_stats, left_on="sector", right_index=True, how="left")
            merged["std"] = merged["std"].replace(0, np.nan)
            merged["z_score"] = (merged["raw_value"] - merged["mean"]) / merged["std"]

            valid = merged.dropna(subset=["z_score"])
            if valid.empty:
                continue

            result = pd.DataFrame({
                "ticker": valid["ticker"].values,
                "trade_date": dt,
                "feature_name": f"{col}_sector_rel",
                "feature_value": valid["z_score"].values,
            })
            all_results.append(result)

    if not all_results:
        return pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"])

    return pd.concat(all_results, ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# IC Analysis
# ═══════════════════════════════════════════════════════════════════════════

def compute_per_feature_ic(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
) -> list[dict]:
    """Compute per-date Spearman IC for each feature, then aggregate."""
    results = []
    dates = sorted(features.index.get_level_values("trade_date").unique())

    for col in feature_columns:
        if col not in features.columns:
            continue

        ic_series = []
        for dt in dates:
            try:
                x = features.xs(dt, level="trade_date")[col].dropna()
                y = labels.xs(dt, level="trade_date").reindex(x.index).dropna()
                common = x.index.intersection(y.index)
                if len(common) < 20:
                    continue
                rho, _ = spearmanr(x.loc[common], y.loc[common])
                if np.isfinite(rho):
                    ic_series.append(rho)
            except (KeyError, ValueError):
                continue

        if not ic_series:
            results.append({"feature": col, "mean_ic": np.nan, "ic_std": np.nan,
                           "icir": np.nan, "n_dates": 0, "sign_consistency": np.nan})
            continue

        arr = np.array(ic_series)
        mean_ic = float(np.mean(arr))
        std_ic = float(np.std(arr, ddof=1)) if len(arr) > 1 else np.nan
        icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
        pos_frac = float(np.mean(arr > 0))

        results.append({
            "feature": col,
            "mean_ic": mean_ic,
            "ic_std": std_ic,
            "icir": icir,
            "n_dates": len(ic_series),
            "sign_consistency": pos_frac,
        })

    return sorted(results, key=lambda r: abs(r.get("mean_ic", 0) or 0), reverse=True)


def compute_sector_neutral_label_ic(
    features: pd.DataFrame,
    labels: pd.Series,
    feature_columns: list[str],
    dates: list[date],
) -> dict:
    """S1.5: Compare IC using sector-demeaned labels vs SPY-excess labels."""
    # Use the actual timestamps from the labels index (not Python date objects)
    label_dates = sorted(labels.index.get_level_values("trade_date").unique())

    chunks: list[pd.Series] = []
    for ts in label_dates:
        try:
            y = labels.xs(ts, level="trade_date")
        except KeyError:
            continue
        if y.empty:
            continue

        dt_safe = ts.date() if hasattr(ts, "date") else ts
        sector_map = load_sector_map_pit(dt_safe)
        sectors = y.index.map(lambda t: sector_map.get(str(t).upper(), "Unknown"))

        df_temp = pd.DataFrame({"ret": y.values, "sector": sectors}, index=y.index)
        sector_means = df_temp.groupby("sector")["ret"].transform("mean")
        demeaned = df_temp["ret"] - sector_means
        demeaned = demeaned.dropna()
        if demeaned.empty:
            continue

        # Build MultiIndex chunk
        idx = pd.MultiIndex.from_arrays(
            [np.full(len(demeaned), ts), demeaned.index],
            names=["trade_date", "ticker"],
        )
        chunks.append(pd.Series(demeaned.values, index=idx, name="sector_demeaned"))

    if chunks:
        sector_demeaned = pd.concat(chunks).sort_index()
    else:
        sector_demeaned = pd.Series(dtype=float, name="sector_demeaned")
        sector_demeaned.index = pd.MultiIndex.from_tuples([], names=["trade_date", "ticker"])

    logger.info("sector-demeaned labels: {} rows across {} dates",
                len(sector_demeaned), len(chunks))

    # Compute IC with sector-demeaned labels for top features
    top_features = feature_columns[:15]  # top 15 features
    ic_spy_excess = compute_per_feature_ic(features, labels, top_features)
    ic_sector_neutral = compute_per_feature_ic(features, sector_demeaned, top_features)

    return {
        "spy_excess_ic": {r["feature"]: r["mean_ic"] for r in ic_spy_excess},
        "sector_neutral_ic": {r["feature"]: r["mean_ic"] for r in ic_sector_neutral},
        "n_dates": len(dates),
        "comparison": [
            {
                "feature": r1["feature"],
                "spy_excess_ic": r1["mean_ic"],
                "sector_neutral_ic": next(
                    (r2["mean_ic"] for r2 in ic_sector_neutral if r2["feature"] == r1["feature"]),
                    None,
                ),
            }
            for r1 in ic_spy_excess
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_existing_feature_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df = df.set_index(["trade_date", "ticker"]).sort_index()
    return df


def load_labels(path: Path) -> pd.Series:
    labels_df = pd.read_parquet(path)
    labels_df["trade_date"] = pd.to_datetime(labels_df["trade_date"])
    labels_df = labels_df.set_index(["trade_date", "ticker"]).sort_index()
    return labels_df["excess_return"]


def align_panel(features: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = features.index.intersection(labels.index)
    X = features.loc[common].sort_index()
    y = labels.loc[common].sort_index()
    mask = y.notna() & X.notna().all(axis=1)
    return X.loc[mask], y.loc[mask]


def build_report(
    ic_results: list[dict],
    ic_baseline: list[dict],
    sector_neutral_ic: dict,
    augmented_features: list[str],
    n_dates: int,
    n_tickers: int,
    combined_shape: tuple,
) -> dict:
    # Features passing IC > 0.01 threshold
    passing = [r for r in ic_results if abs(r.get("mean_ic", 0) or 0) >= 0.01]

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "S1.4/S1.5 Sector-relative features + sector-neutral labels",
        "pit_safety": "Method B: time-conditional sector mapping (2018-09 GICS fix)",
        "panel_stats": {
            "n_dates": n_dates,
            "n_tickers": n_tickers,
            "combined_features": combined_shape[1],
        },
        "s1_4_sector_relative_features": {
            "new_features": augmented_features,
            "n_new_features": len(augmented_features),
            "n_passing_ic_threshold": len(passing),
            "ic_results": ic_results,
            "baseline_ic": ic_baseline,
            "passing_features": [r["feature"] for r in passing],
        },
        "s1_5_sector_neutral_labels": sector_neutral_ic,
    }


def print_summary(
    ic_results: list[dict],
    ic_baseline: list[dict],
    sector_neutral_ic: dict,
) -> None:
    print("\n" + "=" * 70)
    print("  S1.4: Sector-relative features IC analysis")
    print("=" * 70)

    baseline_lookup = {r["feature"]: r["mean_ic"] for r in ic_baseline}

    print(f"\n  {'Feature':<30} {'Sector-Rel IC':>14} {'Raw IC':>10} {'Delta':>10}")
    print(f"  {'-'*30} {'-'*14} {'-'*10} {'-'*10}")
    for r in ic_results:
        feat = r["feature"]
        base_name = feat.replace("_sector_rel", "")
        base_ic = baseline_lookup.get(base_name)
        rel_ic = r.get("mean_ic")
        if rel_ic is None or np.isnan(rel_ic):
            continue
        base_str = f"{base_ic:.4f}" if base_ic and not np.isnan(base_ic) else "N/A"
        delta = (rel_ic - base_ic) if base_ic and not np.isnan(base_ic) else None
        delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
        passing = " *" if abs(rel_ic) >= 0.01 else ""
        print(f"  {feat:<30} {rel_ic:>14.4f} {base_str:>10} {delta_str:>10}{passing}")

    print(f"\n  * = passes |IC| >= 0.01 threshold")

    print(f"\n{'=' * 70}")
    print("  S1.5: Sector-neutral labels comparison")
    print("=" * 70)
    comparison = sector_neutral_ic.get("comparison", [])
    if comparison:
        print(f"\n  {'Feature':<30} {'SPY-Excess IC':>14} {'Sector-Neut IC':>15} {'Delta':>10}")
        print(f"  {'-'*30} {'-'*14} {'-'*15} {'-'*10}")
        for c in comparison[:10]:
            spy = c.get("spy_excess_ic")
            sn = c.get("sector_neutral_ic")
            if spy is None or sn is None:
                continue
            delta = sn - spy if not np.isnan(sn) and not np.isnan(spy) else None
            delta_str = f"{delta:+.4f}" if delta is not None else "N/A"
            print(f"  {c['feature']:<30} {spy:>14.4f} {sn:>15.4f} {delta_str:>10}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S1.4/S1.5 Sector features experiment")
    parser.add_argument("--feature-matrix-path", default=DEFAULT_FEATURE_MATRIX_PATH)
    parser.add_argument("--labels-path", default=DEFAULT_LABELS_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
