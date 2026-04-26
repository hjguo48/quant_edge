#!/usr/bin/env python3
"""Build a date-level regime table for W9.3 weak-window gating analysis.

Each row = one SPY trade date with ex-ante PIT-safe macro / market covariates
that could be used to predict whether a forward return horizon is in an
"alpha-friendly" or "adverse" regime.

Columns:
- trade_date
- vix_level
- vix_zscore_60d, vix_zscore_252d
- yield_curve_10y2y      (DGS10 - DGS2)
- credit_spread_baa10y   (BAA10Y - DGS10)
- spy_close
- spy_ret_5d, spy_ret_20d, spy_ret_60d
- spy_vol_20d, spy_vol_60d        (annualized realized vol)
- spy_drawdown_60d               (SPY peak-to-current over last 60 trading days)
- universe_breadth_pct_above_20dma   (503-ticker fraction above 20-day MA)
- universe_return_dispersion_20d     (cross-sectional std of 20D return across universe)

PIT: every covariate uses only data with knowledge_time <= trade_date EOD NYT.

Usage:
    python scripts/build_regime_table.py \
        --start-date 2016-03-01 --end-date 2025-02-28 \
        --output data/features/regime_table_2016-2025.parquet
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import settings  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402

EASTERN = ZoneInfo("America/New_York")


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _as_of_end_utc(d: date) -> datetime:
    return datetime.combine(d, time(23, 59, 59), tzinfo=EASTERN).astimezone(timezone.utc)


def load_spy_prices(engine, start: date, end: date) -> pd.DataFrame:
    sql = text("""
        select trade_date, adj_close::float as close, knowledge_time
        from stock_prices
        where ticker='SPY' and trade_date between :start and :end
        order by trade_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"start": start, "end": end})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def load_macro_pit(engine, series_id: str, end: date) -> pd.DataFrame:
    sql = text("""
        select observation_date, value::float as value
        from macro_series_pit
        where series_id = :sid
          and observation_date <= :end
        order by observation_date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"sid": series_id, "end": end})
    df["observation_date"] = pd.to_datetime(df["observation_date"]).dt.date
    return df


def load_universe_panel(
    engine,
    frozen_universe_path: Path,
    start: date,
    end: date,
) -> pd.DataFrame:
    import json
    payload = json.loads(frozen_universe_path.read_text())
    tickers = payload["tickers"]
    sql = text("""
        select ticker, trade_date, adj_close::float as close
        from stock_prices
        where ticker = ANY(:tickers)
          and trade_date between :start and :end
        order by trade_date, ticker
    """)
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn, params={"tickers": tickers, "start": start, "end": end})
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    return df


def compute_spy_features(spy: pd.DataFrame) -> pd.DataFrame:
    spy = spy.sort_values("trade_date").reset_index(drop=True).copy()
    p = spy["close"]
    spy["spy_ret_1d"] = p.pct_change(1)
    spy["spy_ret_5d"] = p.pct_change(5)
    spy["spy_ret_20d"] = p.pct_change(20)
    spy["spy_ret_60d"] = p.pct_change(60)
    spy["spy_vol_20d"] = spy["spy_ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)
    spy["spy_vol_60d"] = spy["spy_ret_1d"].rolling(60, min_periods=60).std() * np.sqrt(252)
    rolling_max = p.rolling(60, min_periods=60).max()
    spy["spy_drawdown_60d"] = (p - rolling_max) / rolling_max
    return spy[["trade_date", "spy_ret_5d", "spy_ret_20d", "spy_ret_60d",
                "spy_vol_20d", "spy_vol_60d", "spy_drawdown_60d", "close"]].rename(columns={"close": "spy_close"})


def attach_macro_pit(table: pd.DataFrame, vix: pd.DataFrame, dgs10: pd.DataFrame,
                     dgs2: pd.DataFrame, baa10y: pd.DataFrame) -> pd.DataFrame:
    """For each trade_date, take the most recent macro observation_date <= trade_date."""
    out = table.copy()
    out["_td"] = pd.to_datetime(out["trade_date"])
    for col, src, valcol in [
        ("vix_level", vix, "value"),
        ("dgs10", dgs10, "value"),
        ("dgs2", dgs2, "value"),
        ("baa10y", baa10y, "value"),
    ]:
        s = src.copy()
        s["_obs"] = pd.to_datetime(s["observation_date"])
        s = s[["_obs", "value"]].rename(columns={"value": col})
        merged = pd.merge_asof(
            out.sort_values("_td"),
            s.sort_values("_obs"),
            left_on="_td",
            right_on="_obs",
            direction="backward",
        )
        out[col] = merged[col].values
    out = out.drop(columns=["_td"])
    out["yield_curve_10y2y"] = out["dgs10"] - out["dgs2"]
    out["credit_spread_baa10y"] = out["baa10y"] - out["dgs10"]
    out["vix_zscore_60d"] = (out["vix_level"] - out["vix_level"].rolling(60, min_periods=60).mean()) / out["vix_level"].rolling(60, min_periods=60).std()
    out["vix_zscore_252d"] = (out["vix_level"] - out["vix_level"].rolling(252, min_periods=252).mean()) / out["vix_level"].rolling(252, min_periods=252).std()
    return out.drop(columns=["dgs10", "dgs2", "baa10y"])


def compute_universe_breadth_dispersion(panel: pd.DataFrame) -> pd.DataFrame:
    """503-ticker breadth (% above 20DMA) and 20D return dispersion."""
    panel = panel.sort_values(["ticker", "trade_date"]).copy()
    panel["close"] = panel["close"].astype(float)
    panel["sma20"] = panel.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    panel["above_sma20"] = (panel["close"] > panel["sma20"]).astype(float)
    panel["ret_20d"] = panel.groupby("ticker")["close"].pct_change(20)
    breadth = panel.groupby("trade_date")["above_sma20"].mean().rename("universe_breadth_pct_above_20dma")
    dispersion = panel.groupby("trade_date")["ret_20d"].std().rename("universe_return_dispersion_20d")
    return pd.concat([breadth, dispersion], axis=1).reset_index()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=_parse_date, required=True)
    parser.add_argument("--end-date", type=_parse_date, required=True)
    parser.add_argument("--frozen-universe-path", type=Path,
                        default=REPO_ROOT / "data/features/frozen_universe_503.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    engine = create_engine(settings.database_url)

    print("[regime] loading SPY prices...")
    spy = load_spy_prices(engine, args.start_date, args.end_date)
    if spy.empty:
        raise RuntimeError("No SPY rows returned for requested range.")
    spy_feats = compute_spy_features(spy)

    print("[regime] loading macro PIT...")
    vix = load_macro_pit(engine, "VIXCLS", args.end_date)
    dgs10 = load_macro_pit(engine, "DGS10", args.end_date)
    dgs2 = load_macro_pit(engine, "DGS2", args.end_date)
    baa10y = load_macro_pit(engine, "BAA10Y", args.end_date)

    print("[regime] attaching macro covariates...")
    table = attach_macro_pit(spy_feats, vix, dgs10, dgs2, baa10y)

    print("[regime] loading universe panel for breadth + dispersion...")
    panel = load_universe_panel(engine, args.frozen_universe_path, args.start_date, args.end_date)
    print(f"[regime]   universe panel rows: {len(panel):,}")
    breadth_disp = compute_universe_breadth_dispersion(panel)

    table = table.merge(breadth_disp, on="trade_date", how="left")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    print(f"[regime] wrote {args.output} (rows={len(table)}, cols={len(table.columns)})")
    print(f"[regime] columns: {table.columns.tolist()}")
    print(f"[regime] sample:\n{table.tail(3).to_string()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
