"""Calibration: measure trade-level vs minute-proxy feature magnitude ratio per-sample.

Purpose: before investing 1-2 days to build flat-files trade-level pipeline, measure how much
stronger trade-level features are vs minute-proxy on the SAME (ticker, date) samples.

Approach:
- For each (ticker, date) sample: fetch raw trades (Polygon REST) + minute_aggs (DB)
- Compute 3 features (trade_imbalance_proxy, late_day_aggressiveness, offhours_trade_ratio) twice:
  * Task 7 version from raw trades (trade-level truth)
  * Task 10-alt proxy version from minute aggs
- Report per-sample magnitude ratio |trade| / |minute| and overall average

Decision rule (my judgment):
- avg |trade|/|minute| > 1.5 → flat-files likely pushes IC from 0.013 to 0.020+, worth 2 days
- 1.2 < avg < 1.5 → borderline, invest only in 1 window (W5, ~40h)
- avg < 1.2 → minute-proxy captures most signal, flat-files not worth

Note: N=10 is too small for IC statistics but OK for magnitude comparison on same sample.
"""
from __future__ import annotations

from datetime import date, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.data.db.session import get_engine
from src.data.polygon_trades import PolygonTradesClient
from src.features.trade_microstructure import (
    compute_late_day_aggressiveness,
    compute_offhours_trade_ratio,
    compute_trade_imbalance_proxy,
)
from src.features.trade_microstructure_minute_proxy import (
    compute_late_day_aggressiveness_minute,
    compute_offhours_trade_ratio_minute,
    compute_trade_imbalance_proxy_minute,
)

EASTERN = ZoneInfo("America/New_York")

# 10 samples across 3 weak windows, mix of mega/mid caps
SAMPLES: list[tuple[str, date]] = [
    # W5 (2021-09 → 2022-02): early weak period
    ("JPM", date(2021, 10, 5)),
    ("CAT", date(2021, 11, 15)),
    ("CRM", date(2022, 1, 20)),
    # W10 (2024-03 → 2024-08): AI concentration era
    ("MSFT", date(2024, 4, 10)),
    ("NVDA", date(2024, 6, 12)),
    ("ADBE", date(2024, 7, 8)),
    ("HD", date(2024, 8, 20)),
    # W11 (2024-09 → 2025-02): post-election
    ("JNJ", date(2024, 10, 15)),
    ("BAC", date(2024, 12, 5)),
    ("AAPL", date(2025, 2, 10)),
]


def fetch_minute_aggs(ticker: str, trading_date: date) -> pd.DataFrame:
    """Load minute aggs for (ticker, date) from DB."""
    engine = get_engine()
    with engine.connect() as conn:
        frame = pd.read_sql(
            text(
                """
                SELECT ticker, trade_date AS trading_date, minute_ts,
                       open, high, low, close, volume, transactions, vwap
                FROM stock_minute_aggs
                WHERE ticker = :ticker AND trade_date = :trade_date
                ORDER BY minute_ts
                """,
            ),
            conn,
            params={"ticker": ticker, "trade_date": trading_date},
        )
    if not frame.empty:
        frame["minute_ts"] = pd.to_datetime(frame["minute_ts"], utc=True)
    return frame


def fetch_trades_as_feature_frame(ticker: str, trading_date: date, client: PolygonTradesClient) -> pd.DataFrame:
    """Fetch all trades for (ticker, date) and convert to DataFrame for Task 7 feature functions."""
    trades, api_calls = client.fetch_trades_for_day(ticker, trading_date, page_size=50000, max_pages=100)
    if not trades:
        return pd.DataFrame(columns=["sip_timestamp", "price", "size", "exchange", "trf_id", "trf_timestamp", "conditions"])
    rows = [
        {
            "sip_timestamp": t.sip_timestamp,
            "price": float(t.price),
            "size": float(t.size),
            "exchange": t.exchange,
            "trf_id": t.trf_id,
            "trf_timestamp": t.trf_timestamp,
            "conditions": list(t.conditions) if t.conditions else [],
        }
        for t in trades
    ]
    df = pd.DataFrame(rows)
    print(f"    fetched {len(df):,} trades, {api_calls} API calls")
    return df


def compute_pair(ticker: str, trading_date: date, client: PolygonTradesClient) -> dict[str, Any]:
    print(f"  {ticker} {trading_date}:")
    # Minute proxy (existing DB data)
    minute_df = fetch_minute_aggs(ticker, trading_date)
    if minute_df.empty:
        print(f"    no minute_aggs, skip")
        return {}
    m_imb = compute_trade_imbalance_proxy_minute(minute_df)
    m_late = compute_late_day_aggressiveness_minute(minute_df)
    m_off = compute_offhours_trade_ratio_minute(minute_df)
    print(f"    minute: imb={m_imb:+.4f}  late={m_late:+.4f}  off={m_off:+.4f}")

    # Trade-level via REST
    trades_df = fetch_trades_as_feature_frame(ticker, trading_date, client)
    if trades_df.empty:
        print(f"    no trades, skip")
        return {}
    t_imb = compute_trade_imbalance_proxy(trades_df, condition_allow=None)
    t_late = compute_late_day_aggressiveness(trades_df, late_day_window_et=("15:00", "16:00"))
    t_off = compute_offhours_trade_ratio(trades_df, pre_window_et=("04:00", "09:30"), post_window_et=("16:00", "20:00"))
    print(f"    trade:  imb={t_imb:+.4f}  late={t_late:+.4f}  off={t_off:+.4f}")

    return {
        "ticker": ticker,
        "date": trading_date,
        "trades_count": len(trades_df),
        "minute_imbalance": m_imb,
        "trade_imbalance": t_imb,
        "minute_late": m_late,
        "trade_late": t_late,
        "minute_off": m_off,
        "trade_off": t_off,
    }


def main() -> int:
    print("Calibration: trade-level REST vs minute-proxy on same (ticker, date)")
    print(f"Samples: {len(SAMPLES)}")
    client = PolygonTradesClient(min_request_interval=0.05)
    rows = []
    for ticker, trading_date in SAMPLES:
        r = compute_pair(ticker, trading_date, client)
        if r:
            rows.append(r)

    if not rows:
        print("\nNo valid samples.")
        return 1

    df = pd.DataFrame(rows)
    print("\n=== Results ===")
    print(df.to_string(index=False))

    def ratio(trade_col: str, minute_col: str) -> float:
        """Average |trade| / max(|minute|, epsilon) across samples where both non-null and non-zero."""
        t = df[trade_col].abs()
        m = df[minute_col].abs()
        valid = t.notna() & m.notna() & (m > 1e-6) & (t > 1e-6)
        if not valid.any():
            return float("nan")
        return float((t[valid] / m[valid]).mean())

    r_imb = ratio("trade_imbalance", "minute_imbalance")
    r_late = ratio("trade_late", "minute_late")
    r_off = ratio("trade_off", "minute_off")
    print("\n=== Magnitude ratios (|trade| / |minute|) ===")
    print(f"  imbalance:  {r_imb:.3f}")
    print(f"  late_day:   {r_late:.3f}")
    print(f"  offhours:   {r_off:.3f}" if not np.isnan(r_off) else "  offhours:   N/A (minute all zero)")

    # Decision
    useful_ratios = [r for r in [r_imb, r_late] if not np.isnan(r) and np.isfinite(r)]
    if useful_ratios:
        avg = sum(useful_ratios) / len(useful_ratios)
        print(f"\n=== Average ratio (imbalance + late_day): {avg:.3f} ===")
        if avg > 1.5:
            print("  → FLAT FILES 值得 (预期 trade-level IC > 0.020, 过 Gate 0.015 阈值)")
        elif avg > 1.2:
            print("  → 边缘 — 建议只跑 W5 (1 weak window, ~40h)")
        else:
            print("  → 不值 — minute-proxy 已 capture 主要信号, 放弃 flat-files")

    out = Path("data/reports/week4/calibration_trade_vs_minute.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅ saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
