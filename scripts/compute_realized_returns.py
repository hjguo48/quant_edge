#!/usr/bin/env python3
"""W13.2 paper P&L tracker — compute realized returns for greyscale weeks.

For each `data/reports/greyscale/week_*.json`:
  1. Read `live_outputs.target_weights_after_risk` (the paper portfolio).
  2. Pull realized close prices from stock_prices for signal_date+1 ... signal_date+H.
  3. Compute weighted portfolio return at multiple horizons (1D, 5D, 20D, 60D)
     using only cash position assumption (cash earns 0%).
  4. Compute SPY benchmark return over the same horizon.
  5. Excess = portfolio_return - spy_return.

Output: data/reports/greyscale/greyscale_performance.json (single rolling file)

  {
    "as_of_utc": "...",
    "horizons_supported": [1, 5, 20, 60],
    "per_week": [
      {
        "week_number": 1,
        "signal_date": "2026-04-24",
        "horizons": {
          "1d": {"portfolio_return": 0.0023, "spy_return": 0.0011,
                 "excess": 0.0012, "status": "realized",
                 "tickers_used": 30, "tickers_missing": 0,
                 "horizon_end_date": "2026-04-25"},
          "5d": {... "status": "partial" if not yet 5d ...},
          "20d": {... "status": "pending" ...},
          "60d": {... "status": "pending" ...}
        }
      }
    ],
    "cumulative": {
      "1d": {"return": ..., "spy_return": ..., "excess": ..., "max_drawdown": ...,
             "weeks_realized": N, "winrate_vs_spy": 0.6},
      "5d": {...},
      ...
    }
  }

Status semantics:
  * "realized": full horizon ended on or before today
  * "partial": horizon end > today, but at least 1 trading day available
  * "pending": no trading day in horizon yet

Run:
    python scripts/compute_realized_returns.py [--report-dir ...]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
import json
import sys

from sqlalchemy import text

from src.data.db.session import get_engine

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "reports" / "greyscale"
DEFAULT_BENCHMARK = "SPY"
HORIZONS = (1, 5, 20, 60)


@dataclass(frozen=True)
class HorizonResult:
    horizon_days: int
    horizon_end_date: date | None
    portfolio_return: float | None
    spy_return: float | None
    excess: float | None
    tickers_used: int
    tickers_missing: int
    status: str  # realized / partial / pending


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    p.add_argument("--output-name", default="greyscale_performance.json")
    return p.parse_args(argv)


def list_week_reports(report_dir: Path) -> list[Path]:
    return sorted(
        p for p in report_dir.glob("week_*.json")
        if "bak" not in p.name and "contaminated" not in p.name
    )


def load_week_report(path: Path) -> dict:
    return json.loads(path.read_text())


def load_close_series(
    *,
    tickers: list[str],
    signal_date: date,
    horizon_days: int,
    conn,
) -> dict[str, dict[date, float]]:
    """Return {ticker: {trade_date: close}} from signal_date through signal_date + horizon * 2 calendar days.

    We over-fetch (×2) because horizon is in *trading* days but the cutoff in
    SQL is calendar days. We pick the Nth distinct trade_date >= signal_date+1 below.
    """
    if not tickers:
        return {}
    # For 60D horizon we need ~85 calendar days of margin
    fetch_until_calendar = horizon_days * 2 + 5
    rows = conn.execute(
        text(
            """
            SELECT ticker, trade_date, COALESCE(adj_close, close) AS px
            FROM stock_prices
            WHERE ticker = ANY(:tickers)
              AND trade_date >= :signal_date
              AND trade_date <= :end_date
            ORDER BY ticker, trade_date
            """
        ),
        {
            "tickers": tickers,
            "signal_date": signal_date,
            "end_date": signal_date.fromordinal(signal_date.toordinal() + fetch_until_calendar),
        },
    ).all()
    out: dict[str, dict[date, float]] = {}
    for ticker, td, px in rows:
        if px is None:
            continue
        out.setdefault(ticker, {})[td] = float(px)
    return out


def first_close_on_or_after(series: dict[date, float], anchor: date) -> tuple[date, float] | None:
    """Get the first (date, close) at or after anchor."""
    for d in sorted(series):
        if d >= anchor:
            return d, series[d]
    return None


def compute_horizon_return(
    *,
    series: dict[date, float],
    signal_date: date,
    horizon_days: int,
    today: date,
) -> tuple[float | None, date | None, str]:
    """Compute close-to-close return from signal_date close → N-th trading day after.

    Convention (W13.2):
      * Reference price = close on signal_date itself (or the most recent trading
        day on or before signal_date if signal_date is not a trading day).
      * 1D return = (close on first trading day strictly after signal_date) /
                    (close on signal_date) - 1
      * 5D return = same with 5th trading day after signal_date
      * etc.

    Returns (return_pct, horizon_end_date, status).
    """
    if horizon_days < 1:
        return None, None, "pending"

    # Reference price: the most recent close on or before signal_date.
    ref: tuple[date, float] | None = None
    for d in sorted(series, reverse=True):
        if d <= signal_date:
            ref = (d, series[d])
            break
    if ref is None or ref[1] == 0:
        return None, None, "pending"
    _, ref_px = ref

    # Trading days STRICTLY AFTER signal_date, ordered ascending.
    forward_days = sorted(d for d in series if d > signal_date)
    if not forward_days:
        return None, None, "pending"

    if len(forward_days) >= horizon_days:
        # Realized — take the horizon_days-th trading day after signal_date.
        end_d = forward_days[horizon_days - 1]
        end_px = series[end_d]
        return (end_px - ref_px) / ref_px, end_d, "realized" if end_d <= today else "partial"

    # Partial — fewer than horizon_days trading days available; use latest.
    end_d = forward_days[-1]
    end_px = series[end_d]
    return (end_px - ref_px) / ref_px, end_d, "partial"


def compute_per_week(
    week_report: dict,
    *,
    benchmark: str,
    today: date,
    conn,
) -> dict:
    weights = week_report.get("live_outputs", {}).get("target_weights_after_risk", {}) or {}
    signal_date_str = week_report.get("live_outputs", {}).get("signal_date")
    week_number = int(week_report.get("week_number") or 0)
    if not signal_date_str or not weights:
        return {
            "week_number": week_number,
            "signal_date": signal_date_str,
            "horizons": {f"{h}d": {"status": "pending", "portfolio_return": None,
                                    "spy_return": None, "excess": None,
                                    "tickers_used": 0, "tickers_missing": len(weights),
                                    "horizon_end_date": None}
                          for h in HORIZONS},
        }
    signal_date = datetime.fromisoformat(signal_date_str).date()
    tickers = list(weights.keys())

    max_h = max(HORIZONS)
    series = load_close_series(
        tickers=tickers + [benchmark],
        signal_date=signal_date,
        horizon_days=max_h,
        conn=conn,
    )

    horizon_block: dict[str, dict] = {}
    for h in HORIZONS:
        per_ticker_returns: dict[str, float] = {}
        per_ticker_status: dict[str, str] = {}
        end_dates: list[date] = []
        for tk in tickers:
            tk_series = series.get(tk, {})
            r, end_d, status = compute_horizon_return(
                series=tk_series, signal_date=signal_date, horizon_days=h, today=today,
            )
            if r is not None and status != "pending":
                per_ticker_returns[tk] = r
                per_ticker_status[tk] = status
                if end_d:
                    end_dates.append(end_d)
        if not per_ticker_returns:
            horizon_block[f"{h}d"] = {
                "status": "pending",
                "portfolio_return": None,
                "spy_return": None,
                "excess": None,
                "tickers_used": 0,
                "tickers_missing": len(weights),
                "horizon_end_date": None,
            }
            continue

        # Weighted portfolio return — cash (any weight not in admitted tickers) earns 0
        portfolio_return = sum(weights.get(tk, 0.0) * r for tk, r in per_ticker_returns.items())

        # SPY benchmark
        spy_series = series.get(benchmark, {})
        spy_r, spy_end, spy_status = compute_horizon_return(
            series=spy_series, signal_date=signal_date, horizon_days=h, today=today,
        )
        excess = (portfolio_return - spy_r) if spy_r is not None else None

        # Aggregate status — if any ticker is partial, the bucket is partial
        agg_status = "realized" if all(s == "realized" for s in per_ticker_status.values()) else "partial"
        horizon_end = max(end_dates) if end_dates else None

        horizon_block[f"{h}d"] = {
            "status": agg_status,
            "portfolio_return": portfolio_return,
            "spy_return": spy_r,
            "excess": excess,
            "tickers_used": len(per_ticker_returns),
            "tickers_missing": len(weights) - len(per_ticker_returns),
            "horizon_end_date": horizon_end.isoformat() if horizon_end else None,
        }

    return {
        "week_number": week_number,
        "signal_date": signal_date.isoformat(),
        "horizons": horizon_block,
    }


def aggregate_cumulative(per_week: list[dict]) -> dict[str, dict]:
    """Per-horizon: compound weekly returns, compute drawdown vs running max.

    Note: for non-overlapping horizons (e.g. 1d, 5d) compounding weekly returns
    is straightforward. For overlapping horizons (20d, 60d), the "cumulative"
    is informational only — the same trading day appears in multiple weeks'
    horizon. We still compound for visualization parity.
    """
    cumulative: dict[str, dict] = {}
    for h in HORIZONS:
        key = f"{h}d"
        weekly_records = [
            (w["signal_date"], w["horizons"][key])
            for w in per_week
            if w["horizons"][key]["status"] in ("realized", "partial")
            and w["horizons"][key]["portfolio_return"] is not None
        ]
        if not weekly_records:
            cumulative[key] = {
                "return": None, "spy_return": None, "excess": None,
                "max_drawdown": None, "weeks_realized": 0,
                "winrate_vs_spy": None,
                "weekly_curve": [],
            }
            continue
        cum_port = 1.0
        cum_spy = 1.0
        peak = 1.0
        max_dd = 0.0
        weeks_realized = 0
        wins = 0
        weekly_curve = []
        for sig_date, block in weekly_records:
            r_p = block["portfolio_return"]
            r_s = block.get("spy_return") or 0.0
            cum_port *= (1.0 + r_p)
            cum_spy *= (1.0 + r_s)
            if cum_port > peak:
                peak = cum_port
            else:
                dd = cum_port / peak - 1.0
                if dd < max_dd:
                    max_dd = dd
            if block["status"] == "realized":
                weeks_realized += 1
                if (block.get("excess") or 0.0) > 0:
                    wins += 1
            weekly_curve.append({
                "signal_date": sig_date,
                "weekly_return": r_p,
                "weekly_spy": r_s,
                "weekly_excess": block.get("excess"),
                "cumulative_return": cum_port - 1.0,
                "cumulative_spy": cum_spy - 1.0,
                "cumulative_excess": (cum_port - cum_spy),
            })
        cumulative[key] = {
            "return": cum_port - 1.0,
            "spy_return": cum_spy - 1.0,
            "excess": cum_port - cum_spy,
            "max_drawdown": max_dd,
            "weeks_realized": weeks_realized,
            "winrate_vs_spy": (wins / weeks_realized) if weeks_realized else None,
            "weekly_curve": weekly_curve,
        }
    return cumulative


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    weeks = list_week_reports(report_dir)
    today = datetime.now(timezone.utc).date()

    engine = get_engine()
    per_week_results: list[dict] = []
    with engine.connect() as conn:
        for path in weeks:
            week_data = load_week_report(path)
            result = compute_per_week(
                week_data, benchmark=args.benchmark, today=today, conn=conn,
            )
            per_week_results.append(result)

    cumulative = aggregate_cumulative(per_week_results)

    payload = {
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "today": today.isoformat(),
        "horizons_supported": list(HORIZONS),
        "benchmark": args.benchmark,
        "per_week": per_week_results,
        "cumulative": cumulative,
    }

    out_path = report_dir / args.output_name
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))

    print(f"wrote {out_path}")
    print(f"weeks processed: {len(per_week_results)}")
    for w in per_week_results:
        print(f"  Week {w['week_number']} ({w['signal_date']}):")
        for h_key, h_data in w["horizons"].items():
            r_str = f"{h_data['portfolio_return']:.4%}" if h_data['portfolio_return'] is not None else "—"
            ex_str = f"{h_data['excess']:.4%}" if h_data['excess'] is not None else "—"
            print(f"    {h_key:>4s} [{h_data['status']:>8s}]: paper={r_str:>9s} excess={ex_str:>9s} ({h_data['tickers_used']}/{h_data['tickers_used']+h_data['tickers_missing']} tickers)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(REPO_ROOT))
    raise SystemExit(main())
