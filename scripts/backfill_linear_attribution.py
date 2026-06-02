#!/usr/bin/env python3
"""Backfill linear_attribution.ticker_features for existing week_*.json reports.

Bug context: scripts/run_greyscale_live.py previously wrote ticker_features
only for holdings via `constrained.weights.keys()`, but that loop produced
0 entries (constrained.weights vs ridge_X.index alignment issue). Result:
linear_attribution.ticker_features was {} in all weekly reports, so
/api/predictions/{ticker}/shap returned 404 for every stock.

Wrapper fix in run_greyscale_live.py now iterates ridge_X directly. This
script repairs existing week_*.json files by reloading feature_store data
at each signal_date and computing ticker_features the same way.

Usage:
  python scripts/backfill_linear_attribution.py [--report path] [--all]

  --report PATH : single week_*.json to repair
  --all         : repair all week_*.json in default report dir
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.data.db.session import get_engine  # noqa: E402
from src.utils.io import write_json_atomic  # noqa: E402

DEFAULT_REPORT_DIR = REPO_ROOT / "data" / "reports" / "greyscale"


def load_features_for_signal_date(
    signal_date: date, feature_names: list[str]
) -> pd.DataFrame:
    """Load feature_store rows for the given signal_date + features.

    Returns ticker × feature DataFrame matching the column order of
    feature_names. Missing (ticker, feature) cells are NaN; caller converts
    to 0.0 to match wrapper behavior.
    """
    sql = text(
        """
        SELECT ticker, feature_name, feature_value
        FROM feature_store
        WHERE calc_date = :signal_date
          AND feature_name = ANY(:feature_names)
        """
    )
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            sql, {"signal_date": signal_date, "feature_names": list(feature_names)}
        )
        rows = [dict(row._mapping) for row in result]

    if not rows:
        return pd.DataFrame(columns=feature_names)

    df = pd.DataFrame(rows)
    pivot = df.pivot_table(
        index="ticker", columns="feature_name", values="feature_value", aggfunc="last"
    )
    # Ensure column order matches bundle's feature_names exactly
    pivot = pivot.reindex(columns=feature_names)
    return pivot


def repair_report(report_path: Path, dry_run: bool = False) -> dict[str, int]:
    """Repair one week_*.json. Returns stats dict."""
    raw = report_path.read_text()
    report = json.loads(raw)

    la = report.get("linear_attribution") or {}
    if not isinstance(la, dict):
        return {"status": "skip", "reason": "no_linear_attribution"}

    feature_names = la.get("feature_names") or []
    if not feature_names:
        return {"status": "skip", "reason": "no_feature_names"}

    signal_date_str = report.get("live_outputs", {}).get("signal_date") or report.get(
        "db_state", {}
    ).get("latest_pit_trade_date")
    if not signal_date_str:
        return {"status": "skip", "reason": "no_signal_date"}
    signal_date = date.fromisoformat(signal_date_str)

    before_count = len(la.get("ticker_features") or {})

    df = load_features_for_signal_date(signal_date, feature_names)
    if df.empty:
        return {
            "status": "skip",
            "reason": f"no_feature_store_data_for_{signal_date_str}",
        }

    ticker_features: dict[str, list[float]] = {}
    for ticker, row in df.iterrows():
        feats = []
        for v in row.values:
            try:
                if v is None or pd.isna(v):
                    feats.append(0.0)
                    continue
                vf = float(v)
                feats.append(vf if np.isfinite(vf) else 0.0)
            except (TypeError, ValueError):
                feats.append(0.0)
        ticker_features[str(ticker)] = feats

    la["ticker_features"] = ticker_features
    report["linear_attribution"] = la
    report["_repaired_at_utc"] = datetime.utcnow().isoformat() + "Z"

    after_count = len(ticker_features)

    if not dry_run:
        write_json_atomic(report_path, report)

    return {
        "status": "repaired",
        "signal_date": signal_date_str,
        "ticker_features_before": before_count,
        "ticker_features_after": after_count,
        "features_per_ticker": len(feature_names),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=str, help="single week_*.json path")
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"repair all week_*.json in {DEFAULT_REPORT_DIR}",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.report and not args.all:
        parser.error("--report or --all is required")

    if args.all:
        reports = sorted(DEFAULT_REPORT_DIR.glob("week_*.json"))
        reports = [p for p in reports if "bak" not in p.name and "contaminated" not in p.name]
    else:
        reports = [Path(args.report)]

    overall_ok = True
    for path in reports:
        try:
            stats = repair_report(path, dry_run=args.dry_run)
            print(f"{path.name}: {stats}")
        except Exception as exc:
            print(f"{path.name}: ERROR {type(exc).__name__}: {exc}")
            overall_ok = False
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
