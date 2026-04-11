from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

from loguru import logger
import pandas as pd
import sqlalchemy as sa

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.db.models import FeatureStore, Stock
from src.data.db.session import get_session_factory

MODEL_NAMES = ("ridge", "xgboost", "lightgbm")
DEFAULT_REPORT_PATH = REPO_ROOT / "data/reports/greyscale/week_02.json"
DEFAULT_FOCUS_TICKER = "BKNG"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose greyscale model consistency and single-name outliers.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--focus-ticker", default=DEFAULT_FOCUS_TICKER)
    parser.add_argument("--top-n", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report if args.report.is_absolute() else REPO_ROOT / args.report
    report = json.loads(report_path.read_text())

    score_frame = build_score_frame(report)
    print("=== Overall Rank Correlation ===")
    print(score_frame.corr(method="spearman").round(4).to_string())
    print()

    sectors = load_sector_map(tuple(score_frame.index.astype(str)))
    print("=== Sector Rank Correlation ===")
    print(format_sector_rank_correlation(score_frame, sectors))
    print()

    print("=== Top/Bottom Overlap ===")
    print(format_overlap_table(score_frame, top_n=args.top_n))
    print()

    print(f"=== {args.focus_ticker.upper()} Score Analysis ===")
    print(format_focus_score_analysis(score_frame, args.focus_ticker.upper()))
    print()

    print(f"=== {args.focus_ticker.upper()} Feature Snapshot ===")
    print(format_focus_feature_analysis(report, args.focus_ticker.upper()))
    return 0


def build_score_frame(report: dict[str, object]) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    score_vectors = report.get("score_vectors", {})
    for model_name in MODEL_NAMES:
        values = score_vectors.get(model_name, {})
        series = pd.Series(values, dtype=float)
        series.index = series.index.astype(str).str.upper()
        columns[model_name] = series
    frame = pd.DataFrame(columns).dropna(how="all")
    frame.index.name = "ticker"
    return frame.sort_index()


def load_sector_map(tickers: tuple[str, ...]) -> dict[str, str]:
    if not tickers:
        return {}
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(
            sa.select(Stock.ticker, Stock.sector).where(Stock.ticker.in_(tickers)),
        ).all()
    return {
        str(ticker).upper(): str(sector or "Unknown")
        for ticker, sector in rows
    }


def format_sector_rank_correlation(score_frame: pd.DataFrame, sectors: dict[str, str]) -> str:
    rows: list[dict[str, object]] = []
    sector_series = pd.Series({ticker: sectors.get(str(ticker), "Unknown") for ticker in score_frame.index}, dtype=object)
    for sector, tickers in sector_series.groupby(sector_series):
        sector_frame = score_frame.loc[score_frame.index.intersection(tickers.index)].dropna(how="all")
        if len(sector_frame) < 5:
            continue
        corr = sector_frame.corr(method="spearman")
        rows.append(
            {
                "sector": str(sector),
                "count": int(len(sector_frame)),
                "ridge_xgboost": float(corr.loc["ridge", "xgboost"]),
                "ridge_lightgbm": float(corr.loc["ridge", "lightgbm"]),
                "xgboost_lightgbm": float(corr.loc["xgboost", "lightgbm"]),
            },
        )
    if not rows:
        return "No sector groups with enough names."
    return pd.DataFrame(rows).sort_values("sector").round(4).to_string(index=False)


def format_overlap_table(score_frame: pd.DataFrame, *, top_n: int) -> str:
    rows: list[dict[str, object]] = []
    for left_idx, left_name in enumerate(MODEL_NAMES):
        for right_name in MODEL_NAMES[left_idx + 1 :]:
            left = score_frame[left_name].dropna().sort_values(ascending=False)
            right = score_frame[right_name].dropna().sort_values(ascending=False)
            common = left.index.intersection(right.index)
            if common.empty:
                continue
            left = left.reindex(common)
            right = right.reindex(common)
            top_left = set(left.head(top_n).index)
            top_right = set(right.head(top_n).index)
            bottom_left = set(left.tail(top_n).index)
            bottom_right = set(right.tail(top_n).index)
            rows.append(
                {
                    "pair": f"{left_name}/{right_name}",
                    "top_overlap_ratio": len(top_left & top_right) / float(top_n),
                    "bottom_overlap_ratio": len(bottom_left & bottom_right) / float(top_n),
                },
            )
    if not rows:
        return "No overlap rows available."
    return pd.DataFrame(rows).round(4).to_string(index=False)


def format_focus_score_analysis(score_frame: pd.DataFrame, ticker: str) -> str:
    if ticker not in score_frame.index:
        return f"{ticker} is not present in the score frame."
    rows = []
    for model_name in MODEL_NAMES:
        series = score_frame[model_name].dropna().sort_values(ascending=False)
        rank = int(series.index.get_loc(ticker)) + 1 if ticker in series.index else None
        rows.append(
            {
                "model": model_name,
                "score": float(score_frame.at[ticker, model_name]) if pd.notna(score_frame.at[ticker, model_name]) else None,
                "rank": rank,
            },
        )
    return pd.DataFrame(rows).to_string(index=False)


def format_focus_feature_analysis(report: dict[str, object], ticker: str) -> str:
    signal_date = date.fromisoformat(str(report["live_outputs"]["signal_date"]))
    batch_id = str(report["feature_pipeline"]["batch_id"])
    feature_frame = load_feature_snapshot(signal_date=signal_date, batch_id=batch_id, ticker=ticker)
    if feature_frame.empty:
        return f"No feature snapshot found for {ticker}."
    return feature_frame.sort_values("feature_name").to_string(index=False)


def load_feature_snapshot(*, signal_date: date, batch_id: str, ticker: str) -> pd.DataFrame:
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(
            sa.select(FeatureStore.feature_name, FeatureStore.feature_value)
            .where(FeatureStore.ticker == ticker)
            .where(FeatureStore.calc_date == signal_date)
            .where(FeatureStore.batch_id == batch_id),
        ).all()
    frame = pd.DataFrame(rows, columns=["feature_name", "feature_value"])
    if not frame.empty:
        frame["feature_value"] = pd.to_numeric(frame["feature_value"], errors="coerce")
        return frame

    fallback_path = REPO_ROOT / "data/features/all_features.parquet"
    if not fallback_path.exists():
        return pd.DataFrame(columns=["feature_name", "feature_value"])
    fallback = pd.read_parquet(fallback_path, columns=["ticker", "trade_date", "feature_name", "feature_value"])
    fallback["ticker"] = fallback["ticker"].astype(str).str.upper()
    fallback["trade_date"] = pd.to_datetime(fallback["trade_date"]).dt.date
    fallback = fallback.loc[
        (fallback["ticker"] == ticker)
        & (fallback["trade_date"] == signal_date),
        ["feature_name", "feature_value"],
    ].copy()
    fallback["feature_value"] = pd.to_numeric(fallback["feature_value"], errors="coerce")
    return fallback


if __name__ == "__main__":
    raise SystemExit(main())
