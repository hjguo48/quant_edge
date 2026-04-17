from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_parquet_atomic
from src.data.db.pit import get_prices_pit
from src.labels.forward_returns import compute_forward_returns

DEFAULT_FEATURES_PATH = "data/features/all_features_v5.parquet"
DEFAULT_IC_REPORT_PATH = "data/features/ic_screening_report_v5_60d.csv"
DEFAULT_LABEL_1D_PATH = "data/labels/forward_returns_1d.parquet"
DEFAULT_LABEL_5D_PATH = "data/labels/forward_returns_5d.parquet"
DEFAULT_REBALANCE_WEEKDAY = 4
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_AS_OF = date(2026, 4, 15)
THRESHOLDS = {1: 0.015, 5: 0.025}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    features_path = REPO_ROOT / args.features_path
    ic_report_path = REPO_ROOT / args.ic_report_path

    retained = load_retained_features(ic_report_path)
    if args.limit_features:
        retained = retained[: args.limit_features]
    print(f"Loaded {len(retained)} retained features from {ic_report_path}")

    feature_panel = load_feature_panel(
        features_path=features_path,
        retained_features=retained,
        rebalance_weekday=args.rebalance_weekday,
    )
    if feature_panel.empty:
        raise RuntimeError("No retained Friday features available for quick IC check.")
    print(
        "Feature panel: "
        f"rows={len(feature_panel):,}, dates={feature_panel['trade_date'].nunique():,}, "
        f"tickers={feature_panel['ticker'].nunique():,}, features={feature_panel['feature_name'].nunique():,}",
    )

    feature_start = feature_panel["trade_date"].min().date()
    feature_end = feature_panel["trade_date"].max().date()
    tickers = sorted(feature_panel["ticker"].unique().tolist())
    sign_by_feature = load_orientation_signs(ic_report_path, retained)

    summaries: list[dict[str, Any]] = []
    for horizon, label_path_arg in [(1, args.label_1d_path), (5, args.label_5d_path)]:
        label_path = REPO_ROOT / label_path_arg
        labels = load_or_build_labels(
            label_path=label_path,
            horizon=horizon,
            tickers=tickers,
            start_date=feature_start,
            end_date=feature_end,
            as_of=parse_date(args.as_of),
            benchmark_ticker=args.benchmark_ticker,
            rebalance_weekday=args.rebalance_weekday,
            label_buffer_days=args.label_buffer_days,
        )
        per_feature = compute_feature_ics(
            features=feature_panel,
            labels=labels,
            retained_features=retained,
            sign_by_feature=sign_by_feature,
        )
        summary = summarize_ics(horizon=horizon, per_feature=per_feature)
        summaries.append(summary)
        print_report(horizon=horizon, summary=summary, per_feature=per_feature)

    if args.output:
        output_path = REPO_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "features_path": str(features_path),
            "ic_report_path": str(ic_report_path),
            "retained_feature_count": len(retained),
            "rebalance_weekday": int(args.rebalance_weekday),
            "threshold_metric": "mean_abs_ic",
            "summaries": summaries,
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved quick IC summary to {output_path}")

    return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick all-sample 1D/5D IC check for retained feature sets.")
    parser.add_argument("--features-path", default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--ic-report-path", default=DEFAULT_IC_REPORT_PATH)
    parser.add_argument("--label-1d-path", default=DEFAULT_LABEL_1D_PATH)
    parser.add_argument("--label-5d-path", default=DEFAULT_LABEL_5D_PATH)
    parser.add_argument("--rebalance-weekday", type=int, default=DEFAULT_REBALANCE_WEEKDAY)
    parser.add_argument("--benchmark-ticker", default=DEFAULT_BENCHMARK_TICKER)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF.isoformat())
    parser.add_argument("--label-buffer-days", type=int, default=15)
    parser.add_argument("--limit-features", type=int, default=None, help="Optional debug limit for retained features.")
    parser.add_argument("--output", default=None, help="Optional JSON summary output path.")
    return parser.parse_args(argv)


def load_retained_features(ic_report_path: Path) -> list[str]:
    report = pd.read_csv(ic_report_path)
    retention_column = "retained" if "retained" in report.columns else "passed"
    if retention_column not in report.columns:
        raise RuntimeError(f"No retained/passed column found in {ic_report_path}")
    mask = coerce_bool_series(report[retention_column])
    retained = report.loc[mask, "feature_name"].astype(str).tolist()
    if not retained:
        raise RuntimeError(f"No retained features found in {ic_report_path}")
    return retained


def coerce_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def load_orientation_signs(ic_report_path: Path, retained_features: list[str]) -> dict[str, float]:
    report = pd.read_csv(ic_report_path)
    report = report.loc[report["feature_name"].isin(retained_features)].copy()
    if "window_signed_ic_mean" in report.columns:
        signed = pd.to_numeric(report["window_signed_ic_mean"], errors="coerce")
    elif "signed_ic" in report.columns:
        signed = pd.to_numeric(report["signed_ic"], errors="coerce")
    else:
        signed = pd.Series(np.nan, index=report.index)
    signs = np.sign(signed).replace(0, np.nan).fillna(1.0)
    return dict(zip(report["feature_name"].astype(str), signs.astype(float), strict=False))


def load_feature_panel(
    *,
    features_path: Path,
    retained_features: list[str],
    rebalance_weekday: int,
) -> pd.DataFrame:
    columns = ["ticker", "trade_date", "feature_name", "feature_value"]
    filters = [("feature_name", "in", retained_features)]
    if features_path.exists():
        features = pd.read_parquet(features_path, columns=columns, filters=filters)
    else:
        batch_dir = features_path.with_name(features_path.stem + "_batches")
        if not batch_dir.exists():
            raise FileNotFoundError(f"Neither {features_path} nor {batch_dir} exists.")
        frames = [
            pd.read_parquet(path, columns=columns, filters=filters)
            for path in sorted(batch_dir.glob("*.parquet"))
        ]
        features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    features["ticker"] = features["ticker"].astype(str).str.upper()
    features["trade_date"] = pd.to_datetime(features["trade_date"])
    features = features.loc[features["trade_date"].dt.weekday == rebalance_weekday].copy()
    features["feature_name"] = features["feature_name"].astype(str)
    features["feature_value"] = pd.to_numeric(features["feature_value"], errors="coerce")
    features.dropna(subset=["feature_value"], inplace=True)
    return features


def load_or_build_labels(
    *,
    label_path: Path,
    horizon: int,
    tickers: list[str],
    start_date: date,
    end_date: date,
    as_of: date,
    benchmark_ticker: str,
    rebalance_weekday: int,
    label_buffer_days: int,
) -> pd.DataFrame:
    if label_path.exists():
        labels = pd.read_parquet(label_path)
    else:
        price_end = min(end_date + timedelta(days=label_buffer_days), as_of)
        prices = get_prices_pit(
            tickers=list(dict.fromkeys([*tickers, benchmark_ticker.upper()])),
            start_date=start_date,
            end_date=price_end,
            as_of=datetime.combine(as_of, time.max, tzinfo=timezone.utc),
        )
        if prices.empty:
            raise RuntimeError(f"No PIT prices available to build {horizon}D labels.")
        labels = compute_forward_returns(prices, horizons=(horizon,), benchmark_ticker=benchmark_ticker)
        write_parquet_atomic(labels, label_path)

    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    labels = labels.loc[
        (labels["horizon"].astype(int) == int(horizon))
        & (labels["ticker"] != benchmark_ticker.upper())
        & (labels["trade_date"] >= pd.Timestamp(start_date))
        & (labels["trade_date"] <= pd.Timestamp(end_date))
        & (labels["trade_date"].dt.weekday == rebalance_weekday)
    ].copy()
    labels["excess_return"] = pd.to_numeric(labels["excess_return"], errors="coerce")
    labels.dropna(subset=["excess_return"], inplace=True)
    return labels[["ticker", "trade_date", "excess_return"]]


def compute_feature_ics(
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    retained_features: list[str],
    sign_by_feature: dict[str, float],
) -> pd.DataFrame:
    labels_indexed = labels.set_index(["trade_date", "ticker"])["excess_return"].sort_index()
    rows: list[dict[str, Any]] = []
    for feature_name in retained_features:
        feature_slice = features.loc[features["feature_name"] == feature_name, ["trade_date", "ticker", "feature_value"]]
        if feature_slice.empty:
            rows.append(
                {
                    "feature_name": feature_name,
                    "n_obs": 0,
                    "ic": np.nan,
                    "abs_ic": np.nan,
                    "oriented_ic": np.nan,
                },
            )
            continue
        feature_series = feature_slice.set_index(["trade_date", "ticker"])["feature_value"].sort_index()
        aligned = pd.concat([feature_series.rename("feature"), labels_indexed.rename("label")], axis=1, join="inner")
        aligned.dropna(inplace=True)
        ic = float(aligned["feature"].corr(aligned["label"])) if len(aligned) >= 3 else np.nan
        sign = sign_by_feature.get(feature_name, 1.0)
        rows.append(
            {
                "feature_name": feature_name,
                "n_obs": int(len(aligned)),
                "ic": ic,
                "abs_ic": abs(ic) if np.isfinite(ic) else np.nan,
                "oriented_ic": ic * sign if np.isfinite(ic) else np.nan,
            },
        )
    return pd.DataFrame(rows)


def summarize_ics(*, horizon: int, per_feature: pd.DataFrame) -> dict[str, Any]:
    threshold = THRESHOLDS[horizon]
    mean_ic = float(per_feature["ic"].mean(skipna=True))
    mean_abs_ic = float(per_feature["abs_ic"].mean(skipna=True))
    mean_oriented_ic = float(per_feature["oriented_ic"].mean(skipna=True))
    return {
        "horizon": int(horizon),
        "feature_count": int(len(per_feature)),
        "usable_feature_count": int(per_feature["ic"].notna().sum()),
        "mean_ic": mean_ic,
        "mean_abs_ic": mean_abs_ic,
        "mean_oriented_ic": mean_oriented_ic,
        "threshold": float(threshold),
        "passes_mean_abs_ic_threshold": bool(mean_abs_ic > threshold),
    }


def print_report(*, horizon: int, summary: dict[str, Any], per_feature: pd.DataFrame) -> None:
    print(f"\n{horizon}D quick IC check")
    print(
        f"mean_ic={summary['mean_ic']:.4f} "
        f"mean_abs_ic={summary['mean_abs_ic']:.4f} "
        f"mean_oriented_ic={summary['mean_oriented_ic']:.4f} "
        f"threshold={summary['threshold']:.4f} "
        f"pass={summary['passes_mean_abs_ic_threshold']}",
    )
    table = per_feature.sort_values("abs_ic", ascending=False).copy()
    for column in ["ic", "abs_ic", "oriented_ic"]:
        table[column] = table[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
    print(table[["feature_name", "n_obs", "ic", "abs_ic", "oriented_ic"]].to_string(index=False))


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


if __name__ == "__main__":
    raise SystemExit(main())
