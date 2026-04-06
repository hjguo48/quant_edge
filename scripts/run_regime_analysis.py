from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
import sqlalchemy as sa

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from src.data.db.session import get_session_factory
from src.data.sources.fred import MACRO_SERIES_TABLE

DEFAULT_COMPARISON_REPORT = "data/reports/walkforward_comparison_60d.json"
DEFAULT_OUTPUT_PATH = "data/reports/regime_analysis_60d.json"
DEFAULT_SERIES_ID = "VIXCLS"
DEFAULT_AS_OF = date(2026, 3, 31)
DEFAULT_LOW_THRESHOLD = 20.0
DEFAULT_HIGH_THRESHOLD = 30.0
MODEL_NAMES = ("ridge", "xgboost", "lightgbm")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    comparison_path = REPO_ROOT / args.comparison_report
    output_path = REPO_ROOT / args.output
    payload = json.loads(comparison_path.read_text())
    windows = select_windows(payload.get("windows", []), limit=args.window_limit)
    if not windows:
        raise RuntimeError(f"No windows available in {comparison_path}.")

    as_of = parse_date(args.as_of)
    test_start = min(extract_window_dates(window)["test_start"] for window in windows)
    test_end = max(extract_window_dates(window)["test_end"] for window in windows)
    vix_history = load_macro_series(
        series_id=args.series_id.upper(),
        start_date=test_start,
        end_date=test_end,
        as_of=as_of,
    )
    if vix_history.empty:
        raise RuntimeError(f"No PIT macro observations found for {args.series_id}.")

    per_window = build_window_records(
        windows=windows,
        vix_history=vix_history,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    regime_stats = build_regime_stats(
        per_window,
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
    )
    regime_weights = derive_regime_weights(regime_stats)
    regime_adjusted_ic = build_regime_adjusted_ic(per_window, regime_weights)

    report = {
        "config": {
            "comparison_report": str(comparison_path),
            "series_id": args.series_id.upper(),
            "as_of": as_of.isoformat(),
            "low_vix_threshold": float(args.low_threshold),
            "high_vix_threshold": float(args.high_threshold),
            "analysis_granularity": "test_window_mean_vix",
        },
        "windows": per_window,
        "vix_regime_stats": regime_stats,
        "regime_weights": regime_weights,
        "regime_adjusted_ic": regime_adjusted_ic,
    }
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved regime analysis report to {}", output_path)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 60D walk-forward window ICs by VIX regime and propose regime weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--comparison-report", default=DEFAULT_COMPARISON_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--series-id", default=DEFAULT_SERIES_ID)
    parser.add_argument("--as-of", default=DEFAULT_AS_OF.isoformat())
    parser.add_argument("--low-threshold", type=float, default=DEFAULT_LOW_THRESHOLD)
    parser.add_argument("--high-threshold", type=float, default=DEFAULT_HIGH_THRESHOLD)
    parser.add_argument("--window-limit", type=int)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
    )


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def select_windows(windows: list[dict[str, Any]], *, limit: int | None) -> list[dict[str, Any]]:
    selected = list(windows)
    if limit is not None:
        if limit <= 0:
            raise ValueError("window_limit must be positive.")
        selected = selected[:limit]
    return selected


def extract_window_dates(window: dict[str, Any]) -> dict[str, date]:
    nested = window.get("dates", {})
    return {
        "train_start": extract_date_field(window, nested, "train_start"),
        "train_end": extract_date_field(window, nested, "train_end"),
        "validation_start": extract_date_field(window, nested, "validation_start", aliases=("val_start",)),
        "validation_end": extract_date_field(window, nested, "validation_end", aliases=("val_end",)),
        "test_start": extract_date_field(window, nested, "test_start"),
        "test_end": extract_date_field(window, nested, "test_end"),
    }


def extract_date_field(
    window: dict[str, Any],
    nested: dict[str, Any],
    name: str,
    *,
    aliases: tuple[str, ...] = (),
) -> date:
    for candidate in (name, *aliases):
        if candidate in window:
            return parse_date(str(window[candidate]))
        if candidate in nested:
            return parse_date(str(nested[candidate]))
    raise KeyError(f"Window is missing date field {name!r}.")


def load_macro_series(
    *,
    series_id: str,
    start_date: date,
    end_date: date,
    as_of: date,
) -> pd.Series:
    ranked = (
        sa.select(
            MACRO_SERIES_TABLE.c.series_id,
            MACRO_SERIES_TABLE.c.observation_date,
            MACRO_SERIES_TABLE.c.value,
            sa.func.row_number()
            .over(
                partition_by=(MACRO_SERIES_TABLE.c.series_id, MACRO_SERIES_TABLE.c.observation_date),
                order_by=(MACRO_SERIES_TABLE.c.knowledge_time.desc(), MACRO_SERIES_TABLE.c.id.desc()),
            )
            .label("row_num"),
        )
        .where(
            MACRO_SERIES_TABLE.c.series_id == series_id,
            MACRO_SERIES_TABLE.c.observation_date >= start_date,
            MACRO_SERIES_TABLE.c.observation_date <= end_date,
            MACRO_SERIES_TABLE.c.knowledge_time <= datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc),
        )
    ).subquery()

    statement = (
        sa.select(ranked.c.observation_date, ranked.c.value)
        .where(ranked.c.row_num == 1)
        .order_by(ranked.c.observation_date)
    )

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).fetchall()

    if not rows:
        return pd.Series(dtype=float, name=series_id)

    frame = pd.DataFrame(rows, columns=["observation_date", "value"])
    frame["observation_date"] = pd.to_datetime(frame["observation_date"]).dt.date
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return pd.Series(
        frame["value"].to_numpy(dtype=float),
        index=pd.Index(frame["observation_date"], name="observation_date"),
        name=series_id,
    ).sort_index()


def build_window_records(
    *,
    windows: list[dict[str, Any]],
    vix_history: pd.Series,
    low_threshold: float,
    high_threshold: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for window in windows:
        dates = extract_window_dates(window)
        test_slice = vix_history.loc[(vix_history.index >= dates["test_start"]) & (vix_history.index <= dates["test_end"])]
        mean_vix = float(test_slice.mean()) if not test_slice.empty else float("nan")
        regime = classify_vix(mean_vix, low_threshold=low_threshold, high_threshold=high_threshold)
        record = {
            "window_id": str(window.get("window_id", f"W{len(records) + 1}")),
            "test_start": dates["test_start"].isoformat(),
            "test_end": dates["test_end"].isoformat(),
            "n_vix_dates": int(test_slice.dropna().shape[0]),
            "mean_vix": mean_vix,
            "regime": regime,
            "results": {},
        }
        for model_name in MODEL_NAMES:
            result = window.get("results", {}).get(model_name)
            if not result:
                continue
            record["results"][model_name] = {
                "test_ic": extract_metric(result, "ic"),
                "test_rank_ic": extract_metric(result, "rank_ic"),
                "test_icir": extract_metric(result, "icir"),
                "test_hit_rate": extract_metric(result, "hit_rate"),
            }
        records.append(record)
    return records


def extract_metric(result: dict[str, Any], metric_name: str) -> float:
    if "test_metrics" in result and metric_name in result["test_metrics"]:
        return float(result["test_metrics"][metric_name])
    direct_name = f"test_{metric_name}"
    if direct_name in result:
        return float(result[direct_name])
    if metric_name in result:
        return float(result[metric_name])
    return float("nan")


def classify_vix(value: float, *, low_threshold: float, high_threshold: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < low_threshold:
        return "low"
    if value < high_threshold:
        return "mid"
    return "high"


def build_regime_stats(
    per_window: list[dict[str, Any]],
    *,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    labels = {
        "low": f"<{low_threshold:g}",
        "mid": f"{low_threshold:g}-{high_threshold:g}",
        "high": f">{high_threshold:g}",
        "unknown": "unknown",
    }
    for regime_name, label in labels.items():
        regime_windows = [window for window in per_window if window["regime"] == regime_name]
        model_stats = {
            model_name: mean_ignore_nan(
                [window["results"].get(model_name, {}).get("test_ic", float("nan")) for window in regime_windows],
            )
            for model_name in MODEL_NAMES
        }
        stats[regime_name] = {
            "vix_range": label,
            "n_windows": int(len(regime_windows)),
            "n_dates": int(sum(window["n_vix_dates"] for window in regime_windows)),
            "mean_vix": mean_ignore_nan([window["mean_vix"] for window in regime_windows]),
            **{f"{model_name}_ic": value for model_name, value in model_stats.items()},
        }
    return stats


def derive_regime_weights(regime_stats: dict[str, Any]) -> dict[str, float]:
    def regime_mean_ic(regime_name: str) -> float:
        values = [
            float(regime_stats.get(regime_name, {}).get(f"{model_name}_ic", float("nan")))
            for model_name in MODEL_NAMES
        ]
        return mean_ignore_nan(values)

    low_score = regime_mean_ic("low")
    mid_score = regime_mean_ic("mid")
    high_score = regime_mean_ic("high")

    mid_weight = 1.0
    if np.isfinite(mid_score):
        if mid_score < 0:
            mid_weight = 0.6
        elif np.isfinite(low_score) and mid_score < 0.75 * low_score:
            mid_weight = 0.8

    high_weight = 0.8
    if np.isfinite(high_score):
        if high_score < 0:
            high_weight = 0.4
        elif np.isfinite(low_score) and high_score < 0.5 * low_score:
            high_weight = 0.6

    return {
        "low": 1.0,
        "mid": float(mid_weight),
        "high": float(high_weight),
        "unknown": 1.0,
    }


def build_regime_adjusted_ic(
    per_window: list[dict[str, Any]],
    regime_weights: dict[str, float],
) -> dict[str, float]:
    adjusted: dict[str, float] = {}
    for model_name in MODEL_NAMES:
        weighted_values: list[float] = []
        weights: list[float] = []
        for window in per_window:
            metric = window["results"].get(model_name, {}).get("test_ic", float("nan"))
            if not np.isfinite(metric):
                continue
            weight = float(regime_weights.get(window["regime"], 1.0)) * max(window["n_vix_dates"], 1)
            weighted_values.append(float(metric))
            weights.append(weight)
        adjusted[model_name] = weighted_average(weighted_values, weights)
    return adjusted


def weighted_average(values: list[float], weights: list[float]) -> float:
    if not values or not weights:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(arr) & np.isfinite(w) & (w > 0)
    if not valid.any():
        return float("nan")
    return float(np.average(arr[valid], weights=w[valid]))


def mean_ignore_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, np.generic):
        item = value.item()
        if isinstance(item, float) and (np.isnan(item) or np.isinf(item)):
            return None
        return item
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
