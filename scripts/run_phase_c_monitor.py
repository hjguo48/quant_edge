"""Phase C (S1.13 + S1.15 + S1.16): Live monitoring suite.

S1.13 — PSI feature distribution monitor (raw pre-rank features)
S1.15 — Rolling IC + CUSUM change-point detection
S1.16 — Decile monotonicity calibration check

Usage:
    python scripts/run_phase_c_monitor.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

for env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(env_var, "1")

from loguru import logger
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from src.stats.psi import compute_feature_psi_report

DEFAULT_PREDICTIONS_PATH = "data/backtest/extended_walkforward_predictions.parquet"
DEFAULT_PRICES_PATH = "data/backtest/extended_walkforward_prices.parquet"
DEFAULT_FUSION_REPORT_PATH = "data/reports/fusion_analysis_60d.json"
DEFAULT_COMPARISON_REPORT_PATH = "data/reports/walkforward_comparison_60d.json"
DEFAULT_REPORT_PATH = "data/reports/phase_c_monitor.json"
BENCHMARK_TICKER = "SPY"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    logger.info("Phase C: Live monitoring suite (S1.13 + S1.15 + S1.16)")

    # --- S1.13: PSI Feature Distribution Monitor ---
    logger.info("=== S1.13: PSI Feature Distribution Monitor ===")
    psi_report = run_psi_monitor()

    # --- S1.15: Rolling IC + CUSUM ---
    logger.info("=== S1.15: Rolling IC + CUSUM Change-Point Detection ===")
    fusion_path = REPO_ROOT / args.fusion_report
    comparison_path = REPO_ROOT / args.comparison_report
    cusum_report = run_cusum_monitor(fusion_path, comparison_path)

    # --- S1.16: Decile Monotonicity ---
    logger.info("=== S1.16: Decile Monotonicity Calibration Check ===")
    predictions_df = pd.read_parquet(REPO_ROOT / args.predictions_path)
    prices_df = pd.read_parquet(REPO_ROOT / args.prices_path)
    decile_report = run_decile_monitor(predictions_df, prices_df, benchmark_ticker=BENCHMARK_TICKER)

    # --- Phase C Gate ---
    gate = evaluate_phase_c_gate(psi_report, cusum_report, decile_report)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": current_git_branch(),
        "script": "scripts/run_phase_c_monitor.py",
        "s1_13_psi": psi_report,
        "s1_15_cusum": cusum_report,
        "s1_16_decile": decile_report,
        "phase_c_gate": gate,
    }

    report_path = REPO_ROOT / args.report_path
    write_json_atomic(report_path, json_safe(report))
    logger.info("saved Phase C report to {}", report_path)
    logger.info("Phase C Gate: {}", "PASS" if gate["pass"] else "FAIL")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C monitoring suite")
    parser.add_argument("--predictions-path", default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--prices-path", default=DEFAULT_PRICES_PATH)
    parser.add_argument("--fusion-report", default=DEFAULT_FUSION_REPORT_PATH)
    parser.add_argument("--comparison-report", default=DEFAULT_COMPARISON_REPORT_PATH)
    parser.add_argument("--report-path", default=DEFAULT_REPORT_PATH)
    return parser.parse_args(argv)


# ===================================================================
# S1.13: PSI Feature Distribution Monitor
# ===================================================================

def run_psi_monitor(
    psi_threshold: float = 0.25,
    fill_rate_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compute PSI on raw pre-rank features from feature_store.

    Splits available data into reference (first 70%) and current (last 30%)
    periods to detect distribution drift.
    """
    from src.data.db.session import get_session_factory
    import sqlalchemy as sa

    sf = get_session_factory()
    with sf() as sess:
        # Get date range
        row = sess.execute(sa.text(
            "SELECT MIN(calc_date), MAX(calc_date), COUNT(DISTINCT calc_date) FROM feature_store"
        )).fetchone()
        min_date, max_date, n_dates = row

        if n_dates < 5:
            logger.warning("Insufficient feature_store data ({} dates) for PSI", n_dates)
            return {"status": "insufficient_data", "n_dates": int(n_dates), "alerts": []}

        # Get all distinct dates
        dates = [r[0] for r in sess.execute(
            sa.text("SELECT DISTINCT calc_date FROM feature_store ORDER BY calc_date")
        ).fetchall()]

        # Split: first 70% = reference, last 30% = current
        split_idx = int(len(dates) * 0.7)
        ref_dates = dates[:split_idx]
        cur_dates = dates[split_idx:]
        logger.info("PSI split: reference={} dates ({} to {}), current={} dates ({} to {})",
                    len(ref_dates), ref_dates[0], ref_dates[-1],
                    len(cur_dates), cur_dates[0], cur_dates[-1])

        # Get feature names
        feature_names = [r[0] for r in sess.execute(
            sa.text("SELECT DISTINCT feature_name FROM feature_store ORDER BY feature_name")
        ).fetchall()]

        # Load raw features as pivoted DataFrames
        ref_df = _load_feature_pivot(sess, ref_dates[0], ref_dates[-1])
        cur_df = _load_feature_pivot(sess, cur_dates[0], cur_dates[-1])

    logger.info("loaded reference={} rows, current={} rows, {} features",
                len(ref_df), len(cur_df), len(feature_names))

    psi_results = compute_feature_psi_report(
        reference_df=ref_df,
        current_df=cur_df,
        feature_columns=feature_names,
        psi_alert_threshold=psi_threshold,
        fill_rate_change_threshold=fill_rate_threshold,
    )

    alerts = [r for r in psi_results if r.get("psi_alert") or r.get("fill_rate_alert")]
    psi_values = [r["psi"] for r in psi_results if math.isfinite(r.get("psi", float("nan")))]

    logger.info("PSI results: {} features, {} alerts, mean PSI={:.4f}",
                len(psi_results), len(alerts),
                float(np.mean(psi_values)) if psi_values else 0.0)

    for a in alerts:
        logger.warning("  ALERT: {} PSI={:.4f} fill_rate_delta={:.4f}",
                       a["feature"], a["psi"], a["fill_rate_delta"])

    return {
        "status": "ok",
        "reference_period": f"{ref_dates[0]} to {ref_dates[-1]}",
        "current_period": f"{cur_dates[0]} to {cur_dates[-1]}",
        "n_features": len(psi_results),
        "n_psi_alerts": sum(1 for r in psi_results if r.get("psi_alert")),
        "n_fill_rate_alerts": sum(1 for r in psi_results if r.get("fill_rate_alert")),
        "mean_psi": float(np.mean(psi_values)) if psi_values else 0.0,
        "max_psi": float(np.max(psi_values)) if psi_values else 0.0,
        "alerts": [
            {"feature": a["feature"], "psi": a["psi"],
             "fill_rate_delta": a["fill_rate_delta"],
             "psi_alert": a["psi_alert"], "fill_rate_alert": a["fill_rate_alert"]}
            for a in alerts
        ],
        "per_feature": psi_results,
        "thresholds": {"psi": psi_threshold, "fill_rate_change": fill_rate_threshold},
    }


def _load_feature_pivot(sess: Any, start_date: date, end_date: date) -> pd.DataFrame:
    """Load raw feature values from feature_store as a pivoted DataFrame."""
    import sqlalchemy as sa

    rows = sess.execute(sa.text("""
        SELECT calc_date, ticker, feature_name, feature_value
        FROM feature_store
        WHERE calc_date >= :start AND calc_date <= :end
    """), {"start": start_date, "end": end_date}).fetchall()

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=["calc_date", "ticker", "feature_name", "feature_value"])
    df["feature_value"] = pd.to_numeric(df["feature_value"], errors="coerce")
    pivoted = df.pivot_table(
        index=["calc_date", "ticker"],
        columns="feature_name",
        values="feature_value",
        aggfunc="first",
    )
    pivoted.columns.name = None
    return pivoted.reset_index(drop=True)


# ===================================================================
# S1.15: Rolling IC + CUSUM Change-Point Detection
# ===================================================================

def run_cusum_monitor(
    fusion_report_path: Path,
    comparison_report_path: Path,
    ic_alert_threshold: float = 0.03,
    consecutive_alert_weeks: int = 3,
) -> dict[str, Any]:
    """Analyze IC drift using per-window IC from walk-forward reports."""
    fusion_data = json.loads(fusion_report_path.read_text())
    comparison_data = json.loads(comparison_report_path.read_text())

    # Extract per-window IC series
    model_ics: dict[str, list[float]] = {"ridge": [], "xgboost": [], "lightgbm": [], "fusion": []}
    window_ids: list[str] = []

    for w in fusion_data["per_window"]:
        window_ids.append(w["window_id"])
        model_ics["ridge"].append(float(w.get("ridge_ic", 0.0)))
        model_ics["xgboost"].append(float(w.get("xgboost_ic", 0.0)))
        model_ics["lightgbm"].append(float(w.get("lightgbm_ic", 0.0)))
        model_ics["fusion"].append(float(w.get("ic_weighted_fusion_ic", 0.0)))

    # CUSUM analysis for each model
    cusum_results: dict[str, dict[str, Any]] = {}
    alerts: list[dict[str, Any]] = []

    for model_name, ics in model_ics.items():
        ic_array = np.array(ics, dtype=float)
        cusum = _compute_cusum(ic_array)
        consecutive_low = _count_consecutive_below(ic_array, ic_alert_threshold)
        rolling_mean = _rolling_mean(ic_array, window=3)

        is_alert = consecutive_low >= consecutive_alert_weeks
        cusum_results[model_name] = {
            "window_ics": {wid: float(ic) for wid, ic in zip(window_ids, ics)},
            "mean_ic": float(ic_array.mean()),
            "std_ic": float(ic_array.std()) if len(ic_array) > 1 else 0.0,
            "min_ic": float(ic_array.min()),
            "max_ic": float(ic_array.max()),
            "cusum_values": cusum.tolist(),
            "cusum_max_deviation": float(np.abs(cusum).max()),
            "rolling_mean_3w": rolling_mean.tolist(),
            "consecutive_below_threshold": int(consecutive_low),
            "alert": is_alert,
        }

        if is_alert:
            alerts.append({
                "model": model_name,
                "consecutive_low_ic_weeks": int(consecutive_low),
                "recent_ics": [float(x) for x in ic_array[-consecutive_alert_weeks:]],
            })
            logger.warning("  CUSUM ALERT: {} has {} consecutive windows with IC < {}",
                          model_name, consecutive_low, ic_alert_threshold)
        else:
            logger.info("  {}: mean_ic={:.4f}, min={:.4f}, max={:.4f}, consecutive_low={}",
                       model_name, ic_array.mean(), ic_array.min(), ic_array.max(), consecutive_low)

    # CUSUM change-point detection on fusion IC
    fusion_ics = np.array(model_ics["fusion"])
    change_point = _detect_cusum_change_point(fusion_ics)

    return {
        "n_windows": len(window_ids),
        "window_ids": window_ids,
        "models": cusum_results,
        "fusion_change_point": change_point,
        "n_alerts": len(alerts),
        "alerts": alerts,
        "thresholds": {
            "ic_alert": ic_alert_threshold,
            "consecutive_weeks": consecutive_alert_weeks,
        },
    }


def _compute_cusum(values: np.ndarray) -> np.ndarray:
    """Cumulative sum of deviations from the mean (CUSUM control chart)."""
    mean = values.mean()
    return np.cumsum(values - mean)


def _count_consecutive_below(values: np.ndarray, threshold: float) -> int:
    """Count longest consecutive run of values below threshold (from the end)."""
    count = 0
    for v in reversed(values):
        if v < threshold:
            count += 1
        else:
            break
    return count


def _rolling_mean(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Simple rolling mean with NaN padding for initial values."""
    result = np.full_like(values, float("nan"))
    for i in range(window - 1, len(values)):
        result[i] = values[i - window + 1 : i + 1].mean()
    return result


def _detect_cusum_change_point(values: np.ndarray, drift: float = 0.5) -> dict[str, Any]:
    """Detect change point using CUSUM with a drift parameter.

    Returns the index and direction of the most significant shift.
    """
    n = len(values)
    if n < 4:
        return {"detected": False, "reason": "insufficient_data"}

    mean = values.mean()
    std = values.std()
    if std < 1e-10:
        return {"detected": False, "reason": "zero_variance"}

    # Standardized CUSUM
    z = (values - mean) / std
    cusum_pos = np.zeros(n)
    cusum_neg = np.zeros(n)

    for i in range(1, n):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + z[i] - drift)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - z[i] - drift)

    # Threshold for detection (h = 4 is common)
    h = 4.0
    pos_breach = cusum_pos > h
    neg_breach = cusum_neg > h

    if pos_breach.any():
        idx = int(np.argmax(pos_breach))
        return {
            "detected": True,
            "direction": "upward_shift",
            "change_index": idx,
            "cusum_value": float(cusum_pos[idx]),
        }
    elif neg_breach.any():
        idx = int(np.argmax(neg_breach))
        return {
            "detected": True,
            "direction": "downward_shift",
            "change_index": idx,
            "cusum_value": float(cusum_neg[idx]),
        }

    return {
        "detected": False,
        "max_pos_cusum": float(cusum_pos.max()),
        "max_neg_cusum": float(cusum_neg.max()),
        "threshold": h,
    }


# ===================================================================
# S1.16: Decile Monotonicity Calibration Check
# ===================================================================

def run_decile_monitor(
    predictions_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    benchmark_ticker: str = "SPY",
    horizon_days: int = 60,
    monotonicity_threshold: float = 0.80,
) -> dict[str, Any]:
    """Check decile monotonicity: top decile should beat bottom decile.

    For each test period (signal date), rank stocks into deciles by model score,
    compute forward returns for each decile, and check:
    - top decile > bottom decile in ≥ 80% of periods
    - monotonic ordering (higher score → higher return) frequency
    """
    # Prepare forward returns
    prices_wide = prices_df.pivot_table(index="trade_date", columns="ticker", values="close")
    prices_wide.index = pd.to_datetime(prices_wide.index)

    # Benchmark returns
    if benchmark_ticker in prices_wide.columns:
        benchmark_series = prices_wide[benchmark_ticker]
        stock_prices = prices_wide.drop(columns=[benchmark_ticker], errors="ignore")
    else:
        benchmark_series = None
        stock_prices = prices_wide

    results_by_window: list[dict[str, Any]] = []

    for window_id, window_df in predictions_df.groupby("window_id"):
        for signal_date, date_df in window_df.groupby("trade_date"):
            signal_dt = pd.Timestamp(signal_date)
            scores = date_df.set_index("ticker")["score"].dropna()

            if len(scores) < 20:
                continue

            # Assign deciles (1=lowest score, 10=highest score)
            decile_labels = pd.qcut(scores, q=10, labels=False, duplicates="drop") + 1

            # Compute forward return over horizon
            # 60 calendar days ≈ 42 trading days (5/7 ratio)
            target_calendar_date = signal_dt + pd.Timedelta(days=horizon_days)
            future_dates = stock_prices.index[stock_prices.index > signal_dt]
            if len(future_dates) == 0:
                continue

            # Find the trading date closest to (but not exceeding) target calendar date
            valid_future = future_dates[future_dates <= target_calendar_date]
            if len(valid_future) == 0:
                continue
            target_date = valid_future[-1]

            # Forward return for each stock
            current_prices = stock_prices.loc[signal_dt] if signal_dt in stock_prices.index else None
            future_prices = stock_prices.loc[target_date] if target_date in stock_prices.index else None

            if current_prices is None or future_prices is None:
                continue

            common_tickers = list(set(scores.index) & set(current_prices.dropna().index) & set(future_prices.dropna().index))
            if len(common_tickers) < 20:
                continue

            fwd_returns = (future_prices[common_tickers] / current_prices[common_tickers] - 1.0)

            # Benchmark excess
            if benchmark_series is not None and signal_dt in benchmark_series.index and target_date in benchmark_series.index:
                bench_ret = float(benchmark_series[target_date] / benchmark_series[signal_dt] - 1.0)
                fwd_excess = fwd_returns - bench_ret
            else:
                fwd_excess = fwd_returns

            # Group by decile
            decile_excess = pd.DataFrame({
                "decile": decile_labels.reindex(common_tickers),
                "excess_return": fwd_excess,
            }).dropna()

            decile_means = decile_excess.groupby("decile")["excess_return"].mean()

            if len(decile_means) < 2:
                continue

            top_decile = decile_means.get(decile_means.index.max(), float("nan"))
            bottom_decile = decile_means.get(decile_means.index.min(), float("nan"))

            # Check monotonicity (Spearman rank correlation between decile rank and mean return)
            if len(decile_means) >= 3:
                from scipy.stats import spearmanr
                mono_corr, _ = spearmanr(decile_means.index.to_numpy(), decile_means.values)
            else:
                mono_corr = float("nan")

            results_by_window.append({
                "window_id": str(window_id),
                "signal_date": str(signal_date),
                "top_decile_excess": float(top_decile),
                "bottom_decile_excess": float(bottom_decile),
                "top_beats_bottom": bool(top_decile > bottom_decile),
                "long_short_spread": float(top_decile - bottom_decile),
                "monotonicity_corr": float(mono_corr) if math.isfinite(mono_corr) else None,
                "n_deciles": int(len(decile_means)),
                "n_stocks": len(common_tickers),
            })

    if not results_by_window:
        logger.warning("No valid decile results produced")
        return {"status": "no_data", "pass": False}

    df = pd.DataFrame(results_by_window)

    # Evaluate at window level (not individual signal dates):
    # For each walk-forward window, does the average top-decile exceed bottom-decile?
    window_agg = df.groupby("window_id").agg(
        mean_spread=("long_short_spread", "mean"),
        top_beats_count=("top_beats_bottom", "sum"),
        n_periods=("top_beats_bottom", "count"),
    )
    window_agg["window_top_beats_bottom"] = window_agg["mean_spread"] > 0
    window_level_rate = float(window_agg["window_top_beats_bottom"].mean())

    # Also track per-period rate for diagnostics
    period_level_rate = float(df["top_beats_bottom"].mean())
    mean_spread = float(df["long_short_spread"].mean())
    mean_mono = float(df["monotonicity_corr"].dropna().mean()) if df["monotonicity_corr"].notna().any() else 0.0

    # Gate uses window-level rate (more robust with 60D overlapping horizons)
    passes = window_level_rate >= monotonicity_threshold

    logger.info("Decile monotonicity (window-level): {}/{} windows top > bottom ({:.1%}), threshold={:.0%}",
                int(window_agg["window_top_beats_bottom"].sum()), len(window_agg),
                window_level_rate, monotonicity_threshold)
    logger.info("  period-level: {}/{} ({:.1%}), mean spread={:.4f}, mean mono corr={:.4f}",
                int(df["top_beats_bottom"].sum()), len(df), period_level_rate,
                mean_spread, mean_mono)

    # Per-window summary
    window_summary = {}
    for wid, wdf in df.groupby("window_id"):
        window_summary[str(wid)] = {
            "n_periods": len(wdf),
            "top_beats_bottom_rate": float(wdf["top_beats_bottom"].mean()),
            "mean_spread": float(wdf["long_short_spread"].mean()),
            "mean_monotonicity_corr": float(wdf["monotonicity_corr"].dropna().mean())
            if wdf["monotonicity_corr"].notna().any() else None,
        }

    return {
        "status": "ok",
        "pass": passes,
        "window_level_rate": window_level_rate,
        "period_level_rate": period_level_rate,
        "threshold": monotonicity_threshold,
        "n_windows": len(window_agg),
        "n_periods": len(df),
        "mean_long_short_spread": mean_spread,
        "mean_monotonicity_corr": mean_mono,
        "per_window": window_summary,
        "period_details": results_by_window,
    }


# ===================================================================
# Phase C Gate
# ===================================================================

def evaluate_phase_c_gate(
    psi_report: dict[str, Any],
    cusum_report: dict[str, Any],
    decile_report: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Phase C gate criteria.

    Pass conditions:
    - PSI: no features with PSI > 0.25 alert (or acceptable count)
    - CUSUM: no model with 3+ consecutive IC < 0.03 (fusion especially)
    - Decile: top > bottom in ≥ 80% of periods
    """
    psi_pass = psi_report.get("n_psi_alerts", 0) == 0 or psi_report.get("status") == "insufficient_data"
    # CUSUM gate: fusion IC is the primary metric (individual model alerts are
    # informational — adaptive fusion already handles weak models by zeroing weights)
    fusion_cusum = cusum_report.get("models", {}).get("fusion", {})
    cusum_pass = not fusion_cusum.get("alert", False)
    decile_pass = decile_report.get("pass", False)

    overall = psi_pass and cusum_pass and decile_pass

    details = {
        "psi_pass": psi_pass,
        "psi_alerts": psi_report.get("n_psi_alerts", 0),
        "cusum_pass": cusum_pass,
        "cusum_alerts": cusum_report.get("n_alerts", 0),
        "decile_pass": decile_pass,
        "decile_window_rate": decile_report.get("window_level_rate", 0.0),
    }

    logger.info("Phase C Gate: PSI={} CUSUM={} Decile={} → {}",
                "PASS" if psi_pass else "FAIL",
                "PASS" if cusum_pass else "FAIL",
                "PASS" if decile_pass else "FAIL",
                "PASS" if overall else "FAIL")

    return {"pass": overall, "details": details}


if __name__ == "__main__":
    raise SystemExit(main())
