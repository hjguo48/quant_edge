"""Week 4 Task 9: Gate verification — coverage / feature quality / data-readiness IC / per-reason.

CLI:
    uv run python scripts/run_week4_gate_verification.py \
        --config configs/research/week4_trades_sampling.yaml \
        --features data/features/trade_microstructure_features.parquet \
        --labels data/features/forward_labels_5d.parquet \
        --output data/reports/week4/gate_summary.json

Four gates (sign-aware per plan v3.2):
    1. Coverage Gate — trades_sampling_state completed / (total - skipped_holiday)
    2. Feature Quality Gate — per-feature missing_rate + outlier_rate
    3. Data-readiness IC Gate (diagnostic, not alpha adoption):
       - Split feature date range into n_windows (default 11) equal slices
       - Per feature: mean_ic, abs_t_stat, positive_windows, negative_windows
       - sign_consistent_windows = max(positive_windows, negative_windows) with direction from mean_ic sign
       - Feature passes iff |mean_ic| >= ic_threshold AND abs_t_stat >= abs_tstat_threshold
         AND sign_consistent_windows >= sign_consistent_windows_min
       - Gate passes iff >= min_passing_features pass
    4. Per-reason IC breakdown — report top-1 strongest reason (earnings / gap / weak_window)

Labels parquet expected schema: event_date, ticker, forward_excess_return_5d (or --label-col override).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sqlalchemy import func, select
from sqlalchemy.orm import Session

try:
    import scripts.preflight_trades_estimator as preflight_trades_estimator
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    import preflight_trades_estimator as preflight_trades_estimator

from src.config.week4_trades import Week4TradesConfig
from src.data.db.models import TradesSamplingState
from src.data.db.session import get_session_factory
from src.features.trade_microstructure import TRADE_MICROSTRUCTURE_FEATURE_NAMES

DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_FEATURES_PATH = Path("data/features/trade_microstructure_features.parquet")
DEFAULT_OUTPUT_PATH = Path("data/reports/week4/gate_summary.json")
DEFAULT_LABEL_COL = "forward_excess_return_5d"
DEFAULT_N_WINDOWS = 11
OUTLIER_ZSCORE_THRESHOLD = 5.0  # |z| > 5 counted as outlier (consistent with preprocessing Winsorize)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 4 Gate verification — coverage / quality / IC.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional parquet (event_date, ticker, label_col). If omitted, IC gates report skipped.",
    )
    parser.add_argument("--label-col", type=str, default=DEFAULT_LABEL_COL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--n-windows", type=int, default=DEFAULT_N_WINDOWS)
    parser.add_argument(
        "--stage-note",
        type=str,
        default="pilot",
        help="Stage label for report (pilot / stage2); informational only.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Gate 1: Coverage
# ---------------------------------------------------------------------------


def load_state_counts(session: Session) -> dict[str, int]:
    """Load status-bucketed counts from trades_sampling_state."""
    stmt = (
        select(TradesSamplingState.status, func.count())
        .group_by(TradesSamplingState.status)
    )
    return {str(status): int(count) for status, count in session.execute(stmt).all()}


def compute_coverage_gate(status_counts: dict[str, int], *, threshold_pct: float) -> dict[str, Any]:
    """Coverage = completed / (total - skipped_holiday).
    Both 'completed' and 'partial' (max_pages truncation) count toward covered samples.
    """
    total = sum(status_counts.values())
    skipped_holiday = status_counts.get("skipped_holiday", 0)
    completed = status_counts.get("completed", 0) + status_counts.get("partial", 0)
    denominator = max(total - skipped_holiday, 0)
    value = (completed / denominator * 100.0) if denominator > 0 else 0.0
    return {
        "pass": value >= threshold_pct and denominator > 0,
        "value": round(value, 4),
        "threshold": threshold_pct,
        "completed": completed,
        "denominator": denominator,
        "status_counts": status_counts,
    }


# ---------------------------------------------------------------------------
# Gate 2: Feature Quality
# ---------------------------------------------------------------------------


def compute_feature_quality_gate(
    features_frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...] = TRADE_MICROSTRUCTURE_FEATURE_NAMES,
    missing_max_pct: float,
    outlier_max_pct: float,
) -> dict[str, Any]:
    """For each feature: missing_rate (% NaN), outlier_rate (% |z| > 5). Per-feature pass thresholds."""
    if features_frame.empty:
        per_feature = {
            name: {
                "missing_rate_pct": 100.0,
                "outlier_rate_pct": 0.0,
                "missing_pass": False,
                "outlier_pass": True,
                "pass": False,
            }
            for name in feature_names
        }
        return {
            "pass": False,
            "missing_threshold_pct": missing_max_pct,
            "outlier_threshold_pct": outlier_max_pct,
            "per_feature": per_feature,
        }

    total = len(features_frame)
    per_feature: dict[str, dict[str, float | bool]] = {}
    for name in feature_names:
        if name not in features_frame.columns:
            per_feature[name] = {
                "missing_rate_pct": 100.0,
                "outlier_rate_pct": 0.0,
                "missing_pass": False,
                "outlier_pass": True,
                "pass": False,
            }
            continue
        series = pd.to_numeric(features_frame[name], errors="coerce")
        missing = int(series.isna().sum())
        missing_rate = (missing / total) * 100.0 if total > 0 else 100.0
        valid = series.dropna()
        outlier = 0
        if len(valid) > 1 and valid.std(ddof=0) > 0:
            z = (valid - valid.mean()) / valid.std(ddof=0)
            outlier = int((z.abs() > OUTLIER_ZSCORE_THRESHOLD).sum())
        outlier_rate = (outlier / total) * 100.0 if total > 0 else 0.0
        missing_pass = missing_rate <= missing_max_pct
        outlier_pass = outlier_rate <= outlier_max_pct
        per_feature[name] = {
            "missing_rate_pct": round(missing_rate, 4),
            "outlier_rate_pct": round(outlier_rate, 4),
            "missing_pass": missing_pass,
            "outlier_pass": outlier_pass,
            "pass": missing_pass and outlier_pass,
        }

    overall_pass = all(bool(info["pass"]) for info in per_feature.values())
    return {
        "pass": overall_pass,
        "missing_threshold_pct": missing_max_pct,
        "outlier_threshold_pct": outlier_max_pct,
        "per_feature": per_feature,
    }


# ---------------------------------------------------------------------------
# Gate 3: Data-readiness IC (sign-aware)
# ---------------------------------------------------------------------------


def assign_windows(event_dates: pd.Series, *, n_windows: int) -> pd.Series:
    """Split sorted unique dates into n_windows equal-size buckets (by time, not by count).

    Returns a Series of window ids (0..n_windows-1) aligned with input index.
    """
    if event_dates.empty or n_windows <= 0:
        return pd.Series([pd.NA] * len(event_dates), index=event_dates.index, dtype="object")
    ts = pd.to_datetime(event_dates)
    min_ts = ts.min()
    max_ts = ts.max()
    if min_ts == max_ts:
        return pd.Series([0] * len(event_dates), index=event_dates.index, dtype="int64")
    total_ns = (max_ts - min_ts).value
    offsets_ns = (ts - min_ts).astype("int64")
    # Each window covers ceil(total/n) ns; the last window absorbs exact-max boundary
    span_ns = math.ceil(total_ns / n_windows)
    raw_idx = (offsets_ns // span_ns).clip(upper=n_windows - 1)
    return raw_idx.astype("int64")


def compute_sign_aware_ic_for_feature(
    feature_series: pd.Series,
    label_series: pd.Series,
    window_ids: pd.Series,
    *,
    ic_threshold: float,
    abs_tstat_threshold: float,
    sign_consistent_windows_min: int,
) -> dict[str, Any]:
    """Per-feature sign-aware IC diagnostic.

    Returns dict with: mean_ic, abs_t_stat, positive_windows, negative_windows,
    sign_consistent_windows, direction, pass, per_window_ic.
    """
    frame = pd.DataFrame(
        {"feature": pd.to_numeric(feature_series, errors="coerce"),
         "label": pd.to_numeric(label_series, errors="coerce"),
         "window_id": window_ids},
    ).dropna(subset=["feature", "label", "window_id"])

    per_window: list[float] = []
    per_window_counts: dict[int, int] = {}
    for window_id, group in frame.groupby("window_id"):
        if len(group) < 2 or group["feature"].nunique() < 2 or group["label"].nunique() < 2:
            continue
        rho, _ = stats.spearmanr(group["feature"].to_numpy(), group["label"].to_numpy())
        if not math.isnan(float(rho)):
            per_window.append(float(rho))
            per_window_counts[int(window_id)] = len(group)

    if not per_window:
        return {
            "mean_ic": float("nan"),
            "abs_t_stat": float("nan"),
            "positive_windows": 0,
            "negative_windows": 0,
            "sign_consistent_windows": 0,
            "direction": "unknown",
            "pass": False,
            "per_window_ic_count": 0,
            "reason": "no_valid_windows",
        }

    mean_ic = float(np.mean(per_window))
    positive_windows = int(sum(1 for ic in per_window if ic > 0))
    negative_windows = int(sum(1 for ic in per_window if ic < 0))
    sign_consistent_windows = max(positive_windows, negative_windows)
    direction = "positive" if mean_ic > 0 else ("negative" if mean_ic < 0 else "flat")

    if len(per_window) >= 2 and np.std(per_window, ddof=1) > 0:
        t_stat = float(mean_ic / (np.std(per_window, ddof=1) / math.sqrt(len(per_window))))
    else:
        t_stat = 0.0

    abs_t_stat = abs(t_stat)
    passes = (
        abs(mean_ic) >= ic_threshold
        and abs_t_stat >= abs_tstat_threshold
        and sign_consistent_windows >= sign_consistent_windows_min
    )

    return {
        "mean_ic": round(mean_ic, 6),
        "abs_t_stat": round(abs_t_stat, 4),
        "positive_windows": positive_windows,
        "negative_windows": negative_windows,
        "sign_consistent_windows": sign_consistent_windows,
        "direction": direction,
        "pass": bool(passes),
        "per_window_ic_count": len(per_window),
    }


def compute_data_readiness_ic_gate(
    features_frame: pd.DataFrame,
    labels_frame: pd.DataFrame,
    *,
    label_col: str,
    feature_names: tuple[str, ...] = TRADE_MICROSTRUCTURE_FEATURE_NAMES,
    ic_threshold: float,
    abs_tstat_threshold: float,
    sign_consistent_windows_min: int,
    min_passing_features: int,
    n_windows: int,
) -> dict[str, Any]:
    if features_frame.empty or labels_frame.empty:
        return {
            "pass": False,
            "reason": "features_or_labels_empty",
            "passing_features": [],
            "details": {},
            "min_passing_features": min_passing_features,
            "n_windows_used": 0,
        }

    features = features_frame.copy()
    features["event_date"] = pd.to_datetime(features["event_date"]).dt.date
    features["ticker"] = features["ticker"].astype(str).str.upper()
    labels = labels_frame.copy()
    labels["event_date"] = pd.to_datetime(labels["event_date"]).dt.date
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels = labels[["event_date", "ticker", label_col]].drop_duplicates(["event_date", "ticker"])

    merged = features.merge(labels, on=["event_date", "ticker"], how="inner")
    if merged.empty:
        return {
            "pass": False,
            "reason": "no_overlap_features_labels",
            "passing_features": [],
            "details": {},
            "min_passing_features": min_passing_features,
            "n_windows_used": 0,
        }

    merged["window_id"] = assign_windows(merged["event_date"], n_windows=n_windows)

    details: dict[str, dict[str, Any]] = {}
    passing_features: list[str] = []
    for name in feature_names:
        if name not in merged.columns:
            details[name] = {"pass": False, "reason": "feature_column_missing"}
            continue
        result = compute_sign_aware_ic_for_feature(
            merged[name],
            merged[label_col],
            merged["window_id"],
            ic_threshold=ic_threshold,
            abs_tstat_threshold=abs_tstat_threshold,
            sign_consistent_windows_min=sign_consistent_windows_min,
        )
        details[name] = result
        if result.get("pass"):
            passing_features.append(name)

    return {
        "pass": len(passing_features) >= min_passing_features,
        "passing_features": passing_features,
        "details": details,
        "min_passing_features": min_passing_features,
        "ic_threshold": ic_threshold,
        "abs_tstat_threshold": abs_tstat_threshold,
        "sign_consistent_windows_min": sign_consistent_windows_min,
        "n_windows_used": n_windows,
    }


# ---------------------------------------------------------------------------
# Gate 4: Per-reason IC breakdown
# ---------------------------------------------------------------------------


def load_state_reasons(session: Session) -> pd.DataFrame:
    """Load (ticker, trading_date, sampled_reason, status) for per-reason grouping."""
    stmt = select(
        TradesSamplingState.ticker,
        TradesSamplingState.trading_date,
        TradesSamplingState.sampled_reason,
        TradesSamplingState.status,
    )
    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["ticker", "trading_date", "sampled_reason", "status"])
    frame = pd.DataFrame(rows, columns=["ticker", "trading_date", "sampled_reason", "status"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.date
    return frame


def compute_per_reason_ic(
    features_frame: pd.DataFrame,
    labels_frame: pd.DataFrame,
    state_frame: pd.DataFrame,
    *,
    label_col: str,
    feature_names: tuple[str, ...] = TRADE_MICROSTRUCTURE_FEATURE_NAMES,
) -> dict[str, Any]:
    """For each reason (earnings / gap / weak_window), compute aggregate IC (Spearman)
    of each feature vs labels on the subset of (ticker, event_date) in that reason bucket.

    Returns: {reason: {feature: mean_spearman_rho}} + top_1_reason_per_feature.
    """
    if features_frame.empty or labels_frame.empty or state_frame.empty:
        return {"per_reason": {}, "top_reason_by_feature": {}, "reason": "insufficient_data"}

    features = features_frame.copy()
    features["event_date"] = pd.to_datetime(features["event_date"]).dt.date
    features["ticker"] = features["ticker"].astype(str).str.upper()
    labels = labels_frame.copy()
    labels["event_date"] = pd.to_datetime(labels["event_date"]).dt.date
    labels["ticker"] = labels["ticker"].astype(str).str.upper()
    labels = labels[["event_date", "ticker", label_col]].drop_duplicates(["event_date", "ticker"])
    merged = features.merge(labels, on=["event_date", "ticker"], how="inner")
    if merged.empty:
        return {"per_reason": {}, "top_reason_by_feature": {}, "reason": "no_overlap"}

    state = state_frame.copy()
    state["trading_date"] = pd.to_datetime(state["trading_date"]).dt.date
    state["ticker"] = state["ticker"].astype(str).str.upper()
    state["sampled_reason"] = state["sampled_reason"].astype(str).str.lower()

    per_reason: dict[str, dict[str, float]] = {}
    for reason in sorted(state["sampled_reason"].unique()):
        keys = state.loc[state["sampled_reason"] == reason, ["ticker", "trading_date"]].rename(
            columns={"trading_date": "event_date"},
        )
        if keys.empty:
            continue
        subset = merged.merge(keys.drop_duplicates(), on=["ticker", "event_date"], how="inner")
        if subset.empty:
            continue
        feature_scores: dict[str, float] = {}
        for name in feature_names:
            if name not in subset.columns:
                continue
            s_feat = pd.to_numeric(subset[name], errors="coerce")
            s_label = pd.to_numeric(subset[label_col], errors="coerce")
            valid = s_feat.notna() & s_label.notna()
            if valid.sum() < 2 or s_feat.loc[valid].nunique() < 2 or s_label.loc[valid].nunique() < 2:
                feature_scores[name] = float("nan")
                continue
            rho, _ = stats.spearmanr(s_feat.loc[valid].to_numpy(), s_label.loc[valid].to_numpy())
            feature_scores[name] = float(rho) if not math.isnan(float(rho)) else float("nan")
        per_reason[reason] = {k: round(v, 6) if not math.isnan(v) else None for k, v in feature_scores.items()}

    top_reason_by_feature: dict[str, str | None] = {}
    for name in feature_names:
        best_reason = None
        best_abs = -1.0
        for reason, scores in per_reason.items():
            val = scores.get(name)
            if val is None:
                continue
            if abs(val) > best_abs:
                best_abs = abs(val)
                best_reason = reason
        top_reason_by_feature[name] = best_reason

    return {"per_reason": per_reason, "top_reason_by_feature": top_reason_by_feature}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def evaluate_gates(
    *,
    config: Week4TradesConfig,
    config_hash: str,
    features_frame: pd.DataFrame,
    labels_frame: pd.DataFrame | None,
    state_counts: dict[str, int],
    state_frame: pd.DataFrame | None,
    label_col: str,
    n_windows: int,
    stage_note: str,
) -> dict[str, Any]:
    """Pure evaluator — no DB access. Takes all data as frames, returns the full gate report."""
    coverage = compute_coverage_gate(state_counts, threshold_pct=config.gate.coverage_min_pct)
    feature_quality = compute_feature_quality_gate(
        features_frame,
        missing_max_pct=config.gate.feature_missing_max_pct,
        outlier_max_pct=config.gate.feature_outlier_max_pct,
    )

    if labels_frame is None or labels_frame.empty:
        data_readiness = {
            "pass": False,
            "reason": "labels_not_provided",
            "passing_features": [],
            "details": {},
            "min_passing_features": config.gate.min_passing_features,
            "n_windows_used": 0,
        }
        per_reason = {"per_reason": {}, "top_reason_by_feature": {}, "reason": "labels_not_provided"}
    else:
        data_readiness = compute_data_readiness_ic_gate(
            features_frame,
            labels_frame,
            label_col=label_col,
            ic_threshold=config.gate.ic_threshold,
            abs_tstat_threshold=config.gate.abs_tstat_threshold,
            sign_consistent_windows_min=config.gate.sign_consistent_windows_min,
            min_passing_features=config.gate.min_passing_features,
            n_windows=n_windows,
        )
        per_reason = compute_per_reason_ic(
            features_frame,
            labels_frame,
            state_frame if state_frame is not None else pd.DataFrame(),
            label_col=label_col,
        )

    notes = [
        "SEC event window deferred to Week 5 (see plan Task 3 footnote)",
        f"Stage note: {stage_note}",
        "Data-readiness IC gate is diagnostic-only; not an alpha-adoption criterion.",
    ]

    overall_pass = bool(coverage["pass"] and feature_quality["pass"] and data_readiness["pass"])
    return {
        "config_hash": config_hash,
        "run_stage": stage_note,
        "overall_pass": overall_pass,
        "gates": {
            "coverage": coverage,
            "feature_quality": feature_quality,
            "data_readiness_ic": data_readiness,
            "per_reason_ic": per_reason,
        },
        "notes": notes,
    }


def run_gate_verification(
    *,
    config: Week4TradesConfig,
    config_hash: str,
    features_path: Path,
    labels_path: Path | None,
    label_col: str,
    n_windows: int,
    stage_note: str,
    session_factory: Callable[[], Session] | None = None,
) -> dict[str, Any]:
    if not features_path.exists():
        raise FileNotFoundError(f"features parquet not found: {features_path}")
    features_frame = pd.read_parquet(features_path)

    labels_frame: pd.DataFrame | None = None
    if labels_path is not None:
        if not labels_path.exists():
            logger.warning("labels parquet {} not found; skipping IC gates", labels_path)
        else:
            labels_frame = pd.read_parquet(labels_path)

    factory = session_factory or get_session_factory()
    with factory() as session:
        state_counts = load_state_counts(session)
        state_frame = load_state_reasons(session) if labels_frame is not None else None

    return evaluate_gates(
        config=config,
        config_hash=config_hash,
        features_frame=features_frame,
        labels_frame=labels_frame,
        state_counts=state_counts,
        state_frame=state_frame,
        label_col=label_col,
        n_windows=n_windows,
        stage_note=stage_note,
    )


def _format_console(report: dict[str, Any]) -> str:
    lines = [f"Week 4 Gate Summary — stage={report['run_stage']} config_hash={report['config_hash'][:12]}..."]
    lines.append(f"  Overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    gates = report["gates"]
    lines.append(
        f"  Coverage: {'PASS' if gates['coverage']['pass'] else 'FAIL'} "
        f"value={gates['coverage']['value']}%  threshold={gates['coverage']['threshold']}%",
    )
    lines.append(
        f"  Feature Quality: {'PASS' if gates['feature_quality']['pass'] else 'FAIL'}",
    )
    for name, info in gates["feature_quality"].get("per_feature", {}).items():
        lines.append(
            f"    - {name}: missing={info['missing_rate_pct']}%  outlier={info['outlier_rate_pct']}%  "
            f"{'PASS' if info['pass'] else 'FAIL'}",
        )
    lines.append(
        f"  Data-readiness IC (diagnostic): {'PASS' if gates['data_readiness_ic']['pass'] else 'FAIL'} "
        f"passing={gates['data_readiness_ic'].get('passing_features', [])}",
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = preflight_trades_estimator.load_config(args.config)
    config_hash = preflight_trades_estimator.compute_config_hash(config)

    report = run_gate_verification(
        config=config,
        config_hash=config_hash,
        features_path=args.features,
        labels_path=args.labels,
        label_col=args.label_col,
        n_windows=args.n_windows,
        stage_note=args.stage_note,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str))
    print(_format_console(report))
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":  # pragma: no cover - direct script execution
    sys.exit(main())
