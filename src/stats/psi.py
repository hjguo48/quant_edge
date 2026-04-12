"""Population Stability Index (PSI) for feature distribution monitoring.

PSI measures the shift between a reference (training) distribution and a
current (live/test) distribution.  Interpretation:
  - PSI < 0.10 : no significant shift
  - 0.10 ≤ PSI < 0.25 : moderate shift — investigate
  - PSI ≥ 0.25 : significant shift — alert

IMPORTANT: PSI must be computed on **raw pre-rank** features.
Rank-normalized features compress distributions and suppress real drift signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-4,
) -> float:
    """Compute PSI between reference and current distributions.

    Uses equal-frequency (quantile) binning on the reference distribution
    to define bin edges, then measures how the current distribution has
    shifted relative to those bins.
    """
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) < n_bins or len(current) < n_bins:
        return float("nan")

    # Quantile-based bin edges from reference
    bin_edges = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    # Deduplicate edges (can happen with discrete features)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        return float("nan")

    ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Clip to avoid log(0)
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return psi


def compute_fill_rate(series: np.ndarray) -> float:
    """Fraction of finite (non-NaN, non-inf) values."""
    if len(series) == 0:
        return 0.0
    return float(np.isfinite(series).sum() / len(series))


def compute_feature_psi_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    n_bins: int = 10,
    psi_alert_threshold: float = 0.25,
    fill_rate_change_threshold: float = 0.05,
) -> list[dict]:
    """Compute PSI and fill-rate for each feature, flag alerts.

    Both DataFrames should contain raw pre-rank feature values
    (before Winsorize / rank normalization).
    """
    results = []
    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            results.append({
                "feature": col,
                "psi": float("nan"),
                "ref_fill_rate": float("nan"),
                "cur_fill_rate": float("nan"),
                "fill_rate_delta": float("nan"),
                "psi_alert": False,
                "fill_rate_alert": False,
                "missing_in_data": True,
            })
            continue

        ref_vals = reference_df[col].to_numpy(dtype=float)
        cur_vals = current_df[col].to_numpy(dtype=float)

        psi_value = compute_psi(ref_vals, cur_vals, n_bins=n_bins)
        ref_fill = compute_fill_rate(ref_vals)
        cur_fill = compute_fill_rate(cur_vals)
        fill_delta = cur_fill - ref_fill

        results.append({
            "feature": col,
            "psi": psi_value,
            "ref_fill_rate": ref_fill,
            "cur_fill_rate": cur_fill,
            "fill_rate_delta": fill_delta,
            "psi_alert": bool(np.isfinite(psi_value) and psi_value >= psi_alert_threshold),
            "fill_rate_alert": bool(abs(fill_delta) >= fill_rate_change_threshold),
            "missing_in_data": False,
        })

    return results
