from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.run_week4_gate_verification as gate


def _config_payload() -> dict[str, Any]:
    return {
        "version": 1,
        "stage": "pilot",
        "sampling": {
            "pilot": {
                "reasons": ["earnings", "gap", "weak_window"],
                "earnings_window_days": 3,
                "gap_threshold_pct": 0.03,
                "weak_window_top_n": 100,
                "weak_windows": [{"name": "W5", "start": "2022-10-01", "end": "2023-03-31"}],
            },
            "stage2": {"top_n_liquidity": 200, "top_liquidity_lookback_days": 20},
        },
        "polygon": {
            "entitlement_delay_minutes": 15,
            "rest_max_pages_per_request": 50,
            "rest_page_size": 50000,
            "rest_min_interval_seconds": 0.05,
            "retry_max": 3,
        },
        "budgets": {
            "max_daily_api_calls": 50000,
            "max_storage_gb": 200,
            "max_rows_per_ticker_day": 2000000,
            "expected_pilot_ticker_days": 30000,
        },
        "features": {
            "size_threshold_dollars": 1000000,
            "size_threshold_min_cap_dollars": 250000,
            "condition_allow_list": [],
            "trf_exchange_codes": [4, 202],
            "late_day_window_et": ["15:00", "16:00"],
            "offhours_window_et_pre": ["04:00", "09:30"],
            "offhours_window_et_post": ["16:00", "20:00"],
        },
        "gate": {
            "coverage_min_pct": 95.0,
            "feature_missing_max_pct": 30.0,
            "feature_outlier_max_pct": 5.0,
            "min_passing_features": 2,
            "ic_threshold": 0.015,
            "abs_tstat_threshold": 2.0,
            "sign_consistent_windows_min": 7,
        },
    }


@pytest.fixture
def config(tmp_path: Path):
    path = tmp_path / "week4.yaml"
    path.write_text(yaml.safe_dump(_config_payload()))
    return gate.preflight_trades_estimator.load_config(path)


@pytest.fixture
def config_hash(config):
    return gate.preflight_trades_estimator.compute_config_hash(config)


# ---------------------------------------------------------------------------
# Coverage gate
# ---------------------------------------------------------------------------


def test_coverage_gate_passes_when_completed_share_above_threshold() -> None:
    counts = {"completed": 96, "pending": 4, "skipped_holiday": 5}
    result = gate.compute_coverage_gate(counts, threshold_pct=95.0)
    assert result["pass"] is True
    # completed=96 / (105-5)=100 → 96%
    assert result["value"] == 96.0


def test_coverage_gate_fails_when_completed_share_below_threshold() -> None:
    counts = {"completed": 80, "failed": 20}
    result = gate.compute_coverage_gate(counts, threshold_pct=95.0)
    assert result["pass"] is False
    assert result["value"] == 80.0


def test_coverage_gate_counts_partial_as_covered() -> None:
    """Task 6 I-3 fix: partial state (max_pages truncation) still counts as covered sample."""
    counts = {"completed": 90, "partial": 6, "failed": 4}
    result = gate.compute_coverage_gate(counts, threshold_pct=95.0)
    assert result["pass"] is True
    assert result["completed"] == 96
    assert result["value"] == 96.0


def test_coverage_gate_all_skipped_holiday_fails_gracefully() -> None:
    counts = {"skipped_holiday": 10}
    result = gate.compute_coverage_gate(counts, threshold_pct=95.0)
    assert result["pass"] is False
    assert result["denominator"] == 0
    # F2: denominator=0 → value=None (not 0.0) so consumers distinguish "no coverable" from "0% covered"
    assert result["value"] is None


# ---------------------------------------------------------------------------
# Feature quality gate
# ---------------------------------------------------------------------------


def _feature_frame(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n, freq="D").date
    return pd.DataFrame(
        {
            "event_date": dates,
            "ticker": ["AAPL"] * n,
            "knowledge_time_regular": pd.Timestamp("2026-01-05 21:15", tz="UTC"),
            "knowledge_time_offhours": pd.Timestamp("2026-01-06 01:15", tz="UTC"),
            "trade_imbalance_proxy": rng.normal(0, 0.1, n),
            "large_trade_ratio": rng.uniform(0, 0.1, n),
            "late_day_aggressiveness": rng.uniform(0, 2, n),
            "offhours_trade_ratio": rng.uniform(0, 0.3, n),
            "off_exchange_volume_ratio": rng.uniform(0, 0.4, n),
            "run_config_hash": ["abc"] * n,
        },
    )


def test_feature_quality_gate_passes_on_clean_data() -> None:
    frame = _feature_frame()
    result = gate.compute_feature_quality_gate(frame, missing_max_pct=30.0, outlier_max_pct=5.0)
    assert result["pass"] is True
    for name, info in result["per_feature"].items():
        assert info["pass"], f"{name} unexpectedly failed: {info}"


def test_feature_quality_gate_fails_when_missing_rate_exceeds_threshold() -> None:
    frame = _feature_frame(n=100)
    frame.loc[:40, "trade_imbalance_proxy"] = np.nan  # 41% missing, above 30%
    result = gate.compute_feature_quality_gate(frame, missing_max_pct=30.0, outlier_max_pct=5.0)
    assert result["pass"] is False
    assert result["per_feature"]["trade_imbalance_proxy"]["missing_pass"] is False


def test_feature_quality_gate_fails_on_outliers() -> None:
    """1 extreme outlier in n=1000 clean samples → |z|>5 detected; 0.1% rate > 0.05% threshold."""
    frame = _feature_frame(n=1000)
    # Single extreme value — keeps mean/std from absorbing it
    frame.loc[0, "large_trade_ratio"] = 1000.0
    result = gate.compute_feature_quality_gate(
        frame,
        missing_max_pct=30.0,
        outlier_max_pct=0.05,  # 0.05% threshold; 1/1000 = 0.1%
    )
    assert result["per_feature"]["large_trade_ratio"]["outlier_rate_pct"] > 0.05
    assert result["pass"] is False


# ---------------------------------------------------------------------------
# Data-readiness IC gate
# ---------------------------------------------------------------------------


def test_assign_windows_splits_date_range_evenly() -> None:
    dates = pd.Series(pd.date_range("2026-01-01", periods=11, freq="D").date)
    ids = gate.assign_windows(dates, n_windows=11)
    # With 11 dates and 11 windows, each date likely falls in its own window (or close)
    assert len(ids.unique()) >= 2
    assert ids.min() == 0
    assert ids.max() == 10


def test_sign_aware_ic_passing_feature_positive_ic() -> None:
    rng = np.random.default_rng(42)
    n = 500
    feature = rng.normal(0, 1, n)
    # Label correlated positively with feature (true IC ~ 0.5)
    label = feature * 0.5 + rng.normal(0, 1, n) * 0.5
    window_ids = pd.Series(np.repeat(np.arange(11), n // 11 + 1)[:n])
    result = gate.compute_sign_aware_ic_for_feature(
        pd.Series(feature),
        pd.Series(label),
        window_ids,
        ic_threshold=0.015,
        abs_tstat_threshold=2.0,
        sign_consistent_windows_min=7,
    )
    assert result["pass"] is True
    assert result["direction"] == "positive"
    assert result["positive_windows"] >= 7


def test_sign_aware_ic_passing_feature_negative_ic() -> None:
    """Negative IC features are acceptable (short signal) per plan P2 fix."""
    rng = np.random.default_rng(123)
    n = 500
    feature = rng.normal(0, 1, n)
    label = -feature * 0.5 + rng.normal(0, 1, n) * 0.5
    window_ids = pd.Series(np.repeat(np.arange(11), n // 11 + 1)[:n])
    result = gate.compute_sign_aware_ic_for_feature(
        pd.Series(feature),
        pd.Series(label),
        window_ids,
        ic_threshold=0.015,
        abs_tstat_threshold=2.0,
        sign_consistent_windows_min=7,
    )
    assert result["pass"] is True
    assert result["direction"] == "negative"
    assert result["negative_windows"] >= 7


def test_sign_aware_ic_fails_when_signal_too_weak() -> None:
    rng = np.random.default_rng(7)
    n = 500
    feature = rng.normal(0, 1, n)
    label = rng.normal(0, 1, n)  # uncorrelated
    window_ids = pd.Series(np.repeat(np.arange(11), n // 11 + 1)[:n])
    result = gate.compute_sign_aware_ic_for_feature(
        pd.Series(feature),
        pd.Series(label),
        window_ids,
        ic_threshold=0.015,
        abs_tstat_threshold=2.0,
        sign_consistent_windows_min=7,
    )
    assert result["pass"] is False


def test_data_readiness_ic_gate_skipped_when_no_labels() -> None:
    frame = _feature_frame()
    result = gate.compute_data_readiness_ic_gate(
        frame,
        pd.DataFrame(),
        label_col="forward_excess_return_5d",
        ic_threshold=0.015,
        abs_tstat_threshold=2.0,
        sign_consistent_windows_min=7,
        min_passing_features=2,
        n_windows=11,
    )
    assert result["pass"] is False
    assert result["reason"] == "features_or_labels_empty"


# ---------------------------------------------------------------------------
# Per-reason IC
# ---------------------------------------------------------------------------


def test_per_reason_ic_reports_score_per_reason() -> None:
    rng = np.random.default_rng(11)
    n = 60
    dates = pd.date_range("2026-01-01", periods=n, freq="D").date
    tickers = np.tile(["AAPL", "MSFT", "GOOGL"], n // 3 + 1)[:n]

    # Build features: positive IC for earnings subset, noise elsewhere
    features = pd.DataFrame(
        {
            "event_date": dates,
            "ticker": tickers,
            "knowledge_time_regular": pd.Timestamp("2026-01-01 21:15", tz="UTC"),
            "knowledge_time_offhours": pd.Timestamp("2026-01-02 01:15", tz="UTC"),
            "trade_imbalance_proxy": rng.normal(0, 1, n),
            "large_trade_ratio": rng.uniform(0, 0.1, n),
            "late_day_aggressiveness": rng.uniform(0, 2, n),
            "offhours_trade_ratio": rng.uniform(0, 0.3, n),
            "off_exchange_volume_ratio": rng.uniform(0, 0.4, n),
            "run_config_hash": ["h"] * n,
        },
    )
    # Labels: strongly correlated with trade_imbalance_proxy
    labels = pd.DataFrame(
        {
            "event_date": dates,
            "ticker": tickers,
            "forward_excess_return_5d": features["trade_imbalance_proxy"] * 0.8
            + rng.normal(0, 0.5, n),
        },
    )
    # State: half earnings, half gap
    state = pd.DataFrame(
        {
            "ticker": tickers,
            "trading_date": dates,
            "sampled_reason": ["earnings"] * (n // 2) + ["gap"] * (n - n // 2),
            "status": ["completed"] * n,
        },
    )

    report = gate.compute_per_reason_ic(
        features,
        labels,
        state,
        label_col="forward_excess_return_5d",
    )
    assert "earnings" in report["per_reason"]
    assert "gap" in report["per_reason"]
    # trade_imbalance_proxy 应在至少一个 reason 里 top-1
    assert report["top_reason_by_feature"]["trade_imbalance_proxy"] in {"earnings", "gap"}


# ---------------------------------------------------------------------------
# End-to-end evaluate_gates
# ---------------------------------------------------------------------------


def test_evaluate_gates_end_to_end_labels_missing(config, config_hash) -> None:
    frame = _feature_frame()
    report = gate.evaluate_gates(
        config=config,
        config_hash=config_hash,
        features_frame=frame,
        labels_frame=None,
        state_counts={"completed": 100},
        state_frame=None,
        label_col="forward_excess_return_5d",
        n_windows=11,
        stage_note="pilot",
    )
    assert report["gates"]["coverage"]["pass"] is True
    assert report["gates"]["feature_quality"]["pass"] is True
    assert report["gates"]["data_readiness_ic"]["pass"] is False
    assert report["gates"]["data_readiness_ic"]["reason"] == "labels_not_provided"
    assert report["overall_pass"] is False  # IC gate failure blocks overall


def test_evaluate_gates_end_to_end_all_pass(config, config_hash) -> None:
    rng = np.random.default_rng(21)
    n = 400
    dates = pd.date_range("2022-01-01", periods=n, freq="D").date
    tickers = np.tile(["AAPL", "MSFT"], n // 2 + 1)[:n]
    feat_values = rng.normal(0, 1, n)
    features = pd.DataFrame(
        {
            "event_date": dates,
            "ticker": tickers,
            "knowledge_time_regular": pd.Timestamp("2022-01-05 21:15", tz="UTC"),
            "knowledge_time_offhours": pd.Timestamp("2022-01-06 01:15", tz="UTC"),
            "trade_imbalance_proxy": feat_values,
            "large_trade_ratio": feat_values * 0.9 + rng.normal(0, 0.1, n),
            "late_day_aggressiveness": rng.uniform(0, 2, n),
            "offhours_trade_ratio": rng.uniform(0, 0.3, n),
            "off_exchange_volume_ratio": rng.uniform(0, 0.4, n),
            "run_config_hash": ["h"] * n,
        },
    )
    labels = pd.DataFrame(
        {
            "event_date": dates,
            "ticker": tickers,
            "forward_excess_return_5d": feat_values * 0.7 + rng.normal(0, 0.3, n),
        },
    )
    state = pd.DataFrame(
        {
            "ticker": tickers,
            "trading_date": dates,
            "sampled_reason": ["earnings"] * n,
            "status": ["completed"] * n,
        },
    )
    report = gate.evaluate_gates(
        config=config,
        config_hash=config_hash,
        features_frame=features,
        labels_frame=labels,
        state_counts={"completed": 100},
        state_frame=state,
        label_col="forward_excess_return_5d",
        n_windows=11,
        stage_note="pilot",
    )
    assert report["gates"]["coverage"]["pass"] is True
    assert report["gates"]["feature_quality"]["pass"] is True
    assert report["gates"]["data_readiness_ic"]["pass"] is True
    # At least trade_imbalance_proxy and large_trade_ratio (correlated features) should pass
    passing = set(report["gates"]["data_readiness_ic"]["passing_features"])
    assert "trade_imbalance_proxy" in passing
    assert report["overall_pass"] is True


def test_output_json_schema_contains_expected_keys(config, config_hash) -> None:
    frame = _feature_frame()
    report = gate.evaluate_gates(
        config=config,
        config_hash=config_hash,
        features_frame=frame,
        labels_frame=None,
        state_counts={"completed": 100},
        state_frame=None,
        label_col="forward_excess_return_5d",
        n_windows=11,
        stage_note="pilot",
    )
    assert set(report.keys()) >= {"config_hash", "run_stage", "overall_pass", "gates", "notes"}
    assert set(report["gates"].keys()) == {"coverage", "feature_quality", "data_readiness_ic", "per_reason_ic"}
