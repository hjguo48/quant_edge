from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.run_walkforward_comparison import (
    REBALANCE_WEEKDAY,
    SPLIT_STRATEGY_LEGACY_CONTIGUOUS,
    WINDOWS,
    build_ic_comparison,
    build_window_split_diagnostics,
    resolve_windows_with_embargo,
)


def test_resolve_windows_with_embargo_passes_all_configured_windows() -> None:
    trade_dates = tuple(
        pd.bdate_range(
            min(window.train_start for window in WINDOWS),
            max(window.test_end for window in WINDOWS),
        ).date,
    )
    trade_date_positions = {trade_date: index for index, trade_date in enumerate(trade_dates)}

    resolved_windows = resolve_windows_with_embargo(
        windows=list(WINDOWS),
        trade_dates=trade_dates,
        trade_date_positions=trade_date_positions,
        rebalance_weekday=REBALANCE_WEEKDAY,
        embargo_days=60,
    )

    assert len(resolved_windows) == 11

    for requested_window, effective_window in zip(WINDOWS, resolved_windows, strict=True):
        diagnostics = build_window_split_diagnostics(
            requested_window=requested_window,
            effective_window=effective_window,
            trade_date_positions=trade_date_positions,
            embargo_days=60,
        )

        assert effective_window.train_start.weekday() == REBALANCE_WEEKDAY
        assert effective_window.validation_start.weekday() == REBALANCE_WEEKDAY
        assert effective_window.test_start.weekday() == REBALANCE_WEEKDAY
        assert effective_window.train_end.weekday() == REBALANCE_WEEKDAY
        assert effective_window.validation_end.weekday() == REBALANCE_WEEKDAY
        assert effective_window.test_end.weekday() == REBALANCE_WEEKDAY
        assert effective_window.train_end <= requested_window.train_end
        assert effective_window.validation_end <= requested_window.validation_end
        assert effective_window.test_end <= requested_window.test_end

        for boundary in (
            "train_to_validation",
            "validation_to_test",
            "refit_train_plus_validation_to_test",
        ):
            check = diagnostics["verification"][boundary]
            assert check["passed"] is True
            assert check["gap_trading_days"] >= 60


def test_build_ic_comparison_summarizes_baseline_vs_embargo() -> None:
    baseline_report = {
        "windows": [
            {
                "window_id": "W1",
                "results": {
                    "ridge": {"test_metrics": {"ic": 0.10}},
                    "xgboost": {"test_metrics": {"ic": 0.20}},
                },
            },
            {
                "window_id": "W2",
                "results": {
                    "ridge": {"test_metrics": {"ic": 0.05}},
                    "xgboost": {"test_metrics": {"ic": 0.15}},
                },
            },
        ],
    }
    current_windows = [
        {
            "window_id": "W1",
            "results": {
                "ridge": {"test_metrics": {"ic": 0.08}},
                "xgboost": {"test_metrics": {"ic": 0.18}},
            },
        },
        {
            "window_id": "W2",
            "results": {
                "ridge": {"test_metrics": {"ic": 0.02}},
                "xgboost": {"test_metrics": {"ic": 0.10}},
            },
        },
    ]

    comparison = build_ic_comparison(
        current_windows=current_windows,
        baseline_report=baseline_report,
        baseline_report_path=Path("data/reports/walkforward_comparison_60d.json"),
        model_names=["ridge", "xgboost"],
        embargo_days=60,
    )

    assert comparison["available"] is True
    assert comparison["baseline_strategy"] == SPLIT_STRATEGY_LEGACY_CONTIGUOUS
    assert comparison["summary"]["ridge"]["windows_compared"] == 2
    assert comparison["summary"]["ridge"]["baseline_mean_test_ic"] == pytest.approx(0.075)
    assert comparison["summary"]["ridge"]["embargo_mean_test_ic"] == pytest.approx(0.05)
    assert comparison["summary"]["ridge"]["mean_delta_test_ic"] == pytest.approx(-0.025)
    assert comparison["summary"]["ridge"]["min_delta_test_ic"] == pytest.approx(-0.03)
    assert comparison["summary"]["ridge"]["max_delta_test_ic"] == pytest.approx(-0.02)
    assert comparison["summary"]["xgboost"]["mean_delta_test_ic"] == pytest.approx(-0.035)
    assert comparison["per_window"][0]["models"]["ridge"]["delta_test_ic"] == pytest.approx(-0.02)
    assert comparison["per_window"][1]["models"]["xgboost"]["delta_test_ic"] == pytest.approx(-0.05)
