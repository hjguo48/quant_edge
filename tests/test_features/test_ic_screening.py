from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.run_ic_screening import (
    build_model_feature_sets,
    compute_feature_stability_record,
    prepare_feature_batch,
)
from scripts.run_single_window_validation import (
    load_retained_features as load_validation_retained_features,
    long_to_feature_matrix,
)
from scripts.run_walkforward_comparison import load_retained_features as load_comparison_retained_features


def test_prepare_feature_batch_keeps_missing_indicator_rows() -> None:
    features = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 5),
                "feature_name": "ret_20d",
                "feature_value": 0.25,
                "is_filled": False,
            },
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 5),
                "feature_name": "is_missing_ret_20d",
                "feature_value": 1.0,
                "is_filled": False,
            },
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 5),
                "feature_name": "not_in_screening_set",
                "feature_value": 99.0,
                "is_filled": False,
            },
        ],
    )

    prepared = prepare_feature_batch(features)

    assert set(prepared["feature_name"]) == {"ret_20d", "is_missing_ret_20d"}


def test_long_to_feature_matrix_synthesizes_missing_indicators_from_base_feature_nans() -> None:
    features = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 5),
                "feature_name": "ret_20d",
                "feature_value": np.nan,
            },
            {
                "ticker": "BBB",
                "trade_date": date(2024, 1, 5),
                "feature_name": "ret_20d",
                "feature_value": 0.75,
            },
        ],
    )

    matrix = long_to_feature_matrix(features, ["is_missing_ret_20d"])

    assert matrix.loc[(pd.Timestamp("2024-01-05"), "AAA"), "is_missing_ret_20d"] == pytest.approx(1.0)
    assert matrix.loc[(pd.Timestamp("2024-01-05"), "BBB"), "is_missing_ret_20d"] == pytest.approx(0.0)


def test_legacy_retained_feature_loaders_append_missing_indicator_companions(tmp_path) -> None:
    report_path = tmp_path / "legacy_ic_report.csv"
    pd.DataFrame(
        [
            {"feature_name": "ret_20d", "passed": True},
            {"feature_name": "vol_20d", "passed": False},
        ],
    ).to_csv(report_path, index=False)

    validation_retained = load_validation_retained_features(report_path)
    comparison_retained = load_comparison_retained_features(report_path)

    assert validation_retained == ["ret_20d", "is_missing_ret_20d"]
    assert comparison_retained == ["ret_20d", "is_missing_ret_20d"]


def test_compute_feature_stability_record_keeps_stable_negative_signal() -> None:
    dates = [date(2020, 1, 31), date(2020, 7, 31), date(2021, 1, 29), date(2021, 7, 30)]
    tickers = ["AAA", "BBB", "CCC"]
    rows: list[tuple[date, str, float, float]] = []
    for trade_date in dates:
        for ticker, y_true, y_pred in zip(tickers, [1.0, 0.0, -1.0], [-1.0, 0.0, 1.0], strict=True):
            rows.append((trade_date, ticker, y_true, y_pred))

    aligned = pd.DataFrame(rows, columns=["trade_date", "ticker", "y_true", "y_pred"]).set_index(
        ["trade_date", "ticker"],
    )

    record = compute_feature_stability_record(
        feature_name="ret_20d",
        aligned=aligned,
        ic_threshold=0.10,
        sign_consistency_threshold=0.60,
    )

    assert record["signed_ic"] == pytest.approx(-1.0)
    assert record["window_signed_ic_mean"] == pytest.approx(-1.0)
    assert record["sign_consistency"] == pytest.approx(1.0)
    assert record["dominant_sign"] == "negative"
    assert record["passed"] is True


def test_build_model_feature_sets_applies_corr_dedup_and_model_limits() -> None:
    report = pd.DataFrame(
        [
            {
                "feature_name": "ret_20d",
                "passed": True,
                "stability_score": 0.60,
                "window_signed_ic_mean": 0.60,
                "rank_ic": 0.55,
                "sign_consistency": 1.0,
                "feature_kind": "base",
            },
            {
                "feature_name": "vol_20d",
                "passed": True,
                "stability_score": 0.59,
                "window_signed_ic_mean": 0.59,
                "rank_ic": 0.54,
                "sign_consistency": 1.0,
                "feature_kind": "base",
            },
            {
                "feature_name": "bb_width",
                "passed": True,
                "stability_score": 0.40,
                "window_signed_ic_mean": 0.40,
                "rank_ic": 0.35,
                "sign_consistency": 0.9,
                "feature_kind": "base",
            },
            {
                "feature_name": "amihud",
                "passed": True,
                "stability_score": 0.30,
                "window_signed_ic_mean": 0.30,
                "rank_ic": 0.25,
                "sign_consistency": 0.8,
                "feature_kind": "base",
            },
            {
                "feature_name": "is_missing_ret_20d",
                "passed": False,
                "stability_score": 0.05,
                "window_signed_ic_mean": 0.05,
                "rank_ic": 0.02,
                "sign_consistency": 0.7,
                "feature_kind": "missing_indicator",
            },
        ],
    )
    features = report["feature_name"].tolist()
    correlation = pd.DataFrame(np.eye(len(features)), index=features, columns=features, dtype=float)
    correlation.loc["ret_20d", "vol_20d"] = 0.96
    correlation.loc["vol_20d", "ret_20d"] = 0.96
    correlation.loc["ret_20d", "is_missing_ret_20d"] = 0.30
    correlation.loc["is_missing_ret_20d", "ret_20d"] = 0.30

    selection = build_model_feature_sets(
        report=report,
        correlation_matrix=correlation,
        correlation_threshold=0.85,
        model_feature_limits={"ridge": 4, "xgboost": 2, "lightgbm": 2},
    )

    ridge_features = selection["feature_sets"]["ridge"]
    xgboost_features = selection["feature_sets"]["xgboost"]

    assert "ret_20d" in ridge_features
    assert "vol_20d" not in ridge_features
    assert "is_missing_ret_20d" in ridge_features
    assert len(ridge_features) == 4
    assert xgboost_features == ["ret_20d", "bb_width"]

    for left, right in [("ret_20d", "vol_20d")]:
        assert not ({left, right} <= set(ridge_features))
        assert not ({left, right} <= set(xgboost_features))
