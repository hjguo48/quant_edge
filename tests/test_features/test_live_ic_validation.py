from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import src.features.preprocessing as preprocessing_module
from scripts.run_live_ic_validation import (
    build_cross_sectional_technical_features_for_date,
    expand_requested_feature_names,
    finalize_fusion_date_feature_matrix,
    rank_series_to_unit_interval,
    score_fusion_models_batched,
)
from scripts.run_single_window_validation import fill_feature_matrix, long_to_feature_matrix
from src.features.pipeline import compute_composite_features


def _base_features_for_trade_date(trade_date: str) -> pd.DataFrame:
    rows = [
        ("AAA", trade_date, "ret_20d", 0.10),
        ("BBB", trade_date, "ret_20d", 0.30),
        ("CCC", trade_date, "ret_20d", 0.20),
        ("AAA", trade_date, "ret_60d", 0.40),
        ("BBB", trade_date, "ret_60d", 0.10),
        ("CCC", trade_date, "ret_60d", 0.20),
        ("AAA", trade_date, "vol_20d", 0.30),
        ("BBB", trade_date, "vol_20d", 0.10),
        ("CCC", trade_date, "vol_20d", 0.20),
        ("AAA", trade_date, "volume_ratio_20d", 1.20),
        ("BBB", trade_date, "volume_ratio_20d", 0.80),
        ("CCC", trade_date, "volume_ratio_20d", 1.00),
        ("AAA", trade_date, "pe_ratio", 15.0),
        ("BBB", trade_date, "pe_ratio", 12.0),
        ("CCC", trade_date, "pe_ratio", 18.0),
        ("AAA", trade_date, "pb_ratio", 2.0),
        ("BBB", trade_date, "pb_ratio", 1.5),
        ("CCC", trade_date, "pb_ratio", 2.5),
    ]
    frame = pd.DataFrame(rows, columns=["ticker", "trade_date", "feature_name", "feature_value"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["is_filled"] = False
    frame["_raw_missing"] = frame["feature_value"].isna()
    return frame


def test_finalize_fusion_date_feature_matrix_matches_one_shot_cross_section() -> None:
    base_features = _base_features_for_trade_date("2025-01-03")
    retained_features = [
        "ret_20d",
        "momentum_rank_20d",
        "momentum_rank_60d",
        "vol_rank",
        "value_mom_pe",
        "is_missing_value_mom_pe",
    ]

    actual = finalize_fusion_date_feature_matrix(
        base_features=base_features,
        retained_features=retained_features,
    )

    global_technical = build_cross_sectional_technical_features_for_date(base_features)
    composite_inputs = pd.concat(
        [
            base_features[["ticker", "trade_date", "feature_name", "feature_value"]],
            global_technical[["ticker", "trade_date", "feature_name", "feature_value"]],
        ],
        ignore_index=True,
    )
    composites = compute_composite_features(composite_inputs)
    raw_features = pd.concat(
        [
            base_features,
            global_technical,
            composites.assign(is_filled=False, _raw_missing=lambda frame: frame["feature_value"].isna()),
        ],
        ignore_index=True,
    )
    winsorized = preprocessing_module.winsorize_features(raw_features)
    normalized = preprocessing_module.rank_normalize_features(winsorized)
    raw_missing = normalized.pop("_raw_missing").astype(bool)
    finalized = preprocessing_module.add_missing_flags(normalized, raw_missing)
    expected_long = finalized.loc[
        finalized["feature_name"].astype(str).isin(expand_requested_feature_names(retained_features)),
        ["ticker", "trade_date", "feature_name", "feature_value"],
    ].copy()
    expected = fill_feature_matrix(long_to_feature_matrix(expected_long, retained_features))

    pd.testing.assert_frame_equal(actual.sort_index(), expected.sort_index())


class _ColumnModel:
    def __init__(self, feature_name: str):
        self.feature_name = feature_name

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X[self.feature_name]


def test_score_fusion_models_batched_combines_date_partitions(tmp_path: Path) -> None:
    trade_dates = [pd.Timestamp("2025-01-03"), pd.Timestamp("2025-01-10")]
    retained_features = ["ret_20d", "momentum_rank_20d", "value_mom_pe"]

    for trade_date in trade_dates:
        date_dir = tmp_path / trade_date.date().isoformat()
        date_dir.mkdir(parents=True)
        base_features = _base_features_for_trade_date(trade_date.date().isoformat())
        first_batch = base_features.loc[base_features["ticker"].isin(["AAA", "BBB"])].copy()
        second_batch = base_features.loc[base_features["ticker"].isin(["CCC"])].copy()
        first_batch.to_parquet(date_dir / "batch_001.parquet", index=False)
        second_batch.to_parquet(date_dir / "batch_002.parquet", index=False)

    label_index = pd.MultiIndex.from_tuples(
        [
            (trade_dates[0], "AAA"),
            (trade_dates[0], "BBB"),
            (trade_dates[0], "CCC"),
            (trade_dates[1], "AAA"),
            (trade_dates[1], "BBB"),
            (trade_dates[1], "CCC"),
        ],
        names=["trade_date", "ticker"],
    )
    label_series = pd.Series([0.1, 0.2, 0.3, 0.2, 0.1, 0.0], index=label_index, name="excess_return")

    predictions, feature_diagnostics, sample_size_base, peak_rss_gb = score_fusion_models_batched(
        temp_dir=tmp_path,
        trade_dates=trade_dates,
        tickers=["AAA", "BBB", "CCC"],
        retained_features=retained_features,
        model_specs={
            "ridge": {
                "model_object": _ColumnModel("ret_20d"),
                "feature_names": ["ret_20d"],
            },
        },
        label_series=label_series,
        batch_size=2,
    )

    assert "ridge" in predictions
    assert len(predictions["ridge"]) == 6
    assert feature_diagnostics["batch_mode"] == "ticker_batched_date_partitioned"
    assert feature_diagnostics["rows"] == 6
    assert sample_size_base["aligned_rows_before_filter"] == 6
    assert sample_size_base["friday_dates_before_filter"] == 2
    assert peak_rss_gb > 0.0


def test_rank_series_to_unit_interval_matches_expected_average_ranks() -> None:
    series = pd.Series([3.0, 1.0, 2.0, 2.0], index=list("ABCD"), dtype=float)

    ranked = rank_series_to_unit_interval(series)

    assert ranked["B"] == pytest.approx(0.0)
    assert ranked["C"] == pytest.approx(0.5)
    assert ranked["D"] == pytest.approx(0.5)
    assert ranked["A"] == pytest.approx(1.0)
