from __future__ import annotations

import pandas as pd

import scripts.run_greyscale_live as greyscale_live


def test_compute_layer1_diagnostics_warns_on_per_feature_dropout() -> None:
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2026-04-24"), "AAA"),
            (pd.Timestamp("2026-04-24"), "BBB"),
            (pd.Timestamp("2026-04-24"), "CCC"),
            (pd.Timestamp("2026-04-24"), "DDD"),
        ],
        names=["trade_date", "ticker"],
    )
    matrix = pd.DataFrame(
        {
            "dense_feature": [1.0, 2.0, 3.0, 4.0],
            "dropout_feature": [1.0, None, None, 4.0],
        },
        index=index,
    )

    diagnostics = greyscale_live.compute_layer1_diagnostics(
        raw_feature_matrix=matrix,
        warn_threshold=0.20,
    )

    assert diagnostics["warning_triggered"] is True
    assert diagnostics["warning_name"] == "layer1_per_feature_dropout"
    assert diagnostics["latest_trade_date"] == "2026-04-24"
    assert diagnostics["features_over_threshold"] == {"dropout_feature": 0.5}
    assert diagnostics["per_feature_null_rates"]["dense_feature"] == 0.0


def test_compute_layer1_diagnostics_passes_when_latest_slice_is_dense() -> None:
    latest_tickers = [f"T{i:02d}" for i in range(20)]
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2026-04-17"), "AAA")]
        + [(pd.Timestamp("2026-04-24"), ticker) for ticker in latest_tickers],
        names=["trade_date", "ticker"],
    )
    matrix = pd.DataFrame(
        {
            "stable_feature": [None] + [float(i) for i in range(20)],
            "mostly_dense_feature": [1.0] + [float(i) for i in range(19)] + [None],
        },
        index=index,
    )

    diagnostics = greyscale_live.compute_layer1_diagnostics(
        raw_feature_matrix=matrix,
        warn_threshold=0.20,
    )

    assert diagnostics["warning_triggered"] is False
    assert diagnostics["warning_name"] is None
    assert diagnostics["latest_trade_date"] == "2026-04-24"
    assert diagnostics["features_over_threshold"] == {}
    assert diagnostics["per_feature_null_rates"]["mostly_dense_feature"] == 0.05
