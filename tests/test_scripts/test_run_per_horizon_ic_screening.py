from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

import scripts.run_per_horizon_ic_screening as screening_module
from src.features.registry import FeatureDefinition


class _FakeRegistry:
    def __init__(self, definitions: list[FeatureDefinition]) -> None:
        self._definitions = definitions

    def list_features(self) -> list[FeatureDefinition]:
        return list(self._definitions)


def test_screen_horizon_features_applies_thresholds_and_exclusions(monkeypatch) -> None:
    feature_matrix = pd.DataFrame(
        {
            "alpha": [1.0, 2.0],
            "beta": [3.0, 4.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-05"), "AAA"), (pd.Timestamp("2024-01-12"), "AAA")],
            names=["trade_date", "ticker"],
        ),
    )
    label_series = pd.Series(
        [0.1, 0.2],
        index=feature_matrix.index,
        name="excess_return",
    )
    metrics_by_feature = {
        "alpha": {"mean_ic": 0.03, "t_stat": 2.5, "sign_consistent_windows": 8},
        "beta": {"mean_ic": 0.01, "t_stat": 1.2, "sign_consistent_windows": 5},
    }

    def fake_metrics(*, feature_series, label_series, windows):
        return metrics_by_feature[str(feature_series.name)]

    monkeypatch.setattr(screening_module, "compute_feature_screening_metrics", fake_metrics)

    rows = screening_module.screen_horizon_features(
        feature_matrix=feature_matrix,
        label_series=label_series,
        all_features=["alpha", "beta", "gamma"],
        family_by_feature={"alpha": "technical", "beta": "technical", "gamma": "shorting"},
        included_features={"alpha", "beta"},
        exclusion_map={"gamma": "data_source_block"},
        mean_ic_threshold=0.015,
        t_stat_threshold=2.0,
        sign_window_threshold=7,
        windows=[SimpleNamespace(window_id="W1", test_start=date(2024, 1, 1), test_end=date(2024, 1, 31))],
    )

    row_map = rows.set_index("feature").to_dict(orient="index")
    assert row_map["alpha"]["status"] == "PASS"
    assert row_map["alpha"]["excluded_reason"] == ""
    assert row_map["beta"]["status"] == "FAIL"
    assert row_map["beta"]["excluded_reason"] == ""
    assert row_map["gamma"]["status"] == "FAIL"
    assert row_map["gamma"]["excluded_reason"] == "data_source_block"


def test_csv_output_path_uses_expected_filename() -> None:
    output = screening_module.csv_output_path(Path("/tmp/reports"), "20d")
    assert output == Path("/tmp/reports/ic_screening_v7_20d.csv")


def test_main_writes_reports_and_retained_yaml(tmp_path: Path, monkeypatch) -> None:
    definitions = [
        FeatureDefinition(name="alpha", category="technical", description="x", compute_fn=lambda **_: None),
        FeatureDefinition(name="beta", category="shorting", description="x", compute_fn=lambda **_: None),
    ]
    monkeypatch.setattr(screening_module, "build_feature_registry", lambda: _FakeRegistry(definitions))
    monkeypatch.setattr(
        screening_module,
        "build_registry_feature_maps",
        lambda registry: ({"alpha": "technical", "beta": "shorting"}, ["alpha", "beta"]),
    )
    monkeypatch.setattr(
        screening_module,
        "load_horizon_families",
        lambda path: {
            "1d": {"families": ["technical"], "excluded_families": [], "rationale": "x"},
            "5d": {"families": ["technical"], "excluded_families": [], "rationale": "x"},
            "20d": {"families": ["technical", "shorting"], "excluded_families": [], "rationale": "x"},
            "60d": {"families": ["technical"], "excluded_families": [], "rationale": "x"},
        },
    )
    monkeypatch.setattr(screening_module, "load_missingness_exclusion_map", lambda path: {"beta": "data_source_block"})
    monkeypatch.setattr(
        screening_module,
        "build_panel_context",
        lambda **kwargs: SimpleNamespace(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            sampled_tickers=("AAA",),
            sampled_trade_dates=(date(2024, 1, 5),),
            universe_size=503,
        ),
    )
    monkeypatch.setattr(
        screening_module,
        "horizon_feature_names",
        lambda **kwargs: ["alpha"] if kwargs["horizon_label"] != "20d" else ["alpha"],
    )
    panel = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 5), "feature_name": "alpha", "feature_value": 1.0},
        ],
    )
    monkeypatch.setattr(screening_module, "build_or_load_week7_panel", lambda **kwargs: panel)
    monkeypatch.setattr(
        screening_module,
        "load_label_series",
        lambda **kwargs: pd.Series(
            [0.1],
            index=pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-05"), "AAA")], names=["trade_date", "ticker"]),
            name="excess_return",
        ),
    )
    monkeypatch.setattr(
        screening_module,
        "screen_horizon_features",
        lambda **kwargs: pd.DataFrame(
            [
                {"feature": "alpha", "family": "technical", "mean_ic": 0.02, "t_stat": 2.1, "sign_consistent_windows": 8, "status": "PASS", "excluded_reason": ""},
                {"feature": "beta", "family": "shorting", "mean_ic": None, "t_stat": None, "sign_consistent_windows": 0, "status": "FAIL", "excluded_reason": "data_source_block"},
            ],
        ),
    )

    output_dir = tmp_path / "ic_v7"
    retained_path = tmp_path / "horizon_retained_features.yaml"
    exit_code = screening_module.main(
        [
            "--enable-flags",
            "--output-dir",
            str(output_dir),
            "--retained-output",
            str(retained_path),
        ],
    )

    assert exit_code == 0
    for horizon in ("1d", "5d", "20d", "60d"):
        assert (output_dir / f"ic_screening_v7_{horizon}.csv").exists()
    retained = yaml.safe_load(retained_path.read_text(encoding="utf-8"))
    assert retained["horizons"]["1d"]["retained"] == ["alpha"]
    assert retained["horizons"]["20d"]["top_3_families"] == ["technical"]
