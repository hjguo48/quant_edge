from __future__ import annotations

from datetime import date
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import scripts.run_family_ablation as ablation_module
from src.features.registry import FeatureDefinition


class _FakeRegistry:
    def __init__(self, definitions: list[FeatureDefinition]) -> None:
        self._definitions = definitions

    def list_features(self) -> list[FeatureDefinition]:
        return list(self._definitions)


def test_parse_horizon_labels_accepts_numeric_and_suffix() -> None:
    assert ablation_module.parse_horizon_labels("20,60d") == ["20d", "60d"]


def test_build_family_feature_sets_keeps_only_passed_rows() -> None:
    retained_rows = pd.DataFrame(
        [
            {"feature": "alpha", "family": "technical", "status": "PASS"},
            {"feature": "beta", "family": "technical", "status": "FAIL"},
            {"feature": "gamma", "family": "shorting", "status": "PASS"},
        ],
    )

    family_sets = ablation_module.build_family_feature_sets(retained_rows=retained_rows)

    assert family_sets == {"shorting": ["gamma"], "technical": ["alpha"]}


def test_run_feature_subset_walkforward_returns_metrics_without_engine(monkeypatch) -> None:
    window = SimpleNamespace(
        window_id="W1",
        train_start=date(2024, 1, 5),
        train_end=date(2024, 1, 5),
        validation_start=date(2024, 1, 12),
        validation_end=date(2024, 1, 12),
        test_start=date(2024, 1, 19),
        test_end=date(2024, 1, 19),
    )
    monkeypatch.setattr(ablation_module, "screening_windows", lambda: (window,))

    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-05"), "AAA"),
            (pd.Timestamp("2024-01-05"), "BBB"),
            (pd.Timestamp("2024-01-05"), "CCC"),
            (pd.Timestamp("2024-01-12"), "AAA"),
            (pd.Timestamp("2024-01-12"), "BBB"),
            (pd.Timestamp("2024-01-12"), "CCC"),
            (pd.Timestamp("2024-01-19"), "AAA"),
            (pd.Timestamp("2024-01-19"), "BBB"),
            (pd.Timestamp("2024-01-19"), "CCC"),
        ],
        names=["trade_date", "ticker"],
    )
    feature_matrix = pd.DataFrame(
        {"alpha": [1.0, 2.0, 3.0, 1.1, 2.1, 3.1, 1.2, 2.2, 3.2]},
        index=index,
    )
    label_series = pd.Series(
        [0.1, 0.2, 0.3, 0.11, 0.21, 0.31, 0.12, 0.22, 0.32],
        index=index,
        name="excess_return",
    )

    metrics = ablation_module.run_feature_subset_walkforward(
        feature_matrix=feature_matrix,
        label_series=label_series,
        horizon_label="20d",
        rebalance_weekday=4,
    )

    assert metrics["window_count"] == 1
    assert metrics["feature_count"] == 1
    assert metrics["ic"] is not None
    assert metrics["per_window_ic"]["W1"] > 0.99


def test_generate_family_ablation_report_computes_only_one_and_leave_one(monkeypatch) -> None:
    retained_rows = pd.DataFrame(
        [
            {"feature": "alpha", "family": "technical", "status": "PASS"},
            {"feature": "beta", "family": "shorting", "status": "PASS"},
            {"feature": "gamma", "family": "shorting", "status": "PASS"},
        ],
    )
    feature_matrix = pd.DataFrame(
        {
            "alpha": [1.0, 2.0],
            "beta": [3.0, 4.0],
            "gamma": [5.0, 6.0],
        },
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-05"), "AAA"), (pd.Timestamp("2024-01-12"), "AAA")],
            names=["trade_date", "ticker"],
        ),
    )
    label_series = pd.Series([0.1, 0.2], index=feature_matrix.index, name="excess_return")

    metrics_by_subset = {
        ("alpha", "beta", "gamma"): {"ic": 0.04, "t_stat": 3.1, "window_count": 11},
        ("alpha",): {"ic": 0.02, "t_stat": 2.2, "window_count": 11},
        ("beta", "gamma"): {"ic": 0.03, "t_stat": 2.7, "window_count": 11},
        ("alpha", "beta"): {"ic": 0.01, "t_stat": 1.1, "window_count": 11},
    }

    def fake_walkforward(*, feature_matrix, label_series, horizon_label, rebalance_weekday):
        return metrics_by_subset[tuple(feature_matrix.columns)]

    monkeypatch.setattr(ablation_module, "run_feature_subset_walkforward", fake_walkforward)

    report = ablation_module.generate_family_ablation_report(
        horizon_label="20d",
        feature_matrix=feature_matrix,
        label_series=label_series,
        retained_rows=retained_rows,
    )

    assert report["baseline_full_ic"] == 0.04
    only_one = {row["family"]: row for row in report["only_one_family"]}
    leave_one = {row["family"]: row for row in report["leave_one_family_out"]}
    assert only_one["technical"]["ic"] == 0.02
    assert only_one["shorting"]["ic"] == 0.03
    assert leave_one["technical"]["ic_delta"] == -0.01
    assert leave_one["shorting"]["ic_delta"] == -0.02


def test_main_writes_family_ablation_json(tmp_path: Path, monkeypatch) -> None:
    definitions = [
        FeatureDefinition(name="alpha", category="technical", description="x", compute_fn=lambda **_: None),
    ]
    monkeypatch.setattr(ablation_module, "build_feature_registry", lambda: _FakeRegistry(definitions))
    monkeypatch.setattr(
        ablation_module,
        "build_registry_feature_maps",
        lambda registry: ({"alpha": "technical"}, ["alpha"]),
    )
    monkeypatch.setattr(
        ablation_module,
        "load_horizon_families",
        lambda path: {
            "60d": {"families": ["technical"], "excluded_families": [], "rationale": "x"},
        },
    )
    monkeypatch.setattr(ablation_module, "load_missingness_exclusion_map", lambda path: {})
    monkeypatch.setattr(
        ablation_module,
        "build_panel_context",
        lambda **kwargs: SimpleNamespace(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31),
            sampled_tickers=("AAA",),
            sampled_trade_dates=(date(2024, 1, 5),),
            universe_size=503,
        ),
    )
    monkeypatch.setattr(ablation_module, "horizon_feature_names", lambda **kwargs: ["alpha"])
    monkeypatch.setattr(
        ablation_module,
        "build_or_load_week7_panel",
        lambda **kwargs: pd.DataFrame(
            [{"ticker": "AAA", "trade_date": date(2024, 1, 5), "feature_name": "alpha", "feature_value": 1.0}],
        ),
    )
    monkeypatch.setattr(
        ablation_module,
        "load_retained_rows",
        lambda screening_dir, horizon_label: pd.DataFrame(
            [{"feature": "alpha", "family": "technical", "status": "PASS"}],
        ),
    )
    monkeypatch.setattr(
        ablation_module,
        "load_label_series",
        lambda **kwargs: pd.Series(
            [0.1],
            index=pd.MultiIndex.from_tuples([(pd.Timestamp("2024-01-05"), "AAA")], names=["trade_date", "ticker"]),
            name="excess_return",
        ),
    )
    monkeypatch.setattr(
        ablation_module,
        "generate_family_ablation_report",
        lambda **kwargs: {
            "generated_at": "2026-04-25T00:00:00+00:00",
            "horizon": kwargs["horizon_label"],
            "model_type": "ridge_baseline",
            "baseline_full_ic": 0.03,
            "baseline_full_t_stat": 2.5,
            "baseline_window_count": 11,
            "retained_feature_count": 1,
            "only_one_family": [{"family": "technical", "feature_count": 1, "ic": 0.03, "t_stat": 2.5, "window_count": 11, "rank": 1}],
            "leave_one_family_out": [{"family": "technical", "feature_count_removed": 1, "feature_count_remaining": 0, "ic": None, "t_stat": None, "ic_delta": None, "window_count": 0, "rank": None}],
        },
    )

    output_dir = tmp_path / "reports"
    exit_code = ablation_module.main(
        [
            "--enable-flags",
            "--horizons",
            "60d",
            "--output",
            str(output_dir),
        ],
    )

    assert exit_code == 0
    output_path = output_dir / "family_ablation_60d.json"
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["baseline_full_ic"] == 0.03
