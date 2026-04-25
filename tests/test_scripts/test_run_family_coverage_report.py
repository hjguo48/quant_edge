from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

import scripts.run_family_coverage_report as coverage_module
from scripts._week6_family_utils import SampleContext
from src.features.registry import FeatureDefinition


class _FakeRegistry:
    def __init__(self, definitions: list[FeatureDefinition]) -> None:
        self._definitions = definitions

    def list_features(self) -> list[FeatureDefinition]:
        return list(self._definitions)


def test_generate_family_coverage_report_has_expected_schema() -> None:
    context = SampleContext(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 2, 28),
        universe=("AAA", "BBB", "CCC"),
        sampled_tickers=("AAA", "BBB"),
        sampled_trade_dates=(date(2026, 1, 5), date(2026, 2, 10)),
    )
    observations = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "ret_10d", "family": "technical", "is_missing": False},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "ret_10d", "family": "technical", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 2, 10), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "BBB", "trade_date": date(2026, 2, 10), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 2, 10), "feature_name": "ret_10d", "family": "technical", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 2, 10), "feature_name": "ret_10d", "family": "technical", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "vix", "family": "macro", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "vix", "family": "macro", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 2, 10), "feature_name": "vix", "family": "macro", "is_missing": False},
            {"ticker": "BBB", "trade_date": date(2026, 2, 10), "feature_name": "vix", "family": "macro", "is_missing": False},
        ],
    )
    family_by_feature = {"ret_5d": "technical", "ret_10d": "technical", "vix": "macro"}

    report = coverage_module.generate_family_coverage_report(
        context=context,
        observations=observations,
        family_by_feature=family_by_feature,
    )

    assert report["universe_size"] == 3
    assert report["date_range"] == {"start": "2026-01-01", "end": "2026-02-28"}
    assert set(report["families"]) == {"macro", "technical"}
    assert report["families"]["technical"]["feature_count"] == 2
    assert report["families"]["technical"]["monthly_coverage"][0] == {
        "month": "2026-01",
        "missing_rate": 0.25,
        "tickers_with_data": 2,
    }
    assert report["families"]["technical"]["first_available_date"] == "2026-01-05"
    assert report["families"]["technical"]["last_available_date"] == "2026-02-10"


def test_generate_family_coverage_report_handles_all_missing_family() -> None:
    context = SampleContext(
        start_date=date(2026, 3, 1),
        end_date=date(2026, 3, 31),
        universe=("AAA",),
        sampled_tickers=("AAA",),
        sampled_trade_dates=(date(2026, 3, 3),),
    )
    observations = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 3, 3), "feature_name": "foo", "family": "earnings", "is_missing": True},
        ],
    )

    report = coverage_module.generate_family_coverage_report(
        context=context,
        observations=observations,
        family_by_feature={"foo": "earnings"},
    )

    family = report["families"]["earnings"]
    assert family["monthly_coverage"] == [{"month": "2026-03", "missing_rate": 1.0, "tickers_with_data": 0}]
    assert family["first_available_date"] is None
    assert family["last_available_date"] is None


def test_main_writes_family_coverage_report(tmp_path: Path, monkeypatch) -> None:
    definitions = [
        FeatureDefinition(name="ret_5d", category="technical", description="x", compute_fn=lambda **_: None),
        FeatureDefinition(name="vix", category="macro", description="x", compute_fn=lambda **_: None),
    ]
    fake_context = SampleContext(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        universe=("AAA", "BBB"),
        sampled_tickers=("AAA",),
        sampled_trade_dates=(date(2026, 1, 5),),
    )
    fake_observations = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "vix", "family": "macro", "is_missing": True},
        ],
    )
    monkeypatch.setattr(coverage_module, "build_feature_registry", lambda: _FakeRegistry(definitions))
    monkeypatch.setattr(coverage_module, "build_sample_context", lambda **kwargs: fake_context)
    monkeypatch.setattr(coverage_module, "collect_feature_observations", lambda *args, **kwargs: fake_observations)

    output_path = tmp_path / "family_coverage_report.json"
    exit_code = coverage_module.main(["--output", str(output_path)])

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["families"]["technical"]["feature_count"] == 1
    assert payload["families"]["macro"]["monthly_coverage"][0]["missing_rate"] == 1.0
