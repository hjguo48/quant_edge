from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

import scripts.run_missingness_audit as audit_module
from scripts._week6_family_utils import SampleContext
from src.features.registry import FeatureDefinition


class _FakeRegistry:
    def __init__(self, definitions: list[FeatureDefinition]) -> None:
        self._definitions = definitions

    def list_features(self) -> list[FeatureDefinition]:
        return list(self._definitions)


def test_classify_missingness_uses_abnormal_off_exchange_override() -> None:
    result = audit_module.classify_missingness(
        feature_name="abnormal_off_exchange_shorting",
        family="shorting",
        missing_rate=1.0,
    )

    assert result["category"] == "data_source_block"
    assert result["recommendation"] == "drop_pending_paid_subscription"


def test_generate_missingness_audit_has_expected_summary() -> None:
    context = SampleContext(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        universe=("AAA", "BBB", "CCC"),
        sampled_tickers=("AAA", "BBB"),
        sampled_trade_dates=(date(2026, 1, 5), date(2026, 1, 12)),
    )
    observations = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 1, 12), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "BBB", "trade_date": date(2026, 1, 12), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "abnormal_off_exchange_shorting", "family": "shorting", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "abnormal_off_exchange_shorting", "family": "shorting", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 1, 12), "feature_name": "abnormal_off_exchange_shorting", "family": "shorting", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 1, 12), "feature_name": "abnormal_off_exchange_shorting", "family": "shorting", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "earnings_surprise_latest", "family": "earnings", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "earnings_surprise_latest", "family": "earnings", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 1, 12), "feature_name": "earnings_surprise_latest", "family": "earnings", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 1, 12), "feature_name": "earnings_surprise_latest", "family": "earnings", "is_missing": True},
        ],
    )
    family_by_feature = {
        "ret_5d": "technical",
        "abnormal_off_exchange_shorting": "shorting",
        "earnings_surprise_latest": "earnings",
    }

    report = audit_module.generate_missingness_audit(
        context=context,
        observations=observations,
        family_by_feature=family_by_feature,
    )

    features = {item["feature"]: item for item in report["features"]}
    assert features["ret_5d"]["recommendation"] == "keep"
    assert features["abnormal_off_exchange_shorting"]["category"] == "data_source_block"
    assert features["earnings_surprise_latest"]["category"] == "era"
    assert report["summary"]["keep"] == 1
    assert report["summary"]["drop"] == 1
    assert report["summary"]["convert_to_imputed"] == 1
    assert report["summary"]["vendor_bias"] == 0
    assert report["summary"]["data_source_block"] == 1
    assert report["summary"]["sample_disabled_pipeline"] == 0
    assert report["summary"]["dropped_features"] == ["abnormal_off_exchange_shorting"]
    assert report["summary"]["data_source_block_features"] == ["abnormal_off_exchange_shorting"]
    assert report["summary"]["sample_disabled_features"] == []


def test_generate_missingness_audit_marks_all_missing_family_as_sample_disabled() -> None:
    context = SampleContext(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        universe=("AAA", "BBB"),
        sampled_tickers=("AAA", "BBB"),
        sampled_trade_dates=(date(2026, 1, 5),),
    )
    observations = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "large_trade_ratio", "family": "trade_microstructure", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "large_trade_ratio", "family": "trade_microstructure", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "offhours_trade_ratio", "family": "trade_microstructure", "is_missing": True},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "offhours_trade_ratio", "family": "trade_microstructure", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
            {"ticker": "BBB", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": False},
        ],
    )

    report = audit_module.generate_missingness_audit(
        context=context,
        observations=observations,
        family_by_feature={
            "large_trade_ratio": "trade_microstructure",
            "offhours_trade_ratio": "trade_microstructure",
            "ret_5d": "technical",
        },
    )

    features = {item["feature"]: item for item in report["features"]}
    assert features["large_trade_ratio"]["category"] == "sample_disabled_pipeline"
    assert features["large_trade_ratio"]["recommendation"] == "re_run_with_dependencies"
    assert features["offhours_trade_ratio"]["category"] == "sample_disabled_pipeline"
    assert report["summary"]["sample_disabled_pipeline"] == 2
    assert report["summary"]["sample_disabled_features"] == ["large_trade_ratio", "offhours_trade_ratio"]
    assert report["summary"]["drop"] == 0


def test_generate_missingness_audit_summary_splits_vendor_bias_and_data_source_block() -> None:
    context = SampleContext(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        universe=("AAA",),
        sampled_tickers=("AAA",),
        sampled_trade_dates=(date(2026, 1, 5),),
    )
    observations = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "ret_5d", "family": "technical", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "ret_10d", "family": "technical", "is_missing": False},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "abnormal_off_exchange_shorting", "family": "shorting", "is_missing": True},
        ],
    )

    report = audit_module.generate_missingness_audit(
        context=context,
        observations=observations,
        family_by_feature={
            "ret_5d": "technical",
            "ret_10d": "technical",
            "abnormal_off_exchange_shorting": "shorting",
        },
    )

    assert report["summary"]["vendor_bias"] == 1
    assert report["summary"]["data_source_block"] == 1
    assert report["summary"]["dropped_features"] == ["abnormal_off_exchange_shorting", "ret_5d"]
    assert report["summary"]["data_source_block_features"] == ["abnormal_off_exchange_shorting"]


def test_main_writes_missingness_audit(tmp_path: Path, monkeypatch) -> None:
    definitions = [
        FeatureDefinition(name="ret_5d", category="technical", description="x", compute_fn=lambda **_: None),
        FeatureDefinition(name="earnings_surprise_latest", category="earnings", description="x", compute_fn=lambda **_: None),
        FeatureDefinition(name="earnings_beat_streak", category="earnings", description="x", compute_fn=lambda **_: None),
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
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "earnings_surprise_latest", "family": "earnings", "is_missing": True},
            {"ticker": "AAA", "trade_date": date(2026, 1, 5), "feature_name": "earnings_beat_streak", "family": "earnings", "is_missing": False},
        ],
    )
    monkeypatch.setattr(audit_module, "build_feature_registry", lambda: _FakeRegistry(definitions))
    monkeypatch.setattr(audit_module, "build_sample_context", lambda **kwargs: fake_context)
    monkeypatch.setattr(audit_module, "collect_feature_observations", lambda *args, **kwargs: fake_observations)

    output_path = tmp_path / "missingness_audit.json"
    exit_code = audit_module.main(["--output", str(output_path)])
    full_output_path = audit_module.full_report_output_path(output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    full_payload = json.loads(full_output_path.read_text(encoding="utf-8"))
    feature_map = {item["feature"]: item for item in full_payload["features"]}
    assert exit_code == 0
    assert "features" not in payload
    assert payload["summary"]["keep"] == 2
    assert feature_map["ret_5d"]["recommendation"] == "keep"
    assert feature_map["earnings_surprise_latest"]["recommendation"] == "convert_to_imputed"
