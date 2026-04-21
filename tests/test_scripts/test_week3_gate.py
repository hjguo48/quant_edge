from __future__ import annotations

from pathlib import Path

import scripts.run_week3_gate_verification as gate_module


def test_evaluate_gate1_flags_under_threshold() -> None:
    metrics = [
        gate_module.FeatureCoverageMetric(
            feature_name="gap_pct",
            expected=100,
            actual=97,
            coverage=0.97,
            missing_examples=[],
        ),
        gate_module.FeatureCoverageMetric(
            feature_name="close_to_vwap",
            expected=100,
            actual=90,
            coverage=0.90,
            missing_examples=[{"ticker": "AAA", "trade_date": "2026-01-05"}],
        ),
    ]

    summary = gate_module.evaluate_gate1(metrics)

    assert summary["pass"] is False
    assert summary["features"]["gap_pct"]["pass"] is True
    assert summary["features"]["close_to_vwap"]["pass"] is False
    assert summary["failing_examples"][0]["feature_name"] == "close_to_vwap"


def test_evaluate_gate2_requires_zero_blockers() -> None:
    summary = gate_module.evaluate_gate2(
        blocker_count=1,
        warning_count=5,
        sample_blockers=[{"ticker": "AAPL", "trade_date": "2026-04-17"}],
    )

    assert summary["pass"] is False
    assert summary["blocker_count_last_30d"] == 1
    assert summary["warning_count_last_30d"] == 5


def test_evaluate_gate3_requires_missing_outlier_and_documentation() -> None:
    metrics = [
        gate_module.FeatureQualityMetric(
            feature_name="gap_pct",
            total_rows=100,
            missing_rows=2,
            missing_rate=0.02,
            evaluated_rows=98,
            outlier_rows=0,
            outlier_rate=0.0,
            lag_rule_documented=True,
            missing_examples=[],
            outlier_examples=[],
        ),
        gate_module.FeatureQualityMetric(
            feature_name="close_to_vwap",
            total_rows=100,
            missing_rows=100,
            missing_rate=1.0,
            evaluated_rows=0,
            outlier_rows=0,
            outlier_rate=None,
            lag_rule_documented=False,
            missing_examples=[{"ticker": "AAA", "trade_date": "2026-01-05"}],
            outlier_examples=[],
        ),
    ]

    summary = gate_module.evaluate_gate3(metrics)

    assert summary["pass"] is False
    assert summary["features"]["gap_pct"]["pass"] is True
    assert summary["features"]["close_to_vwap"]["missing_rate_pass"] is False
    assert summary["features"]["close_to_vwap"]["lag_rule_documented"] is False
    assert any(item["issue_type"] == "lag_rule_undocumented" for item in summary["failing_examples"])


def test_load_lag_rule_documentation_recognizes_intraday_rule(tmp_path: Path) -> None:
    lineage_path = tmp_path / "data_lineage.yaml"
    lineage_path.write_text(
        """
intraday_feature_layer:
  applies_to:
    - gap_pct
    - overnight_ret
  knowledge_time_rule: trade_date + 1 business day close (next XNYS session close, T+1)
""".strip()
    )

    payload = gate_module.load_lag_rule_documentation(lineage_path)

    assert payload["exists"] is True
    assert payload["rule_mentions_t_plus_one"] is True
    assert payload["documented_features"] == ["gap_pct", "overnight_ret"]
