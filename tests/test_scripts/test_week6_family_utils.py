from __future__ import annotations

from datetime import date

import pandas as pd

import scripts._week6_family_utils as utils


class _FakePipeline:
    def run(self, **kwargs) -> pd.DataFrame:
        trade_day = kwargs["end_date"]
        return pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "trade_date": trade_day,
                    "feature_name": "high_vix_x_beta",
                    "feature_value": 0.42,
                },
                {
                    "ticker": "AAA",
                    "trade_date": trade_day,
                    "feature_name": "is_missing_high_vix_x_beta",
                    "feature_value": 1.0,
                },
                {
                    "ticker": "AAA",
                    "trade_date": trade_day,
                    "feature_name": "ret_5d",
                    "feature_value": 0.11,
                },
                {
                    "ticker": "AAA",
                    "trade_date": trade_day,
                    "feature_name": "stock_beta_252",
                    "feature_value": 1.2,
                },
                {
                    "ticker": "AAA",
                    "trade_date": trade_day,
                    "feature_name": "is_missing_ret_5d",
                    "feature_value": 0.0,
                },
                {
                    "ticker": "AAA",
                    "trade_date": trade_day,
                    "feature_name": "is_missing_stock_beta_252",
                    "feature_value": 0.0,
                },
            ],
        )


def test_collect_pipeline_missing_observations_prefers_feature_value_for_composite_family(monkeypatch) -> None:
    monkeypatch.setattr(utils, "FeaturePipeline", _FakePipeline)
    monkeypatch.setattr(utils, "_high_vix_context_available", lambda trade_day: True)

    observations = utils.collect_pipeline_missing_observations(
        sampled_tickers=("AAA",),
        sampled_trade_dates=(date(2026, 1, 5),),
        feature_family_map={
            "high_vix_x_beta": "composite",
            "ret_5d": "technical",
        },
    )

    feature_map = {
        row["feature_name"]: bool(row["is_missing"])
        for _, row in observations.iterrows()
    }
    assert feature_map["high_vix_x_beta"] is False
    assert feature_map["ret_5d"] is False
