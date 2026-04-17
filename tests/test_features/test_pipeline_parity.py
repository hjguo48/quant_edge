from __future__ import annotations

from datetime import date
from decimal import Decimal

import pandas as pd

from src.features.pipeline import feature_store_records_from_frame, prepare_feature_export_frame


def test_prepare_feature_export_frame_canonicalizes_and_dedupes() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": "aapl",
                "trade_date": "2026-04-16",
                "feature_name": "vol_60d",
                "feature_value": "0.25",
                "is_filled": None,
            },
            {
                "ticker": "AAPL",
                "trade_date": "2026-04-16",
                "feature_name": "vol_60d",
                "feature_value": "0.30",
                "is_filled": True,
            },
            {
                "ticker": "msft",
                "trade_date": "2026-04-16",
                "feature_name": "curve_inverted_x_growth",
                "feature_value": "1.0",
                "is_filled": False,
            },
        ],
    )

    prepared = prepare_feature_export_frame(raw)

    assert list(prepared.columns) == ["ticker", "trade_date", "feature_name", "feature_value", "is_filled"]
    assert len(prepared) == 2
    assert prepared.iloc[0]["ticker"] == "AAPL"
    assert prepared.iloc[0]["trade_date"] == date(2026, 4, 16)
    assert prepared.iloc[0]["feature_value"] == 0.30
    assert bool(prepared.iloc[0]["is_filled"]) is True


def test_feature_store_records_match_parquet_contract() -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "trade_date": date(2026, 4, 16),
                "feature_name": "curve_inverted_x_growth",
                "feature_value": 0.5,
                "is_filled": False,
            },
            {
                "ticker": "MSFT",
                "trade_date": date(2026, 4, 16),
                "feature_name": "is_missing_overnight_gap",
                "feature_value": 1.0,
                "is_filled": False,
            },
        ],
    )

    prepared = prepare_feature_export_frame(frame)
    records = feature_store_records_from_frame(frame, batch_id="batch-1")

    assert len(records) == len(prepared)
    assert records[0]["ticker"] == prepared.iloc[0]["ticker"]
    assert records[0]["calc_date"] == prepared.iloc[0]["trade_date"]
    assert records[0]["feature_name"] == prepared.iloc[0]["feature_name"]
    assert records[0]["feature_value"] == Decimal(str(prepared.iloc[0]["feature_value"]))
    assert records[0]["batch_id"] == "batch-1"
