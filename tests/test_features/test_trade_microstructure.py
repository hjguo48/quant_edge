from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import settings
from src.features import registry as registry_module
from src.features.alternative import ALTERNATIVE_FEATURE_NAMES
from src.features.fundamental import FUNDAMENTAL_FEATURE_NAMES
from src.features.intraday import INTRADAY_FEATURE_NAMES
from src.features.macro import MACRO_FEATURE_NAMES
from src.features.pipeline import COMPOSITE_FEATURE_NAMES
from src.features.sector_rotation import SECTOR_ROTATION_FEATURE_NAMES
from src.features.technical import TECHNICAL_FEATURE_NAMES
from src.features.trade_microstructure import (
    TRADE_MICROSTRUCTURE_FEATURE_NAMES,
    compute_knowledge_time,
    compute_large_trade_ratio,
    compute_late_day_aggressiveness,
    compute_off_exchange_volume_ratio,
    compute_offhours_trade_ratio,
    compute_trade_imbalance_proxy,
)


def _et(raw: str) -> pd.Timestamp:
    return pd.Timestamp(raw, tz="America/New_York").tz_convert("UTC")


def _trades(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "ticker": "AAPL",
        "price": 100.0,
        "size": 100.0,
        "exchange": 1,
        "trf_id": None,
        "trf_timestamp": None,
        "conditions": [0],
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def test_trade_imbalance_proxy_happy_path_tick_rule() -> None:
    trades = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:01"), "price": 101, "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:02"), "price": 101, "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:03"), "price": 99, "size": 100},
        ],
    )

    assert compute_trade_imbalance_proxy(trades) == pytest.approx(0.25)


def test_trade_imbalance_proxy_returns_nan_without_regular_session_trades() -> None:
    trades = _trades([{"sip_timestamp": _et("2026-01-05 19:55"), "price": 100, "size": 100}])

    assert np.isnan(compute_trade_imbalance_proxy(trades))


def test_condition_filter_drops_non_allowed_trade_conditions() -> None:
    trades = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 100, "conditions": [1]},
            {"sip_timestamp": _et("2026-01-05 10:01"), "price": 101, "size": 100, "conditions": [9]},
            {"sip_timestamp": _et("2026-01-05 10:02"), "price": 102, "size": 100, "conditions": [1]},
        ],
    )

    assert compute_trade_imbalance_proxy(trades, condition_allow={1}) == pytest.approx(0.5)


def test_large_trade_ratio_happy_path_dollar_based() -> None:
    trades = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:01"), "price": 101, "size": 50},
        ],
    )

    expected = 10_000 / (10_000 + 5_050)
    assert compute_large_trade_ratio(trades, size_threshold_dollars=10_000) == pytest.approx(expected)


def test_large_trade_ratio_returns_nan_when_total_dollar_volume_zero() -> None:
    trades = _trades([{"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 0}])

    assert np.isnan(compute_large_trade_ratio(trades))


def test_late_day_aggressiveness_happy_path_clipped_ratio() -> None:
    trades = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:01"), "price": 101, "size": 100},
            {"sip_timestamp": _et("2026-01-05 15:00"), "price": 100, "size": 100},
            {"sip_timestamp": _et("2026-01-05 15:01"), "price": 99, "size": 100},
        ],
    )

    assert compute_late_day_aggressiveness(trades) == pytest.approx(2.0)


def test_late_day_aggressiveness_returns_nan_when_full_imbalance_zero() -> None:
    trades = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:01"), "price": 100, "size": 100},
        ],
    )

    assert np.isnan(compute_late_day_aggressiveness(trades))


def test_offhours_trade_ratio_happy_path() -> None:
    trades = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 08:00"), "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:00"), "size": 200},
            {"sip_timestamp": _et("2026-01-05 19:00"), "size": 100},
        ],
    )

    assert compute_offhours_trade_ratio(trades) == pytest.approx(0.5)


def test_offhours_trade_ratio_returns_nan_when_no_volume() -> None:
    trades = _trades([{"sip_timestamp": _et("2026-01-05 10:00"), "size": 0}])

    assert np.isnan(compute_offhours_trade_ratio(trades))


def test_off_exchange_volume_ratio_happy_path_three_way_detection() -> None:
    trades = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 1, "exchange": 4},
            {"sip_timestamp": _et("2026-01-05 10:01"), "price": 100, "size": 1, "exchange": 1, "trf_id": "T"},
            {
                "sip_timestamp": _et("2026-01-05 10:02"),
                "price": 100,
                "size": 1,
                "exchange": 1,
                "trf_timestamp": _et("2026-01-05 10:02"),
            },
            {"sip_timestamp": _et("2026-01-05 10:03"), "price": 100, "size": 1, "exchange": 1},
        ],
    )

    assert compute_off_exchange_volume_ratio(trades, trf_exchange_codes={4, 202}) == pytest.approx(0.75)


def test_off_exchange_volume_ratio_returns_nan_when_regular_dollar_volume_zero() -> None:
    trades = _trades([{"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 0, "exchange": 4}])

    assert np.isnan(compute_off_exchange_volume_ratio(trades, trf_exchange_codes={4}))


def test_compute_knowledge_time_pit_split_regular_and_offhours() -> None:
    trading_date = date(2026, 1, 5)

    assert compute_knowledge_time(trading_date, "trade_imbalance_proxy").isoformat() == "2026-01-05T21:15:00+00:00"
    assert compute_knowledge_time(trading_date, "offhours_trade_ratio").isoformat() == "2026-01-06T01:15:00+00:00"


def test_regular_session_features_do_not_leak_1955_et_trade() -> None:
    regular_only = _trades(
        [
            {"sip_timestamp": _et("2026-01-05 10:00"), "price": 100, "size": 100},
            {"sip_timestamp": _et("2026-01-05 10:01"), "price": 101, "size": 100},
        ],
    )
    with_late_offhours = pd.concat(
        [
            regular_only,
            _trades([{"sip_timestamp": _et("2026-01-05 19:55"), "price": 1, "size": 10_000}]),
        ],
        ignore_index=True,
    )

    assert compute_trade_imbalance_proxy(with_late_offhours) == compute_trade_imbalance_proxy(regular_only)


def test_trade_microstructure_registry_metadata_is_default_off() -> None:
    registry = registry_module.FeatureRegistry()
    trade_defs = registry.list_features("trade_microstructure")
    default_feature_names = set().union(
        TECHNICAL_FEATURE_NAMES,
        FUNDAMENTAL_FEATURE_NAMES,
        MACRO_FEATURE_NAMES,
        ALTERNATIVE_FEATURE_NAMES,
        SECTOR_ROTATION_FEATURE_NAMES,
        COMPOSITE_FEATURE_NAMES,
        INTRADAY_FEATURE_NAMES,
    )

    assert settings.ENABLE_TRADE_MICROSTRUCTURE_FEATURES is False
    assert {definition.name for definition in trade_defs} == set(TRADE_MICROSTRUCTURE_FEATURE_NAMES)
    assert set(TRADE_MICROSTRUCTURE_FEATURE_NAMES).isdisjoint(default_feature_names)


def test_feature_pipeline_source_does_not_import_trade_microstructure() -> None:
    """Plan P2 contract pin: FeaturePipeline default output MUST NOT include trade_microstructure.

    Task 7 achieves this 'passive gating' by NOT importing trade_microstructure in pipeline.py.
    Any future refactor that adds such an import would activate trade features in the default
    V5 bundle output silently; this test catches that regression.
    """
    from pathlib import Path

    pipeline_source = Path("src/features/pipeline.py").read_text(encoding="utf-8")
    assert "trade_microstructure" not in pipeline_source, (
        "src/features/pipeline.py imports trade_microstructure — default-off gating broken. "
        "Task 8 must make this an explicit opt-in (e.g., guarded by ENABLE_TRADE_MICROSTRUCTURE_FEATURES)."
    )


def test_trade_microstructure_opt_in_documented_not_implemented(monkeypatch) -> None:
    """Plan P2 contract pin: ENABLE_TRADE_MICROSTRUCTURE_FEATURES=True currently does NOT change
    FeaturePipeline default output (Task 7 chose passive gating). Task 8 will wire the flag when
    adding the batch builder. This test locks the current contract so Task 8 knows to revisit.
    """
    from src.features import pipeline as pipeline_module

    monkeypatch.setattr(settings, "ENABLE_TRADE_MICROSTRUCTURE_FEATURES", True)
    pipeline_source = inspect_module_source(pipeline_module)
    assert "trade_microstructure" not in pipeline_source
    assert "ENABLE_TRADE_MICROSTRUCTURE_FEATURES" not in pipeline_source


def inspect_module_source(module) -> str:
    import inspect

    return inspect.getsource(module)


def test_naive_timestamp_raises_value_error() -> None:
    """Plan PIT contract: upstream must supply tz-aware UTC sip_timestamp. Naive datetimes would
    silently coerce via utc=True and get misinterpreted as UTC — ET wall-clock 10:00 would land at
    05:00 ET after round-trip, outside the 09:30-16:00 window, producing silent NaN feature values.
    """
    trades = pd.DataFrame(
        [
            {
                "sip_timestamp": pd.Timestamp("2026-01-05 10:00"),  # naive
                "price": 100.0,
                "size": 100.0,
                "exchange": 1,
                "trf_id": None,
                "trf_timestamp": None,
                "conditions": [0],
            },
        ],
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        compute_trade_imbalance_proxy(trades)
