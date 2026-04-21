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
