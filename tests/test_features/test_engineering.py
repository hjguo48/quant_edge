from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.data.db.session as db_session_module
import src.features.fundamental as fundamental_module
import src.features.macro as macro_module
import src.features.pipeline as pipeline_module
import src.features.preprocessing as preprocessing_module
import src.features.technical as technical_module
from src.features.fundamental import compute_fundamental_features
from src.features.macro import compute_macro_features
from src.features.pipeline import FeaturePipeline, compute_composite_features
from src.features.preprocessing import preprocess_features, winsorize_features
from src.features.registry import FeatureRegistry
from src.features.technical import compute_technical_features
from src.labels.forward_returns import compute_forward_returns


def test_preprocess_features_forward_fills_then_ranks_and_adds_missing_flags() -> None:
    raw_features = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 1), "feature_name": "pe_ratio", "feature_value": 10.0},
            {"ticker": "BBB", "trade_date": date(2024, 1, 1), "feature_name": "pe_ratio", "feature_value": 30.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "pe_ratio", "feature_value": None},
            {"ticker": "BBB", "trade_date": date(2024, 1, 2), "feature_name": "pe_ratio", "feature_value": 40.0},
        ],
    )

    processed = preprocess_features(raw_features)

    aaa_value = processed.loc[
        (processed["ticker"] == "AAA")
        & (processed["trade_date"] == date(2024, 1, 2))
        & (processed["feature_name"] == "pe_ratio"),
        "feature_value",
    ].iloc[0]
    bbb_value = processed.loc[
        (processed["ticker"] == "BBB")
        & (processed["trade_date"] == date(2024, 1, 2))
        & (processed["feature_name"] == "pe_ratio"),
        "feature_value",
    ].iloc[0]
    aaa_missing_flag = processed.loc[
        (processed["ticker"] == "AAA")
        & (processed["trade_date"] == date(2024, 1, 2))
        & (processed["feature_name"] == "is_missing_pe_ratio"),
        "feature_value",
    ].iloc[0]
    bbb_missing_flag = processed.loc[
        (processed["ticker"] == "BBB")
        & (processed["trade_date"] == date(2024, 1, 2))
        & (processed["feature_name"] == "is_missing_pe_ratio"),
        "feature_value",
    ].iloc[0]

    assert aaa_value == pytest.approx(0.0)
    assert bbb_value == pytest.approx(1.0)
    assert aaa_missing_flag == pytest.approx(1.0)
    assert bbb_missing_flag == pytest.approx(0.0)


def test_preprocess_features_keeps_missing_flags_aligned_after_sorting() -> None:
    raw_features = pd.DataFrame(
        [
            {"ticker": "BBB", "trade_date": date(2024, 1, 2), "feature_name": "pe_ratio", "feature_value": 40.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 1), "feature_name": "pe_ratio", "feature_value": 10.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "pe_ratio", "feature_value": None},
            {"ticker": "BBB", "trade_date": date(2024, 1, 1), "feature_name": "pe_ratio", "feature_value": 30.0},
        ],
    )

    processed = preprocess_features(raw_features)

    aaa_missing_flag = processed.loc[
        (processed["ticker"] == "AAA")
        & (processed["trade_date"] == date(2024, 1, 2))
        & (processed["feature_name"] == "is_missing_pe_ratio"),
        "feature_value",
    ].iloc[0]
    bbb_missing_flag = processed.loc[
        (processed["ticker"] == "BBB")
        & (processed["trade_date"] == date(2024, 1, 2))
        & (processed["feature_name"] == "is_missing_pe_ratio"),
        "feature_value",
    ].iloc[0]

    assert aaa_missing_flag == pytest.approx(1.0)
    assert bbb_missing_flag == pytest.approx(0.0)


def test_preprocess_features_executes_required_pipeline_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    execution_order: list[str] = []
    raw_features = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 1), "feature_name": "pe_ratio", "feature_value": 10.0},
        ],
    )

    def fake_forward_fill(features_df: pd.DataFrame, max_days: int = 90) -> pd.DataFrame:
        execution_order.append("forward_fill")
        assert "_raw_missing" in features_df.columns
        assert max_days == 90
        return features_df

    def fake_winsorize(features_df: pd.DataFrame, z_threshold: float = 5.0) -> pd.DataFrame:
        execution_order.append("winsorize")
        assert "_raw_missing" in features_df.columns
        assert z_threshold == 5.0
        return features_df

    def fake_rank_normalize(features_df: pd.DataFrame, method: str = "rank") -> pd.DataFrame:
        execution_order.append("rank_normalize")
        assert "_raw_missing" in features_df.columns
        assert method == "rank"
        return features_df

    def fake_add_missing_flags(features_df: pd.DataFrame, raw_missing_mask: pd.Series) -> pd.DataFrame:
        execution_order.append("missing_flags")
        assert "_raw_missing" not in features_df.columns
        assert raw_missing_mask.tolist() == [False]
        return features_df

    monkeypatch.setattr(preprocessing_module, "forward_fill_features", fake_forward_fill)
    monkeypatch.setattr(preprocessing_module, "winsorize_features", fake_winsorize)
    monkeypatch.setattr(preprocessing_module, "rank_normalize_features", fake_rank_normalize)
    monkeypatch.setattr(preprocessing_module, "add_missing_flags", fake_add_missing_flags)

    preprocess_features(raw_features)

    assert execution_order == [
        "forward_fill",
        "winsorize",
        "rank_normalize",
        "missing_flags",
    ]


def test_winsorize_features_clips_extreme_values() -> None:
    features = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "ret_20d", "feature_value": 1.0},
            {"ticker": "BBB", "trade_date": date(2024, 1, 2), "feature_name": "ret_20d", "feature_value": 1.1},
            {"ticker": "CCC", "trade_date": date(2024, 1, 2), "feature_name": "ret_20d", "feature_value": 1.2},
            {"ticker": "DDD", "trade_date": date(2024, 1, 2), "feature_name": "ret_20d", "feature_value": 1.3},
            {"ticker": "EEE", "trade_date": date(2024, 1, 2), "feature_name": "ret_20d", "feature_value": 50.0},
        ],
    )

    clipped = winsorize_features(features, z_threshold=1.0)
    clipped_outlier = clipped.loc[clipped["ticker"] == "EEE", "feature_value"].iloc[0]

    assert clipped_outlier < 50.0


def test_compute_forward_returns_calculates_excess_returns() -> None:
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 1), "adj_close": 100.0, "close": 100.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "adj_close": 110.0, "close": 110.0},
            {"ticker": "SPY", "trade_date": date(2024, 1, 1), "adj_close": 200.0, "close": 200.0},
            {"ticker": "SPY", "trade_date": date(2024, 1, 2), "adj_close": 210.0, "close": 210.0},
        ],
    )

    labels = compute_forward_returns(prices, horizons=[1], benchmark_ticker="SPY")
    aaa_row = labels.loc[
        (labels["ticker"] == "AAA")
        & (labels["trade_date"] == date(2024, 1, 1))
        & (labels["horizon"] == 1),
    ].iloc[0]

    assert aaa_row["forward_return"] == pytest.approx(0.10)
    assert aaa_row["excess_return"] == pytest.approx(0.05)


def test_compute_fundamental_features_returns_nan_for_missing_liabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pit_frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "fiscal_period": fiscal_period,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "event_time": event_time,
            }
            for fiscal_period, event_time, metric_values in [
                (
                    "2023Q1",
                    date(2023, 3, 31),
                    {
                        "revenue": 100.0,
                        "net_income": 10.0,
                        "total_assets": 80.0,
                        "total_liabilities": 30.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 14.0,
                        "ebitda": 18.0,
                        "eps": 1.0,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q2",
                    date(2023, 6, 30),
                    {
                        "revenue": 105.0,
                        "net_income": 11.0,
                        "total_assets": 85.0,
                        "total_liabilities": 32.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 15.0,
                        "ebitda": 19.0,
                        "eps": 1.1,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q3",
                    date(2023, 9, 30),
                    {
                        "revenue": 110.0,
                        "net_income": 12.0,
                        "total_assets": 90.0,
                        "total_liabilities": 34.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 16.0,
                        "ebitda": 20.0,
                        "eps": 1.2,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q4",
                    date(2023, 12, 31),
                    {
                        "revenue": 115.0,
                        "net_income": 13.0,
                        "total_assets": 95.0,
                        "total_liabilities": None,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 17.0,
                        "ebitda": 21.0,
                        "eps": 1.3,
                        "cash": 5.0,
                    },
                ),
            ]
            for metric_name, metric_value in metric_values.items()
        ],
    )
    prices = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 15),
                "close": 50.0,
            },
        ],
    )

    monkeypatch.setattr(fundamental_module, "get_fundamentals_pit", lambda *args, **kwargs: pit_frame)

    features = compute_fundamental_features("AAA", date(2024, 1, 15), prices)
    feature_map = {
        row.feature_name: row.feature_value
        for row in features.itertuples(index=False)
    }

    assert np.isnan(feature_map["roe"])
    assert np.isnan(feature_map["debt_to_equity"])
    assert np.isnan(feature_map["ev_ebitda"])


def test_compute_fundamental_features_queries_pit_by_trade_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_as_of_dates: list[date] = []
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 15), "close": 50.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 16), "close": 51.0},
        ],
    )

    def fake_get_fundamentals_pit(*, ticker: str, as_of: date, metric_names: tuple[str, ...]) -> pd.DataFrame:
        requested_as_of_dates.append(as_of)
        assert ticker == "AAA"
        assert "eps" in metric_names
        assert "weighted_average_shares_outstanding" in metric_names
        return pd.DataFrame()

    monkeypatch.setattr(fundamental_module, "get_fundamentals_pit", fake_get_fundamentals_pit)

    compute_fundamental_features("AAA", date(2024, 1, 20), prices)

    assert requested_as_of_dates == [date(2024, 1, 15), date(2024, 1, 16)]


def test_compute_fundamental_features_uses_pit_shares_for_market_cap_dependent_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pit_frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "fiscal_period": fiscal_period,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "event_time": event_time,
            }
            for fiscal_period, event_time, metric_values in [
                (
                    "2023Q1",
                    date(2023, 3, 31),
                    {
                        "revenue": 100.0,
                        "net_income": 10.0,
                        "total_assets": 80.0,
                        "total_liabilities": 30.0,
                        "total_debt": 25.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 14.0,
                        "free_cash_flow": 5.0,
                        "ebitda": 18.0,
                        "eps": 1.0,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q2",
                    date(2023, 6, 30),
                    {
                        "revenue": 105.0,
                        "net_income": 11.0,
                        "total_assets": 85.0,
                        "total_liabilities": 32.0,
                        "total_debt": 28.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 15.0,
                        "free_cash_flow": 6.0,
                        "ebitda": 19.0,
                        "eps": 1.1,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q3",
                    date(2023, 9, 30),
                    {
                        "revenue": 110.0,
                        "net_income": 12.0,
                        "total_assets": 90.0,
                        "total_liabilities": 34.0,
                        "total_debt": 30.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 16.0,
                        "free_cash_flow": 7.0,
                        "ebitda": 20.0,
                        "eps": 1.2,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q4",
                    date(2023, 12, 31),
                    {
                        "revenue": 115.0,
                        "net_income": 13.0,
                        "total_assets": 95.0,
                        "total_liabilities": 40.0,
                        "total_debt": 35.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 17.0,
                        "free_cash_flow": 8.0,
                        "ebitda": 21.0,
                        "eps": 1.3,
                        "cash": 5.0,
                    },
                ),
            ]
            for metric_name, metric_value in metric_values.items()
        ],
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 15), "close": 50.0},
        ],
    )

    monkeypatch.setattr(fundamental_module, "get_fundamentals_pit", lambda *args, **kwargs: pit_frame)

    features = compute_fundamental_features("AAA", date(2024, 1, 15), prices)
    feature_map = {row.feature_name: row.feature_value for row in features.itertuples(index=False)}

    assert feature_map["ps_ratio"] == pytest.approx(50.0 / ((100.0 + 105.0 + 110.0 + 115.0) / 100.0))
    assert feature_map["fcf_yield"] == pytest.approx((5.0 + 6.0 + 7.0 + 8.0) / 5000.0)
    assert feature_map["debt_to_equity"] == pytest.approx(35.0 / (95.0 - 40.0))
    assert feature_map["ev_ebitda"] == pytest.approx((5000.0 + 35.0 - 5.0) / (18.0 + 19.0 + 20.0 + 21.0))


def test_compute_fundamental_features_includes_accruals_and_asset_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pit_frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "fiscal_period": fiscal_period,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "event_time": event_time,
            }
            for fiscal_period, event_time, metric_values in [
                (
                    "2023Q1",
                    date(2023, 3, 31),
                    {
                        "revenue": 100.0,
                        "net_income": 8.0,
                        "total_assets": 100.0,
                        "total_liabilities": 40.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 7.0,
                        "ebitda": 15.0,
                        "eps": 1.0,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q2",
                    date(2023, 6, 30),
                    {
                        "revenue": 105.0,
                        "net_income": 9.0,
                        "total_assets": 110.0,
                        "total_liabilities": 42.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 8.0,
                        "ebitda": 16.0,
                        "eps": 1.1,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q3",
                    date(2023, 9, 30),
                    {
                        "revenue": 110.0,
                        "net_income": 10.0,
                        "total_assets": 120.0,
                        "total_liabilities": 44.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 8.5,
                        "ebitda": 17.0,
                        "eps": 1.2,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q4",
                    date(2023, 12, 31),
                    {
                        "revenue": 120.0,
                        "net_income": 12.0,
                        "total_assets": 130.0,
                        "total_liabilities": 46.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 10.0,
                        "ebitda": 18.0,
                        "eps": 1.3,
                        "cash": 5.0,
                    },
                ),
                (
                    "2024Q1",
                    date(2024, 3, 31),
                    {
                        "revenue": 125.0,
                        "net_income": 16.0,
                        "total_assets": 140.0,
                        "total_liabilities": 50.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 12.0,
                        "ebitda": 20.0,
                        "eps": 1.4,
                        "cash": 5.0,
                    },
                ),
            ]
            for metric_name, metric_value in metric_values.items()
        ],
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 5, 15), "close": 50.0},
        ],
    )

    monkeypatch.setattr(fundamental_module, "get_fundamentals_pit", lambda *args, **kwargs: pit_frame)

    features = compute_fundamental_features("AAA", date(2024, 5, 15), prices)
    feature_map = {row.feature_name: row.feature_value for row in features.itertuples(index=False)}

    assert feature_map["accruals"] == pytest.approx((16.0 - 12.0) / 140.0)
    assert feature_map["asset_growth"] == pytest.approx((140.0 - 100.0) / 100.0)


def test_compute_fundamental_features_ignore_current_stock_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pit_frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "fiscal_period": fiscal_period,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "event_time": event_time,
            }
            for fiscal_period, event_time, metric_values in [
                (
                    "2023Q1",
                    date(2023, 3, 31),
                    {
                        "revenue": 100.0,
                        "net_income": 10.0,
                        "total_assets": 80.0,
                        "total_liabilities": 30.0,
                        "total_debt": 25.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 14.0,
                        "free_cash_flow": 5.0,
                        "ebitda": 18.0,
                        "eps": 1.0,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q2",
                    date(2023, 6, 30),
                    {
                        "revenue": 105.0,
                        "net_income": 11.0,
                        "total_assets": 85.0,
                        "total_liabilities": 32.0,
                        "total_debt": 28.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 15.0,
                        "free_cash_flow": 6.0,
                        "ebitda": 19.0,
                        "eps": 1.1,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q3",
                    date(2023, 9, 30),
                    {
                        "revenue": 110.0,
                        "net_income": 12.0,
                        "total_assets": 90.0,
                        "total_liabilities": 34.0,
                        "total_debt": 30.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 16.0,
                        "free_cash_flow": 7.0,
                        "ebitda": 20.0,
                        "eps": 1.2,
                        "cash": 5.0,
                    },
                ),
                (
                    "2023Q4",
                    date(2023, 12, 31),
                    {
                        "revenue": 115.0,
                        "net_income": 13.0,
                        "total_assets": 95.0,
                        "total_liabilities": 40.0,
                        "total_debt": 35.0,
                        "weighted_average_shares_outstanding": 100.0,
                        "operating_cash_flow": 17.0,
                        "free_cash_flow": 8.0,
                        "ebitda": 21.0,
                        "eps": 1.3,
                        "cash": 5.0,
                    },
                ),
            ]
            for metric_name, metric_value in metric_values.items()
        ],
    )
    prices = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 15), "close": 50.0},
        ],
    )

    class FakeCurrentStockSession:
        def __enter__(self) -> FakeCurrentStockSession:
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, *args, **kwargs) -> object:
            raise AssertionError("current stock metadata should not be queried for PIT feature generation")

    monkeypatch.setattr(fundamental_module, "get_fundamentals_pit", lambda *args, **kwargs: pit_frame)
    monkeypatch.setattr(db_session_module, "get_session_factory", lambda: FakeCurrentStockSession)

    first = compute_fundamental_features("AAA", date(2024, 1, 15), prices)
    second = compute_fundamental_features("AAA", date(2024, 1, 15), prices)

    pd.testing.assert_frame_equal(first, second)


def test_compute_technical_features_includes_expected_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(80)]
    prices = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "trade_date": trade_date,
                "open": base + index,
                "high": base + index + 1.0,
                "low": base + index - 1.0,
                "close": base + index + 0.5,
                "adj_close": base + index + 0.5,
                "volume": 1_000_000 + 1_000 * index,
            }
            for ticker, base in [("AAA", 100.0), ("BBB", 120.0)]
            for index, trade_date in enumerate(trade_dates)
        ],
    )

    monkeypatch.setattr(
        technical_module,
        "_load_pit_shares_outstanding",
        lambda date_pairs: pd.DataFrame(columns=["ticker", "trade_date", "pit_shares_outstanding"]),
    )

    features = compute_technical_features(prices)
    latest_ret_5d = features.loc[
        (features["ticker"] == "AAA")
        & (features["trade_date"] == trade_dates[-1])
        & (features["feature_name"] == "ret_5d"),
        "feature_value",
    ].iloc[0]

    assert "macd_histogram" in set(features["feature_name"])
    assert "residual_momentum" in set(features["feature_name"])
    assert "idio_vol" in set(features["feature_name"])
    assert latest_ret_5d == pytest.approx((179.5 - 174.5) / 174.5)


def test_compute_technical_features_turnover_rate_uses_pit_shares_by_trade_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prices = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 1),
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "adj_close": 10.5,
                "volume": 1_000.0,
            },
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 2),
                "open": 10.5,
                "high": 11.5,
                "low": 10.0,
                "close": 11.0,
                "adj_close": 11.0,
                "volume": 1_000.0,
            },
        ],
    )

    monkeypatch.setattr(
        technical_module,
        "_load_pit_shares_outstanding",
        lambda date_pairs: pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "trade_date": date(2024, 1, 1),
                    "pit_shares_outstanding": 100.0,
                },
                {
                    "ticker": "AAA",
                    "trade_date": date(2024, 1, 2),
                    "pit_shares_outstanding": 200.0,
                },
            ],
        ),
    )

    features = compute_technical_features(prices)
    turnover_day_one = features.loc[
        (features["ticker"] == "AAA")
        & (features["trade_date"] == date(2024, 1, 1))
        & (features["feature_name"] == "turnover_rate"),
        "feature_value",
    ].iloc[0]
    turnover_day_two = features.loc[
        (features["ticker"] == "AAA")
        & (features["trade_date"] == date(2024, 1, 2))
        & (features["feature_name"] == "turnover_rate"),
        "feature_value",
    ].iloc[0]

    assert turnover_day_one == pytest.approx(10.0)
    assert turnover_day_two == pytest.approx(5.0)


def test_compute_technical_features_loads_spy_market_proxy_for_residual_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trade_dates = pd.bdate_range("2023-01-02", periods=330).date
    market_returns = np.array([0.0008 + 0.0004 * np.sin(idx / 11.0) for idx in range(len(trade_dates))], dtype=float)
    residual_component = np.array([0.0015 * np.sin(idx / 7.0) for idx in range(len(trade_dates))], dtype=float)
    stock_returns = 0.0004 + 1.2 * market_returns + residual_component

    def build_price_frame(ticker: str, returns: np.ndarray, base_price: float) -> pd.DataFrame:
        close = base_price * np.cumprod(1.0 + returns)
        open_ = np.concatenate([[base_price], close[:-1]])
        return pd.DataFrame(
            {
                "ticker": ticker,
                "trade_date": trade_dates,
                "open": open_,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "adj_close": close,
                "volume": 1_000_000.0,
                "knowledge_time": [
                    pd.Timestamp(trade_date).tz_localize("UTC") + pd.Timedelta(days=1)
                    for trade_date in trade_dates
                ],
            },
        )

    stock_prices = build_price_frame("AAA", stock_returns, 100.0)
    spy_prices = build_price_frame("SPY", market_returns, 400.0)
    requested_tickers: list[str] = []

    monkeypatch.setattr(
        technical_module,
        "_load_pit_shares_outstanding",
        lambda date_pairs: pd.DataFrame(columns=["ticker", "trade_date", "pit_shares_outstanding"]),
    )

    def fake_get_prices_pit(*, tickers: list[str], start_date: date, end_date: date, as_of: datetime) -> pd.DataFrame:
        requested_tickers.extend(tickers)
        assert tickers == ["SPY"]
        assert start_date == trade_dates[0]
        assert end_date == trade_dates[-1]
        assert as_of.tzinfo is not None
        return spy_prices

    monkeypatch.setattr(technical_module, "get_prices_pit", fake_get_prices_pit)

    features = compute_technical_features(stock_prices)
    latest_date = trade_dates[-1]
    residual_momentum = features.loc[
        (features["ticker"] == "AAA")
        & (features["trade_date"] == latest_date)
        & (features["feature_name"] == "residual_momentum"),
        "feature_value",
    ].iloc[0]
    idio_vol = features.loc[
        (features["ticker"] == "AAA")
        & (features["trade_date"] == latest_date)
        & (features["feature_name"] == "idio_vol"),
        "feature_value",
    ].iloc[0]

    market_series = pd.Series(market_returns)
    stock_series = pd.Series(stock_returns)
    x_mean = market_series.rolling(252, min_periods=252).mean()
    y_mean = stock_series.rolling(252, min_periods=252).mean()
    xy_mean = (market_series * stock_series).rolling(252, min_periods=252).mean()
    xx_mean = (market_series * market_series).rolling(252, min_periods=252).mean()
    beta = (xy_mean - (x_mean * y_mean)) / (xx_mean - (x_mean * x_mean))
    alpha = y_mean - beta * x_mean
    residuals = stock_series - (alpha + beta * market_series)
    expected_residual_momentum = residuals.rolling(60, min_periods=60).sum().iloc[-1]
    expected_idio_vol = residuals.rolling(60, min_periods=60).std(ddof=0).iloc[-1]

    assert requested_tickers == ["SPY"]
    assert residual_momentum == pytest.approx(expected_residual_momentum)
    assert idio_vol == pytest.approx(expected_idio_vol)
    assert idio_vol > 0.0


def test_compute_composite_features_outputs_expected_feature_names() -> None:
    base_features = pd.DataFrame(
        [
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "ret_20d", "feature_value": 0.2},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "ret_60d", "feature_value": 0.4},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "volume_ratio_20d", "feature_value": 1.5},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "vol_20d", "feature_value": 0.1},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "vol_60d", "feature_value": 0.2},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "momentum_rank_60d", "feature_value": 0.9},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "pe_ratio", "feature_value": 10.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "pb_ratio", "feature_value": 1.5},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "roe", "feature_value": 0.2},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "roa", "feature_value": 0.1},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "fcf_yield", "feature_value": 0.05},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "debt_to_equity", "feature_value": 0.6},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "gross_margin", "feature_value": 0.4},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "operating_margin", "feature_value": 0.2},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "bb_position", "feature_value": 0.7},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "rsi_14", "feature_value": 60.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "macd_histogram", "feature_value": 0.01},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "stoch_d", "feature_value": 80.0},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "market_ret_20d", "feature_value": 0.03},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "vix_change_5d", "feature_value": 0.01},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "credit_spread", "feature_value": 1.2},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "yield_spread_10y2y", "feature_value": 0.5},
            {"ticker": "AAA", "trade_date": date(2024, 1, 2), "feature_name": "sp500_breadth", "feature_value": 0.55},
        ],
    )

    composite = compute_composite_features(base_features)

    assert len(composite["feature_name"].unique()) == 20
    assert "macro_risk_on" in set(composite["feature_name"])


def test_feature_registry_pre_registers_79_features() -> None:
    registry = FeatureRegistry()
    assert len(registry.list_features()) == 79


def test_compute_macro_features_uses_baa_minus_aaa_credit_spread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation_dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(25)]
    histories = {
        "VIXCLS": pd.Series(np.linspace(20.0, 30.0, 25), index=observation_dates),
        "DGS10": pd.Series(np.linspace(4.0, 4.5, 25), index=observation_dates),
        "DGS2": pd.Series(np.linspace(3.0, 3.2, 25), index=observation_dates),
        "BAA10Y": pd.Series(np.linspace(2.0, 4.4, 25), index=observation_dates),
        "AAA10Y": pd.Series(np.linspace(1.0, 2.2, 25), index=observation_dates),
        "FEDFUNDS": pd.Series(np.linspace(5.0, 5.0, 25), index=observation_dates),
    }

    monkeypatch.setattr(macro_module, "_load_macro_histories", lambda *args, **kwargs: histories)
    monkeypatch.setattr(macro_module, "_sp500_breadth", lambda as_of: 0.55)
    monkeypatch.setattr(macro_module, "_market_return", lambda as_of, benchmark_ticker, horizon: 0.03)

    features = compute_macro_features(date(2024, 1, 25))
    feature_map = {row.feature_name: row.feature_value for row in features.itertuples(index=False)}

    assert feature_map["credit_spread"] == pytest.approx(2.2)
    assert feature_map["credit_spread_change"] == pytest.approx(1.0)


def test_feature_pipeline_save_to_parquet_includes_batch_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fake_to_parquet(frame: pd.DataFrame, path, index: bool = False) -> None:
        captured["columns"] = list(frame.columns)
        captured["batch_values"] = frame["batch_id"].tolist()
        captured["path"] = path
        captured["index"] = index

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    pipeline = FeaturePipeline()
    features = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade_date": date(2024, 1, 2),
                "feature_name": "ret_20d",
                "feature_value": 0.2,
            },
        ],
    )

    output_path = pipeline.save_to_parquet(features, batch_id="batch-123", output_dir=str(tmp_path))

    assert output_path == tmp_path / "batch-123.parquet"
    assert captured["columns"] == ["ticker", "trade_date", "feature_name", "feature_value", "batch_id"]
    assert captured["batch_values"] == ["batch-123"]
    assert captured["index"] is False
    assert "batch_id" not in features.columns


def test_feature_pipeline_warns_when_as_of_equals_end_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warning_messages: list[str] = []

    def capture_warning(message: str, *args: object) -> None:
        warning_messages.append(message.format(*args))

    fake_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=capture_warning,
    )

    monkeypatch.setattr(pipeline_module, "logger", fake_logger)
    monkeypatch.setattr(
        pipeline_module,
        "get_prices_pit",
        lambda *args, **kwargs: pd.DataFrame(
            columns=[
                "ticker",
                "trade_date",
                "open",
                "high",
                "low",
                "close",
                "adj_close",
                "volume",
            ],
        ),
    )

    pipeline = FeaturePipeline()
    result = pipeline.run(
        tickers=["AAA"],
        start_date=date(2024, 1, 1),
        end_date=date(2024, 1, 10),
        as_of=date(2024, 1, 10),
    )

    assert result.empty
    assert any("equal to end_date" in warning for warning in warning_messages)
