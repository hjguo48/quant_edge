from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import src.features.fundamental as fundamental_module
import src.features.macro as macro_module
import src.features.pipeline as pipeline_module
import src.features.technical as technical_module
from src.features.pipeline import FeaturePipeline


def test_feature_pipeline_runs_end_to_end_with_mocked_pit_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tickers = ["AAA", "BBB"]
    full_prices = _build_price_history(tickers=tickers, start=date(2023, 1, 2), end=date(2024, 11, 15))
    fundamentals = _build_fundamentals_history(tickers)

    def fake_get_prices_pit(
        *,
        tickers: list[str],
        start_date: date,
        end_date: date,
        as_of: date | datetime,
    ) -> pd.DataFrame:
        mask = (
            full_prices["ticker"].isin([ticker.upper() for ticker in tickers])
            & (full_prices["trade_date"] >= start_date)
            & (full_prices["trade_date"] <= end_date)
        )
        return full_prices.loc[mask].copy()

    def fake_get_fundamentals_pit(
        *,
        ticker: str,
        as_of: date | datetime,
        metric_names: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts = as_of_ts.tz_convert("UTC")

        frame = fundamentals.loc[
            (fundamentals["ticker"] == ticker.upper()) & (fundamentals["knowledge_time"] <= as_of_ts)
        ].copy()
        if metric_names:
            frame = frame.loc[frame["metric_name"].isin(metric_names)].copy()
        return frame.reset_index(drop=True)

    def fake_load_pit_shares_outstanding(date_pairs: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for row in date_pairs.itertuples(index=False):
            if str(row.ticker).upper() == "AAA":
                shares = 100_000_000
            else:
                shares = 120_000_000
            rows.append(
                {
                    "ticker": str(row.ticker).upper(),
                    "trade_date": row.trade_date,
                    "pit_shares_outstanding": shares,
                },
            )
        return pd.DataFrame(rows)

    def fake_load_macro_histories(as_of: datetime, lookback_days: int) -> dict[str, pd.Series]:
        index = pd.date_range(as_of.date() - timedelta(days=lookback_days), as_of.date(), freq="B").date
        trend = np.linspace(0.0, 1.0, len(index))
        return {
            "VIXCLS": pd.Series(15.0 + 2.0 * trend, index=index),
            "DGS10": pd.Series(4.0 + 0.1 * trend, index=index),
            "DGS2": pd.Series(3.5 + 0.05 * trend, index=index),
            "BAA10Y": pd.Series(2.2 + 0.1 * trend, index=index),
            "AAA10Y": pd.Series(1.4 + 0.08 * trend, index=index),
            "FEDFUNDS": pd.Series(5.25 - 0.05 * trend, index=index),
        }

    monkeypatch.setattr(pipeline_module, "get_prices_pit", fake_get_prices_pit)
    monkeypatch.setattr(fundamental_module, "get_fundamentals_pit", fake_get_fundamentals_pit)
    monkeypatch.setattr(technical_module, "_load_pit_shares_outstanding", fake_load_pit_shares_outstanding)
    monkeypatch.setattr(macro_module, "_load_macro_histories", fake_load_macro_histories)
    monkeypatch.setattr(macro_module, "_sp500_breadth", lambda as_of: 0.55)
    monkeypatch.setattr(macro_module, "_market_return", lambda as_of, benchmark_ticker, horizon: 0.03)

    pipeline = FeaturePipeline()
    result = pipeline.run(
        tickers=tickers,
        start_date=date(2024, 10, 1),
        end_date=date(2024, 10, 31),
        as_of=datetime(2024, 11, 5, 12, tzinfo=timezone.utc),
    )

    assert not result.empty
    assert set(result.columns) == {"ticker", "trade_date", "feature_name", "feature_value", "is_filled"}
    assert sorted(result["ticker"].unique().tolist()) == tickers
    assert result["trade_date"].min() >= date(2024, 10, 1)
    assert result["trade_date"].max() <= date(2024, 10, 31)
    assert {"ret_20d", "pe_ratio", "vix", "risk_sentiment", "is_missing_pe_ratio"}.issubset(
        set(result["feature_name"].unique()),
    )
    assert result["feature_name"].nunique() >= 150
    assert result["feature_value"].isna().mean() < 0.10

    ranked_subset = result.loc[result["feature_name"].isin(["ret_20d", "pe_ratio", "vix"])].copy()
    ranked_subset["feature_value"] = pd.to_numeric(ranked_subset["feature_value"], errors="coerce")
    ranked_subset = ranked_subset.dropna(subset=["feature_value"])
    assert ranked_subset["feature_value"].between(0.0, 1.0).all()


def _build_price_history(
    *,
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="B")
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers):
        base_price = 80.0 + ticker_index * 15.0
        base_volume = 1_000_000 + ticker_index * 250_000
        for day_index, trade_date in enumerate(dates):
            close = base_price + 0.35 * day_index + 2.0 * np.sin(day_index / 8.0 + ticker_index)
            open_price = close - 0.4
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date.date(),
                    "open": round(open_price, 4),
                    "high": round(close + 0.8, 4),
                    "low": round(open_price - 0.8, 4),
                    "close": round(close, 4),
                    "adj_close": round(close * (1.0 - 0.001 * ticker_index), 4),
                    "volume": int(base_volume + 15_000 * (day_index % 10)),
                    "knowledge_time": datetime.combine(
                        trade_date.date(),
                        datetime.min.time(),
                        tzinfo=timezone.utc,
                    ) + timedelta(hours=21),
                    "source": "test",
                },
            )
    return pd.DataFrame(rows)


def _build_fundamentals_history(tickers: list[str]) -> pd.DataFrame:
    quarter_templates = [
        ("2023Q4", date(2023, 12, 31), datetime(2024, 2, 10, 21, tzinfo=timezone.utc)),
        ("2024Q1", date(2024, 3, 31), datetime(2024, 5, 10, 21, tzinfo=timezone.utc)),
        ("2024Q2", date(2024, 6, 30), datetime(2024, 8, 9, 21, tzinfo=timezone.utc)),
        ("2024Q3", date(2024, 9, 30), datetime(2024, 11, 8, 21, tzinfo=timezone.utc)),
    ]
    metric_templates = {
        "eps": [1.8, 1.9, 2.0, 2.1],
        "weighted_average_shares_outstanding": [100_000_000, 100_500_000, 101_000_000, 101_500_000],
        "book_value_per_share": [12.0, 12.4, 12.8, 13.2],
        "revenue": [4_000_000_000, 4_120_000_000, 4_260_000_000, 4_410_000_000],
        "net_income": [620_000_000, 645_000_000, 670_000_000, 700_000_000],
        "total_assets": [12_000_000_000, 12_250_000_000, 12_500_000_000, 12_800_000_000],
        "total_liabilities": [5_400_000_000, 5_450_000_000, 5_500_000_000, 5_560_000_000],
        "total_debt": [1_250_000_000, 1_240_000_000, 1_230_000_000, 1_220_000_000],
        "operating_cash_flow": [820_000_000, 850_000_000, 880_000_000, 910_000_000],
        "capital_expenditure": [160_000_000, 165_000_000, 170_000_000, 175_000_000],
        "free_cash_flow": [660_000_000, 685_000_000, 710_000_000, 735_000_000],
        "ebitda": [1_020_000_000, 1_050_000_000, 1_080_000_000, 1_115_000_000],
        "cash": [1_100_000_000, 1_120_000_000, 1_150_000_000, 1_180_000_000],
        "annual_dividend": [1.2, 1.24, 1.28, 1.32],
        "gross_profit": [1_900_000_000, 1_960_000_000, 2_020_000_000, 2_090_000_000],
        "operating_income": [780_000_000, 805_000_000, 830_000_000, 860_000_000],
        "current_assets": [3_200_000_000, 3_260_000_000, 3_320_000_000, 3_380_000_000],
        "current_liabilities": [1_900_000_000, 1_920_000_000, 1_940_000_000, 1_960_000_000],
        "consensus_eps": [1.7, 1.8, 1.9, 2.0],
    }

    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers):
        adjustment = ticker_index * 0.05
        for quarter_index, (fiscal_period, event_time, knowledge_time) in enumerate(quarter_templates):
            for metric_name, values in metric_templates.items():
                value = values[quarter_index]
                if metric_name.endswith("_shares_outstanding"):
                    value = float(value + ticker_index * 20_000_000)
                elif metric_name in {"annual_dividend", "book_value_per_share", "consensus_eps", "eps"}:
                    value = float(value + adjustment)
                else:
                    value = float(value * (1.0 + 0.03 * ticker_index))
                rows.append(
                    {
                        "ticker": ticker,
                        "fiscal_period": fiscal_period,
                        "metric_name": metric_name,
                        "metric_value": value,
                        "event_time": event_time,
                        "knowledge_time": knowledge_time,
                        "is_restated": False,
                        "source": "test",
                    },
                )
    return pd.DataFrame(rows)
