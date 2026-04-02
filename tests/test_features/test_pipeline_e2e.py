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
    full_prices = _build_price_history(
        tickers=tickers,
        start=date(2023, 1, 2),
        end=date(2024, 11, 15),
    )
    fundamentals = _build_fundamentals_history(tickers)

    def fake_get_prices_pit(
        tickers: list[str],
        start_date: date,
        end_date: date,
        as_of: date | datetime,
    ) -> pd.DataFrame:
        requested = {ticker.upper() for ticker in tickers}
        trade_dates = pd.to_datetime(full_prices["trade_date"]).dt.date
        mask = (
            full_prices["ticker"].isin(requested)
            & (trade_dates >= start_date)
            & (trade_dates <= end_date)
        )
        return full_prices.loc[mask].copy()

    def fake_get_fundamentals_pit(
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
            (fundamentals["ticker"] == ticker.upper())
            & (pd.to_datetime(fundamentals["knowledge_time"], utc=True) <= as_of_ts)
        ].copy()
        if metric_names is not None:
            frame = frame.loc[frame["metric_name"].isin(metric_names)]
        return frame.reset_index(drop=True)

    def fake_load_pit_shares(date_pairs: pd.DataFrame) -> pd.DataFrame:
        frame = date_pairs.copy()
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
        frame["pit_shares_outstanding"] = frame["ticker"].map({"AAA": 100_000_000.0, "BBB": 120_000_000.0})
        return frame

    def fake_load_macro_histories(as_of: datetime, lookback_days: int) -> dict[str, pd.Series]:
        start = as_of.date() - timedelta(days=lookback_days + 30)
        dates = pd.bdate_range(start, as_of.date()).date
        base = np.linspace(0.0, 1.0, len(dates))
        return {
            "VIXCLS": pd.Series(18.0 + 2.0 * base, index=dates),
            "DGS10": pd.Series(4.0 + 0.2 * base, index=dates),
            "DGS2": pd.Series(3.5 + 0.1 * base, index=dates),
            "BAA10Y": pd.Series(5.5 + 0.1 * base, index=dates),
            "AAA10Y": pd.Series(4.8 + 0.1 * base, index=dates),
            "FEDFUNDS": pd.Series(5.25 + 0.0 * base, index=dates),
        }

    monkeypatch.setattr(pipeline_module, "get_prices_pit", fake_get_prices_pit)
    monkeypatch.setattr(fundamental_module, "get_fundamentals_pit", fake_get_fundamentals_pit)
    monkeypatch.setattr(technical_module, "_load_pit_shares_outstanding", fake_load_pit_shares)
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
    assert set(result["ticker"]) == set(tickers)
    assert result["trade_date"].min() >= date(2024, 10, 1)
    assert result["trade_date"].max() <= date(2024, 10, 31)
    assert {"ret_20d", "pe_ratio", "vix", "risk_sentiment", "is_missing_pe_ratio"}.issubset(
        set(result["feature_name"]),
    )
    assert result["feature_name"].nunique() >= 150
    assert result["feature_value"].isna().mean() < 0.10

    ranked_slice = result.loc[result["feature_name"].isin(["ret_20d", "pe_ratio", "vix"]), "feature_value"]
    assert ranked_slice.dropna().between(0.0, 1.0).all()


def _build_price_history(*, tickers: list[str], start: date, end: date) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers, start=1):
        for day_index, trade_date in enumerate(dates, start=1):
            close = 90.0 + ticker_index * 10.0 + day_index * (0.25 + ticker_index * 0.02)
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date.date(),
                    "open": close * 0.995,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000 + ticker_index * 25_000 + day_index * 100,
                    "knowledge_time": datetime.combine(trade_date.date(), datetime.max.time(), tzinfo=timezone.utc),
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
    rows: list[dict[str, object]] = []
    for ticker_index, ticker in enumerate(tickers, start=1):
        for quarter_index, (fiscal_period, event_time, knowledge_time) in enumerate(quarter_templates, start=1):
            base_revenue = 400.0 + ticker_index * 20.0 + quarter_index * 15.0
            shares = 100_000_000 + ticker_index * 20_000_000
            metrics = {
                "eps": 1.0 + 0.05 * quarter_index,
                "weighted_average_shares_outstanding": float(shares),
                "book_value_per_share": 20.0 + ticker_index,
                "revenue": base_revenue,
                "net_income": 40.0 + ticker_index * 3.0 + quarter_index,
                "total_assets": 900.0 + ticker_index * 50.0,
                "total_liabilities": 450.0 + ticker_index * 25.0,
                "total_debt": 220.0 + ticker_index * 10.0,
                "operating_cash_flow": 55.0 + quarter_index,
                "capital_expenditure": -12.0,
                "free_cash_flow": 43.0 + quarter_index,
                "ebitda": 70.0 + quarter_index,
                "cash": 80.0 + ticker_index * 5.0,
                "annual_dividend": 1.2 + 0.1 * ticker_index,
                "gross_profit": 180.0 + quarter_index,
                "operating_income": 60.0 + quarter_index,
                "current_assets": 250.0 + quarter_index,
                "current_liabilities": 150.0 + quarter_index,
                "consensus_eps": 0.95 + 0.05 * quarter_index,
            }
            for metric_name, metric_value in metrics.items():
                rows.append(
                    {
                        "ticker": ticker,
                        "fiscal_period": fiscal_period,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "event_time": event_time,
                        "knowledge_time": knowledge_time,
                        "is_restated": False,
                        "source": "test",
                    },
                )
    return pd.DataFrame(rows)
