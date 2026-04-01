from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import src.universe.builder as builder_module
from src.data.sources.fmp import FMPDataSource


def test_fmp_knowledge_time_per_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    source = FMPDataSource(api_key="test-key")

    def fake_get_endpoint_rows(
        endpoint: str,
        ticker: str,
        *,
        limit: int = 120,
        period: str | None = "quarter",
    ) -> list[dict[str, object]]:
        data = {
            "income-statement": [
                {
                    "date": "2024-03-31",
                    "period": "Q1",
                    "acceptedDate": "2024-05-10 16:00:00",
                    "revenue": 1000,
                    "netIncome": 100,
                    "eps": 1.5,
                },
            ],
            "cash-flow-statement": [
                {
                    "date": "2024-03-31",
                    "period": "Q1",
                    "acceptedDate": "2024-05-20 16:00:00",
                    "operatingCashFlow": 250,
                },
            ],
            "balance-sheet-statement": [],
            "key-metrics": [],
        }
        return data[endpoint]

    monkeypatch.setattr(source, "_get_endpoint_rows", fake_get_endpoint_rows)

    records = source._fetch_ticker_records("AAPL")
    revenue_record = next(record for record in records if record["metric_name"] == "revenue")
    cash_flow_record = next(
        record for record in records if record["metric_name"] == "operating_cash_flow"
    )

    assert revenue_record["event_time"] == date(2024, 3, 31)
    assert revenue_record["fiscal_period"] == "2024Q1"
    assert revenue_record["knowledge_time"] < cash_flow_record["knowledge_time"]


def test_build_universe_historical_path_applies_adv_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixedDate(date):
        @classmethod
        def today(cls) -> FixedDate:
            return cls(2026, 4, 1)

    def fail_sync(*args: object, **kwargs: object) -> None:
        raise AssertionError("Historical universe rebuild should not rewrite membership state.")

    monkeypatch.setattr(builder_module, "date", FixedDate)
    monkeypatch.setattr(
        builder_module,
        "get_historical_members",
        lambda as_of, index_name="SP500": ["AAA", "BBB"],
    )
    monkeypatch.setattr(builder_module, "_sync_membership", fail_sync)
    monkeypatch.setattr(
        builder_module,
        "get_prices_pit",
        lambda tickers, start_date, end_date, as_of: pd.DataFrame(
            [
                *[
                    {
                        "ticker": "AAA",
                        "trade_date": trade_date,
                        "close": 100.0 + index,
                        "adj_close": 100.0 + index,
                        "volume": 700_000,
                        "knowledge_time": as_of,
                        "source": "test",
                    }
                    for index, trade_date in enumerate(
                        [
                            date(2024, 1, 2),
                            date(2024, 1, 3),
                            date(2024, 1, 4),
                            date(2024, 1, 5),
                            date(2024, 1, 8),
                            date(2024, 1, 9),
                            date(2024, 1, 10),
                            date(2024, 1, 11),
                            date(2024, 1, 12),
                            date(2024, 1, 15),
                            date(2024, 1, 16),
                            date(2024, 1, 17),
                            date(2024, 1, 18),
                            date(2024, 1, 19),
                            date(2024, 1, 22),
                            date(2024, 1, 23),
                            date(2024, 1, 24),
                            date(2024, 1, 25),
                            date(2024, 1, 26),
                            date(2024, 1, 29),
                        ],
                    )
                ],
                *[
                    {
                        "ticker": "BBB",
                        "trade_date": trade_date,
                        "close": 20.0,
                        "adj_close": 20.0,
                        "volume": 100_000,
                        "knowledge_time": as_of,
                        "source": "test",
                    }
                    for trade_date in [
                        date(2024, 1, 2),
                        date(2024, 1, 3),
                        date(2024, 1, 4),
                        date(2024, 1, 5),
                        date(2024, 1, 8),
                        date(2024, 1, 9),
                        date(2024, 1, 10),
                        date(2024, 1, 11),
                        date(2024, 1, 12),
                        date(2024, 1, 15),
                        date(2024, 1, 16),
                        date(2024, 1, 17),
                        date(2024, 1, 18),
                        date(2024, 1, 19),
                        date(2024, 1, 22),
                        date(2024, 1, 23),
                        date(2024, 1, 24),
                        date(2024, 1, 25),
                        date(2024, 1, 26),
                        date(2024, 1, 29),
                    ]
                ],
            ],
        ),
    )

    result = builder_module.build_universe(
        as_of=date(2024, 1, 31),
        index_name="SP500",
        min_adv_usd=50_000_000,
    )

    assert result == ["AAA"]
