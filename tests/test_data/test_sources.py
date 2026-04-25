from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

import src.universe.builder as builder_module
from src.data.db.models import Base, UniverseMembership
from src.data.db.pit import get_universe_pit
from src.data.sources.fmp import FMPDataSource
from src.data.sources.polygon import PolygonDataSource, normalize_polygon_ticker, to_polygon_request_ticker


def test_fmp_knowledge_time_per_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    source = FMPDataSource(api_key="test-key")

    def fake_get_endpoint_rows(
        endpoint: str,
        ticker: str,
        *,
        limit: int = 120,
        page: int = 0,
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
            "dividends": [],
            "earnings": [],
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


def test_fmp_fetch_ticker_records_includes_dividend_and_consensus_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = FMPDataSource(api_key="test-key")

    def fake_get_endpoint_rows(
        endpoint: str,
        ticker: str,
        *,
        limit: int = 120,
        page: int = 0,
        period: str | None = "quarter",
    ) -> list[dict[str, object]]:
        data = {
            "income-statement": [
                {
                    "date": "2024-03-31",
                    "period": "Q1",
                    "acceptedDate": "2024-05-10 16:00:00",
                    "revenue": 1000,
                    "eps": 1.5,
                },
            ],
            "balance-sheet-statement": [
                {
                    "date": "2024-03-31",
                    "period": "Q1",
                    "acceptedDate": "2024-05-11 16:00:00",
                    "totalStockholdersEquity": 500,
                },
            ],
            "cash-flow-statement": [],
            "dividends": [
                {
                    "date": "2024-05-10",
                    "declarationDate": "2024-05-01",
                    "dividend": 0.25,
                },
            ],
            "earnings": [
                {
                    "date": "2024-05-02",
                    "epsEstimated": 1.4,
                },
                {
                    "date": "2024-09-15",
                    "epsEstimated": 1.8,
                },
            ],
        }
        return data.get(endpoint, [])

    monkeypatch.setattr(source, "_get_endpoint_rows", fake_get_endpoint_rows)

    records = source._fetch_ticker_records("AAPL")
    by_metric = {
        record["metric_name"]: record
        for record in records
        if record["metric_name"] in {"annual_dividend", "dividend_per_share", "consensus_eps", "eps_consensus"}
    }

    assert set(by_metric) == {"annual_dividend", "dividend_per_share", "consensus_eps", "eps_consensus"}
    assert by_metric["dividend_per_share"]["fiscal_period"] == "2024Q1"
    assert by_metric["dividend_per_share"]["event_time"] == date(2024, 3, 31)
    assert float(by_metric["dividend_per_share"]["metric_value"]) == 0.25
    assert by_metric["dividend_per_share"]["knowledge_time"] == datetime(2024, 5, 1, 20, tzinfo=timezone.utc)
    assert float(by_metric["annual_dividend"]["metric_value"]) == 0.25
    assert float(by_metric["consensus_eps"]["metric_value"]) == 1.4
    assert float(by_metric["eps_consensus"]["metric_value"]) == 1.4
    assert by_metric["consensus_eps"]["event_time"] == date(2024, 3, 31)
    assert by_metric["consensus_eps"]["knowledge_time"] == datetime(2024, 5, 2, 20, tzinfo=timezone.utc)


def test_polygon_ticker_mapping_helpers() -> None:
    assert normalize_polygon_ticker("BF.B") == "BF-B"
    assert normalize_polygon_ticker("BRK.B") == "BRK-B"
    assert to_polygon_request_ticker("BF-B") == "BF.B"
    assert to_polygon_request_ticker("BRK-B") == "BRK.B"
    assert normalize_polygon_ticker("AAPL") == "AAPL"
    assert to_polygon_request_ticker("AAPL") == "AAPL"


def test_polygon_fetch_historical_uses_provider_ticker_but_persists_canonical_ticker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PolygonDataSource(api_key="test-key")
    requested_tickers: list[tuple[str, bool]] = []
    persisted: dict[str, pd.DataFrame] = {}

    def fake_list_aggs(
        ticker: str,
        start_date: date,
        end_date: date,
        *,
        adjusted: bool,
    ) -> dict[date, dict[str, object]]:
        requested_tickers.append((ticker, adjusted))
        return {
            date(2024, 1, 2): {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5 if adjusted else 100.0,
                "volume": 1_000,
            },
        }

    monkeypatch.setattr(source, "_list_aggs", fake_list_aggs)

    def fake_persist_prices(frame: pd.DataFrame, *, batch_size: int = 1_000) -> int:
        persisted["frame"] = frame.copy()
        return len(frame)

    monkeypatch.setattr(source, "persist_prices", fake_persist_prices)

    frame = source.fetch_historical(["BF-B"], date(2024, 1, 2), date(2024, 1, 2))

    assert requested_tickers == [("BF.B", False), ("BF.B", True)]
    assert frame["ticker"].tolist() == ["BF-B"]
    assert persisted["frame"]["ticker"].tolist() == ["BF-B"]


def test_fmp_parse_knowledge_time_rejects_kt_le_event_time() -> None:
    """Vendor sometimes returns acceptedDate <= fiscal-period end (impossible in
    reality — 10-Q files 35-45 days after quarter end). The parser must reject
    those values so the conservative ``_fallback_knowledge_time`` is used, or
    early-period fundamental features leak future information into backtests
    (data audit P0-2).
    """
    # Vendor reports acceptedDate exactly on fiscal period end → reject
    parsed = FMPDataSource._parse_knowledge_time(
        {"acceptedDate": "2025-12-31 00:00:00"},
        event_time=date(2025, 12, 31),
    )
    assert parsed is None

    # Vendor reports acceptedDate before fiscal period end → reject
    parsed = FMPDataSource._parse_knowledge_time(
        {"acceptedDate": "2025-12-30 16:00:00"},
        event_time=date(2025, 12, 31),
    )
    assert parsed is None

    # Realistic acceptedDate (35 days after quarter end) → accept
    parsed = FMPDataSource._parse_knowledge_time(
        {"acceptedDate": "2026-02-04 14:23:55"},
        event_time=date(2025, 12, 31),
    )
    assert parsed == datetime(2026, 2, 4, 19, 23, 55, tzinfo=timezone.utc)

    # event_time omitted → no validation, accept anything parseable
    parsed = FMPDataSource._parse_knowledge_time(
        {"acceptedDate": "2025-12-31 00:00:00"},
    )
    assert parsed is not None


def test_short_interest_knowledge_time_uses_8_business_day_lag() -> None:
    """FINRA publishes short interest ~8 business days after settlement.

    Previous code used a flat 3 calendar-day offset which let backtest "see"
    short-interest 5+ days too early (data audit P1-3). Lock the new convention.
    """
    from src.data.sources.polygon_short_interest import _add_business_days

    # Friday settlement → kt 8 BD later spans weekends, lands on Wednesday
    assert _add_business_days(date(2026, 1, 16), 8) == date(2026, 1, 28)
    # Mid-month settlement on Wednesday → 8 BD later lands on Monday two weeks out
    assert _add_business_days(date(2026, 4, 15), 8) == date(2026, 4, 27)
    # Weekend in middle of n=8 still produces business day result
    result = _add_business_days(date(2025, 12, 31), 8)
    assert result.weekday() < 5


def test_polygon_historical_knowledge_time_is_next_day_market_close() -> None:
    """The historical mode must produce knowledge_time = trade_date + 1 day at 16:00 NYT.

    Without this lag-1 convention, backtest with as_of = trade_date EOD UTC would see
    today's close while computing today's signal (PIT leak). Locking the convention here
    so future refactors cannot silently regress the data audit P0-1 fix.
    """
    kt = PolygonDataSource._historical_knowledge_time(date(2026, 4, 17))
    assert kt == datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    # Friday → Saturday is intentional: kt is a wall-clock convention, not a business day.
    kt_friday = PolygonDataSource._historical_knowledge_time(date(2026, 4, 24))
    assert kt_friday == datetime(2026, 4, 25, 20, 0, tzinfo=timezone.utc)


def test_polygon_fetch_historical_default_mode_emits_lag_one_knowledge_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = PolygonDataSource(api_key="test-key")

    def fake_list_aggs(
        ticker: str,
        start_date: date,
        end_date: date,
        *,
        adjusted: bool,
    ) -> dict[date, dict[str, object]]:
        return {
            date(2026, 4, 17): {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5 if adjusted else 100.0,
                "volume": 1_000,
            },
        }

    persisted: dict[str, pd.DataFrame] = {}

    def fake_persist_prices(frame: pd.DataFrame, *, batch_size: int = 1_000) -> int:
        persisted["frame"] = frame.copy()
        return len(frame)

    monkeypatch.setattr(source, "_list_aggs", fake_list_aggs)
    monkeypatch.setattr(source, "persist_prices", fake_persist_prices)

    frame = source.fetch_historical(["AAPL"], date(2026, 4, 17), date(2026, 4, 17))

    expected_kt = datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    assert list(frame["knowledge_time"]) == [expected_kt]
    assert list(persisted["frame"]["knowledge_time"]) == [expected_kt]


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


def test_fmp_paginates_beyond_first_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    source = FMPDataSource(api_key="test-key")
    page_calls: list[int] = []

    def fake_get_endpoint_rows(
        endpoint: str,
        ticker: str,
        *,
        limit: int = 120,
        page: int = 0,
        period: str | None = "quarter",
    ) -> list[dict[str, object]]:
        page_calls.append(page)
        if endpoint != "income-statement":
            return []

        rows_by_page = {
            0: [
                {
                    "date": "2025-03-31",
                    "period": "Q1",
                    "acceptedDate": "2025-05-01 16:00:00",
                    "revenue": 1000,
                },
                {
                    "date": "2024-12-31",
                    "period": "Q4",
                    "acceptedDate": "2025-02-01 16:00:00",
                    "revenue": 900,
                },
            ],
            1: [
                {
                    "date": "2024-09-30",
                    "period": "Q3",
                    "acceptedDate": "2024-11-01 16:00:00",
                    "revenue": 800,
                },
            ],
            2: [],
        }
        return rows_by_page.get(page, [])

    monkeypatch.setattr(source, "_get_endpoint_rows", fake_get_endpoint_rows)

    rows = source._get_all_endpoint_rows("income-statement", "AAPL", page_limit=2)

    assert [row["date"] for row in rows] == ["2025-03-31", "2024-12-31", "2024-09-30"]
    assert page_calls == [0, 1]


def test_backfill_universe_membership_reconstructs_history(
    db_engine: Engine,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Base.metadata.create_all(bind=db_engine, tables=[UniverseMembership.__table__], checkfirst=True)
    session_factory = sessionmaker(bind=db_engine, expire_on_commit=False)

    with session_factory() as session:
        session.execute(
            sa.delete(UniverseMembership).where(
                UniverseMembership.ticker.in_(["AAA", "BBB", "CCC", "DDD"]),
            ),
        )
        session.commit()

    monkeypatch.setattr(
        builder_module,
        "_fetch_index_constituents",
        lambda index_name="SP500", strict_fmp=False: ["AAA", "CCC"],
    )
    monkeypatch.setattr(
        builder_module,
        "_fetch_index_change_events",
        lambda index_name="SP500", strict_fmp=False: [
            builder_module.UniverseChangeEvent(
                effective_date=date(2024, 3, 1),
                added_ticker="BBB",
                removed_ticker="DDD",
                reason="rebalance",
                source="test",
            ),
            builder_module.UniverseChangeEvent(
                effective_date=date(2025, 6, 15),
                added_ticker="CCC",
                removed_ticker="BBB",
                reason="rebalance",
                source="test",
            ),
        ],
    )

    builder_module.backfill_universe_membership(
        start_date=date(2024, 1, 1),
        end_date=date(2025, 12, 31),
        index_name="SP500",
    )

    january_members = get_universe_pit(
        as_of=datetime(2024, 1, 15, 12, tzinfo=timezone.utc),
        index_name="SP500",
    )
    april_members = get_universe_pit(
        as_of=datetime(2024, 4, 1, 12, tzinfo=timezone.utc),
        index_name="SP500",
    )
    july_members = get_universe_pit(
        as_of=datetime(2025, 7, 1, 12, tzinfo=timezone.utc),
        index_name="SP500",
    )

    assert january_members == ["AAA", "DDD"]
    assert april_members == ["AAA", "BBB"]
    assert july_members == ["AAA", "CCC"]

    with session_factory() as session:
        session.execute(
            sa.delete(UniverseMembership).where(
                UniverseMembership.ticker.in_(["AAA", "BBB", "CCC", "DDD"]),
            ),
        )
        session.commit()


def test_fetch_index_constituents_strict_fmp_raises_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder_module.settings, "FMP_API_KEY", "test-key")
    monkeypatch.setattr(
        builder_module,
        "_fetch_sp500_from_fmp",
        lambda: (_ for _ in ()).throw(RuntimeError("fmp boom")),
    )
    monkeypatch.setattr(builder_module, "_fetch_sp500_from_wikipedia", lambda: ["WIKI"])

    with pytest.raises(RuntimeError, match="fmp boom"):
        builder_module._fetch_index_constituents(index_name="SP500", strict_fmp=True)


def test_fetch_index_change_events_strict_fmp_disables_empty_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(builder_module.settings, "FMP_API_KEY", "test-key")
    monkeypatch.setattr(builder_module, "_fetch_sp500_historical_changes_from_fmp", lambda: [])
    monkeypatch.setattr(
        builder_module,
        "_fetch_sp500_historical_changes_from_wikipedia",
        lambda: (_ for _ in ()).throw(AssertionError("Wikipedia fallback should be disabled")),
    )

    assert builder_module._fetch_index_change_events(index_name="SP500", strict_fmp=True) == []


def test_backfill_universe_membership_forwards_strict_fmp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, bool] = {}

    def fake_fetch_index_constituents(index_name: str = "SP500", *, strict_fmp: bool = False) -> list[str]:
        seen["constituents"] = strict_fmp
        return ["AAA"]

    def fake_fetch_index_change_events(
        index_name: str = "SP500",
        *,
        strict_fmp: bool = False,
    ) -> list[builder_module.UniverseChangeEvent]:
        seen["events"] = strict_fmp
        return []

    monkeypatch.setattr(builder_module, "_fetch_index_constituents", fake_fetch_index_constituents)
    monkeypatch.setattr(builder_module, "_fetch_index_change_events", fake_fetch_index_change_events)
    monkeypatch.setattr(
        builder_module,
        "_reconstruct_membership_rows",
        lambda **kwargs: [
            {
                "ticker": "AAA",
                "index_name": "SP500",
                "effective_date": date(2024, 1, 1),
                "end_date": None,
                "reason": "test",
            },
        ],
    )
    monkeypatch.setattr(builder_module, "_replace_membership_rows", lambda *args, **kwargs: None)

    rows_written = builder_module.backfill_universe_membership(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31),
        index_name="SP500",
        strict_fmp=True,
    )

    assert rows_written == 1
    assert seen == {"constituents": True, "events": True}
