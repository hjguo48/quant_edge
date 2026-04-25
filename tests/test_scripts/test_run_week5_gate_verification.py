from __future__ import annotations

import json
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import scripts.run_week5_gate_verification as gate_module
from src.data.sources.fmp_earnings_calendar import EarningsCalendar
from src.data.sources.fmp_grades import GradesEvent
from src.data.sources.fmp_price_target import PriceTargetEvent
from src.data.sources.fmp_ratings import RatingEvent
from src.data.finra_short_sale import ShortSaleVolume

EASTERN = ZoneInfo("America/New_York")
TEST_TICKERS = tuple(f"TST{i:02d}" for i in range(10))


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_tables(db_engine) -> None:
    for table in (
        ShortSaleVolume.__table__,
        GradesEvent.__table__,
        RatingEvent.__table__,
        PriceTargetEvent.__table__,
        EarningsCalendar.__table__,
    ):
        table.create(bind=db_engine, checkfirst=True)


def _clear_rows(session_factory: sessionmaker) -> None:
    with session_factory() as session:
        session.execute(sa.delete(ShortSaleVolume).where(ShortSaleVolume.ticker.like("TST%")))
        session.execute(sa.delete(GradesEvent).where(GradesEvent.ticker.like("TST%")))
        session.execute(sa.delete(RatingEvent).where(RatingEvent.ticker.like("TST%")))
        session.execute(sa.delete(PriceTargetEvent).where(PriceTargetEvent.ticker.like("TST%")))
        session.execute(sa.delete(EarningsCalendar).where(EarningsCalendar.ticker.like("TST%")))
        session.commit()


@pytest.fixture
def seeded_sf(db_engine) -> sessionmaker:
    _ensure_tables(db_engine)
    sf = _session_factory(db_engine)
    _clear_rows(sf)
    yield sf
    _clear_rows(sf)


def _kt_eod(day: date) -> datetime:
    return datetime.combine(day, time(23, 59), tzinfo=EASTERN).astimezone(timezone.utc)


def _kt_18(day: date) -> datetime:
    return datetime.combine(day, time(18, 0), tzinfo=EASTERN).astimezone(timezone.utc)


def _seed(session_factory: sessionmaker, rows: list[object]) -> None:
    with session_factory() as session:
        session.add_all(rows)
        session.commit()


def test_compute_coverage_gate_passes_with_thresholded_seed(seeded_sf: sessionmaker) -> None:
    trade_day = date(2026, 1, 15)
    rows: list[object] = []
    for ticker in TEST_TICKERS[:9]:
        rows.append(
            ShortSaleVolume(
                ticker=ticker,
                trade_date=trade_day,
                knowledge_time=_kt_18(trade_day),
                market="CNMS",
                short_volume=100,
                short_exempt_volume=0,
                total_volume=1000,
                file_etag="etag",
            ),
        )
    for ticker in TEST_TICKERS[:7]:
        rows.append(
            GradesEvent(
                ticker=ticker,
                event_date=trade_day,
                knowledge_time=_kt_eod(trade_day),
                analyst_firm=f"{ticker}-grades",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
        )
        rows.append(
            RatingEvent(
                ticker=ticker,
                event_date=trade_day,
                knowledge_time=_kt_eod(trade_day),
                rating_score=4,
                rating_recommendation="Buy",
                dcf_rating=Decimal("4.0"),
                pe_rating=Decimal("4.0"),
                roe_rating=Decimal("4.0"),
            ),
        )
        rows.append(
            PriceTargetEvent(
                ticker=ticker,
                event_date=trade_day,
                knowledge_time=_kt_eod(trade_day),
                analyst_firm=f"{ticker}-targets",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        )
    for ticker in TEST_TICKERS:
        rows.append(
            EarningsCalendar(
                ticker=ticker,
                announce_date=trade_day,
                knowledge_time=_kt_eod(trade_day),
                timing="AMC",
                fiscal_period_end=date(2025, 12, 31),
                eps_estimate=Decimal("1.0"),
                eps_actual=Decimal("1.1"),
                revenue_estimate=1000,
                revenue_actual=1100,
            ),
        )
    _seed(seeded_sf, rows)

    result = gate_module.compute_coverage_gate(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        session_factory=seeded_sf,
        universe_fetcher=lambda as_of, index_name="SP500": list(TEST_TICKERS),
    )

    assert result["status"] == "PASS"
    assert result["sources"]["finra_short_sale_volume"]["min_coverage"] == pytest.approx(0.9)
    assert result["sources"]["fmp_grades"]["min_coverage"] == pytest.approx(0.7)
    assert result["sources"]["earnings_calendar"]["min_coverage"] == pytest.approx(1.0)


def test_compute_coverage_gate_fails_below_threshold(seeded_sf: sessionmaker) -> None:
    trade_day = date(2026, 1, 15)
    rows = [
        ShortSaleVolume(
            ticker=ticker,
            trade_date=trade_day,
            knowledge_time=_kt_18(trade_day),
            market="CNMS",
            short_volume=100,
            short_exempt_volume=0,
            total_volume=1000,
            file_etag="etag",
        )
        for ticker in TEST_TICKERS[:8]
    ]
    _seed(seeded_sf, rows)

    result = gate_module.compute_coverage_gate(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        session_factory=seeded_sf,
        universe_fetcher=lambda as_of, index_name="SP500": list(TEST_TICKERS),
    )

    assert result["status"] == "FAIL"
    assert result["sources"]["finra_short_sale_volume"]["status"] == "FAIL"


def test_compute_missing_rate_gate_passes(monkeypatch, seeded_sf: sessionmaker) -> None:
    monkeypatch.setattr(gate_module, "_source_ticker_count_for_feature", lambda *args, **kwargs: 5)

    class FakeRegistry:
        def get_feature(self, name: str):
            if name == "short_sale_ratio_1d":
                return SimpleNamespace(compute_fn=lambda ticker, as_of, session_factory=None: 1.0)
            return SimpleNamespace(
                compute_fn=lambda ticker, as_of, session_factory=None: None
                if (ticker == TEST_TICKERS[0] and as_of.day % 2 == 0)
                else 2.0,
            )

    result = gate_module.compute_missing_rate_gate(
        feature_names=("short_sale_ratio_1d", "net_grade_change_20d"),
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 8),
        universe_asof=date(2026, 1, 8),
        session_factory=seeded_sf,
        registry_builder=FakeRegistry,
        universe_fetcher=lambda as_of, index_name="SP500": list(TEST_TICKERS[:2]),
    )

    assert result["status"] == "PASS"
    rates = {item["feature"]: item["missing_rate"] for item in result["per_feature"]}
    assert rates["short_sale_ratio_1d"] == pytest.approx(0.0)
    assert rates["net_grade_change_20d"] < 0.40


def test_compute_missing_rate_gate_fails_when_too_sparse(monkeypatch, seeded_sf: sessionmaker) -> None:
    monkeypatch.setattr(gate_module, "_source_ticker_count_for_feature", lambda *args, **kwargs: 5)

    class FakeRegistry:
        def get_feature(self, name: str):
            return SimpleNamespace(
                compute_fn=lambda ticker, as_of, session_factory=None: None if ticker == TEST_TICKERS[0] else 1.0,
            )

    result = gate_module.compute_missing_rate_gate(
        feature_names=("short_sale_ratio_1d",),
        start_date=date(2026, 1, 5),
        end_date=date(2026, 1, 8),
        universe_asof=date(2026, 1, 8),
        session_factory=seeded_sf,
        registry_builder=FakeRegistry,
        universe_fetcher=lambda as_of, index_name="SP500": list(TEST_TICKERS[:2]),
    )

    assert result["status"] == "FAIL"
    assert result["per_feature"][0]["missing_rate"] >= 0.40


def test_compute_lag_rule_gate_passes(seeded_sf: sessionmaker) -> None:
    event_day = date(2026, 1, 15)
    _seed(
        seeded_sf,
        [
            ShortSaleVolume(
                ticker=TEST_TICKERS[0],
                trade_date=event_day,
                knowledge_time=_kt_18(event_day),
                market="CNMS",
                short_volume=100,
                short_exempt_volume=0,
                total_volume=1000,
                file_etag="etag",
            ),
            GradesEvent(
                ticker=TEST_TICKERS[0],
                event_date=event_day,
                knowledge_time=_kt_eod(event_day),
                analyst_firm="FirmA",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
        ],
    )

    result = gate_module.compute_lag_rule_gate(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        session_factory=seeded_sf,
    )

    assert result["status"] == "PASS"
    assert result["violations"] == []


def test_compute_lag_rule_gate_detects_offender(seeded_sf: sessionmaker) -> None:
    event_day = date(2026, 1, 15)
    _seed(
        seeded_sf,
        [
            ShortSaleVolume(
                ticker=TEST_TICKERS[0],
                trade_date=event_day,
                knowledge_time=datetime.combine(event_day, time(17, 0), tzinfo=EASTERN).astimezone(timezone.utc),
                market="CNMS",
                short_volume=100,
                short_exempt_volume=0,
                total_volume=1000,
                file_etag="etag",
            ),
        ],
    )

    result = gate_module.compute_lag_rule_gate(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        session_factory=seeded_sf,
    )

    assert result["status"] == "FAIL"
    assert result["violations"][0]["source"] == "short_sale_volume_daily"


def test_compute_source_integrity_gate_passes(seeded_sf: sessionmaker) -> None:
    event_day = date(2099, 1, 15)
    _seed(
        seeded_sf,
        [
            ShortSaleVolume(
                ticker=TEST_TICKERS[0],
                trade_date=event_day,
                knowledge_time=_kt_18(event_day),
                market="CNMS",
                short_volume=100,
                short_exempt_volume=0,
                total_volume=1000,
                file_etag="etag",
            ),
            GradesEvent(
                ticker=TEST_TICKERS[0],
                event_date=event_day,
                knowledge_time=_kt_eod(event_day),
                analyst_firm="FirmA",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
            RatingEvent(
                ticker=TEST_TICKERS[0],
                event_date=event_day,
                knowledge_time=_kt_eod(event_day),
                rating_score=4,
                rating_recommendation="Buy",
                dcf_rating=Decimal("4.0"),
                pe_rating=Decimal("4.0"),
                roe_rating=Decimal("4.0"),
            ),
            PriceTargetEvent(
                ticker=TEST_TICKERS[0],
                event_date=event_day,
                knowledge_time=_kt_eod(event_day),
                analyst_firm="FirmA",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            EarningsCalendar(
                ticker=TEST_TICKERS[0],
                announce_date=event_day,
                knowledge_time=_kt_eod(event_day),
                timing="AMC",
                fiscal_period_end=date(2025, 12, 31),
                eps_estimate=Decimal("1.0"),
                eps_actual=Decimal("1.1"),
                revenue_estimate=1000,
                revenue_actual=1100,
            ),
        ],
    )

    result = gate_module.compute_source_integrity_gate(
        start_date=date(2099, 1, 1),
        end_date=date(2099, 1, 31),
        session_factory=seeded_sf,
        today=date(2099, 2, 1),
    )

    assert result["status"] == "PASS"
    assert result["http_error_rate"] == "not_implemented"


def test_compute_source_integrity_gate_fails_on_sparse_fields(seeded_sf: sessionmaker) -> None:
    event_day = date(2099, 1, 15)
    _seed(
        seeded_sf,
        [
            RatingEvent(
                ticker=TEST_TICKERS[0],
                event_date=event_day,
                knowledge_time=_kt_eod(event_day),
                rating_score=4,
                rating_recommendation=None,
                dcf_rating=Decimal("4.0"),
                pe_rating=Decimal("4.0"),
                roe_rating=Decimal("4.0"),
            ),
            EarningsCalendar(
                ticker=TEST_TICKERS[0],
                announce_date=event_day,
                knowledge_time=_kt_eod(event_day),
                timing="AMC",
                fiscal_period_end=date(2025, 12, 31),
                eps_estimate=Decimal("1.0"),
                eps_actual=None,
                revenue_estimate=1000,
                revenue_actual=None,
            ),
        ],
    )

    result = gate_module.compute_source_integrity_gate(
        start_date=date(2099, 1, 1),
        end_date=date(2099, 1, 31),
        session_factory=seeded_sf,
        today=date(2099, 2, 1),
    )

    assert result["status"] == "FAIL"
    assert result["fmp_field_populate_rate"]["ratings"] == pytest.approx(0.0)


def test_generate_gate_summary_has_expected_schema(tmp_path: Path, monkeypatch, seeded_sf: sessionmaker) -> None:
    config_path = tmp_path / "lineage.yaml"
    config_path.write_text("version: 1\n", encoding="utf-8")

    monkeypatch.setattr(gate_module, "compute_coverage_gate", lambda **kwargs: {"status": "PASS", "sources": {}})
    monkeypatch.setattr(gate_module, "compute_missing_rate_gate", lambda **kwargs: {"status": "PASS", "per_feature": []})
    monkeypatch.setattr(gate_module, "compute_lag_rule_gate", lambda **kwargs: {"status": "PASS", "violations": []})
    monkeypatch.setattr(
        gate_module,
        "compute_source_integrity_gate",
        lambda **kwargs: {"status": "PARTIAL", "http_error_rate": "not_implemented"},
    )

    summary = gate_module.generate_gate_summary(
        feature_names=("short_sale_ratio_1d",),
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        universe_asof=date(2026, 1, 31),
        features_config_path=config_path,
        session_factory=seeded_sf,
        universe_fetcher=lambda as_of, index_name="SP500": list(TEST_TICKERS[:2]),
    )

    assert summary["overall"] == "PASS"
    assert summary["features_covered"] == ["short_sale_ratio_1d"]
    assert set(summary["gates"]) == {"coverage", "missing_rate", "lag_rule", "source_integrity"}


def test_main_dry_run_empty_db_fails_cleanly(tmp_path: Path, capsys, seeded_sf: sessionmaker, monkeypatch) -> None:
    config_path = tmp_path / "lineage.yaml"
    config_path.write_text("version: 1\n", encoding="utf-8")
    monkeypatch.setattr(gate_module, "get_session_factory", lambda: seeded_sf)
    monkeypatch.setattr(gate_module, "get_historical_members", lambda as_of, index_name="SP500": [])

    exit_code = gate_module.main(
        [
            "--features-config",
            str(config_path),
            "--start-date",
            "2026-01-01",
            "--end-date",
            "2026-01-31",
            "--dry-run",
            "--features",
            "short_sale_ratio_1d",
        ],
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["overall"] == "FAIL"
    assert payload["gates"]["coverage"]["status"] == "FAIL"
