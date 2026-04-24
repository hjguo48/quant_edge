from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Iterable
from zoneinfo import ZoneInfo

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from src.data.db.models import StockPrice
from src.data.sources.fmp_grades import GradesEvent
from src.data.sources.fmp_price_target import PriceTargetEvent
from src.data.sources.fmp_ratings import RatingEvent
from src.features.analyst_proxy import (
    compute_consensus_upside,
    compute_coverage_change_proxy,
    compute_downgrade_count,
    compute_financial_health_trend,
    compute_net_grade_change,
    compute_target_dispersion_proxy,
    compute_target_price_drift,
    compute_upgrade_count,
)

EASTERN = ZoneInfo("America/New_York")
TEST_TICKER = "TSTA"


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_tables(db_engine) -> None:
    for table in (
        StockPrice.__table__,
        GradesEvent.__table__,
        RatingEvent.__table__,
        PriceTargetEvent.__table__,
    ):
        table.create(bind=db_engine, checkfirst=True)


def _clear_test_rows(session_factory: sessionmaker) -> None:
    with session_factory() as session:
        session.execute(sa.delete(PriceTargetEvent).where(PriceTargetEvent.ticker.like("TST%")))
        session.execute(sa.delete(RatingEvent).where(RatingEvent.ticker.like("TST%")))
        session.execute(sa.delete(GradesEvent).where(GradesEvent.ticker.like("TST%")))
        session.execute(sa.delete(StockPrice).where(StockPrice.ticker.like("TST%")))
        session.commit()


def _kt(day: date, hour: int = 23, minute: int = 59, second: int = 0) -> datetime:
    return datetime.combine(day, time(hour=hour, minute=minute, second=second), tzinfo=EASTERN).astimezone(timezone.utc)


def _future_kt(day: date) -> datetime:
    return _kt(day, 23, 59, 59) + timedelta(seconds=2)


def _seed(session_factory: sessionmaker, entities: Iterable[object]) -> None:
    with session_factory() as session:
        session.add_all(list(entities))
        session.commit()


@pytest.fixture
def analyst_sf(db_engine) -> sessionmaker:
    _ensure_tables(db_engine)
    sf = _session_factory(db_engine)
    _clear_test_rows(sf)
    return sf


@pytest.mark.parametrize(
    ("horizon_days", "expected"),
    [
        (5, 1),
        (20, -1),
        (60, 0),
    ],
)
def test_compute_net_grade_change_sums_events_by_horizon(analyst_sf: sessionmaker, horizon_days: int, expected: int) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 22),
                knowledge_time=_kt(date(2026, 4, 22)),
                analyst_firm="FirmA",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 10),
                knowledge_time=_kt(date(2026, 4, 10)),
                analyst_firm="FirmB",
                prior_grade="Buy",
                new_grade="Sell",
                action="downgrade",
                grade_score_change=-2,
            ),
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 3, 10),
                knowledge_time=_kt(date(2026, 3, 10)),
                analyst_firm="FirmC",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
        ],
    )

    result = compute_net_grade_change(TEST_TICKER, as_of, horizon_days, session_factory=analyst_sf)

    assert result == expected


def test_compute_net_grade_change_returns_zero_without_events(analyst_sf: sessionmaker) -> None:
    result = compute_net_grade_change(TEST_TICKER, date(2026, 4, 23), 20, session_factory=analyst_sf)
    assert result == 0


def test_compute_net_grade_change_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 22),
                knowledge_time=_future_kt(as_of),
                analyst_firm="FirmA",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
        ],
    )

    result = compute_net_grade_change(TEST_TICKER, as_of, 20, session_factory=analyst_sf)

    assert result == 0


def test_compute_upgrade_count_counts_positive_events(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                analyst_firm="FirmA",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 18),
                knowledge_time=_kt(date(2026, 4, 18)),
                analyst_firm="FirmB",
                prior_grade="Underperform",
                new_grade="Outperform",
                action="upgrade",
                grade_score_change=2,
            ),
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 16),
                knowledge_time=_kt(date(2026, 4, 16)),
                analyst_firm="FirmC",
                prior_grade="Buy",
                new_grade="Hold",
                action="downgrade",
                grade_score_change=-2,
            ),
        ],
    )

    result = compute_upgrade_count(TEST_TICKER, as_of, 20, session_factory=analyst_sf)

    assert result == 2


def test_compute_upgrade_count_returns_zero_without_events(analyst_sf: sessionmaker) -> None:
    result = compute_upgrade_count(TEST_TICKER, date(2026, 4, 23), 20, session_factory=analyst_sf)
    assert result == 0


def test_compute_upgrade_count_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 22),
                knowledge_time=_future_kt(as_of),
                analyst_firm="FirmA",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
        ],
    )

    result = compute_upgrade_count(TEST_TICKER, as_of, 20, session_factory=analyst_sf)

    assert result == 0


def test_compute_downgrade_count_counts_negative_events(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                analyst_firm="FirmA",
                prior_grade="Buy",
                new_grade="Hold",
                action="downgrade",
                grade_score_change=-2,
            ),
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 18),
                knowledge_time=_kt(date(2026, 4, 18)),
                analyst_firm="FirmB",
                prior_grade="Outperform",
                new_grade="Underperform",
                action="downgrade",
                grade_score_change=-2,
            ),
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 16),
                knowledge_time=_kt(date(2026, 4, 16)),
                analyst_firm="FirmC",
                prior_grade="Hold",
                new_grade="Buy",
                action="upgrade",
                grade_score_change=1,
            ),
        ],
    )

    result = compute_downgrade_count(TEST_TICKER, as_of, 20, session_factory=analyst_sf)

    assert result == 2


def test_compute_downgrade_count_returns_zero_without_events(analyst_sf: sessionmaker) -> None:
    result = compute_downgrade_count(TEST_TICKER, date(2026, 4, 23), 20, session_factory=analyst_sf)
    assert result == 0


def test_compute_downgrade_count_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            GradesEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 22),
                knowledge_time=_future_kt(as_of),
                analyst_firm="FirmA",
                prior_grade="Buy",
                new_grade="Sell",
                action="downgrade",
                grade_score_change=-4,
            ),
        ],
    )

    result = compute_downgrade_count(TEST_TICKER, as_of, 20, session_factory=analyst_sf)

    assert result == 0


def test_compute_consensus_upside_uses_latest_consensus_and_latest_close(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 1),
                knowledge_time=_kt(date(2026, 4, 1)),
                analyst_firm=None,
                target_price=Decimal("110.0"),
                prior_target=None,
                price_when_published=None,
                is_consensus=True,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                analyst_firm=None,
                target_price=Decimal("120.0"),
                prior_target=None,
                price_when_published=None,
                is_consensus=True,
            ),
            StockPrice(
                ticker=TEST_TICKER,
                trade_date=date(2026, 4, 22),
                open=Decimal("99.0"),
                high=Decimal("101.0"),
                low=Decimal("98.0"),
                close=Decimal("100.0"),
                adj_close=Decimal("100.0"),
                volume=1_000_000,
                knowledge_time=_kt(date(2026, 4, 22)),
                source="test",
            ),
            StockPrice(
                ticker=TEST_TICKER,
                trade_date=date(2026, 4, 23),
                open=Decimal("79.0"),
                high=Decimal("81.0"),
                low=Decimal("78.0"),
                close=Decimal("80.0"),
                adj_close=Decimal("80.0"),
                volume=1_000_000,
                knowledge_time=_future_kt(as_of),
                source="test",
            ),
        ],
    )

    result = compute_consensus_upside(TEST_TICKER, as_of, session_factory=analyst_sf)

    assert result == pytest.approx(0.2)


def test_compute_consensus_upside_returns_none_without_consensus(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            StockPrice(
                ticker=TEST_TICKER,
                trade_date=date(2026, 4, 22),
                open=Decimal("99.0"),
                high=Decimal("101.0"),
                low=Decimal("98.0"),
                close=Decimal("100.0"),
                adj_close=Decimal("100.0"),
                volume=1_000_000,
                knowledge_time=_kt(date(2026, 4, 22)),
                source="test",
            ),
        ],
    )

    result = compute_consensus_upside(TEST_TICKER, as_of, session_factory=analyst_sf)

    assert result is None


def test_compute_consensus_upside_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_future_kt(as_of),
                analyst_firm=None,
                target_price=Decimal("120.0"),
                prior_target=None,
                price_when_published=None,
                is_consensus=True,
            ),
            StockPrice(
                ticker=TEST_TICKER,
                trade_date=date(2026, 4, 22),
                open=Decimal("99.0"),
                high=Decimal("101.0"),
                low=Decimal("98.0"),
                close=Decimal("100.0"),
                adj_close=Decimal("100.0"),
                volume=1_000_000,
                knowledge_time=_kt(date(2026, 4, 22)),
                source="test",
            ),
        ],
    )

    result = compute_consensus_upside(TEST_TICKER, as_of, session_factory=analyst_sf)

    assert result is None


def test_compute_target_price_drift_returns_normalized_slope(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    rows = []
    for offset, target in zip(range(4, -1, -1), [100, 102, 104, 106, 108], strict=False):
        event_day = as_of - timedelta(days=offset)
        rows.append(
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=event_day,
                knowledge_time=_kt(event_day),
                analyst_firm=f"Firm{offset}",
                target_price=Decimal(str(target)),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        )
    rows.append(
        StockPrice(
            ticker=TEST_TICKER,
            trade_date=as_of,
            open=Decimal("99.0"),
            high=Decimal("101.0"),
            low=Decimal("98.0"),
            close=Decimal("100.0"),
            adj_close=Decimal("100.0"),
            volume=1_000_000,
            knowledge_time=_kt(as_of),
            source="test",
        ),
    )
    _seed(analyst_sf, rows)

    result = compute_target_price_drift(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result == pytest.approx(1.2, rel=1e-6)


def test_compute_target_price_drift_returns_none_when_insufficient_event_days(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    rows = []
    for offset, target in zip(range(3, -1, -1), [100, 102, 104, 106], strict=False):
        event_day = as_of - timedelta(days=offset)
        rows.append(
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=event_day,
                knowledge_time=_kt(event_day),
                analyst_firm=f"Firm{offset}",
                target_price=Decimal(str(target)),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        )
    rows.append(
        StockPrice(
            ticker=TEST_TICKER,
            trade_date=as_of,
            open=Decimal("99.0"),
            high=Decimal("101.0"),
            low=Decimal("98.0"),
            close=Decimal("100.0"),
            adj_close=Decimal("100.0"),
            volume=1_000_000,
            knowledge_time=_kt(as_of),
            source="test",
        ),
    )
    _seed(analyst_sf, rows)

    result = compute_target_price_drift(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result is None


def test_compute_target_price_drift_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    rows = []
    for offset, target in zip(range(4, 0, -1), [100, 102, 104, 106], strict=False):
        event_day = as_of - timedelta(days=offset)
        rows.append(
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=event_day,
                knowledge_time=_kt(event_day),
                analyst_firm=f"Firm{offset}",
                target_price=Decimal(str(target)),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        )
    rows.extend(
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=as_of,
                knowledge_time=_future_kt(as_of),
                analyst_firm="FirmFuture",
                target_price=Decimal("108"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            StockPrice(
                ticker=TEST_TICKER,
                trade_date=as_of,
                open=Decimal("99.0"),
                high=Decimal("101.0"),
                low=Decimal("98.0"),
                close=Decimal("100.0"),
                adj_close=Decimal("100.0"),
                volume=1_000_000,
                knowledge_time=_kt(as_of),
                source="test",
            ),
        ],
    )
    _seed(analyst_sf, rows)

    result = compute_target_price_drift(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result is None


def test_compute_target_dispersion_proxy_uses_latest_per_firm(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 1),
                knowledge_time=_kt(date(2026, 4, 1)),
                analyst_firm="FirmA",
                target_price=Decimal("90"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                analyst_firm="FirmA",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 18),
                knowledge_time=_kt(date(2026, 4, 18)),
                analyst_firm="FirmB",
                target_price=Decimal("110"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 19),
                knowledge_time=_kt(date(2026, 4, 19)),
                analyst_firm="FirmC",
                target_price=Decimal("120"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        ],
    )

    result = compute_target_dispersion_proxy(TEST_TICKER, as_of, session_factory=analyst_sf)

    assert result == pytest.approx(0.07422696, rel=1e-6)


def test_compute_target_dispersion_proxy_returns_none_when_insufficient_firms(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 18),
                knowledge_time=_kt(date(2026, 4, 18)),
                analyst_firm="FirmA",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 19),
                knowledge_time=_kt(date(2026, 4, 19)),
                analyst_firm="FirmB",
                target_price=Decimal("110"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        ],
    )

    result = compute_target_dispersion_proxy(TEST_TICKER, as_of, session_factory=analyst_sf)

    assert result is None


def test_compute_target_dispersion_proxy_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                analyst_firm="FirmA",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 18),
                knowledge_time=_kt(date(2026, 4, 18)),
                analyst_firm="FirmB",
                target_price=Decimal("110"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 19),
                knowledge_time=_kt(date(2026, 4, 19)),
                analyst_firm="FirmC",
                target_price=Decimal("120"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 23),
                knowledge_time=_future_kt(as_of),
                analyst_firm="FirmA",
                target_price=Decimal("300"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        ],
    )

    result = compute_target_dispersion_proxy(TEST_TICKER, as_of, session_factory=analyst_sf)

    assert result == pytest.approx(0.07422696, rel=1e-6)


def test_compute_coverage_change_proxy_returns_delta(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 1, 10),
                knowledge_time=_kt(date(2026, 1, 10)),
                analyst_firm="FirmA",
                target_price=Decimal("90"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 1, 12),
                knowledge_time=_kt(date(2026, 1, 12)),
                analyst_firm="FirmB",
                target_price=Decimal("95"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 3, 10),
                knowledge_time=_kt(date(2026, 3, 10)),
                analyst_firm="FirmA",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 3, 15),
                knowledge_time=_kt(date(2026, 3, 15)),
                analyst_firm="FirmC",
                target_price=Decimal("105"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 1),
                knowledge_time=_kt(date(2026, 4, 1)),
                analyst_firm="FirmB",
                target_price=Decimal("110"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        ],
    )

    result = compute_coverage_change_proxy(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result == 1


def test_compute_coverage_change_proxy_returns_none_without_prior_history(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 3, 10),
                knowledge_time=_kt(date(2026, 3, 10)),
                analyst_firm="FirmA",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 3, 15),
                knowledge_time=_kt(date(2026, 3, 15)),
                analyst_firm="FirmC",
                target_price=Decimal("105"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        ],
    )

    result = compute_coverage_change_proxy(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result is None


def test_compute_coverage_change_proxy_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 1, 10),
                knowledge_time=_kt(date(2026, 1, 10)),
                analyst_firm="FirmA",
                target_price=Decimal("90"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 3, 10),
                knowledge_time=_kt(date(2026, 3, 10)),
                analyst_firm="FirmA",
                target_price=Decimal("100"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 3, 15),
                knowledge_time=_kt(date(2026, 3, 15)),
                analyst_firm="FirmB",
                target_price=Decimal("105"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_future_kt(as_of),
                analyst_firm="FirmC",
                target_price=Decimal("110"),
                prior_target=None,
                price_when_published=None,
                is_consensus=False,
            ),
        ],
    )

    result = compute_coverage_change_proxy(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result == 1


def test_compute_financial_health_trend_returns_delta(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            RatingEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 2, 1),
                knowledge_time=_kt(date(2026, 2, 1)),
                rating_score=2,
                rating_recommendation="Hold",
                dcf_rating=Decimal("2.0"),
                pe_rating=Decimal("2.0"),
                roe_rating=Decimal("2.0"),
            ),
            RatingEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                rating_score=4,
                rating_recommendation="Buy",
                dcf_rating=Decimal("4.0"),
                pe_rating=Decimal("4.0"),
                roe_rating=Decimal("4.0"),
            ),
        ],
    )

    result = compute_financial_health_trend(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result == pytest.approx(2.0)


def test_compute_financial_health_trend_returns_none_without_prior(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            RatingEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                rating_score=4,
                rating_recommendation="Buy",
                dcf_rating=Decimal("4.0"),
                pe_rating=Decimal("4.0"),
                roe_rating=Decimal("4.0"),
            ),
        ],
    )

    result = compute_financial_health_trend(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result is None


def test_compute_financial_health_trend_filters_future_knowledge_time(analyst_sf: sessionmaker) -> None:
    as_of = date(2026, 4, 23)
    _seed(
        analyst_sf,
        [
            RatingEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 2, 1),
                knowledge_time=_kt(date(2026, 2, 1)),
                rating_score=2,
                rating_recommendation="Hold",
                dcf_rating=Decimal("2.0"),
                pe_rating=Decimal("2.0"),
                roe_rating=Decimal("2.0"),
            ),
            RatingEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 20),
                knowledge_time=_kt(date(2026, 4, 20)),
                rating_score=4,
                rating_recommendation="Buy",
                dcf_rating=Decimal("4.0"),
                pe_rating=Decimal("4.0"),
                roe_rating=Decimal("4.0"),
            ),
            RatingEvent(
                ticker=TEST_TICKER,
                event_date=date(2026, 4, 23),
                knowledge_time=_future_kt(as_of),
                rating_score=10,
                rating_recommendation="Strong Buy",
                dcf_rating=Decimal("10.0"),
                pe_rating=Decimal("10.0"),
                roe_rating=Decimal("10.0"),
            ),
        ],
    )

    result = compute_financial_health_trend(TEST_TICKER, as_of, 60, session_factory=analyst_sf)

    assert result == pytest.approx(2.0)
