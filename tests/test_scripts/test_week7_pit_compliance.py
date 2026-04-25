from __future__ import annotations

from datetime import date, datetime, time, timezone
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

import scripts._week7_ic_utils as utils
from scripts._week7_ic_utils import PanelContext
from src.data.db.models import StockPrice
from src.data.finra_short_sale import ShortSaleVolume
from src.data.sources.fmp_grades import GradesEvent
from src.data.sources.fmp_price_target import PriceTargetEvent
from src.data.sources.fmp_ratings import RatingEvent

EASTERN = ZoneInfo("America/New_York")


def _session_factory(db_engine) -> sessionmaker:
    return sessionmaker(bind=db_engine, expire_on_commit=False)


def _ensure_tables(db_engine) -> None:
    for table in (
        ShortSaleVolume.__table__,
        StockPrice.__table__,
        GradesEvent.__table__,
        PriceTargetEvent.__table__,
        RatingEvent.__table__,
    ):
        table.create(bind=db_engine, checkfirst=True)


def _clear_test_rows(session_factory: sessionmaker) -> None:
    with session_factory() as session:
        session.execute(sa.delete(ShortSaleVolume).where(ShortSaleVolume.ticker.like("TST_PIT%")))
        session.execute(sa.delete(PriceTargetEvent).where(PriceTargetEvent.ticker.like("TST_PIT%")))
        session.execute(sa.delete(RatingEvent).where(RatingEvent.ticker.like("TST_PIT%")))
        session.execute(sa.delete(GradesEvent).where(GradesEvent.ticker.like("TST_PIT%")))
        session.execute(sa.delete(StockPrice).where(StockPrice.ticker.like("TST_PIT%")))
        session.commit()


def _kt(day: date, hour: int = 23, minute: int = 59, second: int = 0) -> datetime:
    return datetime.combine(day, time(hour=hour, minute=minute, second=second), tzinfo=EASTERN).astimezone(
        timezone.utc,
    )


def _seed(session_factory: sessionmaker, entities: list[object]) -> None:
    with session_factory() as session:
        session.add_all(entities)
        session.commit()


@pytest.fixture
def week7_sf(db_engine) -> sessionmaker:
    _ensure_tables(db_engine)
    sf = _session_factory(db_engine)
    _clear_test_rows(sf)
    return sf


def test_build_shorting_panel_filters_future_knowledge_time(week7_sf: sessionmaker) -> None:
    ticker = "TST_PIT_S"
    early_trade_date = date(2026, 1, 9)
    later_trade_date = date(2026, 1, 16)
    _seed(
        week7_sf,
        [
            ShortSaleVolume(
                ticker=ticker,
                trade_date=early_trade_date,
                market="CNMS",
                knowledge_time=_kt(date(2026, 1, 15)),
                short_volume=90,
                short_exempt_volume=0,
                total_volume=100,
                file_etag="future-visible",
            ),
            ShortSaleVolume(
                ticker=ticker,
                trade_date=later_trade_date,
                market="CNMS",
                knowledge_time=_kt(later_trade_date),
                short_volume=30,
                short_exempt_volume=0,
                total_volume=100,
                file_etag="later",
            ),
        ],
    )

    panel = utils.build_shorting_panel(
        tickers=[ticker],
        trade_dates=[early_trade_date, later_trade_date],
        requested_features=["short_sale_ratio_1d"],
        session_factory=week7_sf,
    )

    row_map = panel.loc[panel["feature_name"] == "short_sale_ratio_1d"].set_index("trade_date")["feature_value"]
    assert pd.isna(row_map[early_trade_date])
    assert float(row_map[later_trade_date]) == pytest.approx(0.30)


def test_build_analyst_proxy_panel_filters_future_knowledge_time(week7_sf: sessionmaker) -> None:
    ticker = "TST_PIT_A"
    early_trade_date = date(2026, 1, 9)
    later_trade_date = date(2026, 1, 16)
    _seed(
        week7_sf,
        [
            StockPrice(
                ticker=ticker,
                trade_date=early_trade_date,
                knowledge_time=_kt(early_trade_date),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000,
            ),
            StockPrice(
                ticker=ticker,
                trade_date=later_trade_date,
                knowledge_time=_kt(later_trade_date),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1200,
            ),
            PriceTargetEvent(
                ticker=ticker,
                event_date=date(2026, 1, 5),
                knowledge_time=_kt(date(2026, 1, 5)),
                analyst_firm="FirmA",
                target_price=110.0,
                prior_target=105.0,
                price_when_published=100.0,
                is_consensus=False,
            ),
            PriceTargetEvent(
                ticker=ticker,
                event_date=date(2026, 1, 6),
                knowledge_time=_kt(later_trade_date),
                analyst_firm="FirmB",
                target_price=200.0,
                prior_target=190.0,
                price_when_published=100.0,
                is_consensus=False,
            ),
        ],
    )

    panel = utils.build_analyst_proxy_panel(
        tickers=[ticker],
        trade_dates=[early_trade_date, later_trade_date],
        requested_features=["consensus_upside"],
        session_factory=week7_sf,
    )

    value_map = {
        row.trade_date: float(row.feature_value)
        for row in panel.itertuples(index=False)
        if row.feature_name == "consensus_upside" and pd.notna(row.feature_value)
    }
    assert value_map[early_trade_date] == pytest.approx(0.10)
    assert value_map[later_trade_date] == pytest.approx(0.55)


def test_build_or_load_week7_panel_cache_ignores_feature_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = PanelContext(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        sampled_tickers=("AAA",),
        sampled_trade_dates=(date(2026, 1, 9),),
        universe_size=503,
    )
    cache_path = tmp_path / "week7_cache.parquet"
    calls = {"feature_store": 0}

    monkeypatch.setattr(utils, "FeatureRegistry", lambda: object())
    monkeypatch.setattr(
        utils,
        "build_registry_feature_maps",
        lambda registry: ({"alpha": "technical", "beta": "daily"}, ["alpha", "beta"]),
    )
    monkeypatch.setattr(utils, "select_historical_feature_parquet", lambda: None)

    def fake_load_feature_store_panel(*, tickers, trade_dates, feature_names, session_factory=None):
        calls["feature_store"] += 1
        assert set(feature_names) == {"alpha", "beta"}
        return pd.DataFrame(
            [
                {"ticker": "AAA", "trade_date": trade_dates[0], "feature_name": "alpha", "feature_value": 1.0},
                {"ticker": "AAA", "trade_date": trade_dates[0], "feature_name": "beta", "feature_value": 2.0},
            ],
        )

    monkeypatch.setattr(utils, "load_feature_store_panel", fake_load_feature_store_panel)
    monkeypatch.setattr(
        utils,
        "build_week5_feature_panel",
        lambda **kwargs: pd.DataFrame(columns=["ticker", "trade_date", "feature_name", "feature_value"]),
    )

    alpha_panel = utils.build_or_load_week7_panel(
        context=context,
        feature_names=["alpha"],
        cache_path=cache_path,
    )
    beta_panel = utils.build_or_load_week7_panel(
        context=context,
        feature_names=["beta"],
        cache_path=cache_path,
    )

    assert calls["feature_store"] == 1
    assert alpha_panel["feature_name"].tolist() == ["alpha"]
    assert beta_panel["feature_name"].tolist() == ["beta"]

    meta = json.loads(cache_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["cached_feature_count"] == 2
    assert "feature_names" not in meta


def test_compute_financial_health_trend_from_frame_uses_365d_lookback_and_latest_visible_row() -> None:
    trade_date = date(2026, 4, 15)
    frame = pd.DataFrame(
        [
            {
                "event_date": date(2025, 4, 10),
                "knowledge_time": _kt(date(2025, 4, 10)),
                "rating_score": 2.0,
            },
            {
                "event_date": date(2025, 4, 10),
                "knowledge_time": _kt(date(2025, 4, 12)),
                "rating_score": 3.0,
            },
            {
                "event_date": date(2026, 3, 1),
                "knowledge_time": _kt(date(2026, 3, 2)),
                "rating_score": 5.0,
            },
            {
                "event_date": date(2026, 3, 1),
                "knowledge_time": _kt(date(2026, 4, 20)),
                "rating_score": 9.0,
            },
        ],
    )

    result = utils.compute_financial_health_trend_from_frame(frame, trade_date=trade_date)

    assert result == pytest.approx(2.0)
