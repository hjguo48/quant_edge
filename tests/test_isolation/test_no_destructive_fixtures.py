from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import sqlalchemy as sa

from src.data.finra_short_sale import ShortSaleVolume

EASTERN = ZoneInfo("America/New_York")


def _count_recent_short_sale_rows(db_engine) -> int:
    with db_engine.connect() as connection:
        return int(
            connection.execute(
                sa.text(
                    """
                    SELECT COUNT(*)
                    FROM short_sale_volume_daily
                    WHERE trade_date >= '2026-04-01'
                    """,
                ),
            ).scalar_one(),
        )


def test_isolated_session_factory_does_not_mutate_production_short_sale_table(
    db_engine,
    isolated_session_factory,
) -> None:
    ShortSaleVolume.__table__.create(bind=db_engine, checkfirst=True)
    before = _count_recent_short_sale_rows(db_engine)

    with isolated_session_factory([ShortSaleVolume.__table__]) as session_factory:
        with session_factory() as session:
            session.execute(sa.delete(ShortSaleVolume))
            session.add(
                ShortSaleVolume(
                    ticker="AAPL",
                    trade_date=date(2026, 5, 1),
                    knowledge_time=datetime.combine(
                        date(2026, 5, 1),
                        time(hour=18),
                        tzinfo=EASTERN,
                    ).astimezone(timezone.utc),
                    market="ADF",
                    short_volume=100,
                    short_exempt_volume=0,
                    total_volume=1000,
                    file_etag="isolation-test",
                ),
            )
            session.commit()
            isolated_count = session.execute(
                sa.select(sa.func.count()).select_from(ShortSaleVolume),
            ).scalar_one()

        assert isolated_count == 1
        assert _count_recent_short_sale_rows(db_engine) == before

    assert _count_recent_short_sale_rows(db_engine) == before
