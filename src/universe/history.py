from __future__ import annotations

from datetime import date, datetime

import sqlalchemy as sa
from loguru import logger

from src.data.db.models import UniverseMembership
from src.data.db.pit import get_universe_pit
from src.data.db.session import get_session_factory


def get_historical_members(as_of: date | datetime, index_name: str = "SP500") -> list[str]:
    return get_universe_pit(as_of=as_of, index_name=index_name)


def record_membership_change(
    ticker: str,
    index_name: str,
    effective_date: date | datetime,
    reason: str,
) -> None:
    effective = effective_date.date() if isinstance(effective_date, datetime) else effective_date
    normalized_ticker = ticker.upper()
    session_factory = get_session_factory()

    with session_factory() as session:
        try:
            active_membership = (
                session.execute(
                    sa.select(UniverseMembership)
                    .where(
                        UniverseMembership.ticker == normalized_ticker,
                        UniverseMembership.index_name == index_name,
                        UniverseMembership.effective_date <= effective,
                        sa.or_(
                            UniverseMembership.end_date.is_(None),
                            UniverseMembership.end_date > effective,
                        ),
                    )
                    .order_by(UniverseMembership.effective_date.desc()),
                )
                .scalars()
                .first()
            )

            if active_membership is None:
                session.add(
                    UniverseMembership(
                        ticker=normalized_ticker,
                        index_name=index_name,
                        effective_date=effective,
                        reason=reason,
                    ),
                )
                action = "added"
            else:
                active_membership.end_date = effective
                active_membership.reason = reason
                action = "removed"

            session.commit()
            logger.info(
                "universe membership {} {} in {} effective {}",
                action,
                normalized_ticker,
                index_name,
                effective,
            )
        except Exception:
            session.rollback()
            raise
