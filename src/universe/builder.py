from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import UniverseMembership
from src.data.db.pit import get_prices_pit
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSourceError
from src.universe.history import get_historical_members


def build_universe(
    as_of: date | datetime,
    index_name: str = "SP500",
    min_adv_usd: float = 50_000_000,
) -> list[str]:
    as_of_ts = _coerce_as_of(as_of)
    as_of_date = as_of_ts.date()

    if as_of_date < date.today():
        historical_members = get_historical_members(as_of_ts, index_name=index_name)
        if historical_members:
            logger.info(
                "returning {} historical {} members for {}",
                len(historical_members),
                index_name,
                as_of_date,
            )
            return historical_members
        raise ValueError(
            "Historical universe membership is missing. Refusing to backfill from current constituents.",
        )

    constituents = _fetch_index_constituents(index_name=index_name)
    filtered = _filter_by_adv(constituents, as_of_ts, min_adv_usd=min_adv_usd)
    _sync_membership(filtered, as_of_date=as_of_date, index_name=index_name, reason="rebalance")
    logger.info(
        "built {} universe with {} members after ADV filtering",
        index_name,
        len(filtered),
    )
    return filtered


def _fetch_index_constituents(index_name: str) -> list[str]:
    if index_name.upper() != "SP500":
        raise ValueError(f"Unsupported index_name={index_name!r}. Only SP500 is implemented.")

    if settings.FMP_API_KEY:
        try:
            return _fetch_sp500_from_fmp()
        except Exception as exc:
            logger.warning("FMP constituent fetch failed, falling back to Wikipedia: {}", exc)

    return _fetch_sp500_from_wikipedia()


def _fetch_sp500_from_fmp() -> list[str]:
    try:
        import requests
    except ImportError as exc:
        raise DataSourceError(
            "requests is not installed. Add the phase1-week2 dependency group.",
        ) from exc

    response = requests.get(
        "https://financialmodelingprep.com/api/v3/sp500_constituent",
        params={"apikey": settings.FMP_API_KEY},
        timeout=30,
    )
    if response.status_code != 200:
        raise DataSourceError(
            f"FMP constituent request failed with HTTP {response.status_code}: {response.text[:500]}",
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise DataSourceError(f"Unexpected FMP constituent payload: {type(payload).__name__}")

    tickers = [str(item.get("symbol", "")).replace(".", "-").upper() for item in payload]
    return sorted({ticker for ticker in tickers if ticker})


def _fetch_sp500_from_wikipedia() -> list[str]:
    try:
        import requests
    except ImportError as exc:
        raise DataSourceError(
            "requests is not installed. Add the phase1-week2 dependency group.",
        ) from exc

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(response.text)
    if not tables:
        raise DataSourceError("Wikipedia did not return an S&P 500 constituent table.")

    table = tables[0]
    symbol_column = "Symbol" if "Symbol" in table.columns else table.columns[0]
    tickers = table[symbol_column].astype(str).str.replace(".", "-", regex=False).str.upper()
    return sorted(set(tickers))


def _filter_by_adv(tickers: list[str], as_of: datetime, *, min_adv_usd: float) -> list[str]:
    if not tickers:
        return []

    lookback_start = as_of.date() - timedelta(days=40)
    prices = get_prices_pit(
        tickers=tickers,
        start_date=lookback_start,
        end_date=as_of.date(),
        as_of=as_of,
    )
    if prices.empty:
        logger.warning("ADV filter found no PIT prices; returning an empty universe")
        return []

    price_frame = prices.copy()
    close_series = pd.to_numeric(price_frame["close"], errors="coerce")
    adj_close_series = pd.to_numeric(price_frame["adj_close"], errors="coerce")
    volume_series = pd.to_numeric(price_frame["volume"], errors="coerce")
    price_frame["dollar_volume"] = close_series.fillna(adj_close_series) * volume_series
    price_frame.sort_values(["ticker", "trade_date"], inplace=True)
    latest_20 = price_frame.groupby("ticker", group_keys=False).tail(20)
    adv = latest_20.groupby("ticker")["dollar_volume"].mean()
    observation_count = latest_20.groupby("ticker").size()
    liquid = adv[(adv >= min_adv_usd) & (observation_count >= 20)].index.tolist()
    return sorted(liquid)


def _sync_membership(
    tickers: list[str],
    *,
    as_of_date: date,
    index_name: str,
    reason: str,
) -> None:
    desired = set(tickers)
    session_factory = get_session_factory()

    with session_factory() as session:
        try:
            active_memberships = (
                session.execute(
                    sa.select(UniverseMembership).where(
                        UniverseMembership.index_name == index_name,
                        UniverseMembership.effective_date <= as_of_date,
                        sa.or_(
                            UniverseMembership.end_date.is_(None),
                            UniverseMembership.end_date > as_of_date,
                        ),
                    ),
                )
                .scalars()
                .all()
            )
            active_by_ticker = {row.ticker: row for row in active_memberships}

            for ticker, membership in active_by_ticker.items():
                if ticker not in desired:
                    membership.end_date = as_of_date
                    membership.reason = f"{reason}_removed"

            new_rows = [
                {
                    "ticker": ticker,
                    "index_name": index_name,
                    "effective_date": as_of_date,
                    "end_date": None,
                    "reason": reason,
                }
                for ticker in sorted(desired - set(active_by_ticker))
            ]
            if new_rows:
                statement = insert(UniverseMembership).values(new_rows)
                upsert = statement.on_conflict_do_update(
                    constraint="uq_universe_membership_entry",
                    set_={
                        "end_date": statement.excluded.end_date,
                        "reason": statement.excluded.reason,
                    },
                )
                session.execute(upsert)

            session.commit()
        except Exception as exc:
            session.rollback()
            logger.opt(exception=exc).error("failed to synchronize universe membership")
            raise DataSourceError("Failed to write universe membership.") from exc


def _coerce_as_of(as_of: date | datetime) -> datetime:
    if isinstance(as_of, datetime):
        if as_of.tzinfo is None:
            return as_of.replace(tzinfo=timezone.utc)
        return as_of
    return datetime.combine(as_of, datetime.max.time(), tzinfo=timezone.utc)
