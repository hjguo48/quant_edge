"""Polygon Short Interest data source.

Fetches bi-weekly short interest from /stocks/v1/short-interest.
PIT: knowledge_time = settlement_date + 8 business days (FINRA standard).

FINRA publishes short-interest reports roughly 8 business days after each
mid-month and end-of-month settlement. The previous heuristic of
``settlement_date + 3 calendar days`` was too aggressive and let backtests
"see" short-interest data 5+ days before it was actually public — flagged in
the data audit on 2026-04-25 (P1-3, 122k rows affected).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
from loguru import logger
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import Base, TimestampMixin
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource

try:
    from sqlalchemy.orm import Mapped, mapped_column
except ImportError:
    class Mapped:
        @classmethod
        def __class_getitem__(cls, _item: Any) -> Any:
            return Any

    def mapped_column(*args: Any, **kwargs: Any) -> sa.Column[Any]:
        return sa.Column(*args, **kwargs)

import math


class ShortInterest(TimestampMixin, Base):
    __tablename__ = "short_interest"
    __table_args__ = (
        sa.UniqueConstraint("ticker", "settlement_date", name="uq_short_interest_version"),
        sa.Index("idx_short_interest_lookup", "ticker", "settlement_date"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    settlement_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    short_interest: Mapped[int | None] = mapped_column(sa.BigInteger)
    avg_daily_volume: Mapped[int | None] = mapped_column(sa.BigInteger)
    days_to_cover: Mapped[float | None] = mapped_column(sa.Numeric(10, 2))
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    source: Mapped[str | None] = mapped_column(sa.String(20))


class PolygonShortInterestSource(DataSource):
    source_name = "polygon_short"

    def __init__(self, api_key: str | None = None, *, min_request_interval: float = 0.25) -> None:
        super().__init__(api_key or settings.POLYGON_API_KEY, min_request_interval=min_request_interval)
        self._http_session: Any | None = None

    def _get_session(self) -> Any:
        if self._http_session is None:
            import requests
            self._http_session = requests.Session()
        return self._http_session

    @DataSource.retryable()
    def _fetch_ticker(self, ticker: str, limit: int = 1000) -> list[dict[str, Any]]:
        self._before_request(f"short-interest/{ticker}")
        session = self._get_session()
        all_results = []
        next_url = None

        while True:
            if next_url:
                r = session.get(next_url, timeout=30)
            else:
                r = session.get("https://api.polygon.io/stocks/v1/short-interest",
                                params={"ticker": ticker, "limit": min(limit, 1000), "apiKey": self.api_key},
                                timeout=30)
            if not r.ok:
                self.classify_http_error(r.status_code, r.text, context=f"short-interest/{ticker}")
            data = r.json()
            results = data.get("results", [])
            if not results:
                break
            all_results.extend(results)
            if len(all_results) >= limit:
                break
            next_url = data.get("next_url")
            if next_url:
                next_url = f"{next_url}&apiKey={self.api_key}"
            else:
                break
            self._throttle()

        rows = []
        for rec in all_results[:limit]:
            sd = rec.get("settlement_date")
            if not sd:
                continue
            try:
                sdate = date.fromisoformat(sd)
            except (ValueError, TypeError):
                continue
            # PIT: FINRA publishes short interest ~8 business days after settlement
            kt = datetime.combine(_add_business_days(sdate, 8), datetime.max.time(), tzinfo=timezone.utc)
            rows.append({
                "ticker": ticker.upper(),
                "settlement_date": sdate,
                "short_interest": _intv(rec.get("short_interest")),
                "avg_daily_volume": _intv(rec.get("avg_daily_volume")),
                "days_to_cover": _floatv(rec.get("days_to_cover")),
                "knowledge_time": kt,
                "source": "polygon",
            })
        return rows

    def fetch_historical(self, tickers: Sequence[str], start_date: date | datetime, end_date: date | datetime) -> pd.DataFrame:
        start, end = self.coerce_date(start_date), self.coerce_date(end_date)
        all_rows = []
        for t in self.normalize_tickers(tickers):
            for row in self._fetch_ticker(t):
                if start <= row["settlement_date"] <= end:
                    all_rows.append(row)
        cols = ["ticker", "settlement_date", "short_interest", "avg_daily_volume",
                "days_to_cover", "knowledge_time", "source"]
        frame = self.dataframe_or_empty(all_rows, cols)
        if not frame.empty:
            self._persist(frame)
        logger.info("polygon_short fetched {} rows for {} tickers", len(frame),
                     len(set(frame["ticker"])) if not frame.empty else 0)
        return frame

    def fetch_incremental(self, tickers: Sequence[str], since_date: date | datetime) -> pd.DataFrame:
        return self.fetch_historical(tickers, self.coerce_date(since_date) - timedelta(days=30), date.today())

    def health_check(self) -> bool:
        try:
            return len(self._fetch_ticker("AAPL", limit=1)) > 0
        except Exception:
            return False

    def _persist(self, frame: pd.DataFrame) -> None:
        session_factory = get_session_factory()
        with session_factory() as session:
            for _, row in frame.iterrows():
                stmt = insert(ShortInterest).values(
                    ticker=row["ticker"], settlement_date=row["settlement_date"],
                    short_interest=_clean(row["short_interest"]),
                    avg_daily_volume=_clean(row["avg_daily_volume"]),
                    days_to_cover=_clean(row["days_to_cover"]),
                    knowledge_time=row["knowledge_time"], source=row["source"],
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_short_interest_version",
                    set_={"short_interest": stmt.excluded.short_interest,
                           "avg_daily_volume": stmt.excluded.avg_daily_volume,
                           "days_to_cover": stmt.excluded.days_to_cover},
                )
                session.execute(stmt)
            session.commit()
            logger.info("polygon_short persisted {} rows", len(frame))


def _add_business_days(start: date, n: int) -> date:
    cur = start
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            added += 1
    return cur


def _intv(v: Any) -> int | None:
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else int(f)
    except: return None

def _floatv(v: Any) -> float | None:
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except: return None

def _clean(val: Any) -> Any:
    if val is None: return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return None
    return val
