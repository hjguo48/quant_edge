"""FMP Insider Trading data source.

Fetches Form 4 insider trades from FMP /stable/insider-trading/search.
PIT: knowledge_time = filingDate (SEC accepted date).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
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


class InsiderTrade(TimestampMixin, Base):
    __tablename__ = "insider_trades"
    __table_args__ = (
        sa.UniqueConstraint("ticker", "filing_date", "reporting_cik", "transaction_type",
                            "securities_transacted", name="uq_insider_trades_version"),
        sa.Index("idx_insider_trades_lookup", "ticker", "filing_date"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    filing_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    transaction_date: Mapped[date | None] = mapped_column(sa.Date)
    reporting_cik: Mapped[str | None] = mapped_column(sa.String(20))
    reporting_name: Mapped[str | None] = mapped_column(sa.String(200))
    type_of_owner: Mapped[str | None] = mapped_column(sa.String(200))
    transaction_type: Mapped[str | None] = mapped_column(sa.String(20))
    securities_transacted: Mapped[int | None] = mapped_column(sa.BigInteger)
    price: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    securities_owned: Mapped[int | None] = mapped_column(sa.BigInteger)
    acquisition_or_disposition: Mapped[str | None] = mapped_column(sa.String(5))
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    source: Mapped[str | None] = mapped_column(sa.String(20))


class FMPInsiderSource(DataSource):
    source_name = "fmp_insider"
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(self, api_key: str | None = None, *, min_request_interval: float = 0.25) -> None:
        super().__init__(api_key or settings.FMP_API_KEY, min_request_interval=min_request_interval)
        self._http_session: Any | None = None

    def _get_session(self) -> Any:
        if self._http_session is None:
            import requests
            self._http_session = requests.Session()
        return self._http_session

    @DataSource.retryable()
    def _fetch_ticker(self, ticker: str, pages: int = 10) -> list[dict[str, Any]]:
        self._before_request(f"insider-trading/{ticker}")
        session = self._get_session()
        all_rows = []
        for page in range(pages):
            r = session.get(f"{self.base_url}/insider-trading/search",
                            params={"symbol": ticker, "page": page, "limit": 100, "apikey": self.api_key},
                            timeout=30)
            if not r.ok:
                if r.status_code == 404:
                    break
                self.classify_http_error(r.status_code, r.text, context=f"insider-trading/{ticker}")
            data = r.json()
            if not isinstance(data, list) or not data:
                break
            for rec in data:
                fd = rec.get("filingDate")
                if not fd:
                    continue
                try:
                    filing_date = date.fromisoformat(fd[:10])
                except (ValueError, TypeError):
                    continue
                td = rec.get("transactionDate")
                try:
                    trans_date = date.fromisoformat(td[:10]) if td else None
                except (ValueError, TypeError):
                    trans_date = None
                all_rows.append({
                    "ticker": ticker.upper(), "filing_date": filing_date,
                    "transaction_date": trans_date,
                    "reporting_cik": rec.get("reportingCik"),
                    "reporting_name": (rec.get("reportingName") or "")[:200],
                    "type_of_owner": (rec.get("typeOfOwner") or "")[:200],
                    "transaction_type": rec.get("transactionType"),
                    "securities_transacted": _intv(rec.get("securitiesTransacted")),
                    "price": _dec(rec.get("price")),
                    "securities_owned": _intv(rec.get("securitiesOwned")),
                    "acquisition_or_disposition": rec.get("acquisitionOrDisposition"),
                    "knowledge_time": datetime.combine(filing_date, datetime.max.time(), tzinfo=timezone.utc),
                    "source": "fmp",
                })
            if len(data) < 100:
                break
            self._throttle()
        return all_rows

    def fetch_historical(self, tickers: Sequence[str], start_date: date | datetime, end_date: date | datetime) -> pd.DataFrame:
        start, end = self.coerce_date(start_date), self.coerce_date(end_date)
        all_rows = []
        for t in self.normalize_tickers(tickers):
            for row in self._fetch_ticker(t):
                if start <= row["filing_date"] <= end:
                    all_rows.append(row)
        cols = ["ticker", "filing_date", "transaction_date", "reporting_cik", "reporting_name",
                "type_of_owner", "transaction_type", "securities_transacted", "price",
                "securities_owned", "acquisition_or_disposition", "knowledge_time", "source"]
        frame = self.dataframe_or_empty(all_rows, cols)
        if not frame.empty:
            self._persist(frame)
        logger.info("fmp_insider fetched {} rows for {} tickers", len(frame),
                     len(set(frame["ticker"])) if not frame.empty else 0)
        return frame

    def fetch_incremental(self, tickers: Sequence[str], since_date: date | datetime) -> pd.DataFrame:
        return self.fetch_historical(tickers, self.coerce_date(since_date) - timedelta(days=90), date.today())

    def health_check(self) -> bool:
        try:
            return len(self._fetch_ticker("AAPL", pages=1)) > 0
        except Exception:
            return False

    def _persist(self, frame: pd.DataFrame) -> None:
        session_factory = get_session_factory()
        with session_factory() as session:
            for _, row in frame.iterrows():
                try:
                    stmt = insert(InsiderTrade).values(
                        ticker=row["ticker"], filing_date=row["filing_date"],
                        transaction_date=row["transaction_date"],
                        reporting_cik=row["reporting_cik"], reporting_name=row["reporting_name"],
                        type_of_owner=row["type_of_owner"], transaction_type=row["transaction_type"],
                        securities_transacted=_clean(row["securities_transacted"]),
                        price=_clean(row["price"]),
                        securities_owned=_clean(row["securities_owned"]),
                        acquisition_or_disposition=row["acquisition_or_disposition"],
                        knowledge_time=row["knowledge_time"], source=row["source"],
                    ).on_conflict_do_nothing()
                    session.execute(stmt)
                except Exception:
                    continue
            session.commit()
            logger.info("fmp_insider persisted {} rows", len(frame))


def _dec(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else Decimal(str(v))
    except: return None

def _intv(v):
    if v is None: return None
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else int(f)
    except: return None

def _clean(val):
    if val is None: return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)): return None
    return val
