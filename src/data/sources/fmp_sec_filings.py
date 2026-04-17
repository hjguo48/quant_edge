"""FMP SEC Filings data source.

Fetches 8-K and other SEC filings from FMP /stable/sec-filings-8k and /stable/sec-filings-search/symbol.
PIT: knowledge_time = acceptedDate (when SEC accepted the filing).
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


class SecFiling(TimestampMixin, Base):
    __tablename__ = "sec_filings"
    __table_args__ = (
        sa.UniqueConstraint("ticker", "accepted_date", "form_type", name="uq_sec_filings_version"),
        sa.Index("idx_sec_filings_lookup", "ticker", "accepted_date"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    cik: Mapped[str | None] = mapped_column(sa.String(20))
    filing_date: Mapped[date | None] = mapped_column(sa.Date)
    accepted_date: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    form_type: Mapped[str] = mapped_column(sa.String(20), nullable=False)
    has_financials: Mapped[bool | None] = mapped_column(sa.Boolean)
    link: Mapped[str | None] = mapped_column(sa.String(500))
    final_link: Mapped[str | None] = mapped_column(sa.String(500))
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    source: Mapped[str | None] = mapped_column(sa.String(20))


class FMPSecFilingsSource(DataSource):
    source_name = "fmp_sec"
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
    def _fetch_by_symbol(self, ticker: str, start: date, end: date) -> list[dict[str, Any]]:
        """Fetch filings for a ticker using sec-filings-search/symbol (max 90 day range)."""
        self._before_request(f"sec-filings/{ticker}")
        session = self._get_session()
        all_rows = []

        # API has 90-day max range, so chunk
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=89), end)
            for page in range(100):  # max pages
                r = session.get(f"{self.base_url}/sec-filings-search/symbol",
                                params={"symbol": ticker, "from": str(chunk_start), "to": str(chunk_end),
                                        "page": page, "limit": 100, "apikey": self.api_key},
                                timeout=30)
                if not r.ok:
                    break
                data = r.json()
                if not isinstance(data, list) or not data:
                    break
                for rec in data:
                    row = self._parse_filing(ticker, rec)
                    if row:
                        all_rows.append(row)
                if len(data) < 100:
                    break
                self._throttle()
            chunk_start = chunk_end + timedelta(days=1)

        return all_rows

    def _parse_filing(self, ticker: str, rec: dict) -> dict[str, Any] | None:
        ad = rec.get("acceptedDate")
        if not ad:
            return None
        try:
            accepted = datetime.fromisoformat(ad.replace("Z", "+00:00"))
            if accepted.tzinfo is None:
                accepted = accepted.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return None
        fd = rec.get("filingDate")
        try:
            filing_date = date.fromisoformat(fd[:10]) if fd else None
        except (ValueError, TypeError):
            filing_date = None
        return {
            "ticker": ticker.upper(),
            "cik": rec.get("cik"),
            "filing_date": filing_date,
            "accepted_date": accepted,
            "form_type": rec.get("formType", ""),
            "has_financials": rec.get("hasFinancials"),
            "link": (rec.get("link") or "")[:500],
            "final_link": (rec.get("finalLink") or "")[:500],
            "knowledge_time": accepted,  # PIT = SEC acceptance time
            "source": "fmp",
        }

    def fetch_historical(self, tickers: Sequence[str], start_date: date | datetime, end_date: date | datetime) -> pd.DataFrame:
        start, end = self.coerce_date(start_date), self.coerce_date(end_date)
        all_rows = []
        for t in self.normalize_tickers(tickers):
            all_rows.extend(self._fetch_by_symbol(t, start, end))
        cols = ["ticker", "cik", "filing_date", "accepted_date", "form_type",
                "has_financials", "link", "final_link", "knowledge_time", "source"]
        frame = self.dataframe_or_empty(all_rows, cols)
        if not frame.empty:
            self._persist(frame)
        logger.info("fmp_sec fetched {} filings for {} tickers", len(frame),
                     len(set(frame["ticker"])) if not frame.empty else 0)
        return frame

    def fetch_incremental(self, tickers: Sequence[str], since_date: date | datetime) -> pd.DataFrame:
        return self.fetch_historical(tickers, self.coerce_date(since_date) - timedelta(days=30), date.today())

    def health_check(self) -> bool:
        try:
            rows = self._fetch_by_symbol("AAPL", date(2024, 1, 1), date(2024, 3, 1))
            return len(rows) > 0
        except Exception:
            return False

    def _persist(self, frame: pd.DataFrame) -> None:
        session_factory = get_session_factory()
        with session_factory() as session:
            for _, row in frame.iterrows():
                try:
                    stmt = insert(SecFiling).values(
                        ticker=row["ticker"], cik=row["cik"], filing_date=row["filing_date"],
                        accepted_date=row["accepted_date"], form_type=row["form_type"],
                        has_financials=row["has_financials"], link=row["link"],
                        final_link=row["final_link"], knowledge_time=row["knowledge_time"],
                        source=row["source"],
                    ).on_conflict_do_nothing()
                    session.execute(stmt)
                except Exception:
                    continue
            session.commit()
            logger.info("fmp_sec persisted {} rows", len(frame))
