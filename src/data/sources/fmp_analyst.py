"""FMP Analyst Estimates data source.

Fetches forward-looking consensus estimates from FMP /stable/analyst-estimates.
Key fields: epsAvg/High/Low, revenueAvg/High/Low, numAnalystsEps/Revenue.

PIT: knowledge_time approximated as fiscal_date (the quarter end date the estimate targets).
For revision momentum, we compare consecutive snapshots of the same future quarter.
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


class AnalystEstimate(TimestampMixin, Base):
    __tablename__ = "analyst_estimates"
    __table_args__ = (
        sa.UniqueConstraint("ticker", "fiscal_date", "period", name="uq_analyst_estimates_version"),
        sa.Index("idx_analyst_estimates_lookup", "ticker", "fiscal_date"),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    fiscal_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    period: Mapped[str] = mapped_column(sa.String(10), nullable=False)  # quarter/annual
    eps_avg: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    eps_high: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    eps_low: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    revenue_avg: Mapped[int | None] = mapped_column(sa.BigInteger)
    revenue_high: Mapped[int | None] = mapped_column(sa.BigInteger)
    revenue_low: Mapped[int | None] = mapped_column(sa.BigInteger)
    num_analysts_eps: Mapped[int | None] = mapped_column(sa.Integer)
    num_analysts_revenue: Mapped[int | None] = mapped_column(sa.Integer)
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    source: Mapped[str | None] = mapped_column(sa.String(20))


class FMPAnalystSource(DataSource):
    source_name = "fmp_analyst"
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
    def _fetch_ticker(self, ticker: str, period: str = "quarter") -> list[dict[str, Any]]:
        self._before_request(f"analyst-estimates/{ticker}")
        session = self._get_session()
        r = session.get(f"{self.base_url}/analyst-estimates",
                        params={"symbol": ticker, "period": period, "apikey": self.api_key}, timeout=30)
        if not r.ok:
            self.classify_http_error(r.status_code, r.text, context=f"analyst-estimates/{ticker}")
        raw = r.json()
        if not isinstance(raw, list):
            return []
        rows = []
        for rec in raw:
            fd = rec.get("date")
            if not fd:
                continue
            try:
                fiscal_date = date.fromisoformat(fd)
            except (ValueError, TypeError):
                continue
            rows.append({
                "ticker": ticker.upper(), "fiscal_date": fiscal_date, "period": period,
                "eps_avg": _dec(rec.get("epsAvg")), "eps_high": _dec(rec.get("epsHigh")),
                "eps_low": _dec(rec.get("epsLow")),
                "revenue_avg": _intv(rec.get("revenueAvg")), "revenue_high": _intv(rec.get("revenueHigh")),
                "revenue_low": _intv(rec.get("revenueLow")),
                "num_analysts_eps": _intv(rec.get("numAnalystsEps")),
                "num_analysts_revenue": _intv(rec.get("numAnalystsRevenue")),
                "knowledge_time": datetime.combine(fiscal_date, datetime.max.time(), tzinfo=timezone.utc),
                "source": "fmp",
            })
        return rows

    def fetch_historical(self, tickers: Sequence[str], start_date: date | datetime, end_date: date | datetime) -> pd.DataFrame:
        start, end = self.coerce_date(start_date), self.coerce_date(end_date)
        all_rows = []
        for t in self.normalize_tickers(tickers):
            for row in self._fetch_ticker(t):
                if start <= row["fiscal_date"] <= end:
                    all_rows.append(row)
        cols = ["ticker", "fiscal_date", "period", "eps_avg", "eps_high", "eps_low",
                "revenue_avg", "revenue_high", "revenue_low", "num_analysts_eps",
                "num_analysts_revenue", "knowledge_time", "source"]
        frame = self.dataframe_or_empty(all_rows, cols)
        if not frame.empty:
            self._persist(frame)
        logger.info("fmp_analyst fetched {} rows for {} tickers", len(frame),
                     len(set(frame["ticker"])) if not frame.empty else 0)
        return frame

    def fetch_incremental(self, tickers: Sequence[str], since_date: date | datetime) -> pd.DataFrame:
        return self.fetch_historical(tickers, self.coerce_date(since_date) - timedelta(days=180), date.today())

    def health_check(self) -> bool:
        try:
            return len(self._fetch_ticker("AAPL")) > 0
        except Exception:
            return False

    def _persist(self, frame: pd.DataFrame) -> None:
        session_factory = get_session_factory()
        with session_factory() as session:
            for _, row in frame.iterrows():
                stmt = insert(AnalystEstimate).values(
                    ticker=row["ticker"], fiscal_date=row["fiscal_date"], period=row["period"],
                    eps_avg=_clean(row["eps_avg"]), eps_high=_clean(row["eps_high"]), eps_low=_clean(row["eps_low"]),
                    revenue_avg=_clean(row["revenue_avg"]), revenue_high=_clean(row["revenue_high"]),
                    revenue_low=_clean(row["revenue_low"]),
                    num_analysts_eps=_clean(row["num_analysts_eps"]),
                    num_analysts_revenue=_clean(row["num_analysts_revenue"]),
                    knowledge_time=row["knowledge_time"], source=row["source"],
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_analyst_estimates_version",
                    set_={"eps_avg": stmt.excluded.eps_avg, "eps_high": stmt.excluded.eps_high,
                           "eps_low": stmt.excluded.eps_low, "revenue_avg": stmt.excluded.revenue_avg,
                           "revenue_high": stmt.excluded.revenue_high, "revenue_low": stmt.excluded.revenue_low,
                           "num_analysts_eps": stmt.excluded.num_analysts_eps,
                           "num_analysts_revenue": stmt.excluded.num_analysts_revenue},
                )
                session.execute(stmt)
            session.commit()
            logger.info("fmp_analyst persisted {} rows", len(frame))


def _dec(v: Any) -> Decimal | None:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return Decimal(str(v))
    except Exception:
        return None

def _intv(v: Any) -> int | None:
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)
    except Exception:
        return None

def _clean(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val
