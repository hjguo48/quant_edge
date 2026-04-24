"""FMP Earnings data source — analyst estimates, actuals, and surprise calculations.

Fetches per-ticker quarterly earnings data from FMP /stable/earnings endpoint.
Each record contains epsEstimated, epsActual, revenueEstimated, revenueActual.

PIT discipline: knowledge_time = lastUpdated date (when FMP published the data).
For future earnings (epsActual is null), knowledge_time = date the estimate was last updated.
For past earnings (epsActual is not null), knowledge_time = lastUpdated (typically earnings release day).
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


class EarningsEstimate(TimestampMixin, Base):
    """Stores quarterly earnings estimates and actuals with PIT timestamps."""

    __tablename__ = "earnings_estimates"
    __table_args__ = (
        sa.UniqueConstraint(
            "ticker",
            "fiscal_date",
            "knowledge_time",
            name="uq_earnings_estimates_version",
        ),
        sa.Index(
            "idx_earnings_estimates_lookup",
            "ticker",
            "knowledge_time",
            "fiscal_date",
        ),
    )

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(10), nullable=False)
    fiscal_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    eps_estimated: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    eps_actual: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    revenue_estimated: Mapped[int | None] = mapped_column(sa.BigInteger)
    revenue_actual: Mapped[int | None] = mapped_column(sa.BigInteger)
    knowledge_time: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    source: Mapped[str | None] = mapped_column(sa.String(20))


EARNINGS_COLUMNS = [
    "ticker",
    "fiscal_date",
    "eps_estimated",
    "eps_actual",
    "revenue_estimated",
    "revenue_actual",
    "knowledge_time",
    "source",
]


class FMPEarningsSource(DataSource):
    """Fetches earnings estimates/actuals from FMP /stable/earnings endpoint."""

    source_name = "fmp_earnings"
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.25,
    ) -> None:
        super().__init__(
            api_key or settings.FMP_API_KEY,
            min_request_interval=min_request_interval,
        )
        self._http_session: Any | None = None

    def _get_session(self) -> Any:
        if self._http_session is None:
            import requests

            self._http_session = requests.Session()
        return self._http_session

    @DataSource.retryable()
    def _fetch_ticker_earnings(self, ticker: str) -> list[dict[str, Any]]:
        """Fetch all earnings records for a single ticker."""
        self._before_request(f"earnings/{ticker}")
        session = self._get_session()
        url = f"{self.base_url}/earnings"
        params = {"symbol": ticker, "apikey": self.api_key}
        response = session.get(url, params=params, timeout=30)

        if not response.ok:
            self.classify_http_error(
                response.status_code, response.text, context=f"earnings/{ticker}"
            )

        raw = response.json()
        if not isinstance(raw, list):
            logger.warning("fmp_earnings unexpected response type for {}: {}", ticker, type(raw))
            return []

        rows: list[dict[str, Any]] = []
        for record in raw:
            fiscal_date_str = record.get("date")
            if not fiscal_date_str:
                continue

            try:
                fiscal_date = date.fromisoformat(fiscal_date_str)
            except (ValueError, TypeError):
                continue

            last_updated_str = record.get("lastUpdated", fiscal_date_str)
            try:
                last_updated = date.fromisoformat(last_updated_str)
            except (ValueError, TypeError):
                last_updated = fiscal_date

            # PIT: knowledge_time = lastUpdated (when FMP published this data point)
            knowledge_time = datetime.combine(
                last_updated,
                datetime.max.time(),
                tzinfo=timezone.utc,
            )

            rows.append(
                {
                    "ticker": ticker.upper(),
                    "fiscal_date": fiscal_date,
                    "eps_estimated": _to_decimal(record.get("epsEstimated")),
                    "eps_actual": _to_decimal(record.get("epsActual")),
                    "revenue_estimated": _to_int(record.get("revenueEstimated")),
                    "revenue_actual": _to_int(record.get("revenueActual")),
                    "knowledge_time": knowledge_time,
                    "source": "fmp",
                }
            )

        return rows

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        all_rows: list[dict[str, Any]] = []

        for ticker in self.normalize_tickers(tickers):
            records = self._fetch_ticker_earnings(ticker)
            for record in records:
                if start <= record["fiscal_date"] <= end:
                    all_rows.append(record)

        frame = self.dataframe_or_empty(all_rows, EARNINGS_COLUMNS)
        if not frame.empty:
            self._persist_earnings(frame)

        logger.info(
            "fmp_earnings fetched {} records across {} tickers between {} and {}",
            len(frame),
            len(set(frame["ticker"])) if not frame.empty else 0,
            start,
            end,
        )
        return frame

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        cutoff = self.coerce_datetime(since_date)
        lookback_start = self.coerce_date(since_date) - timedelta(days=180)
        frame = self.fetch_historical(tickers, lookback_start, date.today())
        if frame.empty:
            return frame
        incremental = frame.loc[
            pd.to_datetime(frame["knowledge_time"], utc=True) >= cutoff
        ]
        return incremental.reset_index(drop=True)

    def health_check(self) -> bool:
        try:
            records = self._fetch_ticker_earnings("AAPL")
            return len(records) > 0
        except Exception:
            return False

    def _persist_earnings(self, frame: pd.DataFrame) -> None:
        """Upsert earnings records into the database."""
        import math

        def _clean(val: Any) -> Any:
            """Convert nan/inf to None for DB insertion."""
            if val is None:
                return None
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val

        session_factory = get_session_factory()
        with session_factory() as session:
            for _, row in frame.iterrows():
                stmt = insert(EarningsEstimate).values(
                    ticker=row["ticker"],
                    fiscal_date=row["fiscal_date"],
                    eps_estimated=_clean(row["eps_estimated"]),
                    eps_actual=_clean(row["eps_actual"]),
                    revenue_estimated=_clean(row["revenue_estimated"]),
                    revenue_actual=_clean(row["revenue_actual"]),
                    knowledge_time=row["knowledge_time"],
                    source=row["source"],
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_earnings_estimates_version",
                    set_={
                        "eps_estimated": stmt.excluded.eps_estimated,
                        "eps_actual": stmt.excluded.eps_actual,
                        "revenue_estimated": stmt.excluded.revenue_estimated,
                        "revenue_actual": stmt.excluded.revenue_actual,
                        "source": stmt.excluded.source,
                    },
                )
                session.execute(stmt)
            session.commit()
            logger.info(
                "fmp_earnings persisted {} rows",
                len(frame),
            )


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        import math
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return Decimal(str(value))
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        import math
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return int(f)
    except Exception:
        return None
