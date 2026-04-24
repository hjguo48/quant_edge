"""FMP earnings calendar source."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo
import math

from loguru import logger
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import Base
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource, DataSourceError, RetryConfig

try:
    from sqlalchemy.orm import Mapped, mapped_column
except ImportError:
    class Mapped:
        @classmethod
        def __class_getitem__(cls, _item: Any) -> Any:
            return Any

    def mapped_column(*args: Any, **kwargs: Any) -> sa.Column[Any]:
        return sa.Column(*args, **kwargs)


EASTERN = ZoneInfo("America/New_York")
FMP_USER_AGENT = "QuantEdge/1.0 (research; contact: hjguo48@gmail.com)"
EARNINGS_CALENDAR_COLUMNS = [
    "ticker",
    "announce_date",
    "knowledge_time",
    "timing",
    "fiscal_period_end",
    "eps_estimate",
    "eps_actual",
    "revenue_estimate",
    "revenue_actual",
]


class EarningsCalendar(Base):
    __tablename__ = "earnings_calendar"
    __table_args__ = (
        sa.Index("ix_earnings_cal_kt", "ticker", "knowledge_time"),
    )

    ticker: Mapped[str] = mapped_column(sa.String(16), primary_key=True)
    announce_date: Mapped[date] = mapped_column(sa.Date, primary_key=True)
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    timing: Mapped[str | None] = mapped_column(sa.String(16))
    fiscal_period_end: Mapped[date | None] = mapped_column(sa.Date)
    eps_estimate: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    eps_actual: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    revenue_estimate: Mapped[int | None] = mapped_column(sa.BigInteger)
    revenue_actual: Mapped[int | None] = mapped_column(sa.BigInteger)


class FMPEarningsCalendarSource(DataSource):
    source_name = "fmp_earnings_calendar"
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.25,
        retry_config: RetryConfig | None = None,
        http_session: Any | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(
            api_key or settings.FMP_API_KEY,
            min_request_interval=min_request_interval,
            retry_config=retry_config,
        )
        self._http_session = http_session
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def _get_session(self) -> Any:
        if self._http_session is not None:
            return self._http_session
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("requests is not installed.") from exc
        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": FMP_USER_AGENT})
        self._http_session = session
        return session

    @DataSource.retryable()
    def _request_chunk(self, start_date: date, end_date: date) -> list[dict[str, Any]]:
        self._before_request(f"earnings-calendar/{start_date}/{end_date}")
        response = self._get_session().get(
            f"{self.base_url}/earnings-calendar",
            params={"from": start_date.isoformat(), "to": end_date.isoformat(), "apikey": self.api_key},
            timeout=30,
        )
        if response.status_code == 404:
            return []
        if not response.ok:
            self.classify_http_error(
                response.status_code,
                getattr(response, "text", ""),
                context=f"earnings-calendar/{start_date}/{end_date}",
            )
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return payload

    def fetch_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        if start_date > end_date:
            return pd.DataFrame(columns=EARNINGS_CALENDAR_COLUMNS)
        rows: list[dict[str, Any]] = []
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=364), end_date)
            for record in self._request_chunk(chunk_start, chunk_end):
                announce_date = _parse_date(record.get("date"))
                ticker = _clean_text(record.get("symbol"))
                if announce_date is None or ticker is None:
                    continue
                updated_from = _parse_date(record.get("updatedFromDate"))
                eps_actual = _decimal_or_none(record.get("eps"))
                knowledge_date = updated_from if (eps_actual is not None and updated_from is not None) else announce_date
                rows.append(
                    {
                        "ticker": ticker.upper(),
                        "announce_date": announce_date,
                        "knowledge_time": _end_of_day_utc(knowledge_date),
                        "timing": _clean_text(record.get("time")),
                        "fiscal_period_end": _parse_date(record.get("fiscalDateEnding")),
                        "eps_estimate": _decimal_or_none(record.get("epsEstimated")),
                        "eps_actual": eps_actual,
                        "revenue_estimate": _int_or_none(record.get("revenueEstimated")),
                        "revenue_actual": _int_or_none(record.get("revenue")),
                    },
                )
            chunk_start = chunk_end + timedelta(days=1)
        frame = self.dataframe_or_empty(rows, EARNINGS_CALENDAR_COLUMNS)
        if not frame.empty:
            frame.sort_values(["ticker", "announce_date"], inplace=True)
            frame.reset_index(drop=True, inplace=True)
        return frame

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> int:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        frame = self.fetch_range(start, end)
        if frame.empty:
            return 0
        ticker_filter = set(self.normalize_tickers(tickers))
        filtered = frame.loc[frame["ticker"].isin(ticker_filter)].copy()
        if filtered.empty:
            return 0
        return self._persist(filtered, now_dt=_ensure_utc(self._now_fn()))

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        start = self.coerce_date(since_date)
        end = date.today()
        frame = self.fetch_range(start, end)
        if frame.empty:
            return frame
        ticker_filter = set(self.normalize_tickers(tickers))
        filtered = frame.loc[frame["ticker"].isin(ticker_filter)].copy()
        if filtered.empty:
            return filtered
        self._persist(filtered, now_dt=_ensure_utc(self._now_fn()))
        return filtered

    def health_check(self) -> bool:
        today = date.today()
        try:
            self.fetch_range(today - timedelta(days=1), today)
        except Exception as exc:
            logger.warning("fmp_earnings_calendar health check failed: {}", exc)
            return False
        return True

    @staticmethod
    def _persist(frame: pd.DataFrame, *, now_dt: datetime) -> int:
        if frame.empty:
            return 0
        session_factory = get_session_factory()
        with session_factory() as session:
            for row in frame.itertuples(index=False):
                stmt = insert(EarningsCalendar).values(
                    ticker=row.ticker,
                    announce_date=row.announce_date,
                    knowledge_time=row.knowledge_time,
                    timing=row.timing,
                    fiscal_period_end=row.fiscal_period_end,
                    eps_estimate=row.eps_estimate,
                    eps_actual=row.eps_actual,
                    revenue_estimate=row.revenue_estimate,
                    revenue_actual=row.revenue_actual,
                )
                knowledge_time_value = sa.case(
                    (
                        sa.and_(
                            EarningsCalendar.eps_actual.is_(None),
                            stmt.excluded.eps_actual.is_not(None),
                        ),
                        sa.func.greatest(
                            EarningsCalendar.knowledge_time + sa.text("interval '1 second'"),
                            sa.literal(now_dt, type_=sa.DateTime(timezone=True)),
                        ),
                    ),
                    (
                        sa.and_(
                            EarningsCalendar.eps_actual.is_not(None),
                            stmt.excluded.eps_actual.is_(None),
                        ),
                        EarningsCalendar.knowledge_time,
                    ),
                    else_=stmt.excluded.knowledge_time,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[EarningsCalendar.ticker, EarningsCalendar.announce_date],
                    set_={
                        "knowledge_time": knowledge_time_value,
                        "timing": sa.func.coalesce(stmt.excluded.timing, EarningsCalendar.timing),
                        "fiscal_period_end": sa.func.coalesce(
                            stmt.excluded.fiscal_period_end,
                            EarningsCalendar.fiscal_period_end,
                        ),
                        "eps_estimate": stmt.excluded.eps_estimate,
                        "eps_actual": sa.func.coalesce(stmt.excluded.eps_actual, EarningsCalendar.eps_actual),
                        "revenue_estimate": stmt.excluded.revenue_estimate,
                        "revenue_actual": sa.func.coalesce(
                            stmt.excluded.revenue_actual,
                            EarningsCalendar.revenue_actual,
                        ),
                    },
                )
                session.execute(stmt)
            session.commit()
        return len(frame)


def _parse_date(raw_value: Any) -> date | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except (TypeError, ValueError):
        return None


def _clean_text(raw_value: Any) -> str | None:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    return text or None


def _decimal_or_none(raw_value: Any) -> Decimal | None:
    if raw_value is None:
        return None
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return None
    return Decimal(str(raw_value))


def _int_or_none(raw_value: Any) -> int | None:
    if raw_value is None:
        return None
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric_value) or math.isinf(numeric_value):
        return None
    return int(numeric_value)


def _end_of_day_utc(day_value: date) -> datetime:
    local_dt = datetime.combine(day_value, time(hour=23, minute=59), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
