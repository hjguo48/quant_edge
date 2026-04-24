"""FMP ratings history source."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timezone
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
RATINGS_COLUMNS = [
    "ticker",
    "event_date",
    "knowledge_time",
    "rating_score",
    "rating_recommendation",
    "dcf_rating",
    "pe_rating",
    "roe_rating",
]


class RatingEvent(Base):
    __tablename__ = "ratings_events"
    __table_args__ = (
        sa.Index("ix_ratings_kt", "ticker", "knowledge_time"),
    )

    ticker: Mapped[str] = mapped_column(sa.String(16), primary_key=True)
    event_date: Mapped[date] = mapped_column(sa.Date, primary_key=True)
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    rating_score: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False)
    rating_recommendation: Mapped[str | None] = mapped_column(sa.String(32))
    dcf_rating: Mapped[Decimal | None] = mapped_column(sa.Numeric(6, 2))
    pe_rating: Mapped[Decimal | None] = mapped_column(sa.Numeric(6, 2))
    roe_rating: Mapped[Decimal | None] = mapped_column(sa.Numeric(6, 2))


class FMPRatingsSource(DataSource):
    source_name = "fmp_ratings"
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.25,
        retry_config: RetryConfig | None = None,
        http_session: Any | None = None,
    ) -> None:
        super().__init__(
            api_key or settings.FMP_API_KEY,
            min_request_interval=min_request_interval,
            retry_config=retry_config,
        )
        self._http_session = http_session

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
    def _request_ticker(self, ticker: str) -> list[dict[str, Any]]:
        self._before_request(f"ratings-historical/{ticker}")
        # limit=5000 returns ~20 years of daily snapshots. FMP ignores `page` and
        # `from`/`to` on this endpoint — `limit` is the only depth control.
        response = self._get_session().get(
            f"{self.base_url}/ratings-historical",
            params={"symbol": ticker, "limit": 5000, "apikey": self.api_key},
            timeout=60,
        )
        if response.status_code == 404:
            return []
        if not response.ok:
            self.classify_http_error(
                response.status_code,
                getattr(response, "text", ""),
                context=f"ratings-historical/{ticker}",
            )
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return payload

    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        normalized_ticker = self.normalize_tickers([ticker])[0]
        payload = self._request_ticker(normalized_ticker)
        rows: list[dict[str, Any]] = []
        for record in payload:
            event_date = _parse_date(record.get("date"))
            # FMP /stable/ratings-historical schema (2025+): `overallScore` not
            # `ratingScore`; financial ratios use *Score naming.
            rating_score = _int_or_none(record.get("overallScore"))
            if event_date is None or rating_score is None:
                continue
            rows.append(
                {
                    "ticker": normalized_ticker,
                    "event_date": event_date,
                    "knowledge_time": _end_of_day_utc(event_date),
                    "rating_score": rating_score,
                    # The letter grade (A/B/C/D) is the closest analog to a
                    # recommendation tag in this endpoint; no dedicated field exists.
                    "rating_recommendation": _clean_text(record.get("rating")),
                    "dcf_rating": _decimal_or_none(record.get("discountedCashFlowScore")),
                    "pe_rating": _decimal_or_none(record.get("priceToEarningsScore")),
                    "roe_rating": _decimal_or_none(record.get("returnOnEquityScore")),
                },
            )
        # Fail loud on schema drift: if FMP returned rows but we parsed zero,
        # the expected fields (date + overallScore) are renamed/missing.
        if payload and not rows:
            sample_keys = sorted(payload[0].keys()) if isinstance(payload[0], dict) else []
            raise DataSourceError(
                f"fmp_ratings: payload non-empty ({len(payload)} rows) but zero parsed for "
                f"{normalized_ticker}. Likely FMP schema change. Received keys: {sample_keys}. "
                "Required: date + overallScore."
            )
        frame = self.dataframe_or_empty(rows, RATINGS_COLUMNS)
        if not frame.empty:
            frame.sort_values(["ticker", "event_date"], inplace=True)
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
        if start > end:
            return 0
        total_rows = 0
        for ticker in self.normalize_tickers(tickers):
            frame = self.fetch_ticker(ticker)
            if frame.empty:
                continue
            filtered = frame.loc[(frame["event_date"] >= start) & (frame["event_date"] <= end)].copy()
            if filtered.empty:
                continue
            total_rows += self._persist(filtered)
        return total_rows

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        start = self.coerce_date(since_date)
        end = date.today()
        frames: list[pd.DataFrame] = []
        for ticker in self.normalize_tickers(tickers):
            frame = self.fetch_ticker(ticker)
            if frame.empty:
                continue
            filtered = frame.loc[(frame["event_date"] >= start) & (frame["event_date"] <= end)].copy()
            if filtered.empty:
                continue
            self._persist(filtered)
            frames.append(filtered)
        if not frames:
            return pd.DataFrame(columns=RATINGS_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def health_check(self) -> bool:
        try:
            self.fetch_ticker("AAPL")
        except Exception as exc:
            logger.warning("fmp_ratings health check failed: {}", exc)
            return False
        return True

    @staticmethod
    def _persist(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        session_factory = get_session_factory()
        with session_factory() as session:
            for row in frame.itertuples(index=False):
                stmt = insert(RatingEvent).values(
                    ticker=row.ticker,
                    event_date=row.event_date,
                    knowledge_time=row.knowledge_time,
                    rating_score=int(row.rating_score),
                    rating_recommendation=row.rating_recommendation,
                    dcf_rating=row.dcf_rating,
                    pe_rating=row.pe_rating,
                    roe_rating=row.roe_rating,
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[RatingEvent.ticker, RatingEvent.event_date],
                    set_={
                        "knowledge_time": stmt.excluded.knowledge_time,
                        "rating_score": stmt.excluded.rating_score,
                        "rating_recommendation": stmt.excluded.rating_recommendation,
                        "dcf_rating": stmt.excluded.dcf_rating,
                        "pe_rating": stmt.excluded.pe_rating,
                        "roe_rating": stmt.excluded.roe_rating,
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


def _end_of_day_utc(day_value: date) -> datetime:
    local_dt = datetime.combine(day_value, time(hour=23, minute=59), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)
