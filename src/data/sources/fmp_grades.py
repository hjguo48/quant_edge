"""FMP analyst grades events source."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

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
GRADES_COLUMNS = [
    "ticker",
    "event_date",
    "knowledge_time",
    "analyst_firm",
    "prior_grade",
    "new_grade",
    "action",
    "grade_score_change",
]


class GradesEvent(Base):
    __tablename__ = "grades_events"
    __table_args__ = (
        sa.UniqueConstraint("ticker", "event_date", "analyst_firm", name="uq_grade_event"),
        sa.Index("ix_grades_kt", "ticker", "knowledge_time"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    event_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    analyst_firm: Mapped[str] = mapped_column(sa.String(128), nullable=False)
    prior_grade: Mapped[str | None] = mapped_column(sa.String(64))
    new_grade: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    action: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    grade_score_change: Mapped[int] = mapped_column(sa.SmallInteger, nullable=False)


class FMPGradesSource(DataSource):
    source_name = "fmp_grades"
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
        self._before_request(f"grades/{ticker}")
        response = self._get_session().get(
            f"{self.base_url}/grades",
            params={"symbol": ticker, "apikey": self.api_key},
            timeout=30,
        )
        if response.status_code == 404:
            return []
        if not response.ok:
            self.classify_http_error(response.status_code, getattr(response, "text", ""), context=f"grades/{ticker}")
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return payload

    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        normalized_ticker = self.normalize_tickers([ticker])[0]
        rows: list[dict[str, Any]] = []
        for record in self._request_ticker(normalized_ticker):
            event_date = _parse_date(record.get("date"))
            analyst_firm = _clean_text(record.get("gradingCompany"))
            new_grade = _clean_text(record.get("newGrade"))
            if event_date is None or analyst_firm is None or new_grade is None:
                continue
            prior_grade = _clean_text(record.get("previousGrade"))
            action = _clean_text(record.get("action")) or "unknown"
            rows.append(
                {
                    "ticker": normalized_ticker,
                    "event_date": event_date,
                    "knowledge_time": _end_of_day_utc(event_date),
                    "analyst_firm": analyst_firm,
                    "prior_grade": prior_grade,
                    "new_grade": new_grade,
                    "action": action,
                    "grade_score_change": _grade_score(new_grade) - _grade_score(prior_grade),
                },
            )
        frame = self.dataframe_or_empty(rows, GRADES_COLUMNS)
        if not frame.empty:
            frame.sort_values(["ticker", "event_date", "analyst_firm"], inplace=True)
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
            return pd.DataFrame(columns=GRADES_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def health_check(self) -> bool:
        try:
            self.fetch_ticker("AAPL")
        except Exception as exc:
            logger.warning("fmp_grades health check failed: {}", exc)
            return False
        return True

    @staticmethod
    def _persist(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        session_factory = get_session_factory()
        with session_factory() as session:
            for row in frame.itertuples(index=False):
                stmt = insert(GradesEvent).values(
                    ticker=row.ticker,
                    event_date=row.event_date,
                    knowledge_time=row.knowledge_time,
                    analyst_firm=row.analyst_firm,
                    prior_grade=row.prior_grade,
                    new_grade=row.new_grade,
                    action=row.action,
                    grade_score_change=int(row.grade_score_change),
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_grade_event",
                    set_={
                        "knowledge_time": stmt.excluded.knowledge_time,
                        "prior_grade": stmt.excluded.prior_grade,
                        "new_grade": stmt.excluded.new_grade,
                        "action": stmt.excluded.action,
                        "grade_score_change": stmt.excluded.grade_score_change,
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


def _grade_score(raw_grade: str | None) -> int:
    if raw_grade is None:
        return 0
    normalized = raw_grade.strip().lower()
    if not normalized:
        return 0
    if "sell" in normalized:
        return -2
    if "underperform" in normalized or "underweight" in normalized:
        return -1
    if "hold" in normalized or "neutral" in normalized:
        return 0
    if "outperform" in normalized or "overweight" in normalized:
        return 1
    if "buy" in normalized:
        return 2
    return 0


def _end_of_day_utc(day_value: date) -> datetime:
    local_dt = datetime.combine(day_value, time(hour=23, minute=59), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)
