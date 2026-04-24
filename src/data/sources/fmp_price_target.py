"""FMP price target history source with stable consensus fallback."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
PRICE_TARGET_COLUMNS = [
    "ticker",
    "event_date",
    "knowledge_time",
    "analyst_firm",
    "target_price",
    "prior_target",
    "price_when_published",
    "is_consensus",
]


class PriceTargetEvent(Base):
    __tablename__ = "price_target_events"
    __table_args__ = (
        sa.UniqueConstraint("ticker", "event_date", "analyst_firm", name="uq_target_event"),
        sa.Index("ix_target_kt", "ticker", "knowledge_time"),
    )

    id: Mapped[int] = mapped_column(sa.BigInteger, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(sa.String(16), nullable=False)
    event_date: Mapped[date] = mapped_column(sa.Date, nullable=False)
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    analyst_firm: Mapped[str | None] = mapped_column(sa.String(128))
    target_price: Mapped[Decimal] = mapped_column(sa.Numeric(12, 4), nullable=False)
    prior_target: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    price_when_published: Mapped[Decimal | None] = mapped_column(sa.Numeric(12, 4))
    is_consensus: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, server_default=sa.text("false"))


class FMPPriceTargetSource(DataSource):
    source_name = "fmp_price_target"
    stable_base_url = "https://financialmodelingprep.com/stable"
    # Note: retired /api/v4/price-target endpoint returns 403 "Legacy Endpoint"
    # since FMP 2025-08 cut-off. Per-analyst history now comes from
    # /stable/price-target-news via paginated _request_news_page.

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
    def _request_consensus(self, ticker: str) -> dict[str, Any] | None:
        self._before_request(f"price-target-consensus/{ticker}")
        response = self._get_session().get(
            f"{self.stable_base_url}/price-target-consensus",
            params={"symbol": ticker, "apikey": self.api_key},
            timeout=30,
        )
        if response.status_code == 404:
            return None
        if not response.ok:
            self.classify_http_error(
                response.status_code,
                getattr(response, "text", ""),
                context=f"price-target-consensus/{ticker}",
            )
        payload = response.json()
        if isinstance(payload, list):
            if not payload:
                return None
            first = payload[0]
            return first if isinstance(first, dict) else None
        return payload if isinstance(payload, dict) else None

    @DataSource.retryable()
    def _request_news_page(self, ticker: str, page: int, limit: int = 100) -> list[dict[str, Any]]:
        """Fetch one page of /stable/price-target-news for per-analyst history.

        Replaces retired /api/v4/price-target (403 "Legacy Endpoint" since 2025-08).
        FMP hard-caps limit at 100 and ignores from/to — pagination via page=0,1,2,...
        """
        self._before_request(f"price-target-news/{ticker}?page={page}")
        response = self._get_session().get(
            f"{self.stable_base_url}/price-target-news",
            params={"symbol": ticker, "page": page, "limit": limit, "apikey": self.api_key},
            timeout=30,
        )
        response_text = getattr(response, "text", "")
        if response.status_code in {404, 410}:
            return []
        if not response.ok:
            self.classify_http_error(
                response.status_code,
                response_text,
                context=f"price-target-news/{ticker}",
            )
        payload = response.json()
        if not isinstance(payload, list):
            return []
        return payload

    def _request_news_all(self, ticker: str, max_pages: int = 50) -> list[dict[str, Any]]:
        """Pull all available per-analyst events via pagination.

        Terminates when a page returns < page_size records (tail of history).
        Warns loudly if max_pages is reached with a still-full page (i.e. more
        history exists than we pulled) so silent truncation cannot happen.
        """
        page_size = 100
        all_rows: list[dict[str, Any]] = []
        last_full_batch = True
        for page in range(max_pages):
            batch = self._request_news_page(ticker, page=page, limit=page_size)
            all_rows.extend(batch)
            if len(batch) < page_size:
                last_full_batch = False
                break
        if last_full_batch:
            logger.warning(
                "fmp_price_target: news pagination hit max_pages=%s for %s with a still-full "
                "final page; history may be truncated. Bump max_pages if this recurs.",
                max_pages,
                ticker,
            )
        return all_rows

    def fetch_ticker(self, ticker: str) -> pd.DataFrame:
        normalized_ticker = self.normalize_tickers([ticker])[0]
        rows: list[dict[str, Any]] = []

        snapshot_time = _ensure_utc(self._now_fn())
        snapshot_date = snapshot_time.astimezone(EASTERN).date()
        consensus_payload = self._request_consensus(normalized_ticker)
        if consensus_payload:
            target_price = _decimal_or_none(consensus_payload.get("targetConsensus"))
            if target_price is not None:
                rows.append(
                    {
                        "ticker": normalized_ticker,
                        "event_date": snapshot_date,
                        "knowledge_time": snapshot_time,
                        "analyst_firm": None,
                        "target_price": target_price,
                        "prior_target": None,
                        "price_when_published": None,
                        "is_consensus": True,
                    },
                )

        news_records = self._request_news_all(normalized_ticker)
        for record in news_records:
            published_raw = record.get("publishedDate")
            event_date = _parse_date(published_raw)
            # Prefer split-adjusted target; fallback to raw priceTarget.
            target_price = _decimal_or_none(record.get("adjPriceTarget"))
            if target_price is None:
                target_price = _decimal_or_none(record.get("priceTarget"))
            analyst_firm = _clean_text(record.get("analystCompany")) or _clean_text(record.get("analystName"))
            if event_date is None or target_price is None or analyst_firm is None:
                continue
            # Use EOD(event_date) for PIT consistency with the lag-rule gate
            # (which requires knowledge_time >= end_of_day_ET(event_date)).
            # Intraday publication precision adds no value for cross-sectional
            # alpha and would cause spurious lag-rule violations. Preserve the
            # actual publication timestamp as a separate column if ever needed.
            rows.append(
                {
                    "ticker": normalized_ticker,
                    "event_date": event_date,
                    "knowledge_time": _end_of_day_utc(event_date),
                    "analyst_firm": analyst_firm,
                    "target_price": target_price,
                    "prior_target": None,
                    "price_when_published": _decimal_or_none(record.get("priceWhenPosted")),
                    "is_consensus": False,
                },
            )

        frame = self.dataframe_or_empty(rows, PRICE_TARGET_COLUMNS)
        if not frame.empty:
            # Persistence PK is (ticker, event_date, analyst_firm) + is_consensus.
            # News endpoint can emit multiple same-day notes from the same firm.
            # Sort by knowledge_time ASC then drop_duplicates(keep="last") so the
            # most recent revision wins the upsert, not whichever arrived last
            # in HTTP response order.
            frame.sort_values(
                ["ticker", "event_date", "is_consensus", "analyst_firm", "knowledge_time"],
                inplace=True,
                na_position="first",
            )
            frame = frame.drop_duplicates(
                subset=["ticker", "event_date", "is_consensus", "analyst_firm"],
                keep="last",
            ).reset_index(drop=True)
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
            return pd.DataFrame(columns=PRICE_TARGET_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def health_check(self) -> bool:
        try:
            self.fetch_ticker("AAPL")
        except Exception as exc:
            logger.warning("fmp_price_target health check failed: {}", exc)
            return False
        return True

    @staticmethod
    def _persist(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        session_factory = get_session_factory()
        with session_factory() as session:
            for row in frame.itertuples(index=False):
                if bool(row.is_consensus):
                    existing_id = session.execute(
                        sa.select(PriceTargetEvent.id).where(
                            PriceTargetEvent.ticker == row.ticker,
                            PriceTargetEvent.event_date == row.event_date,
                            PriceTargetEvent.is_consensus.is_(True),
                            PriceTargetEvent.analyst_firm.is_(None),
                        ),
                    ).scalar_one_or_none()
                    if existing_id is None:
                        session.execute(
                            insert(PriceTargetEvent).values(
                                ticker=row.ticker,
                                event_date=row.event_date,
                                knowledge_time=row.knowledge_time,
                                analyst_firm=None,
                                target_price=row.target_price,
                                prior_target=row.prior_target,
                                price_when_published=row.price_when_published,
                                is_consensus=True,
                            ),
                        )
                    else:
                        session.execute(
                            sa.update(PriceTargetEvent)
                            .where(PriceTargetEvent.id == existing_id)
                            .values(
                                knowledge_time=row.knowledge_time,
                                target_price=row.target_price,
                                prior_target=row.prior_target,
                                price_when_published=row.price_when_published,
                                is_consensus=True,
                            ),
                        )
                    continue

                stmt = insert(PriceTargetEvent).values(
                    ticker=row.ticker,
                    event_date=row.event_date,
                    knowledge_time=row.knowledge_time,
                    analyst_firm=row.analyst_firm,
                    target_price=row.target_price,
                    prior_target=row.prior_target,
                    price_when_published=row.price_when_published,
                    is_consensus=False,
                )
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_target_event",
                    set_={
                        "knowledge_time": stmt.excluded.knowledge_time,
                        "target_price": stmt.excluded.target_price,
                        "prior_target": stmt.excluded.prior_target,
                        "price_when_published": stmt.excluded.price_when_published,
                        "is_consensus": stmt.excluded.is_consensus,
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


def _end_of_day_utc(day_value: date) -> datetime:
    local_dt = datetime.combine(day_value, time(hour=23, minute=59), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)




def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
