"""FINRA daily short-sale volume source."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import csv
from datetime import date, datetime, time, timezone
import io
from typing import Any, Literal
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
from loguru import logger
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.data.db.models import Base
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource, DataSourceError, DataSourceTransientError, RetryConfig

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
XNYS = xcals.get_calendar("XNYS")
VALID_MARKETS = ("CNMS", "ADF", "BNY")
FINRA_USER_AGENT = "QuantEdge/1.0 (research; contact: hjguo48@gmail.com)"
FINRA_COLUMNS = [
    "ticker",
    "trade_date",
    "knowledge_time",
    "market",
    "short_volume",
    "short_exempt_volume",
    "total_volume",
]
PERSISTED_FINRA_COLUMNS = [
    "ticker",
    "trade_date",
    "market",
    "knowledge_time",
    "short_volume",
    "short_exempt_volume",
    "total_volume",
    "file_etag",
]


class ShortSaleVolume(Base):
    __tablename__ = "short_sale_volume_daily"
    __table_args__ = (
        sa.Index("ix_short_sale_kt", "ticker", "knowledge_time"),
    )

    ticker: Mapped[str] = mapped_column(sa.String(16), primary_key=True)
    trade_date: Mapped[date] = mapped_column(sa.Date, primary_key=True)
    market: Mapped[str] = mapped_column(sa.String(16), primary_key=True)
    knowledge_time: Mapped[datetime] = mapped_column(sa.DateTime(timezone=True), nullable=False)
    short_volume: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    short_exempt_volume: Mapped[int | None] = mapped_column(sa.BigInteger)
    total_volume: Mapped[int] = mapped_column(sa.BigInteger, nullable=False)
    file_etag: Mapped[str | None] = mapped_column(sa.String(64))


class FINRAShortSaleSource(DataSource):
    """FINRA daily short-sale volume public files."""

    source_name = "finra_short_sale"
    base_url = "https://cdn.finra.org/equity/regsho/daily"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.0,
        retry_config: RetryConfig | None = None,
        http_session: Any | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__(
            api_key or "public",
            min_request_interval=min_request_interval,
            retry_config=retry_config or RetryConfig(max_attempts=5),
        )
        self._http_session = http_session
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))

    def _require_api_key(self) -> None:
        # FINRA daily short-sale files are public; keep the retry/throttle
        # contract from DataSource, but do not require a secret.
        return None

    def fetch_day(
        self,
        trade_date: date,
        market: Literal["CNMS", "ADF", "BNY"],
    ) -> tuple[pd.DataFrame, str | None]:
        normalized_market = _normalize_market(market)
        raw_text, etag = self._download_day_text(trade_date, normalized_market)
        if not raw_text:
            return pd.DataFrame(columns=FINRA_COLUMNS), etag
        frame = self._parse_day_text(raw_text, trade_date=trade_date, market=normalized_market)
        return frame, etag

    def fetch_historical(
        self,
        start_date: date,
        end_date: date,
        markets: Sequence[str] = VALID_MARKETS,
        session_factory: Callable | None = None,
        force_refetch: bool = False,
    ) -> int:
        total_rows, _ = self._fetch_and_persist_range(
            start_date=start_date,
            end_date=end_date,
            markets=markets,
            session_factory=session_factory,
            force_refetch=force_refetch,
            collect_frames=False,
        )
        return total_rows

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        if tickers:
            logger.warning("FINRA files are market-wide, 'tickers' filter ignored; returning all rows")
        start = self.coerce_date(since_date)
        end = date.today()
        inserted, frames = self._fetch_and_persist_range(
            start_date=start,
            end_date=end,
            markets=VALID_MARKETS,
            session_factory=None,
            force_refetch=False,
            collect_frames=True,
        )
        logger.info("finra_short_sale incremental fetched {} rows from {} to {}", inserted, start, end)
        if not frames:
            return pd.DataFrame(columns=PERSISTED_FINRA_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        combined = combined[PERSISTED_FINRA_COLUMNS]
        combined.sort_values(["trade_date", "market", "ticker"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined

    def health_check(self) -> bool:
        today = pd.Timestamp(date.today())
        last_session = today if XNYS.is_session(today) else XNYS.previous_session(today)
        try:
            self._head_day(last_session.date(), "CNMS")
        except Exception as exc:
            logger.warning("finra_short_sale health check failed: {}", exc)
            return False
        return True

    @DataSource.retryable()
    def _head_day(self, trade_date: date, market: str) -> tuple[bool, str | None]:
        url = self._build_url(trade_date, market)
        self._before_request(f"FINRA HEAD {market} {trade_date}")
        session = self._get_http_session()
        try:
            response = session.head(url, timeout=30)
        except Exception as exc:
            raise DataSourceTransientError(f"FINRA HEAD failed for {market} {trade_date}: {exc}") from exc
        if response.status_code == 404:
            return False, None
        if not response.ok:
            self.classify_http_error(response.status_code, getattr(response, "text", ""), context=f"FINRA HEAD {url}")
        return True, _extract_etag(response)

    @DataSource.retryable()
    def _download_day_text(self, trade_date: date, market: str) -> tuple[str, str | None]:
        url = self._build_url(trade_date, market)
        self._before_request(f"FINRA GET {market} {trade_date}")
        session = self._get_http_session()
        try:
            response = session.get(url, timeout=30)
        except Exception as exc:
            raise DataSourceTransientError(f"FINRA GET failed for {market} {trade_date}: {exc}") from exc
        if response.status_code == 404:
            return "", None
        if not response.ok:
            self.classify_http_error(response.status_code, getattr(response, "text", ""), context=f"FINRA GET {url}")
        return getattr(response, "text", ""), _extract_etag(response)

    def _get_http_session(self) -> Any:
        if self._http_session is not None:
            return self._http_session
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("requests is not installed.") from exc
        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": FINRA_USER_AGENT})
        self._http_session = session
        return session

    def _fetch_and_persist_range(
        self,
        *,
        start_date: date,
        end_date: date,
        markets: Sequence[str],
        session_factory: Callable | None,
        force_refetch: bool,
        collect_frames: bool,
    ) -> tuple[int, list[pd.DataFrame]]:
        if start_date > end_date:
            return 0, []

        total_rows = 0
        collected_frames: list[pd.DataFrame] = []
        active_session_factory = session_factory or get_session_factory()
        sessions = XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
        for session_label in sessions:
            trade_day = session_label.date()
            for market in markets:
                normalized_market = _normalize_market(market)
                cached_etag = self._get_cached_etag(
                    trade_date=trade_day,
                    market=normalized_market,
                    session_factory=active_session_factory,
                )
                remote_etag: str | None = None
                if not force_refetch:
                    exists, remote_etag = self._head_day(trade_day, normalized_market)
                    if not exists:
                        continue
                    if cached_etag and remote_etag and cached_etag == remote_etag:
                        logger.info(
                            "finra_short_sale skipping {} {} because cached ETag matches {}",
                            trade_day,
                            normalized_market,
                            remote_etag,
                        )
                        continue

                frame, etag = self.fetch_day(trade_day, normalized_market)
                if frame.empty:
                    continue

                effective_etag = etag or remote_etag or cached_etag
                revision_knowledge_time = None
                if cached_etag and effective_etag and cached_etag != effective_etag:
                    revision_knowledge_time = _ensure_utc(self._now_fn())

                total_rows += self._persist(
                    frame,
                    file_etag=effective_etag,
                    session_factory=active_session_factory,
                    revision_knowledge_time=revision_knowledge_time,
                )
                if collect_frames:
                    persisted_frame = frame.copy()
                    persisted_frame["file_etag"] = effective_etag
                    collected_frames.append(persisted_frame[PERSISTED_FINRA_COLUMNS])
        return total_rows, collected_frames

    @staticmethod
    def _build_url(trade_date: date, market: str) -> str:
        return f"{FINRAShortSaleSource.base_url}/{market}shvol{trade_date:%Y%m%d}.txt"

    @staticmethod
    def _parse_day_text(raw_text: str, *, trade_date: date, market: str) -> pd.DataFrame:
        reader = csv.reader(io.StringIO(raw_text), delimiter="|")
        try:
            header = next(reader)
        except StopIteration:
            return pd.DataFrame(columns=FINRA_COLUMNS)

        normalized_header = [str(value).strip() for value in header]
        required = {"Date", "Symbol", "ShortVolume", "TotalVolume"}
        if not required.issubset(normalized_header):
            raise DataSourceError(
                "FINRA short-sale file is missing required columns: "
                f"{', '.join(sorted(required - set(normalized_header)))}",
            )

        rows: list[dict[str, Any]] = []
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(str(cell).strip() for cell in row):
                continue
            if len(row) != len(normalized_header):
                logger.warning(
                    "finra_short_sale skipping malformed line {} for {} {}: expected {} columns, got {}",
                    line_number,
                    market,
                    trade_date,
                    len(normalized_header),
                    len(row),
                )
                continue
            record = dict(zip(normalized_header, row, strict=False))
            ticker = str(record.get("Symbol") or "").strip().upper()
            if not ticker:
                logger.warning(
                    "finra_short_sale skipping line {} for {} {} because Symbol is blank",
                    line_number,
                    market,
                    trade_date,
                )
                continue
            try:
                short_volume = _parse_int(record.get("ShortVolume"), field="ShortVolume", line_number=line_number)
                total_volume = _parse_int(record.get("TotalVolume"), field="TotalVolume", line_number=line_number)
            except ValueError as exc:
                logger.warning("finra_short_sale skipping malformed line {}: {}", line_number, exc)
                continue
            short_exempt_volume = None
            if "ShortExemptVolume" in record:
                raw_short_exempt = str(record.get("ShortExemptVolume") or "").strip()
                if raw_short_exempt:
                    try:
                        short_exempt_volume = _parse_int(
                            raw_short_exempt,
                            field="ShortExemptVolume",
                            line_number=line_number,
                        )
                    except ValueError as exc:
                        logger.warning(
                            "finra_short_sale skipping malformed line {}: {}",
                            line_number,
                            exc,
                        )
                        continue

            row_market = str(record.get("Market") or "").strip().upper() or market
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_date,
                    "knowledge_time": _knowledge_time(trade_date),
                    "market": row_market,
                    "short_volume": short_volume,
                    "short_exempt_volume": short_exempt_volume,
                    "total_volume": total_volume,
                },
            )

        return pd.DataFrame(rows, columns=FINRA_COLUMNS)

    @staticmethod
    def _get_cached_etag(
        *,
        trade_date: date,
        market: str,
        session_factory: Callable,
    ) -> str | None:
        with session_factory() as session:
            etag = session.execute(
                sa.select(ShortSaleVolume.file_etag)
                .where(
                    ShortSaleVolume.trade_date == trade_date,
                    ShortSaleVolume.market == market,
                )
                .limit(1),
            ).scalar_one_or_none()
        if etag is None:
            return None
        cleaned = str(etag).strip()
        return cleaned or None

    @staticmethod
    def _persist(
        frame: pd.DataFrame,
        *,
        file_etag: str | None,
        session_factory: Callable,
        revision_knowledge_time: datetime | None = None,
    ) -> int:
        if frame.empty:
            return 0

        prepared = frame.copy()
        prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
        records = []
        for row in prepared.to_dict(orient="records"):
            records.append(
                {
                    "ticker": row["ticker"],
                    "trade_date": row["trade_date"],
                    "knowledge_time": row["knowledge_time"],
                    "market": row["market"],
                    "short_volume": int(row["short_volume"]),
                    "short_exempt_volume": int(row["short_exempt_volume"]) if pd.notna(row["short_exempt_volume"]) else None,
                    "total_volume": int(row["total_volume"]),
                    "file_etag": file_etag,
                },
            )

        statement = insert(ShortSaleVolume).values(records)
        updated_knowledge_time: Any = statement.excluded.knowledge_time
        if revision_knowledge_time is not None:
            updated_knowledge_time = sa.func.greatest(
                ShortSaleVolume.knowledge_time + sa.text("interval '1 second'"),
                sa.literal(revision_knowledge_time, type_=sa.DateTime(timezone=True)),
            )
        upsert = statement.on_conflict_do_update(
            index_elements=[
                ShortSaleVolume.ticker,
                ShortSaleVolume.trade_date,
                ShortSaleVolume.market,
            ],
            set_={
                "knowledge_time": updated_knowledge_time,
                "short_volume": statement.excluded.short_volume,
                "short_exempt_volume": statement.excluded.short_exempt_volume,
                "total_volume": statement.excluded.total_volume,
                "file_etag": statement.excluded.file_etag,
            },
        )
        with session_factory() as session:
            session.execute(upsert)
            session.commit()
        return len(records)


def _normalize_market(market: str) -> str:
    normalized = str(market).strip().upper()
    if normalized not in VALID_MARKETS:
        raise ValueError(f"Unsupported FINRA market '{market}'. Expected one of {VALID_MARKETS}.")
    return normalized


def _knowledge_time(trade_date: date) -> datetime:
    local_dt = datetime.combine(trade_date, time(hour=18, minute=0), tzinfo=EASTERN)
    return local_dt.astimezone(timezone.utc)


def _extract_etag(response: Any) -> str | None:
    headers = getattr(response, "headers", {}) or {}
    etag = headers.get("ETag") or headers.get("Etag")
    if etag is None:
        return None
    cleaned = str(etag).strip().strip('"')
    return cleaned or None


def _parse_int(raw_value: object, *, field: str, line_number: int) -> int:
    text = str(raw_value).strip()
    if not text:
        raise ValueError(f"{field} is blank on line {line_number}")
    try:
        return int(float(text))
    except Exception as exc:
        raise ValueError(f"{field}='{text}' is not numeric on line {line_number}") from exc


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
