from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.session import get_engine, get_session_factory
from src.data.sources.base import DataSource, DataSourceError, DataSourceTransientError

DEFAULT_SERIES = ("VIXCLS", "DGS10", "DGS2", "BAA10Y", "AAA10Y", "FEDFUNDS")
SERIES_ALIASES = {
    "VIX": "VIXCLS",
    "10Y": "DGS10",
    "UST10Y": "DGS10",
    "CREDIT_SPREAD": "BAA10Y",
    "BAA_10Y": "BAA10Y",
    "AAA_10Y": "AAA10Y",
    "FFR": "FEDFUNDS",
}

MACRO_SERIES_TABLE = sa.Table(
    "macro_series_pit",
    sa.MetaData(),
    sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
    sa.Column("series_id", sa.String(20), nullable=False),
    sa.Column("observation_date", sa.Date, nullable=False),
    sa.Column("value", sa.Numeric(20, 6)),
    sa.Column("knowledge_time", sa.DateTime(timezone=True), nullable=False),
    sa.Column("is_revision", sa.Boolean, nullable=False, server_default=sa.text("FALSE")),
    sa.Column("source", sa.String(20)),
    sa.UniqueConstraint(
        "series_id",
        "observation_date",
        "knowledge_time",
        name="uq_macro_series_pit_version",
    ),
    sa.Index(
        "idx_macro_series_pit_lookup",
        "series_id",
        "knowledge_time",
        "observation_date",
    ),
)

MACRO_COLUMNS = [
    "series_id",
    "observation_date",
    "value",
    "knowledge_time",
    "is_revision",
    "source",
]


class _HttpFredClient:
    root_url = "https://api.stlouisfed.org/fred/series/observations"
    page_size = 1_000

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_series_all_releases(
        self,
        series_id: str,
        *,
        realtime_start: str,
        realtime_end: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0

        while True:
            params = {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "sort_order": "asc",
                "realtime_start": realtime_start,
                "realtime_end": realtime_end,
                "limit": self.page_size,
                "offset": offset,
            }
            request_url = f"{self.root_url}?{urlencode(params)}"
            try:
                with urlopen(request_url, timeout=30) as response:  # noqa: S310
                    payload = json.load(response)
            except HTTPError as exc:
                response_text = exc.read().decode("utf-8", errors="replace")
                DataSource.classify_http_error(exc.code, response_text, context=f"FRED releases request for {series_id}")
                raise
            except URLError as exc:
                raise DataSourceTransientError(
                    f"FRED releases request failed for {series_id}: {exc}",
                ) from exc

            observations = list(payload.get("observations") or [])
            rows.extend(observations)

            total = int(payload.get("count") or len(observations))
            if not observations or len(rows) >= total:
                break
            offset += len(observations)

        return rows


class FredDataSource(DataSource):
    source_name = "fred"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.10,
    ) -> None:
        super().__init__(api_key or settings.FRED_API_KEY, min_request_interval=min_request_interval)
        self._client: Any | None = None
        self._ensure_table_exists()

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = min(self.coerce_date(end_date), self._current_realtime_date())
        rows: list[dict[str, Any]] = []

        for series_id in self._resolve_series_ids(tickers):
            releases = self._fetch_releases(series_id, start, end)
            if releases.empty:
                logger.warning("fred returned no observations for {}", series_id)
                continue

            filtered = releases.loc[
                (releases["observation_date"] >= start) & (releases["observation_date"] <= end)
            ].copy()
            if filtered.empty:
                continue

            filtered.sort_values(["observation_date", "realtime_start"], inplace=True)
            filtered["knowledge_time"] = filtered["realtime_start"].map(self._knowledge_time)
            filtered["is_revision"] = filtered.groupby("observation_date").cumcount() > 0
            filtered["series_id"] = series_id
            filtered["source"] = self.source_name
            rows.extend(filtered[MACRO_COLUMNS].to_dict("records"))

        frame = self.dataframe_or_empty(rows, MACRO_COLUMNS)
        if not frame.empty:
            self.persist_series(frame)
        logger.info(
            "fred fetched {} rows across {} series between {} and {}",
            len(frame),
            len(set(frame["series_id"])) if not frame.empty else 0,
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
        frame = self.fetch_historical(tickers, lookback_start, self._current_realtime_date())
        if frame.empty:
            return frame
        incremental = frame.loc[pd.to_datetime(frame["knowledge_time"], utc=True) >= cutoff]
        return incremental.reset_index(drop=True)

    def health_check(self) -> bool:
        try:
            end = self._current_realtime_date()
            releases = self._fetch_releases("FEDFUNDS", end - timedelta(days=30), end)
            return not releases.empty
        except Exception as exc:
            logger.warning("fred health check failed: {}", exc)
            return False

    @DataSource.retryable()
    def _fetch_releases(
        self,
        series_id: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        client = self._get_client()
        self._before_request(f"all releases for {series_id} between {start_date} and {end_date}")

        frames: list[pd.DataFrame] = []
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=1_500), end_date)
            try:
                releases = client.get_series_all_releases(
                    series_id,
                    realtime_start=chunk_start.isoformat(),
                    realtime_end=chunk_end.isoformat(),
                )
            except DataSourceError:
                raise
            except Exception as exc:
                raise DataSourceTransientError(
                    f"FRED series release request failed for {series_id}: {exc}",
                ) from exc

            frame = pd.DataFrame(releases)
            if not frame.empty:
                frames.append(frame)
            chunk_start = chunk_end + timedelta(days=1)

        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            return pd.DataFrame(columns=["observation_date", "realtime_start", "value"])

        frame.rename(columns={"date": "observation_date"}, inplace=True)
        frame["observation_date"] = pd.to_datetime(frame["observation_date"], errors="coerce").dt.date
        frame["realtime_start"] = frame["realtime_start"].map(self._parse_release_timestamp)
        frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
        frame.dropna(subset=["observation_date", "realtime_start", "value"], inplace=True)
        frame.drop_duplicates(subset=["observation_date", "realtime_start", "value"], inplace=True)
        return frame[["observation_date", "realtime_start", "value"]]

    def persist_series(self, frame: pd.DataFrame, *, batch_size: int = 1_000) -> int:
        if frame.empty:
            return 0

        records = [self._frame_row_to_record(row) for row in frame.itertuples(index=False)]
        session_factory = get_session_factory()

        with session_factory() as session:
            try:
                for index in range(0, len(records), batch_size):
                    chunk = records[index : index + batch_size]
                    statement = insert(MACRO_SERIES_TABLE).values(chunk)
                    upsert = statement.on_conflict_do_update(
                        constraint="uq_macro_series_pit_version",
                        set_={
                            "value": statement.excluded.value,
                            "is_revision": statement.excluded.is_revision,
                            "source": statement.excluded.source,
                        },
                    )
                    session.execute(upsert)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.opt(exception=exc).error("fred failed to persist macro rows")
                raise DataSourceError("Failed to persist FRED macro series data.") from exc

        return len(records)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        self._require_api_key()

        try:
            from fredapi import Fred
        except ImportError as exc:
            logger.info("fredapi is not installed; using HTTP fallback client for FRED releases")
            self._client = _HttpFredClient(self.api_key)
            return self._client

        self._client = Fred(api_key=self.api_key)
        return self._client

    def _ensure_table_exists(self) -> None:
        MACRO_SERIES_TABLE.metadata.create_all(
            get_engine(),
            tables=[MACRO_SERIES_TABLE],
            checkfirst=True,
        )

    @staticmethod
    def _resolve_series_ids(series_ids: Sequence[str]) -> tuple[str, ...]:
        raw_ids = series_ids or DEFAULT_SERIES
        resolved = []
        for series_id in raw_ids:
            normalized = series_id.strip().upper()
            resolved.append(SERIES_ALIASES.get(normalized, normalized))
        return tuple(dict.fromkeys(resolved))

    @staticmethod
    def _parse_release_timestamp(value: Any) -> datetime | None:
        timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            logger.warning("fred skipped malformed realtime_start value: {}", value)
            return None
        if timestamp.tzinfo is None:
            return timestamp.to_pydatetime().replace(tzinfo=timezone.utc)
        return timestamp.tz_convert(timezone.utc).to_pydatetime()

    @staticmethod
    def _knowledge_time(release_time: datetime) -> datetime:
        if release_time.tzinfo is None:
            release_time = release_time.replace(tzinfo=timezone.utc)
        next_day = (release_time + timedelta(days=1)).date()
        return datetime.combine(next_day, time.min, tzinfo=timezone.utc)

    @staticmethod
    def _current_realtime_date() -> date:
        return datetime.now(ZoneInfo("America/New_York")).date()

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        if value is None or pd.isna(value):
            return None
        return Decimal(str(round(float(value), 6)))

    def _frame_row_to_record(self, row: Any) -> dict[str, Any]:
        return {
            "series_id": row.series_id,
            "observation_date": row.observation_date,
            "value": self._to_decimal(row.value),
            "knowledge_time": self.coerce_datetime(row.knowledge_time),
            "is_revision": bool(row.is_revision),
            "source": row.source,
        }
