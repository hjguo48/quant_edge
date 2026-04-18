from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal
from typing import Any
import uuid
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
from loguru import logger
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import StockMinuteAggs
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource, DataSourceError, DataSourceTransientError
from src.data.sources.polygon import normalize_polygon_ticker, to_polygon_request_ticker

MINUTE_COLUMNS = [
    "ticker",
    "trade_date",
    "minute_ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "vwap",
    "transactions",
    "event_time",
    "knowledge_time",
    "batch_id",
]
EASTERN = ZoneInfo("America/New_York")
XNYS = xcals.get_calendar("XNYS")
REGULAR_SESSION_START = pd.Timestamp("09:30").time()
REGULAR_SESSION_END = pd.Timestamp("16:00").time()


class PolygonMinuteClient(DataSource):
    source_name = "polygon_minute"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.25,
    ) -> None:
        super().__init__(api_key or settings.POLYGON_API_KEY, min_request_interval=min_request_interval)
        self._http_session: Any | None = None

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        frames = [self.get_minute_aggs(ticker, start_date, end_date) for ticker in self.normalize_tickers(tickers)]
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=MINUTE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("Minute aggregates incremental fetch is not part of the Week 3.0 smoke path.")

    def health_check(self) -> bool:
        # Use last XNYS session (not date.today()) so weekends/holidays don't
        # produce false-negative "unhealthy" results.
        today = pd.Timestamp(date.today())
        last_session = (
            today
            if XNYS.is_session(today)
            else XNYS.previous_session(today)
        ).date()
        try:
            recent = self.get_minute_aggs("SPY", last_session, last_session)
        except Exception as exc:
            logger.warning("polygon_minute health check failed: {}", exc)
            return False
        return not recent.empty

    def get_minute_aggs(
        self,
        ticker: str,
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        if start > end:
            return pd.DataFrame(columns=MINUTE_COLUMNS)

        canonical_ticker = normalize_polygon_ticker(ticker)
        provider_ticker = to_polygon_request_ticker(canonical_ticker)
        payload_rows = self._fetch_range_results(provider_ticker, start, end)

        rows: list[dict[str, Any]] = []
        batch_id = str(uuid.uuid4())
        for payload in payload_rows:
            timestamp_ms = payload.get("t")
            if timestamp_ms is None:
                continue
            minute_ts_utc = pd.Timestamp(timestamp_ms, unit="ms", tz="UTC")
            if not self._is_regular_session_bar(minute_ts_utc):
                continue

            minute_ts_et = minute_ts_utc.tz_convert(EASTERN).to_pydatetime()
            session_label = pd.Timestamp(minute_ts_et.date())
            next_session = XNYS.next_session(session_label)
            knowledge_time = XNYS.session_close(next_session).to_pydatetime()
            rows.append(
                {
                    "ticker": canonical_ticker,
                    "trade_date": minute_ts_et.date(),
                    "minute_ts": minute_ts_et,
                    "open": payload.get("o"),
                    "high": payload.get("h"),
                    "low": payload.get("l"),
                    "close": payload.get("c"),
                    "volume": payload.get("v"),
                    "vwap": payload.get("vw"),
                    "transactions": payload.get("n"),
                    "event_time": minute_ts_et,
                    "knowledge_time": knowledge_time,
                    "batch_id": batch_id,
                },
            )

        frame = pd.DataFrame(rows, columns=MINUTE_COLUMNS)
        if frame.empty:
            return frame
        frame.sort_values(["ticker", "minute_ts"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
        return frame

    def persist_minute_aggs(self, frame: pd.DataFrame, *, batch_size: int = 10_000) -> int:
        if frame.empty:
            return 0

        prepared = frame.copy()
        if "batch_id" not in prepared.columns:
            prepared["batch_id"] = str(uuid.uuid4())
        prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
        prepared["trade_date"] = pd.to_datetime(prepared["trade_date"]).dt.date
        prepared["minute_ts"] = pd.to_datetime(prepared["minute_ts"], utc=True)
        prepared["event_time"] = pd.to_datetime(prepared["event_time"], utc=True)
        prepared["knowledge_time"] = pd.to_datetime(prepared["knowledge_time"], utc=True)
        prepared.drop_duplicates(["ticker", "minute_ts"], keep="last", inplace=True)
        records = [self._frame_row_to_record(row) for row in prepared.itertuples(index=False)]

        session_factory = get_session_factory()
        with session_factory() as session:
            try:
                for start in range(0, len(records), batch_size):
                    chunk = records[start : start + batch_size]
                    statement = insert(StockMinuteAggs).values(chunk)
                    upsert = statement.on_conflict_do_update(
                        index_elements=[StockMinuteAggs.ticker, StockMinuteAggs.minute_ts],
                        set_={
                            "trade_date": statement.excluded.trade_date,
                            "open": statement.excluded.open,
                            "high": statement.excluded.high,
                            "low": statement.excluded.low,
                            "close": statement.excluded.close,
                            "volume": statement.excluded.volume,
                            "vwap": statement.excluded.vwap,
                            "transactions": statement.excluded.transactions,
                            "event_time": statement.excluded.event_time,
                            "knowledge_time": sa.func.least(
                                sa.func.coalesce(
                                    StockMinuteAggs.knowledge_time,
                                    statement.excluded.knowledge_time,
                                ),
                                statement.excluded.knowledge_time,
                            ),
                            "batch_id": statement.excluded.batch_id,
                        },
                    )
                    session.execute(upsert)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.opt(exception=exc).error("polygon_minute failed to persist minute bars")
                raise DataSourceError("Failed to persist Polygon minute aggregates.") from exc

        return len(records)

    @DataSource.retryable()
    def _fetch_range_results(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        session = self._get_http_session()
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/minute/"
            f"{start_date.isoformat()}/{end_date.isoformat()}"
        )
        params: dict[str, Any] | None = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 50_000,
            "apiKey": self.api_key,
        }
        results: list[dict[str, Any]] = []

        while url:
            self._before_request(f"minute_aggs {ticker} {start_date}->{end_date}")
            try:
                response = session.get(url, params=params, timeout=30)
            except Exception as exc:
                raise DataSourceTransientError(
                    f"Polygon minute aggregates request failed for {ticker}: {exc}",
                ) from exc

            if response.status_code != 200:
                self.classify_http_error(
                    response.status_code,
                    response.text,
                    context=f"Polygon minute aggregates {ticker} {start_date}->{end_date}",
                )

            payload = response.json()
            page_results = payload.get("results") or []
            if not isinstance(page_results, list):
                raise DataSourceError(
                    f"Unexpected Polygon minute aggregates payload for {ticker}: results is "
                    f"{type(page_results).__name__}",
                )
            results.extend(page_results)

            next_url = payload.get("next_url")
            if not next_url:
                break
            url = str(next_url)
            params = {"apiKey": self.api_key}

        return results

    @staticmethod
    def _is_regular_session_bar(minute_ts_utc: pd.Timestamp) -> bool:
        # Polygon bar `t` is the START of a 1-minute window [t, t+1min).
        # Regular session bar ⇔ t ∈ [session_open, session_close).
        # Using XNYS.session_open/close instead of hardcoded 09:30/16:00 handles
        # early-close days (e.g. 13:00 ET close on day after Thanksgiving).
        minute_ts_et = minute_ts_utc.tz_convert(EASTERN)
        session_label = pd.Timestamp(minute_ts_et.date())
        if not XNYS.is_session(session_label):
            return False
        session_open = XNYS.session_open(session_label)
        session_close = XNYS.session_close(session_label)
        return session_open <= minute_ts_utc < session_close

    def _get_http_session(self) -> Any:
        if self._http_session is not None:
            return self._http_session

        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("requests is not installed.") from exc

        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": "QuantEdge/0.1.0"})
        self._http_session = session
        return session

    @staticmethod
    def _to_decimal(value: Any, *, places: int = 6) -> Decimal | None:
        if value is None or pd.isna(value):
            return None
        return Decimal(str(round(float(value), places)))

    def _frame_row_to_record(self, row: Any) -> dict[str, Any]:
        return {
            "ticker": str(row.ticker).upper(),
            "minute_ts": pd.Timestamp(row.minute_ts).to_pydatetime(),
            "trade_date": row.trade_date,
            "open": self._to_decimal(row.open),
            "high": self._to_decimal(row.high),
            "low": self._to_decimal(row.low),
            "close": self._to_decimal(row.close),
            "volume": int(row.volume) if row.volume is not None and not pd.isna(row.volume) else None,
            "vwap": self._to_decimal(row.vwap),
            "transactions": int(row.transactions)
            if row.transactions is not None and not pd.isna(row.transactions)
            else None,
            "event_time": pd.Timestamp(row.event_time).to_pydatetime(),
            "knowledge_time": pd.Timestamp(row.knowledge_time).to_pydatetime(),
            "batch_id": str(row.batch_id),
        }
