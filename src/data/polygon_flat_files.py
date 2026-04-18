from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from hashlib import md5
import gzip
import io
from typing import Any

import pandas as pd
from loguru import logger

from src.config import settings
from src.data.polygon_minute import EASTERN, MINUTE_COLUMNS, XNYS
from src.data.sources.base import DataSource, DataSourceAuthError, DataSourceError, DataSourceTransientError
from src.data.sources.polygon import normalize_polygon_ticker

DEFAULT_BUCKET = "flatfiles"
DEFAULT_PREFIX = "us_stocks_sip/minute_aggs_v1"


@dataclass(frozen=True)
class FlatFileHeader:
    source_file: str
    content_length: int
    etag: str | None


@dataclass(frozen=True)
class FlatFileLoadResult:
    source_file: str
    checksum_md5: str
    rows_raw: int
    rows_kept: int
    tickers_loaded: int
    frame: pd.DataFrame


class PolygonFlatFilesClient(DataSource):
    source_name = "polygon_flat_files"

    def __init__(
        self,
        access_key: str | None = None,
        secret_key: str | None = None,
        *,
        endpoint_url: str | None = None,
        bucket: str | None = None,
        prefix: str = DEFAULT_PREFIX,
        min_request_interval: float = 0.0,
        s3_client: Any | None = None,
    ) -> None:
        super().__init__(access_key or settings.POLYGON_S3_KEY, min_request_interval=min_request_interval)
        self.secret_key = secret_key or settings.POLYGON_S3_SECRET
        self.endpoint_url = endpoint_url or settings.POLYGON_S3_ENDPOINT
        self.bucket = bucket or settings.POLYGON_S3_BUCKET or DEFAULT_BUCKET
        self.prefix = prefix.strip("/")
        self._s3_client = s3_client

    def _require_api_key(self) -> None:
        if not self.api_key or not self.secret_key:
            raise DataSourceAuthError(
                "Polygon flat files require POLYGON_S3_KEY and POLYGON_S3_SECRET from the Massive dashboard.",
            )

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        normalized = set(self.normalize_tickers(tickers))
        current = self.coerce_date(start_date)
        final = self.coerce_date(end_date)
        while current <= final:
            if not XNYS.is_session(pd.Timestamp(current)):
                current += timedelta(days=1)
                continue
            result = self.load_day(current, universe_tickers=normalized)
            if not result.frame.empty:
                frames.append(result.frame)
            current += timedelta(days=1)
        if not frames:
            return pd.DataFrame(columns=MINUTE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("Minute flat-file incremental fetch is orchestrated by run_minute_backfill.py.")

    def health_check(self) -> bool:
        today = pd.Timestamp(date.today())
        probe_day = today if XNYS.is_session(today) else XNYS.date_to_session(today, direction="previous")
        try:
            self.head_day(probe_day.date())
        except Exception as exc:
            logger.warning("polygon_flat_files health check failed: {}", exc)
            return False
        return True

    def build_s3_key(self, trading_date: date | datetime) -> str:
        trade_day = self.coerce_date(trading_date)
        return f"{self.prefix}/{trade_day:%Y/%m/%Y-%m-%d}.csv.gz"

    @DataSource.retryable()
    def head_day(self, trading_date: date | datetime) -> FlatFileHeader:
        trade_day = self.coerce_date(trading_date)
        key = self.build_s3_key(trade_day)
        self._before_request(f"head flat file {key}")
        try:
            response = self._get_s3_client().head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            self._classify_s3_error(exc, context=f"Polygon flat-file head {key}")
        return FlatFileHeader(
            source_file=f"s3://{self.bucket}/{key}",
            content_length=int(response.get("ContentLength") or 0),
            etag=str(response.get("ETag") or "").strip('"') or None,
        )

    @DataSource.retryable()
    def download_day_bytes(self, trading_date: date | datetime) -> tuple[bytes, FlatFileHeader]:
        trade_day = self.coerce_date(trading_date)
        key = self.build_s3_key(trade_day)
        self._before_request(f"download flat file {key}")
        try:
            response = self._get_s3_client().get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            self._classify_s3_error(exc, context=f"Polygon flat-file download {key}")
        body = response["Body"].read()
        header = FlatFileHeader(
            source_file=f"s3://{self.bucket}/{key}",
            content_length=int(response.get("ContentLength") or len(body)),
            etag=str(response.get("ETag") or "").strip('"') or None,
        )
        return body, header

    def sample_day(
        self,
        trading_date: date | datetime,
        *,
        universe_tickers: Iterable[str] | None = None,
        sample_rows: int = 1_000,
    ) -> FlatFileLoadResult:
        payload, header = self.download_day_bytes(trading_date)
        return self.parse_day_bytes(
            payload,
            trading_date=trading_date,
            source_file=header.source_file,
            universe_tickers=universe_tickers,
            nrows=sample_rows,
        )

    def load_day(
        self,
        trading_date: date | datetime,
        *,
        universe_tickers: Iterable[str] | None = None,
    ) -> FlatFileLoadResult:
        payload, header = self.download_day_bytes(trading_date)
        return self.parse_day_bytes(
            payload,
            trading_date=trading_date,
            source_file=header.source_file,
            universe_tickers=universe_tickers,
        )

    def parse_day_bytes(
        self,
        payload: bytes,
        *,
        trading_date: date | datetime,
        source_file: str,
        universe_tickers: Iterable[str] | None = None,
        nrows: int | None = None,
    ) -> FlatFileLoadResult:
        trade_day = self.coerce_date(trading_date)
        checksum = md5(payload).hexdigest()
        universe_set = {normalize_polygon_ticker(ticker) for ticker in universe_tickers} if universe_tickers is not None else None

        with gzip.GzipFile(fileobj=io.BytesIO(payload)) as gz:
            raw = pd.read_csv(gz, nrows=nrows)
        if raw.empty:
            return FlatFileLoadResult(
                source_file=source_file,
                checksum_md5=checksum,
                rows_raw=0,
                rows_kept=0,
                tickers_loaded=0,
                frame=pd.DataFrame(columns=MINUTE_COLUMNS),
            )

        raw.columns = [str(column).strip().lower() for column in raw.columns]
        required = {"ticker", "window_start", "open", "high", "low", "close", "volume"}
        missing = sorted(required - set(raw.columns))
        if missing:
            raise DataSourceError(
                f"Polygon flat file {source_file} is missing expected columns: {', '.join(missing)}",
            )

        raw["ticker"] = raw["ticker"].astype(str).map(normalize_polygon_ticker)
        rows_raw = int(len(raw))
        if universe_set is not None:
            raw = raw.loc[raw["ticker"].isin(universe_set)].copy()
        if "otc" in raw.columns:
            raw = raw.loc[~raw["otc"].fillna(False).astype(bool)].copy()
        if raw.empty:
            return FlatFileLoadResult(
                source_file=source_file,
                checksum_md5=checksum,
                rows_raw=rows_raw,
                rows_kept=0,
                tickers_loaded=0,
                frame=pd.DataFrame(columns=MINUTE_COLUMNS),
            )

        raw["minute_ts_utc"] = self._parse_window_start(raw["window_start"])
        raw = raw.loc[raw["minute_ts_utc"].notna()].copy()
        raw["minute_ts_et"] = raw["minute_ts_utc"].dt.tz_convert(EASTERN)
        raw["trade_date"] = raw["minute_ts_et"].dt.date
        raw = raw.loc[raw["trade_date"] == trade_day].copy()
        raw = raw.loc[self._regular_session_mask(raw["minute_ts_et"], raw["minute_ts_utc"])].copy()
        if raw.empty:
            return FlatFileLoadResult(
                source_file=source_file,
                checksum_md5=checksum,
                rows_raw=rows_raw,
                rows_kept=0,
                tickers_loaded=0,
                frame=pd.DataFrame(columns=MINUTE_COLUMNS),
            )

        raw.sort_values(["ticker", "minute_ts_utc"], inplace=True)
        raw["knowledge_time"] = raw["trade_date"].map(self._next_session_close)
        frame = pd.DataFrame(
            {
                "ticker": raw["ticker"].astype(str).str.upper(),
                "trade_date": raw["trade_date"],
                "minute_ts": raw["minute_ts_et"],
                "open": pd.to_numeric(raw["open"], errors="coerce"),
                "high": pd.to_numeric(raw["high"], errors="coerce"),
                "low": pd.to_numeric(raw["low"], errors="coerce"),
                "close": pd.to_numeric(raw["close"], errors="coerce"),
                "volume": pd.to_numeric(raw["volume"], errors="coerce"),
                "vwap": pd.to_numeric(raw["vwap"], errors="coerce") if "vwap" in raw.columns else pd.NA,
                "transactions": pd.to_numeric(raw["transactions"], errors="coerce") if "transactions" in raw.columns else pd.NA,
                "event_time": raw["minute_ts_et"],
                "knowledge_time": raw["knowledge_time"],
                "batch_id": "",
            },
            columns=MINUTE_COLUMNS,
        )
        return FlatFileLoadResult(
            source_file=source_file,
            checksum_md5=checksum,
            rows_raw=rows_raw,
            rows_kept=int(len(frame)),
            tickers_loaded=int(frame["ticker"].nunique()),
            frame=frame,
        )

    def _get_s3_client(self) -> Any:
        if self._s3_client is not None:
            return self._s3_client

        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("boto3 is not installed.") from exc

        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.api_key,
            aws_secret_access_key=self.secret_key,
            endpoint_url=self.endpoint_url,
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
        )
        return self._s3_client

    @staticmethod
    def _parse_window_start(series: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.dropna().empty:
            return pd.to_datetime(series, utc=True, errors="coerce")
        unit = "ns" if float(numeric.dropna().abs().max()) >= 10**15 else "ms"
        return pd.to_datetime(numeric, unit=unit, utc=True, errors="coerce")

    @staticmethod
    def _regular_session_mask(minute_ts_et: pd.Series, minute_ts_utc: pd.Series) -> pd.Series:
        # Polygon bar `t` = START of minute window [t, t+1min).
        # Regular ⇔ t ∈ [session_open, session_close). Both bounds are needed:
        # - session_open excludes pre-market bars (04:00-09:29 ET)
        # - session_close excludes post-close bars (16:00+ ET, or early-close 13:00+)
        # Consistent with polygon_minute._is_regular_session_bar.
        dates_et = minute_ts_et.dt.date
        session_open_map: dict = {}
        session_close_map: dict = {}
        for day in dates_et.unique():
            ts = pd.Timestamp(day)
            if XNYS.is_session(ts):
                session_open_map[day] = pd.Timestamp(XNYS.session_open(ts))
                session_close_map[day] = pd.Timestamp(XNYS.session_close(ts))
            else:
                session_open_map[day] = pd.NaT
                session_close_map[day] = pd.NaT
        session_opens = dates_et.map(session_open_map)
        session_closes = dates_et.map(session_close_map)
        return (
            session_closes.notna()
            & (minute_ts_utc >= session_opens)
            & (minute_ts_utc < session_closes)
        )

    @staticmethod
    def _next_session_close(trade_day: date) -> datetime:
        session_label = pd.Timestamp(trade_day)
        next_session = XNYS.next_session(session_label)
        return XNYS.session_close(next_session).to_pydatetime()

    @staticmethod
    def _classify_s3_error(exc: Exception, *, context: str) -> None:
        message = str(exc)
        response = getattr(exc, "response", None)
        status_code = None
        if isinstance(response, dict):
            status_code = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if status_code in {401, 403} or "AccessDenied" in message or "InvalidAccessKeyId" in message:
            raise DataSourceAuthError(f"{context} is not authorized: {message}") from exc
        if status_code == 404 or "NoSuchKey" in message:
            raise DataSourceError(f"{context} is not available: {message}") from exc
        raise DataSourceTransientError(f"{context} failed: {message}") from exc
