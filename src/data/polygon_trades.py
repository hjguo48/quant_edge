"""Polygon/Massive trades helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from loguru import logger
import pandas as pd

from src.config import settings
from src.data.sources.base import DataSource, DataSourceError, DataSourceTransientError, RetryConfig
from src.data.sources.polygon import normalize_polygon_ticker, to_polygon_request_ticker

DEFAULT_PAGE_SIZE = 50_000
POLYGON_TRADES_BASE_URL = "https://api.polygon.io"


def stable_sequence_fallback(
    trade_id: str | None,
    price: Decimal,
    size: Decimal,
    participant_timestamp_ns: int | None,
    conditions: tuple[int, ...],
) -> int:
    key = "|".join(
        [
            trade_id or "",
            f"{price:.6f}",
            f"{size:.4f}",
            str(participant_timestamp_ns or 0),
            ",".join(str(condition) for condition in conditions),
        ],
    )
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    int64_from_first_8 = int.from_bytes(digest[:8], "big", signed=False) & ((1 << 62) - 1)
    return -(int64_from_first_8 + 1)


@dataclass(frozen=True)
class TradeRecord:
    ticker: str
    trading_date: date
    sip_timestamp: datetime
    participant_timestamp: datetime | None
    trf_timestamp: datetime | None
    price: Decimal
    size: Decimal
    decimal_size: Decimal | None
    exchange: int
    tape: int | None
    conditions: list[int]
    correction: int | None
    sequence_number: int
    trade_id: str | None
    trf_id: str | None


class PolygonTradesClient(DataSource):
    """Polygon v3 trades REST client for targeted Week 4 sampling."""

    source_name = "polygon_trades"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.05,
        retry_config: RetryConfig | None = None,
        http_session: Any | None = None,
    ) -> None:
        super().__init__(
            api_key or settings.POLYGON_API_KEY,
            min_request_interval=min_request_interval,
            retry_config=retry_config
            or RetryConfig(
                max_attempts=3,
                initial_delay=1.0,
                backoff_factor=2.0,
                max_delay=30.0,
                jitter=0.2,
            ),
        )
        self._http_session = http_session

    def fetch_trades_for_day(
        self,
        ticker: str,
        trading_date: date,
        *,
        page_size: int | None = None,
        max_pages: int | None = None,
    ) -> Iterator[TradeRecord]:
        canonical_ticker = normalize_polygon_ticker(ticker)
        provider_ticker = to_polygon_request_ticker(canonical_ticker)
        limit = page_size or DEFAULT_PAGE_SIZE
        url = f"{POLYGON_TRADES_BASE_URL}/v3/trades/{provider_ticker}"
        params: dict[str, Any] | None = {
            "timestamp": trading_date.isoformat(),
            "limit": limit,
            "apiKey": self.api_key,
        }
        page_count = 0

        while url:
            if max_pages is not None and page_count >= max_pages:
                logger.warning(
                    "polygon_trades reached max_pages={} for {} {} before exhausting next_url",
                    max_pages,
                    canonical_ticker,
                    trading_date,
                )
                break

            payload = self._request_json(
                url,
                params=params,
                context=f"{canonical_ticker} {trading_date.isoformat()} page {page_count + 1}",
            )
            page_count += 1

            results = payload.get("results") or []
            if not isinstance(results, list):
                raise DataSourceError(
                    f"Unexpected Polygon trades payload for {canonical_ticker} {trading_date}: "
                    f"results is {type(results).__name__}",
                )

            for raw_trade in results:
                if not isinstance(raw_trade, dict):
                    raise DataSourceError(
                        f"Unexpected Polygon trade row for {canonical_ticker} {trading_date}: "
                        f"{type(raw_trade).__name__}",
                    )
                yield self._parse_trade_record(canonical_ticker, trading_date, raw_trade)

            next_url = payload.get("next_url")
            if not next_url:
                break
            url = self._append_api_key(str(next_url))
            params = None

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("Bulk trades fetch is handled by the Week 4 sampling executor.")

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        raise NotImplementedError("Incremental trades fetch is handled by the Week 4 sampling executor.")

    def health_check(self) -> bool:
        try:
            list(
                self.fetch_trades_for_day(
                    "SPY",
                    date.today(),
                    page_size=1,
                    max_pages=1,
                ),
            )
        except Exception as exc:
            logger.warning("polygon_trades health check failed: {}", exc)
            return False
        return True

    @DataSource.retryable()
    def _request_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        context: str,
    ) -> dict[str, Any]:
        self._before_request(context)
        session = self._get_http_session()
        try:
            response = session.get(url, params=params, timeout=30)
        except Exception as exc:
            raise DataSourceTransientError(f"Polygon trades request failed for {context}: {exc}") from exc

        if response.status_code != 200:
            self.classify_http_error(
                response.status_code,
                getattr(response, "text", ""),
                context=f"Polygon trades {context}",
            )

        payload = response.json()
        if not isinstance(payload, dict):
            raise DataSourceError(
                f"Unexpected Polygon trades response for {context}: {type(payload).__name__}",
            )
        return payload

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

    def _parse_trade_record(
        self,
        ticker: str,
        trading_date: date,
        payload: dict[str, Any],
    ) -> TradeRecord:
        sip_timestamp_ns = self._required_int(payload.get("sip_timestamp"), "sip_timestamp")
        price = self._required_decimal(payload.get("price"), "price")
        size = self._required_decimal(payload.get("size"), "size")
        participant_timestamp_ns = self._optional_int(payload.get("participant_timestamp"))
        conditions = self._parse_conditions(payload.get("conditions"))
        trade_id = self._optional_str(payload.get("id") or payload.get("trade_id"))
        sequence_raw = payload.get("sequence_number")
        sequence_number = (
            int(sequence_raw)
            if sequence_raw is not None and not pd.isna(sequence_raw)
            else stable_sequence_fallback(
                trade_id,
                price,
                size,
                participant_timestamp_ns,
                tuple(conditions),
            )
        )

        return TradeRecord(
            ticker=ticker,
            trading_date=trading_date,
            sip_timestamp=self._timestamp_ns_to_utc(sip_timestamp_ns, field_name="sip_timestamp"),
            participant_timestamp=self._optional_timestamp_ns_to_utc(participant_timestamp_ns),
            trf_timestamp=self._optional_timestamp_ns_to_utc(payload.get("trf_timestamp")),
            price=price,
            size=size,
            decimal_size=self._optional_decimal(payload.get("decimal_size")),
            exchange=self._optional_int(payload.get("exchange"), default=-1),
            tape=self._optional_int(payload.get("tape")),
            conditions=conditions,
            correction=self._optional_int(payload.get("correction")),
            sequence_number=sequence_number,
            trade_id=trade_id,
            trf_id=self._optional_str(payload.get("trf_id")),
        )

    def _append_api_key(self, url: str) -> str:
        parsed = urlparse(url)
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        if not any(key == "apiKey" for key, _ in query_pairs):
            query_pairs.append(("apiKey", self.api_key))
        return urlunparse(parsed._replace(query=urlencode(query_pairs)))

    @staticmethod
    def _timestamp_ns_to_utc(value: int, *, field_name: str) -> datetime:
        try:
            return datetime.fromtimestamp(int(value) / 1_000_000_000, tz=timezone.utc)
        except Exception as exc:
            raise DataSourceError(f"Invalid Polygon trades {field_name}: {value!r}") from exc

    @classmethod
    def _optional_timestamp_ns_to_utc(cls, value: Any) -> datetime | None:
        numeric = cls._optional_int(value)
        if numeric is None:
            return None
        return cls._timestamp_ns_to_utc(numeric, field_name="timestamp")

    @staticmethod
    def _required_decimal(value: Any, field_name: str) -> Decimal:
        if value is None or pd.isna(value):
            raise DataSourceError(f"Polygon trade is missing required field {field_name}")
        return Decimal(str(value))

    @staticmethod
    def _optional_decimal(value: Any) -> Decimal | None:
        if value is None or pd.isna(value):
            return None
        return Decimal(str(value))

    @staticmethod
    def _required_int(value: Any, field_name: str) -> int:
        if value is None or pd.isna(value):
            raise DataSourceError(f"Polygon trade is missing required field {field_name}")
        return int(value)

    @staticmethod
    def _optional_int(value: Any, default: int | None = None) -> int | None:
        if value is None or pd.isna(value):
            return default
        return int(value)

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(value)

    @staticmethod
    def _parse_conditions(value: Any) -> list[int]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            if pd.isna(value):
                return []
            raise DataSourceError(f"Polygon trade conditions must be a list, got {type(value).__name__}")
        return [int(condition) for condition in value]
