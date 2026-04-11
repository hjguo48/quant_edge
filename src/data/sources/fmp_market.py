from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource, DataSourceError, DataSourceTransientError
from src.data.sources.fred import MACRO_COLUMNS, MACRO_SERIES_TABLE

VIX_SERIES_ALIASES = {
    "VIX": "^VIX",
    "VIXCLS": "^VIX",
    "^VIX": "^VIX",
}


class FMPMarketDataSource(DataSource):
    source_name = "fmp"
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.10,
    ) -> None:
        super().__init__(api_key or settings.FMP_API_KEY, min_request_interval=min_request_interval)
        self._http_session: Any | None = None

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
        *,
        knowledge_time: date | datetime | None = None,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        observed_at = self.coerce_datetime(knowledge_time or datetime.now(timezone.utc))
        rows: list[dict[str, Any]] = []

        for series_id, symbol in self._resolve_series_ids(tickers):
            payload_rows = self._get_eod_rows(symbol=symbol, start_date=start, end_date=end)
            if not payload_rows:
                logger.warning("fmp market returned no EOD rows for {} ({})", series_id, symbol)
                continue

            for payload in payload_rows:
                observation_date = self._parse_date(payload.get("date"))
                close_value = self._to_decimal(payload.get("close"))
                if observation_date is None or close_value is None:
                    continue
                rows.append(
                    {
                        "series_id": series_id,
                        "observation_date": observation_date,
                        "value": close_value,
                        "knowledge_time": observed_at,
                        "is_revision": False,
                        "source": self.source_name,
                    },
                )

        frame = self.dataframe_or_empty(rows, MACRO_COLUMNS)
        if not frame.empty:
            self.persist_series(frame)
        logger.info(
            "fmp market fetched {} macro rows across {} series between {} and {}",
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
        *,
        end_date: date | datetime | None = None,
        knowledge_time: date | datetime | None = None,
    ) -> pd.DataFrame:
        start = self.coerce_date(since_date)
        end = self.coerce_date(end_date or date.today())
        if start > end:
            return pd.DataFrame(columns=MACRO_COLUMNS)
        return self.fetch_historical(
            tickers,
            start,
            end,
            knowledge_time=knowledge_time,
        )

    def health_check(self) -> bool:
        try:
            rows = self._get_quote_rows("^VIX")
            return bool(rows and rows[0].get("price") is not None)
        except Exception as exc:
            logger.warning("fmp market health check failed: {}", exc)
            return False

    @DataSource.retryable()
    def _get_quote_rows(self, symbol: str) -> list[dict[str, Any]]:
        session = self._get_http_session()
        self._before_request(f"quote/{symbol}")
        try:
            response = session.get(
                f"{self.base_url}/quote",
                params={"symbol": symbol, "apikey": self.api_key},
                timeout=30,
            )
        except Exception as exc:
            raise DataSourceTransientError(f"FMP market quote transport failure for {symbol}: {exc}") from exc
        if response.status_code != 200:
            self.classify_http_error(response.status_code, response.text, context=f"FMP market quote {symbol}")
        payload = response.json()
        if not isinstance(payload, list):
            raise DataSourceError(f"Unexpected FMP market quote payload for {symbol}: {type(payload).__name__}")
        return payload

    @DataSource.retryable()
    def _get_eod_rows(
        self,
        *,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[dict[str, Any]]:
        session = self._get_http_session()
        self._before_request(f"historical-price-eod/full/{symbol} {start_date}->{end_date}")
        try:
            response = session.get(
                f"{self.base_url}/historical-price-eod/full",
                params={
                    "symbol": symbol,
                    "from": start_date.isoformat(),
                    "to": end_date.isoformat(),
                    "apikey": self.api_key,
                },
                timeout=30,
            )
        except Exception as exc:
            raise DataSourceTransientError(
                f"FMP market EOD transport failure for {symbol} {start_date}->{end_date}: {exc}",
            ) from exc

        if response.status_code != 200:
            self.classify_http_error(
                response.status_code,
                response.text,
                context=f"FMP market EOD {symbol} {start_date}->{end_date}",
            )

        payload = response.json()
        if not isinstance(payload, list):
            raise DataSourceError(f"Unexpected FMP market EOD payload for {symbol}: {type(payload).__name__}")
        return payload

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
                logger.opt(exception=exc).error("fmp market failed to persist macro rows")
                raise DataSourceError("Failed to persist FMP market macro rows.") from exc
        return len(records)

    def _get_http_session(self) -> Any:
        if self._http_session is not None:
            return self._http_session

        self._require_api_key()

        try:
            import requests
        except ImportError as exc:
            raise DataSourceError("requests is not installed. Add the phase1-week2 dependency group.") from exc

        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": "QuantEdge/0.1.0"})
        self._http_session = session
        return self._http_session

    @staticmethod
    def _resolve_series_ids(series_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
        normalized_pairs: list[tuple[str, str]] = []
        for series_id in series_ids:
            normalized = str(series_id).strip().upper()
            canonical = normalized if normalized == "VIXCLS" else "VIXCLS"
            symbol = VIX_SERIES_ALIASES.get(normalized, VIX_SERIES_ALIASES.get(canonical))
            if symbol is None:
                raise ValueError(f"Unsupported FMP market macro series: {series_id!r}")
            normalized_pairs.append((canonical, symbol))
        return tuple(dict.fromkeys(normalized_pairs))

    @staticmethod
    def _parse_date(raw_value: Any) -> date | None:
        timestamp = pd.to_datetime(raw_value, errors="coerce")
        if pd.isna(timestamp):
            return None
        return timestamp.date()

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
