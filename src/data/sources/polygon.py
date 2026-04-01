from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import StockPrice
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource, DataSourceAuthError, DataSourceError, DataSourceTransientError

PRICE_COLUMNS = [
    "ticker",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "knowledge_time",
    "source",
]


class PolygonDataSource(DataSource):
    source_name = "polygon"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.25,
    ) -> None:
        super().__init__(api_key or settings.POLYGON_API_KEY, min_request_interval=min_request_interval)
        self._client: Any | None = None

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        rows: list[dict[str, Any]] = []

        for ticker in self.normalize_tickers(tickers):
            raw_bars = self._list_aggs(ticker, start, end, adjusted=False)
            adjusted_bars = self._list_aggs(ticker, start, end, adjusted=True)

            if not raw_bars:
                logger.warning("polygon returned no daily bars for {}", ticker)
                continue

            for trade_date, raw_bar in sorted(raw_bars.items()):
                adjusted_bar = adjusted_bars.get(trade_date, {})
                rows.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "open": raw_bar["open"],
                        "high": raw_bar["high"],
                        "low": raw_bar["low"],
                        "close": raw_bar["close"],
                        "adj_close": adjusted_bar.get("close", raw_bar["close"]),
                        "volume": raw_bar["volume"],
                        "knowledge_time": self._knowledge_time(trade_date),
                        "source": self.source_name,
                    },
                )

        frame = self.dataframe_or_empty(rows, PRICE_COLUMNS)
        if not frame.empty:
            self.persist_prices(frame)
        logger.info(
            "polygon fetched {} rows for {} tickers between {} and {}",
            len(frame),
            len(set(frame["ticker"])) if not frame.empty else 0,
            start,
            end,
        )
        return frame

    def fetch_incremental(
        self,
        tickers: Sequence[str],
        since_date: date | datetime,
    ) -> pd.DataFrame:
        return self.fetch_historical(tickers, self.coerce_date(since_date), self._current_market_end_date())

    def health_check(self) -> bool:
        try:
            end = self._current_market_end_date()
            start = end - timedelta(days=10)
            return bool(self._list_aggs("SPY", start, end, adjusted=False))
        except Exception as exc:
            logger.warning("polygon health check failed: {}", exc)
            return False

    @DataSource.retryable()
    def _list_aggs(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        *,
        adjusted: bool,
    ) -> dict[date, dict[str, Any]]:
        client = self._get_client()
        self._before_request(
            f"aggregates {ticker} {start_date.isoformat()}->{end_date.isoformat()} adjusted={adjusted}",
        )

        try:
            aggregates = list(
                client.list_aggs(
                    ticker=ticker,
                    multiplier=1,
                    timespan="day",
                    from_=start_date.isoformat(),
                    to=end_date.isoformat(),
                    adjusted=adjusted,
                    sort="asc",
                    limit=50_000,
                ),
            )
        except Exception as exc:
            error_message = str(exc)
            if "NOT_AUTHORIZED" in error_message or "doesn't include this data timeframe" in error_message:
                raise DataSourceAuthError(
                    f"Polygon aggregates request is not authorized for {ticker}: {error_message}",
                ) from exc
            raise DataSourceTransientError(
                f"Polygon aggregates request failed for {ticker}: {exc}",
            ) from exc

        parsed: dict[date, dict[str, Any]] = {}
        for aggregate in aggregates:
            trade_date = self._extract_trade_date(aggregate)
            if trade_date is None:
                continue

            parsed[trade_date] = {
                "open": self._extract_field(aggregate, "open", "o"),
                "high": self._extract_field(aggregate, "high", "h"),
                "low": self._extract_field(aggregate, "low", "l"),
                "close": self._extract_field(aggregate, "close", "c"),
                "volume": self._extract_field(aggregate, "volume", "v"),
            }

        return parsed

    def persist_prices(self, frame: pd.DataFrame, *, batch_size: int = 1_000) -> int:
        if frame.empty:
            return 0

        records = [self._frame_row_to_record(row) for row in frame.itertuples(index=False)]
        session_factory = get_session_factory()

        with session_factory() as session:
            try:
                for index in range(0, len(records), batch_size):
                    chunk = records[index : index + batch_size]
                    statement = insert(StockPrice).values(chunk)
                    upsert = statement.on_conflict_do_update(
                        index_elements=[StockPrice.ticker, StockPrice.trade_date],
                        set_={
                            "open": statement.excluded.open,
                            "high": statement.excluded.high,
                            "low": statement.excluded.low,
                            "close": statement.excluded.close,
                            "adj_close": statement.excluded.adj_close,
                            "volume": statement.excluded.volume,
                            "knowledge_time": statement.excluded.knowledge_time,
                            "source": statement.excluded.source,
                        },
                    )
                    session.execute(upsert)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.opt(exception=exc).error("polygon failed to persist price rows")
                raise DataSourceError("Failed to persist Polygon price data.") from exc

        return len(records)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        self._require_api_key()

        try:
            from polygon import RESTClient
        except ImportError as exc:
            raise DataSourceError(
                "polygon-api-client is not installed. Add the phase1-week2 dependency group.",
            ) from exc

        self._client = RESTClient(api_key=self.api_key)
        return self._client

    @staticmethod
    def _knowledge_time(trade_date: date) -> datetime:
        next_close_local = datetime.combine(
            trade_date + timedelta(days=1),
            time(16, 0),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return next_close_local.astimezone(timezone.utc)

    @staticmethod
    def _current_market_end_date() -> date:
        return datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)

    @staticmethod
    def _extract_field(payload: Any, *candidate_names: str) -> Any:
        for name in candidate_names:
            if isinstance(payload, dict) and name in payload:
                return payload[name]
            if hasattr(payload, name):
                return getattr(payload, name)
        return None

    @classmethod
    def _extract_trade_date(cls, aggregate: Any) -> date | None:
        timestamp = cls._extract_field(aggregate, "timestamp", "t")
        if timestamp is None:
            return None
        return datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc).date()

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        if value is None or pd.isna(value):
            return None
        return Decimal(str(round(float(value), 4)))

    def _frame_row_to_record(self, row: Any) -> dict[str, Any]:
        return {
            "ticker": row.ticker,
            "trade_date": row.trade_date if isinstance(row.trade_date, date) else row.trade_date.date(),
            "open": self._to_decimal(row.open),
            "high": self._to_decimal(row.high),
            "low": self._to_decimal(row.low),
            "close": self._to_decimal(row.close),
            "adj_close": self._to_decimal(row.adj_close),
            "volume": None if pd.isna(row.volume) else int(row.volume),
            "knowledge_time": self.coerce_datetime(row.knowledge_time),
            "source": row.source,
        }
