from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from loguru import logger
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import Stock, StockPrice
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


def normalize_polygon_ticker(ticker: str) -> str:
    return ticker.strip().upper().replace(".", "-")


def to_polygon_request_ticker(ticker: str) -> str:
    return normalize_polygon_ticker(ticker).replace("-", ".")


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
        self._http_session: Any | None = None

    def fetch_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
        *,
        knowledge_time_mode: str = "historical",
        observed_at: date | datetime | None = None,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        observed_at_ts = (
            self.coerce_datetime(observed_at or datetime.now(timezone.utc))
            if knowledge_time_mode == "observed_at"
            else None
        )
        rows: list[dict[str, Any]] = []

        canonical_tickers = tuple(
            dict.fromkeys(normalize_polygon_ticker(ticker) for ticker in self.normalize_tickers(tickers))
        )

        for ticker in canonical_tickers:
            provider_ticker = to_polygon_request_ticker(ticker)
            raw_bars = self._list_aggs(provider_ticker, start, end, adjusted=False)
            adjusted_bars = self._list_aggs(provider_ticker, start, end, adjusted=True)

            if not raw_bars:
                logger.warning(
                    "polygon returned no daily bars for {} using provider symbol {}",
                    ticker,
                    provider_ticker,
                )
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
                        "knowledge_time": self._resolve_knowledge_time(
                            trade_date,
                            mode=knowledge_time_mode,
                            observed_at=observed_at_ts,
                        ),
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

    def fetch_adjusted_historical(
        self,
        tickers: Sequence[str],
        start_date: date | datetime,
        end_date: date | datetime,
        *,
        knowledge_time_mode: str = "historical",
        observed_at: date | datetime | None = None,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        observed_at_ts = (
            self.coerce_datetime(observed_at or datetime.now(timezone.utc))
            if knowledge_time_mode == "observed_at"
            else None
        )
        rows: list[dict[str, Any]] = []

        canonical_tickers = tuple(
            dict.fromkeys(normalize_polygon_ticker(ticker) for ticker in self.normalize_tickers(tickers))
        )

        for ticker in canonical_tickers:
            provider_ticker = to_polygon_request_ticker(ticker)
            adjusted_bars = self._list_aggs(provider_ticker, start, end, adjusted=True)

            if not adjusted_bars:
                logger.warning(
                    "polygon returned no adjusted daily bars for {} using provider symbol {}",
                    ticker,
                    provider_ticker,
                )
                continue

            for trade_date, adjusted_bar in sorted(adjusted_bars.items()):
                adjusted_close = adjusted_bar["close"]
                rows.append(
                    {
                        "ticker": ticker,
                        "trade_date": trade_date,
                        "open": adjusted_bar["open"],
                        "high": adjusted_bar["high"],
                        "low": adjusted_bar["low"],
                        "close": adjusted_close,
                        "adj_close": adjusted_close,
                        "volume": adjusted_bar["volume"],
                        "knowledge_time": self._resolve_knowledge_time(
                            trade_date,
                            mode=knowledge_time_mode,
                            observed_at=observed_at_ts,
                        ),
                        "source": self.source_name,
                    },
                )

        frame = self.dataframe_or_empty(rows, PRICE_COLUMNS)
        if not frame.empty:
            self.persist_prices(frame)
        logger.info(
            "polygon fetched {} fully adjusted rows for {} tickers between {} and {}",
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
        *,
        knowledge_time_mode: str = "historical",
        observed_at: date | datetime | None = None,
    ) -> pd.DataFrame:
        return self.fetch_historical(
            tickers,
            self.coerce_date(since_date),
            self._current_market_end_date(),
            knowledge_time_mode=knowledge_time_mode,
            observed_at=observed_at,
        )

    def fetch_grouped_daily(
        self,
        trade_date: date | datetime,
        *,
        tickers: Sequence[str] | None = None,
        knowledge_time_mode: str = "historical",
        observed_at: date | datetime | None = None,
    ) -> pd.DataFrame:
        trade_day = self.coerce_date(trade_date)
        observed_at_ts = (
            self.coerce_datetime(observed_at or datetime.now(timezone.utc))
            if knowledge_time_mode == "observed_at"
            else None
        )
        ticker_filter = (
            {normalize_polygon_ticker(ticker) for ticker in tickers}
            if tickers is not None
            else None
        )

        raw_bars = self._get_grouped_daily_rows(trade_day, adjusted=False)
        adjusted_bars = self._get_grouped_daily_rows(trade_day, adjusted=True)
        if not raw_bars:
            logger.warning("polygon grouped daily returned no bars for {}", trade_day)
            return pd.DataFrame(columns=PRICE_COLUMNS)

        adjusted_by_ticker = {
            normalize_polygon_ticker(str(self._extract_field(payload, "ticker", "T"))): payload
            for payload in adjusted_bars
            if self._extract_field(payload, "ticker", "T")
        }

        rows: list[dict[str, Any]] = []
        for payload in raw_bars:
            raw_ticker = self._extract_field(payload, "ticker", "T")
            if not raw_ticker:
                continue
            ticker = normalize_polygon_ticker(str(raw_ticker))
            if ticker_filter is not None and ticker not in ticker_filter:
                continue

            adjusted_payload = adjusted_by_ticker.get(ticker, {})
            rows.append(
                {
                    "ticker": ticker,
                    "trade_date": trade_day,
                    "open": self._extract_field(payload, "open", "o"),
                    "high": self._extract_field(payload, "high", "h"),
                    "low": self._extract_field(payload, "low", "l"),
                    "close": self._extract_field(payload, "close", "c"),
                    "adj_close": self._extract_field(adjusted_payload, "close", "c")
                    or self._extract_field(payload, "close", "c"),
                    "volume": self._extract_field(payload, "volume", "v"),
                    "knowledge_time": self._resolve_knowledge_time(
                        trade_day,
                        mode=knowledge_time_mode,
                        observed_at=observed_at_ts,
                    ),
                    "source": self.source_name,
                },
            )

        frame = self.dataframe_or_empty(rows, PRICE_COLUMNS)
        if not frame.empty:
            self.persist_prices(frame)
        logger.info(
            "polygon grouped daily fetched {} rows for {} between {} and {}",
            len(frame),
            len(set(frame["ticker"])) if not frame.empty else 0,
            trade_day,
            trade_day,
        )
        return frame

    def fetch_grouped_daily_range(
        self,
        start_date: date | datetime,
        end_date: date | datetime,
        *,
        tickers: Sequence[str] | None = None,
        knowledge_time_mode: str = "historical",
        observed_at: date | datetime | None = None,
    ) -> pd.DataFrame:
        start = self.coerce_date(start_date)
        end = self.coerce_date(end_date)
        if start > end:
            return pd.DataFrame(columns=PRICE_COLUMNS)

        frames: list[pd.DataFrame] = []
        current = start
        while current <= end:
            frame = self.fetch_grouped_daily(
                current,
                tickers=tickers,
                knowledge_time_mode=knowledge_time_mode,
                observed_at=observed_at,
            )
            if not frame.empty:
                frames.append(frame)
            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["trade_date", "ticker"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined

    def health_check(self) -> bool:
        try:
            end = self._current_market_end_date()
            start = end - timedelta(days=10)
            return bool(self._list_aggs("SPY", start, end, adjusted=False))
        except Exception as exc:
            logger.warning("polygon health check failed: {}", exc)
            return False

    def resolve_latest_available_trade_date(
        self,
        *,
        benchmark_ticker: str = "SPY",
        reference_time: date | datetime | None = None,
        lookback_days: int = 7,
    ) -> date:
        if lookback_days < 1:
            raise ValueError("lookback_days must be at least 1.")

        anchor = self.coerce_datetime(reference_time or datetime.now(timezone.utc)).astimezone(
            ZoneInfo("America/New_York"),
        ).date()
        provider_ticker = to_polygon_request_ticker(normalize_polygon_ticker(benchmark_ticker))

        for offset in range(lookback_days):
            candidate = anchor - timedelta(days=offset)
            bars = self._list_aggs(provider_ticker, candidate, candidate, adjusted=False)
            if candidate in bars:
                return candidate

        raise DataSourceError(
            f"Polygon did not expose a latest available trade date for {benchmark_ticker} "
            f"within the last {lookback_days} calendar days.",
        )

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

    @DataSource.retryable()
    def _get_grouped_daily_rows(
        self,
        trade_date: date,
        *,
        adjusted: bool,
    ) -> list[dict[str, Any]]:
        session = self._get_http_session()
        self._before_request(f"grouped_daily {trade_date.isoformat()} adjusted={adjusted}")

        try:
            response = session.get(
                f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{trade_date.isoformat()}",
                params={
                    "adjusted": str(adjusted).lower(),
                    "apiKey": self.api_key,
                },
                timeout=30,
            )
        except Exception as exc:
            raise DataSourceTransientError(
                f"Polygon grouped daily request failed for {trade_date.isoformat()}: {exc}",
            ) from exc

        if response.status_code == 200:
            payload = response.json()
            if not isinstance(payload, dict):
                raise DataSourceError(
                    f"Unexpected Polygon grouped daily payload for {trade_date.isoformat()}: "
                    f"{type(payload).__name__}",
                )
            results = payload.get("results") or []
            if not isinstance(results, list):
                raise DataSourceError(
                    f"Unexpected Polygon grouped daily results type for {trade_date.isoformat()}: "
                    f"{type(results).__name__}",
                )
            return results

        if response.status_code == 404:
            return []

        self.classify_http_error(
            response.status_code,
            response.text,
            context=f"Polygon grouped daily {trade_date.isoformat()} adjusted={adjusted}",
        )
        return []

    def persist_prices(self, frame: pd.DataFrame, *, batch_size: int = 1_000) -> int:
        if frame.empty:
            return 0

        frame = self._filter_pre_ipo_rows(frame)
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
                            "knowledge_time": sa.func.least(
                                sa.func.coalesce(
                                    StockPrice.knowledge_time,
                                    statement.excluded.knowledge_time,
                                ),
                                statement.excluded.knowledge_time,
                            ),
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

    def _filter_pre_ipo_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame

        tickers = tuple(
            dict.fromkeys(
                str(ticker).upper()
                for ticker in frame["ticker"].dropna().astype(str)
            ),
        )
        if not tickers:
            return frame

        session_factory = get_session_factory()
        with session_factory() as session:
            rows = session.execute(
                sa.select(Stock.ticker, Stock.ipo_date).where(Stock.ticker.in_(tickers)),
            ).all()

        ipo_dates = {
            str(raw_ticker).upper(): ipo_date
            for raw_ticker, ipo_date in rows
            if ipo_date is not None
        }
        if not ipo_dates:
            return frame

        trade_dates = pd.to_datetime(frame["trade_date"]).dt.date
        ipo_series = frame["ticker"].astype(str).str.upper().map(ipo_dates)
        keep_mask = ipo_series.isna() | (trade_dates >= ipo_series)
        filtered_count = int((~keep_mask).sum())
        if filtered_count > 0:
            logger.warning("polygon filtered {} pre-IPO price rows before persistence", filtered_count)
        return frame.loc[keep_mask].copy()

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

    def _get_http_session(self) -> Any:
        if self._http_session is not None:
            return self._http_session

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
    def _historical_knowledge_time(trade_date: date) -> datetime:
        next_close_local = datetime.combine(
            trade_date + timedelta(days=1),
            time(16, 0),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return next_close_local.astimezone(timezone.utc)

    @staticmethod
    def _market_close_time(trade_date: date) -> datetime:
        close_local = datetime.combine(
            trade_date,
            time(16, 0),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return close_local.astimezone(timezone.utc)

    @classmethod
    def _resolve_knowledge_time(
        cls,
        trade_date: date,
        *,
        mode: str,
        observed_at: datetime | None,
    ) -> datetime:
        if mode == "historical":
            return cls._historical_knowledge_time(trade_date)
        if mode == "observed_at":
            effective_observed_at = observed_at or datetime.now(timezone.utc)
            return max(effective_observed_at, cls._market_close_time(trade_date))
        raise ValueError(f"Unsupported Polygon knowledge_time_mode: {mode!r}")

    def _current_market_end_date(self) -> date:
        return self.resolve_latest_available_trade_date()

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
