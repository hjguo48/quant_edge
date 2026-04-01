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
from src.data.db.models import FundamentalsPIT
from src.data.db.session import get_session_factory
from src.data.sources.base import DataSource, DataSourceError, DataSourceTransientError

FUNDAMENTAL_COLUMNS = [
    "ticker",
    "fiscal_period",
    "metric_name",
    "metric_value",
    "event_time",
    "knowledge_time",
    "is_restated",
    "source",
]

ENDPOINT_CONFIG = {
    "income-statement": {
        "revenue": "revenue",
        "gross_profit": "grossProfit",
        "operating_income": "operatingIncome",
        "net_income": "netIncome",
        "ebitda": "ebitda",
        "eps": "eps",
        "weighted_average_shares_outstanding": "weightedAverageShsOut",
    },
    "balance-sheet-statement": {
        "total_assets": "totalAssets",
        "total_liabilities": "totalLiabilities",
        "total_debt": "totalDebt",
        "cash_and_cash_equivalents": "cashAndCashEquivalents",
        "current_assets": "totalCurrentAssets",
        "current_liabilities": "totalCurrentLiabilities",
        "total_stockholders_equity": "totalStockholdersEquity",
    },
    "cash-flow-statement": {
        "operating_cash_flow": "operatingCashFlow",
        "capital_expenditure": "capitalExpenditure",
        "free_cash_flow": "freeCashFlow",
    },
}


class FMPDataSource(DataSource):
    source_name = "fmp"
    base_url = "https://financialmodelingprep.com/stable"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_request_interval: float = 0.25,
    ) -> None:
        super().__init__(api_key or settings.FMP_API_KEY, min_request_interval=min_request_interval)
        self._http_session: Any | None = None

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
            records = self._fetch_ticker_records(ticker)
            if not records:
                logger.warning("fmp returned no quarterly fundamentals for {}", ticker)
                continue

            for record in records:
                if start <= record["event_time"] <= end:
                    rows.append(record)

        frame = self.dataframe_or_empty(rows, FUNDAMENTAL_COLUMNS)
        if not frame.empty:
            self.persist_fundamentals(frame)
        logger.info(
            "fmp fetched {} PIT fundamental rows across {} tickers between {} and {}",
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
        cutoff = self.coerce_datetime(since_date)
        lookback_start = self.coerce_date(since_date) - timedelta(days=180)
        frame = self.fetch_historical(tickers, lookback_start, date.today())
        if frame.empty:
            return frame
        incremental = frame.loc[pd.to_datetime(frame["knowledge_time"], utc=True) >= cutoff]
        return incremental.reset_index(drop=True)

    def health_check(self) -> bool:
        try:
            rows = self._get_endpoint_rows("income-statement", "AAPL", limit=1, period="quarter")
            return bool(rows)
        except Exception as exc:
            logger.warning("fmp health check failed: {}", exc)
            return False

    @DataSource.retryable()
    def _get_endpoint_rows(
        self,
        endpoint: str,
        ticker: str,
        *,
        limit: int = 120,
        page: int = 0,
        period: str | None = "quarter",
    ) -> list[dict[str, Any]]:
        session = self._get_http_session()
        self._before_request(f"{endpoint}/{ticker}")

        params: dict[str, Any] = {
            "apikey": self.api_key,
            "symbol": ticker,
            "limit": limit,
            "page": page,
        }
        if period:
            params["period"] = period

        try:
            response = session.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                timeout=30,
            )
        except Exception as exc:
            raise DataSourceTransientError(
                f"FMP request transport failure for {endpoint}/{ticker}: {exc}",
            ) from exc
        if response.status_code != 200:
            if (
                response.status_code == 402
                and limit > 5
                and "values for 'limit' must be between 0 and 5" in response.text
            ):
                logger.warning(
                    "FMP plan limits {} to five rows for {}; retrying with limit=5",
                    endpoint,
                    ticker,
                )
                return self._get_endpoint_rows(
                    endpoint,
                    ticker,
                    limit=5,
                    page=page,
                    period=period,
                )
            self.classify_http_error(
                response.status_code,
                response.text,
                context=f"FMP endpoint {endpoint}/{ticker}",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise DataSourceTransientError(
                f"FMP returned invalid JSON for {endpoint}/{ticker}: {exc}",
            ) from exc
        if isinstance(payload, dict) and payload.get("Error Message"):
            raise DataSourceError(
                f"FMP endpoint {endpoint}/{ticker} returned an error: {payload['Error Message']}",
            )
        if not isinstance(payload, list):
            raise DataSourceError(
                f"Unexpected FMP payload for {endpoint}/{ticker}: {type(payload).__name__}",
            )
        return payload

    def _get_all_endpoint_rows(
        self,
        endpoint: str,
        ticker: str,
        *,
        period: str | None = "quarter",
        page_limit: int = 250,
    ) -> list[dict[str, Any]]:
        page = 0
        collected: list[dict[str, Any]] = []
        seen_row_keys: set[tuple[Any, ...]] = set()

        while True:
            rows = self._get_endpoint_rows(
                endpoint,
                ticker,
                limit=page_limit,
                page=page,
                period=period,
            )
            if not rows:
                break

            new_rows = 0
            for row in rows:
                row_key = (
                    row.get("date"),
                    row.get("acceptedDate"),
                    row.get("fillingDate"),
                    row.get("calendarYear"),
                    row.get("period"),
                )
                if row_key in seen_row_keys:
                    continue
                seen_row_keys.add(row_key)
                collected.append(row)
                new_rows += 1

            if len(rows) < page_limit or new_rows == 0:
                break
            page += 1

        return collected

    def _fetch_ticker_records(self, ticker: str) -> list[dict[str, Any]]:
        metric_records: list[dict[str, Any]] = []
        derived_inputs: dict[tuple[str, date], dict[str, Any]] = {}

        for endpoint, metric_mapping in ENDPOINT_CONFIG.items():
            try:
                rows = self._get_all_endpoint_rows(endpoint, ticker)
            except DataSourceTransientError:
                raise
            except Exception as exc:
                raise DataSourceTransientError(
                    f"Failed to retrieve FMP {endpoint} for {ticker}: {exc}",
                ) from exc

            for row in rows:
                event_time = self._parse_event_time(row)
                if event_time is None:
                    continue

                fiscal_period = self._derive_fiscal_period(row, event_time)
                knowledge_time = self._parse_knowledge_time(row) or self._fallback_knowledge_time(event_time)
                derived_bucket = derived_inputs.setdefault(
                    (fiscal_period, event_time),
                    {
                        "fiscal_period": fiscal_period,
                        "event_time": event_time,
                        "equity": None,
                        "equity_knowledge_time": None,
                        "shares": None,
                        "shares_knowledge_time": None,
                    },
                )

                for metric_name, source_field in metric_mapping.items():
                    metric_value = self._to_decimal(row.get(source_field))
                    if metric_value is None:
                        continue
                    if metric_name == "total_stockholders_equity":
                        derived_bucket["equity"] = metric_value
                        derived_bucket["equity_knowledge_time"] = knowledge_time
                    elif metric_name == "weighted_average_shares_outstanding":
                        derived_bucket["shares"] = metric_value
                        derived_bucket["shares_knowledge_time"] = knowledge_time
                    metric_records.append(
                        {
                            "ticker": ticker,
                            "fiscal_period": fiscal_period,
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                            "event_time": event_time,
                            "knowledge_time": knowledge_time,
                            "is_restated": False,
                            "source": self.source_name,
                        },
                    )

        for payload in derived_inputs.values():
            equity = payload["equity"]
            shares = payload["shares"]
            equity_knowledge_time = payload["equity_knowledge_time"]
            shares_knowledge_time = payload["shares_knowledge_time"]
            if equity is None or shares is None or equity_knowledge_time is None or shares_knowledge_time is None:
                continue

            shares_float = float(shares)
            if shares_float <= 0:
                continue

            knowledge_time = max(equity_knowledge_time, shares_knowledge_time)
            metric_records.append(
                {
                    "ticker": ticker,
                    "fiscal_period": payload["fiscal_period"],
                    "metric_name": "book_value_per_share",
                    "metric_value": self._to_decimal(float(equity) / shares_float),
                    "event_time": payload["event_time"],
                    "knowledge_time": knowledge_time,
                    "is_restated": False,
                    "source": self.source_name,
                },
            )

        return sorted(
            metric_records,
            key=lambda record: (
                record["event_time"],
                record["metric_name"],
                record["knowledge_time"],
            ),
        )

    def persist_fundamentals(self, frame: pd.DataFrame, *, batch_size: int = 1_000) -> int:
        if frame.empty:
            return 0

        records = [self._frame_row_to_record(row) for row in frame.itertuples(index=False)]
        session_factory = get_session_factory()

        with session_factory() as session:
            try:
                for index in range(0, len(records), batch_size):
                    chunk = records[index : index + batch_size]
                    statement = insert(FundamentalsPIT).values(chunk)
                    upsert = statement.on_conflict_do_update(
                        constraint="uq_fundamentals_pit_version",
                        set_={
                            "metric_value": statement.excluded.metric_value,
                            "event_time": statement.excluded.event_time,
                            "is_restated": statement.excluded.is_restated,
                            "source": statement.excluded.source,
                        },
                    )
                    session.execute(upsert)
                session.commit()
            except Exception as exc:
                session.rollback()
                logger.opt(exception=exc).error("fmp failed to persist PIT fundamentals")
                raise DataSourceError("Failed to persist FMP fundamentals data.") from exc

        return len(records)

    def _get_http_session(self) -> Any:
        if self._http_session is not None:
            return self._http_session

        self._require_api_key()

        try:
            import requests
        except ImportError as exc:
            raise DataSourceError(
                "requests is not installed. Add the phase1-week2 dependency group.",
            ) from exc

        session = requests.Session()
        # FMP may reject requests routed through the local proxy/VPN path.
        session.trust_env = False
        session.headers.update({"User-Agent": "QuantEdge/0.1.0"})
        self._http_session = session
        return self._http_session

    @staticmethod
    def _parse_event_time(row: dict[str, Any]) -> date | None:
        raw_value = row.get("date") or row.get("fillingDate") or row.get("acceptedDate")
        timestamp = pd.to_datetime(raw_value, errors="coerce")
        if pd.isna(timestamp):
            return None
        return timestamp.date()

    @staticmethod
    def _derive_fiscal_period(row: dict[str, Any], event_time: date) -> str:
        period = str(row.get("period") or "").upper()
        if period in {"Q1", "Q2", "Q3", "Q4"}:
            quarter = period
        else:
            quarter = f"Q{((event_time.month - 1) // 3) + 1}"
        return f"{event_time.year}{quarter}"

    @staticmethod
    def _parse_knowledge_time(row: dict[str, Any]) -> datetime | None:
        for candidate in ("acceptedDate", "fillingDate"):
            raw_value = row.get(candidate)
            if not raw_value:
                continue

            timestamp = pd.to_datetime(raw_value, errors="coerce")
            if pd.isna(timestamp):
                continue

            if timestamp.tzinfo is None:
                localized = timestamp.to_pydatetime().replace(
                    tzinfo=ZoneInfo("America/New_York"),
                )
                return localized.astimezone(timezone.utc)
            return timestamp.tz_convert(timezone.utc).to_pydatetime()

        return None

    @staticmethod
    def _fallback_knowledge_time(event_time: date) -> datetime:
        conservative_time = datetime.combine(
            event_time + timedelta(days=45),
            time(16, 0),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return conservative_time.astimezone(timezone.utc)

    @staticmethod
    def _to_decimal(value: Any) -> Decimal | None:
        if value is None or pd.isna(value):
            return None
        return Decimal(str(round(float(value), 6)))

    def _frame_row_to_record(self, row: Any) -> dict[str, Any]:
        return {
            "ticker": row.ticker,
            "fiscal_period": row.fiscal_period,
            "metric_name": row.metric_name,
            "metric_value": row.metric_value,
            "event_time": row.event_time,
            "knowledge_time": self.coerce_datetime(row.knowledge_time),
            "is_restated": bool(row.is_restated),
            "source": row.source,
        }
