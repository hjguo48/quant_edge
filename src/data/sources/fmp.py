from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
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

DIVIDEND_ENDPOINT = "dividends"
EARNINGS_ENDPOINT = "earnings"
CONSENSUS_METRIC_NAMES = ("consensus_eps", "eps_consensus")
DERIVED_FUNDAMENTAL_METRICS = (
    "book_value_per_share",
    "annual_dividend",
    "dividend_per_share",
    *CONSENSUS_METRIC_NAMES,
)
INGESTED_FUNDAMENTAL_METRICS = frozenset(
    metric_name
    for endpoint_metrics in ENDPOINT_CONFIG.values()
    for metric_name in endpoint_metrics
).union(DERIVED_FUNDAMENTAL_METRICS)


@dataclass(frozen=True)
class FundamentalAnchor:
    fiscal_period: str
    event_time: date


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
        canonical_anchors: dict[str, FundamentalAnchor] = {}

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
                if endpoint == "income-statement" or fiscal_period not in canonical_anchors:
                    canonical_anchors[fiscal_period] = FundamentalAnchor(
                        fiscal_period=fiscal_period,
                        event_time=event_time,
                    )
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
                        self._build_metric_record(
                            ticker=ticker,
                            fiscal_period=fiscal_period,
                            metric_name=metric_name,
                            metric_value=metric_value,
                            event_time=event_time,
                            knowledge_time=knowledge_time,
                        ),
                    )

        anchors = self._sorted_anchors(canonical_anchors)
        metric_records.extend(self._build_dividend_records(ticker=ticker, anchors=anchors))
        metric_records.extend(self._build_consensus_records(ticker=ticker, anchors=anchors))

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
                self._build_metric_record(
                    ticker=ticker,
                    fiscal_period=str(payload["fiscal_period"]),
                    metric_name="book_value_per_share",
                    metric_value=self._to_decimal(float(equity) / shares_float),
                    event_time=payload["event_time"],
                    knowledge_time=knowledge_time,
                ),
            )

        return sorted(
            metric_records,
            key=lambda record: (
                record["event_time"],
                record["metric_name"],
                record["knowledge_time"],
            ),
        )

    def _build_dividend_records(
        self,
        *,
        ticker: str,
        anchors: Sequence[FundamentalAnchor],
    ) -> list[dict[str, Any]]:
        if not anchors:
            return []

        try:
            rows = self._get_all_endpoint_rows(DIVIDEND_ENDPOINT, ticker, period=None)
        except DataSourceTransientError:
            raise
        except Exception as exc:
            raise DataSourceTransientError(
                f"Failed to retrieve FMP {DIVIDEND_ENDPOINT} for {ticker}: {exc}",
            ) from exc

        dividend_history: list[tuple[date, Decimal]] = []
        records: list[dict[str, Any]] = []
        for row in sorted(rows, key=lambda payload: self._parse_date_value(payload.get("date")) or date.min):
            dividend_date = self._parse_date_value(row.get("date"))
            declaration_date = self._parse_date_value(row.get("declarationDate"))
            if dividend_date is None:
                continue

            anchor = self._match_anchor(
                reference_date=declaration_date or dividend_date,
                anchors=anchors,
                max_lag_days=120,
            )
            if anchor is None:
                continue

            dividend_value = self._to_decimal(row.get("dividend"))
            if dividend_value is None:
                dividend_value = self._to_decimal(row.get("adjDividend"))
            if dividend_value is None or float(dividend_value) <= 0:
                continue

            knowledge_time = self._knowledge_time_from_calendar_date(declaration_date or dividend_date)
            if knowledge_time is None:
                knowledge_time = self._fallback_knowledge_time(anchor.event_time)

            dividend_history.append((dividend_date, dividend_value))
            annual_dividend = self._rolling_annual_dividend(dividend_history, as_of_date=dividend_date)
            records.append(
                self._build_metric_record(
                    ticker=ticker,
                    fiscal_period=anchor.fiscal_period,
                    metric_name="dividend_per_share",
                    metric_value=dividend_value,
                    event_time=anchor.event_time,
                    knowledge_time=knowledge_time,
                ),
            )
            if annual_dividend is not None:
                records.append(
                    self._build_metric_record(
                        ticker=ticker,
                        fiscal_period=anchor.fiscal_period,
                        metric_name="annual_dividend",
                        metric_value=annual_dividend,
                        event_time=anchor.event_time,
                        knowledge_time=knowledge_time,
                    ),
                )

        return records

    def _build_consensus_records(
        self,
        *,
        ticker: str,
        anchors: Sequence[FundamentalAnchor],
    ) -> list[dict[str, Any]]:
        if not anchors:
            return []

        try:
            rows = self._get_all_endpoint_rows(EARNINGS_ENDPOINT, ticker, period=None)
        except DataSourceTransientError:
            raise
        except Exception as exc:
            raise DataSourceTransientError(
                f"Failed to retrieve FMP {EARNINGS_ENDPOINT} for {ticker}: {exc}",
            ) from exc

        records: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, datetime]] = set()
        for row in sorted(rows, key=lambda payload: self._parse_date_value(payload.get("date")) or date.min):
            earnings_date = self._parse_date_value(row.get("date"))
            if earnings_date is None:
                continue

            anchor = self._match_anchor(reference_date=earnings_date, anchors=anchors, max_lag_days=120)
            if anchor is None:
                continue

            consensus_value = self._to_decimal(row.get("epsEstimated"))
            if consensus_value is None:
                continue

            knowledge_time = self._knowledge_time_from_calendar_date(earnings_date)
            if knowledge_time is None:
                knowledge_time = self._fallback_knowledge_time(anchor.event_time)

            for metric_name in CONSENSUS_METRIC_NAMES:
                dedupe_key = (anchor.fiscal_period, metric_name, knowledge_time)
                if dedupe_key in seen_keys:
                    continue
                seen_keys.add(dedupe_key)
                records.append(
                    self._build_metric_record(
                        ticker=ticker,
                        fiscal_period=anchor.fiscal_period,
                        metric_name=metric_name,
                        metric_value=consensus_value,
                        event_time=anchor.event_time,
                        knowledge_time=knowledge_time,
                    ),
                )

        return records

    @staticmethod
    def _sorted_anchors(anchors_by_period: dict[str, FundamentalAnchor]) -> list[FundamentalAnchor]:
        return sorted(
            anchors_by_period.values(),
            key=lambda anchor: (anchor.event_time, anchor.fiscal_period),
        )

    @staticmethod
    def _match_anchor(
        *,
        reference_date: date,
        anchors: Sequence[FundamentalAnchor],
        max_lag_days: int,
    ) -> FundamentalAnchor | None:
        matched_anchor: FundamentalAnchor | None = None
        for anchor in anchors:
            if anchor.event_time > reference_date:
                break
            if (reference_date - anchor.event_time).days <= max_lag_days:
                matched_anchor = anchor
        return matched_anchor

    @staticmethod
    def _rolling_annual_dividend(
        dividend_history: Sequence[tuple[date, Decimal]],
        *,
        as_of_date: date,
    ) -> Decimal | None:
        trailing_values = [
            float(value)
            for dividend_date, value in dividend_history
            if as_of_date - timedelta(days=365) <= dividend_date <= as_of_date
        ]
        if not trailing_values:
            return None
        return FMPDataSource._to_decimal(sum(trailing_values))

    @staticmethod
    def _build_metric_record(
        *,
        ticker: str,
        fiscal_period: str,
        metric_name: str,
        metric_value: Decimal | None,
        event_time: date,
        knowledge_time: datetime,
    ) -> dict[str, Any]:
        return {
            "ticker": ticker,
            "fiscal_period": fiscal_period,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "event_time": event_time,
            "knowledge_time": knowledge_time,
            "is_restated": False,
            "source": FMPDataSource.source_name,
        }

    def persist_fundamentals(self, frame: pd.DataFrame, *, batch_size: int = 1_000) -> int:
        if frame.empty:
            return 0

        records = [self._frame_row_to_record(row) for row in frame.itertuples(index=False)]
        deduped_records: dict[tuple[str, str, str, datetime], dict[str, Any]] = {}
        for record in records:
            dedupe_key = (
                str(record["ticker"]),
                str(record["fiscal_period"]),
                str(record["metric_name"]),
                record["knowledge_time"],
            )
            deduped_records[dedupe_key] = record
        records = list(deduped_records.values())
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
    def _parse_date_value(raw_value: Any) -> date | None:
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
    def _knowledge_time_from_calendar_date(raw_value: Any) -> datetime | None:
        observed_date = FMPDataSource._parse_date_value(raw_value)
        if observed_date is None:
            return None
        localized = datetime.combine(
            observed_date,
            time(16, 0),
            tzinfo=ZoneInfo("America/New_York"),
        )
        return localized.astimezone(timezone.utc)

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
