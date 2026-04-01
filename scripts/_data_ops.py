from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
import socket
import sys
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import sqlalchemy as sa
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.data.db.models import (
    CorporateAction,
    FundamentalsPIT,
    Stock,
    StockPrice,
    UniverseMembership,
)
from src.data.db.session import get_engine, get_session_factory
from src.data.quality import DataQualityChecker, QualityReport, QualityStatus

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_TABLES = (
    "stocks",
    "stock_prices",
    "fundamentals_pit",
    "universe_membership",
    "corporate_actions",
    "feature_store",
    "model_registry",
    "predictions",
    "portfolios",
    "backtest_results",
    "audit_log",
)


@dataclass(frozen=True)
class PriceCoverage:
    ticker: str
    min_date: date | None
    max_date: date | None
    row_count: int


@dataclass(frozen=True)
class DateCoverage:
    key: str
    min_date: date | None
    max_date: date | None
    row_count: int


def configure_logging(script_name: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            f"{script_name} | <level>{{message}}</level>"
        ),
    )


def parse_tickers(value: str | None) -> list[str]:
    if not value:
        return []
    return sorted({ticker.strip().upper() for ticker in value.split(",") if ticker.strip()})


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO date: {value!r}") from exc


def check_tcp_port(host: str, port: int, *, timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_http_endpoint(url: str, *, timeout: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:  # noqa: S310
            return 200 <= response.status < 500
    except (OSError, URLError):
        return False


def current_market_data_end_date() -> date:
    market_date = datetime.now(ZoneInfo("America/New_York")).date()
    return market_date - timedelta(days=1)


def fetch_sp500_constituents() -> pd.DataFrame:
    if settings.FMP_API_KEY:
        try:
            frame = _fetch_sp500_from_fmp()
            if not frame.empty:
                logger.info("loaded {} S&P 500 constituents from FMP", len(frame))
                return frame
        except Exception as exc:
            logger.warning("FMP constituent fetch failed; falling back to Wikipedia: {}", exc)

    frame = _fetch_sp500_from_wikipedia()
    logger.info("loaded {} S&P 500 constituents from Wikipedia", len(frame))
    return frame


def ensure_selected_constituents(
    constituents: pd.DataFrame,
    requested_tickers: Sequence[str],
) -> pd.DataFrame:
    normalized_requested = tuple(dict.fromkeys(ticker.upper() for ticker in requested_tickers if ticker))
    if not normalized_requested:
        return constituents.copy()

    selected = constituents.loc[constituents["ticker"].isin(normalized_requested)].copy()
    missing = sorted(set(normalized_requested) - set(selected["ticker"]))
    if missing:
        logger.warning(
            "requested tickers are not current S&P 500 constituents and will use placeholder metadata: {}",
            ",".join(missing),
        )
        selected = pd.concat(
            [
                selected,
                pd.DataFrame(
                    [
                        {
                            "ticker": ticker,
                            "company_name": ticker,
                            "sector": None,
                            "industry": None,
                            "ipo_date": None,
                            "shares_outstanding": None,
                        }
                        for ticker in missing
                    ],
                ),
            ],
            ignore_index=True,
        )

    return selected.sort_values("ticker").reset_index(drop=True)


def filter_constituents_by_range(
    constituents: pd.DataFrame,
    *,
    ticker_start: str | None = None,
    ticker_end: str | None = None,
) -> pd.DataFrame:
    if constituents.empty:
        return constituents.copy()

    filtered = constituents.copy()
    if ticker_start:
        filtered = filtered.loc[filtered["ticker"] >= ticker_start.upper()].copy()
    if ticker_end:
        filtered = filtered.loc[filtered["ticker"] <= ticker_end.upper()].copy()
    return filtered.sort_values("ticker").reset_index(drop=True)


def upsert_stocks(constituents: pd.DataFrame) -> int:
    if constituents.empty:
        return 0

    prepared = constituents.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["company_name"] = prepared["company_name"].fillna(prepared["ticker"]).astype(str)
    if "ipo_date" in prepared.columns:
        prepared["ipo_date"] = pd.to_datetime(prepared["ipo_date"], errors="coerce").dt.date
    else:
        prepared["ipo_date"] = None
    if "shares_outstanding" not in prepared.columns:
        prepared["shares_outstanding"] = None

    records = [
        {
            "ticker": row.ticker,
            "company_name": row.company_name,
            "sector": row.sector,
            "industry": row.industry,
            "ipo_date": None if pd.isna(row.ipo_date) else row.ipo_date,
            "delist_date": None,
            "delist_reason": None,
            "shares_outstanding": int(row.shares_outstanding)
            if row.shares_outstanding is not None and not pd.isna(row.shares_outstanding)
            else None,
        }
        for row in prepared.itertuples(index=False)
    ]

    session_factory = get_session_factory()
    with session_factory() as session:
        statement = insert(Stock).values(records)
        upsert = statement.on_conflict_do_update(
            index_elements=[Stock.ticker],
            set_={
                "company_name": statement.excluded.company_name,
                "sector": statement.excluded.sector,
                "industry": statement.excluded.industry,
                "ipo_date": statement.excluded.ipo_date,
                "delist_date": statement.excluded.delist_date,
                "delist_reason": statement.excluded.delist_reason,
                "shares_outstanding": sa.func.coalesce(
                    statement.excluded.shares_outstanding,
                    Stock.shares_outstanding,
                ),
            },
        )
        session.execute(upsert)
        session.commit()

    logger.info("upserted {} rows into stocks", len(records))
    return len(records)


def ensure_tables_exist(required_tables: Sequence[str] = CORE_TABLES) -> None:
    inspector = sa.inspect(get_engine())
    existing = set(inspector.get_table_names())
    missing = [table_name for table_name in required_tables if table_name not in existing]
    if missing:
        raise RuntimeError(
            "Database schema is incomplete. Run scripts/init_db.py first. "
            f"Missing tables: {missing}",
        )


def get_table_counts(
    table_names: Sequence[str] = (
        "stocks",
        "stock_prices",
        "fundamentals_pit",
        "universe_membership",
        "corporate_actions",
        "macro_series_pit",
    ),
) -> dict[str, int | None]:
    inspector = sa.inspect(get_engine())
    session_factory = get_session_factory()
    counts: dict[str, int | None] = {}

    with session_factory() as session:
        for table_name in table_names:
            if not inspector.has_table(table_name):
                counts[table_name] = None
                continue
            counts[table_name] = int(
                session.execute(sa.text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one(),
            )

    return counts


def get_price_coverage(tickers: Sequence[str]) -> dict[str, PriceCoverage]:
    normalized = tuple(dict.fromkeys(ticker.upper() for ticker in tickers if ticker))
    if not normalized:
        return {}

    session_factory = get_session_factory()
    statement = (
        sa.select(
            StockPrice.ticker.label("ticker"),
            sa.func.min(StockPrice.trade_date).label("min_date"),
            sa.func.max(StockPrice.trade_date).label("max_date"),
            sa.func.count().label("row_count"),
        )
        .where(StockPrice.ticker.in_(normalized))
        .group_by(StockPrice.ticker)
    )
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    coverage = {
        str(row["ticker"]): PriceCoverage(
            ticker=str(row["ticker"]),
            min_date=row["min_date"],
            max_date=row["max_date"],
            row_count=int(row["row_count"]),
        )
        for row in rows
    }
    return coverage


def get_fundamental_coverage(tickers: Sequence[str]) -> dict[str, DateCoverage]:
    normalized = tuple(dict.fromkeys(ticker.upper() for ticker in tickers if ticker))
    if not normalized:
        return {}

    statement = (
        sa.select(
            FundamentalsPIT.ticker.label("key"),
            sa.func.min(FundamentalsPIT.event_time).label("min_date"),
            sa.func.max(FundamentalsPIT.event_time).label("max_date"),
            sa.func.count().label("row_count"),
        )
        .where(FundamentalsPIT.ticker.in_(normalized))
        .group_by(FundamentalsPIT.ticker)
    )
    return _load_date_coverage(statement)


def get_corporate_action_coverage(tickers: Sequence[str]) -> dict[str, DateCoverage]:
    normalized = tuple(dict.fromkeys(ticker.upper() for ticker in tickers if ticker))
    if not normalized:
        return {}

    statement = (
        sa.select(
            CorporateAction.ticker.label("key"),
            sa.func.min(CorporateAction.ex_date).label("min_date"),
            sa.func.max(CorporateAction.ex_date).label("max_date"),
            sa.func.count().label("row_count"),
        )
        .where(
            CorporateAction.ticker.in_(normalized),
            CorporateAction.action_type.in_(["split", "dividend"]),
        )
        .group_by(CorporateAction.ticker)
    )
    return _load_date_coverage(statement)


def get_macro_coverage(series_ids: Sequence[str]) -> dict[str, DateCoverage]:
    normalized = tuple(dict.fromkeys(series_id.upper() for series_id in series_ids if series_id))
    if not normalized:
        return {}

    from src.data.sources.fred import MACRO_SERIES_TABLE

    statement = (
        sa.select(
            MACRO_SERIES_TABLE.c.series_id.label("key"),
            sa.func.min(MACRO_SERIES_TABLE.c.observation_date).label("min_date"),
            sa.func.max(MACRO_SERIES_TABLE.c.observation_date).label("max_date"),
            sa.func.count().label("row_count"),
        )
        .where(MACRO_SERIES_TABLE.c.series_id.in_(normalized))
        .group_by(MACRO_SERIES_TABLE.c.series_id)
    )
    return _load_date_coverage(statement)


def get_tracked_tickers() -> list[str]:
    session_factory = get_session_factory()
    today = date.today()
    with session_factory() as session:
        tickers = (
            session.execute(
                sa.select(Stock.ticker)
                .where(sa.or_(Stock.delist_date.is_(None), Stock.delist_date > today))
                .order_by(Stock.ticker),
            )
            .scalars()
            .all()
        )
    return list(tickers)


def get_latest_price_dates(tickers: Sequence[str]) -> dict[str, date]:
    normalized = tuple(dict.fromkeys(ticker.upper() for ticker in tickers if ticker))
    if not normalized:
        return {}

    session_factory = get_session_factory()
    statement = (
        sa.select(
            StockPrice.ticker.label("ticker"),
            sa.func.max(StockPrice.trade_date).label("latest_trade_date"),
        )
        .where(StockPrice.ticker.in_(normalized))
        .group_by(StockPrice.ticker)
    )
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    return {
        str(row["ticker"]): row["latest_trade_date"]
        for row in rows
        if row["latest_trade_date"] is not None
    }


def _load_date_coverage(statement: sa.Select[Any]) -> dict[str, DateCoverage]:
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    return {
        str(row["key"]): DateCoverage(
            key=str(row["key"]),
            min_date=row["min_date"],
            max_date=row["max_date"],
            row_count=int(row["row_count"]),
        )
        for row in rows
    }


def summarize_quality_reports(
    checker: DataQualityChecker,
    *,
    dataset_name: str,
    frame: pd.DataFrame,
    validate_prices: bool = False,
) -> list[tuple[str, QualityReport]]:
    reports: list[tuple[str, QualityReport]] = []
    if frame.empty:
        logger.warning("quality checks skipped for {} because the dataframe is empty", dataset_name)
        return reports

    reports.append(("missing_rate", checker.check_missing_rate(frame)))
    reports.append(("extreme_values", checker.check_extreme_values(frame)))
    if validate_prices:
        reports.append(("price_validation", checker.validate_price_data(frame)))

    for check_name, report in reports:
        if report.status != QualityStatus.GREEN:
            logger.warning(
                "{} {} status={} message={}",
                dataset_name,
                check_name,
                report.status.value,
                report.message,
            )
        else:
            logger.info(
                "{} {} status={} message={}",
                dataset_name,
                check_name,
                report.status.value,
                report.message,
            )
    return reports


def summarize_row_counts() -> None:
    counts = get_table_counts()
    for table_name, count in counts.items():
        if count is None:
            logger.warning("table {} is missing", table_name)
        else:
            logger.info("table {} row_count={}", table_name, count)


def decimal_to_float_frame(frame: pd.DataFrame) -> pd.DataFrame:
    converted = frame.copy()
    for column in converted.columns:
        if converted[column].dtype == object:
            converted[column] = converted[column].map(
                lambda value: float(value) if isinstance(value, Decimal) else value,
            )
    return converted


def _fetch_sp500_from_fmp() -> pd.DataFrame:
    import requests

    session = requests.Session()
    # FMP may reject requests routed through the local proxy/VPN path.
    session.trust_env = False
    session.headers.update({"User-Agent": "QuantEdge/0.1.0"})
    response = session.get(
        "https://financialmodelingprep.com/stable/sp500-constituent",
        params={"apikey": settings.FMP_API_KEY},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected FMP constituent payload: {type(payload).__name__}")

    frame = pd.DataFrame(payload)
    if frame.empty:
        return pd.DataFrame(
            columns=["ticker", "company_name", "sector", "industry", "ipo_date", "shares_outstanding"],
        )

    company_name_column = "name" if "name" in frame.columns else None
    sector_column = "sector" if "sector" in frame.columns else None
    industry_column = "subSector" if "subSector" in frame.columns else None
    prepared = pd.DataFrame(
        {
            "ticker": frame.get("symbol", pd.Series(dtype=str)).astype(str).str.replace(".", "-", regex=False).str.upper(),
            "company_name": frame[company_name_column] if company_name_column else frame.get("symbol"),
            "sector": frame[sector_column] if sector_column else None,
            "industry": frame[industry_column] if industry_column else None,
            "ipo_date": None,
            "shares_outstanding": None,
        },
    )
    return prepared.dropna(subset=["ticker"]).drop_duplicates(subset=["ticker"]).sort_values("ticker")


def _fetch_sp500_from_wikipedia() -> pd.DataFrame:
    import requests

    response = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "QuantEdge/0.1.0"},
        timeout=30,
    )
    response.raise_for_status()
    parser = _WikipediaConstituentTableParser()
    parser.feed(response.text)
    table = parser.to_frame()
    if table.empty:
        raise RuntimeError("Wikipedia did not return a parseable S&P 500 constituent table.")
    prepared = pd.DataFrame(
        {
            "ticker": table["Symbol"].astype(str).str.replace(".", "-", regex=False).str.upper(),
            "company_name": table["Security"].astype(str),
            "sector": table.get("GICS Sector"),
            "industry": table.get("GICS Sub-Industry"),
            "ipo_date": None,
            "shares_outstanding": None,
        },
    )
    return prepared.drop_duplicates(subset=["ticker"]).sort_values("ticker")


class _WikipediaConstituentTableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_target_table = False
        self._table_complete = False
        self._in_row = False
        self._in_cell = False
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._rows: list[list[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._table_complete:
            return

        attr_map = {key: value or "" for key, value in attrs}
        if tag == "table" and not self._in_target_table:
            classes = attr_map.get("class", "")
            if "wikitable" in classes:
                self._in_target_table = True
            return

        if not self._in_target_table:
            return

        if tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag in {"td", "th"} and self._in_row:
            self._in_cell = True
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._in_target_table and self._in_cell:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self._in_target_table:
            return

        if tag in {"td", "th"} and self._in_cell:
            cell_value = " ".join(part.strip() for part in self._current_cell if part.strip()).strip()
            self._current_row.append(cell_value)
            self._current_cell = []
            self._in_cell = False
        elif tag == "tr" and self._in_row:
            if any(cell for cell in self._current_row):
                self._rows.append(self._current_row)
            self._current_row = []
            self._in_row = False
        elif tag == "table":
            self._in_target_table = False
            self._table_complete = True

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"])

        header = self._rows[0]
        body = self._rows[1:]
        normalized_header = [self._normalize_header(value) for value in header]
        try:
            symbol_index = normalized_header.index("symbol")
            security_index = normalized_header.index("security")
            sector_index = normalized_header.index("gicssector")
            industry_index = normalized_header.index("gicssubindustry")
        except ValueError as exc:
            raise RuntimeError(f"Unexpected Wikipedia constituent table headers: {header}") from exc

        records = []
        for row in body:
            if len(row) <= max(symbol_index, security_index, sector_index, industry_index):
                continue
            records.append(
                {
                    "Symbol": row[symbol_index],
                    "Security": row[security_index],
                    "GICS Sector": row[sector_index],
                    "GICS Sub-Industry": row[industry_index],
                },
            )
        return pd.DataFrame(records)

    @staticmethod
    def _normalize_header(value: str) -> str:
        return "".join(character for character in value.lower() if character.isalnum())
