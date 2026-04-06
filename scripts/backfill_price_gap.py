from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sys
import time as time_module
from typing import Any, Sequence
from zoneinfo import ZoneInfo

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pandas as pd
import requests
import sqlalchemy as sa
import yfinance as yf
from loguru import logger
from sqlalchemy.dialects.postgresql import insert

from _data_ops import configure_logging, ensure_tables_exist
from src.config import settings
from src.data.db.models import Stock, StockPrice, UniverseMembership
from src.data.db.session import get_session_factory

DEFAULT_START_DATE = date(2015, 1, 1)
DEFAULT_END_DATE = date(2016, 4, 5)
INDEX_NAME = "SP500"
BENCHMARK_TICKER = "SPY"
YFINANCE_SOURCE = "yfinance"
FMP_SOURCE = "fmp_price"
FMP_PRICE_ENDPOINT = "https://financialmodelingprep.com/stable/historical-price-eod/full"
NEW_YORK_TZ = ZoneInfo("America/New_York")
PRICE_COLUMNS = (
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
)


@dataclass(frozen=True)
class MembershipWindow:
    start_date: date
    end_date: date


@dataclass(frozen=True)
class PlanItem:
    ticker: str
    is_current_member: bool
    fetch_start: date
    fetch_end: date
    expected_trade_dates: tuple[date, ...]
    existing_trade_dates: tuple[date, ...]
    reason: str

    @property
    def expected_rows(self) -> int:
        return len(self.expected_trade_dates)

    @property
    def existing_rows(self) -> int:
        return len(self.existing_trade_dates)

    @property
    def missing_rows(self) -> int:
        return self.expected_rows - self.existing_rows


@dataclass(frozen=True)
class FailureRecord:
    ticker: str
    start_date: date
    end_date: date
    error_reason: str


@dataclass
class RunSummary:
    planned_tickers: int = 0
    planned_rows: int = 0
    processed_tickers: int = 0
    skipped_tickers: int = 0
    successful_tickers: int = 0
    failed_tickers: int = 0
    yfinance_only_tickers: int = 0
    fmp_only_tickers: int = 0
    hybrid_tickers: int = 0
    yfinance_success_tickers: int = 0
    fmp_success_tickers: int = 0
    attempted_active_tickers: int = 0
    active_yfinance_success_tickers: int = 0
    attempted_ended_tickers: int = 0
    ended_fmp_success_tickers: int = 0
    inserted_rows: int = 0
    failures: list[FailureRecord] = field(default_factory=list)


class YFinancePriceClient:
    def __init__(self, *, min_request_interval: float = 0.20) -> None:
        self._min_request_interval = min_request_interval
        self._last_request_started_at: float | None = None

    def fetch_history(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        self._throttle()
        raw = yf.download(
            tickers=ticker,
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
            timeout=30,
        )
        if raw.empty:
            logger.warning("yfinance returned no rows ticker={} start={} end={}", ticker, start_date, end_date)
            return _empty_price_frame()

        frame = raw.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            if ticker in frame.columns.get_level_values(-1):
                frame = frame.xs(ticker, axis=1, level=-1)
            else:
                frame.columns = frame.columns.get_level_values(0)

        required_columns = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise RuntimeError(
                f"yfinance payload missing columns for {ticker}: {','.join(missing_columns)}",
            )

        normalized = frame.reset_index()
        date_column = "Date" if "Date" in normalized.columns else normalized.columns[0]
        normalized["trade_date"] = pd.to_datetime(normalized[date_column], errors="coerce").dt.date
        normalized.dropna(subset=["trade_date", "Close", "Adj Close"], inplace=True)
        normalized = normalized.loc[
            (normalized["trade_date"] >= start_date) & (normalized["trade_date"] <= end_date)
        ].copy()
        if normalized.empty:
            return _empty_price_frame()

        normalized.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            },
            inplace=True,
        )
        normalized["ticker"] = ticker
        normalized["knowledge_time"] = normalized["trade_date"].map(_make_knowledge_time)
        normalized["source"] = YFINANCE_SOURCE

        return normalized[list(PRICE_COLUMNS)].sort_values("trade_date").reset_index(drop=True)

    def _throttle(self) -> None:
        if self._last_request_started_at is not None:
            elapsed = time_module.monotonic() - self._last_request_started_at
            remaining = self._min_request_interval - elapsed
            if remaining > 0:
                time_module.sleep(remaining)
        self._last_request_started_at = time_module.monotonic()


class FMPPriceClient:
    def __init__(self, api_key: str, *, min_request_interval: float = 0.25) -> None:
        if not api_key:
            raise RuntimeError("FMP_API_KEY is required for FMP price fallback.")
        self._api_key = api_key
        self._min_request_interval = min_request_interval
        self._last_request_started_at: float | None = None
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers.update({"User-Agent": "QuantEdge/0.1.0"})

    def fetch_history(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        self._throttle()
        response = self._session.get(
            FMP_PRICE_ENDPOINT,
            params={
                "apikey": self._api_key,
                "symbol": ticker,
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
            timeout=30,
        )
        if response.status_code != 200:
            raise RuntimeError(
                "FMP price request failed "
                f"ticker={ticker} status={response.status_code} body={response.text[:200]!r}",
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(f"FMP returned invalid JSON for {ticker}") from exc

        if isinstance(payload, dict) and payload.get("Error Message"):
            raise RuntimeError(f"FMP price error for {ticker}: {payload['Error Message']}")
        if not isinstance(payload, list):
            raise RuntimeError(
                f"Unexpected FMP price payload for {ticker}: {type(payload).__name__}",
            )
        if not payload:
            logger.warning("FMP returned no rows ticker={} start={} end={}", ticker, start_date, end_date)
            return _empty_price_frame()

        frame = pd.DataFrame(payload)
        missing_columns = [column for column in ("date", "open", "high", "low", "close", "volume") if column not in frame]
        if missing_columns:
            raise RuntimeError(
                f"FMP price payload missing columns for {ticker}: {','.join(missing_columns)}",
            )

        frame["trade_date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        frame.dropna(subset=["trade_date", "close"], inplace=True)
        frame = frame.loc[(frame["trade_date"] >= start_date) & (frame["trade_date"] <= end_date)].copy()
        if frame.empty:
            return _empty_price_frame()

        frame.sort_values("trade_date", inplace=True)
        frame.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)
        frame["ticker"] = ticker
        frame["adj_close"] = frame["close"]
        frame["knowledge_time"] = frame["trade_date"].map(_make_knowledge_time)
        frame["source"] = FMP_SOURCE

        return frame[list(PRICE_COLUMNS)].reset_index(drop=True)

    def _throttle(self) -> None:
        if self._last_request_started_at is not None:
            elapsed = time_module.monotonic() - self._last_request_started_at
            remaining = self._min_request_interval - elapsed
            if remaining > 0:
                time_module.sleep(remaining)
        self._last_request_started_at = time_module.monotonic()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.end_date < args.start_date:
        raise ValueError("--end-date must be on or after --start-date.")

    configure_logging("backfill_price_gap")
    ensure_tables_exist(required_tables=("stock_prices", "universe_membership"))

    requested_tickers = _parse_tickers_arg(args.tickers)
    membership_by_ticker, current_tickers = _load_membership_windows(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    ipo_dates_by_ticker = _load_ipo_dates(tuple(sorted(membership_by_ticker)))
    benchmark_trade_dates = _load_benchmark_trade_dates(
        start_date=args.start_date,
        end_date=args.end_date,
    )
    existing_dates_by_ticker = _load_existing_trade_dates(
        tickers=tuple(sorted(membership_by_ticker)),
        start_date=args.start_date,
        end_date=args.end_date,
    )

    plan = _build_plan(
        membership_by_ticker=membership_by_ticker,
        current_tickers=current_tickers,
        ipo_dates_by_ticker=ipo_dates_by_ticker,
        benchmark_trade_dates=benchmark_trade_dates,
        existing_dates_by_ticker=existing_dates_by_ticker,
        requested_tickers=requested_tickers,
        limit=args.limit,
    )
    _log_plan(plan=plan, dry_run=args.dry_run)

    if args.dry_run:
        logger.info(
            "price gap dry-run summary planned_tickers={} active={} ended={} est_rows={}",
            len(plan),
            sum(1 for item in plan if item.is_current_member),
            sum(1 for item in plan if not item.is_current_member),
            sum(item.missing_rows for item in plan),
        )
        return 0

    summary = _execute_plan(plan=plan)
    if args.failures_csv:
        _write_failures_csv(Path(args.failures_csv), summary.failures)
    _log_run_summary(summary)
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill the 2015-01-01 to 2016-04-05 stock price gap with yfinance primary and FMP fallback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the gap-fill plan without writing rows.")
    parser.add_argument("--limit", type=int, help="Process only the first N planned tickers.")
    parser.add_argument("--tickers", help="Optional comma-separated ticker filter.")
    parser.add_argument("--failures-csv", help="Write failures to CSV after execution.")
    parser.add_argument(
        "--start-date",
        type=_parse_iso_date,
        default=DEFAULT_START_DATE,
        help="Gap start date in ISO format.",
    )
    parser.add_argument(
        "--end-date",
        type=_parse_iso_date,
        default=DEFAULT_END_DATE,
        help="Gap end date in ISO format.",
    )
    return parser.parse_args(argv)


def _parse_iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid ISO date: {value!r}") from exc


def _parse_tickers_arg(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    tickers = sorted({_normalize_ticker(token) for token in value.split(",") if token.strip()})
    return tuple(ticker for ticker in tickers if ticker)


def _normalize_ticker(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().upper().replace(".", "-")


def _empty_price_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def _make_knowledge_time(trade_date: date) -> datetime:
    next_day = trade_date + timedelta(days=1)
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
    next_close_local = datetime.combine(next_day, time(16, 0), tzinfo=NEW_YORK_TZ)
    return next_close_local.astimezone(timezone.utc)


def _load_membership_windows(
    *,
    start_date: date,
    end_date: date,
) -> tuple[dict[str, tuple[MembershipWindow, ...]], frozenset[str]]:
    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(
            sa.select(
                UniverseMembership.ticker,
                UniverseMembership.effective_date,
                UniverseMembership.end_date,
            )
            .where(UniverseMembership.index_name == INDEX_NAME)
            .where(UniverseMembership.effective_date <= end_date)
            .where(
                sa.or_(
                    UniverseMembership.end_date.is_(None),
                    UniverseMembership.end_date >= start_date,
                ),
            )
            .order_by(UniverseMembership.ticker, UniverseMembership.effective_date),
        ).all()

        current_tickers = frozenset(
            row[0]
            for row in session.execute(
                sa.select(UniverseMembership.ticker)
                .where(UniverseMembership.index_name == INDEX_NAME)
                .where(UniverseMembership.end_date.is_(None)),
            )
        )

    windows_by_ticker: dict[str, list[MembershipWindow]] = defaultdict(list)
    for raw_ticker, effective_date, membership_end in rows:
        ticker = _normalize_ticker(raw_ticker)
        overlap_start = max(start_date, effective_date)
        overlap_end = min(end_date, membership_end or end_date)
        if overlap_start > overlap_end:
            continue
        windows_by_ticker[ticker].append(MembershipWindow(start_date=overlap_start, end_date=overlap_end))

    merged: dict[str, tuple[MembershipWindow, ...]] = {}
    for ticker, windows in windows_by_ticker.items():
        sorted_windows = sorted(windows, key=lambda item: (item.start_date, item.end_date))
        merged_windows: list[MembershipWindow] = []
        for window in sorted_windows:
            if not merged_windows:
                merged_windows.append(window)
                continue
            previous = merged_windows[-1]
            if window.start_date <= previous.end_date + timedelta(days=1):
                merged_windows[-1] = MembershipWindow(
                    start_date=previous.start_date,
                    end_date=max(previous.end_date, window.end_date),
                )
                continue
            merged_windows.append(window)
        merged[ticker] = tuple(merged_windows)

    return merged, current_tickers


def _load_benchmark_trade_dates(*, start_date: date, end_date: date) -> tuple[date, ...]:
    logger.info(
        "loading benchmark trading calendar ticker={} start={} end={}",
        BENCHMARK_TICKER,
        start_date,
        end_date,
    )
    yfinance_client = YFinancePriceClient(min_request_interval=0.0)
    try:
        frame = yfinance_client.fetch_history(BENCHMARK_TICKER, start_date, end_date)
        if not frame.empty:
            dates = tuple(sorted(pd.to_datetime(frame["trade_date"]).dt.date.unique()))
            logger.info("loaded {} benchmark trading dates from yfinance", len(dates))
            return dates
    except Exception as exc:
        logger.warning("yfinance benchmark calendar failed: {}", exc)

    if settings.FMP_API_KEY:
        try:
            frame = FMPPriceClient(settings.FMP_API_KEY, min_request_interval=0.0).fetch_history(
                BENCHMARK_TICKER,
                start_date,
                end_date,
            )
            if not frame.empty:
                dates = tuple(sorted(pd.to_datetime(frame["trade_date"]).dt.date.unique()))
                logger.info("loaded {} benchmark trading dates from FMP", len(dates))
                return dates
        except Exception as exc:
            logger.warning("FMP benchmark calendar failed: {}", exc)

    dates = tuple(pd.bdate_range(start_date, end_date).date)
    logger.warning("falling back to business-day benchmark calendar with {} dates", len(dates))
    return dates


def _load_ipo_dates(tickers: tuple[str, ...]) -> dict[str, date]:
    if not tickers:
        return {}

    session_factory = get_session_factory()
    with session_factory() as session:
        rows = session.execute(
            sa.select(Stock.ticker, Stock.ipo_date).where(Stock.ticker.in_(tickers)),
        ).all()
    return {
        _normalize_ticker(raw_ticker): ipo_date
        for raw_ticker, ipo_date in rows
        if ipo_date is not None
    }


def _load_existing_trade_dates(
    *,
    tickers: tuple[str, ...],
    start_date: date,
    end_date: date,
) -> dict[str, set[date]]:
    if not tickers:
        return {}

    session_factory = get_session_factory()
    existing: dict[str, set[date]] = defaultdict(set)
    with session_factory() as session:
        rows = session.execute(
            sa.select(StockPrice.ticker, StockPrice.trade_date)
            .where(StockPrice.ticker.in_(tickers))
            .where(StockPrice.trade_date >= start_date)
            .where(StockPrice.trade_date <= end_date),
        ).all()

    for raw_ticker, trade_date in rows:
        existing[_normalize_ticker(raw_ticker)].add(trade_date)
    return existing


def _build_plan(
    *,
    membership_by_ticker: dict[str, tuple[MembershipWindow, ...]],
    current_tickers: frozenset[str],
    ipo_dates_by_ticker: dict[str, date],
    benchmark_trade_dates: tuple[date, ...],
    existing_dates_by_ticker: dict[str, set[date]],
    requested_tickers: tuple[str, ...],
    limit: int | None,
) -> list[PlanItem]:
    benchmark_set = frozenset(benchmark_trade_dates)
    requested_set = frozenset(requested_tickers)
    plan: list[PlanItem] = []

    for ticker, windows in membership_by_ticker.items():
        if requested_set and ticker not in requested_set:
            continue

        ipo_date = ipo_dates_by_ticker.get(ticker)
        bounded_windows = tuple(
            MembershipWindow(start_date=max(window.start_date, ipo_date or window.start_date), end_date=window.end_date)
            for window in windows
            if max(window.start_date, ipo_date or window.start_date) <= window.end_date
        )
        if not bounded_windows:
            continue

        expected_dates: list[date] = []
        for trade_date in benchmark_trade_dates:
            if any(window.start_date <= trade_date <= window.end_date for window in bounded_windows):
                expected_dates.append(trade_date)
        if not expected_dates:
            continue

        expected_date_set = frozenset(expected_dates)
        existing_date_set = frozenset(existing_dates_by_ticker.get(ticker, set()) & benchmark_set & expected_date_set)
        if len(existing_date_set) >= len(expected_date_set):
            continue

        fetch_start = min(window.start_date for window in bounded_windows)
        fetch_end = max(window.end_date for window in bounded_windows)
        if not existing_date_set:
            reason = "no_existing_gap_rows"
        else:
            reason = f"partial_gap_coverage {len(existing_date_set)}/{len(expected_date_set)}"

        plan.append(
            PlanItem(
                ticker=ticker,
                is_current_member=ticker in current_tickers,
                fetch_start=fetch_start,
                fetch_end=fetch_end,
                expected_trade_dates=tuple(expected_dates),
                existing_trade_dates=tuple(sorted(existing_date_set)),
                reason=reason,
            ),
        )

    plan.sort(key=lambda item: (not item.is_current_member, item.ticker))
    if limit is not None:
        plan = plan[:limit]
    return plan


def _log_plan(*, plan: list[PlanItem], dry_run: bool) -> None:
    mode = "dry-run" if dry_run else "execute"
    total_rows = sum(item.missing_rows for item in plan)
    logger.info(
        "price gap {} plan tickers={} active={} ended={} estimated_missing_rows={}",
        mode,
        len(plan),
        sum(1 for item in plan if item.is_current_member),
        sum(1 for item in plan if not item.is_current_member),
        total_rows,
    )
    items_to_log = plan if dry_run or len(plan) <= 50 else plan[:25]
    for item in items_to_log:
        logger.info(
            "gap plan status=fetch ticker={} active={} start={} end={} expected_rows={} existing_rows={} missing_rows={} reason={}",
            item.ticker,
            item.is_current_member,
            item.fetch_start,
            item.fetch_end,
            item.expected_rows,
            item.existing_rows,
            item.missing_rows,
            item.reason,
        )
    if len(items_to_log) < len(plan):
        logger.info("omitted {} additional plan rows from execution log", len(plan) - len(items_to_log))


def _execute_plan(*, plan: list[PlanItem]) -> RunSummary:
    summary = RunSummary(
        planned_tickers=len(plan),
        planned_rows=sum(item.missing_rows for item in plan),
    )
    if not plan:
        return summary

    yfinance_client = YFinancePriceClient()
    fmp_client = FMPPriceClient(settings.FMP_API_KEY)

    buffered_records: list[dict[str, Any]] = []
    buffered_ticker_count = 0

    for index, item in enumerate(plan, start=1):
        expected_dates = frozenset(item.expected_trade_dates)
        existing_dates = set(item.existing_trade_dates)
        summary.processed_tickers += 1
        if item.is_current_member:
            summary.attempted_active_tickers += 1
        else:
            summary.attempted_ended_tickers += 1

        logger.info(
            "[{}/{}] filling gap ticker={} active={} start={} end={} missing_rows={}",
            index,
            len(plan),
            item.ticker,
            item.is_current_member,
            item.fetch_start,
            item.fetch_end,
            item.missing_rows,
        )

        ticker_records: list[dict[str, Any]] = []
        source_labels: set[str] = set()
        yfinance_reason: str | None = None
        fmp_reason: str | None = None

        try:
            yf_frame = yfinance_client.fetch_history(item.ticker, item.fetch_start, item.fetch_end)
        except Exception as exc:
            yf_frame = _empty_price_frame()
            yfinance_reason = str(exc)
            logger.warning("yfinance failed ticker={} error={}", item.ticker, exc)

        if not yf_frame.empty:
            yf_frame = _filter_to_expected_dates(yf_frame, expected_dates)
            yf_records = _frame_to_records(yf_frame)
            if yf_records:
                ticker_records.extend(yf_records)
                source_labels.add(YFINANCE_SOURCE)
                summary.yfinance_success_tickers += 1
                if item.is_current_member:
                    summary.active_yfinance_success_tickers += 1

        resolved_dates = existing_dates | {record["trade_date"] for record in ticker_records}
        remaining_dates = expected_dates - resolved_dates

        if remaining_dates:
            fallback_start = min(remaining_dates)
            fallback_end = max(remaining_dates)
            logger.info(
                "attempting FMP fallback ticker={} start={} end={} remaining_rows={}",
                item.ticker,
                fallback_start,
                fallback_end,
                len(remaining_dates),
            )
            try:
                fmp_frame = fmp_client.fetch_history(item.ticker, fallback_start, fallback_end)
            except Exception as exc:
                fmp_frame = _empty_price_frame()
                fmp_reason = str(exc)
                logger.warning("FMP fallback failed ticker={} error={}", item.ticker, exc)

            if not fmp_frame.empty:
                fmp_frame = _filter_to_expected_dates(fmp_frame, remaining_dates)
                fmp_records = _frame_to_records(fmp_frame)
                fmp_records = [
                    record for record in fmp_records if record["trade_date"] in remaining_dates
                ]
                if fmp_records:
                    ticker_records.extend(fmp_records)
                    source_labels.add(FMP_SOURCE)
                    summary.fmp_success_tickers += 1
                    if not item.is_current_member:
                        summary.ended_fmp_success_tickers += 1

        ticker_records = _dedupe_records(ticker_records)
        resolved_dates = existing_dates | {record["trade_date"] for record in ticker_records}
        remaining_dates = expected_dates - resolved_dates

        if ticker_records:
            buffered_records.extend(ticker_records)
            buffered_ticker_count += 1

        if not remaining_dates:
            summary.successful_tickers += 1
            if source_labels == {YFINANCE_SOURCE}:
                summary.yfinance_only_tickers += 1
            elif source_labels == {FMP_SOURCE}:
                summary.fmp_only_tickers += 1
            elif source_labels == {YFINANCE_SOURCE, FMP_SOURCE}:
                summary.hybrid_tickers += 1
            logger.info(
                "gap fill completed ticker={} sources={} rows_buffered={}",
                item.ticker,
                ",".join(sorted(source_labels)) if source_labels else "none",
                len(ticker_records),
            )
        else:
            summary.failed_tickers += 1
            reason_parts: list[str] = [f"missing_rows_after_backfill={len(remaining_dates)}"]
            if yfinance_reason:
                reason_parts.append(f"yfinance={yfinance_reason}")
            elif not any(record["source"] == YFINANCE_SOURCE for record in ticker_records):
                reason_parts.append("yfinance=empty_result")
            if fmp_reason:
                reason_parts.append(f"fmp={fmp_reason}")
            elif remaining_dates:
                reason_parts.append("fmp=empty_result")

            failure = FailureRecord(
                ticker=item.ticker,
                start_date=item.fetch_start,
                end_date=item.fetch_end,
                error_reason="; ".join(reason_parts),
            )
            summary.failures.append(failure)
            logger.warning(
                "gap fill incomplete ticker={} remaining_rows={} error_reason={}",
                item.ticker,
                len(remaining_dates),
                failure.error_reason,
            )

        if buffered_ticker_count >= 50:
            inserted = _persist_records(buffered_records)
            summary.inserted_rows += inserted
            logger.info(
                "persisted buffered gap rows inserted_rows={} buffered_tickers={}",
                inserted,
                buffered_ticker_count,
            )
            buffered_records = []
            buffered_ticker_count = 0

    if buffered_records:
        inserted = _persist_records(buffered_records)
        summary.inserted_rows += inserted
        logger.info(
            "persisted final buffered gap rows inserted_rows={} buffered_tickers={}",
            inserted,
            buffered_ticker_count,
        )

    summary.skipped_tickers = summary.planned_tickers - summary.processed_tickers
    return summary


def _filter_to_expected_dates(frame: pd.DataFrame, expected_dates: frozenset[date] | set[date]) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.loc[frame["trade_date"].isin(expected_dates)].copy()
    filtered.sort_values("trade_date", inplace=True)
    filtered.drop_duplicates(subset=["trade_date"], keep="last", inplace=True)
    return filtered.reset_index(drop=True)


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "ticker": _normalize_ticker(row.ticker),
                "trade_date": row.trade_date if isinstance(row.trade_date, date) else pd.Timestamp(row.trade_date).date(),
                "open": _to_decimal(row.open),
                "high": _to_decimal(row.high),
                "low": _to_decimal(row.low),
                "close": _to_decimal(row.close),
                "adj_close": _to_decimal(row.adj_close),
                "volume": _to_int(row.volume),
                "knowledge_time": _coerce_datetime(row.knowledge_time),
                "source": str(row.source),
            },
        )
    return records


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, date], dict[str, Any]] = {}
    for record in records:
        deduped[(record["ticker"], record["trade_date"])] = record
    return [deduped[key] for key in sorted(deduped, key=lambda item: (item[0], item[1]))]


def _to_decimal(value: Any) -> Decimal | None:
    if value is None or pd.isna(value):
        return None
    return Decimal(str(round(float(value), 4)))


def _to_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    parsed = pd.Timestamp(value).to_pydatetime()
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _persist_records(records: list[dict[str, Any]], *, chunk_size: int = 2_000) -> int:
    if not records:
        return 0

    session_factory = get_session_factory()
    inserted_rows = 0
    with session_factory() as session:
        try:
            for start_index in range(0, len(records), chunk_size):
                chunk = records[start_index : start_index + chunk_size]
                statement = insert(StockPrice).values(chunk)
                upsert = statement.on_conflict_do_nothing(
                    index_elements=[StockPrice.ticker, StockPrice.trade_date],
                )
                result = session.execute(upsert)
                if result.rowcount is not None and result.rowcount > 0:
                    inserted_rows += result.rowcount
            session.commit()
        except Exception as exc:
            session.rollback()
            logger.opt(exception=exc).error("failed to persist price gap rows")
            raise
    return inserted_rows


def _write_failures_csv(path: Path, failures: list[FailureRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ticker", "start_date", "end_date", "error_reason"])
        for failure in failures:
            writer.writerow(
                [
                    failure.ticker,
                    failure.start_date.isoformat(),
                    failure.end_date.isoformat(),
                    failure.error_reason,
                ],
            )
    logger.info("wrote {} failures to {}", len(failures), path)


def _log_run_summary(summary: RunSummary) -> None:
    active_rate = (
        (summary.active_yfinance_success_tickers / summary.attempted_active_tickers) * 100.0
        if summary.attempted_active_tickers
        else 0.0
    )
    logger.info(
        "price gap summary planned_tickers={} successful_tickers={} failed_tickers={} inserted_rows={}",
        summary.planned_tickers,
        summary.successful_tickers,
        summary.failed_tickers,
        summary.inserted_rows,
    )
    logger.info(
        "source mix yfinance_only={} fmp_only={} hybrid={} yfinance_success_tickers={} fmp_success_tickers={}",
        summary.yfinance_only_tickers,
        summary.fmp_only_tickers,
        summary.hybrid_tickers,
        summary.yfinance_success_tickers,
        summary.fmp_success_tickers,
    )
    logger.info(
        "active yfinance success rate {:.1f}% ({}/{})",
        active_rate,
        summary.active_yfinance_success_tickers,
        summary.attempted_active_tickers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
