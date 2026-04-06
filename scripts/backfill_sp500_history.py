from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if TYPE_CHECKING:
    import pandas as pd
    import sqlalchemy as sa
    import src.universe.builder as builder_module

pd = None
sa = None
logger = None
configure_logging = None
current_market_data_end_date = None
ensure_tables_exist = None
get_session_factory = None
FundamentalsPIT = None
StockPrice = None
UniverseMembership = None
FMPDataSource = None
PolygonDataSource = None
INGESTED_FUNDAMENTAL_METRICS = None
REQUIRED_FUNDAMENTAL_COVERAGE_METRICS = None
builder_module = None

INDEX_NAME = "SP500"
DEFAULT_MEMBERSHIP_START = date(2018, 1, 1)


@dataclass(frozen=True)
class MembershipDryRunPlan:
    current_constituent_count: int
    change_event_count: int
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class PriceInterval:
    ticker: str
    effective_date: date
    end_date: date


@dataclass(frozen=True)
class PriceCoverage:
    min_date: date | None
    max_date: date | None
    row_count: int


@dataclass(frozen=True)
class PricePlanItem:
    ticker: str
    effective_date: date
    end_date: date
    estimated_rows: int
    status: str
    reason: str


@dataclass(frozen=True)
class FundamentalCoverage:
    min_date: date | None
    max_date: date | None
    row_count: int
    metric_names: frozenset[str]


@dataclass(frozen=True)
class FundamentalPlanItem:
    ticker: str
    effective_date: date
    end_date: date
    estimated_rows: int
    status: str
    reason: str


@dataclass(frozen=True)
class BackfillFailure:
    ticker: str
    effective_date: date
    end_date: date
    error_reason: str


@dataclass
class PriceRunSummary:
    planned_intervals: int = 0
    successful_intervals: int = 0
    skipped_intervals: int = 0
    failed_intervals: int = 0
    fetched_rows: int = 0
    failures: list[BackfillFailure] = field(default_factory=list)


@dataclass
class FundamentalRunSummary:
    planned_intervals: int = 0
    successful_intervals: int = 0
    skipped_intervals: int = 0
    failed_intervals: int = 0
    fetched_rows: int = 0
    failures: list[BackfillFailure] = field(default_factory=list)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.price_tail_days < 0:
        raise ValueError("--price-tail-days must be non-negative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")

    _ensure_runtime_imports()
    configure_logging("backfill_sp500_history")
    ensure_tables_exist(required_tables=_required_tables_for_phase(args.phase))

    membership_end = _resolve_membership_end(args.membership_end)
    if membership_end < args.membership_start:
        raise ValueError("--membership-end must be on or after --membership-start.")

    requested_tickers = _parse_tickers_arg(args.tickers)

    if args.phase == "membership":
        if requested_tickers or args.limit is not None:
            logger.warning(
                "--tickers and --limit apply to fundamentals/prices only; membership rebuild will ignore them",
            )
        if args.failures_csv:
            logger.warning(
                "--failures-csv applies to fundamentals/prices only; membership rebuild will ignore it",
            )
        _run_membership_phase(
            membership_start=args.membership_start,
            membership_end=membership_end,
            strict_fmp=args.strict_fmp,
            dry_run=args.dry_run,
        )
        return 0

    if args.phase == "fundamentals":
        if args.strict_fmp:
            logger.info("--strict-fmp has no effect in fundamentals-only mode")
        summary = _run_fundamentals_phase(
            dry_run=args.dry_run,
            requested_tickers=requested_tickers,
            limit=args.limit,
            membership_rows=None,
        )
        if args.failures_csv and not args.dry_run:
            _write_failures_csv(Path(args.failures_csv), summary.failures)
        return 0

    if args.phase == "prices":
        if args.strict_fmp:
            logger.info("--strict-fmp has no effect in prices-only mode")
        summary = _run_prices_phase(
            dry_run=args.dry_run,
            price_tail_days=args.price_tail_days,
            requested_tickers=requested_tickers,
            limit=args.limit,
            membership_rows=None,
        )
        if args.failures_csv and not args.dry_run:
            _write_failures_csv(Path(args.failures_csv), summary.failures)
        return 0

    if args.dry_run:
        plan = _run_membership_phase(
            membership_start=args.membership_start,
            membership_end=membership_end,
            strict_fmp=args.strict_fmp,
            dry_run=True,
        )
        _run_fundamentals_phase(
            dry_run=True,
            requested_tickers=requested_tickers,
            limit=args.limit,
            membership_rows=plan.rows,
        )
        _run_prices_phase(
            dry_run=True,
            price_tail_days=args.price_tail_days,
            requested_tickers=requested_tickers,
            limit=args.limit,
            membership_rows=plan.rows,
        )
        return 0

    _run_membership_phase(
        membership_start=args.membership_start,
        membership_end=membership_end,
        strict_fmp=args.strict_fmp,
        dry_run=False,
    )
    fundamental_summary = _run_fundamentals_phase(
        dry_run=False,
        requested_tickers=requested_tickers,
        limit=args.limit,
        membership_rows=None,
    )
    price_summary = _run_prices_phase(
        dry_run=False,
        price_tail_days=args.price_tail_days,
        requested_tickers=requested_tickers,
        limit=args.limit,
        membership_rows=None,
    )
    if args.failures_csv:
        _write_failures_csv(
            Path(args.failures_csv),
            [*fundamental_summary.failures, *price_summary.failures],
        )
    return 0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild S&P 500 membership history and backfill removed-constituent fundamentals/prices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--phase", choices=("membership", "fundamentals", "prices", "all"), required=True)
    parser.add_argument("--membership-start", type=_parse_date_arg, default=DEFAULT_MEMBERSHIP_START)
    parser.add_argument("--membership-end", type=_parse_date_arg, default=date.today())
    parser.add_argument("--price-tail-days", type=int, default=90)
    parser.add_argument("--strict-fmp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        help="Fundamentals/prices phases only: limit processing to the first N tickers.",
    )
    parser.add_argument(
        "--tickers",
        help="Fundamentals/prices phases only: comma-separated ticker filter (for example BRK-B,BF-B).",
    )
    parser.add_argument(
        "--failures-csv",
        help="Fundamentals/prices phases only: write failures to CSV with ticker,effective_date,end_date,error_reason.",
    )
    return parser.parse_args(argv)


def _parse_date_arg(value: str) -> date:
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


def _required_tables_for_phase(phase: str) -> tuple[str, ...]:
    if phase == "membership":
        return ("universe_membership",)
    if phase == "fundamentals":
        return ("universe_membership", "fundamentals_pit")
    if phase == "prices":
        return ("universe_membership", "stock_prices")
    return ("universe_membership", "fundamentals_pit", "stock_prices")


def _resolve_membership_end(requested_end: date) -> date:
    today = date.today()
    if requested_end > today:
        logger.warning(
            "--membership-end={} is in the future; clamping to {}",
            requested_end,
            today,
        )
        return today
    return requested_end


def _run_membership_phase(
    *,
    membership_start: date,
    membership_end: date,
    strict_fmp: bool,
    dry_run: bool,
) -> MembershipDryRunPlan | None:
    if dry_run:
        plan = _build_membership_dry_run_plan(
            membership_start=membership_start,
            membership_end=membership_end,
            strict_fmp=strict_fmp,
        )
        ended_rows = sum(1 for row in plan.rows if row["end_date"] is not None)
        active_rows = len(plan.rows) - ended_rows
        logger.info(
            "membership dry-run index={} start={} end={} strict_fmp={} current_constituents={} change_events={} rows={} active_rows={} ended_rows={}",
            INDEX_NAME,
            membership_start,
            membership_end,
            strict_fmp,
            plan.current_constituent_count,
            plan.change_event_count,
            len(plan.rows),
            active_rows,
            ended_rows,
        )
        for row in plan.rows:
            logger.info(
                "membership plan ticker={} effective_date={} end_date={} reason={}",
                row["ticker"],
                row["effective_date"],
                row["end_date"],
                row["reason"],
            )
        return plan

    logger.info(
        "rebuilding membership index={} start={} end={} strict_fmp={}",
        INDEX_NAME,
        membership_start,
        membership_end,
        strict_fmp,
    )
    rows_written = builder_module.backfill_universe_membership(
        membership_start,
        membership_end,
        index_name=INDEX_NAME,
        strict_fmp=strict_fmp,
    )
    logger.info(
        "membership rebuild completed index={} start={} end={} rows_written={}",
        INDEX_NAME,
        membership_start,
        membership_end,
        rows_written,
    )
    return None


def _build_membership_dry_run_plan(
    *,
    membership_start: date,
    membership_end: date,
    strict_fmp: bool,
) -> MembershipDryRunPlan:
    current_constituents = builder_module._fetch_index_constituents(
        index_name=INDEX_NAME,
        strict_fmp=strict_fmp,
    )
    change_events = builder_module._fetch_index_change_events(
        index_name=INDEX_NAME,
        strict_fmp=strict_fmp,
    )
    rows = builder_module._reconstruct_membership_rows(
        current_constituents=current_constituents,
        change_events=change_events,
        start_date=membership_start,
        end_date=membership_end,
        index_name=INDEX_NAME,
    )
    return MembershipDryRunPlan(
        current_constituent_count=len(current_constituents),
        change_event_count=len(change_events),
        rows=rows,
    )


def _run_prices_phase(
    *,
    dry_run: bool,
    price_tail_days: int,
    requested_tickers: Sequence[str],
    limit: int | None,
    membership_rows: Sequence[dict[str, object]] | None,
) -> PriceRunSummary:
    plan = _build_price_plan(
        price_tail_days=price_tail_days,
        requested_tickers=requested_tickers,
        limit=limit,
        membership_rows=membership_rows,
    )
    summary = PriceRunSummary(planned_intervals=len(plan))
    _log_price_plan(plan=plan, dry_run=dry_run)

    if dry_run:
        logger.info(
            "prices dry-run summary planned_intervals={} fetch_intervals={} skipped_intervals={} estimated_rows={}",
            len(plan),
            sum(1 for item in plan if item.status == "fetch"),
            sum(1 for item in plan if item.status == "skip"),
            sum(item.estimated_rows for item in plan if item.status == "fetch"),
        )
        return summary

    polygon = PolygonDataSource()

    for item in plan:
        if item.status == "skip":
            summary.skipped_intervals += 1
            continue

        logger.info(
            "fetching prices ticker={} effective_date={} end_date={} est_rows={}",
            item.ticker,
            item.effective_date,
            item.end_date,
            item.estimated_rows,
        )
        try:
            frame = polygon.fetch_historical(
                tickers=[item.ticker],
                start_date=item.effective_date,
                end_date=item.end_date,
            )
        except Exception as exc:
            summary.failed_intervals += 1
            summary.failures.append(
                BackfillFailure(
                    ticker=item.ticker,
                    effective_date=item.effective_date,
                    end_date=item.end_date,
                    error_reason=str(exc),
                ),
            )
            logger.error(
                "price fetch failed ticker={} effective_date={} end_date={} error={}",
                item.ticker,
                item.effective_date,
                item.end_date,
                exc,
            )
            continue

        if frame.empty:
            summary.failed_intervals += 1
            summary.failures.append(
                BackfillFailure(
                    ticker=item.ticker,
                    effective_date=item.effective_date,
                    end_date=item.end_date,
                    error_reason="empty_result",
                ),
            )
            logger.warning(
                "price fetch returned no rows ticker={} effective_date={} end_date={}",
                item.ticker,
                item.effective_date,
                item.end_date,
            )
            continue

        summary.successful_intervals += 1
        summary.fetched_rows += len(frame)
        logger.info(
            "price fetch completed ticker={} effective_date={} end_date={} rows={}",
            item.ticker,
            item.effective_date,
            item.end_date,
            len(frame),
        )

    logger.info(
        "prices summary planned_intervals={} successful_intervals={} skipped_intervals={} failed_intervals={} fetched_rows={}",
        summary.planned_intervals,
        summary.successful_intervals,
        summary.skipped_intervals,
        summary.failed_intervals,
        summary.fetched_rows,
    )
    return summary


def _run_fundamentals_phase(
    *,
    dry_run: bool,
    requested_tickers: Sequence[str],
    limit: int | None,
    membership_rows: Sequence[dict[str, object]] | None,
) -> FundamentalRunSummary:
    plan = _build_fundamental_plan(
        requested_tickers=requested_tickers,
        limit=limit,
        membership_rows=membership_rows,
    )
    summary = FundamentalRunSummary(planned_intervals=len(plan))
    _log_fundamental_plan(plan=plan, dry_run=dry_run)

    if dry_run:
        logger.info(
            "fundamentals dry-run summary planned_intervals={} fetch_intervals={} skipped_intervals={} estimated_rows={}",
            len(plan),
            sum(1 for item in plan if item.status == "fetch"),
            sum(1 for item in plan if item.status == "skip"),
            sum(item.estimated_rows for item in plan if item.status == "fetch"),
        )
        return summary

    source = FMPDataSource()
    for item in plan:
        if item.status == "skip":
            summary.skipped_intervals += 1
            continue

        logger.info(
            "fetching fundamentals ticker={} effective_date={} end_date={} est_rows={}",
            item.ticker,
            item.effective_date,
            item.end_date,
            item.estimated_rows,
        )
        try:
            frame = source.fetch_historical(
                tickers=[item.ticker],
                start_date=item.effective_date,
                end_date=item.end_date,
            )
        except Exception as exc:
            summary.failed_intervals += 1
            summary.failures.append(
                BackfillFailure(
                    ticker=item.ticker,
                    effective_date=item.effective_date,
                    end_date=item.end_date,
                    error_reason=str(exc),
                ),
            )
            logger.error(
                "fundamental fetch failed ticker={} effective_date={} end_date={} error={}",
                item.ticker,
                item.effective_date,
                item.end_date,
                exc,
            )
            continue

        if frame.empty:
            summary.failed_intervals += 1
            summary.failures.append(
                BackfillFailure(
                    ticker=item.ticker,
                    effective_date=item.effective_date,
                    end_date=item.end_date,
                    error_reason="empty_result",
                ),
            )
            logger.warning(
                "fundamental fetch returned no rows ticker={} effective_date={} end_date={}",
                item.ticker,
                item.effective_date,
                item.end_date,
            )
            continue

        summary.successful_intervals += 1
        summary.fetched_rows += len(frame)
        logger.info(
            "fundamental fetch completed ticker={} effective_date={} end_date={} rows={}",
            item.ticker,
            item.effective_date,
            item.end_date,
            len(frame),
        )

    logger.info(
        "fundamentals summary planned_intervals={} successful_intervals={} skipped_intervals={} failed_intervals={} fetched_rows={}",
        summary.planned_intervals,
        summary.successful_intervals,
        summary.skipped_intervals,
        summary.failed_intervals,
        summary.fetched_rows,
    )
    return summary


def _build_price_plan(
    *,
    price_tail_days: int,
    requested_tickers: Sequence[str],
    limit: int | None,
    membership_rows: Sequence[dict[str, object]] | None,
) -> list[PricePlanItem]:
    source_rows = list(membership_rows) if membership_rows is not None else _load_ended_membership_rows()
    intervals = _build_price_intervals(
        membership_rows=source_rows,
        price_tail_days=price_tail_days,
        requested_tickers=requested_tickers,
        limit=limit,
        max_end_date=current_market_data_end_date(),
    )
    coverage_by_ticker = _load_price_coverage([interval.ticker for interval in intervals])

    items: list[PricePlanItem] = []
    for interval in intervals:
        coverage = coverage_by_ticker.get(interval.ticker)
        if _is_fully_covered(interval=interval, coverage=coverage):
            status = "skip"
            reason = (
                f"existing coverage {coverage.min_date}->{coverage.max_date}"
                if coverage is not None
                else "existing coverage"
            )
        else:
            status = "fetch"
            if coverage is None or coverage.row_count == 0:
                reason = "no_existing_prices"
            else:
                reason = f"partial coverage {coverage.min_date}->{coverage.max_date}"
        items.append(
            PricePlanItem(
                ticker=interval.ticker,
                effective_date=interval.effective_date,
                end_date=interval.end_date,
                estimated_rows=_estimate_business_rows(interval.effective_date, interval.end_date),
                status=status,
                reason=reason,
            ),
        )

    return items


def _build_fundamental_plan(
    *,
    requested_tickers: Sequence[str],
    limit: int | None,
    membership_rows: Sequence[dict[str, object]] | None,
) -> list[FundamentalPlanItem]:
    source_rows = list(membership_rows) if membership_rows is not None else _load_ended_membership_rows()
    intervals = _build_price_intervals(
        membership_rows=source_rows,
        price_tail_days=0,
        requested_tickers=requested_tickers,
        limit=limit,
        max_end_date=date.today(),
    )
    coverage_by_ticker = _load_fundamental_coverage([interval.ticker for interval in intervals])

    items: list[FundamentalPlanItem] = []
    for interval in intervals:
        coverage = coverage_by_ticker.get(interval.ticker)
        missing_metrics = _missing_fundamental_metrics(coverage)
        if _is_fundamentals_fully_covered(interval=interval, coverage=coverage):
            status = "skip"
            reason = (
                f"existing coverage {coverage.min_date}->{coverage.max_date}"
                if coverage is not None
                else "existing coverage"
            )
        else:
            status = "fetch"
            if coverage is None or coverage.row_count == 0:
                reason = "no_existing_fundamentals"
            elif missing_metrics:
                reason = f"missing metrics {','.join(missing_metrics[:4])}"
            else:
                reason = f"partial coverage {coverage.min_date}->{coverage.max_date}"

        items.append(
            FundamentalPlanItem(
                ticker=interval.ticker,
                effective_date=interval.effective_date,
                end_date=interval.end_date,
                estimated_rows=_estimate_fundamental_rows(interval.effective_date, interval.end_date),
                status=status,
                reason=reason,
            ),
        )

    return items


def _load_ended_membership_rows() -> list[dict[str, object]]:
    session_factory = get_session_factory()
    statement = (
        sa.select(
            UniverseMembership.ticker.label("ticker"),
            UniverseMembership.effective_date.label("effective_date"),
            UniverseMembership.end_date.label("end_date"),
            UniverseMembership.reason.label("reason"),
        )
        .where(
            UniverseMembership.index_name == INDEX_NAME,
            UniverseMembership.end_date.is_not(None),
        )
        .order_by(UniverseMembership.ticker, UniverseMembership.effective_date)
    )
    with session_factory() as session:
        return [dict(row) for row in session.execute(statement).mappings().all()]


def _build_price_intervals(
    *,
    membership_rows: Sequence[dict[str, object]],
    price_tail_days: int,
    requested_tickers: Sequence[str],
    limit: int | None,
    max_end_date: date,
) -> list[PriceInterval]:
    requested = set(requested_tickers)
    intervals_by_ticker: dict[str, list[PriceInterval]] = {}

    for row in membership_rows:
        raw_end_date = row.get("end_date")
        raw_effective_date = row.get("effective_date")
        ticker = _normalize_ticker(row.get("ticker"))
        if not ticker or raw_end_date is None or not isinstance(raw_effective_date, date):
            continue

        if requested and ticker not in requested:
            continue

        interval_end = min(raw_end_date + timedelta(days=price_tail_days), max_end_date)
        interval = PriceInterval(
            ticker=ticker,
            effective_date=raw_effective_date,
            end_date=interval_end,
        )
        if interval.effective_date > interval.end_date:
            continue
        intervals_by_ticker.setdefault(ticker, []).append(interval)

    candidate_tickers = sorted(intervals_by_ticker)
    if limit is not None:
        candidate_tickers = candidate_tickers[:limit]

    merged: list[PriceInterval] = []
    for ticker in candidate_tickers:
        merged.extend(_merge_intervals(intervals_by_ticker[ticker]))

    return merged


def _merge_intervals(intervals: Sequence[PriceInterval]) -> list[PriceInterval]:
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda item: (item.effective_date, item.end_date))
    merged: list[PriceInterval] = [ordered[0]]

    for interval in ordered[1:]:
        last = merged[-1]
        if interval.effective_date <= last.end_date + timedelta(days=1):
            merged[-1] = PriceInterval(
                ticker=last.ticker,
                effective_date=last.effective_date,
                end_date=max(last.end_date, interval.end_date),
            )
            continue
        merged.append(interval)

    return merged


def _load_price_coverage(tickers: Sequence[str]) -> dict[str, PriceCoverage]:
    normalized = tuple(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers if ticker))
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

    return {
        str(row["ticker"]): PriceCoverage(
            min_date=row["min_date"],
            max_date=row["max_date"],
            row_count=int(row["row_count"]),
        )
        for row in rows
    }


def _load_fundamental_coverage(tickers: Sequence[str]) -> dict[str, FundamentalCoverage]:
    normalized = tuple(dict.fromkeys(_normalize_ticker(ticker) for ticker in tickers if ticker))
    if not normalized:
        return {}

    session_factory = get_session_factory()
    statement = (
        sa.select(
            FundamentalsPIT.ticker.label("ticker"),
            FundamentalsPIT.metric_name.label("metric_name"),
            FundamentalsPIT.event_time.label("event_time"),
        )
        .where(FundamentalsPIT.ticker.in_(normalized))
        .order_by(FundamentalsPIT.ticker, FundamentalsPIT.event_time, FundamentalsPIT.metric_name)
    )
    with session_factory() as session:
        rows = session.execute(statement).mappings().all()

    coverage_by_ticker: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        coverage_by_ticker.setdefault(str(row["ticker"]), []).append(dict(row))

    summarized: dict[str, FundamentalCoverage] = {}
    for ticker, ticker_rows in coverage_by_ticker.items():
        event_dates = [
            row["event_time"]
            for row in ticker_rows
            if isinstance(row.get("event_time"), date)
        ]
        metric_names = frozenset(
            _normalize_ticker(row["metric_name"]).replace("-", "_").lower()
            for row in ticker_rows
            if row.get("metric_name")
        )
        summarized[ticker] = FundamentalCoverage(
            min_date=min(event_dates) if event_dates else None,
            max_date=max(event_dates) if event_dates else None,
            row_count=len(ticker_rows),
            metric_names=metric_names,
        )

    return summarized


def _is_fully_covered(*, interval: PriceInterval, coverage: PriceCoverage | None) -> bool:
    if coverage is None or coverage.row_count <= 0:
        return False
    if coverage.min_date is None or coverage.max_date is None:
        return False
    return coverage.min_date <= interval.effective_date and coverage.max_date >= interval.end_date


def _is_fundamentals_fully_covered(
    *,
    interval: PriceInterval,
    coverage: FundamentalCoverage | None,
) -> bool:
    if coverage is None or coverage.row_count <= 0:
        return False
    if coverage.min_date is None or coverage.max_date is None:
        return False
    if _missing_fundamental_metrics(coverage):
        return False
    return coverage.min_date <= interval.effective_date and coverage.max_date >= interval.end_date


def _missing_fundamental_metrics(coverage: FundamentalCoverage | None) -> list[str]:
    if coverage is None:
        return sorted(REQUIRED_FUNDAMENTAL_COVERAGE_METRICS)
    return sorted(set(REQUIRED_FUNDAMENTAL_COVERAGE_METRICS) - set(coverage.metric_names))


def _estimate_business_rows(start_date: date, end_date: date) -> int:
    if end_date < start_date:
        return 0
    return int(pd.bdate_range(start=start_date, end=end_date).size)


def _estimate_fundamental_rows(start_date: date, end_date: date) -> int:
    if end_date < start_date:
        return 0

    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    quarter_count = max(1, total_months // 3 + 1)
    return quarter_count * len(INGESTED_FUNDAMENTAL_METRICS)


def _log_price_plan(*, plan: Sequence[PricePlanItem], dry_run: bool) -> None:
    mode = "prices dry-run plan" if dry_run else "prices plan"
    if not plan:
        logger.info("{} is empty", mode)
        return

    for item in plan:
        logger.info(
            "{} status={} ticker={} effective_date={} end_date={} est_rows={} reason={}",
            mode,
            item.status,
            item.ticker,
            item.effective_date,
            item.end_date,
            item.estimated_rows,
            item.reason,
        )


def _log_fundamental_plan(*, plan: Sequence[FundamentalPlanItem], dry_run: bool) -> None:
    mode = "fundamentals dry-run plan" if dry_run else "fundamentals plan"
    if not plan:
        logger.info("{} is empty", mode)
        return

    for item in plan:
        logger.info(
            "{} status={} ticker={} effective_date={} end_date={} est_rows={} reason={}",
            mode,
            item.status,
            item.ticker,
            item.effective_date,
            item.end_date,
            item.estimated_rows,
            item.reason,
        )


def _write_failures_csv(path: Path, failures: Sequence[BackfillFailure]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ticker", "effective_date", "end_date", "error_reason"],
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "ticker": failure.ticker,
                    "effective_date": failure.effective_date.isoformat(),
                    "end_date": failure.end_date.isoformat(),
                    "error_reason": failure.error_reason,
                },
            )
    logger.info("wrote {} failures to {}", len(failures), path)


def _ensure_runtime_imports() -> None:
    global pd
    global sa
    global logger
    global configure_logging
    global current_market_data_end_date
    global ensure_tables_exist
    global get_session_factory
    global FundamentalsPIT
    global StockPrice
    global UniverseMembership
    global FMPDataSource
    global PolygonDataSource
    global INGESTED_FUNDAMENTAL_METRICS
    global REQUIRED_FUNDAMENTAL_COVERAGE_METRICS
    global builder_module

    if logger is not None:
        return

    import pandas as imported_pandas
    import sqlalchemy as imported_sa
    from loguru import logger as imported_logger

    from _data_ops import (
        configure_logging as imported_configure_logging,
        current_market_data_end_date as imported_current_market_data_end_date,
        ensure_tables_exist as imported_ensure_tables_exist,
    )
    from src.data.db.models import FundamentalsPIT as imported_FundamentalsPIT
    from src.data.db.models import StockPrice as imported_StockPrice
    from src.data.db.models import UniverseMembership as imported_UniverseMembership
    from src.data.db.session import get_session_factory as imported_get_session_factory
    from src.data.sources.fmp import (
        FMPDataSource as imported_FMPDataSource,
        INGESTED_FUNDAMENTAL_METRICS as imported_INGESTED_FUNDAMENTAL_METRICS,
    )
    from src.data.sources.polygon import PolygonDataSource as imported_PolygonDataSource
    import src.universe.builder as imported_builder_module

    pd = imported_pandas
    sa = imported_sa
    logger = imported_logger
    configure_logging = imported_configure_logging
    current_market_data_end_date = imported_current_market_data_end_date
    ensure_tables_exist = imported_ensure_tables_exist
    get_session_factory = imported_get_session_factory
    FundamentalsPIT = imported_FundamentalsPIT
    StockPrice = imported_StockPrice
    UniverseMembership = imported_UniverseMembership
    FMPDataSource = imported_FMPDataSource
    PolygonDataSource = imported_PolygonDataSource
    INGESTED_FUNDAMENTAL_METRICS = tuple(sorted(imported_INGESTED_FUNDAMENTAL_METRICS))
    REQUIRED_FUNDAMENTAL_COVERAGE_METRICS = tuple(
        metric_name
        for metric_name in INGESTED_FUNDAMENTAL_METRICS
        if metric_name not in {"annual_dividend", "dividend_per_share"}
    )
    builder_module = imported_builder_module


if __name__ == "__main__":
    raise SystemExit(main())
