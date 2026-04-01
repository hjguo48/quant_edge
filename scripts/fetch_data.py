from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

if TYPE_CHECKING:
    import pandas as pd
    from src.data.quality import DataQualityChecker

pd = None
logger = None
configure_logging = None
current_market_data_end_date = None
decimal_to_float_frame = None
ensure_selected_constituents = None
ensure_tables_exist = None
fetch_sp500_constituents = None
filter_constituents_by_range = None
get_corporate_action_coverage = None
get_fundamental_coverage = None
get_macro_coverage = None
get_price_coverage = None
parse_date = None
parse_tickers = None
summarize_quality_reports = None
summarize_row_counts = None
upsert_stocks = None
fetch_corporate_actions = None
DataQualityChecker = None
FMPDataSource = None
FredDataSource = None
PolygonDataSource = None
backfill_universe_membership = None

MACRO_SERIES_IDS = ("VIXCLS", "DGS10", "DGS2", "BAA10Y", "AAA10Y", "FEDFUNDS")


@dataclass
class FetchSummary:
    stock_rows: int = 0
    price_rows: int = 0
    macro_rows: int = 0
    fundamental_rows: int = 0
    corporate_action_rows: int = 0
    universe_rows: int = 0
    skipped_price_tickers: list[str] = field(default_factory=list)
    skipped_fundamental_tickers: list[str] = field(default_factory=list)
    skipped_action_tickers: list[str] = field(default_factory=list)
    failed_price_tickers: list[str] = field(default_factory=list)
    failed_fundamental_tickers: list[str] = field(default_factory=list)
    failed_action_tickers: list[str] = field(default_factory=list)
    failed_components: list[str] = field(default_factory=list)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    _ensure_runtime_imports()
    configure_logging("fetch_data")
    ensure_tables_exist()

    start_date = parse_date(args.start_date)
    requested_end_date = parse_date(args.end_date) if args.end_date else None
    price_end_date = (
        current_market_data_end_date()
        if requested_end_date is None
        else min(requested_end_date, current_market_data_end_date())
    )
    fundamentals_end_date = (
        date.today()
        if requested_end_date is None
        else min(requested_end_date, date.today())
    )
    if start_date > fundamentals_end_date:
        logger.error(
            "start_date {} is after the effective fundamentals end_date {}",
            start_date,
            fundamentals_end_date,
        )
        return 1

    checker = DataQualityChecker()
    summary = FetchSummary()

    constituents = fetch_sp500_constituents()
    selected_constituents = ensure_selected_constituents(constituents, parse_tickers(args.tickers))
    selected_constituents = filter_constituents_by_range(
        selected_constituents,
        ticker_start=args.ticker_start,
        ticker_end=args.ticker_end,
    )
    selected_tickers = selected_constituents["ticker"].tolist()
    if not selected_tickers:
        logger.error("no tickers selected for ingestion")
        return 1

    logger.info(
        "selected {} tickers start={} price_end={} fundamentals_end={} dry_run={}",
        len(selected_tickers),
        start_date,
        price_end_date,
        fundamentals_end_date,
        args.dry_run,
    )

    if args.dry_run:
        _log_dry_run(
            tickers=selected_tickers,
            start_date=start_date,
            price_end_date=price_end_date,
            fundamentals_end_date=fundamentals_end_date,
            skip_prices=args.skip_prices,
            skip_fundamentals=args.skip_fundamentals,
            skip_macro=args.skip_macro,
            skip_corporate_actions=args.skip_corporate_actions,
            skip_universe=args.skip_universe,
            partial_run=_is_partial_run(args),
        )
        return 0

    summary.stock_rows = upsert_stocks(selected_constituents)

    if not args.skip_prices:
        _fetch_prices(
            tickers=selected_tickers,
            start_date=start_date,
            end_date=price_end_date,
            checker=checker,
            summary=summary,
            request_interval=args.polygon_request_interval,
        )
    else:
        logger.info("skipping price ingestion by request")

    if not args.skip_macro:
        _fetch_macro(
            start_date=start_date,
            end_date=fundamentals_end_date,
            checker=checker,
            summary=summary,
        )
    else:
        logger.info("skipping macro ingestion by request")

    if not args.skip_fundamentals:
        _fetch_fundamentals(
            tickers=selected_tickers,
            start_date=start_date,
            end_date=fundamentals_end_date,
            checker=checker,
            summary=summary,
            request_interval=args.fmp_request_interval,
        )
    else:
        logger.info("skipping fundamentals ingestion by request")

    if not args.skip_corporate_actions:
        _fetch_corporate_actions(
            tickers=selected_tickers,
            start_date=start_date,
            end_date=price_end_date,
            summary=summary,
            request_interval=args.polygon_request_interval,
        )
    else:
        logger.info("skipping corporate action ingestion by request")

    if args.skip_universe:
        logger.info("skipping universe membership backfill by request")
    elif _is_partial_run(args):
        logger.info("skipping universe membership backfill for partial ticker run")
    else:
        try:
            summary.universe_rows = backfill_universe_membership(
                start_date,
                price_end_date,
                index_name="SP500",
            )
        except Exception as exc:
            logger.opt(exception=exc).error("failed to backfill historical universe membership")
            summary.failed_components.append("universe_backfill")

    _log_summary(summary)
    summarize_row_counts()
    return 0 if not _has_failures(summary) else 2


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch and persist Week 2 historical data with resumable backfill windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tickers",
        help="Optional comma-separated tickers for a partial run. Combined with ticker range filters if provided.",
    )
    parser.add_argument(
        "--ticker-start",
        help="Optional inclusive ticker lower bound for batching large backfills alphabetically.",
    )
    parser.add_argument(
        "--ticker-end",
        help="Optional inclusive ticker upper bound for batching large backfills alphabetically.",
    )
    parser.add_argument(
        "--start-date",
        default="2018-01-01",
        help="Historical backfill start date in ISO format.",
    )
    parser.add_argument(
        "--end-date",
        help="Optional end date in ISO format. Defaults to the latest available provider date.",
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip Polygon daily price ingestion.",
    )
    parser.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Skip FMP PIT fundamentals ingestion.",
    )
    parser.add_argument(
        "--skip-macro",
        action="store_true",
        help="Skip FRED macro series ingestion.",
    )
    parser.add_argument(
        "--skip-corporate-actions",
        action="store_true",
        help="Skip Polygon split and dividend ingestion.",
    )
    parser.add_argument(
        "--skip-universe",
        action="store_true",
        help="Skip historical S&P 500 universe membership backfill.",
    )
    parser.add_argument(
        "--polygon-request-interval",
        type=float,
        default=12.5,
        help="Minimum seconds between Polygon requests. Use a lower value only on paid plans.",
    )
    parser.add_argument(
        "--fmp-request-interval",
        type=float,
        default=0.5,
        help="Minimum seconds between FMP requests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned work without mutating the database or calling providers.",
    )
    return parser.parse_args(argv)


def _log_dry_run(
    *,
    tickers: list[str],
    start_date: date,
    price_end_date: date,
    fundamentals_end_date: date,
    skip_prices: bool,
    skip_fundamentals: bool,
    skip_macro: bool,
    skip_corporate_actions: bool,
    skip_universe: bool,
    partial_run: bool,
) -> None:
    logger.info("dry run only; no provider calls will be executed")
    logger.info("would upsert {} stock records", len(tickers))
    if not skip_prices:
        coverage = get_price_coverage(tickers)
        fully_covered = sum(
            1
            for ticker in tickers
            if not _uncovered_windows(coverage.get(ticker), start_date, price_end_date)
        )
        logger.info(
            "would inspect {} tickers for price resume logic; {} already appear fully covered",
            len(tickers),
            fully_covered,
        )
    if not skip_macro:
        coverage = get_macro_coverage(MACRO_SERIES_IDS)
        fully_covered = sum(
            1
            for series_id in MACRO_SERIES_IDS
            if not _uncovered_windows(coverage.get(series_id), start_date, fundamentals_end_date)
        )
        logger.info(
            "would inspect {} macro series; {} already appear fully covered",
            len(MACRO_SERIES_IDS),
            fully_covered,
        )
    if not skip_fundamentals:
        coverage = get_fundamental_coverage(tickers)
        fully_covered = sum(
            1
            for ticker in tickers
            if not _uncovered_windows(coverage.get(ticker), start_date, fundamentals_end_date)
        )
        logger.info(
            "would inspect {} tickers for PIT fundamentals; {} already appear fully covered",
            len(tickers),
            fully_covered,
        )
    if not skip_corporate_actions:
        coverage = get_corporate_action_coverage(tickers)
        fully_covered = sum(
            1
            for ticker in tickers
            if not _uncovered_windows(coverage.get(ticker), start_date, price_end_date)
        )
        logger.info(
            "would inspect {} tickers for split/dividend resume logic; {} already appear fully covered",
            len(tickers),
            fully_covered,
        )
    if not skip_universe and not partial_run:
        logger.info(
            "would backfill historical SP500 universe membership between {} and {}",
            start_date,
            price_end_date,
        )


def _fetch_prices(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    checker: DataQualityChecker,
    summary: FetchSummary,
    request_interval: float,
) -> None:
    source = PolygonDataSource(min_request_interval=request_interval)
    coverage = get_price_coverage(tickers)

    for index, ticker in enumerate(tickers, start=1):
        logger.info("[{}/{}] ingesting prices for {}", index, len(tickers), ticker)
        windows = _uncovered_windows(coverage.get(ticker), start_date, end_date)
        if not windows:
            summary.skipped_price_tickers.append(ticker)
            logger.info("skipping {} because requested price window is already covered", ticker)
            continue

        try:
            frames: list[pd.DataFrame] = []
            empty_windows: list[str] = []
            for window_start, window_end in windows:
                logger.info("requesting {} prices for {}->{}", ticker, window_start, window_end)
                frame = source.fetch_historical([ticker], window_start, window_end)
                if frame.empty:
                    empty_windows.append(f"{window_start}->{window_end}")
                frames.append(frame)

            non_empty_frames = [frame for frame in frames if not frame.empty]
            combined = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame()
            if combined.empty:
                logger.warning(
                    "requested uncovered price windows for {} but Polygon returned no rows: {}",
                    ticker,
                    ", ".join(f"{window_start}->{window_end}" for window_start, window_end in windows),
                )
                summary.failed_price_tickers.append(ticker)
                continue

            if empty_windows:
                logger.warning(
                    "{} returned no rows for some requested price windows: {}",
                    ticker,
                    ", ".join(empty_windows),
                )

            summary.price_rows += len(combined)
            summarize_quality_reports(
                checker,
                dataset_name=f"prices:{ticker}",
                frame=decimal_to_float_frame(combined),
                validate_prices=True,
            )
        except Exception as exc:
            logger.opt(exception=exc).error("price ingestion failed for {}", ticker)
            summary.failed_price_tickers.append(ticker)


def _fetch_macro(
    *,
    start_date: date,
    end_date: date,
    checker: DataQualityChecker,
    summary: FetchSummary,
) -> None:
    source = FredDataSource()
    coverage = get_macro_coverage(MACRO_SERIES_IDS)

    for series_id in MACRO_SERIES_IDS:
        windows = _uncovered_windows(coverage.get(series_id), start_date, end_date)
        if not windows:
            logger.info("skipping macro series {} because requested window is already covered", series_id)
            continue

        try:
            for window_start, window_end in windows:
                logger.info("requesting macro series {} for {}->{}", series_id, window_start, window_end)
                frame = source.fetch_historical([series_id], window_start, window_end)
                summary.macro_rows += len(frame)
                if frame.empty:
                    logger.warning(
                        "macro series {} returned no rows for requested window {}->{}",
                        series_id,
                        window_start,
                        window_end,
                    )
                    continue
                summarize_quality_reports(
                    checker,
                    dataset_name=f"macro:{series_id}",
                    frame=decimal_to_float_frame(frame),
                    validate_prices=False,
                )
        except Exception as exc:
            logger.opt(exception=exc).error("macro ingestion failed for {}", series_id)
            summary.failed_components.append(f"macro:{series_id}")


def _fetch_fundamentals(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    checker: DataQualityChecker,
    summary: FetchSummary,
    request_interval: float,
) -> None:
    source = FMPDataSource(min_request_interval=request_interval)
    coverage = get_fundamental_coverage(tickers)

    for index, ticker in enumerate(tickers, start=1):
        logger.info("[{}/{}] ingesting fundamentals for {}", index, len(tickers), ticker)
        windows = _uncovered_windows(coverage.get(ticker), start_date, end_date)
        if not windows:
            summary.skipped_fundamental_tickers.append(ticker)
            logger.info("skipping {} because requested fundamentals window is already covered", ticker)
            continue

        try:
            for window_start, window_end in windows:
                logger.info("requesting {} fundamentals for {}->{}", ticker, window_start, window_end)
                frame = source.fetch_historical([ticker], window_start, window_end)
                if frame.empty:
                    logger.warning(
                        "FMP returned no PIT fundamentals for {} in requested window {}->{}",
                        ticker,
                        window_start,
                        window_end,
                    )
                    continue
                summary.fundamental_rows += len(frame)
                summarize_quality_reports(
                    checker,
                    dataset_name=f"fundamentals:{ticker}",
                    frame=decimal_to_float_frame(frame),
                    validate_prices=False,
                )
        except Exception as exc:
            logger.opt(exception=exc).error("fundamental ingestion failed for {}", ticker)
            summary.failed_fundamental_tickers.append(ticker)


def _fetch_corporate_actions(
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    summary: FetchSummary,
    request_interval: float,
) -> None:
    coverage = get_corporate_action_coverage(tickers)

    for index, ticker in enumerate(tickers, start=1):
        logger.info("[{}/{}] ingesting corporate actions for {}", index, len(tickers), ticker)
        windows = _uncovered_windows(coverage.get(ticker), start_date, end_date)
        if not windows:
            summary.skipped_action_tickers.append(ticker)
            logger.info("skipping {} because requested corporate action window is already covered", ticker)
            continue

        try:
            for window_start, window_end in windows:
                logger.info(
                    "requesting {} corporate actions for {}->{}",
                    ticker,
                    window_start,
                    window_end,
                )
                frame = fetch_corporate_actions(
                    [ticker],
                    window_start,
                    window_end,
                    min_request_interval=request_interval,
                )
                if frame.empty:
                    logger.warning(
                        "Polygon returned no corporate actions for {} in requested window {}->{}",
                        ticker,
                        window_start,
                        window_end,
                    )
                    continue
                summary.corporate_action_rows += len(frame)
        except Exception as exc:
            logger.opt(exception=exc).error("corporate action ingestion failed for {}", ticker)
            summary.failed_action_tickers.append(ticker)


def _log_summary(summary: FetchSummary) -> None:
    logger.info(
        "ingestion summary stocks={} prices={} macro={} fundamentals={} corporate_actions={} universe_rows={}",
        summary.stock_rows,
        summary.price_rows,
        summary.macro_rows,
        summary.fundamental_rows,
        summary.corporate_action_rows,
        summary.universe_rows,
    )
    if summary.skipped_price_tickers:
        logger.info("skipped price tickers already covered: {}", ",".join(summary.skipped_price_tickers))
    if summary.skipped_fundamental_tickers:
        logger.info(
            "skipped fundamentals tickers already covered: {}",
            ",".join(summary.skipped_fundamental_tickers),
        )
    if summary.skipped_action_tickers:
        logger.info(
            "skipped corporate action tickers already covered: {}",
            ",".join(summary.skipped_action_tickers),
        )
    if summary.failed_price_tickers:
        logger.warning("failed price tickers: {}", ",".join(summary.failed_price_tickers))
    if summary.failed_fundamental_tickers:
        logger.warning("failed fundamental tickers: {}", ",".join(summary.failed_fundamental_tickers))
    if summary.failed_action_tickers:
        logger.warning("failed corporate action tickers: {}", ",".join(summary.failed_action_tickers))
    if summary.failed_components:
        logger.warning("failed components: {}", ",".join(summary.failed_components))


def _has_failures(summary: FetchSummary) -> bool:
    return any(
        [
            summary.failed_price_tickers,
            summary.failed_fundamental_tickers,
            summary.failed_action_tickers,
            summary.failed_components,
        ],
    )


def _is_partial_run(args: argparse.Namespace) -> bool:
    return bool(args.tickers or args.ticker_start or args.ticker_end)


def _uncovered_windows(
    coverage: Any | None,
    start_date: date,
    end_date: date,
) -> list[tuple[date, date]]:
    if start_date > end_date:
        return []
    if coverage is None or getattr(coverage, "row_count", 0) == 0:
        return [(start_date, end_date)]

    min_date = getattr(coverage, "min_date", None)
    max_date = getattr(coverage, "max_date", None)
    if min_date is None or max_date is None:
        return [(start_date, end_date)]

    windows: list[tuple[date, date]] = []
    if min_date > start_date:
        windows.append((start_date, min_date - timedelta(days=1)))
    if max_date < end_date:
        windows.append((max_date + timedelta(days=1), end_date))
    return [(window_start, window_end) for window_start, window_end in windows if window_start <= window_end]


def _ensure_runtime_imports() -> None:
    global DataQualityChecker
    global FMPDataSource
    global FredDataSource
    global PolygonDataSource
    global backfill_universe_membership
    global configure_logging
    global current_market_data_end_date
    global decimal_to_float_frame
    global ensure_selected_constituents
    global ensure_tables_exist
    global fetch_corporate_actions
    global fetch_sp500_constituents
    global filter_constituents_by_range
    global get_corporate_action_coverage
    global get_fundamental_coverage
    global get_macro_coverage
    global get_price_coverage
    global logger
    global parse_date
    global parse_tickers
    global pd
    global summarize_quality_reports
    global summarize_row_counts
    global upsert_stocks

    if logger is not None:
        return

    import pandas as imported_pandas
    from loguru import logger as imported_logger

    from _data_ops import (
        configure_logging as imported_configure_logging,
        current_market_data_end_date as imported_current_market_data_end_date,
        decimal_to_float_frame as imported_decimal_to_float_frame,
        ensure_selected_constituents as imported_ensure_selected_constituents,
        ensure_tables_exist as imported_ensure_tables_exist,
        fetch_sp500_constituents as imported_fetch_sp500_constituents,
        filter_constituents_by_range as imported_filter_constituents_by_range,
        get_corporate_action_coverage as imported_get_corporate_action_coverage,
        get_fundamental_coverage as imported_get_fundamental_coverage,
        get_macro_coverage as imported_get_macro_coverage,
        get_price_coverage as imported_get_price_coverage,
        parse_date as imported_parse_date,
        parse_tickers as imported_parse_tickers,
        summarize_quality_reports as imported_summarize_quality_reports,
        summarize_row_counts as imported_summarize_row_counts,
        upsert_stocks as imported_upsert_stocks,
    )
    from src.data.corporate_actions import fetch_corporate_actions as imported_fetch_corporate_actions
    from src.data.quality import DataQualityChecker as imported_DataQualityChecker
    from src.data.sources.fmp import FMPDataSource as imported_FMPDataSource
    from src.data.sources.fred import FredDataSource as imported_FredDataSource
    from src.data.sources.polygon import PolygonDataSource as imported_PolygonDataSource
    from src.universe.builder import (
        backfill_universe_membership as imported_backfill_universe_membership,
    )

    pd = imported_pandas
    logger = imported_logger
    configure_logging = imported_configure_logging
    current_market_data_end_date = imported_current_market_data_end_date
    decimal_to_float_frame = imported_decimal_to_float_frame
    ensure_selected_constituents = imported_ensure_selected_constituents
    ensure_tables_exist = imported_ensure_tables_exist
    fetch_sp500_constituents = imported_fetch_sp500_constituents
    filter_constituents_by_range = imported_filter_constituents_by_range
    get_corporate_action_coverage = imported_get_corporate_action_coverage
    get_fundamental_coverage = imported_get_fundamental_coverage
    get_macro_coverage = imported_get_macro_coverage
    get_price_coverage = imported_get_price_coverage
    parse_date = imported_parse_date
    parse_tickers = imported_parse_tickers
    summarize_quality_reports = imported_summarize_quality_reports
    summarize_row_counts = imported_summarize_row_counts
    upsert_stocks = imported_upsert_stocks
    fetch_corporate_actions = imported_fetch_corporate_actions
    DataQualityChecker = imported_DataQualityChecker
    FMPDataSource = imported_FMPDataSource
    FredDataSource = imported_FredDataSource
    PolygonDataSource = imported_PolygonDataSource
    backfill_universe_membership = imported_backfill_universe_membership


if __name__ == "__main__":
    sys.exit(main())
