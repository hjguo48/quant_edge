from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import sys

from loguru import logger

from _data_ops import (
    configure_logging,
    current_market_data_end_date,
    decimal_to_float_frame,
    ensure_tables_exist,
    get_latest_price_dates,
    get_tracked_tickers,
    summarize_quality_reports,
    summarize_row_counts,
)
from src.data.corporate_actions import fetch_corporate_actions
from src.data.quality import DataQualityChecker
from src.data.sources.fmp import FMPDataSource
from src.data.sources.fred import FredDataSource
from src.data.sources.polygon import PolygonDataSource


@dataclass
class UpdateSummary:
    price_rows: int = 0
    macro_rows: int = 0
    fundamental_rows: int = 0
    corporate_action_rows: int = 0
    skipped_price_tickers: list[str] = field(default_factory=list)
    failed_price_tickers: list[str] = field(default_factory=list)
    failed_fundamental_tickers: list[str] = field(default_factory=list)
    failed_action_tickers: list[str] = field(default_factory=list)
    failed_components: list[str] = field(default_factory=list)


def main() -> int:
    configure_logging("daily_update")
    ensure_tables_exist()

    checker = DataQualityChecker()
    summary = UpdateSummary()
    today = date.today()
    market_data_end = current_market_data_end_date()
    tracked_tickers = get_tracked_tickers()
    latest_price_dates = get_latest_price_dates(tracked_tickers)

    logger.info("daily update will process {} tracked tickers", len(tracked_tickers))
    _update_prices(
        tickers=tracked_tickers,
        latest_price_dates=latest_price_dates,
        market_data_end=market_data_end,
        checker=checker,
        summary=summary,
    )
    _update_macro(today=today, checker=checker, summary=summary)
    _update_fundamentals(tickers=tracked_tickers, today=today, checker=checker, summary=summary)
    _update_corporate_actions(tickers=tracked_tickers, today=today, summary=summary)

    _log_summary(summary)
    summarize_row_counts()
    return 0 if not _has_failures(summary) else 2


def _update_prices(
    *,
    tickers: list[str],
    latest_price_dates: dict[str, date],
    market_data_end: date,
    checker: DataQualityChecker,
    summary: UpdateSummary,
) -> None:
    if not tickers:
        logger.warning("no tracked tickers found in stocks; skipping price update")
        return

    source = PolygonDataSource()
    bootstrap_start = market_data_end - timedelta(days=30)

    for index, ticker in enumerate(tickers, start=1):
        latest_date = latest_price_dates.get(ticker)
        logger.info("[{}/{}] updating prices for {} latest_trade_date={}", index, len(tickers), ticker, latest_date)
        try:
            if latest_date is None:
                logger.warning(
                    "{} has no existing stock_prices rows; bootstrapping only the last 30 calendar days",
                    ticker,
                )
                frame = source.fetch_historical([ticker], bootstrap_start, market_data_end)
            elif latest_date >= market_data_end:
                summary.skipped_price_tickers.append(ticker)
                logger.info("skipping {} because prices are already current through {}", ticker, latest_date)
                continue
            else:
                frame = source.fetch_incremental([ticker], latest_date + timedelta(days=1))

            if frame.empty:
                summary.skipped_price_tickers.append(ticker)
                logger.info("no new price rows returned for {}", ticker)
                continue

            summary.price_rows += len(frame)
            summarize_quality_reports(
                checker,
                dataset_name=f"daily_prices:{ticker}",
                frame=decimal_to_float_frame(frame),
                validate_prices=True,
            )
        except Exception as exc:
            logger.opt(exception=exc).error("daily price update failed for {}", ticker)
            summary.failed_price_tickers.append(ticker)


def _update_macro(
    *,
    today: date,
    checker: DataQualityChecker,
    summary: UpdateSummary,
) -> None:
    try:
        source = FredDataSource()
        frame = source.fetch_incremental(
            ["VIXCLS", "DGS10", "DGS2", "BAA10Y", "AAA10Y", "FEDFUNDS"],
            today - timedelta(days=30),
        )
        summary.macro_rows += len(frame)
        summarize_quality_reports(
            checker,
            dataset_name="daily_macro",
            frame=decimal_to_float_frame(frame),
            validate_prices=False,
        )
    except Exception as exc:
        logger.opt(exception=exc).error("daily macro update failed")
        summary.failed_components.append("macro")


def _update_fundamentals(
    *,
    tickers: list[str],
    today: date,
    checker: DataQualityChecker,
    summary: UpdateSummary,
) -> None:
    if not tickers:
        logger.warning("no tracked tickers found in stocks; skipping fundamentals update")
        return

    source = FMPDataSource()
    since_date = today - timedelta(days=120)
    for index, ticker in enumerate(tickers, start=1):
        logger.info("[{}/{}] updating fundamentals for {} since {}", index, len(tickers), ticker, since_date)
        try:
            frame = source.fetch_incremental([ticker], since_date)
            if frame.empty:
                continue
            summary.fundamental_rows += len(frame)
            summarize_quality_reports(
                checker,
                dataset_name=f"daily_fundamentals:{ticker}",
                frame=decimal_to_float_frame(frame),
                validate_prices=False,
            )
        except Exception as exc:
            logger.opt(exception=exc).error("daily fundamental update failed for {}", ticker)
            summary.failed_fundamental_tickers.append(ticker)


def _update_corporate_actions(
    *,
    tickers: list[str],
    today: date,
    summary: UpdateSummary,
) -> None:
    if not tickers:
        logger.warning("no tracked tickers found in stocks; skipping corporate action update")
        return

    start_date = today - timedelta(days=30)
    for index, ticker in enumerate(tickers, start=1):
        logger.info("[{}/{}] updating corporate actions for {} since {}", index, len(tickers), ticker, start_date)
        try:
            frame = fetch_corporate_actions([ticker], start_date, today)
            summary.corporate_action_rows += len(frame)
        except Exception as exc:
            logger.opt(exception=exc).error("daily corporate action update failed for {}", ticker)
            summary.failed_action_tickers.append(ticker)


def _log_summary(summary: UpdateSummary) -> None:
    logger.info(
        "daily update summary prices={} macro={} fundamentals={} corporate_actions={}",
        summary.price_rows,
        summary.macro_rows,
        summary.fundamental_rows,
        summary.corporate_action_rows,
    )
    if summary.skipped_price_tickers:
        logger.info("skipped price tickers: {}", ",".join(summary.skipped_price_tickers))
    if summary.failed_price_tickers:
        logger.warning("failed price tickers: {}", ",".join(summary.failed_price_tickers))
    if summary.failed_fundamental_tickers:
        logger.warning("failed fundamental tickers: {}", ",".join(summary.failed_fundamental_tickers))
    if summary.failed_action_tickers:
        logger.warning("failed corporate action tickers: {}", ",".join(summary.failed_action_tickers))
    if summary.failed_components:
        logger.warning("failed components: {}", ",".join(summary.failed_components))


def _has_failures(summary: UpdateSummary) -> bool:
    return any(
        [
            summary.failed_price_tickers,
            summary.failed_fundamental_tickers,
            summary.failed_action_tickers,
            summary.failed_components,
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
