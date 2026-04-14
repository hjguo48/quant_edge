from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import sys

from loguru import logger
import sqlalchemy as sa

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
from src.config import settings
from src.data.corporate_actions import fetch_corporate_actions
from src.data.db.models import Stock, StockPrice
from src.data.db.session import get_session_factory
from src.data.quality import DataQualityChecker
from src.data.sources.fmp import FMPDataSource
from src.data.sources.fred import FredDataSource
from src.data.sources.polygon import PolygonDataSource, normalize_polygon_ticker


@dataclass
class UpdateSummary:
    price_rows: int = 0
    macro_rows: int = 0
    fundamental_rows: int = 0
    corporate_action_rows: int = 0
    split_adjusted_tickers: list[str] = field(default_factory=list)
    price_anomaly_tickers: list[str] = field(default_factory=list)
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
    _handle_stock_splits(
        tickers=tracked_tickers,
        market_data_end=market_data_end,
        summary=summary,
    )
    _update_prices(
        tickers=tracked_tickers,
        latest_price_dates=latest_price_dates,
        market_data_end=market_data_end,
        checker=checker,
        summary=summary,
    )
    _check_price_anomalies(tickers=tracked_tickers, summary=summary)
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


def _handle_stock_splits(
    *,
    tickers: list[str],
    market_data_end: date,
    summary: UpdateSummary,
    lookback_days: int = 5,
) -> None:
    if not tickers:
        logger.warning("no tracked tickers found in stocks; skipping split detection")
        return
    if not settings.POLYGON_API_KEY:
        logger.warning("POLYGON_API_KEY is not configured; skipping split detection")
        return

    lookback_start = date.today() - timedelta(days=lookback_days)
    try:
        split_rows = _fetch_recent_splits(start_date=lookback_start, end_date=date.today())
    except Exception as exc:
        logger.opt(exception=exc).error("recent split detection failed")
        summary.failed_components.append("split_detection")
        return

    tracked_tickers = {normalize_polygon_ticker(ticker) for ticker in tickers}
    matching_splits = [
        row
        for row in split_rows
        if row["ticker"] in tracked_tickers
    ]
    if not matching_splits:
        logger.info("no tracked-ticker splits detected between {} and {}", lookback_start, date.today())
        return

    source = PolygonDataSource(min_request_interval=0.0)
    split_tickers = sorted({str(row["ticker"]) for row in matching_splits})
    for ticker in split_tickers:
        related_events = [row for row in matching_splits if row["ticker"] == ticker]
        start_date = _resolve_adjusted_backfill_start(ticker=ticker, market_data_end=market_data_end)
        logger.warning(
            "detected split event(s) for {}: {}. Re-fetching fully adjusted history from {} to {}",
            ticker,
            "; ".join(_format_split_event(row) for row in related_events),
            start_date,
            market_data_end,
        )
        try:
            frame = source.fetch_adjusted_historical([ticker], start_date, market_data_end)
            if frame.empty:
                logger.warning("split-adjusted backfill returned no rows for {}", ticker)
                continue
            if ticker not in summary.split_adjusted_tickers:
                summary.split_adjusted_tickers.append(ticker)
        except Exception as exc:
            logger.opt(exception=exc).error("split-adjusted price backfill failed for {}", ticker)
            if ticker not in summary.failed_price_tickers:
                summary.failed_price_tickers.append(ticker)


def _check_price_anomalies(
    *,
    tickers: list[str],
    summary: UpdateSummary,
    threshold: float = 0.50,
) -> None:
    if not tickers:
        return

    rows = _load_latest_closes(tickers=tickers)
    closes_by_ticker: dict[str, list[tuple[date, float]]] = {}
    for ticker, trade_date, close_value in rows:
        if close_value is None:
            continue
        closes_by_ticker.setdefault(str(ticker), []).append((trade_date, float(close_value)))

    for ticker, series in closes_by_ticker.items():
        if len(series) < 2:
            continue

        latest_date, latest_close = series[0]
        previous_date, previous_close = series[1]
        if previous_close == 0:
            continue

        pct_change = (latest_close / previous_close) - 1.0
        if abs(pct_change) <= threshold:
            continue

        logger.warning(
            "price anomaly detected for {} between {} and {}: {:.4f} -> {:.4f} ({:+.2%})",
            ticker,
            previous_date,
            latest_date,
            previous_close,
            latest_close,
            pct_change,
        )
        if ticker not in summary.price_anomaly_tickers:
            summary.price_anomaly_tickers.append(ticker)


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


def _fetch_recent_splits(*, start_date: date, end_date: date) -> list[dict[str, object]]:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("requests is required for split detection") from exc

    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "QuantEdge/0.1.0"})

    url = "https://api.polygon.io/v3/reference/splits"
    params: dict[str, object] | None = {
        "execution_date.gte": start_date.isoformat(),
        "execution_date.lte": end_date.isoformat(),
        "limit": 1_000,
        "sort": "execution_date",
        "order": "asc",
        "apiKey": settings.POLYGON_API_KEY,
    }
    rows: list[dict[str, object]] = []

    while url:
        response = session.get(url, params=params, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(
                f"Polygon splits request failed with HTTP {response.status_code}: {response.text[:500]}",
            )

        payload = response.json()
        results = payload.get("results") or []
        if not isinstance(results, list):
            raise RuntimeError(f"Unexpected Polygon splits payload type: {type(results).__name__}")

        for item in results:
            execution_date = item.get("execution_date")
            ticker = item.get("ticker")
            if not execution_date or not ticker:
                continue
            rows.append(
                {
                    "ticker": normalize_polygon_ticker(str(ticker)),
                    "execution_date": date.fromisoformat(str(execution_date)),
                    "split_from": item.get("split_from"),
                    "split_to": item.get("split_to"),
                },
            )

        next_url = payload.get("next_url")
        if isinstance(next_url, str) and next_url:
            url = next_url
            params = {"apiKey": settings.POLYGON_API_KEY}
        else:
            break

    return rows


def _resolve_adjusted_backfill_start(*, ticker: str, market_data_end: date) -> date:
    fallback_start = market_data_end - timedelta(days=365 * 10)
    session_factory = get_session_factory()
    with session_factory() as session:
        min_trade_date = session.execute(
            sa.select(sa.func.min(StockPrice.trade_date)).where(StockPrice.ticker == ticker),
        ).scalar_one_or_none()
        ipo_date = session.execute(
            sa.select(Stock.ipo_date).where(Stock.ticker == ticker),
        ).scalar_one_or_none()

    candidates = [value for value in (min_trade_date, ipo_date) if value is not None]
    if not candidates:
        return fallback_start
    return min(candidates)


def _load_latest_closes(*, tickers: list[str]) -> list[tuple[str, date, object]]:
    if not tickers:
        return []

    session_factory = get_session_factory()
    latest_rows = (
        sa.select(
            StockPrice.ticker.label("ticker"),
            StockPrice.trade_date.label("trade_date"),
            StockPrice.close.label("close"),
            sa.func.row_number().over(
                partition_by=StockPrice.ticker,
                order_by=StockPrice.trade_date.desc(),
            ).label("row_num"),
        )
        .where(StockPrice.ticker.in_(tickers))
        .subquery()
    )

    with session_factory() as session:
        rows = session.execute(
            sa.select(latest_rows.c.ticker, latest_rows.c.trade_date, latest_rows.c.close)
            .where(latest_rows.c.row_num <= 2)
            .order_by(latest_rows.c.ticker, latest_rows.c.trade_date.desc()),
        ).all()

    return [(str(ticker), trade_date, close_value) for ticker, trade_date, close_value in rows]


def _format_split_event(row: dict[str, object]) -> str:
    split_from = row.get("split_from")
    split_to = row.get("split_to")
    execution_date = row.get("execution_date")
    if split_from and split_to:
        return f"{execution_date} {split_from}:{split_to}"
    return str(execution_date)


def _log_summary(summary: UpdateSummary) -> None:
    logger.info(
        "daily update summary prices={} macro={} fundamentals={} corporate_actions={}",
        summary.price_rows,
        summary.macro_rows,
        summary.fundamental_rows,
        summary.corporate_action_rows,
    )
    if summary.split_adjusted_tickers:
        logger.info("split-adjusted tickers: {}", ",".join(summary.split_adjusted_tickers))
    if summary.price_anomaly_tickers:
        logger.warning("price anomaly tickers: {}", ",".join(summary.price_anomaly_tickers))
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
