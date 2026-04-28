from __future__ import annotations

from datetime import date as date_type
from pathlib import Path

import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.schemas.predictions import (
    ConfidenceStatsResponse,
    ExpectedReturnBand,
    ExpectedReturnsResponse,
    PredictionItem,
    PredictionResponse,
    ShapFeature,
    SignalHistoryPoint,
    SignalHistoryResponse,
    TickerExpectedReturnResponse,
    TickerPredictionResponse,
    TickerShapResponse,
)
from src.api.services.greyscale_reader import GreyscaleReader
from src.api.services.shap_service import get_shap_for_ticker
from src.data.db.models import Stock, StockPrice

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])
GREYSCALE_REPORT_DIR = Path("data/reports/greyscale")
_READER: GreyscaleReader | None = None
_READER_DIR: Path | None = None


def _get_reader() -> GreyscaleReader:
    global _READER, _READER_DIR

    if _READER is None or _READER_DIR != GREYSCALE_REPORT_DIR:
        _READER = GreyscaleReader(report_dir=GREYSCALE_REPORT_DIR)
        _READER_DIR = GREYSCALE_REPORT_DIR
    return _READER


def _normalize_tickers(raw_tickers: str) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for ticker in raw_tickers.split(","):
        candidate = ticker.strip().upper()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _percentile_to_quintile(percentile: float | None) -> int | None:
    if percentile is None:
        return None
    if percentile >= 80:
        return 1
    if percentile >= 60:
        return 2
    if percentile >= 40:
        return 3
    if percentile >= 20:
        return 4
    return 5


RECENT_PRICE_DAYS = 20
RECENT_BENCHMARK_TICKER = "SPY"


async def _get_recent_price_series(
    db: AsyncSession,
    tickers: list[str],
    n_days: int = RECENT_PRICE_DAYS,
    anchor_date: date_type | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Return {TICKER: {"prices": [...], "excess_cum": [...]}} for the given tickers.

    `prices` is the last `n_days` of adj_close (oldest → newest).
    `excess_cum` is cumulative excess vs SPY over the same window (in pct, oldest → newest).

    Codex P3/Finding 3 fix: anchor the price window at the greyscale report's
    signal_date (passed as `anchor_date`) rather than max(stock_prices.trade_date).
    Without this, the sparkline drifts past the signal date as new prices
    accumulate during the week — the user would see post-signal price action
    that the model didn't see when it scored the ticker. When `anchor_date`
    is None we fall back to the legacy max-trade-date behaviour for any
    out-of-band callers.
    """
    if not tickers:
        return {}

    normalized = [t.upper() for t in tickers] + [RECENT_BENCHMARK_TICKER]

    if anchor_date is not None:
        latest_trade_date = anchor_date
    else:
        latest_ts_row = await db.execute(
            sa.select(sa.func.max(StockPrice.trade_date)).where(
                StockPrice.ticker.in_(normalized)
            )
        )
        latest_trade_date = latest_ts_row.scalar()
    if latest_trade_date is None:
        return {}

    fetch_n = n_days + 5
    rows = await db.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT ticker, trade_date,
                       COALESCE(adj_close, close) AS px,
                       ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY trade_date DESC) AS rn
                FROM stock_prices
                WHERE ticker = ANY(:tickers)
                  AND trade_date <= :latest
            )
            SELECT ticker, trade_date, px FROM ranked WHERE rn <= :fetch_n
            ORDER BY ticker, trade_date
            """
        ),
        {"tickers": normalized, "latest": latest_trade_date, "fetch_n": fetch_n},
    )

    by_ticker: dict[str, list[tuple]] = {}
    for ticker, td, px in rows.all():
        if px is None:
            continue
        by_ticker.setdefault(ticker, []).append((td, float(px)))

    bench_series = by_ticker.get(RECENT_BENCHMARK_TICKER, [])
    # Codex P2 fix: align by trade_date, not by array length. Build a
    # date-keyed return map for the benchmark so missing ticker sessions
    # don't silently misalign with SPY returns from different dates.
    bench_returns_by_date = dict(_trade_date_returns(bench_series[-n_days:]))

    out: dict[str, dict[str, list[float]]] = {}
    for tk in tickers:
        upper = tk.upper()
        series = by_ticker.get(upper, [])
        recent = series[-n_days:]
        prices = [p for _, p in recent]
        if len(recent) < 2:
            out[upper] = {"prices": prices, "excess_cum": []}
            continue
        ticker_returns_dated = _trade_date_returns(recent)
        cum_excess: list[float] = []
        running = 0.0
        for td, r_t in ticker_returns_dated:
            r_b = bench_returns_by_date.get(td)
            if r_b is None:
                # SPY missing this date (rare — e.g. early trading halt).
                # Skip without polluting cumulative; preserve series alignment
                # by NOT appending — frontend will see a slightly shorter array
                # but every entry is mathematically meaningful.
                continue
            running += (r_t - r_b)
            cum_excess.append(running)
        out[upper] = {"prices": prices, "excess_cum": cum_excess}

    return out


def _to_returns(prices: list[float]) -> list[float]:
    out: list[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        if prev == 0:
            out.append(0.0)
        else:
            out.append((prices[i] - prev) / prev)
    return out


def _trade_date_returns(series: list[tuple]) -> list[tuple]:
    """Return [(trade_date, daily_return)] preserving date keys."""
    out: list[tuple] = []
    for i in range(1, len(series)):
        prev_td, prev_px = series[i - 1]
        curr_td, curr_px = series[i]
        if prev_px == 0:
            out.append((curr_td, 0.0))
        else:
            out.append((curr_td, (curr_px - prev_px) / prev_px))
    return out




async def _get_stock_info(
    db: AsyncSession, tickers: list[str],
) -> dict[str, dict[str, str | None]]:
    """Return {TICKER: {"sector": ..., "company_name": ...}} for the given tickers."""
    if not tickers:
        return {}

    normalized_tickers = [ticker.upper() for ticker in tickers]
    result = await db.execute(
        sa.select(Stock.ticker, Stock.sector, Stock.company_name).where(
            Stock.ticker.in_(normalized_tickers)
        )
    )
    return {
        ticker.upper(): {"sector": sector, "company_name": company_name}
        for ticker, sector, company_name in result.all()
    }


@router.get("/latest", response_model=PredictionResponse)
async def get_latest_predictions(
    top_n: int | None = Query(default=None, ge=1, le=600, description="Limit to top N predictions"),
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    reader = _get_reader()
    report = reader.get_latest_report()
    predictions = reader.get_all_fusion_scores()

    if top_n is not None:
        predictions = predictions[:top_n]

    pred_tickers = [item["ticker"] for item in predictions]
    stock_info = await _get_stock_info(db, pred_tickers)
    signal_date_str = (report or {}).get("live_outputs", {}).get("signal_date")
    anchor = date_type.fromisoformat(signal_date_str) if signal_date_str else None
    price_series = await _get_recent_price_series(db, pred_tickers, anchor_date=anchor)

    return PredictionResponse(
        signal_date=signal_date_str,
        week_number=report.get("week_number") if report else None,
        universe_size=report.get("db_state", {}).get("stock_universe_size") if report else None,
        predictions=[
            PredictionItem(
                **item,
                sector=(stock_info.get(item["ticker"].upper()) or {}).get("sector"),
                company_name=(stock_info.get(item["ticker"].upper()) or {}).get("company_name"),
                recent_prices=(price_series.get(item["ticker"].upper()) or {}).get("prices") or [],
                recent_excess_cum=(price_series.get(item["ticker"].upper()) or {}).get("excess_cum") or [],
            )
            for item in predictions
        ],
    )


@router.get("/confidence-stats", response_model=ConfidenceStatsResponse)
async def get_confidence_stats() -> ConfidenceStatsResponse:
    reader = _get_reader()
    confidence = reader.get_bootstrap_confidence()
    if confidence is None:
        raise HTTPException(status_code=404, detail="No bootstrap confidence data available")
    return ConfidenceStatsResponse(**confidence)


@router.get("/batch", response_model=PredictionResponse)
async def get_batch_predictions(
    tickers: str = Query(..., description="Comma-separated ticker list"),
    db: AsyncSession = Depends(get_db),
) -> PredictionResponse:
    reader = _get_reader()
    report = reader.get_latest_report()
    normalized_tickers = _normalize_tickers(tickers)
    predictions = reader.get_fusion_scores_for_tickers(normalized_tickers)
    pred_tickers = [item["ticker"] for item in predictions]
    stock_info = await _get_stock_info(db, pred_tickers)
    signal_date_str = (report or {}).get("live_outputs", {}).get("signal_date")
    anchor = date_type.fromisoformat(signal_date_str) if signal_date_str else None
    price_series = await _get_recent_price_series(db, pred_tickers, anchor_date=anchor)

    return PredictionResponse(
        signal_date=signal_date_str,
        week_number=report.get("week_number") if report else None,
        universe_size=report.get("db_state", {}).get("stock_universe_size") if report else None,
        predictions=[
            PredictionItem(
                **item,
                sector=(stock_info.get(item["ticker"].upper()) or {}).get("sector"),
                company_name=(stock_info.get(item["ticker"].upper()) or {}).get("company_name"),
                recent_prices=(price_series.get(item["ticker"].upper()) or {}).get("prices") or [],
                recent_excess_cum=(price_series.get(item["ticker"].upper()) or {}).get("excess_cum") or [],
            )
            for item in predictions
        ],
    )


@router.get("/expected-returns", response_model=ExpectedReturnsResponse)
async def get_expected_returns() -> ExpectedReturnsResponse:
    reader = _get_reader()
    expected_returns = reader.get_expected_returns()
    if expected_returns is None:
        raise HTTPException(status_code=404, detail="No expected-return data available")
    return ExpectedReturnsResponse(
        data_source=expected_returns["data_source"],
        ci_level=expected_returns["ci_level"],
        n_observations=expected_returns["n_observations"],
        annualized_excess=ExpectedReturnBand(**expected_returns["annualized_excess"]),
        sharpe=ExpectedReturnBand(**expected_returns["sharpe"]),
    )


@router.get("/{ticker}", response_model=TickerPredictionResponse)
async def get_ticker_prediction(
    ticker: str,
    db: AsyncSession = Depends(get_db),
) -> TickerPredictionResponse:
    reader = _get_reader()
    normalized_ticker = ticker.upper()
    detail = reader.get_ticker_detail(normalized_ticker)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for ticker '{normalized_ticker}'")

    stock_info = await _get_stock_info(db, [normalized_ticker])
    info = stock_info.get(normalized_ticker) or {}
    confidence = reader.get_ticker_confidence(normalized_ticker) or {}
    return TickerPredictionResponse(
        **detail,
        sector=info.get("sector"),
        confidence=confidence.get("confidence"),
        model_spread=confidence.get("model_spread"),
        model_agreement=confidence.get("model_agreement"),
    )


@router.get("/{ticker}/history", response_model=SignalHistoryResponse)
async def get_signal_history(ticker: str) -> SignalHistoryResponse:
    reader = _get_reader()
    history = reader.get_signal_history(ticker.upper())
    return SignalHistoryResponse(
        ticker=ticker.upper(),
        history=[SignalHistoryPoint(**point) for point in history],
    )


@router.get("/{ticker}/shap", response_model=TickerShapResponse)
async def get_ticker_shap(
    ticker: str,
    top_n: int = Query(default=15, ge=1, le=50, description="Number of top SHAP features to return"),
) -> TickerShapResponse:
    shap_result = get_shap_for_ticker(ticker.upper(), report_dir=GREYSCALE_REPORT_DIR, top_n=top_n)
    if shap_result is None:
        raise HTTPException(status_code=404, detail=f"No SHAP data for ticker '{ticker.upper()}'")
    return TickerShapResponse(
        ticker=shap_result["ticker"],
        signal_date=shap_result.get("signal_date"),
        features=[ShapFeature(**feature) for feature in shap_result.get("features", [])],
    )


@router.get("/{ticker}/expected-return", response_model=TickerExpectedReturnResponse)
async def get_ticker_expected_return(ticker: str) -> TickerExpectedReturnResponse:
    reader = _get_reader()
    normalized_ticker = ticker.upper()
    detail = reader.get_ticker_detail(normalized_ticker)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for ticker '{normalized_ticker}'")

    quintile = _percentile_to_quintile(detail.get("percentile"))
    expected_returns = reader.get_expected_returns(quintile=quintile)
    if expected_returns is None:
        raise HTTPException(status_code=404, detail="No expected-return data available")

    return TickerExpectedReturnResponse(
        ticker=normalized_ticker,
        signal_date=detail.get("signal_date"),
        percentile=detail.get("percentile"),
        quintile=quintile,
        data_source=expected_returns["data_source"],
        ci_level=expected_returns["ci_level"],
        n_observations=expected_returns["n_observations"],
        annualized_excess=ExpectedReturnBand(**expected_returns["annualized_excess"]),
        sharpe=ExpectedReturnBand(**expected_returns["sharpe"]),
    )
