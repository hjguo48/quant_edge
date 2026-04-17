from __future__ import annotations

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
from src.data.db.models import Stock

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

    stock_info = await _get_stock_info(
        db,
        [item["ticker"] for item in predictions],
    )

    return PredictionResponse(
        signal_date=report.get("live_outputs", {}).get("signal_date") if report else None,
        week_number=report.get("week_number") if report else None,
        universe_size=report.get("db_state", {}).get("stock_universe_size") if report else None,
        predictions=[
            PredictionItem(
                **item,
                sector=(stock_info.get(item["ticker"].upper()) or {}).get("sector"),
                company_name=(stock_info.get(item["ticker"].upper()) or {}).get("company_name"),
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
    stock_info = await _get_stock_info(
        db,
        [item["ticker"] for item in predictions],
    )

    return PredictionResponse(
        signal_date=report.get("live_outputs", {}).get("signal_date") if report else None,
        week_number=report.get("week_number") if report else None,
        universe_size=report.get("db_state", {}).get("stock_universe_size") if report else None,
        predictions=[
            PredictionItem(
                **item,
                sector=(stock_info.get(item["ticker"].upper()) or {}).get("sector"),
                company_name=(stock_info.get(item["ticker"].upper()) or {}).get("company_name"),
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
