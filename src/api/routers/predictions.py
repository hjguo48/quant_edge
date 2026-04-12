from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.predictions import (
    PredictionItem,
    PredictionResponse,
    ShapFeature,
    SignalHistoryPoint,
    SignalHistoryResponse,
    TickerPredictionResponse,
    TickerShapResponse,
)
from src.api.services.greyscale_reader import GreyscaleReader
from src.api.services.shap_service import get_shap_for_ticker

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


@router.get("/latest", response_model=PredictionResponse)
async def get_latest_predictions(
    top_n: int | None = Query(default=None, ge=1, le=600, description="Limit to top N predictions"),
) -> PredictionResponse:
    reader = _get_reader()
    report = reader.get_latest_report()
    predictions = reader.get_all_fusion_scores()

    if top_n is not None:
        predictions = predictions[:top_n]

    return PredictionResponse(
        signal_date=report.get("live_outputs", {}).get("signal_date") if report else None,
        week_number=report.get("week_number") if report else None,
        universe_size=report.get("db_state", {}).get("stock_universe_size") if report else None,
        predictions=[PredictionItem(**item) for item in predictions],
    )


@router.get("/{ticker}", response_model=TickerPredictionResponse)
async def get_ticker_prediction(ticker: str) -> TickerPredictionResponse:
    reader = _get_reader()
    detail = reader.get_ticker_detail(ticker.upper())
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for ticker '{ticker.upper()}'")
    return TickerPredictionResponse(**detail)


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
