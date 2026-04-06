from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.stocks import StockDetailResponse

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])


@router.get("/{ticker}", response_model=StockDetailResponse)
async def get_stock_detail(ticker: str) -> StockDetailResponse:
    return StockDetailResponse(
        ticker=ticker.upper(),
        message="not implemented",
    )
