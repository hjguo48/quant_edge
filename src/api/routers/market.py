from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas.market import MarketOverviewResponse

router = APIRouter(prefix="/api/market", tags=["Market"])


@router.get("/overview", response_model=MarketOverviewResponse)
async def get_market_overview() -> MarketOverviewResponse:
    return MarketOverviewResponse(
        message="not implemented",
        as_of=datetime.now(timezone.utc),
        market_status="placeholder",
    )
