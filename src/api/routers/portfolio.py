from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas.portfolio import PortfolioResponse

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])


@router.get("/current", response_model=PortfolioResponse)
async def get_current_portfolio() -> PortfolioResponse:
    return PortfolioResponse(
        message="not implemented",
        as_of=datetime.now(timezone.utc),
        holdings=[],
    )
