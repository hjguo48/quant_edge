from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas.backtest import BacktestRequest, BacktestResponse

router = APIRouter(prefix="/api/backtest", tags=["Backtest"])


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    return BacktestResponse(
        message="not implemented",
        status="placeholder",
        strategy_name=request.strategy_name,
    )
