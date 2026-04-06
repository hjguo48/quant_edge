from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from src.api.schemas.predictions import PredictionResponse

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])


@router.get("/latest", response_model=PredictionResponse)
async def get_latest_predictions() -> PredictionResponse:
    return PredictionResponse(
        message="not implemented",
        as_of=datetime.now(timezone.utc),
        model_name="ic_weighted_fusion_60d",
        predictions=[],
    )
