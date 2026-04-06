from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    ticker: str
    score: float


class PredictionResponse(BaseModel):
    message: str
    as_of: datetime | None = None
    model_name: str | None = None
    predictions: list[PredictionItem] = Field(default_factory=list)
