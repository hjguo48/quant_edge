from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PortfolioHolding(BaseModel):
    ticker: str
    weight: float


class PortfolioResponse(BaseModel):
    message: str
    as_of: datetime | None = None
    holdings: list[PortfolioHolding] = Field(default_factory=list)
