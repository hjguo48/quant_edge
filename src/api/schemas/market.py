from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class MarketOverviewResponse(BaseModel):
    message: str
    as_of: datetime | None = None
    market_status: str
