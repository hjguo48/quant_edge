from __future__ import annotations

from pydantic import BaseModel


class StockDetailResponse(BaseModel):
    ticker: str
    message: str
    company_name: str | None = None
    sector: str | None = None
    industry: str | None = None
