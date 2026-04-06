from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    strategy_name: str = "placeholder"
    start_date: date | None = None
    end_date: date | None = None
    initial_capital: float = Field(default=1_000_000.0, gt=0)
    tickers: list[str] = Field(default_factory=list)


class BacktestResponse(BaseModel):
    message: str
    status: str
    strategy_name: str | None = None
    run_id: str | None = None
