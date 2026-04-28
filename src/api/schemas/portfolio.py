from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PortfolioHolding(BaseModel):
    ticker: str
    weight: float
    score: float | None = None
    sector: str | None = None
    company_name: str | None = None


class PortfolioResponse(BaseModel):
    signal_date: str | None = None
    week_number: int | None = None
    holding_count: int | None = None
    gross_exposure: float | None = None
    cash_weight: float | None = None
    portfolio_beta: float | None = None
    cvar_95: float | None = None
    turnover: float | None = None
    risk_pass: bool | None = None
    holdings: list[PortfolioHolding] = Field(default_factory=list)


class PortfolioSummaryResponse(BaseModel):
    signal_date: str | None = None
    week_number: int | None = None
    holding_count: int | None = None
    gross_exposure: float | None = None
    cash_weight: float | None = None
    portfolio_beta: float | None = None
    cvar_95: float | None = None
    turnover: float | None = None
    risk_pass: bool | None = None


class BudgetAllocation(BaseModel):
    ticker: str
    weight: float
    dollar_amount: float


class BudgetResponse(BaseModel):
    total_budget: float
    allocations: list[BudgetAllocation] = Field(default_factory=list)


class RebalanceOrder(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    weight_prev: float
    weight_new: float
    weight_delta: float


class RebalanceResponse(BaseModel):
    signal_date: str | None = None
    orders: list[RebalanceOrder] = Field(default_factory=list)
