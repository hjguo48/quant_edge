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
    sector: str | None = None


class BudgetResponse(BaseModel):
    total_budget: float
    allocations: list[BudgetAllocation] = Field(default_factory=list)


class RebalanceOrder(BaseModel):
    ticker: str
    action: Literal["buy", "sell", "hold"]
    weight_prev: float
    weight_new: float
    weight_delta: float
    sector: str | None = None


class RebalanceResponse(BaseModel):
    signal_date: str | None = None
    orders: list[RebalanceOrder] = Field(default_factory=list)


class DailyPerformancePoint(BaseModel):
    date: str
    cumulative_portfolio: float
    cumulative_spy: float
    cumulative_excess: float
    tranche_day: int


class DailyPerformanceTranche(BaseModel):
    signal_date: str
    tranche_index: int
    entry_date: str | None = None
    horizon_end_date: str | None = None
    tickers_used: list[str] = Field(default_factory=list)
    tickers_dropped: list[str] = Field(default_factory=list)
    series: list[DailyPerformancePoint] = Field(default_factory=list)


class DailyPortfolioPerformanceResponse(BaseModel):
    horizon: Literal["1d", "5d", "20d", "60d"]
    bundle_version: str | None = None
    weeks_count: int = 0
    latest_horizon_end_date: str | None = None
    tranches: list[DailyPerformanceTranche] = Field(default_factory=list)
