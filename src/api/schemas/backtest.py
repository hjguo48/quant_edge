from __future__ import annotations

from datetime import date, datetime
from typing import Any

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
    task_id: str
    strategy_name: str | None = None
    run_id: str | None = None


class BacktestStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class BacktestResultResponse(BaseModel):
    task_id: str
    status: str
    result: dict[str, Any] | None = None


class ChampionProfile(BaseModel):
    strategy: str
    horizon_days: int
    net_ann_excess: float
    gross_ann_excess: float
    ir: float
    sharpe: float
    max_drawdown: float
    avg_turnover_weekly: float
    cost_drag_ann: float
    n_periods: int
    backtest_as_of: str | None = None
    cost_model: dict[str, float] | str | None = None


class ExpectationStats(BaseModel):
    weekly_excess_mean: float
    weekly_excess_std: float
    source: str = "truth_table_periods"


class ConePoint(BaseModel):
    date: str
    day_index: int
    expected: float
    upper_1s: float
    lower_1s: float
    upper_2s: float
    lower_2s: float


class LiveExcessPoint(BaseModel):
    date: str
    excess_cum_return: float
    is_rebalance: bool = False


class MetricComparison(BaseModel):
    backtest: float | None = None
    live: float | None = None
    unit: str | None = None
    note: str | None = None


class BacktestVsLiveResponse(BaseModel):
    champion: ChampionProfile | None = None
    expectation: ExpectationStats | None = None
    cone: list[ConePoint] = Field(default_factory=list)
    live: list[LiveExcessPoint] = Field(default_factory=list)
    comparison: dict[str, MetricComparison] = Field(default_factory=dict)
