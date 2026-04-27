from __future__ import annotations

from pydantic import BaseModel, Field


class GreyscaleHorizonWeek(BaseModel):
    status: str  # realized / partial / pending
    portfolio_return: float | None = None
    spy_return: float | None = None
    excess: float | None = None
    tickers_used: int = 0
    tickers_missing: int = 0
    horizon_end_date: str | None = None


class GreyscaleWeek(BaseModel):
    week_number: int
    signal_date: str | None = None
    horizons: dict[str, GreyscaleHorizonWeek] = Field(default_factory=dict)


class GreyscaleWeeklyCurvePoint(BaseModel):
    signal_date: str
    weekly_return: float | None = None
    weekly_spy: float | None = None
    weekly_excess: float | None = None
    cumulative_return: float | None = None
    cumulative_spy: float | None = None
    cumulative_excess: float | None = None


class GreyscaleHorizonCumulative(BaseModel):
    return_: float | None = Field(default=None, alias="return")
    spy_return: float | None = None
    excess: float | None = None
    max_drawdown: float | None = None
    weeks_realized: int = 0
    winrate_vs_spy: float | None = None
    weekly_curve: list[GreyscaleWeeklyCurvePoint] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class GreyscalePerformanceResponse(BaseModel):
    as_of_utc: str | None = None
    today: str | None = None
    benchmark: str = "SPY"
    horizons_supported: list[int] = Field(default_factory=list)
    per_week: list[GreyscaleWeek] = Field(default_factory=list)
    cumulative: dict[str, GreyscaleHorizonCumulative] = Field(default_factory=dict)
