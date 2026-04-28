from __future__ import annotations

from typing import Any

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


# Monitor schemas — combined heartbeat + weeks + gate + diagnostics


class GreyscaleHeartbeat(BaseModel):
    status: str | None = None
    bundle_version: str | None = None
    signal_date: str | None = None
    generated_at_utc: str | None = None
    layer3_enforcement_mode: bool | None = None
    ticker_count: int | None = None
    actual_holding_count: int | None = None
    shadow_holding_count: int | None = None
    shadow_cvar_triggered: bool | None = None
    layer1_pass: bool | None = None
    layer2_pass: bool | None = None
    layer3_pass: bool | None = None
    shadow_layer3_pass: bool | None = None
    layer4_pass: bool | None = None
    gate_status: str | None = None
    matured_weeks: int | None = None


class GreyscaleWeekSummary(BaseModel):
    week_number: int
    signal_date: str | None = None
    holding_count: int | None = None
    turnover: float | None = None
    layer1_pass: bool | None = None
    layer2_pass: bool | None = None
    layer3_pass: bool | None = None
    layer4_pass: bool | None = None
    weight_source: str | None = None
    realized_ic_mean: float | None = None


class GreyscaleGateCheck(BaseModel):
    threshold: str | None = None
    value: float | None = None
    passed: bool | None = None
    skipped_reason: str | None = None


class GreyscaleGate(BaseModel):
    gate_rule: str | None = None
    gate_status: str | None = None
    matured_weeks: int = 0
    required_weeks: int = 0
    reports_seen: int = 0
    layer12_halt_count: int = 0
    mean_live_ic: float | None = None
    mean_pairwise_rank_correlation: float | None = None
    mean_turnover: float | None = None
    positive_live_ic_weeks: int = 0
    rolling_live_ic_std: float | None = None
    checks: dict[str, GreyscaleGateCheck] = Field(default_factory=dict)


class GreyscaleShadowReduceEntry(BaseModel):
    ticker: str
    raw_weight: float | None = None
    shadow_weight: float | None = None


class GreyscaleShadowDiagnostics(BaseModel):
    enforcement_mode: bool | None = None
    shadow_holding_count: int | None = None
    shadow_gross_exposure: float | None = None
    shadow_cvar_99: float | None = None
    shadow_cash_weight: float | None = None
    shadow_turnover_vs_previous: float | None = None
    cvar_triggered: bool | None = None
    cvar_haircut_rounds: int | None = None
    tickers_layer3_would_remove: list[str] = Field(default_factory=list)
    tickers_layer3_would_reduce: list[GreyscaleShadowReduceEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)


class GreyscaleMonitorResponse(BaseModel):
    heartbeat: GreyscaleHeartbeat | None = None
    weeks: list[GreyscaleWeekSummary] = Field(default_factory=list)
    gate: GreyscaleGate | None = None
    shadow_diagnostics: GreyscaleShadowDiagnostics | None = None
    layer1_diagnostics: dict[str, Any] | None = None
