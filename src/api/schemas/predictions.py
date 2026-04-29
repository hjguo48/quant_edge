from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    ticker: str
    score: float
    rank: int
    percentile: float
    sector: str | None = None
    company_name: str | None = None
    # W13.1 sparkline data — last 20 trading days. Empty list if data unavailable.
    recent_prices: list[float] = Field(default_factory=list)
    recent_excess_cum: list[float] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    signal_date: str | None = None
    week_number: int | None = None
    # W12 champion is single-model Ridge, not the deprecated 3-model fusion.
    # Naming aligned with bundle.json `version` field for clarity.
    model_name: str = "60d_ridge_swbuf_v3"
    universe_size: int | None = None
    predictions: list[PredictionItem] = Field(default_factory=list)


class TickerPredictionResponse(BaseModel):
    ticker: str
    fusion_score: float
    rank: int | None = None
    total: int | None = None
    percentile: float | None = None
    model_scores: dict[str, float] = Field(default_factory=dict)
    weight: float | None = None
    signal_date: str | None = None
    sector: str | None = None
    confidence: str | None = None
    model_spread: float | None = None
    model_agreement: float | None = None


class SignalHistoryPoint(BaseModel):
    week: int
    signal_date: str | None = None
    score: float
    rank: int | None = None
    total: int | None = None


class SignalHistoryResponse(BaseModel):
    ticker: str
    history: list[SignalHistoryPoint] = Field(default_factory=list)


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class TickerShapResponse(BaseModel):
    ticker: str
    signal_date: str | None = None
    # 'shap' for tree-model SHAP attribution (W11 fusion);
    # 'linear' for Ridge coef × feature attribution (W12 champion).
    attribution_type: str = "shap"
    features: list[ShapFeature] = Field(default_factory=list)


class ConfidenceStatsResponse(BaseModel):
    annualized_excess_ci_lower: float | None = None
    annualized_excess_ci_upper: float | None = None
    annualized_excess_estimate: float | None = None
    sharpe_ci_lower: float | None = None
    sharpe_ci_upper: float | None = None
    sharpe_estimate: float | None = None
    n_bootstrap: int | None = None
    ci_level: float | None = None


class ExpectedReturnBand(BaseModel):
    estimate: float
    ci_lower: float
    ci_upper: float


class ExpectedReturnsResponse(BaseModel):
    data_source: str = "g3_gate_bootstrap"
    ci_level: float
    n_observations: int
    n_bootstrap: int | None = None
    block_size: int | None = None
    annualized_excess: ExpectedReturnBand
    sharpe: ExpectedReturnBand


class TickerExpectedReturnResponse(BaseModel):
    ticker: str
    signal_date: str | None = None
    percentile: float | None = None
    quintile: int | None = None
    data_source: str = "g3_gate_bootstrap"
    ci_level: float
    n_observations: int
    n_bootstrap: int | None = None
    block_size: int | None = None
    annualized_excess: ExpectedReturnBand
    sharpe: ExpectedReturnBand
