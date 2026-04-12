from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    ticker: str
    score: float
    rank: int
    percentile: float
    sector: str | None = None
    company_name: str | None = None


class PredictionResponse(BaseModel):
    signal_date: str | None = None
    week_number: int | None = None
    model_name: str = "ic_weighted_fusion_60d"
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
    features: list[ShapFeature] = Field(default_factory=list)
