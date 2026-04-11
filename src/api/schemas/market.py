from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class MarketSnapshot(BaseModel):
    ticker: str
    trade_date: date
    price: float
    previous_close: float | None = None
    change: float | None = None
    change_pct: float | None = None
    volume: int | None = None


class VolatilitySnapshot(BaseModel):
    series_id: str
    observation_date: date
    value: float
    knowledge_time: datetime | None = None


class MarketBreadth(BaseModel):
    advancing: int = 0
    declining: int = 0
    unchanged: int = 0
    total: int = 0
    advance_decline_ratio: float | None = None
    advance_pct: float | None = None


class SectorPerformanceItem(BaseModel):
    sector: str
    avg_change_pct: float | None = None
    total_volume: int = 0
    ticker_count: int = 0


class IndexPriceBar(BaseModel):
    trade_date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = None
    volume: int | None = None


class MarketOverviewResponse(BaseModel):
    as_of: datetime
    latest_trade_date: date | None = None
    spy: MarketSnapshot | None = None
    breadth: MarketBreadth = Field(default_factory=MarketBreadth)
    sectors: list[SectorPerformanceItem] = Field(default_factory=list)
    vix: VolatilitySnapshot | None = None


class MarketIndicesResponse(BaseModel):
    ticker: str
    as_of: datetime
    days: int
    start_date: date | None = None
    end_date: date | None = None
    prices: list[IndexPriceBar] = Field(default_factory=list)


class MarketSectorsResponse(BaseModel):
    as_of: datetime
    days: int
    start_date: date | None = None
    end_date: date | None = None
    sectors: list[SectorPerformanceItem] = Field(default_factory=list)
