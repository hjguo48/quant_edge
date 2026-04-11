from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class StockQuote(BaseModel):
    trade_date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = None
    volume: int | None = None
    previous_close: float | None = None
    change: float | None = None
    change_pct: float | None = None


class StockPriceBar(BaseModel):
    trade_date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    adj_close: float | None = None
    volume: int | None = None


class StockDetailResponse(BaseModel):
    ticker: str
    company_name: str
    sector: str | None = None
    industry: str | None = None
    ipo_date: date | None = None
    delist_date: date | None = None
    delist_reason: str | None = None
    shares_outstanding: int | None = None
    market_cap: float | None = None
    as_of: datetime
    latest_price: StockQuote | None = None


class StockPricesResponse(BaseModel):
    ticker: str
    as_of: datetime
    days: int
    start_date: date | None = None
    end_date: date | None = None
    prices: list[StockPriceBar] = Field(default_factory=list)


class StockFundamentalsResponse(BaseModel):
    ticker: str
    as_of: datetime
    fiscal_period: str | None = None
    event_time: date | None = None
    knowledge_time: datetime | None = None
    metric_count: int = 0
    metrics: dict[str, float | None] = Field(default_factory=dict)


class StockTechnicalsResponse(BaseModel):
    ticker: str
    as_of: datetime
    trade_date: date | None = None
    close: float | None = None
    history_points: int = 0
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    sma_20: float | None = None
    sma_50: float | None = None
    sma_200: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    bb_width: float | None = None
    bb_position: float | None = None
