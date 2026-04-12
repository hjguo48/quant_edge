from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import pandas as pd
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.schemas.common import ErrorResponse
from src.api.schemas.stocks import (
    StockDetailResponse,
    StockFundamentalsResponse,
    StockPriceBar,
    StockPricesResponse,
    StockQuote,
    StockTechnicalsResponse,
)
from src.data.db.models import Stock, StockPrice
from src.data.db.pit import get_fundamentals_pit

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])

TECHNICAL_HISTORY_ROWS = 260


def _normalize_ticker(ticker: str) -> str:
    return ticker.strip().upper()


def _resolve_as_of(as_of: datetime | None) -> datetime:
    resolved = as_of or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(value)


async def _get_stock_or_404(db: AsyncSession, ticker: str) -> Stock:
    stock = await db.get(Stock, ticker)
    if stock is None:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' not found.")
    return stock


async def _fetch_price_history(
    db: AsyncSession,
    *,
    ticker: str,
    limit: int,
    as_of: datetime,
) -> list[dict[str, Any]]:
    statement = (
        sa.select(
            StockPrice.trade_date,
            StockPrice.open,
            StockPrice.high,
            StockPrice.low,
            StockPrice.close,
            StockPrice.adj_close,
            StockPrice.volume,
        )
        .where(
            StockPrice.ticker == ticker,
            StockPrice.knowledge_time <= as_of,
            StockPrice.trade_date <= as_of.date(),
        )
        .order_by(StockPrice.trade_date.desc())
        .limit(limit)
    )
    rows = (await db.execute(statement)).mappings().all()
    return [dict(row) for row in reversed(rows)]


def _build_quote(rows: list[dict[str, Any]]) -> StockQuote | None:
    if not rows:
        return None

    latest = rows[-1]
    previous = rows[-2] if len(rows) >= 2 else None

    latest_close = _to_float(latest.get("close"))
    if latest_close is None:
        latest_close = _to_float(latest.get("adj_close"))

    previous_close = _to_float(previous.get("close")) if previous else None
    if previous_close is None and previous is not None:
        previous_close = _to_float(previous.get("adj_close"))

    change = None
    change_pct = None
    if latest_close is not None and previous_close not in (None, 0):
        change = latest_close - previous_close
        change_pct = ((latest_close / previous_close) - 1.0) * 100.0

    return StockQuote(
        trade_date=latest["trade_date"],
        open=_to_float(latest.get("open")),
        high=_to_float(latest.get("high")),
        low=_to_float(latest.get("low")),
        close=_to_float(latest.get("close")),
        adj_close=_to_float(latest.get("adj_close")),
        volume=_to_int(latest.get("volume")),
        previous_close=previous_close,
        change=change,
        change_pct=change_pct,
    )


def _build_price_bars(rows: list[dict[str, Any]]) -> list[StockPriceBar]:
    return [
        StockPriceBar(
            trade_date=row["trade_date"],
            open=_to_float(row.get("open")),
            high=_to_float(row.get("high")),
            low=_to_float(row.get("low")),
            close=_to_float(row.get("close")),
            adj_close=_to_float(row.get("adj_close")),
            volume=_to_int(row.get("volume")),
        )
        for row in rows
    ]


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    return macd_line, signal_line


def _bollinger_levels(series: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    middle = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = (upper - lower) / middle.replace(0, pd.NA)
    position = (series - lower) / (upper - lower).replace(0, pd.NA)
    return pd.DataFrame(
        {
            "bb_upper": upper,
            "bb_middle": middle,
            "bb_lower": lower,
            "bb_width": width,
            "bb_position": position,
        },
    )


def _build_technicals_response(
    ticker: str,
    *,
    as_of: datetime,
    rows: list[dict[str, Any]],
) -> StockTechnicalsResponse:
    if not rows:
        return StockTechnicalsResponse(ticker=ticker, as_of=as_of)

    frame = pd.DataFrame(rows)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    for column in ["close", "adj_close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["price"] = frame["adj_close"].fillna(frame["close"])
    frame.sort_values("trade_date", inplace=True)

    price_series = frame["price"]
    macd_line, macd_signal = _macd(price_series)
    bollinger = _bollinger_levels(price_series)

    latest_index = frame.index[-1]
    latest_trade_date = frame.loc[latest_index, "trade_date"]
    latest_close = _to_float(frame.loc[latest_index, "close"])
    if latest_close is None:
        latest_close = _to_float(frame.loc[latest_index, "price"])

    return StockTechnicalsResponse(
        ticker=ticker,
        as_of=as_of,
        trade_date=latest_trade_date,
        close=latest_close,
        history_points=int(len(frame)),
        rsi_14=_to_float(_rsi(price_series, 14).loc[latest_index]),
        macd=_to_float(macd_line.loc[latest_index]),
        macd_signal=_to_float(macd_signal.loc[latest_index]),
        macd_histogram=_to_float((macd_line - macd_signal).loc[latest_index]),
        sma_20=_to_float(price_series.rolling(20, min_periods=20).mean().loc[latest_index]),
        sma_50=_to_float(price_series.rolling(50, min_periods=50).mean().loc[latest_index]),
        sma_200=_to_float(price_series.rolling(200, min_periods=200).mean().loc[latest_index]),
        bb_upper=_to_float(bollinger.loc[latest_index, "bb_upper"]),
        bb_middle=_to_float(bollinger.loc[latest_index, "bb_middle"]),
        bb_lower=_to_float(bollinger.loc[latest_index, "bb_lower"]),
        bb_width=_to_float(bollinger.loc[latest_index, "bb_width"]),
        bb_position=_to_float(bollinger.loc[latest_index, "bb_position"]),
    )


@router.get(
    "/{ticker}/prices",
    response_model=StockPricesResponse,
    response_model_exclude_none=True,
    responses={404: {"model": ErrorResponse, "description": "Ticker not found."}},
    summary="Historical stock prices",
)
async def get_stock_prices(
    ticker: str,
    days: int = Query(default=90, ge=1, le=2520, description="Number of trading days to return."),
    as_of: datetime | None = Query(
        default=None,
        description="Point-in-time cutoff in UTC. Defaults to the current UTC timestamp.",
    ),
    db: AsyncSession = Depends(get_db),
) -> StockPricesResponse:
    normalized_ticker = _normalize_ticker(ticker)
    as_of_ts = _resolve_as_of(as_of)
    await _get_stock_or_404(db, normalized_ticker)
    rows = await _fetch_price_history(db, ticker=normalized_ticker, limit=days, as_of=as_of_ts)
    prices = _build_price_bars(rows)

    return StockPricesResponse(
        ticker=normalized_ticker,
        as_of=as_of_ts,
        days=days,
        start_date=prices[0].trade_date if prices else None,
        end_date=prices[-1].trade_date if prices else None,
        prices=prices,
    )


@router.get(
    "/{ticker}/fundamentals",
    response_model=StockFundamentalsResponse,
    response_model_exclude_none=True,
    responses={404: {"model": ErrorResponse, "description": "Ticker not found."}},
    summary="Latest visible fundamentals",
)
async def get_stock_fundamentals(
    ticker: str,
    as_of: datetime | None = Query(
        default=None,
        description="Point-in-time cutoff in UTC. Defaults to the current UTC timestamp.",
    ),
    db: AsyncSession = Depends(get_db),
) -> StockFundamentalsResponse:
    normalized_ticker = _normalize_ticker(ticker)
    as_of_ts = _resolve_as_of(as_of)
    await _get_stock_or_404(db, normalized_ticker)

    fundamentals = await asyncio.to_thread(get_fundamentals_pit, normalized_ticker, as_of_ts)
    if fundamentals.empty:
        return StockFundamentalsResponse(ticker=normalized_ticker, as_of=as_of_ts)

    fundamentals = fundamentals.copy()
    fundamentals["event_time"] = pd.to_datetime(fundamentals["event_time"]).dt.date
    fundamentals["knowledge_time"] = pd.to_datetime(fundamentals["knowledge_time"], utc=True)
    fundamentals.sort_values(["event_time", "fiscal_period", "knowledge_time", "id"], inplace=True)

    latest_period = str(fundamentals.iloc[-1]["fiscal_period"])
    latest_period_rows = (
        fundamentals.loc[fundamentals["fiscal_period"] == latest_period]
        .sort_values("metric_name")
        .reset_index(drop=True)
    )

    metrics = {
        str(row["metric_name"]): _to_float(row["metric_value"])
        for row in latest_period_rows.to_dict("records")
    }

    knowledge_time = latest_period_rows["knowledge_time"].max()

    return StockFundamentalsResponse(
        ticker=normalized_ticker,
        as_of=as_of_ts,
        fiscal_period=latest_period,
        event_time=latest_period_rows.iloc[-1]["event_time"],
        knowledge_time=None if pd.isna(knowledge_time) else knowledge_time.to_pydatetime(),
        metric_count=len(metrics),
        metrics=metrics,
    )


@router.get(
    "/{ticker}/technicals",
    response_model=StockTechnicalsResponse,
    response_model_exclude_none=True,
    responses={404: {"model": ErrorResponse, "description": "Ticker not found."}},
    summary="Latest technical indicators",
)
async def get_stock_technicals(
    ticker: str,
    as_of: datetime | None = Query(
        default=None,
        description="Point-in-time cutoff in UTC. Defaults to the current UTC timestamp.",
    ),
    db: AsyncSession = Depends(get_db),
) -> StockTechnicalsResponse:
    normalized_ticker = _normalize_ticker(ticker)
    as_of_ts = _resolve_as_of(as_of)
    await _get_stock_or_404(db, normalized_ticker)
    rows = await _fetch_price_history(
        db,
        ticker=normalized_ticker,
        limit=TECHNICAL_HISTORY_ROWS,
        as_of=as_of_ts,
    )
    return _build_technicals_response(normalized_ticker, as_of=as_of_ts, rows=rows)


@router.get(
    "/{ticker}",
    response_model=StockDetailResponse,
    response_model_exclude_none=True,
    responses={404: {"model": ErrorResponse, "description": "Ticker not found."}},
    summary="Stock detail",
)
async def get_stock_detail(
    ticker: str,
    as_of: datetime | None = Query(
        default=None,
        description="Point-in-time cutoff in UTC. Defaults to the current UTC timestamp.",
    ),
    db: AsyncSession = Depends(get_db),
) -> StockDetailResponse:
    normalized_ticker = _normalize_ticker(ticker)
    as_of_ts = _resolve_as_of(as_of)
    stock = await _get_stock_or_404(db, normalized_ticker)
    rows = await _fetch_price_history(db, ticker=normalized_ticker, limit=2, as_of=as_of_ts)
    latest_price = _build_quote(rows)

    market_cap = None
    shares_outstanding = _to_int(stock.shares_outstanding)
    last_close = latest_price.close if latest_price is not None else None
    if shares_outstanding is not None and last_close is not None:
        market_cap = shares_outstanding * last_close

    return StockDetailResponse(
        ticker=normalized_ticker,
        company_name=stock.company_name,
        sector=stock.sector,
        industry=stock.industry,
        ipo_date=stock.ipo_date,
        delist_date=stock.delist_date,
        delist_reason=stock.delist_reason,
        shares_outstanding=shares_outstanding,
        market_cap=market_cap,
        as_of=as_of_ts,
        latest_price=latest_price,
    )
