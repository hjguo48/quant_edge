from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, TypeVar
from zoneinfo import ZoneInfo

import httpx
import sqlalchemy as sa
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from src.config import settings
from src.api.deps import get_db, get_redis
from src.api.schemas.market import (
    IndexPriceBar,
    MarketBreadth,
    MarketIndicesResponse,
    MarketOverviewResponse,
    MarketSectorsResponse,
    MarketSnapshot,
    SectorPerformanceItem,
    VolatilitySnapshot,
)
from src.data.db.models import Stock, StockPrice

router = APIRouter(prefix="/api/market", tags=["Market"])

MACRO_SERIES_TABLE = sa.table(
    "macro_series_pit",
    sa.column("id"),
    sa.column("series_id"),
    sa.column("observation_date"),
    sa.column("value"),
    sa.column("knowledge_time"),
)

CACHE_TTL_SECONDS = 300
LATEST_TRADE_DATE_CACHE_KEY = "quantedge:market:latest_trade_date"
OVERVIEW_CACHE_PREFIX = "quantedge:market:overview"
SECTORS_CACHE_PREFIX = "quantedge:market:sectors"
POLYGON_INTRADAY_BASE_URL = "https://api.polygon.io/v2/aggs/ticker"
MARKET_TZ = ZoneInfo("America/New_York")

ModelT = TypeVar("ModelT", bound=BaseModel)


def _resolve_as_of(as_of: datetime | None) -> datetime:
    resolved = as_of or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        return resolved.replace(tzinfo=timezone.utc)
    return resolved.astimezone(timezone.utc)


def _should_use_cache(as_of: datetime | None) -> bool:
    return as_of is None


def _overview_cache_key(trade_date: date) -> str:
    return f"{OVERVIEW_CACHE_PREFIX}:{trade_date.isoformat()}"


def _sectors_cache_key(trade_date: date, days: int) -> str:
    return f"{SECTORS_CACHE_PREFIX}:{trade_date.isoformat()}:{days}"


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _to_percent(change_ratio: Any) -> float | None:
    value = _to_float(change_ratio)
    if value is None:
        return None
    return value * 100.0


def _resolve_intraday_date(raw_date: str | None) -> str:
    if raw_date:
        return date.fromisoformat(raw_date).isoformat()
    return datetime.now(MARKET_TZ).date().isoformat()


def _resolve_intraday_range(
    *,
    raw_date: str | None,
    raw_from_date: str | None,
    raw_to_date: str | None,
) -> tuple[str, str]:
    if raw_from_date or raw_to_date:
        resolved_to = date.fromisoformat(raw_to_date).isoformat() if raw_to_date else _resolve_intraday_date(None)
        resolved_from = (
            date.fromisoformat(raw_from_date).isoformat()
            if raw_from_date
            else resolved_to
        )
        if resolved_from > resolved_to:
            raise HTTPException(status_code=400, detail="from_date must be on or before to_date")
        return resolved_from, resolved_to

    resolved_date = _resolve_intraday_date(raw_date)
    return resolved_date, resolved_date


async def _load_cached_model(redis: Redis, key: str, model_cls: type[ModelT]) -> ModelT | None:
    payload = await redis.get(key)
    if not payload:
        return None
    try:
        return model_cls.model_validate_json(payload)
    except Exception:
        await redis.delete(key)
        return None


async def _store_cached_model(redis: Redis, key: str, model: BaseModel) -> None:
    await redis.set(key, model.model_dump_json(), ex=CACHE_TTL_SECONDS)


async def _load_cached_latest_trade_date(redis: Redis) -> date | None:
    raw_value = await redis.get(LATEST_TRADE_DATE_CACHE_KEY)
    if not raw_value:
        return None
    try:
        return date.fromisoformat(raw_value)
    except ValueError:
        await redis.delete(LATEST_TRADE_DATE_CACHE_KEY)
        return None


async def _store_cached_latest_trade_date(redis: Redis, trade_date: date) -> None:
    await redis.set(
        LATEST_TRADE_DATE_CACHE_KEY,
        trade_date.isoformat(),
        ex=CACHE_TTL_SECONDS,
    )


async def _fetch_visible_trade_date(
    db: AsyncSession,
    *,
    as_of: datetime,
    before: date | None = None,
) -> date | None:
    conditions: list[Any] = [StockPrice.knowledge_time <= as_of]
    if before is None:
        conditions.append(StockPrice.trade_date <= as_of.date())
    else:
        conditions.append(StockPrice.trade_date < before)

    statement = (
        sa.select(StockPrice.trade_date)
        .where(*conditions)
        .order_by(StockPrice.trade_date.desc())
        .limit(1)
    )
    return (await db.execute(statement)).scalar_one_or_none()


async def _fetch_recent_trade_dates(
    db: AsyncSession,
    *,
    limit: int,
    as_of: datetime,
) -> list[date]:
    trade_dates: list[date] = []
    before: date | None = None

    for _ in range(limit):
        trade_date = await _fetch_visible_trade_date(db, as_of=as_of, before=before)
        if trade_date is None:
            break
        trade_dates.append(trade_date)
        before = trade_date

    return trade_dates


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


async def _fetch_latest_day_market_rows(
    db: AsyncSession,
    *,
    latest_trade_date: date,
    previous_trade_date: date,
    as_of: datetime,
) -> list[dict[str, Any]]:
    current_price = aliased(StockPrice)
    previous_price = aliased(StockPrice)
    current_value = sa.func.coalesce(current_price.adj_close, current_price.close)
    previous_value = sa.func.coalesce(previous_price.adj_close, previous_price.close)

    statement = (
        sa.select(
            current_price.ticker.label("ticker"),
            sa.func.coalesce(Stock.sector, sa.literal("Unknown")).label("sector"),
            current_price.trade_date.label("trade_date"),
            current_price.volume.label("volume"),
            current_value.label("price"),
            previous_value.label("previous_price"),
            ((current_value / sa.func.nullif(previous_value, 0)) - 1).label("change_ratio"),
        )
        .select_from(current_price)
        .join(
            previous_price,
            sa.and_(
                previous_price.ticker == current_price.ticker,
                previous_price.trade_date == previous_trade_date,
                previous_price.knowledge_time <= as_of,
            ),
        )
        .join(Stock, Stock.ticker == current_price.ticker)
        .where(
            current_price.trade_date == latest_trade_date,
            current_price.knowledge_time <= as_of,
            current_value.is_not(None),
            previous_value.is_not(None),
        )
        .order_by(current_price.ticker)
    )
    return [dict(row) for row in (await db.execute(statement)).mappings().all()]


async def _fetch_market_change_rows(
    db: AsyncSession,
    *,
    days: int,
    as_of: datetime,
) -> tuple[list[dict[str, Any]], date | None, date | None]:
    if days == 1:
        latest_trade_date = await _fetch_visible_trade_date(db, as_of=as_of)
        if latest_trade_date is None:
            return [], None, None

        previous_trade_date = await _fetch_visible_trade_date(
            db,
            as_of=as_of,
            before=latest_trade_date,
        )
        if previous_trade_date is None:
            return [], latest_trade_date, latest_trade_date

        rows = await _fetch_latest_day_market_rows(
            db,
            latest_trade_date=latest_trade_date,
            previous_trade_date=previous_trade_date,
            as_of=as_of,
        )
        return rows, latest_trade_date, latest_trade_date

    recent_dates = await _fetch_recent_trade_dates(db, limit=days + 1, as_of=as_of)
    if len(recent_dates) < 2:
        return [], None, None

    price_value = sa.func.coalesce(StockPrice.adj_close, StockPrice.close)
    statement = (
        sa.select(
            StockPrice.ticker.label("ticker"),
            sa.func.coalesce(Stock.sector, sa.literal("Unknown")).label("sector"),
            StockPrice.trade_date.label("trade_date"),
            StockPrice.volume.label("volume"),
            price_value.label("price"),
        )
        .select_from(StockPrice)
        .join(Stock, Stock.ticker == StockPrice.ticker)
        .where(
            StockPrice.trade_date.in_(recent_dates),
            StockPrice.knowledge_time <= as_of,
            price_value.is_not(None),
        )
        .order_by(StockPrice.ticker, StockPrice.trade_date)
    )
    raw_rows = [dict(row) for row in (await db.execute(statement)).mappings().all()]
    if not raw_rows:
        return [], None, None

    change_rows: list[dict[str, Any]] = []
    previous_prices: dict[str, float] = {}

    for row in raw_rows:
        ticker = str(row["ticker"])
        price = _to_float(row["price"])
        if price is None:
            continue

        previous_price = previous_prices.get(ticker)
        if previous_price not in (None, 0):
            change_rows.append(
                {
                    "ticker": ticker,
                    "sector": row["sector"],
                    "trade_date": row["trade_date"],
                    "volume": row["volume"],
                    "change_ratio": (price / previous_price) - 1.0,
                },
            )
        previous_prices[ticker] = price

    if not change_rows:
        return [], None, None

    return change_rows, min(row["trade_date"] for row in change_rows), max(
        row["trade_date"] for row in change_rows
    )


async def _fetch_vix_snapshot(
    db: AsyncSession,
    *,
    as_of: datetime,
) -> VolatilitySnapshot | None:
    statement = (
        sa.select(
            MACRO_SERIES_TABLE.c.series_id,
            MACRO_SERIES_TABLE.c.observation_date,
            MACRO_SERIES_TABLE.c.value,
            MACRO_SERIES_TABLE.c.knowledge_time,
        )
        .where(
            MACRO_SERIES_TABLE.c.series_id == "VIXCLS",
            MACRO_SERIES_TABLE.c.knowledge_time <= as_of,
            MACRO_SERIES_TABLE.c.observation_date <= as_of.date(),
        )
        .order_by(
            MACRO_SERIES_TABLE.c.observation_date.desc(),
            MACRO_SERIES_TABLE.c.knowledge_time.desc(),
            MACRO_SERIES_TABLE.c.id.desc(),
        )
        .limit(1)
    )
    row = (await db.execute(statement)).mappings().first()
    if row is None:
        return None

    value = _to_float(row["value"])
    if value is None:
        return None

    return VolatilitySnapshot(
        series_id=str(row["series_id"]),
        observation_date=row["observation_date"],
        value=value,
        knowledge_time=row["knowledge_time"],
    )


def _build_sector_items(rows: list[dict[str, Any]]) -> list[SectorPerformanceItem]:
    if not rows:
        return []

    buckets: dict[str, dict[str, float | int]] = {}
    for row in rows:
        sector = str(row.get("sector") or "Unknown")
        bucket = buckets.setdefault(
            sector,
            {
                "sum_change_ratio": 0.0,
                "ticker_count": 0,
                "total_volume": 0,
            },
        )
        change_ratio = _to_float(row.get("change_ratio"))
        if change_ratio is None:
            continue
        bucket["sum_change_ratio"] += change_ratio
        bucket["ticker_count"] += 1
        bucket["total_volume"] += _to_int(row.get("volume")) or 0

    items = [
        SectorPerformanceItem(
            sector=sector,
            avg_change_pct=_to_percent(bucket["sum_change_ratio"] / bucket["ticker_count"])
            if bucket["ticker_count"]
            else None,
            total_volume=int(bucket["total_volume"]),
            ticker_count=int(bucket["ticker_count"]),
        )
        for sector, bucket in buckets.items()
        if bucket["ticker_count"]
    ]
    return sorted(
        items,
        key=lambda item: (
            float("-inf") if item.avg_change_pct is None else -item.avg_change_pct,
            item.sector,
        ),
    )


def _build_breadth(rows: list[dict[str, Any]]) -> MarketBreadth:
    if not rows:
        return MarketBreadth()

    advancing = 0
    declining = 0
    unchanged = 0

    for row in rows:
        change_ratio = _to_float(row.get("change_ratio"))
        if change_ratio is None:
            continue
        if abs(change_ratio) < 1e-12:
            unchanged += 1
        elif change_ratio > 0:
            advancing += 1
        else:
            declining += 1

    total = advancing + declining + unchanged
    return MarketBreadth(
        advancing=advancing,
        declining=declining,
        unchanged=unchanged,
        total=total,
        advance_decline_ratio=(advancing / declining) if declining else None,
        advance_pct=(advancing / total) * 100.0 if total else None,
    )


def _build_market_snapshot(ticker: str, rows: list[dict[str, Any]]) -> MarketSnapshot | None:
    if not rows:
        return None

    latest = rows[-1]
    previous = rows[-2] if len(rows) >= 2 else None

    latest_price = _to_float(latest.get("close"))
    if latest_price is None:
        latest_price = _to_float(latest.get("adj_close"))

    previous_close = _to_float(previous.get("close")) if previous else None
    if previous_close is None and previous is not None:
        previous_close = _to_float(previous.get("adj_close"))

    change = None
    change_pct = None
    if latest_price is not None and previous_close not in (None, 0):
        change = latest_price - previous_close
        change_pct = ((latest_price / previous_close) - 1.0) * 100.0

    if latest_price is None:
        return None

    return MarketSnapshot(
        ticker=ticker,
        trade_date=latest["trade_date"],
        price=latest_price,
        previous_close=previous_close,
        change=change,
        change_pct=change_pct,
        volume=_to_int(latest.get("volume")),
    )


def _build_index_bars(rows: list[dict[str, Any]]) -> list[IndexPriceBar]:
    return [
        IndexPriceBar(
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


@router.get(
    "/overview",
    response_model=MarketOverviewResponse,
    response_model_exclude_none=True,
    summary="Market overview",
)
async def get_market_overview(
    as_of: datetime | None = Query(
        default=None,
        description="Point-in-time cutoff in UTC. Defaults to the current UTC timestamp.",
    ),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> MarketOverviewResponse:
    use_cache = _should_use_cache(as_of)
    if use_cache:
        cached_trade_date = await _load_cached_latest_trade_date(redis)
        if cached_trade_date is not None:
            cached_response = await _load_cached_model(
                redis,
                _overview_cache_key(cached_trade_date),
                MarketOverviewResponse,
            )
            if cached_response is not None:
                return cached_response

    as_of_ts = _resolve_as_of(as_of)
    latest_trade_date = await _fetch_visible_trade_date(db, as_of=as_of_ts)
    if use_cache and latest_trade_date is not None:
        await _store_cached_latest_trade_date(redis, latest_trade_date)
        cached_response = await _load_cached_model(
            redis,
            _overview_cache_key(latest_trade_date),
            MarketOverviewResponse,
        )
        if cached_response is not None:
            return cached_response

    market_rows: list[dict[str, Any]] = []
    if latest_trade_date is not None:
        previous_trade_date = await _fetch_visible_trade_date(
            db,
            as_of=as_of_ts,
            before=latest_trade_date,
        )
        if previous_trade_date is not None:
            market_rows = await _fetch_latest_day_market_rows(
                db,
                latest_trade_date=latest_trade_date,
                previous_trade_date=previous_trade_date,
                as_of=as_of_ts,
            )

    spy_rows = await _fetch_price_history(db, ticker="SPY", limit=2, as_of=as_of_ts)
    vix = await _fetch_vix_snapshot(db, as_of=as_of_ts)
    response = MarketOverviewResponse(
        as_of=as_of_ts,
        latest_trade_date=latest_trade_date,
        spy=_build_market_snapshot("SPY", spy_rows),
        breadth=_build_breadth(market_rows),
        sectors=_build_sector_items(market_rows),
        vix=vix,
    )

    if use_cache and latest_trade_date is not None:
        await _store_cached_model(redis, _overview_cache_key(latest_trade_date), response)

    return response


@router.get(
    "/indices",
    response_model=MarketIndicesResponse,
    response_model_exclude_none=True,
    summary="SPY index history",
)
async def get_market_indices(
    days: int = Query(default=30, ge=1, le=2520, description="Number of trading days to return."),
    as_of: datetime | None = Query(
        default=None,
        description="Point-in-time cutoff in UTC. Defaults to the current UTC timestamp.",
    ),
    db: AsyncSession = Depends(get_db),
) -> MarketIndicesResponse:
    as_of_ts = _resolve_as_of(as_of)
    rows = await _fetch_price_history(db, ticker="SPY", limit=days, as_of=as_of_ts)
    prices = _build_index_bars(rows)

    return MarketIndicesResponse(
        ticker="SPY",
        as_of=as_of_ts,
        days=days,
        start_date=prices[0].trade_date if prices else None,
        end_date=prices[-1].trade_date if prices else None,
        prices=prices,
    )


@router.get(
    "/sectors",
    response_model=MarketSectorsResponse,
    response_model_exclude_none=True,
    summary="Sector performance",
)
async def get_market_sectors(
    days: int = Query(default=1, ge=1, le=2520, description="Number of trading days to aggregate."),
    as_of: datetime | None = Query(
        default=None,
        description="Point-in-time cutoff in UTC. Defaults to the current UTC timestamp.",
    ),
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
) -> MarketSectorsResponse:
    use_cache = _should_use_cache(as_of)
    if use_cache:
        cached_trade_date = await _load_cached_latest_trade_date(redis)
        if cached_trade_date is not None:
            cached_response = await _load_cached_model(
                redis,
                _sectors_cache_key(cached_trade_date, days),
                MarketSectorsResponse,
            )
            if cached_response is not None:
                return cached_response

    as_of_ts = _resolve_as_of(as_of)
    rows, start_date, end_date = await _fetch_market_change_rows(db, days=days, as_of=as_of_ts)
    response = MarketSectorsResponse(
        as_of=as_of_ts,
        days=days,
        start_date=start_date,
        end_date=end_date,
        sectors=_build_sector_items(rows),
    )

    if use_cache and end_date is not None:
        await _store_cached_latest_trade_date(redis, end_date)
        await _store_cached_model(redis, _sectors_cache_key(end_date, days), response)

    return response


@router.get(
    "/intraday",
    summary="Intraday aggregated bars from Polygon",
)
async def get_intraday(
    ticker: str = Query(..., min_length=1, description="Ticker symbol."),
    multiplier: int = Query(default=1, ge=1),
    timespan: str = Query(default="minute"),
    date: str | None = Query(default=None, description="Trading date in YYYY-MM-DD. Defaults to today in ET."),
    from_date: str | None = Query(default=None, description="Range start date in YYYY-MM-DD."),
    to_date: str | None = Query(default=None, description="Range end date in YYYY-MM-DD."),
) -> dict[str, Any]:
    if not settings.POLYGON_API_KEY:
        raise HTTPException(status_code=503, detail="POLYGON_API_KEY is not configured")

    normalized_ticker = ticker.strip().upper()
    resolved_from_date, resolved_to_date = _resolve_intraday_range(
        raw_date=date,
        raw_from_date=from_date,
        raw_to_date=to_date,
    )
    url = (
        f"{POLYGON_INTRADAY_BASE_URL}/{normalized_ticker}/range/"
        f"{multiplier}/{timespan}/{resolved_from_date}/{resolved_to_date}"
    )
    params = {
        "adjusted": "true",
        "limit": 50_000,
        "sort": "asc",
        "apiKey": settings.POLYGON_API_KEY,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.get(url, params=params)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Polygon request failed: {exc}") from exc

    if response.status_code >= 400:
        detail: Any
        try:
            payload = response.json()
            detail = payload.get("error") or payload.get("message") or payload
        except ValueError:
            detail = response.text or f"Polygon returned status {response.status_code}"
        raise HTTPException(status_code=response.status_code, detail=detail)

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(status_code=502, detail="Polygon returned invalid JSON") from exc
