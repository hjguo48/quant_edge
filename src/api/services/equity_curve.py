from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import exchange_calendars as xcals
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.portfolio import (
    EquityCurvePoint,
    EquityCurveResponse,
)
from src.data.db.models import PaperPortfolioAudit, StockPrice

BENCHMARK_TICKER = "SPY"
XNYS = xcals.get_calendar("XNYS")


async def compute_portfolio_equity_curve(
    db: AsyncSession,
    *,
    bundle_version: str | None = None,
    as_of: datetime | None = None,
) -> EquityCurveResponse:
    """Rebalance-aware daily NAV curve across all signal_dates in the bundle.

    Each signal_date is a rebalance event at T+1 session (open of next trading day).
    Between rebalances, daily return uses adj_close-to-adj_close on the active weights.
    Missing tickers at rebalance time are dropped + remaining weights renormalized.
    """
    # UI 默认请求加 +2d buffer 看穿 historical PIT 写入延迟 (knowledge_time = T+1 16:00 ET).
    # buffer 只用于 PIT 查询; end_date 仍以真实当前时间为上限, 防止 series 包含未来日期.
    # 显式传 as_of 时两者都用显式值, 保留严格 PIT 语义, 对应 PR #47 stocks router 处理.
    real_now_utc = as_of if as_of is not None else datetime.now(timezone.utc)
    as_of_utc = as_of if as_of is not None else real_now_utc + timedelta(days=2)
    resolved_bundle = bundle_version or await _latest_bundle_version(db)
    if resolved_bundle is None:
        return EquityCurveResponse()

    audit_rows = await _load_paper_portfolio_rows(db, bundle_version=resolved_bundle)
    if not audit_rows:
        return EquityCurveResponse(bundle_version=resolved_bundle)

    grouped: dict[date, dict[str, float]] = defaultdict(dict)
    for signal_date, ticker, weight in audit_rows:
        grouped[signal_date][ticker.upper()] = float(weight)

    signal_dates = sorted(grouped)
    rebalance_map: dict[date, date] = {}
    for sd in signal_dates:
        sd_ts = pd.Timestamp(sd)
        entry = (
            XNYS.next_session(sd_ts)
            if XNYS.is_session(sd_ts)
            else XNYS.date_to_session(sd_ts, direction="next")
        )
        rebalance_map[sd] = pd.Timestamp(entry).date()

    sorted_rebalances = sorted(rebalance_map.values())
    start_date = sorted_rebalances[0]

    real_now_date = real_now_utc.date()
    real_now_ts = pd.Timestamp(real_now_date)
    end_session = (
        real_now_ts
        if XNYS.is_session(real_now_ts)
        else XNYS.date_to_session(real_now_ts, direction="previous")
    )
    end_date = pd.Timestamp(end_session).date()
    if end_date < start_date:
        return EquityCurveResponse(bundle_version=resolved_bundle)

    all_tickers: set[str] = {BENCHMARK_TICKER}
    for weights in grouped.values():
        all_tickers.update(weights)

    price_rows = await _load_adj_close(
        db,
        tickers=sorted(all_tickers),
        start_date=start_date,
        end_date=end_date,
        as_of=as_of_utc,
    )
    price_by_ticker = _price_lookup(price_rows)

    sessions = [
        pd.Timestamp(s).date()
        for s in XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ]
    if not sessions:
        return EquityCurveResponse(bundle_version=resolved_bundle)

    rebalance_set = set(sorted_rebalances)
    sd_by_entry = sorted(
        ((rebalance_map[sd], sd) for sd in signal_dates),
        key=lambda pair: pair[0],
    )

    series: list[EquityCurvePoint] = []
    portfolio_nav = 1.0
    spy_nav = 1.0
    active_weights: dict[str, float] = {}
    prev_session: date | None = None
    sd_idx = 0

    for sess in sessions:
        # Apply any rebalance events scheduled for today (re-snap weights, NAV unchanged)
        while sd_idx < len(sd_by_entry) and sd_by_entry[sd_idx][0] == sess:
            _, sd = sd_by_entry[sd_idx]
            raw_weights = grouped[sd]
            usable = {
                ticker: weight
                for ticker, weight in raw_weights.items()
                if price_by_ticker.get(ticker, {}).get(sess) is not None
            }
            orig_sum = sum(raw_weights.values())
            remain_sum = sum(usable.values())
            if orig_sum > 0.0 and remain_sum > 0.0:
                scale = orig_sum / remain_sum
                active_weights = {ticker: weight * scale for ticker, weight in usable.items()}
            sd_idx += 1

        if prev_session is None:
            daily_port_return = 0.0
            daily_spy_return = 0.0
        else:
            daily_port_return = 0.0
            for ticker, weight in active_weights.items():
                px_today = price_by_ticker.get(ticker, {}).get(sess)
                px_prev = price_by_ticker.get(ticker, {}).get(prev_session)
                if px_today is None or px_prev is None or px_prev <= 0.0:
                    continue
                daily_port_return += weight * ((px_today - px_prev) / px_prev)
            spy_today = price_by_ticker.get(BENCHMARK_TICKER, {}).get(sess)
            spy_prev = price_by_ticker.get(BENCHMARK_TICKER, {}).get(prev_session)
            if spy_today is not None and spy_prev is not None and spy_prev > 0.0:
                daily_spy_return = (spy_today - spy_prev) / spy_prev
            else:
                daily_spy_return = 0.0

        portfolio_nav *= 1.0 + daily_port_return
        spy_nav *= 1.0 + daily_spy_return

        series.append(
            EquityCurvePoint(
                date=sess.isoformat(),
                portfolio_nav=float(portfolio_nav),
                spy_nav=float(spy_nav),
                portfolio_cum_return=float(portfolio_nav - 1.0),
                spy_cum_return=float(spy_nav - 1.0),
                excess_cum_return=float((portfolio_nav - 1.0) - (spy_nav - 1.0)),
                is_rebalance=sess in rebalance_set,
            )
        )
        prev_session = sess

    return EquityCurveResponse(
        bundle_version=resolved_bundle,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        rebalance_dates=[d.isoformat() for d in sorted_rebalances],
        series=series,
    )


async def _latest_bundle_version(db: AsyncSession) -> str | None:
    result = await db.execute(
        sa.select(PaperPortfolioAudit.bundle_version)
        .order_by(PaperPortfolioAudit.generated_at_utc.desc())
        .limit(1)
    )
    value = result.scalar_one_or_none()
    return None if value is None else str(value)


async def _load_paper_portfolio_rows(
    db: AsyncSession,
    *,
    bundle_version: str,
) -> list[tuple[date, str, Decimal]]:
    result = await db.execute(
        sa.select(
            PaperPortfolioAudit.signal_date,
            PaperPortfolioAudit.ticker,
            PaperPortfolioAudit.target_weight,
        )
        .where(PaperPortfolioAudit.bundle_version == bundle_version)
        .order_by(PaperPortfolioAudit.signal_date.asc(), PaperPortfolioAudit.ticker.asc())
    )
    return [(row[0], str(row[1]), row[2]) for row in result.all()]


async def _load_adj_close(
    db: AsyncSession,
    *,
    tickers: list[str],
    start_date: date,
    end_date: date,
    as_of: datetime,
) -> list[tuple[str, date, Decimal]]:
    result = await db.execute(
        sa.select(StockPrice.ticker, StockPrice.trade_date, StockPrice.adj_close)
        .where(
            StockPrice.ticker.in_(tickers),
            StockPrice.trade_date >= start_date,
            StockPrice.trade_date <= end_date,
            StockPrice.knowledge_time <= as_of,
            StockPrice.adj_close.is_not(None),
        )
        .order_by(StockPrice.ticker.asc(), StockPrice.trade_date.asc())
    )
    return [(str(row[0]).upper(), row[1], row[2]) for row in result.all()]


def _price_lookup(rows: list[tuple[str, date, Decimal]]) -> dict[str, dict[date, float]]:
    lookup: dict[str, dict[date, float]] = defaultdict(dict)
    for ticker, trade_date, price in rows:
        lookup[ticker.upper()][trade_date] = float(price)
    return lookup
