from __future__ import annotations

import logging
from pathlib import Path

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db
from src.api.schemas.portfolio import (
    BudgetAllocation,
    BudgetResponse,
    PortfolioHolding,
    PortfolioResponse,
    PortfolioSummaryResponse,
    RebalanceOrder,
    RebalanceResponse,
)
from src.api.services.greyscale_reader import GreyscaleReader
from src.data.db.models import Stock

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])
GREYSCALE_REPORT_DIR = Path("data/reports/greyscale")
_READER: GreyscaleReader | None = None
_READER_DIR: Path | None = None


def _get_reader() -> GreyscaleReader:
    global _READER, _READER_DIR

    if _READER is None or _READER_DIR != GREYSCALE_REPORT_DIR:
        _READER = GreyscaleReader(report_dir=GREYSCALE_REPORT_DIR)
        _READER_DIR = GREYSCALE_REPORT_DIR
    return _READER


async def _enrich_with_stock_info(
    db: AsyncSession, tickers: list[str]
) -> dict[str, dict[str, str | None]]:
    """Return {TICKER: {sector, company_name}} for the given tickers.

    Returns {} on DB failure — caller should treat enrichment as best-effort.
    Core portfolio data comes from greyscale report files, so DB issues should
    not turn /api/portfolio/current into a 500.
    """
    if not tickers:
        return {}
    normalized = [t.upper() for t in tickers]
    try:
        result = await db.execute(
            sa.select(Stock.ticker, Stock.sector, Stock.company_name).where(
                Stock.ticker.in_(normalized)
            )
        )
        return {
            ticker.upper(): {"sector": sector, "company_name": company_name}
            for ticker, sector, company_name in result.all()
        }
    except SQLAlchemyError as exc:
        logger.warning("portfolio sector/company_name enrich skipped (DB error): %s", exc)
        return {}


@router.get("/current", response_model=PortfolioResponse)
async def get_current_portfolio(
    db: AsyncSession = Depends(get_db),
) -> PortfolioResponse:
    reader = _get_reader()
    summary = reader.get_portfolio_summary()
    holdings = reader.get_portfolio_holdings()
    try:
        stock_info = await _enrich_with_stock_info(db, [h["ticker"] for h in holdings])
    except Exception as exc:  # noqa: BLE001 — never let enrich failure kill the page
        logger.warning("portfolio sector enrich raised unexpected: %s", exc)
        stock_info = {}

    return PortfolioResponse(
        signal_date=summary.get("signal_date") if summary else None,
        week_number=summary.get("week_number") if summary else None,
        holding_count=summary.get("holding_count") if summary else None,
        gross_exposure=summary.get("gross_exposure") if summary else None,
        cash_weight=summary.get("cash_weight") if summary else None,
        portfolio_beta=summary.get("portfolio_beta") if summary else None,
        cvar_95=summary.get("cvar_95") if summary else None,
        turnover=summary.get("turnover") if summary else None,
        risk_pass=summary.get("risk_pass") if summary else None,
        holdings=[
            PortfolioHolding(
                ticker=holding["ticker"],
                weight=holding["weight"],
                score=holding.get("score"),
                sector=(stock_info.get(holding["ticker"].upper()) or {}).get("sector"),
                company_name=(stock_info.get(holding["ticker"].upper()) or {}).get("company_name"),
            )
            for holding in holdings
        ],
    )


@router.get("/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary() -> PortfolioSummaryResponse:
    reader = _get_reader()
    summary = reader.get_portfolio_summary()
    if summary is None:
        return PortfolioSummaryResponse()
    return PortfolioSummaryResponse(**summary)


@router.get("/budget", response_model=BudgetResponse)
async def get_budget_allocation(
    total_budget: float = Query(default=100000, ge=1000, le=100_000_000, description="Total investment budget in USD"),
) -> BudgetResponse:
    reader = _get_reader()
    holdings = reader.get_portfolio_holdings()
    allocations = [
        BudgetAllocation(
            ticker=holding["ticker"],
            weight=holding["weight"],
            dollar_amount=round(holding["weight"] * total_budget, 2),
        )
        for holding in holdings
    ]
    return BudgetResponse(total_budget=total_budget, allocations=allocations)


@router.get("/rebalance", response_model=RebalanceResponse)
async def get_rebalance_orders() -> RebalanceResponse:
    reader = _get_reader()
    reports = reader.get_reports()
    weeks = sorted(reports)

    if len(weeks) < 2:
        latest = reader.get_latest_report()
        holdings = reader.get_portfolio_holdings()
        return RebalanceResponse(
            signal_date=latest.get("live_outputs", {}).get("signal_date") if latest else None,
            orders=[
                RebalanceOrder(
                    ticker=holding["ticker"],
                    action="buy",
                    weight_prev=0.0,
                    weight_new=holding["weight"],
                    weight_delta=holding["weight"],
                )
                for holding in holdings
            ],
        )

    previous_weights = reports[weeks[-2]].get("live_outputs", {}).get("target_weights_after_risk", {})
    current_report = reports[weeks[-1]]
    current_weights = current_report.get("live_outputs", {}).get("target_weights_after_risk", {})
    all_tickers = sorted(set(previous_weights) | set(current_weights))

    orders = [
        RebalanceOrder(
            ticker=ticker,
            action=_get_rebalance_action(float(current_weights.get(ticker, 0.0)) - float(previous_weights.get(ticker, 0.0))),
            weight_prev=round(float(previous_weights.get(ticker, 0.0)), 6),
            weight_new=round(float(current_weights.get(ticker, 0.0)), 6),
            weight_delta=round(float(current_weights.get(ticker, 0.0)) - float(previous_weights.get(ticker, 0.0)), 6),
        )
        for ticker in all_tickers
    ]
    orders.sort(key=lambda order: abs(order.weight_delta), reverse=True)
    return RebalanceResponse(
        signal_date=current_report.get("live_outputs", {}).get("signal_date"),
        orders=orders,
    )


def _get_rebalance_action(weight_delta: float) -> str:
    if abs(weight_delta) < 1e-6:
        return "hold"
    if weight_delta > 0.0:
        return "buy"
    return "sell"
