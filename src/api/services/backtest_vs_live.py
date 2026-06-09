from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.schemas.backtest import (
    BacktestVsLiveResponse,
    ChampionProfile,
    ConePoint,
    ExpectationStats,
    LiveExcessPoint,
    MetricComparison,
)
from src.api.services.equity_curve import compute_portfolio_equity_curve

TRUTH_TABLE_PATH = Path("data/reports/w10_truth_table_60d.json")
PERIODS_PARQUET_PATH = Path("data/reports/w10_truth_table_60d_periods.parquet")
G4_GATE_SUMMARY_PATH = Path("data/reports/greyscale/g4_gate_summary.json")

TRADING_DAYS_PER_WEEK = 5
IC_MATURITY_NOTE = "60d realized IC matures 2026-07-17 (W1 signal + 60 trading days)"


async def compute_backtest_vs_live(db: AsyncSession) -> BacktestVsLiveResponse:
    """Champion backtest profile + expectation cone + live excess overlay.

    The cone projects the backtest's per-period net excess distribution forward
    from the live entry date: expected drift k*mu_d with a +/- z*sigma_d*sqrt(k)
    band. Live data comes from the same equity curve service the Portfolio page
    uses, so both pages tell one story.
    """
    truth_table = await asyncio.to_thread(_load_json, TRUTH_TABLE_PATH)
    champion = _build_champion_profile(truth_table)

    expectation: ExpectationStats | None = None
    if truth_table is not None and champion is not None:
        expectation = await asyncio.to_thread(_compute_expectation_stats, truth_table)

    equity = await compute_portfolio_equity_curve(db)
    live_points = [
        LiveExcessPoint(
            date=point.date,
            excess_cum_return=point.excess_cum_return,
            is_rebalance=point.is_rebalance,
        )
        for point in equity.series
    ]

    cone: list[ConePoint] = []
    if expectation is not None and live_points:
        mu_d = expectation.weekly_excess_mean / TRADING_DAYS_PER_WEEK
        sigma_d = expectation.weekly_excess_std / math.sqrt(TRADING_DAYS_PER_WEEK)
        for k, point in enumerate(live_points):
            band_1s = sigma_d * math.sqrt(k)
            expected = mu_d * k
            cone.append(
                ConePoint(
                    date=point.date,
                    day_index=k,
                    expected=expected,
                    upper_1s=expected + band_1s,
                    lower_1s=expected - band_1s,
                    upper_2s=expected + 2.0 * band_1s,
                    lower_2s=expected - 2.0 * band_1s,
                )
            )

    comparison = await _build_comparison(
        champion=champion,
        expectation=expectation,
        equity_series=equity.series,
    )

    return BacktestVsLiveResponse(
        champion=champion,
        expectation=expectation,
        cone=cone,
        live=live_points,
        comparison=comparison,
    )


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _build_champion_profile(truth_table: dict[str, Any] | None) -> ChampionProfile | None:
    if not truth_table:
        return None
    champion_raw = (truth_table.get("verdict") or {}).get("champion")
    if not isinstance(champion_raw, dict):
        return None
    try:
        return ChampionProfile(
            strategy=str(champion_raw["strategy"]),
            horizon_days=int(truth_table.get("horizon_days", 60)),
            net_ann_excess=float(champion_raw["net_ann_excess"]),
            gross_ann_excess=float(champion_raw["gross_ann_excess"]),
            ir=float(champion_raw["ir"]),
            sharpe=float(champion_raw["sharpe"]),
            max_drawdown=float(champion_raw["max_drawdown"]),
            avg_turnover_weekly=float(champion_raw["avg_turnover"]),
            cost_drag_ann=float(champion_raw["cost_drag_ann"]),
            n_periods=int(champion_raw["n_periods"]),
            backtest_as_of=truth_table.get("as_of"),
            cost_model=truth_table.get("cost_model_base"),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _compute_expectation_stats(truth_table: dict[str, Any]) -> ExpectationStats | None:
    champion_raw = (truth_table.get("verdict") or {}).get("champion") or {}
    if not PERIODS_PARQUET_PATH.is_file():
        return None
    try:
        periods = pd.read_parquet(PERIODS_PARQUET_PATH)
    except (OSError, ValueError, ImportError):
        return None

    mask = (
        (periods["strategy"] == champion_raw.get("strategy"))
        & (periods["cost_mult"] == champion_raw.get("cost_mult"))
        & (periods["gate_on"] == champion_raw.get("gate_on"))
    )
    excess = pd.to_numeric(periods.loc[mask, "net_excess_return"], errors="coerce").dropna()
    if len(excess) < 2:
        return None
    return ExpectationStats(
        weekly_excess_mean=float(excess.mean()),
        weekly_excess_std=float(excess.std(ddof=1)),
        source="truth_table_periods",
    )


async def _build_comparison(
    *,
    champion: ChampionProfile | None,
    expectation: ExpectationStats | None,
    equity_series: list[Any],
) -> dict[str, MetricComparison]:
    live_weekly_excess: float | None = None
    live_max_drawdown: float | None = None
    if len(equity_series) >= 2:
        daily_excess_increments = [
            equity_series[i].excess_cum_return - equity_series[i - 1].excess_cum_return
            for i in range(1, len(equity_series))
        ]
        live_weekly_excess = (
            sum(daily_excess_increments) / len(daily_excess_increments)
        ) * TRADING_DAYS_PER_WEEK

        peak = equity_series[0].portfolio_nav
        worst = 0.0
        for point in equity_series:
            peak = max(peak, point.portfolio_nav)
            if peak > 0:
                worst = min(worst, point.portfolio_nav / peak - 1.0)
        live_max_drawdown = worst

    gate_summary = await asyncio.to_thread(_load_json, G4_GATE_SUMMARY_PATH)
    live_turnover: float | None = None
    if gate_summary is not None:
        raw_turnover = (gate_summary.get("summary") or {}).get("mean_turnover")
        if isinstance(raw_turnover, (int, float)):
            live_turnover = float(raw_turnover)

    return {
        "weekly_excess": MetricComparison(
            backtest=expectation.weekly_excess_mean if expectation else None,
            live=live_weekly_excess,
            unit="weekly",
        ),
        "turnover": MetricComparison(
            backtest=champion.avg_turnover_weekly if champion else None,
            live=live_turnover,
            unit="weekly",
        ),
        "max_drawdown": MetricComparison(
            backtest=champion.max_drawdown if champion else None,
            live=live_max_drawdown,
        ),
        "ic": MetricComparison(backtest=None, live=None, note=IC_MATURITY_NOTE),
    }
