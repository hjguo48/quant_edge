from __future__ import annotations

import asyncio
import math
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.api.schemas.backtest import ExpectationStats
from src.api.schemas.portfolio import EquityCurvePoint, EquityCurveResponse
from src.api.services import backtest_vs_live as svc

TRUTH_TABLE_FIXTURE = {
    "as_of": "2026-04-24",
    "horizon_days": 60,
    "cost_model_base": "almgren_chriss_sqrt",
    "verdict": {
        "champion": {
            "strategy": "score_weighted_buffered",
            "cost_mult": 0.75,
            "gate_on": False,
            "net_ann_excess": 0.0735,
            "gross_ann_excess": 0.0875,
            "ir": 0.7167,
            "sharpe": 0.8388,
            "max_drawdown": 0.1404,
            "avg_turnover": 0.0265,
            "cost_drag_ann": 0.0140,
            "n_periods": 328,
        }
    },
}


def _make_equity(n_points: int) -> EquityCurveResponse:
    series = [
        EquityCurvePoint(
            date=f"2026-05-{day + 1:02d}",
            portfolio_nav=1.0 + 0.001 * day,
            spy_nav=1.0,
            portfolio_cum_return=0.001 * day,
            spy_cum_return=0.0,
            excess_cum_return=0.001 * day,
            is_rebalance=day == 0,
        )
        for day in range(n_points)
    ]
    return EquityCurveResponse(
        bundle_version="test_bundle",
        start_date=series[0].date if series else None,
        end_date=series[-1].date if series else None,
        rebalance_dates=[series[0].date] if series else [],
        series=series,
    )


def test_cone_math_anchored_sqrt_band() -> None:
    asyncio.run(_run_cone_math())


async def _run_cone_math() -> None:
    expectation = ExpectationStats(weekly_excess_mean=0.005, weekly_excess_std=0.01)
    with (
        patch.object(svc, "_load_json", side_effect=[TRUTH_TABLE_FIXTURE, None]),
        patch.object(svc, "_compute_expectation_stats", return_value=expectation),
        patch.object(svc, "compute_portfolio_equity_curve", AsyncMock(return_value=_make_equity(26))),
    ):
        resp = await svc.compute_backtest_vs_live(AsyncMock())

    assert resp.champion is not None
    assert resp.champion.n_periods == 328
    assert resp.champion.net_ann_excess == pytest.approx(0.0735)
    assert len(resp.cone) == len(resp.live) == 26

    mu_d = 0.005 / 5
    sigma_d = 0.01 / math.sqrt(5)

    # k=0: anchored at zero with zero band width
    k0 = resp.cone[0]
    assert k0.day_index == 0
    assert k0.expected == pytest.approx(0.0)
    assert k0.upper_1s == pytest.approx(0.0)
    assert k0.lower_2s == pytest.approx(0.0)

    # k=25: band = z * sigma_d * sqrt(25) = z * 5 * sigma_d
    k25 = resp.cone[25]
    assert k25.day_index == 25
    assert k25.expected == pytest.approx(mu_d * 25)
    assert k25.upper_1s == pytest.approx(mu_d * 25 + 5 * sigma_d)
    assert k25.lower_1s == pytest.approx(mu_d * 25 - 5 * sigma_d)
    assert k25.upper_2s == pytest.approx(mu_d * 25 + 10 * sigma_d)
    assert k25.lower_2s == pytest.approx(mu_d * 25 - 10 * sigma_d)

    # comparison: live weekly excess = mean daily increment (0.001) * 5
    assert resp.comparison["weekly_excess"].live == pytest.approx(0.005)
    assert resp.comparison["weekly_excess"].backtest == pytest.approx(0.005)
    assert resp.comparison["ic"].note is not None


def test_missing_truth_table_returns_valid_empty() -> None:
    asyncio.run(_run_missing_truth_table())


async def _run_missing_truth_table() -> None:
    with (
        patch.object(svc, "_load_json", return_value=None),
        patch.object(svc, "compute_portfolio_equity_curve", AsyncMock(return_value=_make_equity(3))),
    ):
        resp = await svc.compute_backtest_vs_live(AsyncMock())

    assert resp.champion is None
    assert resp.expectation is None
    assert resp.cone == []
    assert len(resp.live) == 3
    assert resp.comparison["turnover"].backtest is None


@pytest.mark.skipif(
    not (Path("data/reports/w10_truth_table_60d.json").is_file() and Path("data/reports/w10_truth_table_60d_periods.parquet").is_file()),
    reason="real W10 truth table artifacts not present",
)
def test_expectation_stats_from_real_truth_table() -> None:
    import json

    truth_table = json.loads(Path("data/reports/w10_truth_table_60d.json").read_text())
    stats = svc._compute_expectation_stats(truth_table)
    assert stats is not None
    assert stats.source == "truth_table_periods"
    # champion 子集应是 328 个周度样本, 周度超额均值在合理范围内
    assert abs(stats.weekly_excess_mean) < 0.02
    assert 0.0 < stats.weekly_excess_std < 0.10
