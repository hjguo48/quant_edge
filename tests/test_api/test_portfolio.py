from __future__ import annotations

import asyncio
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import portfolio as portfolio_router
from src.api.services import daily_performance as dp_service


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    base_report = {
        "generated_at_utc": "2026-04-12T00:00:00+00:00",
        "score_vectors": {
            "fusion": {"BKNG": 13.21, "OGN": 6.26, "AA": 0.54, "AAPL": 0.08},
        },
        "risk_checks": {
            "layer3_portfolio": {
                "pass": True,
                "report": {
                    "portfolio_beta": 0.908,
                    "cvar_95": -0.0404,
                    "holding_count": 61,
                    "gross_exposure": 0.80,
                    "cash_weight": 0.20,
                },
            },
        },
        "portfolio_metrics": {
            "turnover_vs_previous": 0.17,
            "holding_count_after_risk": 61,
            "gross_exposure_after_risk": 0.80,
            "cash_weight_after_risk": 0.20,
        },
    }
    report_week_2 = {
        **base_report,
        "week_number": 2,
        "live_outputs": {
            "signal_date": "2026-04-03",
            "target_weights_after_risk": {"BKNG": 0.010, "OGN": 0.012, "AAPL": 0.008},
        },
    }
    report_week_3 = {
        **base_report,
        "week_number": 3,
        "live_outputs": {
            "signal_date": "2026-04-10",
            "target_weights_after_risk": {"BKNG": 0.015, "OGN": 0.012, "AA": 0.010},
        },
    }
    (tmp_path / "week_02.json").write_text(json.dumps(report_week_2))
    (tmp_path / "week_03.json").write_text(json.dumps(report_week_3))
    return tmp_path


@pytest.fixture()
def client(report_dir: Path) -> TestClient:
    app = FastAPI()
    app.include_router(portfolio_router.router)
    with (
        patch.object(portfolio_router, "GREYSCALE_REPORT_DIR", report_dir),
        patch.object(portfolio_router, "_READER", None),
        patch.object(portfolio_router, "_READER_DIR", None),
        TestClient(app) as client,
    ):
        yield client


def test_get_current_portfolio(client: TestClient) -> None:
    response = client.get("/api/portfolio/current")
    assert response.status_code == 200

    payload = response.json()
    assert payload["signal_date"] == "2026-04-10"
    assert payload["week_number"] == 3
    assert payload["holding_count"] == 61
    assert payload["gross_exposure"] == pytest.approx(0.80)
    assert payload["risk_pass"] is True
    assert [holding["ticker"] for holding in payload["holdings"]] == ["BKNG", "OGN", "AA"]
    assert payload["holdings"][0]["score"] == pytest.approx(13.21)


def test_get_portfolio_summary(client: TestClient) -> None:
    response = client.get("/api/portfolio/summary")
    assert response.status_code == 200

    payload = response.json()
    assert payload["signal_date"] == "2026-04-10"
    assert payload["portfolio_beta"] == pytest.approx(0.908)
    assert payload["cvar_95"] == pytest.approx(-0.0404)
    assert "holdings" not in payload


def test_get_budget_allocation(client: TestClient) -> None:
    response = client.get("/api/portfolio/budget", params={"total_budget": 100000})
    assert response.status_code == 200

    payload = response.json()
    assert payload["total_budget"] == pytest.approx(100000.0)
    allocations = {item["ticker"]: item for item in payload["allocations"]}
    assert allocations["BKNG"]["dollar_amount"] == pytest.approx(1500.0)
    assert allocations["AA"]["dollar_amount"] == pytest.approx(1000.0)


def test_get_rebalance_orders(client: TestClient) -> None:
    response = client.get("/api/portfolio/rebalance")
    assert response.status_code == 200

    payload = response.json()
    assert payload["signal_date"] == "2026-04-10"
    actions = {order["ticker"]: order for order in payload["orders"]}
    assert actions["AA"]["action"] == "buy"
    assert actions["AA"]["weight_delta"] == pytest.approx(0.01)
    assert actions["AAPL"]["action"] == "sell"
    assert actions["OGN"]["action"] == "hold"


def test_compute_daily_performance_open_to_open() -> None:
    asyncio.run(_run_open_to_open())


async def _run_open_to_open() -> None:
    """单 tranche: 5d horizon, AAPL 50% + MSFT 50%, 1 名 dropped (NEW), 校验
    OPEN-to-OPEN 加权累计 + 权重重新归一 + SPY excess.
    """
    signal_date = date(2026, 4, 24)
    entry_date = date(2026, 4, 27)
    audit_rows = [
        (signal_date, "AAPL", Decimal("0.50")),
        (signal_date, "MSFT", Decimal("0.50")),
        (signal_date, "NEW", Decimal("0.10")),  # T+1 open 缺失, 应剔除
    ]
    # 价格: entry day + 后续 2 天
    price_rows: list[tuple[str, date, Decimal]] = [
        ("AAPL", entry_date, Decimal("100.0000")),
        ("AAPL", date(2026, 4, 28), Decimal("102.0000")),
        ("AAPL", date(2026, 4, 29), Decimal("101.0000")),
        ("MSFT", entry_date, Decimal("200.0000")),
        ("MSFT", date(2026, 4, 28), Decimal("204.0000")),
        ("MSFT", date(2026, 4, 29), Decimal("210.0000")),
        ("SPY", entry_date, Decimal("500.0000")),
        ("SPY", date(2026, 4, 28), Decimal("505.0000")),
        ("SPY", date(2026, 4, 29), Decimal("507.5000")),
    ]
    as_of = datetime(2026, 4, 30, tzinfo=timezone.utc)

    db = AsyncMock()
    with (
        patch.object(dp_service, "_latest_bundle_version", AsyncMock(return_value="test_bundle")),
        patch.object(dp_service, "_load_paper_portfolio_rows", AsyncMock(return_value=audit_rows)),
        patch.object(dp_service, "_load_open_prices", AsyncMock(return_value=price_rows)),
    ):
        resp = await dp_service.compute_daily_portfolio_performance(
            db, horizon="5d", bundle_version="test_bundle", as_of=as_of
        )

    assert resp.horizon == "5d"
    assert resp.bundle_version == "test_bundle"
    assert resp.weeks_count == 1
    assert len(resp.tranches) == 1
    tranche = resp.tranches[0]
    assert tranche.signal_date == "2026-04-24"
    assert tranche.entry_date == "2026-04-27"
    assert tranche.tickers_used == ["AAPL", "MSFT"]
    assert tranche.tickers_dropped == ["NEW"]
    # 3 个交易日: 4/27 (entry, 0), 4/28, 4/29
    assert len(tranche.series) == 3
    # entry day: cumulative = 0
    assert tranche.series[0].date == "2026-04-27"
    assert tranche.series[0].cumulative_portfolio == pytest.approx(0.0)
    assert tranche.series[0].cumulative_spy == pytest.approx(0.0)
    # day 2 (4/28): NEW(0.1) 剔除后 sum_orig=1.1, sum_remain=1.0, scale=1.1,
    # AAPL 权重 = 0.55, MSFT 权重 = 0.55; AAPL +2%, MSFT +2%
    # portfolio = 0.55*0.02 + 0.55*0.02 = 0.022; SPY = 5/500 = 0.01
    assert tranche.series[1].date == "2026-04-28"
    assert tranche.series[1].cumulative_portfolio == pytest.approx(0.022)
    assert tranche.series[1].cumulative_spy == pytest.approx(0.01)
    assert tranche.series[1].cumulative_excess == pytest.approx(0.012)
    # day 3 (4/29): AAPL +1%, MSFT +5%, portfolio = 0.55*0.01 + 0.55*0.05 = 0.033; SPY = 7.5/500 = 0.015
    assert tranche.series[2].cumulative_portfolio == pytest.approx(0.033)
    assert tranche.series[2].cumulative_spy == pytest.approx(0.015)


def test_compute_daily_performance_empty_bundle() -> None:
    asyncio.run(_run_empty_bundle())


async def _run_empty_bundle() -> None:
    """空 bundle → 返回 valid 空响应, 不抛错."""
    db = AsyncMock()
    with patch.object(dp_service, "_latest_bundle_version", AsyncMock(return_value=None)):
        resp = await dp_service.compute_daily_portfolio_performance(db, horizon="60d")
    assert resp.horizon == "60d"
    assert resp.weeks_count == 0
    assert resp.tranches == []
