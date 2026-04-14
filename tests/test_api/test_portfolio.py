from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import portfolio as portfolio_router


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
