from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import predictions as predictions_router


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    base_report = {
        "generated_at_utc": "2026-04-12T00:00:00+00:00",
        "live_outputs": {
            "top_10_fusion_scores": [
                {"ticker": "BKNG", "score": 13.21},
                {"ticker": "OGN", "score": 6.26},
            ],
            "target_weights_after_risk": {"BKNG": 0.015, "OGN": 0.012, "AA": 0.010},
            "target_weights_raw": {"BKNG": 0.020, "OGN": 0.015},
        },
        "score_vectors": {
            "fusion": {"BKNG": 13.21, "OGN": 6.26, "AA": 0.54, "AAPL": 0.08},
            "ridge": {"BKNG": 10.0, "OGN": 5.0, "AA": 0.3, "AAPL": 0.1},
            "xgboost": {"BKNG": 14.0, "OGN": 7.0, "AA": 0.6, "AAPL": 0.05},
            "lightgbm": {"BKNG": 12.0, "OGN": 6.0, "AA": 0.5, "AAPL": 0.09},
        },
        "fusion": {
            "live_weights": {"ridge": 0.25, "xgboost": 0.40, "lightgbm": 0.35},
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
        "db_state": {
            "stock_universe_size": 503,
        },
    }
    report_week_2 = {
        **base_report,
        "week_number": 2,
        "live_outputs": {
            **base_report["live_outputs"],
            "signal_date": "2026-04-03",
        },
    }
    report_week_3 = {
        **base_report,
        "week_number": 3,
        "live_outputs": {
            **base_report["live_outputs"],
            "signal_date": "2026-04-10",
        },
    }
    (tmp_path / "week_02.json").write_text(json.dumps(report_week_2))
    (tmp_path / "week_03.json").write_text(json.dumps(report_week_3))
    return tmp_path


@pytest.fixture()
def client(report_dir: Path) -> TestClient:
    app = FastAPI()
    app.include_router(predictions_router.router)
    with (
        patch.object(predictions_router, "GREYSCALE_REPORT_DIR", report_dir),
        patch.object(predictions_router, "_READER", None),
        patch.object(predictions_router, "_READER_DIR", None),
        TestClient(app) as client,
    ):
        yield client


def test_get_latest_predictions(client: TestClient) -> None:
    response = client.get("/api/predictions/latest")
    assert response.status_code == 200

    payload = response.json()
    assert payload["signal_date"] == "2026-04-10"
    assert payload["week_number"] == 3
    assert payload["universe_size"] == 503
    assert len(payload["predictions"]) == 4
    assert payload["predictions"][0] == {
        "ticker": "BKNG",
        "score": pytest.approx(13.21),
        "rank": 1,
        "percentile": 100.0,
    }


def test_get_latest_predictions_with_top_n(client: TestClient) -> None:
    response = client.get("/api/predictions/latest", params={"top_n": 2})
    assert response.status_code == 200
    payload = response.json()
    assert [item["ticker"] for item in payload["predictions"]] == ["BKNG", "OGN"]


def test_get_ticker_prediction(client: TestClient) -> None:
    response = client.get("/api/predictions/bkng")
    assert response.status_code == 200

    payload = response.json()
    assert payload["ticker"] == "BKNG"
    assert payload["fusion_score"] == pytest.approx(13.21)
    assert payload["rank"] == 1
    assert payload["total"] == 4
    assert payload["weight"] == pytest.approx(0.015)
    assert payload["model_scores"]["ridge"] == pytest.approx(10.0)


def test_get_ticker_prediction_not_found(client: TestClient) -> None:
    response = client.get("/api/predictions/zzzz")
    assert response.status_code == 404
    assert "ZZZZ" in response.json()["detail"]


def test_get_signal_history(client: TestClient) -> None:
    response = client.get("/api/predictions/BKNG/history")
    assert response.status_code == 200

    payload = response.json()
    assert payload["ticker"] == "BKNG"
    assert [point["week"] for point in payload["history"]] == [2, 3]
    assert all(point["rank"] == 1 for point in payload["history"])
