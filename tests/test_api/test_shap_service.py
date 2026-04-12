from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import predictions as predictions_router
from src.api.services.shap_service import get_shap_for_ticker


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    report = {
        "week_number": 3,
        "live_outputs": {
            "signal_date": "2026-04-10",
            "target_weights_after_risk": {"AAPL": 0.02},
        },
        "score_vectors": {
            "fusion": {"AAPL": 0.08},
            "ridge": {"AAPL": 0.03},
            "xgboost": {"AAPL": 0.10},
            "lightgbm": {"AAPL": 0.07},
        },
        "shap_values": {
            "xgboost": {
                "AAPL": {
                    "base_value": 0.0,
                    "features": {
                        "vol_60d": 0.05,
                        "atr_14": -0.02,
                        "low_52w_ratio": 0.03,
                    },
                },
            },
            "lightgbm": {
                "AAPL": {
                    "base_value": 0.0,
                    "features": {
                        "vol_60d": 0.04,
                        "atr_14": -0.01,
                        "low_52w_ratio": 0.02,
                    },
                },
            },
        },
        "fusion": {
            "live_weights": {"ridge": 0.25, "xgboost": 0.40, "lightgbm": 0.35},
        },
    }
    (tmp_path / "week_03.json").write_text(json.dumps(report))
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


def test_get_shap_for_ticker(report_dir: Path) -> None:
    result = get_shap_for_ticker("AAPL", report_dir=report_dir)
    assert result is not None
    assert result["ticker"] == "AAPL"
    assert result["signal_date"] == "2026-04-10"
    assert len(result["features"]) == 3
    assert result["features"][0] == {"feature": "vol_60d", "shap_value": pytest.approx(0.034)}
    assert result["features"][1]["feature"] == "low_52w_ratio"
    assert result["features"][2]["shap_value"] == pytest.approx(-0.0115)


def test_shap_not_found(report_dir: Path) -> None:
    result = get_shap_for_ticker("ZZZZ", report_dir=report_dir)
    assert result is None


def test_get_ticker_shap_endpoint(client: TestClient) -> None:
    response = client.get("/api/predictions/AAPL/shap", params={"top_n": 2})
    assert response.status_code == 200

    payload = response.json()
    assert payload["ticker"] == "AAPL"
    assert payload["signal_date"] == "2026-04-10"
    assert len(payload["features"]) == 2
    assert payload["features"][0]["feature"] == "vol_60d"
