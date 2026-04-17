from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.deps import get_db
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
    g3_gate = {
        "checks": {
            "bootstrap_ci_positive": {
                "annualized_excess_estimate": 0.12,
                "annualized_excess_ci_lower": 0.02,
                "annualized_excess_ci_upper": 0.22,
                "sharpe_estimate": 0.70,
                "sharpe_ci_lower": 0.10,
                "sharpe_ci_upper": 1.10,
                "n_observations": 278,
                "ci_level": 0.95,
            },
        },
    }
    (tmp_path / "g3_gate_phase_e_v2.json").write_text(json.dumps(g3_gate))
    quintiles = {
        "data_source": "walk_forward_quintile_bootstrap",
        "ci_level": 0.95,
        "quintiles": {
            "1": {
                "annualized_excess": {"estimate": 0.18, "ci_lower": 0.08, "ci_upper": 0.28},
                "sharpe": {"estimate": 0.90, "ci_lower": 0.20, "ci_upper": 1.40},
                "n_observations": 278,
            },
            "4": {
                "annualized_excess": {"estimate": 0.04, "ci_lower": -0.02, "ci_upper": 0.10},
                "sharpe": {"estimate": 0.20, "ci_lower": -0.30, "ci_upper": 0.70},
                "n_observations": 278,
            },
        },
    }
    (tmp_path / "quintile_expected_returns.json").write_text(json.dumps(quintiles))
    return tmp_path


@pytest.fixture()
def sector_map() -> dict[str, str | None]:
    return {
        "BKNG": "Consumer Discretionary",
        "OGN": "Health Care",
        "AA": "Materials",
    }


class FakeResult:
    def __init__(self, rows: list[tuple[str, str | None, str | None]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[str, str | None, str | None]]:
        return self._rows


class FakeAsyncSession:
    def __init__(self, sector_map: dict[str, str | None]) -> None:
        self._sector_map = sector_map
        self.statements: list[object] = []

    async def execute(self, statement: object) -> FakeResult:
        self.statements.append(statement)
        return FakeResult(
            [
                (ticker, sector, f"{ticker} Inc." if sector is not None else None)
                for ticker, sector in self._sector_map.items()
            ],
        )


@pytest.fixture()
def db_session(sector_map: dict[str, str | None]) -> FakeAsyncSession:
    return FakeAsyncSession(sector_map)


@pytest.fixture()
def client(report_dir: Path, db_session: FakeAsyncSession) -> TestClient:
    app = FastAPI()
    app.include_router(predictions_router.router)

    async def override_get_db() -> FakeAsyncSession:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with (
        patch.object(predictions_router, "GREYSCALE_REPORT_DIR", report_dir),
        patch.object(
            predictions_router,
            "_READER",
            predictions_router.GreyscaleReader(
                report_dir=report_dir,
                g3_gate_results_path=report_dir / "g3_gate_phase_e_v2.json",
                quintile_expected_returns_path=report_dir / "quintile_expected_returns.json",
            ),
        ),
        patch.object(predictions_router, "_READER_DIR", report_dir),
        TestClient(app) as client,
    ):
        yield client

    app.dependency_overrides.clear()


def test_get_latest_predictions(client: TestClient, db_session: FakeAsyncSession) -> None:
    response = client.get("/api/predictions/latest")
    assert response.status_code == 200

    payload = response.json()
    assert payload["signal_date"] == "2026-04-10"
    assert payload["week_number"] == 3
    assert payload["universe_size"] == 503
    assert len(payload["predictions"]) == 4
    assert payload["predictions"][0]["ticker"] == "BKNG"
    assert payload["predictions"][0]["score"] == pytest.approx(13.21)
    assert payload["predictions"][0]["rank"] == 1
    assert payload["predictions"][0]["percentile"] == 100.0
    assert payload["predictions"][0]["sector"] == "Consumer Discretionary"
    assert payload["predictions"][0]["company_name"] == "BKNG Inc."
    assert payload["predictions"][-1]["sector"] is None
    assert len(db_session.statements) == 1
    assert " IN " in str(db_session.statements[0]).upper()


def test_get_latest_predictions_with_top_n(client: TestClient) -> None:
    response = client.get("/api/predictions/latest", params={"top_n": 2})
    assert response.status_code == 200
    payload = response.json()
    assert [item["ticker"] for item in payload["predictions"]] == ["BKNG", "OGN"]
    assert [item["sector"] for item in payload["predictions"]] == [
        "Consumer Discretionary",
        "Health Care",
    ]


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
    assert payload["sector"] == "Consumer Discretionary"


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


def test_get_expected_returns_global_fallback(client: TestClient) -> None:
    response = client.get("/api/predictions/expected-returns")
    assert response.status_code == 200
    payload = response.json()
    assert payload["data_source"] == "g3_gate_bootstrap"
    assert payload["annualized_excess"]["estimate"] == pytest.approx(0.12)


def test_get_ticker_expected_return_uses_quintile_payload(client: TestClient) -> None:
    top_response = client.get("/api/predictions/BKNG/expected-return")
    low_response = client.get("/api/predictions/AAPL/expected-return")

    assert top_response.status_code == 200
    assert low_response.status_code == 200

    top_payload = top_response.json()
    low_payload = low_response.json()
    assert top_payload["quintile"] == 1
    assert low_payload["quintile"] == 4
    assert top_payload["data_source"] == "walk_forward_quintile_bootstrap"
    assert top_payload["annualized_excess"]["estimate"] == pytest.approx(0.18)
    assert low_payload["annualized_excess"]["estimate"] == pytest.approx(0.04)
