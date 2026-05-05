from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import greyscale as greyscale_router


def _client_for_performance_file(path: object) -> TestClient:
    app = FastAPI()
    app.include_router(greyscale_router.router)
    patcher = patch.object(greyscale_router, "PERFORMANCE_FILE", path)
    patcher.start()
    client = TestClient(app)
    client._performance_patcher = patcher  # type: ignore[attr-defined]
    return client


def _close_client(client: TestClient) -> None:
    client._performance_patcher.stop()  # type: ignore[attr-defined]
    client.close()


def test_greyscale_performance_returns_payload(tmp_path: Path) -> None:
    performance_file = tmp_path / "greyscale_performance.json"
    performance_file.write_text(
        json.dumps(
            {
                "as_of_utc": "2026-05-05T00:00:00Z",
                "today": "2026-05-05",
                "benchmark": "SPY",
                "horizons_supported": [5],
                "per_week": [
                    {
                        "week_number": 1,
                        "signal_date": "2026-05-01",
                        "horizons": {
                            "5": {
                                "status": "pending",
                                "tickers_used": 0,
                                "tickers_missing": 0,
                            },
                        },
                    },
                ],
                "cumulative": {
                    "5": {
                        "return": 0.0123,
                        "weeks_realized": 1,
                        "weekly_curve": [],
                    },
                },
            },
        ),
    )
    client = _client_for_performance_file(performance_file)
    try:
        response = client.get("/api/greyscale/performance")
    finally:
        _close_client(client)

    assert response.status_code == 200
    payload = response.json()
    assert payload["today"] == "2026-05-05"
    assert payload["cumulative"]["5"]["return"] == 0.0123


def test_greyscale_performance_missing_file_is_controlled_404(tmp_path: Path) -> None:
    client = _client_for_performance_file(tmp_path / "missing.json")
    try:
        response = client.get("/api/greyscale/performance")
    finally:
        _close_client(client)

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "performance_file_missing"


class _PermissionDeniedPath:
    def read_text(self) -> str:
        raise PermissionError("permission denied")


def test_greyscale_performance_permission_error_is_retryable_503() -> None:
    client = _client_for_performance_file(_PermissionDeniedPath())
    try:
        response = client.get("/api/greyscale/performance")
    finally:
        _close_client(client)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "performance_file_unreadable"
    assert detail["retryable"] is True


def test_greyscale_performance_corrupt_json_is_retryable_503(tmp_path: Path) -> None:
    performance_file = tmp_path / "greyscale_performance.json"
    performance_file.write_text("{bad json")
    client = _client_for_performance_file(performance_file)
    try:
        response = client.get("/api/greyscale/performance")
    finally:
        _close_client(client)

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "performance_file_corrupt"
    assert detail["retryable"] is True
