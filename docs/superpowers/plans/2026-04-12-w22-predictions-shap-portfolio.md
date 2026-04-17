# W22-23: AI 预测中心 + SHAP 解释层 + 组合构建器 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace stub prediction and portfolio API endpoints with real data from greyscale weekly reports, add SHAP feature attribution, and connect the frontend to live data instead of hardcoded mock arrays.

**Architecture:** The greyscale pipeline already produces weekly JSON reports containing full score vectors (all ~500 tickers), target weights, risk check results, and model metadata. The API layer reads these JSON files (no new model inference at request time). SHAP values are pre-computed during the greyscale pipeline run and stored in the report. The frontend replaces hardcoded arrays with `useQuery` fetches to the new API endpoints.

**Tech Stack:** FastAPI (async), Pydantic v2 schemas, SHAP (TreeExplainer for XGBoost/LightGBM, linear for Ridge), React + TypeScript + Recharts + TanStack Query.

---

## File Structure

### Backend — New Files
- `src/api/services/greyscale_reader.py` — Reads and caches weekly greyscale JSON reports from `data/reports/greyscale/`
- `src/api/services/shap_service.py` — Loads model artifacts from fusion bundle, computes SHAP values for a given ticker against the latest feature matrix
- `src/api/schemas/predictions.py` — Expanded: full prediction list, per-ticker detail, SHAP attribution, signal history
- `src/api/schemas/portfolio.py` — Expanded: holdings with sector/weight/score, sector allocation, rebalance orders, budget calculator
- `tests/test_api/test_predictions.py` — Tests for prediction endpoints
- `tests/test_api/test_portfolio.py` — Tests for portfolio endpoints
- `tests/test_api/__init__.py` — Package init

### Backend — Modified Files
- `src/api/routers/predictions.py` — Replace stub with 5 real endpoints
- `src/api/routers/portfolio.py` — Replace stub with 5 real endpoints
- `src/api/routers/__init__.py` — No change needed (routers already registered)
- `scripts/run_greyscale_live.py` — Add SHAP values to the report output (+~30 lines in report dict)

### Frontend — Modified Files
- `frontend/src/pages/Signals.tsx` — Replace `SIGNALS_DATA` hardcoded array with API fetch
- `frontend/src/pages/SignalDetail.tsx` — Add SHAP waterfall tab, signal history chart, uncertainty display
- `frontend/src/pages/Portfolio.tsx` — Replace hardcoded `holdings` / `sectorAlloc` / `perfData` with API fetch
- `frontend/src/pages/Index.tsx` — No structural change needed (routing already exists)

### Frontend — New Files
- `frontend/src/components/ShapWaterfall.tsx` — SHAP waterfall bar chart component (Recharts)
- `frontend/src/components/SignalHistory.tsx` — 60-day signal score line chart component
- `frontend/src/hooks/useApi.ts` — Shared fetch helper + API base URL config

---

## Task 1: Greyscale Report Reader Service (Backend)

**Files:**
- Create: `src/api/services/__init__.py`
- Create: `src/api/services/greyscale_reader.py`
- Create: `tests/test_api/__init__.py`
- Create: `tests/test_api/test_greyscale_reader.py`

This service reads and caches the weekly greyscale JSON report files. It is the single data source for both the predictions and portfolio endpoints.

- [ ] **Step 1: Write failing tests for GreyscaleReader**

```python
# tests/test_api/__init__.py
# (empty)

# tests/test_api/test_greyscale_reader.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.services.greyscale_reader import GreyscaleReader


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    report = {
        "week_number": 3,
        "generated_at_utc": "2026-04-12T00:00:00+00:00",
        "live_outputs": {
            "signal_date": "2026-04-10",
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
            "weight_source": "seed_weights",
            "regime": {"vix": 21.04, "regime": "mid", "scalar": 0.8},
        },
        "risk_checks": {
            "layer3_portfolio": {
                "pass": True,
                "checks": {
                    "overall_pass": True,
                    "checks": {
                        "beta": {"pass": True, "value": 0.908},
                        "cvar": {"pass": True, "value": -0.0404},
                    },
                },
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
    (tmp_path / "week_03.json").write_text(json.dumps(report))

    report_w2 = {**report, "week_number": 2, "live_outputs": {**report["live_outputs"], "signal_date": "2026-04-03"}}
    (tmp_path / "week_02.json").write_text(json.dumps(report_w2))
    return tmp_path


def test_latest_report(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    report = reader.get_latest_report()
    assert report is not None
    assert report["week_number"] == 3


def test_all_scores_sorted(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    scores = reader.get_all_fusion_scores()
    assert len(scores) == 4
    assert scores[0]["ticker"] == "BKNG"
    assert scores[0]["score"] > scores[1]["score"]


def test_ticker_detail(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    detail = reader.get_ticker_detail("BKNG")
    assert detail is not None
    assert detail["fusion_score"] == pytest.approx(13.21)
    assert "ridge" in detail["model_scores"]


def test_ticker_not_found(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    detail = reader.get_ticker_detail("ZZZZ")
    assert detail is None


def test_portfolio_holdings(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    holdings = reader.get_portfolio_holdings()
    assert len(holdings) == 3
    assert holdings[0]["ticker"] == "BKNG"


def test_signal_history(report_dir: Path) -> None:
    reader = GreyscaleReader(report_dir=report_dir)
    history = reader.get_signal_history("BKNG")
    assert len(history) == 2
    assert history[0]["signal_date"] < history[1]["signal_date"]


def test_no_reports(tmp_path: Path) -> None:
    reader = GreyscaleReader(report_dir=tmp_path)
    assert reader.get_latest_report() is None
    assert reader.get_all_fusion_scores() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_greyscale_reader.py -v`
Expected: ModuleNotFoundError — `src.api.services.greyscale_reader` does not exist

- [ ] **Step 3: Implement GreyscaleReader**

```python
# src/api/services/__init__.py
# (empty)

# src/api/services/greyscale_reader.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

_WEEK_PATTERN = re.compile(r"week_(\d+)\.json$")


class GreyscaleReader:
    """Reads greyscale weekly report JSONs from a directory."""

    def __init__(self, report_dir: Path | str) -> None:
        self._report_dir = Path(report_dir)
        self._cache: dict[int, dict[str, Any]] = {}

    def _load_all(self) -> dict[int, dict[str, Any]]:
        if self._cache:
            return self._cache
        reports: dict[int, dict[str, Any]] = {}
        if not self._report_dir.is_dir():
            return reports
        for path in sorted(self._report_dir.glob("week_*.json")):
            match = _WEEK_PATTERN.search(path.name)
            if not match:
                continue
            week_num = int(match.group(1))
            try:
                data = json.loads(path.read_text())
                reports[week_num] = data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("failed to load {}: {}", path, exc)
        self._cache = reports
        return reports

    def invalidate_cache(self) -> None:
        self._cache.clear()

    def get_latest_report(self) -> dict[str, Any] | None:
        reports = self._load_all()
        if not reports:
            return None
        latest_week = max(reports)
        return reports[latest_week]

    def get_report(self, week: int) -> dict[str, Any] | None:
        return self._load_all().get(week)

    def get_all_fusion_scores(self) -> list[dict[str, Any]]:
        report = self.get_latest_report()
        if report is None:
            return []
        scores = report.get("score_vectors", {}).get("fusion", {})
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            {"ticker": ticker, "score": score, "rank": i + 1, "percentile": round((1 - i / max(len(ranked), 1)) * 100, 1)}
            for i, (ticker, score) in enumerate(ranked)
        ]

    def get_ticker_detail(self, ticker: str) -> dict[str, Any] | None:
        report = self.get_latest_report()
        if report is None:
            return None
        fusion_scores = report.get("score_vectors", {}).get("fusion", {})
        if ticker not in fusion_scores:
            return None

        all_scores = sorted(fusion_scores.items(), key=lambda item: item[1], reverse=True)
        rank = next((i + 1 for i, (t, _) in enumerate(all_scores) if t == ticker), None)
        total = len(all_scores)

        model_scores = {}
        for model_name in ("ridge", "xgboost", "lightgbm"):
            model_vec = report.get("score_vectors", {}).get(model_name, {})
            if ticker in model_vec:
                model_scores[model_name] = model_vec[ticker]

        weights = report.get("live_outputs", {}).get("target_weights_after_risk", {})

        return {
            "ticker": ticker,
            "fusion_score": fusion_scores[ticker],
            "rank": rank,
            "total": total,
            "percentile": round((1 - (rank - 1) / max(total, 1)) * 100, 1) if rank else None,
            "model_scores": model_scores,
            "weight": weights.get(ticker),
            "signal_date": report.get("live_outputs", {}).get("signal_date"),
        }

    def get_portfolio_holdings(self) -> list[dict[str, Any]]:
        report = self.get_latest_report()
        if report is None:
            return []
        weights = report.get("live_outputs", {}).get("target_weights_after_risk", {})
        fusion_scores = report.get("score_vectors", {}).get("fusion", {})
        sorted_holdings = sorted(weights.items(), key=lambda item: item[1], reverse=True)
        return [
            {
                "ticker": ticker,
                "weight": weight,
                "score": fusion_scores.get(ticker),
            }
            for ticker, weight in sorted_holdings
        ]

    def get_portfolio_summary(self) -> dict[str, Any] | None:
        report = self.get_latest_report()
        if report is None:
            return None
        metrics = report.get("portfolio_metrics", {})
        risk = report.get("risk_checks", {}).get("layer3_portfolio", {})
        risk_report = risk.get("report", {})
        fusion = report.get("fusion", {})
        regime = fusion.get("regime", {})
        return {
            "signal_date": report.get("live_outputs", {}).get("signal_date"),
            "week_number": report.get("week_number"),
            "holding_count": metrics.get("holding_count_after_risk"),
            "gross_exposure": metrics.get("gross_exposure_after_risk"),
            "cash_weight": metrics.get("cash_weight_after_risk"),
            "turnover": metrics.get("turnover_vs_previous"),
            "portfolio_beta": risk_report.get("portfolio_beta"),
            "cvar_95": risk_report.get("cvar_95"),
            "risk_pass": risk.get("pass"),
            "vix": regime.get("vix"),
            "regime": regime.get("regime"),
        }

    def get_signal_history(self, ticker: str) -> list[dict[str, Any]]:
        reports = self._load_all()
        history: list[dict[str, Any]] = []
        for week_num in sorted(reports):
            report = reports[week_num]
            fusion_scores = report.get("score_vectors", {}).get("fusion", {})
            if ticker not in fusion_scores:
                continue
            signal_date = report.get("live_outputs", {}).get("signal_date")
            all_scores = sorted(fusion_scores.values(), reverse=True)
            rank = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)
            ticker_rank = next((i + 1 for i, (t, _) in enumerate(rank) if t == ticker), None)
            history.append({
                "week": week_num,
                "signal_date": signal_date,
                "score": fusion_scores[ticker],
                "rank": ticker_rank,
                "total": len(fusion_scores),
            })
        return history

    def get_sector_allocation(self) -> list[dict[str, Any]]:
        """Requires sector_map from DB. Returns empty for now — filled by router."""
        return []
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_greyscale_reader.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/api/services/__init__.py src/api/services/greyscale_reader.py tests/test_api/__init__.py tests/test_api/test_greyscale_reader.py
git commit -m "feat: add GreyscaleReader service for reading weekly report JSONs"
```

---

## Task 2: Prediction API Endpoints (Backend)

**Files:**
- Modify: `src/api/schemas/predictions.py`
- Modify: `src/api/routers/predictions.py`
- Create: `tests/test_api/test_predictions.py`

Replace the stub `/api/predictions/latest` with real endpoints that read from greyscale reports.

- [ ] **Step 1: Write failing tests for prediction endpoints**

```python
# tests/test_api/test_predictions.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    report = {
        "week_number": 3,
        "generated_at_utc": "2026-04-12T00:00:00+00:00",
        "live_outputs": {
            "signal_date": "2026-04-10",
            "top_10_fusion_scores": [
                {"ticker": "BKNG", "score": 13.21},
            ],
            "target_weights_after_risk": {"BKNG": 0.015, "AA": 0.010},
            "target_weights_raw": {"BKNG": 0.020},
        },
        "score_vectors": {
            "fusion": {"BKNG": 13.21, "AA": 0.54, "AAPL": 0.08},
            "ridge": {"BKNG": 10.0, "AA": 0.3, "AAPL": 0.1},
            "xgboost": {"BKNG": 14.0, "AA": 0.6, "AAPL": 0.05},
            "lightgbm": {"BKNG": 12.0, "AA": 0.5, "AAPL": 0.09},
        },
        "fusion": {
            "live_weights": {"ridge": 0.25, "xgboost": 0.40, "lightgbm": 0.35},
            "weight_source": "seed_weights",
            "regime": {"vix": 21.04, "regime": "mid", "scalar": 0.8},
        },
        "risk_checks": {
            "layer3_portfolio": {"pass": True, "checks": {"overall_pass": True}, "report": {}},
        },
        "portfolio_metrics": {},
        "db_state": {"stock_universe_size": 503},
    }
    (tmp_path / "week_03.json").write_text(json.dumps(report))
    return tmp_path


@pytest.fixture()
def client(report_dir: Path) -> TestClient:
    with patch("src.api.routers.predictions.GREYSCALE_REPORT_DIR", report_dir):
        from src.api.main import app
        return TestClient(app)


def test_get_latest_predictions(client: TestClient) -> None:
    resp = client.get("/api/predictions/latest")
    assert resp.status_code == 200
    data = resp.json()
    assert data["signal_date"] == "2026-04-10"
    assert len(data["predictions"]) == 3
    assert data["predictions"][0]["ticker"] == "BKNG"


def test_get_predictions_top_n(client: TestClient) -> None:
    resp = client.get("/api/predictions/latest?top_n=2")
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_get_ticker_prediction(client: TestClient) -> None:
    resp = client.get("/api/predictions/BKNG")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "BKNG"
    assert data["fusion_score"] == pytest.approx(13.21)
    assert "ridge" in data["model_scores"]


def test_get_ticker_not_found(client: TestClient) -> None:
    resp = client.get("/api/predictions/ZZZZ")
    assert resp.status_code == 404


def test_get_signal_history(client: TestClient) -> None:
    resp = client.get("/api/predictions/BKNG/history")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["history"]) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_predictions.py -v`
Expected: FAIL — endpoints don't return expected structure

- [ ] **Step 3: Expand prediction schemas**

```python
# src/api/schemas/predictions.py
from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    ticker: str
    score: float
    rank: int
    percentile: float


class PredictionResponse(BaseModel):
    signal_date: str | None = None
    week_number: int | None = None
    model_name: str = "ic_weighted_fusion_60d"
    universe_size: int | None = None
    predictions: list[PredictionItem] = Field(default_factory=list)


class ModelScore(BaseModel):
    model: str
    score: float


class TickerPredictionResponse(BaseModel):
    ticker: str
    fusion_score: float
    rank: int | None = None
    total: int | None = None
    percentile: float | None = None
    model_scores: dict[str, float] = Field(default_factory=dict)
    weight: float | None = None
    signal_date: str | None = None


class SignalHistoryPoint(BaseModel):
    week: int
    signal_date: str | None = None
    score: float
    rank: int | None = None
    total: int | None = None


class SignalHistoryResponse(BaseModel):
    ticker: str
    history: list[SignalHistoryPoint] = Field(default_factory=list)
```

- [ ] **Step 4: Implement prediction router**

```python
# src/api/routers/predictions.py
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.api.schemas.predictions import (
    PredictionItem,
    PredictionResponse,
    SignalHistoryPoint,
    SignalHistoryResponse,
    TickerPredictionResponse,
)
from src.api.services.greyscale_reader import GreyscaleReader

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

GREYSCALE_REPORT_DIR = Path("data/reports/greyscale")


def _get_reader() -> GreyscaleReader:
    return GreyscaleReader(report_dir=GREYSCALE_REPORT_DIR)


@router.get("/latest", response_model=PredictionResponse)
async def get_latest_predictions(
    top_n: int | None = Query(default=None, ge=1, le=600, description="Limit to top N predictions"),
) -> PredictionResponse:
    reader = _get_reader()
    all_scores = reader.get_all_fusion_scores()
    report = reader.get_latest_report()

    if top_n is not None:
        all_scores = all_scores[:top_n]

    return PredictionResponse(
        signal_date=report.get("live_outputs", {}).get("signal_date") if report else None,
        week_number=report.get("week_number") if report else None,
        universe_size=report.get("db_state", {}).get("stock_universe_size") if report else None,
        predictions=[PredictionItem(**s) for s in all_scores],
    )


@router.get("/{ticker}", response_model=TickerPredictionResponse)
async def get_ticker_prediction(ticker: str) -> TickerPredictionResponse:
    reader = _get_reader()
    detail = reader.get_ticker_detail(ticker.upper())
    if detail is None:
        raise HTTPException(status_code=404, detail=f"No prediction found for ticker '{ticker.upper()}'")
    return TickerPredictionResponse(**detail)


@router.get("/{ticker}/history", response_model=SignalHistoryResponse)
async def get_signal_history(ticker: str) -> SignalHistoryResponse:
    reader = _get_reader()
    history = reader.get_signal_history(ticker.upper())
    return SignalHistoryResponse(
        ticker=ticker.upper(),
        history=[SignalHistoryPoint(**h) for h in history],
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_predictions.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/schemas/predictions.py src/api/routers/predictions.py tests/test_api/test_predictions.py
git commit -m "feat: implement real prediction API endpoints from greyscale reports"
```

---

## Task 3: Portfolio API Endpoints (Backend)

**Files:**
- Modify: `src/api/schemas/portfolio.py`
- Modify: `src/api/routers/portfolio.py`
- Create: `tests/test_api/test_portfolio.py`

Replace stub `/api/portfolio/current` with real endpoints reading from greyscale reports.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api/test_portfolio.py
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    report = {
        "week_number": 3,
        "generated_at_utc": "2026-04-12T00:00:00+00:00",
        "live_outputs": {
            "signal_date": "2026-04-10",
            "top_10_fusion_scores": [],
            "target_weights_after_risk": {"BKNG": 0.015, "OGN": 0.012, "AA": 0.010},
            "target_weights_raw": {"BKNG": 0.020, "OGN": 0.015, "AA": 0.012},
        },
        "score_vectors": {
            "fusion": {"BKNG": 13.21, "OGN": 6.26, "AA": 0.54},
            "ridge": {}, "xgboost": {}, "lightgbm": {},
        },
        "fusion": {
            "live_weights": {"ridge": 0.25, "xgboost": 0.40, "lightgbm": 0.35},
            "regime": {"vix": 21.04, "regime": "mid", "scalar": 0.8},
        },
        "risk_checks": {
            "layer3_portfolio": {
                "pass": True,
                "checks": {"overall_pass": True},
                "report": {"portfolio_beta": 0.908, "cvar_95": -0.0404, "holding_count": 3, "gross_exposure": 0.80, "cash_weight": 0.20},
            },
        },
        "portfolio_metrics": {
            "turnover_vs_previous": 0.17,
            "holding_count_after_risk": 3,
            "gross_exposure_after_risk": 0.80,
            "cash_weight_after_risk": 0.20,
        },
        "db_state": {"stock_universe_size": 503},
    }
    (tmp_path / "week_03.json").write_text(json.dumps(report))

    report_w2 = {**report, "week_number": 2}
    report_w2["live_outputs"] = {**report["live_outputs"], "target_weights_after_risk": {"BKNG": 0.010, "OGN": 0.010, "AAPL": 0.015}}
    (tmp_path / "week_02.json").write_text(json.dumps(report_w2))
    return tmp_path


@pytest.fixture()
def client(report_dir: Path) -> TestClient:
    with patch("src.api.routers.portfolio.GREYSCALE_REPORT_DIR", report_dir):
        from src.api.main import app
        return TestClient(app)


def test_get_current_portfolio(client: TestClient) -> None:
    resp = client.get("/api/portfolio/current")
    assert resp.status_code == 200
    data = resp.json()
    assert data["holding_count"] == 3
    assert data["portfolio_beta"] == pytest.approx(0.908)
    assert len(data["holdings"]) == 3


def test_portfolio_summary(client: TestClient) -> None:
    resp = client.get("/api/portfolio/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["risk_pass"] is True
    assert data["regime"] == "mid"


def test_budget_calculator(client: TestClient) -> None:
    resp = client.get("/api/portfolio/budget?total_budget=100000")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["allocations"]) == 3
    assert data["allocations"][0]["dollar_amount"] > 0


def test_rebalance_orders(client: TestClient) -> None:
    resp = client.get("/api/portfolio/rebalance")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["orders"]) > 0
    actions = {o["action"] for o in data["orders"]}
    assert "buy" in actions or "sell" in actions or "hold" in actions
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_portfolio.py -v`
Expected: FAIL

- [ ] **Step 3: Expand portfolio schemas**

```python
# src/api/schemas/portfolio.py
from __future__ import annotations

from pydantic import BaseModel, Field


class PortfolioHolding(BaseModel):
    ticker: str
    weight: float
    score: float | None = None


class PortfolioResponse(BaseModel):
    signal_date: str | None = None
    week_number: int | None = None
    holding_count: int | None = None
    gross_exposure: float | None = None
    cash_weight: float | None = None
    portfolio_beta: float | None = None
    cvar_95: float | None = None
    turnover: float | None = None
    risk_pass: bool | None = None
    holdings: list[PortfolioHolding] = Field(default_factory=list)


class PortfolioSummaryResponse(BaseModel):
    signal_date: str | None = None
    week_number: int | None = None
    holding_count: int | None = None
    gross_exposure: float | None = None
    cash_weight: float | None = None
    turnover: float | None = None
    portfolio_beta: float | None = None
    cvar_95: float | None = None
    risk_pass: bool | None = None
    vix: float | None = None
    regime: str | None = None


class BudgetAllocation(BaseModel):
    ticker: str
    weight: float
    dollar_amount: float
    shares_estimate: float | None = None


class BudgetResponse(BaseModel):
    total_budget: float
    allocations: list[BudgetAllocation] = Field(default_factory=list)


class RebalanceOrder(BaseModel):
    ticker: str
    action: str  # "buy", "sell", "hold"
    weight_prev: float
    weight_new: float
    weight_delta: float


class RebalanceResponse(BaseModel):
    signal_date: str | None = None
    orders: list[RebalanceOrder] = Field(default_factory=list)
```

- [ ] **Step 4: Implement portfolio router**

```python
# src/api/routers/portfolio.py
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query

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

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])

GREYSCALE_REPORT_DIR = Path("data/reports/greyscale")


def _get_reader() -> GreyscaleReader:
    return GreyscaleReader(report_dir=GREYSCALE_REPORT_DIR)


@router.get("/current", response_model=PortfolioResponse)
async def get_current_portfolio() -> PortfolioResponse:
    reader = _get_reader()
    holdings_raw = reader.get_portfolio_holdings()
    summary = reader.get_portfolio_summary()
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
        holdings=[PortfolioHolding(**h) for h in holdings_raw],
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
            ticker=h["ticker"],
            weight=h["weight"],
            dollar_amount=round(h["weight"] * total_budget, 2),
        )
        for h in holdings
    ]
    return BudgetResponse(total_budget=total_budget, allocations=allocations)


@router.get("/rebalance", response_model=RebalanceResponse)
async def get_rebalance_orders() -> RebalanceResponse:
    reader = _get_reader()
    reports = reader._load_all()
    weeks = sorted(reports.keys())
    if len(weeks) < 2:
        latest = reader.get_latest_report()
        signal_date = latest.get("live_outputs", {}).get("signal_date") if latest else None
        holdings = reader.get_portfolio_holdings()
        return RebalanceResponse(
            signal_date=signal_date,
            orders=[
                RebalanceOrder(
                    ticker=h["ticker"], action="buy",
                    weight_prev=0.0, weight_new=h["weight"],
                    weight_delta=h["weight"],
                )
                for h in holdings
            ],
        )

    prev_report = reports[weeks[-2]]
    curr_report = reports[weeks[-1]]
    prev_weights = prev_report.get("live_outputs", {}).get("target_weights_after_risk", {})
    curr_weights = curr_report.get("live_outputs", {}).get("target_weights_after_risk", {})
    signal_date = curr_report.get("live_outputs", {}).get("signal_date")

    all_tickers = sorted(set(prev_weights) | set(curr_weights))
    orders = []
    for ticker in all_tickers:
        w_prev = prev_weights.get(ticker, 0.0)
        w_new = curr_weights.get(ticker, 0.0)
        delta = w_new - w_prev
        if abs(delta) < 1e-6:
            action = "hold"
        elif delta > 0:
            action = "buy"
        else:
            action = "sell"
        orders.append(RebalanceOrder(
            ticker=ticker, action=action,
            weight_prev=round(w_prev, 6),
            weight_new=round(w_new, 6),
            weight_delta=round(delta, 6),
        ))
    orders.sort(key=lambda o: abs(o.weight_delta), reverse=True)
    return RebalanceResponse(signal_date=signal_date, orders=orders)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_portfolio.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/api/schemas/portfolio.py src/api/routers/portfolio.py tests/test_api/test_portfolio.py
git commit -m "feat: implement real portfolio API endpoints from greyscale reports"
```

---

## Task 4: SHAP Attribution Service + Greyscale Pipeline Integration (Backend)

**Files:**
- Create: `src/api/services/shap_service.py`
- Modify: `scripts/run_greyscale_live.py` — Add SHAP computation to report output
- Create: `tests/test_api/test_shap_service.py`

SHAP values are pre-computed during the greyscale pipeline and stored in the report JSON. The API service reads them from the report (no live inference at request time). This task has two parts: (a) computing SHAP in the pipeline, (b) serving them via API.

- [ ] **Step 1: Write failing test for SHAP service (reads from report)**

```python
# tests/test_api/test_shap_service.py
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.api.services.shap_service import get_shap_for_ticker


@pytest.fixture()
def report_dir(tmp_path: Path) -> Path:
    report = {
        "week_number": 3,
        "live_outputs": {"signal_date": "2026-04-10"},
        "score_vectors": {"fusion": {"AAPL": 0.08}},
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


def test_get_shap_for_ticker(report_dir: Path) -> None:
    result = get_shap_for_ticker("AAPL", report_dir=report_dir)
    assert result is not None
    assert result["ticker"] == "AAPL"
    assert len(result["features"]) == 3
    assert result["features"][0]["feature"] == "vol_60d"
    assert result["features"][0]["shap_value"] > 0


def test_shap_not_found(report_dir: Path) -> None:
    result = get_shap_for_ticker("ZZZZ", report_dir=report_dir)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_shap_service.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement SHAP service (reads from pre-computed report)**

```python
# src/api/services/shap_service.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.api.services.greyscale_reader import GreyscaleReader


def get_shap_for_ticker(
    ticker: str,
    *,
    report_dir: Path | str,
    top_n: int = 20,
) -> dict[str, Any] | None:
    reader = GreyscaleReader(report_dir=report_dir)
    report = reader.get_latest_report()
    if report is None:
        return None

    shap_values = report.get("shap_values", {})
    if not shap_values:
        return None

    live_weights = report.get("fusion", {}).get("live_weights", {})
    ticker = ticker.upper()

    weighted_features: dict[str, float] = {}
    for model_name, ticker_shaps in shap_values.items():
        if ticker not in ticker_shaps:
            continue
        model_weight = live_weights.get(model_name, 0.0)
        for feature, value in ticker_shaps[ticker].get("features", {}).items():
            weighted_features[feature] = weighted_features.get(feature, 0.0) + value * model_weight

    if not weighted_features:
        return None

    sorted_features = sorted(weighted_features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    return {
        "ticker": ticker,
        "signal_date": report.get("live_outputs", {}).get("signal_date"),
        "features": [
            {"feature": feat, "shap_value": round(val, 6)}
            for feat, val in sorted_features
        ],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/test_shap_service.py -v`
Expected: All 2 tests PASS

- [ ] **Step 5: Add SHAP endpoint to predictions router**

Add to `src/api/routers/predictions.py`:

```python
# Add import at top:
from src.api.services.shap_service import get_shap_for_ticker

# Add endpoint:
@router.get("/{ticker}/shap")
async def get_ticker_shap(
    ticker: str,
    top_n: int = Query(default=15, ge=1, le=50, description="Number of top features to return"),
) -> dict:
    result = get_shap_for_ticker(ticker.upper(), report_dir=GREYSCALE_REPORT_DIR, top_n=top_n)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No SHAP data for ticker '{ticker.upper()}'")
    return result
```

- [ ] **Step 6: Add SHAP computation to greyscale pipeline**

In `scripts/run_greyscale_live.py`, add a function to compute SHAP values for the top holdings and include them in the report dict. This should be added after the `score_live_cross_section()` call and before the report assembly.

Add import at top:
```python
import shap
```

Add function:
```python
def compute_shap_for_top_tickers(
    models: dict[str, Any],
    feature_matrix: pd.DataFrame,
    top_tickers: list[str],
    max_tickers: int = 100,
) -> dict[str, dict[str, Any]]:
    """Compute SHAP values for tree-based models on top tickers."""
    tickers_to_explain = top_tickers[:max_tickers]
    result: dict[str, dict[str, Any]] = {}

    for model_name, model in models.items():
        if model_name == "ridge":
            continue  # Ridge uses coefficients directly, skip SHAP
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            continue

        ticker_shaps: dict[str, Any] = {}
        for ticker in tickers_to_explain:
            if ticker not in feature_matrix.index.get_level_values("ticker"):
                continue
            row = feature_matrix.xs(ticker, level="ticker")
            if row.empty:
                continue
            sv = explainer.shap_values(row)
            if hasattr(sv, "values"):
                values = sv.values[0] if len(sv.values.shape) > 1 else sv.values
                base = float(sv.base_values[0]) if hasattr(sv.base_values, '__len__') else float(sv.base_values)
            else:
                values = sv[0] if len(sv.shape) > 1 else sv
                base = float(explainer.expected_value)
            ticker_shaps[ticker] = {
                "base_value": base,
                "features": {
                    feat: round(float(val), 6)
                    for feat, val in zip(feature_matrix.columns, values)
                },
            }
        result[model_name] = ticker_shaps
    return result
```

In the `main()` function, after `score_live_cross_section()` returns and before assembling `report = {...}`, add:

```python
    # Compute SHAP for top tickers
    top_tickers_for_shap = list(fused_scores_by_ticker.head(100).index.astype(str))
    try:
        shap_data = compute_shap_for_top_tickers(
            models=models,
            feature_matrix=current_feature_matrix,
            top_tickers=top_tickers_for_shap,
        )
    except Exception as exc:
        logger.warning("SHAP computation failed, skipping: {}", exc)
        shap_data = {}
```

Then add `"shap_values": json_safe(shap_data)` to the report dict.

- [ ] **Step 7: Run full test suite**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/test_api/ -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add src/api/services/shap_service.py src/api/routers/predictions.py scripts/run_greyscale_live.py tests/test_api/test_shap_service.py
git commit -m "feat: add SHAP attribution service + greyscale pipeline SHAP computation"
```

---

## Task 5: Frontend API Hook + Signals Page Live Data (Frontend)

**Files:**
- Create: `frontend/src/hooks/useApi.ts`
- Modify: `frontend/src/pages/Signals.tsx`

Replace the hardcoded `SIGNALS_DATA` array with live API data from `/api/predictions/latest`.

- [ ] **Step 1: Create shared API fetch hook**

```typescript
// frontend/src/hooks/useApi.ts
const API_BASE = "";  // Vite proxy handles /api → FastAPI

export async function fetchApi<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body?.detail || `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}
```

- [ ] **Step 2: Update Signals.tsx to fetch from API**

Replace the hardcoded `SIGNALS_DATA` array and add a `useQuery` call:

```typescript
// At the top, add imports:
import { useQuery } from "@tanstack/react-query";
import { fetchApi } from "../hooks/useApi";

// Add interface:
interface PredictionItem {
  ticker: string;
  score: float;
  rank: number;
  percentile: number;
}

interface PredictionsResponse {
  signal_date: string | null;
  universe_size: number | null;
  predictions: PredictionItem[];
}

// Remove SIGNALS_DATA constant entirely.

// Inside the Signals component, before the existing state hooks, add:
const { data: predictionsData, isLoading, isError } = useQuery<PredictionsResponse>({
  queryKey: ["predictions"],
  queryFn: () => fetchApi("/api/predictions/latest"),
  refetchInterval: 60_000,
});

const predictions = predictionsData?.predictions ?? [];
```

Update the filter/sort logic to work with the new `predictions` array shape. Replace `SIGNALS_DATA` references with `predictions`. The `direction` field becomes `score > 0 ? "long" : "short"`. The `confidence` field maps from `percentile`. The `alpha` field maps from `score`.

Add loading and error states:
- While loading: show skeleton rows (reuse the existing card structure with `animate-pulse`)
- On error: show "Unable to load predictions" message

- [ ] **Step 3: Configure Vite proxy**

Check if `frontend/vite.config.ts` already has a proxy for `/api`. If not, add:

```typescript
server: {
  host: "0.0.0.0",
  proxy: {
    "/api": {
      target: "http://localhost:8000",
      changeOrigin: true,
    },
  },
},
```

- [ ] **Step 4: Test in browser**

1. Ensure FastAPI is running: `cd /home/jiahao/quant_edge && uvicorn src.api.main:app --reload --host 0.0.0.0`
2. Ensure Vite dev server is running: `cd /home/jiahao/quant_edge/frontend && npm run dev`
3. Open `http://127.0.0.1:15173/signals` in Chrome (with proxy bypass)
4. Verify: signals load from API (real tickers from greyscale data), filter/sort works, loading state shows skeleton, clicking a signal navigates to detail

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/useApi.ts frontend/src/pages/Signals.tsx frontend/vite.config.ts
git commit -m "feat: connect Signals page to live prediction API"
```

---

## Task 6: SHAP Waterfall Component + Signal Detail Integration (Frontend)

**Files:**
- Create: `frontend/src/components/ShapWaterfall.tsx`
- Create: `frontend/src/components/SignalHistory.tsx`
- Modify: `frontend/src/pages/SignalDetail.tsx`

Add the SHAP waterfall chart and signal history chart to the SignalDetail page tabs.

- [ ] **Step 1: Create ShapWaterfall component**

```typescript
// frontend/src/components/ShapWaterfall.tsx
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from "recharts";

interface ShapFeature {
  feature: string;
  shap_value: number;
}

interface ShapWaterfallProps {
  features: ShapFeature[];
  height?: number;
}

function formatFeatureLabel(name: string): string {
  return name
    .split("_")
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(" ");
}

const ShapWaterfall = ({ features, height = 300 }: ShapWaterfallProps) => {
  const data = features
    .slice(0, 15)
    .map((f) => ({
      feature: formatFeatureLabel(f.feature),
      rawFeature: f.feature,
      value: f.shap_value,
      positive: f.shap_value >= 0,
    }))
    .sort((a, b) => b.value - a.value);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ left: 8, right: 16 }}>
        <XAxis
          type="number"
          tick={{ fill: "#607B96", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fill: "#607B96", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
          width={120}
        />
        <Tooltip
          cursor={{ fill: "rgba(255,255,255,0.03)" }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div className="bg-popover border border-border rounded-lg px-2.5 py-1.5 shadow-custom">
                <p className="text-xs text-muted-foreground">{d.feature}</p>
                <p className={`text-xs font-bold ${d.positive ? "text-bull" : "text-bear"}`}>
                  {d.value >= 0 ? "+" : ""}{d.value.toFixed(4)}
                </p>
              </div>
            );
          }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {data.map((entry) => (
            <Cell key={entry.rawFeature} fill={entry.positive ? "#00C805" : "#FF5252"} fillOpacity={0.85} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
};

export default ShapWaterfall;
```

- [ ] **Step 2: Create SignalHistory component**

```typescript
// frontend/src/components/SignalHistory.tsx
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from "recharts";

interface HistoryPoint {
  week: number;
  signal_date: string | null;
  score: number;
  rank: number | null;
}

interface SignalHistoryProps {
  history: HistoryPoint[];
  height?: number;
}

const SignalHistory = ({ history, height = 200 }: SignalHistoryProps) => {
  const data = history.map((h) => ({
    label: h.signal_date ? h.signal_date.slice(5) : `W${h.week}`,
    score: h.score,
    rank: h.rank,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 5, right: 16, bottom: 0, left: 0 }}>
        <XAxis
          dataKey="label"
          tick={{ fill: "#607B96", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          tick={{ fill: "#607B96", fontSize: 10 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const d = payload[0].payload;
            return (
              <div className="bg-popover border border-border rounded-lg px-2.5 py-1.5 shadow-custom">
                <p className="text-xs text-muted-foreground">{d.label}</p>
                <p className="text-xs font-bold text-primary">Score: {d.score.toFixed(2)}</p>
                {d.rank && <p className="text-xs text-muted-foreground">Rank: #{d.rank}</p>}
              </div>
            );
          }}
        />
        <Line
          type="monotone"
          dataKey="score"
          stroke="hsl(var(--primary))"
          strokeWidth={2}
          dot={{ r: 3, fill: "hsl(var(--primary))" }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default SignalHistory;
```

- [ ] **Step 3: Integrate SHAP and Signal History into SignalDetail.tsx**

Add to the existing SignalDetail page:

1. Add new `useQuery` hooks for `/api/predictions/{ticker}`, `/api/predictions/{ticker}/shap`, and `/api/predictions/{ticker}/history`
2. Replace the "factors" tab placeholder with the SHAP waterfall component showing feature attributions
3. Add a signal history section (either as part of overview or in a new tab)
4. Show the prediction score, rank, and percentile in the header area
5. Show model breakdown (ridge/xgboost/lightgbm individual scores) in a small table

The existing `tabs` array `["overview", "factors", "backtest", "risk"]` — the `factors` tab should render `ShapWaterfall` with data from the SHAP endpoint, and the model score breakdown table below it.

- [ ] **Step 4: Test in browser**

1. Navigate to `http://127.0.0.1:15173/signals/BKNG`
2. Verify: header shows fusion score, rank, percentile
3. Click "factors" tab: SHAP waterfall chart loads (if SHAP data exists in report; graceful fallback if not)
4. Signal history chart shows past weeks' scores

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/ShapWaterfall.tsx frontend/src/components/SignalHistory.tsx frontend/src/pages/SignalDetail.tsx
git commit -m "feat: add SHAP waterfall and signal history to SignalDetail page"
```

---

## Task 7: Portfolio Page Live Data (Frontend)

**Files:**
- Modify: `frontend/src/pages/Portfolio.tsx`

Replace all hardcoded mock data with live API calls to `/api/portfolio/*`.

- [ ] **Step 1: Replace hardcoded data with API queries**

Add `useQuery` calls:
1. `GET /api/portfolio/current` → holdings table + header stats
2. `GET /api/portfolio/summary` → summary cards (beta, CVaR, regime, etc.)
3. `GET /api/portfolio/budget?total_budget=100000` → budget tab (new)
4. `GET /api/portfolio/rebalance` → rebalance orders tab

Remove the hardcoded `holdings`, `sectorAlloc`, and `perfData` arrays.

The stats cards at the top should show real data:
- Total Holdings → `holding_count`
- Gross Exposure → `gross_exposure` as percentage
- Portfolio Beta → `portfolio_beta`
- CVaR (95%) → `cvar_95` as percentage

The holdings table columns become: Ticker, Weight (bar + percentage), Score, Signal (long if score > 0).

The "trades" sub-tab should show rebalance orders from `/api/portfolio/rebalance`: ticker, action (buy/sell/hold), weight change.

- [ ] **Step 2: Add budget calculator section**

Add a simple input field for total budget. When the user enters an amount (default $100,000), call `GET /api/portfolio/budget?total_budget=X` and display: ticker, weight, dollar amount per position.

- [ ] **Step 3: Sector allocation from portfolio data**

For now, the sector allocation pie chart can be hidden or show a "Coming soon" placeholder, since the greyscale report doesn't include per-ticker sector data in the weights output. The portfolio performance chart (vs benchmark) should also show a placeholder since we don't have historical portfolio tracking yet.

- [ ] **Step 4: Test in browser**

1. Navigate to `http://127.0.0.1:15173/portfolio`
2. Verify: stats cards show real data, holdings table shows real tickers/weights, loading states work, budget calculator works

- [ ] **Step 5: Commit**

```bash
git add frontend/src/pages/Portfolio.tsx
git commit -m "feat: connect Portfolio page to live API data"
```

---

## Task 8: End-to-End Integration Test + Polish

**Files:**
- Run all backend tests
- Run TypeScript type check
- Manual browser testing

- [ ] **Step 1: Run full backend test suite**

Run: `cd /home/jiahao/quant_edge && python -m pytest tests/ -v --tb=short`
Expected: All existing tests still pass, all new API tests pass

- [ ] **Step 2: Run frontend type check**

Run: `cd /home/jiahao/quant_edge/frontend && npx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 3: Manual browser testing checklist**

- [ ] Dashboard loads without errors
- [ ] Signals page shows real predictions from API (not mock data)
- [ ] Clicking a signal navigates to detail page with real data
- [ ] SignalDetail "factors" tab shows SHAP waterfall (or graceful "no SHAP data" message)
- [ ] Portfolio page shows real holdings, stats, budget calculator
- [ ] Rebalance orders display correctly
- [ ] All pages show SEC-compliant disclaimers ("Model output only", "Not investment advice")
- [ ] No console errors in browser DevTools

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: W22 complete — predictions + SHAP + portfolio endpoints connected to frontend"
```
