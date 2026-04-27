from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.schemas.greyscale import (
    GreyscalePerformanceResponse,
    GreyscaleHorizonCumulative,
    GreyscaleHorizonWeek,
    GreyscaleWeek,
)

router = APIRouter(prefix="/api/greyscale", tags=["Greyscale"])

GREYSCALE_REPORT_DIR = Path("data/reports/greyscale")
PERFORMANCE_FILE = GREYSCALE_REPORT_DIR / "greyscale_performance.json"


@router.get("/performance", response_model=GreyscalePerformanceResponse)
async def get_greyscale_performance() -> GreyscalePerformanceResponse:
    """W13.2 paper P&L tracker — weekly + cumulative paper return vs SPY.

    Returns 404 if compute_realized_returns.py has not been run yet.
    """
    if not PERFORMANCE_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "greyscale_performance.json not found. "
                "Run scripts/compute_realized_returns.py first."
            ),
        )
    payload = json.loads(PERFORMANCE_FILE.read_text())

    return GreyscalePerformanceResponse(
        as_of_utc=payload.get("as_of_utc"),
        today=payload.get("today"),
        benchmark=payload.get("benchmark", "SPY"),
        horizons_supported=payload.get("horizons_supported", []),
        per_week=[
            GreyscaleWeek(
                week_number=int(w.get("week_number", 0)),
                signal_date=w.get("signal_date"),
                horizons={
                    h_key: GreyscaleHorizonWeek(**h_data)
                    for h_key, h_data in (w.get("horizons") or {}).items()
                },
            )
            for w in payload.get("per_week", [])
        ],
        cumulative={
            h_key: GreyscaleHorizonCumulative(**h_data)
            for h_key, h_data in (payload.get("cumulative") or {}).items()
        },
    )
