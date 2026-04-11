from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException

from src.api.schemas.backtest import (
    BacktestRequest,
    BacktestResponse,
    BacktestResultResponse,
    BacktestStatusResponse,
)
from src.celery_app import celery_app
from src.tasks.backtest_task import run_backtest_task

router = APIRouter(prefix="/api/backtest", tags=["Backtest"])


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest) -> BacktestResponse:
    task_id = str(uuid4())
    run_backtest_task.apply_async(
        kwargs={
            "task_id": task_id,
            "strategy_name": request.strategy_name,
            "start_date": request.start_date.isoformat() if request.start_date else None,
            "end_date": request.end_date.isoformat() if request.end_date else None,
            "initial_capital": request.initial_capital,
            "tickers": request.tickers,
        },
        task_id=task_id,
    )
    return BacktestResponse(
        message="backtest submitted",
        status="submitted",
        task_id=task_id,
        strategy_name=request.strategy_name,
        run_id=task_id,
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _build_status_response(task_result: AsyncResult, task_id: str) -> BacktestStatusResponse:
    info = task_result.info if isinstance(task_result.info, dict) else {}
    if task_result.state == "SUCCESS" and isinstance(task_result.result, dict):
        info = task_result.result

    progress = info.get("progress")
    if isinstance(progress, (int, float)):
        progress_value: int | None = int(progress)
    else:
        progress_value = None

    started_at = _parse_datetime(info.get("started_at"))
    completed_at = _parse_datetime(info.get("completed_at"))

    return BacktestStatusResponse(
        task_id=task_id,
        status=task_result.state,
        progress=progress_value,
        started_at=started_at,
        completed_at=completed_at,
    )


@router.get("/{task_id}/status", response_model=BacktestStatusResponse)
async def get_backtest_status(task_id: str) -> BacktestStatusResponse:
    task_result = AsyncResult(task_id, app=celery_app)
    return _build_status_response(task_result, task_id)


@router.get("/{task_id}/result", response_model=BacktestResultResponse)
async def get_backtest_result(task_id: str) -> BacktestResultResponse:
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.state in {"PENDING", "RECEIVED", "STARTED", "RETRY"}:
        raise HTTPException(
            status_code=409,
            detail="Backtest task is not completed yet.",
        )
    if task_result.state == "FAILURE":
        return BacktestResultResponse(
            task_id=task_id,
            status=task_result.state,
            result={
                "task_id": task_id,
                "error": str(task_result.result),
            },
        )
    if not isinstance(task_result.result, dict):
        raise HTTPException(
            status_code=404,
            detail="Backtest result not found.",
        )
    return BacktestResultResponse(
        task_id=task_id,
        status=task_result.state,
        result=task_result.result,
    )
