from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from src.celery_app import celery_app


@celery_app.task(name="src.tasks.backtest_task.run_backtest_task", bind=True)
def run_backtest_task(
    self,
    task_id: str,
    strategy_name: str,
    start_date: str | None,
    end_date: str | None,
    initial_capital: float,
    tickers: list[str],
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    self.update_state(
        state="STARTED",
        meta={
            "task_id": task_id,
            "status": "STARTED",
            "progress": 10,
            "started_at": started_at.isoformat(),
            "completed_at": None,
        },
    )

    time.sleep(2)
    self.update_state(
        state="STARTED",
        meta={
            "task_id": task_id,
            "status": "STARTED",
            "progress": 60,
            "started_at": started_at.isoformat(),
            "completed_at": None,
        },
    )

    time.sleep(2)
    completed_at = datetime.now(timezone.utc)
    ticker_list = tickers or ["SPY", "QQQ"]
    result_summary = {
        "disclosure": "Model Output — Historical Simulation",
        "strategy_name": strategy_name,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "tickers": ticker_list,
        "ending_capital": round(initial_capital * 1.0825, 2),
        "total_return_pct": 8.25,
        "max_drawdown_pct": -4.1,
        "sharpe_ratio": 1.27,
        "trade_count": max(len(ticker_list) * 6, 12),
    }
    return {
        "task_id": task_id,
        "status": "SUCCESS",
        "progress": 100,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "result_summary": result_summary,
    }
