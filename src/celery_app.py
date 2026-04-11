from __future__ import annotations

from celery import Celery

from src.config import settings

broker_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
result_backend = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1"

celery_app = Celery("quantedge", broker=broker_url, backend=result_backend)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="America/New_York",
    task_track_started=True,
    result_expires=86400,
    imports=("src.tasks.backtest_task",),
)

__all__ = ["celery_app"]
