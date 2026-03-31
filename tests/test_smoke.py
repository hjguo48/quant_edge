from __future__ import annotations

import time
import uuid

from mlflow.tracking import MlflowClient
from sqlalchemy import text

from src.config import settings
from src.mlflow_config import log_experiment, setup_mlflow


def wait_for_mlflow(timeout_seconds: int = 300) -> MlflowClient:
    tracking_uri = setup_mlflow()
    client = MlflowClient(tracking_uri=tracking_uri)
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None

    while time.time() < deadline:
        try:
            client.search_experiments(max_results=1)
            return client
        except Exception as exc:  # pragma: no cover - exercised only when services lag.
            last_error = exc
            time.sleep(5)

    raise RuntimeError("MLflow tracking server did not become ready in time.") from last_error


def test_database_connectivity(db_connection) -> None:
    assert db_connection.execute(text("SELECT 1")).scalar_one() == 1


def test_config_loads() -> None:
    assert settings.POSTGRES_USER == "quantedge"
    assert settings.POSTGRES_PORT == 5433
    assert settings.MLFLOW_TRACKING_URI == "http://127.0.0.1:5001"
    assert settings.database_url.endswith("/quantedge")


def test_mlflow_can_log_experiment() -> None:
    client = wait_for_mlflow()
    experiment_name = f"smoke-{uuid.uuid4().hex[:8]}"
    logged_run = log_experiment(
        experiment_name,
        params={"stage": "smoke"},
        metrics={"score": 1.0},
    )

    experiment = client.get_experiment(logged_run.experiment_id)
    run = client.get_run(logged_run.run_id)

    assert experiment is not None
    assert experiment.name == experiment_name
    assert run.info.experiment_id == logged_run.experiment_id
    assert run.data.params["stage"] == "smoke"
    assert run.data.metrics["score"] == 1.0
