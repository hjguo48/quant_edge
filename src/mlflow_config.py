from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any
from urllib.parse import urlparse

import mlflow

from src.config import get_settings

# MLflow is Phase 1's only non-negotiable MLOps component.


@dataclass(frozen=True)
class LoggedRun:
    tracking_uri: str
    experiment_id: str
    run_id: str


LOCAL_MLFLOW_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _ensure_local_no_proxy(tracking_uri: str) -> None:
    parsed = urlparse(tracking_uri)
    if parsed.hostname not in LOCAL_MLFLOW_HOSTS:
        return

    local_hosts = sorted(LOCAL_MLFLOW_HOSTS)
    for env_var in ("NO_PROXY", "no_proxy"):
        existing = {
            entry.strip()
            for entry in os.environ.get(env_var, "").split(",")
            if entry.strip()
        }
        os.environ[env_var] = ",".join(sorted(existing.union(local_hosts)))


def setup_mlflow() -> str:
    settings = get_settings()
    tracking_uri = settings.MLFLOW_TRACKING_URI
    _ensure_local_no_proxy(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    return tracking_uri


def ensure_experiment(experiment_name: str) -> str:
    setup_mlflow()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(experiment_name)


def log_experiment(
    experiment_name: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> LoggedRun:
    tracking_uri = setup_mlflow()
    experiment_id = ensure_experiment(experiment_name)

    with mlflow.start_run(experiment_id=experiment_id) as run:
        if params:
            mlflow.log_params(dict(params))
        if metrics:
            mlflow.log_metrics(dict(metrics))

    return LoggedRun(
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        run_id=run.info.run_id,
    )
