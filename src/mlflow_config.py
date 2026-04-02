from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
from mlflow.entities import Metric, Param, RunTag
from mlflow.tracking import MlflowClient

from src.config import DEFAULT_MLFLOW_ARTIFACT_ROOT, DEFAULT_MLFLOW_TRACKING_URI, get_settings

# MLflow is Phase 1's only non-negotiable MLOps component.

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILE_STORE_URI = (REPO_ROOT / "mlruns").resolve().as_uri()
DEFAULT_ARTIFACT_LOCATION = DEFAULT_MLFLOW_ARTIFACT_ROOT.resolve().as_uri()
MIGRATION_SOURCE_RUN_TAG = "migration.source_run_id"
MIGRATION_SOURCE_EXPERIMENT_TAG = "migration.source_experiment_id"
MIGRATION_SOURCE_TRACKING_URI_TAG = "migration.source_tracking_uri"


@dataclass(frozen=True)
class LoggedRun:
    tracking_uri: str
    experiment_id: str
    run_id: str


@dataclass(frozen=True)
class MigrationReport:
    source_tracking_uri: str
    target_tracking_uri: str
    experiments_seen: int
    experiments_created: int
    runs_seen: int
    runs_migrated: int
    runs_skipped: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_tracking_uri": self.source_tracking_uri,
            "target_tracking_uri": self.target_tracking_uri,
            "experiments_seen": self.experiments_seen,
            "experiments_created": self.experiments_created,
            "runs_seen": self.runs_seen,
            "runs_migrated": self.runs_migrated,
            "runs_skipped": self.runs_skipped,
        }


LOCAL_MLFLOW_HOSTS = {"localhost", "127.0.0.1", "::1"}


def get_mlflow_tracking_uri() -> str:
    env_value = os.environ.get("MLFLOW_TRACKING_URI")
    if env_value:
        return normalize_tracking_uri(env_value)

    settings = get_settings()
    return normalize_tracking_uri(settings.MLFLOW_TRACKING_URI or DEFAULT_MLFLOW_TRACKING_URI)


def get_mlflow_artifact_root() -> str:
    settings = get_settings()
    configured_root = settings.MLFLOW_ARTIFACT_ROOT or str(DEFAULT_MLFLOW_ARTIFACT_ROOT)
    return Path(configured_root).expanduser().resolve().as_uri()


def normalize_tracking_uri(tracking_uri: str) -> str:
    if not tracking_uri:
        return DEFAULT_MLFLOW_TRACKING_URI

    if tracking_uri.startswith("sqlite:///") and not tracking_uri.startswith("sqlite:////"):
        candidate = tracking_uri.removeprefix("sqlite:///")
        if candidate.startswith(("/", "home/", "Users/")):
            absolute_path = Path("/") / candidate.lstrip("/")
            return f"sqlite:///{absolute_path.as_posix()}"
    return tracking_uri


def _ensure_local_no_proxy(tracking_uri: str) -> None:
    parsed = urlparse(tracking_uri)
    if parsed.scheme in {"sqlite", "file"}:
        return
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


def setup_mlflow(*, tracking_uri: str | None = None) -> str:
    resolved = normalize_tracking_uri(tracking_uri or get_mlflow_tracking_uri())
    _ensure_local_no_proxy(resolved)
    mlflow.set_tracking_uri(resolved)
    return resolved


def default_artifact_location_for_tracking_uri(tracking_uri: str) -> str | None:
    resolved_tracking_uri = normalize_tracking_uri(tracking_uri)
    if resolved_tracking_uri.startswith("sqlite:"):
        return get_mlflow_artifact_root()
    if resolved_tracking_uri == DEFAULT_FILE_STORE_URI:
        return get_mlflow_artifact_root()
    if resolved_tracking_uri.startswith("file://"):
        parsed = urlparse(resolved_tracking_uri)
        if parsed.path:
            return (Path(parsed.path) / "artifacts").resolve().as_uri()
    return None


def ensure_experiment(
    experiment_name: str,
    *,
    tracking_uri: str | None = None,
    artifact_location: str | None = None,
) -> str:
    resolved_tracking_uri = setup_mlflow(tracking_uri=tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    resolved_artifact_location = artifact_location or default_artifact_location_for_tracking_uri(
        resolved_tracking_uri,
    )
    if resolved_artifact_location:
        return mlflow.create_experiment(
            experiment_name,
            artifact_location=resolved_artifact_location,
        )
    return mlflow.create_experiment(experiment_name)


def log_experiment(
    experiment_name: str,
    params: Mapping[str, Any] | None = None,
    metrics: Mapping[str, float] | None = None,
) -> LoggedRun:
    tracking_uri = setup_mlflow()
    experiment_id = ensure_experiment(experiment_name, tracking_uri=tracking_uri)

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


def migrate_file_store_to_sqlite(
    *,
    source_tracking_uri: str | None = None,
    target_tracking_uri: str | None = None,
    max_experiments: int = 500,
    max_runs_per_experiment: int = 10_000,
) -> MigrationReport:
    source_uri = normalize_tracking_uri(source_tracking_uri or DEFAULT_FILE_STORE_URI)
    target_uri = normalize_tracking_uri(target_tracking_uri or get_mlflow_tracking_uri())
    if source_uri == target_uri:
        raise ValueError("source_tracking_uri and target_tracking_uri must be different.")

    source_client = MlflowClient(tracking_uri=source_uri)
    target_client = MlflowClient(tracking_uri=target_uri)

    source_experiments = source_client.search_experiments(max_results=max_experiments)
    target_experiments = {experiment.name: experiment for experiment in target_client.search_experiments(max_results=max_experiments)}

    experiments_created = 0
    runs_seen = 0
    runs_migrated = 0
    runs_skipped = 0

    for source_experiment in source_experiments:
        target_experiment = target_experiments.get(source_experiment.name)
        if target_experiment is None:
            experiment_tags = dict(source_experiment.tags or {})
            experiment_tags.setdefault("migration.source_experiment_name", source_experiment.name)
            experiment_tags.setdefault("migration.source_experiment_id", source_experiment.experiment_id)
            target_experiment_id = target_client.create_experiment(
                source_experiment.name,
                artifact_location=source_experiment.artifact_location,
                tags=experiment_tags or None,
            )
            target_experiment = target_client.get_experiment(target_experiment_id)
            target_experiments[source_experiment.name] = target_experiment
            experiments_created += 1

        source_runs = source_client.search_runs(
            [source_experiment.experiment_id],
            max_results=max_runs_per_experiment,
            order_by=["attributes.start_time ASC"],
        )
        for source_run in source_runs:
            runs_seen += 1
            existing = target_client.search_runs(
                [target_experiment.experiment_id],
                filter_string=f"tags.{MIGRATION_SOURCE_RUN_TAG} = '{source_run.info.run_id}'",
                max_results=1,
            )
            if existing:
                runs_skipped += 1
                continue

            target_tags = {
                key: str(value)
                for key, value in source_run.data.tags.items()
            }
            target_tags[MIGRATION_SOURCE_RUN_TAG] = source_run.info.run_id
            target_tags[MIGRATION_SOURCE_EXPERIMENT_TAG] = source_experiment.experiment_id
            target_tags[MIGRATION_SOURCE_TRACKING_URI_TAG] = source_uri

            created = target_client.create_run(
                experiment_id=target_experiment.experiment_id,
                start_time=source_run.info.start_time,
                tags=target_tags,
                run_name=source_run.data.tags.get("mlflow.runName"),
            )

            params = [
                Param(key=str(key), value=str(value))
                for key, value in source_run.data.params.items()
            ]
            tags = [
                RunTag(key=str(key), value=str(value))
                for key, value in target_tags.items()
            ]
            metrics: list[Metric] = []
            for metric_key in source_run.data.metrics:
                history = source_client.get_metric_history(source_run.info.run_id, metric_key)
                if history:
                    metrics.extend(history)
                else:
                    metrics.append(
                        Metric(
                            key=metric_key,
                            value=float(source_run.data.metrics[metric_key]),
                            timestamp=source_run.info.end_time or source_run.info.start_time or 0,
                            step=0,
                        ),
                    )

            target_client.log_batch(
                created.info.run_id,
                metrics=metrics,
                params=params,
                tags=tags,
            )
            target_client.set_terminated(
                created.info.run_id,
                status=source_run.info.status,
                end_time=source_run.info.end_time,
            )
            runs_migrated += 1

    return MigrationReport(
        source_tracking_uri=source_uri,
        target_tracking_uri=target_uri,
        experiments_seen=len(source_experiments),
        experiments_created=experiments_created,
        runs_seen=runs_seen,
        runs_migrated=runs_migrated,
        runs_skipped=runs_skipped,
    )
