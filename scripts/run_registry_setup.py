from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import pickle
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any

from loguru import logger
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from scripts.run_single_window_validation import configure_logging, current_git_branch, json_safe
from src.config import DEFAULT_MLFLOW_TRACKING_URI
from src.mlflow_config import (
    DEFAULT_FILE_STORE_URI,
    MIGRATION_SOURCE_RUN_TAG,
    migrate_file_store_to_sqlite,
    normalize_tracking_uri,
    setup_mlflow,
)
from src.models.champion_challenger import ChampionChallengerRunner
from src.models.experiment import ExperimentTracker
from src.models.registry import (
    CHAMPION_ALIAS,
    MODEL_STAGE_TAG,
    ModelMetadata,
    ModelRegistry,
    ModelStage,
    default_validation_checks,
)

EXPECTED_BRANCH = "feature/week11-model-registry"
MODEL_NAME = "ridge_60d"
REGISTRY_IMPORT_EXPERIMENT = "registry_imports"
REGISTRY_IMPORT_ARTIFACT_ROOT = (REPO_ROOT / "mlartifacts" / REGISTRY_IMPORT_EXPERIMENT).resolve().as_uri()


class PickledBaseModelPyfunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model: Any):
        self.model = model

    def predict(self, context, model_input, params=None):
        if not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)
        prediction = self.model.predict(model_input)
        if isinstance(prediction, pd.Series):
            return pd.DataFrame({"score": prediction.to_numpy(dtype=float)})
        if isinstance(prediction, pd.DataFrame):
            if "score" in prediction.columns:
                return prediction[["score"]]
            if prediction.shape[1] == 1:
                return prediction.rename(columns={prediction.columns[0]: "score"})
            raise ValueError("Wrapped model returned a DataFrame without a unique score column.")
        return pd.DataFrame({"score": pd.Series(prediction, dtype=float).to_numpy(dtype=float)})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()

    branch = current_git_branch()
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}.")

    sqlite_tracking_uri = normalize_tracking_uri(args.sqlite_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI)
    migration_report = migrate_file_store_to_sqlite(
        source_tracking_uri=args.source_tracking_uri,
        target_tracking_uri=sqlite_tracking_uri,
    )
    logger.info(
        "migration complete source_runs={} migrated={} skipped={}",
        migration_report.runs_seen,
        migration_report.runs_migrated,
        migration_report.runs_skipped,
    )

    extended_report = load_json(REPO_ROOT / args.extended_walkforward_report_path)
    alpha_report = load_json(REPO_ROOT / args.phase1_alpha_report_path)
    source_window = find_source_window(extended_report)
    retained_features = list(extended_report["inputs"]["retained_features"])
    source_run_id = str(source_window["mlflow"]["run_id"])

    source_client = MlflowClient(tracking_uri=normalize_tracking_uri(args.source_tracking_uri))
    source_run = source_client.get_run(source_run_id)

    metadata = build_model_metadata(
        source_window=source_window,
        source_run=source_run,
        retained_features=retained_features,
        alpha_report=alpha_report,
        walkforward_report=extended_report,
    )
    registry_tags = build_registry_tags(
        source_run_id=source_run_id,
        metadata=metadata,
        alpha_report=alpha_report,
    )
    validation_checks = default_validation_checks()
    validation_results = evaluate_validation_checks(validation_checks, metadata)

    registry = ModelRegistry(tracking_uri=sqlite_tracking_uri)
    existing_version = find_existing_version(registry, model_name=MODEL_NAME, source_run_id=source_run_id)
    packaging_info: dict[str, Any] | None = None

    if existing_version is None:
        packaging_info = package_source_run_as_pyfunc(
            source_client=source_client,
            source_run_id=source_run_id,
            source_run=source_run,
            metadata=metadata,
            tags=registry_tags,
            tracking_uri=sqlite_tracking_uri,
            retained_features=retained_features,
            experiment_name=REGISTRY_IMPORT_EXPERIMENT,
        )
        registered = registry.register_model(
            run_id=packaging_info["run_id"],
            model_name=MODEL_NAME,
            model_uri=packaging_info["model_uri"],
            description="Phase 1 Ridge champion candidate packaged from the best 60D file-store run.",
            tags=registry_tags,
            metadata=metadata,
        )
        logger.info("registered {} version {} from packaged run {}", MODEL_NAME, registered.version, packaging_info["run_id"])
    else:
        registered = existing_version
        logger.info("reusing existing {} version {} for source run {}", MODEL_NAME, registered.version, source_run_id)

    promoted = registry.transition_stage(
        model_name=MODEL_NAME,
        version=registered.version,
        target_stage=ModelStage.PRODUCTION,
        validation_checks=validation_checks,
    )
    champion = registry.get_champion(MODEL_NAME)
    challengers = registry.list_challengers(MODEL_NAME)
    runner = ChampionChallengerRunner(registry=registry, model_name=MODEL_NAME)
    comparison = registry.compare_versions(MODEL_NAME, registered.version, registered.version)

    # Verify the registered model can be loaded via the champion alias.
    pyfunc_uri = registry.model_uri(
        model_name=MODEL_NAME,
        alias=CHAMPION_ALIAS if champion else None,
        version=None if champion else registered.version,
    )
    loaded_model = mlflow.pyfunc.load_model(pyfunc_uri)

    sqlite_run_count = len(ExperimentTracker(tracking_uri=sqlite_tracking_uri).search_runs(max_results=5_000))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": branch,
        "script": Path(__file__).name,
        "migration": {
            **migration_report.to_dict(),
            "sqlite_run_count_after_migration": sqlite_run_count,
        },
        "source_model": {
            "source_tracking_uri": normalize_tracking_uri(args.source_tracking_uri),
            "source_run_id": source_run_id,
            "window_id": source_window["window_id"],
            "experiment_name": source_window["mlflow"]["experiment_name"],
            "best_hyperparams": source_window["best_hyperparams"],
            "test_ic": source_window["test_metrics"]["ic"],
        },
        "packaging_run": packaging_info,
        "registered_model": {
            "model_name": MODEL_NAME,
            "version": registered.version,
            "stage": champion.stage.value if champion else registered.stage.value,
            "run_id": champion.run_id if champion else registered.run_id,
            "metadata": asdict(metadata),
            "tags": registry_tags,
        },
        "validation_checks": validation_results,
        "promotion": {
            "promoted": bool(promoted),
            "target_stage": ModelStage.PRODUCTION.value,
            "champion_version": champion.version if champion else None,
            "champion_run_id": champion.run_id if champion else None,
        },
        "champion_challenger": {
            "runner_instantiated": True,
            "runner_model_name": runner.model_name,
            "champion": asdict(champion) if champion else None,
            "challenger_count": len(challengers),
            "comparison_self_check": asdict(comparison),
            "loaded_model_type": type(loaded_model).__name__,
        },
    }

    output_path = REPO_ROOT / args.report_path
    write_json_atomic(output_path, json_safe(report))
    logger.info("saved registry setup report to {}", output_path)
    log_summary(report)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate MLflow metadata to SQLite, package the Phase 1 ridge model, and register it as champion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-tracking-uri", default=DEFAULT_FILE_STORE_URI)
    parser.add_argument("--sqlite-tracking-uri", default=None)
    parser.add_argument("--extended-walkforward-report-path", default="data/reports/extended_walkforward.json")
    parser.add_argument("--phase1-alpha-report-path", default="data/reports/phase1_alpha_report_v2.json")
    parser.add_argument("--report-path", default="data/reports/registry_setup.json")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def find_source_window(extended_report: dict[str, Any]) -> dict[str, Any]:
    for window in extended_report["walkforward"]["windows"]:
        if window["window_id"] == "W1":
            return window
    raise RuntimeError("Unable to find W1 in extended walk-forward report.")


def build_model_metadata(
    *,
    source_window: dict[str, Any],
    source_run: Any,
    retained_features: list[str],
    alpha_report: dict[str, Any],
    walkforward_report: dict[str, Any],
) -> ModelMetadata:
    train_start, train_end = split_period(source_window["train_period"])
    val_start, val_end = split_period(source_window["validation_period"])
    aggregate = walkforward_report["walkforward"]["aggregate"]
    windows_positive_ic = int(sum(window["test_metrics"]["ic"] > 0.0 for window in walkforward_report["walkforward"]["windows"]))
    metrics = {
        "mean_oos_ic": float(aggregate["mean_test_ic"]),
        "mean_rank_ic": float(aggregate["mean_test_rank_ic"]),
        "mean_oos_icir": float(aggregate["mean_test_icir"]),
        "dsr_pvalue": float(alpha_report["statistical_tests"]["dsr"]["p_value"]),
        "net_excess_annualized": float(alpha_report["portfolio"]["annualized_net_excess"]),
        "max_drawdown": float(alpha_report["max_drawdown"]["value"]),
        "windows_positive_ic": float(windows_positive_ic),
    }
    return ModelMetadata(
        model_type="ridge",
        horizon=str(walkforward_report["horizon_experiment"]["optimal_horizon"]),
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        features=list(retained_features),
        n_features=len(retained_features),
        hyperparameters={key: coerce_param_value(value) for key, value in source_run.data.params.items()},
        metrics=metrics,
        walk_forward_windows=int(len(walkforward_report["walkforward"]["windows"])),
    )


def build_registry_tags(
    *,
    source_run_id: str,
    metadata: ModelMetadata,
    alpha_report: dict[str, Any],
) -> dict[str, str]:
    decision = alpha_report["final_decision"]
    return {
        "source_run_id": source_run_id,
        "model_type": metadata.model_type,
        "horizon": metadata.horizon,
        "train_windows": "W1-W8",
        "n_features": str(metadata.n_features),
        "mean_oos_ic": f"{metadata.metrics['mean_oos_ic']:.6f}",
        "mean_rank_ic": f"{metadata.metrics['mean_rank_ic']:.6f}",
        "dsr_pvalue": f"{metadata.metrics['dsr_pvalue']:.6f}",
        "net_excess_annualized": f"{metadata.metrics['net_excess_annualized']:.6f}",
        "max_drawdown": f"{metadata.metrics['max_drawdown']:.6f}",
        "phase1_decision": f"{decision['decision']}_{decision['passed_count']}_{decision['total']}",
        "registered_date": datetime.now(timezone.utc).date().isoformat(),
        "feature_list_artifact": "features/retained_features.json",
    }


def evaluate_validation_checks(checks: list[Any], metadata: ModelMetadata) -> list[dict[str, Any]]:
    metrics = dict(metadata.metrics)
    metrics["walk_forward_windows"] = float(metadata.walk_forward_windows)
    results = []
    for check in checks:
        passed = bool(check.check_fn(metrics))
        results.append(
            {
                "name": check.name,
                "description": check.description,
                "passed": passed,
            },
        )
    return results


def find_existing_version(registry: ModelRegistry, *, model_name: str, source_run_id: str):
    try:
        versions = registry.client.search_model_versions(filter_string=f"name = '{model_name}'")
    except MlflowException:
        return None
    for version in sorted(versions, key=lambda item: int(item.version), reverse=True):
        tags = dict(getattr(version, "tags", {}) or {})
        if tags.get("source_run_id") == source_run_id:
            return registry.get_version(model_name=model_name, version=int(version.version))
    return None


def package_source_run_as_pyfunc(
    *,
    source_client: MlflowClient,
    source_run_id: str,
    source_run: Any,
    metadata: ModelMetadata,
    tags: dict[str, str],
    tracking_uri: str,
    retained_features: list[str],
    experiment_name: str,
) -> dict[str, Any]:
    local_model_path = source_client.download_artifacts(source_run_id, "model/model.pkl")
    with open(local_model_path, "rb") as handle:
        model = pickle.load(handle)

    setup_mlflow(tracking_uri=tracking_uri)
    experiment_id = ensure_sqlite_experiment(experiment_name, REGISTRY_IMPORT_ARTIFACT_ROOT, tracking_uri=tracking_uri)
    import_metrics = {key: float(value) for key, value in metadata.metrics.items()}
    import_metrics["source_test_ic"] = float(source_run.data.metrics.get("test_ic", 0.0))

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"import_{source_run_id}") as run:
        mlflow.set_tags(
            {
                "run_kind": "registry_packaging",
                "source_run_id": source_run_id,
                "source_tracking_uri": normalize_tracking_uri(DEFAULT_FILE_STORE_URI),
                "model_type": metadata.model_type,
                "horizon": metadata.horizon,
            },
        )
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        mlflow.log_params({key: str(value) for key, value in source_run.data.params.items()})
        mlflow.log_metrics(import_metrics)
        mlflow.log_dict({"features": retained_features}, "features/retained_features.json")
        mlflow.log_dict(
            {
                "source_run_id": source_run_id,
                "source_params": dict(source_run.data.params),
                "source_metrics": dict(source_run.data.metrics),
                "metadata": asdict(metadata),
            },
            "metadata/source_run_summary.json",
        )
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=PickledBaseModelPyfunc(model),
            tags={
                "model_type": metadata.model_type,
                "horizon": metadata.horizon,
            },
            model_type=metadata.model_type,
        )

    return {
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "experiment_id": experiment_id,
        "run_id": run.info.run_id,
        "artifact_uri": run.info.artifact_uri,
        "model_uri": model_info.model_uri,
        "model_id": model_info.model_id,
    }


def ensure_sqlite_experiment(experiment_name: str, artifact_location: str, *, tracking_uri: str) -> str:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        return experiment.experiment_id
    return client.create_experiment(experiment_name, artifact_location=artifact_location)


def split_period(period: str) -> tuple[str, str]:
    start, end = [item.strip() for item in period.split("->", maxsplit=1)]
    return start, end


def coerce_param_value(value: str) -> Any:
    lowered = str(value).lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except ValueError:
        return value


def log_summary(report: dict[str, Any]) -> None:
    logger.info(
        "sqlite_runs={} champion={} v{} promoted={}",
        report["migration"]["sqlite_run_count_after_migration"],
        report["registered_model"]["model_name"],
        report["registered_model"]["version"],
        report["promotion"]["promoted"],
    )
    logger.info(
        "validation passes={}/{}",
        sum(1 for row in report["validation_checks"] if row["passed"]),
        len(report["validation_checks"]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
