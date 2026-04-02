from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from typing import Any, Callable

import mlflow
import mlflow.pyfunc
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.mlflow_config import setup_mlflow

MODEL_METADATA_JSON_TAG = "quantedge.metadata_json"
MODEL_STAGE_TAG = "quantedge.stage"
CHAMPION_ALIAS = "champion"


class ModelStage(str, Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


@dataclass(frozen=True)
class ModelMetadata:
    model_type: str
    horizon: str
    train_start: str
    train_end: str
    val_start: str | None
    val_end: str | None
    features: list[str]
    n_features: int
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    walk_forward_windows: int

    def to_tags(self) -> dict[str, str]:
        payload = asdict(self)
        tags = {
            MODEL_METADATA_JSON_TAG: json.dumps(payload, sort_keys=True),
            "model_type": self.model_type,
            "horizon": self.horizon,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "val_start": self.val_start or "",
            "val_end": self.val_end or "",
            "n_features": str(self.n_features),
            "walk_forward_windows": str(self.walk_forward_windows),
            "features_json": json.dumps(self.features, sort_keys=True),
            "hyperparameters_json": json.dumps(self.hyperparameters, sort_keys=True),
            "metrics_json": json.dumps(self.metrics, sort_keys=True),
        }
        return tags

    @classmethod
    def from_tags(cls, tags: dict[str, str]) -> ModelMetadata | None:
        raw = tags.get(MODEL_METADATA_JSON_TAG)
        if raw:
            payload = json.loads(raw)
            payload["n_features"] = int(payload["n_features"])
            payload["walk_forward_windows"] = int(payload["walk_forward_windows"])
            payload["features"] = list(payload.get("features", []))
            payload["hyperparameters"] = dict(payload.get("hyperparameters", {}))
            payload["metrics"] = {key: float(value) for key, value in dict(payload.get("metrics", {})).items()}
            return cls(**payload)

        if "model_type" not in tags or "horizon" not in tags:
            return None
        return cls(
            model_type=tags["model_type"],
            horizon=tags["horizon"],
            train_start=tags.get("train_start", ""),
            train_end=tags.get("train_end", ""),
            val_start=tags.get("val_start") or None,
            val_end=tags.get("val_end") or None,
            features=json.loads(tags.get("features_json", "[]")),
            n_features=int(tags.get("n_features", "0")),
            hyperparameters=json.loads(tags.get("hyperparameters_json", "{}")),
            metrics={key: float(value) for key, value in json.loads(tags.get("metrics_json", "{}")).items()},
            walk_forward_windows=int(tags.get("walk_forward_windows", "0")),
        )


@dataclass(frozen=True)
class RegisteredModelVersion:
    name: str
    version: int
    stage: ModelStage
    run_id: str
    metadata: ModelMetadata | None


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    check_fn: Callable[[dict[str, float]], bool]
    description: str


@dataclass(frozen=True)
class ModelComparison:
    model_name: str
    version_a: int
    version_b: int
    metrics_a: dict[str, float]
    metrics_b: dict[str, float]
    deltas: dict[str, float]
    recommendation: str


class ModelRegistry:
    """MLflow Model Registry wrapper for QuantEdge."""

    def __init__(self, tracking_uri: str | None = None):
        self.tracking_uri = setup_mlflow(tracking_uri=tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def register_model(
        self,
        *,
        run_id: str,
        model_name: str,
        model_uri: str | None = None,
        description: str = "",
        tags: dict[str, str] | None = None,
        metadata: ModelMetadata | None = None,
    ) -> RegisteredModelVersion:
        self._ensure_registered_model(model_name=model_name, description=description)

        version_tags = dict(tags or {})
        if metadata is not None:
            version_tags.update(metadata.to_tags())

        source = model_uri or f"runs:/{run_id}/model"
        model_version = mlflow.register_model(
            model_uri=source,
            name=model_name,
            tags=version_tags or None,
        )
        for key, value in version_tags.items():
            self.client.set_model_version_tag(model_name, model_version.version, key, str(value))

        self._set_stage(model_name=model_name, version=int(model_version.version), stage=ModelStage.STAGING)
        if description:
            self.client.update_model_version(model_name, str(model_version.version), description=description)
        return self._to_registered_model_version(model_version)

    def transition_stage(
        self,
        *,
        model_name: str,
        version: int,
        target_stage: ModelStage,
        validation_checks: list[ValidationCheck] | None = None,
    ) -> bool:
        if target_stage == ModelStage.PRODUCTION:
            candidate = self.get_version(model_name=model_name, version=version)
            candidate_metrics = dict(candidate.metadata.metrics if candidate.metadata else {})
            if candidate.metadata is not None:
                candidate_metrics["walk_forward_windows"] = float(candidate.metadata.walk_forward_windows)
            current_production = self.get_production_model(model_name)
            candidate_metrics["current_production_mean_oos_ic"] = (
                current_production.metadata.metrics.get("mean_oos_ic", 0.0)
                if current_production and current_production.metadata
                else 0.0
            )
            for check in validation_checks or []:
                if not bool(check.check_fn(candidate_metrics)):
                    self.client.set_model_version_tag(model_name, str(version), f"validation.{check.name}", "failed")
                    return False
                self.client.set_model_version_tag(model_name, str(version), f"validation.{check.name}", "passed")

            if current_production is not None and current_production.version != version:
                self._set_stage(
                    model_name=model_name,
                    version=current_production.version,
                    stage=ModelStage.ARCHIVED,
                )
            self.client.set_registered_model_alias(model_name, CHAMPION_ALIAS, str(version))
            self._set_stage(model_name=model_name, version=version, stage=ModelStage.PRODUCTION)
            return True

        if target_stage == ModelStage.STAGING:
            self._set_stage(model_name=model_name, version=version, stage=ModelStage.STAGING)
            return True

        if target_stage == ModelStage.ARCHIVED:
            champion = self.get_production_model(model_name)
            if champion is not None and champion.version == version:
                try:
                    self.client.delete_registered_model_alias(model_name, CHAMPION_ALIAS)
                except MlflowException:
                    pass
            self._set_stage(model_name=model_name, version=version, stage=ModelStage.ARCHIVED)
            return True

        self._set_stage(model_name=model_name, version=version, stage=ModelStage.NONE)
        return True

    def get_production_model(self, model_name: str) -> RegisteredModelVersion | None:
        try:
            version = self.client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
            return self._to_registered_model_version(version)
        except MlflowException:
            pass

        for version in self._search_model_versions(model_name=model_name):
            registered = self._to_registered_model_version(version)
            if registered.stage == ModelStage.PRODUCTION:
                return registered
        return None

    def get_champion(self, model_name: str) -> RegisteredModelVersion | None:
        return self.get_production_model(model_name)

    def list_challengers(self, model_name: str) -> list[RegisteredModelVersion]:
        challengers: list[RegisteredModelVersion] = []
        for version in self._search_model_versions(model_name=model_name):
            registered = self._to_registered_model_version(version)
            if registered.stage == ModelStage.STAGING:
                challengers.append(registered)
        return sorted(challengers, key=lambda item: item.version, reverse=True)

    def compare_versions(
        self,
        model_name: str,
        version_a: int,
        version_b: int,
    ) -> ModelComparison:
        a = self.get_version(model_name=model_name, version=version_a)
        b = self.get_version(model_name=model_name, version=version_b)
        metrics_a = dict(a.metadata.metrics if a.metadata else {})
        metrics_b = dict(b.metadata.metrics if b.metadata else {})
        delta_keys = sorted(set(metrics_a) | set(metrics_b))
        deltas = {
            key: float(metrics_b.get(key, 0.0) - metrics_a.get(key, 0.0))
            for key in delta_keys
        }

        if deltas.get("mean_oos_ic", 0.0) > 0.0 and metrics_b.get("dsr_pvalue", 1.0) <= 0.05:
            recommendation = "promote_b"
        elif deltas.get("mean_oos_ic", 0.0) < 0.0:
            recommendation = "keep_a"
        else:
            recommendation = "inconclusive"

        return ModelComparison(
            model_name=model_name,
            version_a=version_a,
            version_b=version_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
            deltas=deltas,
            recommendation=recommendation,
        )

    def get_version(self, *, model_name: str, version: int) -> RegisteredModelVersion:
        model_version = self.client.get_model_version(model_name, str(version))
        return self._to_registered_model_version(model_version)

    def model_uri(self, *, model_name: str, version: int | None = None, alias: str | None = None) -> str:
        if alias:
            return f"models:/{model_name}@{alias}"
        if version is None:
            raise ValueError("Either version or alias must be provided.")
        return f"models:/{model_name}/{version}"

    def _ensure_registered_model(self, *, model_name: str, description: str) -> None:
        try:
            self.client.get_registered_model(model_name)
        except MlflowException:
            self.client.create_registered_model(model_name, description=description or None)

    def _set_stage(self, *, model_name: str, version: int, stage: ModelStage) -> None:
        self.client.set_model_version_tag(model_name, str(version), MODEL_STAGE_TAG, stage.value)

    def _search_model_versions(self, *, model_name: str) -> list[ModelVersion]:
        return list(self.client.search_model_versions(filter_string=f"name = '{model_name}'"))

    def _to_registered_model_version(self, model_version: ModelVersion) -> RegisteredModelVersion:
        tags = dict(getattr(model_version, "tags", {}) or {})
        metadata = ModelMetadata.from_tags(tags)
        aliases = set(getattr(model_version, "aliases", []) or [])
        stage_value = tags.get(MODEL_STAGE_TAG, ModelStage.NONE.value)
        if CHAMPION_ALIAS in aliases:
            stage_value = ModelStage.PRODUCTION.value
        return RegisteredModelVersion(
            name=model_version.name,
            version=int(model_version.version),
            stage=ModelStage(stage_value),
            run_id=str(model_version.run_id or ""),
            metadata=metadata,
        )


def default_validation_checks() -> list[ValidationCheck]:
    return [
        ValidationCheck(
            name="mean_oos_ic",
            description="Mean out-of-sample IC must exceed 0.03.",
            check_fn=lambda metrics: float(metrics.get("mean_oos_ic", 0.0)) > 0.03,
        ),
        ValidationCheck(
            name="dsr_pvalue",
            description="Deflated Sharpe Ratio p-value must be below 0.05.",
            check_fn=lambda metrics: float(metrics.get("dsr_pvalue", 1.0)) < 0.05,
        ),
        ValidationCheck(
            name="stability",
            description="At least 60% of walk-forward windows must have positive IC.",
            check_fn=lambda metrics: (
                float(metrics.get("windows_positive_ic", 0.0))
                / max(float(metrics.get("walk_forward_windows", 1.0)), 1.0)
            ) >= 0.60,
        ),
        ValidationCheck(
            name="non_regression",
            description="Candidate mean OOS IC must be at least 80% of the current champion.",
            check_fn=lambda metrics: float(metrics.get("mean_oos_ic", 0.0)) >= (
                0.80 * max(float(metrics.get("current_production_mean_oos_ic", 0.0)), 1e-12)
                if float(metrics.get("current_production_mean_oos_ic", 0.0)) > 0.0
                else 0.0
            ),
        ),
    ]
