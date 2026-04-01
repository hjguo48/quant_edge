from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
import pickle
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
from loguru import logger
from mlflow.entities import Run
from mlflow.tracking import MlflowClient
import pandas as pd

from src.mlflow_config import setup_mlflow
from src.models.base import BaseModel
from src.models.evaluation import EvaluationSummary


@dataclass(frozen=True)
class LoggedModelRun:
    tracking_uri: str
    experiment_name: str
    experiment_id: str
    run_id: str


@dataclass(frozen=True)
class ValidationWindowConfig:
    train_start: date = date(2018, 1, 1)
    train_end: date = date(2020, 12, 31)
    validation_start: date = date(2021, 1, 1)
    validation_end: date = date(2021, 6, 30)
    test_start: date = date(2021, 7, 1)
    test_end: date = date(2021, 12, 31)
    rebalance_weekday: int = 4
    target_horizon: str = "5D"
    pass_ic_threshold: float = 0.01
    refit_on_train_plus_validation: bool = True


@dataclass(frozen=True)
class ValidationWindowResult:
    best_alpha: float | None
    validation_metrics: EvaluationSummary
    test_metrics: EvaluationSummary
    passed: bool
    train_rows: int
    validation_rows: int
    test_rows: int
    window_id: str
    logged_run: LoggedModelRun | None = None


class ExperimentTracker:
    def __init__(self, *, tracking_uri: str | None = None) -> None:
        self.tracking_uri = tracking_uri

    def log_training_run(
        self,
        *,
        model: BaseModel,
        target_horizon: str,
        window_id: str,
        metrics: Mapping[str, float],
        params: Mapping[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> LoggedModelRun:
        tracking_uri = self._setup_tracking_uri()
        experiment_name = build_experiment_name(
            model_type=model.model_type,
            target_horizon=target_horizon,
            timestamp=timestamp,
        )
        experiment_id = self._ensure_experiment(experiment_name)

        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.set_tags(
                {
                    "model_type": model.model_type,
                    "horizon": target_horizon,
                    "window_id": window_id,
                },
            )
            if params:
                mlflow.log_params({key: str(value) for key, value in params.items()})
            mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

            with TemporaryDirectory(prefix="quantedge-model-") as temp_dir:
                artifact_path = f"{temp_dir}/model.pkl"
                with open(artifact_path, "wb") as handle:
                    pickle.dump(model, handle)
                mlflow.log_artifact(artifact_path, artifact_path="model")

            mlflow.log_dict(
                {
                    "model_type": model.model_type,
                    "params": dict(params or {}),
                    "metrics": dict(metrics),
                },
                "metadata/run_summary.json",
            )

        logger.info(
            "logged {} experiment={} run_id={}",
            model.model_type,
            experiment_name,
            run.info.run_id,
        )
        return LoggedModelRun(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            experiment_id=experiment_id,
            run_id=run.info.run_id,
        )

    def search_runs(
        self,
        *,
        model_type: str | None = None,
        horizon: str | None = None,
        max_results: int = 100,
    ) -> list[Run]:
        tracking_uri = self._setup_tracking_uri()
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment_ids = [experiment.experiment_id for experiment in client.search_experiments(max_results=500)]
        if not experiment_ids:
            return []

        filters: list[str] = []
        if model_type:
            filters.append(f"tags.model_type = '{model_type}'")
        if horizon:
            filters.append(f"tags.horizon = '{horizon}'")
        filter_string = " and ".join(filters)
        return client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=max_results,
            order_by=["attributes.start_time DESC"],
        )

    def _setup_tracking_uri(self) -> str:
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
            return self.tracking_uri
        return setup_mlflow()

    @staticmethod
    def _ensure_experiment(experiment_name: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return experiment.experiment_id
        return mlflow.create_experiment(experiment_name)


def build_experiment_name(
    *,
    model_type: str,
    target_horizon: str,
    timestamp: str | None = None,
) -> str:
    run_timestamp = timestamp or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"{model_type}_{target_horizon}_{run_timestamp}"


def run_single_window_validation(
    *,
    model: BaseModel,
    X: pd.DataFrame,
    y: pd.Series,
    config: ValidationWindowConfig | None = None,
    tracker: ExperimentTracker | None = None,
    timestamp: str | None = None,
) -> ValidationWindowResult:
    active_config = config or ValidationWindowConfig()
    train_X, train_y = _slice_window(
        X=X,
        y=y,
        start_date=active_config.train_start,
        end_date=active_config.train_end,
        rebalance_weekday=active_config.rebalance_weekday,
    )
    validation_X, validation_y = _slice_window(
        X=X,
        y=y,
        start_date=active_config.validation_start,
        end_date=active_config.validation_end,
        rebalance_weekday=active_config.rebalance_weekday,
    )
    test_X, test_y = _slice_window(
        X=X,
        y=y,
        start_date=active_config.test_start,
        end_date=active_config.test_end,
        rebalance_weekday=active_config.rebalance_weekday,
    )

    if hasattr(model, "select_alpha"):
        selection = getattr(model, "select_alpha")(train_X, train_y, validation_X, validation_y)
        best_alpha = getattr(selection, "best_alpha", None)
    else:
        best_alpha = None

    model.train(train_X, train_y)
    validation_predictions = model.predict(validation_X)
    validation_metrics = model.evaluate(validation_y, validation_predictions)

    final_train_X = train_X
    final_train_y = train_y
    if active_config.refit_on_train_plus_validation:
        final_train_X = pd.concat([train_X, validation_X]).sort_index()
        final_train_y = pd.concat([train_y, validation_y]).sort_index()
    model.train(final_train_X, final_train_y)
    test_predictions = model.predict(test_X)
    test_metrics = model.evaluate(test_y, test_predictions)

    window_id = (
        f"{active_config.train_start.isoformat()}_"
        f"{active_config.validation_end.isoformat()}_"
        f"{active_config.test_end.isoformat()}"
    )
    logged_run = None
    if tracker is not None:
        metrics = {
            "validation_ic": validation_metrics.ic,
            "validation_rank_ic": validation_metrics.rank_ic,
            "validation_icir": validation_metrics.icir,
            "validation_hit_rate": validation_metrics.hit_rate,
            "test_ic": test_metrics.ic,
            "test_rank_ic": test_metrics.rank_ic,
            "test_icir": test_metrics.icir,
            "test_hit_rate": test_metrics.hit_rate,
        }
        params = model.get_params() | {
            "train_start": active_config.train_start.isoformat(),
            "train_end": active_config.train_end.isoformat(),
            "validation_start": active_config.validation_start.isoformat(),
            "validation_end": active_config.validation_end.isoformat(),
            "test_start": active_config.test_start.isoformat(),
            "test_end": active_config.test_end.isoformat(),
            "rebalance_weekday": active_config.rebalance_weekday,
        }
        logged_run = tracker.log_training_run(
            model=model,
            target_horizon=active_config.target_horizon,
            window_id=window_id,
            metrics=metrics,
            params=params,
            timestamp=timestamp,
        )

    passed = pd.notna(test_metrics.ic) and float(test_metrics.ic) > active_config.pass_ic_threshold
    return ValidationWindowResult(
        best_alpha=best_alpha,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        passed=bool(passed),
        train_rows=len(train_X),
        validation_rows=len(validation_X),
        test_rows=len(test_X),
        window_id=window_id,
        logged_run=logged_run,
    )


def _slice_window(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    start_date: date,
    end_date: date,
    rebalance_weekday: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if not isinstance(X.index, pd.MultiIndex) or not isinstance(y.index, pd.MultiIndex):
        raise ValueError("single-window validation requires MultiIndex(date, ticker) inputs.")

    date_level = _date_level_name(X.index)
    feature_dates = pd.to_datetime(pd.Index(X.index.get_level_values(date_level)))
    target_dates = pd.to_datetime(pd.Index(y.index.get_level_values(date_level)))

    feature_mask = (
        (feature_dates >= pd.Timestamp(start_date))
        & (feature_dates <= pd.Timestamp(end_date))
        & (feature_dates.weekday == rebalance_weekday)
    )
    target_mask = (
        (target_dates >= pd.Timestamp(start_date))
        & (target_dates <= pd.Timestamp(end_date))
        & (target_dates.weekday == rebalance_weekday)
    )

    sliced_X = X.loc[feature_mask].sort_index()
    sliced_y = y.loc[target_mask].sort_index()
    aligned_index = sliced_X.index.intersection(sliced_y.index)
    if aligned_index.empty:
        raise ValueError(
            f"window {start_date} -> {end_date} contains no aligned rebalance observations",
        )
    return sliced_X.loc[aligned_index], sliced_y.loc[aligned_index]


def _date_level_name(index: pd.MultiIndex) -> str | int:
    for candidate in ("date", "trade_date", "signal_date", "calc_date"):
        if candidate in index.names:
            return candidate
    return 0
