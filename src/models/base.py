from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import pandas as pd
from loguru import logger

from src.models.evaluation import EvaluationSummary, evaluate_predictions


class BaseModel(ABC):
    model_type = "base"

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        """Fit the model on a feature matrix and aligned target series."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate a continuous score for each row in the supplied feature matrix."""

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> EvaluationSummary:
        """Evaluate aligned predictions with the project's baseline metric set."""

        return evaluate_predictions(y_true=y_true, y_pred=y_pred)

    def get_params(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def validate_features(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("X must not be empty.")
        if X.columns.empty:
            raise ValueError("X must contain at least one feature column.")
        if X.isnull().all(axis=None):
            raise ValueError("X contains only missing values.")
        return X.copy()

    @staticmethod
    def validate_target(y: pd.Series, *, expected_index: pd.Index | None = None) -> pd.Series:
        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series.")
        if y.empty:
            raise ValueError("y must not be empty.")
        if expected_index is not None and not y.index.equals(expected_index):
            raise ValueError("y index must align with X index.")
        return y.copy()


class MLflowLoggingMixin:
    def log_training_run(
        self,
        *,
        target_horizon: str,
        window_id: str,
        metrics: Mapping[str, float],
        tracking_uri: str | None = None,
        timestamp: str | None = None,
        extra_params: Mapping[str, Any] | None = None,
    ) -> Any:
        from src.models.experiment import ExperimentTracker

        if not isinstance(self, BaseModel):
            raise TypeError("MLflowLoggingMixin must be combined with BaseModel.")

        tracker = ExperimentTracker(tracking_uri=tracking_uri)
        params = dict(self.get_params())
        if extra_params:
            params.update(dict(extra_params))
        logger.info(
            "logging {} training run horizon={} window_id={}",
            self.model_type,
            target_horizon,
            window_id,
        )
        return tracker.log_training_run(
            model=self,
            target_horizon=target_horizon,
            window_id=window_id,
            metrics=metrics,
            params=params,
            timestamp=timestamp,
        )
