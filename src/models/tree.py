from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from src.models.base import BaseModel, MLflowLoggingMixin
from src.models.evaluation import EvaluationSummary, information_coefficient

XGBOOST_SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 500, 1000],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [5, 10, 20],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [0.0, 0.1, 1.0],
}

LIGHTGBM_SEARCH_SPACE = {
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 500, 1000],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_samples": [20, 50, 100],
    "reg_alpha": [0.0, 0.01, 0.1],
    "reg_lambda": [0.0, 0.1, 1.0],
}

COMPARISON_METRICS = (
    "ic",
    "rank_ic",
    "icir",
    "hit_rate",
    "top_decile_return",
)


@dataclass(frozen=True)
class TreeSearchResult:
    best_params: dict[str, Any]
    best_ic: float
    trials: pd.DataFrame
    n_iter: int


class _TreeModelBase(MLflowLoggingMixin, BaseModel):
    estimator_name = "tree"
    default_params: dict[str, Any] = {}
    search_space: Mapping[str, Sequence[Any]] = {}

    def __init__(
        self,
        *,
        random_state: int = 42,
        n_jobs: int = 1,
        search_n_iter: int = 75,
        **estimator_params: Any,
    ) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.search_n_iter = search_n_iter
        self.estimator_params = self.default_params | estimator_params
        self.estimator_: Any | None = None
        self.feature_names_: list[str] = []
        self.search_result_: TreeSearchResult | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> _TreeModelBase:
        features = self._validate_training_features(X)
        target = self.validate_target(y, expected_index=features.index)
        estimator = self._make_estimator(self.estimator_params)
        estimator.fit(features, target)
        self.estimator_ = estimator
        self.feature_names_ = list(features.columns)
        logger.info(
            "trained {} rows={} features={} params={}",
            self.model_type,
            len(features),
            len(self.feature_names_),
            self.estimator_params,
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be trained before prediction.")

        features = self.validate_features(X)
        missing_columns = [column for column in self.feature_names_ if column not in features.columns]
        if missing_columns:
            raise ValueError(f"prediction features are missing trained columns: {missing_columns}")

        ordered = features.loc[:, self.feature_names_].replace([np.inf, -np.inf], np.nan)
        if ordered.isnull().any(axis=None):
            raise ValueError(f"{self.__class__.__name__} does not accept missing feature values.")

        predictions = self.estimator_.predict(ordered)
        return pd.Series(predictions, index=ordered.index, name="score", dtype=float)

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> EvaluationSummary:
        return super().evaluate(y_true=y_true, y_pred=y_pred)

    def select_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        *,
        n_iter: int | None = None,
        search_space: Mapping[str, Sequence[Any]] | None = None,
        tracker: Any | None = None,
        target_horizon: str = "5D",
        window_id: str = "search",
        timestamp: str | None = None,
    ) -> TreeSearchResult:
        train_features = self._validate_training_features(X_train)
        train_target = self.validate_target(y_train, expected_index=train_features.index)
        val_features = self._validate_training_features(X_val)
        val_target = self.validate_target(y_val, expected_index=val_features.index)

        active_search_space = dict(search_space or self.search_space)
        active_n_iter = min(
            int(n_iter or self.search_n_iter),
            _total_parameter_combinations(active_search_space),
        )
        search_X = pd.concat([train_features, val_features], axis=0)
        search_y = pd.concat([train_target, val_target], axis=0)
        test_fold = np.concatenate(
            [
                np.full(len(train_features), -1, dtype=int),
                np.zeros(len(val_features), dtype=int),
            ],
        )
        splitter = PredefinedSplit(test_fold=test_fold)
        search = RandomizedSearchCV(
            estimator=self._make_estimator(self.estimator_params),
            param_distributions=active_search_space,
            n_iter=active_n_iter,
            scoring=_validation_ic_scorer,
            cv=splitter,
            refit=False,
            random_state=self.random_state,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(search_X, search_y)

        best_params = {key: _normalize_scalar(value) for key, value in search.best_params_.items()}
        best_score = float(search.best_score_)
        self.estimator_params = self.estimator_params | best_params
        trials = _search_results_frame(search.cv_results_)
        self.search_result_ = TreeSearchResult(
            best_params=best_params,
            best_ic=best_score,
            trials=trials,
            n_iter=active_n_iter,
        )

        if tracker is not None:
            best_trial_index = int(trials["rank_test_score"].idxmin()) if not trials.empty else 0
            tracker.log_search_trials(
                model_type=self.model_type,
                target_horizon=target_horizon,
                window_id=window_id,
                trials=trials,
                best_index=best_trial_index,
                timestamp=timestamp,
                search_method="RandomizedSearchCV",
            )

        logger.info(
            "selected {} params best_validation_ic={:.6f} n_iter={}",
            self.model_type,
            best_score,
            active_n_iter,
        )
        return self.search_result_

    @property
    def feature_importances_(self) -> pd.Series:
        if self.estimator_ is None:
            raise RuntimeError(f"{self.__class__.__name__} must be trained before reading feature importances.")
        importances = getattr(self.estimator_, "feature_importances_", None)
        if importances is None:
            raise AttributeError(f"{self.__class__.__name__} estimator does not expose feature_importances_.")
        return pd.Series(importances, index=self.feature_names_, name="importance", dtype=float)

    def get_params(self) -> dict[str, Any]:
        params = {
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
            "search_n_iter": self.search_n_iter,
        }
        params.update({key: _normalize_scalar(value) for key, value in self.estimator_params.items()})
        if self.search_result_ is not None:
            params["best_validation_ic"] = self.search_result_.best_ic
        return params

    def _validate_training_features(self, X: pd.DataFrame) -> pd.DataFrame:
        features = self.validate_features(X)
        clean = features.replace([np.inf, -np.inf], np.nan)
        if clean.isnull().any(axis=None):
            raise ValueError(f"{self.__class__.__name__} does not accept missing feature values.")
        return clean

    def _make_estimator(self, params: Mapping[str, Any]) -> Any:
        raise NotImplementedError


class XGBoostModel(_TreeModelBase):
    model_type = "xgboost"
    default_params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    search_space = XGBOOST_SEARCH_SPACE

    def _make_estimator(self, params: Mapping[str, Any]) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=0,
            **dict(params),
        )


class LightGBMModel(_TreeModelBase):
    model_type = "lightgbm"
    default_params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    search_space = LIGHTGBM_SEARCH_SPACE

    def _make_estimator(self, params: Mapping[str, Any]) -> Any:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            objective="regression",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbosity=-1,
            **dict(params),
        )


def feature_importance_frame(model: _TreeModelBase) -> pd.DataFrame:
    importances = model.feature_importances_.sort_values(ascending=False)
    return pd.DataFrame(
        {
            "feature": importances.index,
            "importance": importances.to_numpy(dtype=float),
            "rank": np.arange(1, len(importances) + 1, dtype=int),
        },
    )


def export_feature_importance_data(model: _TreeModelBase) -> pd.DataFrame:
    return feature_importance_frame(model)


def zero_contribution_features(model: _TreeModelBase, *, threshold: float = 0.0) -> list[str]:
    importances = model.feature_importances_
    return sorted(importances.loc[importances <= threshold].index.tolist())


def compare_model_metrics(
    baseline_result: EvaluationSummary | Mapping[str, float],
    candidate_result: EvaluationSummary | Mapping[str, float],
    *,
    baseline_name: str = "baseline",
    candidate_name: str = "tree",
) -> pd.DataFrame:
    baseline_metrics = _coerce_metrics_mapping(baseline_result)
    candidate_metrics = _coerce_metrics_mapping(candidate_result)
    rows: list[dict[str, Any]] = []

    for metric_name in COMPARISON_METRICS:
        baseline_value = float(baseline_metrics[metric_name])
        candidate_value = float(candidate_metrics[metric_name])
        delta = candidate_value - baseline_value
        pct_change = np.nan
        if not np.isclose(baseline_value, 0.0):
            pct_change = delta / abs(baseline_value)
        rows.append(
            {
                "metric": metric_name,
                baseline_name: baseline_value,
                candidate_name: candidate_value,
                "delta": delta,
                "pct_change": pct_change,
            },
        )

    return pd.DataFrame(rows).set_index("metric")


def _validation_ic_scorer(estimator: Any, X: Any, y: Any) -> float:
    if isinstance(y, pd.Series):
        y_true = y
    else:
        index = X.index if hasattr(X, "index") else None
        y_true = pd.Series(y, index=index, dtype=float)

    predictions = estimator.predict(X)
    y_pred = pd.Series(
        predictions,
        index=X.index if hasattr(X, "index") else y_true.index,
        dtype=float,
    )
    score = information_coefficient(y_true=y_true, y_pred=y_pred)
    return float(score) if pd.notna(score) else float("-inf")


def _search_results_frame(cv_results: Mapping[str, Sequence[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(cv_results).copy()
    param_columns = sorted(column for column in frame.columns if column.startswith("param_"))
    selected_columns = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time", *param_columns]
    selected = frame.loc[:, [column for column in selected_columns if column in frame.columns]].copy()
    selected.rename(
        columns={
            "mean_test_score": "validation_ic",
            "std_test_score": "validation_ic_std",
        },
        inplace=True,
    )
    for column in list(selected.columns):
        if column.startswith("param_"):
            selected.rename(columns={column: column.replace("param_", "")}, inplace=True)

    for column in selected.columns:
        if selected[column].dtype == object:
            selected[column] = selected[column].map(_normalize_scalar)
    return selected.sort_values("rank_test_score").reset_index(drop=True)


def _normalize_scalar(value: Any) -> Any:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _coerce_metrics_mapping(result: EvaluationSummary | Mapping[str, float]) -> dict[str, float]:
    if isinstance(result, EvaluationSummary):
        return result.to_dict()
    return {str(key): float(value) for key, value in result.items()}


def _total_parameter_combinations(search_space: Mapping[str, Sequence[Any]]) -> int:
    if not search_space:
        return 1
    return int(prod(len(tuple(values)) for values in search_space.values()))
