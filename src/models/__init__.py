from src.models.baseline import FactorRankBaselineModel, RidgeBaselineModel
from src.models.base import BaseModel, MLflowLoggingMixin
from src.models.evaluation import EvaluationSummary
from src.models.experiment import ExperimentTracker, ValidationWindowConfig, run_single_window_validation

try:
    from src.models.tree import (
        LightGBMModel,
        XGBoostModel,
        compare_model_metrics,
        export_feature_importance_data,
        feature_importance_frame,
        zero_contribution_features,
    )
except Exception:  # pragma: no cover - allows importing src.models without Week 6 deps.
    LightGBMModel = None
    XGBoostModel = None
    compare_model_metrics = None
    export_feature_importance_data = None
    feature_importance_frame = None
    zero_contribution_features = None

__all__ = [
    "BaseModel",
    "EvaluationSummary",
    "ExperimentTracker",
    "FactorRankBaselineModel",
    "LightGBMModel",
    "MLflowLoggingMixin",
    "RidgeBaselineModel",
    "ValidationWindowConfig",
    "XGBoostModel",
    "compare_model_metrics",
    "export_feature_importance_data",
    "feature_importance_frame",
    "run_single_window_validation",
    "zero_contribution_features",
]
