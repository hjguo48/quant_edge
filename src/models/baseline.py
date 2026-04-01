from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge

from src.models.base import BaseModel, MLflowLoggingMixin
from src.models.evaluation import EvaluationSummary, information_coefficient

DEFAULT_ALPHA_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_DOMAIN_KEYWORDS = {
    "momentum": ("momentum", "mom", "trend", "rsi", "breakout"),
    "volatility": ("volatility", "vol", "atr", "beta", "drawdown"),
    "value": ("value", "pe", "pb", "ps", "ev", "yield", "fcf"),
    "quality": ("quality", "roe", "roa", "margin", "profit", "accrual", "debt"),
    "macro": ("macro", "vix", "spread", "ffr", "breadth", "term"),
}


@dataclass(frozen=True)
class AlphaSelectionResult:
    best_alpha: float
    best_ic: float
    scores_by_alpha: dict[float, float]


class RidgeBaselineModel(MLflowLoggingMixin, BaseModel):
    model_type = "ridge"

    def __init__(
        self,
        *,
        alpha: float = 1.0,
        alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
        fit_intercept: bool = True,
    ) -> None:
        self.alpha = float(alpha)
        self.alpha_grid = tuple(float(candidate) for candidate in alpha_grid)
        self.fit_intercept = fit_intercept
        self.estimator_: Ridge | None = None
        self.feature_names_: list[str] = []
        self.alpha_selection_: AlphaSelectionResult | None = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> RidgeBaselineModel:
        features = self.validate_features(X)
        target = self.validate_target(y, expected_index=features.index)
        clean = features.copy()
        clean = clean.replace([np.inf, -np.inf], np.nan)
        if clean.isnull().any(axis=None):
            raise ValueError("RidgeBaselineModel does not accept missing feature values.")

        estimator = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        estimator.fit(clean.to_numpy(dtype=float), target.to_numpy(dtype=float))
        self.estimator_ = estimator
        self.feature_names_ = list(clean.columns)
        logger.info("trained ridge baseline alpha={} rows={}", self.alpha, len(clean))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.estimator_ is None:
            raise RuntimeError("RidgeBaselineModel must be trained before prediction.")

        features = self.validate_features(X)
        missing_columns = [column for column in self.feature_names_ if column not in features.columns]
        if missing_columns:
            raise ValueError(f"prediction features are missing trained columns: {missing_columns}")

        ordered = features.loc[:, self.feature_names_].replace([np.inf, -np.inf], np.nan)
        if ordered.isnull().any(axis=None):
            raise ValueError("RidgeBaselineModel does not accept missing feature values.")

        predictions = self.estimator_.predict(ordered.to_numpy(dtype=float))
        return pd.Series(predictions, index=ordered.index, name="score", dtype=float)

    def select_alpha(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> AlphaSelectionResult:
        train_features = self.validate_features(X_train)
        train_target = self.validate_target(y_train, expected_index=train_features.index)
        val_features = self.validate_features(X_val)
        val_target = self.validate_target(y_val, expected_index=val_features.index)

        scores_by_alpha: dict[float, float] = {}
        best_alpha = self.alpha
        best_score = float("-inf")

        for candidate in self.alpha_grid:
            estimator = Ridge(alpha=float(candidate), fit_intercept=self.fit_intercept)
            estimator.fit(
                train_features.to_numpy(dtype=float),
                train_target.to_numpy(dtype=float),
            )
            predictions = pd.Series(
                estimator.predict(val_features.to_numpy(dtype=float)),
                index=val_features.index,
                dtype=float,
            )
            score = information_coefficient(y_true=val_target, y_pred=predictions)
            safe_score = score if pd.notna(score) else float("-inf")
            scores_by_alpha[float(candidate)] = safe_score
            if safe_score > best_score:
                best_score = safe_score
                best_alpha = float(candidate)

        self.alpha = best_alpha
        self.alpha_selection_ = AlphaSelectionResult(
            best_alpha=best_alpha,
            best_ic=best_score,
            scores_by_alpha=scores_by_alpha,
        )
        logger.info("selected ridge alpha={} validation_ic={:.6f}", best_alpha, best_score)
        return self.alpha_selection_

    def get_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "alpha": self.alpha,
            "alpha_grid": ",".join(str(candidate) for candidate in self.alpha_grid),
            "fit_intercept": self.fit_intercept,
        }
        if self.alpha_selection_ is not None:
            params["selected_alpha"] = self.alpha_selection_.best_alpha
        return params


class FactorRankBaselineModel(MLflowLoggingMixin, BaseModel):
    model_type = "factor_rank"

    def __init__(
        self,
        *,
        domain_features: Mapping[str, Sequence[str]] | None = None,
        feature_signs: Mapping[str, int] | None = None,
    ) -> None:
        self.domain_features = {
            domain: tuple(features)
            for domain, features in (domain_features or {}).items()
        }
        self.feature_signs = {feature: int(sign) for feature, sign in (feature_signs or {}).items()}
        self.active_domains_: dict[str, tuple[str, ...]] = {}
        self.feature_names_: list[str] = []

    def train(self, X: pd.DataFrame, y: pd.Series) -> FactorRankBaselineModel:
        features = self.validate_features(X)
        self.validate_target(y, expected_index=features.index)
        self.feature_names_ = list(features.columns)
        self.active_domains_ = self._resolve_domain_features(features.columns)
        if not self.active_domains_:
            raise ValueError("no domain features resolved for factor rank baseline")
        logger.info(
            "prepared factor rank baseline with domains={}",
            ",".join(sorted(self.active_domains_)),
        )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        features = self.validate_features(X)
        if not self.active_domains_:
            self.active_domains_ = self._resolve_domain_features(features.columns)
        if not self.active_domains_:
            raise ValueError("no domain features resolved for factor rank baseline")

        scored = self._score_by_domain(features)
        return scored.mean(axis=1).rename("score")

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> EvaluationSummary:
        return super().evaluate(y_true=y_true, y_pred=y_pred)

    def get_params(self) -> dict[str, Any]:
        return {
            "domains": ",".join(sorted(self.active_domains_)) if self.active_domains_ else "",
            "domain_features": str(self.active_domains_ or self.domain_features),
        }

    def _score_by_domain(self, X: pd.DataFrame) -> pd.DataFrame:
        scored_domains: dict[str, pd.Series] = {}
        grouped = self._group_cross_sections(X)

        for domain, feature_names in self.active_domains_.items():
            domain_parts: list[pd.Series] = []
            for _, frame in grouped:
                domain_frame = frame.loc[:, list(feature_names)].copy()
                for column in domain_frame.columns:
                    domain_frame[column] = domain_frame[column] * self.feature_signs.get(column, 1)
                ranked = domain_frame.rank(axis=0, method="average", ascending=True)
                domain_parts.append(ranked.mean(axis=1))
            scored_domains[domain] = pd.concat(domain_parts).sort_index()

        return pd.DataFrame(scored_domains).loc[X.index]

    def _group_cross_sections(self, X: pd.DataFrame) -> list[tuple[object, pd.DataFrame]]:
        if isinstance(X.index, pd.MultiIndex):
            date_level = _date_level_name(X.index)
            return [
                (date_key, frame)
                for date_key, frame in X.groupby(level=date_level, sort=True)
            ]
        return [(pd.NaT, X)]

    def _resolve_domain_features(self, columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
        resolved: dict[str, tuple[str, ...]] = {}
        available = list(columns)

        if self.domain_features:
            for domain, features in self.domain_features.items():
                matched = tuple(feature for feature in features if feature in available)
                if matched:
                    resolved[domain] = matched
            return resolved

        normalized = {column: str(column).lower() for column in available}
        for domain, keywords in DEFAULT_DOMAIN_KEYWORDS.items():
            matched = tuple(
                column
                for column, lowered in normalized.items()
                if any(keyword in lowered for keyword in keywords)
            )
            if matched:
                resolved[domain] = matched
        return resolved


def _date_level_name(index: pd.MultiIndex) -> str | int:
    for candidate in ("date", "trade_date", "signal_date", "calc_date"):
        if candidate in index.names:
            return candidate
    return 0
