from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.models.evaluation import information_coefficient, rank_information_coefficient
from src.models.registry import ModelRegistry, RegisteredModelVersion


@dataclass(frozen=True)
class ShadowResult:
    window_id: str
    champion_ic: float
    challenger_ic: float
    champion_rank_ic: float
    challenger_rank_ic: float
    delta_ic: float
    timestamp: str


@dataclass(frozen=True)
class AccumulatedComparison:
    results: list[ShadowResult]
    consecutive_challenger_wins: int
    total_periods: int
    champion_mean_ic: float
    challenger_mean_ic: float


@dataclass(frozen=True)
class PromotionDecision:
    recommend_promotion: bool
    reason: str
    consecutive_wins: int
    required_wins: int


class ChampionChallengerRunner:
    """Runs Champion and Challenger models side-by-side for comparison."""

    def __init__(self, registry: ModelRegistry, model_name: str):
        self.registry = registry
        self.model_name = model_name

    def run_shadow_comparison(
        self,
        *,
        X: pd.DataFrame,
        y: pd.Series,
        window_id: str,
    ) -> ShadowResult:
        champion = self.registry.get_champion(self.model_name)
        challengers = self.registry.list_challengers(self.model_name)
        if champion is None:
            raise RuntimeError(f"No champion model registered for {self.model_name}.")
        if not challengers:
            raise RuntimeError(f"No challenger models registered for {self.model_name}.")

        challenger = challengers[0]
        champion_model = mlflow.pyfunc.load_model(self.registry.model_uri(model_name=self.model_name, alias="champion"))
        challenger_model = mlflow.pyfunc.load_model(
            self.registry.model_uri(model_name=self.model_name, version=challenger.version),
        )

        champion_pred = self._predict_series(champion_model, X)
        challenger_pred = self._predict_series(challenger_model, X)
        champion_ic = information_coefficient(y_true=y, y_pred=champion_pred)
        challenger_ic = information_coefficient(y_true=y, y_pred=challenger_pred)
        champion_rank_ic = rank_information_coefficient(y_true=y, y_pred=champion_pred)
        challenger_rank_ic = rank_information_coefficient(y_true=y, y_pred=challenger_pred)

        return ShadowResult(
            window_id=window_id,
            champion_ic=float(champion_ic),
            challenger_ic=float(challenger_ic),
            champion_rank_ic=float(champion_rank_ic),
            challenger_rank_ic=float(challenger_rank_ic),
            delta_ic=float(challenger_ic - champion_ic),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def accumulate_results(self, results: list[ShadowResult]) -> AccumulatedComparison:
        consecutive = 0
        best_streak = 0
        for result in results:
            if result.delta_ic > 0.0:
                consecutive += 1
                best_streak = max(best_streak, consecutive)
            else:
                consecutive = 0

        return AccumulatedComparison(
            results=list(results),
            consecutive_challenger_wins=best_streak,
            total_periods=len(results),
            champion_mean_ic=float(np.mean([result.champion_ic for result in results])) if results else float("nan"),
            challenger_mean_ic=float(np.mean([result.challenger_ic for result in results])) if results else float("nan"),
        )

    def check_promotion_criteria(self, accumulated: AccumulatedComparison) -> PromotionDecision:
        required_wins = 4
        if accumulated.consecutive_challenger_wins >= required_wins and accumulated.challenger_mean_ic > accumulated.champion_mean_ic:
            return PromotionDecision(
                recommend_promotion=True,
                reason="Challenger beat champion for at least four consecutive periods and has higher mean IC.",
                consecutive_wins=accumulated.consecutive_challenger_wins,
                required_wins=required_wins,
            )

        return PromotionDecision(
            recommend_promotion=False,
            reason="Challenger has not cleared the four-period consecutive win requirement.",
            consecutive_wins=accumulated.consecutive_challenger_wins,
            required_wins=required_wins,
        )

    @staticmethod
    def _predict_series(pyfunc_model: Any, X: pd.DataFrame) -> pd.Series:
        output = pyfunc_model.predict(X)
        if isinstance(output, pd.Series):
            return output.rename("score")
        if isinstance(output, pd.DataFrame):
            if "score" in output.columns:
                return pd.Series(output["score"].to_numpy(dtype=float), index=X.index, name="score")
            if output.shape[1] == 1:
                return pd.Series(output.iloc[:, 0].to_numpy(dtype=float), index=X.index, name="score")
            raise ValueError("PyFunc model returned a DataFrame without a unique score column.")

        array = np.asarray(output, dtype=float)
        if array.ndim > 1:
            array = array.reshape(len(X), -1)[:, 0]
        return pd.Series(array, index=X.index, name="score")
