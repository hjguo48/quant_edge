from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS


@dataclass(frozen=True)
class CovarianceResult:
    matrix: np.ndarray
    tickers: list[str]
    method: str
    shrinkage_coefficient: float
    condition_number_before: float
    condition_number_after: float
    n_observations: int
    n_assets: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["matrix"] = self.matrix.tolist()
        return payload


class CovarianceEstimator:
    """Shrinkage covariance estimation for portfolio optimization."""

    SUPPORTED_METHODS = {"ledoit_wolf", "oas", "sample"}

    def __init__(self, method: str = "ledoit_wolf") -> None:
        normalized = str(method).strip().lower()
        if normalized not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported covariance method {method!r}. "
                f"Expected one of {sorted(self.SUPPORTED_METHODS)}."
            )
        self.method = normalized

    def estimate(
        self,
        returns: pd.DataFrame,
        *,
        min_history: int = 60,
    ) -> CovarianceResult:
        if not isinstance(returns, pd.DataFrame):
            raise TypeError("returns must be a pandas DataFrame.")

        clean = returns.copy().replace([np.inf, -np.inf], np.nan)
        clean.columns = clean.columns.astype(str)
        clean = clean.loc[:, clean.notna().sum(axis=0) >= int(min_history)]
        clean = clean.dropna(axis=0, how="all")
        clean = clean.apply(lambda column: column.fillna(column.mean()), axis=0)
        clean = clean.dropna(axis=1, how="any")
        clean = clean.loc[:, clean.std(axis=0, ddof=0).fillna(0.0) > 0.0]

        if clean.shape[0] < int(min_history):
            raise ValueError(
                f"Need at least {min_history} observations for covariance estimation; "
                f"got {clean.shape[0]}."
            )
        if clean.shape[1] < 2:
            raise ValueError("Need at least two assets for covariance estimation.")

        values = clean.to_numpy(dtype=float)
        sample_cov = np.cov(values, rowvar=False, ddof=1)
        condition_before = self.condition_number(sample_cov)

        if self.method == "sample":
            matrix = sample_cov
            shrinkage = 0.0
        elif self.method == "oas":
            estimator = OAS(store_precision=False, assume_centered=False)
            estimator.fit(values)
            matrix = np.asarray(estimator.covariance_, dtype=float)
            shrinkage = float(estimator.shrinkage_)
        else:
            estimator = LedoitWolf(store_precision=False, assume_centered=False)
            estimator.fit(values)
            matrix = np.asarray(estimator.covariance_, dtype=float)
            shrinkage = float(estimator.shrinkage_)

        condition_after = self.condition_number(matrix)
        return CovarianceResult(
            matrix=np.asarray(matrix, dtype=float),
            tickers=list(clean.columns.astype(str)),
            method=self.method,
            shrinkage_coefficient=float(shrinkage),
            condition_number_before=float(condition_before),
            condition_number_after=float(condition_after),
            n_observations=int(clean.shape[0]),
            n_assets=int(clean.shape[1]),
        )

    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        array = np.asarray(matrix, dtype=float)
        if array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError("matrix must be a square 2D array.")

        singular_values = np.linalg.svd(array, compute_uv=False)
        finite = singular_values[np.isfinite(singular_values)]
        if finite.size == 0:
            return float("inf")

        max_sv = float(np.max(finite))
        positive = finite[finite > 1e-12]
        if positive.size == 0:
            return float("inf")
        min_sv = float(np.min(positive))
        return float(max_sv / min_sv)
