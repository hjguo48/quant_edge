from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio.constraints import PortfolioConstraints, apply_weight_constraints
from src.portfolio.equal_weight import select_top_scores


def black_litterman_portfolio(
    scores: pd.Series,
    *,
    trailing_returns: pd.DataFrame,
    dollar_liquidity: pd.Series,
    n_stocks: int | None = None,
    selection_pct: float = 0.20,
    constraints: PortfolioConstraints | None = None,
    lookback_days: int = 60,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    score_scale: float = 0.05,
    covariance_shrinkage: float = 0.25,
    ridge: float = 1e-5,
) -> dict[str, float]:
    active_constraints = constraints or PortfolioConstraints()
    selected = select_top_scores(scores, n_stocks=n_stocks, selection_pct=selection_pct)
    if selected.empty:
        return {}

    tickers = selected.index.astype(str)
    history = trailing_returns.reindex(columns=tickers).tail(lookback_days)
    history = history.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="any")
    if history.shape[1] < 2 or history.shape[0] < 20:
        raw = pd.Series(1.0, index=tickers, dtype=float)
        return apply_weight_constraints(raw, ranking=tickers, constraints=active_constraints)

    tickers = history.columns.astype(str)
    selected = selected.reindex(tickers)
    liquidity = (
        pd.Series(dollar_liquidity, dtype=float)
        .reindex(tickers)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )

    cov = history.cov().to_numpy(dtype=float)
    diag = np.diag(np.diag(cov))
    cov = ((1.0 - covariance_shrinkage) * cov) + (covariance_shrinkage * diag)
    cov = cov + (np.eye(len(tickers)) * ridge)

    if float(liquidity.sum()) <= 0.0:
        market_weights = np.full(len(tickers), 1.0 / len(tickers), dtype=float)
    else:
        market_weights = (liquidity / float(liquidity.sum())).to_numpy(dtype=float)

    implied = risk_aversion * cov.dot(market_weights)
    z_scores = _zscore(selected.to_numpy(dtype=float))
    views = z_scores * score_scale
    omega = np.diag(np.maximum(np.diag(tau * cov), ridge))

    tau_cov = tau * cov
    posterior = np.linalg.solve(
        np.linalg.inv(tau_cov) + np.linalg.inv(omega),
        np.linalg.inv(tau_cov).dot(implied) + np.linalg.inv(omega).dot(views),
    )

    optimized = _solve_long_only_weights(
        posterior_returns=posterior,
        covariance=cov,
        market_weights=market_weights,
        max_weight=active_constraints.max_weight,
        risk_aversion=risk_aversion,
    )
    raw = pd.Series(optimized, index=tickers, dtype=float)
    return apply_weight_constraints(raw, ranking=tickers, constraints=active_constraints)


def _solve_long_only_weights(
    *,
    posterior_returns: np.ndarray,
    covariance: np.ndarray,
    market_weights: np.ndarray,
    max_weight: float,
    risk_aversion: float,
) -> np.ndarray:
    n_assets = int(len(posterior_returns))
    if n_assets == 0:
        return np.array([], dtype=float)

    x0 = np.clip(market_weights, 0.0, max_weight if max_weight > 0.0 else 1.0)
    x0 = x0 / max(float(x0.sum()), 1e-12)
    bounds = [(0.0, max_weight if max_weight > 0.0 else 1.0) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}]

    def objective(weights: np.ndarray) -> float:
        risk_term = 0.5 * risk_aversion * float(weights.T @ covariance @ weights)
        return risk_term - float(posterior_returns @ weights)

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if result.success and result.x is not None:
        weights = np.clip(result.x.astype(float), 0.0, None)
        return weights / max(float(weights.sum()), 1e-12)

    fallback = np.clip(posterior_returns, 0.0, None)
    if float(fallback.sum()) <= 0.0:
        return x0
    fallback = fallback / float(fallback.sum())
    return fallback


def _zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std
