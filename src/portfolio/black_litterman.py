from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.portfolio.constraints import (
    CVXPYOptimizer,
    OptimizationResult,
    PortfolioConstraints,
    apply_weight_constraints,
)
from src.portfolio.equal_weight import select_top_scores
from src.portfolio.shrinkage import CovarianceEstimator, CovarianceResult


@dataclass(frozen=True)
class BlackLittermanPosterior:
    tickers: list[str]
    covariance_result: CovarianceResult
    market_weights: np.ndarray
    implied_returns: np.ndarray
    posterior_returns: np.ndarray
    posterior_covariance: np.ndarray
    views_q: np.ndarray
    omega_diag: np.ndarray
    tau: float
    risk_aversion: float
    score_scale: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["market_weights"] = self.market_weights.tolist()
        payload["implied_returns"] = self.implied_returns.tolist()
        payload["posterior_returns"] = self.posterior_returns.tolist()
        payload["posterior_covariance"] = self.posterior_covariance.tolist()
        payload["views_q"] = self.views_q.tolist()
        payload["omega_diag"] = self.omega_diag.tolist()
        payload["covariance_result"] = self.covariance_result.to_dict()
        return payload


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
    covariance_method: str = "ledoit_wolf",
    use_cvxpy: bool = False,
    prev_weights: dict[str, float] | pd.Series | None = None,
    sector_map: dict[str, str] | None = None,
    benchmark_sector_weights: dict[str, float] | None = None,
    stock_betas: np.ndarray | None = None,
    adv: np.ndarray | None = None,
    portfolio_size: float = 1e7,
    lambda_risk: float = 1.0,
    lambda_turnover: float = 0.005,
) -> dict[str, float]:
    active_constraints = constraints or PortfolioConstraints()
    selected = select_top_scores(scores, n_stocks=n_stocks, selection_pct=selection_pct)
    if selected.empty:
        return {}

    try:
        posterior = build_black_litterman_posterior(
            scores=selected,
            trailing_returns=trailing_returns,
            dollar_liquidity=dollar_liquidity,
            lookback_days=lookback_days,
            tau=tau,
            risk_aversion=risk_aversion,
            score_scale=score_scale,
            covariance_method=covariance_method,
        )
    except ValueError:
        raw = pd.Series(1.0, index=selected.index.astype(str), dtype=float)
        return apply_weight_constraints(raw, ranking=selected.index.astype(str), constraints=active_constraints)

    if use_cvxpy:
        optimizer = CVXPYOptimizer()
        previous = _vectorize_previous_weights(prev_weights=prev_weights, tickers=posterior.tickers)
        adv_vector = _resolve_adv_vector(
            adv=adv,
            dollar_liquidity=dollar_liquidity,
            tickers=posterior.tickers,
        )
        result = optimizer.optimize(
            expected_returns=posterior.posterior_returns,
            covariance=posterior.covariance_result.matrix,
            tickers=posterior.tickers,
            prev_weights=previous,
            sector_map=sector_map,
            benchmark_sector_weights=benchmark_sector_weights,
            stock_betas=stock_betas,
            adv=adv_vector,
            portfolio_size=portfolio_size,
            max_weight=active_constraints.max_weight,
            min_holdings=active_constraints.min_holdings,
            lambda_risk=lambda_risk,
            lambda_turnover=lambda_turnover,
        )
        if result.weights:
            return result.weights

    optimized = _solve_long_only_weights(
        posterior_returns=posterior.posterior_returns,
        covariance=posterior.covariance_result.matrix,
        market_weights=posterior.market_weights,
        max_weight=active_constraints.max_weight,
        risk_aversion=risk_aversion,
    )
    raw = pd.Series(optimized, index=posterior.tickers, dtype=float)
    return apply_weight_constraints(raw, ranking=posterior.tickers, constraints=active_constraints)


def build_black_litterman_posterior(
    *,
    scores: pd.Series,
    trailing_returns: pd.DataFrame,
    dollar_liquidity: pd.Series,
    lookback_days: int = 60,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    score_scale: float = 0.05,
    covariance_method: str = "ledoit_wolf",
) -> BlackLittermanPosterior:
    if scores.empty:
        raise ValueError("scores cannot be empty.")

    selected = pd.Series(scores, dtype=float).dropna().sort_values(ascending=False)
    tickers = selected.index.astype(str)
    history = trailing_returns.reindex(columns=tickers).tail(int(lookback_days)).copy()
    history.columns = history.columns.astype(str)
    history = history.replace([np.inf, -np.inf], np.nan)
    history = history.dropna(axis=1, thresh=max(20, int(lookback_days * 0.6)))
    history = history.dropna(axis=0, how="any")
    if history.shape[0] < 20 or history.shape[1] < 2:
        raise ValueError("Not enough clean history for Black-Litterman.")

    selected = selected.reindex(history.columns).dropna()
    liquidity = (
        pd.Series(dollar_liquidity, dtype=float)
        .reindex(history.columns.astype(str))
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    if selected.empty or len(selected) != history.shape[1]:
        raise ValueError("Selected scores do not align with the cleaned return history.")

    covariance_result = CovarianceEstimator(method=covariance_method).estimate(
        history,
        min_history=min(60, max(20, history.shape[0])),
    )
    covariance = covariance_result.matrix
    n_assets = len(covariance_result.tickers)

    if float(liquidity.sum()) <= 0.0:
        market_weights = np.full(n_assets, 1.0 / n_assets, dtype=float)
    else:
        market_weights = (liquidity.reindex(covariance_result.tickers).fillna(0.0) / float(liquidity.sum())).to_numpy(dtype=float)

    implied_returns = float(risk_aversion) * covariance.dot(market_weights)
    q = _zscore(selected.reindex(covariance_result.tickers).to_numpy(dtype=float)) * float(score_scale)
    p = np.eye(n_assets, dtype=float)
    tau_sigma = float(tau) * covariance
    view_covariance = p @ tau_sigma @ p.T
    omega_diag = np.maximum(np.diag(view_covariance), 1e-8)
    omega = np.diag(omega_diag)

    inv_tau_sigma = np.linalg.inv(tau_sigma + np.eye(n_assets) * 1e-10)
    inv_omega = np.linalg.inv(omega)
    posterior_covariance = np.linalg.inv(inv_tau_sigma + p.T @ inv_omega @ p)
    posterior_returns = posterior_covariance @ (inv_tau_sigma @ implied_returns + p.T @ inv_omega @ q)

    return BlackLittermanPosterior(
        tickers=list(covariance_result.tickers),
        covariance_result=covariance_result,
        market_weights=np.asarray(market_weights, dtype=float),
        implied_returns=np.asarray(implied_returns, dtype=float),
        posterior_returns=np.asarray(posterior_returns, dtype=float),
        posterior_covariance=np.asarray(posterior_covariance, dtype=float),
        views_q=np.asarray(q, dtype=float),
        omega_diag=np.asarray(omega_diag, dtype=float),
        tau=float(tau),
        risk_aversion=float(risk_aversion),
        score_scale=float(score_scale),
    )


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
        risk_term = 0.5 * float(risk_aversion) * float(weights.T @ covariance @ weights)
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


def _vectorize_previous_weights(
    *,
    prev_weights: dict[str, float] | pd.Series | None,
    tickers: list[str],
) -> np.ndarray | None:
    if prev_weights is None:
        return None
    series = pd.Series(prev_weights, dtype=float)
    series.index = series.index.astype(str).str.upper()
    return series.reindex(tickers).fillna(0.0).to_numpy(dtype=float)


def _liquidity_to_adv_vector(
    *,
    dollar_liquidity: pd.Series,
    tickers: list[str],
) -> np.ndarray:
    liquidity = (
        pd.Series(dollar_liquidity, dtype=float)
        .reindex(tickers)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(lower=0.0)
    )
    return liquidity.to_numpy(dtype=float)


def _resolve_adv_vector(
    *,
    adv: np.ndarray | None,
    dollar_liquidity: pd.Series,
    tickers: list[str],
) -> np.ndarray:
    if adv is None:
        return _liquidity_to_adv_vector(dollar_liquidity=dollar_liquidity, tickers=tickers)

    adv_array = np.asarray(adv, dtype=float).reshape(-1)
    if adv_array.shape[0] == len(tickers):
        return adv_array
    return _liquidity_to_adv_vector(dollar_liquidity=dollar_liquidity, tickers=tickers)


def _zscore(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std
