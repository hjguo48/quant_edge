from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised only when cvxpy is unavailable
    cp = None


@dataclass(frozen=True)
class PortfolioConstraints:
    max_weight: float = 0.05
    min_weight: float = 0.0
    min_holdings: int = 20
    turnover_buffer: float = 0.0


@dataclass(frozen=True)
class OptimizationResult:
    weights: dict[str, float]
    expected_return: float
    expected_risk: float
    portfolio_beta: float | None
    turnover: float
    sector_weights: dict[str, float]
    solver_status: str
    objective_value: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CVXPYOptimizer:
    """CVXPY-based portfolio optimizer with long-only portfolio constraints."""

    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance: np.ndarray,
        tickers: list[str],
        *,
        prev_weights: np.ndarray | None = None,
        sector_map: dict[str, str] | None = None,
        benchmark_sector_weights: dict[str, float] | None = None,
        stock_betas: np.ndarray | None = None,
        adv: np.ndarray | None = None,
        portfolio_size: float = 1e7,
        max_weight: float = 0.10,
        max_sector_deviation: float = 0.10,
        beta_bounds: tuple[float, float] = (0.8, 1.2),
        min_holdings: int = 20,
        lambda_risk: float = 1.0,
        lambda_turnover: float = 0.005,
    ) -> OptimizationResult:
        if cp is None:
            raise RuntimeError("cvxpy is not installed. Install it before using CVXPYOptimizer.")

        mu = np.asarray(expected_returns, dtype=float).reshape(-1)
        sigma = np.asarray(covariance, dtype=float)
        names = [str(ticker).upper() for ticker in tickers]
        n_assets = len(names)

        if mu.shape[0] != n_assets:
            raise ValueError("expected_returns length must match tickers length.")
        if sigma.shape != (n_assets, n_assets):
            raise ValueError("covariance must be a square matrix matching tickers length.")
        if n_assets == 0:
            return OptimizationResult(
                weights={},
                expected_return=0.0,
                expected_risk=0.0,
                portfolio_beta=None,
                turnover=0.0,
                sector_weights={},
                solver_status="empty",
                objective_value=0.0,
            )

        sigma = 0.5 * (sigma + sigma.T)
        sigma += np.eye(n_assets) * 1e-8

        prev = (
            np.asarray(prev_weights, dtype=float).reshape(-1)
            if prev_weights is not None
            else np.zeros(n_assets, dtype=float)
        )
        if prev.shape[0] != n_assets:
            prev = np.zeros(n_assets, dtype=float)

        if adv is None:
            adv_caps = np.full(n_assets, float(max_weight), dtype=float)
        else:
            adv_array = np.asarray(adv, dtype=float).reshape(-1)
            if adv_array.shape[0] != n_assets:
                raise ValueError("adv length must match tickers length.")
            liquidity_caps = np.where(
                adv_array > 0.0,
                5.0 * adv_array / max(float(portfolio_size), 1.0),
                float(max_weight),
            )
            adv_caps = np.minimum(float(max_weight), liquidity_caps)
            adv_caps = np.clip(adv_caps, 1e-4, float(max_weight))

        weight = cp.Variable(n_assets, nonneg=True)
        objective = (
            mu @ weight
            - float(lambda_risk) * cp.quad_form(weight, sigma)
            - float(lambda_turnover) * cp.norm1(weight - prev)
        )
        constraints = [
            cp.sum(weight) == 1.0,
            weight <= adv_caps,
        ]

        if sector_map and benchmark_sector_weights:
            for sector in sorted(set(benchmark_sector_weights)):
                members = [index for index, ticker in enumerate(names) if sector_map.get(ticker) == sector]
                if not members:
                    continue
                sector_weight = cp.sum(weight[members])
                benchmark_weight = float(benchmark_sector_weights.get(sector, 0.0))
                constraints.append(sector_weight <= benchmark_weight + float(max_sector_deviation))
                constraints.append(sector_weight >= max(0.0, benchmark_weight - float(max_sector_deviation)))

        if stock_betas is not None:
            betas = np.asarray(stock_betas, dtype=float).reshape(-1)
            if betas.shape[0] != n_assets:
                raise ValueError("stock_betas length must match tickers length.")
            constraints.append(betas @ weight >= float(beta_bounds[0]))
            constraints.append(betas @ weight <= float(beta_bounds[1]))
        else:
            betas = None

        problem = cp.Problem(cp.Maximize(objective), constraints)
        status = "unknown"
        for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
            try:
                problem.solve(solver=solver, verbose=False)
            except Exception:
                continue
            status = str(problem.status)
            if status in {"optimal", "optimal_inaccurate"}:
                break

        if weight.value is None or status not in {"optimal", "optimal_inaccurate"}:
            fallback = pd.Series(np.clip(mu, 0.0, None), index=names, dtype=float)
            if float(fallback.sum()) <= 0.0:
                fallback = pd.Series(1.0, index=names, dtype=float)
            fallback = normalize_weights(fallback)
            fallback = cap_weights(fallback, max_weight=float(max_weight))
            result_weights = _ensure_minimum_holdings_posthoc(
                fallback,
                ranking=names,
                min_holdings=min_holdings,
                max_weight=float(max_weight),
            )
            result_weights = result_weights.reindex(names).fillna(0.0)
            final = result_weights.to_numpy(dtype=float)
            status = "fallback"
        else:
            raw = pd.Series(np.asarray(weight.value, dtype=float).reshape(-1), index=names, dtype=float).clip(lower=0.0)
            raw = normalize_weights(raw)
            raw = cap_weights(raw, max_weight=float(max_weight))
            raw = _ensure_minimum_holdings_posthoc(
                raw,
                ranking=[names[index] for index in np.argsort(-mu)],
                min_holdings=min_holdings,
                max_weight=float(max_weight),
            )
            final = raw.reindex(names).fillna(0.0).to_numpy(dtype=float)
            result_weights = raw

        expected_return_value = float(mu @ final)
        expected_risk_value = float(final.T @ sigma @ final)
        turnover = float(0.5 * np.abs(final - prev).sum())
        sector_weights = _compute_sector_weights(result_weights, sector_map or {})
        portfolio_beta = float(np.dot(final, betas)) if betas is not None else None
        objective_value = float(
            expected_return_value
            - float(lambda_risk) * expected_risk_value
            - float(lambda_turnover) * np.abs(final - prev).sum()
        )
        return OptimizationResult(
            weights={str(ticker): float(weight) for ticker, weight in result_weights.items() if weight > 0.0},
            expected_return=expected_return_value,
            expected_risk=expected_risk_value,
            portfolio_beta=portfolio_beta,
            turnover=turnover,
            sector_weights=sector_weights,
            solver_status=status,
            objective_value=objective_value,
        )


def apply_weight_constraints(
    weights: Mapping[str, float] | pd.Series,
    *,
    ranking: Sequence[str] | pd.Index | None = None,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, float]:
    active = constraints or PortfolioConstraints()
    series = normalize_weights(weights)
    if series.empty:
        return {}

    if active.min_weight > 0.0:
        series = series[series >= active.min_weight]
        series = normalize_weights(series)

    series = cap_weights(series, max_weight=active.max_weight)
    if active.min_holdings > 0:
        series = ensure_min_holdings(
            series,
            ranking=ranking,
            min_holdings=active.min_holdings,
            max_weight=active.max_weight,
        )
        series = cap_weights(series, max_weight=active.max_weight)

    series = normalize_weights(series)
    return {str(ticker): float(weight) for ticker, weight in series.items() if weight > 0.0}


def apply_turnover_buffer(
    target_weights: Mapping[str, float] | pd.Series,
    *,
    current_weights: Mapping[str, float] | pd.Series,
    min_trade_weight: float,
    ranking: Sequence[str] | pd.Index | None = None,
    constraints: PortfolioConstraints | None = None,
) -> dict[str, float]:
    if min_trade_weight <= 0.0:
        return apply_weight_constraints(target_weights, ranking=ranking, constraints=constraints)

    target = normalize_weights(target_weights)
    current = normalize_weights(current_weights)
    union = sorted(set(target.index) | set(current.index))
    fixed: dict[str, float] = {}
    adjustable = []

    for ticker in union:
        delta = float(target.get(ticker, 0.0) - current.get(ticker, 0.0))
        if abs(delta) < min_trade_weight:
            fixed[ticker] = float(current.get(ticker, 0.0))
        else:
            adjustable.append(ticker)

    fixed_total = float(sum(fixed.values()))
    adjustable_target = target.reindex(adjustable).fillna(0.0)
    adjustable_total = float(adjustable_target.sum())
    residual = max(0.0, 1.0 - fixed_total)

    if adjustable and adjustable_total > 0.0:
        scaled = adjustable_target * (residual / adjustable_total)
    elif adjustable and residual > 0.0:
        scaled = pd.Series(residual / len(adjustable), index=adjustable, dtype=float)
    else:
        scaled = pd.Series(dtype=float)

    blended = pd.Series(fixed, dtype=float)
    if not scaled.empty:
        blended = pd.concat([blended, scaled]).groupby(level=0).sum()

    return apply_weight_constraints(blended, ranking=ranking, constraints=constraints)


def normalize_weights(weights: Mapping[str, float] | pd.Series) -> pd.Series:
    series = pd.Series(weights, dtype=float)
    if series.empty:
        return pd.Series(dtype=float)

    series = series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    series = series[series > 0.0]
    if series.empty:
        return pd.Series(dtype=float)

    total = float(series.sum())
    if total <= 0.0:
        return pd.Series(dtype=float)
    return (series / total).sort_values(ascending=False)


def cap_weights(weights: Mapping[str, float] | pd.Series, *, max_weight: float) -> pd.Series:
    series = normalize_weights(weights)
    if series.empty:
        return series
    if max_weight <= 0.0:
        return series

    max_slots_required = int(np.ceil(1.0 / max_weight))
    if len(series) < max_slots_required:
        return pd.Series(1.0 / len(series), index=series.index, dtype=float)

    clipped = series.copy()
    tolerance = 1e-12
    while bool((clipped > max_weight + tolerance).any()):
        over = clipped[clipped > max_weight + tolerance]
        excess = float((over - max_weight).sum())
        clipped.loc[over.index] = max_weight

        under_index = clipped.index[clipped < max_weight - tolerance]
        if len(under_index) == 0 or excess <= tolerance:
            break

        base = clipped.loc[under_index]
        if float(base.sum()) <= tolerance:
            clipped.loc[under_index] += excess / len(under_index)
        else:
            clipped.loc[under_index] += excess * (base / float(base.sum()))

    return normalize_weights(clipped.clip(lower=0.0))


def ensure_min_holdings(
    weights: Mapping[str, float] | pd.Series,
    *,
    ranking: Sequence[str] | pd.Index | None,
    min_holdings: int,
    max_weight: float,
) -> pd.Series:
    series = normalize_weights(weights)
    if min_holdings <= 0 or len(series) >= min_holdings:
        return series
    if ranking is None:
        return series

    ordered = [str(item) for item in ranking]
    target_count = max(min_holdings, int(np.ceil(1.0 / max_weight)) if max_weight > 0.0 else min_holdings)
    target_names = ordered[:target_count]
    if not target_names:
        return series

    broadened = pd.Series(0.0, index=pd.Index(target_names, dtype=object), dtype=float)
    if not series.empty:
        overlap = [ticker for ticker in series.index if ticker in broadened.index]
        broadened.loc[overlap] = series.loc[overlap].astype(float)

    missing = [ticker for ticker in broadened.index if broadened.loc[ticker] <= 0.0]
    if missing:
        seed_weight = min(max_weight, 1.0 / len(broadened))
        broadened.loc[missing] = seed_weight

    return normalize_weights(broadened)


def _ensure_minimum_holdings_posthoc(
    weights: pd.Series,
    *,
    ranking: Sequence[str],
    min_holdings: int,
    max_weight: float,
) -> pd.Series:
    return ensure_min_holdings(
        weights,
        ranking=ranking,
        min_holdings=min_holdings,
        max_weight=max_weight,
    )


def _compute_sector_weights(
    weights: Mapping[str, float] | pd.Series,
    sector_map: Mapping[str, str],
) -> dict[str, float]:
    series = normalize_weights(weights)
    if series.empty:
        return {}

    sectors = pd.Series(
        [sector_map.get(str(ticker).upper(), "Unknown") for ticker in series.index],
        index=series.index,
        dtype=object,
    )
    grouped = series.groupby(sectors).sum().sort_values(ascending=False)
    return {str(sector): float(weight) for sector, weight in grouped.items()}
