from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapCIResult:
    block_size: int
    n_bootstrap: int
    ci_level: float
    n_observations: int
    sharpe_estimate: float
    sharpe_ci_lower: float
    sharpe_ci_upper: float
    sharpe_p_value: float
    mean_excess_estimate: float
    mean_excess_ci_lower: float
    mean_excess_ci_upper: float
    mean_excess_p_value: float
    annualized_excess_estimate: float
    annualized_excess_ci_lower: float
    annualized_excess_ci_upper: float
    annualized_excess_p_value: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sharpe_ratio(excess_returns: pd.Series | np.ndarray, *, annualization: int = 52) -> float:
    values = _coerce_return_array(excess_returns)
    if values.size < 2:
        return float("nan")
    dispersion = float(np.std(values, ddof=1))
    if math.isclose(dispersion, 0.0, abs_tol=1e-12):
        if math.isclose(float(np.mean(values)), 0.0, abs_tol=1e-12):
            return 0.0
        return math.copysign(float("inf"), float(np.mean(values)))
    return float(np.sqrt(annualization) * np.mean(values) / dispersion)


def annualized_excess_return(
    excess_returns: pd.Series | np.ndarray,
    *,
    annualization: int = 52,
) -> float:
    values = _coerce_return_array(excess_returns)
    if values.size == 0:
        return float("nan")
    compounded = float(np.prod(1.0 + values))
    if compounded <= 0.0:
        return -1.0
    return float(compounded ** (annualization / values.size) - 1.0)


def bootstrap_return_statistics(
    excess_returns: pd.Series | np.ndarray,
    *,
    block_size: int = 4,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
    annualization: int = 52,
    seed: int = 42,
) -> BootstrapCIResult:
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive.")

    values = _coerce_return_array(excess_returns)
    n_observations = int(values.size)
    if n_observations < 2:
        raise ValueError("At least two return observations are required for bootstrap inference.")

    rng = np.random.default_rng(seed)
    sharpe_samples = np.empty(n_bootstrap, dtype=float)
    mean_samples = np.empty(n_bootstrap, dtype=float)
    annualized_samples = np.empty(n_bootstrap, dtype=float)

    for idx in range(n_bootstrap):
        sample = _draw_block_bootstrap_sample(values, block_size=block_size, rng=rng)
        sharpe_samples[idx] = sharpe_ratio(sample, annualization=annualization)
        mean_samples[idx] = float(np.mean(sample))
        annualized_samples[idx] = annualized_excess_return(sample, annualization=annualization)

    alpha_tail = (1.0 - ci_level) / 2.0
    lower_q = 100.0 * alpha_tail
    upper_q = 100.0 * (1.0 - alpha_tail)

    sharpe_estimate = sharpe_ratio(values, annualization=annualization)
    mean_excess_estimate = float(np.mean(values))
    annualized_excess_estimate = annualized_excess_return(values, annualization=annualization)

    return BootstrapCIResult(
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        n_observations=n_observations,
        sharpe_estimate=sharpe_estimate,
        sharpe_ci_lower=float(np.nanpercentile(sharpe_samples, lower_q)),
        sharpe_ci_upper=float(np.nanpercentile(sharpe_samples, upper_q)),
        sharpe_p_value=float(np.mean(sharpe_samples <= 0.0)),
        mean_excess_estimate=mean_excess_estimate,
        mean_excess_ci_lower=float(np.nanpercentile(mean_samples, lower_q)),
        mean_excess_ci_upper=float(np.nanpercentile(mean_samples, upper_q)),
        mean_excess_p_value=float(np.mean(mean_samples <= 0.0)),
        annualized_excess_estimate=annualized_excess_estimate,
        annualized_excess_ci_lower=float(np.nanpercentile(annualized_samples, lower_q)),
        annualized_excess_ci_upper=float(np.nanpercentile(annualized_samples, upper_q)),
        annualized_excess_p_value=float(np.mean(annualized_samples <= 0.0)),
    )


def _coerce_return_array(excess_returns: pd.Series | np.ndarray) -> np.ndarray:
    if isinstance(excess_returns, pd.Series):
        values = excess_returns.dropna().to_numpy(dtype=float)
    else:
        values = np.asarray(excess_returns, dtype=float)
        values = values[np.isfinite(values)]
    return values.astype(float, copy=False)


def _draw_block_bootstrap_sample(
    values: np.ndarray,
    *,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = values.size
    indices: list[np.ndarray] = []
    blocks_needed = int(math.ceil(n / block_size))

    for _ in range(blocks_needed):
        start = int(rng.integers(0, n))
        block_index = (start + np.arange(block_size)) % n
        indices.append(block_index)

    sampled_index = np.concatenate(indices)[:n]
    return values[sampled_index]
