from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np
import pandas as pd
from scipy import stats

from src.models.experiment import ExperimentTracker
from src.stats.bootstrap import sharpe_ratio


PERFORMANCE_METRIC_PRIORITY = (
    "annualized_net_excess",
    "annualized_gross_excess",
    "mean_ic",
    "test_ic",
    "validation_ic",
)


@dataclass(frozen=True)
class DeflatedSharpeResult:
    method: str
    observed_sharpe: float
    dsr_stat: float
    p_value: float
    raw_p_value: float
    significant: bool
    n_observations: int
    total_trials: int
    independent_trials: int
    usable_trial_metrics: int
    sigma_sr: float
    e_max_sr: float
    se_sharpe: float
    sample_skew: float
    sample_kurtosis: float
    family_counts: dict[str, int]
    performance_metric_priority: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_deflated_sharpe(
    excess_returns: pd.Series,
    *,
    tracking_uri: str,
    max_results: int = 5_000,
    annualization: int = 52,
    alpha: float = 0.05,
) -> DeflatedSharpeResult:
    returns = pd.Series(excess_returns, dtype=float).dropna()
    if len(returns) < 2:
        raise ValueError("At least two return observations are required for DSR.")

    tracker = ExperimentTracker(tracking_uri=tracking_uri)
    runs = tracker.search_runs(max_results=max_results)

    family_counts: Counter[str] = Counter()
    performance_values: list[float] = []

    for run in runs:
        family = classify_trial_family(run.data.tags)
        family_counts[family] += 1

        performance_value = extract_trial_performance(run.data.metrics)
        if performance_value is not None and math.isfinite(performance_value):
            performance_values.append(float(performance_value))

    observed_sharpe = sharpe_ratio(returns, annualization=annualization)
    sample_skew = float(stats.skew(returns.to_numpy(dtype=float), bias=False))
    sample_kurtosis = float(stats.kurtosis(returns.to_numpy(dtype=float), fisher=False, bias=False))
    se_sharpe = sharpe_standard_error(
        observed_sharpe=observed_sharpe,
        sample_skew=sample_skew,
        sample_kurtosis=sample_kurtosis,
        n_observations=len(returns),
    )

    total_trials = len(runs)
    independent_trials = max(len(family_counts), 1)
    sigma_sr = float(np.std(performance_values, ddof=1)) if len(performance_values) > 1 else 0.0
    e_max_sr = expected_max_sharpe(independent_trials=independent_trials, sigma_sr=sigma_sr)

    if math.isnan(observed_sharpe) or math.isnan(se_sharpe) or math.isclose(se_sharpe, 0.0, abs_tol=1e-12):
        dsr_stat = float("nan")
        raw_p_value = float("nan")
        p_value = float("nan")
        significant = False
    else:
        dsr_stat = float((observed_sharpe - e_max_sr) / se_sharpe)
        raw_p_value = float(stats.norm.sf(dsr_stat))
        p_value = float(min(1.0, raw_p_value * max(total_trials, 1)))
        significant = bool(p_value < alpha)

    return DeflatedSharpeResult(
        method="simplified_dsr_bonferroni",
        observed_sharpe=observed_sharpe,
        dsr_stat=dsr_stat,
        p_value=p_value,
        raw_p_value=raw_p_value,
        significant=significant,
        n_observations=int(len(returns)),
        total_trials=total_trials,
        independent_trials=independent_trials,
        usable_trial_metrics=len(performance_values),
        sigma_sr=sigma_sr,
        e_max_sr=e_max_sr,
        se_sharpe=se_sharpe,
        sample_skew=sample_skew,
        sample_kurtosis=sample_kurtosis,
        family_counts=dict(sorted(family_counts.items())),
        performance_metric_priority=PERFORMANCE_METRIC_PRIORITY,
    )


def classify_trial_family(tags: Mapping[str, str]) -> str:
    if "scheme" in tags:
        return f"portfolio:{tags['scheme']}"

    model_type = tags.get("model_type")
    horizon = tags.get("horizon") or tags.get("target_horizon")
    if model_type or horizon:
        return f"model:{model_type or '?'}:{horizon or '?'}"

    run_kind = tags.get("run_kind", "unknown")
    return f"other:{run_kind}"


def extract_trial_performance(metrics: Mapping[str, float]) -> float | None:
    for key in PERFORMANCE_METRIC_PRIORITY:
        value = metrics.get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def expected_max_sharpe(*, independent_trials: int, sigma_sr: float) -> float:
    if independent_trials <= 1 or not math.isfinite(sigma_sr) or math.isclose(sigma_sr, 0.0, abs_tol=1e-12):
        return 0.0
    return float(math.sqrt(2.0 * math.log(independent_trials)) * sigma_sr)


def sharpe_standard_error(
    *,
    observed_sharpe: float,
    sample_skew: float,
    sample_kurtosis: float,
    n_observations: int,
) -> float:
    if n_observations < 2 or not math.isfinite(observed_sharpe):
        return float("nan")

    variance = (
        1.0
        - (sample_skew * observed_sharpe)
        + (((sample_kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe)
    ) / (n_observations - 1)
    if variance <= 0.0 or not math.isfinite(variance):
        variance = 1.0 / (n_observations - 1)
    return float(math.sqrt(variance))
