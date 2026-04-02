from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ICTestResult:
    t_statistic: float
    p_value: float
    mean_ic: float
    std_ic: float
    n_observations: int
    alternative: str
    significant: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


def run_ic_ttest(
    ic_series: pd.Series,
    *,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> ICTestResult:
    values = pd.Series(ic_series, dtype=float).dropna()
    n_observations = int(len(values))
    mean_ic = float(values.mean()) if n_observations else float("nan")
    std_ic = float(values.std(ddof=1)) if n_observations > 1 else float("nan")

    if alternative not in {"greater", "less"}:
        raise ValueError("alternative must be either 'greater' or 'less'.")

    if n_observations == 0:
        return ICTestResult(
            t_statistic=float("nan"),
            p_value=float("nan"),
            mean_ic=float("nan"),
            std_ic=float("nan"),
            n_observations=0,
            alternative=alternative,
            significant=False,
        )

    if n_observations == 1:
        return ICTestResult(
            t_statistic=float("nan"),
            p_value=1.0,
            mean_ic=mean_ic,
            std_ic=float("nan"),
            n_observations=1,
            alternative=alternative,
            significant=False,
        )

    if math.isclose(float(values.std(ddof=0)), 0.0, abs_tol=1e-12):
        if math.isclose(mean_ic, 0.0, abs_tol=1e-12):
            t_statistic = 0.0
            p_value = 1.0
        else:
            positive = mean_ic > 0.0
            if (alternative == "greater" and positive) or (alternative == "less" and not positive):
                t_statistic = math.copysign(float("inf"), mean_ic)
                p_value = 0.0
            else:
                t_statistic = math.copysign(float("inf"), mean_ic)
                p_value = 1.0
    else:
        test_result = stats.ttest_1samp(values.to_numpy(dtype=float), popmean=0.0, nan_policy="omit")
        t_statistic = float(test_result.statistic)
        p_value = _one_sided_p_value(
            t_statistic=t_statistic,
            two_sided_p=float(test_result.pvalue),
            alternative=alternative,
        )

    return ICTestResult(
        t_statistic=t_statistic,
        p_value=p_value,
        mean_ic=mean_ic,
        std_ic=std_ic,
        n_observations=n_observations,
        alternative=alternative,
        significant=bool(p_value < alpha),
    )


def run_windowed_ic_tests(
    ic_series_by_window: Mapping[str, pd.Series],
    *,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> dict[str, ICTestResult]:
    return {
        window_id: run_ic_ttest(series, alternative=alternative, alpha=alpha)
        for window_id, series in ic_series_by_window.items()
    }


def _one_sided_p_value(*, t_statistic: float, two_sided_p: float, alternative: str) -> float:
    if math.isnan(t_statistic) or math.isnan(two_sided_p):
        return float("nan")
    if alternative == "greater":
        return float(two_sided_p / 2.0) if t_statistic >= 0.0 else float(1.0 - (two_sided_p / 2.0))
    return float(two_sided_p / 2.0) if t_statistic <= 0.0 else float(1.0 - (two_sided_p / 2.0))
