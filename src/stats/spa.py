from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


@dataclass(frozen=True)
class SPAComparisonResult:
    competitor: str
    benchmark_mean_ic: float
    competitor_mean_ic: float
    mean_ic_delta: float
    n_observations: int
    t_statistic: float
    p_value: float
    adjusted_p_value: float
    significant: bool

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class SPATestResult:
    method: str
    null_hypothesis: str
    p_value: float
    significant: bool
    benchmark: str
    comparisons: list[SPAComparisonResult]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["comparisons"] = [comparison.to_dict() for comparison in self.comparisons]
        return payload


def series_from_records(
    records: list[dict[str, object]],
    *,
    date_key: str = "trade_date",
    value_key: str = "value",
) -> pd.Series:
    if not records:
        return pd.Series(dtype=float, name=value_key)

    frame = pd.DataFrame(records)
    series = pd.Series(
        data=pd.to_numeric(frame[value_key], errors="coerce").to_numpy(dtype=float),
        index=pd.to_datetime(frame[date_key]),
        name=value_key,
        dtype=float,
    )
    return series.dropna().sort_index()


def run_spa_fallback(
    benchmark_series: pd.Series,
    competitors: Mapping[str, pd.Series],
    *,
    benchmark_name: str = "ridge",
    alpha: float = 0.05,
) -> SPATestResult:
    raw_results: list[dict[str, float | str | int]] = []
    p_values: list[float] = []

    for competitor_name, competitor_series in competitors.items():
        aligned = pd.concat(
            [
                pd.Series(benchmark_series, dtype=float).rename("benchmark"),
                pd.Series(competitor_series, dtype=float).rename("competitor"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        delta = aligned["competitor"] - aligned["benchmark"]

        if len(delta) < 2:
            raw_results.append(
                {
                    "competitor": competitor_name,
                    "benchmark_mean_ic": float(aligned["benchmark"].mean()) if not aligned.empty else float("nan"),
                    "competitor_mean_ic": float(aligned["competitor"].mean()) if not aligned.empty else float("nan"),
                    "mean_ic_delta": float(delta.mean()) if not delta.empty else float("nan"),
                    "n_observations": int(len(delta)),
                    "t_statistic": float("nan"),
                    "p_value": float("nan"),
                }
            )
            continue

        if math.isclose(float(delta.std(ddof=0)), 0.0, abs_tol=1e-12):
            mean_delta = float(delta.mean())
            if mean_delta > 0.0:
                t_statistic = float("inf")
                p_value = 0.0
            else:
                t_statistic = 0.0 if math.isclose(mean_delta, 0.0, abs_tol=1e-12) else float("-inf")
                p_value = 1.0
        else:
            test_result = stats.ttest_1samp(delta.to_numpy(dtype=float), popmean=0.0, nan_policy="omit")
            t_statistic = float(test_result.statistic)
            p_value = _one_sided_greater_p_value(t_statistic=t_statistic, two_sided_p=float(test_result.pvalue))

        raw_results.append(
            {
                "competitor": competitor_name,
                "benchmark_mean_ic": float(aligned["benchmark"].mean()),
                "competitor_mean_ic": float(aligned["competitor"].mean()),
                "mean_ic_delta": float(delta.mean()),
                "n_observations": int(len(delta)),
                "t_statistic": t_statistic,
                "p_value": p_value,
            }
        )
        p_values.append(p_value)

    adjusted_map: dict[str, float] = {}
    significant_map: dict[str, bool] = {}
    valid_entries = [row for row in raw_results if math.isfinite(float(row["p_value"]))]
    if valid_entries:
        reject, adjusted, _, _ = multipletests(
            [float(row["p_value"]) for row in valid_entries],
            alpha=alpha,
            method="holm",
        )
        for row, adjusted_p, is_rejected in zip(valid_entries, adjusted, reject, strict=True):
            adjusted_map[str(row["competitor"])] = float(adjusted_p)
            significant_map[str(row["competitor"])] = bool(is_rejected)

    comparisons = [
        SPAComparisonResult(
            competitor=str(row["competitor"]),
            benchmark_mean_ic=float(row["benchmark_mean_ic"]),
            competitor_mean_ic=float(row["competitor_mean_ic"]),
            mean_ic_delta=float(row["mean_ic_delta"]),
            n_observations=int(row["n_observations"]),
            t_statistic=float(row["t_statistic"]),
            p_value=float(row["p_value"]),
            adjusted_p_value=adjusted_map.get(str(row["competitor"]), float("nan")),
            significant=significant_map.get(str(row["competitor"]), False),
        )
        for row in raw_results
    ]

    overall_p_value = min(
        (comparison.adjusted_p_value for comparison in comparisons if math.isfinite(comparison.adjusted_p_value)),
        default=float("nan"),
    )
    return SPATestResult(
        method="paired_t_test_holm_fallback",
        null_hypothesis=f"{benchmark_name} is not worse than any competitor on aligned test-period IC.",
        p_value=overall_p_value,
        significant=bool(math.isfinite(overall_p_value) and overall_p_value < alpha),
        benchmark=benchmark_name,
        comparisons=comparisons,
    )


def _one_sided_greater_p_value(*, t_statistic: float, two_sided_p: float) -> float:
    if math.isnan(t_statistic) or math.isnan(two_sided_p):
        return float("nan")
    return float(two_sided_p / 2.0) if t_statistic >= 0.0 else float(1.0 - (two_sided_p / 2.0))
