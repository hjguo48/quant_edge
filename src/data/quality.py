from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import exp, sqrt
from typing import Any

import numpy as np
import pandas as pd


class QualityStatus(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True)
class QualityReport:
    status: QualityStatus
    message: str
    details: dict[str, Any]


class DataQualityChecker:
    """Dataframe-level quality checks used by ingestion jobs."""

    def check_missing_rate(
        self,
        df: pd.DataFrame,
        threshold_warn: float = 0.05,
        threshold_error: float = 0.15,
    ) -> QualityReport:
        if df.empty:
            return QualityReport(
                status=QualityStatus.YELLOW,
                message="Missing-rate check received an empty dataframe.",
                details={"rows": 0, "columns": list(df.columns)},
            )

        column_missing = df.isna().mean().sort_values(ascending=False)
        worst_rate = float(column_missing.iloc[0]) if not column_missing.empty else 0.0
        status = QualityStatus.GREEN
        if worst_rate >= threshold_error:
            status = QualityStatus.RED
        elif worst_rate >= threshold_warn:
            status = QualityStatus.YELLOW

        return QualityReport(
            status=status,
            message=f"Worst missing rate is {worst_rate:.2%}.",
            details={
                "rows": int(len(df)),
                "worst_missing_rate": worst_rate,
                "column_missing_rate": column_missing.to_dict(),
            },
        )

    def check_extreme_values(
        self,
        df: pd.DataFrame,
        z_threshold: float = 10,
    ) -> QualityReport:
        numeric = df.select_dtypes(include=["number"])
        if numeric.empty:
            return QualityReport(
                status=QualityStatus.YELLOW,
                message="Extreme-value check found no numeric columns.",
                details={"columns": list(df.columns)},
            )

        flagged: dict[str, dict[str, Any]] = {}
        for column in numeric.columns:
            series = numeric[column].dropna().astype(float)
            if len(series) < 3:
                continue

            std = float(series.std(ddof=0))
            if std == 0:
                continue

            z_scores = ((series - float(series.mean())) / std).abs()
            hits = series[z_scores > z_threshold]
            if not hits.empty:
                flagged[column] = {
                    "count": int(hits.shape[0]),
                    "max_abs_zscore": float(z_scores.loc[hits.index].max()),
                }

        if not flagged:
            return QualityReport(
                status=QualityStatus.GREEN,
                message="No extreme numeric values detected.",
                details={"checked_columns": list(numeric.columns)},
            )

        flagged_points = sum(item["count"] for item in flagged.values())
        total_points = int(numeric.count().sum()) or 1
        flagged_rate = flagged_points / total_points
        status = QualityStatus.RED if flagged_rate > 0.01 else QualityStatus.YELLOW

        return QualityReport(
            status=status,
            message=f"Detected {flagged_points} extreme values across numeric columns.",
            details={
                "flagged_rate": flagged_rate,
                "flagged_columns": flagged,
            },
        )

    def check_distribution_shift(
        self,
        current: pd.DataFrame | pd.Series,
        historical: pd.DataFrame | pd.Series,
        ks_p_threshold: float = 0.01,
    ) -> QualityReport:
        current_df = current.to_frame() if isinstance(current, pd.Series) else current.copy()
        historical_df = historical.to_frame() if isinstance(historical, pd.Series) else historical.copy()
        common_columns = [
            column
            for column in current_df.select_dtypes(include=["number"]).columns
            if column in historical_df.select_dtypes(include=["number"]).columns
        ]

        if not common_columns:
            return QualityReport(
                status=QualityStatus.YELLOW,
                message="Distribution-shift check found no shared numeric columns.",
                details={},
            )

        details: dict[str, dict[str, float]] = {}
        worst_p_value = 1.0
        worst_statistic = 0.0

        for column in common_columns:
            current_values = current_df[column].dropna().astype(float).to_numpy()
            historical_values = historical_df[column].dropna().astype(float).to_numpy()
            if len(current_values) < 2 or len(historical_values) < 2:
                continue

            statistic, p_value = self._ks_test(current_values, historical_values)
            details[column] = {"ks_statistic": statistic, "p_value": p_value}
            worst_p_value = min(worst_p_value, p_value)
            worst_statistic = max(worst_statistic, statistic)

        if not details:
            return QualityReport(
                status=QualityStatus.YELLOW,
                message="Distribution-shift check lacked enough overlapping observations.",
                details={},
            )

        status = QualityStatus.GREEN
        if worst_p_value < ks_p_threshold:
            status = QualityStatus.RED
        elif worst_p_value < ks_p_threshold * 5:
            status = QualityStatus.YELLOW

        return QualityReport(
            status=status,
            message=(
                f"Worst KS statistic {worst_statistic:.4f}, minimum p-value {worst_p_value:.4g}."
            ),
            details=details,
        )

    def validate_price_data(self, df: pd.DataFrame) -> QualityReport:
        required_columns = {"open", "high", "low", "close", "volume"}
        missing_columns = sorted(required_columns - set(df.columns))
        if missing_columns:
            return QualityReport(
                status=QualityStatus.RED,
                message="Price validation is missing required columns.",
                details={"missing_columns": missing_columns},
            )

        issues: dict[str, Any] = {}
        price_columns = ["open", "high", "low", "close"]

        negative_prices = int((df[price_columns] < 0).sum().sum())
        if negative_prices:
            issues["negative_prices"] = negative_prices

        zero_or_negative_volume = int((df["volume"] <= 0).sum())
        if zero_or_negative_volume:
            issues["non_positive_volume"] = zero_or_negative_volume

        open_gt_high = int((df["open"] > df["high"]).sum())
        if open_gt_high:
            issues["open_gt_high"] = open_gt_high

        low_gt_high = int((df["low"] > df["high"]).sum())
        if low_gt_high:
            issues["low_gt_high"] = low_gt_high

        close_outside_range = int(((df["close"] < df["low"]) | (df["close"] > df["high"])).sum())
        if close_outside_range:
            issues["close_outside_range"] = close_outside_range

        if not issues:
            return QualityReport(
                status=QualityStatus.GREEN,
                message="Price data passed validation checks.",
                details={},
            )

        structural_issue = any(
            key in issues
            for key in ["negative_prices", "open_gt_high", "low_gt_high", "close_outside_range"]
        )
        status = QualityStatus.RED if structural_issue else QualityStatus.YELLOW

        return QualityReport(
            status=status,
            message="Price data contains validation issues.",
            details=issues,
        )

    @staticmethod
    def _ks_test(current: np.ndarray, historical: np.ndarray) -> tuple[float, float]:
        current_sorted = np.sort(current)
        historical_sorted = np.sort(historical)
        combined = np.concatenate([current_sorted, historical_sorted])
        current_cdf = np.searchsorted(current_sorted, combined, side="right") / len(current_sorted)
        historical_cdf = (
            np.searchsorted(historical_sorted, combined, side="right") / len(historical_sorted)
        )
        statistic = float(np.max(np.abs(current_cdf - historical_cdf)))

        effective_n = sqrt((len(current_sorted) * len(historical_sorted)) / (len(current_sorted) + len(historical_sorted)))
        if effective_n == 0:
            return statistic, 1.0

        lambda_value = (effective_n + 0.12 + 0.11 / effective_n) * statistic
        p_value = 0.0
        for idx in range(1, 101):
            term = (-1) ** (idx - 1) * exp(-2 * (idx**2) * (lambda_value**2))
            p_value += 2 * term
            if abs(term) < 1e-10:
                break

        return statistic, float(min(max(p_value, 0.0), 1.0))
