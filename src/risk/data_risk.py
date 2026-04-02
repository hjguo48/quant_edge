from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


class RiskSeverity(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class DataRiskAlert:
    check_name: str
    severity: RiskSeverity
    message: str
    observed_value: float
    threshold_yellow: float
    threshold_red: float
    halt_pipeline: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureDistributionAlert:
    feature_name: str
    severity: RiskSeverity
    p_value: float
    statistic: float
    current_sample_size: int
    historical_sample_size: int
    lookback_days: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ApiHealthAlert:
    severity: RiskSeverity
    mean_response_time: float
    error_count: int
    consecutive_failures: int
    max_consecutive_failures: int
    switch_to_backup: bool
    halt_pipeline: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataRiskReport:
    missing_rate_alert: DataRiskAlert
    feature_distribution_alerts: list[FeatureDistributionAlert]
    api_health_alert: ApiHealthAlert
    overall_severity: RiskSeverity
    halt_pipeline: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "missing_rate_alert": self.missing_rate_alert.to_dict(),
            "feature_distribution_alerts": [alert.to_dict() for alert in self.feature_distribution_alerts],
            "api_health_alert": self.api_health_alert.to_dict(),
            "overall_severity": self.overall_severity.value,
            "halt_pipeline": self.halt_pipeline,
        }


class DataRiskMonitor:
    """Layer 1: Pre-research data quality risk control."""

    def check_missing_rate(
        self,
        data: pd.DataFrame,
        universe_size: int,
        threshold_yellow: float = 0.05,
        threshold_red: float = 0.15,
    ) -> DataRiskAlert:
        if universe_size <= 0:
            raise ValueError("universe_size must be positive.")

        observed_tickers = self._count_observed_tickers(data)
        missing_tickers = max(int(universe_size) - observed_tickers, 0)
        missing_rate = missing_tickers / float(universe_size)

        if missing_rate > threshold_red:
            severity = RiskSeverity.RED
            halt = True
        elif missing_rate > threshold_yellow:
            severity = RiskSeverity.YELLOW
            halt = False
        else:
            severity = RiskSeverity.GREEN
            halt = False

        message = (
            f"Observed {observed_tickers} / {universe_size} tickers; "
            f"missing_rate={missing_rate:.2%}."
        )
        return DataRiskAlert(
            check_name="missing_rate",
            severity=severity,
            message=message,
            observed_value=float(missing_rate),
            threshold_yellow=float(threshold_yellow),
            threshold_red=float(threshold_red),
            halt_pipeline=halt,
        )

    def check_feature_distribution(
        self,
        current_features: pd.DataFrame,
        historical_features: pd.DataFrame,
        lookback_days: int = 60,
        p_threshold: float = 0.01,
    ) -> list[FeatureDistributionAlert]:
        current = self._coerce_feature_frame(current_features)
        historical = self._slice_historical_features(historical_features, lookback_days=lookback_days)
        common_columns = [column for column in current.columns if column in historical.columns]

        alerts: list[FeatureDistributionAlert] = []
        for column in common_columns:
            current_values = pd.to_numeric(current[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            historical_values = (
                pd.to_numeric(historical[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(current_values) < 5 or len(historical_values) < 20:
                continue

            statistic, p_value = ks_2samp(current_values.to_numpy(), historical_values.to_numpy())
            if p_value >= p_threshold:
                continue

            alerts.append(
                FeatureDistributionAlert(
                    feature_name=str(column),
                    severity=RiskSeverity.YELLOW,
                    p_value=float(p_value),
                    statistic=float(statistic),
                    current_sample_size=int(len(current_values)),
                    historical_sample_size=int(len(historical_values)),
                    lookback_days=int(lookback_days),
                    message=(
                        f"Feature {column} drifted relative to the trailing {lookback_days}D "
                        f"distribution (KS p={p_value:.4g})."
                    ),
                ),
            )

        return sorted(alerts, key=lambda alert: alert.p_value)

    def check_api_health(
        self,
        response_times: list[float],
        error_count: int,
        consecutive_failures: int,
        max_consecutive: int = 3,
    ) -> ApiHealthAlert:
        latencies = [float(value) for value in response_times if value is not None]
        mean_latency = float(np.mean(latencies)) if latencies else 0.0
        errors = int(error_count)
        failures = int(consecutive_failures)

        if failures >= max_consecutive:
            severity = RiskSeverity.RED
            switch_to_backup = True
            halt = False
            message = (
                f"API health degraded: consecutive_failures={failures} reached "
                f"the failover threshold {max_consecutive}."
            )
        elif errors > 0 or mean_latency > 2.0:
            severity = RiskSeverity.YELLOW
            switch_to_backup = False
            halt = False
            message = (
                f"API health warning: errors={errors}, mean_response_time={mean_latency:.3f}s."
            )
        else:
            severity = RiskSeverity.GREEN
            switch_to_backup = False
            halt = False
            message = f"API health stable: mean_response_time={mean_latency:.3f}s, errors={errors}."

        return ApiHealthAlert(
            severity=severity,
            mean_response_time=mean_latency,
            error_count=errors,
            consecutive_failures=failures,
            max_consecutive_failures=int(max_consecutive),
            switch_to_backup=switch_to_backup,
            halt_pipeline=halt,
            message=message,
        )

    def run_all_checks(
        self,
        *,
        data: pd.DataFrame,
        universe_size: int,
        current_features: pd.DataFrame,
        historical_features: pd.DataFrame,
        response_times: list[float],
        error_count: int,
        consecutive_failures: int,
        missing_threshold_yellow: float = 0.05,
        missing_threshold_red: float = 0.15,
        feature_p_threshold: float = 0.01,
        feature_lookback_days: int = 60,
        max_consecutive_failures: int = 3,
    ) -> DataRiskReport:
        missing_rate_alert = self.check_missing_rate(
            data=data,
            universe_size=universe_size,
            threshold_yellow=missing_threshold_yellow,
            threshold_red=missing_threshold_red,
        )
        feature_distribution_alerts = self.check_feature_distribution(
            current_features=current_features,
            historical_features=historical_features,
            lookback_days=feature_lookback_days,
            p_threshold=feature_p_threshold,
        )
        api_health_alert = self.check_api_health(
            response_times=response_times,
            error_count=error_count,
            consecutive_failures=consecutive_failures,
            max_consecutive=max_consecutive_failures,
        )

        severities = [
            missing_rate_alert.severity,
            api_health_alert.severity,
            *[alert.severity for alert in feature_distribution_alerts],
        ]
        overall = max(severities, key=_severity_rank)
        halt_pipeline = missing_rate_alert.halt_pipeline or api_health_alert.halt_pipeline

        return DataRiskReport(
            missing_rate_alert=missing_rate_alert,
            feature_distribution_alerts=feature_distribution_alerts,
            api_health_alert=api_health_alert,
            overall_severity=overall,
            halt_pipeline=halt_pipeline,
        )

    @staticmethod
    def _count_observed_tickers(data: pd.DataFrame) -> int:
        if "ticker" in data.columns:
            return int(data["ticker"].astype(str).nunique())
        if isinstance(data.index, pd.MultiIndex) and "ticker" in data.index.names:
            return int(pd.Index(data.index.get_level_values("ticker")).astype(str).nunique())
        return int(len(pd.Index(data.index).unique()))

    @staticmethod
    def _coerce_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame.")
        return features.copy()

    @staticmethod
    def _slice_historical_features(features: pd.DataFrame, *, lookback_days: int) -> pd.DataFrame:
        frame = DataRiskMonitor._coerce_feature_frame(features)
        if not isinstance(frame.index, pd.MultiIndex) or "trade_date" not in frame.index.names:
            return frame

        trade_dates = pd.DatetimeIndex(pd.to_datetime(frame.index.get_level_values("trade_date"))).unique().sort_values()
        trailing_dates = set(trade_dates[-int(lookback_days) :])
        mask = pd.Index(pd.to_datetime(frame.index.get_level_values("trade_date"))).isin(trailing_dates)
        return frame.loc[mask]


def _severity_rank(severity: RiskSeverity) -> int:
    return {
        RiskSeverity.GREEN: 0,
        RiskSeverity.YELLOW: 1,
        RiskSeverity.RED: 2,
    }[severity]
