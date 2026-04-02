from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class RiskSeverity(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class SignalRiskAlert:
    severity: RiskSeverity
    rolling_mean_ic: float
    historical_mean_ic: float
    threshold_ratio: float
    consecutive_breaches: int
    consecutive_limit: int
    switch_model: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CalibrationAlert:
    severity: RiskSeverity
    spearman: float
    monotonic: bool
    n_bins: int
    bin_summary: list[dict[str, float]]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelSwitchAlert:
    severity: RiskSeverity
    champion_ic: float
    challenger_ic: float
    consecutive_challenger_wins: int
    required_wins: int
    recommend_switch: bool
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalRiskReport:
    rolling_ic_alert: SignalRiskAlert
    calibration_alert: CalibrationAlert
    model_switch_alert: ModelSwitchAlert
    overall_severity: RiskSeverity
    recommend_switch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rolling_ic_alert": self.rolling_ic_alert.to_dict(),
            "calibration_alert": self.calibration_alert.to_dict(),
            "model_switch_alert": self.model_switch_alert.to_dict(),
            "overall_severity": self.overall_severity.value,
            "recommend_switch": self.recommend_switch,
        }


class SignalRiskMonitor:
    """Layer 2: Signal quality monitoring."""

    def check_rolling_ic(
        self,
        ic_history: list[float],
        lookback: int = 20,
        threshold_ratio: float = 0.5,
        consecutive_limit: int = 4,
    ) -> SignalRiskAlert:
        history = pd.Series(ic_history, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        if history.empty:
            return SignalRiskAlert(
                severity=RiskSeverity.RED,
                rolling_mean_ic=float("nan"),
                historical_mean_ic=float("nan"),
                threshold_ratio=float(threshold_ratio),
                consecutive_breaches=0,
                consecutive_limit=int(consecutive_limit),
                switch_model=True,
                message="IC history is empty; signal quality cannot be evaluated.",
            )

        rolling_mean = float(history.tail(lookback).mean())
        historical_mean = float(history.mean())
        threshold = historical_mean * float(threshold_ratio) if historical_mean > 0.0 else 0.0
        consecutive_breaches = 0
        for value in reversed(history.tolist()):
            if float(value) < threshold:
                consecutive_breaches += 1
            else:
                break

        switch_model = consecutive_breaches >= int(consecutive_limit)
        if switch_model:
            severity = RiskSeverity.RED
        elif rolling_mean < threshold:
            severity = RiskSeverity.YELLOW
        else:
            severity = RiskSeverity.GREEN

        message = (
            f"rolling_ic={rolling_mean:.4f}, historical_mean={historical_mean:.4f}, "
            f"consecutive_breaches={consecutive_breaches}."
        )
        return SignalRiskAlert(
            severity=severity,
            rolling_mean_ic=rolling_mean,
            historical_mean_ic=historical_mean,
            threshold_ratio=float(threshold_ratio),
            consecutive_breaches=int(consecutive_breaches),
            consecutive_limit=int(consecutive_limit),
            switch_model=switch_model,
            message=message,
        )

    def check_calibration(
        self,
        predicted_scores: pd.Series,
        realized_returns: pd.Series,
        n_bins: int = 10,
    ) -> CalibrationAlert:
        frame = pd.concat(
            [
                pd.Series(predicted_scores, name="score", dtype=float),
                pd.Series(realized_returns, name="realized", dtype=float),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            return CalibrationAlert(
                severity=RiskSeverity.RED,
                spearman=float("nan"),
                monotonic=False,
                n_bins=int(n_bins),
                bin_summary=[],
                message="No aligned predictions and realized returns were available for calibration.",
            )

        effective_bins = max(1, min(int(n_bins), len(frame)))
        ranked = frame["score"].rank(method="first")
        frame["bin"] = pd.qcut(ranked, q=effective_bins, labels=False, duplicates="drop")
        grouped = (
            frame.groupby("bin", observed=True)
            .agg(
                mean_score=("score", "mean"),
                mean_realized=("realized", "mean"),
                count=("score", "size"),
            )
            .sort_index()
        )
        monotonic = bool(grouped["mean_realized"].is_monotonic_increasing)
        if len(grouped) >= 2:
            correlation = spearmanr(grouped["mean_score"], grouped["mean_realized"], nan_policy="omit").statistic
        else:
            correlation = np.nan

        if np.isnan(correlation) or correlation < 0.0:
            severity = RiskSeverity.RED
        elif correlation < 0.3 or not monotonic:
            severity = RiskSeverity.YELLOW
        else:
            severity = RiskSeverity.GREEN

        return CalibrationAlert(
            severity=severity,
            spearman=float(correlation) if not np.isnan(correlation) else float("nan"),
            monotonic=monotonic,
            n_bins=int(len(grouped)),
            bin_summary=[
                {
                    "bin": int(index),
                    "mean_score": float(row["mean_score"]),
                    "mean_realized": float(row["mean_realized"]),
                    "count": float(row["count"]),
                }
                for index, row in grouped.iterrows()
            ],
            message=(
                f"Calibration check completed with spearman={correlation:.4f} "
                f"and monotonic={monotonic}."
            ),
        )

    def check_champion_vs_challenger(
        self,
        champion_ic: float,
        challenger_ic: float,
        consecutive_challenger_wins: int,
        required_wins: int = 4,
    ) -> ModelSwitchAlert:
        recommend = float(challenger_ic) > float(champion_ic) and int(consecutive_challenger_wins) >= int(required_wins)
        if recommend:
            severity = RiskSeverity.YELLOW
            message = (
                f"Challenger IC {challenger_ic:.4f} exceeded champion IC {champion_ic:.4f} "
                f"for {consecutive_challenger_wins} consecutive periods."
            )
        else:
            severity = RiskSeverity.GREEN
            message = (
                f"Champion remains active: challenger_wins={consecutive_challenger_wins}, "
                f"required={required_wins}."
            )

        return ModelSwitchAlert(
            severity=severity,
            champion_ic=float(champion_ic),
            challenger_ic=float(challenger_ic),
            consecutive_challenger_wins=int(consecutive_challenger_wins),
            required_wins=int(required_wins),
            recommend_switch=recommend,
            message=message,
        )

    def run_all_checks(
        self,
        *,
        ic_history: list[float],
        predicted_scores: pd.Series,
        realized_returns: pd.Series,
        champion_ic: float,
        challenger_ic: float,
        consecutive_challenger_wins: int,
        lookback: int = 20,
        threshold_ratio: float = 0.5,
        consecutive_limit: int = 4,
        n_bins: int = 10,
        required_wins: int = 4,
    ) -> SignalRiskReport:
        rolling_ic_alert = self.check_rolling_ic(
            ic_history=ic_history,
            lookback=lookback,
            threshold_ratio=threshold_ratio,
            consecutive_limit=consecutive_limit,
        )
        calibration_alert = self.check_calibration(
            predicted_scores=predicted_scores,
            realized_returns=realized_returns,
            n_bins=n_bins,
        )
        model_switch_alert = self.check_champion_vs_challenger(
            champion_ic=champion_ic,
            challenger_ic=challenger_ic,
            consecutive_challenger_wins=consecutive_challenger_wins,
            required_wins=required_wins,
        )

        overall = max(
            [
                rolling_ic_alert.severity,
                calibration_alert.severity,
                model_switch_alert.severity,
            ],
            key=_severity_rank,
        )
        return SignalRiskReport(
            rolling_ic_alert=rolling_ic_alert,
            calibration_alert=calibration_alert,
            model_switch_alert=model_switch_alert,
            overall_severity=overall,
            recommend_switch=rolling_ic_alert.switch_model or model_switch_alert.recommend_switch,
        )


def _severity_rank(severity: RiskSeverity) -> int:
    return {
        RiskSeverity.GREEN: 0,
        RiskSeverity.YELLOW: 1,
        RiskSeverity.RED: 2,
    }[severity]
