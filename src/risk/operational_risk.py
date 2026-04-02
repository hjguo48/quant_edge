from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RiskSeverity(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class OperationalRiskAlert:
    check_name: str
    severity: RiskSeverity
    message: str
    halt_pipeline: bool
    fail_safe_mode: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditRecord:
    timestamp_utc: str
    action: str
    actor: str
    level: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperationalRiskReport:
    timeout_alert: OperationalRiskAlert
    fail_safe_alert: OperationalRiskAlert
    audit_log: list[AuditRecord]
    overall_severity: RiskSeverity
    halt_pipeline: bool
    fail_safe_mode: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_alert": self.timeout_alert.to_dict(),
            "fail_safe_alert": self.fail_safe_alert.to_dict(),
            "audit_log": [record.to_dict() for record in self.audit_log],
            "overall_severity": self.overall_severity.value,
            "halt_pipeline": self.halt_pipeline,
            "fail_safe_mode": self.fail_safe_mode,
        }


class OperationalRiskMonitor:
    """Layer 4: Runtime and fail-safe control for the live pipeline."""

    def check_timeout(
        self,
        runtime_seconds: float,
        max_runtime_seconds: float = 3_600.0,
    ) -> OperationalRiskAlert:
        runtime = float(runtime_seconds)
        max_runtime = float(max_runtime_seconds)
        if runtime > max_runtime:
            severity = RiskSeverity.RED
            halt = True
            fail_safe = True
            message = f"Pipeline runtime {runtime:.1f}s exceeded the timeout limit {max_runtime:.1f}s."
        elif runtime > 0.8 * max_runtime:
            severity = RiskSeverity.YELLOW
            halt = False
            fail_safe = False
            message = f"Pipeline runtime {runtime:.1f}s is approaching the timeout limit {max_runtime:.1f}s."
        else:
            severity = RiskSeverity.GREEN
            halt = False
            fail_safe = False
            message = f"Pipeline runtime {runtime:.1f}s is within the allowed limit."

        return OperationalRiskAlert(
            check_name="timeout",
            severity=severity,
            message=message,
            halt_pipeline=halt,
            fail_safe_mode=fail_safe,
        )

    def check_fail_safe_mode(
        self,
        *,
        critical_alerts: list[str] | tuple[str, ...] | set[str] | bool,
        hold_positions_on_fail_safe: bool = True,
    ) -> OperationalRiskAlert:
        if isinstance(critical_alerts, bool):
            active = bool(critical_alerts)
            alert_count = int(active)
            names = ["critical_failure"] if active else []
        else:
            names = [str(item) for item in critical_alerts]
            alert_count = len(names)
            active = alert_count > 0

        if active:
            severity = RiskSeverity.RED
            halt = True
            fail_safe = True
            message = (
                f"Fail-safe mode engaged due to {alert_count} critical alerts; "
                f"hold_positions={hold_positions_on_fail_safe}."
            )
        else:
            severity = RiskSeverity.GREEN
            halt = False
            fail_safe = False
            message = "Fail-safe mode not required."

        return OperationalRiskAlert(
            check_name="fail_safe_mode",
            severity=severity,
            message=message,
            halt_pipeline=halt,
            fail_safe_mode=fail_safe,
        )

    def audit_decision(
        self,
        *,
        action: str,
        actor: str = "risk_engine",
        details: dict[str, Any] | None = None,
        level: str = "INFO",
    ) -> AuditRecord:
        return AuditRecord(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            action=str(action),
            actor=str(actor),
            level=str(level).upper(),
            details=dict(details or {}),
        )

    def run_all_checks(
        self,
        *,
        runtime_seconds: float,
        critical_alerts: list[str] | tuple[str, ...] | set[str] | bool,
        audit_events: list[AuditRecord] | None = None,
        max_runtime_seconds: float = 3_600.0,
    ) -> OperationalRiskReport:
        timeout_alert = self.check_timeout(
            runtime_seconds=runtime_seconds,
            max_runtime_seconds=max_runtime_seconds,
        )
        fail_safe_alert = self.check_fail_safe_mode(
            critical_alerts=critical_alerts,
            hold_positions_on_fail_safe=True,
        )
        audit_log = list(audit_events or [])
        audit_log.append(
            self.audit_decision(
                action="operational_risk_evaluated",
                details={
                    "runtime_seconds": float(runtime_seconds),
                    "timeout_severity": timeout_alert.severity.value,
                    "fail_safe": fail_safe_alert.fail_safe_mode,
                },
            ),
        )

        overall = max(
            [timeout_alert.severity, fail_safe_alert.severity],
            key=_severity_rank,
        )
        return OperationalRiskReport(
            timeout_alert=timeout_alert,
            fail_safe_alert=fail_safe_alert,
            audit_log=audit_log,
            overall_severity=overall,
            halt_pipeline=timeout_alert.halt_pipeline or fail_safe_alert.halt_pipeline,
            fail_safe_mode=timeout_alert.fail_safe_mode or fail_safe_alert.fail_safe_mode,
        )


def _severity_rank(severity: RiskSeverity) -> int:
    return {
        RiskSeverity.GREEN: 0,
        RiskSeverity.YELLOW: 1,
        RiskSeverity.RED: 2,
    }[severity]
