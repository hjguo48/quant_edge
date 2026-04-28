from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.api.schemas.greyscale import (
    GreyscaleGate,
    GreyscaleGateCheck,
    GreyscaleHeartbeat,
    GreyscaleHorizonCumulative,
    GreyscaleHorizonWeek,
    GreyscaleMonitorResponse,
    GreyscalePerformanceResponse,
    GreyscaleShadowDiagnostics,
    GreyscaleShadowReduceEntry,
    GreyscaleWeek,
    GreyscaleWeekSummary,
)

router = APIRouter(prefix="/api/greyscale", tags=["Greyscale"])

GREYSCALE_REPORT_DIR = Path("data/reports/greyscale")
PERFORMANCE_FILE = GREYSCALE_REPORT_DIR / "greyscale_performance.json"
HEARTBEAT_FILE = GREYSCALE_REPORT_DIR / "last_success.json"
GATE_FILE = GREYSCALE_REPORT_DIR / "g4_gate_summary.json"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_json_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


@router.get("/performance", response_model=GreyscalePerformanceResponse)
async def get_greyscale_performance() -> GreyscalePerformanceResponse:
    """W13.2 paper P&L tracker — weekly + cumulative paper return vs SPY.

    Returns 404 if compute_realized_returns.py has not been run yet.
    """
    if not PERFORMANCE_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "greyscale_performance.json not found. "
                "Run scripts/compute_realized_returns.py first."
            ),
        )
    payload = json.loads(PERFORMANCE_FILE.read_text())

    return GreyscalePerformanceResponse(
        as_of_utc=payload.get("as_of_utc"),
        today=payload.get("today"),
        benchmark=payload.get("benchmark", "SPY"),
        horizons_supported=payload.get("horizons_supported", []),
        per_week=[
            GreyscaleWeek(
                week_number=int(w.get("week_number", 0)),
                signal_date=w.get("signal_date"),
                horizons={
                    h_key: GreyscaleHorizonWeek(**h_data)
                    for h_key, h_data in (w.get("horizons") or {}).items()
                },
            )
            for w in payload.get("per_week", [])
        ],
        cumulative={
            h_key: GreyscaleHorizonCumulative(**h_data)
            for h_key, h_data in (payload.get("cumulative") or {}).items()
        },
    )


@router.get("/monitor", response_model=GreyscaleMonitorResponse)
async def get_greyscale_monitor() -> GreyscaleMonitorResponse:
    """Combined monitor view: heartbeat + weekly history + 8-week gate + Layer 3 shadow.

    Reads:
      - last_success.json (heartbeat from wrapper write_success)
      - week_*.json (weekly run reports)
      - g4_gate_summary.json (8-week eval gate state)
    """
    heartbeat_payload = _load_json_file(HEARTBEAT_FILE)
    heartbeat = (
        GreyscaleHeartbeat(
            status=heartbeat_payload.get("status"),
            bundle_version=heartbeat_payload.get("bundle_version"),
            signal_date=heartbeat_payload.get("signal_date"),
            generated_at_utc=heartbeat_payload.get("generated_at_utc"),
            layer3_enforcement_mode=heartbeat_payload.get("layer3_enforcement_mode"),
            ticker_count=_safe_int(heartbeat_payload.get("ticker_count")),
            actual_holding_count=_safe_int(heartbeat_payload.get("actual_holding_count")),
            shadow_holding_count=_safe_int(heartbeat_payload.get("shadow_holding_count")),
            shadow_cvar_triggered=heartbeat_payload.get("shadow_cvar_triggered"),
            layer1_pass=heartbeat_payload.get("layer1_pass"),
            layer2_pass=heartbeat_payload.get("layer2_pass"),
            layer3_pass=heartbeat_payload.get("layer3_pass"),
            shadow_layer3_pass=heartbeat_payload.get("shadow_layer3_pass"),
            layer4_pass=heartbeat_payload.get("layer4_pass"),
            gate_status=heartbeat_payload.get("gate_status"),
            matured_weeks=_safe_int(heartbeat_payload.get("matured_weeks")),
        )
        if heartbeat_payload
        else None
    )

    gate_payload = _load_json_file(GATE_FILE)
    gate: GreyscaleGate | None = None
    if gate_payload:
        summary = gate_payload.get("summary", {})
        checks_raw = summary.get("checks", {}) or {}
        checks: dict[str, GreyscaleGateCheck] = {}
        for name, c in checks_raw.items():
            if not isinstance(c, dict):
                continue
            threshold = c.get("threshold")
            checks[name] = GreyscaleGateCheck(
                threshold=str(threshold) if threshold is not None else None,
                value=_safe_float(c.get("value")),
                passed=c.get("passed"),
                skipped_reason=c.get("skipped_reason"),
            )
        gate = GreyscaleGate(
            gate_rule=summary.get("gate_rule"),
            gate_status=summary.get("gate_status"),
            matured_weeks=_safe_int(summary.get("matured_weeks")) or 0,
            required_weeks=_safe_int(gate_payload.get("required_weeks")) or 0,
            reports_seen=_safe_int(summary.get("reports_seen")) or 0,
            layer12_halt_count=_safe_int(summary.get("layer12_halt_count")) or 0,
            mean_live_ic=_safe_float(summary.get("mean_live_ic")),
            mean_pairwise_rank_correlation=_safe_float(
                summary.get("mean_pairwise_rank_correlation")
            ),
            mean_turnover=_safe_float(summary.get("mean_turnover")),
            positive_live_ic_weeks=_safe_int(summary.get("positive_live_ic_weeks")) or 0,
            rolling_live_ic_std=_safe_float(summary.get("rolling_live_ic_std")),
            checks=checks,
        )

    weeks: list[GreyscaleWeekSummary] = []
    if gate_payload:
        for entry in gate_payload.get("per_week", []) or []:
            if not isinstance(entry, dict):
                continue
            risk = entry.get("risk_status", {}) or {}
            realized = entry.get("realized_ics") or {}
            ic_values = [
                _safe_float(v) for v in realized.values() if _safe_float(v) is not None
            ]
            ic_mean = sum(ic_values) / len(ic_values) if ic_values else None
            weeks.append(
                GreyscaleWeekSummary(
                    week_number=_safe_int(entry.get("week_number")) or 0,
                    signal_date=entry.get("signal_date"),
                    holding_count=_safe_int(entry.get("holding_count_after_risk")),
                    turnover=_safe_float(entry.get("turnover_vs_previous")),
                    layer1_pass=risk.get("layer1_pass"),
                    layer2_pass=risk.get("layer2_pass"),
                    layer3_pass=risk.get("layer3_pass"),
                    layer4_pass=risk.get("layer4_pass"),
                    weight_source=entry.get("weight_source"),
                    realized_ic_mean=ic_mean,
                )
            )
    weeks.sort(key=lambda w: w.week_number)

    shadow: GreyscaleShadowDiagnostics | None = None
    layer1_diag: dict[str, Any] | None = None
    if weeks:
        latest_week_path = GREYSCALE_REPORT_DIR / f"week_{weeks[-1].week_number:02d}.json"
        latest_week = _load_json_file(latest_week_path)
        if latest_week:
            shadow_raw = latest_week.get("layer3_shadow_diagnostics") or {}
            if isinstance(shadow_raw, dict):
                reduce_entries = []
                for entry in shadow_raw.get("tickers_layer3_would_reduce") or []:
                    if isinstance(entry, dict) and entry.get("ticker"):
                        reduce_entries.append(
                            GreyscaleShadowReduceEntry(
                                ticker=str(entry.get("ticker")),
                                raw_weight=_safe_float(entry.get("raw_weight")),
                                shadow_weight=_safe_float(entry.get("shadow_weight")),
                            )
                        )
                shadow = GreyscaleShadowDiagnostics(
                    enforcement_mode=shadow_raw.get("enforcement_mode"),
                    shadow_holding_count=_safe_int(shadow_raw.get("shadow_holding_count")),
                    shadow_gross_exposure=_safe_float(shadow_raw.get("shadow_gross_exposure")),
                    shadow_cvar_99=_safe_float(shadow_raw.get("shadow_cvar_99")),
                    shadow_cash_weight=_safe_float(shadow_raw.get("shadow_cash_weight")),
                    shadow_turnover_vs_previous=_safe_float(
                        shadow_raw.get("shadow_turnover_vs_previous")
                    ),
                    cvar_triggered=shadow_raw.get("cvar_triggered"),
                    cvar_haircut_rounds=_safe_int(shadow_raw.get("cvar_haircut_rounds")),
                    tickers_layer3_would_remove=[
                        str(t) for t in (shadow_raw.get("tickers_layer3_would_remove") or [])
                    ],
                    tickers_layer3_would_reduce=reduce_entries,
                    warnings=[str(w) for w in (shadow_raw.get("warnings") or [])],
                    audit_trail=list(shadow_raw.get("audit_trail") or []),
                )
            layer1_raw = latest_week.get("layer1_diagnostics")
            if isinstance(layer1_raw, dict):
                layer1_diag = layer1_raw

    return GreyscaleMonitorResponse(
        heartbeat=heartbeat,
        weeks=weeks,
        gate=gate,
        shadow_diagnostics=shadow,
        layer1_diagnostics=layer1_diag,
    )
