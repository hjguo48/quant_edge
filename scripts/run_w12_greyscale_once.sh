#!/usr/bin/env bash
# W12 weekly greyscale wrapper — one-shot, idempotent.
#
# Usage (manual or scheduler):
#   /home/jiahao/quant_edge/scripts/run_w12_greyscale_once.sh
#
# Behavior:
#   - Acquires lock to prevent overlap
#   - Preflight: DB freshness + bundle existence (12 retries × 10 min)
#   - Runs scripts/run_greyscale_live.py (full universe, --dry-run paper)
#   - Runs scripts/run_greyscale_monitor.py (gate evaluation)
#   - Writes data/reports/greyscale/last_success.json or last_failure.json
#   - Sends email alert on infrastructure failure or red strategy state
#
# Exit codes:
#   0 — completed cleanly, all green
#   1 — unexpected shell error
#   2 — preflight failed (data not ready after 120 minutes)
#   3 — greyscale_live or monitor failed
#   10 — completed but red layer/gate
#   99 — another run is already active

set -Eeuo pipefail

REPO_ROOT="/home/jiahao/quant_edge"
cd "$REPO_ROOT"

source .venv/bin/activate

BUNDLE="data/models/bundles/w12_60d_ridge_swbuf_v3/bundle.json"
REF_FM="data/features/walkforward_v9full9y_fm_60d.parquet"
REPORT_DIR="data/reports/greyscale"
LOG_DIR="$REPORT_DIR/logs"
LOCK_FILE="$REPORT_DIR/.w12_greyscale.lock"
LAST_SUCCESS="$REPORT_DIR/last_success.json"
LAST_FAILURE="$REPORT_DIR/last_failure.json"
GATE_SUMMARY="$REPORT_DIR/g4_gate_summary.json"

mkdir -p "$REPORT_DIR" "$LOG_DIR"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="$LOG_DIR/greyscale_${RUN_ID}.log"

# Tee all output to log and stdout
exec > >(tee -a "$LOG_FILE") 2>&1

echo "==================================================================="
echo "W12 greyscale wrapper start: ${RUN_ID}"
echo "==================================================================="

# Acquire lock
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "ERROR: another greyscale run is active"; exit 99; }

# ----------- Helpers -----------

write_failure() {
    local stage="$1"
    local message="$2"
    python - <<PY
import json
from datetime import datetime, timezone
from src.utils.io import write_json_atomic

payload = {
    "status": "failure",
    "stage": "$stage",
    "message": """$message""",
    "bundle_version": "w12_60d_ridge_swbuf_v3",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "log_file": "$LOG_FILE",
    "run_id": "$RUN_ID",
}
write_json_atomic("$LAST_FAILURE", payload)
PY
}

write_success() {
    python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
from src.utils.io import write_json_atomic

report_dir = Path("data/reports/greyscale")
weeks = sorted(report_dir.glob("week_*.json"))
latest = json.loads(weeks[-1].read_text()) if weeks else {}
gate_path = report_dir / "g4_gate_summary.json"
gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}

# Extract key fields safely
risk = latest.get("risk_checks", {}) or {}
score_vectors = latest.get("score_vectors", {}) or {}
fusion_scores = score_vectors.get("fusion") or score_vectors.get("FUSION") or {}
shadow = latest.get("layer3_shadow_diagnostics", {}) or {}
enforcement_mode = shadow.get("enforcement_mode")

# W13.x — when Layer 3 enforcement is OFF (shadow mode), Layer 3's
# overall_pass is informational and must not gate the wrapper. Surface
# layer3_pass as None so the red-layer check skips it; keep the raw
# pass value under shadow_layer3_pass for audit.
shadow_layer3_pass = risk.get("layer3_portfolio", {}).get("pass")
if enforcement_mode is False:
    layer3_pass_for_alert = None
else:
    layer3_pass_for_alert = shadow_layer3_pass

payload = {
    "status": "success",
    "bundle_version": "w12_60d_ridge_swbuf_v3",
    "layer3_enforcement_mode": enforcement_mode,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "latest_report_path": str(weeks[-1]) if weeks else None,
    "signal_date": latest.get("live_outputs", {}).get("signal_date"),
    "ticker_count": int(len(fusion_scores) if isinstance(fusion_scores, dict) else 0),
    "actual_holding_count": int(latest.get("portfolio_metrics", {}).get("holding_count_after_risk", 0) or 0),
    "shadow_holding_count": shadow.get("shadow_holding_count"),
    "shadow_cvar_triggered": shadow.get("cvar_triggered"),
    "layer1_pass": risk.get("layer1_data", {}).get("pass"),
    "layer2_pass": risk.get("layer2_signal", {}).get("pass"),
    "layer3_pass": layer3_pass_for_alert,
    "shadow_layer3_pass": shadow_layer3_pass,
    "layer4_pass": risk.get("layer4_operational", {}).get("pass"),
    "gate_status": gate.get("summary", {}).get("gate_status"),
    "matured_weeks": gate.get("summary", {}).get("matured_weeks"),
}
write_json_atomic("data/reports/greyscale/last_success.json", payload)
print(f"heartbeat written: signal_date={payload['signal_date']}, "
      f"tickers={payload['ticker_count']}, "
      f"layers=[{payload['layer1_pass']},{payload['layer2_pass']},{payload['layer3_pass']},{payload['layer4_pass']}]")
PY
}

send_alert() {
    local severity="$1"
    local subject="$2"
    local body="$3"
    if [ -f "$HOME/.config/quantedge/w12_alert.env" ]; then
        # 90-second hard timeout — WSL2 IPv6 can transiently flake, the SNI
        # fallback in send_w12_email_alert.py usually recovers within 30s but
        # occasionally hangs longer. Better to log "alert send timeout" than
        # block the wrapper indefinitely on a non-critical email.
        timeout 90 python "$REPO_ROOT/scripts/send_w12_email_alert.py" \
            --severity "$severity" \
            --subject "$subject" \
            --body "$body" || echo "WARNING: alert send failed (or timed out at 90s)"
    else
        echo "WARNING: alert config not found, skipping email"
    fi
}

preflight_once() {
    python - <<'PY'
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

# Bundle exists
bundle_path = Path("data/models/bundles/w12_60d_ridge_swbuf_v3/bundle.json")
if not bundle_path.exists():
    print(f"FAIL: bundle missing: {bundle_path}")
    sys.exit(2)

# DB reachable + PIT trade date freshness vs expected market session
sys.path.insert(0, ".")
try:
    from scripts.run_live_pipeline import load_db_state
    as_of = datetime.now(timezone.utc)
    db = load_db_state(as_of=as_of)
    latest = db["latest_pit_trade_date"]
    if latest is None:
        print("FAIL: no PIT trade date in stock_prices")
        sys.exit(2)

    # Expected latest visible session: account for T+1 PIT knowledge_time lag.
    # At time T, latest fully-published session should be:
    # - if today is a trading session and now is after T+1 publish time → today's previous session (today's data not yet visible)
    # - otherwise → most recent past session
    XNYS = xcals.get_calendar("XNYS")
    ET = ZoneInfo("America/New_York")
    today_et = as_of.astimezone(ET).date()
    today_ts = pd.Timestamp(today_et)
    # Always use the most recent past session as the conservative expected
    # (T+1 lag means today's close not yet PIT-visible, plus today_et may itself
    # be a non-trading day like Saturday 5/2 — previous_session() raises
    # NotSessionError on non-session input, so use date_to_session(direction='previous')
    # which accepts any calendar date).
    if XNYS.is_session(today_ts):
        expected_latest = XNYS.previous_session(today_ts).date()
    else:
        expected_latest = XNYS.date_to_session(today_ts, direction="previous").date()
    # Allow 1 session tolerance for upstream batch delays
    tolerance_session = XNYS.previous_session(pd.Timestamp(expected_latest)).date()

    if latest < tolerance_session:
        sessions_behind = len(XNYS.sessions_in_range(pd.Timestamp(latest), pd.Timestamp(expected_latest))) - 1
        print(f"FAIL: PIT trade date {latest} is {sessions_behind} sessions behind expected {expected_latest} (tolerance: {tolerance_session})")
        sys.exit(2)
    print(f"OK: PIT trade date={latest}, expected={expected_latest}, tolerance={tolerance_session}")

    # W12 audit fix: validate per-feature recency in feature_store
    # (catches silent feature dropout — e.g. shorting backfill ran once and never refreshed)
    from src.data.db.session import get_engine
    from src.models.bundle_validator import BundleValidator
    bv = BundleValidator(bundle_path)
    engine = get_engine()
    with engine.connect() as conn:
        rec = bv.validate_recency(conn, max_stale_days=7, min_coverage_count=100)
    if not rec.passed:
        if rec.stale_features:
            print(f"FAIL: {len(rec.stale_features)} features stale (>7d): {rec.stale_features[:10]}")
        if rec.sparse_features:
            print(f"FAIL: {len(rec.sparse_features)} features sparse (<100 tickers): {rec.sparse_features[:10]}")
        sys.exit(2)
    print(f"OK: feature_store recency: {rec.metadata['feature_count_with_data']}/{rec.metadata['feature_count_checked']} features fresh")
    sys.exit(0)
except SystemExit:
    raise
except Exception as exc:
    print(f"FAIL: preflight exception: {exc}")
    sys.exit(2)
PY
}

# ----------- Main flow -----------

trap 'write_failure "shell_error" "unexpected shell error in wrapper"; send_alert "RED" "[QuantEdge W12] wrapper shell error" "Unexpected shell error. Log: '"$LOG_FILE"'"; exit 1' ERR

# W13.x — Targeted self-heal for shorting features only.
# The daily DAG's update_features_cache task has been failing intermittently
# (KeyError: slice_summaries on 4/27); meanwhile feature_store.short_sale_ratio_5d
# can drift past the BundleValidator's 7-day staleness threshold.
# Codex 2026-04-28 review: keep the self-heal NARROW to the shorting family;
# any other stale feature should still fail-loud and trigger a RED email so
# we don't silently mask broader pipeline regressions.
shorting_self_heal() {
    .venv/bin/python - <<'PY'
import sys
sys.path.insert(0, ".")
from datetime import date, timedelta
from sqlalchemy import text
from src.data.db.session import get_engine

SHORTING_FAMILY = {
    "short_sale_ratio_1d", "short_sale_ratio_5d", "short_sale_accel",
    "abnormal_off_exchange_shorting",
    "is_missing_short_sale_ratio_1d", "is_missing_short_sale_ratio_5d",
    "is_missing_short_sale_accel", "is_missing_abnormal_off_exchange_shorting",
}
THRESHOLD_DAYS = 7

with get_engine().connect() as conn:
    rows = conn.execute(
        text("""
            SELECT feature_name, MAX(calc_date) AS latest
            FROM feature_store
            WHERE feature_name = ANY(:names)
            GROUP BY feature_name
        """),
        {"names": list(SHORTING_FAMILY)},
    ).all()
today = date.today()
stale = [(name, latest) for name, latest in rows if latest is not None and (today - latest).days > THRESHOLD_DAYS]
if not stale:
    print("OK: shorting features fresh, no self-heal needed")
    sys.exit(0)
print(f"WARN: {len(stale)} shorting features stale (> {THRESHOLD_DAYS}d):")
for n, d in stale:
    print(f"  {n}: latest={d} ({(today-d).days}d stale)")
sys.exit(1)
PY
}

if ! shorting_self_heal; then
    echo "----- shorting features stale → triggering targeted backfill -----"
    if .venv/bin/python scripts/backfill_shorting_features.py \
        --use-dynamic-universe --recent-fridays 2 2>&1; then
        echo "shorting backfill OK"
    else
        echo "WARN: shorting backfill exited non-zero — preflight will catch it next"
    fi
fi

echo "----- preflight loop (max 12 × 10min) -----"
MAX_RETRIES=12
SLEEP_SECS=600
READY=0

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "preflight attempt ${attempt}/${MAX_RETRIES}"
    if preflight_once; then
        READY=1
        break
    fi
    if [ "$attempt" -lt "$MAX_RETRIES" ]; then
        echo "not ready, sleeping ${SLEEP_SECS}s..."
        sleep "$SLEEP_SECS"
    fi
done

if [ "$READY" -ne 1 ]; then
    write_failure "preflight" "data freshness not ready after 12 × 10min retries"
    send_alert "RED" "[QuantEdge W12] preflight timeout" \
        "DB freshness did not become ready after 120 minutes. Log: $LOG_FILE"
    exit 2
fi

echo "----- run greyscale_live (full universe, --dry-run) -----"
AS_OF="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"

# Resource discipline (per CLAUDE.md tight memory)
ulimit -v 14000000

if ! .venv/bin/python scripts/run_greyscale_live.py \
        --bundle-path "$BUNDLE" \
        --report-dir "$REPORT_DIR" \
        --reference-feature-matrix-path "$REF_FM" \
        --dry-run \
        --as-of "$AS_OF"; then
    write_failure "greyscale_live" "scripts/run_greyscale_live.py exited non-zero"
    send_alert "RED" "[QuantEdge W12] greyscale_live failed" \
        "Greyscale main run failed. Log: $LOG_FILE"
    exit 3
fi

echo "----- run greyscale_monitor -----"
if ! .venv/bin/python scripts/run_greyscale_monitor.py \
        --report-dir "$REPORT_DIR" \
        --output-path "$GATE_SUMMARY" \
        --required-weeks 4 \
        --live-ic-threshold 0.02 \
        --turnover-threshold 0.08 \
        --as-of "$AS_OF" 2>&1; then
    write_failure "monitor" "scripts/run_greyscale_monitor.py exited non-zero"
    send_alert "RED" "[QuantEdge W12] monitor failed" \
        "Greyscale monitor run failed. Log: $LOG_FILE"
    exit 3
fi

echo "----- compute realized paper P&L (W13.2) -----"
# Non-fatal but consistency-protected: if the script crashes, remove any prior
# greyscale_performance.json so the dashboard can't keep serving a stale snapshot
# alongside a fresh success heartbeat (Codex review Finding 2).
PERFORMANCE_FILE="$REPORT_DIR/greyscale_performance.json"
if PYTHONPATH="$REPO_ROOT" .venv/bin/python scripts/compute_realized_returns.py \
        --report-dir "$REPORT_DIR" 2>&1; then
    echo "realized-returns updated"
else
    echo "WARN: realized-returns computation non-zero exit; removing stale performance snapshot"
    rm -f "$PERFORMANCE_FILE"
fi

echo "----- write success heartbeat -----"
write_success

echo "----- evaluate strategy health -----"
# Disable set -e around the health check: sys.exit(10) is a *signal* (red state)
# not a shell error. Without this, ERR trap fires before HEALTH_EXIT=$? can
# capture the code, so red strategy state ends up in shell_error path.
set +e
python - <<'PY'
import json
import sys
from pathlib import Path

p = Path("data/reports/greyscale/last_success.json")
payload = json.loads(p.read_text())
red_layers = [k for k in ("layer1_pass", "layer2_pass", "layer3_pass", "layer4_pass")
              if payload.get(k) is False]
gate = payload.get("gate_status")

if red_layers or gate == "FAIL":
    print(f"WARN: red layers={red_layers}, gate={gate}")
    sys.exit(10)
print(f"OK: all layers green, gate={gate}")
sys.exit(0)
PY
HEALTH_EXIT=$?
set -e

if [ "$HEALTH_EXIT" -eq 10 ]; then
    LATEST_SIGNAL_DATE=$(python -c "import json; print(json.load(open('$LAST_SUCCESS'))['signal_date'])")
    send_alert "RED" "[QuantEdge W12] strategy red state $LATEST_SIGNAL_DATE" \
        "Wrapper completed but layer or gate is red. Inspect last_success.json + week_*.json. Log: $LOG_FILE"
    echo "==================================================================="
    echo "Wrapper completed with red strategy state (exit 10)"
    echo "==================================================================="
    exit 10
elif [ "$HEALTH_EXIT" -ne 0 ]; then
    write_failure "health_check" "strategy health check exited non-zero ($HEALTH_EXIT)"
    send_alert "RED" "[QuantEdge W12] health check failed" \
        "Wrapper completed but strategy health check failed with exit code $HEALTH_EXIT. Inspect last_success.json + log. Log: $LOG_FILE"
    echo "==================================================================="
    echo "Wrapper health check failed (exit $HEALTH_EXIT)"
    echo "==================================================================="
    exit 1
fi

# All-green path — send confirmation email so silent days = real failure days.
GREEN_BODY=$(python - <<'PY'
import json
from pathlib import Path

p = Path("data/reports/greyscale/last_success.json")
payload = json.loads(p.read_text()) if p.exists() else {}
perf_path = Path("data/reports/greyscale/greyscale_performance.json")
perf = json.loads(perf_path.read_text()) if perf_path.exists() else {}

signal_date = payload.get("signal_date") or "?"
ticker_count = payload.get("ticker_count") or "?"
actual = payload.get("actual_holding_count") or payload.get("holding_count") or "?"
shadow = payload.get("shadow_holding_count") or "?"
shadow_cvar = payload.get("shadow_cvar_triggered")
gate = payload.get("gate_status") or "?"
matured = payload.get("matured_weeks") or 0
mode = payload.get("layer3_enforcement_mode")
mode_label = "shadow" if mode is False else ("enforcement" if mode is True else "?")

# 1D paper P&L if available
cum_1d = perf.get("cumulative", {}).get("1d", {}) if isinstance(perf, dict) else {}
weeks_realized = cum_1d.get("weeks_realized") or 0
cum_1d_return = cum_1d.get("return")
cum_1d_excess = cum_1d.get("excess")

lines = [
    f"Signal date: {signal_date}",
    f"Universe: {ticker_count} tickers",
    f"Actual holdings: {actual}  |  Shadow Layer 3 holdings: {shadow}",
    f"Layer 3 mode: {mode_label}  |  Shadow CVaR triggered: {shadow_cvar}",
    f"Layers: L1={payload.get('layer1_pass')} L2={payload.get('layer2_pass')} L3={payload.get('layer3_pass')} L4={payload.get('layer4_pass')}",
    f"Gate: {gate}  (matured weeks: {matured})",
    "",
    "Paper P&L (1D horizon, cumulative across realized weeks):",
    f"  Weeks realized: {weeks_realized}",
    f"  Cumulative return: {cum_1d_return:.4%}" if isinstance(cum_1d_return, (int, float)) else "  Cumulative return: pending",
    f"  Cumulative excess vs SPY: {cum_1d_excess:.4%}" if isinstance(cum_1d_excess, (int, float)) else "  Cumulative excess: pending",
    "",
    "Reports:",
    "  data/reports/greyscale/week_*.json",
    "  data/reports/greyscale/greyscale_performance.json",
    "  data/reports/greyscale/last_success.json",
]
print("\n".join(lines))
PY
)
LATEST_SIGNAL_DATE=$(python -c "import json; print(json.load(open('$LAST_SUCCESS')).get('signal_date', '?'))")
send_alert "GREEN" "[QuantEdge W12] greyscale OK — $LATEST_SIGNAL_DATE" "$GREEN_BODY"

echo "==================================================================="
echo "W12 greyscale wrapper completed cleanly: ${RUN_ID}"
echo "==================================================================="
exit 0
