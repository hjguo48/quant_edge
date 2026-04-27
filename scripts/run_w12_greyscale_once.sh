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

BUNDLE="data/models/bundles/w12_60d_ridge_swbuf_v2/bundle.json"
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
from pathlib import Path

payload = {
    "status": "failure",
    "stage": "$stage",
    "message": """$message""",
    "bundle_version": "w12_60d_ridge_swbuf_v2",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "log_file": "$LOG_FILE",
    "run_id": "$RUN_ID",
}
Path("$LAST_FAILURE").write_text(json.dumps(payload, indent=2))
PY
}

write_success() {
    python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

report_dir = Path("data/reports/greyscale")
weeks = sorted(report_dir.glob("week_*.json"))
latest = json.loads(weeks[-1].read_text()) if weeks else {}
gate_path = report_dir / "g4_gate_summary.json"
gate = json.loads(gate_path.read_text()) if gate_path.exists() else {}

# Extract key fields safely
risk = latest.get("risk_checks", {}) or {}
score_vectors = latest.get("score_vectors", {}) or {}
fusion_scores = score_vectors.get("fusion") or score_vectors.get("FUSION") or {}

payload = {
    "status": "success",
    "bundle_version": "w12_60d_ridge_swbuf_v2",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "latest_report_path": str(weeks[-1]) if weeks else None,
    "signal_date": latest.get("live_outputs", {}).get("signal_date"),
    "ticker_count": int(len(fusion_scores) if isinstance(fusion_scores, dict) else 0),
    "layer1_pass": risk.get("layer1_data", {}).get("pass"),
    "layer2_pass": risk.get("layer2_signal", {}).get("pass"),
    "layer3_pass": risk.get("layer3_portfolio", {}).get("pass"),
    "layer4_pass": risk.get("layer4_operational", {}).get("pass"),
    "gate_status": gate.get("summary", {}).get("gate_status"),
    "matured_weeks": gate.get("summary", {}).get("matured_weeks"),
}
Path("data/reports/greyscale/last_success.json").write_text(json.dumps(payload, indent=2))
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
        python "$REPO_ROOT/scripts/send_w12_email_alert.py" \
            --severity "$severity" \
            --subject "$subject" \
            --body "$body" || echo "WARNING: alert send failed"
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
bundle_path = Path("data/models/bundles/w12_60d_ridge_swbuf_v2/bundle.json")
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
    # Always use previous_session as the conservative expected (T+1 lag means today's close not yet PIT-visible)
    expected_latest = XNYS.previous_session(today_ts).date()
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

echo "----- write success heartbeat -----"
write_success

echo "----- evaluate strategy health -----"
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

if [ $HEALTH_EXIT -eq 10 ]; then
    LATEST_SIGNAL_DATE=$(python -c "import json; print(json.load(open('$LAST_SUCCESS'))['signal_date'])")
    send_alert "RED" "[QuantEdge W12] strategy red state $LATEST_SIGNAL_DATE" \
        "Wrapper completed but layer or gate is red. Inspect last_success.json + week_*.json. Log: $LOG_FILE"
    echo "==================================================================="
    echo "Wrapper completed with red strategy state (exit 10)"
    echo "==================================================================="
    exit 10
fi

echo "==================================================================="
echo "W12 greyscale wrapper completed cleanly: ${RUN_ID}"
echo "==================================================================="
exit 0
