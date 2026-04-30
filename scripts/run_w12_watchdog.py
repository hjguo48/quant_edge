#!/usr/bin/env python3
"""W12 daily watchdog — heartbeat + service health check.

Runs every day. Sends RED email alert on any of:
  - last_success.json missing or > 8 days old
  - last_failure.json newer than last success
  - DB latest PIT trade date > 4 days stale
  - /api/health unreachable for 2nd day in a row
  - frontend unreachable for 2nd day in a row
  - disk free < 5 GB

Sends YELLOW alert on:
  - first-time API/frontend unavailable
  - disk free < 10 GB

Usage:
  python scripts/run_w12_watchdog.py

Exit codes:
  0 — all green
  1 — at least one alert sent
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from src.utils.io import write_json_atomic

REPO_ROOT = Path("/home/jiahao/quant_edge")
REPORT_DIR = REPO_ROOT / "data" / "reports" / "greyscale"
LAST_SUCCESS = REPORT_DIR / "last_success.json"
LAST_FAILURE = REPORT_DIR / "last_failure.json"
WATCHDOG_STATE = REPORT_DIR / ".watchdog_state.json"

API_URL = "http://127.0.0.1:8000/api/health"
FRONTEND_URL = "http://127.0.0.1:5173/"  # WSL-side; Windows accesses via relay :15173
DB_FRESHNESS_RED_DAYS = 4
SUCCESS_STALE_RED_DAYS = 8
DISK_FREE_RED_GB = 5
DISK_FREE_YELLOW_GB = 10


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def age_days(iso_ts: str) -> float:
    dt = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def check_url(url: str, timeout: int = 5) -> tuple[bool, str]:
    """Return (ok, message). Uses curl to avoid requests dependency."""
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", str(timeout), url],
            capture_output=True, text=True, timeout=timeout + 5,
        )
        code = result.stdout.strip()
        if code == "200":
            return True, "200"
        return False, f"http {code}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, f"error: {exc}"


def get_db_latest_trade_date() -> tuple[bool, str]:
    """Return (ok, message). Calls into project venv to query DB.

    Uses XNYS market calendar to compute expected latest PIT-visible session,
    accounting for T+1 knowledge_time lag.
    """
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        result = subprocess.run(
            [
                str(REPO_ROOT / ".venv/bin/python"),
                "-c",
                "from datetime import datetime, timezone\n"
                "from zoneinfo import ZoneInfo\n"
                "import exchange_calendars as xcals\n"
                "import pandas as pd\n"
                "from scripts.run_live_pipeline import load_db_state\n"
                "as_of = datetime.now(timezone.utc)\n"
                "db = load_db_state(as_of=as_of)\n"
                "latest = db['latest_pit_trade_date']\n"
                "if latest is None:\n"
                "    print('NONE'); raise SystemExit(0)\n"
                "XNYS = xcals.get_calendar('XNYS')\n"
                "today_et = as_of.astimezone(ZoneInfo('America/New_York')).date()\n"
                "today_ts = pd.Timestamp(today_et)\n"
                "if XNYS.is_session(today_ts):\n"
                "    expected = XNYS.previous_session(today_ts).date()\n"
                "else:\n"
                "    expected = XNYS.date_to_session(today_ts, direction='previous').date()\n"
                "tolerance = XNYS.previous_session(pd.Timestamp(expected)).date()\n"
                "print(f'{latest.isoformat()}|{expected.isoformat()}|{tolerance.isoformat()}')\n",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True, text=True, timeout=30, env=env,
        )
        if result.returncode != 0:
            return False, f"DB query failed: {result.stderr.strip()[:200]}"
        out = result.stdout.strip()
        if out == "NONE":
            return False, "no PIT trade date"
        parts = out.split("|")
        if len(parts) != 3:
            return False, f"unexpected output: {out}"
        latest = datetime.fromisoformat(parts[0]).date()
        expected = datetime.fromisoformat(parts[1]).date()
        tolerance = datetime.fromisoformat(parts[2]).date()
        if latest < tolerance:
            return False, f"stale: {latest} (expected {expected}, tolerance {tolerance})"
        return True, f"{latest} (expected {expected})"
    except Exception as exc:
        return False, f"exception: {exc}"


def load_state() -> dict:
    return load_json(WATCHDOG_STATE) or {
        "api_consecutive_fails": 0,
        "frontend_consecutive_fails": 0,
    }


def save_state(state: dict) -> None:
    write_json_atomic(WATCHDOG_STATE, state)


def send_alert(severity: str, subject: str, body: str) -> bool:
    """Call send_w12_email_alert.py. Return True if sent."""
    cmd = [
        str(REPO_ROOT / ".venv/bin/python"),
        str(REPO_ROOT / "scripts/send_w12_email_alert.py"),
        "--severity", severity,
        "--subject", subject,
        "--body", body,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            return True
        print(f"WARN: alert send returned {result.returncode}: {result.stderr.strip()[:200]}")
        return False
    except Exception as exc:
        print(f"WARN: alert send exception: {exc}")
        return False


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    alerts: list[tuple[str, str]] = []  # (severity, message)

    # 1. last_success heartbeat
    success = load_json(LAST_SUCCESS)
    if not success:
        alerts.append(("RED", "last_success.json missing entirely"))
    else:
        ts = success.get("generated_at_utc")
        if ts:
            age = age_days(ts)
            if age > SUCCESS_STALE_RED_DAYS:
                alerts.append(("RED", f"last_success.json stale: {age:.1f} days"))
            else:
                print(f"OK: last_success age = {age:.1f} days")
        else:
            alerts.append(("RED", "last_success.json missing generated_at_utc"))

    # 2. failure newer than success
    failure = load_json(LAST_FAILURE)
    if failure and success:
        fts = failure.get("generated_at_utc")
        sts = success.get("generated_at_utc")
        if fts and sts and fts > sts:
            alerts.append(("RED",
                f"latest run failed (failure {fts}) after last success ({sts}); "
                f"stage={failure.get('stage')}"))

    # 3. DB freshness
    db_ok, db_msg = get_db_latest_trade_date()
    if not db_ok:
        alerts.append(("RED", f"DB freshness: {db_msg}"))
    else:
        print(f"OK: DB freshness {db_msg}")

    # 4. API health
    api_ok, api_msg = check_url(API_URL)
    if api_ok:
        print(f"OK: API {api_msg}")
        state["api_consecutive_fails"] = 0
    else:
        state["api_consecutive_fails"] = state.get("api_consecutive_fails", 0) + 1
        if state["api_consecutive_fails"] >= 2:
            alerts.append(("RED",
                f"API unavailable for {state['api_consecutive_fails']} days: {api_msg}"))
        else:
            alerts.append(("YELLOW", f"API unavailable today: {api_msg}"))

    # 5. Frontend
    fe_ok, fe_msg = check_url(FRONTEND_URL)
    if fe_ok:
        print(f"OK: frontend {fe_msg}")
        state["frontend_consecutive_fails"] = 0
    else:
        state["frontend_consecutive_fails"] = state.get("frontend_consecutive_fails", 0) + 1
        if state["frontend_consecutive_fails"] >= 2:
            alerts.append(("RED",
                f"frontend unavailable for {state['frontend_consecutive_fails']} days: {fe_msg}"))
        else:
            alerts.append(("YELLOW", f"frontend unavailable today: {fe_msg}"))

    # 6. Disk space
    free_gb = shutil.disk_usage(str(REPO_ROOT)).free / (1024 ** 3)
    if free_gb < DISK_FREE_RED_GB:
        alerts.append(("RED", f"disk free critically low: {free_gb:.1f} GB"))
    elif free_gb < DISK_FREE_YELLOW_GB:
        alerts.append(("YELLOW", f"disk free warning: {free_gb:.1f} GB"))
    else:
        print(f"OK: disk free {free_gb:.1f} GB")

    save_state(state)

    if not alerts:
        print("watchdog ALL GREEN")
        return 0

    # Aggregate alerts into single email
    red_msgs = [m for sev, m in alerts if sev == "RED"]
    yellow_msgs = [m for sev, m in alerts if sev == "YELLOW"]
    overall_severity = "RED" if red_msgs else "YELLOW"

    body_lines = [
        f"QuantEdge W12 daily watchdog detected {len(alerts)} alert(s):",
        "",
    ]
    if red_msgs:
        body_lines.append("RED:")
        for m in red_msgs:
            body_lines.append(f"  - {m}")
        body_lines.append("")
    if yellow_msgs:
        body_lines.append("YELLOW:")
        for m in yellow_msgs:
            body_lines.append(f"  - {m}")
        body_lines.append("")
    body_lines.append(f"Watchdog run at: {datetime.now(timezone.utc).isoformat()}")
    body_lines.append(f"Reports dir: {REPORT_DIR}")

    body = "\n".join(body_lines)
    subject = f"[QuantEdge W12] watchdog {overall_severity}: {len(alerts)} alert(s)"

    sent = send_alert(overall_severity, subject, body)
    if sent:
        print(f"alert email sent: {overall_severity}")
    else:
        print(f"alert email FAILED to send (alerts still logged below)")
    print(body)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
