#!/usr/bin/env python3
"""W12 email alert helper — Gmail SMTP via App Password.

Reads config from ~/.config/quantedge/w12_alert.env:
  W12_ALERT_SMTP_HOST=smtp.gmail.com
  W12_ALERT_SMTP_PORT=465
  W12_ALERT_FROM=hjguo48@gmail.com
  W12_ALERT_TO=hjguo48@gmail.com
  W12_ALERT_APP_PASSWORD=xxxxxxxxxxxxxxxx

Usage:
  python scripts/send_w12_email_alert.py \
    --severity RED --subject "..." --body "..."
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path


CONFIG_PATH = Path.home() / ".config" / "quantedge" / "w12_alert.env"


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"alert config missing: {path}\n"
            f"create it from scripts/config_templates/w12_alert.env.template",
        )
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--severity", required=True,
                        choices=["RED", "YELLOW", "GREEN", "TEST", "INFO"])
    parser.add_argument("--subject", required=True)
    parser.add_argument("--body", required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print intended message but do not send")
    args = parser.parse_args(argv)

    env = load_env(args.config)

    required = {"W12_ALERT_SMTP_HOST", "W12_ALERT_SMTP_PORT",
                "W12_ALERT_FROM", "W12_ALERT_TO", "W12_ALERT_APP_PASSWORD"}
    missing = required - set(env.keys())
    if missing:
        print(f"ERROR: missing config keys: {missing}", file=sys.stderr)
        return 2

    msg = EmailMessage()
    msg["From"] = env["W12_ALERT_FROM"]
    msg["To"] = env["W12_ALERT_TO"]
    msg["Subject"] = f"[{args.severity}] {args.subject}"
    msg.set_content(args.body)

    if args.dry_run:
        print("=== DRY RUN ===")
        print(f"From: {msg['From']}")
        print(f"To: {msg['To']}")
        print(f"Subject: {msg['Subject']}")
        print(f"Body: {args.body}")
        return 0

    try:
        with smtplib.SMTP_SSL(env["W12_ALERT_SMTP_HOST"], int(env["W12_ALERT_SMTP_PORT"])) as smtp:
            smtp.login(env["W12_ALERT_FROM"], env["W12_ALERT_APP_PASSWORD"])
            smtp.send_message(msg)
        print(f"alert sent: [{args.severity}] {args.subject}")
        return 0
    except Exception as exc:
        print(f"ERROR: SMTP send failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
