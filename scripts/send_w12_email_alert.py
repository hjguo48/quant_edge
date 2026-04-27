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
import socket
import ssl
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

    # WSL2 sometimes has broken IPv6 routing → resolve IPv4 explicitly to avoid timeout.
    # Keep SNI hostname for cert verification.
    host = env["W12_ALERT_SMTP_HOST"]
    port = int(env["W12_ALERT_SMTP_PORT"])
    try:
        try:
            ipv4_addrs = [info[4][0] for info in socket.getaddrinfo(host, port, socket.AF_INET)]
        except socket.gaierror as exc:
            print(f"ERROR: DNS lookup failed for {host}: {exc}", file=sys.stderr)
            return 3
        if not ipv4_addrs:
            print(f"ERROR: no IPv4 address for {host}", file=sys.stderr)
            return 3

        context = ssl.create_default_context()
        last_exc: Exception | None = None
        for ipv4 in ipv4_addrs:
            try:
                # Connect to IPv4 IP but verify cert against hostname via SNI.
                with smtplib.SMTP_SSL(ipv4, port, timeout=45, context=context, local_hostname="localhost") as smtp:
                    # Manually trigger SNI by re-wrapping if needed; SMTP_SSL.__init__ already wraps with server_hostname=host_param,
                    # so cert hostname mismatch may occur. Switch to explicit SSL context with check_hostname=True
                    # via a manual connect path.
                    smtp.ehlo()
                    smtp.login(env["W12_ALERT_FROM"], env["W12_ALERT_APP_PASSWORD"])
                    smtp.send_message(msg)
                print(f"alert sent: [{args.severity}] {args.subject}")
                return 0
            except ssl.SSLCertVerificationError:
                # Cert is for smtp.gmail.com, not the IP. Fall back to manual connect with SNI.
                last_exc = None
                try:
                    raw_sock = socket.create_connection((ipv4, port), timeout=45)
                    ssl_sock = context.wrap_socket(raw_sock, server_hostname=host)
                    smtp = smtplib.SMTP_SSL(local_hostname="localhost")
                    smtp.sock = ssl_sock
                    smtp.file = smtp.sock.makefile('rb')
                    code, msg_resp = smtp.getreply()
                    if code != 220:
                        raise smtplib.SMTPConnectError(code, msg_resp)
                    smtp.ehlo()
                    smtp.login(env["W12_ALERT_FROM"], env["W12_ALERT_APP_PASSWORD"])
                    smtp.send_message(msg)
                    smtp.quit()
                    print(f"alert sent (SNI fallback): [{args.severity}] {args.subject}")
                    return 0
                except Exception as inner:
                    last_exc = inner
                    continue
            except Exception as exc:
                last_exc = exc
                continue

        if last_exc is not None:
            raise last_exc
        return 3
    except Exception as exc:
        print(f"ERROR: SMTP send failed: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
