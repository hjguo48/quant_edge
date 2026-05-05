from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.exc import SQLAlchemyError

from src.data.db.models import PaperPortfolioAudit
from src.data.db.session import get_engine

logger = logging.getLogger(__name__)


def _parse_date(value: date | str) -> date:
    if isinstance(value, date):
        return value
    return datetime.fromisoformat(str(value)).date()


def _parse_datetime(value: datetime | str | None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _decimal_or_none(value: Any) -> Decimal | None:
    if value is None:
        return None
    return Decimal(str(float(value)))


def derive_bundle_version(report: dict[str, Any]) -> str:
    model_bundle = report.get("model_bundle") or {}
    for value in (
        model_bundle.get("version"),
        report.get("bundle_version"),
        model_bundle.get("bundle_version"),
    ):
        if value:
            return str(value)

    bundle_path = model_bundle.get("path")
    if bundle_path:
        path = Path(str(bundle_path))
        if path.name == "bundle.json" and path.parent.name:
            return path.parent.name

    if model_bundle.get("window_id"):
        return str(model_bundle["window_id"])
    return "unknown"


def save_paper_portfolio_snapshot(
    *,
    bundle_version: str,
    signal_date: date | str,
    target_weights: dict[str, Any],
    raw_scores: dict[str, Any] | None = None,
    generated_at_utc: datetime | str | None = None,
    run_id: str | None = None,
    conn: sa.Connection | None = None,
) -> int:
    signal_dt = _parse_date(signal_date)
    generated_at = _parse_datetime(generated_at_utc)
    raw_scores = raw_scores or {}
    rows = [
        {
            "bundle_version": str(bundle_version),
            "signal_date": signal_dt,
            "ticker": str(ticker).upper(),
            "target_weight": Decimal(str(float(weight))),
            "raw_score": _decimal_or_none(raw_scores.get(ticker) or raw_scores.get(str(ticker).upper())),
            "generated_at_utc": generated_at,
            "run_id": run_id,
        }
        for ticker, weight in sorted((target_weights or {}).items())
    ]

    def _write(connection: sa.Connection) -> int:
        connection.execute(
            sa.delete(PaperPortfolioAudit).where(
                PaperPortfolioAudit.bundle_version == str(bundle_version),
                PaperPortfolioAudit.signal_date == signal_dt,
            ),
        )
        if not rows:
            return 0
        connection.execute(sa.insert(PaperPortfolioAudit), rows)
        return len(rows)

    if conn is not None:
        return _write(conn)
    with get_engine().begin() as connection:
        return _write(connection)


def save_paper_portfolio_report(
    report: dict[str, Any],
    *,
    run_id: str | None = None,
    conn: sa.Connection | None = None,
) -> int:
    live_outputs = report.get("live_outputs") or {}
    signal_date = live_outputs.get("signal_date")
    if not signal_date:
        raise ValueError("greyscale report is missing live_outputs.signal_date")
    target_weights = live_outputs.get("target_weights_after_risk") or {}
    raw_scores = (report.get("score_vectors") or {}).get("fusion") or {}
    return save_paper_portfolio_snapshot(
        bundle_version=derive_bundle_version(report),
        signal_date=signal_date,
        target_weights=target_weights,
        raw_scores=raw_scores,
        generated_at_utc=report.get("generated_at_utc"),
        run_id=run_id,
        conn=conn,
    )


def load_db_paper_portfolio(
    signal_date: date | str,
    *,
    conn: sa.Connection | None = None,
) -> dict[str, float] | None:
    signal_dt = _parse_date(signal_date)

    def _read(connection: sa.Connection) -> dict[str, float] | None:
        rows = connection.execute(
            sa.select(
                PaperPortfolioAudit.bundle_version,
                PaperPortfolioAudit.ticker,
                PaperPortfolioAudit.target_weight,
                PaperPortfolioAudit.generated_at_utc,
            )
            .where(PaperPortfolioAudit.signal_date == signal_dt)
            .order_by(PaperPortfolioAudit.generated_at_utc.desc(), PaperPortfolioAudit.ticker.asc()),
        ).all()
        if not rows:
            return None
        latest_bundle = rows[0].bundle_version
        return {
            row.ticker: float(row.target_weight)
            for row in rows
            if row.bundle_version == latest_bundle
        }

    try:
        if conn is not None:
            return _read(conn)
        with get_engine().connect() as connection:
            return _read(connection)
    except SQLAlchemyError as exc:
        logger.warning("paper_portfolio_audit DB read failed; falling back to JSON: %s", exc)
        return None


def _load_file_portfolio(signal_date: date, report_dir: Path) -> dict[str, float] | None:
    for path in sorted(report_dir.glob("week_*.json")):
        if "bak" in path.name or "contaminated" in path.name:
            continue
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        live_outputs = report.get("live_outputs") or {}
        if live_outputs.get("signal_date") != signal_date.isoformat():
            continue
        weights = live_outputs.get("target_weights_after_risk") or {}
        return {str(ticker).upper(): float(weight) for ticker, weight in weights.items()}
    return None


def load_paper_portfolio(
    signal_date: date | str,
    *,
    report_dir: Path,
    conn: sa.Connection | None = None,
) -> dict[str, float] | None:
    signal_dt = _parse_date(signal_date)
    return load_db_paper_portfolio(signal_dt, conn=conn) or _load_file_portfolio(signal_dt, report_dir)


def load_db_paper_portfolio_reports(conn: sa.Connection) -> list[dict[str, Any]]:
    try:
        rows = conn.execute(
            sa.select(
                PaperPortfolioAudit.signal_date,
                PaperPortfolioAudit.ticker,
                PaperPortfolioAudit.target_weight,
                PaperPortfolioAudit.generated_at_utc,
                PaperPortfolioAudit.bundle_version,
            ).order_by(
                PaperPortfolioAudit.signal_date.asc(),
                PaperPortfolioAudit.generated_at_utc.desc(),
                PaperPortfolioAudit.ticker.asc(),
            ),
        ).all()
    except SQLAlchemyError as exc:
        logger.warning("paper_portfolio_audit DB scan failed; using week_*.json only: %s", exc)
        return []

    latest_by_date: dict[date, tuple[str, datetime]] = {}
    for row in rows:
        current = latest_by_date.get(row.signal_date)
        row_key = (row.bundle_version, row.generated_at_utc)
        if current is None or row_key[1] > current[1]:
            latest_by_date[row.signal_date] = row_key

    weights_by_date: dict[date, dict[str, float]] = defaultdict(dict)
    generated_by_date: dict[date, datetime] = {}
    for row in rows:
        latest = latest_by_date.get(row.signal_date)
        if latest is None or row.bundle_version != latest[0]:
            continue
        weights_by_date[row.signal_date][row.ticker] = float(row.target_weight)
        generated_by_date[row.signal_date] = row.generated_at_utc

    reports: list[dict[str, Any]] = []
    for index, signal_dt in enumerate(sorted(weights_by_date), start=1):
        reports.append(
            {
                "generated_at_utc": generated_by_date[signal_dt].isoformat(),
                "week_number": index,
                "live_outputs": {
                    "signal_date": signal_dt.isoformat(),
                    "target_weights_after_risk": weights_by_date[signal_dt],
                },
                "paper_portfolio_source": "db",
            },
        )
    return reports
