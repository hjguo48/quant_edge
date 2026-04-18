#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from typing import Any
import uuid

import exchange_calendars as xcals
from loguru import logger
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import insert

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_ic_screening import write_json_atomic
from src.data.db.models import MinuteBackfillState
from src.data.db.session import get_engine, get_session_factory
from src.data.polygon_flat_files import FlatFileLoadResult, PolygonFlatFilesClient
from src.data.polygon_minute import PolygonMinuteClient

XNYS = xcals.get_calendar("XNYS")
DEFAULT_REPORT_OUTPUT = "data/reports/minute_backfill_report_20260417.json"


@dataclass(frozen=True)
class ProcessResult:
    trading_date: date
    status: str
    source_file: str | None = None
    rows_raw: int | None = None
    rows_kept: int | None = None
    tickers_loaded: int | None = None
    checksum: str | None = None
    error_message: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill Polygon minute flat files into stock_minute_aggs with PIT-universe filtering.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--universe-from-membership", action="store_true")
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args(argv)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")


def parse_date(raw: str) -> date:
    return date.fromisoformat(raw)


def is_trading_day(trade_day: date) -> bool:
    return bool(XNYS.is_session(pd.Timestamp(trade_day)))


def load_universe_whitelist_for_date(trading_date: date, *, index_name: str = "SP500") -> list[str]:
    """Return historical union of tickers ever in the index.

    TEMPORARY FALLBACK: universe_membership has a P3 data bug where
    always-in-index tickers lack historical anchor rows. True PIT filtering can
    return zero tickers for pre-2026 dates. Until that backfill is repaired, we
    return the union of distinct tickers regardless of trading_date.

    The feature/label layer still applies PIT universe filtering downstream.
    This fallback only controls which tickers enter the raw stock_minute_aggs
    storage layer.
    """
    query = sa.text(
        """
        select distinct ticker
        from universe_membership
        where index_name = :index_name
        order by ticker
        """,
    )
    with get_engine().connect() as conn:
        tickers = conn.execute(
            query,
            {"index_name": index_name},
        ).scalars().all()
    return [str(ticker).upper() for ticker in tickers]


def load_state_map() -> dict[date, dict[str, Any]]:
    query = sa.text(
        """
        select trading_date, source_file, rows_raw, rows_kept, tickers_loaded, checksum,
               started_at, finished_at, status, error_message
        from minute_backfill_state
        """,
    )
    with get_engine().connect() as conn:
        frame = pd.read_sql_query(query, conn, parse_dates=["started_at", "finished_at"])
    if frame.empty:
        return {}
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.date
    return {row["trading_date"]: row for row in frame.to_dict(orient="records")}


def upsert_backfill_state(
    *,
    trading_date: date,
    status: str,
    source_file: str | None = None,
    rows_raw: int | None = None,
    rows_kept: int | None = None,
    tickers_loaded: int | None = None,
    checksum: str | None = None,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    error_message: str | None = None,
) -> None:
    payload = {
        "trading_date": trading_date,
        "source_file": source_file,
        "rows_raw": rows_raw,
        "rows_kept": rows_kept,
        "tickers_loaded": tickers_loaded,
        "checksum": checksum,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "error_message": error_message,
    }
    statement = insert(MinuteBackfillState).values(payload)
    upsert = statement.on_conflict_do_update(
        index_elements=[MinuteBackfillState.trading_date],
        set_={
            "source_file": statement.excluded.source_file,
            "rows_raw": statement.excluded.rows_raw,
            "rows_kept": statement.excluded.rows_kept,
            "tickers_loaded": statement.excluded.tickers_loaded,
            "checksum": statement.excluded.checksum,
            "started_at": statement.excluded.started_at,
            "finished_at": statement.excluded.finished_at,
            "status": statement.excluded.status,
            "error_message": statement.excluded.error_message,
        },
    )
    session = get_session_factory()()
    try:
        session.execute(upsert)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def minute_writer(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    return PolygonMinuteClient(min_request_interval=0.0).persist_minute_aggs(frame)


def process_trade_day(
    *,
    trading_date: date,
    client: PolygonFlatFilesClient,
    universe_tickers: list[str],
    dry_run: bool,
    writer: Any,
) -> ProcessResult:
    if not is_trading_day(trading_date):
        return ProcessResult(trading_date=trading_date, status="skipped_holiday")

    if dry_run:
        sample = client.sample_day(trading_date, universe_tickers=universe_tickers)
        return ProcessResult(
            trading_date=trading_date,
            status="completed",
            source_file=sample.source_file,
            rows_raw=sample.rows_raw,
            rows_kept=sample.rows_kept,
            tickers_loaded=sample.tickers_loaded,
            checksum=sample.checksum_md5,
        )

    result = client.load_day(trading_date, universe_tickers=universe_tickers)
    frame = result.frame.copy()
    if not frame.empty:
        frame["batch_id"] = str(uuid.uuid4())
    writer(frame)
    return ProcessResult(
        trading_date=trading_date,
        status="completed",
        source_file=result.source_file,
        rows_raw=result.rows_raw,
        rows_kept=result.rows_kept,
        tickers_loaded=result.tickers_loaded,
        checksum=result.checksum_md5,
    )


def iter_calendar_days(start_date: date, end_date: date) -> list[date]:
    current = start_date
    days: list[date] = []
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)
    return days


def should_skip_resume(existing_state: dict[date, dict[str, Any]], trading_date: date) -> bool:
    state = existing_state.get(trading_date)
    if not state:
        return False
    return str(state.get("status")) in {"completed", "skipped_holiday"}


def run_backfill(args: argparse.Namespace) -> dict[str, Any]:
    start_date = parse_date(args.start_date)
    end_date = parse_date(args.end_date)
    client = PolygonFlatFilesClient(min_request_interval=0.0)
    state_map = load_state_map() if args.resume else {}
    processed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for trading_date in iter_calendar_days(start_date, end_date):
        if args.resume and should_skip_resume(state_map, trading_date):
            processed.append(
                {
                    "trading_date": trading_date.isoformat(),
                    "status": "skipped_resume",
                },
            )
            continue

        universe_tickers: list[str] | None = None
        if args.universe_from_membership and is_trading_day(trading_date):
            universe_tickers = load_universe_whitelist_for_date(trading_date)
            if not universe_tickers:
                raise RuntimeError(
                    f"universe_membership yields empty whitelist for session day {trading_date.isoformat()}; "
                    "run the PIT universe backfill before minute backfill",
                )

        started_at = datetime.now(timezone.utc)
        if not args.dry_run:
            upsert_backfill_state(trading_date=trading_date, status="in_progress", started_at=started_at)

        try:
            result = process_trade_day(
                trading_date=trading_date,
                client=client,
                universe_tickers=universe_tickers,
                dry_run=bool(args.dry_run),
                writer=minute_writer,
            )
            finished_at = datetime.now(timezone.utc)
            if not args.dry_run:
                upsert_backfill_state(
                    trading_date=trading_date,
                    status=result.status,
                    source_file=result.source_file,
                    rows_raw=result.rows_raw,
                    rows_kept=result.rows_kept,
                    tickers_loaded=result.tickers_loaded,
                    checksum=result.checksum,
                    started_at=started_at,
                    finished_at=finished_at,
                    error_message=None,
                )
            processed.append(
                {
                    "trading_date": trading_date.isoformat(),
                    "status": result.status,
                    "rows_raw": result.rows_raw,
                    "rows_kept": result.rows_kept,
                    "tickers_loaded": result.tickers_loaded,
                    "source_file": result.source_file,
                },
            )
        except Exception as exc:
            finished_at = datetime.now(timezone.utc)
            if not args.dry_run:
                upsert_backfill_state(
                    trading_date=trading_date,
                    status="failed",
                    started_at=started_at,
                    finished_at=finished_at,
                    error_message=str(exc),
                )
            errors.append(
                {
                    "trading_date": trading_date.isoformat(),
                    "error_message": str(exc),
                },
            )
            processed.append(
                {
                    "trading_date": trading_date.isoformat(),
                    "status": "failed",
                    "error_message": str(exc),
                },
            )
            if args.dry_run:
                break

    summary = {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "script_name": "run_minute_backfill.py",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "dry_run": bool(args.dry_run),
            "resume": bool(args.resume),
            "universe_from_membership": bool(args.universe_from_membership),
            "universe_size": None,
        },
        "processed": processed,
        "errors": errors,
        "summary": {
            "completed_days": sum(1 for item in processed if item["status"] == "completed"),
            "skipped_resume_days": sum(1 for item in processed if item["status"] == "skipped_resume"),
            "skipped_holidays": sum(1 for item in processed if item["status"] == "skipped_holiday"),
            "failed_days": len(errors),
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging()
    summary = run_backfill(args)
    output_path = REPO_ROOT / args.report_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if summary["summary"]["failed_days"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
