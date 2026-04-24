from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
import sys
from typing import Any

import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

try:
    import scripts.preflight_trades_estimator as preflight_trades_estimator
except ModuleNotFoundError:  # pragma: no cover - exercised by direct script execution.
    import preflight_trades_estimator as preflight_trades_estimator
from src.config.week4_trades import Week4TradesConfig
from src.data.db.models import TradesSamplingState
from src.data.db.session import get_session_factory
from src.data.event_calendar import SamplingEvent, build_sampling_plan

DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_OUTPUT_PATH = Path("data/reports/week4/trades_sampling_plan.parquet")
DEFAULT_PREFLIGHT_OUTPUT_PATH = Path("data/reports/week4/preflight_estimate.json")
DEFAULT_START_DATE = date(2016, 4, 17)
DEFAULT_END_DATE = date(2026, 4, 17)
PLAN_COLUMNS = ["ticker", "trading_date", "reason"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Week 4 targeted trades sampling universe.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT_OUTPUT_PATH)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--force", action="store_true", help="Continue even if preflight budget check fails.")
    parser.add_argument("--dry-run", action="store_true", help="Write parquet preview only; do not write DB state.")
    return parser.parse_args(argv)


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(item) for item in value]
    return value


def sampling_events_to_frame(events: Sequence[SamplingEvent]) -> pd.DataFrame:
    rows = [
        {
            "ticker": event.ticker.upper(),
            "trading_date": event.trading_date,
            "reason": event.reason,
        }
        for event in events
    ]
    frame = pd.DataFrame(rows, columns=PLAN_COLUMNS)
    if frame.empty:
        return frame
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.date
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["reason"] = frame["reason"].astype(str).str.lower()
    frame.drop_duplicates(PLAN_COLUMNS, inplace=True)
    frame.sort_values(["trading_date", "ticker", "reason"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    return frame


def write_sampling_plan(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def summarize_plan(frame: pd.DataFrame, *, state_rows_inserted: int, dry_run: bool, output_path: Path) -> dict[str, Any]:
    reason_counts = (
        frame.groupby("reason", dropna=False)
        .size()
        .sort_index()
        .astype(int)
        .to_dict()
        if not frame.empty
        else {}
    )
    return {
        "output": str(output_path),
        "total_rows": int(len(frame)),
        "unique_ticker_days": int(frame[["ticker", "trading_date"]].drop_duplicates().shape[0])
        if not frame.empty
        else 0,
        "reason_counts": reason_counts,
        "state_rows_inserted": int(state_rows_inserted),
        "dry_run": dry_run,
    }


def insert_pending_state(
    frame: pd.DataFrame,
    *,
    session_factory: Callable[[], Session] | None = None,
) -> int:
    if frame.empty:
        return 0

    records = [
        {
            "ticker": str(row.ticker).upper(),
            "trading_date": row.trading_date,
            "sampled_reason": str(row.reason).lower(),
            "status": "pending",
            "rows_ingested": None,
            "pages_fetched": None,
            "api_calls_used": None,
            "started_at": None,
            "completed_at": None,
            "error_message": None,
        }
        for row in frame.itertuples(index=False)
    ]
    factory = session_factory or get_session_factory()
    with factory() as session:
        bind = session.get_bind()
        if bind.dialect.name == "postgresql":
            statement = pg_insert(TradesSamplingState).values(records)
            statement = statement.on_conflict_do_nothing(
                index_elements=[
                    TradesSamplingState.ticker,
                    TradesSamplingState.trading_date,
                    TradesSamplingState.sampled_reason,
                ],
            )
            result = session.execute(statement)
            session.commit()
            return int(result.rowcount or 0)

        inserted = 0
        for record in records:
            key = (record["ticker"], record["trading_date"], record["sampled_reason"])
            if session.get(TradesSamplingState, key) is not None:
                continue
            session.add(TradesSamplingState(**record))
            inserted += 1
        session.commit()
        return inserted


def run_preflight_or_exit(
    *,
    config_path: Path,
    start_date: date,
    end_date: date,
    output_path: Path,
    concurrency: int,
    force: bool,
) -> dict[str, Any] | None:
    report = preflight_trades_estimator.run_estimator(
        config_path=config_path,
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        concurrency=concurrency,
    )
    if report.get("pass") is True:
        return report

    warning = {
        "message": "Week 4 trades preflight failed; adjust YAML or rerun with --force.",
        "failure_reasons": report.get("failure_reasons", []),
        "preflight_output": str(output_path),
    }
    print(json.dumps(_normalize_for_json(warning), indent=2, sort_keys=True), file=sys.stderr)
    if not force:
        return None
    print("WARNING: continuing despite failed preflight because --force was supplied.", file=sys.stderr)
    return report


def build_trades_sample_universe(
    *,
    config_path: Path,
    start_date: date,
    end_date: date,
    output_path: Path,
    preflight_output_path: Path,
    concurrency: int = 1,
    force: bool = False,
    dry_run: bool = False,
    session_factory: Callable[[], Session] | None = None,
) -> dict[str, Any] | None:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    preflight_report = run_preflight_or_exit(
        config_path=config_path,
        start_date=start_date,
        end_date=end_date,
        output_path=preflight_output_path,
        concurrency=concurrency,
        force=force,
    )
    if preflight_report is None:
        return None

    config: Week4TradesConfig = preflight_trades_estimator.load_config(config_path)
    events = build_sampling_plan(
        start_date=start_date,
        end_date=end_date,
        config=config,
        session_factory=session_factory,
    )
    frame = sampling_events_to_frame(events)
    write_sampling_plan(frame, output_path)
    state_rows_inserted = 0 if dry_run else insert_pending_state(frame, session_factory=session_factory)
    return {
        "summary": summarize_plan(
            frame,
            state_rows_inserted=state_rows_inserted,
            dry_run=dry_run,
            output_path=output_path,
        ),
        "preflight": {
            "pass": bool(preflight_report.get("pass")),
            "failure_reasons": preflight_report.get("failure_reasons", []),
            "output": str(preflight_output_path),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_trades_sample_universe(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        preflight_output_path=args.preflight_output,
        concurrency=args.concurrency,
        force=args.force,
        dry_run=args.dry_run,
    )
    if result is None:
        return 1
    print(json.dumps(_normalize_for_json(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
