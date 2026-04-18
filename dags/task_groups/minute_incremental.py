from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

try:
    from airflow.exceptions import AirflowException
    from airflow.operators.python import PythonOperator
    from airflow.utils.task_group import TaskGroup
    from airflow.utils.trigger_rule import TriggerRule
except ImportError:
    from dags._airflow_compat import AirflowException, PythonOperator, TaskGroup, TriggerRule
import exchange_calendars as xcals
import pandas as pd
import sqlalchemy as sa

from scripts.run_intraday_smoke import (
    load_daily_prices,
    persist_reconciliation_events,
    validate_minute_to_day_consistency,
)
from src.data.db.models import StockMinuteAggs
from src.data.db.session import get_engine

LOGGER = logging.getLogger(__name__)
XNYS = xcals.get_calendar("XNYS")
DEFAULT_LOOKBACK_DAYS = 7
ALLOWED_MISSING_BARS = 3


def _project_root() -> Path:
    candidates = [
        os.environ.get("QUANTEDGE_REPO_ROOT"),
        str(Path(__file__).resolve().parents[2]),
        "/opt/quantedge",
        "/home/jiahao/quant_edge",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).resolve()
        if (path / "src").exists():
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
            return path
    raise RuntimeError("Unable to locate QuantEdge project root for minute incremental task group.")


def minute_incremental_enabled() -> bool:
    raw = os.environ.get("ENABLE_MINUTE_INCREMENTAL")
    if raw is not None:
        return raw.lower() in {"1", "true", "yes", "on"}
    try:
        from airflow.models import Variable

        value = Variable.get("ENABLE_MINUTE_INCREMENTAL", default_var="false")
    except Exception:
        value = "false"
    return str(value).lower() in {"1", "true", "yes", "on"}


def _now_utc_datetime() -> datetime:
    return datetime.now(timezone.utc)


def _latest_completed_session_date(current_time: datetime | None = None) -> date:
    now_utc = current_time or _now_utc_datetime()
    now_ts = pd.Timestamp(now_utc)
    today = now_ts.date()
    today_label = pd.Timestamp(today)
    if XNYS.is_session(today_label):
        session_close = pd.Timestamp(XNYS.session_close(today_label))
        if now_ts >= session_close:
            return today
        return XNYS.previous_session(today_label).date()
    return XNYS.date_to_session(today_label, direction="previous").date()


def _recent_session_dates(*, reference_date: date, lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> list[date]:
    start = reference_date - timedelta(days=lookback_days - 1)
    sessions = XNYS.sessions_in_range(pd.Timestamp(start), pd.Timestamp(reference_date))
    return [ts.date() for ts in sessions if ts.date() <= reference_date]


def _load_recent_state_rows(session_dates: Sequence[date]) -> list[dict[str, Any]]:
    if not session_dates:
        return []
    query = sa.text(
        """
        select trading_date, status
        from minute_backfill_state
        where trading_date = any(:dates)
        """,
    )
    with get_engine().connect() as conn:
        rows = conn.execute(query, {"dates": list(session_dates)}).mappings().all()
    return [dict(row) for row in rows]


def resolve_minute_dates_to_sync(
    *,
    reference_date: date | None = None,
    current_time: datetime | None = None,
    state_rows: Sequence[dict[str, Any]] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> dict[str, Any]:
    reference = reference_date or _latest_completed_session_date(current_time)
    session_dates = _recent_session_dates(reference_date=reference, lookback_days=lookback_days)
    known_rows = {
        pd.Timestamp(row["trading_date"]).date(): str(row.get("status") or "").lower()
        for row in (state_rows if state_rows is not None else _load_recent_state_rows(session_dates))
    }
    pending_dates = [
        trade_day
        for trade_day in session_dates
        if known_rows.get(trade_day) not in {"completed", "skipped_holiday"}
    ]
    return {
        "status": "ok" if pending_dates else "skipped",
        "reference_date": reference.isoformat(),
        "lookback_days": int(lookback_days),
        "candidate_session_dates": [trade_day.isoformat() for trade_day in session_dates],
        "dates_to_sync": [trade_day.isoformat() for trade_day in pending_dates],
    }


def _normalize_dates(raw_dates: Sequence[str | date] | None) -> list[date]:
    normalized: list[date] = []
    for value in raw_dates or []:
        if isinstance(value, date):
            normalized.append(value)
        else:
            normalized.append(date.fromisoformat(str(value)))
    return sorted(set(normalized))


def _pull_xcom_dates(context: dict[str, Any], *, task_id: str) -> list[date]:
    ti = context.get("ti")
    if ti is None:
        return []
    payload = ti.xcom_pull(task_ids=task_id) or {}
    return _normalize_dates(payload.get("dates_to_sync"))


def sync_polygon_minute_incremental(
    *,
    resolved_dates: Sequence[str | date] | None = None,
    repo_root: Path | None = None,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    dates = _normalize_dates(resolved_dates)
    if not dates:
        return {"status": "skipped", "reason": "no_dates_to_sync", "dates_to_sync": []}

    root = repo_root or _project_root()
    command = [
        sys.executable,
        str(root / "scripts" / "run_minute_backfill.py"),
        "--start-date",
        min(dates).isoformat(),
        "--end-date",
        max(dates).isoformat(),
        "--universe-from-membership",
        "--resume",
    ]
    completed = runner(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    result = {
        "status": "ok" if completed.returncode == 0 else "error",
        "dates_to_sync": [trade_day.isoformat() for trade_day in dates],
        "command": command,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout.strip().splitlines()[-10:],
        "stderr_tail": completed.stderr.strip().splitlines()[-10:],
    }
    if completed.returncode != 0:
        raise AirflowException(
            "Minute incremental sync failed for "
            f"{min(dates).isoformat()} -> {max(dates).isoformat()} with exit_code={completed.returncode}",
        )
    return result


def _load_minute_rows_for_dates(dates_to_check: Sequence[date]) -> pd.DataFrame:
    if not dates_to_check:
        return pd.DataFrame()
    statement = (
        sa.select(
            StockMinuteAggs.ticker,
            StockMinuteAggs.trade_date,
            StockMinuteAggs.minute_ts,
            StockMinuteAggs.open,
            StockMinuteAggs.high,
            StockMinuteAggs.low,
            StockMinuteAggs.close,
            StockMinuteAggs.volume,
            StockMinuteAggs.vwap,
            StockMinuteAggs.transactions,
        )
        .where(StockMinuteAggs.trade_date.in_(list(dates_to_check)))
        .order_by(StockMinuteAggs.ticker, StockMinuteAggs.minute_ts)
    )
    with get_engine().connect() as conn:
        rows = conn.execute(statement).mappings().all()
    return pd.DataFrame(rows)


def validate_minute_internal_quality(
    *,
    resolved_dates: Sequence[str | date] | None = None,
    minute_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    dates = _normalize_dates(resolved_dates)
    if not dates:
        return {"status": "skipped", "reason": "no_dates_to_validate", "dates_to_sync": []}

    frame = minute_frame.copy() if minute_frame is not None else _load_minute_rows_for_dates(dates)
    if frame.empty:
        raise AirflowException("Minute internal quality check found no rows for requested dates.")

    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    frame["minute_ts"] = pd.to_datetime(frame["minute_ts"], utc=True)

    failures: list[dict[str, Any]] = []

    duplicate_groups = frame.loc[frame.duplicated(["ticker", "minute_ts"], keep=False), ["ticker", "trade_date", "minute_ts"]]
    if not duplicate_groups.empty:
        for row in duplicate_groups.head(10).to_dict(orient="records"):
            failures.append({"check": "duplicate_minute_ts", **row})

    for (ticker, trade_day), group in frame.groupby(["ticker", "trade_date"], sort=False):
        group = group.sort_values("minute_ts")
        session_label = pd.Timestamp(trade_day)
        session_open = pd.Timestamp(XNYS.session_open(session_label))
        session_close = pd.Timestamp(XNYS.session_close(session_label))
        expected_bars = int((session_close - session_open).total_seconds() / 60)
        minimum_required_bars = max(expected_bars - ALLOWED_MISSING_BARS, 0)
        if len(group) < minimum_required_bars:
            failures.append(
                {
                    "check": "insufficient_bars",
                    "ticker": ticker,
                    "trade_date": trade_day,
                    "observed_bars": int(len(group)),
                    "expected_bars": expected_bars,
                    "minimum_required_bars": minimum_required_bars,
                },
            )
        deltas = group["minute_ts"].diff().dropna()
        if (deltas <= pd.Timedelta(0)).any():
            failures.append(
                {
                    "check": "non_monotonic_minute_ts",
                    "ticker": ticker,
                    "trade_date": trade_day,
                },
            )
        invalid_prices = group.loc[
            (pd.to_numeric(group["open"], errors="coerce") < 0)
            | (pd.to_numeric(group["high"], errors="coerce") < 0)
            | (pd.to_numeric(group["low"], errors="coerce") < 0)
            | (pd.to_numeric(group["close"], errors="coerce") < 0)
        ]
        if not invalid_prices.empty:
            failures.append(
                {
                    "check": "negative_ohlc",
                    "ticker": ticker,
                    "trade_date": trade_day,
                    "row_count": int(len(invalid_prices)),
                },
            )
        nan_ohlc = group.loc[
            group.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
        ]
        if not nan_ohlc.empty:
            failures.append(
                {
                    "check": "nan_ohlc",
                    "ticker": ticker,
                    "trade_date": trade_day,
                    "row_count": int(len(nan_ohlc)),
                },
            )

    result = {
        "status": "ok" if not failures else "error",
        "dates_to_sync": [trade_day.isoformat() for trade_day in dates],
        "checked_rows": int(len(frame)),
        "failure_count": int(len(failures)),
        "failures": failures[:25],
    }
    if failures:
        preview = ", ".join(
            f"{failure['check']}:{failure.get('ticker')}:{failure.get('trade_date')}"
            for failure in failures[:10]
        )
        raise AirflowException(f"Minute internal quality failed: {preview}")
    return result


def validate_minute_day_reconciliation_aplus(
    *,
    resolved_dates: Sequence[str | date] | None = None,
    minute_frame: pd.DataFrame | None = None,
    daily_prices: pd.DataFrame | None = None,
    persist_fn: Any = persist_reconciliation_events,
) -> dict[str, Any]:
    dates = _normalize_dates(resolved_dates)
    if not dates:
        return {"status": "skipped", "reason": "no_dates_to_validate", "dates_to_sync": []}

    minute_rows = minute_frame.copy() if minute_frame is not None else _load_minute_rows_for_dates(dates)
    if minute_rows.empty:
        raise AirflowException("Minute reconciliation found no rows for requested dates.")
    tickers = tuple(sorted(minute_rows["ticker"].astype(str).str.upper().unique().tolist()))
    reference_prices = daily_prices.copy() if daily_prices is not None else load_daily_prices(
        tickers=tickers,
        start_date=min(dates),
        end_date=max(dates),
    )
    filtered_daily = reference_prices.loc[
        (pd.to_datetime(reference_prices["trade_date"]).dt.date >= min(dates))
        & (pd.to_datetime(reference_prices["trade_date"]).dt.date <= max(dates))
    ].copy()
    reconciliation = validate_minute_to_day_consistency(minute_rows, filtered_daily)
    batch_id = f"minute-incremental-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    warning_count = persist_fn(
        reconciliation.get("warning_events", []),
        batch_id=batch_id,
        detected_at=datetime.now(timezone.utc),
    )
    result = {
        "status": "ok" if reconciliation.get("pass") else "error",
        "dates_to_sync": [trade_day.isoformat() for trade_day in dates],
        "reconciliation": reconciliation,
        "warning_event_count": int(warning_count),
        "reconciliation_batch_id": batch_id,
    }
    if not reconciliation.get("pass"):
        ohl_failures = [
            f"{field}:{payload['max_abs_bp']:.2f}bp"
            for field, payload in reconciliation.get("fields", {}).items()
            if payload.get("severity") == "blocker" and not payload.get("pass")
        ]
        raise AirflowException("Minute A-plus reconciliation blocker failed: " + ", ".join(ohl_failures))
    return result


def _minute_backfill_state_columns() -> set[str]:
    inspector = sa.inspect(get_engine())
    return {column["name"] for column in inspector.get_columns("minute_backfill_state")}


def publish_minute_watermark(
    *,
    resolved_dates: Sequence[str | date] | None = None,
    now: datetime | None = None,
    state_columns: set[str] | None = None,
) -> dict[str, Any]:
    current_time = now or datetime.now(timezone.utc)
    dates = _normalize_dates(resolved_dates)
    with get_engine().begin() as conn:
        watermark = conn.execute(
            sa.text("select max(trading_date) from minute_backfill_state where status = 'completed'"),
        ).scalar()
        columns = state_columns if state_columns is not None else _minute_backfill_state_columns()
        if dates and "published_at" in columns:
            conn.execute(
                sa.text(
                    """
                    update minute_backfill_state
                    set published_at = :published_at
                    where trading_date = any(:dates)
                      and status = 'completed'
                    """,
                ),
                {"published_at": current_time, "dates": list(dates)},
            )
        elif dates and "watermark" in columns and watermark is not None:
            conn.execute(
                sa.text(
                    """
                    update minute_backfill_state
                    set watermark = :watermark
                    where trading_date = any(:dates)
                      and status = 'completed'
                    """,
                ),
                {"watermark": watermark, "dates": list(dates)},
            )
        else:
            LOGGER.info(
                "minute_incremental watermark published via system log only: latest_covered_date=%s",
                watermark.isoformat() if watermark else None,
            )
    return {
        "status": "ok",
        "dates_to_sync": [trade_day.isoformat() for trade_day in dates],
        "latest_covered_date": watermark.isoformat() if watermark else None,
        "published_at_utc": current_time.isoformat(),
    }


def _task_resolve_minute_dates_to_sync(**context: Any) -> dict[str, Any]:
    return resolve_minute_dates_to_sync()


def _task_sync_polygon_minute_incremental(**context: Any) -> dict[str, Any]:
    dates = _pull_xcom_dates(context, task_id="minute_incremental.resolve_minute_dates_to_sync")
    return sync_polygon_minute_incremental(resolved_dates=dates)


def _task_validate_minute_internal_quality(**context: Any) -> dict[str, Any]:
    dates = _pull_xcom_dates(context, task_id="minute_incremental.resolve_minute_dates_to_sync")
    return validate_minute_internal_quality(resolved_dates=dates)


def _task_validate_minute_day_reconciliation_aplus(**context: Any) -> dict[str, Any]:
    dates = _pull_xcom_dates(context, task_id="minute_incremental.resolve_minute_dates_to_sync")
    return validate_minute_day_reconciliation_aplus(resolved_dates=dates)


def _task_publish_minute_watermark(**context: Any) -> dict[str, Any]:
    dates = _pull_xcom_dates(context, task_id="minute_incremental.resolve_minute_dates_to_sync")
    return publish_minute_watermark(resolved_dates=dates)


def build_minute_incremental_task_group(*, dag: Any) -> TaskGroup:
    with TaskGroup(group_id="minute_incremental", dag=dag) as task_group:
        resolve_dates = PythonOperator(
            task_id="resolve_minute_dates_to_sync",
            python_callable=_task_resolve_minute_dates_to_sync,
        )
        sync_incremental = PythonOperator(
            task_id="sync_polygon_minute_incremental",
            python_callable=_task_sync_polygon_minute_incremental,
            trigger_rule=TriggerRule.ALL_DONE,
        )
        validate_internal = PythonOperator(
            task_id="validate_minute_internal_quality",
            python_callable=_task_validate_minute_internal_quality,
            trigger_rule=TriggerRule.ALL_DONE,
        )
        validate_reconciliation = PythonOperator(
            task_id="validate_minute_day_reconciliation_aplus",
            python_callable=_task_validate_minute_day_reconciliation_aplus,
            trigger_rule=TriggerRule.ALL_DONE,
        )
        publish_watermark = PythonOperator(
            task_id="publish_minute_watermark",
            python_callable=_task_publish_minute_watermark,
            trigger_rule=TriggerRule.ALL_DONE,
        )

        resolve_dates >> sync_incremental >> validate_internal >> validate_reconciliation >> publish_watermark

    return task_group
