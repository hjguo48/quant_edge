from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import sqlalchemy as sa
import yaml
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.config.week4_trades import Week4TradesConfig
from src.data.db.models import StockTradesSampled, TradesSamplingState
from src.data.db.session import get_session_factory
from src.data.polygon_trades import PolygonTradesClient, TradeRecord
from src.data.sources.base import DataSourceAuthError

DEFAULT_PLAN_PATH = Path("data/reports/week4/trades_sampling_plan.parquet")
DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_BATCH_SIZE = 10_000
STATE_PENDING = "pending"
STATE_FAILED = "failed"
STATE_COMPLETED = "completed"


class StorageBudgetExceeded(RuntimeError):
    """Raised when stock_trades_sampled exceeds the configured storage budget."""


@dataclass(frozen=True)
class SamplingJob:
    ticker: str
    trading_date: date
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class FetchResult:
    job: SamplingJob
    raw_count: int
    rows: list[dict[str, Any]]
    dropped_conditions: int
    pages_fetched: int
    api_calls_used: int
    partial: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and persist targeted Polygon trades samples.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser.parse_args(argv)


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_for_json(item) for item in value]
    return value


def load_config(config_path: Path) -> Week4TradesConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    return Week4TradesConfig.model_validate(payload)


def load_plan(plan_path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(plan_path)
    required = {"ticker", "trading_date", "reason"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"sampling plan missing required columns: {sorted(missing)}")
    out = frame.loc[:, ["ticker", "trading_date", "reason"]].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper()
    out["trading_date"] = pd.to_datetime(out["trading_date"]).dt.date
    out["reason"] = out["reason"].astype(str).str.lower()
    out.drop_duplicates(["ticker", "trading_date", "reason"], inplace=True)
    out.sort_values(["trading_date", "ticker", "reason"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _reason_label(reasons: Sequence[str]) -> str:
    unique = tuple(sorted({str(reason).lower() for reason in reasons}))
    if len(unique) == 1:
        return unique[0]
    label = "+".join(unique)
    return label if len(label) <= 32 else "multi"


def _conditions_allowed(conditions: Sequence[int], allow_list: set[int]) -> bool:
    if not allow_list:
        return True
    return set(int(condition) for condition in conditions).issubset(allow_list)


def _estimate_pages(row_count: int, *, page_size: int, max_pages: int) -> int:
    if row_count <= 0:
        return 1
    return min(max_pages, max(1, int((row_count + page_size - 1) // page_size)))


def _is_partial(row_count: int, *, page_size: int, max_pages: int) -> bool:
    return row_count >= page_size * max_pages


def trade_to_record(
    trade: TradeRecord,
    *,
    entitlement_delay_minutes: int,
    sampled_reason: str,
) -> dict[str, Any]:
    return {
        "ticker": trade.ticker.upper(),
        "trading_date": trade.trading_date,
        "sip_timestamp": trade.sip_timestamp,
        "participant_timestamp": trade.participant_timestamp,
        "trf_timestamp": trade.trf_timestamp,
        "knowledge_time": trade.sip_timestamp + timedelta(minutes=entitlement_delay_minutes),
        "price": trade.price,
        "size": trade.size,
        "decimal_size": trade.decimal_size,
        "exchange": trade.exchange,
        "tape": trade.tape,
        "conditions": trade.conditions,
        "correction": trade.correction,
        "sequence_number": trade.sequence_number,
        "trade_id": trade.trade_id,
        "trf_id": trade.trf_id,
        "sampled_reason": sampled_reason,
    }


def fetch_and_prepare_job(
    *,
    client: PolygonTradesClient,
    job: SamplingJob,
    config: Week4TradesConfig,
) -> FetchResult:
    page_size = config.polygon.rest_page_size
    max_pages = config.polygon.rest_max_pages_per_request
    trades = list(
        client.fetch_trades_for_day(
            job.ticker,
            job.trading_date,
            page_size=page_size,
            max_pages=max_pages,
        ),
    )
    allow_list = set(config.features.condition_allow_list)
    sampled_reason = _reason_label(job.reasons)
    rows: list[dict[str, Any]] = []
    dropped_conditions = 0
    for trade in trades:
        if not _conditions_allowed(trade.conditions, allow_list):
            dropped_conditions += 1
            continue
        rows.append(
            trade_to_record(
                trade,
                entitlement_delay_minutes=config.polygon.entitlement_delay_minutes,
                sampled_reason=sampled_reason,
            ),
        )
    pages_fetched = _estimate_pages(len(trades), page_size=page_size, max_pages=max_pages)
    return FetchResult(
        job=job,
        raw_count=len(trades),
        rows=rows,
        dropped_conditions=dropped_conditions,
        pages_fetched=pages_fetched,
        api_calls_used=pages_fetched,
        partial=_is_partial(len(trades), page_size=page_size, max_pages=max_pages),
    )


class TradesSamplingRepository:
    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        self.session_factory = session_factory or get_session_factory()

    def load_jobs(self, plan: pd.DataFrame, *, resume: bool, limit: int | None = None) -> list[SamplingJob]:
        if plan.empty:
            return []
        plan_keys = {
            (str(row.ticker).upper(), row.trading_date, str(row.reason).lower())
            for row in plan.itertuples(index=False)
        }
        allowed_statuses = (STATE_PENDING, STATE_FAILED) if resume else (STATE_PENDING,)
        with self.session_factory() as session:
            rows = (
                session.execute(
                    sa.select(TradesSamplingState).where(TradesSamplingState.status.in_(allowed_statuses)),
                )
                .scalars()
                .all()
            )

        grouped: dict[tuple[str, date], set[str]] = defaultdict(set)
        for row in rows:
            key = (row.ticker.upper(), row.trading_date, row.sampled_reason.lower())
            if key in plan_keys:
                grouped[(key[0], key[1])].add(key[2])

        jobs = [
            SamplingJob(ticker=ticker, trading_date=trading_date, reasons=tuple(sorted(reasons)))
            for (ticker, trading_date), reasons in grouped.items()
        ]
        jobs.sort(key=lambda job: (job.trading_date, job.ticker, job.reasons))
        return jobs[:limit] if limit is not None else jobs

    def insert_trades(self, rows: list[dict[str, Any]], *, batch_size: int) -> int:
        if not rows:
            return 0
        inserted = 0
        with self.session_factory() as session:
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                statement = pg_insert(StockTradesSampled).values(chunk)
                statement = statement.on_conflict_do_nothing(
                    index_elements=[
                        StockTradesSampled.ticker,
                        StockTradesSampled.sip_timestamp,
                        StockTradesSampled.exchange,
                        StockTradesSampled.sequence_number,
                    ],
                )
                result = session.execute(statement)
                inserted += int(result.rowcount or 0)
            session.commit()
        return inserted

    def persist_completed(self, result: FetchResult, *, batch_size: int) -> int:
        message_parts: list[str] = []
        if result.partial:
            message_parts.append("partial=true")
            message_parts.append("max_pages_reached")
        if result.dropped_conditions:
            message_parts.append(f"dropped_conditions={result.dropped_conditions}")
        error_message = "; ".join(message_parts) if message_parts else None

        inserted = 0
        with self.session_factory() as session:
            for start in range(0, len(result.rows), batch_size):
                chunk = result.rows[start : start + batch_size]
                if not chunk:
                    continue
                statement = pg_insert(StockTradesSampled).values(chunk)
                statement = statement.on_conflict_do_nothing(
                    index_elements=[
                        StockTradesSampled.ticker,
                        StockTradesSampled.sip_timestamp,
                        StockTradesSampled.exchange,
                        StockTradesSampled.sequence_number,
                    ],
                )
                insert_result = session.execute(statement)
                inserted += int(insert_result.rowcount or 0)

            statement = (
                sa.update(TradesSamplingState)
                .where(
                    TradesSamplingState.ticker == result.job.ticker,
                    TradesSamplingState.trading_date == result.job.trading_date,
                    TradesSamplingState.sampled_reason.in_(result.job.reasons),
                )
                .values(
                    status=STATE_COMPLETED,
                    rows_ingested=inserted,
                    pages_fetched=result.pages_fetched,
                    api_calls_used=result.api_calls_used,
                    completed_at=datetime.now(timezone.utc),
                    error_message=error_message,
                )
            )
            session.execute(statement)
            session.commit()
        return inserted

    def mark_completed(self, result: FetchResult, *, rows_ingested: int) -> None:
        message_parts: list[str] = []
        if result.partial:
            message_parts.append("partial=true")
            message_parts.append("max_pages_reached")
        if result.dropped_conditions:
            message_parts.append(f"dropped_conditions={result.dropped_conditions}")
        self._update_state(
            result.job,
            status=STATE_COMPLETED,
            rows_ingested=rows_ingested,
            pages_fetched=result.pages_fetched,
            api_calls_used=result.api_calls_used,
            completed_at=datetime.now(timezone.utc),
            error_message="; ".join(message_parts) if message_parts else None,
        )

    def mark_failed(self, job: SamplingJob, error_message: str) -> None:
        self._update_state(
            job,
            status=STATE_FAILED,
            completed_at=datetime.now(timezone.utc),
            error_message=error_message[:2000],
        )

    def mark_budget_deferred(self, jobs: Sequence[SamplingJob], error_message: str) -> None:
        for job in jobs:
            self._update_state(job, status=STATE_PENDING, error_message=error_message[:2000])

    def relation_size_bytes(self) -> int:
        with self.session_factory() as session:
            bind = session.get_bind()
            if bind.dialect.name != "postgresql":
                return 0
            return int(
                session.execute(
                    sa.text("select pg_total_relation_size('stock_trades_sampled')"),
                ).scalar()
                or 0,
            )

    def _update_state(self, job: SamplingJob, **values: Any) -> None:
        with self.session_factory() as session:
            statement = (
                sa.update(TradesSamplingState)
                .where(
                    TradesSamplingState.ticker == job.ticker,
                    TradesSamplingState.trading_date == job.trading_date,
                    TradesSamplingState.sampled_reason.in_(job.reasons),
                )
                .values(**values)
            )
            session.execute(statement)
            session.commit()


def _jobs_by_date(jobs: Sequence[SamplingJob]) -> dict[date, list[SamplingJob]]:
    grouped: dict[date, list[SamplingJob]] = defaultdict(list)
    for job in jobs:
        grouped[job.trading_date].append(job)
    return dict(sorted(grouped.items()))


def _check_storage_budget(repository: TradesSamplingRepository, config: Week4TradesConfig) -> float:
    size_bytes = repository.relation_size_bytes()
    storage_gb = size_bytes / (1024**3)
    if size_bytes > config.budgets.max_storage_gb * (1024**3):
        raise StorageBudgetExceeded(
            f"stock_trades_sampled storage {storage_gb:.3f}GB exceeds "
            f"{config.budgets.max_storage_gb:.3f}GB budget",
        )
    return storage_gb


def run_sampling(
    *,
    plan_path: Path,
    config_path: Path,
    max_workers: int = 4,
    resume: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    repository: TradesSamplingRepository | None = None,
    client_factory: Callable[[], PolygonTradesClient] | None = None,
) -> dict[str, Any]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    config = load_config(config_path)
    plan = load_plan(plan_path)
    repo = repository or TradesSamplingRepository()
    jobs = repo.load_jobs(plan, resume=resume, limit=limit)
    client = client_factory() if client_factory else PolygonTradesClient(
        min_request_interval=config.polygon.rest_min_interval_seconds,
    )

    summary: dict[str, Any] = {
        "total_jobs": len(jobs),
        "completed_jobs": 0,
        "failed_jobs": 0,
        "deferred_jobs": 0,
        "rows_ingested": 0,
        "api_calls_used": 0,
        "dry_run": dry_run,
        "resume": resume,
        "stopped_reason": None,
        "storage_gb": 0.0,
    }
    daily_api_calls: dict[str, int] = defaultdict(int)
    grouped_jobs = _jobs_by_date(jobs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for trading_date, day_jobs in grouped_jobs.items():
            day_key = trading_date.isoformat()
            idx = 0
            while idx < len(day_jobs):
                remaining_budget = config.budgets.max_daily_api_calls - daily_api_calls[day_key]
                if remaining_budget <= 0:
                    remaining = day_jobs[idx:]
                    repo.mark_budget_deferred(remaining, "daily_api_budget_exhausted")
                    summary["deferred_jobs"] += len(remaining)
                    summary["stopped_reason"] = "daily_api_budget_exhausted"
                    return summary

                batch_jobs = day_jobs[idx : idx + min(max_workers, remaining_budget, len(day_jobs) - idx)]
                futures = {
                    executor.submit(fetch_and_prepare_job, client=client, job=job, config=config): job
                    for job in batch_jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        result = future.result()
                    except DataSourceAuthError:
                        raise
                    except Exception as exc:
                        repo.mark_failed(job, str(exc))
                        summary["failed_jobs"] += 1
                        continue

                    daily_api_calls[day_key] += result.api_calls_used
                    summary["api_calls_used"] += result.api_calls_used

                    if result.raw_count > config.budgets.max_rows_per_ticker_day:
                        repo.mark_failed(job, "row_limit_exceeded")
                        summary["failed_jobs"] += 1
                        continue

                    inserted = len(result.rows) if dry_run else repo.persist_completed(result, batch_size=batch_size)
                    if not dry_run:
                        summary["storage_gb"] = _check_storage_budget(repo, config)
                    summary["rows_ingested"] += inserted
                    summary["completed_jobs"] += 1

                    processed = summary["completed_jobs"] + summary["failed_jobs"]
                    if processed and processed % 100 == 0:
                        print(
                            json.dumps(
                                _normalize_for_json(
                                    {
                                        "completed": summary["completed_jobs"],
                                        "total": summary["total_jobs"],
                                        "storage_gb": summary["storage_gb"],
                                        "api_calls_today": dict(daily_api_calls),
                                    },
                                ),
                                sort_keys=True,
                            ),
                        )

                idx += len(batch_jobs)

                if daily_api_calls[day_key] >= config.budgets.max_daily_api_calls and idx < len(day_jobs):
                    remaining = day_jobs[idx:]
                    repo.mark_budget_deferred(remaining, "daily_api_budget_exhausted")
                    summary["deferred_jobs"] += len(remaining)
                    summary["stopped_reason"] = "daily_api_budget_exhausted"
                    return summary

    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_sampling(
        plan_path=args.plan,
        config_path=args.config,
        max_workers=args.max_workers,
        resume=args.resume,
        dry_run=args.dry_run,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(json.dumps(_normalize_for_json(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
