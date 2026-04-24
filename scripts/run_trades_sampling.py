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
from src.features.trade_microstructure import (
    TRADE_MICROSTRUCTURE_FEATURE_NAMES,
    compute_knowledge_time,
    compute_large_trade_ratio,
    compute_late_day_aggressiveness,
    compute_off_exchange_volume_ratio,
    compute_offhours_trade_ratio,
    compute_trade_imbalance_proxy,
)

DEFAULT_PLAN_PATH = Path("data/reports/week4/trades_sampling_plan.parquet")
DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_FEATURES_OUTPUT = Path("data/features/trade_microstructure_features.parquet")
DEFAULT_BATCH_SIZE = 10_000
DEFAULT_STREAMING_FLUSH_EVERY = 200  # flush features parquet every N completed jobs
STATE_PENDING = "pending"
STATE_FAILED = "failed"
STATE_COMPLETED = "completed"
STATE_PARTIAL = "partial"

FEATURE_OUTPUT_COLUMNS = [
    "event_date",
    "ticker",
    "knowledge_time_regular",
    "knowledge_time_offhours",
    *TRADE_MICROSTRUCTURE_FEATURE_NAMES,
    "run_config_hash",
]


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
    rows: list[dict[str, Any]]  # DB-mode: trade rows for insert; streaming-mode: empty
    dropped_conditions: int
    pages_fetched: int
    api_calls_used: int
    partial: bool
    features_row: dict[str, Any] | None = None  # streaming-mode: computed feature row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and persist targeted Polygon trades samples.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--streaming-mode",
        action="store_true",
        help="Compute 5 trade features from fetched trades in-memory and write parquet; skip DB persist.",
    )
    parser.add_argument(
        "--features-output",
        type=Path,
        default=DEFAULT_FEATURES_OUTPUT,
        help="Output parquet for streaming mode features (ignored in DB mode).",
    )
    parser.add_argument(
        "--streaming-flush-every",
        type=int,
        default=DEFAULT_STREAMING_FLUSH_EVERY,
        help="Checkpoint features parquet every N completed jobs in streaming mode.",
    )
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


def _trades_to_feature_dataframe(trades: list[TradeRecord]) -> pd.DataFrame:
    """Convert TradeRecord list to DataFrame expected by Task 7 feature functions."""
    if not trades:
        return pd.DataFrame(
            columns=["sip_timestamp", "price", "size", "exchange", "trf_id", "trf_timestamp", "conditions"],
        )
    return pd.DataFrame(
        [
            {
                "sip_timestamp": trade.sip_timestamp,
                "price": trade.price,
                "size": trade.size,
                "exchange": trade.exchange,
                "trf_id": trade.trf_id,
                "trf_timestamp": trade.trf_timestamp,
                "conditions": trade.conditions,
            }
            for trade in trades
        ],
    )


def compute_features_inline(
    trades: list[TradeRecord],
    ticker: str,
    trading_date: date,
    config: Week4TradesConfig,
    config_hash: str,
) -> dict[str, Any]:
    """Streaming-mode: compute 5 trade microstructure features directly from in-memory trades.

    Returns dict with feature columns suitable for FEATURE_OUTPUT_COLUMNS schema.
    """
    frame = _trades_to_feature_dataframe(trades)
    condition_allow = set(config.features.condition_allow_list) or None
    trf_exchange_codes = set(config.features.trf_exchange_codes)
    late_day_window = tuple(config.features.late_day_window_et)
    pre_window = tuple(config.features.offhours_window_et_pre)
    post_window = tuple(config.features.offhours_window_et_post)
    size_threshold = float(config.features.size_threshold_dollars)

    return {
        "event_date": trading_date,
        "ticker": ticker.upper(),
        "knowledge_time_regular": compute_knowledge_time(trading_date, "trade_imbalance_proxy"),
        "knowledge_time_offhours": compute_knowledge_time(trading_date, "offhours_trade_ratio"),
        "trade_imbalance_proxy": compute_trade_imbalance_proxy(frame, condition_allow=condition_allow),
        "large_trade_ratio": compute_large_trade_ratio(frame, size_threshold_dollars=size_threshold),
        "late_day_aggressiveness": compute_late_day_aggressiveness(frame, late_day_window_et=late_day_window),
        "offhours_trade_ratio": compute_offhours_trade_ratio(
            frame, pre_window_et=pre_window, post_window_et=post_window,
        ),
        "off_exchange_volume_ratio": compute_off_exchange_volume_ratio(
            frame, trf_exchange_codes=trf_exchange_codes,
        ),
        "run_config_hash": config_hash,
    }


def fetch_and_prepare_job(
    *,
    client: PolygonTradesClient,
    job: SamplingJob,
    config: Week4TradesConfig,
    streaming_mode: bool = False,
    config_hash: str = "",
) -> FetchResult:
    page_size = config.polygon.rest_page_size
    max_pages = config.polygon.rest_max_pages_per_request
    trades, api_calls_used = client.fetch_trades_for_day(
        job.ticker,
        job.trading_date,
        page_size=page_size,
        max_pages=max_pages,
    )
    allow_list = set(config.features.condition_allow_list)
    sampled_reason = _reason_label(job.reasons)
    dropped_conditions = 0
    # In DB mode: materialize trade rows; in streaming mode: skip to save memory + CPU
    rows: list[dict[str, Any]] = []
    features_row: dict[str, Any] | None = None

    if streaming_mode:
        # Still count dropped conditions from the same allow-list (feature functions apply filter too,
        # but state tracking expects dropped count).
        if allow_list:
            dropped_conditions = sum(
                1 for trade in trades if not _conditions_allowed(trade.conditions, allow_list)
            )
        features_row = compute_features_inline(
            trades,
            ticker=job.ticker,
            trading_date=job.trading_date,
            config=config,
            config_hash=config_hash,
        )
    else:
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
    return FetchResult(
        job=job,
        raw_count=len(trades),
        rows=rows,
        dropped_conditions=dropped_conditions,
        pages_fetched=api_calls_used,
        api_calls_used=api_calls_used,
        partial=_is_partial(len(trades), page_size=page_size, max_pages=max_pages),
        features_row=features_row,
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
                    status=STATE_PARTIAL if result.partial else STATE_COMPLETED,
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
        if result.dropped_conditions:
            message_parts.append(f"dropped_conditions={result.dropped_conditions}")
        self._update_state(
            result.job,
            status=STATE_PARTIAL if result.partial else STATE_COMPLETED,
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


def _flush_streaming_features(
    features_accumulator: list[dict[str, Any]],
    output_path: Path,
) -> int:
    """Write accumulated feature rows to parquet atomically (full overwrite)."""
    if not features_accumulator:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(features_accumulator, columns=FEATURE_OUTPUT_COLUMNS)
    frame["event_date"] = pd.to_datetime(frame["event_date"]).dt.date
    frame = frame.drop_duplicates(["event_date", "ticker"], keep="last")
    frame = frame.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(output_path)
    return len(frame)


def _load_streaming_features_checkpoint(output_path: Path) -> list[dict[str, Any]]:
    """Load prior features parquet as list (resume-friendly)."""
    if not output_path.exists():
        return []
    existing = pd.read_parquet(output_path)
    if existing.empty:
        return []
    existing["event_date"] = pd.to_datetime(existing["event_date"]).dt.date
    return existing.to_dict(orient="records")


def run_sampling(
    *,
    plan_path: Path,
    config_path: Path,
    max_workers: int = 4,
    resume: bool = False,
    dry_run: bool = False,
    limit: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    streaming_mode: bool = False,
    features_output: Path = DEFAULT_FEATURES_OUTPUT,
    streaming_flush_every: int = DEFAULT_STREAMING_FLUSH_EVERY,
    repository: TradesSamplingRepository | None = None,
    client_factory: Callable[[], PolygonTradesClient] | None = None,
) -> dict[str, Any]:
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if streaming_mode and dry_run:
        raise ValueError("--streaming-mode and --dry-run are incompatible")

    config = load_config(config_path)
    # Compute config_hash for streaming mode parquet traceability (import here to avoid cycle)
    config_hash = ""
    if streaming_mode:
        try:
            import scripts.preflight_trades_estimator as _pf
        except ModuleNotFoundError:  # pragma: no cover
            import preflight_trades_estimator as _pf
        config_hash = _pf.compute_config_hash(config)

    plan = load_plan(plan_path)
    repo = repository or TradesSamplingRepository()
    jobs = repo.load_jobs(plan, resume=resume, limit=limit)
    client = client_factory() if client_factory else PolygonTradesClient(
        min_request_interval=config.polygon.rest_min_interval_seconds,
    )

    features_accumulator: list[dict[str, Any]] = (
        _load_streaming_features_checkpoint(features_output) if streaming_mode and resume else []
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
        "streaming_mode": streaming_mode,
        "features_output": str(features_output) if streaming_mode else None,
        "features_rows_written": len(features_accumulator),
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

                batch_jobs = day_jobs[idx : idx + min(max_workers, len(day_jobs) - idx)]
                futures = {
                    executor.submit(
                        fetch_and_prepare_job,
                        client=client,
                        job=job,
                        config=config,
                        streaming_mode=streaming_mode,
                        config_hash=config_hash,
                    ): job
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

                    if result.raw_count > config.budgets.max_rows_per_ticker_day and not streaming_mode:
                        # Row-limit only enforced in DB mode (protects stock_trades_sampled storage)
                        repo.mark_failed(job, "row_limit_exceeded")
                        summary["failed_jobs"] += 1
                        continue

                    if streaming_mode:
                        # Accumulate feature row; no DB insert; skip storage kill-switch
                        if result.features_row is not None:
                            features_accumulator.append(result.features_row)
                        repo.mark_completed(result, rows_ingested=0)
                    else:
                        inserted = (
                            len(result.rows)
                            if dry_run
                            else repo.persist_completed(result, batch_size=batch_size)
                        )
                        if not dry_run:
                            summary["storage_gb"] = _check_storage_budget(repo, config)
                        summary["rows_ingested"] += inserted

                    summary["completed_jobs"] += 1

                    processed = summary["completed_jobs"] + summary["failed_jobs"]
                    if streaming_mode and processed and processed % streaming_flush_every == 0:
                        summary["features_rows_written"] = _flush_streaming_features(
                            features_accumulator, features_output,
                        )
                    if processed and processed % 100 == 0:
                        print(
                            json.dumps(
                                _normalize_for_json(
                                    {
                                        "completed": summary["completed_jobs"],
                                        "total": summary["total_jobs"],
                                        "storage_gb": summary["storage_gb"],
                                        "features_rows_written": summary["features_rows_written"],
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
                    # final flush before returning
                    if streaming_mode:
                        summary["features_rows_written"] = _flush_streaming_features(
                            features_accumulator, features_output,
                        )
                    return summary

    # final flush at end of successful run
    if streaming_mode:
        summary["features_rows_written"] = _flush_streaming_features(
            features_accumulator, features_output,
        )
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
        streaming_mode=args.streaming_mode,
        features_output=args.features_output,
        streaming_flush_every=args.streaming_flush_every,
    )
    print(json.dumps(_normalize_for_json(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
