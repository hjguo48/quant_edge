from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import sqlalchemy as sa
import yaml
from sqlalchemy.engine import Engine

from src.config.week4_trades import Week4TradesConfig
from src.data.db.session import get_engine

DEFAULT_CONFIG_PATH = Path("configs/research/week4_trades_sampling.yaml")
DEFAULT_OUTPUT_PATH = Path("data/reports/week4/preflight_estimate.json")
DEFAULT_START_DATE = date(2016, 4, 17)
DEFAULT_END_DATE = date(2026, 4, 17)
BYTES_PER_TRADE_UNCOMPRESSED = 150
BYTES_PER_TRADE_COMPRESSED = 40
XNYS = xcals.get_calendar("XNYS")


@dataclass(frozen=True)
class CandidateEvent:
    ticker: str
    trading_date: date
    reason: str

    @property
    def key(self) -> tuple[str, date]:
        return (self.ticker.upper(), self.trading_date)


@dataclass(frozen=True)
class TransactionEstimate:
    ticker: str
    trading_date: date
    transactions: int
    has_minute_data: bool = True

    @property
    def key(self) -> tuple[str, date]:
        return (self.ticker.upper(), self.trading_date)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate Week 4 trade sampling API/storage budget.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--concurrency", type=int, default=1)
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


def load_config(config_path: Path) -> Week4TradesConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    assert_no_placeholders(payload)
    return Week4TradesConfig.model_validate(payload)


def assert_no_placeholders(payload: Any, path: str = "root") -> None:
    placeholder_strings = {"...", "todo", "tbd", "placeholder", "changeme"}
    if isinstance(payload, dict):
        for key, value in payload.items():
            assert_no_placeholders(value, f"{path}.{key}")
        return
    if isinstance(payload, list):
        for idx, value in enumerate(payload):
            assert_no_placeholders(value, f"{path}[{idx}]")
        return
    if isinstance(payload, str) and payload.strip().lower() in placeholder_strings:
        raise ValueError(f"placeholder value found at {path}: {payload!r}")


def compute_config_hash(config: Week4TradesConfig) -> str:
    normalized = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _sessions_in_range(start_date: date, end_date: date) -> list[date]:
    if end_date < start_date:
        return []
    sessions = XNYS.sessions_in_range(pd.Timestamp(start_date), pd.Timestamp(end_date))
    return [pd.Timestamp(session).date() for session in sessions]


def _add_event(
    events: list[CandidateEvent],
    ticker: Any,
    trading_date: Any,
    reason: str,
    *,
    start_date: date,
    end_date: date,
) -> None:
    if ticker is None or trading_date is None:
        return
    event_date = pd.Timestamp(trading_date).date()
    if event_date < start_date or event_date > end_date:
        return
    events.append(CandidateEvent(ticker=str(ticker).upper(), trading_date=event_date, reason=reason))


def _load_earnings_events(engine: Engine, start_date: date, end_date: date, window_days: int) -> list[CandidateEvent]:
    query = sa.text(
        """
        select distinct ticker, fiscal_date
        from earnings_estimates
        where fiscal_date between :query_start and :query_end
        order by ticker, fiscal_date
        """,
    )
    query_start = start_date - timedelta(days=window_days + 7)
    query_end = end_date + timedelta(days=window_days + 7)
    events: list[CandidateEvent] = []
    with engine.connect() as conn:
        rows = conn.execute(query, {"query_start": query_start, "query_end": query_end}).mappings().all()
    for row in rows:
        fiscal_date = pd.Timestamp(row["fiscal_date"]).date()
        for trading_date in _sessions_in_range(
            fiscal_date - timedelta(days=window_days),
            fiscal_date + timedelta(days=window_days),
        ):
            _add_event(events, row["ticker"], trading_date, "earnings", start_date=start_date, end_date=end_date)
    return events


def _load_gap_events(engine: Engine, start_date: date, end_date: date, gap_threshold_pct: float) -> list[CandidateEvent]:
    query = sa.text(
        """
        with px as (
            select
                ticker,
                trade_date,
                open,
                lag(close) over (partition by ticker order by trade_date) as prev_close
            from stock_prices
            where trade_date between :query_start and :end_date
        )
        select distinct ticker, trade_date
        from px
        where trade_date between :start_date and :end_date
          and open is not null
          and prev_close is not null
          and prev_close <> 0
          and abs((open - prev_close) / prev_close) >= :gap_threshold_pct
        order by ticker, trade_date
        """,
    )
    events: list[CandidateEvent] = []
    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {
                "query_start": start_date - timedelta(days=10),
                "start_date": start_date,
                "end_date": end_date,
                "gap_threshold_pct": gap_threshold_pct,
            },
        ).mappings().all()
    for row in rows:
        _add_event(events, row["ticker"], row["trade_date"], "gap", start_date=start_date, end_date=end_date)
    return events


def _load_top_liquidity_events(
    engine: Engine,
    start_date: date,
    end_date: date,
    *,
    top_n: int,
    lookback_days: int,
    reason: str,
) -> list[CandidateEvent]:
    # Task 0 is a budget preflight, not the final PIT liquidity implementation.
    # Use daily dollar volume as a fast proxy; Task 2 owns exact rolling ADV + PIT membership.
    _ = lookback_days
    query = sa.text(
        """
        with px as (
            select
                ticker,
                trade_date,
                (close::numeric * volume::numeric) as dollar_volume
            from stock_prices sp
            where sp.trade_date between :history_start and :end_date
              and sp.close is not null
              and sp.volume is not null
        ),
        ranked as (
            select
                ticker,
                trade_date,
                row_number() over (partition by trade_date order by dollar_volume desc nulls last) as rn
            from px
            where trade_date between :start_date and :end_date
        )
        select ticker, trade_date
        from ranked
        where rn <= :top_n
        order by trade_date, rn
        """,
    )
    events: list[CandidateEvent] = []
    with engine.connect() as conn:
        rows = conn.execute(
            query,
            {
                "history_start": start_date - timedelta(days=max(lookback_days * 3, 30)),
                "start_date": start_date,
                "end_date": end_date,
                "top_n": top_n,
            },
        ).mappings().all()
    for row in rows:
        _add_event(events, row["ticker"], row["trade_date"], reason, start_date=start_date, end_date=end_date)
    return events


def build_candidate_events_from_db(
    config: Week4TradesConfig,
    start_date: date,
    end_date: date,
    *,
    engine: Engine | None = None,
) -> list[CandidateEvent]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    db_engine = engine or get_engine()
    events: list[CandidateEvent] = []
    pilot = config.sampling.pilot
    if "earnings" in pilot.reasons:
        events.extend(_load_earnings_events(db_engine, start_date, end_date, pilot.earnings_window_days))
    if "gap" in pilot.reasons:
        events.extend(_load_gap_events(db_engine, start_date, end_date, pilot.gap_threshold_pct))
    if "weak_window" in pilot.reasons:
        for window in pilot.weak_windows:
            window_start = max(start_date, window.start)
            window_end = min(end_date, window.end)
            if window_end >= window_start:
                events.extend(
                    _load_top_liquidity_events(
                        db_engine,
                        window_start,
                        window_end,
                        top_n=pilot.weak_window_top_n,
                        lookback_days=config.sampling.stage2.top_liquidity_lookback_days,
                        reason="weak_window",
                    ),
                )
    if config.stage == "stage2":
        events.extend(
            _load_top_liquidity_events(
                db_engine,
                start_date,
                end_date,
                top_n=config.sampling.stage2.top_n_liquidity,
                lookback_days=config.sampling.stage2.top_liquidity_lookback_days,
                reason="top_liquidity",
            ),
        )
    return events


def load_transaction_estimates_from_db(
    keys: set[tuple[str, date]],
    *,
    engine: Engine | None = None,
) -> list[TransactionEstimate]:
    if not keys:
        return []
    db_engine = engine or get_engine()
    tickers = sorted({ticker for ticker, _ in keys})
    min_date = min(trading_date for _, trading_date in keys)
    max_date = max(trading_date for _, trading_date in keys)
    estimates: list[TransactionEstimate] = []
    with db_engine.connect() as conn:
        rows = conn.execute(
            sa.text(
                """
                select
                    ticker,
                    trade_date,
                    coalesce(sum(coalesce(transactions, 0)), 0)::bigint as transactions,
                    count(minute_ts)::bigint as minute_rows
                from stock_minute_aggs
                where ticker = any(:tickers)
                  and trade_date between :min_date and :max_date
                group by ticker, trade_date
                """,
            ),
            {"tickers": tickers, "min_date": min_date, "max_date": max_date},
        ).mappings().all()
    for row in rows:
        estimates.append(
            TransactionEstimate(
                ticker=str(row["ticker"]).upper(),
                trading_date=pd.Timestamp(row["trade_date"]).date(),
                transactions=int(row["transactions"] or 0),
                has_minute_data=int(row["minute_rows"] or 0) > 0,
            ),
        )
    return estimates


def estimate_from_candidates(
    config: Week4TradesConfig,
    candidates: list[CandidateEvent],
    transaction_estimates: list[TransactionEstimate],
    *,
    config_hash: str,
    start_date: date,
    end_date: date,
    concurrency: int = 1,
) -> dict[str, Any]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    unique_keys = {candidate.key for candidate in candidates}
    reason_counts: dict[str, int] = {}
    for candidate in candidates:
        reason_counts[candidate.reason] = reason_counts.get(candidate.reason, 0) + 1
    tx_map = {
        estimate.key: max(int(estimate.transactions), 0)
        for estimate in transaction_estimates
        if estimate.has_minute_data
    }
    missing_keys = sorted(unique_keys - set(tx_map.keys()), key=lambda item: (item[1], item[0]))
    rows_by_key = {key: tx_map[key] for key in sorted(unique_keys) if key in tx_map}
    api_calls_by_key = {
        key: int(math.ceil(rows / config.polygon.rest_page_size))
        for key, rows in rows_by_key.items()
    }
    daily_api_calls: dict[str, int] = {}
    for (_, trading_date), calls in api_calls_by_key.items():
        day_key = trading_date.isoformat()
        daily_api_calls[day_key] = daily_api_calls.get(day_key, 0) + calls
    daily_call_overflows = [
        {"trading_date": trading_date, "api_calls": calls}
        for trading_date, calls in sorted(daily_api_calls.items())
        if calls > config.budgets.max_daily_api_calls
    ]
    row_limit_overflows = [
        {"ticker": ticker, "trading_date": trading_date.isoformat(), "rows": rows}
        for (ticker, trading_date), rows in rows_by_key.items()
        if rows > config.budgets.max_rows_per_ticker_day
    ]
    total_rows = int(sum(rows_by_key.values()))
    api_calls = int(sum(api_calls_by_key.values()))
    # TODO(Task 4): populate probe_calls once Polygon client defines ticker-existence probe policy.
    probe_calls = 0
    uncompressed_storage_gb = total_rows * BYTES_PER_TRADE_UNCOMPRESSED / (1024**3)
    compressed_storage_gb = total_rows * BYTES_PER_TRADE_COMPRESSED / (1024**3)
    wall_time_hours = (
        (api_calls + probe_calls) * config.polygon.rest_min_interval_seconds / concurrency / 3600
    )
    failure_reasons: list[str] = []
    if not unique_keys:
        failure_reasons.append("no_sampling_candidates")
    if missing_keys:
        failure_reasons.append("missing_transaction_estimates")
    if uncompressed_storage_gb > config.budgets.max_storage_gb:
        failure_reasons.append("storage_budget_exceeded")
    if daily_call_overflows:
        failure_reasons.append("daily_api_call_budget_exceeded")
    if row_limit_overflows:
        failure_reasons.append("ticker_day_row_limit_exceeded")
    verdict = "PASS" if not failure_reasons else "FAIL"
    return {
        "metadata": {
            "script": "scripts/preflight_trades_estimator.py",
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "config_hash": config_hash,
            "stage": config.stage,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "bytes_per_trade_uncompressed": BYTES_PER_TRADE_UNCOMPRESSED,
            "bytes_per_trade_compressed": BYTES_PER_TRADE_COMPRESSED,
        },
        "inputs": {
            "candidate_event_rows": len(candidates),
            "unique_ticker_days": len(unique_keys),
            "reason_counts": reason_counts,
            "transaction_estimate_rows": len(transaction_estimates),
            "missing_transaction_estimate_count": len(missing_keys),
            "missing_transaction_estimate_examples": [
                {"ticker": ticker, "trading_date": trading_date.isoformat()}
                for ticker, trading_date in missing_keys[:20]
            ],
        },
        "estimates": {
            "rows": total_rows,
            "api_calls": api_calls,
            "probe_calls": probe_calls,
            "max_daily_api_calls_estimated": max(daily_api_calls.values(), default=0),
            "daily_api_call_overflow_examples": daily_call_overflows[:20],
            "row_limit_overflow_examples": row_limit_overflows[:20],
            "storage_uncompressed_gb": uncompressed_storage_gb,
            "storage_compressed_gb": compressed_storage_gb,
            "wall_time_hours": wall_time_hours,
        },
        "budgets": {
            "max_daily_api_calls": config.budgets.max_daily_api_calls,
            "max_storage_gb": config.budgets.max_storage_gb,
            "max_rows_per_ticker_day": config.budgets.max_rows_per_ticker_day,
            "expected_pilot_ticker_days": config.budgets.expected_pilot_ticker_days,
        },
        "verdict": verdict,
        "pass": verdict == "PASS",
        "failure_reasons": failure_reasons,
    }


def run_estimator(
    *,
    config_path: Path,
    start_date: date,
    end_date: date,
    output_path: Path,
    concurrency: int = 1,
    engine: Engine | None = None,
) -> dict[str, Any]:
    config = load_config(config_path)
    config_hash = compute_config_hash(config)
    db_engine = engine or get_engine()
    candidates = build_candidate_events_from_db(config, start_date, end_date, engine=db_engine)
    unique_keys = {candidate.key for candidate in candidates}
    transaction_estimates = load_transaction_estimates_from_db(unique_keys, engine=db_engine)
    report = estimate_from_candidates(
        config,
        candidates,
        transaction_estimates,
        config_hash=config_hash,
        start_date=start_date,
        end_date=end_date,
        concurrency=concurrency,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_normalize_for_json(report), indent=2, sort_keys=True))
    return report


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_estimator(
        config_path=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=args.output,
        concurrency=args.concurrency,
    )
    summary = {
        "verdict": report["verdict"],
        "failure_reasons": report["failure_reasons"],
        "unique_ticker_days": report["inputs"]["unique_ticker_days"],
        "rows": report["estimates"]["rows"],
        "api_calls": report["estimates"]["api_calls"],
        "storage_uncompressed_gb": report["estimates"]["storage_uncompressed_gb"],
        "output": str(args.output),
    }
    print(json.dumps(_normalize_for_json(summary), indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
